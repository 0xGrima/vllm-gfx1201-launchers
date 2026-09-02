#!/bin/bash
# kv-memory-calibrate.sh -- measure a real --kv-cache-memory pin for THIS host/batch-shape via a
# boot-and-push search, so startup-qwen3.8-27b-vllm.sh stops leaving KV cache unclaimed.
#
# CREDIT: this script is adapted from Brian's (ggz14) calibrate-kv.sh in
# codeberg.org/ggz14/radiance-vllm-mxfp4 (commit ffe8ea4) -- the search strategy, the "profile
# then push" two-pass design, and the insight that vLLM's own profiling step under-reports real
# capacity are all his. This is a light port for a single-GPU (TP=1) host and this repo's own
# startup-qwen3.8-27b-vllm.sh (different env var names/port/served-model-name than upstream's
# serve-mxfp4.sh), not a rewrite of the idea. If you find this useful, go look at the rest of his
# repo -- the runtime patch delta this whole launcher repo depends on (see README's Overview) is
# also his work.
#
# NOT THE SAME THING as calibrating fp8 KV-cache SCALES (accuracy, k_scale/v_scale) -- see the
# README's "KV cache: scale calibration and sizing" section for that, a completely different axis
# (correctness of what's stored) from what this script does (how much of it fits).
#
#   ./kv-memory-calibrate.sh            measure at the default batch shape and save the result
#   ./kv-memory-calibrate.sh --dry-run  show the plan, run nothing
#   ./kv-memory-calibrate.sh --quick    pass 1 only: pin what vLLM profiles, skip the search
#
# Needs the GPU to itself and takes ~15-20 minutes. Result lands in
# ~/.cache/radiance-mxfp4/kv-profiles.local.tsv and is NOT read automatically by the startup
# script -- apply a measured value by hand: KV_MEM=<bytes> ./startup-qwen3.8-27b-vllm.sh.
#
# WHY THIS IS A SEARCH AND NOT A FORMULA
# vLLM sizes the cache as requested_memory - non_kv_cache_memory - cudagraph_estimate, where
# non_kv_cache_memory is measured during a profile run and carries that run's TRANSIENT
# activation peak. Steady-state serving never needs that peak at the same time as a full cache,
# so the profiled figure underestimates what the card can actually hold. How much depends on the
# activation peak at CHUNK, the cudagraph capture set MAXSEQS produces, and allocator
# fragmentation on this specific card -- none of it is in the log. The only honest way to find
# the margin is to raise the pin until startup stops surviving, and back off.
#
# WHAT COUNTS AS SURVIVING
# Reaching "GPU KV cache size" is NOT enough: the cache is allocated before cudagraph capture,
# and capture is where an over-committed pin actually dies. A pass counts as passing only if the
# server answers /health AND completes one CHUNK-sized prefill plus a short decode.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
STANDALONE="$SCRIPT_DIR/startup-qwen3.8-27b-vllm.sh"

die() { echo "[kv-memory-calibrate] ERROR: $1" >&2; shift; for l in "$@"; do echo "  $l" >&2; done; exit 1; }
say() { echo "[kv-memory-calibrate] $*"; }

[ -x "$STANDALONE" ] || die "startup-qwen3.8-27b-vllm.sh not found next to this script" "expected: $STANDALONE"

DRY_RUN_ONLY=0; QUICK=0
for a in "$@"; do case "$a" in
  --dry-run) DRY_RUN_ONLY=1 ;;
  --quick)   QUICK=1 ;;
  -h|--help) sed -n '2,32p' "$0" | sed 's/^# \?//'; exit 0 ;;
  *) die "unknown argument: $a" "usage: ./kv-memory-calibrate.sh [--dry-run] [--quick]" ;;
esac; done

# The shape the pin will be valid for -- MUST match what you'll later run startup-qwen3.8-27b-vllm.sh
# with, because the lookup key includes them. Defaults mirror that script's own current defaults.
NAME=${NAME:-r9700-kvcal}
PORT=${PORT:-9319}
MODELNAME=${MODELNAME:-qwen3.8-27b-mxfp4}
MAXSEQS=${MAXSEQS:-3}
CHUNK=${CHUNK:-2560}
MAXLEN=${MAXLEN:-190000}
GPUUTIL=${GPUUTIL:-0.97}

# Hardware signature -- read straight from sysfs like upstream's gpu-detect.sh; TP=1 assumed
# (single-GPU host). Adjust if you're running this on a multi-GPU box.
RAD_PCIID=$(cat /sys/class/drm/renderD128/device/device 2>/dev/null | sed 's/^0x//' || echo 0000)
RAD_MIB=$(( $(cat /sys/class/drm/renderD128/device/mem_info_vram_total 2>/dev/null || echo 0) / 1048576 ))
RAD_SIG="1x${RAD_PCIID}-${RAD_MIB}"

# Search shape. STEP is a fraction of the profiled figure; 2% is fine grain against a margin
# measured at 5.7% on ggz14's own 2x R9700 reference box -- expect a similar order here.
STEP=${STEP:-0.02}
MAX_STEPS=${MAX_STEPS:-6}
# Back off one full step from the last size that passed. The failure this protects against is
# not startup -- that was just tested -- but a long-context prefill months later hitting a
# fragmentation pattern the calibration run never produced.
BACKOFF_STEPS=${BACKOFF_STEPS:-1}

LOCAL_TABLE=${KV_TABLE_LOCAL:-${XDG_CACHE_HOME:-$HOME/.cache}/radiance-mxfp4/kv-profiles.local.tsv}

say "hardware:  $RAD_SIG (device 0x$RAD_PCIID, ${RAD_MIB} MiB)"
say "shape:     maxseqs=$MAXSEQS chunk=$CHUNK maxlen=$MAXLEN util=$GPUUTIL"
say "table:     $LOCAL_TABLE"
if [ -r "$LOCAL_TABLE" ] && grep -qF "$(printf '%s\t%s\t%s\t%s' "$RAD_SIG" "$MAXSEQS" "$CHUNK" "$MAXLEN")" "$LOCAL_TABLE" 2>/dev/null; then
  say "note:      a pin already exists for this exact key in $LOCAL_TABLE -- this run will replace it"
fi

if [ "$DRY_RUN_ONLY" = 1 ]; then
  say "pass 1: serve with KV_MEM unset (profiling), read 'Available KV cache memory' from the log"
  if [ "$QUICK" = 0 ]; then
    say "pass 2: retry at +$(awk -v s="$STEP" 'BEGIN{printf "%.0f", s*100}')% steps, up to $MAX_STEPS, keeping the largest that serves"
    say "        then back off $BACKOFF_STEPS step(s) for margin"
  fi
  say "would write to: $LOCAL_TABLE"
  exit 0
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  die "port $PORT is in use -- calibration needs the GPU to itself" \
      "stop whatever's running on it first: docker stop $NAME (or your own container name)"
fi

cleanup() { docker stop -t 20 "$NAME" >/dev/null 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# Start a server at a given pin ("" = let vLLM profile) and report what happened.
# Echoes "PASS <available_gib> <toks>" / "FAIL <reason>" on stdout; everything else to stderr.
attempt() {
  local pin=$1 log; log=$(mktemp)
  cleanup
  CONTAINER_NAME="$NAME" PORT="$PORT" MAX_NUM_SEQS="$MAXSEQS" \
    MAX_MODEL_LEN="$MAXLEN" GPU_MEM_UTIL="$GPUUTIL" KV_MEM="$pin" \
    "$STANDALONE" >"$log" 2>&1
  local i ready=0
  for i in $(seq 1 180); do
    if [ "$(curl -s -o /dev/null -w %{http_code} -m3 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = 200 ]; then
      ready=1; break
    fi
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || break
    sleep 5
  done

  if [ "$ready" != 1 ]; then
    local why="did not become healthy"
    docker logs "$NAME" 2>&1 | grep -qiE 'out of memory|HIP out of memory|hipErrorOutOfMemory' "$log" 2>/dev/null && why="out of memory"
    echo "FAIL $why"; echo "--- last 15 log lines ---" >&2; docker logs --tail 15 "$NAME" 2>&1 >&2
    rm -f "$log"; return 0
  fi

  # Health alone does not prove the steady state fits -- force one CHUNK-sized prefill + decode.
  local prompt; prompt=$(python3 -c "print('the quick brown fox jumps over the lazy dog. ' * $((CHUNK / 10)))")
  local code
  code=$(curl -s -o /dev/null -w %{http_code} -m 300 -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "
import json,sys
print(json.dumps({'model':sys.argv[1],'prompt':sys.argv[2],'max_tokens':32,'temperature':0}))" "$MODELNAME" "$prompt")" 2>/dev/null || echo 000)

  if [ "$code" != 200 ]; then
    echo "FAIL prefill probe returned HTTP $code"
    echo "--- last 15 log lines ---" >&2; docker logs --tail 15 "$NAME" 2>&1 >&2
    rm -f "$log"; return 0
  fi

  local avail toks
  avail=$(docker logs "$NAME" 2>&1 | sed -n 's/.*Available KV cache memory: \([0-9.]*\) GiB.*/\1/p' | tail -1)
  toks=$(docker logs "$NAME" 2>&1 | sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' | tail -1)
  echo "PASS ${avail:-0} ${toks:-?}"
  rm -f "$log"
}

# ---------------------------------------------------------------- pass 1: profile
say "pass 1/2: profiling run (vLLM sizes the cache itself)"
res=$(attempt "")
set -- $res
[ "$1" = PASS ] || die "the profiling run itself failed: ${*:2}" \
    "this is not a calibration problem -- the configuration does not serve on this host at all" \
    "try: MAXLEN=32768 CHUNK=1536 $STANDALONE   and read the log"
base_gib=$2; base_toks=$3
base=$(awk -v g="$base_gib" 'BEGIN{printf "%d", g*1073741824}')
say "pass 1: profiled ${base_gib} GiB, ${base_toks} KV tokens"

best=$base; best_toks=$base_toks
if [ "$QUICK" = 1 ]; then
  say "--quick: keeping the profiled figure, skipping the search"
else
  # ---------------------------------------------------------------- pass 2: push
  say "pass 2/2: raising the pin until it stops serving"
  step=0
  while [ "$step" -lt "$MAX_STEPS" ]; do
    step=$((step + 1))
    try=$(awk -v b="$base" -v s="$STEP" -v n="$step" 'BEGIN{printf "%d", b*(1+s*n)}')
    pct=$(awk -v s="$STEP" -v n="$step" 'BEGIN{printf "%.0f", s*n*100}')
    say "  +${pct}%: $try bytes ($(awk -v b="$try" 'BEGIN{printf "%.2f", b/1073741824}') GiB)"
    r=$(attempt "$try"); set -- $r
    if [ "$1" = PASS ]; then
      say "  +${pct}%: served, $3 KV tokens"
      best=$try; best_toks=$3
    else
      say "  +${pct}%: ${*:2} -- stopping the search here"
      break
    fi
  done

  if [ "$best" != "$base" ] && [ "$BACKOFF_STEPS" -gt 0 ]; then
    backed=$(awk -v b="$best" -v s="$STEP" -v n="$BACKOFF_STEPS" 'BEGIN{printf "%d", b/(1+s*n)}')
    if [ "$backed" -gt "$base" ]; then
      say "backing off $BACKOFF_STEPS step(s) from the largest that served, for margin"
      best=$backed
    fi
  fi
fi

cleanup

gain=$(awk -v b="$best" -v a="$base" 'BEGIN{printf "%.1f", (b/a-1)*100}')
mkdir -p "$(dirname "$LOCAL_TABLE")"
if [ ! -s "$LOCAL_TABLE" ]; then
  cat > "$LOCAL_TABLE" <<'HDR'
# KV cache pins measured on THIS host by kv-memory-calibrate.sh.
# Not auto-consulted by startup-qwen3.8-27b-vllm.sh -- apply by hand:
# KV_MEM=<bytes> ./startup-qwen3.8-27b-vllm.sh
# Re-run after changing MAXSEQS or CHUNK: a pin is only valid for the shape it was measured at.
# sig	maxseqs	chunk	maxlen	bytes	note
HDR
fi
tmp=$(mktemp)
awk -F'\t' -v s="$RAD_SIG" -v q="$MAXSEQS" -v c="$CHUNK" -v l="$MAXLEN" \
  '$1 ~ /^#/ || !($1==s && $2==q && $3==c && $4==l)' "$LOCAL_TABLE" > "$tmp"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$RAD_SIG" "$MAXSEQS" "$CHUNK" "$MAXLEN" \
  "$best" "measured $(date +%Y-%m-%d) by kv-memory-calibrate.sh; ${best_toks} KV tokens, +${gain}% over profiled" >> "$tmp"
mv "$tmp" "$LOCAL_TABLE"

say "done. pin = $best bytes ($(awk -v b="$best" 'BEGIN{printf "%.2f", b/1073741824}') GiB), +${gain}% over profiling"
say "written to $LOCAL_TABLE -- apply by hand, this is not auto-consulted"
