# vllm-gfx1201-launchers

Standalone vLLM launchers for the AMD Radeon AI PRO R9700 (RDNA4/gfx1201, 32GB) — a showcase of
how two models are actually run on this card: **Qwen3.8-27B** at native MXFP4 (W4A8) with a
DFlash2 speculative drafter, and **Ornith-1.5-35B-A3B** (MoE, GDN linear-attention hybrid) at
plain AutoRound-produced GPTQ W4A16, full native context, vision enabled.


## Overview

Both models run on the same image — one 32 GB card fits one resident model at a time. Base is the
published [`stilldeadcode/vllm-radiance:0.9.3`](https://hub.docker.com/r/stilldeadcode/vllm-radiance)
(vLLM 0.27.1, DFlash2 and `libr4d` 0.5.0 already in-tree), with a runtime patch delta from
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) @ `dba9def` applied
on top for the native MXFP4 (W4A8) kernel path.

| model | quant | speculative decoding | context | decode (weighted combined) | prefill@16K |
|---|---|---|---|---:|---:|
| `qwen3.8-27b-mxfp4` | native MXFP4 (W4A8) | DFlash2-FP8 n=7 | 220,000 | **77.7-81.4 t/s**¹ | 2536-2560 t/s¹ |
| `ornith-1.5-35b-a3b` | AutoRound-produced GPTQ W4A16 (group_size 128) | **none** — this checkpoint's own MTP head isn't good enough to be worth it | 262,144, vision enabled | **~78 t/s** | ~5700-5800 t/s |

Both measured with BetterBench, 300 W, single-stream, weighted combined score across BetterBench's
own category weights (code 0.30 / reasoning 0.20 / prose 0.15 / json 0.15 / file_edit 0.10 /
summarization 0.10). Qwen3.8's range reflects several independent runs of this config — this
checkpoint's GDN-hybrid architecture shows real run-to-run KV/scheduling noise of a couple
percent; treat any single number as a point in that band, not a fixed constant.

**Concurrency** (see the section below the environment-variable table for the full numbers and
Ornith's own comparison): aggregate throughput scales ~1.8x going from 1 to 2 concurrent
streams (measured 66.6 t/s -> 121.5 t/s aggregate) and ~2.4x by 3 streams (161.3 t/s), with only
a modest per-stream decode cost (82.9 -> 80.3 -> 71.4 t/s).

¹ Decode/prefill numbers above were measured at this checkpoint's prior 190,000-context config
(`GPU_MEM_UTIL=0.97`, `KV_MEM=8232441724`). The context ceiling was since pushed to 220,000
(`GPU_MEM_UTIL=0.98`, `KV_MEM=9126805504`, see the KV cache section below) — not yet re-run
through a full BetterBench pass at the new ceiling, though throughput on this architecture has
consistently measured context-length-independent elsewhere in this repo's own testing.

### Benchmark (Qwen, production config)

Measured with BetterBench against the live production launcher: `Qwen3.8-27B-MXFP4` at its prior
190,000-token context config (`GPU_MEM_UTIL=0.97`, `KV_MEM=8232441724` — since pushed to
220,000/0.98/9126805504, see the KV cache section above; not yet re-run through a full
BetterBench pass at the new ceiling), DFlash2-FP8 speculative decoding (n=7), fp8 KV cache on
both target and drafter (real calibrated scales, not the uncalibrated fallback), `MAX_NUM_SEQS=3`,
300 W. Single-stream weighted-combined median decode: **80.9 t/s**. Median prefill:
**2771.8 t/s @2K** / **2538.8 t/s @16K**.

#### Single-stream decode per category (median)

| category | decode t/s (median) |
|---|---:|
| code | 91.8 |
| file_edit | 101.4 |
| json | 94.5 |
| prose | 49.1 |
| reasoning | 71.1 |
| summarization | 75.1 |
| **weighted combined** | **80.9** |

#### Decode per concurrency level

`MAX_NUM_SEQS=3` is the admission cap. Measured with BetterBench's concurrency sweep, same
production config, 300 W:

| concurrent streams | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-stream decode t/s (median) |
|--:|--:|--:|--:|--:|--:|
| 1 | 48/48 | 66.6 | ~106 ms | ~138 ms | 82.9 |
| 2 | 48/48 | 121.5 | ~163 ms | ~226 ms | 80.3 |
| 3 | 48/48 | 161.3 | ~170 ms | ~221 ms | 71.4 |

Aggregate throughput scales **~2.4x** going from 1 to 3 concurrent streams, for a real ~14%
per-stream decode cost at the full admission cap — near-linear, not a cliff. At the full
`MAX_NUM_SEQS=3` admission cap, the card sustains **161.3 t/s** combined decode across all three
concurrent streams. TTFT p50 grows through concurrency 2 then roughly holds at 3; the p99 tail
stays in the same band through 2 and 3, no cliff there either.


## Quickstart: from scratch

Expect this whole thing to take **30-60 minutes**, mostly download time
(~60 GB across both models) plus ~15 minutes for the FP8-MTP conversion step.

```bash
set -euo pipefail

# 0. checkpoint root -- one tree, both models as siblings under it
MODELS=/path/to/models
mkdir -p "$MODELS"

# 1. clone + build the image
git clone https://github.com/malicz/vllm-gfx1201-launchers.git vllm-gfx1201-launchers
cd vllm-gfx1201-launchers
docker build -t local-radiance-mxfp4:0.9.3 -f Dockerfile .

# 2. Qwen3.8-27B MXFP4 base checkpoint
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --local-dir "$MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4"

# 3. fp8_mtp.py -- MTP head to FP8 (standalone, host-side, needs torch: pip install torch
#    --index-url https://download.pytorch.org/whl/cpu). ~15 min, ~19 GB.
curl -fsSL "https://codeberg.org/ggz14/radiance-vllm-mxfp4/raw/commit/dba9defefb2de7f914fe9cb45ffdf49989c923d6/fp8_mtp.py" \
  -o /tmp/fp8_mtp.py
python3 /tmp/fp8_mtp.py "$MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4" "$MODELS/Qwen3.8-27B-MXFP4-mtpfp8"
# rel-err rows should read ~0.02-0.03; ~0.116 means that layer wasn't converted

# 4. per-token variant (what the launcher serves) -- symlink farm + config.json rewrite
SRC="$MODELS/Qwen3.8-27B-MXFP4-mtpfp8"
DST="$MODELS/Qwen3.8-27B-MXFP4-mtpfp8-pertoken"
mkdir -p "$DST" && cd "$DST"
for f in model.safetensors tokenizer.json tokenizer_config.json merges.txt \
         generation_config.json chat_template.jinja preprocessor_config.json processor_config.json; do
  [ -e "$SRC/$f" ] && ln -sf "../$(basename "$SRC")/$f" .
done
python3 - "$SRC/config.json" "$DST/config.json" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
n = 0
for name, spec in cfg["quantization_config"]["layer_quant_config"].items():
    it = spec.get("input_tensors")
    if not name.startswith("mtp") or not it:
        continue
    it["qscheme"]      = "per_channel"
    it["ch_axis"]      = 0
    it["observer_cls"] = "PerChannelMinMaxObserver"
    n += 1
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(f"rewrote {n} mtp input_tensors to per_channel")  # expect 8
EOF
cd - >/dev/null

# 5. DFlash2 drafter
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir "$MODELS/Qwen3.8-27B-DFlash2-FP8"

# 6. Ornith-1.5-35B-A3B
hf download SergiioB/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16 \
  --local-dir "$MODELS/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16"

# 7. run either model
chmod +x startup-qwen3.8-27b-vllm.sh startup-ornith-1.5-35b-vllm.sh
MODELS_DIR="$MODELS" ./startup-qwen3.8-27b-vllm.sh
# or:
MODELS_DIR="$MODELS" ./startup-ornith-1.5-35b-vllm.sh

docker logs -f r9700-qwen3.8-mxfp4   # boot log; or r9700-ornith-1.5
```

## The additions on top of the two upstreams

### 1. ggz14's MXFP4 runtime delta (fetched, not shipped here)

Only the *runtime* half of that repo applies — its *build-time* patches
(`patch_gfx1201`, `patch_r4d`, `patch_dflash2`, `patch_dflash_base`, `patch_skinny_gemm`,
`patch_gdn_metadata`) target a vanilla vLLM 0.27.1 tree; the published radiance base already
carries their equivalents, so re-applying them fails the drift check or double-applies. Runtime
delta actually used (see the Dockerfile referenced in "Building the image" below for the exact
`COPY`/apply order):

| file | purpose |
|---|---|
| `patch_quark_mxfp4.py` | registers the W4A8 kernel into `_POSSIBLE_MXFP4_KERNELS`; without it MXFP4 silently runs emulated in high precision |
| `patch_dflash_mxfp4_kv.py` | generalizes DFlash2's fused-KV precompute from "fp8 or dense" to "dense or anything else" |
| `patch_topk_triton_rows.py` | fixes a >1900µs top-k sort stall on gfx1201 under speculative decode; bit-identical output |
| `patch_ar_maxbytes.py` | restores `RADIANCE_AR_MAX_KB`; inert at TP=1, kept for fidelity |
| `patch_dflash_calib.py` | activation-statistics hooks for drafter quantization; dark unless `RADIANCE_DFLASH_CALIB` set |
| `patch_rmsquant_fusion.py` | rms+quant fusion op; dark unless `RADIANCE_RMS_QUANT_FUSION=1` — measured within-noise at lower context, and breaks the KV boot margin at 163840 ctx |
| `patch_qwen3_thinkoff.py` | non-fatal fix for `content=null` on thinking-off requests |
| `radiance_mxfp4.py` / `radiance_gdn.py` / `radiance_rmsquant.py` | the runtime modules the patches above wire into |
| `mxfp4-configs/` | aiter GEMM tile configs, pinned to `matrix_instr_nonkdim=16` (aiter ships gfx950/gfx1250 tables only; gfx1250's bands ask for 32, which gfx1201's WMMA 16×16×16 can't lower) |
| `radiance_mxfp4_fp8.hip` | the fp8-WMMA W4A8 GEMM kernel, compiled at build time with `hipcc --offload-arch=gfx1201` |

`RADIANCE_SKINNY_GEMM` is set to `all` (not the narrower `1`) in the launcher, matching
[`zzpanic/qwen3.6-vllm-gfx1201-launchers`](https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers)'s
own default — both settings measure the same on this backend within noise, so this is a
consistency choice, not a performance one.

### 2. `fp8_mtp.py` (also from ggz14's repo — checkpoint prep, not a runtime patch)

AMD's published `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` ships its MTP head in bf16 but doesn't
`exclude` it, so vLLM's quark config falls through to MXFP4 for `mtp.*` and dies loading a
full-width bf16 tensor into a half-width packed buffer. `fp8_mtp.py` rewrites just those eight
MTP projections to `float8_e4m3fn` (FP8 was chosen over re-quantizing to MXFP4 after both plain
RTN and AWQ calibration measurably hurt drafter acceptance — RTN cost acceptance 2.5 → 2.21, and
AWQ calibration didn't rescue it either). CPU-only, no GPU needed. Produces the checkpoint the
Qwen launcher script actually serves.

## Speculative drafter

The launcher ships `tcclaviger/Qwen3.8-27B-DFlash2-FP8` as the DFlash2 drafter, with its KV cache
also in `fp8` to match. It measures a real, healthy, non-zero acceptance rate in production
(roughly 20-80% depending on prompt category).

**Why this matters more than the drafter choice itself**: DFlash2 drafters hook a fixed set of
the *target's* internal hidden-state layers (this drafter's own
`target_layer_ids: [5, 19, 33, 47, 61]`), calibrated against one exact target checkpoint's
activation distribution. **"Same base model" is not automatically "same drafter
compatibility"** — a drafter qualified against one checkpoint is not presumptively safe against
even a lightly-modified derivative of it, because the hook layers see a shifted activation
distribution the drafter was never calibrated for. The origin release name is not a
compatibility contract; the target checkpoint's own exact modification history is what actually
matters.


## KV cache: scale calibration and sizing

Two separate, unrelated axes live under "KV cache" — this launcher addresses both.

### Scale calibration (accuracy — what's actually stored)

`--kv-cache-dtype fp8` alone is not enough for a correct fp8 KV cache. fp8 needs a real
**k_scale/v_scale** per attention layer to map the target's native activation range into fp8's
narrow dynamic range; without one, vLLM silently falls back to `scale=1.0` and logs `Using KV
cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling
factors are properly set in the checkpoint.` on every boot — easy to miss in a long startup log,
and a real accuracy risk, not just a missed optimization. **This checkpoint carries real
calibrated k_scale/v_scale**, and that warning is absent from its boot log — check your own boot
log for its absence before trusting fp8 KV in production, especially if you swap checkpoints.

Producing real scales needs a calibration pass against real activation data — either
[llm-compressor](https://github.com/vllm-project/llm-compressor)'s `QuantizationModifier(scheme=None,
kv_cache_scheme=...)` recipe (works when the checkpoint has separate `k_proj`/`v_proj` modules)
or [AMD Quark](https://github.com/amd/Quark)'s `quantize_quark.py --kv_cache_dtype fp8` (needed
for a fused `qkv_proj`, and the only path that also supports calibrating against one checkpoint
and grafting the resulting scales onto a different, already-quantized one — what we actually
used here). Two real gotchas worth knowing if you do this yourself:

- **Quark's own naming for the scale tensors is not what vLLM expects.** It exports
  `self_attn.{k,v}_proj.output_scale`, not `k_scale`/`v_scale` — and a naive string-rename onto
  that same prefix lands at the *wrong* parameter path entirely if the checkpoint has any
  wrapper nesting (e.g. a `language_model.` prefix on a VL-wrapped checkpoint). The only safe
  approach is deriving the real output path from the *target* checkpoint's own tensor tree (its
  real `k_proj`/`v_proj` weight names), never from a transform of Quark's own naming. A green
  "tensors present" check passed here with the wrong path once — only a real boot with a
  completion request, checking that the `scale=1.0` warning is actually gone, caught it.
- **Not every layer necessarily gets a scale.** One attention layer's k_scale/v_scale can end up
  missing from a calibration pass for reasons not fully root-caused yet — check your own boot log
  for the warning line even after "successful" calibration; its presence for even one layer means
  that layer is still running uncalibrated.

### Sizing the KV pool once scales are right (capacity — how much fits)

Separately from accuracy: vLLM's own `--gpu-memory-utilization`-driven profiling under-reports
how much KV cache actually fits, because it sizes against a *profiling run's* transient
activation peak — a peak steady-state serving never actually needs at the same time as a full
cache. The only honest way to find the real ceiling is to push an explicit `--kv-cache-memory`
pin until boot stops surviving, and back off for margin.

**`kv-memory-calibrate.sh`** (in this repo's root, a light port for single gpu of Brian's
[**ggz14**](https://codeberg.org/ggz14) calibrate-kv.sh) does exactly that — a boot-and-push search
against `startup-qwen3.8-27b-vllm.sh` at your shape (`MAXSEQS`/`CHUNK`/`MAXLEN`), verified by a
real prefill+decode probe at each step, not just a health check (the cache is allocated *before*
cudagraph capture, so an over-committed pin can pass `/health` and still die at capture).

```bash
./kv-memory-calibrate.sh                # ~15-20 min, needs the GPU to itself
# result: KV_MEM=<bytes> printed and saved to ~/.cache/radiance-mxfp4/kv-profiles.local.tsv
KV_MEM=<bytes> ./startup-qwen3.8-27b-vllm.sh
```

At `MAX_NUM_SEQS=3`, `--max-model-len 220000`, `--gpu-memory-utilization 0.98`, this measures
`GPU KV cache size: 226,790 tokens` — a 1.03x margin, reproduced identically across 3 separate
clean boots (not just a lucky single run). That pin (already the default in
`startup-qwen3.8-27b-vllm.sh`'s `KV_MEM`) is specific to this exact
`MAXSEQS`/`CHUNK`/`MAXLEN`/`GPU_MEM_UTIL` shape — re-run the calibration yourself if you change
any of those, or the checkpoint.

**Note on using `kv-memory-calibrate.sh` itself at this shape**: its pass-1 gate (serving once
with `KV_MEM` unset, to get a baseline before searching upward) FAILS outright at
`MAX_MODEL_LEN=220000`/`GPU_MEM_UTIL=0.98` — the unpinned auto-profile only frees ~6.99 GiB,
enough for an estimated ~179,712 tokens, well under both this pin and the checkpoint's own
prior 190,000-token config. The 226,790-token result above was found by hand instead, pushing
the `KV_MEM` pin directly against `startup-qwen3.8-27b-vllm.sh` the same way the script's own
pass-2 search would, just without requiring pass-1 to succeed first. This is expected, not a
bug in the script: a manually forced pin can always exceed what an unpinned auto-profile
computes, because the profile run's own transient activation peak is never actually needed at
the same instant as a full KV cache during real serving (see the script's own header).


## Model downloads & checkpoint prep

Steps to download the models and do an mtp conversion.

```bash
MODELS=/path/to/models   # wherever you point MODELS_DIR at

# --- Qwen3.8-27B-MXFP4 target ---
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --local-dir $MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4

curl -fsSL "https://codeberg.org/ggz14/radiance-vllm-mxfp4/raw/commit/dba9defefb2de7f914fe9cb45ffdf49989c923d6/fp8_mtp.py" \
  -o /tmp/fp8_mtp.py
python3 /tmp/fp8_mtp.py $MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4 $MODELS/Qwen3.8-27B-MXFP4-mtpfp8
# then generate a per-token variant if desired (config-only, ~64KB of symlinks + a rewritten
# config.json moving the MTP layers' input quantization from per-tensor to per-channel).

# --- DFlash2 drafter ---
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 --local-dir $MODELS/Qwen3.8-27B-DFlash2-FP8

# --- Ornith-1.5-35B-A3B ---
hf download SergiioB/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16 \
  --local-dir $MODELS/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16
# NOTE: this checkpoint's own bundled MTP head is community-reported as near-random-init quality
# (~13-22% acceptance) -- the launcher script doesn't use it at all (see its own header for the
# bake-off numbers against two independently-retrained replacement heads). No config.json fixes
# needed for the no-MTP path this launcher runs. Verify every downloaded shard's size against the
# HF API's authoritative blob size -- a truncated mid-transfer download can look complete in a
# casual `ls`.
# Also generate a tuned MoE kernel config before serving -- see the launcher script's own header
# for the exact benchmark_moe.py command; this checkpoint's MoE shape ships with no tuned config
# bundled in vLLM, and both decode and prefill measurably suffer without one.
```

## Building the image

The image (`local-radiance-mxfp4:0.9.3`) is built by the `Dockerfile` in this directory — no
external pieces to assemble yourself.

```bash
git clone https://github.com/malicz/vllm-gfx1201-launchers.git vllm-gfx1201-launchers && cd vllm-gfx1201-launchers
docker build -t local-radiance-mxfp4:0.9.3 -f Dockerfile .
```

## Running

```bash
chmod +x startup-qwen3.8-27b-vllm.sh startup-ornith-1.5-35b-vllm.sh

./startup-qwen3.8-27b-vllm.sh
# or
./startup-ornith-1.5-35b-vllm.sh
```

Both are env-var configurable (image tag, container name, model/cache dirs, port, context
length, GPU memory fraction) — see each script's header for the full list and defaults. Neither
script builds the image; run the `docker build` above first.

### Environment variables

| var | default | meaning |
|---|---|---|
| `IMAGE` | `local-radiance-mxfp4:0.9.3` | image tag to run |
| `CONTAINER_NAME` | `r9700-qwen3.8-mxfp4` / `r9700-ornith-1.5` | docker container name |
| `MODELS_DIR` | `/models` | host dir mounted at `/models` — must contain the WHOLE tree, not just one model's subdir (see the Qwen script's MOUNT REQUIREMENT comment) |
| `CACHE_DIR` | `./vllm-cache-mxfp4` | Triton/inductor/aiter compile cache — persist this or every boot re-autotunes |
| `PORT` | `9300` | host port, also passed to `--port` |
| `MAX_MODEL_LEN` | `220000` (Qwen) / `262144` (Ornith) | KV allocation ceiling — see the boot-log checklist below before raising |
| `GPU_MEM_UTIL` | `0.98` (Qwen) / `0.97` (Ornith) | vLLM's `--gpu-memory-utilization` |
| `MAX_NUM_SEQS` | `3` (Qwen) / `4` (Ornith) | concurrent-sequence cap — see the concurrency section below for what this actually buys you at full context |
| `NUM_SPEC_TOKENS` | `7` (Qwen only) | DFlash2 draft depth — the drafter was trained at this exact value |
| `REASONING_EFFORT` | `medium` (Qwen only) | Qwen3.8's graded thinking-effort default |
| `KV_MEM` | `9126805504` (Qwen only, ~8.5 GiB) | explicit `--kv-cache-memory` pin, measured by hand for `MAXSEQS=3`/`CHUNK=2560`/`MAXLEN=220000`/`GPU_MEM_UTIL=0.98` — see the KV cache section above; re-measure if you change any of those |
| `TUNED_CONFIG_DIR` | `./moe-tuned-configs` (Ornith only) | where a tuned MoE kernel config (see script header) is picked up from, if present |
| `POWER_CAP_W` | unset | optional GPU power cap in microwatts (e.g. `240000000` = 240 W) |

### Concurrency (Ornith)

`MAX_NUM_SEQS=4` is the admission cap, but real *parallel* concurrency depends on how much
context each request actually uses relative to the KV pool (measured ~320K tokens at full
262,144 context / 0.97 utilization):

- Short/moderate prompts: genuinely 4 concurrent requests, no penalty — measured 4× ~20K-token
  prompts all completing together in the same ~16.6s.
- As combined context approaches the pool size, the scheduler gracefully **queues/staggers**
  rather than erroring — measured 4× ~85K-token prompts (340K combined, over budget) all
  succeeding but spread 75-101s instead of finishing together.
- Two genuinely near-max-context requests (240K tokens each) will serialize almost completely
  (measured 124.5s / 249s — the second visibly waits for the first) — you get correctness, not
  concurrency, at that extreme. `floor(pool / MAX_MODEL_LEN)` is the real number of truly
  parallel full-length requests this card supports, and at full 262,144 context that's 1.

## Boot-log checklist

- `Using RadianceMxfp4W4A8LinearKernel` and `304/304 on our kernel, 0 FORCED ONTO AITER` —
  confirms the native MXFP4 kernel is actually engaged, not silently emulated (Qwen only).
- `GPU KV cache size: N tokens` — the real, measured ceiling for this boot; it swings between
  boots (Qwen measured 7.16–8.12 GiB free KV across identical boots, historically at the old
  163,840-token config — the same real boot-to-boot variance applies at other lengths, expect
  it in general, though the current 220,000-token/9126805504-byte pin reproduced identically
  across 3 separate clean boots when it was measured) — if a boot fails where the last one
  succeeded at the same `MAX_MODEL_LEN`, this is why; lower it or back off the `KV_MEM` pin one
  step (e.g. back to the prior 190,000/8232441724 config, itself already verified at real
  production concurrency).
- `creating MTP draft context against the target model` (only relevant if you ever re-enable
  Ornith's MTP) confirms the checkpoint's MTP tensors were actually found and loaded.
- **Cold cache first boot**: an empty `CACHE_DIR` at `MAX_MODEL_LEN` above ~131072 can fail with
  `estimated maximum model length is ~139776` — cold Triton/inductor profiling uses ~1 GiB more
  peak memory than warm. Either warm the cache once at a lower `MAX_MODEL_LEN` first, or retry.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'` | you swapped in a GPTQ-style/compressed-tensors drafter with no dense `.weight` tensor (only `qweight`/`qzeros`/`scales`) | this repo doesn't ship a patch for this — the FP8 drafter it serves carries a real dense `.weight` tensor and doesn't hit it; pick a drafter checkpoint that carries a real `.weight` tensor (fp8/dense formats do), or write an equivalent fix yourself |
| `failed to tokenize reasoning strings: reasoning_start_str='', reasoning_end_str=''` (Qwen) | only the per-token model dir mounted, not the whole `models/` tree — its files are symlinks into a sibling dir | mount the entire `MODELS_DIR`, not a per-model subdirectory |
| `AssertionError: In Mamba cache align mode, block_size (N) must be <= max_num_batched_tokens` (Ornith) | `--max-num-batched-tokens` set below this checkpoint's own Mamba alignment block size (measured 1088 tokens for this specific checkpoint at this max-model-len — varies by checkpoint/context) | keep `--max-num-batched-tokens` at `8192` as shipped in the script, or check your own boot log's `Setting attention block size to N tokens` line before lowering it |
| `WARNING max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` silently truncating prefill even with a higher `--max-num-batched-tokens` set | only fires when `--speculative-config` is active — vLLM caps prefill chunks to 2048 regardless of the flag in that case | this launcher runs no speculative decoding by design, so it doesn't apply; if you re-enable MTP, expect this cap unless it's addressed separately |
| decode much slower than the numbers in this README despite an otherwise-identical config | `VLLM_TUNED_CONFIG_FOLDER`/`TUNED_CONFIG_DIR` not populated — check the boot log for `Using default MoE config. Performance might be sub-optimal!` | run the `benchmark_moe.py` tuning step in the script's header once for your GPU |
| `content: null` with a fully-populated `reasoning` field | a known upstream vLLM bug class in token-ID-based `</think>` boundary detection ([vllm-project/vllm#15758](https://github.com/vllm-project/vllm/issues/15758)) — architecture-agnostic, not fixable by patching this model | drop `--reasoning-parser` entirely; content lands raw with inline `<think>` tags instead |
| MXFP4 running much slower than expected, no `RadianceMxfp4W4A8LinearKernel` in the log | `RADIANCE_MXFP4`/`RADIANCE_MXFP4_W4A8` not set, or `patch_quark_mxfp4.py` missing from the image | confirm both env vars are `1` and rebuild if the log never shows the kernel line |
| container exits immediately, log mentions `/opt/templates/froggeric-qwen.jinja` not found | build step 5 (fetching the chat template) was skipped | add it to the Dockerfile, or drop `--chat-template ...` from the script's `VLLM_ARGS` to use the checkpoint's own template instead |
