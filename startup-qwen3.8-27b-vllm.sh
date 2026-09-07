#!/usr/bin/env bash
# startup-qwen3.8-27b-vllm.sh — standalone launcher for Qwen3.8-27B: native MXFP4 (W4A8) +
# DFlash2-FP8 n=7 speculative decoding, calibrated fp8 KV cache, 220k context, on one gfx1201
# R9700.
#
# Runs directly with `docker run`, no router/orchestration layer in front — useful for isolated
# benchmarking, a second host, or debugging a boot failure without extra machinery in the way.
#
# See ../README.md for the full setup story (image build, checkpoint prep, why each flag exists)
# — this script only wires it up, it doesn't explain it.
set -euo pipefail

# ---------------------------------------------------------------------- configuration (env vars)
IMAGE="${IMAGE:-local-radiance-mxfp4:0.9.3}"       # image tag built by ../Dockerfile
CONTAINER_NAME="${CONTAINER_NAME:-r9700-qwen3.8-mxfp4}"  # model-specific, avoids clashing with other launchers
MODELS_DIR="${MODELS_DIR:-/models}"                # host dir holding checkpoints, mounted into the container
CACHE_DIR="${CACHE_DIR:-./vllm-cache-mxfp4}"       # host dir for Triton/inductor/aiter compile caches, persists across restarts
PORT="${PORT:-9300}"                               # vLLM's HTTP port
MAX_MODEL_LEN="${MAX_MODEL_LEN:-220000}"           # ceiling verified against the KV_MEM pin below (226,790-token
                                                    # real KV pool, 1.03x margin, reproduced across 3 separate clean
                                                    # boots); re-run kv-memory-calibrate.sh if you change MAXSEQS,
                                                    # CHUNK, or the checkpoint. NOTE: at this shape (MAXSEQS=3,
                                                    # util=0.98) kv-memory-calibrate.sh's own pass-1 gate (letting
                                                    # vLLM auto-profile with KV_MEM unset) FAILS outright before it
                                                    # can even start its search -- the unpinned auto-profile only
                                                    # frees ~6.99 GiB, good for an estimated ~179,712 tokens, well
                                                    # under this pin. This value was found by pushing the pin by
                                                    # hand instead (the same technique the script automates, just
                                                    # without its stricter starting-point requirement) -- see
                                                    # kv-memory-calibrate.sh's own header for why the profiled
                                                    # figure always underestimates real capacity.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.98}"               # fraction of VRAM vLLM may claim — high enough to leave room
                                                    # for the KV_MEM pin, low enough to avoid OOM on boot
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"                  # admission cap — cudagraph_capture_sizes below already covers
                                                    # batch=4, so no graph-capture crash risk at this value
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"            # speculative depth — matches the drafter's own training depth (block_size 8)
REASONING_EFFORT="${REASONING_EFFORT:-medium}"     # default thinking-budget hint for the chat template — balances latency vs quality
KV_MEM="${KV_MEM:-9126805504}"                     # ~8.5 GiB — pin measured by hand for this exact (MAXSEQS=3,
                                                    # CHUNK=2560, MAXLEN=220000, util=0.98) shape; re-measure
                                                    # if any of those change, the pin is shape-specific
POWER_CAP_W="${POWER_CAP_W:-}"                     # optional GPU power cap in microwatts (e.g. 240000000), unset = no cap

mkdir -p "$CACHE_DIR"

if [ -n "$POWER_CAP_W" ]; then
  HWMON=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1)
  [ -n "$HWMON" ] && echo "$POWER_CAP_W" > "$HWMON/power1_cap" 2>/dev/null || true
fi

# ---------------------------------------------------------------------- container setup
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

RUN_ARGS=(
  -d --name "$CONTAINER_NAME"
  --device=/dev/kfd --device=/dev/dri
  --group-add "$(getent group render | cut -d: -f3)"
  --group-add "$(getent group video | cut -d: -f3)"
  --ipc=host --shm-size=16g
  -e HIP_VISIBLE_DEVICES=0                        # single-GPU host, always device 0
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces VRAM fragmentation under long-running KV allocation churn
  -e GPU_MAX_HW_QUEUES=1                          # fewer HW queues measured more stable on gfx1201 than the driver default
  -e HSA_ENABLE_INTERRUPT=1                       # interrupt-driven completion signaling, lower latency than polling
  -e HSA_ENABLE_MWAITX=1                          # lets the ROCm runtime use MWAITX for wait-loops, cheaper than a spin
  -e VLLM_USE_V2_MODEL_RUNNER=1                   # required runner version for DFlash2 speculative decoding support
  -e VLLM_ROCM_USE_AITER=1                        # enables the aiter kernel library as the ROCm backend
  # CHANGED 1 -> 0, 2026-09-06: superseded by --attention-backend R4D below, see the adoption
  # note further down for the measured numbers and why this differs from the 2026-08-27 R4D
  # rejection (this repo's own AGENTS.md rule 10 / production's swap-mxfp4/README.md).
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0      # superseded by --attention-backend R4D below
  -e VLLM_ROCM_USE_AITER_MHA=0                    # unified attention above supersedes aiter's separate MHA path
  -e VLLM_ROCM_USE_AITER_MLA=0                    # this checkpoint isn't MLA, nothing to enable
  -e VLLM_ROCM_USE_AITER_MOE=0                    # not an MoE checkpoint, nothing to enable
  -e VLLM_ROCM_USE_AITER_LINEAR=0                 # radiance's own MXFP4 GEMM kernels below replace aiter's linear path
  -e VLLM_ROCM_USE_AITER_FP8BMM=0                 # radiance's kernels handle the fp8 batched matmuls instead
  -e VLLM_ROCM_USE_AITER_FP4BMM=0                 # radiance's kernels handle the mxfp4 batched matmuls instead
  -e VLLM_ROCM_USE_AITER_RMSNORM=0                # RADIANCE_FUSE_RMS_QUANT below replaces aiter's RMSNorm kernel
  -e RADIANCE_USE_R4D=1                           # enables libr4d, the DFlash2 speculative-decoding runtime
  -e RADIANCE_R4D_REPORT=1                        # logs draft-acceptance-rate stats — needed to verify drafter health
  -e RADIANCE_PRESHUFFLE=1                        # pre-shuffles MXFP4 weight layout at load time for faster GEMM
  -e RADIANCE_FUSE_RMS_QUANT=1                    # fuses RMSNorm with the following quantize step, fewer kernel launches
  -e RADIANCE_SKINNY_GEMM=all                     # "1" measures the same within noise on this backend; "all" matches
                                                    # zzpanic/qwen3.6-vllm-gfx1201-launchers' own default
  -e RADIANCE_TOPK_TRITON_MIN_ROWS=1              # always route top-k to the Triton kernel, correctness patch requirement
  -e RADIANCE_FAST_DRAFT=0                        # this drafter shape doesn't benefit from the fast-draft path
  -e RADIANCE_DYNAMIC_DRAFT=0                     # fixed NUM_SPEC_TOKENS depth used instead of dynamic draft-length tuning
  -e RADIANCE_MXFP4=1                             # enables the MXFP4 quantized weight path
  -e RADIANCE_MXFP4_W4A8=1                        # this checkpoint is W4A8, not the plain W4A4 MXFP4 variant
  -e RADIANCE_MXFP4_W4A8_MIN_M=0                  # correctness, not tuning — aiter's W4A4 fallback returns WRONG
                                                    # results for the GDN N=5120,K=3072 projection; 0 = never fall back
  -e RADIANCE_MXFP4_TN4_MIN_M=2048                # threshold above which the TN4 GEMM tiling kicks in, radiance default
  -e RADIANCE_MXFP4_DECODE_MAX_M=64                # DFlash2: M = batch x spec-tokens, up to 3 x 8 = 24, headroom to 64
  -e RADIANCE_MXFP4_SANITIZE=0                    # skips extra NaN/inf checks on the quantized path once verified stable
  -e RADIANCE_RMS_QUANT_FUSION=0                   # breaks the KV boot margin if set to 1
  -e RADIANCE_MXFP4_HOIST_QUANT=0                  # same — breaks the KV boot margin together with the above
  # ADDED 2026-09-03, upstream sync to a5d68ba: real 300W BetterBench A/B on the internal
  # production fleet (same target checkpoint, same drafter) measured WPERM+DECODE_NT as a genuine
  # win — +2.7% weighted decode, +3.9% prefill@16K, +1.6% concurrency@1 vs. the a5d68ba baseline,
  # every metric moved the same direction, no crash across 48 concurrent requests. GDN_NORM_QUANT
  # + STRIDED_GATES measured NEUTRAL on the same run (kept on anyway, upstream's own verdict
  # matched, no downside found). GDN_EMPTY_OUT is deliberately NOT set here — see this repo's own
  # Dockerfile comment: it needs a rebuilt libr4d (rx5) to safely zero cudagraph pad rows, which
  # this image's build does not do.
  -e RADIANCE_MXFP4_WPERM=1                        # fragment-order weight layout, WMMA-aligned reads instead of strided (measured win)
  -e RADIANCE_MXFP4_DECODE_NT=1                     # streaming (non-temporal) weight loads in the decode kernel (measured win)
  -e RADIANCE_GDN_NORM_QUANT=1                      # fused gated-norm + fp8 quant, one HIP kernel replacing RMSNormGated + traced quant on each GDN layer (neutral, kept on)
  -e RADIANCE_GDN_STRIDED_GATES=1                   # skip redundant .contiguous() copies on GDN gate tensors (neutral, kept on)
  # ADOPTED 2026-09-06, following up on this repo's own comparison against
  # zzpanic/qwen3.6-vllm-gfx1201-launchers the same day. Plain R4D (stock f16 attention legs) was
  # tried and REJECTED on 2026-08-27: +4.4% prefill but ~2.5x KV cost/token, capping context to
  # ~16K on a 32 GiB card. The fp8 legs are a DIFFERENT trade -- independently re-measured on the
  # PRODUCTION fleet's qwen3.8-27b-mxfp4-uncensored-orcarouter entry (fp8 KV, same base image,
  # same DFlash2 FP8 drafter, standalone containers, production stopped/restored around each run):
  #   KV pool:      151,318 tok / 6.87 GiB (AITER)  ->  153,560 tok / 6.9 GiB (R4D+fp8 legs), +1.5%
  #   Prefill @16k: 2,598 -> 2,770 t/s (+6.6%)   @32k: 2,364 -> 2,635 (+11.5%)   @64k: 1,967 -> 2,433 (+23.7%)
  #   Decode:       82.3 -> 83.0 t/s (+0.8%, noise)
  # Reproduces zzpanic's own claimed shape ("+1.7/5.6/24.9% at 4k/16k/64k") independently. The
  # 2026-08-27 memory blocker was specific to the f16 attention legs, not R4D itself -- the fp8
  # legs bring KV memory to near-parity with AITER while keeping the prefill win.
  # NOT independently re-benchmarked on THIS checkpoint (Qwen3.8-27B-MXFP4-mtpfp8-pertoken) --
  # carried over from the uncensored-orcarouter measurement on the strength of matching
  # architecture/kernels/drafter/fp8-KV. Re-verify with BetterBench on this exact checkpoint
  # before trusting the numbers above to transfer exactly.
  # 3 = both O_QK8 and O_PV8 fp8 legs (see production's r4d_radiance_extras.patch); fp8 KV only,
  # matches --kv-cache-dtype fp8 below. Also re-confirmed the same day that RADIANCE_FP8_STREAM
  # correctly stays unset here: radiance_arnq.py's own install() unconditionally skips at TP=1
  # (line 262, "tp=1, skipping"), a hard source-level guard, not a stale assumption.
  -e R4D_ATTN_FP8=3                                 # 8-bit prefill attention legs for the R4D backend below
  -e VLLM_CACHE_ROOT=/cache/vllm                  # vLLM's own compile/config cache, mapped to the persistent CACHE_DIR
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor       # torch.compile cache, mapped to the persistent CACHE_DIR
  -e TRITON_CACHE_DIR=/cache/triton               # Triton JIT cache, mapped to the persistent CACHE_DIR
  -e AITER_JIT_DIR=/cache/aiter                  # aiter's kernel build cache, mapped to the persistent CACHE_DIR
  -e TRITON_CACHE_AUTOTUNING=1                    # persists autotuning results across restarts, not just compiled kernels
  -v "$MODELS_DIR:/models"
  -v "$CACHE_DIR:/cache"
  -p "${PORT}:${PORT}"
  "$IMAGE"
)

# ---------------------------------------------------------------------- vLLM args
# MOUNT REQUIREMENT: this checkpoint (Qwen3.8-27B-MXFP4-mtpfp8-pertoken) ships everything but
# config.json as a symlink into the SIBLING dir Qwen3.8-27B-MXFP4-mtpfp8/ — the whole models/
# tree must be mounted (done above), not just this one model subdirectory, or the tokenizer
# silently fails to resolve <think>/</think> and ReasoningConfig dies at startup.
#
# --kv-cache-dtype fp8 on the TARGET is backed by REAL calibrated k_scale/v_scale in this
# checkpoint -- not vLLM's uncalibrated scale=1.0 fallback. See README's "KV cache: scale
# calibration and sizing" section for what that means and how to check it (grep your own boot
# log for "Using KV cache scaling factor 1.0" -- it should be ABSENT; if you ever swap in a
# checkpoint that doesn't carry real scales, that line comes back and fp8 KV silently degrades).
#
# The drafter below (tcclaviger/Qwen3.8-27B-DFlash2-FP8) measures a real, healthy, non-zero
# acceptance rate against this target. ALWAYS verify the real acceptance rate on your own
# checkpoint before trusting any drafter swap (grep the boot/runtime log for "Avg Draft
# acceptance rate"): DFlash2 drafters hook a fixed set of the target's internal hidden-state
# layers, calibrated against one exact target's activations, so "same base model" is not
# automatically "same drafter compatibility" -- a 0%-acceptance drafter still produces coherent
# output, just slower than no speculation at all, so this failure is invisible without checking
# that exact log line.
VLLM_ARGS=(
  /models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken         # the target checkpoint (see MOUNT REQUIREMENT above)
  --served-model-name qwen3.8-27b-mxfp4 qwen-mxfp4 qwen3.8-mxfp4  # aliases for API clients/routers
  --kv-cache-dtype fp8                              # halves KV memory vs bf16; safe here because this
                                                     # checkpoint carries real calibrated k_scale/v_scale
  --tensor-parallel-size 1                          # single GPU, no sharding
  --gpu-memory-utilization "$GPU_MEM_UTIL"          # see GPU_MEM_UTIL above
  --override-generation-config "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20}"  # this checkpoint's
                                                     # own recommended sampling defaults
  --default-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}"  # see REASONING_EFFORT above
  --max-model-len "$MAX_MODEL_LEN"                  # see MAX_MODEL_LEN above
  --max-num-seqs "$MAX_NUM_SEQS"                    # see MAX_NUM_SEQS above
  --max-num-batched-tokens 2560                     # prefill chunk size the KV_MEM pin above was calibrated against
  --attention-backend R4D                           # CHANGED from ROCM_AITER_UNIFIED_ATTN, 2026-09-06 -- see
                                                     # the R4D_ATTN_FP8 adoption note above for the measured
                                                     # numbers and why this differs from the 2026-08-27 rejection
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'  # covers up to batch=4 at MAX_NUM_SEQS=3,
                                                     # see MAX_NUM_SEQS above
  --enable-prefix-caching                           # reuses KV for repeated prompt prefixes; unlike Ornith
                                                     # below, this architecture doesn't hit the GDN zero-hit-rate issue
  --mamba-cache-mode align                          # required cache layout for this checkpoint's hybrid attention state
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-FP8\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"fp8\"}"
                                                     # DFlash2 drafter; see the acceptance-rate warning above
  --chat-template /opt/templates/froggeric-qwen.jinja  # baked into the image, fixes upstream template bugs
  --reasoning-parser qwen3                          # splits <think> blocks into the response's reasoning field
  --tool-call-parser qwen3_xml                      # this checkpoint emits tool calls in Qwen3's XML format
  --enable-auto-tool-choice                         # required for --tool-call-parser to actually engage
  --no-async-scheduling                             # `--async-scheduling` fails to start on this stack
  --trust-remote-code                               # checkpoint ships custom modeling code (GDN, MXFP4 wiring)
  --host 0.0.0.0 --port "$PORT"                     # listen on all interfaces, see PORT above
)
[ -n "$KV_MEM" ] && VLLM_ARGS+=(--kv-cache-memory "$KV_MEM")

echo "Starting $CONTAINER_NAME on port $PORT (max-model-len=$MAX_MODEL_LEN, max-num-seqs=$MAX_NUM_SEQS, spec-tokens=$NUM_SPEC_TOKENS, kv-mem=${KV_MEM:-auto})..."
docker run "${RUN_ARGS[@]}" "${VLLM_ARGS[@]}"
echo "Boot log: docker logs -f $CONTAINER_NAME"
echo "COLD START IS ~4+ MINUTES; a first boot against an empty $CACHE_DIR is slower still (Triton/inductor autotuning)."
