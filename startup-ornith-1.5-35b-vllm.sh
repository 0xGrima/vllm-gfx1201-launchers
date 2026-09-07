#!/usr/bin/env bash
# startup-ornith-1.5-35b-vllm.sh — standalone launcher for Ornith-1.5-35B-A3B: MoE (256 experts,
# GDN linear-attention hybrid), plain AutoRound-produced GPTQ W4A16 (group_size 128, symmetric),
# no speculative decoding (this checkpoint's own MTP head isn't good enough to be worth the
# context cost — see README), full native 262144-token context, vision enabled, on one gfx1201
# R9700.
#
# Served from vLLM's native GPTQ WNA16 MoE path — no MXFP4 kernel involved. Runs on the SAME
# `local-radiance-mxfp4` image as the Qwen3.8 launcher (one 32GB card fits one resident model at
# a time).
#
# See ../README.md for the --dtype/--kv-cache-dtype pairing quirk and the MoE kernel tuning step
# below.
set -euo pipefail

# ---------------------------------------------------------------------- configuration (env vars)
IMAGE="${IMAGE:-local-radiance-mxfp4:0.9.3}"       # image tag built by ../Dockerfile, shared with the Qwen3.8 launcher
CONTAINER_NAME="${CONTAINER_NAME:-r9700-ornith-1.5}"  # model-specific, avoids clashing with other launchers
MODELS_DIR="${MODELS_DIR:-/models}"                # host dir holding checkpoints, mounted into the container
CACHE_DIR="${CACHE_DIR:-./vllm-cache-mxfp4}"       # host dir for Triton/inductor/aiter compile caches, persists across restarts
PORT="${PORT:-9300}"                               # vLLM's HTTP port
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # this checkpoint's native max_position_embeddings
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.97}"              # full native context needs the extra headroom over 0.90-0.95
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"                 # see README's concurrency section — 4 is the
                                                   # measured sweet spot; short/moderate prompts run
                                                   # fully parallel, only near-max-context requests
                                                   # (each close to MAX_MODEL_LEN) serialize
POWER_CAP_W="${POWER_CAP_W:-}"                     # optional GPU power cap in microwatts (e.g. 240000000), unset = no cap

mkdir -p "$CACHE_DIR"

if [ -n "$POWER_CAP_W" ]; then
  HWMON=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1)
  [ -n "$HWMON" ] && echo "$POWER_CAP_W" > "$HWMON/power1_cap" 2>/dev/null || true
fi

# ---------------------------------------------------------------------- MoE kernel tuning (recommended)
# This checkpoint's MoE shape (E=256 experts, N=512 intermediate, int4_w4a16) has no tuned Triton
# config bundled with vLLM by default -- every boot without one logs "Using default MoE config.
# Performance might be sub-optimal!" and both decode and prefill measurably suffer. Generate one
# once per GPU model with vLLM's own official tuning script (not bundled in this repo or the
# image -- fetch the version matching your vLLM):
#
#   curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/benchmarks/kernels/benchmark_moe.py" \
#     -o /tmp/benchmark_moe.py
#   docker run --rm --device=/dev/kfd --device=/dev/dri \
#     -v "$MODELS_DIR:/models" -v "$(pwd)/moe-tuned-configs:/tune/out" -v /tmp/benchmark_moe.py:/tune/benchmark_moe.py \
#     "$IMAGE" bash -c "pip install -q ray && python3 /tune/benchmark_moe.py \
#       --model /models/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16 \
#       --dtype int4_w4a16 --tp-size 1 --trust-remote-code --tune \
#       --batch-size 1 2 4 8 16 --save-dir /tune/out"
#
# This is a genuinely long, CPU-JIT-compile-bound run (hours, not minutes -- expect near-zero GPU
# utilization for most of it, that's normal for this tool). The result is picked up automatically
# via VLLM_TUNED_CONFIG_FOLDER below if present; the script still runs correctly without it, just
# slower. Measured on our own R9700: ~78 t/s decode / ~5700-5800 t/s prefill tuned, vs
# proportionally worse untuned (exact untuned numbers not re-measured after the fact).
TUNED_CONFIG_DIR="${TUNED_CONFIG_DIR:-./moe-tuned-configs}"
mkdir -p "$TUNED_CONFIG_DIR"

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
  -e VLLM_ROCM_USE_AITER=1                        # enables the aiter kernel library as the ROCm backend
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1      # matches --attention-backend ROCM_AITER_UNIFIED_ATTN below
  -e VLLM_ROCM_USE_AITER_MHA=0                    # unified attention above supersedes aiter's separate MHA path
  -e VLLM_ROCM_USE_AITER_MLA=0                    # this checkpoint isn't MLA, nothing to enable
  -e VLLM_ROCM_USE_AITER_MOE=0                    # this checkpoint's GPTQ W4A16 MoE runs vLLM's native path, not aiter's
  -e VLLM_ROCM_USE_AITER_LINEAR=0                 # no MXFP4 GEMM here (this is GPTQ W4A16), nothing for aiter linear to accelerate
  -e VLLM_ROCM_USE_AITER_FP8BMM=0                 # this checkpoint runs bfloat16, not fp8
  -e VLLM_ROCM_USE_AITER_FP4BMM=0                 # this checkpoint runs bfloat16, not mxfp4
  -e VLLM_ROCM_USE_AITER_RMSNORM=0                # RADIANCE_FUSE_RMS_QUANT below replaces aiter's RMSNorm kernel
  -e RADIANCE_PRESHUFFLE=1                        # pre-shuffles weight layout at load time for faster GEMM
  -e RADIANCE_ATTN_TUNE=1                         # enables radiance's attention kernel autotuning
  -e RADIANCE_GDN_WMMA=1                          # WMMA-accelerated kernel for this checkpoint's GDN linear-attention layers
  -e RADIANCE_VIT_FLASH=1                         # flash-attention kernel for the vision tower (this checkpoint has vision enabled)
  -e RADIANCE_FUSE_RMS_QUANT=1                    # fuses RMSNorm with the following quantize step, fewer kernel launches
  -e RADIANCE_USE_R4D=1                           # enables libr4d — required by radiance's runtime even with no drafter here
  -e RADIANCE_R4D_REPORT=1                        # logs libr4d runtime stats
  -e VLLM_TUNED_CONFIG_FOLDER=/tune-config         # picks up the MoE kernel tuning results generated below, if present
  -e VLLM_CACHE_ROOT=/cache/vllm                  # vLLM's own compile/config cache, mapped to the persistent CACHE_DIR
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor       # torch.compile cache, mapped to the persistent CACHE_DIR
  -e TRITON_CACHE_DIR=/cache/triton               # Triton JIT cache, mapped to the persistent CACHE_DIR
  -e AITER_JIT_DIR=/cache/aiter                  # aiter's kernel build cache, mapped to the persistent CACHE_DIR
  -e TRITON_CACHE_AUTOTUNING=1                    # persists autotuning results across restarts, not just compiled kernels
  -v "$MODELS_DIR:/models"
  -v "$CACHE_DIR:/cache"
  -v "$TUNED_CONFIG_DIR:/tune-config"
  -p "${PORT}:${PORT}"
  "$IMAGE"
)

# ---------------------------------------------------------------------- vLLM args
# MTP deliberately absent: this checkpoint's own bundled MTP head is near-random-init quality
# (~13-22% acceptance), and no replacement head available for it beats running without a drafter
# at all once you account for its real cost — a drafter roughly halves the usable KV budget, and
# this checkpoint's decode ceiling doesn't need the assist enough to justify giving up that much
# context. See README if you want to trade context back for a small decode edge.
#
# --dtype bfloat16 is REQUIRED alongside --kv-cache-dtype bfloat16, not optional: this checkpoint
# is a genuine mixed F16/BF16/I32 tensor bag (config.json's top-level dtype says float16, its
# text_config says bfloat16) and resolves to fp16 compute by default. Setting kv-cache-dtype to
# bfloat16 WITHOUT also forcing --dtype bfloat16 crashes engine init with `AssertionError: Both
# operands must be same dtype. Got fp16 and bf16` inside a Triton MoE kernel.
VLLM_ARGS=(
  serve /models/Ornith-1.5-35B-A3B-AutoRound-W4A16-sym-G128-MTP-BF16  # the target checkpoint
  --served-model-name ornith-1.5-35b-a3b ornith    # aliases for API clients/routers
  --dtype bfloat16                                  # required alongside --kv-cache-dtype bfloat16, see comment above
  --kv-cache-dtype bfloat16                         # this checkpoint carries no calibrated fp8 KV scales, bf16 is the safe default
  --tensor-parallel-size 1                          # single GPU, no sharding
  --gpu-memory-utilization "$GPU_MEM_UTIL"          # see GPU_MEM_UTIL above
  --max-model-len "$MAX_MODEL_LEN"                  # see MAX_MODEL_LEN above
  --max-num-seqs "$MAX_NUM_SEQS"                    # see MAX_NUM_SEQS above
  --max-num-batched-tokens 8192                     # prefill chunk size — no KV_MEM pin to calibrate against here,
                                                     # this is vLLM's own default for this context length
  --mamba-cache-mode align                          # required cache layout for this checkpoint's GDN linear-attention state
  --attention-backend ROCM_AITER_UNIFIED_ATTN       # matches VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION above
  --reasoning-parser qwen3                          # splits <think> blocks into the response's reasoning field
  --chat-template /opt/templates/froggeric-qwen.jinja  # baked into the image, fixes upstream template bugs
  --tool-call-parser qwen3_xml                      # this checkpoint emits tool calls in Qwen3's XML format
  --enable-auto-tool-choice                         # required for --tool-call-parser to actually engage
  --trust-remote-code                               # checkpoint ships custom modeling code (GDN, MoE wiring)
  --host 0.0.0.0 --port "$PORT"                     # listen on all interfaces, see PORT above
)
# NOTE: no --enable-prefix-caching -- tested (boots fine, no correctness issue) but measured a
# genuine 0% cache hit rate even across identical back-to-back prompts on this GDN-hybrid
# architecture. Root cause: vLLM's HybridKVCacheCoordinator vetoes an ENTIRE request's cache reuse
# if the Mamba-state KV group misses, even when the attention groups would have hit
# (vllm-project/vllm#45238) -- the documented fix (VLLM_PREFIX_CACHE_RETENTION_INTERVAL) exists in
# vLLM 0.27.1 but did not resolve it here either. Not worth the "experimental" support risk vLLM's
# own boot log flags Mamba-layer prefix caching with, for a feature that measurably does nothing.

echo "Starting $CONTAINER_NAME on port $PORT (max-model-len=$MAX_MODEL_LEN, max-num-seqs=$MAX_NUM_SEQS, vision enabled, no speculative decoding)..."
docker run "${RUN_ARGS[@]}" "${VLLM_ARGS[@]}"
echo "Boot log: docker logs -f $CONTAINER_NAME"
echo "Watch for 'content: null' in traffic — if seen, fallback is to drop --reasoning-parser entirely (see README)."
