#!/usr/bin/env bash
# startup-ornith-1.5-35b-vllm.sh — standalone launcher for Ornith-1.5-35B-A3B: MoE (256 experts,
# GDN linear-attention hybrid), compressed-tensors W4A16 (experts-only), no speculative decoding
# (measured WORSE on every axis for this checkpoint — see README), full native 262144-token
# context on one gfx1201 R9700.
#
# Unlike the Qwen3.8 launcher, this model needs NO custom MXFP4 patch at all — it's served from
# the vLLM/radiance-native compressed-tensors WNA16 MoE path. It still runs on the SAME
# `local-radiance-mxfp4` image as Qwen3.8 (one 32GB card fits one resident model at a time).
#
# See ../README.md for the two on-disk checkpoint config fixes this model needed (MTP-head
# ignore-list, a corrupted safetensors shard) — this script assumes the checkpoint at MODELS_DIR
# is already patched per that section.
set -euo pipefail

# ---------------------------------------------------------------------- configuration (env vars)
IMAGE="${IMAGE:-local-radiance-mxfp4:0.9.3}"
CONTAINER_NAME="${CONTAINER_NAME:-r9700-ornith-1.5}"
MODELS_DIR="${MODELS_DIR:-/models}"
CACHE_DIR="${CACHE_DIR:-./vllm-cache-mxfp4}"
PORT="${PORT:-9300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # this checkpoint's native max_position_embeddings
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
POWER_CAP_W="${POWER_CAP_W:-}"                     # optional: e.g. 240000000 (microwatts)

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
  -e HIP_VISIBLE_DEVICES=0
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_ROCM_USE_AITER=1
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  -e VLLM_ROCM_USE_AITER_MHA=0
  -e VLLM_ROCM_USE_AITER_MLA=0
  -e VLLM_ROCM_USE_AITER_MOE=0
  -e VLLM_ROCM_USE_AITER_LINEAR=0
  -e VLLM_ROCM_USE_AITER_FP8BMM=0
  -e VLLM_ROCM_USE_AITER_FP4BMM=0
  -e VLLM_ROCM_USE_AITER_RMSNORM=0
  -e RADIANCE_PRESHUFFLE=1
  -e RADIANCE_ATTN_TUNE=1
  -e RADIANCE_GDN_WMMA=1
  -e RADIANCE_VIT_FLASH=1
  -e RADIANCE_FUSE_RMS_QUANT=1
  -e RADIANCE_USE_R4D=1
  -e RADIANCE_R4D_REPORT=1
  -e VLLM_CACHE_ROOT=/cache/vllm
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  -e TRITON_CACHE_DIR=/cache/triton
  -e AITER_ROOT_DIR=/cache/aiter
  -e TRITON_CACHE_AUTOTUNING=1
  -v "$MODELS_DIR:/models"
  -v "$CACHE_DIR:/cache"
  -p "${PORT}:${PORT}"
  "$IMAGE"
)

# ---------------------------------------------------------------------- vLLM args
# MTP deliberately absent: --speculative-config was measured WORSE on every axis for this
# checkpoint (59.5 t/s decode / 6152 t/s prefill@16K WITH qwen3_5_mtp n=4, vs 76.8 / 7286 and a
# lower TTFT WITHOUT it) — do not add it back without re-measuring on this exact checkpoint.
VLLM_ARGS=(
  serve /models/ornith-1.5-35b-a3b-w4a16-sym
  --served-model-name ornith-1.5-35b-a3b ornith
  --kv-cache-dtype fp8
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 4
  --max-num-batched-tokens 2560               # REQUIRED: mamba-cache-align asserts
                                                 # block_size(2096) <= this value, or the boot dies
  --attention-backend ROCM_AITER_UNIFIED_ATTN
  --enable-prefix-caching
  --reasoning-parser qwen3
  --chat-template /opt/templates/froggeric-qwen.jinja
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --trust-remote-code
  --host 0.0.0.0 --port "$PORT"
)

echo "Starting $CONTAINER_NAME on port $PORT (max-model-len=$MAX_MODEL_LEN, no speculative decoding)..."
docker run "${RUN_ARGS[@]}" "${VLLM_ARGS[@]}"
echo "Boot log: docker logs -f $CONTAINER_NAME"
echo "Watch for 'content: null' in traffic — if seen, fallback is to drop --reasoning-parser entirely (see README)."
