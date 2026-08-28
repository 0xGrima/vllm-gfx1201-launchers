#!/usr/bin/env bash
# startup-qwen3.8-27b-vllm.sh — standalone launcher for Qwen3.8-27B: native MXFP4 (W4A8) +
# DFlash2 n=7 speculative decoding, on one gfx1201 R9700, 160k ctx.
#
# Runs directly with `docker run`, no router/orchestration layer in front — useful for isolated
# benchmarking, a second host, or debugging a boot failure without extra machinery in the way.
#
# See ../README.md for the full setup story (image build, checkpoint prep, why each patch/flag
# exists) — this script only wires it up, it doesn't explain it.
set -euo pipefail

# ---------------------------------------------------------------------- configuration (env vars)
IMAGE="${IMAGE:-local-radiance-mxfp4:0.9.3}"
CONTAINER_NAME="${CONTAINER_NAME:-r9700-qwen3.8-mxfp4}"
MODELS_DIR="${MODELS_DIR:-/models}"
CACHE_DIR="${CACHE_DIR:-./vllm-cache-mxfp4}"
PORT="${PORT:-9300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"          # 160k VERIFIED ceiling — do not raise, see README
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"               # 0.92 does not fit the drafter's ~6.7 GiB bf16 KV
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"            # drafter trained at this depth (block_size 8)
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
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
  -e GPU_MAX_HW_QUEUES=1
  -e HSA_ENABLE_INTERRUPT=1
  -e HSA_ENABLE_MWAITX=1
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_ROCM_USE_AITER=1
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
  -e VLLM_ROCM_USE_AITER_MHA=0
  -e VLLM_ROCM_USE_AITER_MLA=0
  -e VLLM_ROCM_USE_AITER_MOE=0
  -e VLLM_ROCM_USE_AITER_LINEAR=0
  -e VLLM_ROCM_USE_AITER_FP8BMM=0
  -e VLLM_ROCM_USE_AITER_FP4BMM=0
  -e VLLM_ROCM_USE_AITER_RMSNORM=0
  -e RADIANCE_USE_R4D=1
  -e RADIANCE_R4D_REPORT=1
  -e RADIANCE_PRESHUFFLE=1
  -e RADIANCE_FUSE_RMS_QUANT=1
  -e RADIANCE_SKINNY_GEMM=1
  -e RADIANCE_TOPK_TRITON_MIN_ROWS=1
  -e RADIANCE_FAST_DRAFT=0
  -e RADIANCE_DYNAMIC_DRAFT=0
  -e RADIANCE_MXFP4=1
  -e RADIANCE_MXFP4_W4A8=1
  -e RADIANCE_MXFP4_W4A8_MIN_M=0                  # correctness, not tuning — aiter's W4A4
                                                    # fallback returns WRONG results for the GDN
                                                    # N=5120,K=3072 projection; 0 = never fall back
  -e RADIANCE_MXFP4_TN4_MIN_M=2048
  -e RADIANCE_MXFP4_DECODE_MAX_M=64                # DFlash2: M = batch x 8, up to 64
  -e RADIANCE_MXFP4_SANITIZE=0
  -e RADIANCE_RMS_QUANT_FUSION=0                   # tried+reverted at 163840 ctx, see README §Parameters
  -e RADIANCE_MXFP4_HOIST_QUANT=0                  # same — breaks the KV boot margin together with the above
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
# MOUNT REQUIREMENT: this checkpoint (Qwen3.8-27B-MXFP4-mtpfp8-pertoken) ships everything but
# config.json as a symlink into the SIBLING dir Qwen3.8-27B-MXFP4-mtpfp8/ — the whole models/
# tree must be mounted (done above), not just this one model subdirectory, or the tokenizer
# silently fails to resolve <think>/</think> and ReasoningConfig dies at startup.
VLLM_ARGS=(
  serve /models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken
  --served-model-name qwen3.8-27b-mxfp4 qwen-mxfp4 qwen3.8-mxfp4
  --kv-cache-dtype fp8
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --override-generation-config "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20}"
  --default-chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 2
  --max-num-batched-tokens 2560
  --attention-backend ROCM_AITER_UNIFIED_ATTN
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'
  --no-enable-prefix-caching
  --mamba-cache-mode align
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.8-27B-DFlash2-W4A16\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"bfloat16\"}"
  --chat-template /opt/templates/froggeric-qwen.jinja
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --no-async-scheduling
  --trust-remote-code
  --host 0.0.0.0 --port "$PORT"
)

echo "Starting $CONTAINER_NAME on port $PORT (max-model-len=$MAX_MODEL_LEN, spec-tokens=$NUM_SPEC_TOKENS)..."
docker run "${RUN_ARGS[@]}" "${VLLM_ARGS[@]}"
echo "Boot log: docker logs -f $CONTAINER_NAME"
echo "COLD START IS ~4+ MINUTES; a first boot against an empty $CACHE_DIR is slower still (Triton/inductor autotuning)."
