# local-radiance-mxfp4:0.9.3 — the image both startup-*.sh launchers in this directory run.
#
#
# Two things are fetched at build time rather than vendored in this repo, same policy for both:
# they're third-party/upstream content, not this repo's own work.
#   1. codeberg.org/ggz14/radiance-vllm-mxfp4 @ dba9def   (the MXFP4 runtime delta)
#   2. huggingface.co/froggeric/Qwen-Fixed-Chat-Templates (the chat template both launchers need)
#
# Build:
#   docker build -t local-radiance-mxfp4:0.9.3 -f Dockerfile .
#
# No ENTRYPOINT/CMD is set here — deliberately. The base image's own `vllm` entrypoint is what
# lets both startup-*.sh scripts pass `serve /models/... <flags>` directly as `docker run`
# trailing args; overriding it here would break both launchers.

# Digest-pinned, not just tag-pinned 
ARG RADIANCE_IMAGE=stilldeadcode/vllm-radiance:0.9.3@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
ARG RUNTIME_DELTA_REF=dba9def
ARG GFX_ARCH=gfx1201

# --- Stage 1: fetch ggz14's runtime delta + the third-party chat template, one stage, one
# apt-get (git and curl are both needed here and nowhere else) ---------------------------------
FROM debian:12-slim AS fetch
ARG RUNTIME_DELTA_REF
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --quiet https://codeberg.org/ggz14/radiance-vllm-mxfp4 /delta \
    && git -C /delta checkout --quiet "${RUNTIME_DELTA_REF}"
RUN mkdir -p /templates && curl -fsSL \
    https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja \
    -o /templates/froggeric-qwen.jinja

# --- Stage 3: the actual image -----------------------------------------------------------------
FROM ${RADIANCE_IMAGE}
ARG SP=/opt/vllm/lib/python3.12/site-packages
ARG GFX_ARCH

# Base sanity check, same as the internal build this repo is a showcase of: fail loudly at build
# time if DFlash2 isn't already in the base image, rather than a confusing runtime failure later.
RUN test -f ${SP}/vllm/model_executor/models/qwen3_dflash2.py \
 && python -c "import torch, r4d; assert r4d.__version__ == '0.5.0', r4d.__version__" \
 && echo 'base carries DFlash2 in-tree + libr4d 0.5.0'

# Step 5: bake the chat template. Both launcher scripts point --chat-template here; a missing
# file is a fatal boot error, not a silent fallback (see README's Building the image section).
COPY --from=fetch /templates/froggeric-qwen.jinja /opt/templates/froggeric-qwen.jinja

# Step 2: copy in the runtime-delta files from ggz14's repo (see ../README.md's table for what
# each one does) plus this repo's own patches/patch_dflash_w4a16_kv.py + patches/_patchlib.py.
RUN mkdir -p /opt/patches-mxfp4
COPY --from=fetch \
    /delta/radiance_mxfp4.py /delta/radiance_gdn.py /delta/radiance_rmsquant.py \
    ${SP}/
COPY --from=fetch /delta/mxfp4-configs/ ${SP}/aiter/ops/triton/configs/gemm/
COPY --from=fetch \
    /delta/patch_quark_mxfp4.py \
    /delta/patch_dflash_mxfp4_kv.py \
    /delta/patch_topk_triton_rows.py \
    /delta/patch_ar_maxbytes.py \
    /delta/patch_dflash_calib.py \
    /delta/patch_rmsquant_fusion.py \
    /delta/patch_qwen3_thinkoff.py \
    /delta/radiance_mxfp4_fp8.hip \
    /delta/_patchlib.py \
    /opt/patches-mxfp4/
COPY patches/patch_dflash_w4a16_kv.py /opt/patches-mxfp4/

# Step 3: apply every patch, in order. patch_dflash_w4a16_kv MUST run after patch_dflash_mxfp4_kv
# -- it rewrites that patch's own output (see the patch's own docstring, and patches/README.md).
# Step 4: compile the fp8-WMMA W4A8 GEMM kernel with hipcc.
RUN cd /opt/patches-mxfp4 \
 && for p in patch_quark_mxfp4 patch_dflash_mxfp4_kv patch_dflash_w4a16_kv \
             patch_topk_triton_rows patch_ar_maxbytes patch_dflash_calib patch_rmsquant_fusion; do \
      echo "== applying $p =="; python "$p.py"; \
    done \
 && (python patch_qwen3_thinkoff.py || echo "WARNING: thinkoff did not apply (non-fatal)") \
 && INC=$(python -m pybind11 --includes) \
 && hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=${GFX_ARCH} -Wno-unused-result \
      $INC radiance_mxfp4_fp8.hip -o ${SP}/radiance_mxfp4_fp8.so \
 && python -c "import torch, radiance_mxfp4_fp8 as m, radiance_mxfp4, radiance_rmsquant; \
assert hasattr(m, 'launch'); print('mxfp4 w4a8 kernel built + all runtime modules importable')" \
 && python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; \
print('radiance modules parse OK')"
