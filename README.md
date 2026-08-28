# vllm-gfx1201-launchers

Standalone vLLM launchers for the AMD Radeon AI PRO R9700 (RDNA4/gfx1201, 32GB) — a showcase of
how two models are actually run on this card: **Qwen3.8-27B** at native MXFP4 (W4A8) with a
DFlash2 speculative drafter, and **Ornith-1.5-35B-A3B** (MoE, GDN linear-attention hybrid) at
compressed-tensors W4A16. 

Both scripts run standalone, without any router/orchestration layer in front — just `docker run`
and vLLM's own CLI flags.


## Overview

Both models run on the same image — one 32 GB card fits one resident model at a time. Base is the
published [`stilldeadcode/vllm-radiance:0.9.3`](https://hub.docker.com/r/stilldeadcode/vllm-radiance)
(vLLM 0.27.1, DFlash2 and `libr4d` 0.5.0 already in-tree), with a runtime patch delta from
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4) @ `dba9def` applied
on top for the native MXFP4 (W4A8) kernel path — plus the one patch in `patches/` that neither of
those upstreams ship.

| model | quant | speculative decoding | context | decode | prefill@11.8K |
|---|---|---|---|---:|---:|
| `qwen3.8-27b-mxfp4` | native MXFP4 (W4A8) | DFlash2 n=7 | 163,840 | **78-81 t/s** | 2340-2375 t/s |
| `ornith-1.5-35b-a3b` | compressed-tensors W4A16 (MoE experts only) | **none** — measured worse | 262,144 | **76.8 t/s** | 7286 t/s @16K |

Both measured with BetterBench, 300 W, single-stream. Qwen3.8 range reflects four independent
boots of the identical config — this checkpoint's GDN-hybrid architecture shows real boot-to-boot
KV/scheduling noise of a couple percent; treat any single number as a point in that band, not a
fixed constant.

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

### 2. `fp8_mtp.py` (also from ggz14's repo — checkpoint prep, not a runtime patch)

AMD's published `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` ships its MTP head in bf16 but doesn't
`exclude` it, so vLLM's quark config falls through to MXFP4 for `mtp.*` and dies loading a
full-width bf16 tensor into a half-width packed buffer. `fp8_mtp.py` rewrites just those eight
MTP projections to `float8_e4m3fn` (FP8 was chosen over re-quantizing to MXFP4 after both plain
RTN and AWQ calibration measurably hurt drafter acceptance — RTN cost acceptance 2.5 → 2.21, and
AWQ calibration didn't rescue it either). CPU-only, no GPU needed. Produces the checkpoint the
Qwen launcher script actually serves.

### 3. `patches/patch_dflash_w4a16_kv.py` — original work

The one file in this directory that is genuinely original, not fetched from anywhere. Full
explanation in `patches/README.md`. Short version: ggz14's `patch_dflash_mxfp4_kv.py` handles a
drafter whose `qkv_proj.weight` exists but is packed MXFP4 uint8. The drafter this repo actually
serves, `syvai/Qwen3.8-27B-DFlash2-W4A16`, is a **different quant family** (GPTQ-style
compressed-tensors) that has **no `weight` attribute at all** — only `qweight`/`qzeros`/`scales`
— so reading `.weight.dtype` raises `AttributeError` before ggz14's own fix can even run. Without
this patch the drafter cannot be loaded, period.

## Model downloads & checkpoint prep

**Requires the image built in "Building the image" below** — the FP8-MTP conversion step runs
inside it (`docker run --entrypoint python ... local-radiance-mxfp4:0.9.3 /src/fp8_mtp.py ...`).
Build the image first if starting from scratch.

```bash
MODELS=/path/to/models   # wherever you point MODELS_DIR at

# --- Qwen3.8-27B-MXFP4 target ---
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --local-dir $MODELS/Qwen3.8-27B-Quark-AWQ-MXFP4

git clone https://codeberg.org/ggz14/radiance-vllm-mxfp4 /tmp/mxfp4-src
git -C /tmp/mxfp4-src checkout dba9def
docker run --rm --entrypoint python -v $MODELS:/models -v /tmp/mxfp4-src:/src \
  local-radiance-mxfp4:0.9.3 \
  /src/fp8_mtp.py /models/Qwen3.8-27B-Quark-AWQ-MXFP4 /models/Qwen3.8-27B-MXFP4-mtpfp8
# then generate a per-token variant if desired (config-only, ~64KB of symlinks + a rewritten
# config.json moving the MTP layers' input quantization from per-tensor to per-channel).

# --- DFlash2 drafter (needs patch_dflash_w4a16_kv.py above to load) ---
hf download syvai/Qwen3.8-27B-DFlash2-W4A16 --local-dir $MODELS/Qwen3.8-27B-DFlash2-W4A16

# --- Ornith-1.5-35B-A3B ---
hf download MIRALABS/Ornith-1.5-35B-A3B-W4A16-SYM --local-dir $MODELS/ornith-1.5-35b-a3b-w4a16-sym
# NOTE: the release's own config.json quantization_config.ignore list omits the MTP head, which
# is plain bf16 -- vLLM instantiates it quantized and fails to load without a fix. Append
# "mtp.fc" and the regex "re:mtp\\." to that ignore list before serving (not needed for the
# no-MTP config the launcher below runs, but required if MTP is ever re-enabled). Verify every
# downloaded shard's size against the HF API's authoritative blob size, too -- a truncated
# mid-transfer download can look complete in a casual `ls`.
```

## Building the image

The image (`local-radiance-mxfp4:0.9.3`) is built from a Dockerfile that fetches ggz14's runtime
delta and applies every patch listed above, including this directory's `patch_dflash_w4a16_kv.py`.
That Dockerfile isn't included in this directory — bring your own, or write one that:

1. Starts `FROM stilldeadcode/vllm-radiance:0.9.3`
2. `COPY`s in the runtime-delta files from `ggz14/radiance-vllm-mxfp4` @ `dba9def` (see table
   above) plus this directory's `patches/patch_dflash_w4a16_kv.py` and `patches/_patchlib.py`
3. Applies each `patch_*.py` in order — **`patch_dflash_w4a16_kv` must run after
   `patch_dflash_mxfp4_kv`**, it rewrites that patch's own output (see the patch's docstring)
4. Compiles `radiance_mxfp4_fp8.hip` with `hipcc --offload-arch=gfx1201`
5. Fetches [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)'s
   `chat_template.jinja` and bakes it to `/opt/templates/froggeric-qwen.jinja` inside the image
   — both launcher scripts point `--chat-template` there, and **the container fails to boot
   without it** (`--chat-template` pointing at a missing file is fatal, not a silent fallback).
   It's a third-party template (covers Qwen 3.5/3.6/3.8), so it's fetched at build time rather
   than vendored into this repo, same policy as the MXFP4 runtime delta above. If you'd rather
   not depend on it, drop `--chat-template ...` from both scripts' `VLLM_ARGS` — vLLM then uses
   each checkpoint's own shipped `chat_template.jinja` instead (untested by the numbers in this
   README; the froggeric template is what they were actually measured against).

```bash
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
| `MAX_MODEL_LEN` | `163840` (Qwen) / `262144` (Ornith) | KV allocation ceiling — see the boot-log checklist below before raising |
| `GPU_MEM_UTIL` | `0.95` | vLLM's `--gpu-memory-utilization` |
| `NUM_SPEC_TOKENS` | `7` (Qwen only) | DFlash2 draft depth — the drafter was trained at this exact value |
| `REASONING_EFFORT` | `medium` (Qwen only) | Qwen3.8's graded thinking-effort default |
| `POWER_CAP_W` | unset | optional GPU power cap in microwatts (e.g. `240000000` = 240 W) |

## Boot-log checklist

- `Using RadianceMxfp4W4A8LinearKernel` and `304/304 on our kernel, 0 FORCED ONTO AITER` —
  confirms the native MXFP4 kernel is actually engaged, not silently emulated (Qwen only).
- `GPU KV cache size: N tokens` — the real, measured ceiling for this boot; it swings between
  boots (Qwen measured 7.16–8.12 GiB free KV across identical boots at 163840) — if a boot fails
  where the last one succeeded at the same `MAX_MODEL_LEN`, this is why; lower it.
- `creating MTP draft context against the target model` (only relevant if you ever re-enable
  Ornith's MTP) confirms the checkpoint's MTP tensors were actually found and loaded.
- **Cold cache first boot**: an empty `CACHE_DIR` at `MAX_MODEL_LEN` above ~131072 can fail with
  `estimated maximum model length is ~139776` — cold Triton/inductor profiling uses ~1 GiB more
  peak memory than warm. Either warm the cache once at a lower `MAX_MODEL_LEN` first, or retry.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'` | `patch_dflash_w4a16_kv.py` not applied — building from an image that skipped it | rebuild, confirm the patch's `OK` line in the build log |
| `failed to tokenize reasoning strings: reasoning_start_str='', reasoning_end_str=''` (Qwen) | only the per-token model dir mounted, not the whole `models/` tree — its files are symlinks into a sibling dir | mount the entire `MODELS_DIR`, not a per-model subdirectory |
| `AssertionError: In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens` (Ornith) | `--max-num-batched-tokens` below 2096 | keep it at `2560` as shipped in the script — do not lower it |
| `content: null` with a fully-populated `reasoning` field | a known upstream vLLM bug class in token-ID-based `</think>` boundary detection ([vllm-project/vllm#15758](https://github.com/vllm-project/vllm/issues/15758)) — architecture-agnostic, not fixable by patching this model | drop `--reasoning-parser` entirely; content lands raw with inline `<think>` tags instead |
| MXFP4 running much slower than expected, no `RadianceMxfp4W4A8LinearKernel` in the log | `RADIANCE_MXFP4`/`RADIANCE_MXFP4_W4A8` not set, or `patch_quark_mxfp4.py` missing from the image | confirm both env vars are `1` and rebuild if the log never shows the kernel line |
| container exits immediately, log mentions `/opt/templates/froggeric-qwen.jinja` not found | build step 5 (fetching the chat template) was skipped | add it to the Dockerfile, or drop `--chat-template ...` from the script's `VLLM_ARGS` to use the checkpoint's own template instead |
