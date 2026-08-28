# patches/

Genuinely original source patches — **not** copies of anything from
`codeberg.org/ggz14/radiance-vllm-mxfp4` (which ships its own, larger patch set — `patch_quark_mxfp4`,
`patch_dflash_mxfp4_kv`, `patch_topk_triton_rows`, etc.) or from `StillDeadcode/vllm-radiance`
upstream. Both of those are applied at build time — see `../README.md`'s "additions on top of the
two upstreams" section — but only the file below was authored here; everything else is fetched at
build time from its own upstream source.

| file | what it does |
|---|---|
| `patch_dflash_w4a16_kv.py` | Lets a **GPTQ-style W4A16 DFlash2 drafter** (no dense `.weight` tensor — only `qweight`/`qzeros`/`scales`) load at all. Without it, `vllm/model_executor/models/qwen3_dflash.py`'s fused-KV precompute reads `qkv_proj.weight.dtype` directly and raises `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'` before any guard can act, because `nn.Module.__getattr__` doesn't return `None` for a missing attribute — it raises. Fixes three call sites (the defer decision, the row reader, the identity path's device lookup) to treat "no dense weight" the same as "quantized weight" and recover the K/V rows through the drafter layer's own `apply()` instead of slicing a tensor that doesn't exist. This is the patch that makes `syvai/Qwen3.8-27B-DFlash2-W4A16` — the drafter actually served here — loadable in the first place; ggz14's own `patch_dflash_mxfp4_kv.py` only generalized "fp8 vs dense", not "dense vs no-weight-at-all". |
| `_patchlib.py` | Shared `apply()` helper `patch_dflash_w4a16_kv.py` imports (`from _patchlib import apply`) — an idempotent one-shot string-replace-and-`ast.parse`-verify pattern used by every patch file in this build, this being the only one of those actually authored here. Kept alongside so the patch is runnable standalone (`python patch_dflash_w4a16_kv.py` from this directory, against a checked-out vLLM tree's `site-packages`). |

## Why this is the *only* file here

Every other environment variable, kernel module (`radiance_mxfp4.py`, `radiance_gdn.py`,
`radiance_mxfp4_fp8.hip`), and patch (`patch_quark_mxfp4.py`, `patch_topk_triton_rows.py`, …) the
image bakes in is **fetched, not written** — pulled at build time from
`codeberg.org/ggz14/radiance-vllm-mxfp4` @ `dba9def` (see `../README.md` for the full list and
what each one does). Copying those files into this repo would misrepresent them as original work.
The one gap ggz14's own patch set left — a W4A16-format drafter, as opposed to their MXFP4-format
target model — is what required writing something new, and that's the only thing that belongs here.

## Applying it

Same mechanism as every `patch_*.py` in the build (see the Dockerfile guidance in `../README.md`):
run from this directory against an installed vLLM tree, idempotent, verifies its own anchor
before writing:

```sh
cd patches/
python3 patch_dflash_w4a16_kv.py
```

It must run **after** ggz14's own `patch_dflash_mxfp4_kv.py` — this patch rewrites that patch's
own output (see the docstring at the top of the file). Build order:
`patch_quark_mxfp4 → patch_dflash_mxfp4_kv → patch_dflash_w4a16_kv → patch_topk_triton_rows → ...`
