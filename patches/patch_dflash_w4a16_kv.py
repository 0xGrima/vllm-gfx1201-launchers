#!/usr/bin/env python3
"""Let DFlash2's fused KV projection survive a drafter whose qkv_proj has NO `.weight` at all.

Runs AFTER patch_dflash_mxfp4_kv.py and finishes the job it starts. That patch generalized the
fused-KV precompute from "fp8 or dense" to "dense or anything else", which is right for a Quark
MXFP4 checkpoint: there `weight` exists and holds packed uint8 [N, K/2], so testing its dtype is
safe and only the branch taken was wrong.

A GPTQ-style W4A16 checkpoint (auto_gptq / the Qwen3.8-27B-DFlash2-W4A16 drafter this launcher
speculates with) does not store `weight` under any dtype. It stores qweight/qzeros/scales,
and QKVParallelLinear inherits nn.Module.__getattr__, so the dtype test itself raises

    AttributeError: 'QKVParallelLinear' object has no attribute 'weight'

from vllm/model_executor/models/qwen3_dflash.py:478, during load_weights -> _build_fused_kv_buffers
-> _build_context_kv_buffers, before any guard can act on it.

Fix: treat "no dense weight" exactly as "quantized weight" -- defer the build, then recover the K/V
rows through the identity trick, which is scheme-agnostic because it runs the layer's own apply().
Same exactness argument as the MXFP4 case: the identity's 0.0/1.0 entries are representable without
rounding, so the recovered rows equal the dequantized weight rather than approximating it.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
DF = SP / "vllm" / "model_executor" / "models" / "qwen3_dflash.py"

# Site 1: the row reader. Absent weight -> fall through to the identity path below it.
ANCHOR_ROWS = '''    weight = qkv_proj.weight
    # radiance (patch_dflash_mxfp4_kv.py): the direct slice is valid only for a dense
    # compute-dtype weight. Upstream tested "is it fp8"; an MXFP4 weight is packed uint8 of shape
    # [N, K/2] and passed that test, then blew up as a 2560-wide bf16 matrix. Test for the dense
    # case instead, so every quantized scheme takes the identity path below.
    if weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
'''
NEW_ROWS = '''    # radiance (patch_dflash_w4a16_kv.py): a GPTQ-style W4A16 layer has no `weight` at all
    # (qweight/qzeros/scales instead), and nn.Module.__getattr__ raises rather than returning None.
    weight = getattr(qkv_proj, "weight", None)
    # radiance (patch_dflash_mxfp4_kv.py): the direct slice is valid only for a dense
    # compute-dtype weight. Upstream tested "is it fp8"; an MXFP4 weight is packed uint8 of shape
    # [N, K/2] and passed that test, then blew up as a 2560-wide bf16 matrix. Test for the dense
    # case instead, so every quantized scheme takes the identity path below.
    if weight is not None and weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
'''

# Site 2: the defer decision. Absent weight -> defer, same as any quantized weight.
ANCHOR_DEFER = '''        # radiance (patch_dflash_mxfp4_kv.py): defer for ANY quantized weight, not just fp8 --
        # the dense rows cannot be read until the quant method has processed the weights.
        if layers_attn[0].qkv_proj.weight.dtype not in (
            torch.bfloat16, torch.float16, torch.float32
        ):
'''
NEW_DEFER = '''        # radiance (patch_dflash_mxfp4_kv.py): defer for ANY quantized weight, not just fp8 --
        # the dense rows cannot be read until the quant method has processed the weights.
        # radiance (patch_dflash_w4a16_kv.py): ... and defer for a layer with no dense `weight`
        # at all, which is what a GPTQ-style W4A16 drafter has. Reading .dtype here is what threw.
        _w0 = getattr(layers_attn[0].qkv_proj, "weight", None)
        if _w0 is None or _w0.dtype not in (
            torch.bfloat16, torch.float16, torch.float32
        ):
'''


# Site 3: the identity path reads its device off `weight`, which is exactly the object that does
# not exist here. Take it from whatever parameter the layer does own (qweight / scales / ...).
ANCHOR_DEV = """    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)
    eye = torch.eye(
        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device
    )
"""
NEW_DEV = """    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)
    # radiance (patch_dflash_w4a16_kv.py): no dense `weight` means no `weight.device` either.
    _dev = weight.device if weight is not None else next(qkv_proj.parameters()).device
    eye = torch.eye(
        qkv_proj.input_size_per_partition, dtype=dtype, device=_dev
    )
"""

def main():
    apply(DF, ANCHOR_ROWS, NEW_ROWS, "patch_dflash_w4a16_kv.py): a GPTQ-style",
          "dflash: fused-KV rows when qkv_proj has no dense weight")
    apply(DF, ANCHOR_DEFER, NEW_DEFER, "patch_dflash_w4a16_kv.py): ... and defer",
          "dflash: defer fused-KV build when qkv_proj has no dense weight")
    apply(DF, ANCHOR_DEV, NEW_DEV, "no dense `weight` means no `weight.device`",
          "dflash: identity-path device when qkv_proj has no dense weight")


if __name__ == "__main__":
    main()
