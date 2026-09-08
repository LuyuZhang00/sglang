# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead.
"""兼容转发：NSA/DSA 的 Triton kernel（act_quant、prefill flash 等）。

实现已迁到 ``kernels.ops.attention.dsa.triton_kernel``。可用
``SGLANG_DSA_TRITON_PREFILL=1`` 让 FP8 sparse-MLA prefill 走这条路径。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use sglang.kernels.ops.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.triton_kernel import *  # noqa: F401, F403
