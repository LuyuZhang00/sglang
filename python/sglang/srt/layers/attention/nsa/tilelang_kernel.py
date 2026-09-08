# [Deprecated] Re-export shim for backward compatibility. Use dsa.tilelang_kernel instead.
"""兼容转发：NSA/DSA 的 TileLang kernel，默认 decode / 部分 prefill 走这里。

实现已迁到 ``kernels.ops.attention.dsa.tilelang_kernel``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.tilelang_kernel is deprecated; "
    "use sglang.kernels.ops.attention.dsa.tilelang_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.tilelang_kernel import *  # noqa: F401, F403
