# [Deprecated] Re-export shim for backward compatibility. Use dsa.dequant_k_cache instead.
"""兼容转发：从 FP8 / FP4 paged index-K cache 反量化，供 dense 或 gather 路径用。

实现已迁到 ``kernels.ops.attention.dsa.dequant_k_cache``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.dequant_k_cache is deprecated; "
    "use sglang.kernels.ops.attention.dsa.dequant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.dequant_k_cache import *  # noqa: F401, F403
