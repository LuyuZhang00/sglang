# [Deprecated] Re-export shim for backward compatibility. Use dsa.quant_k_cache instead.
"""兼容转发：把 bf16/FP16 index-K 量化进 FP8 index cache。

Indexer 写入 cache 前走这条路径；也可与 store 融合成
``fused_store_index_k_cache``。实现在 ``kernels.ops.attention.dsa.quant_k_cache``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.quant_k_cache is deprecated; "
    "use sglang.kernels.ops.attention.dsa.quant_k_cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import *  # noqa: F401, F403
