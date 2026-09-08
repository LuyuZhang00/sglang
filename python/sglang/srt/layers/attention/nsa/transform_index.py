# [Deprecated] Re-export shim for backward compatibility. Use dsa.transform_index instead.
"""兼容转发：Top-K 局部下标 → 物理 page table / ragged 偏移。

Indexer 给出的是某行 logits 里的列号；MLA kernel 要的是 KV cache 槽位。
``transform_index_page_table_decode`` / ``_prefill`` 完成这次地址变换。
实现已迁到 ``kernels.ops.attention.dsa.transform_index``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.transform_index is deprecated; "
    "use sglang.kernels.ops.attention.dsa.transform_index instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.transform_index import *  # noqa: F401, F403
