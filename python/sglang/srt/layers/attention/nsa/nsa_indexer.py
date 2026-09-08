# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_indexer instead.
"""兼容转发：原 NSA Indexer，现位于 ``dsa.dsa_indexer.Indexer``。

Indexer 是稀疏注意力的第一段：把 hidden / QLoRA 投影成多 head query 和
单 head index-K，写入 FP8 index-K cache，再对 KV 做加权 MQA logits，
选出每个 query 的 Top-K 物理槽位。``DeepseekSparseAttnBackend`` 只消费
这些下标，不再扫全序列。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.nsa_indexer is deprecated; "
    "use sglang.srt.layers.attention.dsa.dsa_indexer instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import *  # noqa: F401, F403
