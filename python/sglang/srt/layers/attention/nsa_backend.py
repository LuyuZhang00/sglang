# [Deprecated] nsa_backend.py is a thin re-export shim for backward compatibility.
# Use dsa_backend.py instead. This file will be removed in a future release.
"""兼容转发：原 ``NativeSparseAttnBackend``，现为 ``DeepseekSparseAttnBackend``。

这是 NSA/DSA 的 ``AttentionBackend`` 入口（真正实现见 ``dsa_backend.py``）：

* ``__init__``：选 prefill/decode kernel、分配 workspace、接 Top-K backend。
* ``init_forward_metadata``：每拍 plan 一次 page table / cu_seqlens。
* ``forward_extend`` / ``forward_decode``：每层只对 Indexer 选出的 Top-K 槽做 MLA。

没有 ``run_scheduler_process``：调度在 ``managers.scheduler``，本文件只服务
ModelRunner 的 GPU 前向。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa_backend is deprecated; "
    "use sglang.srt.layers.attention.dsa_backend instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa_backend import *  # noqa: F401, F403
from sglang.srt.layers.attention.dsa_backend import (  # noqa: F401
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    DSAFlashMLAMetadata,
    DSAIndexerMetadata,
    DSAMetadata,
    NativeSparseAttnBackend,
    NativeSparseAttnMultiStepBackend,
    NSAFlashMLAMetadata,
    NSAIndexerMetadata,
    NSAMetadata,
)
