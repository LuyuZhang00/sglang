# [Deprecated] attention/nsa/ is a thin re-export shim for backward compatibility.
# Use attention/dsa/ instead. This directory will be removed in a future release.
"""NSA 目录的兼容转发层，不是 Sparse Attention 的实现本体。

历史名字是 Native Sparse Attention（NSA）。上游已把实现重命名为 DeepSeek
Sparse Attention（DSA），代码迁到 ``attention/dsa/`` 与
``attention/dsa_backend.py``。本目录每个文件只 ``warnings.warn`` 再
``from dsa... import *``，让旧 import 路径暂时还能用。

本包没有 ``run_scheduler_process``。那个函数是 Scheduler 子进程入口
（``sglang.srt.managers.scheduler``），负责 continuous batching。NSA/DSA
活在 GPU 前向里：Scheduler 选出 ``ScheduleBatch`` 之后，ModelRunner 调
``DeepseekSparseAttnBackend``。二者不在同一层。

真实实现落点::

    attention/nsa/*.py          # 本目录：弃用 shim
    attention/nsa_backend.py    # shim → dsa_backend.py
            │
            ▼
    attention/dsa_backend.py    # AttentionBackend：plan + sparse MLA kernel
    attention/dsa/dsa_indexer.py
            │                    # Indexer：投影 / index-K cache / Top-K
            ▼
    kernels/ops/attention/dsa/  # Triton / TileLang / quant / transform_index

serving 热路径（注意 Scheduler 与 DSA 的边界）::

    run_scheduler_process                # CPU 进程，不在本目录
            │
            ▼
    Scheduler.run_event_loop
            │  ScheduleBatch
            ▼
    ModelRunner.forward
            │
            ├─ 每拍一次: DeepseekSparseAttnBackend.init_forward_metadata
            │       填 page_table / cu_seqlens / topk 变换元数据
            │
            └─ 每一层:
                    Indexer.forward_cuda(hidden, q_lora, ...)
                        → topk_indices  (每个 query 选 index_topk 个 KV 槽)
                    DeepseekSparseAttnBackend.forward_extend / forward_decode
                        → 只对 topk 槽做 MLA，而不是全序列 attention

为什么需要 DSA（原 NSA）
------------------------
DeepSeek-V3.2 / GLM-5 一类模型用 indexer 先打分再稀疏 attend，把 decode
的 KV 访问从 O(context) 降到 O(topk)。没有 Indexer + 本 backend，模型
只能退回 dense MLA，长上下文会把 HBM 带宽打满。

关键对象
--------
* ``Indexer``：从 hidden/QLoRA 得到 query 与 index-K，写入 FP8 index cache，
  再对 paged/ragged KV 算 MQA logits，选出 Top-K。
* ``DeepseekSparseAttnBackend``：``AttentionBackend`` 实现。``__init__``
  选 prefill/decode kernel；``init_forward_metadata`` 每拍 plan 一次；
  ``forward_extend`` / ``forward_decode`` 每层只跑稀疏 MLA。
* ``DSAMetadata``：plan 产物（page table、cu_seqlens、topk 偏移），各层共享。

文件对照（旧 nsa 名 → 新位置）::

    nsa/nsa_indexer.py              → dsa/dsa_indexer.py
    nsa/utils.py                    → dsa/utils.py
    nsa/nsa_backend_mtp_precompute  → dsa/dsa_backend_mtp_precompute.py
    nsa/transform_index.py          → kernels/ops/attention/dsa/transform_index.py
    nsa/triton_kernel.py            → kernels/ops/attention/dsa/triton_kernel.py
    nsa/tilelang_kernel.py          → kernels/ops/attention/dsa/tilelang_kernel.py
    nsa/quant_k_cache.py            → kernels/ops/attention/dsa/quant_k_cache.py
    nsa/dequant_k_cache.py          → kernels/ops/attention/dsa/dequant_k_cache.py
    nsa/index_buf_accessor.py       → kernels/ops/attention/dsa/index_buf_accessor.py
    nsa_backend.py                  → dsa_backend.py
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa is deprecated; "
    "use sglang.srt.layers.attention.dsa instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa import *  # noqa: F401, F403
