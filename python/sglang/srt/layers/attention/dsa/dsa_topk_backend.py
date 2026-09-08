"""DSA logits 的 Top-K 选择及索引布局变换后端。

Indexer 产出的列下标只是某行 logits 中的局部位置；注意力 kernel 最终
需要物理分页槽位或 ragged KV 偏移。本模块统一封装 SGL Kernel、Torch 和
FlashInfer 三种实现，并在满足条件时融合“选择 + 布局变换”。
"""

from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_spec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):
    """描述 Top-K 局部下标应转换到哪一种 KV 地址域。"""

    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


class DSATopKBackend(Enum):
    """可执行 DSA Top-K 的算子后端及其统一调度接口。"""

    # SGL Kernel 提供融合变换和 DeepSeek-V4 Top-K v2 快路径。
    SGL_KERNEL = "sgl-kernel"
    # Torch 仅用于通用、易调试的非融合回退。
    TORCH = "torch"
    # FlashInfer 提供 graph-safe 的 Top-K 及分页/ragged 变换。
    FLASHINFER = "flashinfer"

    @classmethod
    def resolve(cls, model_runner: ModelRunner) -> DSATopKBackend:
        """Resolve the DSA top-k backend for one model runner.

        ``--dsa-topk-backend`` selects the target backend, while
        ``--speculative-dsa-topk-backend`` independently selects the draft.
        """
        if model_runner.is_draft_worker:
            return cls(get_spec().speculative_dsa_topk_backend)
        return cls(get_exec().kernel.dsa_topk_backend)

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def should_use_topk_v2(self) -> bool:
        """仅在 SGL backend 且优化开关开启时允许进入 v2 JIT kernel。"""
        return self.is_sgl_kernel() and envs.SGLANG_OPT_USE_TOPK_V2.get()

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """只执行有效区间内的 Top-K，不做页表或 ragged 地址变换。

        ``lengths`` 和可选 ``row_starts`` 共同定义每行 logits 的合法窗口。
        返回的是以该窗口起点为零的局部索引，无效/不足 Top-K 的位置填 -1。
        """
        # SGL 路径使用针对 DSA 形状优化的 fast_topk_v2。
        if self.is_sgl_kernel():
            from sgl_kernel import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        # Torch 路径先 mask 非法列，再调用 torch.topk。
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        # FlashInfer 的选择算子复用相同的非融合窗口与 -1 padding 逻辑。
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        # Enum 扩展后若忘记实现分支，应在这里显式失败。
        raise RuntimeError(f"Unsupported {self = }.")

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        force_unfused_topk: bool = False,
    ) -> torch.Tensor:
        """选择 Top-K 并按 attention backend 需要转换索引地址域。

        关闭融合或显式强制回退时，仅返回局部下标。融合路径根据 PAGED/
        RAGGED 布局以及所选 backend 调用相应 kernel；decode 形状还可进入
        v2 kernel，直接消费紧凑真实页表而不物化宽 page-size=1 页表。
        """
        # 非融合模式刻意跳过地址变换，调用方负责理解返回的局部索引。
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)

        # Decode-shaped PAGED top-k for the SGL backend (plain decode AND spec
        # verify / draft-extend, whose expanded rows match the same shape) routes
        # to the DeepSeek-V4 top-k v2 JIT kernel. It fuses top-k selection and the
        # page-table transform in one launch and consumes the indexer's own
        # page_size>=1 table directly, so no page_size=1 table is materialized.
        # Shared by DeepSeek-V3.2 and GLM DSA.
        # This is a deterministic dispatch on the work shape, not a best-effort
        # attempt: the fused-decode CUDA graph drops the page_size=1 table for
        # exactly this case (see dsa_drop_wide_page_table), so once the shape
        # matches we commit to v2 and never silently fall back to the legacy
        # page_size=1 path from here.
        # 严格匹配 decode-shaped PAGED 输入后，提交给 v2 融合选择/页表转换。
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)

        # Extend-shaped RAGGED top-k for the SGL backend routes to the same v2
        # kernel through its ragged entry point: no page table (the columns are
        # already flattened-KV positions), no plan (prefill has enough rows that
        # the cluster path never applies), just a per-row window and an additive
        # output transform. `batch_idx_list` is not None only on the prefill-CP
        # path, whose `topk_indices_offset` is built from cu_seqlens_q rather
        # than the KV bases -- leave that one on the legacy kernel.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.RAGGED
            and topk_indices_offset is not None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0] == logits.shape[0] == topk_indices_offset.shape[0]
        ):
            return _topk_transform_v2_ragged(
                logits, lengths, topk, topk_indices_offset, row_starts
            )

        # The legacy transforms below read attn_metadata.page_table_1 (page_size=1),
        # which is always present here: the fold only drops it for the decode case
        # dispatched to v2 above.
        # 旧融合 kernel 按 token 直接 gather，因此必须保留宽 page-size=1 页表。
        assert attn_metadata.page_table_1 is not None

        # SGL Kernel 分别提供分页与 ragged 的融合变换实现。
        if self.is_sgl_kernel():
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                # 子 batch/CP 路径只选择参与计算的请求页表行。
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                # kernel 在选出局部位置后立刻查页表得到物理 token slot。
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                # ragged 地址变换必须知道每行所属请求在扁平 KV 中的起点。
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        # FlashInfer 路径采用同样的 PAGED/RAGGED 语义。
        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
<<<<<<< Updated upstream
                row_to_batch, page_table_row_starts = _build_flashinfer_paged_args(
=======
                if row_starts is not None:
                    # Packed PAGED extend uses batch-global score offsets with
                    # request-local page tables. FlashInfer applies row_starts
                    # to both, so reuse the SGL transform.
                    from sgl_kernel import fast_topk_transform_fused

                    # packed extend 的 score 偏移是 batch-global，改用 SGL 变换避免歧义。
                    page_table_size_1 = (
                        attn_metadata.page_table_1[batch_idx_list]
                        if batch_idx_list is not None
                        else attn_metadata.page_table_1
                    )
                    return fast_topk_transform_fused(
                        score=logits,
                        lengths=lengths,
                        page_table_size_1=page_table_size_1,
                        cu_seqlens_q=cu_seqlens_q_topk,
                        topk=topk,
                        row_starts=row_starts,
                    )

                # 把 SGL 元数据转换为 FlashInfer 所需的逐行 batch 映射和局部起点。
                row_to_batch, local_row_starts = _build_flashinfer_paged_args(
>>>>>>> Stashed changes
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                # contiguous 约束来自 FlashInfer custom op 的输入 ABI。
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                    page_table_row_starts=page_table_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                # 直接把局部 Top-K 位置加上逐行 ragged 偏移。
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """以 mask + 通用 Top-K 算子实现不带地址变换的回退路径。

    每行只允许访问 ``[row_start, row_start + length)``；输出再减去
    ``row_start`` 恢复局部索引。空行或候选不足时用 -1 填满固定 Top-K
    宽度，确保 CUDA Graph 和下游 attention 接口形状稳定。
    """
    # 从二维 score 取得行数与统一的物理列宽。
    batch_size, max_score_len = score.shape
    # 先以 -1 初始化，可自然覆盖空输入和 topk 大于有效长度的尾部。
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    # 避免对零尺寸 Tensor 调用某些 backend 不支持的 top-k kernel。
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    # 未提供行起点表示每行合法窗口都从第 0 列开始。
    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    # 长度和起点必须与 score 同设备，统一 int32 便于比较与后续 kernel。
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    # 构造广播列坐标并生成逐行合法窗口 mask。
    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    # 非法位置设为 -inf，保证不会压过任何真实有限 logits。
    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    # 实际调用宽度不能超过 score 的物理列数。
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    # topk_op 可替换为 torch.topk 或 FlashInfer top_k。
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    # 后端返回全局列号，减起点转换为每行窗口内的局部位置。
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    # 若选中了 -inf，说明该行真实候选不足，需要按协议写 -1。
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    # 写入预先固定宽度的输出前缀，其余列继续保持 -1。
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:
    """Fused top-k + page-table transform via the DeepSeek-V4 v2 JIT kernel.

    Returns the transformed page indices ``(num_rows, topk)`` int32 (physical
    page_size=1 KV slots, ``-1`` padded) -- identical in meaning to
    ``fast_topk_transform_fused`` / ``flashinfer.top_k_page_table_transform``.
    The kernel selects, per row, the top-k of ``logits[row, :lengths[row]]`` and
    maps each selected position ``p`` through the page table as
    ``real_page_table[row, p // page_size] * page_size + (p % page_size)``. Feeding
    it the indexer's compact ``real_page_table`` (page_size = pool page size,
    typically 64) yields the same physical slots as gathering the page_size=1
    table, without materializing that wide table.

    This is a committed contract, not a best-effort path: ``topk_transform`` routes
    here only for the decode-shaped PAGED case, and the fused-decode CUDA graph
    drops the page_size=1 table for exactly this case (see
    ``dsa_drop_wide_page_table``). The preconditions below are therefore
    invariants the caller must uphold -- they assert (raise) on violation rather
    than fall back to the slow legacy path (which may not even have a page_size=1
    table to fall back to) or silently paper over bad input (padding, recomputing
    the plan) at the cost of the performance this path exists to deliver.

    ``lengths`` entries must be NON-NEGATIVE: the kernel reads them as
    ``uint32_t``, so a negative row length (DP-padded / idle-companion rows)
    reinterprets as ~4e9 tokens and illegal-addresses. Metadata producers clamp
    padded rows to 0 (see ``fused_dsa_draft_extend_metadata`` /
    ``seqlens_expand_kernel``); 0 takes the trivial all-(-1) output path.
    """
    # 延迟导入 JIT kernel，避免未选择 v2 时产生编译/依赖开销。
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512_v2

    # 每个 logits 行对应一个 decode/verify query。
    num_rows = logits.shape[0]

    # The indexer (DeepGEMM) emits fp32 scores with unit row stride and a 16B-aligned
    # row stride (a multiple of 4), which is exactly the kernel's ABI (it checks
    # score_stride % 4 == 0 with strides {S, 1}). This holds even though the scores
    # may be a padded view (stride(0) > width, so not `is_contiguous()`); assert the
    # real requirement rather than force a contiguous copy of the wide score buffer.
    assert (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), f"v2 top-k expects fp32 scores with unit row stride and 16B-aligned score_stride, got {logits.dtype=} {logits.stride()=}"
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    # v2 直接读取按真实 page size 压缩的页表。
    page_table = attn_metadata.real_page_table
<<<<<<< Updated upstream
=======
    assert page_table.dtype == torch.int32
    # kernel ABI 将长度解释为 uint32 底层值，因此生产方必须保证非负。
    lengths_i32 = lengths.to(torch.int32)
>>>>>>> Stashed changes

    # The plan is preprocessed once per forward (DSAMetadata.topk_v2_plan,
    # refreshed in-place under CUDA graph) and reused across layers. A missing or
    # mismatched plan means the caller skipped that preprocessing -- fail loudly
    # rather than silently recompute it per layer.
    plan = attn_metadata.topk_v2_plan
    assert (
        plan is not None and plan.shape[0] == num_rows + 1
    ), "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"

    # page_size 用于把“页号 + 页内偏移”还原成物理 token slot。
    page_size = attn_metadata.page_size
<<<<<<< Updated upstream
    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_512_v2(logits, lengths, page_table, out, page_size, plan)
    return out


def _topk_transform_v2_ragged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_indices_offset: torch.Tensor,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ragged top-k via the DeepSeek-V4 v2 JIT kernel.

    ``logits`` is written in place: the kernel reads from a 16-byte-aligned base
    and masks the <= 3 columns that pulls in ahead of the window. Those columns
    belong to a preceding request of the same row, and the score buffer is dead
    after the top-k (see ``DSAIndexer._get_topk_ragged``).

    Preconditions match the paged helper: fp32 scores with unit row stride and a
    16B-aligned row stride (DeepGEMM's contiguous-KV output satisfies this by
    construction), int32 non-negative lengths, and ``0 < topk <= 2048``.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    topk_transform_ragged_v2(
        logits,
        lengths,
        out_offsets=topk_indices_offset,
        out_indices=out,
        row_starts=row_starts,
    )
=======
    # 固定输出形状并以 -1 初始化零长度/候选不足的行。
    out = logits.new_full((num_rows, topk), -1, dtype=torch.int32)
    # 单 kernel 完成选择和紧凑页表地址变换。
    topk_transform_512_v2(logits, lengths_i32, page_table, out, page_size, plan)
>>>>>>> Stashed changes
    return out


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """把 batch/累积长度元数据转换为 FlashInfer PAGED Top-K 参数。

    ``row_to_batch`` 指明每个 logits 行使用哪一行页表；``local_row_starts``
    则把 SGL 的 batch-global score 起点转换为请求内部起点。Decode 一请求
    一行时可省略映射，以减少图捕获中的动态 Tensor 构造。
    """
    # 子 batch 调用先把 Python 请求下标搬到 score 所在设备。
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )

    # Both dynamic mappings contain one entry per logit row. Supplying the known
    # size avoids synchronizing CUDA to infer the sum of the repeat counts.
    # batch_idx_list 若是请求级映射而 logits 是 token 级行，需要按 q_len 展开。
    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        # 相邻累积边界之差即每个请求的 query 行数。
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(
            row_to_batch, q_lens, output_size=num_rows
        )

    # 无显式子 batch 时，可从 query 累积长度构造完整逐行映射。
    if row_to_batch is None and cu_seqlens_q_topk is not None:
        # Decode-like case (one query row per batch) does not need an explicit mapping.
        # Avoid dynamic tensor construction in this branch to keep CUDA graph capture safe.
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )

    # 全局 row_starts 没有所属 batch 信息就无法转换为页表局部坐标。
    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

<<<<<<< Updated upstream
    page_table_row_starts = row_starts
    if page_table_row_starts is not None and row_to_batch is not None:
        page_table_row_starts = (
            page_table_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
=======
    # 默认沿用输入；只有 packed PAGED 行才需要减去请求的全局 K 起点。
    local_row_starts = row_starts
    if local_row_starts is not None and row_to_batch is not None:
        local_row_starts = (
            local_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
>>>>>>> Stashed changes
        )

    return row_to_batch, page_table_row_starts


def _flashinfer_tie_break_value() -> int:
    """解析 FlashInfer Top-K 相同分数时的确定性破平策略。

    未设置返回 0，交由 FlashInfer 使用默认行为；``small``/``large`` 分别
    映射为选较小/较大索引的整数 ABI。非法值在首次调用时明确报错。
    """
    # 环境变量未配置时不强制额外的 tie-break 语义。
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    # 环境变量大小写不敏感。
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
