"""Native Sparse Attention backend 的多步 MTP 元数据预计算工具。

多步推测解码会创建多个 backend 实例，但它们使用相同的序列长度、页表和
DSA 有效长度。这里将这些值一次算好后复制给各实例，避免每一步重复启动
小 kernel，并让 CUDA Graph replay 使用稳定的张量形状与地址。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.utils import seqlens_expand_triton
from sglang.srt.layers.attention.dsa.utils import compute_dsa_seqlens
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_is_cuda = is_cuda()
_is_hip = is_hip()


@dataclass
class PrecomputedMetadata:
    """多个 MTP backend 实例共享的一组预计算元数据。

    其中同时保存原始 KV 长度、page-size=1 页表、真实页表，以及按 DSA
    Top-K 上限裁剪后的长度。Decode 的展开大小是 batch size；Target Verify
    则会按每个请求的 draft token 数展开。

    """

    # Basic seqlens
    cache_seqlens: torch.Tensor  # int32, [bs]
    cu_seqlens_k: torch.Tensor  # int32, [bs+1]

    # Page table
    page_indices: torch.Tensor  # int32, [bs, max_len] or [expanded_bs, max_len]
    real_page_table: Optional[torch.Tensor]  # int32, transformed version

    # DSA seqlens
    seqlens_expanded: torch.Tensor  # int32, [expanded_size]
    dsa_cache_seqlens: torch.Tensor  # int32, [expanded_size]
    dsa_cu_seqlens_k: torch.Tensor  # int32, [expanded_size+1]
    seqlens_expanded_size: int

    # Dimensions
    max_len: int  # for decode/draft_extend
    max_seqlen_k: int  # for target_verify

    # FlashMLA (optional)
    flashmla_metadata: Optional[torch.Tensor] = None


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """生成 ragged kernel 使用的 int32 累积序列边界。

    输入 ``[l0, l1, ...]`` 经 cumsum 后在头部补零，得到
    ``[0, l0, l0+l1, ...]``；相邻元素即可确定每条序列的起止位置。
    """
    # 多数 DSA/FlashMLA kernel 的 ABI 固定使用 int32 长度。
    assert seqlens.dtype == torch.int32
    # cumsum 保持设备端计算，pad 在首位加入 ragged offset 的零起点。
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


class DeepseekSparseAttnBackendMTPPrecomputeMixin:
    """为 DSA backend 提供多步推测解码元数据预计算。

    Mixin 依赖宿主 backend 的页表、真实 page size、Top-K 配置及 FlashMLA
    元数据构造函数。它不改变注意力计算，仅把各 replay 实例重复的准备工作
    提升到共享阶段完成。
    """

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> PrecomputedMetadata:
        """为一个实际 batch 预计算所有 MTP replay 共享元数据。

        CUDA Graph buffer 往往按最大 batch 分配，因此首先裁到 ``bs``。
        随后只接受普通 Decode/Idle 或 Target Verify 两种长度语义，并分派到
        对应实现；其他 forward mode 若误入会立即报错。

        Args:
            bs: Batch size
            req_pool_indices: Request pool indices [bs]
            seq_lens: Sequence lengths [bs]
            seq_lens_cpu: Sequence lengths on CPU [bs]
            forward_mode: Forward mode (decode/target_verify)

        Returns:
            PrecomputedMetadata containing all shared intermediate results
        """
        # CUDA Graph 输入可能带静态 padding，所有元数据只能观察真实 batch 前缀。
        seq_lens = seq_lens[:bs]
        # CPU 镜像在纯设备 replay 中可为空；存在时保持与设备长度同样裁剪。
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        # 请求池下标也必须裁剪，否则会从页表读取 padding 请求。
        req_pool_indices = req_pool_indices[:bs]

        # Decode 每个请求通常只有一个新 query，元数据无需按 draft token 展开。
        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        # Target Verify 同时验证多枚 draft token，需要请求级元数据逐 token 展开。
        elif forward_mode.is_target_verify():
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        # Extend 等模式具有不同页表/长度定义，禁止静默复用错误逻辑。
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """预计算普通 Decode/Idle 模式的一 token-per-request 元数据。

        NVIDIA CUDA 使用单个融合 kernel 同时生成长度、累积长度和两种页表；
        ROCm/其他平台使用等价的 PyTorch 回退。可选的 FlashMLA 元数据也在
        此处基于裁剪后的 DSA cache 长度一次生成。
        """
        # 复用当前 batch size 对应的静态 graph 页表宽度，保证 replay 形状一致。
        max_len = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]

        # 仅 NVIDIA CUDA 有 fused_dsa_decode_metadata；HIP 走后面的通用实现。
        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_decode_metadata,
            )

            # 以下输出由融合 kernel 完整覆盖，使用 empty 避免无意义清零。
            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            # page_size=1 页表保留逐 token 物理槽位，供通用路径与转换使用。
            page_indices = torch.empty(
                (bs, max_len), dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            dsa_cu_seqlens_k = torch.empty(
                bs + 1, dtype=torch.int32, device=self.device
            )
            # 分页 kernel 需要按真实 page size 压缩后的页表列数。
            if self.real_page_size > 1:
                real_cols = (max_len + self.real_page_size - 1) // self.real_page_size
                real_page_table = torch.empty(
                    (bs, real_cols), dtype=torch.int32, device=self.device
                )
                real_page_table_arg = real_page_table
            else:
                # 真实页大小为 1 时 page_indices 本身就是 kernel 所需页表。
                real_page_table = None
                real_page_table_arg = page_indices

            # 单次 kernel 遍历请求池映射并填充所有长度/页表输出。
            fused_dsa_decode_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_len=max_len,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
            )
            # Decode 每个请求对应一行 query，因此展开长度与 cache 长度相同。
            seqlens_expanded = cache_seqlens
            seqlens_expanded_size = bs

            # 仅 FlashMLA-KV decode 实现需要额外的 tile 调度元数据。
            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            # 将所有设备 Tensor 打包，后续多个 replay backend 只需引用/复制。
            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=max_len,
                max_seqlen_k=max_len,
                flashmla_metadata=flashmla_metadata,
            )

        # 通用回退首先统一长度 dtype，再构造 ragged KV 的累积边界。
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # req_to_token 保存请求逻辑位置到物理 token slot 的映射。
        page_indices = self.req_to_token[req_pool_indices, :max_len].contiguous()

        # 稀疏 Indexer 最多访问 Top-K 个历史 token，长度需按该上限截断。
        dsa_cache_seqlens = compute_dsa_seqlens(
            cache_seqlens, dsa_index_topk=self.dsa_index_topk
        )
        # Decode 不发生 token 级展开，直接复用请求级 cache 长度。
        seqlens_expanded = cache_seqlens
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # 为 DSA ragged logits/attention kernel 构造自己的累积边界。
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # page_size>1 的 kernel 不能直接消费逐 token 页表，需要转换成物理页号。
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None  # Will use page_indices directly

        # FlashMLA-KV 路径按每请求一个 query 预计算调度信息。
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """预计算 Target Verify 的逐 draft-token 展开元数据。

        每个请求会验证 ``speculative_num_draft_tokens`` 个连续位置，因此页表
        和有效 K 长度都要展开到 ``bs * next_n`` 行。融合 CUDA kernel 与
        通用回退生成相同布局，供每个验证 token 独立执行 DSA Top-K。
        """
        # graph 元数据的页表宽度是验证阶段允许访问的最大 K 长度。
        max_seqlen_k = self.decode_cuda_graph_metadata[bs].page_table_1.shape[1]
        # 每个请求展开 next_n 行，对应依次增长的 draft 验证位置。
        seqlens_expanded_size = bs * self.speculative_num_draft_tokens

        # NVIDIA CUDA 使用融合元数据 kernel；HIP 保留通用 PyTorch/Triton 路径。
        if _is_cuda and not _is_hip:
            from sglang.kernels.ops.attention.dsa_metadata import (
                fused_dsa_target_verify_metadata,
            )

            # 请求级 cache 长度仍只需 bs 行。
            cache_seqlens = torch.empty(bs, dtype=torch.int32, device=self.device)
            cu_seqlens_k = torch.empty(bs + 1, dtype=torch.int32, device=self.device)
            # 页表按每个验证 token 复制，因此第一维使用展开大小。
            page_indices = torch.empty(
                (seqlens_expanded_size, max_seqlen_k),
                dtype=torch.int32,
                device=self.device,
            )
            # 每行记录该 draft 位置可见的实际 K 长度。
            seqlens_expanded = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cache_seqlens = torch.empty(
                seqlens_expanded_size, dtype=torch.int32, device=self.device
            )
            dsa_cu_seqlens_k = torch.empty(
                seqlens_expanded_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # 为真实分页 kernel 预分配压缩页表；page_size=1 时直接复用原表。
            if self.real_page_size > 1:
                real_cols = (
                    max_seqlen_k + self.real_page_size - 1
                ) // self.real_page_size
                real_page_table = torch.empty(
                    (seqlens_expanded_size, real_cols),
                    dtype=torch.int32,
                    device=self.device,
                )
                real_page_table_arg = real_page_table
            else:
                real_page_table = None
                real_page_table_arg = page_indices

            # 融合完成请求查表、next_n 展开、长度裁剪、cumsum 与页表转换。
            fused_dsa_target_verify_metadata(
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                req_to_token=self.req_to_token,
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_table_1=page_indices,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                real_page_table=real_page_table_arg,
                bs=bs,
                max_seqlen_k=max_seqlen_k,
                dsa_index_topk=self.dsa_index_topk,
                real_page_size=self.real_page_size,
                next_n=self.speculative_num_draft_tokens,
            )

            # FlashMLA 针对每个展开行仍视为单 query 进行调度。
            flashmla_metadata = None
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )

            return PrecomputedMetadata(
                cache_seqlens=cache_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                page_indices=page_indices,
                real_page_table=real_page_table,
                seqlens_expanded=seqlens_expanded,
                dsa_cache_seqlens=dsa_cache_seqlens,
                dsa_cu_seqlens_k=dsa_cu_seqlens_k,
                seqlens_expanded_size=seqlens_expanded_size,
                max_len=-1,
                max_seqlen_k=max_seqlen_k,
                flashmla_metadata=flashmla_metadata,
            )

        # Target 模型执行时 draft token 已进入可见 KV，因此请求级长度加 next_n。
        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        # 请求级累积长度仍供主 attention 元数据使用。
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # 先按请求读取页表，再为该请求的每个 draft token 重复一行。
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()

        # Generate expanded seqlens on device. seq_lens_cpu is optional for DSA
        # CUDA graph replay, so this fallback must not require a host mirror.
        # 每个请求具有固定 next_n 个待验证 token，完全在设备端构造长度。
        extend_seq_lens = torch.full(
            (bs,),
            self.speculative_num_draft_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # 展开后同一请求内各行的可见 K 长度随 draft 位置递增。
        seqlens_expanded = seqlens_expand_triton(
            extend_seq_lens,
            cache_seqlens,
            bs * self.speculative_num_draft_tokens,
            self.speculative_num_draft_tokens,
        )

        # 每个验证位置再按 Indexer Top-K 上限裁剪有效历史范围。
        dsa_cache_seqlens = compute_dsa_seqlens(seqlens_expanded, self.dsa_index_topk)
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # 构造展开后 ragged DSA 序列的累积边界。
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # 需要时把逐 token slot 页表转换为真实物理页表。
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA-KV 以展开后的逐 query DSA 长度生成调度信息。
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,  # Not used in this mode
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
        )


# 保留旧导入名绑定，避免外部代码因 mixin 重构而失效。
DeepseekSparseAttnBackendMTPPrecomputeMixin = (
    DeepseekSparseAttnBackendMTPPrecomputeMixin
)
