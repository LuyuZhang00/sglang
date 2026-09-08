"""DSA Indexer 与注意力后端之间的元数据适配层。

Indexer 只依赖本文件定义的抽象访问接口，而不直接绑定某一种 attention
backend 的内部字段。具体实现同时承载序列长度、页表以及 Top-K 结果变换
所需的信息，从而让稀疏索引结果能被后续注意力 kernel 直接消费。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata


class BaseIndexerMetadata(ABC):
    """Indexer 所需元数据的最小接口，供不同 attention backend 实现。"""

    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 1.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache(token,xxx) for each token.
        """

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_indexer_seq_len(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_dsa_extend_len_cpu(self) -> List[int]:
        """
        Return: extend seq lens for each batch.
        """

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """
        Return: batch idx for each token.
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


@dataclass(frozen=True)
class DSAIndexerMetadata(BaseIndexerMetadata):
    """标准 DSA backend 的不可变 Indexer 元数据视图。

    该对象不复制张量，只把 ``DSAMetadata`` 中的字段暴露给 Indexer。
    Top-K backend 和变换方式也随元数据绑定，保证 logits 选择结果与
    当前注意力后端约定的索引布局一致。
    """

    # attention backend 构造的原始运行时元数据。
    attn_metadata: DSAMetadata
    # Top-K 结果应采用的索引布局变换方式。
    topk_transform_method: TopkTransformMethod
    # 执行 Top-K 选择/融合变换的算子后端。
    topk_backend: DSATopKBackend = DSATopKBackend.SGL_KERNEL
    # 分页 MQA kernel 可复用的调度元数据，未启用对应路径时为空。
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    # 分页 MQA 使用的二维上下文长度视图。
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None
    # 调试或兼容场景下强制关闭融合 Top-K。
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.dsa_seqlens_expanded

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.dsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """执行 Top-K，并转换成 attention backend 约定的索引布局。

        调用方可以传入局部 query 长度或直接覆盖累积偏移；否则复用
        ``DSAMetadata`` 在 batch 构建阶段预计算的值。最终选择的长度、
        行起点及偏移会统一交给 ``DSATopKBackend`` 完成融合或非融合计算。
        """
        # 显式偏移用于 CP/切片等已自行计算局部布局的调用路径。
        if topk_indices_offset_override is not None:
            # 直接采用调用者给出的偏移，避免再次按完整 batch 推导。
            cu_topk_indices_offset = topk_indices_offset_override
            # 已有精确偏移时，融合变换不再需要 query 累积长度。
            cu_seqlens_q_topk = None
        # 传入 query 长度时，为当前子 batch 动态构造偏移。
        elif cu_seqlens_q is not None:
            # 后端 kernel 的长度 ABI 统一要求 int32。
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            # 得到每个请求在扁平 query 张量中的累积起点。
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            # 把请求级起点扩展到 token 级，供每一行 Top-K 索引做归一化。
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
                # Avoid reading sum(cu_seqlens_q) back to the host.
                output_size=logits.shape[0],
            )
        # 常规路径直接复用 attention backend 已预计算的 batch 元数据。
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        # 某些分块/CP 路径会传入当前块的实际 key 结束位置作为有效长度。
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        # 否则使用按 query token 展开的 DSA 上下文长度。
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        # 后端同时负责选择 Top-K 和必要的索引布局变换。
        return self.topk_backend.topk_transform(
            logits=logits,
            lengths=seq_lens_topk,
            topk=topk,
            topk_transform_method=self.topk_transform_method,
            attn_metadata=self.attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            topk_indices_offset=cu_topk_indices_offset,
            row_starts=ks,
            batch_idx_list=batch_idx_list,
            force_unfused_topk=self.force_unfused_topk,
        )
