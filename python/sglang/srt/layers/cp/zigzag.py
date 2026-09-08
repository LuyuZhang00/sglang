# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Zigzag context parallel strategy shell.

For ``cp_size = 4``, each sequence is split into ``2 * cp_size`` blocks. Each
rank owns one early block and one late block:

    cp0: block0, block7
    cp1: block1, block6
    cp2: block2, block5
    cp3: block3, block4

After all-gather, the blocks are reranged back to their original order:

    block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4
      -> block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.padding import pad_local_rows
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_device, get_parallel


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None

    # Per-rank aggregate lists have length cp_size.
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs + 1]).
    kv_len_prev_tensor: Optional[Any] = None
    kv_len_next_tensor: Optional[Any] = None
    cu_seqlens_kv_prev_tensor: Optional[Any] = None
    cu_seqlens_kv_next_tensor: Optional[Any] = None
    actual_seq_q_prev_tensor: Optional[Any] = None
    actual_seq_q_next_tensor: Optional[Any] = None
    cu_seqlens_q_prev_tensor: Optional[Any] = None
    cu_seqlens_q_next_tensor: Optional[Any] = None

    # Combined prev-then-next TRT-LLM geometry (shape [2 * bs] or [2 * bs + 1]).
    actual_seq_q_combined_tensor: Optional[Any] = None
    kv_len_combined_tensor: Optional[Any] = None
    cu_seqlens_q_combined_tensor: Optional[Any] = None
    cu_seqlens_kv_combined_tensor: Optional[Any] = None

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0
    max_seqlen_q_combined: int = 0

    # Per-sequence CPU lists, useful for indexers and diagnostics.
    kv_len_prev_list: Optional[List[int]] = None
    kv_len_next_list: Optional[List[int]] = None
    actual_seq_q_prev_list: Optional[List[int]] = None
    actual_seq_q_next_list: Optional[List[int]] = None


ContextParallelMetadata = ZigzagContextParallelMetadata


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        """检查 zigzag 双端分块是否能覆盖当前所有 extend 序列。

        每个 CP rank 需要一个前向 block 和一个后向 block，因此全局及每条请求都
        至少应有 ``2 * cp_size`` 个 token。短请求保留在非 CP 路径，避免空 block
        造成 attention metadata 不一致；没有逐请求长度的场景只检查全局长度。
        """
        # 单 rank 或不足两个 CP block 时没有有效 zigzag 切分。
        if self.cp_size <= 1 or num_tokens < self.cp_size * 2:
            return False
        # zigzag 只在 context-parallel extend prefill 运行。
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        # 缺少逐请求信息时只能按全局长度决定是否可用。
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True
        return all(int(length) >= self.cp_size * 2 for length in extend_lens)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        """构造 zigzag 的 token 分块、attention 几何和 gather 反向重排索引。

        每条序列先均分为 ``2 * cp_size`` 个连续 block；rank ``r`` 持有第 ``r``
        个前块和对称的后块。metadata 同时编码 FlashAttention/TRT-LLM 所需的前后
        query/KV 长度与 cumulative seqlens，使 attention 无需重新理解全局布局。
        """
        # 没有逐请求 extend 长度时把平铺 token 当成单条序列。
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        # 统一为 CPU Python int，后续列表索引和长度计算不触发 GPU 同步。
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        # 将 scheduler 填充并入最后一条请求，保证 split 覆盖全部输入行。
        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len
            if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
                seqs_len = list(seqs_len)
                seqs_len[-1] += pad_len

        # 每条序列被切为一前一后共两轮 CP block。
        bs = len(extend_seqs_len)
        cp_segment_num = self.cp_size * 2
        # 若已知完整序列长度，记录 extend 段在完整 KV 序列中的前缀偏移。
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        # TODO: move these per-request layout/index computations to a Triton
        # kernel if Python-side metadata construction becomes a bottleneck.
        # 对每条序列做尽量均匀的连续分块，余数从低 block id 起分配。
        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in extend_seqs_len:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            # 每个 block 的尺寸只相差至多一个 token。
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        # 一个 rank 聚合自己前 block 和镜像后 block 的真实 token 数。
        per_rank_actual_token = []
        for rank in range(self.cp_size):
            per_rank_actual_token.append(
                sum(
                    block_sizes[rank] + block_sizes[cp_segment_num - 1 - rank]
                    for block_sizes in per_seq_block_sizes
                )
            )
        # 所有 rank 的物理 buffer 初始按最大真实长度规划，padding helper 会最终对齐。
        max_rank_len = [max(per_rank_actual_token)] * self.cp_size

        # 本 rank 的输出顺序为所有请求的前块，随后是所有请求的镜像后块。
        cp_rank = self.cp_rank
        zigzag_index = list(
            range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
        ) + list(
            range(
                cp_segment_num - cp_rank - 1,
                bs * cp_segment_num,
                cp_segment_num,
            )
        )

        # all-gather 后 chunk 的顺序按 rank 聚合；该索引将其恢复为全局序列顺序。
        cp_reverse_index: List[int] = []
        for batch_id in range(bs):
            cp_reverse_index.extend(
                list(range(batch_id, cp_segment_num * bs, 2 * bs))
                + list(
                    range(
                        (cp_segment_num - 1) * bs + batch_id,
                        0,
                        -2 * bs,
                    )
                )
            )

        # 与 gather 后各 chunk 对应的长度，顺序必须匹配 rank-major 输出布局。
        reverse_split_len: List[int] = []
        for rank in range(self.cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][rank])
            for batch_id in range(bs):
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - rank]
                )

        # 为前/后 attention 分别计算每条请求可见的 KV 长度及本地 query 长度。
        kv_len_prev_list: List[int] = []
        kv_len_next_list: List[int] = []
        actual_seq_q_prev_list: List[int] = []
        actual_seq_q_next_list: List[int] = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - cp_rank])
            )
            actual_seq_q_prev_list.append(block_sizes[cp_rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - cp_rank - 1])

        # 运行时优先采用 SGLang 当前 device；独立测试时安全回退。
        try:
            device = torch.device(get_device().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FlashAttention 的 cumulative seqlens 从零起始，组合版供 TRT-LLM 一次调用。
        cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
        cu_next = [0] + list(accumulate(actual_seq_q_next_list))
        cu_kv_prev = [0] + list(accumulate(kv_len_prev_list))
        cu_kv_next = [0] + list(accumulate(kv_len_next_list))
        actual_seq_q_combined_list = actual_seq_q_prev_list + actual_seq_q_next_list
        kv_len_combined_list = kv_len_prev_list + kv_len_next_list
        cu_q_combined = [0] + list(accumulate(actual_seq_q_combined_list))
        cu_kv_combined = [0] + list(accumulate(kv_len_combined_list))

        # 这些断言锁定分块、索引和 token 守恒，防止错误 metadata 进入 GPU kernel。
        total_seq_lens = sum(extend_seqs_len)
        assert len(split_list) == bs * cp_segment_num
        assert sum(split_list) == total_seq_lens
        assert len(zigzag_index) == 2 * bs
        assert len(cp_reverse_index) == bs * cp_segment_num
        assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
        assert sum(per_rank_actual_token) == total_seq_lens

        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            kv_len_prev_tensor=torch.tensor(
                kv_len_prev_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=torch.tensor(
                kv_len_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_prev_tensor=torch.tensor(
                cu_kv_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_next_tensor=torch.tensor(
                cu_kv_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_prev_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=torch.tensor(
                actual_seq_q_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=torch.tensor(
                cu_next, device=device, dtype=torch.int32
            ),
            actual_seq_q_combined_tensor=torch.tensor(
                actual_seq_q_combined_list, device=device, dtype=torch.int32
            ),
            kv_len_combined_tensor=torch.tensor(
                kv_len_combined_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_combined_tensor=torch.tensor(
                cu_q_combined, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_combined_tensor=torch.tensor(
                cu_kv_combined, device=device, dtype=torch.int32
            ),
            total_q_prev_tokens=cu_prev[-1],
            total_q_next_tokens=cu_next[-1],
            max_seqlen_q_prev=(
                max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
            ),
            max_seqlen_q_next=(
                max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
            ),
            max_seqlen_q_combined=(
                max(actual_seq_q_combined_list) if actual_seq_q_combined_list else 0
            ),
            kv_len_prev_list=kv_len_prev_list,
            kv_len_next_list=kv_len_next_list,
            actual_seq_q_prev_list=actual_seq_q_prev_list,
            actual_seq_q_next_list=actual_seq_q_next_list,
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        """切出当前 rank 的前后 zigzag block，并补到统一物理长度。

        输入先依各序列/各 block 的 ``split_list`` 切成连续块，再按 ``zigzag_index``
        选取当前 rank 的前块和镜像后块。局部顺序与 build_metadata 生成的 attention
        geometry 一致，尾部 padding 仅用于固定 collective / kernel 形状。
        """
        # 忽略调度超过真实请求范围的输入尾行。
        metadata = forward_batch.attn_cp_metadata
        x = x[: metadata.total_seq_lens]
        # 先按全局 block 边界拆分，再挑选当前 CP rank 拥有的块。
        chunks = torch.split(x, metadata.split_list, dim=0)
        local_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        """按 zigzag token 布局切分位置编码，并在 token 维补零。

        positions 的 token 维位于最后一维，因而 split/cat 与 hidden states 的第 0
        维操作相同但维度不同。这样前后两段 query 使用的 RoPE position 与它们在
        全局序列中的原始位置保持一致。
        """
        metadata = forward_batch.attn_cp_metadata
        positions = positions[..., : metadata.total_seq_lens]
        chunks = torch.split(positions, metadata.split_list, dim=-1)
        local_positions = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=-1)
        return pad_local_rows(local_positions, metadata, dim=-1)

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        """收集局部 hidden states，并把 rank-major zigzag 布局恢复为全局顺序。

        通信 helper 先删除每 rank 的物理 padding，再按 rank 顺序拼接有效行；本函数
        随后依据 ``reverse_split_len`` 重建 block，并用 ``cp_reverse_index`` 反排。
        该顺序正是 logits processor 与上游全局 input_ids 的 token 顺序。
        """
        gathered = self._all_gather_reorganized(x, forward_batch)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
        )

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        """以与 hidden states 相同的重排规则收集 KV/indexer payload。

        KV 的 token 维也遵循 zigzag token 所有权，因此通信和反向重排不可省略；
        唯一不同是调用时机通常在缓存物化阶段。保留独立函数使策略基类能清晰表达
        logits 输出与缓存数据的两个使用场景。
        """
        gathered = self._all_gather_reorganized(x, forward_batch)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
        )

    def get_supported_attention_backend(self):
        return [
            CPAttentionBackendKind.FLASH_ATTENTION,
            CPAttentionBackendKind.TRTLLM_MHA,
        ]

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        """按后端约定执行 zigzag 局部 attention，并保留物理 padding 行。

        FlashAttention 需要将前/后 block 分两次调用，因为二者可见 KV 长度不同；
        TRT-LLM MHA 支持组合后的 metadata，可一次处理两段。attention 的有效输出
        后补零到输入 ``q`` 的物理长度，保持后续层张量形状稳定。
        """
        # 策略只实现其 metadata 能正确表达的两类 attention 后端。
        assert (
            attention_backend in self.get_supported_attention_backend()
        ), f"{self.name} CP does not support {attention_backend=}"

        # 局部 q 的逻辑部分按“所有前块、所有后块”连续排布。
        meta = forward_batch.attn_cp_metadata
        q_prev = q[: meta.total_q_prev_tokens]
        logical_tokens = meta.total_q_prev_tokens + meta.total_q_next_tokens
        q_next = q[meta.total_q_prev_tokens : logical_tokens]

        prev_kwargs = {}
        next_kwargs = {}
        # TRT-LLM 可接受前后拼接的 cumulative seqlens 和 zigzag page table。
        if attention_backend == CPAttentionBackendKind.TRTLLM_MHA:
            result = attn_fn(
                q[:logical_tokens],
                meta.cu_seqlens_q_combined_tensor,
                meta.kv_len_combined_tensor,
                meta.max_seqlen_q_combined,
                cu_seqlens_kv=meta.cu_seqlens_kv_combined_tensor,
                use_zigzag_page_table=True,
            )
        else:
            # FlashAttention 分别处理前段与后段，以匹配各自因果可见 KV 长度。
            result_prev = attn_fn(
                q_prev,
                meta.cu_seqlens_q_prev_tensor,
                meta.kv_len_prev_tensor,
                meta.max_seqlen_q_prev,
                **prev_kwargs,
            )
            result_next = attn_fn(
                q_next,
                meta.cu_seqlens_q_next_tensor,
                meta.kv_len_next_tensor,
                meta.max_seqlen_q_next,
                **next_kwargs,
            )
            result = torch.cat([result_prev, result_next], dim=0)

        # 输入可能已经被 pad_local_rows 补齐，输出也必须保留同样的尾部零行。
        pad_size = q.shape[0] - logical_tokens
        assert pad_size >= 0
        if pad_size > 0:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        """收集 zigzag 局部 K/V 后按全局 token 顺序写入普通 KV cache。

        K 和 V 先沿 feature 维拼接以共用一次 all-gather/重排，随后按原末维切开；
        cache location 以真实 token 数为界，SWA location 若存在也同步裁剪。最终由
        token-to-KV pool 写入，调用层无需感知 CP 的局部 token 所有权。
        """
        # cross-attention 使用 encoder cache location，其余层使用 decoder location。
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        # 只保留与本次真实 cache 写入 token 对应的 SWA location。
        if swa_loc is not None:
            swa_loc = swa_loc[: cache_loc.shape[0]]
        # 合并 K/V 可将两份同布局数据合并为一次 CP collective。
        k_dim = k.shape[-1]
        v_dim = v.shape[-1]
        kv_cache = torch.cat([k, v], dim=-1).contiguous()
        # 反交织后沿 feature 维恢复 K、V，并保证缓存写入需要的连续内存。
        key_cache_full, value_cache_full = self.gather_kv_cache(
            kv_cache, forward_batch
        ).split([k_dim, v_dim], dim=-1)
        key_cache_full = key_cache_full.contiguous()
        value_cache_full = value_cache_full.contiguous()
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def materialize_full_mla_kv(
        self, forward_batch, layer: Any, k_nope: Any, k_rope: Any
    ) -> None:
        """收集 MLA 的 nope/rope latent KV，并写入 MLA 专用缓存池。

        MLA 缓存将两个分量视作同一 token 布局的 latent 向量，先拼接可复用 zigzag
        gather 的通信和重排。收集完按 ``kv_lora_rank`` 分开，再交给缓存池保持其
        内部的 MLA 槽位格式。
        """
        # nope 的 feature 维是可逆切分点。
        kv_lora_rank = k_nope.shape[-1]
        latent = torch.cat([k_nope, k_rope], dim=-1).contiguous()
        latent_full = self.gather_kv_cache(latent, forward_batch)
        get_token_to_kv_pool().set_mla_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            latent_full[..., :kv_lora_rank],
            latent_full[..., kv_lora_rank:],
        )

    def _all_gather_reorganized(self, x: torch.Tensor, forward_batch):
        """执行等长 CP all-gather，并移除 padding 形成 rank-major 有效 token 流。

        每个 rank 的真实长度可能不同，collective 前先补到最大物理长度；收集后再
        按 metadata 裁掉各 rank 的无效尾行。调用者可把结果按 ``reverse_split_len``
        切块并重排，而无需处理通信形状与逻辑 token 数的差异。
        """
        # logical token 长度存在时优先使用，以剥离为 collective 添加的 padding。
        meta = forward_batch.attn_cp_metadata
        per_rank_token = meta.per_rank_logical_token or meta.per_rank_actual_token
        max_len = max(per_rank_token)
        # 输入可包含物理 padding，先仅保留当前 rank 的有效逻辑前缀。
        if per_rank_token == meta.per_rank_actual_token:
            local_len = x.shape[0]
        else:
            local_len = per_rank_token[self.cp_rank]
        assert x.shape[0] >= local_len
        x = x[:local_len]
        # collective 需要所有 rank 输入等长，较短 rank 在尾部补零。
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

        # CUDA/NVSHMEM 路径申请对称内存；CPU 测试路径使用空上下文。
        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        # 结果按 rank-major 固定块排列，每块长度为 max_len。
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                device=x.device,
                dtype=x.dtype,
            )
        group.all_gather_into_tensor(gathered, x)

        # 逐块裁剪到各 rank 有效长度，保留 rank-major 供调用者后续重排。
        chunks = torch.split(gathered, [max_len] * self.cp_size, dim=0)
        return torch.cat(
            [
                chunks[rank][:per_rank_len]
                for rank, per_rank_len in enumerate(per_rank_token)
            ],
            dim=0,
        )
