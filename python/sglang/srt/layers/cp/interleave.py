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

"""Interleave context parallel strategy shell.

For ``cp_size = 4``, each rank owns every fourth token:

    cp0: token0, token4, token8,  token12, token16, ...
    cp1: token1, token5, token9,  token13, token17, ...
    cp2: token2, token6, token10, token14, token18, ...
    cp3: token3, token7, token11, token15, token19, ...

After all-gather, tokens are restored to the original order:

    token0, token1, token2, token3, token4, token5, token6, token7, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

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
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None


class InterleaveCPStrategy(ContextParallelStrategy):
    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        """检查当前 prefill 是否满足轮询 CP 的最小几何条件。

        Interleave 只服务 context-parallel extend；decode 或普通 prefill 没有需要
        沿 token 维切分的完整上下文。至少每个 CP rank 分到一个 token 才有意义，
        因而总 extend token 数必须不小于 ``cp_size``。
        """
        # 只允许 prefill extend 进入 DSA 的 interleave CP 路径。
        if not forward_batch.forward_mode.is_context_parallel_extend():
            return False
        # 使用 attention-CP 规模，而不是总 GPU 数或 DP 规模。
        cp_size = self.cp_size
        # 多请求 batch 已平铺，按每条请求 extend 长度求总逻辑 token。
        seq_len = sum(forward_batch.extend_seq_lens_cpu)
        return seq_len > 0 and seq_len >= cp_size and cp_size > 1

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> InterleaveContextParallelMetadata:
        """构造轮询 token 布局所需的每-rank 长度元数据。

        全局第 ``i`` 个 token 属于 ``i % cp_size`` rank，因此各 rank 的逻辑长度
        仅取决于总 token 数除以 CP 大小后的商和余数。调度产生的尾部 padding 会被
        归入最后一个请求，保证 metadata 覆盖输入张量实际携带的所有行。
        """
        # 缺少逐请求 extend 长度时，把整个平铺输入看成一条序列。
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        # 转为 Python int，避免后续 divmod/list 操作产生 device 同步。
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        # 使逐请求长度之和与实际输入 token 行数一致。
        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len

        # 轮询布局下前 ``extra`` 个 rank 各多获得一个尾部 token。
        total_seq_lens = sum(extend_seqs_len)
        base_len, extra = divmod(total_seq_lens, self.cp_size)
        per_rank_actual_token = [
            base_len + (rank < extra) for rank in range(self.cp_size)
        ]

        return InterleaveContextParallelMetadata(
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=[max(per_rank_actual_token)] * self.cp_size,
            total_seq_lens=total_seq_lens,
            bs=len(extend_seqs_len),
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        """按全局 token 下标取模，把 hidden states 发给当前 CP rank。

        先裁掉调度外的尾部行，再取 ``cp_rank, cp_rank + cp_size, ...``；最后补齐
        到 collective 的物理长度。这样 DSA attention 收到的是本 rank 的稀疏、
        轮询 token 序列，而通信层仍可使用固定张量形状。
        """
        # metadata 的 total_seq_lens 是本次请求的真实逻辑 token 上界。
        metadata = forward_batch.attn_cp_metadata
        local_x = self._interleave_shard(x[: metadata.total_seq_lens])
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        """使用与 hidden states 相同的轮询索引切分 position IDs。

        位置必须和 Q/K/V token 行严格同序，否则 RoPE 会写入错误 KV 槽位。返回值
        同样补齐为物理长度，使 position 张量可直接与局部 hidden states 配对。
        """
        metadata = forward_batch.attn_cp_metadata
        local_positions = self._interleave_shard(positions[: metadata.total_seq_lens])
        return pad_local_rows(local_positions, metadata, dim=0)

    def _interleave_shard(self, input_: Any) -> Any:
        """实现通用的第 ``cp_rank`` 路轮询切分。

        list/tuple 用 Python range 保持容器类型的索引语义；tensor 在可整除时采用
        view 后选列，避免显式索引张量。长度不整除时需要 arange，以正确处理前几个
        rank 多一个 token 的不均衡尾部；零长度分支避免生成不合法的 view。
        """
        # CP rank 决定本地保留全局 token 下标的同余类。
        cp_size = self.cp_size
        cp_rank = self.cp_rank
        # 非 tensor 元数据使用 Python 下标序列直接筛选。
        if isinstance(input_, (tuple, list)):
            indices = range(cp_rank, len(input_), cp_size)
            return input_[indices]

        # 不整除时各 rank 长度不同，不能 reshape 成规则二维矩阵。
        tokens = len(input_)
        if tokens % cp_size != 0:
            # rank 较小时获得一个余数 token。
            cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
            if cur_len == 0:
                return input_.new_empty(0, *input_.shape[1:])
            # 构造当前同余类的全局 token 下标并做 gather。
            indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
            return input_[indices]

        # 整除时把 token 维视为 [round, rank]，选 rank 列最省开销。
        return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()

    def shard_local_tokens(self, input_: Any) -> Any:
        return self._interleave_shard(input_)

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ):
        """按请求构造 DSA attention 所需的局部 query 长度与有效请求索引。

        Python 侧先模拟连续平铺 token 的余数传递，得到每请求的局部长度；GPU 侧再
        运行同一规则的 Triton kernel 填充 device tensor，从而可被 CUDA graph 使用。
        空请求会被过滤，并通过 ``bs_idx`` 保留其在原 batch 中的位置映射。
        """
        from sglang.kernels.ops.attention.dsa.cp_split import (
            dsa_cp_round_robin_split_q_seqs_kernel,
        )

        cp_size = self.cp_size
        cp_rank = self.cp_rank

        # ``extra_seq`` 将前一请求未凑满一轮 CP 的 token 传递到下一请求。
        extra_seq = 0
        q_lens_cpu: List[int] = []
        for cur_len in extend_seqs_cpu:
            cur_len += extra_seq
            # 当前 rank 在本请求（含前序余数）中获得的 token 数。
            cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
            q_lens_cpu.append(cur_seq)
            extra_seq = cur_len - cur_seq * cp_size
        # DSA kernel 只接收有本地 query 的请求，并保存原 batch 下标。
        bs_idx_cpu = [i for i, q_len in enumerate(q_lens_cpu) if q_len > 0]
        q_lens_cpu = [q_len for q_len in q_lens_cpu if q_len > 0]

        # 预分配与 CPU 结果等长的 device 输出，避免在 graph 内动态分配。
        q_lens = torch.empty(
            (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
        )
        bs_idx = torch.empty(
            (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
        )
        # Triton 实现重复 CPU 规则，提供给后续 GPU attention metadata。
        dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
            extend_seqs, q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
        )
        return q_lens_cpu, q_lens, bs_idx_cpu, bs_idx

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._gather_interleaved_tensor(x, forward_batch)

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._gather_interleaved_tensor(x, forward_batch)

    def _gather_interleaved_tensor(self, x: Any, forward_batch) -> Any:
        """all-gather 各 rank 的轮询局部张量，并恢复全局 token 递增顺序。

        collective 需要等长输入，所以先按物理长度补零；收集结果的布局为
        ``[rank0 rows, rank1 rows, ...]``。最后用 ``global_index % cp_size`` 计算
        owner rank、用整除计算该 rank 内行号，从而一次 index_select 完成反交织。
        """
        # gather 依赖 prepare_cp_forward 构建的 logical/physical 长度。
        metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if metadata is None:
            raise RuntimeError("Interleave CP gather requires attn_cp_metadata.")

        # 总逻辑长度决定最终输出，不能接受损坏的负值 metadata。
        total_tokens = int(metadata.total_seq_lens)
        if total_tokens < 0:
            raise RuntimeError(
                f"Invalid interleave CP total_seq_lens={total_tokens}; expected >= 0."
            )

        # logical 长度用于去 padding；旧 metadata 则回退到 actual 长度。
        logical_rank_lens = (
            metadata.per_rank_logical_token or metadata.per_rank_actual_token
        )
        # 当前局部输入至少要包含本 rank 的有效 token 行。
        local_logical_len = logical_rank_lens[self.cp_rank]
        if x.shape[0] < local_logical_len:
            raise RuntimeError(
                "Interleave CP gather received an unexpected local token count: "
                f"rank={self.cp_rank}, got={x.shape[0]}, "
                f"expected_at_least={local_logical_len}, "
                f"total={total_tokens}, cp_size={self.cp_size}."
            )

        # physical 长度是 collective 输入长度，所有 rank 均相等。
        physical_rank_len = max(metadata.per_rank_actual_token)
        if physical_rank_len == 0:
            return x.new_empty((0, *x.shape[1:]))

        # 先构造等长零填充本地 buffer，避免把无效行参与输出。
        padded_x = x.new_zeros((physical_rank_len, *x.shape[1:]))
        padded_x[:local_logical_len] = x[:local_logical_len]

        # CUDA 时申请对称内存，适配 NVSHMEM/NCCL 的高效通信路径。
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            gathered = x.new_empty((self.cp_size * physical_rank_len, *x.shape[1:]))
        # 使用 DP-attention 包装的 CP collective，处理其运行时通信后端。
        attn_cp_all_gather_into_tensor(gathered, padded_x.contiguous())

        # 枚举输出中每个全局 token 的递增下标。
        flat_indices = torch.arange(total_tokens, device=x.device)
        # 将全局下标映射到 all-gather 扁平布局中的 ``rank * physical + row``。
        gather_indices = (
            flat_indices % self.cp_size
        ) * physical_rank_len + flat_indices // self.cp_size
        return gathered.index_select(0, gather_indices)

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.DSA]

    def materialize_full_indexer_k_cache(self, key: Any, forward_batch) -> Any:
        return self.gather_kv_cache(
            key.contiguous(), forward_batch, torch.cuda.current_stream()
        )

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
        **kwargs,
    ) -> Any:
        # No-op: run_attention is the FlashAttention/zigzag dispatch hook.
        # Interleave serves the DSA backend, which runs attention itself.
        return None

    def all_gather_dsa_trtllm_fp8_kv(self, forward_batch, k: Any, k_rope: Any) -> Any:
        """收集 FP8 DSA 的 MLA KV，并恢复 K-nope/K-rope 两个分量。

        部分通信实现无法直接可靠处理 FP8 元素类型，因此先把拼接后的 KV 重解释为
        uint8 字节流再复用普通 CP gather。gather 后恢复原 dtype 并依据原末维大小
        split，避免任何数值转换或额外量化。
        """
        # 记录分割点，因 bytes view 会隐藏原始 feature 维。
        kv_lora_rank = k.shape[-1]
        qk_rope_head_dim = k_rope.shape[-1]
        kv_dtype = k.dtype
        # Pack → gather in raw bytes to avoid dtype issues with FP8
        # 拼接两个分量并按字节通信，规避 FP8 collective dtype 限制。
        kv = torch.cat((k, k_rope), dim=-1).view(torch.uint8)
        kv = self.gather_kv_cache(
            kv.contiguous(), forward_batch, torch.cuda.current_stream()
        ).view(kv_dtype)
        return kv.split((kv_lora_rank, qk_rope_head_dim), dim=-1)

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} strategy does not support dense K/V materialization"
        )

    def materialize_full_mla_kv(
        self,
        forward_batch,
        layer: Any,
        k_nope: Any,
        k_rope: Any,
    ) -> Any:
        """把 interleave 局部 MLA latent KV 收集后还原为缓存接口形状。

        MLA cache 接口期望 ``[token, 1, dim]``，而 gather 为减少通信维度先处理
        squeeze 后的 ``[token, dim]``。收集完成后按 ``kv_lora_rank`` 切回 nope/rope
        并重新插入 head 维，以保持下游缓存池的既有约定。
        """
        # 保留 nope 的末维，作为拼接 latent 的可逆分割点。
        kv_lora_rank = k_nope.shape[-1]
        latent_cache = torch.cat([k_nope, k_rope], dim=-1).squeeze(1)
        full_latent = self.gather_kv_cache(
            latent_cache.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        k_nope = full_latent[..., :kv_lora_rank].unsqueeze(1)
        k_rope = full_latent[..., kv_lora_rank:].unsqueeze(1)
        return k_nope, k_rope
