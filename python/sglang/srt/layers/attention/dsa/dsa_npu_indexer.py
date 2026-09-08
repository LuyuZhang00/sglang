"""DSA Indexer 的昇腾 NPU 实现。

本模块负责生成带 RoPE 的 query/index-key、写入独立 index-K cache，并调用
``npu_lightning_indexer`` 选出稀疏 Top-K。它还处理 NPU 多流重叠、TP
all-gather 以及 Prefill Context Parallel 的双半区负载均衡。
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


class DSANPUIndexerMixin:
    """为通用 ``Indexer`` 注入 NPU 专用 forward 与 CP 辅助实现。"""

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """执行 NPU 上的一次 DSA 索引计算。

        该流程并行计算 query 与 head gate，构造并缓存当前 token 的 index-K，
        随后根据 Prefill/Decode/推测模式准备真实序列长度。普通路径调用一次
        Lightning Indexer；Prefill CP 路径拆成前后两半以平衡各 rank 负载。
        """
        # 后端可能已经缓存 CPU 长度；优先使用它可减少设备同步。
        if get_attn_backend().forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = get_attn_backend().forward_metadata.seq_lens_cpu_int
        # draft extend v2 与 target verify 有独立长度语义，不视作普通 Prefill。
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        )

        # q_lora 第一维即本次实际处理的 token 数。
        bs = q_lora.shape[0]

        # NeoX RoPE 的 cos/sin 排列与标准 interleaved 路径不同，需要手工处理。
        if self.rotary_emb.is_neox_style:
            # 同一 ForwardBatch 的所有层共享位置缓存，首层构造一次即可。
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                # 按 token position 取 RoPE 表，并拆成 cosine/sine 两部分。
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                # NPU rotary_mul 需要显式的 token/head/rope_dim 广播布局。
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                # 挂到 batch 上供后续 Indexer 层复用。
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            # alt_stream 存在时，让 Q 投影和主流上的 K/weights 计算重叠。
            if self.alt_stream is not None:
                # 先等待主流完成 q_lora 生产，避免跨流读到未就绪数据。
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    # 动态量化线性层以 (Tensor, scale) 形式接收输入。
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    # 从低秩 query 表示投影到所有 Indexer heads。
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    # 仅 q_pe 子空间应用 RoPE，q_nope 保持内容特征不变。
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    # 使用 NPU 融合算子旋转位置子空间。
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    # 恢复 Indexer query 的完整 head_dim。
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    # 告知 allocator q 仍被 alt_stream 使用，防止过早复用内存。
                    q.record_stream(self.alt_stream)
                    # 记录事件，主流在调用 Indexer kernel 前精确等待。
                    q_rope_event = self.alt_stream.record_event()
            else:
                # 单流路径保持相同数学过程，只省略事件同步。
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            # head gate 投影可放到独立 weight stream 与 K 投影重叠。
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                # 确保该流看到主流已经生产完成的 x。
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    # weights_proj 使用 FP32 输入保证 gate 精度，随后转 BF16 供算子消费。
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            # 每个 token 只生成一个共享 index-key head。
            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            # K 投影后归一化，稳定 query-key logits 的数值范围。
            k = self.k_norm(k_proj)
            # QLoRA 后仍为 scattered 布局时，K 必须先恢复到完整 attention TP 域。
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            # K 与 Q 使用一致的位置/非位置子空间划分。
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            # index-key 的位置子空间同样施加 RoPE。
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            # Interleaved RoPE 路径复用 RotaryEmbedding 自身的联合 Q/K 实现。
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            # 首层保存本 batch 的 sin/cos 片段，供 NPU 后续融合算子复用。
            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            # 一次调用同时旋转 Q/K，确保二者采用完全相同的位置编码。
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        # Prefill CP 下每个 rank 先生成局部 K，再 all-gather 并恢复全局 token 顺序。
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        # index-K 使用独立于主 MLA KV 的缓存，由 token slot 映射定位写入位置。
        get_token_to_kv_pool().set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )
        # 根据 forward 模式构造 Lightning Indexer 所需的累积 Q/K 长度。
        if is_prefill:
            if (
                self.dsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                # CP 平衡布局分别保存前半段与后半段 query 的累积长度。
                get_attn_backend().forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                # 命中前缀缓存时，K 有效长度还需加上每个请求的 prefix 长度。
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_kv
                )
            else:
                # 普通 Prefill 的 K 长度是完整 seq_len，Q 长度是 extend 的累积和。
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            # Decode/推测路径可能由 attention backend 预先提供 query 累积长度。
            if get_attn_backend().forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                ):
                    # 每个请求包含固定数量 draft token，按该步长生成累积边界。
                    num_draft_tokens = get_attn_backend().speculative_num_draft_tokens
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    # 普通 decode 每个 batch 元素仅有一个 query token。
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    get_attn_backend().forward_metadata.actual_seq_lengths_q
                )

        # 读取整层物理 index-K cache；block table 决定每个请求可见的页。
        past_key_states = get_token_to_kv_pool().get_index_k_buffer(layer_id)

        # 在真正消费 q/weights 前等待两个辅助流完成。
        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        # weights 同样需要从 scattered 域 all-gather 到完整 attention TP 域。
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        # 页表把逻辑 token 序列映射到物理 index-K cache 页面。
        block_table = get_attn_backend().forward_metadata.block_tables
        # Prefill CP 使用双半区长度元组，需要分别调用两次 Indexer 实现负载均衡。
        if (
            is_prefill
            and self.dsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            # 静态页表可能带 padding，只保留本次非空请求对应的行。
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            # 普通 Prefill 同样裁掉 batch padding；Decode 保留完整固定页表。
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            # NPU 融合算子完成分页 K gather、加权 QK logits 与稀疏 Top-K。
            topk_indices = torch_npu.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                    torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            # Keep DSA top-k as [T, K]; NPU attention expands it when needed.
            return topk_indices[0].squeeze(1)

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        """把 CP Prefill query 拆成前后两半并分别执行 NPU Indexer。

        CP 元数据为两半分别维护 Q/K 累积长度，以抵消因果上下文随位置增长
        带来的算力不均。两次 Lightning Indexer 共享同一 K cache 和页表，
        最后按原 token 顺序拼接为标准 ``[T, TopK]`` 结果。
        """
        # 奇数 token 时前半段多一个元素，与上游 CP 平衡切分约定一致。
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        # gate 可以为空；存在时必须和 query 使用相同切分边界。
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        # 解包两半各自的累积 Q/K 长度元数据。
        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        # 前半 query 通常对应较短上下文，独立调用避免与后半互相 padding。
        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        # 后半 query 使用自己的较长 K 有效长度。
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        # 去掉算子保留的单 head 维，恢复 DSA 统一的二维 Top-K 格式。
        return torch.cat([topk_indices_prev[0], topk_indices_next[0]], dim=0).squeeze(1)


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    """把当前 rank 的 scattered token 特征 all-gather 为 TP attention 全量布局。

    输出第一维使用完整 ``input_ids`` 长度；通信操作把各 rank 的连续局部
    片段写入该 buffer。该转换用于 QLoRA 后的 K 与 head gate 对齐。
    """
    # 同时保留输入别名，避免分配输出后丢失当前 rank 的局部 Tensor。
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    # 通信原语要求局部输入连续，并直接写入预分配的完整输出。
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
