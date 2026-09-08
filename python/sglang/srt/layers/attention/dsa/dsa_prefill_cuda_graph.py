"""DSA Indexer 在 piecewise/breakable CUDA Graph 中的 Prefill 入口。

图内路径把 Indexer 封装成 split custom op，以固定地址的 ``topk_result``
传递结果，并允许图编译器在该边界执行 eager 计算。文件中的 fake impl
只负责形状传播；真实实现仍复用 Indexer 的投影、缓存写入和 Top-K 逻辑。
"""

from __future__ import annotations

import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

GRAPH_WEIGHTS_PROJ_LORA_ERROR = (
    "DSA indexer weights_proj LoRA is incompatible with "
    "piecewise/breakable CUDA graph; remove the explicit "
    "prefill cuda-graph backend override or drop "
    "indexer.weights_proj from the LoRA target modules."
)


def _is_in_piecewise_or_breakable_cuda_graph() -> bool:
    """返回当前线程是否正在两类可拆分 CUDA Graph 上下文中。"""
    return is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph()


if _is_cuda:

    def _scale_head_gate_graph_fake_impl(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """为编译器推导融合 head gate 输出的形状、dtype 与设备。"""
        # 最后一维来自量化 scale 的分组数，而非 weights_raw 的形状。
        return torch.empty(
            (weights_raw.shape[0], weights_raw.shape[1], q_scale.shape[-1]),
            dtype=torch.float32,
            device=weights_raw.device,
        )

    # In-graph (PCG/BCG) head gate for the fused path: weights_proj is folded
    # into wk_weights_proj, so weights_raw is precomputed and there is no GEMM.
    @register_custom_op(fake_impl=_scale_head_gate_graph_fake_impl)
    def scale_head_gate_graph(
        weights_raw: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """图内融合路径的 head gate 缩放。

        ``weights_proj`` 已折叠进 K 投影，所以输入是预计算的逐 head 权重。
        这里依次施加 head 数归一化、query 量化 scale 与 attention scale，
        返回 FP32 gate 以保持后续 logits 加权精度。
        """
        # 先按 head 数平方根归一化，和非融合路径的投影结果保持一致。
        weights = weights_raw * n_heads_inv_sqrt
        # 扩展分组维并一次性乘入 query scale 与 softmax scale。
        return weights.unsqueeze(-1) * q_scale * softmax_scale

    def _logits_head_gate_graph_fake_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """为非融合 head gate GEMM 提供无数据依赖的 fake 输出。"""
        # 输出 head 数来自投影权重行数，分组数来自 q_scale。
        return torch.empty(
            (x.shape[0], weight.shape[0], q_scale.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )

    # In-graph (PCG/BCG) head gate for the NON-prefill path
    @register_custom_op(fake_impl=_logits_head_gate_graph_fake_impl)
    def logits_head_gate_graph(
        x: torch.Tensor,
        weight: torch.Tensor,
        n_heads_inv_sqrt: float,
        softmax_scale: float,
        q_scale: torch.Tensor,
    ) -> torch.Tensor:
        """图内非 Prefill/非融合路径的完整 head gate 计算。

        先以 FP32 累加执行 ``x @ weight.T``，再应用 head 归一化、query
        量化 scale 和 softmax scale。独立 custom op 可稳定图捕获边界。
        """
        # 显式指定 FP32 输出，避免 BF16 投影损失 gate 精度。
        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        # 与融合路径采用相同的 head 数归一化。
        weights = out * n_heads_inv_sqrt
        # 广播到量化分组维，得到每个 token/head/group 的最终 gate。
        weights = weights.unsqueeze(-1) * q_scale * softmax_scale
        return weights


@register_custom_op(mutates_args=["topk_result"])
@register_split_op()
def pcg_dsa_indexer_prefill_split(
    layer_id: int,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    topk_result: torch.Tensor,
) -> None:
    """在 CUDA Graph split-op 边界执行一次完整 DSA Prefill Indexer。

    函数从 piecewise graph context 取得 ``ForwardBatch``、Indexer 和元数据，
    按 K-only、融合或普通路径执行。它不返回新 Tensor，而是原地写入调用方
    预分配的静态 ``topk_result``，从而满足下游捕获图对地址稳定性的要求。
    """
    # Default in-graph indexer path for non-CP prefill: runs the whole indexer
    # (q/k proj, head gate, k-cache store, topk) as one eager split op. PCG calls
    # this as a split op; BCG uses the explicit eager wrapper below.
    #
    # Output contract (differs from the eager `forward` path): a split op returns
    # None, so results are delivered only by mutating `topk_result` in place. The
    # call site pre-allocates it at a static, padded shape and a downstream
    # captured graph reads it at a fixed address; eager code instead allocates
    # and returns a fresh, naturally-sized tensor each call.
    # split op 依赖 CUDA 专用量化和图捕获语义，其他平台属于内部调度错误。
    assert _is_cuda, "Internal error: DSA graph dispatch is only supported on CUDA"
    # 延迟导入避免非 CUDA 进程在模块加载阶段触碰 Triton CUDA kernel。
    from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

    # 图上下文保存当前 batch 及按 layer 注册的 Indexer 实例。
    forward_context = get_tc_piecewise_forward_context()
    forward_batch = forward_context.forward_batch
    indexer = forward_context.dsa_indexers[layer_id]
    # attention backend 负责提供与当前层页表/长度布局匹配的元数据视图。
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)

    # 图输入可能按静态上限补齐，真实 token 数用于切掉 padding 部分。
    extend_num_tokens = forward_batch.extend_num_tokens
    # Empty buffer encodes return_indices=False for graph dispatch.
    # 空 Tensor 是图内稳定的布尔协议：表示调用方只要求写 K、不需要索引。
    return_indices = topk_result.numel() != 0
    # 可跳过 logits 时选择更轻量的 K-only 路径；Prefill CP 例外需完整索引。
    k_only = not return_indices or (
        indexer._should_skip_logits_computation(forward_batch)
        and not indexer.dsa_enable_prefill_cp
    )
    if k_only:
        # K-only 路径仍负责投影并写 index-K cache，必要时复用 seed 索引。
        indexer._forward_cuda_k_only(
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            metadata=metadata,
            return_indices=return_indices,
            num_tokens=extend_num_tokens,
            topk_result=topk_result,
        )
        # 已完成本 split op 的全部副作用，避免进入 q/logits 计算。
        return

    # Fused path stores K (no-Hadamard) and computes q_fp8 + head gate in the
    # fused kernels, sliced to the unpadded count. Single stream: the split op is
    # captured, so the dual-stream overlap is disabled.
    if indexer.use_dsa_indexer_fusion:
        # 融合 kernel 同时准备量化 Q、head gate 并写入未旋转 K cache。
        q_fp8, weights = indexer._fused_q_prepare_and_store(
            x,
            q_lora,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            num_tokens=extend_num_tokens,
            enable_dual_stream=False,
        )
        # 基于 ragged 序列元数据计算 logits，并原地写入静态 Top-K buffer。
        indexer._get_topk_ragged(
            False,
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            topk_result,
        )
        return

    # 普通路径分别生成 BF16 query/key；图内关闭双 stream 以保持捕获稳定。
    query, key, _ = indexer._get_q_k_bf16(
        q_lora,
        x,
        positions,
        enable_dual_stream=False,
        forward_batch=forward_batch,
    )
    # MQA logits kernel 使用 FP8 query，并需要 scale 恢复每个量化分组的幅值。
    q_fp8, q_scale = act_quant(query, indexer.block_size, indexer.scale_fmt)
    # Reuse the compiled head-gate util shared with the eager path.
    # 计算逐 head gate，并把 query scale 合并进去供 logits 阶段直接使用。
    weights = indexer._get_logits_head_gate(x, q_scale)
    # Store K cache + ragged top-k, sliced to the unpadded count and writing into
    # the static padded topk_result buffer (the graph contract). Mirrors the eager
    # path's store + _get_topk_ragged.
    # 只写真实 token 对应的 K，禁止 padding token 占用或覆盖 KV 槽位。
    indexer._store_index_k_cache(
        forward_batch=forward_batch,
        layer_id=layer_id,
        key=key[:extend_num_tokens],
        act_quant=act_quant,
        out_cache_loc=forward_batch.out_cache_loc[:extend_num_tokens],
    )
    # 同样切掉 q 的 padding 行，结果写进预分配 buffer 的有效前缀。
    indexer._get_topk_ragged(
        False,
        forward_batch,
        layer_id,
        q_fp8[:extend_num_tokens],
        weights,
        metadata,
        topk_result,
    )


# BCG 要求在图中以 eager wrapper 执行同一实现；两条路径共享全部语义。
bcg_dsa_indexer_prefill_split = eager_on_graph(True)(pcg_dsa_indexer_prefill_split)
