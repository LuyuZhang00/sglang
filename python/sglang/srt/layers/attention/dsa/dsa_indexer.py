"""DeepSeek Sparse Attention（DSA）的核心稀疏索引器。

Indexer 将隐藏状态投影成多 head query 和单 head index-key，计算加权 MQA
logits 后选出每个 query 要访问的 Top-K KV 槽位。实现同时覆盖 CUDA/HIP/
NPU、Decode 分页路径、Prefill ragged/CP 路径、FP8 index-K cache 与图捕获。
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.attention.fused_store_index_cache import (
    can_use_dsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import DSANPUIndexerMixin
from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
    GRAPH_WEIGHTS_PROJ_LORA_ERROR,
    _is_in_piecewise_or_breakable_cuda_graph,
    bcg_dsa_indexer_prefill_split,
    pcg_dsa_indexer_prefill_split,
)
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_in_seq_split,
    is_graph_dsa_split_op_surface,
)
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    configured_pp_size,
    get_device,
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import (
    add_prefix,
    ceil_align,
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_xpu,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()

if not _is_npu:
    from sglang.kernels.ops.attention.dsa import (
        aiter_paged_mqa_logits,
        cutedsl_paged_mqa_logits,
        deepgemm_paged_mqa_logits_native,
        deepgemm_paged_mqa_logits_split,
    )
else:
    aiter_paged_mqa_logits = None
    cutedsl_paged_mqa_logits = None
    deepgemm_paged_mqa_logits_native = None
    deepgemm_paged_mqa_logits_split = None

if _is_cuda:
    from sglang.kernels.ops.attention.dsa import pick_dsl_expand
else:
    pick_dsl_expand = None

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()
# Whether the aiter preshuffle paged-MQA path (page_size=64 + Preshuffle=True +
# KVBlockSize=64) can be used. Falls back to the legacy page_size=1 / KVBlockSize=1
# path when the gluon kernel is unavailable (Triton<3.5 and no AOT bundle).
_use_aiter_preshuffle = aiter_can_use_preshuffle_paged_mqa()
if _use_aiter and not _use_aiter_preshuffle:
    logger.warning(
        "ROCm DSA indexer: aiter preshuffle paged-MQA path is unavailable "
        "(needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1); "
        "falling back to legacy page_size=1 / KVBlockSize=1 path."
    )
if _is_cuda:
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if _use_aiter:
    from aiter.ops.cache import indexer_k_quant_and_cache

from sglang.srt.distributed import (
    get_attn_tp_group,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0


if _is_cuda:
    from sglang.kernels.ops.attention.dsv4 import fused_q_indexer_rope_first_quant
    from sglang.kernels.ops.quantization.dsv32 import (
        fused_k_indexer_norm_rope,
        fused_k_indexer_norm_rope_store,
    )
    from sglang.srt.layers.attention.dsa.dsa_prefill_cuda_graph import (
        logits_head_gate_graph,
        scale_head_gate_graph,
    )

    @register_custom_op(mutates_args=["topk_indices"])
    @register_split_op()
    def broadcast_indexer_topk_from_rank0_(topk_indices: torch.Tensor) -> None:
        """图捕获友好的原地 Top-K 广播 custom op。"""
        _broadcast_indexer_topk_from_rank0_impl(topk_indices)


def _broadcast_indexer_topk_from_rank0_impl(topk_indices: torch.Tensor) -> None:
    """把 attention TP rank 0 的最终 Top-K 原地广播到所有 rank。

    CUDA Graph 捕获中必须使用 PyNCCL，普通 eager 路径可调用通信组的标准
    broadcast。只广播最终输出，避免各 rank 因内部分块不同产生 collective
    次序不一致。
    """
    # 单 rank 无通信需求，直接保留本地结果。
    group = get_attn_tp_group()
    if group.world_size == 1:
        return

    # 捕获期间 torch.distributed broadcast 不一定 graph-safe，改用 PyNCCL。
    if topk_indices.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        if group.pynccl_comm is None:
            raise RuntimeError(
                "SGLANG_DSA_TOPK_BROADCAST requires PyNCCL during CUDA graph capture."
            )
        # 临时启用通信器捕获状态，并以 rank 0 为唯一源。
        with group.pynccl_comm.change_state(enable=True):
            group.pynccl_comm.broadcast(topk_indices, src=0)
    else:
        # eager/非 CUDA 捕获场景使用通信组的普通 broadcast。
        group.broadcast(topk_indices, src=0)


def _broadcast_indexer_topk_from_rank0(
    topk_indices: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """按环境开关同步最终 Indexer 结果，并保持 Optional 返回协议。

    ``None`` 表示调用方不需要索引；未启用广播时完全零开销返回。piecewise
    CUDA Graph 通过注册的 split custom op 暴露通信副作用，其他场景直接执行。
    """
    # Sync only the finalized indexer output. Internal topk_transform calls can
    # be chunked differently across ranks, which would make collectives diverge.
    # 没有输出或功能关闭时无需参与 collective。
    if topk_indices is None or not envs.SGLANG_DSA_TOPK_BROADCAST.get():
        return topk_indices

    # piecewise graph 需要由编译器识别的 custom op 边界。
    if is_in_tc_piecewise_cuda_graph():
        broadcast_indexer_topk_from_rank0_(topk_indices)
    else:
        _broadcast_indexer_topk_from_rank0_impl(topk_indices)
    return topk_indices


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """对最后一维执行保持点积不变的归一化 Hadamard 旋转。

    旋转使 FP8 量化前的异常值分散到各维，降低量化误差；Q 和 K 使用相同
    正交变换，因此未量化的 QK 点积不变。不同平台选择各自优化的实现。
    """
    # from sgl_kernel import hadamard_transform
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    elif _is_xpu:
        from sgl_kernel import hadamard_transform
    else:
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform

    # Fast Hadamard kernel 要求变换维度是 2 的幂。
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    # 乘 1/sqrt(hidden_size) 使变换矩阵正交，从而保持向量范数与点积。
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(DSANPUIndexerMixin, BaseFusedOp):
    """DSA 的 Q/K 投影、index-K 缓存维护及 Top-K 稀疏选择模块。

    Decode 使用分页 MQA logits 直接读取物理 index-K cache；普通 Prefill
    将相关 K gather 成 ragged 连续缓存后计算。类中还负责量化、RoPE、
    head gate、CUDA Graph、双流重叠与 Context Parallel 等执行策略。
    """

    # FP8 MQA logits 以 FP32 输出，每个矩阵元素占 4 字节。
    _MQA_LOGITS_BYTES_PER_ELEM = 4
    # 小于该元素数时直接计算，避免为小 batch 查询显存带来的同步开销。
    _MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
    # 即使空闲显存很多，logits 也最多使用设备总显存的 30%。
    _MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
    _mqa_logits_budget_bytes: Dict[int, int] = {}

    @staticmethod
    def _mqa_logits_free_mem_fraction() -> float:
        return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        config=None,
    ):
        """构造 Indexer 的投影层、归一化、RoPE 与平台执行策略。

        ``wq_b`` 从 QLoRA 表示生成多 head query；K 与 head gate 可使用两个
        独立投影，也可在 CUDA 非 NeoX RoPE 下折叠为融合投影。初始化阶段
        同时解析 CP 拓扑、DeepGEMM SM 配额和分页 MQA logits 后端。

        参数含义
        --------
        hidden_size: 残差流宽度，K / head-gate 投影的输入。
        index_n_heads: indexer query 头数（MQA logits 的 Q 头）。
        index_head_dim: 每个 indexer head 的维度，决定 softmax_scale。
        rope_head_dim: 对 indexer Q/K 做 RoPE 的子空间宽度。
        index_topk: 每个 query 选出多少个 KV 槽，后面 MLA 只看这些槽。
        q_lora_rank: QLoRA 压缩 rank，``wq_b`` 的输入宽度。
        max_position_embeddings / rope_theta / rope_scaling / is_neox_style:
            与模型 RoPE 配置对齐，必须和主干 attention 一致。
        layer_id: 当前 transformer 层，用来索引 index-K cache。
        scale_fmt: FP8 量化 scale 布局。
        block_size: index-K 量化块大小，默认 128。
        quant_config: 投影层量化；None 则 bf16/FP16。
        alt_stream: 第二条 CUDA stream，用来重叠 Q/K 投影与 cache store。
        config: HF config，可改 k_norm 类型和 init/local token 数。
        """
        super().__init__()
        # 保存模型结构超参数，后续 reshape 和 kernel 调度均依赖这些值。
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        # 融合路径要求 CUDA 且当前 RoPE 布局受 fused kernel 支持。
        self.use_dsa_indexer_fusion = (
            _is_cuda
            and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION.get()
            and not is_neox_style
        )
        # alt_stream 用于重叠 Q/K 投影、旋转、量化和缓存写入。
        self.alt_stream = alt_stream
        # Prefill CP 能力由运行时拓扑与模型架构共同决定。
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None
        # CUDA 需要依据可用 SM 数为 DeepGEMM 及 PP 通信做资源规划。
        if _is_cuda:
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
            pp_size = configured_pp_size()
            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
        else:
            self.logits_with_pp_recv = False

        # QLoRA B 投影输出所有 index heads 的 query 向量。
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        # 融合模式把单 head K 与 n_heads 个 gate 合并为一次 GEMM。
        if self.use_dsa_indexer_fusion:
            self.wk_weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim + self.n_heads,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("wk_weights_proj", prefix),
            )
        else:
            # 非融合模式保留独立 K 投影和 head gate 投影。
            self.wk = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wk", prefix),
            )
            self.weights_proj = ReplicatedLinear(
                self.hidden_size,
                self.n_heads,
                bias=False,
                params_dtype=torch.bfloat16,
                prefix=add_prefix("weights_proj", prefix),
            )
        # 模型配置可选择 RMSNorm；默认使用与原模型权重匹配的 LayerNorm。
        if (
            config is not None
            and getattr(config, "index_k_norm_type", "layer") == "rms"
        ):
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.k_norm = LayerNorm(
                self.head_dim, dtype=torch.bfloat16 if _use_aiter else torch.float32
            )
        # RoPE wrapper 统一处理标准/NeoX 布局及长上下文 scaling。
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=get_device().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        # MQA logits 的标准缩放因子 1/sqrt(head_dim)。
        self.softmax_scale = self.head_dim**-0.5
        self.num_init_tokens = self.num_local_tokens = 0
        if config is not None:
            self.num_init_tokens = getattr(config, "index_init_tokens", 0)
            self.num_local_tokens = getattr(config, "index_local_tokens", 0)

        # 启动时完成后端和硬件校验，forward 中只做确定性分派。
        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
            get_exec().kernel.dsa_paged_mqa_logits_backend
        )

    @contextlib.contextmanager
    def _with_real_sm_count(self):
        """在 DeepGEMM logits 计算期间扣除与 PP recv 重叠占用的一个 SM。

        非最后 PP rank 会同时接收上一 stage 数据，通信 kernel 需要常驻一个
        SM。上下文管理器临时调整 DeepGEMM 的 SM 数，退出后自动恢复配置。
        """
        # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
        # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
        # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
        # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
        # 只有确实存在并行 PP recv 的 rank 才需减少计算资源。
        if self.logits_with_pp_recv:
            pp_recv_sm_count = 1
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.sm_count - pp_recv_sm_count
            ):
                yield
        else:
            yield

    @property
    def _indexer_cos_sin_cache(self) -> torch.Tensor:
        return self.rotary_emb.cos_sin_cache

    def _weights_proj_bf16_in_fp32_out(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        """执行 head gate 投影，并尽可能保持 FP32 输出精度。

        CUDA 通过 ``torch.mm(..., out_dtype=float32)`` 直接 FP32 累加；ROCm
        gfx95 可从融合量化三元组取 BF16 旁路，避免昂贵的 FP8 反量化。
        """
        # aiter (ROCm gfx95): extract the passthrough bf16 tensor from the
        # 3-tuple (fp8, scale, bf16) produced by fused_rms_fp8_group_quant,
        # avoiding an expensive FP8-to-bf16 dequantization.
        if _use_aiter and _is_gfx95_supported and isinstance(x, tuple) and len(x) == 3:
            x = x[2]
        # CUDA 直接读取权重，绕过 Linear wrapper 以指定 FP32 输出 dtype。
        if _is_cuda:
            return torch.mm(x, self.weights_proj.weight.t(), out_dtype=torch.float32)

        weights, _ = self.weights_proj(x)
        if _is_hip:
            # Return bf16; multiplying with q_scale promotes back to fp32.
            return weights
        return weights.float()

    @torch.compile(dynamic=True)
    def _project_and_scale_head_gates(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        """投影逐 head gate，并施加 ``1/sqrt(n_heads)`` 归一化。"""
        # 先得到 token × head 的原始投影值。
        weights = self._weights_proj_bf16_in_fp32_out(x)
        # head 数归一化控制多 head 加权求和的方差。
        weights = weights * self.n_heads**-0.5
        return weights

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], q_scale: torch.Tensor
    ):
        """生成已合并量化尺度与 softmax scale 的最终 logits head gate。

        输出形状为 ``[tokens, heads, quant_groups]``，可直接与分组反量化的
        QK 部分积相乘，避免在宽 logits Tensor 上再做额外缩放。
        """
        # 计算 token × head 的 FP32 gate。
        weights = self._weights_proj_bf16_in_fp32_out(x)
        # 先按 head 数归一化，再广播到 FP8 Q 的 scale group 维。
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    @torch.compile(dynamic=True)
    def _apply_q_scale_and_softmax_scale(
        self, weights: torch.Tensor, q_scale: torch.Tensor
    ):
        return weights.unsqueeze(-1) * q_scale * self.softmax_scale

    @torch.compile(dynamic=True)
    def _scale_head_gates(self, weights_raw: torch.Tensor, q_scale: torch.Tensor):
        weights = weights_raw * self.n_heads**-0.5
        return weights.unsqueeze(-1) * q_scale * self.softmax_scale

    def _fused_k_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次融合 GEMM 同时返回原始 K 投影与逐 head gate。"""
        # 输出最后一维布局固定为 [head_dim 的 K | n_heads 的 gate]。
        kw, _ = self.wk_weights_proj(x)
        return kw.split([self.head_dim, self.n_heads], dim=-1)

    def _maybe_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """仅在旧非融合量化布局中应用 Hadamard 旋转。"""
        # Fusion drops the (logit-preserving) Hadamard rotation; without it the
        # index-K cache here matches the fused path that decode reads back.
        return x if self.use_dsa_indexer_fusion else rotate_activation(x)

    def _should_skip_logits_computation(self, forward_batch: ForwardBatch) -> bool:
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.seq_lens_cpu is not None
        ):
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            return max_kv_len <= self.index_topk

        return False

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        """生成完成 RoPE/可选 Hadamard 的 BF16 query 与 index-key。

        小 token batch 可在双 CUDA stream 上重叠 Q 投影与 K/gate 投影；
        Prefill CP 还会 all-gather 局部 K 并恢复全局 token 顺序。返回的
        ``weights_raw`` 仅在融合投影启用时非空。
        """
        # 融合投影才会同时产生尚未缩放的 gate。
        weights_raw = None
        # 双流阶段：当前流做 Q，alt stream 做 K 与 gate。
        if enable_dual_stream:
            # alt stream 先等待 x/q_lora 在当前流准备完成。
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                # 限制 DeepGEMM 使用半数 SM，为并发 K 投影保留资源。
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                # Indexer 只对 query 的前 rope_head_dim 施加位置编码。
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                # alt stream 根据配置执行融合或独立 K 投影。
                if self.use_dsa_indexer_fusion:
                    key, weights_raw = self._fused_k_weights(x)
                else:
                    key, _ = self.wk(x)
                # K 归一化后再进入 RoPE 与 FP8 量化。
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            # 两个投影都完成后才能联合执行 Q/K RoPE。
            current_stream.wait_stream(self.alt_stream)
        else:
            # 单流路径保持相同的投影、reshape、norm 和切分顺序。
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            if self.use_dsa_indexer_fusion:
                key, weights_raw = self._fused_k_weights(x)
            else:
                key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        # 同一次 RoPE 调用保证 Q/K 使用相同 position 与数值实现。
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        # 只覆盖原 Tensor 的 RoPE 子空间，保留后半部分 no-PE 特征。
        self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)

        # Hadamard 也可拆到双流上并行执行。
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = self._maybe_rotate(query)

            with torch.cuda.stream(self.alt_stream):
                key = self._maybe_rotate(key)
            current_stream.wait_stream(self.alt_stream)
        # 未启用常规双流时，旧 CP 路径仍可重叠 Q 旋转与 K all-gather。
        elif (
            self.alt_stream is not None
            and forward_batch.attn_cp_metadata is not None
            and self.dsa_enable_prefill_cp
        ):
            key = self._maybe_rotate(key)
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = self._maybe_rotate(query)

            # Gather the full key on alt_stream so the CP all-gather overlaps
            # with the query rotate above on the current stream.
            with torch.cuda.stream(self.alt_stream):
                # CP-v2 由策略对象物化完整 K；旧 CP 手工 gather 并重排。
                if is_cp_v2_active(forward_batch):
                    key = get_cp_strategy().materialize_full_indexer_k_cache(
                        key, forward_batch
                    )
                else:
                    key = cp_all_gather_rerange_output(
                        key.contiguous(),
                        self.cp_size,
                        forward_batch,
                        torch.cuda.current_stream(),
                    )
            current_stream.wait_stream(self.alt_stream)
            # 此分支已经完成 CP K 物化，避免落到下方重复 all-gather。
            return query, key, weights_raw
        else:
            query = self._maybe_rotate(query)
            key = self._maybe_rotate(key)

        # allgather+rerrange
        # 常规路径在旋转完成后统一物化 Prefill CP 的完整 index-K。
        if is_cp_v2_active(forward_batch):
            key = get_cp_strategy().materialize_full_indexer_k_cache(key, forward_batch)
        elif forward_batch.attn_cp_metadata is not None and self.dsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return query, key, weights_raw

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        """K-only 快路径生成与非融合 Decode 读取布局一致的 BF16 key。

        该函数仅用于未启用融合投影的场景：依次执行 K 投影、归一化、RoPE
        子空间回写与 Hadamard 旋转，不计算 query 或 head gate。
        """
        # Non-fusion path only; self.wk does not exist when fusion is on.
        # 单独执行 K 投影并归一化。
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
        key = rotate_activation(key)

        return key

    def _fused_k_prepare_and_store(
        self,
        key_raw: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        """融合完成 K norm、RoPE、FP8 量化并写入物理 index-K cache。

        支持的 CUDA/page-size/dtype 组合走单 kernel；否则先生成 BF16 K，再
        复用通用缓存写入函数。分层 cache split 下非 owner rank 会跳过写入。
        """
        # 图内调用可传真实 token 槽位前缀；默认使用 ForwardBatch 全量位置。
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc
        # DSA pool 同时管理主 KV 与独立的 index-K-with-scale buffer。
        pool = get_token_to_kv_pool()
        page_size = pool.page_size
        # 写入 owner buffer 后，先使可能存在的广播 scratch 视图失效。
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return
        # E4M3FN + 受支持页布局可一次性完成全部 K 准备与 store。
        if (
            not _is_fp8_fnuz
            and out_cache_loc is not None
            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)
        ):
            fused_k_indexer_norm_rope_store(
                key_raw,
                pool.get_index_k_with_scale_buffer(layer_id=layer_id),
                out_cache_loc,
                self.k_norm.weight,
                self.k_norm.bias,
                self.k_norm.variance_epsilon,
                self._indexer_cos_sin_cache,
                positions,
                page_size,
            )
            return

        # Fallback: separate K kernel + store kernel.
        # 回退仍融合 norm+RoPE，但量化与 scatter store 单独执行。
        key = fused_k_indexer_norm_rope(
            key_raw,
            self.k_norm.weight,
            self.k_norm.bias,
            self.k_norm.variance_epsilon,
            self._indexer_cos_sin_cache,
            positions,
        )
        self._store_index_k_cache(
            forward_batch=forward_batch,
            layer_id=layer_id,
            key=key,
            act_quant=act_quant,
            out_cache_loc=out_cache_loc,
        )

    def _fused_q_prepare_and_store(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        *,
        num_tokens: Optional[int] = None,
        enable_dual_stream: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备融合路径的 FP8 Q/head gate，并并行写入 index-K cache。

        单流模式顺序执行 K 投影/写缓存与 Q kernel；双流模式让 Q GEMM 和
        K+gate GEMM 重叠，再让 fused-Q 与 fused-K-store 重叠。图 split-op
        可用 ``num_tokens`` 切掉静态 buffer 中的 padding 行。
        """
        # num_tokens (graph split-op contract) slices q/k/positions/out_cache_loc
        # to the unpadded count; the returned q_fp8/weights are sliced to match.
        # fused-Q kernel 直接把 attention scale 和 head 归一化乘入 gate。
        q_scale_gate = self.softmax_scale * self.n_heads**-0.5
        out_cache_loc = forward_batch.out_cache_loc
        # graph 静态形状可能大于真实 token 数，位置与写槽位必须同步裁剪。
        if num_tokens is not None:
            positions = positions[:num_tokens]
            out_cache_loc = out_cache_loc[:num_tokens]

        # 无辅助流时使用清晰的顺序路径。
        if self.alt_stream is None or not enable_dual_stream:
            kw, _ = self.wk_weights_proj(x)
            key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)
            if num_tokens is not None:
                key = key[:num_tokens]
                weights_raw = weights_raw[:num_tokens]
            # K 在返回 Q 结果前写入 pool，供本次/后续 Decode 读取。
            self._fused_k_prepare_and_store(
                key,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )
            q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
            if num_tokens is not None:
                q = q[:num_tokens]
            # 单 kernel 完成 Q RoPE、FP8 首维量化及 gate 缩放。
            return fused_q_indexer_rope_first_quant(
                q.contiguous(),
                weights_raw,
                q_scale_gate,
                self._indexer_cos_sin_cache,
                positions,
            )

        # Two overlap stages: wq_b GEMM (alt) || wk_weights_proj GEMM (current),
        # then fused Q kernel (current) || fused K kernel + cache store (alt).
        # wait_stream calls are ordered by issue position so each side waits only
        # on the GEMMs it consumes, not on the other side's fused kernel.
        # 第一阶段：alt stream 做 Q GEMM，当前流做融合 K+gate GEMM。
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            q = self.wq_b(q_lora)[0].view(-1, self.n_heads, self.head_dim)
            if num_tokens is not None:
                q = q[:num_tokens]

        kw, _ = self.wk_weights_proj(x)
        key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)
        if num_tokens is not None:
            key = key[:num_tokens]
            weights_raw = weights_raw[:num_tokens]

        # 第二阶段开始前，当前流等待 Q GEMM；alt stream 等待 K+gate GEMM。
        current_stream.wait_stream(self.alt_stream)
        self.alt_stream.wait_stream(current_stream)
        # 当前流执行 fused-Q，产生 logits kernel 立即需要的 Q 与 gate。
        q_fp8, weights = fused_q_indexer_rope_first_quant(
            q.contiguous(),
            weights_raw,
            q_scale_gate,
            self._indexer_cos_sin_cache,
            positions,
        )
        # alt stream 同时准备 K 并 scatter 写入 cache。
        with torch.cuda.stream(self.alt_stream):
            self._fused_k_prepare_and_store(
                key,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )

        # 返回前等待缓存副作用完成，避免下游立即读取未完成的 index-K。
        current_stream.wait_stream(self.alt_stream)
        return q_fp8, weights

    @staticmethod
    def _update_rope_guarded(dst: torch.Tensor, src: torch.Tensor) -> None:
        """安全地把 RoPE 结果写回原 Q/K 子空间。

        AMD 原地 RoPE 可能让 ``src`` 与 ``dst`` 指向同一地址，此时 copy_ 会
        形成无意义甚至不受支持的自拷贝。编译态不能比较 data_ptr，故跳过检查。
        """
        # On AMD with in-place RoPE kernels, self-aliasing can occur;
        # skip write-back when src/dst tensors point to a single memory.
        # data_ptr() is not comparable inside torch.compile, so skip the guard there.
        if not torch.compiler.is_compiling() and src.data_ptr() == dst.data_ptr():
            return
        dst.copy_(src)

    @staticmethod
    def _get_index_k_read_buffer(pool, layer_id: int) -> torch.Tensor:
        """取得当前 rank 可读取的 index-K-with-scale buffer。

        DSA cache layer split 时非 owner rank 从 owner 广播后的 scratch buffer
        读取；普通 pool 直接读取本层自有 buffer。写路径始终写 owner buffer。
        """
        # Read path: prefer the owner-broadcast scratch buffer under DSA cache
        # layer split; fall back to the owned buffer for plain pools. Stores go
        # through get_index_k_with_scale_buffer() (owned buffer) instead.
        if hasattr(pool, "get_broadcastable_index_k_with_scale_buffer"):
            return pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return pool.get_index_k_with_scale_buffer(layer_id=layer_id)

    @staticmethod
    def _pad_heads_for_deep_gemm(q_fp8, weights):
        """head 数不足 32 时补零以满足 DeepGEMM 的 block 对齐约束。

        DeepGEMM 的 ``block_q = 128 / num_heads`` 不能超过序列对齐 4。
        补出的 Q/head gate 全为零，不改变真实 heads 的 logits；返回原 head 数
        便于调用方在需要时恢复逻辑维度。
        """
        # 32 heads 及以上已满足 kernel tile 要求，无需分配新 Tensor。
        num_heads = q_fp8.shape[1]
        if num_heads >= 32:
            return q_fp8, weights, num_heads
        # Q 在 head 维补零，weights 在对应 head 维同步补零。
        target_heads = 32
        q_fp8 = torch.nn.functional.pad(q_fp8, (0, 0, 0, target_heads - num_heads))
        weights = torch.nn.functional.pad(weights, (0, target_heads - num_heads))
        return q_fp8, weights, num_heads

    def _mask_init_and_local_tokens(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        row_starts: Optional[torch.Tensor] = None,
    ):
        """把全局初始 token 与最近局部 token 强制纳入 Top-K。

        配置的 init/local token 对应 attention sink 与滑动局部窗口。通过把
        这些列 logits 写为 ``+inf``，可复用同一个 Top-K kernel 保证其入选；
        ``row_starts`` 支持 packed ragged logits 的全局列坐标。
        """
        # 两种保留策略均关闭时保持 logits 原样，避免构造索引 Tensor。
        if self.num_init_tokens == 0 and self.num_local_tokens == 0:
            return logits
        # 分页 logits 每行从零开始；ragged packed 行会显式传入起点。
        if row_starts is None:
            row_starts = lengths.new_zeros(lengths.shape[0])
        num_init_tokens = self.num_init_tokens
        num_local_tokens = self.num_local_tokens
        # 初始 token 索引为 row_start 起的固定前缀。
        if num_init_tokens > 0:
            init_idxs = (
                torch.arange(
                    num_init_tokens, dtype=lengths.dtype, device=lengths.device
                )[None, :]
                + row_starts[:, None]
            )
            # 短序列或窄物理 buffer 下将越界位置钳到最后一列。
            init_idxs.clamp_max_(logits.shape[-1] - 1)
            logits.scatter_(dim=1, index=init_idxs, value=float("inf"))
        # 局部 token 从每行有效区间末尾向前枚举。
        if num_local_tokens > 0:
            local_idxs = (
                lengths[:, None]
                - 1
                + row_starts[:, None]
                - torch.arange(
                    num_local_tokens, dtype=lengths.dtype, device=lengths.device
                )[None, :]
            )
            local_idxs.clamp_min_(0)
            logits.scatter_(dim=1, index=local_idxs, value=float("inf"))
        return logits

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        """在 Decode/Verify 类模式上执行分页 MQA logits 与 Top-K。

        该路径直接按 page table 从物理 FP8 index-K cache gather，不物化连续
        KV。它根据平台/配置选择 AITER、CuteDSL 或 DeepGEMM，并处理推测
        token 展开、静态 DP padding 以及融合页表索引变换。
        """
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        # 页大小决定缓存物理布局及可用的 paged-MQA kernel ABI。
        page_size = get_token_to_kv_pool().page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        if _is_hip:
            if _use_aiter_preshuffle:
                assert (
                    page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
            else:
                assert (
                    page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {page_size}"
        else:
            assert page_size == 64, "only support page size 64"
        # NOTE(dark): this support extend/decode/decode+graph
        # ROCm 旧布局读取逐 token 页表；其余分页实现读取真实物理页表。
        if _is_hip and not _use_aiter_preshuffle:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        # 页表列数乘 page size 得到 logits buffer 的最大上下文宽度。
        max_seq_len = block_tables.shape[1] * page_size
        # 读取包含 FP8 K 与 scale 字节的层级物理缓存。
        kv_cache_fp8 = self._get_index_k_read_buffer(get_token_to_kv_pool(), layer_id)

        blocksize = page_size
        # Verify/draft-extend 每个请求展开多行，需使用逐 query 的可见 K 长度。
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()
        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
        # otherwise fall back to computing it here.
        # backend 初始化阶段可能已经按当前长度预计算 DeepGEMM 调度元数据。
        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        assert len(q_fp8.shape) == 3
        # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the
        # hidden states; q_offset is the real (unpadded) q length.
        # q_offset 是去除 DP/静态图 padding 后的真实 query 行数。
        q_offset = sum(metadata.get_dsa_extend_len_cpu())

        # B 为请求数，next_n 为每请求展开的 verify/draft token 数。
        B = metadata.get_seqlens_int32().shape[0]
        next_n = q_offset // B if B > 0 else 0
        # draft_extend_v2 尚不满足 CuteDSL 的输入布局，需回退 DeepGEMM。
        use_cute_dsl = (
            self.paged_mqa_logits_backend.is_cutedsl()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        )
        dsl_expand_factor, dsl_atom = 1, 1
        # Target Verify 可按 next_n/上下文/SM 数选择更合适的 DSL 展开原子。
        if (
            use_cute_dsl
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
        ):
            assert pick_dsl_expand is not None, "CuTe DSL paged MQA is CUDA-only."
            dsl_expand_factor, dsl_atom = pick_dsl_expand(
                next_n,
                batch_size=B,
                max_ctx=max_seq_len,
                num_sms=self.sm_count,
                kernel_atoms=(1, 2, 3, 4),
                num_heads=self.n_heads,
            )
        ctx_2d = getattr(metadata, "paged_mqa_ctx_lens_2d", None)
        # DeepGEMM native 多 query 接口要求已有严格的 [B, next_n] 长度矩阵。
        use_dg_native = (
            not use_cute_dsl
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )

        # 所有 DeepGEMM 路径最终统一为二维长度布局。
        if use_dg_native:
            seqlens_32_2d = ctx_2d
        elif seqlens_32.dim() == 2:
            seqlens_32_2d = seqlens_32
        else:
            seqlens_32_2d = seqlens_32.unsqueeze(-1)
        # 未预计算时按实际二维长度生成分页 MQA work schedule。
        if _is_cuda:
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )

        assert len(kv_cache_fp8.shape) == 2
        block_kv = page_size
        num_heads_kv = 1
        head_dim_with_sf = 132
        # 原始字节 buffer 解释成 [physical_page, token, kv_head=1, K+scale]。
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        # head gate 的尾部单例 scale 维在分页 kernel ABI 中省略。
        weights = weights.squeeze(2)

        # 按初始化时解析的 backend 以及当前 forward 形状确定性分派。
        if self.paged_mqa_logits_backend.is_aiter():
            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32,
                block_tables,
                max_seq_len,
                preshuffle=_use_aiter_preshuffle,
                kv_block_size=block_kv,
            )
        elif use_cute_dsl:
            logits = cutedsl_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                metadata.get_seqlens_int32(),
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                dsl_expand_factor=dsl_expand_factor,
                dsl_atom=dsl_atom,
                blocksize=blocksize,
                sm_count=self.sm_count,
                get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
            )
        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        else:
            # 通用 split 实现覆盖普通 Decode 及不满足 native 条件的推测模式。
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )

        # NOTE(dark): logits should be cleaned in topk_transform
        # 在 Top-K 前强制保留 sink/local token，再完成选择和页表变换。
        self._mask_init_and_local_tokens(logits, seqlens_32)
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        # CUDA 静态 batch 的 padding query 以全 -1 行恢复到输入物理长度。
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        return topk_result

    def _get_mqa_logits_budget_bytes(self, device_index: int) -> int:
        """为宽 Prefill MQA logits 矩阵计算并缓存安全显存预算。

        预算同时受设备总显存比例、服务静态内存占比及当前空闲显存约束。
        CUDA Graph 捕获期间禁止 ``mem_get_info`` 主机同步，只返回静态预算；
        首次普通 Prefill 再测量真实空闲显存并按设备缓存。
        """
        # 用户配置控制可使用空闲显存的比例。
        free_mem_fraction = self._mqa_logits_free_mem_fraction()
        # 非捕获阶段已经测量过的设备直接复用，避免每层同步。
        cached_budget = self._mqa_logits_budget_bytes.get(device_index)
        if cached_budget is not None:
            return cached_budget

        # 总显存用于设定与瞬时空闲值无关的绝对上限。
        total_mem = torch.cuda.get_device_properties(device_index).total_memory

        # logits 最多占设备总显存固定比例，防止挤压 KV cache/工作区。
        total_mem_budget = int(total_mem * self._MQA_LOGITS_TOTAL_MEM_FRACTION)
        mem_fraction_static = get_schedule().mem_fraction_static
        if mem_fraction_static is None:
            static_budget = total_mem_budget
        else:
            static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
            static_budget = min(
                int(static_free_mem * free_mem_fraction),
                total_mem_budget,
            )
        static_budget = max(1, static_budget)

        # Keep the static serving-memory guard during CUDA graph capture without
        # caching it. The first non-capture prefill path will cache the real
        # free-memory budget below.
        # 捕获期间查询 free memory 会同步并破坏图构建的稳定性。
        if get_is_capture_mode():
            return static_budget

        # Match the original free-memory guard: logits_bytes * 2 > free_mem.
        # torch.cuda.mem_get_info synchronizes the host, so cache the result,
        # capped by the workload-independent serving-memory headroom.
        # eager 阶段取得瞬时空闲值，并与服务静态 headroom 取更小者。
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget_bytes = min(int(free_mem * free_mem_fraction), static_budget)

        budget_bytes = max(1, budget_bytes)
        self._mqa_logits_budget_bytes[device_index] = budget_bytes
        return budget_bytes

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device_index: int
    ) -> Tuple[bool, int]:
        """判断 Prefill MQA logits 是否需要按 query 行分块以避免 OOM。

        小矩阵直接跳过预算查询；大矩阵按 FP32 元素数估算完整 buffer 字节数，
        与当前设备预算比较。返回 ``(是否分块, 可用字节预算)``。
        """
        # Quick static check for normal batches
        # 小工作负载的矩阵显存可控，直接计算比查询空闲显存更便宜。
        if num_q * num_k < self._MQA_LOGITS_STATIC_SKIP_ELEMS:
            return False, 0

        # logits 始终为 FP32，因此按 4 字节精确估算完整矩阵。
        logits_bytes = num_q * num_k * self._MQA_LOGITS_BYTES_PER_ELEM
        logits_budget_bytes = self._get_mqa_logits_budget_bytes(device_index)

        need_chunk = logits_bytes > logits_budget_bytes
        return need_chunk, logits_budget_bytes

    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        topk_result: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """执行普通 Prefill 的 ragged MQA logits 与 Top-K。

        先按页表把当前 batch 所需的 FP8 index-K/scale gather 成连续 ragged
        缓存，再计算所有 query 对可见 K 的 logits。若完整 FP32 logits 超出
        显存预算，则按 query 行分块，并为每块保留正确的地址变换元数据。
        """
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        # 推测/Decode 使用分页路径，本函数只接受普通 extend。
        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = get_token_to_kv_pool().page_size
        if _is_hip:
            if _use_aiter_preshuffle:
                assert (
                    page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {page_size}"
            else:
                assert (
                    page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {page_size}"
        else:
            assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        # ragged DeepGEMM 接口使用二维 token × head gate。
        weights = weights.squeeze(-1)

        if _is_hip and not _use_aiter_preshuffle:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        # q_fp8 第一维包括本 rank 上的物理 query 行（可能带 DP padding）。
        batch_size = len(block_tables)
        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device
        device_index = device.index
        assert device_index is not None, "q_fp8 must be on an indexed CUDA device"

        # eager 路径新建自然形状输出；图 split-op 可传入固定地址 buffer。
        if topk_result is None:
            topk_result = torch.full(
                (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
            )
        # 空 batch 保持全 -1 输出，不触碰 max/sum 等不支持空 Tensor 的操作。
        if batch_size == 0:
            return topk_result

        # ks/ke 定义每个 query 在拼接后 ragged K buffer 中的合法列区间。
        ks, ke = metadata.get_indexer_kvcache_range()

        # 先由 CPU 元数据得到 gather buffer 的总长度与最大请求长度。
        indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
        seq_len_sum = torch.sum(indexer_seq_lens_cpu).item()
        max_seq_len = torch.max(indexer_seq_lens_cpu).item()
        # 根据页表 gather 本 batch 可见的 index-K 与 scale 到连续内存。
        k_fp8, k_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
            layer_id,
            metadata.get_indexer_seq_len(),
            block_tables,
            seq_len_sum,
            max_seq_len,
        )
        # AMD 与 NVIDIA 使用不同 FP8 E4M3 语义，按平台恢复正确 dtype 视图。
        if _is_fp8_fnuz:
            k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
        else:
            k_fp8 = k_fp8.view(torch.float8_e4m3fn)

        # scale 在缓存中按原始字节存储，这里恢复为 FP32 向量。
        k_scale = k_scale.view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)

        # Check if we need to chunk to avoid OOM
        # token 级有效长度和 batch 归属用于分块后的 Top-K 地址变换。
        seq_lens_expanded = metadata.get_seqlens_expanded()
        token_to_batch_idx = metadata.get_token_to_batch_idx()
        q_offset = ks.shape[0]
        k_offset = k_fp8.shape[0]
        # 完整 logits 形状为 [真实 query 数, gather 后 K 总数]。
        need_chunk, logits_budget_bytes = self._should_chunk_mqa_logits(
            q_offset, k_offset, device_index
        )

        # 显存允许时一次 kernel 计算全部 query，减少 launch 开销。
        if not need_chunk:
            assert q_fp8[:q_offset].shape[0] != 0
            with self._with_real_sm_count():
                # ROCm 使用 AITER Triton，CUDA 使用 DeepGEMM。
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    # Match the CUDA deep_gemm path (clean_logits=False): the topk
                    # transform masks invalid positions via ks/ke/lengths, so the
                    # -inf pre-fill of the [tokens x seq_len_kv] logits buffer is
                    # redundant and grows quadratically with context length.
                    logits = fp8_mqa_logits(
                        q_fp8[:q_offset],
                        kv,
                        scale,
                        weights[:q_offset],
                        ks,
                        ke,
                        clean_logits=False,
                    )
                else:
                    # 少于 32 heads 时补零满足 DeepGEMM tile 对齐。
                    q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                        q_fp8[:q_offset], weights[:q_offset]
                    )
                    logits = deep_gemm.fp8_mqa_logits(
                        q_padded,
                        kv_fp8,
                        w_padded,
                        ks,
                        ke,
                        clean_logits=False,
                    )
            assert logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset

            # packed ragged logits 使用 ks 作为每行全局列起点。
            self._mask_init_and_local_tokens(logits, seq_lens_expanded, ks)
            # Top-K transform 把选中列转换为 ragged offset 或物理页槽位。
            raw_topk_result = metadata.topk_transform(logits, self.index_topk, ks=ks)
            topk_result[:q_offset] = raw_topk_result
            return topk_result

        # 分块只沿 query 行切；每行仍覆盖完整 gather K 宽度。
        bytes_per_row = k_offset * self._MQA_LOGITS_BYTES_PER_ELEM
        # 预算至少容纳一行，并且不得超过真实 query 数。
        max_rows = max(1, int(logits_budget_bytes // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)

        # RAGGED 元数据带全局 offset；PAGED 元数据则需逐块构造 query 映射。
        global_topk_offset = metadata.attn_metadata.topk_indices_offset
        cu_seqlens_q_full = None
        if global_topk_offset is None:
            cu_seqlens_q_full = torch.ones(q_offset, dtype=torch.int32, device=device)

        assert (
            seq_lens_expanded.shape[0] == q_offset
        ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        if global_topk_offset is not None:
            assert (
                global_topk_offset.shape[0] >= q_offset
            ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

        # 顺序处理 query 块并直接写入最终输出对应切片，峰值显存受预算限制。
        start = 0
        while start < q_offset:
            # 末块可能少于 max_rows。
            end = min(start + max_rows, q_offset)

            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    # clean_logits=False: topk transform handles masking (see above)
                    logits_chunk = fp8_mqa_logits(
                        q_fp8[start:end],
                        kv,
                        scale,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )
                else:
                    q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(
                        q_fp8[start:end], weights[start:end]
                    )
                    logits_chunk = deep_gemm.fp8_mqa_logits(
                        q_padded,
                        kv_fp8,
                        w_padded,
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )

            # 每块保留自身逐行有效长度和 packed K 起点。
            lengths_chunk = seq_lens_expanded[start:end]
            self._mask_init_and_local_tokens(logits_chunk, lengths_chunk, ks[start:end])

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                topk_offset_chunk = None
                cu_seqlens_q_chunk = cu_seqlens_q_full[start:end]
                batch_idx_chunk = token_to_batch_idx[start:end]

            # 传入覆盖参数，确保局部块仍转换到完整 batch 的正确地址域。
            raw_topk_chunk = metadata.topk_transform(
                logits_chunk,
                self.index_topk,
                ks=ks[start:end],
                cu_seqlens_q=cu_seqlens_q_chunk,
                ke_offset=lengths_chunk,
                batch_idx_list=batch_idx_chunk,
                topk_indices_offset_override=topk_offset_chunk,
            )
            # 原地写入块结果后推进游标，旧 logits_chunk 即可释放。
            topk_result[start:end] = raw_topk_chunk
            start = end

        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
        *,
        num_tokens: Optional[int] = None,
        topk_result: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """在所有可见 KV 都必然入选时只计算并存储 index-K。

        该快路径跳过 Q、head gate 和真实 logits。若调用方仍需要索引，则用
        零 logits 触发 Top-K 的连续索引快速生成；图内调用还可裁掉 padding
        token，并把有效结果写入固定地址的 ``topk_result``。
        """
        # Shared by the eager path and the graph DSA split-op dispatch. The two
        # keyword args carry the graph contract and default to the eager behavior:
        #   - num_tokens: slice key/out_cache_loc to the unpadded count (the graph
        #     runs at a static padded shape). None => full (eager) shape.
        #   - topk_result: pre-allocated padded buffer to fill in place (a downstream
        #     captured graph reads it at a fixed address). None => return a fresh,
        #     naturally-sized tensor.
        assert forward_batch.forward_mode.is_extend_without_speculative()
        x_meta = x[0] if isinstance(x, tuple) else x

        # Fast path: only compute and store k cache, skip all q and weights ops.
        # num_tokens (graph contract) slices to the unpadded count.
        # 图 split-op 在静态形状上运行，实际写槽位只取真实 token 前缀。
        out_cache_loc = None
        if num_tokens is not None:
            assert num_tokens <= forward_batch.out_cache_loc.shape[0]
            out_cache_loc = forward_batch.out_cache_loc[:num_tokens]
        elif not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        # Write the same K representation the decode path reads back: fused
        # (no-Hadamard) when fusion is on, else the legacy Hadamard path.
        # K 的数值/量化布局必须与常规路径一致，否则 Decode 读取会不匹配。
        if self.use_dsa_indexer_fusion:
            key_raw, _ = self._fused_k_weights(x)
            if num_tokens is not None:
                assert num_tokens <= key_raw.shape[0]
                key_raw = key_raw[:num_tokens]
                positions = positions[:num_tokens]
            self._fused_k_prepare_and_store(
                key_raw,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                out_cache_loc=out_cache_loc,
            )
        else:
            key = self._get_k_bf16(x, positions)
            if num_tokens is not None:
                assert num_tokens <= key.shape[0]
                key = key[:num_tokens]
            self._store_index_k_cache(
                forward_batch=forward_batch,
                layer_id=layer_id,
                key=key,
                act_quant=act_quant,
                out_cache_loc=out_cache_loc,
            )

        # MHA doesn't need topk_indices
        # MHA 等调用者只依赖缓存副作用，不需要 DSA Top-K。
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        # dummy logits 只提供固定形状，fast Top-K 会按 lengths 生成连续索引。
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x_meta.device,
        )
        raw_topk_result = metadata.topk_transform(dummy_logits, self.index_topk)
        # 图内输出必须保留调用方分配的地址，只覆盖有效行前缀。
        if topk_result is not None:
            # PCG/BCG: fill the valid prefix of the padded static buffer and
            # leave padded rows at the -1 sentinel.
            topk_result[: raw_topk_result.shape[0]] = raw_topk_result
            return None
        return raw_topk_result

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """为旧版 Prefill CP 分片计算 ragged DSA Top-K。

        ``cp_index`` 存在时支持多个指定序列区间的拼接；否则处理单 batch 的
        连续前/后半片。函数从页表 gather 所需 K，构造每个 query 的 ks/ke，
        再调用 DeepGEMM 和 metadata 地址变换恢复物理 KV 索引。
        """
        # 当前 CP 实现未适配 piecewise/breakable graph 的静态副作用协议。
        assert not _is_in_piecewise_or_breakable_cuda_graph(), (
            "DSA context parallel (_get_topk_ragged_with_cp) not supported under "
            "piecewise/breakable CUDA graph"
        )
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        page_size = get_token_to_kv_pool().page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        # 多区间路径先收集每段 K/scale 与 query 元数据，随后拼接一次计算。
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            # 每个三元组描述原 batch 中一个当前 rank 负责的 query 区间。
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                # 因果 query 需要从序列开头到本区间终点的连续历史 K。
                k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                # 拼接后每段 query 的合法 logits 窗口从该段 K buffer 起点开始。
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                # 随 query 位置递增的因果 K 有效长度。
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            # 拼接全部区间，恢复缓存中 FP8 K 与 FP32 scale 的 dtype 视图。
            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            # 一次 DeepGEMM 覆盖所有拼接 query/区间。
            with self._with_real_sm_count():
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
                logits = deep_gemm.fp8_mqa_logits(
                    q_padded,
                    kv_fp8,
                    w_padded,
                    ks,
                    ke,
                    clean_logits=False,
                )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            # 单 batch 半区路径把 prefix 长度加到当前 CP 分片终点。
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            k_fp8 = get_token_to_kv_pool().get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = get_token_to_kv_pool().get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            # 单连续 K buffer 从第 0 列开始，所有 query 的 ks 相同。
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            with self._with_real_sm_count():
                q_padded, w_padded, _ = self._pad_heads_for_deep_gemm(q_fp8, weights)
                logits = deep_gemm.fp8_mqa_logits(
                    q_padded,
                    kv_fp8,
                    w_padded,
                    ks,
                    ke,
                    clean_logits=False,
                )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def _store_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        key: torch.Tensor,
        *,
        act_quant=None,  # fallback only
        out_cache_loc: Optional[torch.Tensor] = None,
    ) -> None:
        """量化并把当前步骤的 DSA index-key 写入物理缓存槽位。

        CUDA 优先以 fused store 同时量化并 scatter；ROCm/AITER 使用其专用
        preshuffle 或 legacy kernel；其他组合回退到 ``act_quant`` 后分别写
        FP8 K 与 scale。该缓存独立于主 MLA KV，但共享 token slot 映射。

        ``out_cache_loc`` 未显式传入时使用 ForwardBatch 的物理槽位。
        """

        # eager 路径默认写入当前 batch 分配的全部 token slot。
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc

        # pool 提供层级 owner/broadcast 语义及 index-K-with-scale 物理 buffer。
        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        # layer split 下只有 owner rank 执行写入，其他 rank 稍后读取广播副本。
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return

        # NVIDIA E4M3FN 且 dtype/page size 受支持时使用最快的融合 scatter store。
        if (
            _is_cuda
            and (not _is_fp8_fnuz)
            and can_use_dsa_fused_store(
                key.dtype,
                out_cache_loc.dtype,
                pool.page_size,
            )
        ):
            # NOTE: wrapper already normalizes shape/contiguity and asserts dtypes.
            # owner buffer 将每个 token 的 128B FP8 K 与 4B scale 打包存储。
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            fused_store_index_k_cache(
                key,
                buf,
                out_cache_loc,
                pool.page_size,
            )
            return

        # Fast path: AITER fused quant + cache store
        # When _use_aiter_preshuffle is True we use the new MFMA 16x16 preshuffle
        # layout (page_size>=16). Otherwise we fall back to the legacy row-major
        # layout with page_size=1; the same kv_cache.view works for both cases
        # because page_size is 1 there.
        # ROCm 使用 AITER 的量化+缓存写入，布局与后续 paged MQA 配套。
        if _use_aiter:
            page_size = pool.page_size
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            kv_cache = buf.view(-1, page_size, 132).view(fp8_dtype)
            # AITER 路径目前按 ForwardBatch 槽位写入，并要求索引连续。
            out_loc = forward_batch.out_cache_loc
            if not out_loc.is_contiguous():
                out_loc = out_loc.contiguous()
            indexer_k_quant_and_cache(
                key,
                kv_cache,
                out_loc,
                self.block_size,
                self.scale_fmt,
                preshuffle=_use_aiter_preshuffle,
            )
            return

        # Fallback: original path
        # 最通用回退先量化为 FP8 K 与逐组 scale。
        assert act_quant is not None
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        if not out_cache_loc.is_contiguous():
            out_cache_loc = out_cache_loc.contiguous()

        # pool 根据 loc 将 K/scale scatter 到对应物理 token slot。
        pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

    def forward_native(self, *args, **kwargs):
        """Indexer 没有纯 Torch 参考实现，防止错误地落入 native dispatch。"""
        # The indexer has no pure-torch reference path; it only runs on
        # platforms with a dedicated forward below.
        raise NotImplementedError("Indexer has no native (pure-torch) path")

    def forward_xpu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """XPU 暂复用与 CUDA 路径相同的高层调度和平台算子分支。"""
        return self.forward_cuda(
            x, q_lora, positions, forward_batch, layer_id, return_indices
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """执行 CUDA/HIP/XPU 上的完整 DSA Indexer 前向。

        函数先选择平台量化算子和 graph-safe 元数据获取方式，再按 K-only、
        融合、图 split-op、双流或普通路径准备 Q/K/head gate。最后依据
        forward mode 分派到 paged Decode 或 ragged Prefill/CP Top-K。

        参数含义
        --------
        x: 残差流 hidden，投影 index-K 与 head gate。
        q_lora: 本层 QLoRA 激活，经 ``wq_b`` 得到 indexer query。
        positions: 每个 token 的位置，给 indexer RoPE。
        forward_batch: 本拍 batch，用来选 paged/ragged 以及读 page table。
        layer_id: 写/读哪一层的 index-K cache。
        return_indices: False 时只更新 cache 不跑 Top-K（部分 MTP 预计算）。
        """
        # HIP 与 CUDA 使用各自的 FP8 activation quant kernel。
        if _is_hip:
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant
        elif not _is_npu:
            from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        # When upstream uses fused FP8 RMSNorm+quant, activations may be passed as
        # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
        # 融合 RMSNorm+量化可能返回 tuple；形状/设备查询使用其中的主 Tensor。
        x_meta = x[0] if isinstance(x, tuple) else x

        # 两类图模式都要求避免在 Python 侧绑定每次 replay 新建的 metadata 对象。
        in_piecewise_or_breakable_cuda_graph = (
            _is_in_piecewise_or_breakable_cuda_graph()
        )

        # In piecewise/breakable CUDA graph mode, metadata is fetched inside
        # custom ops via get_tc_piecewise_forward_context() to prevent Dynamo
        # from guarding on forward_metadata identity, which changes each replay
        # when init_forward_metadata creates a new ForwardMetadata object.
        if not in_piecewise_or_breakable_cuda_graph:
            # eager/普通 graph 路径可直接从 attention backend 取得层级元数据。
            metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
            # backend 返回 None 表示当前层/模式不需要 DSA Indexer。
            if metadata is None:
                return None
        else:
            metadata = None

        # 双流只用于捕获期的小 batch；大 batch 共享 SM 往往得不偿失。
        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        # piecewise/breakable graph 内不能读取 CPU 长度做数据相关分支。
        skip_logits_computation = False
        if not in_piecewise_or_breakable_cuda_graph:
            skip_logits_computation = self._should_skip_logits_computation(
                forward_batch
            )

        # Optimization: fast path when skipping topk computation
        # 短 Prefill 的全部历史 token 都会入选，仅写 K 并生成连续索引。
        if skip_logits_computation and (not self.dsa_enable_prefill_cp):
            topk_result = self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                metadata,
                return_indices,
            )
            # 多 TP rank 可选择只由 rank 0 计算，再同步最终固定形状结果。
            topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
            return maybe_capture_indexer_topk(layer_id, topk_result)

        # When weights_proj is LoRA-wrapped, use an eager module call so the
        # wrapper owns base+delta and no LoRA kernel runs under torch.compile.
        # Fusion folds weights_proj into wk_weights_proj, so weights_proj is
        # absent then; short-circuit before touching it.
        # LoRA wrapper 必须拥有 base+delta 调用，不能绕过模块直接读取 weight。
        weights_proj_lora = not self.use_dsa_indexer_fusion and getattr(
            self.weights_proj, "set_lora", False
        )

        # eager、非 CP 的融合路径一次准备 Q/gate 并完成 K cache 写入。
        if (
            self.use_dsa_indexer_fusion
            and not in_piecewise_or_breakable_cuda_graph
            and forward_batch.attn_cp_metadata is None
        ):
            q_fp8, weights = self._fused_q_prepare_and_store(
                x, q_lora, positions, forward_batch, layer_id, act_quant
            )
        # 普通图内 Prefill 交给完整 Indexer split-op，保持输出地址稳定。
        elif (
            is_graph_dsa_split_op_surface(forward_batch)
            and not self.dsa_enable_prefill_cp
        ):
            # Default path for non-CP prefill under PCG/BCG: run the whole indexer
            # (q/k proj, head gate, k-cache store, topk) as a single eager split op
            # instead of capturing it piecemeal in the graph. The split op is
            # fusion-aware, so this also covers the fused path here.
            if weights_proj_lora:
                raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
            # 返回索引时按图静态 token 数预分配并以 -1 初始化 padding 行。
            if return_indices:
                topk_result = torch.full(
                    (x.shape[0], self.index_topk),
                    -1,
                    device=x.device,
                    dtype=torch.int32,
                )
            else:
                # 空首维作为“不需要索引、只写 K”的 graph-safe 协议。
                topk_result = torch.empty(
                    (0, self.index_topk), device=x.device, dtype=torch.int32
                )
            # BCG 通过 eager wrapper 执行，PCG 直接调用注册的 split op。
            graph_dispatch_fn = (
                bcg_dsa_indexer_prefill_split
                if is_in_breakable_cuda_graph()
                else pcg_dsa_indexer_prefill_split
            )
            # split op 原地填充 topk_result，不返回动态分配 Tensor。
            graph_dispatch_fn(
                layer_id=layer_id,
                x=x,
                q_lora=q_lora,
                positions=positions,
                topk_result=topk_result,
            )
            result = _broadcast_indexer_topk_from_rank0(
                topk_result if return_indices else None
            )
            return maybe_capture_indexer_topk(layer_id, result)

        # Decode 小 batch 用双流重叠 Q/K 投影、量化与缓存 store。
        elif enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            # 当前流与 alt stream 建立起始依赖。
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            # 非融合 gate 投影可先计算，与后续 Q/K 准备重叠。
            if not self.use_dsa_indexer_fusion:
                if weights_proj_lora:
                    weights = self.weights_proj(x)[0].float() * self.n_heads**-0.5
                else:
                    weights = self._project_and_scale_head_gates(x)
            # 双流生成完成 RoPE/旋转的 BF16 Q/K。
            query, key, weights_raw = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )
            # 当前流量化 Q，alt stream 同时量化并写入 K cache。
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            current_stream.wait_stream(self.alt_stream)
            # 根据 gate 来源选择只缩放 raw gate 或合并已投影 gate。
            if self.use_dsa_indexer_fusion:
                weights = self._scale_head_gates(weights_raw, q_scale)
            else:
                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
        else:
            # 其余 eager/CP/非默认图形状使用通用 Q/K BF16 准备路径。
            query, key, weights_raw = self._get_q_k_bf16(
                q_lora,
                x,
                positions,
                enable_dual_stream,
                forward_batch=forward_batch,
            )

            if enable_dual_stream:
                # 通用双流路径同样重叠 Q 量化与 K cache store。
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)

                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                with torch.cuda.stream(self.alt_stream):
                    self._store_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        key=key,
                        act_quant=act_quant,
                    )
                current_stream.wait_stream(self.alt_stream)
            elif not in_piecewise_or_breakable_cuda_graph:
                # eager 单流顺序完成 Q 量化与 K 写入。
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            else:
                # Graph paths not handled by the full DSA indexer split op
                # still need q_fp8 for paged topk and q_scale for
                # logits_head_gate_graph. K-cache storage is handled by the
                # full graph split path when prefill requires it.
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)

            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
            # fused_rms_fp8_group_quant is passed directly to _get_logits_head_gate,
            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
            # completely skipping the FP8 dequantization path below.
            # gfx95 AITER 三元组携带 BF16 旁路，避免反量化用于 gate GEMM。
            if (
                _use_aiter
                and _is_gfx95_supported
                and isinstance(x, tuple)
                and len(x) == 3
            ):
                x_for_gate = x
            elif isinstance(x, tuple):
                # 一般 FP8 tuple 需要按 group scale 恢复 BF16 gate 输入。
                assert len(x) in (
                    2,
                    3,
                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
                x_q, x_s = x[0], x[1]
                if (
                    x_s is not None
                    and x_q.dim() == 2
                    and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]
                ):
                    m, n = x_q.shape
                    ng = x_s.shape[1]
                    # scale group 能整除 hidden dim 时执行精确分组反量化。
                    if ng > 0 and n % ng == 0:
                        group = n // ng
                        x_for_gate = (
                            x_q.to(torch.float32)
                            .view(m, ng, group)
                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
                            .view(m, n)
                            .to(torch.bfloat16)
                        )
                    else:
                        # 元数据异常时至少保留 FP8 值的 BF16 转换，避免非法 reshape。
                        x_for_gate = x_q.to(torch.bfloat16)
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x

            # graph 内使用注册 custom op，避免 Dynamo 内联权重/GEMM 细节。
            if in_piecewise_or_breakable_cuda_graph:
                if self.use_dsa_indexer_fusion:
                    # 融合投影已有 weights_raw，只需 graph-safe 缩放。
                    weights = scale_head_gate_graph(
                        weights_raw,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )
                else:
                    # 非融合路径由 graph-safe custom op 完成 gate GEMM 与缩放。
                    if weights_proj_lora:
                        raise RuntimeError(GRAPH_WEIGHTS_PROJ_LORA_ERROR)
                    weights = logits_head_gate_graph(
                        x_for_gate,
                        self.weights_proj.weight,
                        self.n_heads**-0.5,
                        self.softmax_scale,
                        q_scale,
                    )
            elif self.use_dsa_indexer_fusion:
                # eager 融合路径缩放随 K 一同投影出的 raw gate。
                weights = self._scale_head_gates(weights_raw, q_scale)
            elif weights_proj_lora:
                # LoRA 场景必须调用模块 forward，确保 delta 权重被应用。
                weights = self.weights_proj(x_for_gate)[0].float() * self.n_heads**-0.5
                weights = self._apply_q_scale_and_softmax_scale(weights, q_scale)
            else:
                # 常规非融合路径使用 torch.compile 的投影+缩放 helper。
                weights = self._get_logits_head_gate(x_for_gate, q_scale)

        # CUDA/HIP 共用分页与 ragged Top-K 高层调度。
        if _is_cuda or _is_hip:
            # In piecewise/breakable CUDA graph, any access to seq_lens_cpu
            # creates a Dynamo shape guard. These graph modes never have empty
            # batches.
            if not in_piecewise_or_breakable_cuda_graph:
                if forward_batch.seq_lens.numel() == 0:
                    # this seems b/c max-pad, no worries?
                    # if x.shape[0] != 0:
                    #     print(
                    #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                    #     )
                    # DP max-pad 的空伴随 rank 返回固定形状的全无效索引。
                    topk_result = torch.full(
                        (x_meta.shape[0], self.index_topk),
                        -1,
                        dtype=torch.int,
                        device=x_meta.device,
                    )
                    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
                    return maybe_capture_indexer_topk(layer_id, topk_result)

            # Decode、Target Verify 和 Draft Extend v2 都可直接读取分页 K cache。
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                # 普通 Prefill 使用 ragged 路径；in-seq CP 需分前后半区平衡负载。
                if (
                    forward_batch.attn_cp_metadata is not None
                    and is_dsa_prefill_cp_in_seq_split()
                ):
                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
                    actual_seq_q_prev = (
                        forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
                    )
                    actual_seq_q_next = (
                        forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]
                    )

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    # Q 与 gate 必须使用同一半区边界，保持 token 对齐。
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    # 两个半区使用各自因果 KV 长度独立计算 Top-K。
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    # 按原 query 顺序拼接，再进行可选 TP 广播与状态捕获。
                    topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)
                    topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
                    return maybe_capture_indexer_topk(layer_id, topk_result)
                else:
                    # In-graph (PCG/BCG) non-CP prefill is handled earlier by the
                    # graph DSA split-op dispatch, so only the eager path reaches
                    # here.
                    assert not in_piecewise_or_breakable_cuda_graph, (
                        "Internal error: in-graph DSA prefill must go through the "
                        "graph DSA split-op dispatch"
                    )
                    # 常规 eager Prefill 从分页 pool gather K 后计算 ragged logits。
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
        else:
            raise NotImplementedError("DSA indexer only supports CUDA, HIP, and NPU")
        # 所有路径最后统一同步和捕获，禁止在内部分块循环中插入 collective。
        topk_result = _broadcast_indexer_topk_from_rank0(topk_result)
        return maybe_capture_indexer_topk(layer_id, topk_result)
