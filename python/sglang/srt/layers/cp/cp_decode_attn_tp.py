"""CP Decode Attention TP context.

When CP (Context Parallel) mode sets tp_size=1 (repeat weights), decode can
partition attention weights across CP ranks matching normal TP behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# HF architectures whose attention linears are replicated with tp_size=1 under
# context parallelism, so slicing them to the local CP partition during decode is
# equivalent to a normal TP-layout GEMM. Any model not on this list must not use
# CP decode attention TP. Single arch-string source of truth for the whitelist.
CP_DECODE_ATTN_TP_SUPPORTED_ARCHS: Tuple[str, ...] = (
    # DeepSeek-V4
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
    "DeepseekV4ForCausalLMDSpark",
    # GLM-5.x (inherits DeepseekV2 attention; DSA path)
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
)

_global_cp_decode_attn_tp_ctx: CpDecodeAttnTpContext | None = None


def get_cp_decode_attn_tp_ctx() -> CpDecodeAttnTpContext:
    """返回 decode-attention-TP 的进程级上下文单例。

    attention 权重的临时切片缓存必须在同一进程内复用，才能避免每层、每 token
    反复创建 view。延迟创建也使没有启用该优化的进程不承担初始化依赖和状态开销。
    单例只保存可恢复的切片状态，实际是否启用仍按每个 forward batch 判断。
    """
    global _global_cp_decode_attn_tp_ctx
    # 首次访问时读取当前 parallel context 并建立切片缓存容器。
    if _global_cp_decode_attn_tp_ctx is None:
        _global_cp_decode_attn_tp_ctx = CpDecodeAttnTpContext()
    return _global_cp_decode_attn_tp_ctx


class CpDecodeAttnTpContext:
    """Slices replicated attention weights across CP ranks during decode."""

    def __init__(self):
        """读取 CP 通信拓扑，初始化 decode 时的 TP 切片坐标与缓存。

        当正常 TP 为 1 且 attention 权重在 CP rank 上复制时，CP rank 可临时充当
        decode 的 TP rank；否则保持关闭。该对象不在此决定 prefill/decode，而由
        ``set_decode_attn_tp`` 根据当前 batch 和 DSA CP 状态作最终判定。
        """
        # 用户开关和多 rank attention-CP 组是该优化的必要条件。
        enable_attn_tp = get_parallel().enable_cp_decode_attn_tp

        if enable_attn_tp and get_parallel().attn_cp_size > 1:
            # 将 attention-CP rank/size 映射为 decode 阶段的逻辑 TP 坐标。
            self.decode_tp_rank = get_parallel().attn_cp_rank
            self.decode_tp_size = get_parallel().attn_cp_size
            logger.info("Enable CP decode attention TP")
        else:
            self.decode_tp_rank = None
            self.decode_tp_size = None
            logger.info("Disable CP decode attention TP")

        # 每个 forward 动态更新；默认关闭，避免 prefill 误用切片权重。
        self.use_decode_attn_tp = False
        self._slice_cache: Dict = {}

    @property
    def is_enabled(self) -> bool:
        return self.decode_tp_size is not None and self.decode_tp_size > 1

    def set_decode_attn_tp(self, forward_batch: ForwardBatch):
        """依据当前 forward 决定本次是否临时启用 attention weight 切片。

        CP prefill 需要完整 attention head 布局，DSA prefill CP 亦自行处理局部
        attention，因此二者必须排除。其他 forward（主要是 decode）可以让复制的
        权重按 CP rank 切成等价 TP 分片，减少每 rank 的 attention GEMM 规模。
        """
        # 拓扑未满足条件时明确关闭，防止复用前一次 batch 的状态。
        if not self.is_enabled:
            self.use_decode_attn_tp = False
            return
        # 非 CP-prefill 且非 DSA-prefill 才能安全将 CP 作为 decode TP。
        self.use_decode_attn_tp = not is_cp_v2_active(
            forward_batch
        ) and not dsa_use_prefill_cp(forward_batch)

    def _slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """返回当前 decode-TP rank 在指定维度上的连续等分 view。

        ColumnParallelLinear 沿输出维切，RowParallelLinear 沿输入维切；调用方已
        校验可整除。row 切片需要 contiguous，以满足后续 GEMM 对权重内存布局的
        要求；column 切片可保留 narrow view，避免不必要复制。
        """
        assert dim in (0, 1)
        chunk = tensor.shape[dim] // self.decode_tp_size
        sliced = tensor.narrow(dim, self.decode_tp_rank * chunk, chunk)
        return sliced if dim == 0 else sliced.contiguous()

    # ==================== Unified activate/restore ====================

    def _activate(self, obj, attr_name: str, dim: int):
        """将对象属性临时替换为当前 CP rank 对应的 decode-TP 切片。

        Parameter 需要改 ``.data`` 保持模块属性对象不变，普通 tensor 则直接替换
        属性。首次切片缓存原 tensor、局部 view 和类型标记；之后复用缓存，最终由
        ``_restore`` 无条件恢复，确保不同 forward 不会看到残留分片权重。
        """
        # 吸收式权重/scale 可选，不存在时无需参与切片。
        tensor = getattr(obj, attr_name, None)
        if tensor is None:
            return
        # Parameter 的 data 才是实际参与 GEMM 的 tensor。
        is_param = isinstance(tensor, torch.nn.Parameter)
        raw = tensor.data if is_param else tensor
        # 防御性校验：维度和可整除性错误不能静默产生错误 head 分片。
        assert isinstance(raw, torch.Tensor) and raw.dim() > dim, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name} is not sliceable "
            f"(type={type(tensor).__name__}, dim={raw.dim()}, required_dim>{dim})"
        )
        assert raw.shape[dim] % self.decode_tp_size == 0, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name}.shape[{dim}]={raw.shape[dim]} "
            f"not divisible by decode_tp_size={self.decode_tp_size}"
        )

        # 使用对象 identity + 属性名缓存，同一层的 view 只创建一次。
        cache_key = (id(obj), attr_name)
        cache = self._slice_cache.get(cache_key)
        if cache is None:
            # 缓存原值、当前 rank view 以及 Parameter 标志，供 finally 恢复。
            cache = (raw, self._slice(raw, dim), is_param)
            self._slice_cache[cache_key] = cache

        # Parameter 与普通 tensor 的替换方式不同，避免破坏 module 注册关系。
        if cache[2]:
            tensor.data = cache[1]
        else:
            setattr(obj, attr_name, cache[1])

    def _restore(self, obj, attr_name: str):
        """恢复 ``_activate`` 前缓存的完整权重或 scale 属性。

        context manager 的 finally 会按逆序调用此函数，即使模型 forward 异常也能
        还原模块。不存在缓存说明属性未被激活（例如可选 scale），可安全 no-op。
        """
        cache = self._slice_cache.get((id(obj), attr_name))
        if cache is None:
            return
        orig, _, is_param = cache
        if is_param:
            getattr(obj, attr_name).data = orig
        else:
            setattr(obj, attr_name, orig)

    # ==================== Linear helpers ====================

    def _get_linear_attrs(self, linear_instance) -> List[Tuple]:
        """列出一个并行线性层中需按 decode-TP 维度切分的张量属性。

        ColumnParallelLinear 沿输出维（0）切分，RowParallelLinear 沿输入维（1）
        切分，与正常 TP 权重布局一致。量化权重的 scale 若存在必须沿同维切，
        否则局部权重与 scale 形状不匹配。
        """
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        # 仅支持这两种 SGLang 并行线性层，未知实现保守地不切分。
        if isinstance(linear_instance, RowParallelLinear):
            dim = 1
        elif isinstance(linear_instance, ColumnParallelLinear):
            dim = 0
        else:
            return []

        # 主 weight 必须切分；可选量化 scale 与其保持同一分片轴。
        attrs = [(linear_instance, "weight", dim)]
        for scale_name in ("weight_scale_inv", "weight_scale"):
            if getattr(linear_instance, scale_name, None) is not None:
                attrs.append((linear_instance, scale_name, dim))
        return attrs

    # ==================== Context manager ====================

    @contextmanager
    def maybe_use_decode_attn_tp(
        self,
        forward_batch: ForwardBatch,
        modules: list,
        tensor_attrs: List[Tuple] = None,
        radix_attn: Optional[RadixAttention] = None,
    ):
        """在 context 内临时启用 decode attention 的 CP-as-TP 权重布局。

        Args:
            modules: Linear layers (ColumnParallel/RowParallel) to slice.
            tensor_attrs: (obj, attr_name, dim) tuples for absorbed weights.
            radix_attn: RadixAttention instance whose tp_q_head_num should be
                overridden to match the sliced head count during decode TP.

        进入时替换 attention linear/吸收式张量并同步分片尺寸；退出时按逆序恢复
        所有状态，确保异常不会污染下一次 prefill 或 decode。该局部改写只影响
        当前 forward 的 GEMM 和 head 元数据，不修改检查点的原始权重。
        """
        # 每个 batch 重新判断，避免 prefill 沿用上一 decode 的启用状态。
        self.set_decode_attn_tp(forward_batch)
        if not self.use_decode_attn_tp:
            yield
            return

        # 记录所有可变状态，finally 中无论异常与否都要恢复。
        all_attrs = []  # (obj, attr_name) pairs to restore
        size_overrides = []  # (linear, size_attr, orig_size)
        row_parallel_decode_flags = []  # (RowParallelLinear, orig_flag) to restore
        orig_tp_q_head_num = None
        try:
            # 逐个 attention linear 切 weight/scale，并缩小其局部 feature 尺寸。
            for linear in modules:
                for obj, attr_name, dim in self._get_linear_attrs(linear):
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))
                from sglang.srt.layers.linear import RowParallelLinear

                # Row/ColumnParallelLinear 的本地尺寸字段分别表示输入/输出分片。
                size_attr = (
                    "input_size_per_partition"
                    if isinstance(linear, RowParallelLinear)
                    else "output_size_per_partition"
                )
                # 模块内部 GEMM 形状也要匹配已经切分的权重。
                orig_size = getattr(linear, size_attr)
                setattr(linear, size_attr, orig_size // self.decode_tp_size)
                size_overrides.append((linear, size_attr, orig_size))

                # RowParallelLinear 需切换其 decode 专用的通信/归约分支。
                if isinstance(linear, RowParallelLinear):
                    row_parallel_decode_flags.append(
                        (linear, linear.use_decode_attn_tp)
                    )
                    linear.use_decode_attn_tp = True

            # 吸收式 Q/K/V 等不在标准 linear 对象中的张量同样需要切片。
            if tensor_attrs:
                for obj, attr_name, dim in tensor_attrs:
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))

            # RadixAttention 的 query-head 计数必须与局部 Q 投影一致。
            if radix_attn is not None:
                orig_tp_q_head_num = radix_attn.tp_q_head_num
                radix_attn.tp_q_head_num = orig_tp_q_head_num // self.decode_tp_size
            yield
        finally:
            # 按与激活相反的依赖顺序恢复 head 数、线性标志、尺寸和完整权重。
            if radix_attn is not None and orig_tp_q_head_num is not None:
                radix_attn.tp_q_head_num = orig_tp_q_head_num
            for linear, orig_flag in reversed(row_parallel_decode_flags):
                linear.use_decode_attn_tp = orig_flag
            for linear, size_attr, orig_size in reversed(size_overrides):
                setattr(linear, size_attr, orig_size)
            for obj, attr_name in reversed(all_attrs):
                self._restore(obj, attr_name)
