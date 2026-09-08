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

"""Base types and process-wide helpers for context parallel strategies.

The strategy implementation is split across:

* ``base.py``: base ABC, base metadata dataclass, enums, and singleton helpers.
* ``zigzag.py``: former in-seq-split strategy and zigzag metadata.
* ``interleave.py``: former round-robin-split strategy and interleave metadata.
* ``utils.py``: public re-exports for import convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


class ContextParallelStrategyKind(IntEnum):
    """Context parallel strategy identifiers."""

    NONE = 0
    ZIGZAG = 1
    INTERLEAVE = 2

    @classmethod
    def from_string(cls, value: str) -> ContextParallelStrategyKind:
        """将 CLI 字符串转换为内部策略枚举。

        ServerArgs 保持字符串形式以便序列化和展示，而执行路径使用枚举做
        分支。这里集中校验可选值，避免各个 attention 调用点重复解析。
        """
        # ``zigzag`` 将序列切成前后成对的 block。
        if value == "zigzag":
            return cls.ZIGZAG
        # ``interleave`` 按全局 token 下标对 CP rank 做轮询分配。
        if value == "interleave":
            return cls.INTERLEAVE
        raise ValueError(
            f"Unknown cp_strategy={value!r}; expected one of "
            "{'zigzag', 'interleave'}"
        )

    @property
    def cli_value(self) -> str:
        return {
            ContextParallelStrategyKind.NONE: "none",
            ContextParallelStrategyKind.ZIGZAG: "zigzag",
            ContextParallelStrategyKind.INTERLEAVE: "interleave",
        }[self]


class CPAttentionBackendKind(IntEnum):
    """Attention backend calling convention used by CP strategy dispatch."""

    FLASH_ATTENTION = 0
    DSA = 1
    TRTLLM_MHA = 2

    @classmethod
    def from_string(cls, value: str) -> CPAttentionBackendKind:
        if value in ("fa3", "fa4", "flashinfer"):
            return cls.FLASH_ATTENTION
        if value in ("dsa"):
            return cls.DSA
        if value == "trtllm_mha":
            return cls.TRTLLM_MHA
        raise ValueError(
            f"Unsupported attention_backend={value!r} for CP strategy; expected one "
            "of {'fa3', 'fa4', 'flashinfer', 'dsa', 'trtllm_mha'}"
        )


@dataclass
class BaseContextParallelMetadata:
    total_seq_lens: int = 0
    bs: int = 1


class ContextParallelStrategy(ABC):
    """Owns process-wide policy for one context parallel layout."""

    name: str
    kind: ContextParallelStrategyKind

    def __init__(self, cp_size: int):
        self.cp_size = cp_size

    @property
    def cp_rank(self) -> int:
        """返回当前进程在 attention-CP 通信组内的局部 rank。

        不能使用全局 rank：DP/TP 与 CP 可以组合，只有 ``attn_cp_rank``
        才是切分 token、构造 all-gather 索引时使用的 rank 坐标。
        """
        return get_parallel().attn_cp_rank

    @property
    def per_layer_attn_cp_comm(self) -> bool:
        return _is_dsa_active()

    @abstractmethod
    def can_apply(self, num_tokens: int, forward_batch: ForwardBatch) -> bool:
        """Return True if this strategy can shard the current forward."""

    @abstractmethod
    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> BaseContextParallelMetadata:
        """Build per-forward metadata for this strategy."""

    @abstractmethod
    def shard_hidden_states(self, x: Any, forward_batch: ForwardBatch) -> Any:
        """Shard hidden states to the current CP rank, usually at the first layer."""

    @abstractmethod
    def shard_position_ids(self, positions: Any, forward_batch: ForwardBatch) -> Any:
        """Shard KV-cache slot position IDs for each token to the current CP rank."""

    @abstractmethod
    def gather_hidden_states(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local hidden states, usually at the last layer."""

    @abstractmethod
    def gather_kv_cache(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local KV payloads back to full token order."""

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ) -> Tuple[List[int], Any, List[int], Any]:
        raise NotImplementedError(
            f"{self.name} strategy does not support per-request sharding"
        )

    def shard_local_tokens(self, input_: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} strategy does not support local-token sharding"
        )

    def materialize_full_indexer_k_cache(
        self, key: Any, forward_batch: ForwardBatch
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} strategy does not support DSA indexer key gather"
        )

    def all_gather_dsa_trtllm_fp8_kv(
        self, forward_batch: ForwardBatch, k: Any, k_rope: Any
    ) -> Any:
        raise NotImplementedError(
            f"{self.name} strategy does not support DSA trtllm FP8 KV gather"
        )

    def split_before_forward(
        self,
        forward_batch: ForwardBatch,
        input_ids: Optional[Any],
        positions: Any,
        input_embeds: Optional[Any] = None,
    ) -> Optional[Any]:
        """在进入 model.forward 前把全局输入改写为本 CP rank 的局部输入。

        ``cp_v2_input_ids`` 保留 CP 局部 token ID，供模型内部需要 token 的路径使用；
        positions 同步采用相同布局，保证 RoPE 与 KV slot 对齐。若调用方已算好
        embedding，则返回对应局部 embedding，避免模型重复做 embedding 查表。
        """
        # 输入 ID 与 hidden states 的 token 切分规则必须完全一致。
        if input_ids is not None:
            forward_batch.cp_v2_input_ids = self.shard_hidden_states(
                input_ids, forward_batch
            )
        # 无论是否有 input_ids，位置编码都要随局部 token 一起切分。
        forward_batch.positions = self.shard_position_ids(positions, forward_batch)
        # embedding 是可选的：prefill CUDA graph 路径会在图外先完成查表。
        if input_embeds is not None:
            return self.shard_hidden_states(input_embeds, forward_batch)
        return None

    @abstractmethod
    def run_attention(
        self,
        q: Any,
        forward_batch: ForwardBatch,
        device: Any,
        attn_fn: Callable[[Any, Any, Any, int], Any],
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        """Dispatch CP attention using the selected backend convention."""

    @abstractmethod
    def materialize_full_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        """Write full-layout K/V to the backend cache if needed."""

    @abstractmethod
    def materialize_full_mla_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any,
        k_nope: Any,
        k_rope: Any,
    ) -> Any:
        """Materialize full-layout MLA K/V for the strategy."""

    def reindex_attn_metadata(self, core_attn_metadata: Any) -> None:
        """Optional attention metadata rewrite for strategies that need it."""
        return None


def _is_dsa_active() -> bool:
<<<<<<< Updated upstream
    # Placeholder: a real answer needs the model architecture, not config.
    return False
=======
    """判断当前进程是否声明了需要逐层 CP attention 通信的 DSA 模型。

    该谓词只影响策略的通信时机，不参与普通 CP-v2 的启停判断。模型架构标记
    由运行时 ServerArgs 提供，因而延迟导入以避免模块初始化环。当前保留原有
    兼容字段，便于后续接入实际 DSA 架构检测时不改变调用点。
    """
    from sglang.srt.runtime_context import get_parallel, get_server_args

    # `_is_dsa_model_arch` is set nowhere in the tree, so this predicate is
    # inert today (the getattr default makes it False). Kept verbatim rather
    # than "fixed" here, because deciding what it should name is the CP path's
    # call; the ratchet exempts it with that reason.
    return bool(
        get_parallel().enable_prefill_cp
        and getattr(get_server_args(), "_is_dsa_model_arch", False)
    )
>>>>>>> Stashed changes


_STRATEGY: Optional[ContextParallelStrategy] = None


def init_cp_strategy(server_args: ServerArgs) -> None:
    """依据当前进程的 ServerArgs 创建 CP 策略单例。

    只有 prefill CP 打开且 attention-CP 组规模大于 1 才会实例化策略；其余
    情况显式清空单例，调用方即可退化为普通 prefill。具体策略延迟导入，避免
    base 模块与实现模块在服务参数解析阶段形成循环依赖。
    """
    global _STRATEGY

    # 关闭开关时必须清空旧实例，避免测试或 worker 复用进程遗留策略。
    if not getattr(server_args, "enable_prefill_cp", False):
        _STRATEGY = None
        return

    # CP 组只有一个 rank 时不存在切分和 collective，直接走非 CP 路径。
    cp_size = getattr(server_args, "attn_cp_size", 1)
    if cp_size <= 1:
        _STRATEGY = None
        return

    # 将用户字符串变成枚举，后续分支避免依赖拼写。
    kind = ContextParallelStrategyKind.from_string(server_args.cp_strategy)
    if kind == ContextParallelStrategyKind.ZIGZAG:
        # Zigzag 服务 FlashAttention/TRT-LLM MHA 的双端块布局。
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        _STRATEGY = ZigzagCPStrategy(cp_size=cp_size)
    elif kind == ContextParallelStrategyKind.INTERLEAVE:
        # Interleave 服务 DSA 的轮询 token 布局。
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        _STRATEGY = InterleaveCPStrategy(cp_size=cp_size)
    else:
        raise ValueError(
            f"Unsupported cp_strategy kind {kind} for "
            f"cp_strategy={server_args.cp_strategy!r}"
        )


def get_cp_strategy() -> Optional[ContextParallelStrategy]:
    """Return the configured strategy, initializing lazily on first call.

    Subprocesses re-import this module with ``_STRATEGY = None`` and never
    re-run ``ServerArgs.__post_init__`` because the pickled instance bypasses
    ``__init__``. Lazy init lets worker processes recover the singleton from
    global server args.
    """
    global _STRATEGY

    # 主进程通常已初始化；只有 worker 子进程需要从全局参数懒恢复。
    if _STRATEGY is None:
        from sglang.srt.runtime_context import get_server_args

        # 运行时上下文尚未就绪时（如独立单测）没有可恢复的策略。
        try:
            server_args = get_server_args()
        except ValueError:
            return None
        # 与初始化时一致，只在 prefill CP 已启用时建立策略。
        if server_args is not None and get_parallel().enable_prefill_cp:
            init_cp_strategy(server_args)
    return _STRATEGY


def get_cp_strategy_kind() -> ContextParallelStrategyKind:
    strategy = get_cp_strategy()
    if strategy is None:
        return ContextParallelStrategyKind.NONE
    return strategy.kind


def is_cp_enabled() -> bool:
    return get_cp_strategy() is not None


def is_zigzag() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.ZIGZAG


def is_interleave() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.INTERLEAVE
