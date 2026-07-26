from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)

# 本文件定义了前缀缓存(Prefix Cache)的抽象基类和相关数据结构。
# 所有缓存实现(如 RadixCache)都需要继承 BasePrefixCache 并实现其抽象方法。
# 文件中还定义了匹配、插入、驱逐、锁引用等操作的参数和结果数据类，
# 为不同缓存类型提供统一的接口规范。
# MatchResult 描述了前缀匹配操作的返回结果，包括设备索引和主机命中信息。

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        ComponentType,
    )


# PrefixCacheTrait 定义了前缀缓存必须具备的属性接口。
# 所有实现前缀缓存的类都必须拥有请求到token池、token到KV池分配器、页大小等属性。
@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


# MatchPrefixParams 封装了前缀匹配操作所需的参数。
# key 是用于查找缓存前缀的 RadixKey，包含 token 序列和可选的额外标识。
# cow_mamba 和 req 是 Mamba 模型特定的参数。
@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


# InsertParams 封装了缓存插入操作所需的参数。
# key 和 value 分别是插入的键(token序列)和值(KV缓存索引)。
# mamba_value 用于 Mamba 模型的状态缓存。
# prev_prefix_len 和 swa_evicted_seqlen 用于滑动窗口注意力(SWA)场景。
# chunked 标识是否为分块插入，priority 用于优先级感知的驱逐策略。
@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0


# InsertResult 封装了缓存插入操作的返回结果。
# prefix_len 表示插入前已存在的前缀长度(即有多少token已被缓存)。
# last_device_node 指向插入后树中最后匹配的节点，用于后续的锁引用管理。
@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    inserted_host_node: Any = None


# EvictParams 封装了缓存驱逐操作的参数。
# num_tokens 指定需要驱逐的 token 数量，用于释放KV缓存空间。
# swa_num_tokens 和 mamba_num 分别用于滑动窗口注意力和 Mamba 模型的驱逐。
@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


# EvictResult 封装了缓存驱逐操作的返回结果。
# 记录了实际驱逐的 token 数量，供调用方确认释放了多少资源。
@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""

    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    # Component nodes that were tombstones at acquire time. Replaying this set
    # at release prevents a short-lived lock from consuming a later load-back or
    # request lock after that tombstone becomes a valid device value.
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            swa_uuid_for_host_lock=self.swa_uuid_for_host_lock,
            skip_lock_node_ids={
                component_type: set(node_ids)
                for component_type, node_ids in self.skip_lock_node_ids.items()
            },
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""

    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


# MatchResult 封装了前缀匹配操作的返回结果，是一个命名元组。
# device_indices 是匹配到的KV缓存在设备上的索引张量。
# last_device_node 和 last_host_node 分别指向设备和主机上最后匹配的树节点。
# best_match_node 是所有组件验证器接受的最深节点，作为加载回传的锚点。
# host_hit_length 表示需要从主机(CPU)加载回设备的 token 数量。
# swa_host_hit_length 和 mamba_host_hit_length 分别记录滑动窗口注意力和 Mamba 的主机命中数。
class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None


def zero_match_result(tree_cache, match_result: MatchResult) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
    )


# BasePrefixCache 是前缀缓存的抽象基类，定义了缓存系统的核心接口。
# 所有具体的缓存实现（如 RadixCache、ChunkCache）都需要继承此类。
# 它提供了前缀匹配、插入、驱逐、锁引用管理等核心操作的抽象方法。
# 同时包含了大小查询、打印、加载回传等辅助方法的默认实现。
class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )

    def init_metrics_collector(self):
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        labels = {"cache_type": self.__class__.__name__}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
            server_args,
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    def release_host_resources(self) -> None:
        """Release pinned host buffers in userspace on graceful shutdown.

        Kernel-side unpinning during process reclaim can stall teardown for
        tens of seconds (see HostKVCache.destroy). Idempotent.
        """

    # reset 重置缓存到初始状态，清空所有缓存的数据结构和统计信息。
    @abstractmethod
    def reset(self):
        pass

    # match_prefix 在缓存中查找给定 key 的最长匹配前缀。
    # 返回 MatchResult，包含匹配到的KV缓存索引和最后匹配的节点。
    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    def supports_fast_match_prefix(self) -> bool:
        return False

    # cache_finished_req 在请求完成时缓存其KV缓存。
    # 将已完成请求的 token 序列和对应的KV缓存索引插入缓存树中。
    # 释放请求占用的临时锁引用，并回收已存在于树中的重复部分。
    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    # cache_unfinished_req 在请求尚未完成时缓存其当前的KV缓存。
    # 用于分块预填充(chunked prefill)场景，允许中间结果被缓存和复用。
    # 更新请求的前缀索引和锁引用，以便后续继续生成时复用已计算的KV缓存。
    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    # evict 从缓存中驱逐指定数量的 token 以释放内存空间。
    # 驱逐策略通常基于LRU(最近最少使用)或优先级，选择最不重要的节点进行清理。
    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    # inc_lock_ref 增加节点的锁引用计数，防止该节点被驱逐。
    # 沿树路径向上遍历，将引用链上所有节点标记为受保护状态。
    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    # dec_lock_ref 减少节点的锁引用计数，当计数降为零时节点可被驱逐。
    # 与 inc_lock_ref 对应，沿树路径向上遍历更新引用计数和可驱逐大小。
    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    # evictable_size 返回可被驱逐的 token 总数。
    # 这些 token 当前没有被任何请求引用，可以在内存不足时被回收。
    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    # protected_size 返回受保护的 token 总数。
    # 这些 token 正被活跃请求引用，不能被驱逐。
    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def flush_write_through_acks(self) -> None:
        """Release lock_ref on radix-tree nodes whose write-through has completed.

        Lightweight operation that only processes finished write acks.
        No-op for caches without hierarchical write-through support.
        """
        pass

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []

    def supports_swa(self) -> bool:
        return False

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        return False

    def supports_streaming_session(self) -> bool:
        return False

    def release_session(self, session_id: str) -> None:
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
