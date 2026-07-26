from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

# 本文件定义了 Radix Cache 的可插拔淘汰策略（Eviction Policy）。
# 每个策略类实现了 get_priority 方法，返回一个优先级值用于排序；
# 优先级越低的节点越先被淘汰。支持 LRU、LFU、FIFO、MRU、FILO、
# 按优先级淘汰以及分段 LRU（SLRU）等多种策略。


class EvictionStrategy(ABC):
    """淘汰策略的抽象基类，所有具体策略必须实现 get_priority 方法。"""

    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    """最近最少使用策略：优先淘汰最久未被访问的节点。"""

    def get_priority(self, node: TreeNode) -> float:
        # 返回节点的最后访问时间，时间越早（值越小）越先被淘汰
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    """最不经常使用策略：优先淘汰命中次数最少的节点，命中次数相同时按 LRU 决定。"""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # 返回 (命中次数, 最后访问时间)，命中次数少的优先淘汰
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    """先进先出策略：优先淘汰最早创建的节点。"""

    def get_priority(self, node: TreeNode) -> float:
        # 返回节点创建时间，创建越早越先被淘汰
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    """最近最常使用策略：优先淘汰最近被访问过的节点（与 LRU 相反）。"""

    def get_priority(self, node: TreeNode) -> float:
        # 取负值使得最近访问的节点优先级更低（更先被淘汰）
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    """后进先出策略：优先淘汰最近创建的节点（与 FIFO 相反）。"""

    def get_priority(self, node: TreeNode) -> float:
        # 取负值使得最近创建的节点优先级更低（更先被淘汰）
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        # 返回 (优先级值, 最后访问时间)，优先级值低的先淘汰，相同优先级按 LRU
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    """分段 LRU 策略：将节点分为试用段和保护段，命中次数超过阈值的节点进入保护段，不易被淘汰。"""

    def __init__(self, protected_threshold: int = 2):
        # 命中次数达到此阈值的节点将从试用段晋升到保护段
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        # 根据命中次数判断节点属于试用段(0)还是保护段(1)
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        # 试用段的节点总是先于保护段被淘汰，同一段内按访问时间排序
        return (is_protected, node.last_access_time)
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
