from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        TreeComponent,
    )

# 此文件定义了 KV 缓存初始化参数的数据类 CacheInitParams。
# 该数据类封装了前缀缓存（Radix Cache）创建所需的全部配置，
# 包括内存池引用、页大小、驱逐策略、分布式通信组、推测解码标志等。
# 在服务器启动时由 Scheduler 构造并传递给缓存管理器。

@dataclasses.dataclass
class CacheInitParams:
    # 缓存是否禁用；禁用时跳过前缀缓存逻辑
    disable: bool
    # 请求到 token 的映射池，管理请求级别的 KV 索引分配
    req_to_token_pool: ReqToTokenPool
    # token 到 KV 缓存的分配器，管理底层物理内存的分配与回收
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    # 页大小；大于1时启用页式内存管理以减少碎片
    page_size: int

    # 是否使用 Eagle 推测解码，影响缓存的 KV 层数和分配策略
    is_eagle: bool = False
    # 分布式通信组：用于张量并行（TP）、上下文并行（CP）和流水线并行（PP）的缓存同步
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    pp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    # 缓存驱逐策略，支持 lru/lfu/fifo/mru/filo/priority/slru
    eviction_policy: str = "lru"
    # 是否禁止将已完成的请求插入前缀缓存（用于调试或特殊场景）
    disable_finished_insert: bool = False

    # 是否启用缓存命中率等可观测性指标收集
    enable_metrics: bool = False
    # 是否启用 KV 缓存事件追踪（用于调试和性能分析）
    enable_kv_cache_events: bool = False
    # 是否启用会话级 Radix 缓存（跨请求的前缀复用）
    enable_session_radix_cache: bool = False

    # 是否为 Mamba 架构模型分配额外的 ping-pong 状态缓冲区
    enable_mamba_extra_buffer: bool = False
    # 是否使用懒加载模式分配 Mamba 缓冲区（按需分配以节省内存）
    enable_mamba_extra_buffer_lazy: bool = False

    # 流水线并行的当前 rank 和总 size，用于确定缓存的层级归属
    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    # 滑动窗口大小（token 数），用于滑动窗口注意力模型的缓存管理
    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    # 树缓存的组件类型列表，用于模块化的缓存架构（如 KV 组件、Mamba 组件等）
    tree_components: Optional[tuple[ComponentType, ...]] = None
    # 组件注册表覆盖，允许自定义特定组件的实现类
    component_registry_override: Optional[dict[ComponentType, type[TreeComponent]]] = (
        None
    )
