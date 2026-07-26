from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.kernels.ops.memory.common import (
    _get_last_loc_safe_kernel as _get_last_loc_safe_kernel,
)
from sglang.kernels.ops.memory.common import get_last_loc_kernel as get_last_loc_kernel
from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
    maybe_evict_dsv4_state_on_swa,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# 此文件提供 KV 缓存管理的核心公共函数。
# 主要功能包括：KV 索引与页索引之间的转换、滑动窗口注意力（SWA）的过期槽位释放、
# 树缓存的驱逐触发、以及请求完成时 KV 缓存的释放与回收逻辑。
# 这些函数是调度器（Scheduler）与缓存管理器之间的桥梁，被热路径频繁调用。

# Needs 2 + 1 slots for mamba request with prefix cache. 2 for ping pong cache, 1 for running mamba state.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
# Lazy mode: 1 + 1 slots (1 ping-pong + 1 running), second ping-pong allocated on demand at boundary.
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
# Mamba 状态槽位常量：分别定义了有前缀缓存、懒加载模式和无缓存三种场景下
# 每个请求所需的 Mamba 状态槽位数量，用于 Mamba 架构模型的内存预算计算。
MAMBA_STATE_PER_REQ_NO_CACHE = 1

logger = logging.getLogger(__name__)


def kv_to_page_indices(kv_indices: torch.Tensor, page_size: int) -> np.ndarray:
    # 将连续的 KV 缓存索引转换为页索引数组。
    # 按 page_size 步长采样并整除，用于页式内存池中快速定位物理页。
    return (kv_indices[::page_size] // page_size).cpu().numpy()


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # 向上取整除法计算给定 KV 索引数量所需的页数。
    return (num_kv_indices + page_size - 1) // page_size


def page_align_floor(length: int, page_size: int) -> int:
    # 将长度向下对齐到页边界，用于确保 KV 缓存操作不会跨越页边界。
    return (length // page_size) * page_size


def free_swa_out_of_window_slots(
    req: Req,
    pre_len: int,
    *,
    sliding_window_size: int,
    page_size: int,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    is_chunk_cache: bool = False,
) -> None:
    # 释放滑动窗口注意力（SWA）中超出窗口范围的 KV 缓存槽位。
    # 随着序列增长，旧的 KV 缓存不再被注意力机制访问，需要及时回收以节省显存。
    # 函数根据 Radix 缓存或 Chunk 缓存模式计算不同的驱逐阈值，确保不破坏缓存一致性。
    if req.kv is None:
        return

    # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
    assert (
        req.cache_protected_len % page_size == 0
    ), "cache_protected_len must be page aligned"
    evict_floor = max(req.cache_protected_len, getattr(req, "swa_evict_floor", 0))
    if page_size > 1 and evict_floor > req.cache_protected_len:
        evict_floor = -(-evict_floor // page_size) * page_size
    req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, evict_floor)

    if is_chunk_cache:
        # Chunk cache builds no radix tree, so no tombstone-leaf concern; evict
        # up to the window boundary (the trailing floor keeps it page-aligned).
        evict_threshold = pre_len - sliding_window_size
    else:
        # Radix cache: keep max(window, page). The trailing floor page-aligns the
        # frontier, and subtracting at least one page keeps it below the insert
        # boundary (page_floor(seq_len)) so the last leaf is never all-tombstone.
        # No extra page margin is needed.
        evict_threshold = pre_len - max(sliding_window_size, page_size)
    new_swa_evicted_seqlen = max(
        req.kv.swa_evicted_seqlen,
        evict_threshold,
    )

    if page_size > 1:
        new_swa_evicted_seqlen = (new_swa_evicted_seqlen // page_size) * page_size

    if new_swa_evicted_seqlen > req.kv.swa_evicted_seqlen:
        free_slots = req_to_token_pool.req_to_token[
            req.req_pool_idx, req.kv.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        token_to_kv_pool_allocator.free_swa(free_slots)
        maybe_evict_dsv4_state_on_swa(
            token_to_kv_pool_allocator, req_to_token_pool, req, new_swa_evicted_seqlen
        )
        req.kv.swa_evicted_seqlen = new_swa_evicted_seqlen


def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    # 当可用 KV 缓存空间不足时，从树缓存中驱逐指定数量的 token 以释放空间。
    # 对于 SWA 混合分配器，需同时考虑全注意力和滑动窗口两层的可用空间；
    # 对于标准分配器，仅驱逐差额部分。
    if tree_cache is None:
        return

    if tree_cache.is_chunk_cache():
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    if isinstance(allocator, SWATokenToKVPoolAllocator):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(
                EvictParams(num_tokens=full_num_tokens, swa_num_tokens=swa_num_tokens)
            )
    else:
        # Standard allocator: evict only the shortfall (mirrors the SWA arm)
        available_size = allocator.available_size()
        if available_size < num_tokens:
            tree_cache.evict(EvictParams(num_tokens=num_tokens - available_size))


def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # 释放请求的 KV 缓存资源，是缓存生命周期管理的核心函数。
    # 处理流程：先将已完成的 KV 缓存插入前缀缓存树（如启用）、
    # 再释放多分配的 KV 索引、最后回收请求池和 Mamba 状态（如有）。
    # the two resources currently have the same lifecycle, thus simplify logic below
    assert (req.req_pool_idx is None) == (req.kv is None)
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        # TODO (csy, hanming): clean up this early allocation logic
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    effective_kv_committed_len = req.effective_kv_committed_len()
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=effective_kv_committed_len,
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (req.req_pool_idx is None) == (req.kv is None)
    if req.req_pool_idx is None and req.kv is None:
        return

    start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert (
            req.mamba_pool_idx is not None
        ), "mamba state is freed while the tree cache does not manage mamba states"
        tree_cache.req_to_token_pool.free_mamba_cache(req)
    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv = None


def _release_overallocated_kv_indices(
    req: Req, start_p: int, end_p: int, tree_cache: BasePrefixCache
) -> None:
    # 释放请求中多分配的 KV 缓存索引（committed_len 到 allocated_len 之间的部分）。
    # 这些"溢出"的索引通常由推测解码或 strip_thinking_cache 策略产生。
    global_server_args = get_server_args()
    page_size = global_server_args.page_size
    spec_algo = global_server_args.speculative_algorithm

    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not global_server_args.strip_thinking_cache:
        assert (
            start_p == end_p
        ), f"Unexpected overallocated KV cache, {req.kv_committed_len=}, {req.kv.kv_allocated_len=}"

    if page_size > 1:
        start_p = ceil_align(start_p, page_size)

    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx][
            start_p:end_p
        ]
        tree_cache.token_to_kv_pool_allocator.free(indices_to_free)


def available_and_evictable_str(tree_cache: BasePrefixCache) -> str:
    # 返回树缓存的可用和可驱逐空间的描述字符串，用于日志和可观测性报告。
    return tree_cache.available_and_evictable_str()
