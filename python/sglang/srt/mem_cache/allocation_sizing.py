from __future__ import annotations

from typing import Optional

from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs

# 此文件提供 KV 缓存内存池的分配大小计算函数。
# 核心功能是根据服务器配置（特别是推测解码参数）计算每步解码所需的
# KV 缓存分配长度、预留长度以及 req_to_token 行的额外空间。
# 这些计算直接影响内存预算和 OOM 防护策略的准确性。


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    # 计算每次解码步骤中每个请求需要分配的 KV 缓存长度（token 数）。
    # 非推测解码时固定为1；推测解码时取 max(topk * num_steps, num_draft_tokens)，
    # 以覆盖最坏情况下的草稿 token 空间需求。
    if server_args is None:
        server_args = get_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        # 简单情况：页大小为1或单分支，直接取 token 数量的最大值
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    # 每次解码步骤为每个请求预留的 KV 缓存长度。
    # 2倍系数是双缓冲机制：在重叠模式下吸收 kv_committed_len 的延迟更新，
    # 确保推测解码的提交和分配不会因时序问题导致越界。
    return 2 * get_alloc_len_per_decode(server_args)


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom.
    """
    # 计算 req_to_token 矩阵每行在模型上下文长度之外的额外空间。
    # 这些空间用于容纳推测解码的多分配（over-allocation），防止写入越界到相邻行。
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    # 基础余量 4 + 最大草稿 token 数；推测解码且页大小>1 时需额外考虑页对齐的溢出
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if server_args.speculative_algorithm is not None and server_args.page_size > 1:
        # kv_allocated_len is page-aligned (eagle_prepare_for_decode), so near
        # the context limit the aligned reserve can overshoot by page_size - 1;
        # without the headroom the row write silently lands in the neighbor row.
        extra = max(
            extra,
            get_alloc_reserve_per_decode(server_args) + server_args.page_size - 1,
        )
    return extra
