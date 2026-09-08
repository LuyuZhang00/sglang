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

"""Padding helpers shared by context-parallel strategies."""

from typing import Any

import torch

from sglang.srt.runtime_context import get_parallel


def get_cp_padding_align_size() -> int:
    """返回 CP collective 所需的 token 行数对齐粒度。

    Zigzag/in-sequence DSA 每个 rank 同时承载序列前后两个 block，因此其局部
    行数必须对 ``2 * cp_size`` 对齐；普通轮询布局只需对 ``cp_size`` 对齐。
    统一对齐后每个 rank 可以使用相同形状的 all-gather 缓冲区。
    """
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
    from sglang.srt.layers.utils.cp_utils import is_prefill_cp_in_seq_split

    # 从 parallel context 读取 attention-CP 组，而不是 TP/DP 组规模。
    attn_cp_size = get_parallel().attn_cp_size
    # 双端块布局要求每轮能同时容纳两个 CP block。
    if is_prefill_cp_in_seq_split() or is_dsa_prefill_cp_in_seq_split():
        return attn_cp_size * 2
    return attn_cp_size


def pad_logical_token_to_physical(metadata: Any) -> None:
    """把不均衡的逻辑 token 数扩展为等长的物理通信行数。

    ``per_rank_logical_token`` 保存真实 token 数，供 gather 后去除 padding；
    ``per_rank_actual_token`` 则改为每 rank 相同的物理长度，供 kernel 和
    collective 分配固定形状。metadata 中的原值在这里被视作逻辑长度。
    """
    # 先复制原始分配，后续 gather 仍需要知道每个 rank 的有效行数。
    logical_tokens = list(metadata.per_rank_actual_token)
    # 对齐粒度随 CP 策略/DSA 模式变化。
    align_size = get_cp_padding_align_size()
    # 向上取整到能被通信布局整除的最大局部长度。
    physical_rank_len = (
        (max(logical_tokens) + align_size - 1) // align_size * align_size
    )
    # 写回真实长度，作为反向重排和去 padding 的依据。
    metadata.per_rank_logical_token = logical_tokens
    # 所有 rank 使用相同物理长度，满足 all-gather 的输入形状要求。
    metadata.per_rank_actual_token = [physical_rank_len] * len(logical_tokens)
    metadata.max_rank_len = [physical_rank_len] * len(logical_tokens)


def pad_local_rows(x: torch.Tensor, metadata: Any, dim: int) -> torch.Tensor:
    """沿 token 维把当前 rank 的局部张量补齐到物理通信长度。

    仅补零，不改变逻辑 token 的相对顺序；attention 会用 metadata 忽略这些尾部
    行。没有启用物理 padding 或二者长度本来相等时直接返回原张量，避免额外复制。
    ``dim`` 支持 hidden states 的第 0 维和 position IDs 的最后一维。
    """
    # 未建立 logical/physical 区分时，策略不需要固定形状 collective。
    if (
        metadata.per_rank_logical_token is None
        or metadata.per_rank_logical_token == metadata.per_rank_actual_token
    ):
        return x

    # 各 rank 的 physical 长度相同，取第 0 项即可。
    target_len = metadata.per_rank_actual_token[0]
    # 只在局部逻辑 token 少于通信 buffer 时追加尾部零行。
    pad_size = target_len - x.shape[dim]
    assert pad_size >= 0
    if pad_size == 0:
        return x

    # 根据目标 token 维构造零 padding 的完整张量形状。
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
