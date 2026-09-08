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

"""Public import facade and runtime helpers for context parallel strategies."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
    {
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "GptOssForCausalLM",
        "MiMoV2FlashForCausalLM",
        "MiMoV2ForCausalLM",
        "Qwen3MoeForCausalLM",
        "DeepseekV3ForCausalLM",
    }
)


def is_glm_dsa_cache_layer_split_enabled(model_runner: "ModelRunner") -> bool:
    """Whether DSA GPU KV/indexer cache layers are sharded across CP ranks.

    Layer split is a prefill-CP-only optimization for DSA (DeepSeek Sparse
    Attention) MLA models (e.g. GLM-5.2). Draft workers keep the full cache.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    # 草稿模型需保留完整层缓存，不能参与跨 CP rank 的层级拆分。
    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )


def get_glm_dsa_cp_layer_shard_info(
    model_runner: "ModelRunner",
) -> Tuple[Optional[int], int]:
    """Return ``(layer_shard_rank, layer_shard_size)`` for the DSA KV pool.

    ``(None, 1)`` disables sharding (feature off or only one CP rank).
    """
    # 特性未开启时用 ``None`` 表示 KV pool 不按层分片。
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return None, 1
    # 分片轴就是 attention-CP 组；DP/TP rank 不参与层归属计算。
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return None, 1
    # 返回当前 CP rank，使缓存池只为本 rank 负责的层分配主存储。
    return get_parallel().attn_cp_rank, shard_size


def get_glm_dsa_layer_split_effective_num_layers(
    model_runner: "ModelRunner", num_layers: int
) -> int:
    """Per-rank owned layer count used when sizing the DSA KV cell.

    Under layer split each CP rank only stores ``ceil(num_layers / shard_size)``
    layers, plus one extra layer for the remote scratch buffer used when reading
    a layer owned by another CP rank.
    """
    # 非 DSA/未开启特性时，缓存容量仍按模型完整层数计算。
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return num_layers
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return num_layers
    # 向上取整，覆盖无法整除时拥有最多层的 rank。
    owned_layers_upper_bound = (num_layers + shard_size - 1) // shard_size
    # 额外一层用于访问远端 owner 层时的临时 scratch buffer。
    return max(1, owned_layers_upper_bound + 1)


def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> Tuple[int, int]:
    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.

    Layers are split as evenly as possible; the first ``total_layers %
    shard_size`` ranks own one extra layer.
    """
    # 每个 rank 至少拥有 ``base`` 层。
    base = total_layers // shard_size
    # 余数层从低 rank 起每 rank 多分配一层。
    rem = total_layers % shard_size
    # 先累加前面 rank 的基础层和额外层，得到本 rank 起点。
    start = rank * base + min(rank, rem)
    # 区间采用 Python 常用的左闭右开语义。
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
    """返回连续层切分下拥有指定模型层的 CP rank。

    DSA 层切分缓存需要据此决定本地读取还是从 owner rank 拉取。这里复用
    ``get_layer_shard_range``，保证容量规划、层映射和远端访问使用同一切分规则。
    无归属说明层号超出模型范围，立即报错而不是静默访问错误 KV 槽位。
    """
    # 小规模 CP 组下线性扫描足够清晰，并避免重复实现不均匀切分公式。
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
    )


def enable_cp_v2() -> bool:
    """读取进程级开关，判断是否允许进入 CP-v2 实现路径。

    CP-v2 与旧 CP-v1 并存，环境变量只表示代码路径可用，并不意味着当前 batch
    一定能切分；逐 batch 的几何和 forward mode 由 ``is_cp_v2_active`` 判断。
    将开关封装在此处可保持其他模块不直接依赖 envs 的具体字段。
    """
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_CP_V2.get())


def is_cp_v2_active(forward_batch) -> bool:
    """判断当前 forward batch 是否应走 CP-v2。

    判断顺序从进程开关、forward 类型、策略实例到输入和布局约束逐层收紧；任一
    条件不满足都返回 False，使调用方可安全退回普通或 CP-v1 路径。最后委托策略
    校验 token 数和序列长度，因为 zigzag 与 interleave 的最小几何不同。
    """
    # 环境变量关闭时无需继续读取 batch 元数据。
    if not enable_cp_v2():
        return False
    # 仅 extend prefill 有完整上下文可供沿序列维度切分。
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False

    # 策略可能因 worker 懒初始化失败或服务未启用 prefill CP 而为空。
    strategy = get_cp_strategy()
    if strategy is None:
        return False

    # 没有 token ID 说明不是标准模型输入，不能构造 CP 布局。
    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False

    # 由具体策略检查最小长度、每条序列约束等布局前提。
    return strategy.can_apply(len(input_ids), forward_batch)


def prepare_cp_forward(forward_batch) -> None:
    """为一个活跃 CP-v2 prefill batch 建立并规范化布局元数据。

    策略先按逻辑 token 构造 rank 分配、attention 几何和反向重排索引，再由 padding
    统一成 collective 所需的物理长度。该函数也同步 DP-attention 缓冲区和 cache
    写入位置，使模型层只需消费已经局部化的 ``forward_batch``。
    """
    # 该 helper 只能由已通过逐 batch 几何校验的 CP-v2 路径调用。
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    # 转为 Python int 列表，兼容 CPU tensor、list 及缺省 metadata。
    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    # 有逐请求 extend 长度时以其求和；否则回退到平铺 input_ids 的长度。
    num_tokens = (
        sum(extend_lens_cpu)
        if extend_lens_cpu is not None
        else len(forward_batch.input_ids)
    )
    # 同一 batch 可被多个入口使用，metadata 只在首次准备时构建一次。
    if forward_batch.attn_cp_metadata is None:
        forward_batch.attn_cp_metadata = strategy.build_metadata(
            num_tokens=num_tokens,
            seqs_len=seq_lens_cpu,
            extend_seqs_len=extend_lens_cpu,
        )
        # 保存逻辑长度后补齐物理行数，供后续 all-gather 使用固定形状。
        pad_logical_token_to_physical(forward_batch.attn_cp_metadata)

    # DP attention 需要知道本 DP rank 实际承载的 CP 物理缓冲区大小。
    if getattr(forward_batch, "global_num_tokens_cpu", None) is not None:
        from sglang.srt.layers.dp_attention import set_local_dp_buffer_len

        set_local_dp_buffer_len(
            sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
        )

    # 调度 padding 不是请求真实 token，不能写入 KV cache。
    if getattr(forward_batch, "out_cache_loc", None) is not None:
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]


def cp_split_before_forward(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
) -> Tuple[Optional[Any], Optional[Any]]:
    """把完整 embedding 与 position IDs 同步切为当前 CP rank 的局部输入。

    两个输出必须共享同一 metadata：embedding 的第 0 维 token 与 positions 中的
    token 维一一对应。函数只做模型边界的编排，具体 zigzag/interleave 索引由策略
    实现，从而让 ModelRunner 不需要了解不同 CP 布局的细节。
    """
    assert is_cp_v2_active(forward_batch)
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    # 先后委托同一策略，确保 hidden states 与位置编码采用一致 token 顺序。
    return (
        cp_shard_hidden_states(complete_hidden_states, forward_batch),
        cp_shard_position_ids(complete_position_ids, forward_batch),
    )


def cp_shard_hidden_states(complete_hidden_states: Any, forward_batch):
    """委托当前策略切分 token 维为 CP 局部 hidden states。

    前置断言将错误限制在模型边界：如果 CP-v2 未激活或 metadata 尚未建立，不能
    误把普通 batch 当作 CP 张量切分。具体实现还会按需追加物理 padding 行。
    """
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_hidden_states(complete_hidden_states, forward_batch)


def cp_shard_position_ids(complete_position_ids: Any, forward_batch):
    """委托当前策略切分与 hidden states 对齐的 position IDs。

    在 RoPE/DSA 路径中，位置的 token 排列必须和 Q/K/V 写入 KV cache 的排列一致。
    断言确保该函数不会被 decode 或未准备 metadata 的普通 prefill 错误调用。
    """
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_position_ids is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_position_ids(complete_position_ids, forward_batch)


def cp_round_robin_input_ids_v2(input_ids: Any, forward_batch):
    """为 CP-v2 的 MoE 调度准备与物理布局一致的局部 token ID 顺序。

    有 MoE A2A 后端时，后端本身理解策略切分，直接复用标准 hidden-state 切分；
    否则将全局 token 先补到所有 rank 的物理总长度，再按 rank 转置取得轮询顺序。
    这保证 token ID 与已经固定形状的 CP 通信缓冲区逐行对应。
    """
    assert is_cp_v2_active(forward_batch)
    # A2A 后端需要策略原始局部布局，而非下面的扁平转置布局。
    if not get_moe_a2a_backend().is_none():
        return cp_shard_hidden_states(input_ids, forward_batch)

    # 物理总长度包含为 collective 增补的零 token 槽位。
    physical_tokens = sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
    # 保留原始 token 在前，padding token ID 为零且不会进入真实 cache slot。
    padded_input_ids = input_ids.new_zeros(physical_tokens)
    padded_input_ids[: input_ids.shape[0]] = input_ids
    return padded_input_ids.view(-1, get_parallel().attn_cp_size).T.flatten()


def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
    """在模型末端把各 CP rank 的局部输出恢复为全局逻辑 token 顺序。

    logits processor 以原 batch token 顺序消费 hidden states，因此 gather 必须在它
    之前执行。输出可以是模型额外携带的 tuple；每个非空元素独立 gather，并保留
    MiMo 文本路径对 ``(hidden_states, None)`` 返回纯 tensor 的兼容行为。
    """
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    # 辅助输出（如 capture 的 states）与主输出共享相同 CP token 维。
    if isinstance(x, tuple):
        gathered = tuple(
            (
                strategy.gather_hidden_states(item, forward_batch, stream)
                if item is not None
                else None
            )
            for item in x
        )
        # MiMo 文本 body 的第二项为空，而 logits 接口期望直接收到 tensor。
        if len(gathered) == 2 and gathered[1] is None:
            return gathered[0]
        return gathered

    return strategy.gather_hidden_states(x, forward_batch, stream)


def cp_materialize_global_token_order(
    x: Any, forward_batch, stream: Optional[Any] = None
):
    """将 CP 张量物化为原始全局逻辑 token 顺序。

    CP-v2 由策略处理其各自的 padding 和重排；旧 CP-v1 仍临时保留原有 rerange
    实现，以免混合部署时改变语义。该入口主要用于 KV/indexer 一类必须按全局
    token 位置写回或读取的数据，而不局限于最终 hidden states。
    """
    # 新路径的 metadata 已记录每个 rank 的有效长度和重排索引。
    if is_cp_v2_active(forward_batch):
        strategy = get_cp_strategy()
        assert strategy is not None
        return strategy.gather_kv_cache(x, forward_batch, stream)

    # TODO(hzh0425): Keep the legacy gather temporarily for CP-v1 compatibility. Remove it
    # with the follow-up CP-v1 cleanup.
    from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output

    # 旧路径没有策略对象，继续使用 CP-v1 的通信及重排实现。
    return cp_all_gather_rerange_output(
        x, get_parallel().attn_cp_size, forward_batch, stream
    )


@contextmanager
def cp_shard_model_inputs(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
):
    """在上下文块内提供局部模型输入，并在退出时恢复共享 speculative 状态。

    ModelRunner 可以在 ``with`` 块内使用 CP 局部 embedding/positions，而 batch 的
    其他字段仍保持全局视角。speculative hidden states 恰好覆盖完整 token 数时也
    需要一起切分；finally 恢复原引用，保证后续 logits/调度不会看到局部张量。
    """
    assert is_cp_v2_active(forward_batch)
    # 主输入以策略布局切分，供当前 CP rank 执行 transformer body。
    sharded_hidden_states = cp_shard_hidden_states(
        complete_hidden_states, forward_batch
    )
    sharded_positions = cp_shard_position_ids(complete_position_ids, forward_batch)

    # speculative 信息是可选的，先安全取得嵌套字段。
    spec_info = getattr(forward_batch, "spec_info", None)
    spec_hidden_states = getattr(spec_info, "hidden_states", None)
    spec_hidden_states_backup = None
    # 仅在其 token 维确实对应完整 batch 时切分，避免误处理局部草稿张量。
    if (
        spec_hidden_states is not None
        and spec_hidden_states.shape[0] == complete_hidden_states.shape[0]
    ):
        # 覆盖为局部视图；退出 context 后必须恢复原始全局张量。
        spec_hidden_states_backup = spec_hidden_states
        spec_info.hidden_states = cp_shard_hidden_states(
            spec_hidden_states, forward_batch
        )

    try:
        yield sharded_hidden_states, sharded_positions
    finally:
        # 即使模型 forward 抛错，也恢复共享 ForwardBatch 的全局语义。
        if spec_hidden_states_backup is not None:
            spec_info.hidden_states = spec_hidden_states_backup


def _to_int_list(values) -> Optional[list[int]]:
    """把可选的 CPU tensor/序列规范成 Python ``int`` 列表。

    metadata 构造位于 Python 侧，不应依赖 tensor 标量的隐式转换；``tolist`` 也
    让输入兼容 numpy-like 容器。缺失值保持 ``None``，由调用者选择回退语义。
    """
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "ContextParallelMetadata",
    "ContextParallelStrategy",
    "ContextParallelStrategyKind",
    "InterleaveCPStrategy",
    "InterleaveContextParallelMetadata",
    "ZigzagCPStrategy",
    "ZigzagContextParallelMetadata",
    "CP_V2_DEFAULT_MODEL_CLASSES",
    "enable_cp_v2",
    "get_cp_strategy",
    "is_cp_v2_active",
    "cp_gather_after_forward",
    "cp_materialize_global_token_order",
    "cp_round_robin_input_ids_v2",
    "cp_shard_hidden_states",
    "cp_shard_model_inputs",
    "cp_shard_position_ids",
    "cp_split_before_forward",
    "prepare_cp_forward",
    "is_glm_dsa_cache_layer_split_enabled",
    "get_glm_dsa_cp_layer_shard_info",
    "get_glm_dsa_layer_split_effective_num_layers",
    "get_layer_shard_range",
    "get_layer_owner",
]
