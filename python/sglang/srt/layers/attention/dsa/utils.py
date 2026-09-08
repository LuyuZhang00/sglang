"""DeepSeek Sparse Attention（DSA）的公共判定、切分与布局工具。

这里集中处理平台能力探测、Prefill Context Parallel（CP）切分、DP/CP
补齐长度，以及 FP8 分页 MQA 缓存的物理布局转换。把这些规则放在同一处
可以保证 Indexer、attention backend 和 CUDA Graph 路径采用相同语义。
"""

from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    process_model_config,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip
from sglang.srt.utils.common import ceil_align, ceil_div


@lru_cache(maxsize=1)
def aiter_can_use_preshuffle_paged_mqa() -> bool:
    """Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime.

    aiter's ``deepgemm_fp8_paged_mqa_logits`` only supports ``KVBlockSize > 1`` and
    ``Preshuffle=True`` on its gluon kernel path. The gluon path is enabled when
    Triton >= 3.5.0, OR when ``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`` is set
    (which additionally requires that the AOT gluon kernel artifacts ship inside
    the aiter wheel/image). Otherwise aiter asserts ``KVBlockSize == 1`` and
    refuses ``Preshuffle=True``.

    sglang's DSA indexer uses this single decision to pick:
      * ``page_size``: 64 (preshuffle) vs 1 (legacy) on ROCm
      * ``Preshuffle`` / ``preshuffle`` flags on the aiter MQA + cache kernels
      * ``get_page_table_64`` vs ``get_page_table_1`` on the metadata
      * whether ``GetKAndS.execute`` uses the aiter or the triton implementation

    The result is cached so the cost is paid once per process.

    Set ``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1`` to force the legacy path even when
    the gluon kernel would otherwise be available (useful for CI bisection).
    ``SGLANG_NSA_HIP_DISABLE_PRESHUFFLE`` is a deprecated alias.
    """
    # preshuffle 分页 MQA 是 AITER/ROCm 专属能力，CUDA 不参与探测。
    if not is_hip():
        return False
    # 即使处于 HIP 平台，也必须显式启用 SGLANG_USE_AITER。
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return False
    # 调试开关优先级最高，可强制回退到 page_size=1 的旧布局。
    if envs.SGLANG_DSA_HIP_DISABLE_PRESHUFFLE.get():
        return False
    # 随 wheel 分发的 AOT Gluon kernel 不依赖本机 Triton 版本。
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):
        return True
    try:
        from packaging.version import Version

        # Triton 3.5 起提供满足 preshuffle 路径要求的 Gluon kernel 支持。
        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception:
        # 版本解析或依赖导入失败时保守回退，确保服务仍可使用旧路径启动。
        return False


# Tile size for the indexer FP8 K-cache preshuffle layout. Store and gather
# kernels reorganize each page into (tile x tile) blocks so the aiter preshuffle
# paged-MQA gather can consume the cache directly.
INDEXER_K_CACHE_PRESHUFFLE_TILE = 16


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int):
    """把每个请求的有效 DSA 长度截断到 Top-K 上限。

    Indexer 不需要为超过 ``dsa_index_topk`` 的候选保留输出空间。
    该结果常被扩展到 token 维度，并作为 Top-K kernel 的逐行有效长度。
    """
    # 短序列保留真实长度，长序列只允许选择配置的最大候选数。
    return original_seq_lens.clamp(max=dsa_index_topk)


<<<<<<< Updated upstream
def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
=======
def should_remap_pd_dsa_seed_to_local_slots(server_args: "ServerArgs") -> bool:
    """判断 PD 传来的 DSA seed 是否要重映射到本地槽位。

    Decode 端的 KV allocator 与 Prefill 端不共享逻辑槽位编号，融合 Top-K
    因而必须在满足 CUDA、非 HiSparse、非 DCP 等约束时使用本地域。
    """
>>>>>>> Stashed changes
    return (
        is_cuda()
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and get_disagg().disaggregation_mode == "decode"
        and not get_memory().enable_hisparse
        and not get_parallel().dcp_enabled
    )


<<<<<<< Updated upstream
def should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend: bool) -> bool:
    """Select fused TopK for PD IndexShare.
=======
def should_use_dsa_fused_topk(
    server_args: "ServerArgs", seed_dsa_topk_from_draft_extend: bool
) -> bool:
    """为 PD IndexShare 选择融合 Top-K。
>>>>>>> Stashed changes

    PD Prefill worker:
    - Target prefill: fused TopK enabled.
    - Draft extend: fused TopK disabled.

    PD Decode worker:
    - Draft decode / target verify / draft extend: fused TopK enabled.
    """
    # 只有 PD 模式且 draft extend 提供 seed 时，才属于跨阶段 IndexShare seed。
    pd_index_share_seed = (
        get_disagg().disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    # 普通请求直接融合；跨 PD seed 只有可映射到本地槽位时才能融合。
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots()
    )


def is_dsa_enable_prefill_cp():
    """根据运行时拓扑判断 DSA Prefill CP 是否真正启用。

    旧 CP 路径直接读取显式开关；CP-v2 则以实际 CP size 和模型架构为准。
    这样可避免配置标志与新拓扑初始化结果不一致。
    """
    # 旧实现沿用预先计算的 DSA Prefill CP 开关。
    if not envs.SGLANG_ENABLE_CP_V2.get():
        return get_parallel().enable_dsa_prefill_context_parallel

    # Derive from the runtime CP topology + model arch rather than the legacy
    # flag under CP-v2: DSA prefill CP is active when the CP group is on for a
    # DeepSeek Sparse Attention model.
    # 单 rank 不存在上下文并行，无需继续检查模型类型。
    if get_parallel().attn_cp_size <= 1:
        return False
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    # CP-v2 仅在支持稀疏注意力的 DeepSeek DSA/V4 架构上打开此路径。
    hf_config = process_model_config().hf_config
    return is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)


def is_dsa_prefill_cp_in_seq_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_parallel().dsa_prefill_cp_mode == "in-seq-split"
    )


def is_dsa_prefill_cp_round_robin_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_parallel().dsa_prefill_cp_mode == "round-robin-split"
    )


# Structural surface where the graph DSA split-op dispatch (DSA indexer) and the
# MLA BMM-into-attention fusion apply: a non-speculative extend (prefill) running
# inside a piecewise/breakable CUDA graph. Both fusions are now on by default on
# this surface (no feature flag); each adds its own extra carve-outs at its call
# site (e.g. the indexer also excludes DSA prefill context parallelism).
def is_graph_dsa_split_op_surface(forward_batch: "ForwardBatch") -> bool:
    """判断当前 forward 是否处在可使用 DSA 图内拆分算子的执行面。

    该优化只覆盖 CUDA 上的 piecewise/breakable graph，且必须是非推测
    extend（普通 prefill）。调用点还会继续叠加 CP 等局部限制。
    """
    return (
        is_cuda()
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    """检查当前 prefill batch 能否进行 DSA CP 轮询切分。

    除配置模式外，还要求 forward 已进入 CP extend，token 数不少于 CP rank
    数且 CP size 大于 1，保证每次切分具有实际计算价值。
    """
    # 非 context-parallel extend 不允许在这里改变 token 排列。
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    # 从已初始化的 attention 并行拓扑读取 CP rank 数。
    cp_size = get_parallel().attn_cp_size
    # CPU 长度列表已由 batch 构造阶段准备，可避免 GPU 同步。
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    # 每个 CP rank 取得全局 token 流中下标满足 idx % cp_size == cp_rank 的元素。
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # Python 容器用于 CPU 侧长度/索引元数据，直接按步长切片。
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    # Tensor 路径需要同时处理 token 数无法被 CP size 整除的尾部。
    tokens = len(input_)
    if tokens % cp_size != 0:
        # 排名靠前的 rank 至多比后续 rank 多取得一个尾部 token。
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        # 当前 rank 没有 token 时返回形状正确的空 Tensor。
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        # arange 直接生成当前 rank 的轮询下标，避免构造完整 mask。
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # 可整除时 reshape 后选列，通常比高级索引更便于生成高效 kernel。
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    """复现 DP/CP batch 准备阶段的 token 补齐规则。

    DSA 元数据长度必须和实际送入 attention 的 q 张量物理长度完全一致。
    本函数镜像 ``prepare_mlp_sync_batch`` 的选择逻辑，并在轮询 CP 模式下
    进一步换算为当前 rank 的局部 token 数。
    """
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.layers.cp.utils import enable_cp_v2, is_cp_v2_active

    # CP-v2 already pads each rank-local shard to its physical size
    # CP-v2 已在元数据中记录各 rank 的物理长度，直接复用可避免二次补齐。
    if is_cp_v2_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]

    # 复制列表，后续对齐不能污染 ForwardBatch 保存的原始长度。
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_parallel().attn_cp_size
    # Must mirror ForwardBatch.prepare_mlp_sync_batch, which applies cp_align_size only when
    # CP-v2 is disabled. Under enable_cp_v2() the speculative forwards (TARGET_VERIFY /
    # DRAFT_EXTEND_V2) reach here with is_cp_v2_active False, and q is padded to attn_tp_size only
    # (not cp-aligned). Applying cp_align here over-pads the flashmla metadata past q, so
    # num_splits ends up longer than q -> fwd_kvcache_mla fails "num_splits must have shape (b+1)".
    # (attn_cp analog of the attn_tp fix in PR #30642 / issue #30296.)
    # 仅旧 CP 路径需要在此按 CP kernel 的 tile 大小补齐。
    if not enable_cp_v2():
        cp_align_size = get_cp_padding_align_size()
        for i in range(sync_group_size):
            global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
    # Reuse the mode selected when the DP buffer was prepared.
    # 优先复用 DP buffer 构建时已确定的 padding 模式，保持两端一致。
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    # MAX_LEN 模式让所有 DP rank 对齐到本组最大 token 数。
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    # 非 MAX_LEN 的多 DP 场景采用本 attention DP rank 对应的长度。
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_parallel().attn_dp_rank]
    # 单元素列表表示没有跨 DP rank 的长度差异。
    else:
        tokens = global_num_tokens[0]
    # 轮询 CP 会把全局补齐后的 token 再平均分给各 CP rank。
    if can_dsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    """把 DSA cache 长度张量补到 attention 实际消费的物理 token 数。

    仅 DP padding 或 CP 轮询切分存在时需要补齐；新增行全部置零，表示
    padding token 没有可访问的历史 cache，避免 Top-K 读到无效页。
    """
    # CP 是否会改变当前 rank 的物理 token 数。
    attn_cp_size = get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(
        forward_batch
    )
    # global_num_tokens_cpu 存在说明 DP batch 已进入统一 buffer 补齐流程。
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return dsa_cache_seqlens
    # 计算与 q 张量相同的目标物理长度。
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - dsa_cache_seqlens.shape[0]
    # 只允许向后补零，不在这里截断上游已经构造的数据。
    if pad_len > 0:
        dsa_cache_seqlens = torch.cat(
            [
                dsa_cache_seqlens,
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),
            ]
        )
    return dsa_cache_seqlens


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    """判断一条序列是否满足 DSA Prefill CP 切分条件。

    该检查同时约束 CP 拓扑、forward mode、DSA 开关和最小 token 数。
    返回 True 只表示当前 rank 能获得非空分片，具体布局由所选 CP 模式决定。
    """
    # 任一基础条件不满足都必须保留原始序列布局。
    if (
        cp_size <= 1
        or not use_dsa
        or not forward_batch.forward_mode.is_context_parallel_extend()
        or not is_dsa_enable_prefill_cp()
        or sum(forward_batch.extend_seq_lens_cpu) < cp_size
    ):
        return False

    # 轮询模式要求上游先补齐到 CP size 的整数倍。
    if is_dsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when dsa_prefill_cp_mode is round-robin-split"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
    return cur_cp_seq_len != 0


from sglang.kernels.ops.attention.dsa.cp_split import (
    dsa_cp_round_robin_split_q_seqs_kernel,
)


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    """在 CPU 侧计算当前 CP rank 的逐请求 query 长度及非空请求下标。

    ``extra_seq`` 把前一请求留下的轮询相位延续到后一请求，保持整个
    扁平 token 流的 ``token_idx % cp_size`` 分配规则不被请求边界打断。
    """
    # 读取当前 attention CP 拓扑。
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # 记录跨请求延续的轮询余量。
    extra_seq = 0
    q_seqs = []
    # 逐请求计算当前 rank 实际接收的 token 数。
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    # kernel 只处理当前 rank 上非空的请求。
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    # CPU 结果用于控制流和 Python 列表，GPU 结果供后续 kernel 直接消费。
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    # 先确定输出大小，避免设备端动态分配或回读。
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    # 为非空请求的局部长度分配设备输出。
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    # 同时生成原 batch 中的请求下标，供后续 gather 使用。
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    # 该 kernel 内部遍历小规模 batch 元数据，使用单 program 即可。
    grid = (1,)
    dsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    """判断当前 batch 是否应进入 DSA Prefill CP 执行路径。

    除全局能力开关外，batch 必须已经带有 CP 元数据且 forward mode 为
    context-parallel extend；三者缺一时都应使用普通 Indexer 路径。
    """
    # 未显式传入时，根据当前拓扑和模型配置动态判断。
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def fp8_mqa_logits_ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """将正尺度向上取整为 UE8M0 可表达的 2 的整数次幂。"""
    # abs 兼容意外的负输入，ceil 保证量化尺度不会低估动态范围。
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def fp8_mqa_logits_make_fused_kv(
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    block_kv: int,
    head_dim: int,
) -> torch.Tensor:
    """把 FP8 K 数据和逐 token FP32 scale 打包为分页 MQA 融合布局。

    每个物理 block 先连续存放 ``block_kv * head_dim`` 字节的 FP8 数据，
    再存放每个 token 的 4 字节 scale。最后 reshape 成 kernel 约定的
    ``[num_blocks, block_kv, 1, head_dim + 4]`` 逻辑视图。
    """
    # 物理页数量决定融合 buffer 的第一维。
    num_phys_blocks = kv_fp8.shape[0]
    # 每个 token 的逻辑负载为 head_dim 字节 FP8 加 4 字节 FP32 scale。
    per_token_size = head_dim + 4
    # 单个物理页在融合格式中占用的总字节数。
    block_bytes = block_kv * per_token_size
    # scale 区紧跟在该页全部 FP8 K 数据之后。
    scale_offset = block_kv * head_dim

    # 以 uint8 分配可精确控制不同 dtype 数据的字节拼接位置。
    fused = torch.zeros(
        num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    # 逐物理页复制 K 字节和 scale 字节，保持页表索引不变。
    for blk in range(num_phys_blocks):
        # FP8 Tensor 仅改变视图为字节，不做数值转换。
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        # scale 先规范为连续 FP32，再按原始 4 字节表示写入尾部。
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    # 返回分页 MQA kernel 期望的四维逻辑布局。
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)
