# GLM-5.2 Prefill Context Parallel（CP）详解

> 本文基于当前检出的 SGLang 源码，面向 GLM-5.2 的 GlmMoeDsaForCausalLM 架构和单机 H200/Hopper 环境。SGLang、镜像和其依赖更新很快，部署前应以目标镜像的 sglang serve --help 与最小启动结果为准。

## 1. 结论与正确拓扑

以下三项共同开启的是 **仅针对 prefill/extend 阶段** 的 Context Parallelism：

~~~bash
--attn-cp-size 8 \
--enable-prefill-cp \
--cp-strategy interleave
~~~

它们不会把模型权重切成八份；权重切分属于 TP（tensor parallelism）。CP 切的是**同一条长输入的 token 维度**：8 个 GPU 先各自处理约 1/8 的 token 行，之后只在需要完整上下文、全局 KV 或完整 hidden states 的边界做 collective 通信。

GLM-5.2 DSA 的推荐单机 8 卡拓扑是：

~~~text
tp_size = 8
dp_size = 1
attn_cp_size = 8
attn_tp_size = tp_size / (dp_size × attn_cp_size) = 1
cp_strategy = interleave
~~~

对应启动片段：

~~~bash
sglang serve \
  --model-path /data/models/GLM-5.2-FP8 \
  --tp 8 \
  --dp 1 \
  --attn-cp-size 8 \
  --enable-prefill-cp \
  --cp-strategy interleave \
  --moe-a2a-backend deepep \
  ...
~~~

不能与之前的下列组合混用：

~~~bash
--tp 8 --dp 4 --attn-cp-size 8
~~~

原因有两层：

1. 通用检查要求 tp_size 能被 dp_size × attn_cp_size 整除。此处是 8 % (4 × 8) ≠ 0，参数解析阶段就会抛出 AssertionError。
2. 对 GLM-5.2/DSA 的 interleave prefill CP，模型专用覆盖逻辑进一步要求 dp_size == 1。

这并不矛盾于源码在内部将 enable_dp_attention 设为 True：它用于建立 attention/MLP 的通信语义；但 interleave 的**数据并行大小**仍必须是 DP=1，不能因此把 --dp 4 变为合法。

**对应代码：参数整除校验**
文件：`python/sglang/srt/server_args.py`，`ServerArgs._handle_context_parallelism()`。

~~~python
if view.attn_cp_size > 1:
    assert self.tp_size % view.attn_cp_size == 0, \
        "tp_size must be divisible by attn_cp_size"
    assert self.tp_size % (self.dp_size * view.attn_cp_size) == 0, \
        "tp_size must be divisible by dp_size * attn_cp_size"
~~~

这正是 `TP=8, DP=4, CP=8` 失败的直接原因；条件不成立时，模型权重加载前就会退出。

## 2. 三个参数逐项解释

| 参数 | 默认值 | 直接作用 | GLM-5.2 上的含义 |
|---|---:|---|---|
| --attn-cp-size 8 | 1 | 将 attention context parallel group 的大小设为 8。 | 8 个 rank 分担同一段 prompt 的 token 行；影响 attn_cp_rank、attn_cp_group、attn_tp_size 与 MoE 通信组。 |
| --enable-prefill-cp | False | 开启 prefill CP 总开关。 | 使长 prefill/extend batch 构造 CP metadata、切分 embedding/position，并走 DSA CP 路径。 |
| --cp-strategy interleave | None | 选择 token 的布局。 | round-robin 切分；token i 归属 i mod 8 对应的 CP rank。当前 DSA CP-v2 只支持这一策略。 |

三者需作为整体理解：

- 只设置 --attn-cp-size 8：只定义 attention CP 拓扑；不等价于实际把 prefill token 以 interleave 方式切开。
- 设置 --enable-prefill-cp 但不设置 --cp-strategy：会被校验拒绝。
- 设置 --cp-strategy 但未启用 --enable-prefill-cp：策略对象不会用于 prefill。

**对应代码：CLI 字段定义**
文件：`python/sglang/srt/server_args.py`。

~~~python
attn_cp_size: A[int, Arg(
    help="The attention context parallelism size.",
    aliases=["--attention-context-parallel-size"],
)] = 1

enable_prefill_cp: A[bool,
    "Enable context parallelism for the prefill phase. "
    "Select the layout with --cp-strategy."] = False

cp_strategy: A[Optional[str], Arg(
    choices=("zigzag", "interleave"),
)] = None
~~~

### 2.1 --attn-cp-size：重新分解 attention 并行空间

SGLang 在这条路径中将 attention 的并行空间概念化为：

~~~text
tp_size = attn_dp_size × attn_cp_size × attn_tp_size
~~~

推荐组合 tp=8、dp=1、attn-cp=8 得到：

~~~text
8 = 1 × 8 × 1
~~~

也就是说，8 张卡组成一个 attention CP group；每一个 attention TP group 的大小为 1。

~~~text
物理 GPU/rank:   [0] [1] [2] [3] [4] [5] [6] [7]
TP world:         └────────── TP = 8 ─────────────┘
Attention CP:     └────────── CP = 8 ─────────────┘
Attention TP:      [0] [1] [2] [3] [4] [5] [6] [7]  （每组大小 = 1）
Data parallel:      DP = 1
~~~

attn_tp_size=1 是当前实现的必要限制，而不是漏配 TP。DSACPLayerCommunicator 不会在每层 attention 的 o_proj 之后，为仍需要复制布局的 dense FFN 补齐 attention-TP 的 partial output all-reduce；因此 GLM-5.2 DSA CP 强制将 attention TP 收敛为 1，避免把不完整的 attention 输出交给后续层。

**对应代码：GLM-5.2 DSA 的专用覆盖**
文件：`python/sglang/srt/arg_groups/overrides.py`，`_deepseek_family_overrides()`。

~~~python
if server_args.enable_prefill_cp:
    overrides["enable_dp_attention"] = True
    overrides["moe_dense_tp_size"] = 1
    if server_args.cp_strategy == "zigzag":
        ...
    else:
        assert server_args.dp_size == 1, \
            "interleave DSA CP does not support DP attention."
    assert server_args.tp_size <= 8, \
        "Context parallel only supports single machine (tp_size <= 8)."

    # Keep attn_tp_size == 1 under DSA CP.
    attn_cp_size = server_args.tp_size // server_args.dp_size
    overrides["attn_cp_size"] = attn_cp_size
~~~

最后两行令 `attn_cp_size=tp_size/dp_size`；因此推荐拓扑里 `8/1=8`，而 attention TP 的余数为 1。

### 2.2 --enable-prefill-cp：只优化长输入，不让 decode 每步都做 CP

SGLang 对每一个 forward 检查它是否是 context_parallel_extend。只有 prefill/extend 才进入 CP；decode 通常一次仅新增一个 token，八卡通信的固定开销远大于收益。

它主要改善：

- 长系统提示词、长文档、长代码库等冷缓存请求的 TTFT/prefill latency；
- 长 prefill 的单卡 attention、MoE token batch 和中间激活压力；
- 超长上下文中单 GPU 处理所有 token 的尾延迟。

它不直接改善：

- decode 的逐 token 生成速度；
- EAGLE 的接受率；
- 权重读取、DeepGEMM 首次 JIT；
- 已命中 Radix Cache 的共同前缀。后者不需要重新 prefill，CP 收益自然会变小。

CP prefill 结束前，KV 必须被 materialize 成 decode 可读的 paged KV cache 布局。随后 decode 使用常规路径继续生成，而非在每一个新 token 上重新进行八卡 context parallel。

**对应代码：只有 CP prefill batch 才启用**
文件：`python/sglang/srt/layers/attention/dsa/utils.py`。

~~~python
def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    return (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    )
~~~

`is_context_parallel_extend()` 是关键门限：普通 decode forward 不满足它，因此不会走此 prefill CP 数据路径。

### 2.3 --cp-strategy interleave：按 token 下标轮转分配

interleave 是旧称 round-robin-split 的统一名称。它不按连续段切分，而是按全局 token 下标对 cp_size 取模。

以 cp_size=8、16 个 token 为例：

~~~text
原始顺序:   t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15

CP rank 0:  t0  t8
CP rank 1:  t1  t9
CP rank 2:  t2  t10
CP rank 3:  t3  t11
CP rank 4:  t4  t12
CP rank 5:  t5  t13
CP rank 6:  t6  t14
CP rank 7:  t7  t15
~~~

形式化地：

~~~text
rank r 持有 { token[i] | i mod cp_size = r }
~~~

若总 token 数为 T，每个 rank 的逻辑 token 数为 floor(T/8) 或 ceil(T/8)，差异最多一个，天然负载均衡。若 T 不能被 8 整除，CP-v2 把较短 rank 补零到相同的物理行数，再在输出时忽略 padding 行，以适配 all-gather/reduce-scatter 的固定形状。

与 zigzag 的区别：

| 策略 | token 布局 | 在 GLM-5.2 DSA CP-v2 中 |
|---|---|---|
| interleave | 按 token_index mod cp_size 轮转分发，负载最均匀。 | 推荐且当前支持。 |
| zigzag | 原 in-seq-split，交错地拥有连续首尾块，通常需要按 2 × cp_size 对齐。 | 其他模型/旧路径可能使用；当前 DSA CP-v2 不应选它。 |

**对应代码：interleave 切分**
文件：`python/sglang/srt/layers/cp/interleave.py`，`InterleaveCPStrategy._interleave_shard()`。

~~~python
def _interleave_shard(self, input_):
    cp_size = self.cp_size
    cp_rank = self.cp_rank
    tokens = len(input_)
    if tokens % cp_size != 0:
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()
~~~

`torch.arange(cp_rank, tokens, cp_size)` 就是“取 `cp_rank, cp_rank+8, cp_rank+16, ...`”的代码实现。整除时的 `view(... )[:, cp_rank]` 是等价但更高效的张量视图路径。

**对应代码：各 rank 长度不同的 padding**
文件：`python/sglang/srt/layers/cp/padding.py`。

~~~python
logical_tokens = list(metadata.per_rank_actual_token)
physical_rank_len = (
    (max(logical_tokens) + align_size - 1) // align_size * align_size
)
metadata.per_rank_logical_token = logical_tokens
metadata.per_rank_actual_token = [physical_rank_len] * len(logical_tokens)
~~~

`per_rank_logical_token` 是真实 token 数；`per_rank_actual_token` 是为了 collective 对齐后的物理行数。padding 行为零，最终结果不会被当成真实输入 token。

## 3. GLM-5.2 为什么会从 CP 受益：DSA + MLA + MoE

当前仓库把 GLM-5.2 映射为 GlmMoeDsaForCausalLM，该类复用 DeepseekV2ForCausalLM 的主体实现。它不是普通的密集 MHA + 单 FFN 模型：

1. **DSA（DeepSeek Sparse Attention）**：indexer 为每个 query 找相关 KV，再执行稀疏 attention。长上下文虽然避开了完全密集的 attention 成本，但 indexer、Q/KV 准备、metadata 与 KV 访问在 prefill 仍很重。
2. **MLA（Multi-head Latent Attention）**：KV 以 latent/nope/rope 组织，节省 KV 空间但增加了 KV 组织和 materialization 的复杂性。
3. **MoE FFN**：router 给每个 token 选择专家，DeepEP 把 token all-to-all 分发到专家 owner、执行专家 FFN、再组合返回。长 prefill 的 token 数会放大该成本。
4. **FP8/Hopper**：FP8 能加速 GEMM，但不会改变“所有长 prompt token 都挤在单卡”的 token 维度瓶颈；CP 正是在此维度加并行度。

CP 的原则是：先把 token 流切开，使 token 级的计算在 8 张卡分摊；在数学上需要完整 token 流、完整 KV 或 expert token 交换的位置再通信。总 FLOPs 不会消失，但关键路径的每卡 token 行数下降。

~~~mermaid
flowchart TD
    A[完整 prefill token 序列] --> B[Interleave: token 下标 mod 8 分给 8 个 CP rank]
    B --> C[本地 embedding、Norm、QKV 和 DSA indexer]
    C --> D[DSA/MLA Attention：按需取得全局 KV/索引]
    D --> E[CP communicator：按层 all-gather 或 reduce-scatter]
    E --> F[MoE：DeepEP dispatch、专家 FFN、combine]
    F --> G[下一 Transformer 层]
    G --> H[恢复全局 token 顺序，materialize KV]
    H --> I[普通 decode 使用 paged KV Cache]
~~~

## 4. 从 CLI 到代码执行：整体架构

### 4.1 参数与合法性

入口在 python/sglang/srt/server_args.py：

- attn_cp_size：默认 1；
- enable_prefill_cp：prefill CP 总开关；
- cp_strategy：可选 zigzag 或 interleave。

ServerArgs._handle_context_parallelism() 的关键检查等价于：

~~~text
enable_prefill_cp => cp_strategy 不能为 None
attn_cp_size > 1 => tp_size % attn_cp_size == 0
                    tp_size % (dp_size × attn_cp_size) == 0
~~~

GLM-5.2 被列入 CP-v2 默认模型集合。对 DSA 模型，只有 enable_prefill_cp 与 cp_strategy=interleave 同时成立时，源码才自动打开 CP-v2。

**对应代码：GLM-5.2 选择 CP-v2**
文件：`python/sglang/srt/server_args.py`，`_handle_context_parallelism()`；模型集合定义在 `python/sglang/srt/layers/cp/utils.py`。

~~~python
# cp/utils.py
CP_V2_DEFAULT_MODEL_CLASSES = frozenset({
    "DeepseekV32ForCausalLM",
    "GlmMoeDsaForCausalLM",
    ...
})

# server_args.py
is_dsa_default_model = is_deepseek_dsa(hf_config)
enable_default_cp_v2 = not is_dsa_default_model or (
    self.enable_prefill_cp and self.cp_strategy == "interleave"
)
if enable_default_cp_v2 and not envs.SGLANG_ENABLE_CP_V2.is_set():
    envs.SGLANG_ENABLE_CP_V2.set(True)
~~~

这说明 GLM-5.2 并不是任意 CP strategy 都可用：DSA 默认模型只有显式选 `interleave` 才进入当前 CP-v2 实现。

### 4.2 DSA 专用覆盖与限制

python/sglang/srt/arg_groups/overrides.py 中 DeepSeek/DSA 分支为 GLM-5.2 加上了模型专用规则：

~~~text
当前仅在 Hopper 验证过，仍属实验性功能
interleave DSA CP 要求 dp_size == 1
tp_size <= 8，当前只支持单机；跨机存在精度问题
attn_cp_size = tp_size / dp_size
moe_dense_tp_size = 1
enable_dp_attention = True（内部通信语义）
prefill CUDA Graph backend = disabled
~~~

因此，对单机 8×H200，最终是在 TP world 内建立 8-rank CP group；它不是 TP8、DP4、CP8 这三个维度相乘后的 256-way 并行。

**对应代码：禁用 prefill CUDA Graph**
文件：`python/sglang/srt/server_args.py`。

~~~python
if self.enable_prefill_cp:
    self.cuda_graph_config.prefill.backend = Backend.DISABLED
~~~

这是该文所说的“CP 可能失去 prefill CUDA Graph 固定调度收益”的直接来源；它是当前实现限制，不应理解为 CUDA Graph 永远无法与 CP 共存。

### 4.3 通信组

python/sglang/srt/distributed/parallel_state.py 创建：

- _TP：常规 tensor parallel 组；
- _ATTN_CP：相同 attention-TP lane 上的 CP ranks；
- _ATTN_TP：相同 CP/DP cell 内的 attention TP ranks；
- _MOE_DP：当 attn_cp_size 大于 moe_dp_size 时，直接指向 _ATTN_CP，使 MoE 能跨 CP partner 共享/交换 token。

**对应代码：计算 attention TP 并建立 CP group**
文件：`python/sglang/srt/distributed/parallel_state.py`，`initialize_model_parallel()`。

~~~python
attn_dp_size = attention_data_parallel_size
attn_cp_size = attention_context_model_parallel_size
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

for dp_idx in range(attn_dp_size):
    for attn_tp_idx in range(attn_tp_size):
        st = dp_idx * attn_tp_size * attn_cp_size + attn_tp_idx
        en = (dp_idx + 1) * attn_tp_size * attn_cp_size + attn_tp_idx
        ranks = list(range(st, en, attn_tp_size))
        group_ranks.append(ranks)
_ATTN_CP = init_model_parallel_group(group_ranks, ..., group_name="attn_cp")
~~~

在 `TP=8, DP=1, CP=8, attention-TP=1` 时，上面 `range(0, 8, 1)` 产生 `[0,1,2,3,4,5,6,7]`，恰好是单个 8-rank CP group。

**对应代码：CP partner 复用到 MoE DP group**
文件：`python/sglang/srt/distributed/parallel_state.py`。

~~~python
if attn_cp_size > moe_dp_size:
    # CP ranks must share tokens before MoE.
    _MOE_DP = _ATTN_CP
~~~

### 4.4 CP-v2 的 batch 处理

核心实现分布在：

- python/sglang/srt/layers/cp/base.py：策略抽象、枚举、策略单例；
- python/sglang/srt/layers/cp/interleave.py：round-robin shard、padding、all-gather、全局顺序恢复；
- python/sglang/srt/layers/cp/utils.py：CP-v2 激活判断、metadata、输入 shard、输出 gather；
- python/sglang/srt/model_executor/runner/eager_runner.py：实际调用顺序。

一次 active prefill batch 的主路径为：

1. eager runner 调用 prepare_cp_forward()；
2. InterleaveCPStrategy.build_metadata() 记录总 token、每 rank 的逻辑/物理 token 数；
3. cp_shard_model_inputs() 将 embedding hidden states 和 position ids 按 interleave 切到本 rank；
4. Transformer 层运行，attention/communicator/MoE 根据布局完成通信；
5. cp_gather_after_forward() all-gather 并重排末端 hidden states 为原始 token 顺序，供 logits 等模型边界逻辑使用。

interleave 恢复全局行序的索引等价于：

~~~text
全局 token i 的数据在：
  source CP rank = i % cp_size
  source rank 内行号 = i // cp_size
~~~

所以 gather 后得到的仍是 t0,t1,t2,...，而不是“rank0 所有 token 再接 rank1 所有 token”。

**对应代码：runner 的 CP-v2 边界**
文件：`python/sglang/srt/model_executor/runner/eager_runner.py`。

~~~python
cp_v2_active = is_cp_v2_active(forward_batch)
if cp_v2_active:
    prepare_cp_forward(forward_batch)

with cp_shard_model_inputs(
    input_embeds, forward_batch.positions, forward_batch
) as (sharded_input_embeds, sharded_positions):
    model_kwargs = {"input_embeds": sharded_input_embeds}
    hidden_states = model.model(
        forward_batch.input_ids,
        sharded_positions,
        forward_batch,
        **model_kwargs,
    )

hidden_states = cp_gather_after_forward(hidden_states, forward_batch, stream)
~~~

这里的顺序很重要：输入在模型边界切分，模型内部执行本 rank 的 token 行，模型输出在 logits 前重新拼回全局顺序。

**对应代码：metadata 与 CP 策略对象**
文件：`python/sglang/srt/layers/cp/utils.py`、`python/sglang/srt/layers/cp/base.py`。

~~~python
# cp/utils.py
if forward_batch.attn_cp_metadata is None:
    forward_batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens=num_tokens,
        seqs_len=seq_lens_cpu,
        extend_seqs_len=extend_lens_cpu,
    )
    pad_logical_token_to_physical(forward_batch.attn_cp_metadata)

# cp/base.py
if kind == ContextParallelStrategyKind.INTERLEAVE:
    _STRATEGY = InterleaveCPStrategy(cp_size=cp_size)
~~~

metadata 保留本次 forward 的实际长度、padding 和 CP rank 布局；它附着在 `ForwardBatch`，因此同一服务内 CP 是否真正使用是逐 batch 决定的，而不是每个请求永久固定。

## 5. Attention 的具体机制与影响

### 5.1 输入、RoPE 与 QKV

进入模型前，每个 rank 只保留自身的 interleave token 行。位置 id 不能重新从零编号：例如 rank 3 持有的是 t3,t11,t19,...，它必须保留这些 token 的全局 position。CP 因此同时切分真实的 position ids，DSA/MLA 的 RoPE 仍按全局位置旋转，数学语义与未切分时一致。

**对应代码：hidden states 与 position ids 使用同一策略切分**
文件：`python/sglang/srt/layers/cp/interleave.py`。

~~~python
def shard_hidden_states(self, x, forward_batch):
    metadata = forward_batch.attn_cp_metadata
    local_x = self._interleave_shard(x[:metadata.total_seq_lens])
    return pad_local_rows(local_x, metadata, dim=0)

def shard_position_ids(self, positions, forward_batch):
    metadata = forward_batch.attn_cp_metadata
    local_positions = self._interleave_shard(
        positions[:metadata.total_seq_lens]
    )
    return pad_local_rows(local_positions, metadata, dim=0)
~~~

这两段同用 `_interleave_shard()`，保证 `hidden_states[j]` 与其对应 `position_ids[j]` 始终属于同一个全局 token；CP 不会破坏 RoPE 的位置信息。

### 5.2 DSA sparse attention 与 KV

GLM-5.2 的 DSA backend 位于 python/sglang/srt/layers/attention/dsa_backend.py。InterleaveCPStrategy 只声明支持 DSA backend；其通用 run_attention() 是 no-op，因为实际 sparse attention 由 DSA backend 自己调度。

对每个 local query token，DSA 仍需要按模型规则选择/访问全局历史中的相关 KV。CP 不会让 GPU 只“看见”自己的 token，切分的是计算责任和中间布局。关键处理包括：

- 依据 local query 的真实位置生成 DSA indexer/attention metadata；
- 对要写入 decode cache 的 DSA MLA KV，合并 local k_nope 与 k_rope，并在 CP group all-gather；
- FP8 KV 按原始字节打包再 gather，避免 dtype 转换破坏数据；
- gather 后恢复全局 token 顺序，写入 paged KV cache；
- 随后的 decode 读取已经 materialize 的 KV，不在每一步重复 prefill CP。

**对应代码：DSA/MLA FP8 KV 的 gather 后写入 cache**
文件：`python/sglang/srt/layers/attention/dsa_backend.py`，TRT-LLM MLA forward 路径。

~~~python
rope_positions = forward_batch.positions
if dsa_use_prefill_cp(forward_batch):
    rope_positions = get_cp_strategy().shard_position_ids(
        rope_positions, forward_batch
    )

q, k, k_rope = mla_quantize_and_rope_for_fp8(
    q, q_rope, k.squeeze(1), k_rope.squeeze(1), rope_positions, ...
)
if save_kv_cache and dsa_use_prefill_cp(forward_batch):
    k, k_rope = get_cp_strategy().all_gather_dsa_trtllm_fp8_kv(
        forward_batch, k, k_rope
    )

if save_kv_cache:
    self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
~~~

上段先对 local query 使用正确的全局 RoPE position；在写入 paged KV pool 前再 gather。因此 CP 中间态可以分片，而 decoder 需要的 KV cache 仍以完整请求顺序可查询。

**对应代码：FP8 KV 为什么按 byte gather**
文件：`python/sglang/srt/layers/cp/interleave.py`，`all_gather_dsa_trtllm_fp8_kv()`。

~~~python
kv = torch.cat((k, k_rope), dim=-1).view(torch.uint8)
kv = self.gather_kv_cache(
    kv.contiguous(), forward_batch, torch.cuda.current_stream()
).view(kv_dtype)
return kv.split((kv_lora_rank, qk_rope_head_dim), dim=-1)
~~~

先 view 为 `uint8` 是为了让 collective 搬运 FP8 的原始字节布局；gather 后再恢复原 dtype 并拆回 `k`、`k_rope`。

### 5.3 CUDA Graph 的取舍

DSA 开启 prefill CP 后，当前参数处理显式将 prefill CUDA Graph backend 设为 disabled。原因是 CP 的 rank-local token 数、padding、动态 metadata 和 collective 使静态 capture/replay 更难稳定支持。

这意味着 CP 以 8 卡 token 并行换取长序列规模收益，但会失去 prefill CUDA Graph 的部分固定开销收益。短 prompt、热缓存或很小 batch 下，通信和失去 CUDA Graph 的代价可能超过 CP 收益；应测端到端 TTFT，而非仅看单个 attention kernel。

## 6. FFN、MoE 与 CP 通信

### 6.1 DSACPLayerCommunicator

GLM-5.2 复用的 Transformer block 在 DSA prefill CP 活跃时，使用：

~~~text
python/sglang/srt/layers/communicator_dsa_cp.py
DSACPLayerCommunicator
~~~

而不是普通 LayerCommunicator。其工作是协调每层 attention 与 MLP 边界的 hidden_states/residual 布局：

~~~text
local token shard
  -> prepare_attn（Norm、attention 输入准备）
  -> DSA/MLA attention
  -> prepare_mlp（按需 gather + Norm）
  -> dense FFN 或 MoE
  -> postprocess / 保持下一层所需的 shard 布局
~~~

它依据 LayerScatterModes 选择：

- **SCATTERED**：每个 CP rank 只保留本地 token 分片；
- **FULL**：参与 rank 获得聚合后的完整 token 流；
- 其他与 attention TP 兼容的布局。

当消费者需要完整 token 流时，dsa_cp_gather_hidden_states() 在 attn_cp_group 做 all-gather；当需要从完整流回到本地分片时，dsa_cp_reduce_scatter_hidden_states() 做 reduce-scatter。这两个函数都断言 attn_dp_size==1 且 attn_tp_size==1，正是推荐拓扑的直接代码依据。

**对应代码：attention/MLP 边界的 CP collective**
文件：`python/sglang/srt/layers/communicator_dsa_cp.py`。

~~~python
def dsa_cp_gather_hidden_states(hidden_states):
    assert get_parallel().attn_dp_size == 1
    assert get_parallel().attn_tp_size == 1
    output, local = get_local_dp_buffer(get_parallel().attn_cp_group), hidden_states
    attn_cp_all_gather_into_tensor(output, local)
    return output

def dsa_cp_reduce_scatter_hidden_states(hidden_states):
    cp_size, cp_rank = get_parallel().attn_cp_size, get_parallel().attn_cp_rank
    output = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(output, hidden_states)
    return output
~~~

实现中的 all-gather 与 reduce-scatter 并非随意插入：它们是 `DSACPLayerCommunicator` 依据下一段需要 `FULL` 还是 `SCATTERED` token layout 所调用的布局转换。

**对应代码：什么时候 gather、什么时候 scatter**
文件：`python/sglang/srt/layers/communicator_dsa_cp.py`。

~~~python
if hidden_states_output_mode == ScatterMode.FULL:
    return _gather_hidden_states_and_residual

def _gather_hidden_states_and_residual(...):
    hidden_states, residual = layernorm(hidden_states, residual)
    if dsa_use_prefill_cp(forward_batch):
        hidden_states = dsa_cp_gather_hidden_states(hidden_states)
    return hidden_states, residual

if hidden_states_input_mode == ScatterMode.FULL \
        and output_mode == ScatterMode.SCATTERED:
    return _scatter_hidden_states
~~~

因此文中所谓“Attention 后可能 gather、下一层前可能 scatter”是由 layer data layout 驱动，而非对每一层无条件复制完整 hidden states。

### 6.2 Dense FFN

一个 token 的 dense FFN 仍是独立函数：

~~~text
FFN(x_i) = W2 × activation(W1 × x_i)
~~~

CP 不会把同一个 token 的 FFN 激活拆到八张卡；它让每张卡首先面对更少 token 行。因此可以减少该卡的 token-batch GEMM 规模、激活占用和关键路径时间，但总计算量只是分摊而非消除。

当前 DSA CP 路径将 moe_dense_tp_size 设为 1，再由 communicator 在需要的位置做 gather 或 reduce-scatter。这避免 attention TP 的 partial o_proj 与 dense FFN 的复制数据布局混在一起。

### 6.3 MoE FFN 与 DeepEP

MoE 增加了 token-to-expert 的重排：

~~~text
local hidden states
  -> router / top-k experts
  -> DeepEP all-to-all dispatch
  -> 每个专家执行 FFN
  -> DeepEP combine 返回原 token 所在 rank
~~~

开启 --moe-a2a-backend deepep 时，LayerScatterModes 让 sparse MoE 的 token dispatch/combine 交给 MoE 后端，避免普通 communicator 再做重复 gather。当 attn_cp_size > moe_dp_size 时，源码令 MOE_DP 复用 ATTN_CP，让 CP partner 可参与 token 共享。

CP 不会免掉 MoE all-to-all；它带来的主要变化是每 rank 进入路由阶段的 token 数较均匀。最终专家 batch 仍由路由分布决定，专家倾斜、网络通信或 DeepEP 调度可能成为新的瓶颈。因此必须比较端到端 prefill，而不是只看 attention 时间。

**对应代码：Transformer block 把 communicator、attention、MLP 串起来**
文件：`python/sglang/srt/models/deepseek_v2.py`，`DeepseekV2DecoderLayer.forward()`。

~~~python
hidden_states, residual = self.layer_communicator.prepare_attn(...)
hidden_states = self.self_attn(
    positions=positions, hidden_states=hidden_states, forward_batch=forward_batch, ...
)
hidden_states, residual = self.layer_communicator.prepare_mlp(
    hidden_states, residual, forward_batch
)
hidden_states = self.mlp(hidden_states, forward_batch, ...)
~~~

这四步是 CP 对 attention 与 FFN/MoE 产生影响的真正连接点：communicator 在进入 attention 和 MLP 前后转换 token layout，而 attention 和 MLP 的数学主体保持原有实现。

**对应代码：MoE 选择 DeepEP 路径**
文件：`python/sglang/srt/models/deepseek_v2.py`，`DeepseekV2MoE.forward()`。

~~~python
if not self._enable_a2a_moe:
    return self.forward_normal(hidden_states, ...)
else:
    return self.forward_deepep(
        hidden_states, forward_batch, input_ids_global=input_ids_global
    )
~~~

`--moe-a2a-backend deepep` 会使 `_enable_a2a_moe` 为真，从而进入 `forward_deepep()`；这就是本文所说的 router 后 token dispatch/combine 由 DeepEP 完成的代码入口。

## 7. KV Cache、Radix/HiCache 与 layer split

### 7.1 与 KV Cache 的关系

interleave prefill 的本地 token shard 最终必须转成 decode 能消费的 paged KV cache。InterleaveCPStrategy 为 MLA/DSA 提供 KV gather 与 materialize：各 rank 的 KV 聚合、恢复全局 token 顺序后，再写入 token-to-KV pool。

结论：

- prefill 中间态是分片的；
- 普通 decode 读取的是可按全局请求位置查询的 KV；
- 不应把单体服务的 decode 理解为“每卡仅保存第 1/8 token KV”。

**对应代码：interleave 还原全局 token 顺序**
文件：`python/sglang/srt/layers/cp/interleave.py`，`_gather_interleaved_tensor()`。

~~~python
gathered = x.new_empty((self.cp_size * physical_rank_len, *x.shape[1:]))
attn_cp_all_gather_into_tensor(gathered, padded_x.contiguous())

flat_indices = torch.arange(total_tokens, device=x.device)
gather_indices = (
    flat_indices % self.cp_size
) * physical_rank_len + flat_indices // self.cp_size
return gathered.index_select(0, gather_indices)
~~~

all-gather 后的内存布局是“rank0 的局部行、rank1 的局部行……”。`gather_indices` 按 `rank=i%CP` 与 `local_row=i//CP` 把它重新排列成全局 `t0,t1,t2,...`，这也是 KV materialize 能与常规 decode 兼容的关键。

### 7.2 与 Radix Cache / HiCache 的边界

Radix Cache 已命中共同长前缀时，新请求通常只需计算短 suffix；这时 CP 的边际收益较低。CP 的重点是**冷缓存或未命中部分很长**的 prefill。

若还使用 HiCache/Mooncake，需分别观察：

- 未缓存 prompt 的 TTFT 与 prefill token/s；
- cache hit rate；
- H2D/D2H/远端存储 I/O 时延；
- DeepEP/NCCL 通信时间和专家负载；
- KV token usage 与显存。

不要把 HiCache I/O、请求排队或首次 kernel JIT 误判为 CP 的效果。

### 7.3 可选的 DSA cache layer split

--enable-dsa-cache-layer-split 是独立的内存优化，不是本文三项的默认搭配。它把 DSA GPU KV/indexer cache 的 **layer 维度**分给 CP ranks，每卡通常仅保有约 ceil(num_layers/cp_size) 个 layer，加一个远端读取 scratch layer。

其限制更严格：

- 仅 DSA 模型；
- 必须 --enable-prefill-cp --cp-strategy interleave；
- 仅 PD prefill worker；
- 当前需 Mooncake transfer backend；
- 不支持普通既 prefill 又 decode 的单体服务，也不支持 pp_size > 1。

它适合分离式 prefill 的超长上下文 KV 显存优化，不应在当前单体 sglang serve 命令中顺手添加。

**对应代码：layer split 开关及其有效条件**
文件：`python/sglang/srt/layers/cp/utils.py`。

~~~python
def is_glm_dsa_cache_layer_split_enabled(model_runner):
    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )

def get_glm_dsa_layer_split_effective_num_layers(model_runner, num_layers):
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return num_layers
    shard_size = get_parallel().attn_cp_size
    owned_layers_upper_bound = (num_layers + shard_size - 1) // shard_size
    return max(1, owned_layers_upper_bound + 1)  # +1: remote scratch layer
~~~

这里的 `+1` 是远端 layer 的临时 scratch，不代表该 rank 永久保存所有其他层的 KV；它解释了 layer split 为什么能降低每卡 DSA KV/indexer cache 占用、但不会精确到 `1/CP`。

## 8. 一次长 prompt 的概念时序

~~~mermaid
sequenceDiagram
    participant S as Scheduler
    participant R0 as CP rank 0
    participant R1 as CP ranks 1..7
    participant A as DSA/MLA Attention
    participant M as DeepEP MoE
    participant K as Paged KV Cache

    S->>R0: 逻辑 prefill batch / token 顺序 / positions
    S->>R1: 同一逻辑 batch 的元数据
    Note over R0,R1: interleave：rank r 取 token[i]，i mod 8 = r
    R0->>R0: local embedding、Norm、QKV/indexer
    R1->>R1: local embedding、Norm、QKV/indexer
    R0->>A: local query 与 DSA metadata
    R1->>A: local query 与 DSA metadata
    A-->>R0: local attention output
    A-->>R1: local attention output
    R0->>R1: 每层按 scatter mode 做 all-gather/reduce-scatter
    R0->>M: router + token dispatch
    R1->>M: router + token dispatch
    M-->>R0: expert result combine
    M-->>R1: expert result combine
    R0->>K: CP gather/reorder 后 materialize MLA KV
    R1->>K: 协同写入 KV
    R0->>R1: 末端 hidden states gather 并恢复全局顺序
    R0-->>S: logits / 首 token
~~~

## 9. 性能收益、代价和适用范围

### 适合尝试

- 32K、64K、128K 以上的冷缓存或低命中率长 prompt；
- 单机 8×H100/H200，NVLink/NVSwitch 互联良好；
- prefill 是端到端时延主要部分；
- 可以接受 DP=1，目标是降低少量超长请求的 TTFT；
- DeepEP/NCCL 通信健康、专家负载不过度倾斜。

### 可能不增反降

- 输入短，或绝大多数 token 命中 Radix Cache；
- 主要瓶颈是 decode、模型启动 JIT、HiCache I/O 或排队；
- 业务目标是大量独立请求吞吐，必须保持 --dp 4；
- 多机 CP：当前 GLM DSA 路径限制 tp_size<=8，并提示跨机精度问题；
- 强依赖 prefill CUDA Graph；
- MoE 专家负载高度偏斜，straggler 和 A2A 通信掩盖了 attention 的分摊收益。

### 正确的 A/B 方法

保持模型、权重、chunked-prefill-size、缓存状态、EAGLE 参数和请求集不变：

| 组别 | 配置 | 比较目标 |
|---|---|---|
| A：基线 | --tp 8 --dp 1，不开 prefill CP | TTFT、prefill tok/s、显存、输出一致性。 |
| B：CP | --tp 8 --dp 1 --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave | 同上，并记录 NCCL/DeepEP 时间。 |
| C：吞吐基线 | 原 --dp 4 方案，不使用本 CP | 多独立请求总吞吐和排队；不要与 B 的单长请求 TTFT 直接等价比较。 |

至少测 4K、32K、64K、128K 四档，并分别做冷缓存与热缓存。若只测热缓存，大部分结果是 KV cache 命中效率，而非 CP prefill 效果。

## 10. 代码文件地图

| 文件 | 主要职责 |
|---|---|
| python/sglang/srt/server_args.py | CLI 参数声明、CP-v2 自动开启、tp/dp/attn_cp 整除检查和 DSA CP 的通用限制。 |
| python/sglang/srt/arg_groups/overrides.py | GLM-5.2/DSA 专用覆盖：Hopper 实验性限制、interleave 要求 dp=1、attn_cp=tp/dp、moe_dense_tp_size=1。 |
| python/sglang/srt/distributed/parallel_state.py | 创建 TP、ATTN_CP、ATTN_TP、MOE_DP 通信组。 |
| python/sglang/srt/layers/cp/base.py | CP strategy 抽象、interleave/zigzag 枚举与策略单例。 |
| python/sglang/srt/layers/cp/interleave.py | round-robin token shard、padding、CP all-gather、原全局顺序恢复、DSA MLA FP8 KV gather。 |
| python/sglang/srt/layers/cp/padding.py | 将每 rank 的逻辑 token 数对齐为 collective 需要的物理大小。 |
| python/sglang/srt/layers/cp/utils.py | CP-v2 active 判断、forward metadata、输入 shard、模型输出 gather；也含 GLM DSA 可选 cache layer split。 |
| python/sglang/srt/model_executor/runner/eager_runner.py | 执行 prepare_cp_forward、cp_shard_model_inputs、cp_gather_after_forward。 |
| python/sglang/srt/layers/attention/dsa/utils.py | 当前 batch 是否 DSA CP、round-robin 辅助和 DSA sequence length 处理。 |
| python/sglang/srt/layers/attention/dsa_backend.py | DSA indexer、稀疏 MLA attention、RoPE、KV gather/materialize。 |
| python/sglang/srt/models/glm4_moe.py | 声明 GlmMoeDsaForCausalLM。 |
| python/sglang/srt/models/deepseek_v2.py | GLM-5.2 复用的 Transformer block、DSA attention、dense/MoE MLP 与 CP communicator 接入。 |
| python/sglang/srt/layers/communicator_dsa_cp.py | DSACPLayerCommunicator：每层 attention/MLP 边界上的 gather/reduce-scatter 协调。 |
| python/sglang/srt/layers/dp_attention.py | CP/TP collective 封装和 local DP buffer。 |
| python/sglang/srt/layers/moe/mega_moe.py | MoE 路径及 CP 下的 token 计数/通信协作。 |

## 11. 部署检查清单

1. 确认模型 config 中的 architecture 为 GlmMoeDsaForCausalLM。
2. 确认八张 H200 在同一 NVLink/NVSwitch 节点；不要把当前 GLM DSA CP 扩到多机。
3. 改为 --tp 8 --dp 1 --attn-cp-size 8；不要保留 --dp 4。
4. 成对打开 --enable-prefill-cp --cp-strategy interleave。
5. MoE 保留 --moe-a2a-backend deepep，并观察专家负载与 A2A 通信。
6. 注意最终日志：DSA + DP attention 可能把用户设定的 chunked-prefill-size 自动调低；以生效值为准。
7. 先做固定采样的正确性 A/B，再做冷/热缓存压力测试。
8. 观察 Application startup complete 和 Uvicorn running，而不是仅看进程是否仍存在。

如果目标是提升四份独立流量的总吞吐，原 DP=4 更贴近该目标；如果目标是降低少数条超长 GLM-5.2 prompt 的冷缓存 TTFT，本文的 TP8 + CP8 + DP1 才是应与基线比较的方案。两者优化对象不同，不能直接叠加。

## 12. `python/sglang/srt/layers/cp/` 的实际调用顺序

本节说明的是本文目标配置（GLM-5.2 DSA、`--enable-prefill-cp --cp-strategy interleave`）的一次 **extend/prefill** 的真实调用链。`cp/` 目录不是一个单独的模型层：它在服务参数初始化、ModelRunner 边界、DSA attention/KV cache 路径和可选 CUDA Graph/Decode 优化处被分别调用。

### 12.1 总览：GLM-5.2 + DSA + interleave 的主路径

~~~mermaid
flowchart TD
    A[ServerArgs.__post_init__] --> B[base.init_cp_strategy]
    B --> C[InterleaveCPStrategy 单例]

    D[EagerRunner._execute_extend] --> E[utils.is_cp_v2_active]
    E -->|True| F[utils.prepare_cp_forward]
    F --> G[InterleaveCPStrategy.build_metadata]
    G --> H[padding.pad_logical_token_to_physical]

    H --> I[EagerRunner._execute_extend_cp_v2]
    I --> J[utils.cp_shard_model_inputs]
    J --> K[interleave.shard_hidden_states / shard_position_ids]
    K --> L[InterleaveCPStrategy._interleave_shard]
    L --> M[padding.pad_local_rows]

    M --> N[GLM/DeepSeek 模型 body]
    N --> P[DSA backend: strategy.shard_per_request]
    P --> Q[本 rank DSA attention / MoE]
    Q --> R[dsa_backend.materialize_full_kv_cp]
    R --> S[interleave.materialize_full_mla_kv]
    S --> T[interleave.gather_kv_cache]
    T --> U[_gather_interleaved_tensor: all-gather + 反交织]

    U --> V[utils.cp_gather_after_forward]
    V --> W[interleave.gather_hidden_states]
    W --> X[_gather_interleaved_tensor]
    X --> Y[logits processor]

    N -. DeepSeek-V4 等额外路径 .-> O[utils.cp_round_robin_input_ids_v2]
~~~

箭头中的 `all-gather + 反交织` 有两类用途，不能混淆：

- `materialize_full_mla_kv()` 的 gather 用于把每个 layer 的局部 MLA KV 恢复为全局 token 顺序并写入 paged KV cache；
- `cp_gather_after_forward()` 的 gather 发生在模型 body 末端，用于把局部 hidden states 恢复为 logits processor 所需的原始请求顺序。

### 12.2 阶段 0：策略单例的创建

服务参数完成校验后，`python/sglang/srt/server_args.py` 会调用：

~~~python
from sglang.srt.layers.cp.base import init_cp_strategy

init_cp_strategy(self)
~~~

对应的 `python/sglang/srt/layers/cp/base.py` 调用顺序为：

~~~text
init_cp_strategy(server_args)
  ├─ enable_prefill_cp 为 False 或 attn_cp_size <= 1：_STRATEGY = None
  ├─ ContextParallelStrategyKind.from_string(server_args.cp_strategy)
  └─ cp_strategy == "interleave"
       └─ InterleaveCPStrategy(cp_size=attn_cp_size)
            └─ 进程级 _STRATEGY 单例
~~~

之后各运行时入口统一经 `get_cp_strategy()` 取得该实例。worker 子进程若没有继承已初始化的 `_STRATEGY`，`get_cp_strategy()` 会从运行时 ServerArgs 延迟调用 `init_cp_strategy()` 恢复它。`cp/__init__.py` 仅导出这些公共类型与函数，不包含额外运行逻辑。

### 12.3 阶段 1：EagerRunner 判定并构造 CP metadata

入口为 `python/sglang/srt/model_executor/runner/eager_runner.py`：

~~~python
cp_v2_active = is_cp_v2_active(forward_batch)
if cp_v2_active:
    prepare_cp_forward(forward_batch)
~~~

完整顺序如下：

~~~text
EagerRunner._execute_extend(forward_batch)
  ├─ utils.is_cp_v2_active(forward_batch)
  │    ├─ enable_cp_v2()：检查 SGLANG_ENABLE_CP_V2
  │    ├─ forward_mode.is_context_parallel_extend()
  │    ├─ base.get_cp_strategy() -> InterleaveCPStrategy
  │    └─ InterleaveCPStrategy.can_apply(...)
  │         └─ 总 extend token >= attn_cp_size，且 cp_size > 1
  └─ utils.prepare_cp_forward(forward_batch)
       ├─ InterleaveCPStrategy.build_metadata(...)
       │    └─ 计算每个 rank 的逻辑 token 数 per_rank_actual_token
       ├─ padding.pad_logical_token_to_physical(metadata)
       │    ├─ 保存 per_rank_logical_token
       │    └─ 对齐 per_rank_actual_token / max_rank_len
       ├─ （需要 DP attention buffer 时）set_local_dp_buffer_len(...)
       └─ 裁剪 out_cache_loc 到真实 num_tokens
~~~

这里 `build_metadata()` 的 `per_rank_actual_token` 在 padding 前是**逻辑长度**；`pad_logical_token_to_physical()` 随后把它改写为每 rank 相同的**物理通信长度**，并把原值保存到 `per_rank_logical_token`。后续的 CP collective 使用物理长度，反交织输出时使用逻辑长度去掉尾部零行。

### 12.4 阶段 2：模型边界的输入切分

仍在 `EagerRunner._execute_extend_cp_v2()`，模型 body 运行前的调用顺序为：

~~~text
EagerRunner._execute_extend_cp_v2(...)
  ├─ model.get_input_embeddings()(forward_batch.input_ids)
  └─ with utils.cp_shard_model_inputs(input_embeds, positions, forward_batch)
       ├─ utils.cp_shard_hidden_states(...)
       │    └─ InterleaveCPStrategy.shard_hidden_states(...)
       │         ├─ _interleave_shard(x[:total_seq_lens])
       │         └─ padding.pad_local_rows(..., dim=0)
       ├─ utils.cp_shard_position_ids(...)
       │    └─ InterleaveCPStrategy.shard_position_ids(...)
       │         ├─ _interleave_shard(positions[:total_seq_lens])
       │         └─ padding.pad_local_rows(..., dim=0)
       ├─ 若 speculative hidden_states 覆盖完整 batch：按同一规则切分
       └─ yield (sharded_hidden_states, sharded_positions)
  └─ model.model(..., input_embeds=sharded_hidden_states)
~~~

`_interleave_shard()` 的局部规则是：全局 token `i` 归属 `i % cp_size` 对应的 rank。长度恰好整除 CP size 时，它使用 reshape/选列；长度不整除时，使用 `arange(cp_rank, tokens, cp_size)`，使前 `remainder` 个 rank 各多持有一个 token。`pad_local_rows()` 只在尾部补零，绝不改变真实 token 的相对次序。

### 12.5 阶段 3：进入 GLM-5.2/DSA 模型 body 后的 CP 调用

`python/sglang/srt/models/glm4_moe.py` 中的 `GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM`。对本文的 GLM-5.2 主路径，`python/sglang/srt/models/deepseek_v2.py` 会看到 `is_cp_v2_active(forward_batch) == True`，因此跳过模型内部 CP-v1 的 metadata 创建和 `cp_split_and_rebuild_*`；输入的 embedding/positions 已在阶段 2 由 EagerRunner 切为本 rank 的布局。

随后 DSA backend 仍需为 attention 的每条请求准备局部 query 长度：

~~~text
GLM-5.2: DeepseekV2ForCausalLM.forward(...)
  ├─ is_cp_v2_active(forward_batch) 为 True
  │    └─ 不执行 CP-v1 的 prepare_context_parallel_metadata / cp_split_and_rebuild_*
  └─ DeepSeekModel.forward(...) 使用 EagerRunner 已切分的 input_embeds / positions
       └─ DSA backend 构造本地 attention metadata
~~~

在 `python/sglang/srt/layers/attention/dsa_backend.py`，DSA attention 的实际 CP 调用为：

~~~text
DSA backend 构造 prefill metadata
  ├─ can_dsa_prefill_cp_round_robin_split(forward_batch)
  ├─ is_cp_v2_active(forward_batch)
  ├─ strategy.shard_local_tokens(seqlens_expanded)
  │    └─ InterleaveCPStrategy._interleave_shard(...)
  └─ strategy.shard_per_request(extend_seq_lens_cpu, extend_seq_lens)
       ├─ CPU：计算 q_lens_cpu、bs_idx_cpu
       └─ Triton：dsa_cp_round_robin_split_q_seqs_kernel
            └─ 填充 device 侧 q_lens、bs_idx
~~~

`shard_per_request()` 中的 `extra_seq` 会把前一请求未填满一轮 CP 的余数带入下一请求；这是因为输入在 batch 内是平铺的，全局 round-robin 不会在每个请求边界重新从 rank 0 开始。

注意 `InterleaveCPStrategy.run_attention()` 在本配置中返回 `None`：它是 zigzag/FlashAttention 的策略接口占位。实际 DSA 稀疏 MLA attention 由 `dsa_backend.py` 自己执行，并消费上面准备好的 local query、局部长度和 indexer metadata。

#### 补充：`cp_round_robin_input_ids_v2()` 的独立调用点

`utils.cp_round_robin_input_ids_v2()` 属于 `cp/` 的公共辅助函数，但不是上述 GLM-5.2 `DeepseekV2ForCausalLM` 主路径的必经调用。它在 `python/sglang/srt/models/deepseek_v4.py` 和 `deepseek_v4_nextn.py` 的 CP-v2 分支中用于需要额外重排 token ID 的 DeepSeek-V4 系列：

~~~python
if use_prefill_cp:
    if cp_v2_active:
        input_ids = cp_round_robin_input_ids_v2(input_ids, forward_batch)
~~~

其调用链为：

~~~text
模型 body
  └─ utils.cp_round_robin_input_ids_v2(input_ids, forward_batch)
       ├─ MoE A2A backend 非 none（如 deepep）
       │    └─ cp_shard_hidden_states(input_ids, forward_batch)
       │         └─ InterleaveCPStrategy._interleave_shard(...)
       └─ 否则
            ├─ 按 physical token 总长度补零
            └─ view(-1, attn_cp_size).T.flatten() 得到 rank-major token ID 布局
~~~

DeepEP 存在时走第一条分支：token ID 的布局与策略的 local hidden states 一致，随后由 MoE A2A 后端完成 dispatch/combine。没有 A2A 后端时，第二条分支显式构造与物理 CP buffer 匹配的 rank-major token ID 顺序。

### 12.6 阶段 4：每层 MLA KV 的全局物化

在 DeepSeek MLA attention forward 中，DSA prefill CP 需要把本层局部 K-nope/K-rope 恢复为全局 token 顺序后写入缓存：

~~~text
forward_mla.py
  └─ dsa_backend.materialize_full_kv_cp(attn_mla, forward_batch, ...)
       ├─ utils.is_cp_v2_active(forward_batch)
       └─ InterleaveCPStrategy.materialize_full_mla_kv(...)
            ├─ concat([k_nope, k_rope]).squeeze(1)
            ├─ InterleaveCPStrategy.gather_kv_cache(...)
            │    └─ _gather_interleaved_tensor(...)
            │         ├─ 按 physical_rank_len 补零
            │         ├─ attn_cp_all_gather_into_tensor(...)
            │         └─ index_select(gather_indices) 恢复 t0,t1,...
            ├─ 按 kv_lora_rank 切回 k_nope / k_rope
            └─ token_to_kv_pool.set_mla_kv_buffer(...)
~~~

反交织的核心索引是：

~~~python
gather_indices = (
    flat_indices % self.cp_size
) * physical_rank_len + flat_indices // self.cp_size
~~~

第 `i` 个全局 token 的 owner 是 `i % cp_size`，它在该 owner 局部 buffer 内的行号是 `i // cp_size`；上式把这对坐标转换为 all-gather 后 rank-major 大张量的一维行号。因此 KV cache 最终仍可按原始全局 token 顺序供后续 decode 查询。

### 12.7 阶段 5：模型 body 结束后的 hidden-state gather 与 logits

`EagerRunner._execute_extend_cp_v2()` 只在 pipeline 最后一个 rank 上执行这一阶段：

~~~text
model.model(...) 返回本 CP rank 的局部 hidden_states
  └─ utils.cp_gather_after_forward(hidden_states, forward_batch, stream)
       └─ InterleaveCPStrategy.gather_hidden_states(...)
            └─ _gather_interleaved_tensor(...)
                 ├─ local logical rows -> physical-length padded_x
                 ├─ attn_cp_all_gather_into_tensor(gathered, padded_x)
                 └─ index_select(gather_indices) -> 全局逻辑 token 顺序
  └─ model.logits_processor(forward_batch.input_ids, hidden_states, ...)
~~~

KV gather 与 hidden-state gather 共用 `_gather_interleaved_tensor()`，但触发次数、数据类型和目的不同：前者通常在每个 attention layer 的 KV materialize 阶段发生，后者每个请求在 transformer body 结束后发生一次。若模型捕获了辅助 hidden states，EagerRunner 会对它们使用相同的 `cp_gather_after_forward()` 规则。

### 12.8 `cp/` 中另外三个分支的调用位置

| 模块 | 何时进入 | 主要调用顺序 | 与本文 interleave 主路径的关系 |
|---|---|---|---|
| `zigzag.py` | `--cp-strategy zigzag`，FlashAttention 或 TRT-LLM MHA prefill | `build_metadata` → `shard_hidden_states/position_ids` → `run_attention` → `_all_gather_reorganized` → `gather_hidden_states/gather_kv_cache` | 与 interleave 二选一；GLM-5.2 DSA 的本文配置不走此策略。 |
| `bcg.py` | 已启用 CP-v2 且满足 `supports_prefill_cp_bcg()` 的 prefill Breakable CUDA Graph | `PrefillCPBCGInput.prepare` → `prepare_cp_forward` → `cp_split_before_forward` → 固定地址 buffer copy → `execute_prefill_cp_bcg` replay → `cp_gather_after_forward` → logits | 当前仅支持 zigzag + TRT-LLM MHA 等受限组合；本文 interleave + DSA 正常 eager 主路径不走它。 |
| `cp_decode_attn_tp.py` | `enable_cp_decode_attn_tp` 打开、attention-CP 大于 1，且当前不是 CP prefill/DSA prefill | `get_cp_decode_attn_tp_ctx` → `set_decode_attn_tp` → `maybe_use_decode_attn_tp` → 临时切片 linear weight/scale 与 head 数 → `finally` 恢复 | 面向 decode 的 CP-as-TP 权重切片优化，不参与本文 prefill interleave 的本地 token 切分。 |

### 12.9 阅读源码时的推荐顺序

若要从入口连续追踪一次 GLM-5.2 interleave prefill，建议按以下文件和函数顺序阅读：

1. `python/sglang/srt/server_args.py`：`ServerArgs.__post_init__` 末尾的 `init_cp_strategy(self)`。
2. `python/sglang/srt/layers/cp/base.py`：`init_cp_strategy()`、`get_cp_strategy()`、`ContextParallelStrategy` 抽象接口。
3. `python/sglang/srt/model_executor/runner/eager_runner.py`：`_execute_extend()`、`_execute_extend_cp_v2()`。
4. `python/sglang/srt/layers/cp/utils.py`：`is_cp_v2_active()`、`prepare_cp_forward()`、`cp_shard_model_inputs()`、`cp_round_robin_input_ids_v2()`、`cp_gather_after_forward()`。
5. `python/sglang/srt/layers/cp/interleave.py`：`build_metadata()`、`_interleave_shard()`、`shard_per_request()`、`_gather_interleaved_tensor()`、`materialize_full_mla_kv()`。
6. `python/sglang/srt/layers/cp/padding.py`：`pad_logical_token_to_physical()` 与 `pad_local_rows()`，理解逻辑/物理 token 长度差异。
7. `python/sglang/srt/layers/attention/dsa_backend.py`：`shard_per_request()` 的实际消费和 `materialize_full_kv_cp()` 入口。
8. `python/sglang/srt/models/glm4_moe.py`、`python/sglang/srt/models/deepseek_v2.py`、`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`：GLM 继承关系、CP-v2 模型 body 分支及每层 MLA KV materialize 调用点；`deepseek_v4.py` 的 `cp_round_robin_input_ids_v2()` 为额外的模型族调用示例。

这个顺序将“启动时创建策略 → batch 决定是否启用 → 输入切分 → DSA 每层运行/写 KV → 末端恢复 logits 顺序”的控制流串起来；MoE 的 DeepEP dispatch/combine 位于模型 body 内部，与 CP 的 token 局部所有权并行协作，但不是 `layers/cp/` 目录本身负责的调度逻辑。
