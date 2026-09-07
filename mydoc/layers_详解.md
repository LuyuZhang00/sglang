# SGLang `python/sglang/srt/layers` 目录逐文件详解

> **文档说明**：本文档对 `python/sglang/srt/layers/` 下全部 339 个 Python 文件逐一说明「作用 / 关键内容 / 实现逻辑要点」，重点面向想读懂 SGLang 推理栈源码的读者，作为阅读时的索引地图。
>
> - 生成日期：2026-09-07；依据代码分支：`main`（近期 commit `b5c9b68f03` 附近）
> - 顶层 35 个文件 + 子目录 304 个文件（`attention/` 112、`quantization/` 101、`moe/` 62、`cp/` 8、`rotary_embedding/` 8、`dcp/` 5、`deep_gemm_wrapper/` 4、`utils/` 4）
> - 术语约定：TP=张量并行、EP=专家并行、DP=数据并行、CP=上下文并行、DCP=decode 上下文并行、PP=流水线并行、SP=序列并行；`sglang.kernels`（下称 sgl-kernel）是 CUDA/C++/Triton kernel 的实现层，本目录只做 Python 层的封装、调度与数据搬运。
> - 代码演进很快：本文档是快照式描述，类名/注册名以实际代码为准。

---

## 一、目录总览

### 1.1 这一层解决什么问题

`layers` 是 SGLang 运行时（`srt`）的「模型层构建与执行」核心目录：`srt/models/*` 里的各类模型（Qwen、DeepSeek、Kimi、GLM 等）在 `__init__` 中拼装本目录提供的算子模块，在 `forward` 中按本目录提供的通信编排执行。它不实现真正的 CUDA kernel（那些在 `sglang.kernels`、`sgl-kernel`、FlashInfer、DeepGEMM、aiter 等下层），而是负责：

1. **多平台算子封装**：几乎所有算子类继承 `sglang.kernels.fused_op.BaseFusedOp`，按「平台 `forward_<device>` / 后端 `forward_<backend>` / `forward_native`」的优先级做静态分派（CUDA / HIP / NPU / XPU / CPU-AMX / MUSA）。
2. **张量并行（TP）切分与权重加载**：`linear.py`、`vocab_parallel_embedding.py`、`parameter.py` 构成一套带量化感知的权重加载协议（`weight_loader_v2` + vLLM 风格参数类）。
3. **量化方法体系**：`quantization/` 的 Config → Method → Scheme 三层结构，让线性层/MoE/attention/KV cache 都能按 checkpoint 的量化格式执行对应 kernel。
4. **注意力后端注册表**：`attention/` 的 `attention_registry.py` + `base_attn_backend.py` 定义 `AttentionBackend` 接口，`RadixAttention`（顶层）按后端名分发到 FlashInfer / Triton / FA3/FA4 / DSA / 各类 MLA 内核。
5. **跨层数据搬运（通信编排）**：`communicator.py` 用 `ScatterMode` 状态机管理 hidden_states 的 ALL-REDUCE / LayerNorm 融合 / DP gather-scatter / MoE 布局转换，尽量把通信次数压到最低。
6. **解码采样管线**：`logits_processor.py` → `sampler.py` → `logprob_processor.py` 把最后一层 hidden_states 变成 next token 与各类 logprob，全程 CUDA-graph 友好。

### 1.2 一份简要的调用链地图

```
                     srt/models/<Model>.forward
        ┌───────────────┼───────────────────┬────────────────────┐
        ▼               ▼                   ▼                    ▼
  linear.py      RadixAttention      FusedMoE(layers/moe/)   LogitsProcessor
 (quant_method    │ attention backend   │ 路由 topk.py         │ lm_head + TP/DP 归约
  apply→scheme)   ▼ registry 选择       ▼ dispatcher(dense/   ▼
        │      flashinfer/triton/    EP all2all → MoeRunner   Sampler ──► next token
        │      fa3/fa4/dsa/...        → runner core × kernel    ▲
        └────── communicator.py ◄─────┘                        logprob_processor
                (ScatterMode 状态机，融合 AR+LayerNorm+DP gather/scatter)
```

一次 decode 的典型数据流：模型层 forward → 各算子的 GEMM 走 `quant_method.apply`（未量化则 `UnquantizedLinearMethod`）→ 注意力走 `RadixAttention` → backend；hidden_states 与残差在层间按 `ScatterMode` 被 `LayerCommunicator` 搬运（TP 下 ALL-REDUCE，DP attention 下先 DP gather 再做 LayerNorm，EP MoE 下做 all2all）；最后一层输出交给 `LogitsProcessor` 过 LM head 算 logits，再由 `Sampler` 采样并回填 logprob。

### 1.3 贯穿全目录的设计习惯（读代码前值得先知道）

- **注册表 + 工厂**：attention 后端（`ATTENTION_BACKENDS`）、量化方法（`BASE_QUANTIZATION_METHODS`）、MoE runner 后端（`register_fused_func` 等）、KV cache 量化（`KV_CACHE_QUANT_REGISTRY`）都以「字符串名 → 类」注册表组织，由 ServerArgs 或 checkpoint 配置选择。
- **自定义 op + fake_impl**：凡是要进 torch.compile / CUDA graph 的算子，几乎都通过 `@register_custom_op(op_name=..., fake_impl=...)` 注册，graph 捕获期用 fake 实现产出形状。
- **CUDA graph 约定**：metadata 分「图外 host 逻辑」与「图内静态形状 GPU op」两段（`init_forward_metadata_out_graph` / `_in_graph`）；replay 时原地刷新静态缓冲（grow-only buffer）；`needs_cpu_seq_lens=False` 表示页表在设备端构建、避免 D2H 同步。
- **对称内存（symmetric memory）**：NCCL 多 rank 共享虚拟地址，`use_symmetric_memory` 上下文内分配的张量可被 NVLS/Multicast/零拷贝通信直接消费（与 PDL、MNNVL 深度融合）。
- **数据容器**：新代码一律用 `msgspec.Struct`/frozen struct（本文档简述时不再逐一标注）。

### 1.4 顶层文件分组速览

| 分组 | 文件 |
|---|---|
| 张量并行与基础算子 | `linear.py` `conv.py` `activation.py` `layernorm.py` `layernorm_sp.py` `vocab_parallel_embedding.py` `clippable_linear.py` `model_parallel.py` `n_gram_embedding.py` |
| 参数与平台/量化工具 | `parameter.py` `amx_utils.py` `int4fp8_utils.py` `modelopt_utils.py` `rocm_linear_utils.py` |
| 层间通信与 DP attention | `communicator.py` `communicator_dsa_cp.py` `communicator_mhc.py` `dp_attention.py` `zero_copy_context.py` `aux_hidden_states.py` |
| 注意力封装与 kernel 融合 | `radix_attention.py` `radix_linear_attention.py` `flashinfer_comm_fusion.py` `flashinfer_mnnvl_cutedsl.py` `k3_ar_fusion.py` `k3_gemm_ar.py` `k3_sp_collective.py` `attn_residual.py` `mova.py` |
| logits / 采样 / 池化 | `logits_processor.py` `logprob_processor.py` `logsumexp.py` `sampler.py` `pooler.py` `sparse_pooler.py` |

子目录：`attention/`（注意力后端全集）、`moe/`（MoE 路由/分发/后端）、`quantization/`（量化方法全集）、`cp/`（prefill 上下文并行）、`dcp/`（decode 上下文并行）、`rotary_embedding/`（RoPE 及变体）、`deep_gemm_wrapper/`（DeepGEMM JIT 封装）、`utils/`（杂项工具）。

---

## 二、顶层文件详解（35 个）

> 本节按主题分 5 组，逐一介绍文件的作用、关键内容与实现逻辑。信息来源为对源码的通读，重要类名/函数名与代码一致。

### 2.1 张量并行与基础算子

这组文件提供 TP 切分下的 GEMM/嵌入主干（承自 vLLM 0.6.x 并深度定制）、归一化/激活/卷积的融合算子，以及 torch DTensor 原生 TP 与 n-gram 投机解码等支线。

#### linear.py

- **作用**：TP 感知线性层体系（承自 vLLM 0.6.4），几乎所有 Transformer 模块权重加载与 GEMM 调度的中枢。
- **关键内容**：`LinearBase` 及并行类：`ReplicatedLinear`、`ColumnParallelLinear`、`MergedColumnParallelLinear`、`QKVParallelLinear`、`RowParallelLinear`、`MergedColumnParallelRepeatedLinear`、`ColumnParallelBatchedLinear`；`WEIGHT_LOADER_V2_SUPPORTED`（19 个量化方法名）；`adjust_marlin_shard`、`adjust_bitsandbytes_4bit_shard`、`adjust_shard_offsets` 等。
- **实现逻辑要点**：
  1. 类职责：`ColumnParallelLinear` 沿输出维切分（`gather_output=True` 时 all-gather）；`QKVParallelLinear` 按 head 维切分融合 QKV 投影，支持 `kv_tp_rank/kv_tp_size`、`v_head_size`、MQA/GQA 的 KV head 跨 rank 复制（`num_kv_head_replicas`）；`RowParallelLinear` 沿输入维切分、输出 all-reduce；`MergedColumnParallelLinear` 按 `shard_id` 分别 narrow（gate/up 等拼接权重）。
  2. 权重加载双轨：旧 `weight_loader`（`narrow` 切分）与 `weight_loader_v2`（`BasevLLMParameter` 子类协议），处理 GGUF `UninitializedParameter` 物化、bitsandbytes 4bit 偏移、Marlin tile/pack_factor、`PerTensorScaleParameter` 标量、`BlockQuantScaleParameter` 块级切分、checkpoint 已融合的 Phi-3 权重、qwen2_5_VL 非 8 对齐 padding 等大量特例。
  3. 前向优化点：`RowParallelLinear.forward` 支持 `input_is_parallel`、`skip_all_reduce`、`output_tensor`（量化方法 `apply_into` 直写调用方缓冲）、`use_dp_attention_reduce`（走 attn-TP 组）、prefill 下 `enable_quant_communications` 的量化通信 all-reduce；对称内存经 `use_symmetric_memory` 分配。
  4. SP 集成：`get_forward().sp_active` 时 Column/Row 并行层分别委托 `layernorm_sp` 的融合 matmul。
- **备注**：量化方法通过 `scheme` 属性（如 compressed-tensors）挂到层上；`quant_method.apply` 外包 kernel 日志（`wrap_method_with_debug_kernel_once`）。

#### conv.py

- **作用**：patch embedding 用的 `Conv2dLayer` / `Conv3dLayer`，核心优化是当 `kernel_size == stride`、无 padding/dilation、groups==1 时用 unfold + `F.linear` 等价替换卷积。
- **实现逻辑要点**：unfold 后 reshape 与 `weight.reshape(out_channels, -1)` 做线性变换，CUDA 上更快，且规避 PyTorch 2.9.1 + CuDNN<9.15 的 Conv3d bug；支持字符串 padding `"same"/"valid"` 与四种 padding_mode；`Conv3dLayer.forward_cpu` 走 AMX 的 `conv3d_embed_cpu`（`is_vnni=True`）并挂权重打包。
- **备注**：被多模态模型（Qwen2-VL 类）的 patch embedding 使用。

#### activation.py

- **作用**：融合激活算子（act-and-mul 门控类为主）的多后端统一实现。
- **关键内容**：`SiluAndMul`、`GeluAndMul`（tanh/none 近似）、`NewGELU`、`ReLU2`、`QuickGELU`、`SituAndMul`（Kimi K3 的 SituGLU：`beta*tanh(gate/beta)*sigmoid(gate)*up`）、`XIELU`（可学习参数）、`ScaledActivation`（AWQ 式 `act(x)/scales`）；工厂 `get_act_fn`（`_ACTIVATION_REGISTRY`）。
- **实现逻辑要点**：CUDA 优先走 JIT kernel（hidden size 整除向量宽度时才可用，否则回退 sgl-kernel AOT kernel）；HIP 可走 aiter；NPU 走 `torch_npu.npu_swiglu` 等；CPU AMX 走 `sgl_kernel.*_cpu`；RL on-policy 确定性目标下强制 `forward_native`。CUDA graph `TC_PIECEWISE` 后端下回退 native。

#### layernorm.py

- **作用**：归一化算子全家桶（RMSNorm/LayerNorm/Gemma 系）的多后端融合实现，本目录后端分派最复杂的文件之一。
- **关键内容**：`RMSNorm`（主力：residual 融合、批不变模式、FP8 量化融合、allreduce 融合）、`LayerNorm`（带 bias）、`GemmaRMSNorm`（存储权重=1+x）、`Gemma3RMSNorm`、`Gemma4RMSNorm`（`scale_shift`）、`RMSNormWithoutScale`；模块级 `_forward_with_allreduce_fusion(_quant_per_group)` 等。
- **实现逻辑要点**：
  1. CUDA 走 sgl-kernel `rmsnorm/fused_add_rmsnorm/gemma_rmsnorm`；HF 语义的 `cast_x_before_out_mul` 走 JIT kernel。
  2. **通信融合**：`_forward_with_allreduce_fusion` 把 ALL-REDUCE+residual+norm 融进 FlashInfer 单 kernel（AMD 走 aiter fused AR）；gfx95 上还有「AR+RMSNorm+per-1x128 FP8 量化」单 kernel 变体（供 GDN 同时出 bf16 与 fp8），失败逐级回退 native。
  3. **量化融合**：`forward_with_per_tensor_quant_fusion` 用 flashinfer 的 fused rmsnorm+quant 把「norm+静态逐张量 FP8 量化」合一，仅当下游 linear 是原生 `Fp8LinearMethod`（非 block/mxfp8/marlin）时触发。
  4. 平台覆盖：HIP（aiter / vLLM `_custom_ops` 回退）、NPU（`npu_rms_norm` 系列）、CPU AMX、MUSA。

#### layernorm_sp.py

- **作用**：Megatron 风格 LayerNorm 序列并行（SP，arXiv:2205.05198）：把 row 并行层的 all_reduce 拆成 reduce_scatter（g-bar）+ all_gather（g），让归一化在序列切分后的激活上执行。
- **关键内容**：`SP_SUPPORTED_ARCHITECTURES = {"Qwen3ForCausalLM"}`；`layernorm_sp_enabled`、`runs_sp`、`set_sp_num_tokens`、`sp_entry_scatter`、`sp_exit_gather`、`maybe_exit_gather`、`column_parallel_g_matmul`、`row_parallel_gbar_matmul`。
- **实现逻辑要点**：仅 prefill 生效、默认关闭；token 数存实例属性（避免 dynamo 按序列长度重编译）；入口把 `[M,h]` 沿 token 维 `tensor_split`、出口 all-gather 后 narrow 回真实 token 数；融合快路径基于 PyTorch symmetric memory 的 `fused_matmul_reduce_scatter`/`fused_all_gather_matmul`（要求未量化、bf16/fp16、无 bias）。

#### vocab_parallel_embedding.py

- **作用**：词表维 TP 切分的 embedding 与 LM head，含 padding、LoRA 扩展词表与融合 kernel。
- **关键内容**：`VocabParallelEmbedding`、`ParallelLMHead`、`pad_vocab_size`、`get_masked_input_and_mask`、`get_embedding_tp_kwargs`、`VocabParallelEmbeddingShardIndices`；`DEFAULT_VOCAB_PADDING_SIZE = 64`。
- **实现逻辑要点**：词表先 pad 到 64 的倍数再按 TP 均分，基础词表与 LoRA 新增词表分开 pad/切分后放同一张表；tp>1 且条件满足时走融合 Triton kernel `fused_vocab_parallel_embedding`；输出在对称内存分配，forward 末尾经 TP all-reduce 或 DP attention 的 attn-TP 归约；`maybe_detect_oob` 把越界 token 转成带位置的异步断言；`get_embedding_tp_kwargs` 统一 DeepSeek-V2 目标族与 EAGLE3/NextN draft 的嵌入布局；`ParallelLMHead.forward` 直接抛 `RuntimeError`（权重只供 Sampler 使用）。

#### clippable_linear.py

- **作用**：为 Gemma 4 视觉/音频编码器提供带逐张量激活 clamp（input/output min/max，默认 ±inf 即 no-op）的 TP 线性层包装。
- **关键内容**：`ClippableRowParallelLinear`、`ClippableColumnParallelLinear`、`ClippableQKVParallelLinear`、`ClippableGLUParallelLinear`、`ClippableGateUpParallelLinear`。
- **实现逻辑要点**：每个包装类内嵌一个 `linear.py` 并行层并持有 4 个 clamp 边界 Buffer（值从 checkpoint 加载）；QKV 变体共享一对 input 边界、输出边界按 q/k/v 各自独立；GLU 变体用 `MergedColumnParallelLinear` 对 value/gate 两半独立切分（普通列并行会把两半混排跨 rank），并动态包裹 `weight_loader` 把融合权重按 shard 自动切分。

#### model_parallel.py

- **作用**：基于 torch DTensor 的原生张量并行工具（与自研 TP 并存，服务特定模型的 torch TP 路线）。
- **关键内容**：`_shard_tensor`、`ColwiseParallelSharded`、`RowwiseParallelMaybeWait`、`tensor_parallel`、`apply_torch_tp`。
- **实现逻辑要点**：`_shard_tensor` 直接算本地分片构造 DTensor，跳过 `distribute_tensor` 的通信（适用于各 rank 持有相同全量张量的推理加载）；`RowwiseParallelMaybeWait` 在输出准备末尾 `wait_tensor` 建立 comm→compute 依赖；`tensor_parallel` 递归遍历模块 `_tp_plan` 属性（"Colwise"/"Rowwise"/"Colwise_Sharded"）。

#### n_gram_embedding.py

- **作用**：n-gram 投机解码模型的嵌入层：word embedding + over-embedding n-gram 表 + 投影。
- **关键内容**：`NgramEmbedding`（`word_embeder`、`oe_embeder`、`oe_projection`、滚动哈希参数 `oe_mods`/`oe_weights`）。
- **实现逻辑要点**：构造时预计算各 n-gram embedder 的 vocab 偏移前缀和与滚动哈希参数（`pow(num_embeddings, delta, mod)` 查表）；forward 用 JIT CUDA kernel `compute_n_gram_ids` 批量算 n-gram id，word 与 n-gram 两种 embedding 拼接后 `torch.bmm` 投影再 mean 融合输出；请求级 buffer 按 `chunked_prefill_size`/`max_running_requests` 预分配。

### 2.2 参数与平台/量化工具

这组文件是顶层其余算子与 `quantization/` 之间的「胶水」：`parameter.py` 定义量化感知的 TP 权重加载协议，其余 4 个文件按平台（Intel AMX / ROCm）或工具链（NVIDIA ModelOpt、Quark）提供专项支撑。

#### parameter.py

- **作用**：vLLM 风格参数类体系，把「TP 切分 + 量化打包」的权重加载协议内嵌到 `nn.Parameter` 子类中（`linear.py` 的 `weight_loader_v2` 依赖的类型协议，`__all__` 导出 9 个类）。
- **关键内容**：`BasevLLMParameter`、`_ColumnvLLMParameter`、`RowvLLMParameter`、`ModelWeightParameter`、`GroupQuantScaleParameter`、`ChannelQuantScaleParameter`、`BlockQuantScaleParameter`、`PerTensorScaleParameter`、`PackedColumnParameter`、`PackedvLLMParameter`；函数 `copy_with_check`、`permute_param_layout_`、`_adjust_shard_indexes_for_packing`/`_adjust_shard_indexes_for_marlin`。
- **实现逻辑要点**：
  1. `BasevLLMParameter` 持 `weight_loader` 属性与 `load_column/row/merged_column/qkv_parallel_weight` 协议方法；`_ColumnvLLMParameter`（`output_dim`）/`RowvLLMParameter`（`input_dim`）实现按 `tp_rank * shard_size` narrow 的切分加载，含 CPU 的 `narrow_padded_param_and_loaded_weight` 与 qwen2_5_VL 非 8 对齐的 `pad_or_narrow_weight` 填充。
  2. `PerTensorScaleParameter` 面向融合层（QKV）逐张量 scale：`qkv_idxs={"q":0,"k":1,"v":2}`，`_load_into_shard_id` 把标量/形状 [1] 的 scale 存入对应槽位。
  3. `PackedColumnParameter`/`PackedvLLMParameter` 携带 `packed_factor`（可 `Fraction`）、`packed_dim`、`marlin_tile_size`；`_adjust_shard_indexes_for_packing` 先把 shard 尺寸/偏移除以 pack_factor 再乘 Marlin tile。
  4. `copy_with_check` 用 dtype 排序（fp8 族 < fp16/bf16 < fp32 < fp64）禁止隐式降精度拷贝（除非 `SGLANG_QUANT_ALLOW_DOWNCASTING`）；fp8_e8m0fnu 只允许拷入同类型或 fp32。
- **备注**：另有 `permute_param_layout_` 把参数量化布局重排到指定维度并断言不可 repacking。

#### amx_utils.py

- **作用**：Intel AMX（CPU 高级矩阵扩展）后端支撑：权重打包、量化算法枚举与图像预处理加速。
- **关键内容**：枚举 `CPUQuantMethod`（UNQUANT/INT8_W8A8/FP8_W8A16/INT4_W4A8/MXFP4）、`CPUQuantAlgo`（AWQ/GPTQ）；函数 `fast_preprocess_cpu`、`amx_process_weight_after_loading`、`dim_is_supported`/`dtype_is_supported`/`is_dim_conv_weight`、`_init_amx_conv_state`、类 `PackWeightMethod`。
- **实现逻辑要点**：
  1. 权重加载后若在 CPU 且支持 AMX，调 `torch.ops.sgl_kernel.convert_weight_packed`（普通权重）或 `conv3d_embed_weight_pack`/`causal_conv1d_weight_pack`（卷积权重）预打包。
  2. 打包 GEMM 的维度约束：OC % TILE_N(16)==0 或 OC<16，IC % TILE_K(32)==0；支持 fp16/bf16/uint8/int8/fp8_e4m3fn。
  3. `PackWeightMethod.process_weights_after_loading` 支持 AWQ/GPTQ 的 qweight/qzeros/scales 三件套经 `convert_weight_packed_scale_zp` 统一打包，bias 转 fp32。
- **备注**：被 `conv.py`、`vocab_parallel_embedding.py`（LMHead）及 CPU 量化方法引用；`use_intel_amx_backend` 标志挂在模块上。

#### int4fp8_utils.py

- **作用**：Quark 量化方案的工具函数集（FP8 张量级 / INT4 列级量化与打包）。
- **关键内容**：`quantize_fp8_scale_tensorwise`、`quantize_int4_scale_columnwise`、`pack_int4_to_int32`。
- **实现逻辑要点**：FP8 量化 `scale = amax/FP8_MAX(448.0)` 后 clamp 到 float8_e4m3fn；INT4 列级量化每列独立 `scale = amax/7`（S4_MAX=7）并 clamp 到 [-7,7] 存 int8；`pack_int4_to_int32` 把 8 个 int4 打包进一个 int32，支持 `reorder=True` 的 GPTQ 风格交错序 `order_map=[0,2,4,6,1,3,5,7]`——先 `& 0x0F` 清高 4 位（否则负数符号扩展使 bit_or 出错），再逐 4 bit 左移拼接。
- **备注**：仅支持 ≤2 维张量；供 Quark 系列量化方法在加载期使用。

#### modelopt_utils.py

- **作用**：NVIDIA ModelOpt 量化算法名与 sglang 运行时方法的映射常量。
- **关键内容**：`QUANT_CFG_CHOICES`（fp8/int4_awq/w4a8_awq/nvfp4/nvfp4_awq → ModelOpt 配置名）、类型别名 `ModelOptQuantMethod`（`"modelopt_fp8" | "modelopt_fp4" | "mxfp8"`）、`_MODELOPT_QUANT_ALGO_TO_METHOD`、`canonicalize_modelopt_quant_algo`。
- **实现逻辑要点**：映射采用精确 allowlist（大小写不敏感的全名匹配），刻意避免 `MXFP8` 因含 "FP8" 子串被误判为普通 FP8；W4A16_NVFP4/NVFP4_AWQ 等统一归入 `modelopt_fp4` 族；运行时能力检查留给各消费方。
- **备注**：非字符串输入直接返回 None。

#### rocm_linear_utils.py

- **作用**：AMD（ROCm/aiter）GEMM 与 MLA 融合 kernel 的再导出与 DeepSeek-V3 专项工具（与 `linear.py` 解耦的 AMD 补丁集）。
- **关键内容**：`aiter_dsv3_router_gemm`、`get_dsv3_gemm_output_zero_allocator_size`；`__all__` 再导出 `fused_qk_rope_cat`、`fused_qk_rope_cat_and_cache_mla`。
- **实现逻辑要点**：`aiter_dsv3_router_gemm` 用 `aiter.tuned_gemm.tgemm.mm` 自动调优选择 GEMM kernel；`get_dsv3_gemm_output_zero_allocator_size` 仅在 `embedding_dim==7168` 且 256 routed experts 时按 `256*(allocate_size+256)*num_moe_layers` 估算 router GEMM 零输出分配器大小（DSv3 结构特化）。
- **备注**：文件极小。

### 2.3 层间通信与 DP attention

这组文件解决「单层前向后 hidden_states/residual 如何在多个并行组之间搬运」：`communicator.py` 是总调度台，`dp_attention.py` 提供 DP 维集体通信原语，`communicator_dsa_cp.py`/`communicator_mhc.py` 是 DSA-CP 与 MHC 模型的通信特化，`zero_copy_context.py` 与 `aux_hidden_states.py` 分别是 MoE 输出零拷贝与 draft 模型辅助 hidden states 采集。

#### communicator.py

- **作用**：层间通信编排核心——按 ScatterMode 状态机搬运 hidden_states/residual，并把 all-reduce、LayerNorm、DP 采集/散射、MoE 布局转换融合进最少次数的集体通信。
- **关键内容**：`ScatterMode` 枚举；`LayerScatterModes`（`init_new` 由 `_LayerModeComputationContext` 推导 layer_input/attn/mlp/middle_residual/layer_output 五种 mode）；`LayerCommunicator`（`prepare_attn`/`prepare_mlp`/`postprocess_layer`/`should_use_reduce_scatter`/`should_fuse_mlp_allreduce_with_next_layer`）；`AttentionInputs`/`AttnTpContext`（attn_tp_input_scattered 模式与 qkv_latent 缓存）；三个 fn 工厂 `CommunicateSimpleFn`/`CommunicateWithAllReduceAndLayerNormFn`/`CommunicateSummableTensorPairFn`；`tp_reduce_scatter`、`apply_flashinfer_allreduce_fusion`、`apply_aiter_all_reduce_fusion`、`_fused_rmsnorm_fp8_per_token_quant`。
- **实现逻辑要点**：
  1. 三个 fn 工厂按 (input_mode, output_mode) 查表绑定静态方法，同 group size 走 `_trivial` 零通信。
  2. `prepare_attn`：input-scattered 先 TP reduce-scatter；带 `_sglang_needs_allreduce_fusion` 标记时走 FlashInfer/aiter 融合「AR+RMSNorm」（`layernorm.py` 的 `forward_with_allreduce_fusion(_quant_per_group)`，gfx95 上还有 mxfp4/fp8 fused quant 分支），否则 `moe_tensor_model_parallel_all_reduce` + LayerNorm。
  3. `prepare_mlp` 的 `_gather_hidden_states_and_residual`：attention-TP all-reduce（普通或 `attention_tensor_model_parallel_quant_all_reduce`）；DP>1 时按 `force_layernorm_before_dp_gather` 决定先 LN 再 `dp_gather_replicate`，或先 `dp_gather_partial`+`dp_scatter` 再 LN；`_gather_hidden_states_and_residual_moe` 追加 moe_cp all-gather（zigzag 不均时 pad 到 per-rank max）。
  4. `postprocess_layer` 的 `_scatter_hidden_states` 用 `reduce_scatterv`/`dp_reduce_scatter_tensor`/`dp_scatter` 打回本地；融合决策：`FUSE_ALLREDUCE_MAX_BATCH_SIZE=2048`，要求 SM90/SM100+FlashInfer+非 DP attention；MOE_FULL、EP+TP hybrid 双组归约、input_scattered、Eagle 投机下强制关闭 `should_fuse_mlp_allreduce_with_next_layer`。
  5. LayerNorm SP 激活时构造全 SCATTERED 的 `_sp_variant` 兄弟实例代理。
- **备注**：被几乎所有 Transformer 模型及 `base_runner.py`、`forward_batch_info.py`、`two_batch_overlap.py` 引用。

#### communicator_dsa_cp.py

- **作用**：DSA（含 MLA）prefill context parallel 场景下 `LayerCommunicator` 的特化。
- **关键内容**：`DSACPLayerCommunicator`、`DSACPCommunicateSimpleFn`、`DSACPCommunicateWithAllReduceAndLayerNormFn`、`DSACPCommunicateSummableTensorPairFn`、`dsa_cp_gather_hidden_states`/`dsa_cp_reduce_scatter_hidden_states`、`maybe_prefetch_next_full_attention_kv`。
- **实现逻辑要点**：prefill CP 下 attn 输出 scatter→full 走 `attn_cp_all_gather_into_tensor`、post-MLP full→scatter 走 `attn_cp_reduce_scatter_tensor`（各自要求 attn_dp_size==attn_tp_size==1）；decode 保持 full 零通信；`DSACPCommunicateSummableTensorPairFn.get_fn` 先精确匹配 (FULL,SCATTERED→SCATTERED) 枚举再查 group size——CP 下 FULL 与 SCATTERED 数据布局不同，即使组大小巧合相等也不能按组匹配；`maybe_prefetch_next_full_attention_kv` 在层拆分下经 KV pool `prefetch_kv_buffer` 提前一层广播下一 full-attention 层的 DSA KV，与当前层计算重叠。
- **备注**：与 `attention/dsa/utils.py`、`cp/utils.py` 的 prefill-CP 判定联动。

#### communicator_mhc.py

- **作用**：MHC（多头压缩）模型的层通信特化，把 hc_attn_pre/hc_ffn_pre/hc_post 压缩-解压 kernel 嵌入通信流程。
- **关键内容**：`MHCState`（`attn_split`/`attn_to_mlp`/`mlp_combine`/`reset_aux`，持 `h_res`/`h_post` scratch）、`MHCLayerCommunicator`、`MHCCommunicateWithAllReduceAndLayerNormFn`、`MHCCommunicateSummableTensorPairFn`、`tp_all_gather_hidden_states`。
- **实现逻辑要点**：`get_fn` 复用基类查表后以 replacement 表把 `_simple`/`_gather_hidden_states_and_residual`/`_scatter_hidden_states_and_residual` 换成带 `mhc` 参数的版本；input-scattered 分支走 reduce-scatter→attn_to_mlp→`attn_tp_all_gather_into_tensor`；首层 `hc_expand(hc_mult)` 扩展宽度、末层 `hc_contract` 收窄并全 gather，中间层按 `hc_mult` 加宽 DP 缓冲；构造期拒绝 MOE_FULL 组合；`should_fuse_mlp_allreduce_with_next_layer` 恒 False。
- **备注**：压缩/解压内核实现在 `sglang/kernels/ops/layernorm/mhc.py`。

#### dp_attention.py

- **作用**：DP attention 的 collective 原语 + 全局/本地 DP 缓冲管理，是 `communicator.py` 各 fn 的底层执行者。
- **关键内容**：`DpPaddingMode`（MAX_LEN/SUM_LEN 及成本驱动的 `get_dp_padding_mode`）、`_DpGatheredBufferWrapper`（`set_metadata`/`set_dp_buffer_len`/`get_global_dp_buffer`/`get_local_dp_buffer(_mhc)`）、`initialize_dp_attention`/`compute_dp_attention_world_info`、`disable_dp_size`、`dp_gather_partial`/`dp_gather_replicate`/`dp_scatter`/`dp_reduce_scatter_tensor`、`_dp_gather_via_all_reduce`/`_dp_gather_via_all_gather`/`_dp_gather_via_all_gatherv(_fp8)`；TBO 异步版 `dp_gather_partial_async`/`dp_reduce_scatterv_async`/`get_dp_tbo_comm_stream`/`get_tbo_persistent_buffer`；`attn_tp_*`/`attn_cp_*`/`attn_cp_overlap_*` 封装、`moe_cp_*` 系列与 `mask_dp_pad_moe_topk_ids`。
- **实现逻辑要点**：
  1. padding 决策：pplx a2a 与 DeepEP-V2 强制 MAX_LEN（防设备侧握手死锁），extend 且 dp>1 用 SUM_LEN，否则按 `sum_len*2 >= max_len*dp_size` 选 MAX_LEN（省通信）或 SUM_LEN；CUDA graph 默认 MAX_LEN。
  2. 三种 gather：all_reduce（零填充 SUM 语义）、all_gather（max_len，attn_tp>1 时先 `reduce_scatter_tensor` 去重再 gather）、all_gatherv（`SGLANG_DP_USE_GATHERV` 变长，可选 fp8 group-128 量化 wire 格式：gatherv 传 uint8 视图 + 独立 scale gatherv + Triton 反量化 kernel；combine 腿保持 bf16，因 NCCL SUM 不支持 fp8）。
  3. MAX_LEN 模式下经 `use_symmetric_memory` 分配 buffer 实现零拷贝；TBO 两 ubatch 共享一条 `dp_tbo_comm` stream、异步 gather/combine 后返回按 (kind, subbatch) 复用的持久化 CUDA event（防事件池耗尽），`get_tbo_persistent_buffer` 提供 grow-only 缓冲。
  4. Elastic EP 下 `update_dp_attention_post_scale` 切到 WORLD group 并用 joiner skip 标志（`world_dp_gather_enabled`/`enable_joiner_all_gather`）。
- **备注**：被 `communicator.py`、`scheduler.py`、`data_parallel_controller.py`、`logits_processor.py`、Eagle draft runners 等广泛引用。

#### zero_copy_context.py

- **作用**：进程级发布/获取 MoE 输出目标张量，实现零拷贝写回（frozen `msgspec.Struct` 单例）。
- **关键内容**：`ZeroCopyContext`（字段 `moe_output`）、模块级单例 `ctx`、`set_moe_output`（contextmanager）、`get_moe_output`、`get_moe_output_spec`。
- **实现逻辑要点**：`msgspec.structs.force_setattr` 在 with 块内写入、退出恢复旧值（嵌套/异常安全）；`get_moe_output` 仅在 shape/dtype/device 匹配且 contiguous 时返回该缓冲（等价于 `empty_like` 替身），否则 None；`get_moe_output_spec` 供无法物化参考张量的调用方（如 trtllm-gen runner 的 fp4-packed 激活）按 spec 匹配。
- **备注**：被 MoE runner（`moe/moe_runner/flashinfer_trtllm.py`、`moe/fused_moe_triton/fused_marlin_moe.py`）读取；`models/kimi_k3.py` 以 `with zero_copy_context.set_moe_output(latent)` 发布 latent 输出。

#### aux_hidden_states.py

- **作用**：Eagle3/DFlash draft 模型的辅助 hidden states 采集与打包。
- **关键内容**：`AuxHiddenStates`（Tensor | List 联合类型）、`AuxHiddenStatePacker`、`AuxHiddenStateAccumulator`、`pack_aux_hidden_states`。
- **实现逻辑要点**：packer 作为 list 的 drop-in 替代，`append` 写入预分配的 `[tokens, K*hidden]` 缓冲对应分片，避免 list 路径 `torch.cat` 约 2x HBM 的瞬态峰值；`copies_on_append=True` 使生产者无需克隆后续会变异的张量；`finalize` 校验捕获数量后返回打包张量；对 Tensor/单元素 list 直通。
- **备注**：被 `communicator.py` 的 `prepare_attn_and_capture_last_layer_outputs` 与 `logits_processor.py` 消费，`models/deepseek_v2.py` 已迁移到 packer 路径；两种表示在代码库中并存。

### 2.4 注意力封装与 kernel 融合

这组文件分两类：`RadixAttention`/`RadixLinearAttention` 是所有模型的注意力统一入口（并把注意力注册为 tc-piecewise CUDA graph 的 split op），其余 7 个是通信与 kernel 的深度融合（FlashInfer 融合 allreduce、Kimi-K3 的 MNNVL/NVLS 融合家族、MoVA 路由投影）。

#### radix_attention.py

- **作用**：通用注意力层封装，并注册 tc-piecewise CUDA graph 的注意力 split op；几乎所有模型的注意力层都继承它。
- **关键内容**：`RadixAttention`（nn.Module）、`AttentionType`（DECODER/DECODER_BIDIRECTIONAL/ENCODER_ONLY）、`unified_attention_with_output`、`unified_attention_with_output_and_lse`、`unified_sparse_attention_with_output`、`attention_with_output_extra_kwargs`、`breakable_*`（`eager_on_graph(True)` 包装）、`force_eager_attention`、`_zero_padded_pcg_tail`。
- **实现逻辑要点**：
  1. 构造保存 head 数/head_dim/scaling/logit_cap/滑动窗口/cross-attention/pos_encoding/use_irope 及 MiniMax-M3 fp8 attn-GEMM scales；quant_method 由 quant_config 注入。
  2. forward 在 extend + tc-piecewise 下走 split custom op：预分配 output 供 backend 直写，把 Q/K/V/`out_cache_loc`/`positions` 窄化到 `global_num_token_non_padded_cpu`，PCG replay 后 `_zero_padded_pcg_tail` 清零 padding 尾巴，防 NaN/Inf 污染残差/MoE 路由/allreduce。
  3. 支持 MLA `mha_companion_layers` 双实例同 layer_id、sparse（idx_q/idx_k）双输出、`return_lse`（chunked-prefix MHA 合并用）；`score_mod`/`aux_tensors`/`rel_bias`/mxfp8 descale 等无法过 custom-op schema 的 kwargs 走 `attention_with_output_extra_kwargs` 普通函数 + breakable 包装。
  4. decode 与其余路径直接 `get_attn_backend().forward`。
- **备注**：与 `tc_piecewise_cuda_graph.py`、`breakable_cuda_graph.py` 及 `compilation_config.register_split_op` 联动。

#### radix_linear_attention.py

- **作用**：线性注意力（hybrid GDN/KDA 模型）层封装，为 tc-piecewise graph 注册 split op；与 `RadixAttention` 同构，输出约定 4D `(1, seq, v_heads, v_dim)`。
- **关键内容**：`RadixLinearAttention`、`unified_linear_attention_with_output`、`_linear_attention_with_output_impl`、`bcg_unified_linear_attention_with_output`。
- **实现逻辑要点**：构造持有 q/k/v 头数与维度、GDN/KDA 共享卷积权重 `conv_weights`/`bias`、`activation`、`A_log`/`dt_bias`/`lower_bound`；forward 在 extend + tc-piecewise 时经 split op 输出形状 `(1, seq_len, num_v_heads, head_v_dim)`（BCG 下走 `bcg_` 包装）；非 PCG 但 DP 同步引入物理 padding 行时先对 real 前缀调 backend 再清零 padding 尾部；FlashInfer GDN backend 可直接写进 logical_output 免拷贝，其余 backend 走 copy 兜底。
- **备注**：被 GDN/KDA hybrid 模型使用。

#### flashinfer_comm_fusion.py

- **作用**：FlashInfer unified `allreduce_fusion`（trtllm/mnnvl 双后端）的封装、workspace 生命周期管理与安全兜底。
- **关键内容**：`FlashInferWorkspaceManager`（`initialize`/`is_buffer_size_sufficient`/`cleanup`）、`ensure_workspace_initialized`、`flashinfer_allreduce_residual_rmsnorm`（kARResidualRMSNorm pattern）、`flashinfer_allreduce`（kAllReduce）、`can_use_flashinfer_allreduce`、`resolve_flashinfer_allreduce_fusion_backend`、`_preflight_check_workspace_memory`、`_sync_allreduce_unavailable_across_tp`、`pre_initialize_workspaces`/`cleanup_flashinfer_workspace`、`_FixedTorchDistBackend`/`TorchDistributedCommBackend`。
- **实现逻辑要点**：
  1. 后端矩阵：trtllm 限单机 SM90/SM100，mnnvl 限 Blackwell（SM90 仅单机），auto 按平台/多节点解析，非 SM90/SM10X 拒绝。
  2. workspace 由 `flashinfer.comm.create_allreduce_fusion_workspace` 创建，按 `use_attn_tp_group` 分 attn/moe 两组存于 `resources.buffers`，按高水位只增不减。
  3. 创建前 preflight：按 trtllm SymmDeviceMemory 尺寸公式（buffer/flag/lamport，2MiB 对齐 + signal pad + 分配粒度）先用 `cuMemCreate` 试分配，再在 cpu_group 上 `all_reduce(BAND)` 集体表决，防单 rank 失败使其余 rank 卡死在 handle 交换；`_sync_allreduce_unavailable_across_tp` 用 MAX 归约保证全 TP rank 一致禁用，避免 CUDA graph capture 期间状态不一致挂死。
  4. `flashinfer_allreduce_residual_rmsnorm` 是注册 custom op（`mutates_args` + fake_impl），返回 `(None, None)` 即回退；`can_use_flashinfer_allreduce` 把 rank-invariant 检查移出 op 供纯 Python 决策；`_FixedTorchDistBackend` 修复 FlashInfer bcast 的 global-rank bug，用 GLOO cpu_group 避开 capture 期 NCCL 集体操作。
- **备注**：被 `layernorm.py`、`communicator.py`、`base_runner.py`（`pre_initialize_workspaces`）引用。

#### flashinfer_mnnvl_cutedsl.py

- **作用**：FlashInfer MNNVL CuTe DSL AR 融合 workspace 的进程内访问层（Qwen3.5 用）。
- **关键内容**：`FlashInferMNNVLCuteDSLARFusion`（`moe_finalize_all_reduce_rms_norm`/`all_reduce_residual_rms_norm`/`supports`/`destroy`）、`get_flashinfer_mnnvl_cutedsl_ar_fusion`、`_WorkspaceSignature`、`_with_early_finalize_shared_load`。
- **实现逻辑要点**：一个 workspace 服务两种 pattern：`kMoEFinalizeARResidualRMSNorm`（把 MoE finalize——`expanded_idx_to_permuted_idx` 还原、expert weights 加权、gated shared expert 输出、residual 加法、RMSNorm——全部融进一次 all-reduce）与 `kARResidualRMSNorm`（attention 侧 AR+residual+norm）；构造前必须选定 PyTorch symmetric memory backend（NCCL）并在 mailbox barrier + `dist.barrier` 之后才发布，防 Lamport 阶段失步；非投机解码时用 `_with_early_finalize_shared_load` 把 shared expert 加载提前到 PDL 之前（投机时保持安全顺序）；workspace 按 `(hidden_size, top_k, rms_eps, weight_bias, device, process_group)` 签名缓存、`max_m` 可复用时取最小者，禁止在 CUDA graph capture 期间创建，实例数受 `SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES` 限制。
- **备注**：被 `models/qwen3_5.py` 与 `moe/qwen35_flashinfer_fusion.py` 使用。

#### k3_ar_fusion.py

- **作用**：Kimi-K3 的 MNNVL 融合 all-reduce 派发层，连接模型与 `kernels/ops/kimi_k3/all_reduce`（仅被 `models/kimi_k3.py` 使用）。
- **关键内容**：`_get_state`（`cache_once` 探测 CustomAllReduceV2 + multicast）、`enabled`、`symm_buffer`、`find_mc_ptr`/`get_mc_ptr`、`all_reduce`、`all_reduce_low_sm`、`all_reduce_norm`、`finalize_push_fits`/`finalize_all_reduce_push_norm`、`gemm_ag_up_fits`/`gemm_ag_up_proj`、`_State`/`_Buffer`。
- **实现逻辑要点**：SM100/SM103 且 CustomAllReduceV2 带 multicast 时自动启用（`SGLANG_K3_AR_FUSION` 可双向强制，symm-mem 或 EP a2a 下自动跳过）；消息 ≤512KB 且 ≤`max_push_size` 走 1shot multicast push，大消息走 NVLS 2shot pull（pull 要求输入是 `symm_buffer` 切片，原地归约 multicast 别名，否则回退）；持久化对称缓冲用 `_SymmetricMemory.empty_strided_p2p` + `rendezvous` 一次建好（每 rank 必须解析同一 buffer/offset）；`all_reduce_norm` 在 AR 内核内融合首 num_tokens 行的 RMSNorm；`finalize_all_reduce_push_norm` 把 MoE finalize 融进 push AR+norm；`gemm_ag_up_proj` 是列并行 GEMV + multicast all-gather + add3 单 kernel（限 TP8、MAX_TOKENS、push 槽容量）。
- **备注**：`NORM_DIM=3584` 与 csrc 的 kNormDim 硬约定。

#### k3_gemm_ar.py

- **作用**：K3 o_proj 的 GEMM+all-reduce 单 kernel 融合开关（`SGLANG_K3_GEMM_AR`）。
- **关键内容**：`_init`、`maybe_wrap_o_proj`。
- **实现逻辑要点**：环境变量开启且 2≤TP≤8、SM100+ 才启用；`maybe_wrap_o_proj` 在模型构建期把 `RowParallelLinear.forward` 替换为 `_gemm_ar_forward`：decode 形状、bf16 且 `mod.fits(x)` 时调单 kernel `mod.o_proj_gemm_ar(x, weight)`，否则回退原 forward；`mod.init`（comm region 分配）与 per-K JIT 编译都在 build 期完成，保证 CUDA graph capture 只看到 ready-to-launch 路径；权重需满足 shape 约束（K%128==0）。
- **备注**：仅 eager 模式；详见 `kernels/ops/kimi_k3/GEMM_AR_README.md`。

#### k3_sp_collective.py

- **作用**：K3 SP-MoE 的 MNNVL reduce-scatter / all-gather 派发，含与 attention-residual 的二次融合。
- **关键内容**：`_init_state`/`enabled`、`requires_symmetric_rs`、`get_o_proj_output_buffer`、`o_proj_output_rows`（ContextVar 上下文）、`register_o_proj_output`、`finish_padded_o_proj_output`、`reduce_scatter_res`、`reduce_scatter_attn_res`、`all_gather`、`attn_res_all_gather`。
- **实现逻辑要点**：仅 SM103 + TP4/TP8 + MegaMoE/DeepEP a2a + CustomAllReduceV2 multicast 且存在 checked-in tuning table 时启用，否则 NCCL（`SGLANG_K3_SP_COLLECTIVE` 可显式开关）；RS 两策略：push 直算 / pull 经 o_proj 输出的 NVLS multicast 别名原地归约（`register_o_proj_output` 把模型可见 view 关联到持久化对称 backing）；`reduce_scatter_attn_res`（`attn_res.attn_res_fused_pull_rs`）把 o_proj RS + 本地残差加法 + 注意力残差聚合（打分/softmax/混合/norm）熔成一 kernel，需 `SGLANG_K3_SP_ATTN_RES`；`all_gather` 两策略 push / direct（multicast 直写持久化对称缓冲）；`attn_res_all_gather`（`attn_res.attn_res_fused_direct_ag`）融合本地聚合 + direct multicast AG + 可选 bank 写。
- **备注**：被 `models/kimi_k3.py` 与 `attn_residual.py` 互相调用。

#### attn_residual.py

- **作用**：Kimi-K3 注意力残差流的「快照银行 + 聚合」（score rows → softmax → 加权和 → RMSNorm）。
- **关键内容**：`AttnResidual`（每次 forward 一个实例，含 `write`/`forward`/`forward_sp_all_gather`/`forward_sp_reduce_scatter`）、`aggregate_stream`（预 norm 聚合，供 dspark aux 采集）、`aggregate_stream_torch`（eager 参考实现）、`_aggregate`/`_aggregate_fast`/`_aggregate_fused`/`_aggregate_hip`/`_aggregate_fused_add`、Triton kernel `_score_kernel`/`_combine_kernel`、`get_cw`（缓存 norm_weight⊙proj_weight）。
- **实现逻辑要点**：银行 `block_residual` 形状 [T, NB, H]，PP 跨 rank 继承前一段快照；按硬件分派三档——SM100+（排除 SM12x）且 H=7168 走 TMA warp-specialized kernel `attn_res_fused_tma`（每 SM 一个持久 CTA、double-buffer chunk ring、输出 norm 融合、可内核内写 bank 行），ROCm 走单 kernel `attn_res_hip`（bank 驻留寄存器、残差加法与快照同 launch），其余走 `_score_kernel`+`_combine_kernel` 两段 Triton（每个 H-chunk CTA 冗余算 softmax 换取 H 并行）；`H % 1024 != 0` 回退 eager；SP-MoE 下 `forward_sp_all_gather`/`forward_sp_reduce_scatter` 委托 `k3_sp_collective` 融合通信与聚合。
- **备注**：仅被 `models/kimi_k3.py` 使用；NORM_DIM/打分公式与 `csrc/kimi_k3` 内核严格对应。

#### mova.py

- **作用**：Mixture-of-Value attention（MoVA）的推理原语：路由 top-k + routed value 投影（value 侧按专家混合，取代普通 attention 的输出投影）。
- **关键内容**：`mova_router_topk`、`routed_linear`/`_routed_linear_cuda_chunk`、`routed_linear_reference`（torch oracle）、`RoutedValueExperts(nn.Module)`；常量 `_ROUTED_LINEAR_CHUNK_SIZE = 64*1024`。
- **实现逻辑要点**：
  1. `mova_router_topk` 实现 xLLM 的「selection-only」router-bias 语义：bias 只影响选择，混合系数从无偏 score 里 gather；打分 fp32；CUDA 且（sigmoid 或无 bias）时复用 `moe/topk.py` 的 `fused_topk`（`renormalize = renormalize and top_k > 1`，原生 xLLM 的 top-1 保留原始概率）；结果乘 `scaling_factor` 返回。
  2. `routed_linear` 复用 fused-MoE 的第一段 GEMM：`try_get_optimal_moe_config` 用伪造的第二段形状 `(num_experts, input_size, output_size)` 描述目标输出（MoVA 没有第二段 GEMM），config 先 pop 掉 `USE_TMA`（普通两段式 runner 才构造 TMA 描述符）；再 `moe_align_block_size` 对齐；kernel 输出先 SiLU 再按 routing weights 加权求和；超过 64K token 分块执行防 OOM。
  3. 通过 `@register_custom_op(op_name="mova_routed_linear_cuda", fake_impl=...)` 注册，fake 实现只产出形状（CUDA graph / torch.compile 兼容）；非 CUDA 走 `routed_linear_reference`。
  4. `RoutedValueExperts` 是持久化的 TP 输出切分权重 `[experts, out//tp, hidden]`；`weight_loader` 同时支持整块 packed 权重（按输出维 narrow）与 per-expert `loaded_shard_id` 两种加载。
- **备注**：推理路径只支持 fp16/bf16 连续输入；供带 MoVA 注意力（multi-value 专家）的模型使用。

### 2.5 logits / 采样 / 池化

这 6 个文件构成解码采样管线的最后一环：model_runner 在 `forward_batch_generation` 拿到最后一层 hidden_states 后，先经 `LogitsProcessor.forward` 过 LM head 与 TP/DP 归约得到 `next_token_logits`，再由 `Sampler.forward` 做温度缩放、top-k/top-p/min-p 过滤与采样，并把输出 logprob 回填同一 `LogitsProcessorOutput` 结构；`logprob_processor.py` 负责 prefill 输入 logprob 与 decode 输出 logprob 的提取，`logsumexp.py` 是其 fast path 的底层 Triton kernel。`pooler.py`/`sparse_pooler.py` 不属生成路径，服务于 embedding/分类/奖励模型。

#### logits_processor.py

- **作用**：把模型最后一层 hidden_states 变成 next-token logits（及 prefill 输入 logprob 所需的 logits），hidden_states → logits 的最后一跳。
- **关键内容**：`LogitsProcessorOutput`（管线中枢结果结构：Part 1 `next_token_logits`/`hidden_states` 本文件填充；Part 2 `next_token_logprobs`/`next_token_top_logprobs_val/idx`/`sampling_mask_*` 由 sampler 填充；Part 3 输入 logprob 仅 prefill；Part 4 `full_logits`；Part 5 `customized_info`；Part 6 临时字段）；`LogitsMetadata`（`from_forward_batch` 聚合 forward_mode、capture_hidden_mode、extend logprob 标志与 DP attention 元数据）；`LogitsProcessor` 主类（`forward` 与 `_get_pruned_states`/`_get_logits`/`_compute_lm_head`/`_gather_dp_attn_hidden_states`/`_gather_attn_tp_logits`/`_tp_lm_head_all_to_all`/`_scatter_dp_attn_logits`/`_copy_logits_to_buffer` 等）；`should_apply_lm_head_quant_method`、`_reassemble_tp_lm_head_all_to_all_output`、`compute_logprobs_for_multi_item_scoring`。
- **实现逻辑要点**：
  1. `forward` 分支顺序：autotune dummy run（FlashInfer 调优时跳过 LM head，避免 DP attention 下 `[batch*dp_size, vocab]` all-gather OOM）→ `layernorm_sp.maybe_exit_gather` → MIS prefill-only 走 `compute_logprobs_for_multi_item_scoring` → DLLM extend 走 `_get_dllm_logits` → 常规路径。
  2. `_get_pruned_states` 三模式：decode 等取整行（或按 `draft_extend_select_index` 挑行）；extend 无输入 logprob 用 `cumsum(extend_seq_lens)-1` 只取每序列末 token；extend 有输入 logprob 时逐序列构造 pruned_states/sample_indices/input_logprob_indices/token_to_seq_idx 四组索引，索引用 pinned memory + `non_blocking=True` 上传。
  3. `_get_logits` 顺序：DP gather（`dp_gather_replicate`）→ `_compute_lm_head`（LoRA 包装 / ModelOpt fp4·fp8 量化 `apply` / 普通 matmul / fp32 LM head 用 `torch.mm(out_dtype=float32)` / Intel AMX `weight_packed_linear` / GGUF）→ `logit_scale` 缩放 → TP 归约三选一（attn_tp all-gather、`all_to_all_single` + 输出重排、`MultimemAllGatherer`）→ DP scatter → 写共享 buffer → softcapping（CUDA 走 `fused_softcap` kernel）。
  4. `_copy_logits_to_buffer` 是 CUDA-graph 友好点：logits 拷入按 (rows, vocab) 复用的静态 buffer。
- **备注**：`should_apply_lm_head_quant_method` 在文件内定义两次，后一个版本（含 `quant_mode == "w4a16"` 分支）覆盖前者生效。

#### logprob_processor.py

- **作用**：实现 prefill 输入 logprob 与 decode 输出 logprob 的计算/提取（top-k logprob、指定 token id logprob、采样 token 的 logprob）。
- **关键内容**：`LogprobStage`（PREFILL/DECODE）、`LogprobResult`（`write_input_to`/`write_output_to` 回写 `LogitsProcessorOutput`）、`compute_row_log_normalizer`、`get_top_logprobs_raw`/`get_token_ids_logprobs_raw`（及 chunk 版）、`compute_spec_logprobs`、`InputLogprobProcessor`、`get_token_ids_logprobs_batch_optimized`、`OutputLogprobProcessor`。
- **实现逻辑要点**：
  1. `InputLogprobProcessor._forward_by_chunk` 逐 chunk：`lm_head.set_lm_head_pass(i)` 切换 LoRA batch_info → 注入的 `get_logits_fn` 算 chunk logits → 回填 `sampled_logits` → 提取三类 logprob；结尾 `reset_lm_head_pass()`。
  2. Fast path（`SGLANG_ENABLE_FAST_INPUT_LOGPROBS`，deterministic inference 下自动禁用）不做全词表 log_softmax，只算每行 normalizer；`max_k <= FUSED_TOPK_MAX_K` 时用 `row_logsumexp_topk` 融合 kernel 一趟同时拿 normalizer 与 top-k，否则退到 `compute_row_log_normalizer`。
  3. 内存优化：未启用 fast path 时先 `del chunk_logits` 再 out-of-place `log_softmax`，避免 raw logits + bf16 + fp32 三份的 3x 峰值 OOM。
  4. `OutputLogprobProcessor.compute_logprobs`：先 `clamp_(min=finfo.min)` 防 -inf，再 top-k、指定 token id、`logprobs[arange, batch_next_token_ids]` gather 采样 token 的 logprob。
  5. `compute_spec_logprobs` 按 accept_index 或 chain_stride 布局处理投机解码被接受 token 的 logprob。
- **备注**：所有 topk/gather 结果 `no_copy_to_cpu=True` 保留 GPU tensor，由下游延迟转 CPU。

#### logsumexp.py

- **作用**：单趟在线行 logsumexp 的 Triton kernel（可融合小 top-k），输出每行 (max, logsumexp-max) 的 fp32 对，是 fast input logprob 路径的底层算子。
- **关键内容**：`_accumulate_block_lse`/`_combine_lse`、`_row_logsumexp_kernel`、`row_logsumexp`；融合 top-k 版：`FUSED_TOPK_MAX_K = 8`、`_fpval_to_key`/`_key_to_fpval`、`_pack_key`/`_unpack_val`/`_unpack_idx`（值+下标打包 int64 key）、`_row_logsumexp_topk_kernel`、`row_logsumexp_topk`。
- **实现逻辑要点**：
  1. 每行一个 program 单趟扫描；`num_cols` 以 int 而非 constexpr 传入避免 Triton 展开循环；BLOCK_N 取 `next_power_of_2(min(num_cols, 16384))`，8 warps。
  2. 输出 (max, log_sum_exp−max) 分离保存，保证调用方 `(logit − max) − log_sum` 与 log-softmax 同阶平移不变，避免大公共偏移舍掉小 log_sum 项。
  3. top-k 版把 fp32 值 + int32 下标打包成排序编码 int64 key（高 31 位放值、低 31 位放 `2147483647 − idx`），寄存器内维护 K_PAD（≥2 的 2 幂）个降序 slot，空槽用 (−inf, 2147483647) 哨兵；块内按 `[K_PAD, SEG]` 重塑 + minor-axis 归约浮出每段最佳候选；非首块且块 max 不超过运行 k-th 值时直接跳过。
  4. `row_logsumexp_topk` 断言 `1 <= k <= FUSED_TOPK_MAX_K`；k 是 `tl.constexpr`，不同 k 编译独立实例；与 `row_logsumexp` 的 (max, log_sum) 逐位一致。
- **备注**：调优注释记录 GB300 上 vocab=151936 时融合 kernel 在 k≈6 内优于分离实现，k≈32 时低于纯 `torch.topk`。

#### sampler.py

- **作用**：从 `LogitsProcessorOutput.next_token_logits` 采样出 next token ids，并计算输出侧 logprob（top-k、指定 token id、采样 mask），回填同一结构。
- **关键内容**：`Sampler` 主类（`forward` 与 `_preprocess_logits`/`_sample_from_probs`/`_sample_from_logprobs`/`_sample_from_logits`/`_sync_token_ids_across_tp`/`_attach_sampling_mask_to_output`/`compute_logprobs_only`）；`create_sampler`/`register_sampler_backend`（`_BUILT_IN_SAMPLING_BACKENDS = {"flashinfer", "pytorch", "ascend"}`）、`top_k_top_p_min_p_sampling_from_probs_torch`、`multinomial_with_seed`、`apply_custom_logit_processor`；环境变量 `SYNC_TOKEN_IDS_ACROSS_TP`、`SGLANG_RETURN_ORIGINAL_LOGPROB`。
- **实现逻辑要点**：
  1. `forward` 流水线：`_preprocess_logits`（`apply_custom_logit_processor` 就地修改 + `sanitize_nan_logits`）→ `is_all_greedy` 快路径（ROCm 可选 aiter `greedy_sample`，否则 `torch.argmax`，temperature 免算）→ 否则按 `simple_sampling_case`（无 top-k/top-p/min-p）分流。
  2. 简单情形 `sampling_from_probs_torch`：默认 `torch.multinomial`；`SGLANG_OPT_USE_GUMBEL_SAMPLE` 下用指数分布 Gumbel-max 技巧，避开 multinomial 的 CPU 侧 philox bookkeeping 与合法性断言。
  3. 复杂情形按 backend：FlashInfer 的 min-p 走 `top_k_renorm_prob` → `top_p_renorm_prob` → `min_p_sampling_from_probs`，否则一次 joint 过滤；pytorch backend 退化为排序 + cumsum + 掩码的 torch 实现；Ascend 直接从 logits 采样（softmax 在 kernel 内，优先 `torch_npu.npu_top_k_top_p`）。
  4. RL on-policy：`use_log_softmax_logprob=True` 时先 `log_softmax(logits/temperature)` 再从 logprobs 采样；deterministic inference 要求 logprob 必须来自 `F.log_softmax`（与 prefill rescore 同 kernel），不能用 `log(softmax(x))`（二者在 ~1e-6 处不一致，破坏逐位对齐）。
  5. 确定性采样 `multinomial_with_seed`：`murmur_hash32(seed, positions, col_indices)` 生成每行独立随机源，Gumbel 噪声全程 float64（两端 clamp 保证 x==1 时有限），带 `@torch.compile(dynamic=True)`。
  6. TP 一致性：`_sync_token_ids_across_tp` 默认不同步（省一次 all-reduce，依赖各环节确定性），使用 grammar 时对 token ids 做 `all_reduce(MIN)`；采样时同步捕获截断/重归一化后的支撑集与选中权重，`_attach_sampling_mask_to_output` 校验采样 token 落在支撑集内并写回 mask idx 与 `log(selected_weight/support_mass)`。
- **备注**：CUDA graph 兼容性体现在全程避免动态分配与 CPU 同步：复用 logits buffer、原地运算、`non_blocking` H2D。

#### pooler.py

- **作用**：对 hidden_states 做池化（LAST/CLS/MEAN）并可选接打分头，服务 embedding/分类/奖励模型（qwen2_rm、llama_reward、roberta 等）。
- **关键内容**：`PoolingType`（IntEnum：LAST=0/CLS=1/MEAN=2）、`EmbeddingPoolerOutput`、`pool_hidden_states`、`pool_at_delimiter_positions`、`score_and_pool`、`Pooler`、`CrossEncodingPooler`。
- **实现逻辑要点**：LAST 用 `cumsum(extend_seq_lens)−1` 取每请求末 token；CLS 构造首 token flat 索引；MEAN 用前缀和相减除以 `prompt_lens`，全 batch 向量化无循环；MIS 路径在 CPU 组装 delimiter 索引后一次 `non_blocking=True` 传 GPU；`score_and_pool` 在 MIS prefill-only 时只在 delimiter 前位置打分省 head 计算；`Pooler.forward` 支持 per-request matryoshka 维度截断（维度不同时返回 tensor list）与 `normalize=True` L2 归一化；`CrossEncodingPooler` 整批一次性过 `classifier` + 激活函数再 squeeze(-1)。
- **备注**：文件头注明 adapted from vLLM `model_executor/layers/pooler.py`。

#### sparse_pooler.py

- **作用**：把 hidden states 池化为稀疏的词表空间 embedding，输出 `[batch_size, vocab_size]`。
- **关键内容**：`SparseEmbeddingOutput`、`SparsePooler(nn.Module)`（`forward`/`load_weights`）。
- **实现逻辑要点**：`sparse_linear`（hidden→1）+ ReLU 得每 token 权重 → `torch.repeat_interleave` 按 `extend_seq_lens` 构造 batch 索引 → `scatter_reduce_(reduce="amax")` 把权重按 (batch, input_ids) 散射进零初始化的 `[batch, vocab]`（重复 token 取 max 聚合）。
- **备注**：`__init__` 校验 config 必须含 vocab_size/hidden_size；`_weights_loaded` 强制先 `load_weights(state_dict)` 再 forward。

---

**顶层小结**：35 个顶层文件呈现清晰的分层——模型层直接接触的是 2.1 的线性/归一化/激活与 2.3 的通信编排、2.4 的注意力封装；2.2 的参数与工具文件为它们提供加载协议与平台支撑；2.5 构成采样管线末端。真正的算法多样性（attention 后端、MoE、量化）都在下一节的三级子目录里。

---

## 三、子目录详解（8 个目录，304 个文件）

> 逐文件条目按字母序跟随四个阅读批次组织；与顶层章节相比，这里更侧重「家族归属 + 职责 + 关键逻辑」，篇幅从简。目录顺序：`attention/` → `moe/` → `quantization/` → `cp/` → `dcp/` → `rotary_embedding/` → `deep_gemm_wrapper/` → `utils/`。

### 3.1 attention/（112 个文件）

**总览**：attention 目录的全部后端共享一套接口——`attention_registry.py` 的 `@register_attention_backend(name)` 把字符串注册进全局 `ATTENTION_BACKENDS` 工厂表，`base_attn_backend.py` 的 `AttentionBackend` ABC 规定 CUDA-graph 元数据契约（`init_forward_metadata_out_graph` / `_in_graph` 两段式、`needs_cpu_seq_lens` 等类属性），模型侧统一经顶层 `RadixAttention` 按 `ServerArgs.attention_backend` 拿到具体后端。文件可粗分为：通用 MHA/MLA 后端（flashinfer/flashattention/fa3/fa4/cutlass_mla 等）、AMD aiter 生态、DSV4/DSA 稀疏注意力家族（含 indexer/compressor/kpool 子模块）、hybrid 线性注意力家族（Mamba2/GDN/KDA/短卷积，`linear/` 子目录及多后端 kernel 矩阵）、mini-max/minicpm/mova/nsa 等模型特定后端、vision/decoder-only 共用的工具与 `minicpm` 系。以下按四个批次逐文件说明（字母序，批次边界为人工划分）。

#### attention/aiter_backend.py

- **作用**：AMD ROCm 端到端 attention 后端（注册名 `aiter`）。
- **关键内容**：`AiterAttnBackend`、`ForwardMetadata`、`WrapperDispatch`、`AiterIndicesUpdaterPrefill`/`AiterMlaIndicesUpdaterPrefill`、`AiterMultiStepDraftBackend`。
- **实现逻辑**：依赖 aiter 的 `mla_decode_fwd`/`mla_prefill_fwd`/`paged_attention_ragged`/`unified_attention` 等及 sglang Triton kernel；decode 按 MLA→`_forward_mla_decode`（优先 `mla_gluon_decode`，退化 zero-pad + `mla_decode_fwd`）、非 MLA→`paged_attention_ragged`/`unified_attention`/5D `pa_decode_gluon` 分路；fp8 KV 走 persist MLA 设计（`SGLANG_AITER_MLA_PERSIST`）。
- **备注**：`_MLA_REDUCE_V1_HEADS` 表从 aiter reduce.cu 抄录。

#### attention/aiter_mla_gluon.py

- **作用**：低 head 数 MLA（如 Kimi K3 TP8 每 GPU 12 heads）decode 的 aiter Gluon 封装。
- **关键内容**：`_gluon_fn()`（lru_cache 探测：env 开关 + 导入 + Triton≥3.7 `cga_layout`）、`mla_gluon_decode()`、`prefer_mla_gluon_decode()`。
- **实现逻辑**：要求 head_pad_mode=zero、num_head=12、fp8 KV cache；把 q 拆 nope/pe 后直接调 `mla_gluon` 写输出；不可用返回 None 由调用方回退。

#### attention/aiter_utils.py

- **作用**：SHUFFLE 5D KV 布局（`SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`）专用路径，独立成文件保持主后端聚焦 legacy NHD 路径。
- **关键内容**：`forward_extend_vectorized_5d`、`forward_decode_vectorized_5d`。
- **实现逻辑**：extend 分 fresh-prompt 捷径（直接吃 bf16 k/v）与 gather-and-linearize（逆 shuffle 聚合线性缓冲走 LINEAR 模式）两条子路；decode 用 `pa_decode_gluon`，full 层走 kv_indices + `get_recommended_splits`，SWA 层走 swa_page_table。

#### attention/attention_registry.py

- **作用**：所有 attention backend 的注册表与工厂（字符串名 → 构造函数），ModelRunner.init_attention_backend 调用。
- **关键内容**：`ATTENTION_BACKENDS` 字典 + `register_attention_backend` 装饰器；注册条目含 flashinfer/trtllm_mla/tokenspeed_mla/cutedsl_mla/aiter/wave/ascend/dsa/nsa（→dsa 弃用别名）/dsv4/triton/torch_native/flex_attention/flashmla/fa3/fa4/cutlass_mla/hpc_ops/intel_amx/dual_chunk_flash_attn/minicpm_*/intel_xpu 等。
- **实现逻辑**：`attn_backend_wrapper` 为 hybrid GDN/Mamba2/KDA/Lightning/Inkling/MiniMax-sparse/Dots3 等特殊模型包一层线性/稀疏后端；dsv4 按平台分发 NPU/HIP/CUDA 三实现；另有 draft-extend/decode 的包装函数。

#### attention/base_attn_backend.py

- **作用**：`AttentionBackend` 抽象基类，定义全部后端的接口契约。
- **关键内容**：`SharedReadEnds` 枚举（调度器共享数据读取结束时机）；三方法元数据契约 `init_forward_metadata` / `init_forward_metadata_out_graph`（捕获外 host 逻辑）/ `init_forward_metadata_in_graph`（可录图的静态 GPU op）；`forward()` 按 ForwardMode 分发。
- **实现逻辑**：类属性 `needs_cpu_seq_lens`、`supports_ragged_verify_graph`、`use_captured_forward_metadata_for_breakable_cuda_graph` 供 runner 全局决策；`verify_mask` property、`get_indexer_metadata` 默认 None。
- **备注**：legacy capture/replay CUDA graph 覆写已从 ABC 移除。

#### attention/cutedsl_mla_backend.py

- **作用**：flashinfer cute-dsl MLA decode 后端，专长 DCP（decode context parallel）。
- **关键内容**：`CuteDslMLABackend(TRTLLMMLABackend)`、`CuteDslMLAMultiStepDraftBackend`。
- **实现逻辑**：`_run_decode_kernel` 在 `cp_world>1` 时调 `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl", enable_dcp=True)` 返回 rank-local `(out, lse)`；零 KV 行用 `fixup_zero_kv_rows` 置中性值供跨 rank 合并。

#### attention/cutlass_mla_backend.py

- **作用**：Cutlass MLA decode 后端，固定 page_size=128。
- **关键内容**：`CutlassMLADecodeMetadata`、`CutlassMLABackend(FlashInferMLAAttnBackend)`。
- **实现逻辑**：`create_flashmla_kv_indices_triton` 构建页表 + `cutlass_mla_get_workspace_size` 分配 workspace；decode 拆 q_nope/q_rope 后调 sgl_kernel `cutlass_mla_decode`（num_kv_splits=1），prefill 复用 FlashInfer MLA 路径。

#### attention/deepseek_v4_backend.py

- **作用**：DeepSeek-V4 压缩注意力 CUDA 主后端（注册名 `dsv4`）。
- **关键内容**：`DSV4AttnMetadata` 等元数据、`_GraphBucket`、`DeepseekV4AttnBackend`、`DeepseekV4MultiStepBackend`；常量 `SWA_WINDOW=128, C4_TOPK=512`。
- **实现逻辑**：组装 SWA 环形窗口 + c4/c128 压缩 extra_k_cache + attn_sink，调 `flash_mla_with_kvcache`（SM120 走 `_sm120` 变体）；大 query 或 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` 走 `flash_mla_sparse_fwd`，fp8 走 q8kv8 变体；组合 `C4IndexerBackendMixin` + `CompressorBackendMixin`（v2），支持 DSpark 草稿。
- **备注**：k 与 v 共享（`assert k is v`）。

#### attention/deepseek_v4_backend_hip_radix.py

- **作用**：DSV4 的 HIP/ROCm 后端，基于 radix 感知的 SWA + unified_kv 池（bf16）。
- **关键内容**：`UnifiedKvMetadata`、`DeepseekV4HipRadixBackend(AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin)`、`DeepseekV4MultiStepBackend`。
- **实现逻辑**：硬编码 page_size=256、swa_page_size=128；`_forward_unified_kv` 调 unified_kv_kernels 的 `store_swa_into_unified`/`decode`，按 compress_ratio 选 swa/hca/csa 索引流；`tbo_supports_cuda_graph=False` 防 TBO 泄漏 ROCm HSA 资源。

#### attention/dots_hybrid_backend.py

- **作用**：dots.note.omni 模型的逐层 DSA/SWA 混合分发。
- **关键内容**：`DotsSWAMLAAttnBackend`、`DotsHybridAttnBackend`、`DotsSWAMLAPrefillMetadata`。
- **实现逻辑**：`backend_for_layer` 按 `layer.sliding_window_size` 选后端；SWA 尾窗经 `translate_loc_from_full_to_swa` 翻译；`forward_swa_mla_expanded` 用 `flash_attn_varlen_func`、absorbed 版走 page64 latent cache 的 torch fallback。

#### attention/dsa/（12 个文件：DSA 子模块）

以下为 `dsa/` 子目录文件（indexer 与其元数据、topk、kpool 压缩池、CUDA graph、NPU 路径）；主后端 `dsa_backend.py` 见其后条目。

#### attention/dsa/dsa_backend_kpool.py

- **作用**：DSA 后端的 kpool（压缩池）元数据与 tail 处理 mixin（被 `DeepseekSparseAttnBackend` 多继承）。
- **实现逻辑**：`_init_kpool_metadata` 按 forward mode 分派（extend→`init_kpool_extend_metadata`，decode→pooled-paged MQA + 写计划）；`_update_kpool_metadata_replay` 在 CUDA graph replay 时原地刷新；`_resolve_kpool_tail_backend` 把带 tail 的 `flashmla_sparse` 升级为 trtllm(SM≥10)/fa3(SM90)。

#### attention/dsa/dsa_backend_mtp_precompute.py

- **作用**：多步投机解码的元数据预计算（算一次、拷 N 份）。
- **关键内容**：`PrecomputedMetadata`、`DeepseekSparseAttnBackendMTPPrecomputeMixin`、`compute_cu_seqlens`。
- **实现逻辑**：decode 用融合 kernel `fused_dsa_decode_metadata` 一次产出全部元数据；target_verify 用 `fused_dsa_target_verify_metadata`（含 next_n 展开）；flashmla_kv 实现时顺带算 flashmla metadata。

#### attention/dsa/dsa_indexer.py

- **作用**：DSA lightning indexer 核心实现（Q/K 投影、FP8 量化、MQA logits、topk）。
- **关键内容**：`Indexer(DSANPUIndexerMixin, BaseFusedOp)`；`rotate_activation`（Hadamard 旋转）、`broadcast_indexer_topk_from_rank0_`。
- **实现逻辑**：`_get_topk_paged`/`_get_topk_ragged` 两路 topk，MQA logits 按平台选 deep_gemm/aiter gluon/cutedsl/tilelang/XPU；`_should_chunk_mqa_logits` 按显存预算分块防 OOM；`forward_cuda` 处理 PCG/BCG split-op 表面、dual-stream（token≤1024）、k-only 快路径与 rank0 广播。

#### attention/dsa/dsa_indexer_kpool.py

- **作用**：带 kpool 压缩池的 DSA indexer（DeepSeek-V3.2 系配置）。
- **关键内容**：`IndexerKPool(MultiPlatformOp)`。
- **实现逻辑**：构造断言 `index_kpool>1`、`index_topk % index_kpool == 0`、`64 % index_kpool == 0`；`_compress_write_*` 把 KV 按池压缩写入；`_topk_from_kpool_logits`/`_get_topk_ragged_kpool` 从池化历史 logits 取 topk 并 append tail token。

#### attention/dsa/dsa_indexer_metadata.py

- **作用**：indexer 与 attention backend 之间的元数据抽象层（模型层只依赖此接口，与后端解耦）。
- **关键内容**：`BaseIndexerMetadata`（`get_seqlens_int32`/`get_page_table_64`/`get_page_table_1`/`get_seqlens_expanded`/`topk_transform`）、`DSAIndexerMetadata`。

#### attention/dsa/dsa_npu_indexer.py

- **作用**：NPU（Ascend）平台 indexer mixin。
- **实现逻辑**：`torch_npu.npu_rotary_mul` 做 RoPE、`npu_lightning_indexer`（layout TND/PA_BSND, sparse_mode=3）出 topk；prefill CP 时先 all-gather 全量 K 再切分 query 平衡计算；支持多流与事件同步。

#### attention/dsa/dsa_prefill_cuda_graph.py

- **作用**：DSA indexer 在 piecewise/breakable CUDA graph 下的 split-op。
- **关键内容**：`pcg_dsa_indexer_prefill_split`（custom op + `register_split_op`）、`bcg_dsa_indexer_prefill_split`、`scale_head_gate_graph`/`logits_head_gate_graph`。
- **实现逻辑**：整条 indexer（q/k 投影、head gate、K cache 存储、topk）作为单个 eager split op，结果通过原地修改静态 padded `topk_result` 交付（图契约）；拒绝 LoRA 组合（`GRAPH_WEIGHTS_PROJ_LORA_ERROR`）。

#### attention/dsa/dsa_topk_backend.py

- **作用**：DSA topk 选择与索引变换的后端抽象。
- **关键内容**：`TopkTransformMethod`（PAGED/RAGGED）、`DSATopKBackend`（SGL_KERNEL/TORCH/FLASHINFER，`resolve()` 区分 target/draft worker）。
- **实现逻辑**：topk 底层为 `sgl_kernel.fast_topk_v2`/torch.topk/flashinfer.top_k；`SGLANG_DSA_FUSE_TOPK` 下 decode 形 PAGED/RAGGED 各路由到 DSV4 topk v2 JIT kernel（限定 0<topk≤2048，要求 fp32 score、单位行步长、16B 对齐）。

#### attention/dsa/kpool_fp8_index.py

- **作用**：kpool 压缩池的 Triton 内核集合（FP8 index 域；与 kpool_plan.py 分工：此文件是内核，彼是计划编排）。
- **关键内容**：`build_pooled_page_table_64`、`topk_from_pooled_history_logits`、`kpool_softmax_rotate_write_cache`（Hadamard 旋转 + softmax 池化写缓存，`_hadamard128` 手工展开）、`kpool_write_tail_and_maybe_compress`、`all_gather_and_scatter_pool_slots`（CP 跨 rank 收集池 slot）等。

#### attention/dsa/kpool_plan.py

- **作用**：kpool 压缩池的写入计划构建与调度编排。
- **关键内容**：`PoolWriteRows`/`TailWriteRows`/`KPoolCpInfo`/`KPoolExtendPlan`/`KPoolWritePlan`。
- **实现逻辑**：CPU 上把每个 request 的 KV 分解为「完整压缩池行 + 未闭合 tail 行」再传 GPU；`init_pooled_paged_mqa_metadata` 管理 DeepGEMM schedule；写计划服务 CUDA graph 捕获与 replay 原地刷新。

#### attention/dsa/paged_mqa_logits_backend.py

- **作用**：paged MQA logits 计算后端选择枚举（极小文件，对应 server arg `--dsa-paged-mqa-logits-backend`）。
- **实现逻辑**：`DSAPagedMQALogitsBackend(Enum)`：DEEPGEMM/CUTEDSL/AITER；ROCm 只允许 aiter，cutedsl 要求 SM100，auto→deepgemm。

#### attention/dsa/utils.py

- **作用**：DSA 体系共享工具函数。
- **关键内容**：`aiter_can_use_preshuffle_paged_mqa`（探测决定 page_size 64 vs 1 与 preshuffle 开关）、`compute_dsa_seqlens`、`is_dsa_enable_prefill_cp`/`dsa_cp_round_robin_split_data`、`cal_padded_tokens`/`pad_dsa_cache_seqlens`、`should_use_dsa_fused_topk`、`fp8_mqa_logits_make_fused_kv`。

#### attention/dsa_backend.py

- **作用**：DeepSeek Native Sparse Attention（DSA）主后端（注册名 `dsa`，`nsa` 为弃用别名）。
- **关键内容**：`DSAMetadata`、`DeepseekSparseAttnBackend(KPoolMixin, MTPPrecomputeMixin, AttentionBackend)`、`DeepseekSparseAttnMultiStepBackend`。
- **实现逻辑**：按 `dsa_impl`（flashmla_sparse/q8、flashmla_kv、flashinfer_sparse_mla、fa3、tilelang、trtllm、aiter、intel_xpu 等）分派 `_forward_*`；decode 用 `transform_index_page_table_decode` 把 topk_indices 转稀疏页表；`q8kv8_*` 管理 born-fp8 Q 缓冲；`get_indexer_metadata` 返回 `DSAIndexerMetadata`；`needs_cpu_seq_lens=False` 避免 D2H。

#### attention/dsv4/__init__.py

- 空文件，仅作 Python 包标记。

#### attention/dsv4/compressor.py

- **作用**：DSV4 压缩器核心模块与后端 mixin（legacy 版）。
- **关键内容**：`FusedCompressMetadata`、`CompressorBackendMixin`（压缩结果经 `set_extra_key_buffer_fused`/`set_index_k_fused` 写入池）、`Compressor(BaseFusedOp)`。
- **实现逻辑**：模块持有 `ape`（ratio × coff×head_dim，overlap 时 coff=2）+ `wkv_gate` + RMSNorm；`make_compressor_plan`/`create_paged_compressor_data` 生成写位置计划（c4 用 req_to_token 定位、c128 用 req_slot×ring）；`prelaunch_kv_score` 在 HIP CP 下预取 kv_score 隐藏投影延迟。
- **备注**：v2 版（compressor_v2.py）才是 CUDA 后端现在使用的 mixin。

#### attention/dsv4/compressor_v2.py

- **作用**：compressor 的 v2 实现（在线压缩 + 统一 forward；`deepseek_v4_backend.py` 使用此版）。
- **实现逻辑**：`_use_online_compress` 仅 c128 且 `SGLANG_OPT_USE_ONLINE_COMPRESS` 开启；`_forward_compress_all_in_one` 单 kernel 完成 norm+rope+compress 全融合；`_compress_forward_c128_fallback` 为 HIP wave64 提供 PyTorch 回退；`_extract_positions_from_plan` 从 plan 前 4 字节取 seq_len 反推 RoPE 位置。

#### attention/dsv4/compress_hip.py

- **作用**：HIP/ROCm 专用 Compressor 实现。
- **关键内容**：`CompressorHip(_CompressorBase)`、`DeepseekRefRMSNorm`。
- **实现逻辑**：`compress_extend_paged`/`compress_decode_paged` 用 `fused_norm_rope_inplace_triton` + `fused_softmax_pool_triton`；`compress_fused` 融合 norm+rope+压缩单 kernel；`overlap_transform` 处理 ratio=4 的 overlap 布局。

#### attention/dsv4/indexer.py

- **作用**：DSV4 的 C4 indexer 实现与后端 mixin。
- **关键内容**：`C4IndexerBackendMixin`（`forward_c4_indexer` 主流程 + `_can_use_nonpaged_indexer`/`_get_nonpaged_indexer_plan` 非 paged 快路径）、`C4Indexer(nn.Module)`（wq_b + weights_proj + ratio=4 时的 Compressor）。
- **实现逻辑**：`compute_q` 在 fp4 indexer 时走 `fused_q_indexer_rope_hadamard_fp4_quant`，否则 `fused_q_indexer_rope_hadamard_quant`；nonpaged 快路径仅限 CUDA、batch=1、EXTEND、无 CP/图捕获。

#### attention/dsv4/metadata.py

- **作用**：定义 DSV4 稀疏注意力后端（paged indexer）的元数据结构与复制工具，是元数据约定源头（模块 docstring 解释 topk_lengths/page_table/c4_/c128_ 等术语）。
- **关键内容**：`copy_metadata()`（check_eq/copy/assign 三类泛型字段复制）、`PagedIndexerMetadata`（page_table、c4_seq_lens、use_topk_v2、deep_gemm_metadata、topk_metadata、nonpaged_plan）、`NonPagedIndexerPlan`、`maybe_copy_inplace()`。
- **实现逻辑**：`__post_init__` 按平台（HIP/XPU/FP8 torch 路径）决定是否构建 deep_gemm metadata，SM120 且 query 超阈值时分 chunk 构建；use_topk_v2 调 `plan_topk_v2`；硬断言 page_size=256；`copy_()` 用于 CUDA graph replay 原地刷新。

#### attention/dsv4/sparse_prefill_utils.py

- **作用**：DSV4 FlashMLA 稀疏 prefill 的 per-query topk+SWA 索引合并工具（适配 vLLM `combine_topk_swa_indices`）。
- **关键内容**：`SparsePrefillWorkspace`（后端持有的 KV 反量化 scratch）、`combine_topk_swa_indices()`（Triton kernel 拼行并 pad 到 128 对齐）、`build_swa_token_ids()`、`SparsePrefillChunkCache`（chunk 级不变量的 c0/c128/c4 combined indices 与 bases 预计算）、`use_dsv4_q8kv8_sparse_prefill()`。
- **实现逻辑**：workspace 宽 512 = 448 dequant nope + 64 rope；SWA-only 层传 topk=0；chunk 内跨 61 层复用预分配 buffer。
- **备注**：`SGLANG_DSV4_Q8KV8_PREFILL` 为运行时硬化期 debug 覆盖。

#### attention/dual_chunk_flashattention_backend.py

- **作用**：DeepSeek Dual Chunk Attention（DCA）+ vertical/slash 稀疏注意力的 FlashAttention backend。
- **关键内容**：`DualChunkFlashAttentionMetadata`（intra/succ/inter 三段 seq_lens 与 block tables）、`DualChunkFlashAttentionBackend`、`_dual_chunk_flash_attn_decoding`、`_dual_chunk_flash_attn_prefill`（支持稀疏 vertical_and_slash 模式）。
- **实现逻辑**：Q 沿最后一维拆成 5 份（main/succ/inter/succ_critical/inter_critical）；decode 三段输出用 `return_softmax_lse` 的 exp-sum 归一化在线合并；DCA 缩放 `0.1*log(len/max_pos)+1` 乘在 K（prefill）或 Q（decode）；序列超 `sparse_attention_threshold`（默认 32768）才启用稀疏。

#### attention/flashattention_backend.py

- **作用**：SGLang 主力 FlashAttention backend，支持 FA3/FA4 两代与 MHA/MLA/本地窗口/CP/投机解码全场景。
- **关键内容**：`FlashAttentionMetadata`（含 `LocalAttentionMetadata`）、`FlashAttentionBackend`、`FlashAttentionMultiStepBackend`、`make_local_attention_virtual_batches`、`merge_state_v2_wrapper`。
- **实现逻辑**：`fa_impl_ver` 选 FA3（sgl_kernel.flash_attn，scheduler_metadata 预计算）或 FA4（SM120 特化/通用 cute 实现）；KV 写支持 MLA `set_mla_kv_buffer`、CP 全量物化、mxfp8 独立 descale；topk>1 target verify 走 cascade attention（两段 + merge_state）；local attention 拆虚拟 batch 项；`needs_cpu_seq_lens=False`。
- **备注**：embedding 模式有 `fa_skip_kv_cache` 直通；rel_bias（Inkling sheared bias）仅 FA4。

#### attention/flashinfer_backend.py

- **作用**：FlashInfer paged/ragged attention backend，decode 与 extend 主力实现之一。
- **关键内容**：`FlashInferAttnBackend`、`FlashInferIndicesUpdaterDecode/Prefill`（含 SWA、cross-attention、DCP、spec decode 分支）、`DecodeMetadata`/`PrefillMetadata`、`fast_prefill_plan()`（免 D2H 的 graph 重放 plan）、`FlashInferMultiStepDraftBackend`。
- **实现逻辑**：按 sliding-window/cross-attention 决定 num_wrappers（1 或 2）并按层分发；decode 用 paged wrapper，extend 无 prefix 时走 ragged wrapper（与 paged 结果 `_safe_merge_state` 合并，head 过大回退 Triton merge_state）；支持 FP4/nvfp4 dequant workspace、multi-item scoring、DLLM；Qwen 系自动调大 workspace（512MB）。
- **备注**：`fast_prefill_plan` 专门消除 EAGLE draft-extend graph 重放时三次阻塞 `.to("cpu")`。

#### attention/flashinfer_mla_backend.py

- **作用**：FlashInfer MLA 后端，为 DeepSeek 类 MLA 模型提供 paged MLA prefill/decode wrapper。
- **关键内容**：`FlashInferMhaChunkKVRunner`（MLA prefix-chunk 缓存 MHA 计算）、`FlashInferMLAAttnBackend`、`FlashInferMLAIndicesUpdaterDecode/Prefill`、`FlashInferMLAMultiStepDraftBackend`。
- **实现逻辑**：decode 用 `BatchMLAPagedAttentionWrapper`（q 拆 q_nope/q_rope，k_cache 单 buffer 存 nope+rope）；extend 默认无 prefix 走 ragged、否则 paged（tc_piecewise/breakable graph 下强制 paged）；SM100 用 cutlass fmha；`fast_mla_decode_plan`/`fast_mla_prefill_plan` 提供免同步重放 plan。
- **备注**：支持 DCP 时 decode 返回 lse 供在线 softmax。

#### attention/flashmla_backend.py

- **作用**：基于 sgl-kernel FlashMLA 稀疏 decode kernel 的 MLA 后端（继承 FlashInferMLA，prefill 仍走父类）。
- **关键内容**：`FlashMLADecodeMetadata`（flashmla_metadata、num_splits、block_kv_indices、seq_lens_k）、`FlashMLABackend`、`FlashMLAMultiStepDraftBackend`（topk=1 限制）。
- **实现逻辑**：decode/verify/draft_extend_v2 元数据全在设备端构建（`create_flashmla_kv_indices_triton` 填 block table、`get_mla_metadata` 生成 tile scheduler 元数据），`needs_cpu_seq_lens=False`；decode 支持 FP8 KV cache（`scaled_fp8_quant` + descale）；DCP 时返回 (out, lse)。

#### attention/hip_flash_mla.py

- **作用**：AMD HIP 平台 FlashMLA 入口与 torch 参考实现（测试/对照用）。
- **关键内容**：`flash_mla_with_kvcache_entrypoint(backend, ...)`（按 backend 字符串分发 torch/tilelang/triton/kernel/comparison）、`flash_mla_with_kvcache_torch()`、`_assert_close()`。
- **实现逻辑**：HIP 上 backend 由 `SGLANG_HACK_FLASHMLA_BACKEND` 强制指定；comparison 模式并行跑参考与 kernel 实现并断言一致。

#### attention/hpc_ops_backend.py

- **作用**：腾讯 HunYuan AI Infra 的 HPC-Ops 算子库 attention backend（SM90/H20 优化）。
- **关键内容**：`HPCOpsMetadata`、`HPCOpsAttnBackend`、`hpc_ops_fp8_rope_store_kv`（graph-splitting 包装的融合 QKNorm+RoPE+FP8 quant+StoreKV op）。
- **实现逻辑**：直接读 SGLang token-major KV pool 为零拷贝 NHD 视图；硬约束 SM90、page-size 64、bf16/fp8 KV、head_dim 128、GQA ∈ {4,8}；decode 用动态调度 task map 平衡长尾 seq（min_process_len 512）。
- **备注**：不支持滑窗/logit cap/投机解码；RMSNorm eps 硬编码 1e-6。

#### attention/hybrid_attn_backend.py

- **作用**：prefill 与 decode 使用不同后端的组合 backend（如 prefill=trtllm_mla、decode=flashmla）。
- **关键内容**：`HybridAttnBackend(model_runner, prefill_backend, decode_backend)`、`_select_backend(forward_mode)`。
- **实现逻辑**：`needs_cpu_seq_lens` 取 decode 与（spec 走 prefill 时）prefill 的 OR，避免无谓 D2H；`init_mha_chunk_metadata` 委托 prefill 侧（修复 MLA MHA ragged 计划陈旧 bug）。
- **备注**：该文件 forward 定义出现两次，实际生效的是签名较简的版本。

#### attention/hybrid_linear_attn_backend.py

- **作用**：full attention + 线性注意力（Mamba2/GDN/KDA/短卷积）的组合 backend 与 mamba 元数据基座。
- **关键内容**：`MambaAttnBackendBase`（track indices、ReplaySSM 游标、`update_mamba_state_after_mtp_verify` 三提交路径）、`Mamba2AttnBackend`、`HybridLinearAttnBackend`（按 layer_id 分发）、`ShortConvHybridAttnBackend`（ZAYA1 CCA/LFM2）。
- **实现逻辑**：mamba slot 统一做 virtual→physical translate，padding 行置 -1；extend 按 chunk 对齐与否选 track 最终状态或中间 h 快照（radix 前缀缓存）；CUDA graph 用 per-bs 静态 buffer + replay 原地刷新；ReplaySSM 写游标 decode 时快照推进（GDN 支持 radix 边界 force-flush，KDA 只自然回绕）。

#### attention/index_topk_share.py

- **作用**：DSV4/DSA 稀疏注意力在 MTP 多轮迭代间共享 topk_indices 的状态机（投机解码索引复用基础设施）。
- **关键内容**：`IndexTopKShareState`（from_mtp_carry/enabled/should_publish/topk_indices/update/publish + `mtp_iteration` 上下文管理器）。
- **实现逻辑**：`reuse_dsa_topk_indices` 开启时从 `spec_info.dsa_topk_indices` 继承上一轮索引；CP v2 下 update 时对 topk_indices 做 `cp_gather_after_forward`；迭代结束清空携带、恢复标志位。

#### attention/intel_amx_backend.py

- **作用**：Intel CPU（AMX）上的注意力 backend，走 sgl_kernel CPU 算子。
- **关键内容**：`IntelAMXAttnBackend`（调 `extend_attention_cpu`/`decode_attention_cpu`）、`IntelAMXMultiStepDraftBackend`。
- **实现逻辑**：attn_logits 缓冲 `[bs, num_head, 8, v_head_dim+1]` 复用；TARGET_VERIFY 时从 spec_info 派生 extend 元数据，tree_topk!=1 才传 tree_mask（EAGLE/NGRAM）；decode 中 FP8 KV 先写池再交 kernel 内核写。
- **备注**：支持 encoder/cross attention；`support_triton()` 返回 False。

#### attention/linear/（线性注意力后端与 `kernels/` 多后端矩阵）

`linear/` 子目录共 17 个文件：`__init__.py`（空）、`gdn_backend.py`、`kda_backend.py`、`inkling_sconv_backend.py`、`lightning_backend.py`、`linear_metadata.py`、`short_conv_backend.py`、`utils.py` + `kernels/`（12 个：`__init__.py` 空 + `kernel_backend.py` 抽象基类 + 10 个具体内核封装）。

#### attention/linear/__init__.py

- 空文件，仅作包标记。

#### attention/linear/gdn_backend.py

- **作用**：GDN（Gated Delta Network）线性注意力后端：kernel 分发 + conv1d + 层实现。
- **关键内容**：`GDNKernelDispatcher`（decode/prefill/verify 三向选择 triton/cutedsl/flashinfer）、`GDNAttnBackend`、`fused_qkvzba_*` 融合 decode 路径、`flashinfer_gdn_prefill_default()`。
- **实现逻辑**：decode 优先 `packed_decode`（conv 输出直进融合 kernel），ReplaySSM 环形缓存启用时改走 replayssm decode；extend 走 `fused_gdn_gating` 得 g/beta 再进 chunk kernel，page-major 池先 gather 再 scatter 回；verify 分 ReplaySSM fold/spec 重放/recurrent 三种协议，FlashInfer verify 仅线性链（树状回退 Triton）。

#### attention/linear/inkling_sconv_backend.py

- **作用**：Inkling 模型的短卷积状态 sidecar backend（每层 4 个短卷积流共享一步元数据）。
- **关键内容**：`InklingShortConvMetadata`、`InklingShortConvAttnBackend`、`InklingShortConvHybridAttnBackend`（full attention + sidecar，无线性层）。
- **实现逻辑**：元数据拆 in-graph（decode/target_verify/draft_extend 录制）与 out-of-graph（full-CG prefill 与统一池 translate）两段；`_alloc_graph_buffers` 按 capture 形状分配图静态缓冲；MTP verify 后按流提交最后接受步的 conv 状态。

#### attention/linear/kda_backend.py

- **作用**：KDA（Kimi Delta Attention，Kimi-K2/K3）线性注意力后端：kernel 分发 + 层实现 + 深度融合路径。
- **关键内容**：`KDAKernelDispatcher`（decode：triton/helion/cutedsl/flashinfer；verify：triton/flashinfer/nv_cutedsl；prefill：triton/helion/flashkda/cutedsl/ptx_kda/nvidia_kda）、`KDAAttnBackend`。
- **实现逻辑**：Kimi-K3 的 fused 参数移交触发 `kda_fused_decode`（conv+delta rule+门控 RMSNorm 单 kernel，含 aiter HIP 变体）；target_verify 支持 fused chain-verify、DSpARK CuTe MTP（SM100、draft 2-8）、ragged dense scatter 快路径；fused-accept 下 verify 内核按 `num_accepted_tokens` 自举跳过 SSM 提交 scatter。
- **备注**：KDA conv 状态池为转置布局 `[kernel-1, dim]`，与 GDN/Mamba2 相反。

#### attention/linear/kernels/__init__.py

- 空文件，仅作包标记。

#### attention/linear/kernels/gdn_cutedsl.py

- **作用**：GDN 的 CuTe DSL 内核封装（decode SM90+，prefill SM100+）。
- **实现逻辑**：decode 调 `cutedsl_fused_sigmoid_gating_delta_rule_update`；extend 懒加载 `chunk_gated_delta_rule_cutedsl`（Blackwell，head_k_dim 必须 128），Q/K 先 `l2norm_fwd`，状态池读写融合进 TMA（-1 padding 重映射 sentinel slot）；SM90 `supports_prefill=False` 由 dispatcher 回退 Triton。

#### attention/linear/kernels/gdn_flashinfer.py

- **作用**：GDN 的 FlashInfer 内核封装（SM90/SM100 decode+prefill+MTP verify；要求 flashinfer≥0.6.14）。
- **实现逻辑**：SM100 用 state pool 直通（`initial_state_indices`），SM90 用 gather/scatter；extend 支持 state checkpoints（`checkpoint_cu_starts/every_n_tokens`，供 radix track 非对齐快照）；动态输入做 32B 对齐检查，不满足回退 Triton；SM120 prefill 需 fp32 状态、SM100 可用 bf16 池。

#### attention/linear/kernels/gdn_triton.py

- **作用**：GDN 的 Triton 参考内核（全平台 fallback）。
- **实现逻辑**：按平台选内核（CPU/NPU/XPU 用平台实现，其余 FLA Triton）；`packed_decode` 把 conv1d 输出直喂融合 packed-decode kernel（ReplaySSM ring 参数齐全时走 replayssm 版）；target_verify 带 `disable_state_update=True` 与 intermediate states 缓冲供 MTP 回滚。

#### attention/linear/kernels/kda_cutedsl.py

- **作用**：KDA 的 CuTe DSL 内核封装（decode SM90+，prefill SM100+）。
- **实现逻辑**：decode 调 `cutedsl_fused_sigmoid_gating_kda_update`，带 lower_bound（safe gate）时抛 NotImplementedError 指路 triton；extend 懒加载 `chunk_kda_cutedsl`（SM100、head_k_dim=128），`return_intermediate_states` 时拒绝并指路 triton。
- **备注**：pool 状态 `[slots,HV,V,K]` 与 cutedsl `[V,K]` 布局一致无需 transpose。

#### attention/linear/kernels/kda_flashinfer.py

- **作用**：KDA 的 FlashInfer 内核封装（SM100 decode + MTP verify，topk=1）。
- **实现逻辑**：包裹 `flashinfer.kda_decode.recurrent_kda`（SM100 CuTe DSL）；契约：门激活为 `-exp(A_log)*softplus(a+dt_bias)`、beta 由包装层 sigmoid、q/k 内核内 L2 norm；`_check_state_stride_contract` 一次性校验 pool stride 与 16/32B 对齐；输入全 cast bf16（内核 cp.async 硬编码 2 字节）。
- **备注**：prefill 无 FlashInfer kernel，dispatcher 保证 extend 留 Triton/CuTe DSL。

#### attention/linear/kernels/kda_flashkda.py

- **作用**：MoonshotAI FlashKDA 外部包（CUTLASS 全融合 KDA prefill）的封装。
- **实现逻辑**：内核内融合 q/k L2 norm、beta sigmoid 与门激活（传 RAW 张量 + A_log/dt_bias/lower_bound）；限制 bf16、K==V==128、HV==H、必须有 lower_bound，序列长度 [64, 2048]；`return_intermediate_states`、spec decode 与超界 batch 回退 Triton。

#### attention/linear/kernels/kda_helion.py

- **作用**：Helion 包实现的 KDA packed decode 与 prefill（helion==1.4.0 外部依赖）。
- **实现逻辑**：`packed_decode` 用 `helion_fused_recurrent_kda_packed_decode`（ReplaySSM ring 齐全时走 replayssm 版）；extend 用 Helion `chunk_kda`；通用 decode 与投机 verify 路由回 Triton。

#### attention/linear/kernels/kda_nvidia.py

- **作用**：NVIDIA 提供的 split KDA prefill 管线（SM100/103，tcgen05+TMEM）的 serving 封装。
- **实现逻辑**：packed prefill 批次按 ≤8 序列分组、pad 到 2k/4k/8k/16k 桶（pad 行状态中性：k/v/beta 置零、g 置 -1000）；单序列、需 interior chunk 快照与 spec decode 留 Triton；执行前备份状态，异常恢复并回退 Triton。

#### attention/linear/kernels/kda_ptx.py

- **作用**：手写 CUDA/PTX（tcgen05）的 KDA chunked prefill（GB300/sm_103a），`--linear-attn-prefill-backend ptx_kda`。
- **实现逻辑**：支持 K=V=128、chunk 64；单序列非 64 倍数 pad 到 1k-32k 桶，多序列走内核自带 varlen grid（host cu_seqlens 免 D2H）；`return_intermediate_states` 仅当 `mamba_cache_chunk_size()==64` 才支持；首次调用 JIT 编译（~1-2 分钟）。
- **备注**：仅 SM103。

#### attention/linear/kernels/kda_triton.py

- **作用**：KDA 的 Triton 参考内核（全后端 fallback 与正确性对照）。
- **实现逻辑**：packed_decode 调 `fused_recurrent_kda_packed_decode`（ReplaySSM ring 齐全且无 lower_bound 时走通用 replayssm，is_kda=True）；decode/verify 调 `fused_sigmoid_gating_delta_rule_update`；extend 调 `chunk_kda`（A_log/dt_bias/lower_bound/beta_is_raw 透传）。
- **备注**：是 KDA 正确性测试断言对照的参考实现。

#### attention/linear/kernels/kernel_backend.py

- **作用**：线性注意力内核实现的抽象基类（纯接口契约，具体实现在 `sglang.kernels.ops.attention.linear`）。
- **关键内容**：`LinearAttnKernelBase(ABC)`，类属性 `uses_state_checkpoints`、`supports_fused_chain_verify`；抽象方法 `decode`（含 A_log/dt_bias/ssm_states）、`extend`（含 g/beta）；`target_verify` 默认抛 `NotImplementedError`——spec-decode 验证路径通过默认未实现声明能力。

#### attention/linear/lightning_backend.py

- **作用**：Lightning（Bailing）线性注意力后端，混合模型中线性层的 prefill/decode 执行器。
- **关键内容**：`LightningAttentionBackend(MambaAttnBackendBase)`；`_build_slope_tensor`（ALiBi 式 decay slope，支持 `lightning_layerwise_decay` 逐层缩放并按 TP 切分）。
- **实现逻辑**：按 `linear_backend` 配置在 `minimax`（BailingLinearKernel 的 JIT 前向 / decode Triton）与 `seg_la`（`SegLaMeta`+`seg_la_fwd`）间分派；init 时直接拒绝 topk>1 的 spec decode（seg_la 按链式处理 draft）；target_verify 用 `intermediate_ssm` 暂存并生成 `intermediate_state_indices`。

#### attention/linear/linear_metadata.py

- **作用**：Bailing 线性注意力的 per-forward 元数据。
- **关键内容**：`BailingLinearMetadata(ForwardMetadata)`（字段 `num_prefills`/`num_prefill_tokens`/`num_decodes`/`batch_size`/`has_initial_states`/`q_lengths`）；工厂 `prepare_decode`（CUDA graph 捕获路径，prefill 数为 0）与 `prepare_mixed`（混批，从 `extend_prefix_lens` 构造 `has_initial_states`）。

#### attention/linear/short_conv_backend.py

- **作用**：short-conv 注意力后端（sidecar），只做每请求 conv 状态的管道，不跑 conv 内核本身；服务 LFM2 深度门控 short conv 与 ZAYA1 CCA 双阶段 group conv。
- **关键内容**：`ShortConvMetadata`（NamedTuple）、`ShortConvAttnBackend(MambaAttnBackendBase)`，核心入口 `conv_state_metadata(layer_id, forward_batch)`。
- **实现逻辑**：`_cache_indices_buf` 是 grow-only 的 CUDA/CPU 图静态缓冲，replay 时原地重填保持地址稳定；`forward_decode/forward_extend` 显式抛 `NotImplementedError` 防误走全量分派。

#### attention/linear/utils.py

- **作用**：线性注意力内核后端的选择与工具。
- **关键内容**：`LinearAttnKernelBackend` 枚举（TRITON/CUTEDSL/NV_CUTEDSL/FLASHINFER/FLASHKDA/NVIDIA_KDA/PTX_KDA/HELION/INTEL_XPU/CUSTOM，未知值经 `_missing_` 落 CUSTOM）、`LinearAttnBackends`（frozen msgspec.Struct：decode/prefill/verify 三阶段）、`resolve_linear_attn_backends(prefill_default)`（从 `get_exec().mamba` 读配置；verify 未设时跟随 decode 或落 triton）、`build_verify_intermediate_state_indices`（为 MTP/target_verify 生成 intermediate_ssm 行索引，padding 行 clamp）。
- **备注**：三阶段独立选择使 target 与 draft worker 可用不同内核。

#### attention/mamba/（5 个文件：Mamba2 家族）

`mamba/` 提供 Mamba2 mixer 层、causal conv1d、共享元数据与融合 norm。

#### attention/mamba/causal_conv1d.py

- **作用**：causal conv1d 的 CUDA/Triton 双实现分派包装（改编自 vLLM/Dao-AILab causal-conv1d）。
- **关键内容**：`causal_conv1d_fn`（varlen prefill 版）、`causal_conv1d_update`（逐 token decode 版）。
- **实现逻辑**：CUDA 用 sglang.kernels 编译内核、其余设备 Triton；x 非连续且调用方预传 `seq_lens_cpu` 时优先 Triton，避免大 prefill 批的 contiguous 拷贝（>0.6ms/layer）；都支持 `cache_indices`/`conv_state_indices` 连续批处理与 `pad_slot_id` 跳过填充行。

#### attention/mamba/mamba.py

- **作用**：Mamba2 的 mixer 模块（PyTorch 层）。
- **关键内容**：`mamba_v2_sharded_weight_loader`（n_groups 无法整除 TP 时 group 复制的分片加载，含 CPU 零填充路径）、`MambaMixer2`（in_proj/conv1d/out_proj、参数 A/D/dt_bias、norm=Mixer2RMSNormGated）。
- **实现逻辑**：prefill 走 `mamba_chunk_scan_combined`（chunked prefill、`initial_states`、track_states 支持），decode 走 `selective_state_update`；target_verify 用 `causal_conv1d_update_triton`+`intermediate_conv_window` 和 `disable_state_update=True`+`intermediate_ssm` 缓冲处理 draft 树。

#### attention/mamba/mamba2_metadata.py

- **作用**：Mamba2 各层共享的 forward 元数据。
- **关键内容**：`ForwardMetadata` dataclass（mamba_cache_indices、mamba_track_indices、GDN 的 replayssm_write_pos/force_flush、radix track 的 track_ssm_*、KDA fused-accept 字段等）、`Mamba2Metadata`（含 `MixedMetadata`：has_initial_states、seq_idx、chunk_indices/chunk_offsets）、`prepare_decode`/`prepare_mixed`。
- **实现逻辑**：`_query_start_loc_to_chunk_indices_offsets` 把变长序列切成「逻辑 chunk」（跨序列或物理 chunk 边界即断）；`chunk_offsets/chunk_indices` 仅在存在 initial states 时计算并全局共享。

#### attention/mamba/mixer2_rms_norm_gated.py

- **作用**：Mamba2 的 gate+SiLU+分组 RMSNorm 融合层。
- **关键内容**：`Mixer2RMSNormGated(BaseFusedOp)`（`forward_native`/`forward_cuda`）。
- **实现逻辑**：SiLU 在归一化前对 gate 应用；TP 分三种情形——n_groups==1 沿归约维切分并 AllReduce 平方和、tp_size 整除 n_groups 时本地归约、一般情形 AllGather 冗余计算后按 rank 切回；CUDA 可整除时调融合内核 `rms_norm_gated`；DP attention 下用 `attn_tp_all_reduce`。

#### attention/mamba/replay_state_indices_validator.py

- **作用**：Mamba CUDA graph replay 状态索引的 CPU 侧调试校验器。
- **实现逻辑**：`validate_replay_state_indices_cpu`：live 行必须落在 `[0, num_state_slots)` 且槽位唯一（slot 0 是合法存储行），padding 行必须全等于 `pad_slot_id`；错误聚合后一次抛 `AssertionError`。
- **备注**：刻意 CPU-only，避免把 host 读与动态断言放进 graph 捕获。

#### attention/merge_state.py

- **作用**：radix 缓存 prefix/suffix 注意力结果的状态合并入口。
- **关键内容**：`merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse)`。
- **实现逻辑**：CUDA 且 fp32/fp16/bf16、headdim 满足 4/8 整除时走 `sgl_kernel.merge_state_v2`；否则（AMD、FP8、特殊 headdim）回退 `merge_state_triton`。

#### attention/minicpm/（7 个文件：MiniCPM 稀疏注意力）

`minicpm/` 支撑 MiniCPM-Hybrid 类模型的 K1/K2 两级键压缩 + top-k 块稀疏注意力。

#### attention/minicpm/__init__.py

- 空文件，仅作包标记。

#### attention/minicpm/attention_adapter.py

- **作用**：MiniCPM 稀疏后端的实际注意力执行适配器（flash-attention 与 FlashInfer 两个实现）。
- **关键内容**：`MiniCPMFlashAttentionAdapter`（把稀疏页表/seqlens 传入 `flash_attn_with_kvcache`）、`MiniCPMFlashInferAdapter`（维护 kv_indptr/kv_indices 缓冲，prefill 用 paged wrapper、decode 用 decode wrapper 或 graph wrapper）。
- **实现逻辑**：decode 的 kv_indices 由 `create_flashinfer_kv_indices_triton` 按 group-major 行填充（每 KV head 一个 row）；支持 sinks/k_descale 透传（FlashInfer 版不支持 sinks）。

#### attention/minicpm/backend.py

- **作用**：MiniCPM 稀疏注意力后端主体（叠加在标准 FlashAttentionBackend 上）。
- **关键内容**：`MiniCPMSparseBackend(AttentionBackend)`、`_transpose_head_group_layout`、`_gather_compressed_keys` 等。
- **实现逻辑**：要求 MiniCPMHybridConfig，校验 kernel_stride 整除 kernel_size/block_size、每 KV head 16 个 query head；`attach_compressed_cache` 挂 K1/K2 压缩缓存；top-k 经 `compressed_attention`（infllmv2 内核）或 `compressed_attention_tilelang`（Blackwell+FlashInfer 时 fused）得块索引后用 `get_block_table` 生成稀疏 page table；CUDA graph 用静态缓冲绑定 + replay 原地刷新；不支持 spec decode 与 sliding window。

#### attention/minicpm/cache.py

- **作用**：MiniCPM 压缩 K 缓存的槽位分配器。
- **关键内容**：`MiniCPMCompressedCache`（`reset_allocator`/`alloc_to_lengths`/`free`/`clear`）、`attach_compressed_cache`。
- **实现逻辑**：在 ReqToTokenPool 上新增 K1/K2 压缩 token 表（K1 每 stride 一个、K2 每 4*stride 一个）；预留最多 `1/s + 1/(4s)` 槽位；`alloc_to_lengths` 失败时回滚；要求 page_size=1。

#### attention/minicpm/fuse_kernel.py

- **作用**：Blackwell 上 fused 注意力+池化+online top-k 的 tilelang 内核。
- **关键内容**：`@tilelang.jit` 的 `_fused_attn_pooling_online_topk`（含 `bitonic_sort` macro）；`fused_attn_pooling_online_topk_prefill` 与 `_decode` 两个 partial。
- **实现逻辑**：在线 softmax（logsum/scores_max）后按 infllmv2 块池化窗口聚合分数、`atomic_max` 取块最大值；双调排序维护 topk 的 (值, 索引) 对；pass config 关闭 TMA/warp-specialized。
- **备注**：仅 bf16/fp16，要求正数 `--chunked-prefill-size`。

#### attention/minicpm/sparse_kernels.py

- **作用**：K 缓存压缩的 Triton 内核。
- **实现逻辑**：`compress_k_complete_kernel_new` 单内核融合历史 chunk 拷贝与新增 token 的 kernel_size 窗口 mean pooling 压缩，写回 key_cache 与 full_compressed_k；grid (batch, max_grid_chunks, head_num_k)，超限块内循环；目前仅支持 page_size==1。

#### attention/minicpm/sparse_utils.py

- **作用**：MiniCPM 稀疏注意力的元数据规划与压缩/评分工具集。
- **关键内容**：`compress_k_core_new`、`get_compress_k_v2`、`compressed_attention`（infllmv2 stage1 + max_pooling + topk）、`compressed_attention_tilelang`、`CompressionLevelMetadata`/`MiniCPMSparseMetadata`、`_plan_sparse_prefill`/`_plan_sparse_decode`。
- **实现逻辑**：两级压缩分别按 (kernel_size, stride) 与 (4×, 4×) 生成 cu-seqlens/history/new token 计数；plan 把 batch 拆成 sparse 行（按 head_group 展开、每行 KV 长度截到 `sparse_topk*block_size`）与 dense 行（走普通 FA）。

#### attention/minimax_sparse_backend.py

- **作用**：MiniMax-M3 稀疏注意力的服务后端（含 hybrid 组合）。
- **关键内容**：`MiniMaxSparseAttnBackend(AttentionBackend)`（forward_extend/forward_decode 返回 `(idx_o, o)` 两路输出）、`MiniMaxHybridAttnBackend`（按 `sparse_layer_ids` 路由 dense/sparse 后端并桥接 verify buffer 接口）。
- **实现逻辑**：从 `MiniMaxSparseKVPool` 读主 KV 与 index KV 两套缓存（`set_fused_kv_index_buffer` 一次性写入）；MSA 仅在 SM100、block_size_k==128==page_size、topk∈{4,8,16,32}、非 fp8 缓存或 fp8 attn-GEMM 一致时启用，且与 CUDA graph+spec 解码互斥；fp8 模式下 KV 写后把 q/idx_q 量化成 e4m3；decode 可选 dense-sparse 路径；NPU 走专用 `_forward_npu_triton_*`。

#### attention/minimax_sparse_ops/minimax_sparse.py

- **作用**：MiniMax 稀疏注意力的三段式编排（indexer → topk 归约 → 稀疏主注意力）。
- **关键内容**：`minimax_sparse_prefill`、`minimax_sparse_decode`、`_warn_msa_fallback`。
- **实现逻辑**：Step1 用 index head 跑 `flash_prefill_with_topk_index`/`flash_decode_with_topk_idx` 产出 topk 块索引（decode 且 dense main 时直接产 page table）；Step2 `num_idx_heads > num_kv_heads` 时用 `topk_index_reduce` 归约；Step3 有 sink 或 MSA 不可用时走 Triton `flash_*_with_gqa_share_sparse`，否则 `use_msa` 走 `msa_sparse_*_main` 并在 `MSAUnavailableError` 时回落；cu_seqblocks 预计算一次供 indexer 与主注意力共享。

#### attention/minimax_sparse_ops/msa.py

- **作用**：fmha_sm100（MSA）作为 MiniMax-M3 主稀疏注意力步骤的 drop-in 实现（只替换 Step3，indexer 不变）。
- **关键内容**：`MSAUnavailableError`、`msa_available`（仅 SM100/103 且 fmha_sm100 可导入）、`_build_page_table`、`msa_sparse_prefill_main`、`msa_sparse_decode_main`、`build_msa_decode_cg_plan`/`update_msa_decode_cg_meta`。
- **实现逻辑**：dtype 必须统一 bf16 或 fp8_e4m3（fmha_sm100 按 q.dtype 选内核，混排会被静默重解释）；decode 用 build-once 持久 plan，`update_msa_decode_cg_meta` 全设备侧原地刷新四个长度张量与 page table（无 host-device sync）；注释记录 2026-07 根因：topk 生产者曾按分数序输出块 id，违反 fmha_sm100 严格升序契约导致静默错结果。

#### attention/minimax_sparse_ops/naive/（2 个参考实现）

- `flash_with_topk_idx.py`：indexer 解码的纯 torch 参考（einsum 全量 qk 打分 → block max 池化，init/local blocks 高分强制 → 逐 (batch, head) topk 取块），仅供测试对照。
- `topk_sparse.py`：GQA 共享稀疏主注意力的纯 torch 参考（按 (batch, kv_head) 收集 topk 块，GQA 组内 query head 共用同一组块，sink 参与归一化）。

#### attention/minimax_sparse_ops/tests/（4 个测试文件）

- `test_flash_with_topk_idx.py`：indexer 内核对照 torch 参考的输出与 topk 索引、score-only 模式、trivial 行跳过等回归用例。
- `test_fp8_attn_gemm.py`：Step3 稀疏主注意力与 indexer 的 decode/prefill 在 all-fp8、scale 语义上与 dequant 参考对齐，验证 e5m2 q 被 dtype 契约拒绝。
- `test_msa_fp8_parity.py`：MSA fp8 decode/prefill 与 Triton fp8 一致性，含 capture/replay 与 eager 逐位一致（bitexact）。
- `test_sparse_gqa.py`：Triton GQA 共享稀疏注意力对照 torch 参考，覆盖 paged 布局、sink、不同块大小参数化。

#### attention/nsa/__init__.py

- 弃用转发 shim：导入时发 `DeprecationWarning` 并 `from sglang.srt.layers.attention.dsa import *`（Native Sparse Attention 已迁至 dsa 目录），后续版本将删除。

#### attention/nsa/（9 个文件：弃用转发壳）

`nsa/` 子包 9 个文件（`dequant_k_cache.py`、`index_buf_accessor.py`、`dsa_backend_mtp_precompute.py`、`dsa_indexer.py`、`quant_k_cache.py`、`tilelang_kernel.py`、`transform_index.py`、`triton_kernel.py`、`utils.py`）全部是同构的 DeprecationWarning 转发壳：`import *` 自迁移后的新位置（`sglang.kernels.ops.attention.dsa.*` 或 `sglang.srt.layers.attention.dsa.*`），保持旧导入路径不破坏。

#### attention/nsa_backend.py

- 废弃转发壳：发 DeprecationWarning 后从 `dsa_backend.py` 导入全部符号，并显式重导出 `DeepseekSparseAttnBackend`、`NativeSparseAttnBackend` 及其 MultiStep 变体与 `DSAMetadata`/`NSAMetadata` 等元数据类，保证 `nsa_backend` 旧名称可用。

#### attention/swa_mla_fallback/（3 个文件：SWA MLA 兜底）

#### attention/swa_mla_fallback/__init__.py

- 仅一行 docstring「Fallback operations for sliding-window MLA attention paths.」，无导出内容。

#### attention/swa_mla_fallback/forward.py

- **作用**：dense KV-LoRA 解码的 SWA torch 兜底 forward（`forward_dense_kvlora_swa_torch_fallback`）。
- **实现逻辑**：前置校验 `tp_k_head_num==1`、`qk_dim==kv_cache_dim`、`s_q ∈ {1,4}`；`gather_page64_kv_latent` 按 window_size 收集 KV latent 与有效掩码，`torch.bmm` 算 scores，`apply_swa_score_mask` 屏蔽窗口外位置后 softmax+BMM；窗口大小向上对齐 8 倍。

#### attention/swa_mla_fallback/ops.py

- **作用**：SWA MLA 兜底的两个 Triton kernel 及包装（`PAGE_SIZE=64`）。
- **实现逻辑**：`_gather_page64_kv_latent_kernel` 按 block_table 做逻辑→物理 token 转换；`_apply_swa_score_mask_kernel` 对窗口外/无效位置写 -FLT_MAX，约束 `kv_pos <= q_pos` 且 `kv_pos >= q_pos - WINDOW_SIZE + 1`。

#### attention/tbo_backend.py

- **作用**：two-batch overlap（TBO）注意力后端包装器 `TboAttnBackend`。
- **实现逻辑**：持有 primary 全批后端 + 2 个 child 后端，metadata 按 `forward_batch.tbo_children` 切片分派给 child，`forward` 全委托 primary；`__getattr__` 兜底转发 DSV4 的自定义方法；`_children_use_cuda_graph` 依 `tbo_supports_cuda_graph` 决定 child 是否参与图捕获（避免 DSV4 在 ROCm 上泄漏 HSA 资源）。

#### attention/tokenspeed_mla_backend.py

- **作用**：TokenSpeed-MLA（LightSeek，MIT 协议）Blackwell SM100 CuTe DSL 内核后端，`TokenspeedMLABackend(TRTLLMMLABackend)`。
- **实现逻辑**：要求 FP8 e4m3 KV 且 page_size∈{32,64}；`__init__` 预 JIT prefill kernel 变体（避免首次请求触发 300s watchdog）；`_fused_rope_fp8_quantize` 融合 RoPE+FP8 量化；decode 原生支持 DCP 的 cp_world/cp_rank 并返回 partial LSE 供跨 rank 归并；另有 `TokenspeedMLAMultiStepDraftBackend` 供 EAGLE。

#### attention/torch_flex_backend.py

- **作用**：`TorchFlexAttnBackend`，基于 `torch.nn.attention.flex_attention` 的纯 PyTorch 后端。
- **实现逻辑**：`torch.compile(flex_attention, dynamic=True)` 并放大 dynamo cache 上限；`init_forward_metadata` 逐序列用 `create_block_mask` 预建 extend（下三角）/decode（单行）block mask；经 `req_to_token` 取 K/V；不支持非因果与 cross-attention。

#### attention/torch_native_backend.py

- **作用**：`TorchNativeAttnBackend`，用 `F.scaled_dot_product_attention` 的参考后端。
- **实现逻辑**：extend/decode 逐序列循环，把冗余 query（prefix 补零占位）与 cache 中 K/V 送 SDPA；支持 cross-attention（encoder_lens 定位 K/V 段）、GQA、sliding window（布尔掩码取代 causal）；写缓存时把 full→SWA 翻译后的 `swa_out_cache_loc` 传给 KVWriteLoc。

#### attention/triton_backend.py

- **作用**：sglang 最完整的 Triton 注意力后端 `TritonAttnBackend`（2544 行）及其多步草稿包装器。
- **实现逻辑**：decode 用 split-KV `decode_attention_fwd`（支持 Lean/Work-Centric 持久网格、PDL、MLA、DCP all-gather + `cp_lse_ag_out_rs_mha` 归并）；extend 支持 dense FP8 chunked MHA（Kimi-K3 专属条件）、one-shot dense、DCP、deterministic unified 单阶段 kernel、SWA 窗口缓冲及 gfx95 快速 target-verify（`verify_splitkv_fwd`/`verify_shared_kv_fwd`）；`TritonMultiStepDraftBackend` 用 `generate_draft_decode_kv_indices` 一次生成多步草稿 KV 索引。

#### attention/trtllm_mha_backend.py

- **作用**：`TRTLLMHAAttnBackend(FlashInferAttnBackend)`，flashinfer 的 TRT-LLM MHA 内核后端。
- **实现逻辑**：prefill 按架构分派 `trtllm_fmha_v2_prefill`（SM90/SM120，Q_PAGED_KV_NHD）或 `trtllm_batch_context_with_kv_cache`（SM100，HND）；decode 用 XQA（SM90/SM120）或 TRTLLM-GEN（SM100）；支持 attention sink、滑动窗口、fused FP8 qkv+KV cache 写、nvfp4 KV 读缩放、CP-v2 zigzag 页表；`needs_cpu_seq_lens=False`。

#### attention/trtllm_mla_backend.py

- **作用**：`TRTLLMMLABackend(FlashInferMLAAttnBackend)`，TRT-LLM MLA 内核后端。
- **实现逻辑**：decode 走 trtllm-gen 多 CTA 内核；FP8 路径用 `mla_quantize_and_rope_for_fp8` 融合量化，另有 `_set_kv_and_concat_q_fused`/`_fp8_fused` 单 launch 融合「KV 写入 + q 拼接」；支持 DCP 局部解码返回 `(out, lse)` 并 `fixup_zero_kv_rows` 修正无分片 rank；prefill metadata 支持 `fallback_to_flashinfer_impl` 回退；`TRTLLMMLAMultiStepDraftBackend` 提供 EAGLE 多步草稿。

#### attention/verify_mask.py

- **作用**：投机解码 target-verify 树掩码的缓冲区契约（`VerifyMask` msgspec.Struct）。
- **实现逻辑**：`tree_mask_numel` 按 `TreeMaskMode.QLEN_ONLY`（num_draft_tokens²）或 FULL_MASK（num_draft_tokens×(max_context_len+num_draft_tokens)）计算；因 `build_tree_kernel_efficient` 在草稿后原地写 buffer 而与布局强绑定；`maybe_create_verify_mask` 在非 draft runner、非 skip_prefill 且配置了 draft tokens 时按捕获最大批分配。

#### attention/vision.py

- **作用**：视觉编码器（ViT 类）无缓存注意力的统一实现与后端分发（1567 行）。
- **关键内容**：`VisionAttention` 主类（QKV 并行投影、可选 qk-norm（internvl/GLM-OCR 两种）、RoPE 与自定义位置嵌入）；后端实现类 `VisionSdpaAttention`/`VisionTritonAttention`/`VisionFlash3/4Attention`/`VisionFlashInferAttention`（cuDNN graph，分桶）/`VisionAiterAttention`/`VisionAscendAttention`/`VisionAMXAttention`/`VisionIntelXPUAttention`。
- **实现逻辑**：`_determine_attention_backend` 规则 server args > 构造参数 > 平台默认（SM90→fa3、SM100→fa4、其余 CUDA→triton、NPU→ascend、HIP→aiter…）；`STRIDED_QKV_BACKENDS` 可消费 qkv_proj 的跨步视图省一次拷贝。

#### attention/vision_utils.py

- **作用**：ViT 注意力 TP 假头（dummy heads）工具。
- **实现逻辑**：`update_vit_attn_dummy_heads_config` 在 `num_heads % tp_size != 0` 时计算 `num_dummy_heads`；`pad_vit_attn_dummy_heads` 按权重名（qkv_proj/q/k/v_proj/proj/q_norm/k_norm）零填充对应维度，保证加载权重形状与 TP 切分兼容。

#### attention/wave_backend.py

- **作用**：`WaveAttnBackend`，基于 AMD `wave_lang` DSL 的注意力后端（当前针对 ROCm）。
- **实现逻辑**：懒导入 wave kernel 并为每进程设独立编译缓存目录（防写竞争）；decode 走 `wave_ops.decode_attention`（两阶段 phase_0/phase_1），extend 走 `extend_attention_wave`，均支持 target-verify 与 CUDA graph；`get_num_kv_splits` 按序列数/核心数经 Triton kernel 计算。

#### attention/wave_ops/（3 个文件）

- `decode_attention.py`：wave_lang 分页 decode 内核封装（支持 page size=1）；按 MHA/GQA 选 mfma_variant，编译 phase_0（QK）/phase_1（SV 归并）两 kernel，中间结果放外部 `attn_logits` 缓冲；`WAVE_DUMP_MLIR=1` 可导出 MLIR。
- `extend_attention.py`：分页 extend 内核（MFMA `F32_16x16x32_K8_F16` 变体），同时接受本次扩展的 k/v 与 paged KV cache，支持 causal、layer_scaling、logit_cap。
- `prefill_attention.py`：非分页 prefill 内核；`b_start_loc`/`b_seq_len` 描述变长批次；q 预乘 `1/sqrt(head_dim)` 与 log2e 配合 kernel 的 exp2 计算。

#### attention/xpu_backend.py

- **作用**：`XPUAttentionBackend`，Intel GPU 的 FlashAttention 后端（基于 FlashAttentionBackend 形态，计划重构）。
- **实现逻辑**：prefill/decode 用 `flash_attn_varlen_func`/`flash_attn_with_kvcache`，MLA 用 `flash_mla_decode`（`merge_state_v2` 归并）；复用 FlashAttentionMetadata 并扩展 local attention、SWA spec、encoder-decoder 页表；支持 topk>1 cascade attention；deterministic 模式固定 num_splits=1。
- **备注**：TODO：暂不支持投机解码、XPU graph 与 MLA prefill。

### 3.2 moe/（MoE 路由与执行，62 个文件）

本目录是 SGLang 混合专家（MoE）推理的完整分层实现，按「路由 → 分发 → 计算 → 归并」的数据流组织：`topk.py`/`hash_topk.py` 把 router logits 变成每 token 的 top-k 专家与权重；`token_dispatcher/` 负责 EP（专家并行）下跨 rank 的 token all-to-all 通信，抽象出 `dispatch → run_moe_core → combine` 三段式协议；`moe_runner/` 是核心后端分发框架，用 `register_fused_func`/`register_pre_permute`/`register_post_permute` 注册表把任意 A2A 后端与任意计算后端（Triton、DeepGEMM、FlashInfer TRT-LLM/CUTLASS/CuteDSL、Marlin、Humming、AITER、Ascend、HPC-Ops、triton-kernels）两两配对；`fused_moe_triton/` 存放主 `FusedMoE` 层与 Triton/Marlin 算子；量化直通路径（`cutlass_moe`、`cutlass_w4a8_moe`、`flashinfer_cutedsl_moe`、`flashinfer_trtllm_moe`）为 FP8/W4A8/NVFP4 提供单 kernel 实现。EP 扩展族还包括：`dwdp/`（DWDP：MoE prefill 时 token 留在本 rank、对端专家权重经 NVLink 预取进复合虚拟地址空间）、`waterfill.py`（共享专家作为第 9 个 routed 专家做负载均衡）、`mega_moe.py`（DeepGEMM 单体 Mega-MoE 核）、`ep_moe/layer.py`（DeepEPMoE 层）。数据流一句话串起来：router logits 经 topk 选出专家 → dispatcher.dispatch 做 EP 通信 → MoeRunner 按注册表把 dispatch 输出 permute 成后端输入并调用 kernel → post-permute 还原 → dispatcher.combine 加权归并出最终 hidden states。

#### moe/__init__.py

- **作用**：转发型包入口，构成 moe 包的公共 API。
- **关键内容**：从 `moe_runner` 导出 `MoeRunner`/`MoeRunnerConfig`；从 `utils` 导出 `DeepEPMode`、`MoeA2ABackend`、`MoeRunnerBackend`、`initialize_moe_config`、`get_deepep_mode` 及一系列 `should_*` 决策函数。

#### moe/cutlass_moe.py

- **作用**：CUTLASS 风格 FP8 fused MoE 内核封装。
- **关键内容**：核心函数 `cutlass_fused_experts_fp8`：对激活做 per-token-group-128 FP8 量化后用 `fp8_blockwise_scaled_grouped_mm` 完成两次 grouped GEMM，中间 `silu_and_mul`（支持 swiglu_limit clamp），最后 `apply_shuffle_mul_sum` 加权归并。
- **实现逻辑要点**：SM90 走 expert-specialization 核 `es_fp8_blockwise_scaled_grouped_mm`；SM100 MXFP8 走 `es_sm100_mxfp8_*`（UE8M0 尺度、128 对齐、专有 blockscale_offsets）；`prepare_moe_input` 生成 a_map/c_map 置换。

#### moe/cutlass_moe_params.py

- **作用**：CUTLASS MoE 参数容器。
- **关键内容**：`CutlassMoEType` 枚举（BlockscaledFP8/FP4）；`CutlassMoEParams` dataclass 在构造时预分配 strides、expert_offsets、problem_sizes、指针容器（a_ptrs/b_ptrs/out_ptrs/scales_ptrs）、alpha_ptrs、layout_sfa/sfb 等 GPU 张量，通过 `to_gemm1_args()`/`to_gemm2_args()` 分别给两次 GEMM 取参。FP4 类型额外分配 blockscale_offsets。

#### moe/cutlass_w4a8_moe.py

- **作用**：CUTLASS W4A8（int4 权重 + FP8 激活）MoE 内核封装，供 W4AFp8 量化方法调用。
- **关键内容**：三个入口：`cutlass_w4a8_moe`（standard 路径，`pre_reorder_for_cutlass_moe` 重排 + `cutlass_w4a8_moe_mm` grouped GEMM）、`cutlass_w4a8_moe_deepep_normal`（DeepEP normal，先 `deepep_permute_triton_kernel` 置换再 per-tensor 量化）、`cutlass_w4a8_moe_deepep_ll`（low-latency，masked 布局，`silu_and_mul_masked_post_per_tensor_quant_fwd` 融合激活与量化）。
- **备注**：EP 下 topk_ids 中 -1 被映射到 sink slot。

#### moe/dwdp/（DWDP 预取族，7 个文件）

- **目录说明**：DWDP（DeepSeek Weight-Direct Prefetch？/ MoE prefill 专用）让 token 留在本 rank、对端专家权重经 NVLink 预取到复合虚拟地址（VMM）空间再直接读写，省去 all-to-all 通信与重排。
- **顺序**：`__init__.py`、`dwdp_manager.py`、`layout.py`、`page_pool.py`、`transport.py`、`weight_buffer.py`、`weight_manager.py`。

#### moe/dwdp/__init__.py

- **作用**：转发型。导出 `DwdpManager` 及全局 get/set 函数；docstring 说明 DWDP 语义：MoE prefill 时 token 留在本 rank，peer 专家权重经 NVLink 预取到复合 VMM 地址空间。

#### moe/dwdp/dwdp_manager.py

- **作用**：DWDP 全局单例编排器，管理从 `setup(model)` 到 `cleanup()` 的生命周期。
- **实现逻辑要点**：`setup` 收集模型中所有 `FusedMoE` 层、校验专家数可被 dwdp_size 整除，构建 `DwdpExpertLayout`，依次创建 `DWDPTransport`（句柄交换）、`WeightBuffer`（复合 VA）、填充边界字节、`DWDPWeightManager`（异步预取），并对小参数做 all_gather；随后把每层权重 rebind 为全量张量。`prefetch_first_layers`/`wait_prefetch`/`record_compute_and_prefetch_next` 供前向调用。

#### moe/dwdp/layout.py

- **作用**：DWDP 专家归属与页对齐内存布局计算（改编自 TensorRT-LLM）。
- **关键内容**：`DwdpExpertLayout` 算每个 rank 的本地专家区间 `[local_expert_start, local_expert_end)`、预取专家数、peer_ranges；`PageAlignedLayout.compute` 把本地权重区间对齐到设备 granularity，算出 pre/mnnvl/post 三段布局与 leading/trailing edge；`MnnvlHandleSet` 管理 (layer, name) → handle 映射；`compute_peer_ranges`/`lookup_owner` 按预取粒度切分专家所有权。

#### moe/dwdp/page_pool.py

- **作用**：双缓冲本地 VMM 页池，支撑复合 VA 中的远端区域。
- **关键内容**：`PagePool` 用 `VmmReservation` 为每个 slot 预留粒度对齐的页句柄（默认页大小 = 8×granularity），`map_pages` 把页映射进给定 reservation；`compute_slot_sizes` 按层权重布局之和算双缓冲 slot 大小。

#### moe/dwdp/transport.py

- **作用**：跨 rank 专家权重句柄交换（FABRIC 或 POSIX fd）与 peer view 导入。
- **实现逻辑要点**：`_copy_local_weights_to_handles` 把本 rank 专家权重拷入对齐的 VMM reservation 并导出句柄（拷贝后释放原参数存储）；`DWDPTransport.create` 经 `export_shareable_handles` + all_gather_object（或 `exchange_posix_fds`）交换句柄，`import_peer_handle` 导入后在 `VmmReservation` 上 `map_existing`，用 `tensor_from_pointer` 构造 peer 权重 view。

#### moe/dwdp/weight_buffer.py

- **作用**：复合虚拟地址空间：为每个 (layer, weight) 呈现连续的「全量专家权重」张量。
- **实现逻辑要点**：`WeightBuffer.create` 对每层算 `PageAlignedLayout`，分配覆盖全专家区间的 VmmReservation：pre/post 区由 PagePool 页映射（双缓冲可复用），中间 mnnvl 区直接 `map_existing` 本 rank 的 fabric 句柄；`get_remote_slices` 返回本地区间之外的远端切片供预取。`buffer_index_for_layer` 按层序 mod 2 分槽实现双缓冲。

#### moe/dwdp/weight_manager.py

- **作用**：双缓冲异步预取 peer 专家权重。
- **实现逻辑要点**：`DWDPWeightManager` 在独立 copy stream 上按 `peer_views` 分段 `copy_` 到复合 VA 的远端切片；`prefetch_layer` 等待该槽 consume event 后拷贝并记录 prefetch event，`wait_prefetch` 让计算流等待，`record_compute_and_prefetch_next` 记录 consume event 并预取「隔两层」的下一层（与当前层复用同一缓冲槽）。预置 consume events 避免首次预取 stall。

#### moe/ep_moe/（DeepEP EP 层族，2 个文件）

- **目录说明**：基于 DeepEP 的 EP MoE 层，供 Mooncake/Mori/Nixl/PPLX/DeepEPv2 等 A2A 后端共用。

#### moe/ep_moe/__init__.py

- **作用**：空文件（0 行），无导出。

#### moe/ep_moe/layer.py

- **作用**：基于 DeepEP 的 EP MoE 层 `DeepEPMoE(FusedMoE)`。
- **关键内容**：构造时按 a2a 后端、quant 方法、runner 后端组合计算 `deprecate_flag`（决定是否退回父类 FusedMoE 路径）；前向为 `dispatcher.dispatch → run_moe_core → dispatcher.combine` 三段。
- **实现逻辑要点**：`run_moe_core` 按 dispatch 格式（deepep_normal/deepep_ll）分派到 `forward_cutlass_w4afp8`/`forward_cutlass_w4afp8_masked`；支持 breakable/piecewise CUDA graph 的 eager 兜底。`get_moe_impl_class` 为 DeepEP 族后端选 `DeepEPMoE`。

#### moe/expert_pack.py

- **作用**：SGLANG-EXPERTPACK-v1 的运行时读取器与 VRAM 缓存（专家权重打包离线格式的在线加载）。
- **关键内容**：`ExpertPackStore` 校验 pack 头部（MAGIC、SHA-256、identity triplet 布局），`acquire(layer, topk_ids)` 按路由所需专家做 LRU+LFU（reuse-lfu-lru-v2）缓存：多线程 `preadv` 分段读入 pinned staging（可 O_DIRECT），再在 transfer stream 上拷贝进 GPU cache，返回 slot 索引供 MoE 层取权重；`mark_used` 记录 last_use 事件防覆盖。`KimiGGMLExpertPackStore` 子类解析 Kimi 的 GGMLMOEPACKv1 布局（Q2_K gate/up、Q3_K down、top_k 恒 16、layer 1..92）。

#### moe/flashinfer_cutedsl_moe.py

- **作用**：FlashInfer CuteDSL 的 masked FP4 MoE 单函数 `flashinfer_cutedsl_moe_masked`。
- **实现逻辑要点**：输入为 (bf16 或已量化的 fp4, scale) 元组与 `masked_m`；用 `grouped_gemm_nt_masked` 做两次 float4_e2m1fn GEMM（w1/w2 以 (…, experts) 布局 permute 传入），中间 `silu_and_mul_scaled_nvfp4_experts_quantize`（gated）或 relu²+NVFP4 量化（non-gated）；支持 down GEMM 的 sm_count/dst_signals 信号重叠（SBO）。

#### moe/flashinfer_trtllm_moe.py

- **作用**：FlashInfer TRT-LLM MoE 的 custom op 包装层。
- **关键内容**：三个 `@register_custom_op` 包装：`trtllm_fp8_block_scale_moe_out_wrapper`（logits 路由、block-quant FP8）、`trtllm_fp8_block_scale_routed_moe_out_wrapper`（packed topk_ids 路由）、`trtllm_fp8_per_tensor_scale_moe_wrapper`（per-tensor scale FP8）。
- **实现逻辑要点**：均带 fake impl 供 torch.compile / CUDA graph 捕获，支持 fp8_quantization_type（MXFP8/DeepSeekFP8）与 activation_type 透传。

#### moe/fused_moe_native.py

- **作用**：torch-native 的 FusedMoE 实现（源自 gpt-fast），供 torch.compile 使用。
- **关键内容**：`fused_moe_forward_native` 用 einsum 直接算 gate/up/down 加权和；`moe_forward_native` 仿 DeepSeek-V2 参考实现：按 topk_ids 排序 token、逐专家 `F.linear` + `SiluAndMul`/`GeluAndMul`/`SituAndMul` 激活、支持 bias 与 `swiglu_gpt_oss_sigmoid_alpha`，最后按 idxs 还原并乘 topk 权重求和。

#### moe/fused_moe_triton/（主 MoE 层族，4 个文件）

- **目录说明**：主 `FusedMoE` 层 + Triton/Marlin 内核封装 + 外部 triton-kernels 适配。

#### moe/fused_moe_triton/__init__.py

- **作用**：转发型。导出 `FusedMoE`、`FusedMoeWeightScaleSupported` 与 triton_utils 的 `fused_experts`、`get_config`、`get_config_file_name`、`moe_align_block_size`、`override_config`、`try_get_optimal_moe_config`。

#### moe/fused_moe_triton/aiter_mxfp4_w4a8_moe.py

- **作用**：AMD gfx1250（RDNA4）GPT-OSS MXFP4 W4A8 的专用路径（改编自 vLLM #46516）。
- **实现逻辑要点**：`prepare_w4a8_gfx1250_weights` 把 Quark 布局转成 `moe_gemm_a8w4` 需要的 [E, K, N] contraction-major 布局（gate/up 交错）；`aiter_w4a8_gfx1250_forward` 用 aiter triton `routing` + 两次 `moe_gemm_a8w4`（fused SwiGLU + requant）。两个 gfx1250 怪癖：禁用 TDM 内核 gather（改为 torch 手工 gather，`gather_indx=None`）、scale 不 swizzle（`swizzle_mx_scale=None`）。

#### moe/fused_moe_triton/fused_marlin_moe.py

- **作用**：Marlin 系 MoE 主算子 `fused_marlin_moe`（custom op）。
- **实现逻辑要点**：用 `moe_align_block_size` 对齐后两次 `moe_wna16_marlin_gemm`（第一次 mul_topk_weights=False、第二次 True）完成 grouped GEMM；激活支持 silu/situ（`_situ_and_mul_kernel`）/GPT-OSS sigmoid-alpha/relu²/swiglu_limit。
- **备注**：`get_scalar_type` 识别 E8M0 MXFP4 / NVFP4 global scale / uint4b8 / uint8b128；单 token decode 走 `moe_align_single_token` 快速排序；MXFP4 输出用 `moe_topk_sum` 快速归并。

#### moe/fused_moe_triton/layer.py

- **作用**：主 `FusedMoE(torch.nn.Module)` 层（约 1800 行，本目录枢纽）。
- **实现逻辑要点**：构造时计算本地/全局专家分片（含 per-rank shared slots 与 elastic EP join）、按 runner 后端做 intermediate/hidden 对齐（trtllm 128、mxfp4 256）、创建 quant_method 与 `MoeRunner`、经 `create_moe_dispatcher` 按 a2a 后端选 dispatcher（standard/DeepEP 族/DeepEPv2/Flashinfer/AscendTP）。
- **关键内容**：前向 `forward_impl` 为 dispatch → run_moe_core(quant_method.apply) → combine，DWDP 层间插入 wait_prefetch/record_compute_and_prefetch_next，支持 SGLANG_OPT_MOE_QUANT_ONCE 预量化透传与 deferred finalize。后半为整套权重加载机（per-tensor/per-channel/group scale、w13/w2、GGUF 物化、`make_expert_params_mapping*`）与 piecewise CUDA graph 的 custom op 入口。

#### moe/fused_moe_triton/triton_kernels_moe.py

- **作用**：外部 triton-kernels 包的 MoE 适配（改编自 vLLM #18595）。
- **关键内容**：`triton_kernel_fused_experts`/`..._with_bias` 用 `triton_kernels.matmul.matmul` 配合 ragged metadata、gather/scatter 索引与 gate_scal 完成两次 GEMM（可 fused activation、FP4 权重、PrecisionConfig）。
- **备注**：断言只支持 bf16 且拒绝 fp8/per-channel/block 量化；SM120 上禁用 persistent 优化。

#### moe/hash_topk.py

- **作用**：DeepSeek-V4/V3.2 的 hash 路由 `HashTopK`：token → 专家映射由词表级查表 `tid2eid`（nn.Parameter int32）决定，而非实时 top-k。
- **实现逻辑要点**：`_forward_torch` 对 router logits 用 softmax/sigmoid/sqrtsoftplus 打分后按输入 token id 查表取权重并归一化（共享专家权重 = routed 和 / routed_scaling_factor）；XPU 有 `hash_topk` AOT kernel，CUDA 可走 fused JIT 版本。
- **备注**：支持 waterfill 扩展、EPLB logical→physical 重映射、per-rank shared slots 与 `expert_location_dispatch` LP 求解。

#### moe/kt_ep_wrapper.py

- **作用**：KTransformers CPU-GPU 异构专家并行包装。
- **实现逻辑要点**：`KTEPWrapperMethod(FusedMoEMethodBase)` 包裹任意 GPU 量化方法：`apply` 先异步 `submit_forward` 提交 CPU（AMX/AVX）专家计算，同时把 `topk_ids >= num_gpu_experts` 掩成 -1 让 GPU kernel 跳过，GPU 算完后 `sync_forward` 取回 CPU 结果相加（submit-compute-sync 模式）。`KTConfig` 由 ServerArgs 的 kt_* 参数生成。

#### moe/mega_moe.py

- **作用**：DeepGEMM 单体 Mega-MoE 前向（DeepSeek-V2/V4 共享）。
- **实现逻辑要点**：`forward_mega_moe` 先算共享专家（可与 routed 部分在 alt stream 重叠），再走 `_run_mega_routed`：gate/topk → `deep_gemm.get_symm_buffer_for_mega_moe` 复用缓冲 → `mega_moe_pre_dispatch`（或 deep_gemm 的 mxf4xmxf4 版本）→ `deep_gemm.fp8_fp4_mega_moe`（recipe (1,1,32)、fast_math）。
- **关键内容**：`build_mega_moe_experts_weights` 把 w13 scale 转 UE8M0 布局、gate/up 按 gran-8/16 交错、UTCCP 转置 scale；`_configure_mega_moe_deep_gemm_num_sms` 为 clustered launch 保留 SM。

#### moe/mega_moe_sm90.py

- **作用**：SM90 专用的 FP8 Mega-MoE。
- **关键内容**：`is_sm90_fp8_mega_moe_available` 探测 deep_gemm 的 `fp8_mega_moe`/`mega_moe_pre_dispatch_sm90`；`run_sm90_mega_routed` 用 group_size=128 的 pre-dispatch + `deep_gemm.fp8_mega_moe`（recipe (128,128,128)）；`build_sm90_mega_moe_experts_weights` 只做 gate/up 按 gran-8 交错，无需 scale 变换（沿用 fp32 scale）。

#### moe/moe_runner/（后端分发框架族，17 个文件）

- **目录说明**：核心注册表机制：`FusedOpPool`（(a2a, runner) → fused func）与 `PermuteMethodPool`（pre/post permute）+ 三个装饰器；任一 (dispatcher 格式, 计算后端) 组合要么有整段 fused kernel，要么拆成 pre_permute → core.run → post_permute 三段。

#### moe/moe_runner/__init__.py

- **作用**：转发型。导出 `MoeRunnerConfig`、`MoeRunner`、`register_moe_runner_core`。

#### moe/moe_runner/aiter.py

- **作用**：AITER（AMD ROCm）runner。
- **实现逻辑要点**：`AiterRunnerCore.run` 调 `aiter.fused_moe.fused_moe`（quant_type per_Token/per_128x128/per_1x32、activation Silu/Swiglu/Situv2、gate_mode、swiglu_limit、no_combine）；`_pre_permute_deepep_to_aiter` 处理 mori/deepep 输出的 dtype 匹配：FP8/W4A4/FP4 dispatch 与权重量化不匹配时用 `upscale`/`upscale_mxfp4` 反量化成 bf16，-1 sink slot 重路由，并按 `SGLANG_MORI_MOE_MAX_INPUT_TOKENS`/`SGLANG_MORI_RECV_BOUND` 截断接收缓冲。

#### moe/moe_runner/ascend.py

- **作用**：Ascend NPU runner。
- **实现逻辑要点**：`AscendRunnerCore.__init__` 按 NPU 量化方法（NPUMXFP8MoEMethod 等）与 a2a 后端选择激活实现（NPUSwiglu/NPUSitu/NPUSwigluDeepEPKernel/NPUSwigluOAI，GGUF 时包 AllGatherActivationWrapper）；`run` 依次调 w13_kernel.apply → 激活 → w2_kernel.apply（MXFP8 时 gmm1 融合 swiglu+requant 跳过激活步）。pre/post permute 挂 ascend_tp、deepep_normal、deepep_ll 三种格式。

#### moe/moe_runner/base.py

- **作用**：moe_runner 框架的基座。
- **关键内容**：`MoeRunnerConfig` dataclass（激活、inplace、no_combine、routed_scaling_factor、gate_up_interleaved 等）；`RunnerInput`/`RunnerOutput`/`MoeQuantInfo`/`MoeRunnerCore` 抽象基类（带 `runner_backend_is_triton` TypeGuard）；`DispatchMoeRunnerCore` 为直接消费 dispatch 表示的 runner（Marlin 等）；核心注册表：`FusedOpPool`（(a2a, runner) → fused func）、`PermuteMethodPool`（pre/post permute 方法池）+ 三个装饰器 `register_fused_func`/`register_pre_permute`/`register_post_permute`。

#### moe/moe_runner/deep_gemm.py

- **作用**：DeepGEMM runner（最大 runner 文件之一）。
- **实现逻辑要点**：`DeepGemmRunnerCore.run` 按权重 dtype 与布局分四条路径：contiguous/masked × bf16/fp8。FP8 路径要点：`grouped_gemm_nt_f8f8bf16_{contig,masked}` 两次 grouped GEMM，中间融合激活+per-group 量化（SiTU 用 `_situ_mul_quant_contig_kernel`/`_varlen_deep_gemm_situ_mul_quant`，DSV4 clamped/swizzle 用 `silu_and_mul_masked_post_quant`）；UE8M0 scale 经 `_cast_to_e8m0_with_rounding_up` 向上取整转 e8m0；masked/compact 布局由内存预算 `_should_use_masked_standard_layout` 自动决策。
- **关键内容**：文件后半为多种 dispatch 格式（standard/flashinfer/deepep_ll/deepep_normal/deepep_v2）的 pre/post permute 注册（ep_scatter/ep_gather 重排）。

#### moe/moe_runner/deep_gemm_sm120.py

- **作用**：SM120（消费级 Blackwell）专用 DeepGEMM 路径。
- **实现逻辑要点**：SM120 的 contiguous grouped GEMM 只吃标准布局激活，故 `maybe_pre_permute` 用 `ep_scatter` 做连续散射（token ≥1024、非 bf16 权重、无 EP 才启用），`maybe_post_permute` 用 `ep_gather` 还原；`use_swizzle`/`allows_masked_standard_layout` 在 SM120 上恒 False；其余架构全部 no-op 回落到共享 masked 路径。

#### moe/moe_runner/flashinfer_cutedsl.py

- **作用**：FlashInfer CuteDSL FP4 runner（ModelOpt FP4 专用）。
- **关键内容**：前半为权重/尺度准备：`interleave_w13_halves`（GEMM1 SwiGLU 布局）、`cutedsl_quant_scale_to_scalar`（per-expert 尺度坍缩为标量，对齐 TRTLLM 约定）、`resolve_cutedsl_standard_scales`（由 g1/g2_alphas 与输入尺度导出 w1/w2 alpha）、`ensure_cutedsl_wrapper`（惰性创建 `CuteDslMoEWrapper`，预分配 CUDA graph 缓冲）。
- **实现逻辑要点**：后半三个 fused func：v2 标准路径（wrapper.run，w4a4/w4a16 两模式）、flashinfer A2A 路径（NVFP4 直通或本地量化）、v1 DeepEP-LL 路径（`flashinfer_cutedsl_moe_masked` + SBO 信号重叠）。

#### moe/moe_runner/flashinfer_cutlass.py

- **作用**：FlashInfer `cutlass_fused_moe` 的 fused func 集合，服务 bf16/ModelOpt FP8/NVFP4/MXFP4 四条路径。
- **关键内容**：`FlashInferCutlassMoeQuantInfo`（quant_type bf16/fp8/fp4）与 `FlashInferCutlassMxfp4MoeQuantInfo`（SM90 交错权重 + E8M0 尺度；SM120 packed + MXFP8 激活；Humming W4A8 残差尺度三元组；GPT-OSS 偏置与 swiglu 参数）。
- **实现逻辑要点**：`_run_flashinfer_cutlass` 统一调 kernel，flashinfer A2A 时 `enable_alltoall=True` 并写进 `moe_output` workspace；`fused_experts_none_to_flashinfer_mxfp4` 按 quant_info 类型分发到 CUTLASS 或 TRT-LLM SM100 实现。

#### moe/moe_runner/flashinfer_trtllm.py

- **作用**：FlashInfer TRT-LLM MoE 的最大 fused func 集合（FP8/MXFP4/FP4/BF16 四种量化）。
- **关键内容**：权重对齐工具：`align_fp8_moe_weights_for_flashinfer_trtllm`（E,N,K 重排、128 对齐 padding、shuffle）、`align_mxfp8_moe_weights...`（K 32 对齐）、`align_fp4_moe_weights...`（NVFP4 布局、g1_scale_c）。
- **实现逻辑要点**：`fused_experts_none_to_flashinfer_trtllm_fp8` 支持 block-quant（DeepSeekFP8/MXFP8）与 per-tensor 两条子路径，含 deferred finalize（do_finalize=False 延迟 GEMM2 物化）与 PDL；MXFP4 走 `trtllm_fp4_block_scale_{moe,routed_moe}`（SiTU/routed 两模式）；FP4/BF16 各自实现；`fused_experts_flashinfer_to_flashinfer_trtllm` 桥接 flashinfer A2A（要求 BF16 payload、routed kernel）。

#### moe/moe_runner/hpc_ops.py

- **作用**：腾讯 HPC-Ops 的 FP8 fused-MoE 后端（仅 SM90、主打 H20）。
- **实现逻辑要点**：`fused_experts_none_to_hpc_ops` 按 `block_quant` 选择 `hpc.fuse_moe_blockwise`（128x128 block 权重 + per-token-group-128 激活量化）或 `hpc.fuse_moe`（per-tensor 尺度、dequant alpha）；两 kernel 都融合 gather → gate_up GEMM → SiLU-mul → down GEMM → 加权归约并直接消费全局 top-k id + rank_ep/num_expert_total，EP 无需本地重映射。`pad_hpc_ops_block_scale` 把 K 维尺度 pad 到 4 的倍数；`_check_runner_config_supported` 严格校验配置。

#### moe/moe_runner/humming.py

- **作用**：Humming（自研 MoE 框架，支持 W4A16/W4A8/W8A8 等）runner。
- **关键内容**：`HummingRunnerCore` 提供三种 GEMM 形态：INDEXED（`moe_align_block_size` 对齐 + `moe_fused_mul_sum` 归并）、GROUPED_CONTIGUOUS（`moe_permute[_with_scale]`/`moe_unpermute`）、GROUPED_MASKED（per-expert token 计数、masked 激活+量化）；工作区为双 workspace 轮转复用（`get_buffer_metas`/`prepare_buffers`）。
- **实现逻辑要点**：`run` 经 custom op `humming_moe_runner_core_run` 执行以兼容 torch.compile 动态 shape；`HummingMethod.forward_layer`/`may_quant_input` 完成实际 GEMM 与量化。

#### moe/moe_runner/marlin.py

- **作用**：Marlin runner 的 fused func 入口。
- **实现逻辑要点**：`fused_experts_none_to_marlin` 调 `fused_marlin_moe`，组装 `MarlinMoeQuantInfo`（GPTQ g_idx/act-order、AWQ qzeros、E8M0 MXFP4、expert_map、bias、global scale）；每次调用新建 workspace 防 CUDA graph 捕获死锁；fp16+E8M0 时上转 bf16 再转回；对 fused gate+topk 的 PackedTopKOutput 用 Triton `_unpack_packed_topk_kernel` 单 launch 解包。

#### moe/moe_runner/runner.py

- **作用**：`MoeRunner` 类，按后端名实例化 runner_core（Triton/Ascend/TritonKernels/DeepGemm/Humming/Aiter/Marlin…；trtllm/cutlass/cutedsl/hpc_ops 等纯 fused 后端 core 为 None），查 `FusedOpPool` 取 fused func。
- **实现逻辑要点**：`run(dispatch_output, quant_info)`：有 fused func 直接调用；否则经 `PermuteMethodPool` 的 pre_permute → core.run → post_permute，支持 LoRA hooks 与 down-gemm overlap 参数注入。`register_moe_runner_core` 允许扩展注册自定义后端工厂。

#### moe/moe_runner/triton.py

- **作用**：SGLang 自带 Triton MoE runner。
- **关键内容**：`TritonRunnerCore.run` 调 `_fused_moe_kernel_sequence`（HIP gfx95 MXFP8 走 `fused_experts_mxfp8` 专用核）；`fused_experts_none_to_triton` 是 fused 路径（含 SGLANG_OPT_MOE_QUANT_ONCE 的预量化输入直通）；`pre_permute_standard_to_triton` 用 `_prepare_fused_moe_run` 选 config/对齐路由。量化参数由 `TritonMoeQuantInfo` 承载（fp8/int8/int4、per-channel、block_shape、fuse_swiglu_interleaved）。

#### moe/moe_runner/triton_kernels.py

- **作用**：外部 triton-kernels 包的 runner。
- **关键内容**：`TritonKernelsRunnerCore.run` 调 `triton_kernel_fused_experts[_with_bias]`，按 `no_combine` 决定 scatter 索引与输出 reshape/sum；pre_permute 要求 `TritonKernelTopKOutput`（ragged metadata + gather/scatter + gate_scal），post_permute 应用 routed_scaling_factor。

#### moe/moe_runner/triton_utils/（Triton 内核族，4 个文件）

- **目录说明**：Triton fused-MoE kernel 本体与配置选择、对齐算子。转发型 `__init__` 持有模块级 `_config`，提供 `override_config` 上下文管理器与 `get_config`，并转发 `fused_experts`、`get_config_file_name`、`moe_align_block_size`、`try_get_optimal_moe_config`。

#### moe/moe_runner/triton_utils/fused_moe.py

- **作用**：Triton fused MoE 的完整实现（约 1250 行），本族核心。
- **关键内容**：custom op `inplace_fused_experts`/`outplace_fused_experts` → `fused_experts_impl`。
- **实现逻辑要点**：`_prepare_fused_moe_run` 选 kernel config（含上下投影分离的 TMA config）与 `moe_align_block_size` 路由对齐，`_fused_moe_kernel_sequence` 依序执行「可选激活量化 → 上投影 → 融合激活(+量化) → 下投影 → moe_sum_reduce/mul_sum 归并」，支持 fp8/int8/int4/block 量化、filter_expert（EP）、no_combine、hooks（LoRA）、GPT-OSS sigmoid-alpha 激活等。公开 `fused_moe`/`fused_experts` 兼容旧接口。

#### moe/moe_runner/triton_utils/fused_moe_triton_config.py

- **作用**：Triton fused MoE kernel 的配置选择。
- **关键内容**：`get_moe_configs` 按 (E,N,dtype,block_shape,device,triton 版本) 从 `configs/triton_x_y/` JSON 加载调优配置（找不到时回退最新可用版本），`get_default_config` 提供启发式默认（低共享内存 fp8 特化），`try_get_optimal_moe_config` 就近取 batch 网格并强制 down config 与 up 共用 BLOCK_SIZE_M；`get_config_dtype_str` 映射量化类型字符串。

#### moe/moe_runner/triton_utils/moe_align_block_size.py

- **作用**：token→专家对齐算子 `moe_align_block_size`：把 topk_ids 展平排序、按专家分桶并 pad 到 block_size 的倍数，返回 sorted_token_ids/expert_ids/num_tokens_post_padded 供分组 GEMM。
- **实现逻辑要点**：为无效专家（EP 下 -1）额外建桶；极小 batch 走 `moe_align_small_numel` 单 CTA 快路径（bucket 数超 64 时）；底层为 sgl_kernel AOT 实现。

#### moe/qwen35_flashinfer_fusion.py

- **作用**：Qwen3.5 与 FlashInfer MNNVL CuTe DSL AllReduce 融合的集成层。
- **关键内容**：`Qwen35FlashInferFusionService` 持有融合 workspace，`finalize`/`all_reduce_residual_rms_norm` 把 MoE 加权归并、AllReduce、残差相加与 RMSNorm 融进一个 MNNVL 核；`Qwen35MoeFinalizeHandoff` 从 FlashInfer deferred 输出携带 gemm2_out/expert_weights/permuted_indices；`Qwen35FlashInferLayerCommunicator.prepare_attn/prepare_mlp` 在条件满足时（非 DP attention、无 CP、a2a=none、tp>1）拦截 LayerCommunicator 流程。

#### moe/route_quant_handoff.py

- **作用**：K3「路由+量化」融合的 attempt-and-verify 交接机制。
- **实现逻辑要点**：KimiK3MoE 层在 topk 前 `stage(x)` 激活、topk 路径中 `try_route_quant_fused` 用单个 fused kernel 一次完成 route_radix top-16 + (id<<16)|bf16(weight) pack + mxfp8 量化，`Mxfp4MoEMethod.apply` 用 `take(x)` 按 data_ptr/shape/stride 校验后取走产物跳过重复量化与 pack；任何一步 miss 均安全回退到未融合链路。
- **关键内容**：`_Handoff` msgspec.Struct 存储 staged/published 状态；`take` 用 data_ptr/shape/dtype/stride 全等校验防误消费；stage/clear 由模型层以 try/finally 配对，防止分配器复用地址导致串层泄漏。

#### moe/token_dispatcher/（分发器族，13 个文件）

- **目录说明**：dispatcher 框架基座 + 11 个具体分发器（standard/DeepEP/DeepEPv2/Mooncake/Mori/Nixl/Pplx/FlashInfer/AscendTP + flashinfer_utils）+ 汇总 `__init__`。协议核心：`dispatch → run_moe_core → combine`，输出/输入用格式枚举 + Protocol/TypeGuard 做类型契约。

#### moe/token_dispatcher/__init__.py

- **作用**：转发型。汇总导出 BaseDispatcher、各 dispatcher（AscendTP/DeepEP/DeepEPv2/Flashinfer/Mooncake/MoriEP/NixlEP/Pplx/Standard）与全部 DispatchOutput/CombineInput 类型。

#### moe/token_dispatcher/ascend_tp.py

- **作用**：Ascend NPU 的 TP 分发器。
- **实现逻辑要点**：`AscendTPDispatcher.dispatch` 调 `NPUMoEInitRouting_v2` 把 token 按专家重排（`expanded_row_idx`/`expert_tokens`），combine 用 `NPUFinalizeRouting` 加权还原；按 quant config 选输出 dtype（BF16/INT8/MXFP8），GGUF + TP>1 时 finalize 外包 `AllGatherFinalizeRoutingWrapper`。输出格式为 `ASCEND_TP`。

#### moe/token_dispatcher/base.py

- **作用**：dispatcher 框架基座。
- **关键内容**：定义 `DispatchOutputFormat`/`CombineInputFormat` 枚举（standard/deepep_normal/deepep_ll/flashinfer/deepep_v2/ascend_tp）、`DispatchOutput`/`CombineInput` Protocol 与 `DispatchOutputChecker`/`CombineInputChecker` TypeGuard；`BaseDispatcher` 提供 dispatch/combine 抽象及 pre/post dispatch、pre/post combine 四类 hook 注册（懒替换原函数，返回可移除 handle）；`BaseDispatcherConfig` 为空基类。

#### moe/token_dispatcher/deepep.py

- **作用**：DeepEP A2A 的完整实现（约 1100 行）。
- **关键内容**：`DeepEPBuffer` 管理进程级 Buffer（NVL/RDMA 字节按 dispatch/combine config hint 计算，low_latency 需 num_experts%group==0，MNNVL fabric 支持）；两个 impl：`_DeepEPDispatcherImplNormal`（`get_dispatch_layout` + `buffer.dispatch`，async_finish 用 capture/event 流水，expert_alignment=128）与 `_DeepEPDispatcherImplLowLatency`（`low_latency_dispatch` 返回 masked_m + packed_recv_count，支持 return_recv_hook 零等待、FP8/NVFP4/UE8M0 量化参数、SBO overlap 的 low_latency_combine）。
- **实现逻辑要点**：`DeepEPDispatcher` 以 dispatch_a/b、combine_a/b 四段状态机（_Stage）把 prefill（NORMAL）与 decode（LOW_LATENCY）按 `DeepEPMode.resolve` 路由到对应 impl。

#### moe/token_dispatcher/deepep_v2.py

- **作用**：DeepEP v2（`ElasticBuffer`）分发器。
- **关键内容**：`DeepEPv2Buffer` 以 key 管理进程级 ElasticBuffer（hybrid 模式、FP8 dispatch 固定 128 块 + UE8M0/TMA 对齐尺度格式）；`_DeepEPv2Impl.dispatch`：decode 用 expanded/masked 布局（异步、免 CPU sync，配 masked_max_m/expected_m 提示），prefill 用 contiguous 布局（CPU-synced，空 batch 补 dummy token 通知空闲 rank），`psum_num_recv_tokens_per_expert` 由 handle 提供；combine 单次消费 handle（finally 释放）。
- **备注**：仅支持 BF16 模型激活 + DeepGEMM runner。

#### moe/token_dispatcher/flashinfer.py

- **作用**：FlashInfer MNNVL 单边 A2A 分发器。
- **实现逻辑要点**：`FlashinferDispatcher` 持 `MoeAlltoAll` workspace（按 chunked_prefill_size/4096 上界定容，eagle 用更大尺寸），dispatch 为 decode 单边 A2A（BF16 或 NVFP4 载荷、`invalid_token_expert_id` 清洗 padding slot、`runtime_max_tokens_per_rank` 必须 rank 一致）；prefill 走 `_dispatch_prefill_allgather`：TP 组 all_gatherv 后交 `StandardDispatcher`，combine 用 reduce_scatterv 还原。`payload_in_workspace`（cutlass）时把 kernel 输出直接写进 combine workspace。

#### moe/token_dispatcher/flashinfer_utils.py

- **作用**：FlashInfer MNNVL 通信初始化的小工具。
- **关键内容**：`TorchDistributedCommBackend(flashinfer.comm.mnnvl.CommBackend)` 用 torch.distributed 替代 MPI 完成 allgather/bcast/barrier（Split 返回自身），flashinfer 未安装时提供占位基类。

#### moe/token_dispatcher/mooncake.py

- **作用**：Mooncake EP 分发器（接口对齐 DeepEP）。
- **关键内容**：`EPBuffer` 懒创建 `mooncake.mooncake_ep_buffer.Buffer`（仅支持 low_latency）；`_MooncakeEPDispatcherImpl` 复用 DeepEP 的 dispatch_a/b + combine_a/b 状态机，dispatch 带 `active_ranks`（elastic EP 动态成员）与首次执行 -1 超时；输出复用 `MooncakeDispatchOutput`（DEEPEP_LL 格式，含 masked_m/expected_m）。

#### moe/token_dispatcher/moriep.py

- **作用**：AMD ROCm Mori EP 分发器（约 1270 行）。
- **实现逻辑要点**：`init_mori_op`（lru_cache）创建 `mori.ops.EpDispatchCombineOp`：按世界规模选 INTRA_NODE/INTER_NODE/LOW_LATENCY(AsyncLL) 内核，dispatch dtype 支持 bf16/fp8/fp4/mxfp8（发送前量化：per_1x128 fp8、per_1x32 fp4、e8m0 尺度 mxfp8），combine dtype 支持 fp8/fp8_direct_cast/fp4。
- **关键内容**：`_MoriEPDispatcherImplNormal` 支持 TBO 双流（CommStreamPool）与 SDMA（dispatch_send/recv 分离）；输出携带 origin_topk_ids/weights（供 AITER runner 的 combine 归并）。dtype 选择有 env 覆盖与旧变量弃用告警。

#### moe/token_dispatcher/nixl.py

- **作用**：NIXL EP 分发器（NVIDIA Dynamo 生态）。
- **关键内容**：`NixlEPBuffer` 用全局 TCPStore 构建 `Buffer`（RDMA 尺寸按 nixl_max_ranks×num_local_experts 计算），支持 elastic EP 的增量 `connect_ranks`/scale 调度；`_NixlEPDispatcherImpl` 仅 low_latency，dispatch 支持 fp8（`SGLANG_NIXL_EP_BF16_DISPATCH` 关闭时）与 UE8M0 尺度，combine 后 `query_mask_buffer` 更新 active_ranks；输出复用 DeepEPLL 类型。

#### moe/token_dispatcher/pplx.py

- **作用**：Perplexity pplx-kernels 的 MoE A2A。
- **实现逻辑要点**：`PplxAllToAllManager` 初始化 NVSHMEM，按跨节点与否选 `AllToAll.internode/intranode`（intranode 经 c10d 注册 "pplx_ep" 组）；dispatch 输出 per-expert 掩码布局 `[num_local_experts, max_batch_tokens, hidden]`（fp8 时带 scale），combine 反向归约；输出复用 DEEPEP_LL 格式使 masked expert 计算路径不变；仅支持 low_latency 模式。

#### moe/token_dispatcher/standard.py

- **作用**：无 EP 通信的标准分发器。
- **实现逻辑要点**：`StandardDispatchOutput`/`StandardCombineInput` NamedTuple；`StandardDispatcher.dispatch` 在 EP>1 时构建 `local_expert_mapping` 把全局专家 id 映射到本地（非本地为 -1，可顺带 `mask_dp_pad_moe_topk_ids` 丢弃 DP pad 行；cutlass/cutedsl/trtllm/hpc_ops/mxfp4 后端跳过映射），FP4 all-gather 路径先量化再 `all_gatherv`；combine 时 reduce_scatterv 还原 DP 分片。

#### moe/topk.py

- **作用**：路由核心（约 2600 行，目录最大文件）。
- **关键内容**：`TopK(BaseFusedOp)` 按 runner 后端选输出格式（STANDARD/TRITON_KERNEL/BYPASSED），`select_experts` 按路由种类分发：`grouped_topk`（DeepSeek-V2/V3 分组 top-k，torch.compile）、`biased_grouped_topk_gpu`（sigmoid+bias，含 flashinfer `fused_topk_deepseek`、JIT Triton `moe_fused_gate`、aiter、K3 radix4 等多级后端选择）、`fused_topk`（softmax/sigmoid）、`kimi_k2_biased_topk_impl`；`_post_process_topk_ids` 统一做 EPLB logical→physical 映射、pad 区掩码（CUDA -1/HIP 0+零权重）、共享专家 append/per-rank shared-slot remap（aiter 走 fused Triton kernel）；`build_precomputed_topk_output` 支持路由器自产结果。
- **备注**：另有 benchmark 用的均匀/round-robin 路由模拟。

#### moe/utils.py

- **作用**：MoE 全局配置与决策函数。
- **关键内容**：`MoeA2ABackend`/`MoeRunnerBackend` 枚举（含谓词混入类与扩展注册 `RegisteredMoeRunnerBackend`）；`DeepEPMode`（normal/low_latency/auto，`resolve` 按是否 extend 选择）；`get_deepep_output_dtype`（server arg → 弃用 env → NVFP4 → quant_config → 后端默认 → FP8 的优先级链）；`initialize_moe_config` 把 exec 配置种子进 flags。
- **实现逻辑要点**：关键决策：`should_use_flashinfer_cutlass_moe_fp4_allgather`、`should_use_dp_reduce_scatterv`、`should_skip_post_experts_all_reduce`（fuse/RS/A2A 已归约时跳过 all-reduce）、`is_moe_input_scattered_across_dp_ranks`；`RoutingMethodType` IntEnum 与 flashinfer 对齐；末尾为 aiter/triton padding 与 XPU L3 aliasing 行填充工具。

#### moe/waterfill.py

- **作用**：Waterfill 负载均衡：把共享专家当作第 9 个 routed 专家分发到负载最轻的 rank（topk 8→9）。
- **实现逻辑要点**：`WaterfillBalancer.expand_topk` 先用 `_count_routed_per_rank_kernel` 统计各 rank routed 计数：小 batch（<64，static 模式）本地扩展（`expand_topk_with_shared_expert`），大 batch 走 `materialize_waterfill_dispatch_fused` fused kernel（rank_load + allow_all_ranks + target_total 决定共享专家落点）；dynamic 模式先 EP all-reduce 全局负载。
- **备注**：共享专家权重 = 1/routed_scaling_factor。

### 3.3 quantization/（量化体系，101 个文件）

本目录是从 vLLM 迁移并深度扩展的量化子系统，采用清晰的三层架构：**Config → Method → Scheme**。最底层 `base_config.py` 定义 `QuantizationConfig` 抽象基类与三类方法接口（`QuantizeMethodBase`/`LinearMethodBase`/`FusedMoEMethodBase`）；`base_scheme.py` 定义 per-format 的 `BaseLinearScheme`（`create_weights`/`process_weights_after_loading`/`apply_weights`）与 `BaseMoEScheme`（多一个 `create_moe_runner`），scheme 实例挂在 `layer.scheme` 上，Method 类只是把调用委托给 scheme 的薄壳；`__init__.py` 用 `BASE_QUANTIZATION_METHODS` 注册表把方法名字符串（fp8、awq、gptq、compressed-tensors、mxfp4…）映射到 Config 类，并按平台（CPU/NPU/XPU/MPS）覆盖，`get_quantization_config()` 是唯一入口。awq、gptq、compressed_tensors、quark、modelslim 五个方法目录结构同构；MoE 的 `apply` 最终通过 `MoeRunner` 按后端分发 quant info 与 kernel。顶层还散布大量独立工具：marlin_utils 系列（Marlin 形状校验/repack/GEMM）、fp8.py + fp8_utils.py（FP8/MXFP8 家族与 GEMM dispatch）、mxfp4.py（OCP MXFP4 生态中枢，按硬件分发到 FlashInfer CUTLASS/TRT-LLM、Marlin、Humming、AITER、DeepGEMM、CPU AMX、XPU 等 MoE kernel 入口）、kv_cache.py + fp4_kv_cache_quant_method.py + kvfp4_tensor.py（KV cache 量化独立体系）、gguf.py（llama.cpp GGUF）、humming.py（inclusionAI humming）、modelopt_quant.py（NVIDIA ModelOpt NVFP4 家族）、quark/ 与 modelslim/（AMD/Ascend 工具链）等。`unquant.py` 提供所有量化路径共用的未量化回退实现，`utils.py` 是共享底层函数库。按字母序逐文件如下（schemes/ 子目录见各自家族条目）。

#### quantization/__init__.py

- **作用**：量化方法注册表与工厂入口。
- **关键内容**：`BASE_QUANTIZATION_METHODS` 字典登记 fp8/mxfp8/blockwise_int8/awq/awq_marlin/gptq/gptq_marlin/gguf/bitsandbytes/compressed-tensors/mxfp4 等 30 余种方法；按平台追加覆盖（NPU 的 gptq→GPTQAscendConfig、mxfp4→Mxfp4W4A4Config，XPU 的 gptq/awq，MPS 的 mlx_q4/q8）；`CPU_QUANTIZATION_METHODS` 是 AMX CPU 子集。
- **实现逻辑要点**：`get_quantization_config()` 校验方法名、CPU+AMX 时查子集表、out-of-tree 平台可提供自定义 config。

#### quantization/auto_round.py

- **作用**：AutoRound 量化适配层，本身不实现 kernel，而是把 auto_gptq/auto_awq 检查点翻译成 GPTQ/AWQ 配置。
- **关键内容**：`AutoRoundConfig` 支持 2/3/4/8 bit、`backend`（auto/gptq/gptq:marlin/awq/awq:marlin/marlin）、逐层 `extra_config` 正则覆盖与 `block_name_to_quantize`。
- **实现逻辑要点**：核心是 `apply_gptq_quant_layer`/`apply_awq_quant_layer`：按层取位宽后构造对应 Config（支持自动转 Marlin，MoE 不支持 Marlin 时回退 `MoeWNA16Config`）；CPU 路径仅支持 4-bit 且要求 Intel AMX，NPU 走 GPTQAscendConfig。

#### quantization/awq/（AWQ 家族，8 个文件）

- **家族说明**：结构同 gptq/：`awq.py` 放 Config+Method（含 CPU/XPU/Marlin 平台变体 Config），`schemes/` 放各平台 Linear/MoE scheme；Method 类均为薄壳，create/process/apply 全部委托 `layer.scheme`。

#### quantization/awq/__init__.py

- **作用**：转发导出：AWQConfig/AWQCPUConfig/AWQMarlinConfig/AWQXPUConfig、AWQLinearMethod/AWQMoEMethod、各 AWQ scheme 及 awq_dequantize_triton/decomposition。

#### quantization/awq/awq.py

- **作用**：AWQ 的 Config+Method。
- **关键内容**：`AWQConfig` 仅支持 4-bit（pack_factor=8）、group_size/zero_point；`AWQCPUConfig`（Intel AMX）、`AWQXPUConfig`（降级到 torch 原生 int4pack mm，不支持 MoE）；`AWQMarlinConfig` 支持 4/8 bit（uint4/uint8），构造时 `verify_marlin_supported`，`get_quant_method` 对不满足 Marlin 形状的层回退普通 AWQ、MoE 回退 `MoeWNA16Config`，并支持 `override_quantization_method` 自动把 awq 检查点升级为 awq_marlin。
- **备注**：末尾为 `sgl_kernel::awq_marlin_repack` 注册 torch.compile fake。

#### quantization/awq/schemes/__init__.py

- **作用**：转发导出全部 AWQ scheme（AWQLinearSchemeBase/AWQLinearScheme/AWQAscend/XPU/IntelAMX/AWQMarlinLinearScheme/AWQMoEScheme 等）。

#### quantization/awq/schemes/awq_scheme.py

- **作用**：`AWQLinearSchemeBase`/`AWQMoESchemeBase` 抽象类，继承 base_scheme 并重述三/四接口，仅作类型分派用。

#### quantization/awq/schemes/awq_cpu.py

- **关键内容**：`AWQIntelAMXLinearScheme`/`AWQIntelAMXMoEScheme` 继承 GPU 版本，仅把 `_init_kernel` 换成 `hardware_backend/cpu` 的 Intel AMX kernel；MoE 版额外透传 `create_moe_runner`。

#### quantization/awq/schemes/awq_linear.py

- **关键内容**：`AWQLinearScheme`：校验 group_size/pack_factor 对齐后注册 `qweight`/`qzeros`（int32 PackedvLLMParameter，按 8 个 4bit 打包）与 `scales`（GroupQuantScaleParameter），apply 委托 `hardware_backend/gpu` 的 AWQLinearKernel；`AWQAscendLinearScheme`（NPU）/`AWQXPULinearScheme`（XPU）仅替换 kernel 类。

#### quantization/awq/schemes/awq_marlin.py

- **关键内容**：`AWQMarlinLinearScheme`：`create_weights` 先 `verify_marlin_supports_shape`（min_thread_n=64/k=128 等约束），再注册 qweight/qzeros/scales（group_size=-1 时退化为 channelwise），并记录 input/output_size_per_partition、num_groups 供 Marlin kernel 使用；apply 委托 AWQMarlinLinearKernel。

#### quantization/awq/schemes/awq_moe.py

- **关键内容**：`AWQMoEScheme`（Marlin MoE）：注册 w13/w2 的 qweight（transposed 布局）、scales、qzeros，权重标为 `FusedMoeWeightScaleSupported.GROUP`；`create_moe_runner` 固定 `MoeRunner(MARLIN)`，apply 由 AWQMoEKernel 完成。`AWQAscendMoEScheme` 改用 `NPUWNA16Int4MoEMethod` + ASCEND runner，apply 时组 `AscendQuantInfo`。

#### quantization/base_config.py

- **作用**：量化体系抽象基类。
- **关键内容**：`QuantizeMethodBase`（create_weights/apply/process_weights_after_loading）、`LinearMethodBase`（线性层带 partition 尺寸签名）、`FusedMoEMethodBase`（含 `runner` 字段、`create_moe_runner`、`get_triton_quant_info`/`get_moe_quant_info` 供 LoRA/runner 复用）。`QuantizationConfig` 抽象 get_name/get_supported_act_dtypes/get_min_capability/get_config_filenames/from_config/get_quant_method/get_scaled_act_names。
- **实现逻辑要点**：提供 `override_quantization_method`（含 `_modelopt_override_quantization_method` 自动识别 modelopt_fp8/fp4）、`packed_modules_mapping` 更新接口、`get_from_keys(_or)` 配置取值工具和 `method_has_implemented_embedding`。

#### quantization/base_scheme.py

- **作用**：per-format scheme 抽象。
- **关键内容**：`BaseLinearScheme` 定义 create_weights/process_weights_after_loading/apply_weights 三接口，另设 `requires_weight_loader_v2` 标志让 LinearBase 统一路由 v2 loader；`BaseMoEScheme` 额外要求 create_moe_runner。所有方法目录的 scheme 基类都继承它。

#### quantization/bitsandbytes.py

- **作用**：bitsandbytes 集成（要求版本 ≥0.46.1）。
- **关键内容**：`BitsAndBytesConfig` 含 load_in_8bit/4bit 与 bnb_4bit 系列参数。`BitsAndBytesLinearMethod`：8bit 路径用 `Int8Params`+`MatmulLtState` 调 bnb `matmul`，4bit 路径用 uint8 打包权重+`apply_bnb_4bit`（注册为 mutates_args=["out"] 的 custom op，内部调 `matmul_4bit`）；`is_layer_skipped_bnb` 做前缀/子串匹配。
- **备注**：`BitsAndBytesMoEMethod`：8bit 未实现，4bit 在 apply 时 `dequantize_4bit` 出 w13/w2 后走通用 `fused_moe`。

#### quantization/blockwise_int8.py

- **作用**：块级 W8A8 INT8。
- **关键内容**：`BlockInt8Config` 要求 int8 序列化 checkpoint + dynamic activation scheme + 二维 weight_block_size；`BlockInt8LinearMethod` 注册 int8 `weight` 与 fp32 `weight_scale_inv`（BlockQuantScaleParameter），apply 调 `apply_w8a8_block_int8_linear`（per_token_group_quant_int8 + w8a8_block_int8_matmul）；`BlockInt8MoEMethod` 同样注册 w13/w2 权重与 `_scale_inv`，`get_triton_quant_info` 输出 BLOCK 量化信息，runner 固定 TRITON。

#### quantization/compressed_tensors/（compressed-tensors 家族，16 个文件）

- **家族说明**：`compressed_tensors.py` 解析 llm-compressor 的 `config_groups`，按 target（名字/regex/模块类）为不同层配不同格式；`schemes/` 下 13 个 scheme 实现各格式（FP8 家族、WNA16、NVFP4、W4AFP8、W4A8/MXINT4 等），`utils.py` 提供 target/ignore 匹配工具。

#### quantization/compressed_tensors/compressed_tensors.py

- **作用**：compressed-tensors（llm-compressor）全家桶入口，全目录核心。
- **关键内容**：`CompressedTensorsConfig` 解析 `config_groups` 得到 `target_scheme_map`（每 target 的 weights/input_activations QuantizationArgs 与 per-group format），同时解析 sparsity、ignore、`linear_fp8_config`（Linear 用 FP8、MoE 用其他格式的混合量化）、`kv_cache_scheme`。文件含大量 `_is_*` 谓词判定各格式（wNa16、fp8_w8a8/16、static/dynamic token w8a8、fp4a4_nvfp4、wint4afp8、wint4abf16、mxint4a16、mxfp4_moe 等）。
- **实现逻辑要点**：`get_quant_method` 按 Linear/ParallelLMHead/RadixAttention/FusedMoE 分发（MXFP4 MoE 直接给 `Mxfp4MoEMethod`，KV cache 仅支持 static per-tensor FP8）；`get_moe_scheme` 按 runner 后端选择 Marlin/Triton/flashinfer_trtllm mxint4/NPU 变体。`CompressedTensorsLinearMethod`/`FusedMoEMethod` 为薄壳，另有 `apply_weights_with_router_logits`/`apply_without_routing_weights` 透传。

#### quantization/compressed_tensors/schemes/__init__.py

- **作用**：转发导出全部 compressed-tensors scheme（W8A8Fp8/Fp8MoE、W8A16Fp8、W8A8Int8 系列、WNA16 系列、W4A4Fp4/Nvfp4MoE、MxInt4MoE、W4AFP8MoE 等）。

#### quantization/compressed_tensors/schemes/compressed_tensors_scheme.py

- **作用**：`CompressedTensorsLinearScheme`（含类方法 get_min_capability）与 `CompressedTensorsMoEScheme`（含 `load_up_proj_weight_first` 类属性，控制 gate/up 加载顺序）两个抽象基类。

#### quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py

- **关键内容**：`CompressedTensorsW8A8Fp8`：支持 CHANNEL/TENSOR/BLOCK 三种 weight strategy；BLOCK 用 BlockQuantScaleParameter 且 apply 走 `dispatch_w8a8_block_fp8_linear`（按初始化 config 选 cutlass/flashinfer/deepgemm 等）。
- **实现逻辑要点**：process 中 TENSOR 用 `requantize_with_max_scale` 合并、fnuz（ROCm）归一化、`SGLANG_USE_AITER` 时 `shuffle_weight`、BLOCK 且在 Blackwell DeepGEMM runner 时 `requant_block_scale_ue8m0_for_deepgemm` 把 scale 转为 UE8M0；apply 支持融合 RMSNorm 产出的预量化 tuple 输入。

#### quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py

- **关键内容**：`CompressedTensorsW8A8Fp8MoE`：支持 per-tensor/per-channel/block 三种 weight strategy 的 FP8 MoE。
- **实现逻辑要点**：process 合并静态输入 scale 为 max、fnuz 归一化、per-tensor 时逐专家反量化再量化、aiter 时 shuffle_weight、flashinfer_trtllm 时 `swap_w13_to_w31`；runner 可为 AITER/TRITON/FLASHINFER_TRTLLM；apply 依 strategy/后端组 `TritonMoeQuantInfo`/`AiterMoeQuantInfo`/`FlashInferTrtllmFp8MoeQuantInfo`。

#### quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py

- **关键内容**：`CompressedTensorsW8A16Fp8`：FP8 权重-only 量化（activation 保持 BF16/FP16）。支持 CHANNEL/TENSOR 两种 strategy；process 时 TENSOR 用 `convert_to_channelwise` 展开、权重转置后 `prepare_fp8_layer_for_marlin`；apply 走 `apply_fp8_marlin_linear`（Marlin 的 FP8 权重-only GEMM，主要用于无 FP8 硬件的老 GPU）。

#### quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py

- **关键内容**：`CompressedTensorsW8A8Int8`：CHANNEL/TENSOR int8 权重；静态输入时额外加载 `input_zero_point`（AZP），process 里重建 range 并计算 `azp_adj`（权重和调整项，参照 CUTLASS w8a8 epilogue 文档）；apply 用 `per_token_quant_int8` + `int8_scaled_mm`。`NPUCompressedTensorsW8A8Int8` 委托 `NPUW8A8Int8DynamicLinearMethod`（暂不支持静态）。

#### quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py

- **关键内容**：`NPUCompressedTensorsW8A8Int8DynamicMoE`（Ascend）：仅支持 channelwise 权重 + 动态 per-token 激活；注册 int8 w13/w2 权重与 per-channel scale，`NPUW8A8Int8MoEMethod` 处理后走 ASCEND runner + `AscendQuantInfo`。

#### quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py

- **关键内容**：`CompressedTensorsWNA16`：权重-only 2/4/8 bit 量化（对称 4→uint4b8、8→uint8b128，非对称 →uint4/uint8+zero point），支持 actorder(g_idx)。`create_weights` 构造 `MarlinLinearLayerConfig` 并注册 weight_packed/weight_scale/weight_shape（及 weight_zero_point/weight_g_idx）；`process_weights_after_loading` 用 `gptq_marlin_repack`/`marlin_permute_scales`/`marlin_zero_points` 变换成 Marlin 布局；apply 调 `apply_gptq_marlin_linear`。文件头导出 `WNA16_SUPPORTED_BITS=[4,8]`。

#### quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py

- **关键内容**：`CompressedTensorsWNA16MoE`：pack_quantized 的 wNa16 MoE。process 做 `gptq_marlin_moe_repack`、`marlin_moe_permute_scales`、`moe_awq_to_marlin_zero_points`，记录 `_original_shapes` 供 `restore_weights_before_loading`（hot reload 回滚），`is_marlin_converted` 防重复打包；runner=MARLIN，`get_marlin_quant_info` 输出 g_idx/zeros 等信息。
- **备注**：`CompressedTensorsWNA16TritonMoE` 改为转置成 uint8 打包布局走 TRITON；`NPUCompressedTensorsW4A16Int4DynamicMoE` 走 ASCEND。

#### quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py

- **关键内容**：`CompressedTensorsMxInt4MoE`：MXINT4（int4、group_size=32、pack_quantized）MoE，仅支持 flashinfer_trtllm 后端。
- **实现逻辑要点**：`process_weights_after_loading` 把 HF 的 (w/s+8) 格式 repack 成 TRT-LLM 格式（`prepare_static_weights_for_kernel`：nibble 重排、w3/w1 与 scale 的 permute 索引缓存、`block_scale_interleave`、`convert_to_block_layout`）；apply 调 `trtllm_mxint4_block_scale_moe`，DeepSeekV3 路由需 fp32 router logits，输出走对称内存分配。

#### quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py

- **关键内容**：`CompressedTensorsW4A4Fp4`（Linear）：NVFP4 双级量化（uint8 packed fp4 权重 + fp8 e4m3 per-group16 scale + fp32 global scale）。
- **实现逻辑要点**：process 时合并全局 scale 为标量、按 FP4 GEMM 后端选择 `shuffle_matrix_a`/`shuffle_matrix_sf_a`（FlashInfer TRTLLM 布局）或 `swizzle_blockscale`（CUTLASS 布局），并计算 `alpha = 1/(input_global_scale × weight_global_scale)`；apply 用 `fp4_quantize` 量化激活后调 `fp4_gemm`。

#### quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py

- **关键内容**：`CompressedTensorsW4A4Nvfp4MoE`：NVFP4 MoE。create_weights 注册 w13/w2 packed uint8 权重、group16 fp8 块 scale、每专家 global scale 与 input global scale；`load_up_proj_weight_first` 依后端（CUTLASS 需 [up;gate]，TRT-LLM 加载后重排）。
- **实现逻辑要点**：process 计算 `g1_alphas`/`g2_alphas`、`w13/w2_input_scale_quant`，TRTLLM 后端做 `prepare_static_weights_for_trtllm_fp4_moe` 洗牌并生成 `g1_scale_c`，CUTLASS 后端则 `swizzle_blockscale`；runner 对应 FLASHINFER_TRTLLM 或 FLASHINFER_CUTLASS。

#### quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py

- **关键内容**：`CompressedTensorsW4AFP8MoE`：INT4 group 权重 + 动态 8bit 激活，走 CUTLASS W4A8 grouped GEMM。
- **实现逻辑要点**：`_unpack_repack_int32_to_cutlass_int8` 把 pack_to_int32 的 8 个无符号 int4 重排为 CUTLASS 的成对符号 int4→int8 打包；process 时 `interleave_scales` 并设置 `is_w4afp8_converted` 幂等标志；`_init_cutlass_buffers` 预分配 stride/problem_sizes 缓冲；apply 调 `cutlass_w4a8_moe`（仅 SiLU，group_size 硬编码 128）。

#### quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int8_moe.py

- **关键内容**：`NPUCompressedTensorsW4A8Int8DynamicMoE`（Ascend 专用）：per-channel int8 权重 + scale + offset；支持 `activation_use_clip`（注册 w13/w2_bias、w2_alpha）或二次量化参数（weight_scale/offset_second、scale_bias）；用 `NPUW4A8Int8MoEMethod` 做权重后处理，runner 为 ASCEND，apply 组 `AscendQuantInfo`。

#### quantization/compressed_tensors/utils.py

- **作用**：compressed-tensors 的 target/ignore 匹配工具。
- **关键内容**：`check_equal_or_regex_match`（`re:` 前缀正则，普通 target 支持点分路径后缀匹配、禁止前缀误匹配）、`should_ignore_layer`（fused 层拆 shard 逐项检查并要求各 shard 方案一致）、`find_matched_target`（依次按层名、模块类名、fused 映射匹配）、`is_activation_quantization_format`（列出含激活量化的 5 种 CompressionFormat）。

#### quantization/dequantization.py

- **作用**：权重反量化工具。
- **关键内容**：`dequantize_fp8`（`block_quant_dequant` 到 bf16，UE8M0 scale 先 `inverse_transform_scale_ue8m0`）；`dequantize_nvfp4`（用内置 E2M1 16 值 LUT 解码打包 nibble，fp8 块 scale × 可选全局 scale，支持 high_nibble_first）；`copy_missing_attrs`。

#### quantization/expert_pack.py

- **作用**：GGUF 稠密权重 + 流式 GGUF-MXFP4 routed experts 混合方案。
- **关键内容**：`ExpertPackConfig` 继承 GGUFConfig，对 FusedMoE 返回 `ExpertPackMoEMethod`（不注册常规权重参数，仅注册非持久 buffer 标记）；apply 时从 `ExpertPackStore` 按 topk 获取专家槽位，用 `mxfp4_matvec`/`mxfp4_matvec_dual` kernel 计算。
- **备注**：`KimiGGMLExpertPackStore` 路径用 `ggml_moe_a8_vec` + SiTU 激活（β=4.0/25.0 常量校验）。

#### quantization/fp4_kv_cache_quant_method.py

- **作用**：KV cache 量化的策略模式（三玩家设计：quant_method 纯计算 → Pool 管 buffer → Backend 做视图适配）。
- **关键内容**：`KV_CACHE_QUANT_REGISTRY` 注册 cpu_fp8_e4m3/nvfp4/fp4_mx_block16 三种方法；`KV_CACHE_ATTENTION_ACCESS_REGISTRY` 为每个 recipe 声明 prefill/decode 各 attention 后端的访问方式（PLAIN/DEQUANT_WORKSPACE/NATIVE_FP4，例如 NVFP4：prefill 用 flashinfer 反量化 workspace、decode 由 trtllm_mha 原生消费 FP4）。`KVCacheQuantMethodBase` 抽象 create_buffers/quantize_and_store/dequantize_prev_kv/compute_cell_size；`NVFP4KVCacheMethod.load_scales_from_model` 从每层 k_scale/v_scale 取全局 scale（SM100 乘 E2M1_MAX=6.0 桥接 TRT-LLM）；`resolve_kv_cache_quant` 校验并拒绝过时的 `fp4_e2m1`/`mxfp4` 取值。

#### quantization/fp4_utils.py

- **作用**：FP4 GEMM 公共配置。
- **关键内容**：`fp4_quantize` 封装 flashinfer 的 fp4_quantize（注册 custom op + torch.compile fake，SM100 用 cute-dsl 后端）；`Fp4GemmRunnerBackend` 枚举 AUTO/FLASHINFER_CUDNN/FLASHINFER_CUTEDSL/FLASHINFER_CUTLASS/FLASHINFER_TRTLLM/MARLIN，`initialize_fp4_gemm_config` 按平台自动选择（SM100→cutedsl、Ampere~CUDA→marlin、其他→cutlass），`get_fp4_gemm_runner_backend` 供 scheme 查询。

#### quantization/fp8.py

- **作用**：FP8 量化主文件（全目录最大）。
- **关键内容**：`cast_e2m1fn_to_e4m3fn`（DeepSeek-V4 的 fp4 专家转 fp8 无损转换）；`Fp8Config` 支持 activation_scheme、weight_block_size、`use_mxfp8`（block 固定 [1,32]）、`is_fp4_experts`（DeepSeek-V4 MXFP4 专家），get_quant_method 在 is_fp4_experts 时按 MoE runner 后端分发到 Mxfp4Marlin/Humming/FlashinferCUTLASS/TRTLLM 变体，RadixAttention 给 Fp8KVCacheMethod。`Fp8LinearMethod` 提供可复用的静态 `create_fp8_weight_`，apply 覆盖 marlin/mxfp8/block/普通 scaled_mm 四条路径。
- **实现逻辑要点**：`Fp8MoEMethod` 提供静态 `create_fp8_moe_weight_`，process 处理在线量化、fnuz 归一化、w13 双 shard scale 合并、DeepGEMM UE8M0、FlashInfer-TRTLLM 权重对齐、hip int4/hpc_ops；apply 按 TRITON/DeepGEMM/FlashInferTRTLLM/CUTLASS/AITER/XPU/CPU-AMX 后端分发 quant info。`Fp8KVCacheMethod` 继承 BaseKVCacheMethod。

#### quantization/fp8_utils.py

- **作用**：FP8 工具集与 GEMM dispatch 中枢。
- **关键内容**：`Fp8GemmRunnerBackend`/`Mxfp8DenseGemmBackend` 枚举 + `initialize_fp8_gemm_config` 全局初始化；`dispatch_w8a8_block_fp8_linear`/`dispatch_w8a8_mxfp8_linear` 按配置解析具体实现（cutlass/flashinfer/deepgemm/aiter/triton，含 fallback 链）；`apply_fp8_linear`（支持预量化 tuple 输入、`torch._scaled_mm` 融合 GEMM、aiter bpreshuffle、per-token-per-channel 路径）、`apply_fp8_ptpc_linear`。
- **备注**：量化/反量化工具：`input_to_float8`、`mxfp8_group_quantize`、`dequant_mxfp4`、`block_quant_dequant`、`requant_weight_ue8m0`（DeepGEMM UE8M0 重量化 + TMA 对齐）、`normalize_e4m3fn_to_e4m3fnuz`、`bmm_fp8`、`validate_fp8_block_shape` 等。

#### quantization/gguf.py

- **作用**：llama.cpp GGUF 格式量化。
- **关键内容**：`GGUFConfig`+`GGUFLinearMethod`（`GGUFUninitializedParameter`+`qweight_type`，process 时对融合层做 padding 并建 `shard_offset_map`，apply 调 `fused_mul_mat_gguf`）；`fused_mul_mat_gguf` 按量化类型分派 MMVQ（`ggml_mul_mat_vec_a8`，batch≤阈值）、MMQ（`ggml_mul_mat_a8`）或反量化回退；`GGUFMoEMethod` 用 `fused_moe_gguf`（`ggml_moe_a8`/`ggml_moe_a8_vec` 或逐 token 回退）；`GGUFEmbeddingMethod`。
- **备注**：类型集合区分 STANDARD_QUANT_TYPES/KQUANT_TYPES/IMATRIX_QUANT_TYPES；文件后半是 Ascend 变体（CPU 上用 gguf 库参考实现反量化）。

#### quantization/gptq/（GPTQ 家族，8 个文件）

- **家族说明**：与 awq/ 同构：`gptq.py` 放 Config+Method（含 Ascend/CPU/XPU/Marlin 变体），`schemes/` 放各平台 scheme；Method 类均为薄壳委托 `layer.scheme`。

#### quantization/gptq/__init__.py

- **作用**：转发导出 GPTQ 全家族：GPTQConfig/GPTQAscendConfig/CPUGPTQConfig/GPTQMarlinConfig/GPTQXPUConfig、四种 Method、全部 scheme 与 check_marlin_format。

#### quantization/gptq/gptq.py

- **作用**：GPTQ 的 Config+Method。
- **关键内容**：`GPTQConfig` 支持 2/3/4/8 bit、`dynamic` 正则逐模块配置（`+:/-:` 前缀）、checkpoint_format（gptq_v2），普通 GPTQ 不支持 MoE；`GPTQAscendConfig`/`CPUGPTQConfig`/`GPTQXPUConfig` 为平台变体；`GPTQMarlinConfig` TYPE_MAP 仅 (4,True)→uint4b8、(8,True)→uint8b128，`override_quantization_method` 自动把兼容检查点升级为 gptq_marlin。
- **备注**：末尾为 gptq_gemm/gptq_marlin_repack/gptq_shuffle 注册 torch.compile fake。

#### quantization/gptq/schemes/__init__.py

- **作用**：转发导出全部 GPTQ scheme。

#### quantization/gptq/schemes/gptq_scheme.py

- **作用**：`GPTQLinearSchemeBase`/`GPTQMoESchemeBase` 抽象类，与 AWQ 版本同构。

#### quantization/gptq/schemes/gptq_cpu.py

- **关键内容**：`GPTQIntelAMXLinearScheme`（先校验：仅 4bit、desc_act 需 true_sequential+static_groups、不支持 gptq_v2，再用 AMX kernel）与 `GPTQIntelAMXMoEScheme`。

#### quantization/gptq/schemes/gptq_linear.py

- **关键内容**：`GPTQLinearScheme`：注册 qweight（int32 PackedvLLMParameter，packed_dim=0）、g_idx、qzeros、scales（desc_act 或行并行时 scale 按 partition 分割且禁用 shuffle）；apply 委托 GPU GPTQLinearKernel。`GPTQAscendLinearScheme`（禁 desc_act）、`GPTQXPULinearScheme` 仅替换 kernel。

#### quantization/gptq/schemes/gptq_marlin.py

- **关键内容**：`GPTQMarlinLinearScheme`：构造 `MarlinLinearLayerConfig`，按 `marlin_repeat_scales_on_all_ranks` 决定 scales/qzeros 是否按 partition 切分，注册 qweight/g_idx/scales/qzeros；apply 委托 GPTQMarlinLinearKernel。

#### quantization/gptq/schemes/gptq_moe.py

- **关键内容**：`GPTQMoEAscendScheme`（Ascend：w13/w2 qweight+scales+qzeros，NPUWNA16Int4MoEMethod + ASCEND runner）与 `GPTQMarlinMoEScheme`（GPU Marlin MoE：transposed 布局权重，desc_act 时 w2 scales/qzeros 全量加载，注册 g_idx/g_idx_sort_indices，委托 GPTQMarlinMoEKernel）。

#### quantization/humming.py

- **作用**：inclusionAI humming 库的集成（W4A8/W4AFP8/block-fp8 等格式的 PyTorch 端）。
- **关键内容**：`HummingConfig` 惰性导入 humming，解析 compressed-tensors/modelopt/w4afp8 配置为 `BaseWeightSchema`，支持 `is_layer_skipped`（re:/dynamic 负匹配）、`SGLANG_HUMMING_ONLINE_QUANT_CONFIG` 在线量化；内置 `_W4AFp8CheckpointWeightSchema`（把检查点转 humming 格式）、`_StackedBlockFp8CheckpointWeightSchema`。
- **实现逻辑要点**：`HummingLinearMethod`：weight_loader 内联量化（`quantize_weight`）、未量化层 fallback 时原位改造成 `UnquantizedLinearMethod`，process 用 `HummingMethod.prepare_layer_meta`+`transform_humming_layer` 转格式，apply 调 `HummingMethod.forward_layer`；`HummingMoEMethod` 按 w13/w2 子层同样处理，支持 force requant。

#### quantization/humming_utils.py

- **作用**：humming 的 MoE 辅助。
- **关键内容**：`humming_is_layer_skipped`（ignore 列表 + dynamic 负正则）；`configure_humming_deepep_dispatch`（DeepEP 时按配置设 dispatcher output dtype 为 fp8 或 bf16）；`make_humming_deepep_input_schema`（fp8 输入 schema，K 须整除 128）；`prepare_humming_moe_layer`（把整层转换为 humming 标准格式并注册 locks buffer）。

#### quantization/int8_utils.py

- **作用**：INT8 块量化 GEMM 工具。
- **关键内容**：`apply_w8a8_block_int8_linear`（`per_token_group_quant_int8` 量化激活 + `w8a8_block_int8_matmul`）；`block_dequant`（逐 tile 乘 scale 的反量化）。

#### quantization/kv_cache.py

- **关键内容**：`BaseKVCacheMethod`：为 attention 层创建 `k_scale`/`v_scale` 参数（初始 -1.0 哨兵、`_skip_weight_check`），`process_weights_after_loading` 处理三种情形：分别加载的 k/v scale、都无效时默认 1.0、只有单 kv_scale 时复制到 v；fnuz 平台 scale 乘 2；最终写入 `k_scale_float`/`v_scale_float` 供 attention 前向使用。

#### quantization/kvfp4_tensor.py

- **作用**：KV cache FP4 量化内核实现。
- **关键内容**：`FP4MXBlock16KVQuantizeUtil`：block16 E2M1 量化（块内 absmax→uint8 指数 scale=ceil(log2(max/6))+127），`@torch.compile` 的 batched_quantize/batched_dequantize（CUDA graph 安全，纯 tensor 实现）；`NVFP4KVQuantizeUtil`：两级 scale（fp32 全局 + fp8 e4m3 块16），SM100/SM120 用 flashinfer 的 `nvfp4_kv_quantize`/`nvfp4_kv_dequantize`，SM90 回退 `fp4_quantize` 或纯 PyTorch LUT；文件头定义 E2M1_VALUES/E2M1_BOUNDS/E2M1_MAX 常量。

#### quantization/marlin_utils.py

- **作用**：Marlin kernel 公共工具。
- **关键内容**：`MarlinLinearLayerConfig`；`query_marlin_supported_quant_types`（带 zp 仅 uint4；不带 zp 有 uint4b8/uint8b128/fp8/fp4，group_size 限 -1/32/64/128，fp4 限 16/32）；`verify_marlin_supports_shape`（min_thread_n=64、min_thread_k=128）；`check_moe_marlin_supports_layer`（形状/group_size/激活/路由约束）；repack 工具 `marlin_permute_scales`/`marlin_moe_permute_scales`/`marlin_zero_points`/`awq_to_marlin_zero_points`/`marlin_sort_g_idx`/`marlin_make_workspace`。
- **实现逻辑要点**：`apply_gptq_marlin_linear`/`apply_awq_marlin_linear`（直接调 gptq_marlin_gemm，CUDA graph piecewise 上下文下走 `unified_apply_gptq_marlin_gemm` custom op）；`should_use_atomic_add_reduce` 按形状选择原子归约。

#### quantization/marlin_utils_fp4.py

- **作用**：为 Marlin 后端准备 NVFP4 与 MXFP4 权重（dense 与 MoE）的 repack/scale 处理工具。
- **关键内容**：`nvfp4_marlin_process_scales`/`nvfp4_marlin_process_global_scale`（NVFP4 E4M3 块 scale 转 Marlin 布局、fp32 global scale 指数偏置）；`apply_fp4_marlin_linear`（`gptq_marlin_gemm` 自定义算子，`b_q_type=float4_e2m1f`）；`prepare_nvfp4_layer_for_marlin`（group_size=16，pad N/K 后 `gptq_marlin_repack`）；`mxfp4_marlin_process_scales`（uint8→E8M0 重解释 + fp8 输入时 +6 指数偏置）；`prepare_moe_mxfp4_layer_for_marlin`（group_size=32）与 `prepare_moe_nvfp4_layer_for_marlin`（group_size=16），逐 expert repack 并释放 `weight_scale_inv`。
- **实现逻辑要点**：NVFP4 scale 被 `*2**7` 转 int16 后重解释为 e4m3fn 取奇数位；MXFP4 scale 重解释为 `float8_e8m0fnu`；Marlin 要求 N/K 对齐 64/128，TP 分片需 pad。
- **备注**：另含 `deinterleave_moe_w13_w4a4_for_marlin`（GPT-OSS gate/up 行交错转连续两半）。

#### quantization/marlin_utils_fp8.py

- **作用**：无 FP8 硬件加速的 GPU 上用 Marlin kernel 做 weight-only FP8 推理的工具。
- **关键内容**：`fp8_fused_exponent_bias_into_scales`（按 FP16/BF16 目标指数给 scale 乘 2^bias）；`apply_fp8_marlin_linear`（`gptq_marlin_gemm`，`b_q_type=float8_e4m3fn`，带 fake_impl）；`prepare_fp8_layer_for_marlin`（`pack_fp8_to_int32` 打包 + repack + scale 置换）。
- **实现逻辑要点**：支持 tensor-wise/channel-wise/block-wise scale 到 Marlin channel/group-wise 的转换（scale 按 `weight_block_size` 展开）；加载时打印"性能可能下降"警告。
- **备注**：bias 保持逻辑输出通道顺序，不做 Marlin tile 置换。

#### quantization/mlx.py

- **作用**：Apple MLX 后端的量化标记配置与 HF 仓库自动探测。
- **关键内容**：`MlxQuantizationConfig`（preset 为 `mlx_q4`/`mlx_q8`），`override_quantization_method` 识别 mlx-community 的 `{"group_size", "bits"}` 无 `quant_method` 配置并映射到对应 preset。
- **实现逻辑要点**：真实量化由 `mlx_lm.utils.quantize_model` 在硬件后端完成；PyTorch 路径的 `from_config`/`get_quant_method` 抛 `NotImplementedError` 并提示设置 `SGLANG_USE_MLX=1`；CLI 显式 `--quantization` 优先于自动探测。

#### quantization/modelopt_quant.py

- **作用**：NVIDIA ModelOpt 量化（FP8、NVFP4、混合精度）的配置与方法集，本批最大文件（3032 行）。
- **关键内容**：`ModelOptQuantConfig` 基类（排除层、kv_cache_quant_algo、`_modelopt_override_quantization_method`）；`ModelOptFp8Config/LinearMethod/KVCacheMethod/MoEMethod`；`ModelOptFp4Config`（4 条路径：序列化+per-tensor FP32 激活 scale、序列化+per-token、BF16/FP16/FP8 MoE 加载期量化、`nvfp4_online`）；`ModelOptFp4LinearMethod`、`ModelOptNvFp4A16LinearMethod`（W4A16_NVFP4）；`ModelOptNvFp4FusedMoEMethod`（Blackwell 或 SM80+ Marlin fallback，含 cutedsl v1/v2 路径）；`ModelOptNvFp4EmbeddingMethod`；`HybridFp8NvFp4Config`（FP8 线性层 + NVFP4 MoE，如 DeepSeek-V4-Pro-NVFP4）；`ModelOptMixedPrecisionConfig`。
- **实现逻辑要点**：`fp4_gemm` 自定义算子优先尝试 `try_qwen3x_nvfp4_gemm`（KDA），否则走 FlashInfer `mm_fp4`（w4a4）或 `mm_bf16_fp4`（w4a16）；`pad_nvfp4_weight` 按 CUTLASS(N/K%32) 与 TRTLLM(N%128) 对齐；序列化 NVFP4 权重为 [N, K/2] 打包格式 + E4M3 块 scale（group 16）+ FP32 per-tensor scale。

#### quantization/modelslim/（ModelSlim 家族，14 个文件）

- **家族说明**：华为 ModelSlim 量化，Ascend NPU 专属：`modelslim.py` 是 scheme 调度总入口，`schemes/`（13 个）按 `quant_model_description.json` 的 scheme 名实现各线性/MoE 格式。

#### quantization/modelslim/modelslim.py

- **作用**：华为 ModelSlim 量化的总入口，NPU 专属的 scheme 调度框架。
- **关键内容**：`ModelSlimConfig`（`quant_model_description.json`，DeepSeek-V4 hc_head 前缀重映射、DSpark MTP `mtp.*`→`stages.*` 别名、`block_sparse_moe`/`mlp` 双命名）；`get_linear_scheme` 匹配 W4A4_DYNAMIC/W8A8/W8A8_DYNAMIC/W8A8_MXFP8/W4A8_MXFP/W4A4_MXFP4；`get_moe_scheme` 匹配 6 种 MoE scheme 并支持 `gate_proj/up_proj` 与 `w1/w3` 两套命名；`ModelSlimLinearMethod`（委托 `layer.scheme`）与 `ModelSlimFusedMoEMethod`（Ascend runner + `AscendQuantInfo`）。
- **实现逻辑要点**：`npu_wrapper_rmsnorm_init/forward` 为 w8a8_int8 在 RMSNorm 中加 bias；`is_layer_skipped` 判断 `quant_description` 中 weight 是否为 "FLOAT"。

#### quantization/modelslim/schemes/__init__.py

- **作用**：转发型包初始化，按依赖顺序导出 10 个 scheme 类（基类必须最先导入避免循环依赖）。

#### quantization/modelslim/schemes/modelslim_scheme.py

- **作用**：ModelSlim 的线性/MoE scheme 抽象基类。
- **关键内容**：`ModelSlimLinearScheme`（抽象 `create_weights`/`process_weights_after_loading`/`apply_weights`）、`ModelSlimMoEScheme`（抽象 `create_weights`/`process_weights_after_loading`），继承自 `BaseLinearScheme`/`BaseMoEScheme`。

#### quantization/modelslim/schemes/modelslim_w4a4_int4.py

- **关键内容**：`ModelSlimW4A4Int4`：W4A4_DYNAMIC int4 线性 scheme（NPU）。创建 int8 权重 + per-channel `weight_scale`/`weight_offset`，委托 `NPU_W4A4DynamicLinearMethod`；受 `SGLANG_NPU_W4A4_NEW_PACKING` 环境变量控制新打包（output_size 减半 + pack_factor=2）。

#### quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py

- **关键内容**：`ModelSlimW4A4Int4MoE`：W4A4 int4 MoE scheme（NPU）。按 w13/w2 前缀创建 int8 权重、fp32 scale 与 offset（CHANNEL quant method），委托 `NPUW4A4Int4MoEMethod`；新打包路径下 w13 权重 out_features 减半、scale 形状 [E, 2*out, 1]。

#### quantization/modelslim/schemes/modelslim_w4a4_mxfp4_moe.py

- **关键内容**：`ModelSlimW4A4MXFP4MoE`：W4A4_MXFP4 MoE scheme（NPU）。创建 uint8 打包 FP4 权重 [E, N, K//2]（BLOCK quant method）与 UE8M0 块 scale [E, N, ceil(K/32)]，委托 `NPUW4A4MXFP4MoEMethod`。

#### quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py

- **关键内容**：`ModelSlimW4A8Int8MoE`：W4A8 int8 MoE scheme（NPU）。创建 int8 权重、fp32 scale/offset 及可选 per-group 二级 scale/offset（`weight_scale_second`/`weight_offset_second`）与 `scale_bias`（activation clip），委托 `NPUW4A8Int8MoEMethod`（可传 `is_per_channel_weight`、`activation_use_clip`）。

#### quantization/modelslim/schemes/modelslim_w4a8_mxfp4_moe.py

- **关键内容**：`ModelSlimW4A8MXFP4MoE`：W4A8_MXFP MoE scheme（NPU）。创建 uint8 打包 FP4 权重 [E, N, K//2] 与 UE8M0 scale [E, N, ceil(K/32)]（BLOCK quant method），委托 `NPUW4A8MXFP4MoEMethod`。

#### quantization/modelslim/schemes/modelslim_w8a8_int8.py

- **关键内容**：`ModelSlimW8A8Int8`：W8A8/W8A8_DYNAMIC int8 线性 scheme（NPU）。创建 int8 权重、channel scale/offset；静态（非 dynamic）时额外创建 `input_scale`/`input_offset`、int32 `quant_bias` 与 `deq_scale`（dtype 随 params_dtype 选 fp32/int64）；委托 `NPUW8A8Int8LinearMethod` 或 `NPUW8A8Int8DynamicLinearMethod`。

#### quantization/modelslim/schemes/modelslim_w8a8_int8_moe.py

- **关键内容**：`ModelSlimW8A8Int8MoE`：W8A8 int8 MoE scheme（NPU）。创建 int8 权重 [E, N, K]、fp32 channel scale 与 offset，委托 `NPUW8A8Int8MoEMethod`。

#### quantization/modelslim/schemes/modelslim_mxfp8.py

- **关键内容**：`ModelSlimMXFP8Scheme`：W8A8_MXFP8 离线 scheme（e4m3fn 权重 + uint8 E8M0 块 scale）。创建 float8_e4m3fn 权重 [out, in] 与 uint8 scale [out, in/32]，委托 `NPUMXFP8LinearMethod`；其 `process_weights_after_loading` 识别预量化 e4m3fn 权重后走 offline（仅转置）分支。

#### quantization/modelslim/schemes/modelslim_mxfp8_moe.py

- **关键内容**：`ModelSlimMXFP8MoEScheme`：MXFP8 MoE 离线 scheme（NPU）。按 `weight_prefix`（"w13"/"w2"）创建 [E, N, K] e4m3fn 权重与 [E, N, K//32] uint8 块 scale，委托 `NPUMXFP8MoEMethod`；注册 `weight_offset=None` 占位以适配统一 `AscendQuantInfo`。

#### quantization/modelslim/schemes/modelslim_mxfp4.py

- **关键内容**：`ModelSlimMXFP4Scheme`：W4A4_MXFP4 离线 scheme（Ascend NPU 单级 MXFP4）。创建 uint8 打包 FP4 权重 [out, in//2] 与 UE8M0 块 scale [out, in//32]（group=32），委托 `NPUSingleLevelMXFP4OfflineLinearMethod` 做转置与 `npu_quant_matmul`（`x1_dtype=x2_dtype=float4_e2m1fn_x2`）。
- **实现逻辑要点**：与在线单级 MXFP4 线性方法共享同一 matmul，仅权重来源（checkpoint 打包 FP4）不同。

#### quantization/modelslim/schemes/modelslim_mxfp4_w4a8.py

- **关键内容**：`ModelSlimMXFP4W4A8Scheme`：W4A8_MXFP 离线 scheme（打包 FP4 权重 + MXFP8 激活）。创建 uint8 权重 [out, in//2] 与 UE8M0 scale [out, in//32]，委托 `NPUMXFP4W4A8OfflineLinearMethod`：`npu_format_cast` 转 FRACTAL_NZ 后以 `x2_dtype=float4_e2m1fn_x2` 执行 matmul（group_sizes=[0,0,32]）。
- **备注**：需较新 torch_npu（Ascend 950/A5）以支持 FP4 NZ matmul。

#### quantization/moe_wna16.py

- **作用**：MoE WNA16（W8A16/W4A16）量化，兼容 GPTQ/AWQ 权重格式。
- **关键内容**：`MoeWNA16Config`（`quantize_config.json`，`linear_quant_method` 仅支持 gptq/awq；linear 层可走 GPTQMarlin/GPTQ/AWQ config，FusedMoE 走 `MoeWNA16Method`）；`get_weight_perm`（生成 GPTQ 重排 perm）；`MoeWNA16Method.create_weights` 创建 uint8 打包 `w13_qweight`/`w2_qweight`（hidden//bit8_pack_factor）、scales、可选 qzeros。
- **实现逻辑要点**：group_size 自动折半直到整除 intermediate/hidden（下限 32）；`get_weight_loader` 把 AWQ int32 打包解包重排、GPTQ qzeros +1 对齐 AWQ 约定后写入标准格式；`apply` 构造 `TritonMoeQuantInfo` 走 TRITON runner，仅支持 SiLU。
- **备注**：`override_quantization_method` 在 `user_quant == "moe_wna16"` 且配置兼容（GPTQ 无 desc_act 4/8bit，或 AWQ 4bit 且满足算力）时接管。

#### quantization/mxfp4.py

- **作用**：OCP MXFP4 量化的核心文件，MoE 按平台/后端分发到各 kernel。
- **关键内容**：`Mxfp4Config`（`mxfp4`，序列化判断、gfx95 检查）；`Mxfp4MoEMethod`（序列化 checkpoint：`create_weights` 按 `_fi_kernel` 选择 cutlass_sm90/cutlass_sm120/trtllm_sm100/marlin/aiter/triton-kernels/deep_gemm 布局；`apply` 依序分发 use_deep_gemm→CPU AMX→XPU→use_marlin→cutlass_sm90→cutlass_sm120→use_flashinfer(trtllm gen)→_use_aiter→triton_kernels/triton）；`Mxfp4DynamicQuantMoEMethod`（在线量化：加载后 `dynamic_mxfp4_quant` + `e8m0_shuffle`，gfx95 上 `shuffle_weight(16,16)`）。
- **实现逻辑要点**：权重打包为 uint8 [E, N, K//2]，scale 为 E8M0/UE8M0（127=1.0）；AITER 路径把 "silu" 翻译为 "swiglu"；`_UE8M0_ONE=127` 用于 padding。辅助函数 `dequant_mxfp4`（amd-quark 的 `mx.dq_mxfp4`）、`_swizzle_mxfp4`（triton_kernels OAI layout）、`_prepare_flashinfer_mxfp8_activations`（route_quant_handoff 或 FlashInfer/per-token-group 量化）。

#### quantization/mxfp4_flashinfer_cutlass_moe.py

- **作用**：DeepSeek-V4 MXFP4 expert 的 FlashInfer CUTLASS MoE 后端（SM90 W4A16/W4A8、SM120 W4A8）。
- **关键内容**：`Mxfp4FlashinferCutlassMoEMethod`：委托 FP8 基础 method 创建权重（scale dtype 保持原生 `float8_e8m0fnu`）；`process_weights_after_loading` 中 SM120 走 `block_scale_interleave`，SM90 走 `interleave_moe_weights/scales_for_sm90_mixed_gemm`（或 humming 精度时 `preprocess_moe_weights_for_sm90_mixed_gemm_humming` + residual scale）。
- **实现逻辑要点**：hidden/intermediate 须 128 对齐；`load_up_proj_weight_first=True`（[up;gate] 顺序）；SM120 注册 per-expert global scale；SwigluStep 变体支持 `gemm1_clamp_limit`。

#### quantization/mxfp4_flashinfer_trtllm_moe.py

- **作用**：SM100 上 MXFP4 expert 的 FlashInfer TRT-LLM 后端。
- **关键内容**：`Mxfp4FlashinferTrtllmMoEMethod`：`process_weights_after_loading` 先 `reorder_w1w3_to_w3w1`，再逐 expert 用 `_maybe_get_cached_w3_w1_permute_indices`/`get_w2_permute_indices_with_cache` + `block_scale_interleave` 打散权重与 scale（可用 `shuffle_matrix_a/sf_a` 旧路径，由 `SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE` 控制）；`apply` 直接调用 `trtllm_fp4_block_scale_routed_moe`（不走 MoeRunner）。
- **实现逻辑要点**：激活按 precision=bf16 直接传入或 `flashinfer_mxfp8_quantize`（cute-dsl/cuda 后端）量化；注册 `output1/2_scale_scalar` 静态 buffer；末尾的 `maybe_fuse_routed_scale_and_shared_add` 把 routed scaling factor 折叠进 shared-expert 加法。

#### quantization/mxfp4_humming_moe.py

- **作用**：MXFP4 expert 的 Humming runner 后端（DeepSeek-V4，`--moe-runner-backend humming` + `SGLANG_DSV4_FP4_EXPERTS=1`）。
- **关键内容**：`Mxfp4HummingMoEMethod` 包装 FP8 基础 method；`process_weights_after_loading` 把 `w{13,2}_weight_scale_inv` 转 `float8_e8m0fnu` 后调 `prepare_humming_moe_layer(layer, {"quant_method": "mxfp4"})`；`apply` 构造 `HummingMoeQuantInfo`。

#### quantization/mxfp4_marlin_moe.py

- **作用**：MXFP4 expert 的 Marlin runner 后端（SM90/SM120）。
- **关键内容**：`build_marlin_moe_quant_info`（含 EP 的 `expert_map` 映射，供 `Mxfp4MoEMethod` 与本类共用）；`Mxfp4MarlinMoEMethod`：权重 pad 到 hidden 256/intermediate 128 对齐，scale 以 uint8 127（=1.0）初始化并标记 `format_ue8m0=False`；`process_weights_after_loading` 调 `prepare_moe_mxfp4_layer_for_marlin`（Marlin 消费 [w1;w3] 原生顺序，不做 w3w1 重排；有 `gemm1_alpha` 时先 `deinterleave_moe_mxfp4_w13_for_marlin`）；`apply` 对 hidden 做 0-pad。

#### quantization/mxfp4_tensor.py

- **作用**：MXFP4 张量量化/反量化的纯 torch 参考实现（源自 TensorRT ModelOptimizer）。
- **关键内容**：`MXFP4QuantizeUtil.quantize`（E2M1 值表 [0,0.5,1,1.5,2,3,4,6]，按 32 元素块取 amax 计算 E8M0 scale，位运算打包 uint4→uint8）与 `dequantize`（unfuse 高低 nibble、查表、乘 2^(scale-127)）。
- **实现逻辑要点**：块默认 32；scale 以 +127 偏置存 uint8；符号位与 3 位 magnitude 组成 4bit 编码。

#### quantization/mxfp8_block_convert.py

- **作用**：AMD gfx942（CDNA3/MI300）上将 MXFP8 checkpoint 转为 block-FP8 [128,128] 的工具。
- **关键内容**：`dequant_mxfp8_2d_to_bf16`（e4m3fn × 2^(UE8M0-127) → bf16）；`bf16_to_block_fp8_128`（按 128×128 块 amax/448 量化，与 DeepSeek-V3 block-fp8 约定一致）；`convert_mxfp8_weight_to_block_fp8` 串联两者。
- **备注**：gfx942 无 MX-scaled matmul，需借道原生 block-fp8 kernel（aiter/triton）。

#### quantization/npu_mxfp4.py

- **作用**：Ascend NPU 的 MXFP4 W4A8 在线量化配置（`--quantization mxfp_w4a8`）。
- **关键内容**：`Mxfp4W4A8Config`：权重加载后在线量化为 MXFP4，激活动态量化为 MXFP8（e4m3fn + UE8M0 块 scale），matmul 走 `npu_quant_matmul`；`get_quant_method` 按设备分发，目前仅 Ascend 950/A5 的 `NPUMXFP4W4A8LinearMethod`；FusedMoE 回退未量化并告警。

#### quantization/npu_mxfp4_w4a4.py

- **作用**：Ascend NPU 的 MXFP4 W4A4 在线量化配置（`--quantization mxfp4`，仅 is_npu 时注册）。
- **关键内容**：`Mxfp4W4A4Config`：权/激都做单级 MXFP4（`float4_e2m1fn_x2`），实际在线路径使用双级 MXFP4（`NPUDualLevelMXFP4LinearMethod`，细粒度 FP8 E4M3 L0 scale + 粗 L1 scale，经 `npu_dual_level_quant_matmul`），因单级 RTN 在 greedy 解码下精度退化；离线 msmodelslim 单级 checkpoint 由 ModelSlim scheme 处理。

#### quantization/nvfp4_online.py

- **作用**：`--quantization nvfp4_online`：加载期把 MoE expert 权重在线转 NVFP4 + 运行时 per-token FP32 激活 scale。
- **关键内容**：`NvFp4OnlineConfig`（继承 `ModelOptQuantConfig`，group_size=16，`SGLANG_FP4_IGNORED_LAYERS` 可追加排除层）；`_ModelOptFp4OnlineConfig`（`modelopt_fp4` 的 per-tensor 激活 scale 变体）；`ModelOptNvFp4OnlineFusedMoEMethod`：`_quantize_weight_nvfp4` 用 FlashInfer `nvfp4_quantize`（weight_scale_2 = amax/(e4m3max*6)，gated w1/w3 成对共享同一 amax 派生 scale），FP8 源 checkpoint 经 `_dequantize_fp8_weight`（per-tensor 或 block_quant_dequant）反量化后重量化。
- **实现逻辑要点**：`get_online_weight_loader` 以 `CopyNumelCounter` 计量已加载字节数，pending 字典 + 锁处理 w1/w3 成对与 FP8 weight/scale 分片到达顺序；支持 EPLB 的 expert 跳过判断。

#### quantization/online_quantization.py

- **作用**：在线量化加载流程的计量工具（24 行）。
- **关键内容**：`CopyNumelCounter`：一个 `TorchDispatchMode`，拦截 `aten.copy_` 累计被写入元素数，用于追踪经过 narrow/变换的权重实际加载进度（Quark/NVFP4 在线重量化的触发器）。

#### quantization/petit.py

- **作用**：AMD GPU 上 Petit kernel（gfx90a/gfx942）的 NVFP4 量化配置。
- **关键内容**：`PetitNvFp4Config`（`petit_nvfp4`，读 `hf_quant_config.json` 的 quant_algo=NVFP4、group_size、kv_cache_quant_algo、exclude_modules；`is_petit_nvfp4_compatible` 要求 HIP 且 quant_method=modelopt）；`PetitNvFp4LinearMethod`：权重 [out, in//2] uint8、E4M3 块 scale、FP32 `input_scale`/`weight_scale_2`。
- **实现逻辑要点**：`process_weights_after_loading` 把 input/weight scale 合成 `alpha` 后调 `prepare_nvfp4_layer_for_petit`；`apply` 走 `apply_petit_nvfp4_linear`。

#### quantization/petit_utils.py

- **作用**：Petit kernel 的封装与支持性校验。
- **关键内容**：`verify_petit_nvfp4_supported`（仅 NVFP4、group_size=16）；`prepare_nvfp4_layer_for_petit`（`repack_nvfp4` + `process_nvfp4_scales`）；`apply_petit_nvfp4_linear`（`mul_nvfp4_a16`，solution_id=-1 自动调优）。
- **备注**：import 失败时提供抛出"请安装 petit-kernel"的占位实现。

#### quantization/quark/（Quark 家族，11 个文件）

- **家族说明**：AMD Quark 工具链（ROCm GPU）：`quark.py` 为总配置（预量化 checkpoint 加载 + NVFP4/FP8→MXFP4 在线重量化），`schemes/` 7 个 scheme 实现各格式（线性/MoE、W4A4/W4A8/W8A8），`utils.py`/`weights.py` 提供底层工具与 GPT-OSS 加载器。

#### quantization/quark/__init__.py

- **作用**：空文件（0 行），仅作为包标记。

#### quantization/quark/quark.py

- **作用**：AMD Quark 量化的主配置：既有预量化 checkpoint 加载，也做 NVFP4/FP8→MXFP4 的在线重量化。
- **关键内容**：`QuarkConfig`（`exclude` 层过滤、`pack_method`、kv_cache_group、`_find_matched_config` 按 layer_quant_config→layer_type_quant_config→global_quant_config 三级匹配、`_is_mx_fp4`/`_is_fp8_w8a8`/`_is_mx_w4a8` 方案判定）；`QuarkLinearMethod`/`QuarkFusedMoEMethod`（委托 `layer.scheme`）；`QuarkKVCacheMethod`（仅 fp8_e4m3 per-tensor KV cache）。
- **实现逻辑要点**：`from_config` 以 `requantization_method=quark_mxfp4` 分派——`_detect_nvfp4_source` 识别 ModelOpt（quant_method=modelopt/modelopt_fp4/nvfp4）与 Quark（global_quant_config 为 [fp4_per_group_gs16, fp8_e4m3_per_tensor]）NVFP4 源；`_build_mixed_precision_layer_quant_config` 把 MIXED_PRECISION 的 per-layer quant_algo 折叠成 fnmatch 模式（NVFP4→MXFP4 目标 spec，FP8 保持）；纯 FP8 源走 `Fp8Config` 反量化路径；`can_fuse_shared_expert` 校验 routed/shared expert 方案一致性。

#### quantization/quark/schemes/__init__.py

- **作用**：转发型包初始化，导出 6 个 scheme 类（QuarkLinearScheme/QuarkMoEScheme 及 4 个具体实现）。

#### quantization/quark/schemes/quark_scheme.py

- **作用**：Quark scheme 抽象基类。
- **关键内容**：`QuarkLinearScheme`（抽象 `get_min_capability`/`create_weights`/`process_weights_after_loading`/`apply_weights`）与 `QuarkMoEScheme`（多出 `create_moe_runner`/`apply_weights(dispatch_output)`），继承 `BaseLinearScheme`/`BaseMoEScheme`。

#### quantization/quark/schemes/quark_w4a4_mxfp4.py

- **作用**：Quark MXFP4（OCP fp4 + e8m0，group=32）线性 scheme，支持序列化、在线与 NVFP4/FP8 重量化三种来源。
- **关键内容**：`QuarkW4A4MXFP4`：HIP 下注册 `aiter_gemm_afp4wfp4`、`aiter_gemm_afp4wfp4_pre_quant`、`aiter_fused_gemm_split_cat`、`aiter_dynamic_mxfp4_quant` 自定义算子；`_create_weights_from_nvfp4`（`dequantize_nvfp4`→`dynamic_mxfp4_quant`，丢弃 input_scale）；FP8 源在 meta device 上建 FP8 参数、加载完成后 `dequantize_fp8`→重量化。
- **实现逻辑要点**：`apply_weights` 支持 tuple 输入（MLA 的 (x,x_s)/(x,x_s,y)/(x,y,S1,S2,out_dtype) 三种形式）分别走 fused/pre-quant/split-cat GEMM；`AITER_FORCE_A8W4` 时（gfx1250 无 fp4 WMMA）退化为加载期 `_dequant_mxfp4_to_bf16`（LUT 查表 + 2^(e8m0-127)）。

#### quantization/quark/schemes/quark_w4a4_mxfp4_moe.py

- **作用**：Quark MXFP4 W4A4 MoE scheme（AMD，AITER kernel）。
- **关键内容**：`QuarkW4A4MXFp4MoE`：序列化路径按 `get_moe_weight_sizes`（aiter concat/packed）分配 w13/w2 uint8 权重与 E8M0 scale；在线路径 meta device 分派 + `_quantize_w13/w2_online`（逐 expert `dynamic_mxfp4_quant`）；NVFP4 源走 `_requantize_nvfp4_to_mxfp4`（含 AITER 对齐的 intermediate 0-pad）；FP8 源走 `get_online_fp8_to_mxfp4_weight_loader`。
- **实现逻辑要点**：`process_weights_after_loading` 预打散：gfx1250 用 `moe_shuffle_scale/moe_shuffle_weight`（n32k4 布局 + GUGU 行交错），其余 gfx95 用 `e8m0_shuffle` + `shuffle_weight(16,16)`；`apply_weights` 构造 `AiterMoeQuantInfo(PER_1X32)` 并把权重 view 为 `float4_e2m1fn_x2`。

#### quantization/quark/schemes/quark_w4a8_mxfp4_moe.py

- **作用**：Quark W4A8 方案（MXFP4 权重 + 静态 FP8 激活）的 MoE scheme（GPT-OSS/DSv4 用）。
- **关键内容**：`QuarkW4A8MXFp4MoE`：校验权重为静态 per-group fp4 gs=32 e8m0、激活为 per-tensor fp8_e4m3(fn) 静态；创建 uint8 权重、E8M0 scale、fp32 `w13/w2_weight_bias` 与 per-expert `w13/w2_input_scale`。
- **实现逻辑要点**：gfx1250 走 `prepare_w4a8_gfx1250_weights` + triton `moe_gemm_a8w4`（绕开 CK/ASM 融合路径）；gfx950 走 `shuffle_weight_a16w4`/`shuffle_scale_a16w4`（gate-up 交错布局，`SGLANG_USE_AITER_MOE_GU_ITLV=0` 时退回分离布局）；激活 scale 取各 expert 最大值；AITER runner 强制 activation="swiglu"。

#### quantization/quark/schemes/quark_w8a8_fp8.py

- **作用**：Quark W8A8 FP8 线性 scheme。
- **关键内容**：`QuarkW8A8Fp8`：支持 per_tensor/per_channel 权重 scale 与动态/静态 per-token 激活；`process_weights_after_loading` 做 fnuz 归一化（ROCm）、per-tensor 时 `requantize_with_max_scale` 融合 QKV 多 shard scale、`_use_aiter` 时 `shuffle_weight(16,16)` 预打散；`apply_weights` 调 `apply_fp8_linear`。
- **备注**：min capability 89（Ada Lovelace 及以上）。

#### quantization/quark/schemes/quark_w8a8_fp8_moe.py

- **作用**：Quark W8A8 FP8 MoE scheme（TRITON runner 或 ROCm tkw1 kernel）。
- **关键内容**：`QuarkW8A8FP8MoE`：创建 e4m3fn w13/w2 权重与 per_tensor（[E,2]/[E]）或 per_channel 权重 scale；`process_weights_after_loading` 做 fnuz 归一化，per-tensor 时逐 expert 反量化后按 max scale 重新 `scaled_fp8_quant`；AITER + per-channel + router_weight_on_input 时预打散并走 `rocm_fused_experts_tkw1`，否则 `TritonMoeQuantInfo(use_fp8_w8a8=True)`。

#### quantization/quark/utils.py

- **作用**：Quark 的通用工具函数。
- **关键内容**：`Nvfp4SourceConfig`（在线 NVFP4→MXFP4 重量化的分派标记 dataclass）；`deep_compare`；`should_ignore_layer`/`check_equal_or_regex_match`（支持 `re:` 前缀正则的层排除匹配）；`b_dynamic_mxfp4_quant`（3D→2D 动态量化）；`mxfp4_to_f32`（uint8 打包 FP4 nibble 解包 + 16 值查表）；`e8m0_to_f32`（2^(x-127)，255 为 NaN sentinel）；`quark_post_load_weights`（MLA 的 kv_b_proj 加载后按 head dim 拆分并量化 w_kc/w_vc）。
- **备注**：`dynamic_mxfp4_quant` 来自 `aiter.ops.triton.quant`，缺失时抛 ImportError。

#### quantization/quark/weights.py

- **作用**：GPT-OSS per-expert 布局的 Quark MoE 权重加载器。
- **关键内容**：`load_gptoss_weight_quark`：用正则匹配 `model.layers.{L}.mlp.experts.{N}.{gate_up_proj|down_proj}.{weight|weight_scale|input_scale|bias}`，专家权重单独加载、其余走普通加载；`_load_gptoss_quark_expert_weights` 处理 gate/up 行解交错（[g0,u0,g1,u1]→[g…;u…]）、TP 窗口切片、EP 专家划分、FP4 打包维减半与 down_proj bias 仅 TP rank 0 持有。

#### quantization/quark_int4fp8_moe.py

- **作用**：`quark_int4fp8_moe` 在线量化（INT4 权重 + FP8 计算），仅 AMD GPU。
- **关键内容**：`QuarkInt4Fp8Config`（含 tqdm 进度条）；`QuarkInt4Fp8MoEMethod`：加载期把 BF16/FP16 expert 权重 `quantize_fp8_scale_tensorwise` + `quantize_int4_scale_columnwise`（int4 scale /= fp8 scale）后 `pack_int4_to_int32`；权重为 uint32 打包 [E, N, K//8]。
- **实现逻辑要点**：加载器先按 TP 窄切再量化（未预切分 checkpoint 临时置 `use_presharded_weights`）；`process_weights_after_loading` 中非 gfx950 做 fp8 fnuz 的 0.5/2.0 因子校正、`shuffle_weight(16,16)`、并把 w13 双 fp8 scale 折入 int4 scale；`apply` 走 `AiterMoeQuantInfo(PER_TOKEN)`。

#### quantization/rocm_mxfp4_utils.py

- **作用**：ROCm/AITER MXFP4 工具再导出（14 行）。
- **关键内容**：从 `aiter.ops.triton` 转发导出 `fused_rms_mxfp4_quant`、`fused_flatten_mxfp4_quant`、`batched_gemm_afp4wfp4_pre_quant` 三个符号。

#### quantization/unquant.py

- **作用**：未量化回退实现全集 + BF16 GEMM 后端选择，是所有量化方法排除层与 fallback 的公共底座。
- **关键内容**：`Bf16GemmBackend` 枚举（auto/cutedsl/flashinfer_pr4266/gemv/torch）与 `initialize_bf16_gemm_config`/`bf16_gemm_dispatch`（含 GB300 TP16 实测的 `_FLASHINFER_PR4266_TUNED_TACTICS` 调优表与 `_can_accumulate_into_addend`）；`UnquantizedEmbeddingMethod`、`UnquantizedLinearMethod`（`apply`/`apply_with_addend`/`apply_into`，CPU 走 AMX 处理）；`UnquantizedFusedMoEMethod`（继承 `BaseFusedOp`，按设备分派 `forward_cuda`/`forward_cpu`/`forward_xpu`/`forward_npu`/`forward_tpu`/`forward_musa`，含 `_maybe_interleave_w13_for_fused_swiglu`、flashinfer trtllm BF16 权重形状恢复、`_torch_compile_forward`）。
- **备注**：MoE runner 自动选择 aiter/ck 与 triton 后端逻辑也在此。

#### quantization/utils.py

- **作用**：跨方案共享的量化工具函数库。
- **关键内容**：`get_scalar_types`（sgl_kernel ScalarType，带 Mock 兜底）；`is_layer_skipped`（含 `_module_path_match` 路径边界匹配与 `_FALLBACK_FUSED_SHARDS` 默认融合映射）；`per_tensor_dequantize`、`convert_to_channelwise`、`requantize_with_max_scale`、`all_close_1d`；参数工具 `replace_parameter`、`update_tensor_inplace`、`override_config`/`get_dynamic_override`（`config.json` 的 `quantization_config.override` 支持）。
- **备注**：GPTQ 侧 `get_pack_factor`/`permute_rows`/`pack_cols`/`pack_rows`/`unpack_cols`/`quantize_weights`/`gptq_quantize_weights`/`sort_weights`；以及 `swizzle_blockscale`、`swap_w13_to_w31`/`reorder_w1w3_to_w3w1`、`prepare_static_weights_for_trtllm_fp4_moe`。

#### quantization/w4afp8.py

- **作用**：MIXED_PRECISION W4AFP8 量化（4bit 权重 + FP8 激活，MoE）。
- **关键内容**：`W4AFp8Config`（group_size=128，`weight_block_size=[128,128]`；linear 层走 `Fp8LinearMethod`）；`W4AFp8MoEMethod`：int8 打包权重 [E, N, K//2]、group-wise `weight_scale_inv`、per-expert input_scale；`interleave_scales`（TRT-LLM 风格 4 组交错）。
- **实现逻辑要点**：`process_weights_after_loading` 交错 scale、input scale 取 max；预置 DeepEP 的 strides/expert_offsets/problem_sizes；`apply` 走 `cutlass_w4a8_moe`，另有 `apply_deepep_ll`（FP8 payload + per-token-group scale）与 `apply_deepep_normal`（BF16）两条 DeepEP 路径。

#### quantization/w8a8_fp8.py

- **作用**：通用 W8A8 FP8 量化（权重静态 per-channel，激活动态 per-token，对称）。
- **关键内容**：`W8A8Fp8Config`（`is_checkpoint_fp8_serialized` 判定 "compressed-tensors"/"w8a8_fp8"）；`W8A8Fp8LinearMethod`（无离线量化时加载期 `per_token_group_quant_fp8` 在线量化权重，CUTLASS 支持时 per-channel 否则 per-tensor）；`W8A8FP8MoEMethod`（e4m3fn w13/w2 + per-tensor fp32 scale，TRITON runner + `TritonMoeQuantInfo(use_fp8_w8a8=True)`）。
- **备注**：min capability 89。

#### quantization/w8a8_int8.py

- **作用**：通用 W8A8 INT8 量化（compressed-tensors 风格 `quant_model_description.json`）。
- **关键内容**：`W8A8Int8Config`（`is_dynamic`、ignore、packed_modules_mapping）；`W8A8Int8LinearMethod`（CUDA 上 `sgl_kernel.int8_scaled_mm`，CPU 走 AMX `_amx_process_weight_after_loading`，支持 per-channel scale + offset、动态 per-token 量化）；`W8A8Int8MoEMethod`（int8 权重 + per-channel fp32 scale/offset + per-expert input scale，TRITON runner `TritonMoeQuantInfo(use_int8_w8a16=True)`）。
- **备注**：min capability 75；CUDA 下注册 `int8_scaled_mm` 的 fake 实现。

### 3.4 cp/（prefill 上下文并行，8 个文件）

本目录是 prefill 上下文并行（Context Parallel，CP）的策略抽象层：以 `ContextParallelStrategy` 抽象基类 + 进程级单例为中心，提供 zigzag（in-seq-split，每序列切成 2×cp_size 块）与 interleave（round-robin 按 token 取模）两种切分布局；`bcg.py` 支持 zigzag 布局下的可打断 CUDA graph（breakable CUDA graph），`cp_decode_attn_tp.py` 把 decode 阶段的复制权重按 CP rank 切片模拟 TP。

#### cp/__init__.py

- **作用**：转发型：从 base/interleave/zigzag 聚合导出策略类、metadata 类与 `get_cp_strategy` 等辅助函数。

#### cp/base.py

- **作用**：CP 策略的抽象基类与进程级单例管理。
- **关键内容**：`ContextParallelStrategyKind`（NONE/ZIGZAG/INTERLEAVE）与 `CPAttentionBackendKind`（FLASH_ATTENTION/DSA/TRTLLM_MHA）两个 IntEnum；`BaseContextParallelMetadata` dataclass；ABC `ContextParallelStrategy` 定义 shard/gather/attention 等抽象接口；`init_cp_strategy`/`get_cp_strategy` 维护 `_STRATEGY` 单例，`is_cp_enabled`/`is_zigzag`/`is_interleave` 为判定辅助。
- **实现逻辑要点**：`cp_rank` 属性直接读 `get_parallel().attn_cp_rank`；`get_cp_strategy` 在 worker 子进程里懒初始化（从全局 server args 恢复单例），失败时静默返回 None；`split_before_forward` 提供 CP-v2 路径的默认实现（切 input_ids/positions/input_embeds）。

#### cp/bcg.py

- **作用**：zigzag 布局下 prefill CP 的可打断 CUDA graph（BCG）支持。
- **关键内容**：`supports_prefill_cp_bcg`（要求 pp_size=1、attn_cp_size==tp_size、zigzag、trtllm_mha）；`filter_prefill_cp_bcg_capture_num_tokens` 过滤太小的 token bucket；dataclass `PrefillCPBCGInput` 持有定址 CP-local 输入缓冲与各 bucket 的 captured 容量，`required_local_tokens` 计算 zigzag 布局需要的对齐行数，`prepare` 把全局 prefill 输入切分填入定址缓冲；`execute_prefill_cp_bcg` 重放 CP-local 图段后 eager 执行全局 gather 与 logits 尾段。
- **实现逻辑要点**：replay 时强制 `metadata.per_rank_actual_token/max_rank_len` 重写为 capture 容量以保证图段行几何不变；`_slice_output_rows` 递归裁剪输出到 live_local_tokens。

#### cp/cp_decode_attn_tp.py

- **作用**：CP 模式下（tp_size=1 权重复制）让 decode 阶段按 CP rank 切片注意力权重，等效于 TP 布局。
- **关键内容**：`CP_DECODE_ATTN_TP_SUPPORTED_ARCHS` 白名单（DeepSeek-V4 系列、GLM-5.x DSA 系列）；全局单例 `CpDecodeAttnTpContext`，`maybe_use_decode_attn_tp` context manager 在进入时用 `_activate` 把 ColumnParallel/RowParallelLinear 的 weight（及 scale）narrow 到本 rank 份额、缩小 size_per_partition、设置 RowParallelLinear 的 use_decode_attn_tp 标志、覆盖 radix_attn 的 tp_q_head_num，退出时全部还原。
- **实现逻辑要点**：`_slice_cache` 按 (id(obj), attr_name) 缓存切片与原始张量；prefill CP（`is_cp_v2_active` 或 dsa prefill CP）时跳过不激活；只支持 dim 0/1 切片。

#### cp/interleave.py

- **作用**：round-robin（interleave）CP 策略：rank r 持有第 r、r+cp、r+2cp…个 token。
- **关键内容**：`InterleaveContextParallelMetadata`（per_rank_actual_token/max_rank_len/per_rank_logical_token）；`InterleaveCPStrategy` 实现 `can_apply`（要求 is_context_parallel_extend 且总长 ≥ cp_size）、`build_metadata`（均分 token 数）、`_interleave_shard`（view+索引切分）、`_gather_interleaved_tensor`（all-gather 后用 `flat_indices % cp_size * max_len + flat_indices // cp_size` 索引还原）、`shard_per_request`（调 DSA kernel `dsa_cp_round_robin_split_q_seqs_kernel`）、`materialize_full_mla_kv` 等。
- **实现逻辑要点**：gather 时先 pad 到 max_rank_len 再 attn_cp_all_gather；`run_attention` 是 no-op（interleave 专供 DSA 后端自己跑 attention）；FP8 KV 以 uint8 视图按字节 gather 规避 dtype 问题。

#### cp/padding.py

- **作用**：CP 策略共用的 token 对齐填充工具。
- **关键内容**：`get_cp_padding_align_size`（zigzag 返回 2×cp_size，其余 cp_size）；`pad_logical_token_to_physical` 把 per_rank_logical_token 记录为真实长度、per_rank_actual_token/max_rank_len 对齐到 align_size；`pad_local_rows` 把本地张量从 logical 长度补零到 physical 长度。
- **实现逻辑要点**：对齐保证 CP 集合通信的 uniform shape；logical 与 physical 相等时零拷贝返回。

#### cp/utils.py

- **作用**：CP 策略的公共 facade 与运行时辅助函数（对外 import 入口）。
- **关键内容**：CP-v2 判定链 `enable_cp_v2`（非 HIP/NPU/MUSA）、`is_cp_v2_active`、`prepare_cp_forward`（构建并 pad metadata、设置 DP 缓冲长度、裁剪 out_cache_loc）；切分/回收 `cp_split_before_forward`、`cp_shard_hidden_states/position_ids`、`cp_gather_after_forward`（tuple 感知，兼容 MiMo 的 (hidden, None)）、`cp_materialize_global_token_order`、context manager `cp_shard_model_inputs`（含 spec_info.hidden_states 备份还原）；GLM DSA cache 层切分辅助 `is_glm_dsa_cache_layer_split_enabled`/`get_layer_shard_range`/`get_layer_owner` 等。
- **实现逻辑要点**：`cp_round_robin_input_ids_v2` 在 MoE A2A backend 非 none 时按 rank 步长取子序列，否则 reshape+转置；legacy 平台 fallback 到 `cp_all_gather_rerange_output`。

#### cp/zigzag.py

- **作用**：zigzag（in-seq-split）CP 策略：每序列切成 2×cp_size 块，rank r 持有 block r 与 block 2×cp_size-1-r。
- **关键内容**：`ZigzagContextParallelMetadata`（split_list/zigzag_index/cp_reverse_index/reverse_split_len 及 prev/next/combined 三套 FA/TRTLLM 几何张量，`ContextParallelMetadata` 为其别名）；`ZigzagCPStrategy` 的 `build_metadata` 完整推导块大小、zigzag 索引、反向排列与 kv_len/cu_seqlens；`run_attention` 对 FLASH_ATTENTION 分 prev/next 两次调用，TRTLLM_MHA 用 combined 几何 + `use_zigzag_page_table`；`materialize_full_kv/_mla_kv` 用 `get_token_to_kv_pool().set_kv_buffer/set_mla_kv_buffer` 写全量 KV。
- **实现逻辑要点**：`_all_gather_reorganized` 先 pad 到 max_rank_len 再 all_gather，按 per_rank_token 拼接去 pad；`can_apply` 要求每序列长度 ≥ 2×cp_size。

### 3.5 dcp/（decode 上下文并行，5 个文件）

与 prefill CP 不同，DCP 按「owner 规则」（位置 mod dcp_size == dcp_rank）在 rank 间分配 KV，通过 LSE 合并（all-gather + all-reduce/reduce-scatter，或 A2A 交换 + Triton 局部合并）还原全局注意力输出；曾是 PR #14194（FlashInfer-MLA）与 #25090（Triton/MHA）两条线的整合点。

#### dcp/__init__.py

- **作用**：转发型：从 comm/layout/metadata 及 sgl_kernel 导出 DCP 外部接口（all_gather_kv_cache_for_*、cp_lse_ag_out_rs_*、dcp_a2a_lse_reduce、DecodeContextParallelMetadata 等）。
- **备注**：planner 被刻意不导入——它依赖 server_args，避免包初始化引入加载期 import 边；`dcp_enabled`/`get_attention_dcp_*` 是带 DeprecationWarning 的兼容导出。

#### dcp/comm.py

- **作用**：DCP 的组访问器、LSE 合并与 KV all-gather 集合通信实现。
- **关键内容**：两个 LSE 合并变体——`cp_lse_ag_out_rs_mha`（torch all-gather LSE → logsumexp → 加权 all-reduce → 按 rank 切头）与 `cp_lse_ag_out_rs_mla`（Triton `correct_attn_out` + reduce_scatter，区分 base-e/base-2 LSE）；`all_gather_kv_cache_for_dcp`（按 owner 规则计算 left/right pad、pad 后 all-gather、按 prefix_starts 重组为 query 顺序）；`all_gather_kv_cache_for_mha_extend/_mla_extend`、`all_gather_q_for_mla_decode`；A2A 后端 `dcp_a2a_lse_reduce`（输出与 fp32 LSE 打包进一次 all_to_all，`_lse_pack_dim` 决定每 D 列夹带位数）及 `_dcp_fi_a2a_lse_reduce`（委托 FlashInfer MNNVL 交换）+ `init_fi_a2a_workspace`。
- **实现逻辑要点**：fp8 数据一律以 uint8 视图走字节通信；fi_a2a workspace 必须在 CUDA graph capture 前初始化并 barrier。

#### dcp/layout.py

- **作用**：DCP 的纯索引数学：每 rank 可见长度与 owner 规则过滤。
- **关键内容**：`get_dcp_lens`（支持 start 偏移的通式：`pos % dcp_size == dcp_rank` 下的本地可见数）；`filter_dcp_local_kv_indices`（按 index % dcp_size == dcp_rank 保留，仍是 widened 索引）；`filter_dcp_local_chunk_kv_indices`（chunk 前缀场景按 `(rank - start) % dcp_size` 步长取样）；`update_local_kv_lens_for_dcp`（start=0 特例的就地覆写，与 floor((len-rank-1)/N)+1 位级一致）。
- **备注**：注释指出该恒等式有单测 test_dcp_layout_unit.py 保证。

#### dcp/metadata.py

- **作用**：DCP 每 forward 的元数据容器。
- **关键内容**：`DecodeContextParallelMetadata` dataclass，含 dcp_kv_indptr/dcp_kv_buffer/dcp_kv_indices/dcp_local_prefix_kv_indices/dcp_extend_prefix_lens_sum 五个字段。
- **备注**：刻意不继承 prefill CP 的 `BaseContextParallelMetadata`，存于 ForwardBatch 独立字段 `attn_dcp_metadata`。

#### dcp/planner.py

- **作用**：DCP metadata 构建器（PR #14194 保留的函数式实现）。
- **关键内容**：`prepare_decode_context_parallel_metadata`（用 chunked prefix kernel 生成 dcp_prefix_kv_indices、`create_dcp_kv_indices` 生成全量 indices、经 `KVIndexTranslator.translate_dcp_read_ids` 翻译本地前缀索引、分配 dcp_kv_buffer）；`plan_dcp_decode_metadata`（就地改写 kv_lens/kv_indptr 为 per-rank 长度，调 Triton kernel `update_kv_lens_and_indices` 把本 rank ids 压缩进 kv_indices 前缀，返回 total_local_len）。
- **实现逻辑要点**：docstring 详述 dcp_kv_buffer 布局（各 rank 前缀 token 交错 + extend token）；dp 未启用时直接返回 None。

### 3.6 rotary_embedding/（RoPE 与变体，8 个文件）

沿袭 vLLM 结构：`base.py` 的 `RotaryEmbedding` 基类覆盖 CUDA/CPU/NPU/XPU 多平台前向，`factory.py` 的 `get_rope` 按 `rope_scaling.rope_type` 分派到 linear/llama3/dynamic/yarn/longrope/proportional 等变体，另有 MRoPE（多模态 3D 位置）与各模型 get_rope_index 实现（Qwen2-VL/Qwen3-VL、Qwen3-Omni、GLM4V、Ernie4.5）。

#### rotary_embedding/__init__.py

- **作用**：公共 API 入口（原 rotary_embedding.py 的 drop-in 替代）：导出 RotaryEmbedding、get_rope、get_rope_wrapper、MRotaryEmbedding、Ernie4_5_VLRotaryEmbedding、apply_rotary_pos_emb 及 `_yarn_*` 别名。

#### rotary_embedding/base.py

- **作用**：`RotaryEmbedding` 基类与 `LinearScalingRotaryEmbedding`。
- **关键内容**：`RotaryEmbedding(BaseFusedOp)` 构造时算 cos_sin_cache（默认 GPU 计算、FP32 缓存）、按 head_size 选择主 kernel 或 fallback；`forward_cuda`（inplace `apply_rope_with_cos_sin_cache_inplace`，或 HIP 上带 KV cache 写入的 fused_qk_rope_reshape_and_cache，或 fallback kernel）、`forward_cpu/_npu/_xpu`、`forward_native`（纯 torch，`_force_native` 用于 RL on-policy 确定性路径）；`_ensure_cos_sin_cache_length` 支持超长位置增量扩容。
- **实现逻辑要点**：`LinearScalingRotaryEmbedding` 把多个 scaling factor 的 cache 拼成一个张量、用 `scaling_factor_to_offset` 偏移索引，支持多 LoRA 批量；`_compute_inv_freq` 在 CPU 上算以精确对齐 HF（force_native 时）；cache dtype 非 CUDA/XPU 时转模型 dtype。

#### rotary_embedding/factory.py

- **作用**：RoPE 实例工厂 `get_rope`/`get_rope_cpu`/`get_rope_wrapper`。
- **关键内容**：`get_rope` 按 `rope_scaling` 的 rope_type 分派——llama3（Llama3RotaryEmbedding）、default 下 mrope_section→MRotaryEmbedding、use_fope→FourierRotaryEmbedding、linear、dynamic（alpha 有无分两个类）、yarn（含 mrope 版）、deepseek_yarn、longrope、proportional（Gemma4RotaryEmbedding）；进程级 `_ROPE_DICT` 缓存 + `_get_live_rope_cache_entry` 丢弃 buffer 已死的条目；`get_rope_wrapper` 在 HIP+AITER 时走 aiter_get_rope（gfx1250 除外），CPU 走 `get_rope_cpu`（仅支持 deepseek_yarn）。
- **实现逻辑要点**：partial_rotary_factor<1 时先缩放 rotary_dim；`_get_rope_param` 对缺失 key 告警而非静默默认（适配 transformers v5 配置）。

#### rotary_embedding/utils.py

- **作用**：RoPE 原始算子。
- **关键内容**：`rotate_neox`/`rotate_gptj`（两种半旋转排列）；`apply_rotary_emb`（token-level 通用旋转，x: [num_tokens, num_heads, head_size]）；`rotate_half`；`apply_rotary_pos_emb_native`（torch.compile 包装）与 `apply_rotary_pos_emb_npu`（torch_npu.npu_rotary_mul，超 1000 头或 896 维回退 native）；按平台条件导出 `apply_rotary_pos_emb`（NPU/CPU-AMX kernel/native）。

#### rotary_embedding/yarn.py

- **作用**：YaRN 缩放方法。
- **关键内容**：`yarn_find_correction_dim/range`（按旋转次数反推维度区间）、`yarn_linear_ramp_mask`、`yarn_get_mscale_simple/_mscale`；`YaRNScalingRotaryEmbedding` 的 `_compute_inv_freq` 把高频外推、低频插值（beta_fast/beta_slow 分界）混合，cache 乘 mscale（支持 HF 的 mscale/mscale_all_dim 组合再乘 attn_factor）。

#### rotary_embedding/mrope.py

- **作用**：多模态 3D RoPE（MRoPE）。
- **关键内容**：`MRotaryEmbedding`——mrope_section 和为 rotary_dim//2 的自动纠错、`_build_axis_map` 生成 t/h/w 轴归属（支持 interleaved 与 GLM interleaved 两种交错布局）、`get_cos_sin_with_position` 按 3 通道位置索引 cache 并按 section 重排/交错；`forward_triton`（triton_mrope_fused）、`forward_cuda/_npu/_xpu/_cpu`；静态方法 `get_rope_index`/`get_rope_index_qwen3_omni`/`get_rope_index_glm4v`/`get_rope_index_ernie45` 均懒 import 转发到 mrope_rope_index；`YaRNScalingMRotaryEmbedding`（YaRN 版 MRoPE）；`Ernie4_5_VLRotaryEmbedding`（h/w/t 排列的 3D RoPE，triton_ernie45_rope_fused_inplace）。
- **实现逻辑要点**：interleaved 路径在 prefill/decode 双后端都支持 Triton 时用 Triton 重排；`apply_interleaved_rope` 为 fallback 交错实现。

#### rotary_embedding/mrope_rope_index.py

- **作用**：各多模态模型的 mrope position ids 计算。
- **关键内容**：`get_rope_index`（Qwen2/2.5/3-VL、InternS2、Cosmos3 系：扫描 vision_start token、按 image/video 的 grid_thw 生成 t/h/w 索引、文本段顺序递增，返回 [3, bs, seq] position_ids 与 mrope_position_deltas）；`get_rope_index_qwen3_omni`（加入 audio：按 audio_seqlens 估算特征长度、t 轴乘以 position_id_per_seconds，含 audio-in-video 交错合并）；`get_rope_index_glm4v`（按 attention_mask 过滤、video_start/end token 标记视频区间、逐帧生成）；`get_rope_index_ernie45`（按 spatial_conv_size/temporal_conv_size 下采样）；辅助 `_get_feat_extract_output_lengths`（音频卷积输出长度）与 `_get_llm_pos_ids_for_vision`。
- **实现逻辑要点**：qwen3_vl 系列视频 grid 按帧 repeat_interleave；无多模态输入时退化为 3 行顺序 arange。

#### rotary_embedding/rope_variant.py

- **作用**：各 RoPE 缩放变体集合。
- **关键内容**：`Phi3LongRoPEScaledRotaryEmbedding`（short/long 因子双 cache 拼接，位置超过 original_max 时加偏移跳转 long cache）；`FourierRotaryEmbedding`（FoPE：可学习 cos_coef/sin_coef 按头混合、可选 per-head 分离，num_inv_freq 截断低频）；`DeepseekScalingRotaryEmbedding`（YaRN 变体 + NPU 专用缓存）；`Llama3RotaryEmbedding`（波长分段 low/high_freq_factor 平滑缩放）；`Llama4VisionRotaryEmbedding`（2D patch 复数频率、含 CLS 特殊索引）；`DynamicNTKAlphaRotaryEmbedding`/`DynamicNTKScalingRotaryEmbedding`；`DualChunkRotaryEmbedding`（DCA：q/qc/k 五套 cache，输出 5 段拼接的 query）；`Gemma4RotaryEmbedding`（[0,R/2) 与 [D/2, D/2+R/2) 交叉旋转）。
- **实现逻辑要点**：Phi3 仅支持 neox_style；Gemma4 把 head_size 当 rotary_dim 传入基类并零填充不旋转维度。

### 3.7 deep_gemm_wrapper/（DeepGEMM JIT 封装，4 个文件）

对 DeepSeek 开源 deep_gemm JIT GEMM 库的封装：负责启用判定（SM 版本/平台）、启动期预编译（按 M 列表 warmup 生成 cubin 到共享缓存）、以及对 grouped/normal GEMM 入口的统一包装（sanity check、num_sms 配置、执行 hook 触发编译）。

#### deep_gemm_wrapper/__init__.py

- **作用**：转发型：`from .entrypoint import *`。

#### deep_gemm_wrapper/compile_utils.py

- **作用**：DeepGEMM JIT 预编译与 warmup 机制。
- **关键内容**：`DeepGemmKernelType` IntEnum（6 种 kernel）；`update_deep_gemm_config`（生成 `_BUILTIN_M_LIST`：fast warmup 用逐步长采样，否则编译 1..m_max 全部 M；仅节点首 rank 全量编译）；`_maybe_compile_deep_gemm_one_type_all` + `_local_rank_compile_lock`（fcntl 文件锁串行化同 cache 目录下多 rank 的 nvcc 编译）；`_BaseWarmupExecutor` 及各子类（`_NormalWarmupExecutor`/`_GroupedContWarmupExecutor` 等）按 M 列表执行 dummy GEMM 触发编译；`deep_gemm_execution_hook`（执行期按 (kernel_type,n,k,num_groups) 懒编译）；`pp_parallel_deep_gemm_warmup`（PP 并行 dummy decode+extend forward 预热）。
- **实现逻辑要点**：强制 `DG_JIT_CACHE_DIR` 指向 SGLANG_CACHE_DIR、默认关 NVRTC；编译期临时禁用 symmetric memory；内存不足时对半削减 max_m。

#### deep_gemm_wrapper/configurer.py

- **作用**：DeepGEMM 可用性判定与平台能力常量。
- **关键内容**：`_compute_enable_deep_gemm`（CUDA 需 SM≥90、MUSA 需 SM≥31、SM120 需探测新入口、再查 `SGLANG_ENABLE_JIT_DEEPGEMM` 环境变量）；模块级常量 `ENABLE_JIT_DEEPGEMM`、`DEEPGEMM_BLACKWELL`（SM100）、`DEEPGEMM_SCALE_UE8M0`（SM100/120）、`DEEPGEMM_NEED_TMA_ALIGNED_SCALES`。
- **实现逻辑要点**：import 失败静默回退 False；SM120 支持来自 DeepGEMM#324，需探测 entry point 防旧版安装。

#### deep_gemm_wrapper/entrypoint.py

- **作用**：对 deep_gemm 各 GEMM 入口的统一包装层。
- **关键内容**：`get_mn_major_tma_aligned_tensor`（修复 sgl-deep-gemm ≤0.1.4.post1 非拥有别名导致的 use-after-free：data_ptr 相同则回传原张量）；`grouped_gemm_nt_f8f8bf16_masked/_contig`、`grouped_gemm_nt_bf16_masked/_contig`、`gemm_nt_f8f8bf16`、`gemm_nt_mxfp8_f8f8bf16`（recipe=(1,32) 的 fp8_fp4 路径）、`gemm_nt_bf16bf16f32`、`tf32_hc_prenorm_gemm`；`update_deep_gemm_config`（PDL 开关 + 委托 compile_utils）；context manager `configure_deep_gemm_num_sms` 与 `contiguous_layout_alignment_scope`；`_sanity_check_input`（可选校验 ue8m0 scale 对齐）。
- **实现逻辑要点**：所有入口都套 `deep_gemm_execution_hook` 触发懒编译；masked 版支持 overlap 参数与 max_block_n；`get_contiguous_layout_alignment` 用 theoretical alignment 按 per_group_m 收缩。

### 3.8 utils/（层相关杂项工具，4 个文件）

`common.py` 提供权重加载辅助（pad_or_narrow_weight、copy_or_rebind_param）与 PP 缺失层占位符；`cp_utils.py` 是给 HIP/NPU/MUSA 保留的 legacy prefill-CP 切分与 all-gather 工具集；`multi_platform.py` 是已废弃的 MultiPlatformOp 兼容 shim。

#### utils/__init__.py

- **作用**：转发型：`from .common import *` 外加 `MultiPlatformOp` 的废弃兼容导出。

#### utils/common.py

- **作用**：权重加载与模型构建小工具。
- **关键内容**：`get_layer_id`（正则解析 weight_name 的层号）；`pad_or_narrow_weight`（shard 截取 + 零填充，处理非对齐权重如 qwen2_5_VL MLP）；`is_strict_contiguous`/`strict_contiguous`；`copy_or_rebind_param`（形状匹配时 copy_ 保参数身份以兼容 CUDA graph 复用，否则 rebind）；`alias_or_bind_derived_param`（派生张量可广播时写回源 storage 并别名，省内存）；`PPMissingLayer`（PP 缺失层的 Identity 占位，可返回元组）。

#### utils/cp_utils.py

- **作用**：legacy prefill-CP 工具（保留给 HIP/NPU/MUSA 调用方，CUDA 已走 cp/ 策略层）。
- **关键内容**：`ContextParallelMetadata` dataclass（zigzag 布局列表 + FA 几何张量，无 combined 字段）；`can_cp_split`（2×cp_size 分块可行性判定，含 MIXED 模式排除与每序列阈值检查）；`cp_split_and_rebuild_data/position`（按 split_list+zigzag_index 切分，DSA round-robin 时走专用 kernel）；`cp_round_robin_input_ids`；`cp_all_gather_reorganized_into_tensor(_kv_cache)`（pad→all_gather→按 max_rank_len 拆分去 pad）；`cp_all_gather_rerange_launch/finish`（comm stream 上异步 all-gather 以重叠计算）；`cp_all_gather_rerange_output/_kv_cache`、`cp_allgather_and_save_kv_cache`（聚合后写全量 KV 进 memory pool）；`prepare_context_parallel_metadata`（zigzag 元数据完整推导，支持 radix 前缀 offset 与 NSA 模式区别）。
- **备注**：与 cp/zigzag.py 的 build_metadata 逻辑同源（此为旧版，含大量 ASCII 布局图注释）。

#### utils/multi_platform.py

- **作用**：废弃兼容 shim。
- **关键内容**：`MultiPlatformOp` 是 `sglang.kernels.fused_op.BaseFusedOp` 的别名（RFC #29630 已将多平台算子抽象统一），`__init_subclass__` 对被继承时发 DeprecationWarning，保留旧平台默认转发（forward_hip→forward_cuda 等）。
- **备注**：仅供 out-of-tree 插件迁移期使用，未来将删除。

## 四、结语与阅读建议

- **覆盖说明**：本文档按「作用 / 关键内容 / 实现逻辑要点」逐一说明了 `layers/` 下全部 **339** 个 Python 文件（顶层 35 + 子目录 304），其中 `attention/`、`moe/`、`quantization/` 三个大目录的同类文件（如各 attention 后端的转发壳、`nsa/` 的 9 个弃用 shim、`minimax_sparse_ops/naive|tests`、quantization 各家的 `schemes/`）按家族合并叙述并逐文件点名，同构内容不重复铺开。
- **推荐阅读顺序**：先读「1.1 这一层解决什么问题」与「1.2 调用链地图」建立整体观，再按需要深入某层——想理解模型权重如何加载与量化，从 `parameter.py` + `quantization/base_config.py` 入手；想理解一次 prefill/decode 的算子调度，看 `radix_attention.py`、`logits_processor.py` 与 `sampler.py`；做 MoE 性能分析则先读 `moe/moe_runner/base.py` 的注册表与 `moe/topk.py`。
- **演进警示**：SGLang 迭代极快（本目录在 `main` 分支 commit `b5c9b68f03` 附近统计），目录结构、类名与注册名随时可能变化：阅读代码时如发现与本文不符，以代码为准；本文档适合作为「第一遍建立索引」的入口而非长期引用。
- **反向使用**：如果你要新增一个注意力后端或 MoE 计算后端，本文档 3.1/3.2 中相应家族的条目（`attention_registry.py`、`moe_runner` 的 `register_fused_func`/`register_pre_permute`/`register_post_permute`）就是最直接的扩展点清单。
