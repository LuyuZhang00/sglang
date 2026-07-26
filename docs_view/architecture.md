# SGLang 项目架构与实现详解

> 本文档详细分析 SGLang 项目的整体架构、核心模块、实现逻辑以及各组件所承担的工作内容。

---

## 目录

- [第一部分：总体架构概览](#第一部分总体架构概览)
  - [1.1 项目定位与核心能力](#11-项目定位与核心能力)
  - [1.2 顶层目录结构](#12-顶层目录结构)
  - [1.3 三大核心子系统](#13-三大核心子系统)
  - [1.4 请求处理全流程](#14-请求处理全流程)
- [第二部分：python/sglang — 主 Python 包](#第二部分pythonsglang--主-python-包)
  - [2.1 公共 API 层](#21-公共-api-层)
  - [2.2 srt/ — SGLang Runtime 核心运行时](#22-srt--sglang-runtime-核心运行时)
  - [2.3 lang/ — 前端 DSL 语言](#23-lang--前端-dsl-语言)
  - [2.4 cli/ — 命令行接口](#24-cli--命令行接口)
  - [2.5 kernels/ — 自定义 GPU 内核](#25-kernels--自定义-gpu-内核)
  - [2.6 multimodal_gen/ — 多模态生成](#26-multimodal_gen--多模态生成)
- [第三部分：sgl-kernel — CUDA/C++ 内核库](#第三部分sgl-kernel--cudac-内核库)
  - [3.1 构建系统](#31-构建系统)
  - [3.2 内核分类详解](#32-内核分类详解)
  - [3.3 Python 绑定层](#33-python-绑定层)
- [第四部分：其他关键组件](#第四部分其他关键组件)
  - [4.1 Rust 工作空间](#41-rust-工作空间)
  - [4.2 sgl-model-gateway](#42-sgl-model-gateway)
  - [4.3 测试与 CI](#43-测试与-ci)

---

# 第一部分：总体架构概览

## 1.1 项目定位与核心能力

SGLang 是一个面向大语言模型（LLM）和多模态模型的高性能推理服务框架。其核心设计目标是实现**低延迟、高吞吐**的模型推理，支持从单 GPU 到大规模分布式集群的部署场景。

**核心特性：**
- **RadixAttention**：基于基数树的前缀缓存机制，自动复用相同前缀的 KV 缓存
- **零开销 CPU 调度器**：CPU 调度与 GPU 计算重叠执行
- **投机解码**：支持 EAGLE、EAGLE3、DFLASH、N-gram、MTP 等多种算法
- **连续批处理**：动态合并请求以最大化 GPU 利用率
- **分页注意力**：高效管理 KV 缓存内存
- **多种并行策略**：张量并行（TP）、流水线并行（PP）、数据并行（DP）、专家并行（EP）、上下文并行（CP）
- **结构化输出**：基于语法的约束解码（XGrammar、Outlines、LLGuidance）
- **量化支持**：FP4/FP8/INT4/AWQ/GPTQ/GGUF 等多种量化方案
- **多 LoRA 批处理**：运行时动态加载和切换 LoRA 适配器
- **Prefill-Decode 解耦**：将预填充和解码阶段分离到不同服务器

**支持的硬件平台：**
- NVIDIA GPU（GB200/B300/H100/A100，SM80-SM120a）
- AMD GPU（MI355/MI300，ROCm）
- Intel Xeon / Gaudi
- Google TPU
- 华为 Ascend NPU
- Moore Threads MUSA
- Apple Metal（macOS）
- CPU（x86_64 + aarch64）

## 1.2 顶层目录结构

```
sglang/
├── python/                  # 主 Python 包（sglang）
├── sgl-kernel/              # CUDA/C++ 内核库（独立包：sglang-kernel）
├── rust/                    # Rust 工作空间（gRPC 服务器、PyO3 绑定）
├── sgl-model-gateway/       # Rust 模型网关（axum + tonic + k8s）
├── benchmark/               # 46 个基准测试子目录
├── test/                    # 顶层测试目录（pytest）
├── docs_new/                # Mintlify 文档
├── docs_view/               # 本文档所在目录
├── docker/                  # Docker 镜像（CUDA、ROCm、ARM、NPU 等）
├── scripts/                 # CI 脚本、发布工具
├── examples/                # 使用示例
├── experimental/            # 实验性代码
├── proto/                   # Protobuf 定义
├── 3rdparty/                # 第三方代码
└── .claude/                 # Claude Code 规则和技能
```

## 1.3 三大核心子系统

SGLang 由三个主要子系统构成：

### （1）Python 运行时（`python/sglang/srt/`）
核心推理引擎，采用**三进程架构**，通过 ZMQ IPC 通信：
- **TokenizerManager**（主进程）：接收请求、分词、管理会话
- **Scheduler**（子进程）：调度批次、执行前向传播、管理 KV 缓存
- **DetokenizerManager**（子进程）：将 token ID 转换回文本

### （2）CUDA/C++ 内核库（`sgl-kernel/`）
提供高性能的底层计算原语：
- 注意力内核（FlashAttention 3、FlashMLA、CUTLASS MLA）
- 矩阵乘法（FP8/INT8/AWQ/GPTQ/GGUF 量化 GEMM）
- MoE 路由与分发
- 采样与归一化
- 投机解码辅助内核

### （3）前端 DSL 语言（`python/sglang/lang/`）
用于编写结构化 LLM 程序的领域特定语言：
- `gen()`、`select()` 等生成原语
- 角色标记（system/user/assistant）
- 多模态输入（image/video）
- 多后端支持（SGLang SRT、OpenAI、Anthropic、LiteLLM）

## 1.4 请求处理全流程

```
                          ┌─────────────────────────────────────────────────┐
                          │                  HTTP 请求                      │
                          └──────────────────────┬──────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │     HTTP Server (FastAPI, OpenAI 兼容 API)       │
                          │     python/sglang/srt/entrypoints/http_server.py │
                          └──────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │              TokenizerManager                    │
                          │     python/sglang/srt/managers/tokenizer_manager │
                          │     - 文本分词                                    │
                          │     - 多模态数据处理                               │
                          │     - 会话管理                                    │
                          │     - LoRA 注册                                   │
                          └──────────────────────┬───────────────────────────┘
                                                 │ ZMQ: TokenizedGenerateReqInput
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │      DataParallelController（可选）               │
                          │     python/sglang/srt/managers/data_parallel_ctrl│
                          │     - 跨多个调度器的负载均衡                        │
                          └──────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │                 Scheduler                        │
                          │     python/sglang/srt/managers/scheduler.py      │
                          │     - 请求队列管理                                │
                          │     - 调度策略（优先级、公平性）                     │
                          │     - 批次构建（ScheduleBatch）                    │
                          │     - 前向执行调度                                │
                          │     - KV 缓存管理（RadixCache）                   │
                          │     - 语法约束解码                                │
                          └──────────┬─────────────────────┬─────────────────┘
                                     │                     │
                                     ▼                     ▼
                          ┌─────────────────┐   ┌──────────────────────────┐
                          │  TpModelWorker   │   │  投机解码 Worker（可选）   │
                          │  srt/managers/   │   │  srt/speculative/        │
                          │  tp_worker.py    │   │  EAGLE/DFLASH/N-gram/MTP│
                          └────────┬────────┘   └────────────┬─────────────┘
                                   │                         │
                                   ▼                         ▼
                          ┌──────────────────────────────────────────────────┐
                          │               ModelRunner                        │
                          │     python/sglang/srt/model_executor/            │
                          │     - 模型加载与管理                              │
                          │     - CUDA Graph 捕获与回放                       │
                          │     - 前向传播执行                                │
                          │     - 注意力后端管理                              │
                          └──────────────────────┬───────────────────────────┘
                                                 │ ForwardBatch → model.forward()
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │          model.forward() → Logits               │
                          │     python/sglang/srt/models/                    │
                          │     210+ 模型架构实现                             │
                          └──────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │          LogitsProcessor + Sampler               │
                          │     srt/layers/logits_processor.py               │
                          │     srt/layers/sampler.py                        │
                          │     - 词汇表并行嵌入收集                          │
                          │     - 温度/Top-K/Top-P/Min-P 采样                 │
                          │     - 惩罚应用                                    │
                          │     - 语法约束掩码                                │
                          └──────────────────────┬───────────────────────────┘
                                                 │ ZMQ: BatchTokenIDOutput
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │           DetokenizerManager                     │
                          │     python/sglang/srt/managers/detokenizer_mgr   │
                          │     - 增量反分词                                  │
                          │     - 流式文本输出                                │
                          └──────────────────────┬───────────────────────────┘
                                                 │ ZMQ: BatchStrOutput
                                                 ▼
                          ┌──────────────────────────────────────────────────┐
                          │         TokenizerManager → HTTP Response         │
                          │     - 流式 SSE 返回                               │
                          │     - 完整响应返回                                │
                          └──────────────────────────────────────────────────┘
```

**关键数据流转换：**
1. `GenerateReqInput` → `TokenizedGenerateReqInput`（分词后）
2. `TokenizedGenerateReqInput` → `Req`（调度器内部请求对象）
3. `Req` → `ScheduleBatch`（CPU 侧批次，调度器管理）
4. `ScheduleBatch` → `ForwardBatch`（GPU 侧张量，ModelRunner 管理）
5. `ForwardBatch` → `LogitsProcessorOutput`（模型输出）
6. `LogitsProcessorOutput` → `BatchTokenIDOutput`（采样后的 token ID）
7. `BatchTokenIDOutput` → `BatchStrOutput`（反分词后的文本）

---

# 第二部分：python/sglang — 主 Python 包

## 2.1 公共 API 层

`python/sglang/__init__.py` 是整个项目的公共 API 入口，导出两个层次的接口：

### 前端语言 API
```python
# 结构化生成原语
from sglang import gen, gen_int, gen_string, select

# 角色标记
from sglang import system, user, assistant
from sglang import system_begin, system_end  # 显式开始/结束变体

# 多模态输入
from sglang import image, video

# 后端管理
from sglang import set_default_backend, flush_cache, get_server_info

# 运行时连接
from sglang import Runtime, RuntimeEndpoint

# 选择策略
from sglang import greedy_token_selection, token_length_normalized
```

### 运行时引擎 API
```python
# 核心引擎（延迟导入）
from sglang import Engine, ServerArgs

# 第三方后端（延迟导入）
from sglang import OpenAI, Anthropic, LiteLLM, VertexAI
```

**平台适配：** 在 macOS/ARM64 上，`__init__.py` 会在任何下游导入之前安装 `triton` 和 `torch.mps` 的存根（stub），确保代码在无 GPU 环境下也能导入。

## 2.2 srt/ — SGLang Runtime 核心运行时

`srt/` 是整个项目的核心，包含约 40 个子包和 5 个顶层 Python 文件。以下按功能域逐一分析。

### 2.2.1 入口层（`srt/entrypoints/`）

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `engine.py` | `Engine(EngineBase, EngineScoreMixin)` | **程序化入口**。启动子进程（Scheduler、Detokenizer），初始化 TokenizerManager，提供 `generate()`、`flush_cache()`、`update_weights_from_tensor()` 等方法 |
| `EngineBase.py` | `EngineBase`（ABC） | 引擎抽象基类，定义接口契约 |
| `http_server.py` | FastAPI 应用 | **HTTP 服务入口**。提供 OpenAI 兼容的 REST API（`/v1/chat/completions`、`/v1/completions` 等） |
| `grpc_server.py` | gRPC 服务 | gRPC 服务入口 |
| `openai/` | 多个 serving 模块 | OpenAI API 兼容层：chat、completions、embedding、classify、rerank、score、responses |
| `anthropic/` | | Anthropic Messages API 兼容 |
| `ollama/` | | Ollama API 兼容 |
| `search/` | | 搜索工具集成（Exa 客户端） |

**Engine 启动流程：**
1. 解析 `ServerArgs` 参数
2. `_launch_subprocesses()` 启动 Scheduler 和 DetokenizerManager 子进程
3. 初始化 ZMQ IPC 通道
4. 创建 TokenizerManager 实例
5. 注册 `atexit` 关闭处理器

### 2.2.2 管理器层（`srt/managers/`）

这是运行时的核心调度层，包含三个主要进程和共享数据结构。

#### TokenizerManager（`tokenizer_manager.py`，142KB）

**职责：**
- 接收来自 HTTP 层的请求
- 文本分词（tokenization）
- 多模态数据处理（图像、视频、音频）
- 会话（session）管理
- LoRA 适配器注册
- 权重更新协调
- 解耦（disaggregation）引导
- 通过 ZMQ 将分词后的请求发送给 Scheduler

**关键类：**
- `TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin)`（line 265）
- `ReqState`（line 172）：每个请求的状态跟踪
- `InputFormat`（line 257）：输入格式枚举

#### Scheduler（`scheduler.py`，203KB — 最大的文件）

**职责：**
- 接收来自 TokenizerManager 的请求
- 管理等待队列
- 应用调度策略（优先级、公平性）
- 构建 `ScheduleBatch`
- 调用 `TpModelWorker` 执行前向传播
- 处理批次结果
- 将 token ID 发送给 DetokenizerManager
- 管理 KV 缓存（RadixCache）
- 协调投机解码
- 处理解耦预填充/解码

**关键类：**
```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,   # 解耦解码
    SchedulerDisaggregationPrefillMixin,  # 解耦预填充
    SchedulerMultiplexMixin,              # 多路复用
    SchedulerPPMixin,                     # 流水线并行
    SchedulerDllmMixin,                   # 离散 LLM
    SchedulerMlxOverlapMixin,             # MLX 重叠
):
```

**主事件循环（`event_loop_normal`，line 1523）：**
```python
while True:
    recv_requests()              # 1. 接收请求
    process_input_requests()     # 2. 处理输入请求
    get_next_batch_to_run()      # 3. 获取下一个批次
    run_batch()                  # 4. 执行批次
    process_batch_result()       # 5. 处理批次结果
    on_idle()                    # 6. 空闲时执行清理
```

**重叠事件循环（`event_loop_overlap`，line 1557）：**
将上一批次的 CPU 结果处理与当前批次的 GPU 计算重叠执行，通过 `result_queue` 双端队列实现。

#### DetokenizerManager（`detokenizer_manager.py`，22KB）

**职责：**
- 接收来自 Scheduler 的 token ID 批次
- 执行增量反分词（incremental detokenization）
- 将文本结果发送回 TokenizerManager
- 维护每个请求的解码状态

**关键类：**
- `DetokenizerManager(MultiHttpWorkerDetokenizerMixin)`（line 91）
- `DecodeStatus`（line 64）：跟踪已解码文本、解码 ID、代理/读取偏移量
- `LimitedCapacityDict`（line 499）：LRU 缓存（默认 65536 条目）

#### IO 结构（`io_struct.py`，90KB）

定义进程间传输的所有数据结构，使用 `msgspec.Struct` 实现高效序列化。

**主要请求类型（100+ 个类）：**

| 类别 | 关键类 | 说明 |
|------|--------|------|
| 用户请求 | `GenerateReqInput` | 用户面向的生成请求（文本、采样参数、多模态数据、LoRA、会话、优先级） |
| 内部请求 | `TokenizedGenerateReqInput` | 分词后传递给调度器的内部形式 |
| 批次请求 | `BatchTokenizedGenerateReqInput` | 批量版本 |
| 嵌入请求 | `EmbeddingReqInput` | 嵌入请求 |
| 输出 | `BatchTokenIDOutput`、`BatchStrOutput`、`BatchEmbeddingOutput` | 调度器到反分词器的输出 |
| 权重更新 | `UpdateWeightFromDiskReqInput`、`UpdateWeightsFromTensorReqInput` | 热权重更新 |
| 控制 | `AbortReq`、`FlushCacheReqInput`、`ShutdownReq`、`ProfileReq` | 控制命令 |
| 会话 | `OpenSessionReqInput`、`CloseSessionReqInput` | 会话管理 |
| LoRA | `LoadLoRAAdapterReqInput`、`UnloadLoRAAdapterReqInput` | LoRA 管理 |

#### 调度批次（`schedule_batch.py`，132KB）

**核心数据结构：**

```python
class Req(ReqDllmMixin):
    """单个请求对象"""
    rid: str                          # 请求 ID
    origin_input_ids: List[int]       # 原始输入 token ID
    sampling_params: SamplingParams   # 采样参数
    return_logprob: bool              # 是否返回对数概率
    lora_id: int                      # LoRA 适配器 ID
    priority: int                     # 请求优先级
    # ... KV 缓存状态、输出 token ID、完成原因等

class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """调度器管理的批次"""
    reqs: List[Req]                   # 请求列表
    req_to_token_pool: ReqToTokenPool # 请求到 token 的映射池
    token_to_kv_pool_allocator: ...   # KV 缓存分配器
    tree_cache: RadixCache            # 基数树缓存
    # ... 批次级状态（是否满、分块预填充、DP 注意力等）
```

**数据流：** `ScheduleBatch`（CPU，调度器管理）→ `ForwardBatch`（GPU 张量，ModelRunner 管理）

#### TpModelWorker（`tp_worker.py`，26KB）

**职责：**
- 封装 `ModelRunner`，向调度器暴露前向传播接口
- 管理 NCCL 通信组
- 处理权重更新
- 管理 LoRA 适配器

```python
class TpModelWorker(BaseTpWorker):
    def __init__(self, server_args, gpu_id, tp_rank, ...):
        # 初始化模型配置、ModelRunner、多层 EAGLE Runner、DLLM 算法

    def forward_batch_generation(self, batch: ScheduleBatch) -> ModelRunnerOutput:
        # 执行前向传播
```

#### 其他管理器组件

| 文件 | 职责 |
|------|------|
| `schedule_policy.py` | 请求调度策略和前缀缓存匹配 |
| `data_parallel_controller.py` | 跨多个数据并行工作者的请求分发（轮询、总请求数、总 token 数） |
| `communicator.py` | `FanOutCommunicator` — ZMQ 扇出通信模式 |
| `scheduler_components/` | 调度器模块化子组件：`request_receiver.py`、`output_streamer.py`、`batch_result_processor.py`、`metrics_reporter.py`、`weight_updater.py`、`ipc_channels.py` 等 |

### 2.2.3 模型执行层（`srt/model_executor/`）

#### ModelRunner（`model_runner.py`，76KB）

**职责：**
- 加载模型权重
- 管理 GPU 内存池
- 管理注意力后端
- 管理 CUDA Graph
- 执行前向传播
- 处理 MLA（Multi-head Latent Attention）
- 处理投机解码
- 处理弹性专家并行
- 处理 LoRA
- 处理多模态模型

```python
class ModelRunner:
    def __init__(self, model_config, ...):
        # 加载模型、初始化注意力后端、创建内存池、捕获 CUDA Graph

    def forward(self, forward_batch: ForwardBatch) -> ModelRunnerOutput:
        # 执行前向传播
```

#### ForwardBatch（`forward_batch_info.py`，73KB）

**GPU 侧的批次表示，从 `ScheduleBatch` 构建：**

```python
class ForwardMode(IntEnum):
    EXTEND = 0          # 预填充
    DECODE = 1          # 解码
    TARGET_VERIFY = 2   # 投机验证
    DRAFT_EXTEND = 3    # 草稿扩展
    # ...

class ForwardBatch(ForwardBatchDeepSeekMHAMixin):
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    # ... 多模态输入、采样信息、投机信息、LoRA ID、DP 注意力状态
```

#### Runner 实现（`runner/`）

| 文件 | 类 | 职责 |
|------|-----|------|
| `base_runner.py` | `BaseRunner`（ABC） | 基类，处理前向传播编排、上下文并行、DP 注意力填充 |
| `eager_runner.py` | `EagerRunner` | 无 CUDA Graph 的即时执行模式 |
| `decode_cuda_graph_runner.py` | `DecodeCudaGraphRunner` | 解码阶段的 CUDA Graph 捕获与回放 |
| `prefill_cuda_graph_runner.py` | `PrefillCudaGraphRunner` | 预填充阶段的 CUDA Graph 捕获与回放 |

**CUDA Graph 后端模式：**
- `full`：每个形状一个完整图
- `breakable`：分段捕获
- `tc_piecewise`：张量核心分段

### 2.2.4 神经网络层（`srt/layers/`）

#### 注意力后端（`layers/attention/`，40+ 文件）

SGLang 支持多种注意力后端，针对不同硬件和算法优化：

| 后端 | 文件 | 适用场景 |
|------|------|----------|
| FlashInfer | `flashinfer_backend.py`（100KB） | NVIDIA GPU 默认后端 |
| FlashInfer MLA | `flashinfer_mla_backend.py`（42KB） | DeepSeek MLA 注意力 |
| FlashAttention | `flashattention_backend.py`（166KB） | FlashAttention 3 |
| Triton | `triton_backend.py`（82KB） | Triton 实现 |
| DeepSeek V4 | `deepseek_v4_backend.py`（88KB） | DeepSeek-V4 专用 |
| DSA | `dsa_backend.py`（138KB） | 动态稀疏注意力 |
| TRT-LLM | `trtllm_mha_backend.py`、`trtllm_mla_backend.py` | TensorRT-LLM 集成 |
| FlashMLA | `flashmla_backend.py`（26KB） | FlashMLA 注意力 |
| AITER | `aiter_backend.py`（117KB） | AMD GPU |
| XPU | `xpu_backend.py`（61KB） | Intel XPU |
| Hybrid | `hybrid_attn_backend.py` | 混合注意力 |

**核心抽象：**
- `RadixAttention`（`radix_attention.py`）：与 RadixCache 集成的注意力层，自动管理 KV 缓存的分配和复用

#### 线性层（`linear.py`，70KB）

支持张量并行的线性层实现：

| 类 | 用途 |
|----|------|
| `LinearBase` | 基类 |
| `ColumnParallelLinear` | 列并行线性层（权重按列切分） |
| `MergedColumnParallelLinear` | 合并列并行（多个线性层合并） |
| `QKVParallelLinear` | QKV 投影的并行实现 |
| `RowParallelLinear` | 行并行线性层（权重按行切分） |
| `ReplicatedLinear` | 复制线性层（无并行） |

#### 层归一化（`layernorm.py`，45KB）

跨平台的归一化实现：
- `RMSNorm`：标准 RMS 归一化
- `LayerNorm`：标准层归一化
- `GemmaRMSNorm`、`Gemma3RMSNorm`、`Gemma4RMSNorm`：Gemma 系列专用变体
- 所有类继承 `MultiPlatformOp`，支持 CUDA/ROCm/XPU/NPU/CPU

#### 量化层（`layers/quantization/`，50+ 文件）

支持多种量化方案：

| 量化方式 | 关键文件 | 说明 |
|----------|----------|------|
| FP8 | `fp8.py`（110KB） | 8 位浮点量化，支持在线和离线量化 |
| INT8 | `w8a8_int8.py` | 8 位整数量化 |
| FP4/NVFP4 | `fp4_utils.py`、`nvfp4_online.py` | 4 位浮点量化 |
| MXFP4/MXFP8 | `mxfp4.py`（59KB） | 微格式量化 |
| AWQ | `awq/` | 激活感知权重量化 |
| GPTQ | `gptq/` | GPTQ 量化 |
| GGUF | `gguf.py`（36KB） | GGML/GGUF 格式量化 |
| Marlin | `marlin_utils.py` | Marlin 量化 GEMM |
| BitsAndBytes | `bitsandbytes.py` | BitsAndBytes 量化 |
| ModelOpt | `modelopt_quant.py`（109KB） | NVIDIA ModelOpt 量化 |

#### MoE 层（`layers/moe/`，20+ 文件）

混合专家模型的实现：

| 组件 | 文件 | 职责 |
|------|------|------|
| Top-K 选择 | `topk.py`（89KB） | 专家路由的 Top-K 选择算法 |
| CUTLASS MoE | `cutlass_moe.py` | 基于 CUTLASS 的 MoE GEMM |
| FlashInfer MoE | `flashinfer_trtllm_moe.py` | FlashInfer/TRT-LLM MoE |
| 专家并行 | `ep_moe/` | EP MoE 实现 |
| Token 分发 | `token_dispatcher/` | Token 到专家的分发策略 |

#### 采样与 Logits 处理

| 文件 | 职责 |
|------|------|
| `sampler.py`（33KB） | Token 采样：Top-K、Top-P、Min-P、贪心采样，使用 FlashInfer 内核 |
| `logits_processor.py`（41KB） | 原始 logits 处理：词汇表并行嵌入收集、对数概率计算、softcapping、DP 注意力聚合 |
| `srt/sampling/sampling_params.py` | `SamplingParams` 类：temperature、top_k、top_p、min_p、频率/存在/重复惩罚、停止字符串、最大 token 数 |
| `srt/sampling/sampling_batch_info.py` | `SamplingBatchInfo`：批量采样状态 |
| `srt/sampling/penaltylib/` | 惩罚库 |

#### 其他层组件

| 文件 | 职责 |
|------|------|
| `vocab_parallel_embedding.py`（27KB） | 词汇表并行嵌入 |
| `communicator.py`（57KB） | 张量/模型并行通信原语 |
| `dp_attention.py`（30KB） | 数据并行注意力实现 |
| `rotary_embedding/` | RoPE 实现（base、factory、mrope、yarn、rope_variant） |
| `activation.py` | 激活函数 |
| `pooler.py` | 嵌入池化层 |
| `parameter.py` | 自定义参数类 |

### 2.2.5 内存管理（`srt/mem_cache/`）

SGLang 采用两级内存池架构：

```
请求 → ReqToTokenPool → TokenToKVPoolAllocator → KVCache（物理存储）
         请求到token映射      token索引管理           实际KV数据
```

#### 核心组件

| 文件 | 类 | 职责 |
|------|-----|------|
| `memory_pool.py`（205KB） | `ReqToTokenPool` | 请求 ID 到 token 槽位索引的映射 |
| | `MHATokenToKVPool` | 多头注意力 KV 池（标准 Transformer） |
| | `MLATokenToKVPool` | 多头潜在注意力 KV 池（DeepSeek MLA） |
| | `DSATokenToKVPool` | 动态稀疏注意力 KV 池 |
| | `MambaPool` | Mamba 状态空间模型池 |
| `radix_cache.py`（31KB） | `RadixCache` | **基数树前缀缓存**。自动跨请求复用相同前缀的 KV 缓存 |
| `base_prefix_cache.py` | `BasePrefixCache`（ABC） | 前缀缓存抽象基类 |
| `allocation.py`（25KB） | | 内存分配逻辑（`alloc_for_decode`、`alloc_for_extend`） |
| `multi_ended_allocator.py`（111KB） | | 多策略高级分配器 |
| `kv_cache_configurator.py`（84KB） | | KV 缓存布局和大小配置 |
| `unified_memory_pool.py`（53KB） | | 统一内存管理 |
| `unified_radix_cache.py`（122KB） | | 统一基数缓存 |

#### 分层缓存

| 文件 | 职责 |
|------|------|
| `hiradix_cache.py`（78KB） | 分层基数缓存（GPU + 主机内存） |
| `hicache_storage.py`（25KB） | 分层缓存存储后端 |
| `swa_radix_cache.py`（59KB） | 滑动窗口注意力缓存 |
| `mamba_radix_cache.py`（57KB） | Mamba 专用缓存 |

### 2.2.6 投机解码（`srt/speculative/`）

投机解码使用较小/较快的"草稿"模型提出 token，然后由"目标"模型并行验证。

#### 支持的算法

| 算法 | 文件 | 说明 |
|------|------|------|
| EAGLE | `eagle_worker_v2.py`（64KB） | EAGLE v2 投机解码 |
| EAGLE3 | `eagle_worker_common.py` | EAGLE v3 变体 |
| DFLASH | `dflash_worker_v2.py`（79KB） | DFLASH v2 投机解码 |
| N-gram | `ngram_worker.py`（22KB） | 基于语料库匹配的 N-gram 投机解码 |
| Frozen KV MTP | `frozen_kv_mtp_worker_v2.py`（31KB） | 冻结 KV 多 token 预测 |
| Standalone | `standalone_worker_v2.py` | 独立投机工作者 |

**架构设计：**
- `SpeculativeAlgorithm` 枚举定义内置算法
- `CustomSpecAlgo` 支持插件注册的自定义算法
- `BaseSpecWorker` 和 `EagleDraftWorkerBase` 定义抽象接口
- 每个算法有独立的 Worker、Info、CUDA Graph Runner

### 2.2.7 分布式通信（`srt/distributed/`）

| 文件 | 职责 |
|------|------|
| `parallel_state.py`（113KB） | 管理所有分布式进程组（TP、PP、DP、EP、CP） |
| `device_communicators/pynccl.py` | NCCL Python 封装 |
| `device_communicators/custom_all_reduce.py` | 自定义 All-Reduce（IPC 共享内存） |
| `device_communicators/shm_broadcast.py` | 共享内存广播 |
| `device_communicators/mooncake_transfer_engine.py` | Mooncake 传输引擎 |
| `device_communicators/pymscclpp.py` | MSCCL++ 集成 |

**支持的并行策略：**
- **张量并行（TP）**：列/行并行线性层、词汇表并行嵌入
- **流水线并行（PP）**：通过 `SchedulerPPMixin`、`PPProxyTensors`
- **数据并行（DP）**：DP 注意力、数据并行控制器
- **专家并行（EP）**：MoE 模型的专家分布，支持弹性 EP
- **上下文并行（CP）**：长序列处理

### 2.2.8 约束解码（`srt/constrained/`）

基于语法的约束解码，确保模型输出符合指定的格式（JSON Schema、正则表达式、EBNF 等）。

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `base_grammar_backend.py` | `BaseGrammarObject` | 语法对象基类：`accept_token`、`rollback`、`try_jump_forward`、`fill_vocab_mask` |
| `grammar_manager.py` | `GrammarManager` | 与调度器集成的语法管理器 |
| `xgrammar_backend.py` | | XGrammar 后端（高性能语法引擎） |
| `llguidance_backend.py` | | LLGuidance 后端 |
| `outlines_backend.py` | | Outlines 后端 |
| `reasoner_grammar_backend.py` | | 推理模型的严格思考模式语法 |

### 2.2.9 解耦服务（`srt/disaggregation/`）

将预填充（prefill）和解码（decode）阶段分离到不同服务器，以优化资源利用。

**生命周期：**

**预填充服务器：**
1. Bootstrap Queue → 初始化发送器，与解码服务器握手
2. Waiting Queue → 弹出请求，执行前向传播，加入进行中队列
3. Inflight Queue → 轮询传输完成，返回请求

**解码服务器：**
1. PreallocQueue → 初始化接收器，握手，预分配 KV
2. TransferQueue → 轮询传输完成
3. WaitingQueue → 构建 PrebuiltExtendBatch（跳过预填充，仅填充元数据）
4. RunningBatch → 合并到运行批次进行解码

**传输后端：**

| 后端 | 目录 | 说明 |
|------|------|------|
| NIXL | `nixl/` | NVIDIA 传输库 |
| Mooncake | `mooncake/` | Mooncake 传输引擎 |
| Mori | `mori/` | Mori 传输引擎 |
| Ascend | `ascend/` | 华为 Ascend NPU |
| Fake | `fake/` | 测试用模拟传输 |

### 2.2.10 其他 SRT 子系统

| 目录 | 职责 |
|------|------|
| `srt/models/` | 210+ 模型架构实现（LLaMA、Qwen、DeepSeek、Gemma、Mistral、BERT、CLIP 等） |
| `srt/model_loader/` | 模型权重加载 |
| `srt/lora/` | LoRA 适配器管理、加载、驱逐、重叠加载 |
| `srt/configs/` | 模型配置解析器 |
| `srt/server_args.py`（396KB） | `ServerArgs` — 所有服务器配置（~400+ 参数） |
| `srt/environ.py`（60KB） | 环境变量定义（`Envs` 类） |
| `srt/hardware_backend/` | 硬件特定后端（GPU、CPU、NPU、XPU、MUSA、MLX） |
| `srt/multimodal/` | 多模态数据处理 |
| `srt/function_call/` | 函数/工具调用支持 |
| `srt/observability/` | 指标、追踪、CPU 监控 |
| `srt/eplb/` | 专家并行负载均衡 |
| `srt/elastic_ep/` | 弹性专家并行 |
| `srt/dllm/` | 离散 LLM 混入 |
| `srt/batch_overlap/` | 批次重叠调度 |
| `srt/compilation/` | 图编译支持 |
| `srt/plugins/` | 插件系统 |
| `srt/session/` | 会话管理 |
| `srt/weight_sync/` | 跨 rank 权重同步 |
| `srt/tokenizer/` | 分词器工具 |
| `srt/parser/` | 推理解析器、模板检测 |
| `srt/ray/` | Ray 分布式集成 |

### 2.2.11 模型注册（`srt/models/registry.py`）

```python
class _ModelRegistry:
    """模型注册表，从 vLLM 适配"""
    models: Dict[str, Type[nn.Module]]  # 架构名 → 模型类

# 使用方式：
ModelRegistry.register("sglang.srt.models")  # 自动扫描所有模型文件
ModelRegistry.resolve_model_cls(["LlamaForCausalLM"])  # 解析模型类
```

**特性：**
- 自动扫描包内所有模块，查找 `EntryClass` 属性
- 支持通过 `SGLANG_EXTERNAL_MODEL_PACKAGE` 环境变量注册外部模型
- 支持通过 `SGLANG_DISABLED_MODEL_ARCHS` 禁用特定架构
- 回退到 `TransformersForCausalLM` 通用实现

## 2.3 lang/ — 前端 DSL 语言

SGLang 提供了一种领域特定语言（DSL），用于编写结构化的 LLM 程序。

### 核心 API

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

@sgl.function
def structured_output(s, question):
    s += sgl.user(question)
    s += sgl.assistant(
        sgl.select("answer", ["yes", "no"])
    )
```

### 架构组件

| 文件 | 职责 |
|------|------|
| `api.py` | 公共 API：`function()`、`gen()`、`select()`、`image()`、`video()`、`set_default_backend()` |
| `ir.py` | 中间表示（IR）：`SglFunction`、`SglGen`、`SglSelect`、`SglImage`、`SglVideo`、`SglRoleBegin/End` |
| `interpreter.py` | `StreamExecutor` 和 `ProgramState` — 执行 IR 树，管理生成状态，与后端通信 |
| `tracer.py` | 将 Python 函数追踪为 IR |
| `chat_template.py` | 聊天模板处理 |
| `choices.py` | 选择策略：贪心、token 长度归一化、无条件似然归一化 |

### 后端适配器（`lang/backend/`）

| 后端 | 文件 | 说明 |
|------|------|------|
| SGLang SRT | `runtime_endpoint.py` | 连接到 SGLang 自己的 SRT 服务器 |
| OpenAI | `openai.py` | OpenAI API |
| Anthropic | `anthropic.py` | Anthropic API |
| LiteLLM | `litellm.py` | LiteLLM 统一接口 |
| VertexAI | `vertexai.py` | Google Vertex AI |
| Crusoe | `crusoe.py` | Crusoe Cloud |

## 2.4 cli/ — 命令行接口

```bash
sglang serve --model-path meta-llama/Llama-3-8B --port 30000  # 启动服务器
sglang generate --model-path ... --prompt "Hello"              # 一次性生成
sglang killall                                                  # 杀死所有 SGLang 进程
sglang version                                                  # 显示版本
```

| 文件 | 命令 | 职责 |
|------|------|------|
| `main.py` | | CLI 入口（argparse 分发） |
| `serve.py` | `sglang serve` | 启动 HTTP 推理服务器 |
| `generate.py` | `sglang generate` | 一次性生成 |
| `killall.py` | `sglang killall` | 杀死所有 SGLang 进程 |

## 2.5 kernels/ — 自定义 GPU 内核

`python/sglang/kernels/` 包含 Triton 和 JIT 编译的自定义内核。

```
kernels/
├── __init__.py          # 内核注册表和初始化
├── fused_op.py          # 融合操作封装
├── registry.py          # 内核变体注册表
├── selector.py          # 运行时内核选择逻辑
├── spec.py              # 内核规范
├── ops/                 # 18 个子目录的内核实现
│   ├── attention/       # 注意力内核
│   ├── gemm/            # 矩阵乘法内核
│   ├── sampling/        # 采样内核
│   ├── moe/             # MoE 内核
│   ├── kvcache/         # KV 缓存操作
│   ├── mamba/           # Mamba/SSM 内核
│   ├── quantization/    # 量化内核
│   ├── activation/      # 激活函数
│   ├── layernorm/       # 层归一化
│   ├── embeddings/      # 嵌入操作
│   ├── grammar/         # 语法约束
│   ├── memory/          # 内存操作
│   ├── communication/   # 通信操作
│   ├── elementwise/     # 逐元素操作
│   ├── diffusion/       # 扩散模型
│   ├── speculative/     # 投机解码
│   └── ...
└── jit/                 # JIT 编译基础设施
    ├── csrc/            # C++ 源码
    ├── include/         # 头文件
    ├── utils/           # 工具
    └── benchmark/       # 基准测试
```

## 2.6 multimodal_gen/ — 多模态生成

独立的图像/视频生成子系统，基于扩散模型：

```
multimodal_gen/
├── registry.py          # 模型/应用注册表（42KB）
├── envs.py              # 环境变量
├── utils.py             # 共享工具（27KB）
├── runtime/             # 20 个子目录：扩散管线、调度器、编码器、解码器等
├── apps/                # 应用层封装
├── configs/             # 生成模型配置
├── csrc/                # C++ 扩展
├── test/                # 测试
└── benchmarks/          # 基准测试
```

---

# 第三部分：sgl-kernel — CUDA/C++ 内核库

`sgl-kernel`（PyPI 包名 `sglang-kernel`，Python 导入路径 `sgl_kernel`）是 SGLang 的高性能 CUDA/C++ 内核库，版本 0.4.5。

## 3.1 构建系统

**构建工具：** scikit-build-core + CMake（>=3.26）

**构建产物：** 四个独立的共享库

| 共享库 | 目标架构 | 安装路径 | 说明 |
|--------|----------|----------|------|
| `common_ops`（SM90） | Hopper GPU | `sgl_kernel/sm90/` | 使用 `-use_fast_math` |
| `common_ops`（SM100+） | Blackwell+ GPU | `sgl_kernel/sm100/` | 使用精确数学 |
| `flash_ops`（可选） | SM80/SM86/SM90a | | FlashAttention 3（需 CUDA >= 12.4） |
| `flashmla_ops` | SM90a/SM100 | | FlashMLA 内核 |
| `infllm_ops` | | | InfLLM-V2 FlashAttention |
| `spatial_ops` | | | Green Context CUDA 流管理 |

**GPU 架构目标：**
- SM80、SM89（可选，旧 GPU）
- SM90、SM90a（Hopper — 始终启用）
- SM100a、SM120a、SM103a、SM110a、SM121a（Blackwell+，需 CUDA >= 12.8）

**第三方依赖（通过 CMake FetchContent 获取）：**
- NVIDIA CUTLASS（GEMM/注意力内核模板）
- fmtlib（格式化）
- Triton（Python 包安装）
- FlashInfer（归一化、重归一化、采样内核）
- FlashAttention（sgl-attn 分支，稀疏注意力）
- FlashMLA（DeepSeek MLA 注意力）

## 3.2 内核分类详解

### 3.2.1 注意力内核（`csrc/attention/`）

| 内核 | 文件 | 功能 |
|------|------|------|
| 合并注意力状态 | `merge_attn_states.cu` | 合并来自不同 KV 分割的两个注意力状态（value + logsumexp） |
| CUTLASS MLA | `cutlass_mla_kernel.cu` | 基于 CUTLASS 的 Multi-head Latent Attention 解码内核（DeepSeek 模型） |
| SM100 MLA | `cutlass_sm100_mla/` | SM100 专用 MLA 实现（使用 TMA 和 warp 特化） |
| 垂直/斜线索引 | `vertical_slash_index.cu` | 转换块稀疏注意力的索引格式 |

### 3.2.2 FlashAttention（`csrc/flash_extension.cc`、`csrc/flashmla_extension.cc`）

**FlashAttention 3（flash_ops）：**
- 通过 sgl-attn 分支实现
- 支持 bf16/fp16/fp8
- 分页 KV 缓存、变长序列、因果/非因果、滑动窗口、softcapping、稀疏掩码

**FlashMLA（flashmla_ops）：**
- SM90 和 SM100 的密集/稀疏 MLA 解码/预填充
- 支持 fp16、bf16、fp8 KV 缓存
- 包含元数据调度内核

### 3.2.3 逐元素/归一化（`csrc/elementwise/`）

| 内核 | 文件 | 功能 |
|------|------|------|
| 融合 RMSNorm | `fused_add_rms_norm_kernel.cu` | RMSNorm 和融合 add+RMSNorm（标准和 Gemma 变体） |
| 激活函数 | `activation.cu` | SiLU-and-mul、GELU-tanh-and-mul、GELU-and-mul |
| 旋转位置编码 | `pos_enc.cu` | RoPE（Rotary Position Embedding） |
| GPU 复制 | `copy.cu` | 无复制引擎的 GPU 复制（用于 PDL 重叠） |
| MLA 拼接 | `concat_mla.cu` | MLA 辅助拼接（k_nope + k_rope、absorb-q） |
| Top-K | `topk.cu` | 快速 Top-K 选择（带页表变换） |
| DSV4 归一化+RoPE | `dsv4_norm_rope.cu` | DeepSeek-V4 融合 Q/K 归一化 + RoPE |
| DSV4 Top-K | `deepseek_v4_topk.cu` | DeepSeek-V4 Top-K 变换（ROCm 变体） |

### 3.2.4 GEMM/量化（`csrc/gemm/`）

| 内核 | 文件 | 功能 |
|------|------|------|
| FP8 GEMM | `fp8_gemm_kernel.cu` | FP8 缩放矩阵乘法（通过 CUTLASS） |
| INT8 GEMM | `int8_gemm_kernel.cu` | INT8 缩放矩阵乘法 |
| AWQ 反量化 | `awq_kernel.cu` | AWQ 反量化 |
| Per-token FP8 量化 | `per_token_quant_fp8.cu` | 逐 token FP8 量化 |
| Per-token-group 量化 | `per_token_group_quant_8bit.cu` | 逐 token 组 8 位量化（FP8/INT8） |
| GPTQ | `gptq/gptq_kernel.cu` | GPTQ 量化 GEMM（2/3/4/8 位） |
| Marlin | `marlin/` | Marlin 量化 GEMM 模板 |

### 3.2.5 MoE（`csrc/moe/`）

| 内核 | 文件 | 功能 |
|------|------|------|
| MoE 对齐 | `moe_align_kernel.cu` | 将 token 对齐到专家块 |
| Top-K Softmax | `moe_topk_softmax_kernels.cu` | MoE 路由的 Top-K Softmax 门控 |
| Top-K Sigmoid | `moe_topk_sigmoid_kernels.cu` | MoE 路由的 Top-K Sigmoid 门控 |
| MoE 求和 | `moe_sum.cu` | 专家输出求和 |
| FP8 块级 MoE | `fp8_blockwise_moe_kernel.cu` | FP8 块级缩放分组矩阵乘法 |
| 准备 MoE 输入 | `prepare_moe_input.cu` | 准备 MoE GEMM 的排列偏移和问题大小 |
| CUTLASS W4A8 MoE | `cutlass_moe/w4a8/` | 4 位权重、8 位激活的分组 GEMM |

### 3.2.6 投机解码（`csrc/speculative/`）

| 内核 | 文件 | 功能 |
|------|------|------|
| 投机采样 | `speculative_sampling.cu` | 树状投机采样（仅目标验证） |
| EAGLE 工具 | `eagle_utils.cu` | EAGLE 投机解码工具（树构建、索引重建） |
| N-gram 工具 | `ngram_utils.cu` | N-gram 投机解码工具 |
| Packbit | `packbit.cu` | 分段 packbits（压缩位掩码表示） |

### 3.2.7 其他内核

| 类别 | 目录 | 功能 |
|------|------|------|
| 采样 | FlashInfer 集成 | `top_k_renorm_probs`、`top_p_renorm_probs` |
| 语法约束 | `csrc/grammar/` | 应用 token 位掩码到 logits |
| KV 缓存 I/O | `csrc/kvcacheio/` | 在不同内存布局间传输 KV 缓存数据 |
| All-Reduce | `csrc/allreduce/` | 自定义 All-Reduce（IPC 共享内存） |
| Mamba/SSM | `csrc/mamba/` | 因果 1D 卷积更新和前向传播 |
| GGUF | `csrc/quantization/gguf/` | GGML/GGUF 量化操作 |
| InfLLM-V2 | `csrc/infllm_v2/` | 长上下文推理的最大池化和 FlashAttention |
| Green Context | `csrc/spatial/` | SM 分区的 Green Context CUDA 流 |
| CPU | `csrc/cpu/` | 完整的 CPU 推理实现（aarch64 NEON + x86_64） |
| Metal | `csrc/metal/` | Apple Metal 着色器（融合 RoPE + 池化） |
| MUSA | `csrc/musa/` | Moore Threads MUSA 变体 |

## 3.3 Python 绑定层

`python/sgl_kernel/__init__.py` 在运行时根据检测到的 GPU 架构动态加载正确的 `common_ops` 共享库。

**主要 Python 模块：**

| 模块 | 导出 |
|------|------|
| `allreduce.py` | `init_custom_ar`、`all_reduce` |
| `attention.py` | `merge_state_v2`、`cutlass_mla_decode` |
| `elementwise.py` | `rmsnorm`、`fused_add_rmsnorm`、`silu_and_mul`、`rotary_embedding` 等 |
| `gemm.py` | `awq_dequantize`、`fp8_scaled_mm`、`int8_scaled_mm`、`gptq_gemm` 等 |
| `moe.py` | `moe_align_block_size`、`topk_softmax`、`moe_sum`、`fp8_blockwise_scaled_grouped_mm` 等 |
| `sampling.py` | `top_k_renorm_prob`、`top_p_renorm_prob` |
| `speculative.py` | `tree_speculative_sampling_target_only`、`verify_tree_greedy` 等 |
| `grammar.py` | `apply_token_bitmask_inplace_cuda` |
| `kvcacheio.py` | KV 缓存传输函数 |
| `flash_attn.py` | FlashAttention 3 封装 |
| `flash_mla.py` | FlashMLA 封装 |
| `top_k.py` | `fast_topk`、`fast_topk_v2` |

**架构模式：**
- **双架构构建：** SM90 使用 `-use_fast_math`；SM100+ 使用精确数学。Python 加载器在运行时选择正确的 `.so`
- **PyTorch 自定义算子：** 所有内核通过 `TORCH_LIBRARY_FRAGMENT` 注册，支持 `torch.compile`
- **Torch Shim：** `include/sgl_kernel_torch_shim.h` 自动转换 C++ 原生类型和 PyTorch 类型

---

# 第四部分：其他关键组件

## 4.1 Rust 工作空间

`rust/` 目录包含一个 Rust 工作空间，提供高性能的原生组件：

```
rust/
├── Cargo.toml           # 工作空间配置
├── sglang-grpc/         # gRPC 服务器实现
├── sglang-mm/           # 多模态处理
└── sglang-server/       # 服务器实现
```

- **版本：** Rust 2024 Edition
- **Python 绑定：** 通过 PyO3 实现
- **构建：** 通过 `setuptools-rust` 自动构建（设置 `SGLANG_BUILD_RUST_EXTS` 环境变量）

## 4.2 sgl-model-gateway

`sgl-model-gateway/` 是一个独立的 Rust 项目，提供模型网关功能：

- **Web 框架：** axum
- **gRPC：** tonic
- **Kubernetes 集成：** 原生 k8s 支持
- **功能：** 模型路由、负载均衡、健康检查

## 4.3 测试与 CI

### 测试框架

- **Python：** pytest + unittest（`asyncio_mode = auto`）
- **Rust：** `cargo test`
- **sgl-kernel：** pytest（`sgl-kernel/tests/`，38 个测试文件）

### 运行测试

```bash
# 单个测试文件
python3 test/srt/<test_file>.py

# 单个测试函数
python3 test/srt/<test_file>.py TestClass.test_method

# pytest 直接调用
pytest test/srt/<test_file>.py -k test_name

# CI 套件
python3 test/run_suite.py --hw cuda --suite base-a-test-1-gpu-small

# sgl-kernel 测试
cd sgl-kernel && pytest tests/

# Rust 测试
cd rust && cargo test --workspace
```

### CI 流水线

CI 流水线（`.github/workflows/pr-test.yml`）分三个阶段：

1. **Stage A**（预检，~3 分钟）：快速冒烟测试
2. **Stage B**（基础，~30 分钟）：核心测试
3. **Stage C**（高级，~30 分钟）：多 GPU 测试

### 代码质量

所有代码质量检查通过 pre-commit 管理：

```bash
pre-commit run --all-files  # 运行所有检查
```

工具链：black（格式化）、ruff（lint F401/F821/UP037）、isort（导入排序）、codespell（拼写检查）、clang-format（C++/CUDA）、rustfmt + clippy（Rust）。

---

# 附录：关键设计模式总结

## A.1 ScheduleBatch vs ForwardBatch

这是 SGLang 最核心的设计模式之一：

- **ScheduleBatch**：CPU 侧数据结构，由调度器管理，包含请求列表、内存池引用、调度状态
- **ForwardBatch**：GPU 侧张量表示，由 ModelRunner 管理，包含实际的 GPU 张量

数据流：`ScheduleBatch` → `ForwardBatch` → `model.forward()` → `LogitsProcessorOutput`

**规则：** 永远不要就地修改 `ScheduleBatch` 字段；构建新值并重新绑定。

## A.2 RadixCache 前缀缓存

基于基数树的前缀缓存机制：
- 自动检测和复用相同前缀的 KV 缓存
- 跨请求共享前缀，减少重复计算
- 支持驱逐策略和引用计数
- 支持分层缓存（GPU + 主机内存）

## A.3 重叠调度

CPU 处理与 GPU 计算重叠执行：
- `event_loop_overlap`：上一批次的 CPU 结果处理与当前批次的 GPU 计算并行
- 通过 `result_queue` 双端队列协调
- 最大化 GPU 利用率

## A.4 模块化注意力后端

通过抽象基类和注册表模式实现可插拔的注意力后端：
- `BaseAttentionBackend` 定义接口
- 每个硬件/算法有独立的后端实现
- 运行时根据配置自动选择后端

## A.5 环境变量管理

所有 SGLang 环境变量在 `python/sglang/srt/environ.py` 的 `Envs` 类中定义：
- 使用 `EnvBool`、`EnvInt`、`EnvFloat`、`EnvStr`、`EnvTuple` 类型描述符
- 通过 `envs.SGLANG_FOO.get()` 访问（不能直接使用 `os.environ`）
- 通过 `envs.SGLANG_FOO.override(value)` 在测试中覆盖
- `SGL_*` 是已弃用的旧别名，自动转换为 `SGLANG_*`
