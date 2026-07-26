# SGLang Managers 模块详解

> 本文档详细分析 `python/sglang/srt/managers/` 目录下的所有文件，涵盖架构设计、核心类、关键方法和数据流。

---

## 目录

- [第一部分：总体架构](#第一部分总体架构)
  - [1.1 模块定位](#11-模块定位)
  - [1.2 架构图](#12-架构图)
  - [1.3 文件总览](#13-文件总览)
- [第二部分：三大核心进程](#第二部分三大核心进程)
  - [2.1 TokenizerManager](#21-tokenizermanager)
  - [2.2 Scheduler](#22-scheduler)
  - [2.3 DetokenizerManager](#23-detokenizermanager)
- [第三部分：核心数据结构](#第三部分核心数据结构)
  - [3.1 io_struct.py — IPC 消息定义](#31-io_structpy--ipc-消息定义)
  - [3.2 schedule_batch.py — 调度批次与请求](#32-schedule_batchpy--调度批次与请求)
  - [3.3 schedule_policy.py — 调度策略](#33-schedule_policypy--调度策略)
- [第四部分：模型执行](#第四部分模型执行)
  - [4.1 tp_worker.py — 张量并行工作者](#41-tp_workerpy--张量并行工作者)
  - [4.2 data_parallel_controller.py — 数据并行控制器](#42-data_parallel_controllerpy--数据并行控制器)
  - [4.3 communicator.py — 扇出通信器](#43-communicatorpy--扇出通信器)
- [第五部分：调度器子组件（scheduler_components/）](#第五部分调度器子组件scheduler_components)
- [第六部分：混入类（Mixin）](#第六部分混入类mixin)
- [第七部分：辅助模块](#第七部分辅助模块)

---

# 第一部分：总体架构

## 1.1 模块定位

`managers/` 是 SGLang Runtime 的**核心调度层**，负责将用户的推理请求转化为 GPU 上的模型前向传播，并将结果返回给用户。它实现了 SGLang 的三进程架构：

- **TokenizerManager**（主进程）：请求入口，负责分词和响应
- **Scheduler**（子进程）：核心调度，负责批次构建和前向执行
- **DetokenizerManager**（子进程）：结果出口，负责反分词

三个进程通过 **ZMQ IPC** 通信，数据结构由 `io_struct.py` 定义。

## 1.2 架构图

### 1.2.1 进程间通信架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           主进程 (Main Process)                             │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        TokenizerManager                              │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │  │
│  │  │ generate_request │  │  handle_loop    │  │ TokenizerControlMixin│  │  │
│  │  │  (入口方法)       │  │  (接收循环)      │  │  (控制平面操作)      │  │  │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────────────────┘  │  │
│  │           │                    │                                      │  │
│  │           ▼                    ▲                                      │  │
│  │  ┌─────────────────┐  ┌───────┴────────┐                             │  │
│  │  │ _tokenize_one   │  │ _handle_batch  │                             │  │
│  │  │ _request()      │  │ _output()      │                             │  │
│  │  └────────┬────────┘  └───────▲────────┘                             │  │
│  │           │                    │                                      │  │
│  └───────────┼────────────────────┼──────────────────────────────────────┘  │
│              │                    │                                         │
│         ZMQ PUSH            ZMQ PULL                                        │
│    (send_to_scheduler)  (recv_from_detokenizer)                              │
│              │                    │                                         │
└──────────────┼────────────────────┼─────────────────────────────────────────┘
               │                    │
               ▼                    │
┌──────────────────────────────────────────────────────────────────────────────┐
│                        子进程: Scheduler                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           Scheduler                                   │  │
│  │                                                                       │  │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │  │
│  │  │ RequestReceiver   │    │ event_loop_normal│    │ OutputStreamer  │  │  │
│  │  │ (请求接收)        │───▶│ (主事件循环)      │───▶│ (结果输出)      │  │  │
│  │  └──────────────────┘    └────────┬─────────┘    └────────┬────────┘  │  │
│  │                                   │                       │           │  │
│  │                                   ▼                       │           │  │
│  │                          ┌────────────────┐               │           │  │
│  │                          │ get_next_batch │               │           │  │
│  │                          │ _to_run()      │               │           │  │
│  │                          └────────┬───────┘               │           │  │
│  │                                   │                       │           │  │
│  │                                   ▼                       │           │  │
│  │                          ┌────────────────┐               │           │  │
│  │                          │   run_batch()  │               │           │  │
│  │                          └────────┬───────┘               │           │  │
│  │                                   │                       │           │  │
│  │                                   ▼                       │           │  │
│  │                          ┌────────────────┐               │           │  │
│  │                          │TpModelWorker   │               │           │  │
│  │                          │.forward_batch  │               │           │  │
│  │                          │_generation()   │               │           │  │
│  │                          └────────────────┘               │           │  │
│  │                                                           │           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │           │  │
│  │  │              scheduler_components/                    │ │           │  │
│  │  │  BatchResultProcessor  MetricsReporter  WeightUpdater │ │           │  │
│  │  │  LogprobProcessor      InvariantChecker ProfilerMgr   │ │           │  │
│  │  │  PoolStatsObserver     KvEventsPublisher LoadInquirer │ │           │  │
│  │  │  IdleSleeper           RecvSkipper       FlushWrapper │ │           │  │
│  │  └──────────────────────────────────────────────────────┘ │           │  │
│  └───────────────────────────────────────────────────────────┼───────────┘  │
│                                                              │              │
│                                                         ZMQ PUSH            │
│                                                    (to_detokenizer)          │
└──────────────────────────────────────────────────────────────┼───────────────┘
                                                               │
                                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     子进程: DetokenizerManager                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                       DetokenizerManager                              │  │
│  │                                                                       │  │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │  │
│  │  │  event_loop()    │───▶│ _decode_batch    │───▶│ handle_batch    │  │  │
│  │  │  (主循环)         │    │ _token_id_output │    │ _token_id_out() │  │  │
│  │  └──────────────────┘    └──────────────────┘    └────────┬────────┘  │  │
│  │                                                           │           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │           │  │
│  │  │  DecodeStatus (增量反分词状态)                         │ │           │  │
│  │  │  LimitedCapacityDict (LRU 缓存, 65536 条目)           │ │           │  │
│  │  └──────────────────────────────────────────────────────┘ │           │  │
│  └───────────────────────────────────────────────────────────┼───────────┘  │
│                                                              │              │
│                                                         ZMQ PUSH            │
│                                                    (to_tokenizer)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2.2 请求数据流

```
用户请求 (HTTP)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GenerateReqInput (API 层请求)                                           │
│  fields: text, input_ids, image_data, sampling_params, stream, ...      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ TokenizerManager.generate_request()
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TokenizedGenerateReqInput (分词后 IPC 消息)                             │
│  fields: input_ids, mm_inputs, sampling_params, return_logprob, ...     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ ZMQ PUSH → Scheduler
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Req (调度器内部请求对象)                                                │
│  fields: rid, origin_input_ids, sampling_params, prefix_indices,        │
│          token_to_kv_pool, finish_reason, output_ids, ...               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ ScheduleBatch.prepare_for_extend/decode()
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ScheduleBatch (CPU 侧批次)                                             │
│  fields: reqs, input_ids, req_pool_indices, seq_lens, out_cache_loc,    │
│          sampling_info, spec_info, ...                                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ TpModelWorker.forward_batch_generation()
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ForwardBatch (GPU 侧张量) → model.forward() → Logits                  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ Sampler → GenerationBatchResult
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BatchTokenIDOutput (token ID 输出)                                      │
│  fields: rids, output_ids, finished_reasons, prompt_tokens, ...         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ ZMQ PUSH → DetokenizerManager
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BatchStrOutput (文本输出)                                               │
│  fields: rids, output_strs, output_ids, token_counts, ...               │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ ZMQ PUSH → TokenizerManager
                             ▼
                          HTTP 响应
```

### 1.2.3 Scheduler 内部事件循环

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     event_loop_normal (line 1523)                        │
│                                                                         │
│  while True:                                                            │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ 1. recv_requests()  ← RequestReceiver 从 ZMQ 接收请求           │  │
│    └────────────────────────────┬────────────────────────────────────┘  │
│                                 ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ 2. process_input_requests()                                     │  │
│    │    - handle_generate_request() → 创建 Req 对象                   │  │
│    │    - handle_embedding_request() → 创建嵌入请求                   │  │
│    │    - 加入 waiting_queue                                         │  │
│    └────────────────────────────┬────────────────────────────────────┘  │
│                                 ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ 3. get_next_batch_to_run()                                      │  │
│    │    - process_pending_chunked_abort()                            │  │
│    │    - _abort_on_waiting_timeout()                                │  │
│    │    - get_new_batch_prefill() → PrefillAdder 选择请求             │  │
│    │    - update_running_batch() → 检查解码内存、合并批次              │  │
│    │    → 返回 NextBatchPlan                                         │  │
│    └────────────────────────────┬────────────────────────────────────┘  │
│                                 ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ 4. run_batch(batch)                                             │  │
│    │    - model_worker.forward_batch_generation(batch)                │  │
│    │    → 返回 GenerationBatchResult                                  │  │
│    └────────────────────────────┬────────────────────────────────────┘  │
│                                 ▼                                       │
│    ┌─────────────────────────────────────────────────────────────────┐  │
│    │ 5. process_batch_result(batch, result)                          │  │
│    │    - BatchResultProcessor 处理输出                               │  │
│    │    - OutputStreamer 发送到 Detokenizer                           │  │
│    │    - MetricsReporter 记录指标                                    │  │
│    └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.3 文件总览

`managers/` 目录包含 **45 个 Python 文件**，分为以下几类：

### 核心进程文件（3 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `tokenizer_manager.py` | 142KB, 3301 行 | 主进程：请求入口、分词、响应 |
| `scheduler.py` | 203KB, 4684 行 | 子进程：核心调度、批次管理、前向执行 |
| `detokenizer_manager.py` | 22KB, 534 行 | 子进程：反分词、文本输出 |

### 核心数据结构（2 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `io_struct.py` | 90KB, 2282 行 | IPC 消息定义（100+ 个类） |
| `schedule_batch.py` | 132KB, 3163 行 | `Req` 和 `ScheduleBatch` 核心数据结构 |

### 调度策略（1 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `schedule_policy.py` | 50KB, 1272 行 | 调度策略和 `PrefillAdder` |

### 模型执行（2 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `tp_worker.py` | 26KB, 656 行 | 张量并行模型工作者 |
| `data_parallel_controller.py` | 36KB, 858 行 | 数据并行控制器 |

### 通信（1 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `communicator.py` | 5KB, 110 行 | `FanOutCommunicator` 扇出通信 |

### 混入类（3 个）
| 文件 | 大小 | 职责 |
|------|------|------|
| `scheduler_pp_mixin.py` | 60KB, 1500+ 行 | 流水线并行事件循环 |
| `tokenizer_control_mixin.py` | 38KB | 控制平面操作（权重更新、LoRA、会话等） |
| `tokenizer_manager_score_mixin.py` | 30KB | 评分/重排序功能 |

### 调度器子组件（19 个）
`scheduler_components/` 目录下的模块化子组件。

### 辅助模块（14 个）
多模态处理、缓存控制、解耦服务等。

---

# 第二部分：三大核心进程

## 2.1 TokenizerManager

**文件：** `tokenizer_manager.py`（142KB, 3301 行）

### 2.1.1 类定义

```python
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
```

- **TokenizerControlMixin**（`tokenizer_control_mixin.py`）：控制平面操作 — 权重更新、缓存管理、LoRA、性能分析、会话管理
- **TokenizerManagerScoreMixin**（`tokenizer_manager_score_mixin.py`）：评分/重排序功能

### 2.1.2 初始化流程（`__init__`，line 278）

```
__init__(server_args, port_args)
  ├── init_model_config()           # line 331: 模型配置、上下文长度、图像 token ID
  ├── init_tokenizer_and_processor() # line 359: HF 分词器、多模态处理器
  ├── init_ipc_channels()           # line 414: ZMQ 套接字（PULL/PUSH）
  ├── init_running_status()         # line 447: rid_to_state 字典、事件循环
  ├── init_request_logging()        # line 464: 请求日志和转储
  ├── init_weight_update()          # line 491: 读写锁、权重更新状态
  ├── init_lora()                   # line 508: LoRA 注册表
  ├── init_disaggregation()         # line 527: 解耦模式、引导服务器
  ├── init_metric_collector()       # line 565: 指标收集器、CPU 监控
  └── init_request_dispatcher()     # line 607: TypeBasedDispatcher
```

### 2.1.3 核心方法

#### `generate_request()`（line 630）— 主入口

```python
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
) -> AsyncIterator:
```

**流程：**
1. 调用 `auto_create_handle_loop()` 确保事件循环运行
2. 标准化批次参数
3. 调用 `_init_req_state()` 创建请求状态
4. 获取模型更新读锁
5. 验证和解析 LoRA 适配器
6. **单请求路径**：`_tokenize_one_request()` → `_send_one_request()` → `_wait_one_response()`
7. **批请求路径**：`_handle_batch_request()`

#### `_tokenize_one_request()`（line 834）— 分词

**流程：**
1. 确定输入源：`input_embeds`、`input_ids` 或文本
2. 如果是文本：调用 `_tokenize_texts()` 获取 `input_ids`
3. 如果有多模态数据：调用 `mm_processor.process_mm_data_async()`
4. 验证请求：`_validate_one_request()`
5. 创建分词对象：`_create_tokenized_object()`

#### `_send_one_request()`（line 1373）— 发送到 Scheduler

通过 ZMQ PUSH 套接字将 `TokenizedGenerateReqInput` 发送给 Scheduler。

#### `_wait_one_response()`（line 1488）— 等待响应

```python
async def _wait_one_response(self, req_obj, rid, state):
    # 等待 state.event
    # 遍历 state.out_list
    # 对于流式输出：合并增量块
    # 对于非流式：解析完整文本
    # yield 输出字典
```

#### `handle_loop()`（line 1890）— 接收循环

```python
async def handle_loop(self):
    while True:
        recv_obj = await async_sock_recv(self.recv_from_detokenizer)
        if isinstance(recv_obj, (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput)):
            await self._handle_batch_output(recv_obj)
        else:
            self._result_dispatcher(recv_obj)
```

#### `_handle_batch_output()`（line 1905）— 处理批次输出

这是最大的方法（~290 行），处理来自 Detokenizer 的批次响应：
1. 遍历 `recv_obj.rids`
2. 查找每个请求的 `ReqState`
3. 构建 `meta_info` 字典
4. 对于 `BatchStrOutput`：处理增量流式输出
5. 对于 `BatchEmbeddingOutput`：提取嵌入
6. 请求完成时：从 `rid_to_state` 删除，释放 LoRA
7. 通过 `state.event.set()` 通知等待的协程

### 2.1.4 关键辅助方法

| 方法 | 行号 | 职责 |
|------|------|------|
| `_tokenize_texts()` | 747 | 文本分词（支持单字符串、批次、交叉编码器对） |
| `_create_tokenized_object()` | 1153 | 构建 `TokenizedGenerateReqInput` |
| `_batch_tokenize_and_process()` | 1283 | 批量分词 |
| `_validate_one_request()` | 989 | 验证输入长度 |
| `_validate_and_resolve_lora()` | 2914 | 验证和解析 LoRA 适配器 |
| `_calculate_spec_decoding_metrics()` | 2393 | 计算投机解码接受率 |
| `abort_request()` | 1717 | 中止请求 |
| `update_weights_from_disk()` | 1757 | 从磁盘更新权重 |
| `scale_elastic_ep()` | 2864 | 弹性 EP 扩缩容 |

### 2.1.5 关键数据结构

```python
class ReqState:
    """每个请求的状态跟踪"""
    out_list: list              # 输出列表
    finished: bool              # 是否完成
    event: asyncio.Event        # 通知事件
    # ... 计时、日志概率、隐藏状态等

class InputFormat(Enum):
    SINGLE_STRING = 0
    BATCH_STRINGS = 1
    CROSS_ENCODER_PAIRS = 2
```

---

## 2.2 Scheduler

**文件：** `scheduler.py`（203KB, 4684 行）— 整个项目最大的文件

### 2.2.1 类定义

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,   # 解耦解码
    SchedulerDisaggregationPrefillMixin,  # 解耦预填充
    SchedulerMultiplexMixin,              # 多路复用
    SchedulerPPMixin,                     # 流水线并行
    SchedulerDllmMixin,                   # 离散 LLM
    SchedulerMlxOverlapMixin,             # MLX 重叠（Apple Silicon）
):
```

### 2.2.2 初始化流程（`__init__`，line 312）

```
__init__(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, ...)
  │
  ├── 1. 看门狗初始化 (line 328)
  ├── 2. 参数解析 (line 331): 调度策略、优先级、LoRA、重叠模式、投机算法
  ├── 3. 分布式 rank 信息 (line 369): attn_tp_rank/size, attn_dp_rank/size
  ├── 4. 模型配置 (line 401)
  ├── 5. 指标收集器 (line 404)
  ├── 6. IPC 通道 (line 407): ZMQ 套接字
  ├── 7. 分词器 (line 418)
  ├── 8. MoE/GEMM 配置 (line 421)
  ├── 9. 模型工作者 (line 431): TpModelWorker
  ├── 10. KV 缓存构建 (line 437): req_to_token_pool, token_to_kv_pool_allocator, tree_cache
  ├── 11. 运行状态 (line 502)
  ├── 12. 分块预填充 (line 505)
  ├── 13. 指标报告器 (line 510)
  ├── 14. 调度策略 (line 513)
  ├── 15. 解耦服务 (line 522)
  ├── 16. 重叠调度 (line 525)
  ├── 17. 权重更新器 (line 533)
  ├── 18. LoRA 排空器 (line 539)
  ├── 19. 语法管理器 (line 545)
  ├── 20. 请求接收器 (line 549)
  ├── 21. DP 注意力适配器 (line 551)
  ├── 22. 池统计观察器 (line 553)
  ├── 23. 不变量检查器 (line 555)
  ├── 24. KV 事件发布器 (line 557)
  ├── 25. 负载查询器 (line 559)
  ├── 26. 输出流式器 (line 561)
  └── 27. 批次结果处理器 (line 563)
```

### 2.2.3 主事件循环

#### `event_loop_normal()`（line 1523）— 标准模式

```python
def event_loop_normal(self):
    while True:
        # 1. 接收请求
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2. 获取下一个批次
        batch = self.get_next_batch_to_run(
            running_batch=self.running_batch,
            last_batch=self.last_batch
        )

        # 3. 执行批次
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()

        self.last_batch = batch
```

#### `event_loop_overlap()`（line 1557）— 重叠模式

将上一批次的 CPU 结果处理与当前批次的 GPU 计算重叠执行：

```python
def event_loop_overlap(self):
    result_queue = deque()
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run(...)

        if batch:
            result = self.run_batch(batch)
            result_queue.append((batch.copy(), result))

        if last_batch and not is_disable_overlap:
            last_batch_copy, last_result = result_queue.popleft()
            self.process_batch_result(last_batch_copy, last_result)

        self.last_batch = batch
```

### 2.2.4 核心方法详解

#### `handle_generate_request()`（line 2090）— 处理生成请求

```python
def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    # 1. 根据 session_id 路由
    # 2. 创建 Req 对象（~60 个参数）
    # 3. 设置 req.tokenizer
    # 4. 对于解耦模式：验证 bootstrap_room
    # 5. 对于已有会话：从会话状态创建请求
```

#### `get_next_batch_to_run()`（line 2705）— 获取下一个批次

```python
def get_next_batch_to_run(self, running_batch, last_batch) -> NextBatchPlan:
    # 1. 处理待处理的分块中止
    # 2. 检查等待/运行超时
    # 3. 处理 DLLM 暂存队列
    # 4. 分块请求排除
    # 5. HiSparse 路径或标准路径
    # 6. get_new_batch_prefill() → PrefillAdder 选择请求
    # 7. update_running_batch() → 检查解码内存
    # → 返回 NextBatchPlan
```

#### `run_batch()`（line 3308）— 执行批次

```python
def run_batch(self, batch: ScheduleBatch, pp_proxy_tensors=None):
    # 1. 增加 forward_ct
    # 2. 脚本化调度器钩子
    # 3. 性能分析器
    # 4. 预构建批次处理
    # 5. 重叠模式：进入 forward_stream_ctx，调用 forward_batch_generation
    # 6. 非重叠模式：直接调用 forward_batch_generation
```

#### `process_batch_result()`（line 3584）— 处理批次结果

```python
def process_batch_result(self, batch, result):
    # 根据前向模式分发：
    # - DECODE → batch_result_processor.process_batch_result_decode()
    # - EXTEND → batch_result_processor.process_batch_result_prefill()
    # - PREBUILT → batch_result_processor.process_batch_result_prebuilt()
    # - IDLE → batch_result_processor.process_batch_result_idle()
    # 记录指标、清理多模态输入
```

### 2.2.5 公共方法索引

| 行号 | 方法 | 职责 |
|------|------|------|
| 567 | `init_zbal_on_npu` | NPU 负载均衡初始化 |
| 577 | `init_model_config` | 模型配置初始化 |
| 605 | `init_ipc_channels` | IPC 通道初始化 |
| 669 | `init_tokenizer` | 分词器初始化 |
| 850 | `init_model_worker` | 模型工作者初始化 |
| 967 | `init_running_status` | 运行状态初始化 |
| 989 | `init_chunked_prefill` | 分块预填充初始化 |
| 1041 | `init_schedule_policy` | 调度策略初始化 |
| 1114 | `init_disaggregation` | 解耦服务初始化 |
| 1253 | `init_overlap` | 重叠调度初始化 |
| 1472 | `run_event_loop` | 启动事件循环 |
| 1523 | `event_loop_normal` | 标准事件循环 |
| 1557 | `event_loop_overlap` | 重叠事件循环 |
| 1681 | `process_input_requests` | 处理输入请求 |
| 2090 | `handle_generate_request` | 处理生成请求 |
| 2373 | `handle_batch_generate_request` | 处理批量生成请求 |
| 2537 | `handle_embedding_request` | 处理嵌入请求 |
| 2705 | `get_next_batch_to_run` | 获取下一个批次 |
| 2847 | `get_new_batch_prefill` | 获取新预填充批次 |
| 3158 | `update_running_batch` | 更新运行批次 |
| 3308 | `run_batch` | 执行批次 |
| 3554 | `launch_batch_sample_if_needed` | 启动批次采样 |
| 3584 | `process_batch_result` | 处理批次结果 |
| 3666 | `on_idle` | 空闲处理 |
| 3868 | `flush_cache` | 刷新缓存 |
| 4054 | `abort_request` | 中止请求 |
| 4189 | `pause_generation` | 暂停生成 |
| 4278 | `continue_generation` | 继续生成 |
| 4378 | `load_lora_adapter` | 加载 LoRA 适配器 |
| 4439 | `open_session` | 打开会话 |
| 4445 | `close_session` | 关闭会话 |

**模块级函数：**

| 行号 | 函数 | 职责 |
|------|------|------|
| 4502 | `dispatch_event_loop` | 分发事件循环类型 |
| 4533 | `configure_scheduler_process` | 配置调度器进程 |
| 4598 | `run_scheduler_process` | 运行调度器进程入口 |

---

## 2.3 DetokenizerManager

**文件：** `detokenizer_manager.py`（22KB, 534 行）

### 2.3.1 类定义

```python
class DetokenizerManager(MultiHttpWorkerDetokenizerMixin):
```

### 2.3.2 初始化流程（`__init__`，line 94）

```
__init__(server_args, port_args)
  ├── init_ipc_channels()       # line 111: ZMQ PULL from scheduler, PUSH to tokenizer
  ├── init_tokenizer()          # line 124: 加载 HF 分词器
  ├── init_running_status()     # line 141: LimitedCapacityDict (65536 条目)
  └── init_request_dispatcher() # line 156: TypeBasedDispatcher
```

### 2.3.3 核心方法

#### `event_loop()`（line 166）— 主循环

```python
def event_loop(self):
    while True:
        recv_obj = sock_recv(self.recv_from_scheduler)
        self._request_dispatcher(recv_obj)  # 按类型分发
        # 发送结果到 TokenizerManager
```

#### `_decode_batch_token_id_output()`（line 290）— 核心反分词逻辑

```python
def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
    # 1. 遍历 recv_obj.rids
    # 2. 对于每个请求：
    #    - 获取或创建 DecodeStatus
    #    - 计算代理/读取 ID
    #    - 执行批量解码
    #    - 处理流式 vs 完成请求
    # 3. 构建 BatchStrOutput
```

#### `_grouped_batch_decode()`（line 226）— 高效批量解码

按 `(skip_special_tokens, spaces_between_special_tokens)` 分组，使用分词器的批量解码功能。

### 2.3.4 关键数据结构

```python
class DecodeStatus:
    """每个请求的增量解码状态"""
    decoded_text: str          # 已解码文本
    decode_ids: List[int]      # 解码 ID
    surr_offset: int           # 代理偏移
    read_offset: int           # 读取偏移
    sent_offset: int           # 已发送偏移
    decoded_text_chunks: list  # 解码文本块

class LimitedCapacityDict(OrderedDict):
    """LRU 缓存（默认 65536 条目）"""
```

---

# 第三部分：核心数据结构

## 3.1 io_struct.py — IPC 消息定义

**文件：** `io_struct.py`（90KB, 2282 行）

定义了进程间传输的所有数据结构，使用 `msgspec.Struct` 实现高效序列化。

### 3.1.1 基类

```python
class BaseReq(msgspec.Struct):
    """单请求 IPC 载荷基类"""
    rid: str = ""
    http_worker_ipc: str = ""

class BaseBatchReq(msgspec.Struct):
    """批量 IPC 载荷基类"""
    rids: List[str] = []
    http_worker_ipcs: List[str] = []

class PickleWrapper(msgspec.Struct):
    """将任意 Python 对象包装为 pickle 序列化字节"""
    data: bytes = b""
```

### 3.1.2 请求结构

| 类 | 行号 | 职责 |
|----|------|------|
| `GenerateReqInput` | 155 | API 层生成请求（text、input_ids、image_data、sampling_params、stream、session_params、lora_path、priority 等） |
| `TokenizedGenerateReqInput` | 788 | 分词后 IPC 消息（input_ids、mm_inputs、sampling_params、return_logprob、stream、lora_id 等） |
| `BatchTokenizedGenerateReqInput` | 895 | 批量版本 |
| `EmbeddingReqInput` | 911 | 嵌入请求 |
| `TokenizedEmbeddingReqInput` | 1143 | 分词后嵌入请求 |

### 3.1.3 输出结构

| 类 | 行号 | 职责 |
|----|------|------|
| `BatchTokenIDOutput` | 1209 | 调度器输出：rids、output_ids、finished_reasons、token_counts、logprobs、hidden_states、routed_experts |
| `BatchStrOutput` | 1300 | 反分词后输出：rids、output_strs、output_ids、token_counts、logprobs |
| `BatchEmbeddingOutput` | 1382 | 嵌入输出：embeddings、pooled_hidden_states |

### 3.1.4 控制结构

| 类 | 行号 | 职责 |
|----|------|------|
| `FlushCacheReqInput/Output` | 1418/1422 | 缓存刷新 |
| `AbortReq` | 1795 | 请求中止 |
| `ProfileReq` | 1862 | 性能分析 |
| `UpdateWeightFromDiskReqInput` | 1533 | 磁盘权重更新 |
| `LoadLoRAAdapterReqInput` | 2002 | LoRA 加载 |
| `OpenSessionReqInput` | 1915 | 会话打开 |
| `CloseSessionReqInput` | 1922 | 会话关闭 |
| `ElasticScaleUpdateReq` | 1812 | 弹性 EP 扩缩容 |

---

## 3.2 schedule_batch.py — 调度批次与请求

**文件：** `schedule_batch.py`（132KB, 3163 行）

### 3.2.1 完成原因

```python
class FINISH_MATCHED_TOKEN:   # 匹配到停止 token
class FINISH_MATCHED_STR:     # 匹配到停止字符串
class FINISHED_MATCHED_REGEX: # 匹配到正则表达式
class FINISH_LENGTH:          # 达到最大长度
class FINISH_ABORT:           # 请求被中止
```

### 3.2.2 多模态数据

```python
class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

class MultimodalDataItem:
    """单个多模态数据项"""
    modality: Modality
    hash: str
    data: Any
    pad_value: int

class MultimodalInputs:
    """运行时多模态输入"""
    pixel_values: torch.Tensor
    hashes: List[str]
```

### 3.2.3 Req 类（line 713）

```python
class Req(ReqDllmMixin):
    """单个请求的完整生命周期状态"""

    # 核心字段
    rid: str                          # 请求 ID
    origin_input_ids: List[int]       # 原始输入 token ID
    sampling_params: SamplingParams   # 采样参数
    return_logprob: bool              # 是否返回对数概率
    stream: bool                      # 是否流式输出

    # 内存池状态
    req_pool_idx: int                 # 请求池索引
    token_indices: torch.Tensor       # token 索引
    prefix_indices: torch.Tensor      # 前缀索引
    last_node: TreeNode               # 基数树节点
    num_matched_prefix_tokens: int    # 匹配的前缀 token 数

    # 输出状态
    output_ids: List[int]             # 输出 token ID
    finished_reason: BaseFinishReason # 完成原因
    origin_output_ids: List[int]      # 原始输出 ID

    # 日志概率
    logprob_return_val: Optional[Dict] # 日志概率返回值

    # 关键方法
    def seqlen(self) -> int:           # 总序列长度
    def finished(self) -> bool:        # 是否完成
    def set_extend_range(self, ...):   # 设置预填充范围
    def get_fill_ids(self) -> List:    # 获取填充 ID
    def init_next_round_input(self):   # 准备下一轮输入
    def update_finish_state(self):     # 检查停止条件
    def reset_for_retract(self):       # 重置以回退
```

### 3.2.4 ScheduleBatch 类（line 1822）

```python
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """调度器管理的批次"""

    # 请求列表
    reqs: List[Req]

    # 共享内存资源
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    tree_cache: RadixCache

    # GPU 张量
    input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    sampling_info: SamplingBatchInfo
    spec_info: Optional[SpecInfo]

    # 批次状态
    batch_is_full: bool
    chunked_prefill_budget: int

    # 关键方法
    @classmethod
    def init_new(cls, reqs, ...):           # 创建新批次
    def prepare_for_extend(self):           # 准备预填充（ForwardMode.EXTEND）
    def prepare_for_decode(self):           # 准备解码（ForwardMode.DECODE）
    def check_decode_mem(self) -> bool:     # 检查解码内存
    def retract_decode(self):               # 回退解码请求
    def filter_batch(self, ...):            # 过滤批次
    def merge_batch(self, other):           # 合并批次
    def release_req(self, req):             # 释放请求资源
```

### 3.2.5 数据流转换

```
ScheduleBatch (CPU, 调度器管理)
    │
    │ prepare_for_extend() 或 prepare_for_decode()
    ▼
ForwardBatch (GPU 张量, ModelRunner 管理)
    │
    │ model.forward()
    ▼
LogitsProcessorOutput → Sampler → GenerationBatchResult
```

---

## 3.3 schedule_policy.py — 调度策略

**文件：** `schedule_policy.py`（50KB, 1272 行）

### 3.3.1 调度策略枚举

```python
class CacheAwarePolicy(Enum):
    LPM = "lpm"           # 最长前缀匹配
    DFS_WEIGHT = "dfs_weight"  # 深度优先搜索权重

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"         # 先来先服务
    LOF = "lof"           # 最长输出优先
    RANDOM = "random"     # 随机
    ROUTING_KEY = "routing_key"  # 路由键
```

### 3.3.2 SchedulePolicy 类（line 163）

```python
class SchedulePolicy:
    """调度策略管理"""

    def __init__(self, policy, tree_cache, ...):
        # 初始化策略、树缓存、优先级调度配置

    def calc_priority(self, waiting_queue):
        """主入口：按策略排序等待队列"""
        # 1. 确定活跃策略（LPM 在队列 > 128 时回退到 FCFS）
        # 2. 计算前缀匹配
        # 3. 按策略排序

    def _sort_by_longest_prefix(waiting_queue):
        """按最长前缀排序"""
    def _sort_by_dfs_weight(waiting_queue):
        """按 DFS 树权重排序"""
    def _sort_by_longest_output(waiting_queue):
        """按最长输出排序"""
    def _sort_randomly(waiting_queue):
        """随机排序"""
    def _sort_by_routing_key(waiting_queue, running_bs):
        """按路由键频率排序"""
```

### 3.3.3 PrefillAdder 类（line 441）

```python
class PrefillAdder:
    """管理预填充批次的 token 预算和准入逻辑"""

    def __init__(self, page_size, tree_cache, allocator, running_batch,
                 new_token_ratio, remaining_token_budgets, ...):
        # 初始化预算、分配器、缓存

    def rem_total_tokens(self) -> int:
        """剩余总 token 数（可用 + 可驱逐 - 偏移）"""

    def cur_rem_tokens(self) -> int:
        """当前剩余 token 数"""

    def budget_state(self) -> AddReqResult:
        """返回预算状态：CONTINUE / NO_TOKEN / OTHER"""

    def add_one_req(self, req) -> AddReqResult:
        """主准入方法：检查 token 预算、锁定树节点、处理分块预填充"""

    def preempt_to_schedule(self, req):
        """抢占低优先级运行请求以腾出空间"""
```

### 3.3.4 关键函数

```python
def match_prefix_for_req(req, tree_cache):
    """匹配请求的 token ID 与树缓存"""
    # 设置 req.prefix_indices, req.last_node, req.num_matched_prefix_tokens
```

---

# 第四部分：模型执行

## 4.1 tp_worker.py — 张量并行工作者

**文件：** `tp_worker.py`（26KB, 656 行）

### 4.1.1 类层次

```python
class BaseTpWorker(ABC):
    """抽象基类"""
    def forward_batch_generation(self, batch): ...
    @property
    def model_runner(self): ...

class TpModelWorker(BaseTpWorker):
    """具体实现"""
```

### 4.1.2 TpModelWorker 初始化（line 276）

```python
def __init__(self, server_args, gpu_id, ps, nccl_port, ...):
    # 1. 模型配置
    # 2. ModelRunner 创建
    # 3. 分词器/处理器
    # 4. NCCL 通信组
    # 5. 随机种子同步
    # 6. 重叠/投机标志
```

### 4.1.3 核心方法

#### `forward_batch_generation()`（line 529）— 主前向方法

```python
def forward_batch_generation(self, batch, ...):
    # 1. 从 ScheduleBatch 创建 ForwardBatch
    # 2. 调用 model_runner.forward(forward_batch)
    # 3. 采样下一个 token
    # 4. 处理 DLLM、重叠调度、仅预填充模式
    # → 返回 GenerationBatchResult
```

#### `get_worker_info()`（line 481）— 获取工作者信息

返回：`(max_total_num_tokens, max_prefill_tokens, max_running_requests, max_queued_requests, max_req_len, random_seed, device, forward_stream, pool_sizes)`

---

## 4.2 data_parallel_controller.py — 数据并行控制器

**文件：** `data_parallel_controller.py`（36KB, 858 行）

### 4.2.1 负载均衡方法

```python
class LoadBalanceMethod(Enum):
    ROUND_ROBIN = "round_robin"              # 轮询
    FOLLOW_BOOTSTRAP_ROOM = "follow_bootstrap_room"  # 按引导房间
    TOTAL_REQUESTS = "total_requests"        # 最少请求
    TOTAL_TOKENS = "total_tokens"            # 最少 token
```

### 4.2.2 DataParallelController 类（line 130）

```python
class DataParallelController:
    """数据并行控制器"""

    def __init__(self, server_args, port_args):
        # 1. 解析参数
        # 2. 设置 ZMQ 上下文
        # 3. 选择分发方法
        # 4. 启动 DP 调度器

    def event_loop(self):
        """非阻塞接收循环"""
        while True:
            recv_obj = sock_recv(self.recv_from_tokenizer)
            self._dispatcher(recv_obj)

    def round_robin_scheduler(self, recv_obj):
        """轮询分发"""
    def total_requests_scheduler(self, recv_obj):
        """分发到最少请求的工作者"""
    def total_tokens_scheduler(self, recv_obj):
        """分发到最少 token 的工作者"""
    def follow_bootstrap_room_scheduler(self, recv_obj):
        """按 bootstrap_room % num_workers 分发"""
```

---

## 4.3 communicator.py — 扇出通信器

**文件：** `communicator.py`（5KB, 110 行）

### 4.3.1 FanOutCommunicator 类（line 13）

```python
class FanOutCommunicator[T]:
    """将单个请求扇出到 N 个接收者并收集所有响应"""

    def __init__(self, send, fan_out, mode="queueing"):
        # send: 发送函数
        # fan_out: 扇出数量
        # mode: "queueing"（FIFO）或 "watching"（共享）

    async def queueing_call(self, obj):
        """队列模式：FIFO 串行化"""
        # 获取锁 → 发送 → 等待所有响应

    async def watching_call(self, obj):
        """观察模式：多个调用者共享单个进行中的请求"""
        # 第一个调用者发送并创建事件
        # 后续调用者仅等待
        # 所有人获得结果的 deepcopy

    def handle_recv(self, value):
        """处理接收到的响应"""
        # 追加到 _result_values
        # 收集完所有响应时设置事件

    @staticmethod
    def merge_results(results):
        """合并多个结果对象"""
```

---

# 第五部分：调度器子组件（scheduler_components/）

`scheduler_components/` 目录包含 19 个模块化子组件，将 Scheduler 的职责分解为独立的、可测试的单元。

### 5.1 请求接收

#### `request_receiver.py`（line 46）

```python
class SchedulerRequestReceiver:
    """从 Tokenizer 接收分词后的请求"""
    # - 通过 ZMQ PULL 接收
    # - 跨 TP/DP/PP 广播
    # - 解包 pickle 包装字段
    # - 应用多模态接收器
    # - 完成共享内存特性
```

### 5.2 输出处理

#### `output_streamer.py`（line 39）

```python
class SchedulerOutputStreamer:
    """将批次输出流式传输到 Detokenizer"""
    # - 收集完成/流式请求输出
    # - 使用 _GenerationStreamAccumulator 累积字段
    # - 构建 BatchTokenIDOutput 载荷
    # - 通过 ZMQ PUSH 发送
```

#### `output_sender.py`（line 8）

```python
class SenderWrapper:
    """ZMQ 套接字封装，传播 http_worker_ipc"""
```

### 5.3 批次结果处理

#### `batch_result_processor.py`（line 65）

```python
class SchedulerBatchResultProcessor:
    """处理模型前向传播结果的核心枢纽"""

    def process_batch_result_prefill(self, batch, result):
        """处理预填充结果：更新输出 ID、完成状态、日志概率、语法 FSM"""

    def process_batch_result_decode(self, batch, result):
        """处理解码结果：解析投机 token、推进语法 FSM、处理完成状态"""

    def process_batch_result_prebuilt(self, batch):
        """处理预构建（解耦解码）批次"""

    def advance_grammar_fsm(self, reqs, next_token_ids):
        """推进语法 FSM"""
```

### 5.4 指标与监控

#### `metrics_reporter.py`（line 90）

```python
class SchedulerMetricsReporter:
    """收集和报告调度器指标"""
    # - 预填充/解码吞吐量日志
    # - 投机解码接受统计
    # - MFU 估计（TFLOPS/s、内存带宽）
    # - CUDA Graph 使用率
    # - LoRA 池利用率
    # - Prometheus 指标发射
```

#### `pool_stats_observer.py`（line 142）

```python
class SchedulerPoolStatsObserver:
    """观察和报告内存池统计"""
    # - 全注意力、滑动窗口、Mamba、HiSparse 池
    # - 会话持有 token
    # - token 使用率、可用大小、可驱逐大小
```

#### `load_inquirer.py`（line 33）

```python
class SchedulerLoadInquirer:
    """构建每个 DP rank 的负载快照"""
    # - 运行/等待请求数
    # - token 使用率
    # - 内存指标
    # - 投机解码统计
    # - LoRA 指标
    # - 解耦队列统计
```

#### `kv_events_publisher.py`（line 45）

```python
class SchedulerKvEventsPublisher:
    """发布 KV 缓存事件和指标"""
    # - ZMQ 发布
    # - 可配置事件发布器（如 NATS）
```

### 5.5 权重管理

#### `weight_updater.py`（line 76）

```python
class SchedulerWeightUpdaterManager:
    """管理在线模型权重更新"""
    # - 从磁盘/分布式/张量/IPC 更新
    # - GPU 内存释放/恢复
    # - 跨 rank 权重校验和验证
    # - 投机解码草稿模型权重更新
```

### 5.6 IPC 通道

#### `ipc_channels.py`（line 17）

```python
class SchedulerIpcChannels:
    """持有所有 ZMQ 套接字"""
    # - PULL from tokenizer
    # - DEALER for RPC
    # - PUSH to tokenizer
    # - PUSH to detokenizer
    # - PUSH for metrics（可选）
```

### 5.7 调度辅助

#### `idle_sleeper.py`（line 8）

```python
class IdleSleeper:
    """空闲期间降低 CPU 功耗"""
    # - 使用 zmq.Poller 阻塞直到请求到达
    # - 可选定期 empty_cache() 释放 GPU 内存
```

#### `recv_skipper.py`（line 13）

```python
class SchedulerRecvSkipper:
    """速率限制请求接收"""
    # - 使用加权计数器
    # - 解码模式下接收频率更低
    # - 预填充模式下接收更频繁
```

#### `new_token_ratio_tracker.py`（line 14）

```python
class NewTokenRatioTracker:
    """跟踪新 token 比率"""
    # - 估计请求将生成的新 token 数
    # - 防止 KV 缓存溢出
    # - 可配置初始比率、最小值、衰减
```

#### `flush_wrapper.py`（line 11）

```python
class SchedulerFlushWrapper:
    """延迟缓存刷新"""
    # - 等待空闲状态后执行
    # - 可配置超时
```

#### `scheduler_input_blocker.py`（line 25）

```python
class SchedulerInputBlocker:
    """阻塞/解除阻塞调度器输入"""
    # - 基于轮询的屏障
    # - 确保所有 DP rank 同时解除阻塞
```

### 5.8 DP 注意力

#### `dp_attn.py`（line 364）

```python
class SchedulerDPAttnAdapter:
    """协调数据并行注意力"""
    # - all-gather 每个 rank 的批次元数据
    # - 确定 CUDA Graph 资格
    # - 管理空闲批次
```

### 5.9 不变量检查

#### `invariant_checker.py`（line 44）

```python
class SchedulerInvariantChecker:
    """内存池不变量验证"""
    # - KV 缓存池不变量：available + evictable + protected + session_held == total
    # - 请求池一致性
    # - KV 页双重释放/释放后使用检查
```

### 5.10 性能分析

#### `profiler_manager.py`（line 51）

```python
class SchedulerProfilerManager:
    """管理 PyTorch 性能分析会话"""
    # - 步骤触发和阶段触发
    # - Chrome 跟踪导出
    # - ROCm RPD 跟踪
    # - CUDA 内存快照
    # - 多 rank 配置文件合并
```

### 5.11 日志概率处理

#### `logprob_result_processor.py`（line 21）

```python
class SchedulerLogprobResultProcessor:
    """处理日志概率结果"""
    # - 支持常规和多项目评分模式
    # - 增量累积（跨分块预填充）
    # - 输入/输出日志概率、top-k 日志概率
```

---

# 第六部分：混入类（Mixin）

## 6.1 SchedulerPPMixin — 流水线并行

**文件：** `scheduler_pp_mixin.py`（60KB, 1500+ 行）

```python
class SchedulerPPMixin:
    """流水线并行事件循环"""

    def event_loop_pp(self):
        """核心 PP 调度器循环，微批次交织"""
    def event_loop_pp_disagg_prefill(self):
        """解耦预填充服务器的 PP 循环"""
    def event_loop_pp_disagg_decode(self):
        """解耦解码服务器的 PP 循环"""
    def predict_next_chunk_size(self):
        """预测动态分块大小"""
```

**ChunkSizePredictor 类**（line 1445）：二次延迟模型，用于动态分块大小预测。

## 6.2 TokenizerControlMixin — 控制平面

**文件：** `tokenizer_control_mixin.py`（38KB）

```python
class TokenizerControlMixin:
    """TokenizerManager 的控制平面操作"""

    def init_communicators(self):
        """为每个规范创建 FanOutCommunicator 实例"""
    def flush_cache(self):
        """刷新调度器缓存"""
    def start_profile(self) / stop_profile():
        """性能分析控制"""
    def update_weights_from_distributed(self):
        """从分布式源更新权重"""
    def update_weights_from_tensor(self):
        """从张量更新权重"""
    def load_lora_adapter(self):
        """加载 LoRA 适配器（LRU 驱逐）"""
    def open_session(self) / close_session():
        """会话管理"""
```

## 6.3 TokenizerManagerScoreMixin — 评分

**文件：** `tokenizer_manager_score_mixin.py`（30KB）

```python
class TokenizerManagerScoreMixin:
    """评分/重排序功能"""

    def score_request(self):
        """主入口：支持单项目和多项目评分"""
    def score_prompts(self):
        """评分给定提示和标签 token ID 的概率"""
    def _build_multi_item_token_sequence(self):
        """构建带分隔符的组合 token 序列"""
    def _convert_logprobs_to_scores(self):
        """将日志概率转换为有序分数列表"""
```

---

# 第七部分：辅助模块

## 7.1 多模态处理

#### `multimodal_processor.py`

```python
PROCESSOR_MAPPING: Dict  # 模型架构名 → 处理器类
def get_mm_processor(model_config) -> BaseMultimodalProcessor:
    """工厂函数：返回适当的多模态处理器"""
```

#### `mm_utils.py`

```python
class TransportProxyTensor(torch.Tensor):
    """支持 CUDA IPC 序列化的张量子类"""
class ShmPointerMMData:
    """共享内存包装器"""
def embed_mm_inputs(...):
    """嵌入多模态输入并集成到文本嵌入中"""
def general_mm_embed_routine(...):
    """完整的多模态嵌入 + 语言模型前向例程"""
```

## 7.2 缓存控制

#### `cache_controller.py`

```python
class HiCacheController:
    """分层 KV 缓存控制器（L1 设备、L2 主机、L3 存储）"""
    def write(self):       # 从设备备份到主机
    def load(self):        # 从主机加载到设备
    def prefetch(self):    # 从存储预取到主机
    def attach_storage_backend(self):  # 运行时附加存储后端
```

#### `hisparse_coordinator.py`

```python
class HiSparseCoordinator:
    """协调分层稀疏 KV 缓存管理"""
    def admit_request_into_staging(self):  # 暂存请求
    def alloc_device_buffer(self):         # 分配设备缓冲区
    def swap_in_selected_pages(self):      # 交换页面到设备
```

## 7.3 解耦服务

#### `disagg_service.py`

```python
def start_disagg_service():
    """启动 KV 引导服务器（解耦预填充模式）"""
```

## 7.4 负载快照

#### `load_snapshot.py`

```python
class LoadSnapshot(msgspec.Struct):
    """每个 DP rank 的负载指标"""
class ShmLoadSnapshotWriter:
    """写入共享内存 mmap 文件"""
class ZmqLoadSnapshotWriter:
    """通过 ZMQ PUSH 发送"""
class ShmLoadSnapshotReader:
    """从共享内存读取"""
```

## 7.5 重叠调度

#### `overlap_utils.py`

```python
class FutureMap:
    """池索引的跨迭代值中继"""
    def publish(self):           # 发布 new_seq_lens 和 confidence
    def stash(self):             # 暂存输出 token 和投机 extras
    def resolve_seq_lens_cpu(self):  # 解析 CPU seq_lens
class ConfidenceRelay:
    """管理投机解码置信度缓冲区"""
```

## 7.6 预填充延迟

#### `prefill_delayer.py`

```python
class PrefillDelayer:
    """跨 DP rank 延迟预填充准入"""
    # - 通过 all-gather 协商
    # - 避免将预填充分散为小批次
```

#### `min_free_slots_delayer.py`

```python
class MinFreeSlotsDelayer:
    """延迟新预填充直到最小空闲槽数"""
```

## 7.7 其他

#### `embed_types.py`

```python
class PositionalEmbeds(msgspec.Struct):
    """位置嵌入注入"""
```

#### `configure_logging.py`

命令行脚本，通过 HTTP POST 配置 SGLang 服务器的日志设置。

#### `utils.py`

```python
class GenerationBatchResult:
    """前向传播结果"""
    def copy_to_cpu(self):  # 重叠调度的 CPU 副本
class EmbeddingBatchResult:
    """嵌入结果"""
```

#### `multi_tokenizer_mixin.py`

```python
class MultiTokenizerRouter:
    """多 HTTP 工作者模式的请求路由"""
class MultiDetokenizerRouter:
    """多 Detokenizer 路由"""
```

#### `async_dynamic_batch_tokenizer.py`

```python
class AsyncDynamicbatchTokenizer:
    """异步动态批量分词器"""
    # - 收集并发编码请求
    # - 批量处理以减少分词开销
```

---

# 附录：关键设计模式

## A.1 消息类型分发（TypeBasedDispatcher）

TokenizerManager 和 DetokenizerManager 都使用 `TypeBasedDispatcher` 将接收到的消息按类型路由到处理方法：

```python
self._result_dispatcher = TypeBasedDispatcher([
    (BatchStrOutput, self._handle_batch_output),
    (AbortReq, self._handle_abort_req),
    (OpenSessionReqOutput, self._handle_open_session_req_output),
    ...
])
```

## A.2 FanOutCommunicator 模式

控制平面操作（权重更新、缓存刷新等）使用 `FanOutCommunicator` 将请求扇出到所有 DP/TP 调度器并收集响应：

```python
# 队列模式：FIFO 串行化
await self.flush_cache_communicator(flush_req)

# 观察模式：多个调用者共享
await self.update_weights_communicator(weight_req)
```

## A.3 ScheduleBatch → ForwardBatch 转换

这是 SGLang 最核心的数据流模式：
- **ScheduleBatch**：CPU 侧，调度器管理，包含请求列表和内存池引用
- **ForwardBatch**：GPU 侧张量，ModelRunner 管理
- 转换由 `ScheduleBatch.prepare_for_extend()` 和 `prepare_for_decode()` 完成

## A.4 增量反分词

DetokenizerManager 使用 `DecodeStatus` 跟踪每个请求的增量解码状态，通过 `surr_offset` 和 `read_offset` 管理代理字符边界，确保流式输出的正确性。

## A.5 模块化子组件

Scheduler 将职责分解为 19 个独立的子组件（`scheduler_components/`），每个组件通过 `frozen dataclass` 实现，接收 Scheduler 的引用，保持单一职责。
