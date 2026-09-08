# GLM-5.2 FP8 PD 分离 NIXL 测试报告（H200-95 / H200-96）

## 1. 测试结论

本次测试最终通过。

- Prefill：`zly-h200-95`，Pod IP `172.20.28.215`，监听 `30000`，TP=8、NSA Prefill CP=8。
- Decode：`zly-h200-96`，Pod IP `172.20.5.107`，监听 `30001`，TP=8、DP=8、DP Attention。
- Router：运行于 `zly-h200-96`，监听 `8000`。
- 传输后端：NIXL 1.0.0，UCX/RoCE 设备 `mlx5_10`～`mlx5_17`。
- 模型：`/dev/shm/GLM-5.2-FP8`，服务模型名 `zai-org/GLM-5.2`。
- 完整 benchmark：7/7 组 `rc=0`，所有请求成功。
- 最终健康状态：P=200、D=200、Router=200。
- P/D 最终日志中未发现 Traceback、Scheduler 异常、段错误、CUDA OOM 或 RuntimeError。

完整测试时间为 2026-08-17 20:28:04～20:42:38（北京时间；服务日志使用 UTC，对应 12:28:04～12:42:38）。

## 2. 环境与版本

P/D 两端使用完全相同的镜像和代码版本：

```text
image: registry.cn-sh-02.sensecore.cn/ccr-axion/sglang:dev-ae5b7040-cu129
image digest: sha256:c47d7267e8ae0c1506afd1413e779436657c277f6c5fd39ddbc5d9ce48378ffc
git commit: 1519acf37c23f2189adb93f57ca9cd2db1bebf18
nixl-cu12: 1.0.0
```

Decode 启动后的实际关键参数：

```text
tp_size=8
dp_size=8
enable_dp_attention=True
max_total_num_tokens=369408
max_running_requests=64
attention_backend=nsa
```

## 3. 结果与日志路径

### 3.1 最终 benchmark 结果

结果位于 Decode 节点 `zly-h200-96`：

```text
/data/zhangluyu/glm52/bench_results/glm52-20260817-122804/
/data/zhangluyu/glm52/bench_results/glm52-20260817-122804/summary.txt
```

目录包含 7 组测试各自的 `.log`、`.jsonl` 和总 `summary.txt`，总大小约 368 KiB。

### 3.2 本次执行过程文件

P/D 容器均保留同名运行目录：

```text
/data/zhangluyu/glm52/test_runs/20260817-193920/
```

P 节点主要文件：

```text
start_p.sh                         最终 P 启动脚本
prefill.log                        最终 P 日志
health_p.txt                       最终 P 健康码
nixl_conn_before.py                修改前 NIXL 源码备份
prefill_attempt1.log               初次 UCX/NIXL 失败日志
prefill_before_source_fix.log      协议兼容修复前日志
prefill_before_protocol_fix.log    协议修复前日志
prefill_before_empty_cp_fix.log    空 CP 分片修复前日志
prefill_before_source_rank_fix.log 来源编号修复前日志
```

D 节点主要文件：

```text
start_d.sh                         最终 D 启动脚本
start_router.sh                    最终 Router 启动脚本
basic_bench_nixl.sh                最终 benchmark 脚本
decode.log                         最终 D 日志
router.log                         最终 Router 日志
health_d_router.txt                最终 D/Router 健康码
smoke_short_response.json          短请求响应
smoke_64k.log                      单条 64K 冒烟测试日志
smoke_64k.jsonl                    单条 64K 冒烟测试结果
benchmark_suite.log                完整 7 组测试驱动日志
benchmark.rc                       完整测试退出码，内容为 0
nixl_conn_before.py                修改前 NIXL 源码备份
```

## 4. 启动命令中的必要修正

### 4.1 Decode 与 Router 端口

给定 Decode 命令监听的是 `30001`，因此 Router 的 Decode 地址必须为：

```text
--decode http://172.20.5.107:30001
```

不能使用原命令中的 `http://172.20.5.107:30000`。Decode 本地健康检查也相应改为：

```bash
curl http://127.0.0.1:30001/health
```

### 4.2 限定 UCX 使用有效 RoCE 设备

初次启动时 NIXL/UCX 自动选择到了 `mlx5_z0`，GPU 内存注册阶段报错并段错误：

```text
ibv_create_ah ... on mlx5_z0 failed: No such device
ucp_ep_rkey_unpack
Segmentation fault
```

检查两机 GID 后确认 `mlx5_10`～`mlx5_17` 的 port 1 可用，IPv4 RoCE v2 位于 GID index 3。P/D 两端增加：

```bash
export UCX_NET_DEVICES=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1,mlx5_14:1,mlx5_15:1,mlx5_16:1,mlx5_17:1
export UCX_IB_GID_INDEX=3
```

这两个变量只约束本次 NIXL 使用的 UCX 传输。原命令未设置 `NCCL_IB_GID_INDEX`，本次没有擅自增加与 NIXL 问题无直接关系的 NCCL GID 配置。

### 4.3 benchmark 的本地模型路径与服务名分离

`bench_serving` 中 `--model` 既参与读取模型配置，也默认作为请求里的模型名。直接设置：

```text
--model zai-org/GLM-5.2
```

会尝试访问 Hugging Face；节点无外网时出现：

```text
Network is unreachable ... huggingface.co/zai-org/GLM-5.2
```

最终测试脚本使用：

```bash
--model /dev/shm/GLM-5.2-FP8 \
--served-model-name zai-org/GLM-5.2 \
--tokenizer /dev/shm/GLM-5.2-FP8
```

这样配置、tokenizer 全部从本地加载，请求中的 `model` 又与 SGLang 的 `--served-model-name` 一致。

## 5. 镜像内 NIXL 源码兼容问题与修复

P/D 镜像虽然完全一致，但镜像内以下文件的调用方与实现版本不一致：

```text
/sgl-workspace/sglang/python/sglang/srt/disaggregation/nixl/conn.py
```

`decode.py`、`prefill.py` 已按较新的 NIXL 协议传参，而镜像中的 `nixl/conn.py` 仍是旧接口。修复在 P/D 两端同时应用，修改后均通过：

```bash
python3 -m py_compile /sgl-workspace/sglang/python/sglang/srt/disaggregation/nixl/conn.py
```

### 5.1 补齐 Decode cache-hit 协议字段

问题一：D 调用 `NixlKVReceiver.init()` 时多传了 `disagg_decode_cache_hit_len`：

```text
TypeError: NixlKVReceiver.init() takes from 2 to 4 positional arguments but 5 were given
```

修复内容：

```python
@dataclasses.dataclass
class TransferInfo:
    # ...
    disagg_decode_cache_hit_len: int

def init(
    self,
    kv_indices,
    aux_index=None,
    state_indices=None,
    disagg_decode_cache_hit_len: int = 0,
):
    # ZMQ 第 9 个 frame
    str(disagg_decode_cache_hit_len).encode("ascii")
```

`TransferInfo.from_zmq()` 同步解析第 9 个 frame，旧消息缺少该字段时回退为 0。该修复消除了 P 侧随后出现的：

```text
AttributeError: 'TransferInfo' object has no attribute 'disagg_decode_cache_hit_len'
```

### 5.2 CP 页过滤使用完整请求页数

NSA Prefill CP=8 时，每个 chunk 的页面属于完整请求的全局页坐标。发送端由：

```python
filter_kv_indices_for_cp_rank(self.kv_mgr, kv_indices, index_slice)
```

调整为：

```python
filter_kv_indices_for_cp_rank(
    self.kv_mgr,
    kv_indices,
    index_slice,
    total_pages=self.num_kv_indices,
)
```

避免 chunk 边界处按当前 chunk 页数重新划分 CP 归属。`total_pages` 参数由容器中此前已存在的公共 CP 页过滤实现提供，本次没有再次修改公共文件。

### 5.3 空 CP 分片不能交给 NIXL 1.0 构造 descriptor

短 Prompt 只有 1 个 KV page，分到 CP=8 后会有 7 个 rank 的 `kv_indices` 为空。NIXL 1.0 对空数组执行 `get_xfer_descs()` 会触发：

```text
IndexError: index 0 is out of bounds for axis 0 with size 0
```

发送端修复为：

```python
if len(kv_indices) > 0:
    # 创建并提交 KV transfer
    handles.append(kv_xfer_handle)

if len(kv_indices) == 0 and not is_last:
    return

if len(kv_indices) > 0:
    self.chunk_id += 1
```

空的非末尾 chunk 不提交、不消耗 `chunk_id`，从而保证接收端收到的 chunk ID 连续。

### 5.4 空的末尾分片仍需发送完成元数据

空的最后分片虽然没有 KV 数据，但 D 必须知道该来源预期收到多少个 KV chunk。P 发送：

```python
aux_notif = f"{req.room}_aux"
if len(kv_indices) == 0:
    aux_notif += f"_nokv_{self.transfer_source_rank}_{chunk_id}"
```

D 解析 `nokv` 通知并写入完成计数：

```python
pp_rank = int(components[3])
expected = int(components[4]) if len(components) > 4 else 0
status.expected_kvs_per_pp[pp_rank] = expected
```

这使“某 CP rank 没有数据”成为一个可完成的 0-chunk 状态，而不是永久等待。

### 5.5 每个 CP/engine rank 必须有唯一来源编号

旧实现把通知来源写成 `self.kv_args.pp_rank`。本拓扑 PP=1，因此 8 个 CP rank 的 `pp_rank` 都是 0：CP0 的有效 KV 计数会被 CP1～7 的 `nokv_0_0` 覆盖，结果是 P 已完成并返回 200，但 D 永远等待，最终 `waiting_timeout`。

按当前上游 SGLang 实现增加：

```python
self.transfer_source_rank = (
    self.kv_args.pp_rank * self.server_args.tp_size
    + self.kv_args.engine_rank
)
```

并在 KV、state、aux-nokv 三类通知中统一使用 `transfer_source_rank`。本拓扑得到来源 0～7，恰好对应 D 侧 `required_prefill_response_num=8` 的完成条件。

这是短请求从“P 返回、D 超时”恢复为 Router HTTP 200 的关键修复。

## 6. 验证过程

### 6.1 服务启动与健康检查

应用 UCX 环境变量和 NIXL 兼容补丁后，P/D 均完成模型加载、KV cache 分配和预热：

```text
P health: 200
D health: 200
Router health: 200
```

### 6.2 短请求端到端验证

通过 Router 请求 `/v1/chat/completions`：

```text
HTTP code: 200
total time: 5.886925 s
prompt tokens: 15
completion tokens: 8
```

这一步验证了 Router → P Prefill → NIXL KV transfer → D Decode → Router 响应闭环。

### 6.3 单条 64K 输入验证

在运行完整压测前执行 1 条 64K 输入、1 token 输出的冒烟测试：

```text
Successful requests: 1/1
Input tokens: 64000
Benchmark duration: 3.54 s
Input token throughput: 18073.20 tok/s
Mean TTFT: 3532.53 ms
```

该测试证明大请求下 CP=8 的页面划分、8 路来源编号、空分片完成通知和 NIXL 数据传输均能闭环。

### 6.4 完整 benchmark

最终执行：

```bash
bash /data/zhangluyu/glm52/test_runs/20260817-193920/basic_bench_nixl.sh
```

套件退出码：

```text
/data/zhangluyu/glm52/test_runs/20260817-193920/benchmark.rc = 0
```

## 7. 最终性能结果

| 用例 | 成功请求 | 时长 s | 请求吞吐 req/s | 输入吞吐 tok/s | 输出吞吐 tok/s | Mean TTFT ms | Mean TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冷随机 64×64K，C=16 | 64/64 | 213.42 | 0.30 | 19,192.51 | 0.30 | 39,307.28 | 0.00 |
| 90% 共享前缀 64×64K，C=16 | 64/64 | 33.17 | 1.93 | 128,275.88 | 1.93 | 7,709.18 | 0.00 |
| 32K+1K，4 请求，C=1 | 4/4 | 42.50 | 0.09 | 3,140.15 | 94.12 | 864.18 | 9.77 |
| 32K+1K，64 请求，C=16 | 64/64 | 71.22 | 0.90 | 30,002.63 | 898.63 | 6,021.95 | 10.70 |
| 32K+1K，128 请求，C=32 | 128/128 | 70.47 | 1.82 | 60,532.35 | 1,816.42 | 1,919.96 | 14.13 |
| 32K+1K，256 请求，C=64 | 256/256 | 89.42 | 2.86 | 95,332.61 | 2,862.93 | 2,521.63 | 17.16 |
| 32K+1K，512 请求，C=128 | 512/512 | 168.44 | 3.04 | 101,303.88 | 3,039.69 | 18,117.91 | 17.53 |

共享前缀场景的输入吞吐约为冷随机场景的 6.68 倍。Decode 输出吞吐随并发从 94.12 tok/s 增长到 3,039.69 tok/s；并发 128 已超过服务端 `max-running-requests=64`，额外请求进入排队，因此 Mean TTFT 从并发 64 的 2.52 秒升至 18.12 秒，而总输出吞吐仅从 2,862.93 增长到 3,039.69 tok/s。对该配置而言，并发 64 已接近吞吐与首 token 时延的较优平衡点。

## 8. 日志告警说明

Router 在第一组 64K 冷随机高负载测试期间记录过一次 P 健康探测超时：

```text
2026-08-17 12:32:39 HTTP health check failed for http://172.20.28.215:30000/health: TimedOut
```

该事件对应两行 WARN（worker 与 manager 各一行），没有导致 worker 被永久摘除：

- 后续所有 7 组测试均完成且无失败请求；
- Router 后续请求均返回 200；
- 测试结束后 P、D、Router 健康码均为 200。

因此它属于 P 在 64K Prefill 高负载期间健康接口响应超过 Router 探测窗口的瞬时告警，不是 NIXL 传输失败。若生产环境不希望因长 Prefill 出现误摘除，应单独调整 Router worker health probe 的超时/失败阈值，而不是降低推理 watchdog。

另外，`User lacks permission to set NUMA affinity` 和 `cutlass.cute.experimental` package-walk 均为非致命 warning，本次没有影响服务启动与测试结果。

## 9. 最终核验与当前状态

```text
benchmark suite exit code: 0
benchmark cases: 7
rc=0 cases: 7
failed cases: 0
P final health: 200
D final health: 200
Router final health: 200
P final fatal matches: 0
D final fatal matches: 0
```

测试结束后服务保持运行，便于继续检查：

```text
P sglang PID: 97775
D sglang PID: 200363
Router PID: 205872
```

运行时源码修改位于 Pod 可写层，Pod 重建或替换镜像后会丢失。投入长期使用前，应把 `nixl/conn.py` 的协议、空分片和唯一来源编号修复合入正式分支并重新构建镜像；启动环境中还需要保留 UCX 设备/GID 设置及：

```bash
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
```

## 10. 最终命令入口

为避免复制长命令时再次出现端口、UCX 或 benchmark 模型名错误，本次实际成功命令已固化为脚本：

```bash
# P：zly-h200-95
bash /data/zhangluyu/glm52/test_runs/20260817-193920/start_p.sh

# D：zly-h200-96
bash /data/zhangluyu/glm52/test_runs/20260817-193920/start_d.sh

# Router：zly-h200-96
bash /data/zhangluyu/glm52/test_runs/20260817-193920/start_router.sh

# Benchmark：zly-h200-96
bash /data/zhangluyu/glm52/test_runs/20260817-193920/basic_bench_nixl.sh
```

