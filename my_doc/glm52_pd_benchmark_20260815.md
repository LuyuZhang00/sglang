# GLM-5.2 FP8 PD 分离测试报告（H200-95 / H200-96）

## 1. 测试结论

本次测试最终通过。

- Prefill 节点：`zly-h200-95`，Pod IP `172.20.28.215`，8 张 H200，TP=8，NSA Prefill CP=8。
- Decode 节点：`zly-h200-96`，Pod IP `172.20.5.107`，8 张 H200，TP=8、DP=8、EP=8、DeepEP low-latency。
- Router：运行在 Decode Pod，监听 `0.0.0.0:8000`。
- SGLang 源码版本：`1519acf`（两端一致）。
- 模型：`/dev/shm/GLM-5.2-FP8`。
- 最终 benchmark：7/7 个用例 `rc=0`，无失败请求。
- 最终健康状态：P=200、D=200、Router=200。
- 修复版日志中没有 CP/Mooncake 索引异常、transfer thread 崩溃、CUDA OOM 或 router 健康检查失败。

测试时间：

- 原始配置启动：2026-08-15 14:55（北京时间，日志 UTC 06:55）。
- 修复版 benchmark：2026-08-15 15:08:18 至 15:24:08（北京时间）。

## 2. 测试文件路径

最终 benchmark 结果位于 Decode 节点 `zly-h200-96`：

```text
/data/zhangluyu/glm52/bench_results/glm52-20260815-070818/
/data/zhangluyu/glm52/bench_results/glm52-20260815-070818/summary.txt
```

本次测试运行日志目录（P、D 是各自容器内的同名目录）：

```text
/data/zhangluyu/glm52/test_runs/20260815-145459/
```

主要日志：

```text
# P 节点 zly-h200-95
prefill.log                 # 原始配置，复现 CP/Mooncake 错误
prefill-fixed.log           # 修复版 P 服务日志

# D 节点 zly-h200-96
decode.log                  # 原始配置 D 服务日志
router.log                  # 原始配置 router 日志
benchmark-driver.log        # 首次失败 benchmark 的客户端日志
decode-fixed.log            # 修复版 D 服务日志
router-fixed.log            # 修复版 router 日志
benchmark-fixed-driver.log  # 最终完整 benchmark 客户端日志
benchmark-fixed.rc          # 最终脚本退出码，值为 0
```

首次失败 benchmark 的不完整结果保留在：

```text
/data/zhangluyu/glm52/bench_results/glm52-20260815-065953/
```

## 3. 测试过程

### 3.1 环境检查

启动前完成以下检查：

1. 两个 Pod 均为 `Running`，Pod IP 与 router 配置一致。
2. 两端 8 张 H200 显存使用均为 0 MiB。
3. `/dev/shm/GLM-5.2-FP8` 在两端均存在。
4. `30000`、`8000`、`58809` 没有被旧服务占用。
5. `basic_bench_example.sh` 位于 Decode 节点，且 `BASE=http://127.0.0.1:8000`，适合从 Router 所在 Pod 发起测试。

### 3.2 按原始命令启动

首先严格按给定 P/D 参数启动，仅移除了 Decode 命令中重复出现的第二个 `--disaggregation-mode decode`，该整理不改变语义。

原始配置启动成功：

- P `/health` 返回 200。
- D `/health` 返回 200。
- Router 使用 `--prefill http://172.20.28.215:30000` 和 `--decode http://172.20.5.107:30000` 注册成功。
- Mooncake Transfer Engine 与 RDMA 设备初始化没有报错。
- 短请求能够经过 Router 完成 P→D 推理。

原始配置的运行时容量为：

| 节点 | mem-fraction-static | max_total_num_tokens |
|---|---:|---:|
| P | 0.80 | 95,872 |
| D | 0.80 | 39,168 |

需要注意，D 启用了 DP attention 后，SGLang 在运行时把 Decode 的 `chunked_prefill_size` 调整为 4096；这是参数后处理结果，不是 benchmark 脚本修改。

### 3.3 原始配置复现的问题

首次运行 64K 冷前缀用例时，Prompt 被拆成两个 chunk：

```text
chunk 1: 32768 tokens -> 512 KV pages
chunk 2: 31232 tokens -> 488 KV pages
```

P 端记录到错误的目标页数：

```text
len(chunked_dst_kv_indice) = 1, len(kv_chunk.prefill_kv_indices) = 512
len(chunked_dst_kv_indice) = 0, len(kv_chunk.prefill_kv_indices) = 488
```

最终在 `group_concurrent_contiguous()` 中触发：

```text
operands could not be broadcast together with shapes (999,) (0,)
Transfer thread failed ... Prefill instance ... is dead
```

benchmark 客户端没有为单请求设置超时，因此一直等待已经无法完成的 warmup 请求。

问题根因有两部分：

1. CP 页过滤按“当前 chunk 的页数”计算 CP rank 所属页，而目标 offset 属于完整请求，第二个 chunk 开始后源/目标页域不一致。
2. `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` 默认是 False，NSA Prefill CP=8 时只有 CP rank 0 参与传输，无法正确表达完整的 CP 分片 KV 布局。

另外，D 原始 `max_total_num_tokens=39168`，低于 benchmark 的 64000-token 输入；即使传输索引不崩溃，也无法可靠承载该用例。

## 4. 已实施的修复

### 4.1 P 端 CP/Mooncake 源码修复

修改 P 容器运行时代码：

```text
/sgl-workspace/sglang/python/sglang/srt/disaggregation/utils.py
/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py
```

具体修改：

1. `filter_kv_indices_for_cp_rank()` 增加可选 `total_pages` 参数，按完整请求页数计算 CP rank 页面归属。
2. `MooncakeKVSender.send()` 传入 `total_pages=self.num_kv_indices`，避免每个 chunk 重新从零划分页域。
3. `maybe_send_extra()` 对 NSA/SWA 的 `prefill_state_indices` 与 `dst_state_indices` 做双向等长裁剪，确保 `_send_kvcache_generic()` 的源/目标索引长度一致。

修改后执行 `python3 -m py_compile` 成功。

### 4.2 CP rank 传输参数

在 P、D 两端增加：

```bash
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
```

使 8 个 CP rank 都参与对应 KV 分片传输，而不是只有 CP rank 0 传输。

### 4.3 D 端 KV 容量

将 D 的参数由：

```text
--mem-fraction-static 0.80
```

调整为：

```text
--mem-fraction-static 0.87
```

调整后实际 `max_total_num_tokens` 从 39,168 增加到 142,720，能够覆盖 64K Prompt 和 32K+1K Decode 用例。DeepGEMM warmup 与 CUDA graph 捕获均成功，没有发生 OOM。

## 5. 最终性能结果

| 用例 | 成功请求 | 请求吞吐 req/s | 输入吞吐 tok/s | 输出吞吐 tok/s | Mean TTFT ms | Mean TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| P 冷前缀，64×64K，C=16 | 64/64 | 0.32 | 20,623.51 | 0.32 | 37,180.92 | 0.00 |
| P 90% 共享前缀，64×64K，C=16 | 64/64 | 2.63 | 175,101.57 | 2.63 | 5,601.86 | 0.00 |
| D 32K+1K，4 请求，C=1 | 4/4 | 0.10 | 3,305.88 | 99.34 | 918.36 | 9.15 |
| D 32K+1K，64 请求，C=16 | 64/64 | 0.99 | 33,130.75 | 994.44 | 1,567.27 | 13.11 |
| D 32K+1K，128 请求，C=32 | 128/128 | 1.81 | 60,225.72 | 1,809.50 | 2,064.52 | 13.27 |
| D 32K+1K，256 请求，C=64 | 256/256 | 1.85 | 61,549.14 | 1,847.27 | 14,475.97 | 14.22 |
| D 32K+1K，512 请求，C=128 | 512/512 | 2.18 | 72,357.22 | 2,177.53 | 37,851.45 | 13.30 |

共享前缀场景的输入吞吐由 20,623.51 tok/s 提升到 175,101.57 tok/s，约为冷前缀的 8.49 倍。

Decode 输出吞吐随并发从 99.34 tok/s 提升到 2,177.53 tok/s；并发 128 的吞吐最高，但 Mean TTFT 增加到 37.85 秒，说明高并发主要以排队和首 token 延迟换取总吞吐。

## 6. 最终核验

```text
benchmark exit code: 0
benchmark cases: 7
rc=0 cases: 7
FAILED cases: 0
P health: 200
D health: 200
Router health: 200
P transfer/index fatal lines: 0
D OOM/traceback lines: 0
Router health failure lines: 0
```

## 7. 当前状态与注意事项

测试结束后，修复版 P、D 和 Router 保持运行，便于继续检查结果。

本次 P 端源码修改发生在运行中 Pod 的 `/sgl-workspace/sglang` 中；Pod 重建或更换镜像后会丢失。要长期使用该组合，应将两处源码修复合入镜像或正式代码分支，并在启动环境中保留：

```bash
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
```

D 端需要保留足以覆盖最大单请求上下文的 KV 容量。本次 H200 环境验证可用值为：

```text
--mem-fraction-static 0.87
```
