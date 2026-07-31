# SGLang PD 分离部署测试指南

## 1. 环境信息

| 项目 | 信息 |
|------|------|
| SGLang 版本 | 0.5.16 |
| GPU | 4x NVIDIA GeForce RTX 5090 (32GB) |
| CUDA 版本 | 13.0 |
| Python | 3.12 |
| 模型 | Qwen3-4B-Instruct-2507 |
| Mooncake 版本 | 0.3.12.post1 |
| NIXL 版本 | 1.3.2 |
| 操作系统 | Linux 6.8.0-60-generic |

## 2. 遇到的问题及修复

### 问题 1：Mooncake `libcudart.so.12` 导入失败

**错误信息：**

```
ImportError: Please install mooncake by following the instructions at
https://kvcache-ai.github.io/Mooncake/getting_started/build.html to run
SGLang with MooncakeTransferEngine.
```

**根因分析：**

系统安装的是 CUDA 13.0，但 Mooncake 的 `engine.so` 是基于 CUDA 12 编译的，运行时需要 `libcudart.so.12` 的版本符号。系统中的软链接 `/usr/local/cuda/lib64/libcudart.so.12 -> libcudart.so.13` 缺少该版本符号，导致动态链接失败。

```bash
# 验证方式
ldd /usr/local/lib/python3.12/dist-packages/mooncake/engine.so 2>&1 | grep cudart
# 输出：libcudart.so.12: version `libcudart.so.12' not found
```

**修复方法：**

安装 CUDA 12.8 运行时库，并将其路径加入 `LD_LIBRARY_PATH`：

```bash
apt-get install -y cuda-cudart-12-8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

验证修复：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
  python3 -c "from mooncake.engine import TransferEngine; print('OK')"
# 输出：OK
```

---

### 问题 2：RDMA 内存注册失败

**错误信息：**

```
E0731 10:23:08.345825 35648 rdma_transport.cpp:656]
Memory region not registered by any active device(s): 0x70257c006000
Session 10.119.219.12:16948 failed.
```

**根因分析：**

Mooncake 默认使用 RDMA 传输，但当前环境中的 RDMA 设备（mlx5_bond_0）使用 Ethernet 链路层而非 InfiniBand，导致 CUDA 内存区域注册到 RDMA 设备时失败。

```bash
# 查看 RDMA 设备状态
ibstat
# Link layer: Ethernet（非 InfiniBand）
```

**修复方法：**

设置环境变量 `MC_FORCE_TCP=1`，强制 Mooncake 使用 TCP 传输替代 RDMA：

```bash
export MC_FORCE_TCP=1
```

参考代码（`mooncake_transfer_engine.py:125-128`）：

```python
# MC_FORCE_TCP=1 makes mooncake install TcpTransport instead of RDMA,
# in which case RDMA HCA selection is irrelevant; pass empty device.
```

---

### 问题 3：Router 未指定 Prefill 的 Bootstrap 端口

**错误信息：**

```
Error fetching prefill server info from bootstrap:
HTTPConnectionPool(host='127.0.0.1', port=8999): Connection refused
```

**根因分析：**

Decode server 启动时会尝试连接 prefill server 的 bootstrap 端口来获取路由信息。如果 router 未将 prefill 的 bootstrap 端口传递给 decode server，decode server 会尝试连接自己的 bootstrap 端口（8999），而非 prefill 的（8998）。

**修复方法：**

在 `--prefill` 参数后追加 bootstrap 端口号：

```bash
# 错误写法
--prefill http://127.0.0.1:30000

# 正确写法（8998 是 prefill server 的 bootstrap 端口）
--prefill http://127.0.0.1:30000 8998
```

---

### 问题 4：Prometheus 端口冲突

**错误信息：**

```
pyo3_runtime.PanicException: failed to install Prometheus metrics exporter:
FailedToCreateHTTPListener("Address already in use (os error 98)")
```

**根因分析：**

Router 默认的 Prometheus metrics 端口（9090）被其他进程占用。

**修复方法：**

通过 `--prometheus-port` 参数指定一个空闲端口：

```bash
--prometheus-port 9100
```

---

## 3. 最终启动命令

```bash
# ============================================
# 1) 设置环境变量
# ============================================
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export MC_FORCE_TCP=1

# ============================================
# 2) 启动 Prefill Server（GPU 0, port 30000）
# ============================================
python3 -m sglang.launch_server \
  --model-path /root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 \
  --port 30000 \
  --base-gpu-id 0 &

# ============================================
# 3) 启动 Decode Server（GPU 1, port 30001）
# ============================================
python3 -m sglang.launch_server \
  --model-path /root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  --host 0.0.0.0 \
  --port 30001 \
  --base-gpu-id 1 &

# ============================================
# 4) 启动 Router（port 8000）
# ============================================
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 8998 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --prometheus-port 9100 &
```

## 4. 测试结果

### 测试 1：中文对话

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master",
    "messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}],
    "max_tokens": 64
  }'
```

**结果：** 成功

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "你好！我是Qwen，是阿里云研发的超大规模语言模型。我可以帮助你回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等，还能表达观点，玩游戏等。我支持多种语言，包括但不限于中文、英文、德"
    },
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 13, "total_tokens": 77, "completion_tokens": 64}
}
```

### 测试 2：英文问答

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master",
    "messages": [{"role": "user", "content": "What is 2+3? Answer briefly."}],
    "max_tokens": 32
  }'
```

**结果：** 成功

```json
{
  "choices": [{
    "message": {"content": "2 + 3 = 5."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 18, "total_tokens": 27, "completion_tokens": 9}
}
```

### 测试 3：流式输出

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master",
    "messages": [{"role": "user", "content": "用Python写一个hello world"}],
    "max_tokens": 128,
    "stream": true
  }'
```

**结果：** 成功，逐 token 流式返回。

### 测试 4：Completions API

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/modelscope/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master",
    "prompt": "The capital of France is",
    "max_tokens": 16
  }'
```

**结果：** 成功

```json
{
  "choices": [{
    "text": " Paris. Paris is a major city in France and has a rich cultural heritage.",
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 5, "total_tokens": 21, "completion_tokens": 16}
}
```

## 5. 架构说明

```
用户请求
   │
   ▼
┌──────────┐    port 8000
│  Router   │ ◄── sglang_router (PD 模式)
└────┬─────┘
     │
     ├──────────────────┐
     ▼                  ▼
┌──────────┐     ┌──────────┐
│ Prefill  │     │  Decode  │
│ Server   │     │  Server  │
│ GPU 0    │     │  GPU 1   │
│ :30000   │     │  :30001  │
│ bport:8998│     │ bport:8999│
└──────────┘     └──────────┘
      │                  │
      └──── Mooncake ────┘
         (TCP 传输)
```

- **Router**：接收用户请求，将请求同时发送到 prefill 和 decode server
- **Prefill Server**：负责计算 KV cache，通过 Mooncake 将 KV cache 传输给 decode server
- **Decode Server**：接收 KV cache，执行自回归解码生成 token
- **Mooncake**：KV cache 传输引擎，本环境使用 TCP 模式（`MC_FORCE_TCP=1`）

## 6. 注意事项

1. **环境变量必须在启动 server 之前设置**，否则 Mooncake 会因找不到 CUDA 12 运行库或尝试 RDMA 而失败
2. **`MC_FORCE_TCP=1` 仅适用于没有正确配置 InfiniBand 的环境**，生产环境建议使用 RDMA 以获得更低延迟
3. **Router 的 `--prefill` 参数后必须指定 bootstrap 端口**，否则 decode server 无法正确连接 prefill server
4. **服务启动顺序**：先启动 prefill server，再启动 decode server，最后启动 router
5. **端口规划**：
   - 30000: Prefill Server HTTP
   - 30001: Decode Server HTTP
   - 8998: Prefill Bootstrap
   - 8999: Decode Bootstrap
   - 8000: Router HTTP
   - 9100: Prometheus Metrics
