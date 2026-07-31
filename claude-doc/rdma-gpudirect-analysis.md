# RDMA + GPUDirect RDMA 容器内分析报告

## 1. 环境信息

| 项目 | 信息 |
|------|------|
| NIC | Mellanox ConnectX-6 (MT2894, vendor_part_id=4127) |
| RDMA 设备 | mlx5_bond_0 |
| 传输类型 | InfiniBand (0) — 硬件支持 IB |
| 链路层 | Ethernet（当前配置为以太网模式） |
| 端口状态 | ACTIVE |
| GPU | 4x NVIDIA RTX 5090 |
| GPU NUMA 节点 | Node 0 (PCIe: 16:00.0, 27:00.0, 49:00.0, 5A:00.0) |
| NIC NUMA 节点 | Node 1 (PCIe: a8:00.0) |
| nvidia_peermem | 已加载，peerdirect_support=0 |
| 容器权限 | 非特权，无 cap_sys_module / cap_sys_admin |

## 2. 测试结果总览

### 2.1 ibv_reg_mr 内存注册测试

| 内存类型 | ibv_reg_mr 结果 | errno |
|---------|:--------------:|-------|
| 普通 CPU 内存 | ✅ OK | — |
| CUDA pinned 内存 | ❌ 失败 | EFAULT (Bad address) |
| GPU 设备内存 | ❌ 失败 | EFAULT (Bad address) |

### 2.2 Mooncake TransferEngine 测试

| 协议 | 创建引擎 | 注册 GPU 内存 |
|------|:-------:|:------------:|
| rdma | ✅ | ❌ EFAULT |
| tcp (MC_FORCE_TCP=1) | ✅ | ✅（不需要注册） |
| nvlink | ✅ | ❌ Segfault |
| nvlink_intra | ✅ | ❌ Segfault |

### 2.3 nixl (UCX 后端) 测试

| 内存类型 | registerMem 结果 |
|---------|:----------------:|
| CPU 内存 | ✅ NIXL_SUCCESS |
| GPU 内存 | ❌ ibv_reg_mr failed: Bad address |

### 2.4 容器权限测试

| 操作 | 结果 | 原因 |
|------|:----:|------|
| 写入 peerdirect_support 参数 | ❌ | 文件只读 (`-r--r--r--`) |
| modprobe 重载 nvidia_peermem | ❌ | 无 cap_sys_module |
| sysfs 写入 | ❌ | read-only file system |

## 3. 根因分析

### 3.1 问题链路

```
GPU 内存申请 (CUDA)
       │
       ▼
ibv_reg_mr() 尝试注册 GPU 内存到 RDMA
       │
       ▼
mlx5 驱动检查是否有 peer memory 客户端可处理 GPU 内存
       │
       ▼
nvidia_peermem 是 peer memory 客户端，但 peerdirect_support=0
       │
       ▼
mlx5 无法通过 GPUDirect RDMA 访问 GPU 内存页表
       │
       ▼
返回 EFAULT (Bad address)
```

### 3.2 关键参数说明

| 参数 | 当前值 | 含义 |
|------|:------:|------|
| `peerdirect_support` | **0** | 禁用 PeerDirect API，mlx5 无法使用 GPUDirect RDMA 注册 GPU 内存 |
| `persistent_api_support` | 1 | 启用持久化 peer memory API，但仅此一项不足以工作 |
| `nvidia_peermem refcount` | 0 | 模块已加载但从未被任何 RDMA 应用成功使用 |

### 3.3 为什么 CPU 内存可以，GPU 不行

- **CPU 内存**：由内核页表管理，`ibv_reg_mr` 可以直接通过 `get_user_pages()` 获取物理页信息，注册到 mlx5 的 Memory Region
- **GPU 内存**：由 NVIDIA 驱动的页表管理（不在内核页表中），`ibv_reg_mr` 需要通过 `nvidia_peermem` 模块作为 peer memory 客户端来获取 GPU 物理页信息。`peerdirect_support=0` 阻断了这条路径

### 3.4 拓扑问题

```
         NUMA Node 0                    NUMA Node 1
  ┌──────────────────────┐      ┌──────────────────┐
  │ GPU0 (16:00.0)       │      │ NIC (a8:00.0)    │
  │ GPU1 (27:00.0)       │ QPI  │ mlx5_bond_0      │
  │ GPU2 (49:00.0)       │◄────►│                  │
  │ GPU3 (5A:00.0)       │      │                  │
  └──────────────────────┘      └──────────────────┘
```

- NIC 和 GPU 跨 NUMA 节点，拓扑为 `SYS`（最差路径）
- 即使 GPUDirect RDMA 能工作，跨 NUMA 也会增加 ~50-100ns 延迟
- 但这不是 GPU 内存注册失败的原因——根因是 `peerdirect_support=0`

## 4. 容器内尝试的所有方法

| # | 方法 | 结果 | 说明 |
|---|------|:----:|------|
| 1 | 写入 `peerdirect_support=1` | ❌ | sysfs 参数文件只读，无写权限 |
| 2 | `modprobe nvidia_peermem peerdirect_support=1` | ❌ | 无 cap_sys_module 权限 |
| 3 | Mooncake `rdma` 协议 | ❌ | 底层 ibv_reg_mr 失败 |
| 4 | Mooncake `nvlink` / `nvlink_intra` 协议 | ❌ | 注册内存时 Segfault |
| 5 | nixl UCX 后端 | ❌ | 同样 ibv_reg_mr GPU 内存失败 |
| 6 | CUDA pinned (page-locked) 内存 | ❌ | CUDA 管理的内存同样 EFAULT |
| 7 | 不同 ibv_reg_mr access flags | ❌ | 所有 flag 组合均失败 |
| 8 | Mooncake allocate_managed_buffer | ❌ | Segfault |
| 9 | 触发 CUDA UVM 初始化后重试 | ❌ | nvidia_peermem refcount 仍为 0 |
| 10 | CPU 内存注册 | ✅ | 不涉及 GPU，正常工作 |
| 11 | `MC_FORCE_TCP=1` | ✅ | 绕过 RDMA，使用 TCP 传输 |

## 5. 解决方案

### 5.1 当前可行方案：TCP 传输（容器内）

```bash
export MC_FORCE_TCP=1
```

- ✅ 无需任何权限或配置改动
- ✅ 已验证可用
- ⚠️ 延迟比 RDMA 高，但对小模型（如 Qwen3-4B）影响有限

### 5.2 根本解决方案：宿主机操作

需要宿主机管理员执行以下操作之一：

#### 方案 A：重新加载 nvidia_peermem（推荐）

```bash
# 宿主机上执行
modprobe -r nvidia_peermem
modprobe nvidia_peermem peerdirect_support=1

# 验证
cat /sys/module/nvidia_peermem/parameters/peerdirect_support
# 应输出 1
```

#### 方案 B：以特权模式重启容器

```bash
docker run --privileged ...
# 或
docker run --cap-add SYS_MODULE --cap-add SYS_ADMIN ...
```

#### 方案 C：修改容器启动参数透传 sysfs 写权限

```bash
docker run \
  --device=/dev/infiniband \
  --device=/dev/nvidia0 \
  --device=/dev/nvidia1 \
  --device=/dev/nvidiactl \
  --cap-add SYS_MODULE \
  ...
```

### 5.3 完整 RDMA 方案（宿主机 + 容器）

如果要实现完整的 RDMA + GPUDirect RDMA，还需要：

#### 步骤 1：切换网卡到 InfiniBand 模式（宿主机）

```bash
apt-get install -y mstflink
mst status
mlxconfig -d /dev/mst/mt*_pciconf0 set LINK_TYPE_P1=1
reboot
```

#### 步骤 2：启用 peerdirect_support（宿主机）

```bash
modprobe -r nvidia_peermem
modprobe nvidia_peermem peerdirect_support=1
```

#### 步骤 3：启动 Subnet Manager（宿主机）

InfiniBand 模式需要 SM 来分配 LID：

```bash
# 如果有 IB 交换机，交换机自带 SM
# 如果直连，需要运行 OpenSM
apt-get install -y opensm
opensm &
```

#### 步骤 4：验证（宿主机）

```bash
ibstat
# 期望: Link layer: InfiniBand, Port LID: <非零值>

ibv_devinfo
# 期望: link_layer: InfiniBand
```

#### 步骤 5：验证 GPUDirect（容器内）

```python
import torch
from mooncake.engine import TransferEngine

te = TransferEngine()
t = torch.zeros(1024, dtype=torch.uint8, device='cuda:0')
result = te.register_memory(t.data_ptr(), 1024)
print(f"GPU register_memory: {result}")  # 应为 0 (成功)
```

## 6. 可选优化：调整 NIC 到同 NUMA 节点

当前 NIC 在 NUMA Node 1，GPU 在 Node 0。优化方法：

- 将 ConnectX-6 网卡物理拔出，插入到 Node 0 对应的 PCIe 插槽
- 通过 `lspci -vvv` 确认 PCIe 插槽所属 NUMA 节点
- 重新启动后验证 `nvidia-smi topo -m` 中 NIC-GPU 拓扑变为 `NODE` 或 `PHB`

## 7. 参考信息

| 资源 | 说明 |
|------|------|
| nvidia_peermem 参数 | `/sys/module/nvidia_peermem/parameters/` |
| RDMA 设备信息 | `/sys/class/infiniband/mlx5_bond_0/` |
| GPU-NIC 拓扑 | `nvidia-smi topo -m` |
| RDMA 设备列表 | `ibv_devices` / `ibv_devinfo` |
| IB 端口状态 | `ibstat` |
| Mooncake 协议配置 | `MOONCAKE_PROTOCOL` 环境变量 |
| SGLang Mooncake 配置 | `python/sglang/srt/environ.py` 中 `MOONCAKE_*` 变量 |
