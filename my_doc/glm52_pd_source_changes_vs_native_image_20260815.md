# GLM-5.2 PD 测试源码修改说明：与原生镜像逐项对比

## 1. 结论

本次 PD 测试对源码的实际修改只有：

- 修改节点：仅 Prefill Pod `zly-h200-95`。
- 修改文件：2 个。
- 逻辑修改点：3 个。
- Decode Pod `zly-h200-96`：没有源码修改。
- `group_concurrent_contiguous()`、`prefill.py`、router 源码：均未修改。

两个修改文件是：

```text
/sgl-workspace/sglang/python/sglang/srt/disaggregation/utils.py
/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py
```

三个逻辑修改点是：

1. `filter_kv_indices_for_cp_rank()` 支持显式传入完整请求的 `total_pages`。
2. `MooncakeKVSender.send()` 使用 `self.num_kv_indices` 传入完整请求页数。
3. NSA/SWA extra state 传输前，对源、目标 state indices 做双向等长裁剪。

除此以外，测试还修改了一个环境变量和一个启动参数，但它们不属于源码修改：

```bash
# P、D 两端新增
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

# D 端启动参数
--mem-fraction-static 0.80  ->  0.87
```

## 2. 对比基线与识别方法

P、D 使用同一个镜像名称、同一个镜像 digest：

```text
image: registry.cn-sh-02.sensecore.cn/ccr-axion/sglang:dev-ae5b7040-cu129
digest: sha256:c47d7267e8ae0c1506afd1413e779436657c277f6c5fd39ddbc5d9ce48378ffc
embedded git HEAD: 1519acf37c23f2189adb93f57ca9cd2db1bebf18
```

本次测试只在 P 端写入源码，D 端没有写入源码，因此使用 D 端作为“该镜像实际出厂文件”的对照副本。同时使用本次修改前保存的 P 文件备份复核。

对全部 tracked files 做 SHA256 对比后，只有以下两个文件不同：

| 文件 | 原生镜像/D SHA256 | 修改后/P SHA256 |
|---|---|---|
| `python/sglang/srt/disaggregation/utils.py` | `cbe49379ee38813a583206b73c579dbd9b98267b825d062a269e9d9c99481399` | `c590dd6f5d9f44bfea42be7ba1a18ac347733764d752d3284274e58ddc3326c2` |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | `df3d9893887ee441621333de460e3dc8319964e0e7069d994c2ec109fd0510df` | `d4a4e1b5915e03f5ca40c0917b46694d8723132ba8f7f985e8c237052d241704` |

进一步将 `python/`、`benchmark/`、`scripts/`、`test/` 下所有 tracked 和 untracked 源文件一起比较，并排除运行生成的 `__pycache__` 和 `*.pyc`，结果仍然只有上述两个文件不同。

### 2.1 为什么不能直接把 `git diff HEAD` 当作本次改动

该镜像的工作树在容器刚启动时就不是 clean 状态。P、D 两端都显示：

```text
514 个 tracked 文件为 M
53 个 untracked 项为 ??
```

这些 514+53 项在 P、D 两端一致，是镜像构建或镜像打包时已经存在的基线内容，不是本次测试产生的改动。因此：

- “相对嵌入的 Git commit `1519acf`”：镜像本身已经有大量差异。
- “相对实际原生镜像文件”：本次测试只有 2 个文件、3 个逻辑修改点。

本文所说的“原生镜像代码”指镜像 digest `c47d...8ffc` 实际携带的文件，而不是单纯使用 `git show HEAD` 得到的内容。

## 3. 修改点一：按完整请求页数划分 CP 页面

文件：

```text
python/sglang/srt/disaggregation/utils.py
```

函数：

```text
filter_kv_indices_for_cp_rank()
```

修改后 P 端大约位于第 473 行。

### 3.1 原生镜像代码

```python
def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
```

### 3.2 修改后代码

```python
def filter_kv_indices_for_cp_rank(
    kv_mgr: CommonKVManager,
    kv_indices: np.ndarray,
    index_slice: slice,
    total_pages: Optional[int] = None,
) -> Tuple[np.ndarray, slice]:
    """Filters kv_indices and index_slice for the current CP rank."""
    # A chunk can contain only part of the request. Split against the full
    # request page count so every CP rank uses the same destination offsets.
    if total_pages is None:
        total_pages = len(kv_indices)
    cp_rank = kv_mgr.attn_cp_rank
    cp_size = kv_mgr.attn_cp_size

    rank_page_indices = page_indices_to_cp_rank_page_indices(
        page_indices=kv_indices,
        total_pages=total_pages,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
```

### 3.3 修改原因

原生代码把 `len(kv_indices)` 当成 `total_pages`。这只在 `kv_indices` 覆盖完整请求时成立；开启 chunked prefill 后，`kv_indices` 只代表当前 chunk。

64K 测试的真实拆分为：

```text
完整请求：64000 tokens / page_size 64 = 1000 pages
chunk 1：32768 tokens = 512 pages
chunk 2：31232 tokens = 488 pages
```

但是 `index_slice` 和 D 端预分配的目标页面 offset 属于完整的 1000-page 请求。如果每个 chunk 分别用 512 或 488 作为 `total_pages`，每个 chunk 都会重新计算 CP rank 的局部起点，源页域和请求全局目标 offset 便不再一致。

增加 `total_pages` 参数后，调用者可以始终传入 1000，CP rank 页面划分在两个 chunk 之间保持同一个全局坐标系。

### 3.4 兼容性

`total_pages` 是可选参数。没有传入时仍回退到 `len(kv_indices)`，因此未修改的调用点保持原行为。

## 4. 修改点二：Sender 传入请求级总页数

文件：

```text
python/sglang/srt/disaggregation/mooncake/conn.py
```

类和函数：

```text
MooncakeKVSender.send()
```

修改后 P 端大约位于第 1149～1165 行。

### 4.1 原生镜像代码

```python
if self.kv_mgr.enable_all_cp_ranks_for_transfer:
    kv_indices, index_slice = filter_kv_indices_for_cp_rank(
        self.kv_mgr,
        kv_indices,
        index_slice,
    )
```

### 4.2 修改后代码

```python
if self.kv_mgr.enable_all_cp_ranks_for_transfer:
    kv_indices, index_slice = filter_kv_indices_for_cp_rank(
        self.kv_mgr,
        kv_indices,
        index_slice,
        total_pages=self.num_kv_indices,
    )
```

### 4.3 修改原因和依据

`MooncakeKVSender` 在请求级别维护：

```python
self.num_kv_indices
self.curr_idx
```

其中 `self.num_kv_indices` 是当前完整请求需要传输的 KV page 数；`send()` 的 `kv_indices` 只是某次 chunk 调用携带的页。

因此：

- `self.num_kv_indices` 适合作为请求级 `total_pages`。
- `len(kv_indices)` 只能作为 chunk 级页数。
- `index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))` 已经按请求全局累计，所以页面划分也必须使用请求全局总数。

实际失败日志直接证明了原映射错误：

```text
len(chunked_dst_kv_indice) = 1, len(kv_chunk.prefill_kv_indices) = 512
len(chunked_dst_kv_indice) = 0, len(kv_chunk.prefill_kv_indices) = 488
```

修复后相同 64K warmup 和 64 个 64K 请求均成功完成，修复版 P 日志中上述长度异常计数为 0。

## 5. 修改点三：NSA/SWA state indices 双向等长裁剪

文件：

```text
python/sglang/srt/disaggregation/mooncake/conn.py
```

函数：

```text
maybe_send_extra()
```

对应 `state_type in ["swa", "nsa"]` 分支，修改后 P 端大约位于第 623～654 行。

### 5.1 原生镜像代码

```python
if len(prefill_state_indices) < len(req.dst_state_indices):
    logger.warning(
        f"len(prefill_state_indices) = {len(prefill_state_indices)}, "
        f"len(dst_state_indices) = {len(req.dst_state_indices)}"
    )
    prefill_state_indices = prefill_state_indices[
        : len(req.dst_state_indices)
    ]

prefill_state_indices = np.array(prefill_state_indices, dtype=np.int32)
dst_state_indices = np.array(req.dst_state_indices, dtype=np.int32)
```

### 5.2 修改后代码

```python
dst_state_indices = req.dst_state_indices
if len(prefill_state_indices) > len(dst_state_indices):
    logger.warning(
        f"len(prefill_state_indices) = {len(prefill_state_indices)}, "
        f"len(dst_state_indices) = {len(dst_state_indices)}"
    )
    prefill_state_indices = prefill_state_indices[: len(dst_state_indices)]
elif len(prefill_state_indices) < len(dst_state_indices):
    logger.warning(
        f"len(prefill_state_indices) = {len(prefill_state_indices)}, "
        f"len(dst_state_indices) = {len(dst_state_indices)}"
    )
    dst_state_indices = dst_state_indices[: len(prefill_state_indices)]

prefill_state_indices = np.array(prefill_state_indices, dtype=np.int32)
dst_state_indices = np.array(dst_state_indices, dtype=np.int32)
```

### 5.3 原生逻辑的问题

原生代码只有在：

```text
len(prefill_state_indices) < len(dst_state_indices)
```

时进入分支，但随后却对更短的 `prefill_state_indices` 使用更长的目标长度切片。Python 切片不会补齐数据，所以源数组仍然更短，目标数组也没有被裁剪，两个数组依旧不等长。

原生代码也完全没有处理：

```text
len(prefill_state_indices) > len(dst_state_indices)
```

### 5.4 为什么必须等长

两个数组最终传入：

```python
_send_kvcache_generic(
    prefill_data_indices=prefill_state_indices,
    dst_data_indices=dst_state_indices,
)
```

内部 `group_concurrent_contiguous()` 同时执行：

```python
(np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1)
```

两个索引数组长度不一致时，`np.diff()` 结果无法广播。例如测试中出现：

```text
src/dst 约为 1000 对 1
np.diff 后为 shapes (999,) 和 (0,)
```

因此修复逻辑以两者较短长度为准：

- 源长、目标短：裁剪源。
- 源短、目标长：裁剪目标。
- 已等长：不修改。

这保证传入 `_send_kvcache_generic()` 的源、目标索引一一对应。

## 6. 为什么还要启用 all-CP-ranks transfer

这不是源码修改，但它决定修改点一、二是否生效。

原生环境定义为：

```python
SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
```

当它为 False，Prefill CP size=8 时：

- CP rank 0 负责发送。
- CP rank 1～7 被标记为 dummy CP rank。

本次测试增加：

```bash
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
```

使所有 CP rank 都进入：

```python
filter_kv_indices_for_cp_rank(...)
```

每个 rank 只传输属于自己的请求级页面范围。仅修改函数但不打开该环境变量，NSA Prefill CP=8 的运行路径不会完整使用新的 CP 分片逻辑。

## 7. 为什么只修改 P，不修改 D

这三处逻辑都位于发送侧：

- P 持有 prefill 计算生成的 KV/state 数据。
- P 的 `MooncakeKVSender.send()` 决定哪些源页面映射到 D 预分配的目标 offset。
- P 的 `maybe_send_extra()` 组织 NSA/SWA state 的源、目标索引。

D 端继续负责预分配目标 KV/state 页面并把目标信息返回给 P。当前错误发生在 P 组织源/目标索引并调用 Mooncake 发送时，因此 D 端不需要同步修改这两个文件。

环境变量 `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` 在 P、D 两端都设置，是为了让两端使用一致的 CP transfer 协议状态；这不等同于 D 端源码修改。

## 8. 明确未修改的代码

以下文件在 P、D 的 SHA256 完全一致，本次没有修改：

```text
python/sglang/srt/disaggregation/common/utils.py
python/sglang/srt/disaggregation/prefill.py
python/sglang/srt/disaggregation/common/conn.py
router/model_gateway 源码
```

特别说明：没有在 `group_concurrent_contiguous()` 中仅增加长度检查。长度检查只能把 NumPy broadcast error 改写成更清晰的 ValueError，不能修复错误的源/目标页面映射，所以本次直接修复上游索引生成逻辑。

## 9. 完整 unified diff

```diff
--- python/sglang/srt/disaggregation/utils.py
+++ python/sglang/srt/disaggregation/utils.py
@@
 def filter_kv_indices_for_cp_rank(
-    kv_mgr: CommonKVManager, kv_indices: np.ndarray, index_slice: slice
+    kv_mgr: CommonKVManager,
+    kv_indices: np.ndarray,
+    index_slice: slice,
+    total_pages: Optional[int] = None,
 ) -> Tuple[np.ndarray, slice]:
     """Filters kv_indices and index_slice for the current CP rank."""
-    total_pages = len(kv_indices)
+    # A chunk can contain only part of the request. Split against the full
+    # request page count so every CP rank uses the same destination offsets.
+    if total_pages is None:
+        total_pages = len(kv_indices)

--- python/sglang/srt/disaggregation/mooncake/conn.py
+++ python/sglang/srt/disaggregation/mooncake/conn.py
@@
         if self.kv_mgr.enable_all_cp_ranks_for_transfer:
             kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                 self.kv_mgr,
                 kv_indices,
                 index_slice,
+                total_pages=self.num_kv_indices,
             )
@@
-            if len(prefill_state_indices) < len(req.dst_state_indices):
+            dst_state_indices = req.dst_state_indices
+            if len(prefill_state_indices) > len(dst_state_indices):
                 logger.warning(
-                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(req.dst_state_indices)}"
+                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(dst_state_indices)}"
                 )
-                prefill_state_indices = prefill_state_indices[
-                    : len(req.dst_state_indices)
-                ]
+                prefill_state_indices = prefill_state_indices[: len(dst_state_indices)]
+            elif len(prefill_state_indices) < len(dst_state_indices):
+                logger.warning(
+                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(dst_state_indices)}"
+                )
+                dst_state_indices = dst_state_indices[: len(prefill_state_indices)]
             prefill_state_indices = np.array(prefill_state_indices, dtype=np.int32)
-            dst_state_indices = np.array(req.dst_state_indices, dtype=np.int32)
+            dst_state_indices = np.array(dst_state_indices, dtype=np.int32)
```

## 10. 验证依据

修复前：

```text
64K warmup 卡住
chunked destination pages = 1 / 0
prefill source pages = 512 / 488
Transfer thread failed
```

修复后：

```text
python3 -m py_compile: 成功
P/D 全源文件哈希差异: 仅 2 个目标文件
64K warmup: 成功
64 个 64K 冷前缀请求: 64/64 成功
64 个 64K 90%共享前缀请求: 64/64 成功
完整 benchmark: 7/7 case rc=0
P transfer/index fatal lines: 0
D OOM/traceback lines: 0
Router health failure lines: 0
```

完整性能结果和日志路径参见：

```text
my_doc/glm52_pd_benchmark_20260815.md
/data/zhangluyu/glm52/bench_results/glm52-20260815-070818/summary.txt
/data/zhangluyu/glm52/test_runs/20260815-145459/
```

## 11. 持久化与回退

当前改动是写入 P Pod 容器文件系统的热修改：

- 当前 P 服务已加载修改后的实现。
- P Pod 重建或更换镜像后，修改会丢失。
- 只恢复磁盘文件、不重启 P 服务时，正在运行的 Python 进程仍使用已加载的旧实现。

要长期使用，应把上述两个文件的三个修改点合入正式镜像或代码分支，并保留 P、D 两端的：

```bash
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
```
