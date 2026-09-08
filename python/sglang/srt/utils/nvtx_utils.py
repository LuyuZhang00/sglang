# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler span helpers for hot SGLang code paths.

本文件的作用是为 SGLang 中的热点代码路径提供统一的性能标记工具。

它把“性能追踪”拆成了两条独立的输出通道：

* ``record_function``：只要 PyTorch profiler 处于启用状态，就会输出一个
  span。这样即便不显式打开 NVTX，也可以在 torch/Perfetto 跟踪中看到
  这些调用段落，适合在线程级 / operator 级的性能概览。
* ``nvtx`` range：只有在调用方显式设置 ``nvtx_enabled=True``，并且该
  子系统对应的 ``SGLANG_ENABLE_NVTX_*`` 开关开启、同时 ``nvtx`` 包可用时，
  才会输出 Nsight Systems 时间线中的范围标记。它更偏向于系统级/组件级
  的大粒度时序观察。

把两者解耦的好处在于：无论是调度器 stage、batch overlap 操作，还是
  speculative decoding / forward 相关的 span，都可以复用同一套 API，
  但根据需要分别输出到 torch profiler 和 NVTX 时间线里。

简单来说，这个文件负责“统一地给关键代码段打标记”，以便在性能分析时
知道某个阶段、某个函数、某个 batch 的执行时间分布，帮助定位热点和
等待/重叠问题。
"""

import logging
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial, wraps
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# 读取环境变量开关：不同子系统可以独立控制是否启用 NVTX 标记。
# 例如调度器、批处理/运算阶段都可以各自单开，避免全局打开导致噪声过大。
_SCHEDULER_NVTX = envs.SGLANG_ENABLE_NVTX_SCHEDULER.get()
_OPERATIONS_NVTX = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

# 如果有任何 NVTX 开关被启用，就尝试导入 NVIDIA 的 nvtx 库。
# 这层保护很重要：有些环境中并没有安装 nvtx；即使开关打开，也不能直接
# 把程序打崩，而是退化为只保留 torch profiler 的输出。
_nvtx_module = None
if _SCHEDULER_NVTX or _OPERATIONS_NVTX:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        logger.warning(
            "An SGLANG_ENABLE_NVTX_* flag is set, but the `nvtx` package is "
            "missing. NVTX markers are disabled; torch profiler spans still emit."
        )

# 只有在 nvtx 包确实可用时，才认为 NVTX 可用。
# 这里和 record_function 的逻辑并不绑定：Profiler 仍然可以在没有 nvtx 的
# 情况下输出 trace，NVTX 只是额外的高层时间线标记。
NVTX_AVAILABLE = _nvtx_module is not None
# Per-subsystem nvtx gates: emit nvtx ranges only when the flag is set AND the
# package is importable. The record_function path is independent of both.
# 也就是说，每个子系统都需要同时满足：
#   1) 对应环境变量开关打开
#   2) nvtx 包已经成功导入
# 才会真正发射 NVTX 区间。
NVTX_SCHEDULER_ENABLED = _SCHEDULER_NVTX and NVTX_AVAILABLE
NVTX_OPERATIONS_ENABLED = _OPERATIONS_NVTX and NVTX_AVAILABLE

# 为静态命名的 span 提供默认颜色。这部分只用于 NVTX 路径，不影响
# torch.profiler.record_function 的输出。
# 颜色有助于在 Nsight Systems 里快速看出不同模块类别的时间轴区间。
_NVTX_COLOR_MAP = {
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

_NULL_CONTEXT = nullcontext()


@contextmanager
def _profile_range_impl(
    debug_name: str, color: Optional[str], record: bool, nvtx_enabled: bool
):
    """真正执行 span 记录的内部实现。

    这个函数用 ExitStack 来组合多个事件源：

    - torch.profiler.record_function(debug_name)：如果 profiler 开启，则输出
      torch trace 中的 span。
    - _nvtx_module.annotate(debug_name, color=...)：如果 NVTX 开关打开，则输出
      Nsight Systems 中的时间范围标记。

    这样所有的性能标记逻辑都集中在一个地方，外部调用者只需要决定是否
    需要记录、是否需要 NVTX，而不必重复写相同的上下文管理逻辑。
    """
    with ExitStack() as stack:
        if record:
            # 只要 profiler 在启用时，record_function 就会自动记录。
            # 这部分完全不依赖环境变量或额外 Python 包。
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx_enabled:
            if color is None:
                color = _NVTX_COLOR_MAP.get(debug_name)
            # 对于 Nvtx，使用 annotate 形成一个范围区间；在 Nsight Systems 中
            # 这个区间会以颜色和名称的方式显示在时间线上。
            stack.enter_context(_nvtx_module.annotate(debug_name, color=color))
        yield


def profile_range(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Context manager emitting a profiler span for ``debug_name``.

    这是最核心的公共接口：在一个 with 语句块中对一段代码打 span。

    语义上它同时支持两种输出：
    - torch profiler span：在 profiler 开启时自动记载
    - nvtx range：在 ``nvtx_enabled=True`` 且环境开关/依赖满足时额外输出

    如果两者都未启用，则返回空 context，避免热点路径产生额外开销。

    这也是此文件“轻量化”的关键：off-profile 时不做任何实际工作，只做一个
    简单的布尔判断。
    """
    record = torch.autograd._profiler_enabled()
    if not record and not nvtx_enabled:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, color, record, nvtx_enabled)


def profile_method(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Decorator form of ``profile_range``.

    这个装饰器适用于需要给整个函数调用打 span 的场景。它会在函数进入时
    创建一个 range，并在函数退出时自动结束，适合快速埋点而不改动函数体。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_range(debug_name, color=color, nvtx_enabled=nvtx_enabled):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-bound per-subsystem helpers: torch spans always (under a profiler), nvtx
# ranges only when that subsystem's gate is on.
# 这里预先绑定了不同子系统的开关配置，调用方可以直接拿到 ready-to-use 的
# helper，而不需要每次重复传 nvtx_enabled。
# 例如：scheduler 相关的方法可以直接使用 scheduler_nvtx_method，
# 而运算/forward 层相关的标记可以用 operations_nvtx_range。
scheduler_nvtx_method = partial(profile_method, nvtx_enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(profile_range, nvtx_enabled=NVTX_OPERATIONS_ENABLED)
