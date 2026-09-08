# [Deprecated] Re-export shim for backward compatibility. Use dsa.utils instead.
"""兼容转发：DSA 公共判定与 CP 切分工具，现位于 ``dsa.utils``。

包含 preshuffle 探测、Prefill Context Parallel 切分、DP/CP 补齐长度，
以及 FP8 分页 MQA 的物理布局转换。Indexer 与 backend 必须共用同一套规则。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.utils is deprecated; "
    "use sglang.srt.layers.attention.dsa.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.utils import *  # noqa: F401, F403
