# [Deprecated] Re-export shim for backward compatibility. Use dsa.index_buf_accessor instead.
"""兼容转发：按 page / token 读写 NSA index-K buffer 的访问器。

把物理布局（含 HIP preshuffle tile）藏在访问接口后面。实现已迁到
``kernels.ops.attention.dsa.index_buf_accessor``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.index_buf_accessor is deprecated; "
    "use sglang.kernels.ops.attention.dsa.index_buf_accessor instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.kernels.ops.attention.dsa.index_buf_accessor import *  # noqa: F401, F403
