# [Deprecated] Re-export shim for backward compatibility. Use dsa.dsa_backend_mtp_precompute instead.
"""兼容转发：投机解码（MTP / EAGLE）下把 DSA metadata 提到 CUDA Graph 外预计算。

避免每步 draft decode 在 graph 内做动态 shape 的 page-table 变换。
实现已迁到 ``dsa.dsa_backend_mtp_precompute``。
"""
import warnings

warnings.warn(
    "sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute is deprecated; "
    "use sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import *  # noqa: F401, F403
