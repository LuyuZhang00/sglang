"""DSA 分页 MQA logits 算子的后端选择。

该文件只负责把用户配置解析为确定的后端枚举；真正的 logits 计算位于
Indexer 及各后端算子实现中。这里提前校验硬件与后端的组合，避免运行到
稀疏索引阶段才以难以理解的算子错误失败。
"""

from __future__ import annotations

from enum import Enum

from sglang.srt.utils import is_hip, is_sm100_supported


class DSAPagedMQALogitsBackend(Enum):
    """DSA 分页 MQA logits 当前可用的三类算子实现。"""

    # NVIDIA 通用路径，使用 DeepGEMM 计算分页 MQA logits。
    DEEPGEMM = "deepgemm"
    # Blackwell（SM100）专用的 CuteDSL 实现。
    CUTEDSL = "cutedsl"
    # AMD ROCm 平台使用的 AITER 实现。
    AITER = "aiter"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        """把命令行字符串解析成与当前硬件匹配的后端。

        ROCm 目前只能选择 AITER；CUDA 默认选择 DeepGEMM。
        CuteDSL 依赖 Blackwell 的 SM100 指令能力，因此在这里做前置校验。
        返回值会被 Indexer 用来决定后续导入和调用哪套分页 logits 算子。
        """
        # HIP/ROCm 平台没有 DeepGEMM 与 CuteDSL 的 DSA 分页实现。
        if is_hip():
            # auto 会自然落到 AITER；显式填写其他后端属于配置错误。
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            # ROCm 上选择唯一已实现的 AITER 路径。
            return DSAPagedMQALogitsBackend.AITER

        # CUDA 的 auto 与显式 deepgemm 都走成熟的默认实现。
        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        # AITER 是 ROCm 算子，不能在 CUDA 设备上使用。
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        # CuteDSL 目前只为 Blackwell SM100 提供了对应 kernel。
        if value == "cutedsl":
            # 启动阶段尽早拒绝旧架构 GPU，避免延迟到首次请求才报错。
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            # 硬件能力满足后才允许启用 CuteDSL。
            return DSAPagedMQALogitsBackend.CUTEDSL
        # 最后统一处理拼写错误或尚未实现的后端名称。
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
