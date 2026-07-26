import logging
from typing import Optional

import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

# 此文件负责 KV 缓存数据类型的配置与解析。
# 根据用户指定的 --kv-cache-dtype 参数（如 auto、fp8_e4m3、bf16、nvfp4 等），
# 结合模型量化配置和硬件平台（NVIDIA/AMD），确定实际使用的 KV 缓存 torch.dtype。
# 同时处理推测解码（Eagle/DFLASH）场景下的特殊 dtype 兼容性问题。

# PyTorch dtype 到 KV 缓存字符串标识的映射表，用于日志和配置序列化。
TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def configure_kv_cache_dtype(
    *,
    server_args_kv_cache_dtype: str,
    model: nn.Module,
    model_dtype: torch.dtype,
    is_draft_worker: bool,
    is_dflash: bool,
    speculative_draft_attention_backend: str,
) -> tuple[Optional[str], torch.dtype]:
    # 核心配置函数：根据服务器参数和模型信息确定 KV 缓存的实际数据类型。
    # "auto" 模式下自动检测模型量化配置（如 FP8 量化）来选择 dtype；
    # 其他模式直接映射到对应的 torch dtype。
    # 返回值为 (解析后的 dtype 字符串, torch.dtype)，字符串用于指标报告。
    resolved_kv_cache_dtype: Optional[str] = None
    if server_args_kv_cache_dtype == "auto":
        # "auto" 模式：从模型的量化配置中自动推断 KV 缓存 dtype
        quant_config = getattr(model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            kv_cache_dtype = fp8_dtype if _is_hip else torch.float8_e4m3fn
            resolved_kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[kv_cache_dtype]
        else:
            kv_cache_dtype = model_dtype
    elif server_args_kv_cache_dtype == "fp8_e5m2":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e5m2
    elif server_args_kv_cache_dtype == "fp8_e4m3":
        if _is_hip:  # Using natively supported format
            kv_cache_dtype = fp8_dtype
        else:
            kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype == "mxfp8":
        kv_cache_dtype = torch.float8_e4m3fn
    elif server_args_kv_cache_dtype in ("bf16", "bfloat16"):
        kv_cache_dtype = torch.bfloat16
    elif server_args_kv_cache_dtype == "fp4_e2m1":
        raise ValueError(
            "--kv-cache-dtype=fp4_e2m1 is deprecated. "
            "Use --kv-cache-dtype=fp4_mx_block16."
        )
    elif server_args_kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
        if hasattr(torch, "float4_e2m1fn_x2"):
            kv_cache_dtype = torch.float4_e2m1fn_x2
            logger.warning(
                "%s KV Cache might lead to an accuracy drop!",
                server_args_kv_cache_dtype.upper(),
            )
        else:
            raise ValueError(
                f"--kv-cache-dtype={server_args_kv_cache_dtype} requires "
                "torch.float4_e2m1fn_x2 support. Please use PyTorch 2.8.0+ "
                "with CUDA 12.8+."
            )
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {server_args_kv_cache_dtype}.")

    # DFLASH 推测解码兼容性处理：fa4 draft 注意力无法读取 target 的 fp8 KV（要求 K.dtype == Q.dtype），
    # 因此将 draft worker 的 KV 缓存 dtype 回退到模型计算 dtype（通常为 bf16/fp16）。
    if (
        is_draft_worker
        and is_dflash
        and speculative_draft_attention_backend == "fa4"
        and kv_cache_dtype != model_dtype
    ):
        logger.info(
            "DFLASH fa4 draft: overriding KV cache dtype %s -> %s "
            "(fa4 needs K.dtype == Q.dtype; cannot read the target's quantized KV).",
            kv_cache_dtype,
            model_dtype,
        )
        kv_cache_dtype = model_dtype

    return resolved_kv_cache_dtype, kv_cache_dtype
