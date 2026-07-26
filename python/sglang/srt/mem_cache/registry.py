"""Registry for pluggable RadixCache factories.

If `--radix-cache-backend` is unset (by default), the built-in selection
chain is used to pick a cache implementation.

To plug in a custom backend, register it under a string name via
`register_radix_cache_backend(name, factory)`, then select it with
`--radix-cache-backend <name>` (the flag accepts only registered names).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.utils.tensor_bridge import use_mlx

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# 本文件实现了 Radix Cache 的可插拔工厂注册机制。
# 通过 register_radix_cache_backend 注册自定义缓存后端，
# 通过 --radix-cache-backend 命令行参数选择后端。
# 未指定后端时，default_radix_cache_factory 按优先级链自动选择
# 最适合当前模型配置的缓存实现。


@dataclass
class TreeCacheBuildContext:
    """Radix Cache 构建上下文，封装了创建缓存所需的所有参数。"""
    """Radix Cache construction arguments."""

    server_args: ServerArgs
    params: CacheInitParams
    is_hybrid_swa: bool
    is_hybrid_ssm: bool
    enable_hierarchical_cache: bool
    disable_radix_cache: bool
    effective_chunked_prefill_size: Optional[int]
    tp_worker: Any
    model_config: ModelConfig
    tp_size: int
    tp_rank: int
    tp_group: Any
    full_tokens_per_layer: Optional[int] = None
    is_dsa: bool = False


RadixCacheFactory = Callable[[TreeCacheBuildContext], BasePrefixCache]

# 全局注册表，存储名称到工厂函数的映射
_RADIX_CACHE_REGISTRY: dict[str, RadixCacheFactory] = {}


def register_radix_cache_backend(name: str, factory: RadixCacheFactory) -> None:
    """注册一个 Radix Cache 后端工厂函数，名称不可为空或重复。"""
    """Register a radix-cache factory under `name`.

    Raises ValueError if `name` is empty/whitespace-only or already
    registered.
    """
    if not name.strip():
        raise ValueError(
            f"register_radix_cache_backend: name must be non-empty, got {name!r}"
        )
    if name in _RADIX_CACHE_REGISTRY:
        raise ValueError(
            f"register_radix_cache_backend: {name!r} is already registered"
        )
    _RADIX_CACHE_REGISTRY[name] = factory


def get_radix_cache_factory(name: str) -> Optional[RadixCacheFactory]:
    """根据名称获取已注册的缓存工厂函数，未注册则返回 None。"""
    return _RADIX_CACHE_REGISTRY.get(name)


def registered_radix_cache_backends() -> list[str]:
    return list(_RADIX_CACHE_REGISTRY.keys())


def default_radix_cache_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Built-in Radix Cache selection chain."""
    # 内置的缓存选择链，按优先级依次判断当前配置，返回最合适的缓存实现
    server_args = ctx.server_args
    params = ctx.params

    # 禁用 Radix Cache 时使用 ChunkCache 系列
    if ctx.effective_chunked_prefill_size is not None and ctx.disable_radix_cache:
        if not ctx.is_hybrid_swa:
            from sglang.srt.mem_cache.chunk_cache import ChunkCache

            return ChunkCache(params)
        if ctx.full_tokens_per_layer == 0:
            from sglang.srt.mem_cache.chunk_cache import PureSWAChunkCache

            return PureSWAChunkCache(params)
        from sglang.srt.mem_cache.chunk_cache import SWAChunkCache

        return SWAChunkCache(params)

    if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
        # lazy import to avoid JIT overhead
        from sglang.srt.mem_cache.radix_cache_cpp import RadixCacheCpp

        logger.info("Using experimental C++ radix tree implementation.")
        return RadixCacheCpp(params=params, server_args=server_args)

    if envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx():
        return _create_unified_radix_cache(ctx, server_args, params)

    if ctx.is_hybrid_swa:
        if ctx.full_tokens_per_layer == 0:
            from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache

            return PureSWARadixCache(params=params)
        return _create_unified_radix_cache(ctx, server_args, params)

    if ctx.is_hybrid_ssm:
        return _create_unified_radix_cache(ctx, server_args, params)

    if ctx.enable_hierarchical_cache:
        if ctx.is_hybrid_ssm or ctx.is_hybrid_swa or ctx.is_dsa:
            # HybridModel and DSA (e.g. DeepSeek V3.2 / GLM-5.1) launch
            # HiCache via UnifiedRadixCache by default.
            return _create_unified_radix_cache(ctx, server_args, params)
        else:
            from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

            cache = HiRadixCache(params=params, server_args=server_args)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
        return cache

    if server_args.enable_lmcache:
        from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
            LMCRadixCache,
        )

        return LMCRadixCache(
            params=params,
            model_config=ctx.model_config,
            tp_size=ctx.tp_size,
            rank=ctx.tp_rank,
            tp_group=ctx.tp_group,
        )

    if server_args.enable_flexkv:
        # Importing the package side-effect registers the explicit
        # ``--radix-cache-backend=flexkv`` factory; we then call the
        # factory directly so --enable-flexkv stands on its own.
        import os

        from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory

        # Honor a CLI --flexkv-config-file by forwarding it via the env
        # var that FlexKV's config loader actually reads.
        if server_args.flexkv_config_file and not os.environ.get("FLEXKV_CONFIG_PATH"):
            os.environ["FLEXKV_CONFIG_PATH"] = server_args.flexkv_config_file
        return _flexkv_factory(ctx)

    from sglang.srt.mem_cache.radix_cache import RadixCache

    return RadixCache(params)


def _create_unified_radix_cache(
    ctx: TreeCacheBuildContext,
    server_args: ServerArgs,
    params: CacheInitParams,
) -> BasePrefixCache:
    """Initialize a UnifiedRadixCache with proper components and optional HiCache."""
    from sglang.srt.mem_cache.unified_cache_components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree_components = [ComponentType.FULL]
    if ctx.is_hybrid_swa:
        tree_components.append(ComponentType.SWA)
    if ctx.is_hybrid_ssm:
        tree_components.append(ComponentType.MAMBA)

    params.tree_components = tuple(tree_components)
    if use_mlx() and ctx.is_hybrid_ssm:
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
            MlxAuxiliaryStateComponent,
        )

        params.component_registry_override = {
            ComponentType.MAMBA: MlxAuxiliaryStateComponent,
        }
    cache = UnifiedRadixCache(params)
    if ctx.enable_hierarchical_cache:
        cache.init_hicache(server_args, params)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
    return cache


def create_tree_cache(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Route to the matching factory to construct Radix Cache."""
    # 创建缓存的顶层入口：优先使用用户指定的后端，否则走默认选择链
    name = ctx.server_args.radix_cache_backend
    if name:
        factory = get_radix_cache_factory(name)
        if factory is None:
            raise ValueError(
                f"--radix-cache-backend={name!r} is not registered. "
                f"Registered backends: {registered_radix_cache_backends()}. "
                "External backends must call register_radix_cache_backend(...) at import time."
            )
        cache = factory(ctx)
        source = f"registered({name!r})"
    else:
        cache = default_radix_cache_factory(ctx)
        source = "default"

    streaming_wrapped = False
    if (
        ctx.server_args.enable_streaming_session
        and not cache.supports_streaming_session()
    ):
        from sglang.srt.session.streaming_session import StreamingSession

        cache = StreamingSession(cache)
        streaming_wrapped = True

    logger.info(
        "Tree cache initialized: source=%s impl=%s hybrid_swa=%s hybrid_ssm=%s "
        "hierarchical=%s streaming_wrapped=%s",
        source,
        type(cache).__name__,
        ctx.is_hybrid_swa,
        ctx.is_hybrid_ssm,
        ctx.enable_hierarchical_cache,
        streaming_wrapped,
    )
    return cache
