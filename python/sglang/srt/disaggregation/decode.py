"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributed import ProcessGroup

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVReceiver
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    _is_fake_transfer,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce,
    poll_and_all_reduce_pp,
    poll_and_all_reduce_with_staging,
    prepare_abort,
    setup_state_kv_args,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    page_align_floor,
    release_kv_cache,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    KVCache,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.utils import get_num_new_pages, is_npu
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

CLIP_MAX_NEW_TOKEN = envs.SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION.get()


def _bootstrap_addr(req: Req) -> str:
    # FIXME: make a property of a req
    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()


class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
        self.max_context_len = max_context_len
        self.device = device
        self.pre_alloc_size = pre_alloc_size
        with memory_saver_adapter.region(tag=GPU_MEMORY_TYPE_KV_CACHE):
            self.req_to_token = torch.zeros(
                (self._alloc_size, max_context_len),
                dtype=torch.int32,
                device=device,
            )

        self.free_slots = list(range(1, self._alloc_size))
        # Slot-reuse generation counter; mirrors ReqToTokenPool. Required even
        # here: HybridMambaDecodeReqToTokenPool borrows this __init__ while
        # inheriting ReqToTokenPool.alloc, which bumps it.
        self.req_generation = torch.zeros(self._alloc_size, dtype=torch.int64)

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: List[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        assert (
            len(reusing) <= 1
        ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                self.req_generation[r.req_pool_idx] += 1
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.req_generation.zero_()


class HybridMambaDecodeReqToTokenPool(HybridReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: List[int],
        speculative_num_draft_tokens: int,
        enable_mamba_extra_buffer: bool,
        pre_alloc_size: int,
        enable_overlap_schedule: bool,
        mamba_size: int = None,
        start_layer: int = None,
        speculative_eagle_topk: Optional[int] = None,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
        enable_gdn_replayssm_spec: bool = False,
    ):
        DecodeReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            pre_alloc_size=pre_alloc_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_memory_saver = enable_memory_saver
        # Each request needs 1 main mamba slot + ping-pong slots when extra_buffer is enabled.
        # Cap the pool at max concurrent requests * slots_per_req to avoid allocating failed.
        slots_per_req = 1 + (
            self.mamba_ping_pong_track_buffer_size if enable_mamba_extra_buffer else 0
        )
        max_slots_needed = (size + pre_alloc_size) * slots_per_req
        if mamba_size is not None:
            effective_mamba_size = max(mamba_size, max_slots_needed)
            if mamba_size < max_slots_needed:
                logger.warning(
                    "mamba_size (%d) is less than decode side's max_slots_needed (%d = %d reqs * %d slots/req), "
                    "raising effective_mamba_size to %d",
                    mamba_size,
                    max_slots_needed,
                    size + pre_alloc_size,
                    slots_per_req,
                    effective_mamba_size,
                )
        else:
            effective_mamba_size = max_slots_needed
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self._init_mamba_pool(
            mamba_size=effective_mamba_size,
            mamba_spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=self.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
            enable_gdn_replayssm_spec=enable_gdn_replayssm_spec,
        )

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))
        self.mamba_allocator.clear()


@dataclass
class DecodeRequest:
    req: Req
    kv_receiver: CommonKVReceiver
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1
    is_rebootstrap: bool = False

    # HiCache Status
    prefix_match: Optional[DecodePrefixMatch] = None
    hicache_restored_kv_indices: Optional[torch.Tensor] = None
    hicache_restored_node: Any = None
    hicache_load_consumer_index: int = -1
    hicache_restore_status: HiCacheRestoreResult = HiCacheRestoreResult.PENDING

    @property
    def seqlen(self) -> int:
        return self.req.seqlen

    @property
    def priority(self) -> Optional[int]:
        return self.req.priority


class DecodePreallocQueue(DecodeHiCachePreallocMixin):
    """
    Store the requests that are preallocating.
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        transfer_queue: DecodeTransferQueue,
        tree_cache: BasePrefixCache,
        gloo_group: ProcessGroup,
        tp_rank: int,
        tp_size: int,
        dp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        max_total_num_tokens: int,
        pp_rank: int,
        num_reserved_decode_tokens: int,
        transfer_backend: TransferBackend,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(self.token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.scheduler = scheduler
        self.transfer_queue = transfer_queue
        self.tree_cache = tree_cache
        self.gloo_group = gloo_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.max_total_num_tokens = max_total_num_tokens
        self.pp_rank = pp_rank
        self.pp_size = scheduler.ps.pp_size
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        self.transfer_backend = transfer_backend
        # Queue for requests pending pre-allocation
        self.queue: List[DecodeRequest] = []
        self.retracted_queue: List[Req] = []
        self.pending_reqs: List[DecodeRequest] = []
        self._ensure_retry_count: Dict[str, int] = {}
        self._max_ensure_retries: int = 15  # scheduling cycles
        self._ensure_last_attempt_time: Dict[str, float] = {}
        self._ensure_retry_interval: float = 1.0  # seconds
        # Retracted requests staged for rebootstrap while generation is paused.
        # Enqueued into ``self.queue`` only on ``continue_generation`` so the
        # prefix KV is recomputed under the post-retract (updated) weights.
        # NOTE: requests held here are not reachable by ``/abort_request``; to
        # support aborting them we would need an additional fix in the
        # scheduler. In practice this shouldn't arise in the RL scenario.
        self.held_rebootstrap_reqs: List[Req] = []
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        if self.enable_staging and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()
        if self.enable_staging:
            self.transfer_queue._init_staging_handler(self.kv_manager)

        if (
            self.scheduler.tp_worker.is_hybrid_swa
            and not self._uses_swa_tail_prealloc()
        ):
            # Fallback for SWA allocators that still allocate the SWA pool at
            # full prompt length.
            self.max_total_num_tokens = min(
                self.max_total_num_tokens,
                self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
            )

    def _uses_swa_tail_prealloc(self) -> bool:
        return (
            isinstance(self.token_to_kv_pool, (SWAKVPool, DeepSeekV4TokenToKVPool))
            and self.token_to_kv_pool_allocator.page_size > 1
            and hasattr(self.token_to_kv_pool_allocator, "alloc_extend_swa_tail")
        )

    def _swa_tail_len(self, seq_len: int) -> int:
        if not self._uses_swa_tail_prealloc() or seq_len <= 0:
            return max(seq_len, 0)

        window_size = self.scheduler.sliding_window_size
        if window_size is None or window_size <= 0:
            return seq_len

        page_size = self.token_to_kv_pool_allocator.page_size
        window_start = max(0, seq_len - window_size)
        window_start = (window_start // page_size) * page_size
        return seq_len - window_start

    def _swa_retractable_len(self, req: Req) -> int:
        if not self._uses_swa_tail_prealloc():
            return len(req.origin_input_ids) + len(req.output_ids)
        return self._swa_tail_len(len(req.origin_input_ids)) + len(req.output_ids)

    def _prealloc_kv_lens(self, req: Req) -> Tuple[int, int]:
        allocated_kv_len = self._pre_alloc_fill_len(req)
        if self._uses_swa_tail_prealloc():
            return allocated_kv_len, self._swa_tail_len(allocated_kv_len)
        return allocated_kv_len, allocated_kv_len

    def _prealloc_required_tokens(self, req: Req) -> Tuple[int, int]:
        full_len, swa_len = self._prealloc_kv_lens(req)
        swa_reserved = self.num_reserved_decode_tokens
        if self.scheduler.server_args.disable_radix_cache:
            swa_reserved = 0
        return (
            full_len + self.num_reserved_decode_tokens,
            swa_len + swa_reserved,
        )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()

        attn_tp_size = get_parallel().attn_tp_size
        kv_args.engine_rank = self.tp_rank % (attn_tp_size)

        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        kv_args.kv_cache_dtype_str = (
            self.scheduler.tp_worker.model_runner.kv_cache_dtype_str
        )
        transfer_kv_pool = (
            self.scheduler.hisparse_coordinator.mem_pool_host
            if self.scheduler.enable_hisparse
            else self.token_to_kv_pool
        )
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            transfer_kv_pool.get_contiguous_buf_infos()
        )
        kv_data_mem_kinds = (
            ["DRAM"] * len(kv_data_ptrs)
            if self.scheduler.enable_hisparse
            else ["VRAM"] * len(kv_data_ptrs)
        )
        if self.scheduler.enable_hisparse and isinstance(
            self.token_to_kv_pool, DeepSeekV4TokenToKVPool
        ):
            device_kv_data_ptrs, device_kv_data_lens, device_kv_item_lens = (
                self.token_to_kv_pool.get_contiguous_buf_infos()
            )
            c4_layer_num = self.scheduler.hisparse_coordinator.mem_pool_host.layer_num
            kv_data_ptrs += device_kv_data_ptrs[c4_layer_num:]
            kv_data_lens += device_kv_data_lens[c4_layer_num:]
            kv_item_lens += device_kv_item_lens[c4_layer_num:]
            kv_data_mem_kinds += ["VRAM"] * len(device_kv_data_ptrs[c4_layer_num:])
        if self.draft_token_to_kv_pool is not None:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = (
                self.draft_token_to_kv_pool.get_contiguous_buf_infos()
            )
            kv_data_ptrs += draft_kv_data_ptrs
            kv_data_lens += draft_kv_data_lens
            kv_item_lens += draft_kv_item_lens
            kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_layer_ids = (
            self.token_to_kv_pool.get_kv_layer_ids()
            if self.draft_token_to_kv_pool is None
            and hasattr(self.token_to_kv_pool, "get_kv_layer_ids")
            else []
        )
        if self.transfer_backend == TransferBackend.NIXL:
            kv_args.kv_data_mem_kinds = kv_data_mem_kinds
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )

        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool,
            total_kv_layers=self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=getattr(self, "req_to_token_pool", None),
        )

        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id
        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.DECODE,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        # Staging buffer setup (only when heterogeneous TP staging is enabled)
        if self.enable_staging and not self.is_mla_backend:
            kv_pool_for_heads = self.token_to_kv_pool
            if hasattr(kv_pool_for_heads, "full_kv_pool"):
                kv_pool_for_heads = kv_pool_for_heads.full_kv_pool
            per_rank_kv_heads = getattr(kv_pool_for_heads, "head_num", 0)
            if per_rank_kv_heads > 0:
                kv_args.kv_head_num = per_rank_kv_heads
                kv_args.total_kv_head_num = per_rank_kv_heads * attn_tp_size
            if hasattr(kv_manager, "set_kv_buffer_tensors"):
                kv_pool = kv_pool_for_heads
                if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                    kv_manager.set_kv_buffer_tensors(
                        kv_pool.k_buffer, kv_pool.v_buffer, kv_pool.page_size
                    )
        return kv_manager

    def add(
        self, req: Req, is_retracted: bool = False, is_rebootstrap: bool = False
    ) -> None:
        """Add a request to the pending queue.

        ``is_rebootstrap`` marks a PD true-retraction request whose prefix KV
        must be recomputed by the original prefill worker under the current
        weights (rather than resumed from stale CPU KV). It otherwise follows the
        same bootstrap-handshake path as a fresh request; the ``/generate``
        dispatch happens later, after preallocation and ``send_metadata`` (see
        ``pop_preallocated``).
        """
        if self._check_if_req_exceed_kv_capacity(req):
            return

        if is_retracted:
            req.retraction_mb_id = None
            self.retracted_queue.append(req)
        else:
            decode_req = self._create_receiver_and_enqueue(
                req, is_rebootstrap=is_rebootstrap
            )

            # NOTE: fake transfer does not need to resolve prefill dp rank in the pending queue
            if _is_fake_transfer(req, self.scheduler.server_args):
                decode_req.kv_receiver.init(0)
                return

            # Fast path: cache-only lookup, no network calls
            prefill_dp_rank = self._resolve_prefill_dp_rank(req)
            logger.debug(f"prefill_dp_rank: {prefill_dp_rank}")
            if prefill_dp_rank is not None:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return

            self.pending_reqs.append(decode_req)

    def _match_prefix_and_lock(self, req: Req) -> DecodePrefixMatch:
        """
        Match a request against the decode-side radix cache, lock the matched
        node to prevent eviction, and return the matched prefix information.
        """
        result = match_prefix_for_req(
            self.tree_cache,
            req,
            req.origin_input_ids,
            cow_mamba=self.tree_cache.supports_mamba(),
            include_req=True,
        )
        # Always lock to match aggregated scheduling behavior
        self.tree_cache.inc_lock_ref(result.last_device_node)
        return self._build_decode_prefix_match(req, result)

    def _resolve_prefill_dp_rank(self, req: Req) -> Optional[int]:
        prefill_info = self.kv_manager.prefill_info_table.get(_bootstrap_addr(req))
        # If None, it will go to the slow path and resolve prefill_info by _ensure_prefill_info then cache it
        if prefill_info is None:
            return None

        if req.disagg_prefill_dp_rank is not None:
            return req.disagg_prefill_dp_rank

        if prefill_info.dp_size == 1:
            return 0

        if (
            prefill_info.follow_bootstrap_room
            and not envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.get()
        ):
            return req.bootstrap_room % prefill_info.dp_size

        return None

    def _create_receiver_and_enqueue(
        self, req: Req, is_rebootstrap: bool = False
    ) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(
            req=req, kv_receiver=kv_receiver, is_rebootstrap=is_rebootstrap
        )
        self.queue.append(decode_req)
        return decode_req

    def hold_rebootstrap(self, req: Req) -> None:
        """Stage a retracted request for rebootstrap without enqueuing it yet.

        Retraction is always paired with a weight update
        (``pause_generation(mode="retract")`` -> ``update_weights`` ->
        ``continue_generation``). Enqueuing the rebootstrap into ``self.queue``
        here would leave the preallocation queue non-empty, which makes the
        scheduler non-idle so ``update_weights``' post-update cache flush
        asserts and crashes the decode worker. Instead we hold the request and
        enqueue it from ``enqueue_held_rebootstrap`` on resume, so its prefix KV
        is recomputed by the prefill worker under the updated weights.
        """
        self.held_rebootstrap_reqs.append(req)

    def enqueue_held_rebootstrap(self) -> None:
        """Enqueue all staged rebootstrap requests when generation resumes."""
        held = self.held_rebootstrap_reqs
        self.held_rebootstrap_reqs = []
        for req in held:
            self.add(req, is_rebootstrap=True)

    @staticmethod
    def _rebootstrap_prefill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids)

    @staticmethod
    def _pre_alloc_fill_len(req: Req) -> int:
        if getattr(req, "pd_rebootstrap_in_progress", False):
            # pause_generation(retract) already popped the boundary token out of
            # output_ids (it is replayed via the decode-side override at commit
            # time), so output_ids here is prompt + emitted-tokens-minus-boundary,
            # i.e. the original seqlen - 1. The prefill recomputes KV for *all* of
            # these tokens, leaving no just-sampled "pending" token in the list, so
            # we allocate exactly len(origin)+len(output_ids) with no -1 (unlike
            # normal decode, where the last token's KV has not been written yet).
            # This is the same token count as offloading-based retraction, where
            # offload_kv_cache saves seqlen-1 tokens; the boundary token's KV is
            # (re)computed on the decode side once generation resumes.
            return len(req.origin_input_ids) + len(req.output_ids)
        return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        input_len = self._rebootstrap_prefill_len(req)
        if input_len > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {input_len} > {self.max_total_num_tokens}"
            logger.error(message)
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        if self._uses_swa_tail_prealloc():
            _, swa_required = self._prealloc_required_tokens(req)
            swa_capacity = self.token_to_kv_pool_allocator.size_swa
            if swa_required > swa_capacity:
                message = (
                    f"Request {req.rid} requires too many SWA KV tokens for "
                    f"decode preallocation: {swa_required} > {swa_capacity}"
                )
                logger.error(message)
                prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
                self.scheduler.output_streamer.stream_output([req], req.return_logprob)
                return True
        return False

    def extend(self, reqs: List[Req], is_retracted: bool = False) -> None:
        """Add a request to the pending queue."""
        for req in reqs:
            self.add(req, is_retracted=is_retracted)

    def release_memory_occupation(self):
        self.queue.clear()
        self.retracted_queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()

    def resume_retracted_reqs(
        self, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        # TODO refactor the scheduling part, reuse with the unified engine logic as much as possible

        # allocate memory
        resumed_reqs = []
        indices_to_remove = set()
        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        if uses_swa_tail_prealloc:
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(count_retracted=False)
            )
        else:
            full_allocatable_tokens = self._allocatable_token_budgets(
                count_retracted=False
            )

        for i, req in enumerate(self.retracted_queue):
            if rids_to_check is not None and req.rid not in rids_to_check:
                continue

            if self.req_to_token_pool.available_size() <= 0:
                break

            full_required, swa_required = self._prealloc_required_tokens(req)
            if full_required > full_allocatable_tokens:
                break
            if uses_swa_tail_prealloc and swa_required > swa_allocatable_tokens:
                break

            resumed_reqs.append(req)
            indices_to_remove.add(i)
            req.is_retracted = False
            self._pre_alloc(req)
            full_allocatable_tokens -= full_required
            if uses_swa_tail_prealloc:
                swa_allocatable_tokens -= swa_required

            # load from cpu, release the cpu copy
            req.load_kv_cache(self.req_to_token_pool, self.token_to_kv_pool_allocator)

        self.retracted_queue = [
            entry
            for i, entry in enumerate(self.retracted_queue)
            if i not in indices_to_remove
        ]

        return resumed_reqs

    def _update_handshake_waiters(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> None:
        if not self.queue:
            return

        # Still poll if any receiver was aborted, otherwise it stays stuck.
        if (
            self.pp_size <= 1
            and all(decode_req.waiting_for_input for decode_req in self.queue)
            and not any(
                decode_req.kv_receiver.conclude_state == KVPoll.Failed
                for decode_req in self.queue
            )
        ):
            return

        if self.pp_size > 1:
            polls = poll_and_all_reduce_pp(
                (decode_req.req.rid for decode_req in self.queue),
                KVPoll.WaitingForInput,
                pp_good_rids,
                pp_bad_rids,
            )
        else:
            polls = poll_and_all_reduce(
                [decode_req.kv_receiver for decode_req in self.queue], self.gloo_group
            )

        for decode_req, poll in zip(self.queue, polls):
            if poll is None:
                continue
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            if poll == KVPoll.Bootstrapping:
                pass
            elif poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
                decode_req.req.time_stats.set_bootstrap_done_time()
            elif poll == KVPoll.Failed:
                error_message = f"Decode handshake failed for request rank={self.tp_rank} {decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                is_propagated = False
                try:
                    decode_req.kv_receiver.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

    def _ensure_prefill_info(
        self, addr_to_reqs: Dict[str, List[DecodeRequest]]
    ) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
        """Non-blocking ensure parallel info for each addr.
        Returns (ready_addrs, remaining_reqs)."""
        ready: Dict[str, List[DecodeRequest]] = {}
        remaining: List[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and (
                now - last_attempt < self._ensure_retry_interval
            ):
                remaining.extend(reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now

            if self.kv_manager.try_ensure_parallel_info(bootstrap_addr):
                if bootstrap_addr in self._ensure_retry_count:
                    del self._ensure_retry_count[bootstrap_addr]
                if bootstrap_addr in self._ensure_last_attempt_time:
                    del self._ensure_last_attempt_time[bootstrap_addr]
                ready[bootstrap_addr] = reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count

            if count >= self._max_ensure_retries:
                error_msg = f"Could not fetch prefill parallel info from {bootstrap_addr} after {count} attempts"
                logger.error(error_msg)
                for decode_req in reqs:
                    # kv_receiver may be None from a prior self.queue cleanup
                    if decode_req.kv_receiver is not None:
                        decode_req.kv_receiver.abort()
                del self._ensure_retry_count[bootstrap_addr]
                del self._ensure_last_attempt_time[bootstrap_addr]
            else:
                remaining.extend(reqs)

        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        """Batch-resolve prefill_dp_ranks for pending requests and initialize receivers."""
        if not self.pending_reqs:
            return

        # Group pending requests by bootstrap_addr
        addr_to_reqs: Dict[str, List[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr = _bootstrap_addr(decode_req.req)
            addr_to_reqs.setdefault(addr, []).append(decode_req)

        # Pass 1: ensure parallel info for each addr
        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)

        resolved: List[Tuple[DecodeRequest, int]] = []
        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: List[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            # Pass 2: resolve dp rank for addrs whose info is available
            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = CommonKVReceiver.query_prefill_dp_ranks(
                    bootstrap_addr, rooms
                )
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(
                        str(decode_req.req.bootstrap_room)
                    )
                    if prefill_dp_rank is not None:
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        remaining.append(decode_req)

        self.pending_reqs = remaining

        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def pop_preallocated(
        self,
        rids_to_check: Optional[List[str]] = None,
        pp_good_rids: Optional[List[str]] = None,
        pp_bad_rids: Optional[List[str]] = None,
    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:
        """从预分配队列中挑选本轮可以进入 KV 传输阶段的请求。

        这个函数位于 Decode 侧 PD 请求生命周期的关键边界：请求已经建立了
        bootstrap/handshake，但 Prefill 还不知道应该把 KV 写到 Decode 的哪些页。
        因此这里需要先完成资源准入和目标页预分配，再通过 ``send_metadata`` 把
        目标页号、状态张量页号以及 Decode 侧前缀命中长度发给 Prefill。

        函数按如下顺序工作：

        1. 汇总 TP/PP rank 上的握手结果，并清理已经失败或被取消的请求；
        2. 计算 full-KV、SWA、HiCache 恢复区和 HiSparse buffer 的可用预算；
        3. 按队列顺序（可选优先级顺序）检查请求是否能够安全准入；
        4. 预分配 req slot、KV 页和 metadata slot，并启动 Decode HiCache 恢复；
        5. 把接收地址元数据发给 Prefill，随后把请求交给 transfer queue。

        ``preallocated_reqs`` 表示已完成预分配、可以等待 Prefill 写入 KV 的请求；
        ``failed_reqs`` 表示本轮发现握手失败或已 abort、需要上层回收的请求。
        未满足握手或资源条件的请求继续留在 ``self.queue``，不会被丢失。
        """
        is_pp_mode = self.pp_size > 1
        # PP>1 时每个 pipeline rank 都会观察同一请求；必须使用跨 rank 共识，
        # 否则某个 rank 单独准入会让各 stage 的请求队列和 KV 地址发生错位。
        if is_pp_mode and (pp_good_rids is None or pp_bad_rids is None):
            # good/bad 两个集合共同描述 PP 共识结果，缺少任一集合都无法安全决策。
            raise ValueError("PP consensus is required when pp_size > 1")
            # 直接报错比让不同 PP rank 继续执行更安全，因为后者可能造成通信死锁。
        if is_pp_mode and rids_to_check is not None:
            # PP 模式必须以共识集合为唯一过滤来源，不能再叠加本 rank 的局部 rid 集合。
            raise ValueError("rids_to_check cannot be used in PP mode")
            # 禁止混用两种过滤语义，避免各 PP rank 遍历不同的请求子集。

        self._resolve_pending_reqs()
        # 某些 receiver 尚不知道 Prefill DP rank；这里批量解析并初始化其连接端点。
        self._update_handshake_waiters(rids_to_check, pp_good_rids, pp_bad_rids)
        # 轮询 receiver/PP 共识，把握手完成、失败等状态写回 DecodeRequest。
        if is_pp_mode:
            # 在 PP 模式下，后续只处理已进入本轮共识结果的 rid。
            rids_to_check = set(pp_good_rids) | set(pp_bad_rids)
            # good 与 bad 都要扫描：good 参与预分配，bad 则需要在失败清理阶段移除。

        failed_reqs = []
        # 单独返回失败请求，让 scheduler 执行生命周期收尾而不是静默丢弃。
        preallocated_reqs = []
        # 这里收集已经拿到 Decode 端资源且 metadata 已发送成功的请求。
        indices_to_remove = set()
        # 先记录索引、最后一次性重建队列，避免遍历过程中删除元素导致索引漂移。

        # We need to make sure that the sum of inflight tokens and allocatable tokens is greater than maximum input+output length of each inflight request
        # Otherwise it is possible for one request running decode out of memory, while all other requests are in the transfer queue that cannot be retracted.
        retractable_tokens = sum(
            len(r.origin_input_ids) + len(r.output_ids)
            for r in self.scheduler.running_batch.reqs
        )
        # running batch 可以通过 retract 释放上述 KV。准入计算把它视为应急容量，
        # 但仍要保证至少一个在途请求能够持续 decode，避免不可回退的 transfer 请求
        # 占满显存并与 running batch 形成 OOM/等待死锁。

        uses_swa_tail_prealloc = self._uses_swa_tail_prealloc()
        # 分页 SWA allocator 只需为滑窗尾部保留 SWA KV，因此 full 与 SWA 要分开预算。
        swa_allocatable_tokens = 0
        # 非 SWA-tail 路径不会读取该预算，先置零可保持后续分支变量总是已定义。
        if uses_swa_tail_prealloc:
            # SWA 池能通过 retract 回收的只是每个请求当前滑窗覆盖的尾部，而非完整序列。
            retractable_swa_tokens = sum(
                self._swa_retractable_len(r) for r in self.scheduler.running_batch.reqs
            )
            # 同时计算 full attention 池与 SWA 池预算；任一池不足都不能准入请求。
            full_allocatable_tokens, swa_allocatable_tokens = (
                self._swa_aware_allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    retractable_swa_tokens=retractable_swa_tokens,
                    count_retracted=True,
                )
            )
            # count_retracted=True 会预留 retracted_queue 将来恢复时所需的 KV 空间。
        else:
            # 普通注意力模型只有一套完整 KV 池，不需要第二套 SWA 物理预算。
            retractable_swa_tokens = 0
            # 保持变量存在，使后面的 SWA 条件块无需额外的 Optional 分支。
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens, count_retracted=True
            )
            # 预算已包含 active 请求的 decode 余量以及可驱逐 radix cache 页。
        reserved_restore_tokens = self._hicache_pending_restore_tokens()
        # 已经启动但尚未完成的 L2/L3 -> L1 恢复也会消耗设备页，必须先行预留。
        full_allocatable_tokens -= reserved_restore_tokens
        # 防止本轮新请求把 HiCache 恢复目标页抢走，造成恢复完成时无处落盘。
        # Sort by priority before any index-based bookkeeping so that both the
        # abort-scan loop and the preallocation loop operate on the same order.
        if self.scheduler.enable_priority_scheduling:
            # priority 数值的高低语义由配置决定，因此用 sign 统一转换为升序 sort key。
            priority_sign = (
                1 if self.scheduler.schedule_low_priority_values_first else -1
            )
            # sign=1 表示较小数值先执行；sign=-1 则把较大数值排到前面。
            self.queue.sort(key=lambda r: r.req.priority * priority_sign)
            # 必须在记录 indices_to_remove 之前排序，保证两次遍历使用完全相同的索引。

        # First, remove all failed requests from the queue
        for i, decode_req in enumerate(self.queue):
            # 第一遍只负责失败清理；成功请求的资源准入留给下面的第二遍。
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                # TP 局部轮询或 PP 共识未覆盖的请求本轮不动，继续等待后续调度周期。
                continue
            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                # 握手失败会由 _update_handshake_waiters 标记 FINISH_ABORT，用户主动
                # abort 也走同一路径，因此这里统一完成输出通知和 receiver 清理。
                if not getattr(decode_req.req, "finished_output", False):
                    # 若终止结果尚未返回客户端，必须先输出一次，避免请求永久悬挂。
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    # return_logprob 保持请求原始输出契约，即使最终结果是错误/取消。
                decode_req.kv_receiver.clear()
                # 清理 bootstrap/transfer backend 状态，释放该请求占用的通信资源。
                decode_req.kv_receiver = None
                # 显式断开引用，防止后续阶段误用一个已经 clear 的 receiver。
                failed_reqs.append(decode_req)
                # 返回给调用方做其余请求级回收和统计。
                indices_to_remove.add(i)
                # 延迟到函数末尾统一从 self.queue 删除，保证当前 enumerate 稳定。

        # DecodeRequest is shared between self.queue and self.pending_reqs;
        # drop failed reqs from both
        if failed_reqs:
            # pending_reqs 与 queue 保存同一 DecodeRequest 对象，而不是对象副本。
            failed_ids = {id(r) for r in failed_reqs}
            # 使用对象 identity 而不是 rid，避免重复 rid 或 Req 自定义相等语义误删。
            self.pending_reqs = [
                r for r in self.pending_reqs if id(r) not in failed_ids
            ]
            # 否则后续 _resolve_pending_reqs 可能重新初始化已经失败的 receiver。

        # HiSparse physical constraint: max requests by device buffer capacity.
        # Each admitted req needs padded_buffer_size from hisparse device pool.
        # waiting_queue reqs already have device buffers (allocated in admit_request_direct),
        # only transfer_queue reqs are pending device buffer allocation.
        hisparse_req_budget = float("inf")
        # 非 HiSparse 路径没有“每请求一个 padded device buffer”的约束，用无穷大
        # 让统一的准入循环无需额外分叉。
        if self.scheduler.enable_hisparse:
            # HiSparse attention 的设备 buffer 与普通 logical KV 页是不同的物理资源。
            hisparse_avail = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            # available_size 按 token/slot 计量，要除以每请求固定的 padded buffer 大小。
            hisparse_req_budget = max(
                0,
                hisparse_avail // self.scheduler.hisparse_coordinator.padded_buffer_size
                - len(self.transfer_queue.queue),
            )
            # transfer_queue 中的请求尚未拿到这块 device buffer，也要提前占用名额；
            # max(0, ...) 避免历史在途量导致负预算并污染后面的递减逻辑。

        # Then, preallocate the remaining requests if possible
        for i, decode_req in enumerate(self.queue):
            # 第二遍严格按 FIFO 或优先级顺序准入；遇到共享资源不足时会 break，
            # 从而避免后来的请求越过队首，破坏调度公平性。
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                # 该请求没有本轮可用的 handshake/PP 共识结果，暂时留在队列。
                continue

            if i in indices_to_remove:
                # 第一遍已判定失败的条目只等待统一删除，不能再进行资源分配。
                continue

            if not decode_req.waiting_for_input:
                # Prefill 尚未进入 WaitingForInput 时不能发送目标页 metadata，
                # 否则 Decode 与 Prefill 的 bootstrap 状态机会发生乱序。
                continue

            if self.req_to_token_pool.available_size() <= 0:
                # 每个准入请求至少需要一个 req slot 保存 token -> KV 页映射。
                break
                # req slot 是全局硬约束，后续请求同样无法绕过，因此直接停止扫描。

            if self.req_to_metadata_buffer_idx_allocator.available_size() <= 0:
                # metadata slot 用于传输 backend 的同步/辅助数据，KV 页充足也不能缺它。
                break
                # allocator 没有空位时所有后续请求都会失败，所以保持 FIFO 并退出。

            if hisparse_req_budget <= 0:
                # HiSparse 的按请求物理 buffer 名额耗尽，不能再接收新的 KV transfer。
                break
                # 该约束与请求长度无关，检查后续短请求也没有意义。

            # Memory estimation: don't add if the projected memory cannot be met
            # TODO: add new_token ratio
            origin_input_len = self._rebootstrap_prefill_len(decode_req.req)
            # 普通请求只传原 prompt；true rebootstrap 还要让 Prefill 重算已有输出 token。
            prefix_match: Optional[DecodePrefixMatch] = None
            # 默认没有 Decode 侧命中；成功匹配后对象还承载锁定节点和 HiCache 恢复信息。
            use_decode_radix_cache = (
                self.scheduler.server_args.disaggregation_decode_enable_radix_cache
                and not decode_req.is_rebootstrap
            )
            # rebootstrap 必须按当前权重重算完整前缀，不能复用可能由旧权重生成的缓存。
            if use_decode_radix_cache:
                # Match prefix against decode's radix cache.
                prefix_match = self._match_prefix_and_lock(decode_req.req)
                # 匹配函数会增加 radix 节点 lock ref，保证准入决策期间命中页不被驱逐。
                prefix_indices = prefix_match.prefix_indices
                # 这些是已经位于 Decode GPU(L1) 的前缀页，可直接写入 req_to_token 映射。
                # prefix_len: tokens already on device (L1 hit).
                # total_prefix_len: full prefix promised to prefill
                # (L1 + L2 host hit + L3 storage hit), sent as PD
                # protocol's `decode_prefix_len`. The [prefix_len, total)
                # gap is filled by HiCache loadback later.
                prefix_len = prefix_match.l1_prefix_len
                # prefix_len 只计算当前已在设备上的连续命中，决定实际还需分配多少 L1 页。
                total_prefix_len = prefix_match.decode_prefix_len
                # total_prefix_len 还包含 L2/L3 命中；Prefill 会跳过这段，Decode 后续自行恢复。

                fill_len = self._pre_alloc_fill_len(decode_req.req)
                # fill_len 是接收 Prefill KV 后应当已经 committed 的 token 数，通常不含
                # 最后一个尚未执行 forward、仅刚采样出的 output token。
                required_alloc_tokens = self._required_alloc_tokens(
                    fill_len=fill_len, prefix_len=prefix_len
                )
                # 按 allocator page size 向上取页，计算 L1 命中之外仍需占用的真实物理量。
                # Matching may lock previously-evictable radix pages, so refresh
                # the admission budget against the post-lock pool state before we
                # decide whether this request still fits.
                full_allocatable_tokens = self._allocatable_token_budgets(
                    retractable_tokens=retractable_tokens,
                    count_retracted=True,
                    extra_reserved_reqs=len(preallocated_reqs),
                    hicache_reserved_tokens=reserved_restore_tokens,
                )
                # radix 匹配刚锁住的页原本可能是 evictable，旧预算已不准确，必须重算。
            else:
                prefix_indices = None
                # 不使用 Decode radix cache 时，req_to_token 前缀没有可复用页号。
                prefix_len = 0
                # L1 命中长度为零，因此整个 fill 区间都要新分配。
                total_prefix_len = 0
                # 也不会向 Prefill 宣称 Decode 能从 L2/L3 恢复任何前缀。
                required_alloc_tokens = self._pre_alloc_fill_len(decode_req.req)
                # 无 prefix 且传统 token allocator 下，所需 token 数就是完整 fill 长度。

            required_tokens_for_request = (
                required_alloc_tokens + self.num_reserved_decode_tokens
            )
            # 除接收 prompt KV 外，还为刚准入请求预留若干 decode token，避免它一进入
            # running batch 就因没有生成空间而立即触发 retraction。

            if (
                max(
                    required_tokens_for_request,
                    origin_input_len
                    - prefix_len
                    + min(
                        decode_req.req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKEN,
                    )
                    - retractable_tokens,
                )
                > full_allocatable_tokens
            ):
                # 第一项保证当前请求能完成预分配；第二项保证最坏情况下仍能通过
                # retract running batch 为其腾出“prompt 剩余 + 最大输出”所需空间。
                # 取 max 是为了同时覆盖立即分配安全和长期 decode 可推进性。
                if prefix_len > 0:
                    # 本请求未获准进入 transfer queue，必须撤销前面 radix match 加的锁。
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    # 不解锁会让命中节点永久不可驱逐，逐轮侵蚀可用 KV 容量。
                break
                # full pool 是队列共享硬约束；为保持顺序，不让后续请求越过当前请求。
            if required_tokens_for_request > full_allocatable_tokens:
                # 这是一个显式的即时容量保护。它与上面的 max 条件看似重复，但保留
                # 独立检查可清晰约束“预分配 + decode reserve”绝不能超过当前预算。
                if prefix_len > 0:
                    # 与所有准入失败路径一样，归还此次 prefix match 持有的 radix 锁。
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    # last_node 是 _match_prefix_and_lock 保存到 Req 上的设备命中节点。
                break
                # 当前请求无法分配时，停止 FIFO 扫描，等待回收或下一调度周期。

            if uses_swa_tail_prealloc:
                # hybrid SWA 模型还要独立验证 SWA 尾部池，full KV 足够并不代表 SWA 足够。
                _, swa_required = self._prealloc_required_tokens(decode_req.req)
                # swa_required 包含当前滑窗尾部和必要的后续 decode reserve。
                _, swa_len = self._prealloc_kv_lens(decode_req.req)
                # swa_len 是本次 prompt/rebootstrap 真正需要落到 SWA 池的有效尾部长度。
                max_new_tokens = min(
                    decode_req.req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKEN,
                )
                # 用全局上限截断用户配置，避免不现实的超大 max_new_tokens 阻塞所有请求。
                if (
                    max(
                        swa_required,
                        swa_len + max_new_tokens - retractable_swa_tokens,
                    )
                    > swa_allocatable_tokens
                ):
                    # 同 full pool：同时检查即时分配量和 retract 后完成生成的最坏需求。
                    if prefix_len > 0:
                        # SWA 预算失败也意味着整个请求未准入，必须释放 radix 节点锁。
                        self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                        # 即便命中页位于 full pool，失败路径仍不能保留其保护引用。
                    break
                    # SWA buffer 是共享硬资源，因此不跳过队首去尝试后续请求。

            if total_prefix_len != 0 and hasattr(
                self.token_to_kv_pool_allocator, "c4_attn_allocator"
            ):
                # DSV4 NPU chunked-prefill/C4 allocator 尚不能表达 Decode 侧前缀复用的
                # 多池映射；若继续，Prefill 跳过的 prefix 将无法正确写回 Decode。
                if prefix_len > 0:
                    # 抛错前撤销 radix match 的锁，避免异常被上层捕获后留下资源泄漏。
                    self.tree_cache.dec_lock_ref(decode_req.req.last_node)
                    # total_prefix_len 可能仅来自 L2/L3，而 prefix_len>0 才实际持有 L1 锁。
                raise RuntimeError(
                    "DSV4 NPU PD disaggregation does not support decode-side "
                    "prefix cache yet; disable disaggregation decode radix/HiCache "
                    "for PD + chunked prefill."
                )
                # 明确拒绝不受支持的组合，比生成错误 KV 或错误 token 更容易定位问题。

            dst_kv_indices = self._pre_alloc(
                decode_req.req,
                prefix_indices,
                prefix_len,
                total_prefix_len,
            )
            # 真正分配 req slot 和 KV 目标页，并把 L1 命中页写进 req_to_token 映射；
            # 返回值在 HiSparse 下还用于构造 direct-to-host 的目标索引。
            decode_req.prefix_match = prefix_match
            # transfer 阶段需要此对象判断 HiCache 恢复是否完成，并在结束时管理锁引用。
            if self.scheduler.enable_decode_hicache:
                # 只有 Decode HiCache 完整开启时才需要把 L2/L3 命中区恢复到 L1。
                self._start_hicache_prefetch(decode_req.req, prefix_match)
                # 恢复与 Prefill 计算/传输并行启动，以隐藏远端或 host cache 读取延迟。
            hisparse_req_budget -= 1
            # 非 HiSparse 时预算为 inf，递减后仍为 inf；HiSparse 时消耗一个请求名额。
            # Recompute from actual pool state for the next queue entry.
            # This accounts for page rounding and newly locked evictable cache.
            if prefix_match is not None:
                # 已启动恢复的 token 未来要占据本轮预分配的目标页，加入全局恢复预留量。
                reserved_restore_tokens += prefix_match.restore_token_count
                # 后续请求预算必须看见该占用，不能把同一恢复空间重复承诺出去。
            full_allocatable_tokens = self._allocatable_token_budgets(
                retractable_tokens=retractable_tokens,
                count_retracted=True,
                extra_reserved_reqs=len(preallocated_reqs) + 1,
                hicache_reserved_tokens=reserved_restore_tokens,
            )
            # 使用 allocator 的实际状态重算而非简单减 token 数，因为分页向上取整、
            # radix 锁变化和 evictable size 都可能让理论差值与真实容量不同。
            if uses_swa_tail_prealloc:
                # SWA budget uses simple decrement (no radix cache eviction in
                # the SWA pool, so page-rounding drift is negligible).
                swa_allocatable_tokens -= swa_required
                # SWA 池没有 radix 驱逐/锁状态变化，可用已计算的物理需求直接递减。
            decode_req.req.cache_protected_len = total_prefix_len
            # 标记这段前缀已由 Decode 命中承诺，后续 cache/release 逻辑不能提前覆盖它。

            page_size = self.token_to_kv_pool_allocator.page_size
            # allocator page size 用于把 token 索引压缩成传输协议中的页索引。
            kv_transfer_page_size = page_size
            # 普通路径传输页大小与设备 allocator 一致；HiSparse 会在下方改成压缩页大小。
            if self.scheduler.enable_hisparse:
                # Direct-to-host sends host/C4 rows; keep allocator.page_size
                # logical and use the compressed page size only for these indices.
                kv_transfer_page_size = getattr(
                    self.token_to_kv_pool_allocator,
                    "hisparse_page_size",
                    page_size,
                )
                # 某些 allocator 暂无专用属性时回退到普通 page_size，保持兼容性。
                kv_indices = dst_kv_indices[: origin_input_len - prefix_len]
                # HiSparse 直接写 host/C4 目标，只发送本次 Prefill 需要填充的有效区间。
            else:
                # Only send delta indices (beyond prefix) to prefill.
                kv_indices = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx
                ][total_prefix_len:origin_input_len]
                # Prefill 已收到 decode_prefix_len，会跳过 Decode 的 L1/L2/L3 命中；
                # 因此这里只回传 [total_prefix_len, prompt_len) 的增量目标页。

            seq_len = origin_input_len
            # 下方各类模型状态 payload 都必须以同一个逻辑序列长度构造，确保 P/D 对齐。

            def _mamba_payload():
                # Mamba state 按 request slot 而非普通 KV page 编址，需要独立映射。
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        decode_req.req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]
                # ZMQ 元数据需要 CPU numpy；外层 list 与 state transfer 接口的分段格式一致。

            def _swa_payload():
                # SWA 只传当前窗口覆盖的页，窗口之前的 KV 已不参与后续 attention。
                window_size = self.scheduler.sliding_window_size
                # 使用 scheduler 的模型级窗口配置，保证与实际 forward attention 一致。
                window_start = max(0, seq_len - window_size)
                # 短序列从 0 开始，长序列只保留最后 window_size 个 token。
                window_start = page_align_floor(window_start, page_size)
                # 向下页对齐以包含跨页窗口边界，避免漏传窗口首 token 所在的整页。
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, window_start:seq_len
                ]
                # req_to_token 保存的是 full-pool 逻辑索引，先截取窗口的连续位置。
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                # hybrid allocator 的 SWA 池有独立物理编号，必须从 full 编号翻译后再发送。
                return kv_to_page_indices(window_kv_indices_swa, page_size)
                # 协议按页传输，压缩为每页一个索引以减少 bootstrap metadata 体积。

            def _dsa_payload():
                # DSA/indexer 状态与主 KV 共用 device page location，但需要单独声明状态类型。
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx, :seq_len
                ]
                # DSA 需要完整逻辑序列的页位置，不能像主 KV 一样只传 prefix 之后的 delta。
                # Indexer lives on device pool; always use device page_size
                device_page_size = self.token_to_kv_pool.page_size
                # HiSparse 主传输可能使用 host 压缩页，indexer 仍驻留设备，页大小不能混用。
                return kv_to_page_indices(kv_indices_full, device_page_size)
                # 结果与 Prefill 侧相同 state type 的 source page 顺序一一对应。

            def _swa_ring_payload():
                # Mirror of prefill _swa_ring_payload using this side's req_pool_idx.
                # Same window positions and order -> positional match with prefill.
                ring_stride = self.token_to_kv_pool.unified_swa_ring_size
                # 每个 request slot 在统一 SWA ring 中占用固定 stride 的行区间。
                window_size = self.token_to_kv_pool.unified_swa_window
                # ring 使用 KV pool 的实际窗口配置，而不是假定与普通 SWA payload 相同。
                window_start = max(0, seq_len - window_size)
                # 只为仍在滑窗中的 token 构造 ring row，旧位置已经可以被循环覆盖。
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                # 显式 int64 防止长序列位置计算时溢出；最终传输前再压成 int32。
                state_slot = int(decode_req.req.req_pool_idx)
                # req_pool_idx 决定该请求在全局 ring buffer 中的独立基址。
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                # 取模实现环形复用，加 slot 基址保证不同请求的 ring 行互不重叠。
                return ring_rows.astype(np.int32)
                # state metadata 使用紧凑 int32，且与接收端索引 dtype 保持一致。

            def _c128_state_payload():
                # DSV4 C128 state 在线与离线模式的 ring 编址不同，需要运行时选择。
                online = is_dsv4_c128_online_enabled()
                # online 模式只维护当前 state，因此逻辑 ring size 固定为 1。
                ring_size = 1 if online else self.token_to_kv_pool.get_ring_size(128)
                # offline 模式从 KV pool 获取真实 C128 ring 大小，避免硬编码模型布局。
                return get_dsv4_c128_state_indices(
                    int(decode_req.req.req_pool_idx),
                    seq_len,
                    online=online,
                    ring_size=ring_size,
                )
                # helper 同时编码 request slot、序列进度和模式，返回 P/D 对齐的 state 行号。

            state_types = self.kv_manager.kv_args.state_types
            # transfer backend 在初始化时根据模型注册需要随主 KV 一起传输的额外状态类型。
            if StateType.C128_STATE in state_types:
                # C128 buffer 可能复用旧 request slot；接收新状态前要先清除残留的 ring 内容。
                clear_c128_state = getattr(
                    self.token_to_kv_pool, "clear_c128_req_state", None
                )
                # getattr 允许不实现该清理接口的兼容 KV pool 继续工作。
                if clear_c128_state is not None:
                    # 仅在 pool 明确提供清理能力时执行，避免对通用 KVCache 做类型假设。
                    clear_c128_state(int(decode_req.req.req_pool_idx))
                    # 以本次刚分配的 req slot 为粒度清理，不影响其他并发请求。
            # MINIMAX_INDEX_K reuses _dsa_payload: index rows live at the same loc
            # as main KV on the same page_size.
            payloads = {
                StateType.MAMBA: _mamba_payload,
                StateType.SWA: _swa_payload,
                StateType.DSA: _dsa_payload,
                StateType.MINIMAX_INDEX_K: _dsa_payload,
                StateType.SWA_RING: _swa_ring_payload,
                StateType.C128_STATE: _c128_state_payload,
            }
            # 使用“状态类型 -> 延迟构造函数”映射，只计算当前模型真正声明的 payload，
            # 避免无关模型访问不存在的 pool 属性或做不必要的 GPU->CPU 索引复制。
            if hasattr(self.req_to_token_pool, "req_to_token_c4"):
                # req_to_token_c4 标识 DSV4 NPU 的多池页表；该组合目前只支持无前缀命中。
                # DSV4 on NPU: per-pool dst page indices, produced by the same
                # shared builder prefill uses so src/dst line up positionally.
                if total_prefix_len != 0:
                    # 这里再次防御是因为 state payload 构造必须与 Prefill 的 source 排列一致。
                    raise RuntimeError(
                        "DSV4 NPU PD disaggregation does not support decode-side "
                        "prefix cache yet; disable disaggregation decode radix/HiCache "
                        "for PD + chunked prefill."
                    )
                    # 若允许继续，C4/C128 各池的 prefix offset 会不一致并写错目标状态。
            if _is_npu and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
                # DSV4 NPU 还有通用字典未覆盖的专用 state pools，按平台惰性导入 helper。
                from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                    dsv4_state_payloads,
                )
                # 局部 import 避免 CUDA/其他硬件环境加载 NPU 专用模块及其依赖。

                payloads.update(
                    dsv4_state_payloads(
                        self.req_to_token_pool,
                        decode_req.req.req_pool_idx,
                        seq_len,
                        self.token_to_kv_pool_allocator.page_size,
                        self.scheduler.sliding_window_size,
                        prefix_len=total_prefix_len,
                    )
                )
                # shared builder 同时被 Prefill/Decode 使用，保证每种 DSV4 state 的
                # source/destination 页列表在长度、顺序和 prefix offset 上严格对齐。
            state_indices: Optional[List] = [
                payloads[st]() if st in payloads else None for st in state_types
            ]
            # 顺序必须严格跟随 kv_args.state_types；None 表示该 state type 无额外索引，
            # backend 会采用默认布局或跳过，而不能打乱其他 state 的位置。

            decode_req.metadata_buffer_index = (
                self.req_to_metadata_buffer_idx_allocator.alloc()
            )
            # 每个在途请求独占一个 metadata buffer slot，供传输完成信号和辅助数据使用。
            assert decode_req.metadata_buffer_index is not None
            # 前面已检查 available_size；这里失败说明估算与 allocator 状态不同步，是内部 bug。
            # int32 for ZMQ serialization -- from_zmq reads np.int32.
            page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size).astype(
                np.int32
            )
            # 把 token location 压成页首索引，并固定为接收端 ZMQ 反序列化期待的 int32。
            device_page_indices = None
            # 只有 HiSparse DSV4 同时具有 host C4 页和 device logical 页，普通路径不需要它。
            if (
                self.scheduler.enable_hisparse
                and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
                and not _is_fake_transfer(decode_req.req, self.scheduler.server_args)
            ):
                # fake backend 不进行真实 RDMA，也不需要携带第二套 device 目标页信息。
                # alloc_logical_only() already allocated the shared logical pages
                # used by C4 indexer and C128 KV. These device buffers do not use
                # the C4 sparse physical-slot mapping; carry their logical page IDs
                # alongside the independently allocated C4 host page IDs.
                full_kv_indices = self.req_to_token_pool.req_to_token[
                    decode_req.req.req_pool_idx,
                    prefix_len:origin_input_len,
                ]
                # device states 使用 logical full-KV 页表；切掉 L1 prefix 后只传本轮增量区间。
                device_page_indices = kv_to_page_indices(
                    full_kv_indices,
                    page_size,
                ).astype(np.int32)
                # device pool 必须使用原 allocator page_size，不能使用 host C4 压缩页大小。
                if self.transfer_backend != TransferBackend.MOONCAKE:
                    # 当前只有 Mooncake backend 实现了同时携带 host/device 两套目标页的协议。
                    raise NotImplementedError(
                        "DSV4 HiSparse direct PD transfer currently requires "
                        "the Mooncake backend"
                    )
                    # 及早拒绝其他 backend，防止它们忽略 device_kv_indices 后写入错误地址。
            metadata_kwargs = {"decode_prefix_len": total_prefix_len}
            # 告诉 Prefill：Decode 已承诺拥有 [0, total_prefix_len) 的 KV，P 只需发送 delta。
            if device_page_indices is not None:
                # HiSparse DSV4 的 device-resident states 需要第二套目标页号。
                metadata_kwargs["device_kv_indices"] = device_page_indices
                # 通过 kwargs 扩展协议，使普通 backend/模型不承担无关字段。
            if (
                self.transfer_queue.enable_staging
                and hasattr(decode_req.kv_receiver, "require_staging")
                and decode_req.kv_receiver.require_staging
            ):
                # heterogeneous TP 可能要求先把 Prefill KV 写入 staging，再重排到最终 KV pool。
                # Register before send_metadata, which triggers the STAGING_REQ
                # prefetch (dropped for an unregistered room); tiny race, correct order.
                self.transfer_queue.staging_handler.register_decode_req(
                    decode_req.req.bootstrap_room, decode_req
                )
                # 必须先按 bootstrap_room 注册：send_metadata 会立即触发对端的 staging 请求，
                # 如果消息先到而本地映射尚不存在，该请求会被当作未知 room 丢弃。
            decode_req.kv_receiver.send_metadata(
                page_indices,
                decode_req.metadata_buffer_index,
                state_indices,
                **metadata_kwargs,
            )
            # 这是预分配阶段的提交点：Prefill 收到目标页后即可向 Decode 发起真实 KV 写入。
            if decode_req.is_rebootstrap:
                # true rebootstrap 不是普通 prompt 请求；它要求原 Prefill worker 用当前权重
                # 重算 prompt + 已生成 token 的 KV，因此 metadata 之后再提交重算任务。
                self.kv_manager.submit_prefill_recompute(
                    decode_req.kv_receiver,
                    decode_req.req.build_rebootstrap_payload(),
                )
                # payload 携带恢复生成所需的完整逻辑输入；复用已建立的 receiver 保证路由一致。
            preallocated_reqs.append(decode_req)
            # 调用方会把这些请求移入 transfer queue，等待 Prefill 完成数据传输。
            indices_to_remove.add(i)
            # 已提交 metadata 的请求不能继续留在 prealloc queue，否则下一轮会重复分配/发送。
            decode_req.req.time_stats.set_decode_transfer_queue_entry_time()
            # 记录进入 transfer 阶段的时间，用于拆分握手、排队和 KV 传输延迟指标。

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]
        # 一次性移除失败和已预分配条目；未握手或资源不足的请求保持原相对顺序等待下轮。

        return preallocated_reqs, failed_reqs
        # 返回值把“可进入 transfer”和“需要失败收尾”分流，队列中剩余请求无需返回。

    @property
    def num_tokens_pre_allocated(self):
        return sum(
            decode_req.req.extend_range.end for decode_req in self.transfer_queue.queue
        )

    def _need_space_for_single_req(
        self, retractable_tokens: Optional[int] = None
    ) -> int:
        need_space_for_single_req = (
            max(
                [
                    min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                    + len(x.origin_input_ids)
                    - retractable_tokens
                    for x in self.scheduler.running_batch.reqs
                ]
            )
            if retractable_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
            else 0
        )
        return need_space_for_single_req

    def _active_req_count(self, extra_reserved_reqs: int = 0) -> int:
        return (
            len(self.scheduler.running_batch.reqs)
            + len(self.transfer_queue.queue)
            + len(self.scheduler.waiting_queue)
            + extra_reserved_reqs
        )

    def _active_reserved_tokens(
        self, n_active: Optional[int] = None, extra_reserved_reqs: int = 0
    ) -> int:
        if n_active is None:
            n_active = self._active_req_count(extra_reserved_reqs)
        return self.num_reserved_decode_tokens * n_active

    def _swa_aware_allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
    ) -> Tuple[int, int]:
        n_active = self._active_req_count()
        reserved_tokens = self._active_reserved_tokens(n_active)

        full_allocatable_tokens = self._allocatable_token_budgets(
            retractable_tokens=retractable_tokens,
            count_retracted=count_retracted,
            reserved_tokens=reserved_tokens,
        )

        return full_allocatable_tokens, self._swa_tail_allocatable_token_budget(
            retractable_tokens=retractable_tokens,
            retractable_swa_tokens=retractable_swa_tokens,
            count_retracted=count_retracted,
            n_active=n_active,
            reserved_tokens=reserved_tokens,
        )

    def _allocatable_token_budgets(
        self,
        retractable_tokens: Optional[int] = None,
        count_retracted: bool = True,
        extra_reserved_reqs: int = 0,
        reserved_tokens: Optional[int] = None,
        hicache_reserved_tokens: int = 0,
    ) -> int:
        need_space_for_single_req = self._need_space_for_single_req(retractable_tokens)
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(
                extra_reserved_reqs=extra_reserved_reqs
            )

        if self.scheduler.enable_hisparse:
            logical_allocator = self.token_to_kv_pool_allocator.logical_attn_allocator
            if self._uses_swa_tail_prealloc() and hasattr(
                logical_allocator, "full_available_size"
            ):
                available_size = logical_allocator.full_available_size()
            else:
                # HiSparse pre-alloc only allocates logical indices, so the
                # logical pool is the binding constraint for admission control.
                available_size = logical_allocator.available_size()
        elif self._uses_swa_tail_prealloc():
            available_size = self.token_to_kv_pool_allocator.full_available_size()
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        else:
            available_size = self.token_to_kv_pool_allocator.available_size()
            # Include evictable decode-radix cache entries in the budget -- they
            # can be freed on demand before allocation.
            if self.scheduler.server_args.disaggregation_decode_enable_radix_cache:
                available_size += self.tree_cache.evictable_size()
        allocatable_tokens = available_size - max(
            reserved_tokens, need_space_for_single_req
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            allocatable_tokens -= self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )

        if count_retracted:
            for req in self.retracted_queue:
                full_required, _ = self._prealloc_required_tokens(req)
                allocatable_tokens -= full_required

        allocatable_tokens -= hicache_reserved_tokens
        return allocatable_tokens

    def _swa_tail_allocatable_token_budget(
        self,
        retractable_tokens: Optional[int] = None,
        retractable_swa_tokens: Optional[int] = None,
        count_retracted: bool = True,
        n_active: Optional[int] = None,
        reserved_tokens: Optional[int] = None,
    ) -> int:
        need_swa_space_for_single_req = self._need_space_for_single_req(
            retractable_tokens
        )
        if (
            retractable_swa_tokens is not None
            and len(self.scheduler.running_batch.reqs) > 0
        ):
            need_swa_space_for_single_req = max(
                self._swa_tail_len(len(x.origin_input_ids))
                + min(x.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKEN)
                - retractable_swa_tokens
                for x in self.scheduler.running_batch.reqs
            )

        if n_active is None:
            n_active = self._active_req_count()
        if reserved_tokens is None:
            reserved_tokens = self._active_reserved_tokens(n_active)

        # SWA growth is bounded by the sliding window: once a req's SWA
        # footprint reaches `sliding_window_size`, further decode tokens
        # evict old ones and net growth is zero. The linear reservation
        # `num_reserved_decode_tokens * n_active` (correct for the full
        # pool) over-reserves SWA in steady state. Cap by the actual
        # remaining headroom up to per-req window cap.
        window_size = self.scheduler.sliding_window_size or 0
        swa_total = self.token_to_kv_pool_allocator.size_swa
        swa_used = swa_total - self.token_to_kv_pool_allocator.swa_available_size()
        swa_growth_potential = max(0, n_active * window_size - swa_used)
        swa_reserved_tokens = min(reserved_tokens, swa_growth_potential)
        swa_allocatable_tokens = (
            self.token_to_kv_pool_allocator.swa_available_size()
            - max(swa_reserved_tokens, need_swa_space_for_single_req)
        )

        # Note: if the last prebuilt extend just finishes, and we enter `pop_preallocated` immediately in the next iteration
        #       the extend batch is not in any queue, so we need to explicitly add the tokens slots here
        if (
            self.scheduler.last_batch
            and self.scheduler.last_batch.forward_mode.is_prebuilt()
        ):
            prebuilt_reserved_tokens = self.num_reserved_decode_tokens * len(
                self.scheduler.last_batch.reqs
            )
            prebuilt_n = len(self.scheduler.last_batch.reqs)
            prebuilt_swa_growth = max(0, prebuilt_n * window_size - swa_used)
            swa_allocatable_tokens -= min(prebuilt_reserved_tokens, prebuilt_swa_growth)

        if count_retracted:
            for req in self.retracted_queue:
                _, swa_required = self._prealloc_required_tokens(req)
                swa_allocatable_tokens -= swa_required

        return swa_allocatable_tokens

    def _required_alloc_tokens(self, *, fill_len: int, prefix_len: int) -> int:
        page_size = self.token_to_kv_pool_allocator.page_size
        if page_size == 1:
            return fill_len - prefix_len

        num_new_pages = get_num_new_pages(
            seq_lens=torch.tensor([fill_len], dtype=torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int64),
            page_size=page_size,
        )
        return num_new_pages * page_size

    def _pre_alloc(
        self,
        req: Req,
        prefix_indices: Optional[torch.Tensor] = None,
        prefix_len: Optional[int] = None,
        total_prefix_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pre-allocate the memory for req_to_token and token_kv_pool.

        ``prefix_len`` is the L1 device-resident prefix length (already
        backed by ``prefix_indices``). ``total_prefix_len`` is the full
        prefix committed to prefill as ``decode_prefix_len`` (L1 + L2 + L3);
        the ``[prefix_len, total_prefix_len)`` gap is filled later by HiCache
        loadback.
        """
        if prefix_len is None:
            prefix_len = 0
        if total_prefix_len is None:
            total_prefix_len = prefix_len

        req_pool_indices = self.req_to_token_pool.alloc([req])

        assert (
            req_pool_indices is not None
        ), "req_pool_indices is full! There is a bug in memory estimation."

        fill_len = self._pre_alloc_fill_len(req)
        req.kv_committed_len = fill_len

        if prefix_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, prefix_len)), prefix_indices
            )

        # TODO(retraction): when retraction is implemented with radix cache
        # awareness, a retracted request should re-match the tree here
        # instead of re-allocating from scratch. See resume_retracted_reqs.
        delta_len = fill_len - total_prefix_len
        required_alloc_tokens = self._required_alloc_tokens(
            fill_len=fill_len, prefix_len=prefix_len
        )

        # Evict cached entries if the pool doesn't have enough free pages.
        if (
            self.scheduler.server_args.disaggregation_decode_enable_radix_cache
            and self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens
        ):
            num_to_evict = (
                required_alloc_tokens - self.token_to_kv_pool_allocator.available_size()
            )
            result = self.tree_cache.evict(EvictParams(num_tokens=num_to_evict))
            if self.token_to_kv_pool_allocator.available_size() < required_alloc_tokens:
                logger.warning(
                    f"Eviction insufficient: needed {required_alloc_tokens} tokens, "
                    f"available {self.token_to_kv_pool_allocator.available_size()} "
                    f"after evicting {result.num_tokens_evicted}/{num_to_evict} tokens. "
                    f"evictable_size={self.tree_cache.evictable_size()}, "
                    f"protected_size={self.tree_cache.protected_size()}, "
                    f"fill_len={fill_len}, prefix_len={prefix_len}, "
                    f"total_prefix_len={total_prefix_len}, delta_len={delta_len}, "
                    f"page_size={self.token_to_kv_pool_allocator.page_size}, "
                    f"req={req.rid}"
                )

        allocator = self.token_to_kv_pool_allocator
        if self.scheduler.enable_hisparse:
            # HiSparse is incompatible with decode-side L1 radix cache. Keep
            # this path on the upstream full-allocation semantics.
            assert prefix_len == 0

            # Direct-to-host path: only allocate logical indices (no hisparse
            # device indices) and allocate host indices for RDMA destination.
            coordinator = self.scheduler.hisparse_coordinator
            kv_loc = alloc_for_decode_prealloc_hisparse(
                allocator,
                req=req,
                fill_len=fill_len,
                uses_swa_tail=self._uses_swa_tail_prealloc(),
                swa_tail_len=self._swa_tail_len(fill_len),
            )
            # Allocate host indices for the RDMA transfer target.
            host_indices = coordinator.mem_pool_host.alloc_paged_token_slots(
                coordinator.req_to_host_pool,
                coordinator.req_to_host_pool_allocated_len,
                req.req_pool_idx,
                0,
                coordinator.host_token_len(fill_len),
            )
        else:
            uses_swa_tail = self._uses_swa_tail_prealloc() and prefix_len == 0
            swa_tail_len = self._swa_tail_len(fill_len)
            kv_loc = alloc_for_decode_prealloc(
                allocator,
                req=req,
                fill_len=fill_len,
                delta_len=delta_len,
                prefix_len=prefix_len,
                total_prefix_len=total_prefix_len,
                prefix_indices=prefix_indices,
                uses_swa_tail=uses_swa_tail,
                swa_tail_len=swa_tail_len,
                req_to_token_pool=self.req_to_token_pool,
            )
        assert kv_loc is not None, (
            f"KV cache is full! Bug in memory estimation. "
            f"available={self.token_to_kv_pool_allocator.available_size()}, "
            f"evictable={self.tree_cache.evictable_size()}, "
            f"protected={self.tree_cache.protected_size()}, "
            f"required_alloc={required_alloc_tokens}, delta={delta_len}, "
            f"fill={fill_len}, prefix={prefix_len}, total_prefix={total_prefix_len}, "
            f"page_size={self.token_to_kv_pool_allocator.page_size}, "
            f"req={req.rid}"
        )

        self.req_to_token_pool.write(
            (
                req.req_pool_idx,
                slice(total_prefix_len, total_prefix_len + len(kv_loc)),
            ),
            kv_loc,
        )

        # Truncate fill_len to kv_committed_len so cache_unfinished_req only
        # inserts committed KV into the radix tree. The last output token
        # hasn't had KV committed yet (output_ids is 1 ahead).
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        # Set prefix_indices so downstream consumers (init_next_round_input,
        # prepare_for_extend) see the correct prefix length. In the agg path
        # this is done inside init_next_round_input, but decode-disagg needs
        # allocation info before batch assembly so we set it here.
        req.prefix_indices = (
            prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
        )
        req.set_extend_range(total_prefix_len, req.kv_committed_len)

        # Return the transfer destination indices:
        if self.scheduler.enable_hisparse:
            return host_indices
        return kv_loc


def alloc_for_decode_prealloc_hisparse(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    uses_swa_tail: bool,
    swa_tail_len: int,
) -> torch.Tensor:
    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len
    device = allocator.device
    prefix_lens = torch.tensor([0], dtype=torch.int64, device=device)
    prefix_lens_cpu = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([fill_len], dtype=torch.int64, device=device)
    seq_lens_cpu = torch.tensor([fill_len], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64, device=device)
    if uses_swa_tail:
        kv_loc = allocator.alloc_extend_swa_tail(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
            swa_tail_len=swa_tail_len,
        )
        req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
    else:
        kv_loc = allocator.alloc_logical_only(
            prefix_lens=prefix_lens,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
            extend_num_tokens=fill_len,
        )
    return kv_loc


def alloc_for_decode_prealloc(
    allocator: BaseTokenToKVPoolAllocator,
    *,
    req: Req,
    fill_len: int,
    delta_len: int,
    prefix_len: int,
    total_prefix_len: int,
    prefix_indices: Optional[torch.Tensor],
    uses_swa_tail: bool,
    swa_tail_len: int,
    req_to_token_pool: Optional[ReqToTokenPool] = None,
) -> torch.Tensor:
    if req.kv is None:
        req.kv = ReqKvInfo(kv_allocated_len=fill_len, swa_evicted_seqlen=0)
    else:
        req.kv.kv_allocated_len = fill_len
    if allocator.page_size == 1:
        kv_loc = allocator.alloc(delta_len)
    else:
        device = allocator.device
        last_loc = (
            prefix_indices[-1:].to(dtype=torch.int64, device=device)
            if prefix_len > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )
        extra_kwargs = {}
        dsv4_unwrap_prealloc = None
        if hasattr(allocator, "c4_attn_allocator"):
            assert req_to_token_pool is not None
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import (
                dsv4_prealloc_kwargs,
                dsv4_unwrap_prealloc,
            )

            extra_kwargs = dsv4_prealloc_kwargs(
                allocator,
                req,
                fill_len,
                req_to_token_pool,
                device=device,
            )
        if uses_swa_tail:
            # Tail-only SWA allocation: only valid when prefix_len == 0.
            # When prefix_len > 0 (radix cache hit), we fall back to
            # alloc_extend which allocates SWA at full page count; the
            # SWA budget in that case may slightly under-estimate.
            kv_loc = allocator.alloc_extend_swa_tail(
                prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=fill_len,
                swa_tail_len=swa_tail_len,
                **extra_kwargs,
            )
            req.kv.swa_evicted_seqlen = fill_len - swa_tail_len
        else:
            kv_loc = allocator.alloc_extend(
                prefix_lens=torch.tensor(
                    [total_prefix_len], dtype=torch.int64, device=device
                ),
                prefix_lens_cpu=torch.tensor([total_prefix_len], dtype=torch.int64),
                seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=delta_len,
                **extra_kwargs,
            )
        if dsv4_unwrap_prealloc is not None:
            kv_loc = dsv4_unwrap_prealloc(
                kv_loc, req_to_token_pool, req, total_prefix_len, fill_len
            )
    return kv_loc


class DecodeTransferQueue(DecodeHiCacheTransferMixin):
    """
    Store the requests that is polling kv
    """

    def __init__(
        self,
        gloo_group: ProcessGroup,
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        tp_rank: int,
        metadata_buffers: MetadataBuffers,
        scheduler: Scheduler,
        tree_cache: BasePrefixCache,
    ):
        self.queue: List[DecodeRequest] = []
        self.gloo_group = gloo_group
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.metadata_buffers = metadata_buffers
        self.scheduler = scheduler
        self.tree_cache = tree_cache
        self.spec_algorithm = scheduler.spec_algorithm
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.staging_handler = None

    def add(self, decode_req: DecodeRequest) -> None:
        self.queue.append(decode_req)

    def extend(self, decode_reqs: List[DecodeRequest]) -> None:
        self.queue.extend(decode_reqs)

    def _commit_transfer_to_req(self, decode_req: DecodeRequest):
        idx = decode_req.metadata_buffer_index
        (
            output_id,
            cached_tokens,
            output_token_logprobs_val,
            output_token_logprobs_idx,
            output_top_logprobs_val,
            output_top_logprobs_idx,
            output_token_sampling_mask_len,
            output_token_sampling_mask_idx,
            output_token_sampling_logprobs,
            output_topk_p,
            output_topk_index,
            output_hidden_states,
            output_dsa_topk_indices,
            output_bootstrap_room,
        ) = self.metadata_buffers.get_buf(idx)

        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return

        self._commit_hicache_local_restore_to_req(decode_req)

        # Case 3: Success - commit the transfer
        # PD true-retraction rebootstrap: the prefill recomputed the prefix KV
        # under the current weights and sampled a fresh handoff token, but when
        # there is a remembered boundary token we are *replaying* an
        # already-emitted token. Override the handoff with it, and skip
        # re-committing a logprob for it -- it keeps its original behavior
        # logprob from before the retract (we never re-score generated tokens
        # under the new policy). A rebootstrap with no boundary token (retracted
        # before emitting any output) falls through to the normal path so its
        # first token and logprob are committed as usual.
        replayed_boundary = (
            decode_req.is_rebootstrap
            and decode_req.req.pd_rebootstrap_forced_output_id is not None
        )
        if replayed_boundary:
            committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id
            decode_req.req.pd_rebootstrap_forced_output_id = None
        else:
            committed_output_id = output_id[0].item()
        decode_req.req.output_ids.append(committed_output_id)
        decode_req.req.cached_tokens = cached_tokens[0].item()
        # The prefill node already reported its prefix-cache hit in
        # cached_tokens[0]. Seed already_computed with it so that
        # prepare_for_prebuilt's `cached_tokens += pre_len - already_computed`
        # only adds decode-side reuse *beyond* what prefill counted, instead of
        # double-counting the shared prompt prefix (which would make
        # cached_tokens exceed prompt_tokens when decode radix cache is on).
        decode_req.req.already_computed = decode_req.req.cached_tokens
        decode_req.req.cached_tokens_device = cached_tokens[1].item()
        decode_req.req.cached_tokens_host = cached_tokens[2].item()
        decode_req.req.cached_tokens_storage = cached_tokens[3].item()
        # Multimodal prompt token counts packed into cached_tokens slots 4-6
        # by the prefill node (see MetadataBuffers.set_buf).
        decode_req.req.mm_image_tokens = cached_tokens[4].item()
        decode_req.req.mm_audio_tokens = cached_tokens[5].item()
        decode_req.req.mm_video_tokens = cached_tokens[6].item()
        if not self.spec_algorithm.is_none():
            decode_req.req.output_topk_p = output_topk_p
            decode_req.req.output_topk_index = output_topk_index
            decode_req.req.hidden_states_tensor = output_hidden_states
            if (
                output_dsa_topk_indices is not None
                and torch.all(output_dsa_topk_indices < 0).item()
            ):
                output_dsa_topk_indices = None
            decode_req.req.output_dsa_topk_indices = output_dsa_topk_indices

        if decode_req.req.return_logprob and not replayed_boundary:
            decode_req.req.logprob.output_token_logprobs_val.append(
                output_token_logprobs_val[0].item()
            )
            decode_req.req.logprob.output_token_logprobs_idx.append(
                output_token_logprobs_idx[0].item()
            )
            decode_req.req.logprob.output_top_logprobs_val.append(
                output_top_logprobs_val[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
            decode_req.req.logprob.output_top_logprobs_idx.append(
                output_top_logprobs_idx[
                    : decode_req.req.logprob.top_logprobs_num
                ].tolist()
            )
        if decode_req.req.return_sampling_mask:
            assert (
                output_token_sampling_mask_idx is not None
            ), "sampling mask buffer disabled on decode side"
            sampling_mask_len = int(output_token_sampling_mask_len[0].item())
            if sampling_mask_len < 0:
                decode_req.req.output_token_sampling_mask.append(None)
                decode_req.req.output_token_sampling_logprobs.append(None)
            else:
                decode_req.req.output_token_sampling_mask.append(
                    output_token_sampling_mask_idx[:sampling_mask_len].cpu().tolist()
                )
                decode_req.req.output_token_sampling_logprobs.append(
                    float(output_token_sampling_logprobs[0].item())
                )

        decode_req.kv_receiver.clear()
        decode_req.kv_receiver = None
        decode_req.req.time_stats.set_wait_queue_entry_time()
        return

    def _poll_with_metadata_gate(self) -> List[int]:
        pollers = (
            [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
            if self.scheduler.enable_decode_hicache
            else [dr.kv_receiver for dr in self.queue]
        )
        return poll_and_all_reduce(
            pollers,
            self.gloo_group,
            decode_reqs=self.queue,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _poll_with_staging(self) -> list:
        return poll_and_all_reduce_with_staging(
            self.queue,
            self.staging_handler,
            self.gloo_group,
            metadata_buffers=self.metadata_buffers,
            server_args=self.scheduler.server_args,
        )

    def _init_staging_handler(self, kv_manager):
        """Create staging handler from kv_manager. Must be called exactly once."""
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingHandler,
        )

        self.staging_handler = DecodeStagingHandler.create(
            kv_manager, self.scheduler, self.tp_rank
        )
        kv_manager._staging_handler = self.staging_handler

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        if not self.queue:
            return []

        if self.scheduler.enable_decode_hicache:
            self._process_hicache_local_restores(
                [
                    decode_req
                    for decode_req in self.queue
                    if rids_to_check is None or decode_req.req.rid in rids_to_check
                ]
            )

        if self.enable_staging:
            polls = self._poll_with_staging()
        else:
            polls = self._poll_with_metadata_gate()

        transferred_reqs = []
        indices_to_remove = set()
        for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
                continue

            hicache_restore_status = decode_req.hicache_restore_status
            if (
                poll == KVPoll.Failed
                or hicache_restore_status == HiCacheRestoreResult.FAILED
            ):
                error_message = (
                    f"Decode transfer failed for request rank={self.tp_rank} "
                    f"{decode_req.req.rid=} {decode_req.req.bootstrap_room=}"
                )
                is_propagated = False
                if poll == KVPoll.Failed:
                    try:
                        decode_req.kv_receiver.failure_exception()
                    except Exception as e:
                        error_message += f" with exception {e}"
                        is_propagated = getattr(e, "is_from_another_rank", False)
                self._clean_hicache_prefetch_resources(decode_req)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.error(error_message)
                prepare_abort(
                    decode_req.req,
                    error_message,
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                self.scheduler.output_streamer.stream_output(
                    [decode_req.req],
                    decode_req.req.return_logprob,
                )
                if self.scheduler.enable_hisparse:
                    self.scheduler.hisparse_coordinator.request_finished(decode_req.req)
                # release pre-allocated kv cache, but don't insert into the tree since it's failed
                release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                decode_req.kv_receiver.clear()
                decode_req.kv_receiver = None
                indices_to_remove.add(i)
                if self.scheduler.metrics_reporter.enable_metrics:
                    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                continue
            elif poll == KVPoll.Success:
                if (
                    self.scheduler.enable_decode_hicache
                    and hicache_restore_status == HiCacheRestoreResult.PENDING
                ):
                    continue
                self._commit_transfer_to_req(decode_req)
                indices_to_remove.add(i)
                # Check if request was aborted due to corruption
                if isinstance(decode_req.req.finished_reason, FINISH_ABORT):
                    self.scheduler.output_streamer.stream_output(
                        [decode_req.req],
                        decode_req.req.return_logprob,
                    )
                    if self.scheduler.enable_hisparse:
                        self.scheduler.hisparse_coordinator.request_finished(
                            decode_req.req
                        )
                    self._clean_hicache_prefetch_resources(decode_req)
                    release_kv_cache(decode_req.req, self.tree_cache, is_insert=False)
                    if self.scheduler.metrics_reporter.enable_metrics:
                        self.scheduler.metrics_collector.increment_transfer_failed_reqs()
                else:
                    transferred_reqs.append(decode_req.req)
            elif poll in [
                KVPoll.Bootstrapping,
                KVPoll.WaitingForInput,
                KVPoll.Transferring,
            ]:
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            if self.enable_staging and self.staging_handler.is_staging_room(
                self.queue[i].req.bootstrap_room
            ):
                self.staging_handler.unregister_decode_req(
                    self.queue[i].req.bootstrap_room
                )
            idx = self.queue[i].metadata_buffer_index
            assert idx != -1
            # Reset so the next owner sees actual_room == 0 ("not yet written")
            # instead of the stale value, avoiding a false-positive mismatch.
            self.metadata_buffers.bootstrap_room[idx] = 0
            self.req_to_metadata_buffer_idx_allocator.free(idx)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        return transferred_reqs

    def release_memory_occupation(self):
        """Clean up in-flight transfers before releasing GPU memory."""
        self.queue.clear()

    def resume_memory_occupation(self):
        """Queues are already cleared on release; new transfers can be accepted."""
        pass


class SchedulerDisaggregationDecodeMixin:
    @torch.no_grad()
    def event_loop_normal_disagg_decode(self: Scheduler):
        """A normal scheduler loop for decode worker in disaggregation mode."""

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self: Scheduler):
        self.result_queue = deque()
        self.last_batch: Optional[ScheduleBatch] = None

        def pop_and_process():
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
            self.process_decode_queue()

            # Get the next batch to run
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=self.chunked_req
            )
            self.cur_batch_for_debug = batch
            # overlap + spec + grammar is unsupported (would desync DP ranks).
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            if disable_overlap_for_batch and self.last_batch:
                pop_and_process()

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self._apply_war_barrier()
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

    def _run_batch_prebuilt(
        self: Scheduler, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.inner_idle_batch is not None:
            idle_batch = batch.inner_idle_batch
            # Reset the inner idle batch to avoid reusing it.
            batch.inner_idle_batch = None
            return self.run_batch(idle_batch)

        return GenerationBatchResult()

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_decode_batch_to_run(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        """Process prebuilt batch and schedule the next decode batch."""
        # Process pending prebuilt batch: output processing + filter + merge
        new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
        if new_prebuilt_batch:
            assert self.chunked_req is None
            self.batch_result_processor.process_batch_result_prebuilt(
                new_prebuilt_batch
            )
            new_prebuilt_batch.filter_batch()
            if not new_prebuilt_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = new_prebuilt_batch
                    if self.enable_hisparse:
                        running_batch.hisparse_coordinator = self.hisparse_coordinator
                else:
                    running_batch.merge_batch(new_prebuilt_batch)

        # Schedule decode batch
        if running_batch.is_empty():
            ret = None
        else:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None

        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
        if ret:
            set_schedule_time_batch(ret)
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_new_prebuilt_batch(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Create a schedulebatch for fake completed prefill"""
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if len(self.waiting_queue) == 0:
            return None

        if self.enable_priority_scheduling:
            self.policy.calc_priority(self.waiting_queue, running_batch)

        curr_batch_size = running_batch.batch_size()

        batch_size = min(self.req_to_token_pool.size, self.max_running_requests)

        num_not_used_batch = batch_size - curr_batch_size

        # pop req from waiting queue
        can_run_list: List[Req] = []
        waiting_queue: List[Req] = []

        for i in range(len(self.waiting_queue)):
            req = self.waiting_queue[i]
            # we can only add at least `num_not_used_batch` new batch to the running queue
            if i < num_not_used_batch:
                can_run_list.append(req)
                # Decode-radix path: new requests already matched in
                # `pop_preallocated`. Retracted requests reset `last_node`,
                # so re-match only when that state is missing.
                if get_disagg().disaggregation_decode_enable_radix_cache:
                    tree_cache = self.tree_cache if req.last_node is None else None
                else:
                    tree_cache = self.tree_cache
                req.init_next_round_input(tree_cache)
                # Truncate fill_len to kv_committed_len so cache_unfinished_req
                # only sees committed KV (full array includes one uncommitted
                # token because init_next_round_input rebuilt it as full).
                if req.kv_committed_len is not None:
                    req.set_extend_range(len(req.prefix_indices), req.kv_committed_len)
            else:
                waiting_queue.append(req)

        self.waiting_queue = waiting_queue
        if len(can_run_list) == 0:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")

        # construct a schedule batch with those requests and mark as decode
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )

        # construct fake completed prefill
        new_batch.prepare_for_prebuilt()
        new_batch.process_prebuilt(self.server_args, self.future_map)

        return new_batch

    def process_decode_queue(self: Scheduler):
        if self.enable_decode_hicache:
            self.tree_cache.check_hicache_events()

        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.decode_offload_manager.check_offload_progress()

        # try to resume retracted requests if there are enough space for another `num_reserved_decode_tokens` decode steps
        resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
        self.waiting_queue.extend(resumed_reqs)
        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
            # if there are still retracted requests, we do not allocate new requests
            return

        if not hasattr(self, "polling_count"):
            self.polling_count = 0
            self.polling_interval = get_disagg().disaggregation_decode_polling_interval

        self.polling_count = (self.polling_count + 1) % self.polling_interval

        if self.polling_count % self.polling_interval == 0:
            req_conns, _ = self.disagg_decode_prealloc_queue.pop_preallocated()
            self.disagg_decode_transfer_queue.extend(req_conns)
            transferred_reqs = (
                self.disagg_decode_transfer_queue.pop_transferred()
            )  # the requests which kv has arrived
            if self.enable_hisparse:
                for req in transferred_reqs:
                    # Direct-to-host: KV data already in host pool, skip staging
                    self.hisparse_coordinator.admit_request_direct(req)
            self.waiting_queue.extend(transferred_reqs)
