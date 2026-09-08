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
"""GPU-side continuous-batching orchestrator for one parallel rank.

HTTP / Engine entrypoints tokenize on CPU. Model forward, KV allocation, and
sampling happen on GPUs that must stay saturated. ``Scheduler`` is the per-rank
process that owns one ``TpModelWorker`` / ``ModelRunner`` and, every iteration,
decides which requests form the next ``ScheduleBatch``.

TokenizerManager never talks to ModelRunner directly. It only sends already-
tokenized IPC messages (``TokenizedGenerateReqInput`` and control RPCs) to this
process. Without Scheduler there is no continuous batching: prefills would
block decodes, KV pages would leak, and TP / PP / DP ranks would not share a
single control plane.

This file is the composition root. Domain logic lives in collaborators
(``TpModelWorker``, radix cache, overlap future map, PD-disagg queues, ...).
``__init__`` only sequences ``init_*`` / ``maybe_init_*`` helpers so forks can
override one piece. Mixins cover PD-disagg, pipeline parallel, multiplex,
diffusion-LLM, and MLX overlap.

Entry point: :func:`run_scheduler_process` is the ``multiprocessing.Process``
target spawned by ``Engine._launch_scheduler_processes`` (or the DP controller
/ Ray actor). It publishes runtime context, constructs ``Scheduler``,
handshakes with the parent over ``pipe_writer``, then blocks in
:meth:`Scheduler.run_event_loop`.

Process topology::

    Client ──HTTP/gRPC──► TokenizerManager ──ZMQ req──► Scheduler (this process)
                              ▲                            │
                              │ ZMQ text                   │ ZMQ output ids
                              └──────── Detokenizer ◄──────┘
                                                           │
    Engine ──mp.Process / Pipe handshake───────────────────┤
                                                           ▼
                                                    TpModelWorker
                                                           ▼
                                                     ModelRunner / GPU

When ``dp_size > 1`` (or elastic EP), Engine launches a DataParallelController
instead of Scheduler ranks; the controller then spawns one Scheduler process
per (DP, TP, PP) replica.

Event-loop tick (normal)::

    recv_requests → process_input_requests → get_next_batch_to_run
          → run_batch (GPU forward) → process_batch_result → stream output

Overlap mode pipelines CPU result processing of batch N-1 with GPU forward of
batch N. Prefill is preferred over decode when both are eligible (better TTFT
while keeping the decode batch alive via merge).

Parallel ranks passed into this process (outermost → innermost)::

    Attention:  global TP  →  DP-attn  →  ATTN_CP  →  ATTN_TP
    MoE:        global TP  →  MOE_DP   →  EP       →  MOE_TP

----------------------------------------------------------------------
中文说明：本文件为什么存在、它在整条 serving 链路里做什么
----------------------------------------------------------------------

HTTP/gRPC 入口和 TokenizerManager 跑在 CPU 进程里，负责把文本变成 token。
真正的模型前向、KV 分配和采样必须在 GPU 上持续进行，否则 GPU 会空转。
``Scheduler`` 就是「每个并行 rank 一个」的 GPU 侧编排进程：它独占一个
``TpModelWorker`` / ``ModelRunner``，并在每一拍决定哪些请求组成下一个
``ScheduleBatch``。

TokenizerManager 从不直接调用 ModelRunner。它只通过 ZMQ 把已经 tokenize
好的 ``TokenizedGenerateReqInput`` 以及控制 RPC 发给本进程。如果没有
Scheduler，就没有 continuous batching：新的 prefill 会堵住正在 decode 的
请求，KV page 无法统一回收，TP / PP / DP 各 rank 也没有统一的控制面。

本文件是 composition root（组装根），领域逻辑放在协作对象里
（``TpModelWorker``、radix cache、overlap future map、PD-disagg 队列等）。
``__init__`` 只按顺序调用 ``init_*`` / ``maybe_init_*``，方便下游 fork
只覆盖其中一个组件。Mixin 覆盖 PD 分离、流水线并行、PD-multiplex、
diffusion-LLM 和 MLX overlap。

进程入口是 :func:`run_scheduler_process`，由 Engine 用 ``mp.Process`` 拉起
（``dp_size > 1`` 时由 DataParallelController 再 spawn；Ray 模式下由 actor
代替）。它先 ``publish`` 运行时配置、构造 ``Scheduler``、用 ``pipe_writer``
向父进程握手就绪，然后阻塞在 :meth:`Scheduler.run_event_loop`。

进程拓扑::

    客户端 ──HTTP/gRPC──► TokenizerManager ──ZMQ 已分词请求──► Scheduler（本进程）
                                ▲                                  │
                                │ ZMQ 文本                         │ ZMQ token id
                                └──────── Detokenizer ◄────────────┘
                                                                   │
    Engine ──mp.Process / Pipe 握手────────────────────────────────┤
                                                                   ▼
                                                            TpModelWorker
                                                                   ▼
                                                             ModelRunner / GPU

``dp_size > 1``（或 Elastic EP）时，Engine 不直接 spawn Scheduler，而是
启动 DataParallelController；控制器再为每个 (DP, TP, PP) 副本 spawn 一个
Scheduler 进程。

正常事件循环每一拍::

    收请求 → 处理输入 → 选出下一批
          → GPU 前向 → 处理结果 → 流式回传 token

Overlap 模式把第 N-1 批的 CPU 结果处理与第 N 批的 GPU 前向重叠。
prefill 和 decode 都就绪时优先跑 prefill（改善 TTFT），decode batch 通过
merge 继续活着。

传入本进程的并行 rank（由外到内）::

    Attention:  全局 TP  →  DP-attn  →  ATTN_CP  →  ATTN_TP
    MoE:        全局 TP  →  MOE_DP   →  EP       →  MOE_TP
"""

import dataclasses
import faulthandler
import logging
import os
import signal
import sys
import time
from array import array
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set, Tuple, Union

from sglang.srt.runtime_context import (
    attention_backends,
    configured_attn_cp_size,
    configured_dcp_size,
    configured_moe_dp_size,
    configured_pp_size,
    configured_tp_size,
    get_device,
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_mm,
    get_model,
    get_observability,
    get_parallel,
    get_schedule,
    get_server_args,
    get_serving,
    get_spec,
)

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.distributed import barrier

if TYPE_CHECKING:
    from torch.cuda import Stream as CudaStream

try:
    from sglang.kernels.ops.mamba.triton_ops import (
        initialize_mamba_selective_state_update_backend,
    )
except ImportError:
    initialize_mamba_selective_state_update_backend = None
from sglang.srt.configs.model_config import (
    ModelConfig,
    ModelImpl,
    is_deepseek_v4,
    is_minimax_sparse,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encoder.receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    get_dsa_seed_metadata_dim,
    prepare_abort,
    unified_memory_disagg_move_gate,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.disagg_service import maybe_create_ascend_config_store
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FinishReasonDict,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    ShutdownReq,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
    sock_send,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer
from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    decide_needs_confidence_relay,
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.rust_server import RustServer
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    NextBatchPlan,
    Req,
    ScheduleBatch,
    retract_all,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import (
    IdleSleeper,
    RustServerIdleSleeper,
)
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.memory_usage import (
    build_memory_usage,
    combine_graph_memory_usage,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
    retraction_discard,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    flush_trace_batch,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.startup_time import build_scheduler_startup_time
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.runtime_context import get_context, publish
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import PortArgs, ServerArgs, compute_world_size
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_mps,
    kill_itself_when_parent_died,
    rank_consensus_checker,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
    triton_load_watch,
)
from sglang.srt.utils.common import is_npu
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
    resolve_image_processor_backend,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_versions import (
    compute_weight_version_spans,
    record_weight_version_events,
)
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin
else:
    from torch.cuda import StreamContext as CudaStreamContext

    class SchedulerMlxOverlapMixin:
        pass


logger = logging.getLogger(__name__)


def _prewarm_hccl_group(device, group, device_module):
    """NPU 上用一次 all_reduce 预热 HCCL communicator。

    第一次集合通信会触发底层建链，放在 serving 热路径会把首包延迟打高；
    仅 DeepSeek-V4 + NPU 需要，见 ``maybe_init_hccl_dp_prewarm``。
    """
    warmup_tensor = torch.zeros(1, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(warmup_tensor, group=group)
    device_module.synchronize()


# Test retract decode for debugging purposes
# 人为制造 decode retract，用来测 KV 满时把请求踢回 waiting_queue 的路径。
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()


STEP_MAX_US = 2_000_000


def _accumulate_decode_moment(
    totals: list[float],
    batch_size: int,
    step_us: int,
    generated: int,
) -> None:
    if batch_size <= 0 or step_us <= 0:
        return
    b = float(batch_size)
    t = float(step_us)
    g = float(generated)
    totals[0] += 1.0
    totals[1] += b
    totals[2] += t
    totals[3] += b * b
    totals[4] += b * t
    totals[5] += g


_is_npu = is_npu()
_is_hip = is_hip()


class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerPPMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """Per-rank continuous-batching loop that owns one GPU worker.

    Exists so serving can interleave prefill admission, decode continuation,
    KV-cache management, and IPC in a single thread next to the GPU. Ranks
    share work via NCCL; only rank 0 of each attention-TP / CP / PP slice
    owns the tokenizer ZMQ sockets (see :meth:`init_ipc_channels`).

    Core state after :meth:`init_running_status`:

    * ``waiting_queue`` — admitted but not yet running ``Req`` objects.
    * ``running_batch`` — live decode (or prefill-only) ``ScheduleBatch``.
    * ``last_batch`` — previous forward; overlap mode processes its result
      while the current forward is in flight.
    * ``chunked_req`` — long prefill currently split across iterations.

    Mixins add PD-disagg bootstrap/transfer loops, pipeline-parallel
    microbatches, PD-multiplex, diffusion-LLM staging, and MLX overlap.

    中文：每个并行 rank 上的 continuous-batching 循环，独占一块 GPU worker。

    存在的理由：推理 serving 必须在「贴近 GPU 的单线程」里交织四件事——
    接纳新 prefill、延续已有 decode、管理 KV cache、收发 IPC。多线程拆开这些
    步骤会引入锁和跨流同步；放进独立进程则让 Tokenizer/HTTP 崩溃不影响 GPU
    侧状态。各 rank 用 NCCL 共享计算；每个 attention-TP / CP / PP 切片里
    只有 rank 0 持有 tokenizer ZMQ socket（见 :meth:`init_ipc_channels`），
    其余 rank 通过 CPU 进程组广播拿到同一批请求，保证并行集合通信对齐。

    :meth:`init_running_status` 之后的核心状态：

    * ``waiting_queue`` — 已准入、尚未进入 running 的 ``Req``。
    * ``running_batch`` — 正在 decode（或纯 prefill）的 ``ScheduleBatch``。
    * ``last_batch`` — 上一拍前向；overlap 模式下在当前 GPU 前向飞行时
      处理它的 CPU 结果。
    * ``chunked_req`` — 被切成多拍执行的超长 prefill。

    Mixin 负责 PD 分离的 bootstrap/传输循环、PP microbatch、PD-multiplex、
    diffusion-LLM staging 和 MLX overlap，避免把这些变体塞进主循环。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        """Wire collaborators for this rank. No event loop yet.

        Args:
            server_args: Full launch config. :func:`publish` already ran in
                :func:`run_scheduler_process`, so bag getters (``get_schedule``,
                ``get_parallel``, ...) are valid.
            port_args: ZMQ IPC paths and the NCCL init port for this instance
                (or this DP replica). Built by ``PortArgs.init_new``.
            gpu_id: Device index this process binds to, after
                ``maybe_reindex_device_id`` CUDA_VISIBLE_DEVICES remapping.
            tp_rank: This process's tensor-parallel rank in ``[0, tp_size)``.
                Also the key used to derive ATTN_CP / MOE_DP / MOE_EP ranks.
            moe_ep_rank: Expert-parallel rank inside the MoE hierarchy
                (global TP → MOE_DP → EP → MOE_TP).
            pp_rank: Pipeline-parallel stage. Rank 0 is the first stage
                (receives tokenized inputs); the last stage produces logits.
            attn_cp_rank: Attention context-parallel rank. Splits the KV
                sequence across ranks; 0 when CP is disabled.
            moe_dp_rank: MoE data-parallel rank (replicas of the expert
                mesh). Distinct from serving ``dp_rank``.
            dp_rank: Data-parallel replica id, or ``None`` when Engine
                launched this process directly (``dp_size == 1``). Set by
                DataParallelController / ``SGLANG_DP_RANK`` for multi-DP.

        中文：为本 rank 接线全部协作对象，此时还不进入事件循环。

        构造函数是编排器，不是算法本体：按固定顺序调用 ``init_*`` /
        ``maybe_init_*``，让 fork 可以只覆盖某一个组件（tokenizer、KV、
        IPC、worker）。真正的调度策略、前向、结果处理都在后续方法里。

        参数含义：
            server_args: 完整启动配置。:func:`run_scheduler_process` 里已经
                调用过 :func:`publish`，因此 ``get_schedule`` / ``get_parallel``
                等 bag getter 在这里是合法的。
            port_args: 本实例（或本 DP 副本）的 ZMQ IPC 路径和 NCCL 初始化端口，
                由 ``PortArgs.init_new`` 生成。
            gpu_id: 本进程绑定的设备号，已经过 ``maybe_reindex_device_id``
                对 CUDA_VISIBLE_DEVICES 的重映射。
            tp_rank: 本进程在 ``[0, tp_size)`` 中的张量并行 rank，也是推导
                ATTN_CP / MOE_DP / MOE_EP rank 的主键。
            moe_ep_rank: MoE 层级里的 expert-parallel rank
                （全局 TP → MOE_DP → EP → MOE_TP）。
            pp_rank: 流水线并行阶段。0 是第一段（收 tokenized 输入）；
                最后一段产出 logits。
            attn_cp_rank: Attention context-parallel rank，把 KV 序列切到
                多个 rank；关闭 CP 时为 0。
            moe_dp_rank: MoE 数据并行 rank（expert mesh 的副本），与 serving
                侧的 ``dp_rank`` 不是同一件事。
            dp_rank: 数据并行副本 id。Engine 直接拉起本进程时（``dp_size == 1``）
                为 ``None``；多 DP 时由 DataParallelController 或
                ``SGLANG_DP_RANK`` 填入。
        """
        # NOTE: KEEP THE FOLLOWING CODE STYLE for this function:
        # Keep __init__ as an orchestrator: sequence init_* and maybe_init_* calls
        # with minimal glue. Move substantial component-specific logic into
        # dedicated methods instead of adding inline blocks here.
        # 中文：__init__ 只做编排。领域逻辑放到独立 init_* 方法里，fork 才能按组件覆盖。
        self.init_startup_timing_begin()
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        # 软 watchdog 的守护线程在第一拍就会读这两个字段，必须在启动线程前赋初值。
        self.forward_ct: int = 0
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        self.init_soft_watchdog(server_args)

        # Parse args
        # 把频繁读取的开关从 bag getter 快照到 self.*，避免热路径反复查运行时上下文。
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = get_schedule().schedule_policy
        self.enable_priority_scheduling = get_schedule().enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            get_schedule().abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            get_schedule().schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            get_schedule().priority_scheduling_preemption_threshold
        )
        self.enable_lora = get_lora().enable_lora
        self.enable_lora_overlap_loading = get_lora().enable_lora_overlap_loading
        self.max_loras_per_batch = get_lora().max_loras_per_batch
        self.enable_overlap = (
            not get_schedule().disable_overlap_schedule and not use_mlx()
        )
        self.enable_overlap_mlx = (
            not get_schedule().disable_overlap_schedule and use_mlx()
        )
        self.enable_pdmux = get_disagg().enable_pdmux
        self.skip_tokenizer_init = get_serving().skip_tokenizer_init
        self.stream_interval = get_serving().stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.page_size = get_schedule().page_size
        self.enable_hierarchical_cache = get_memory().enable_hierarchical_cache
        self.enable_session_radix_cache = get_memory().enable_session_radix_cache
        self.enable_hicache_storage = get_memory().hicache_storage_backend is not None
        self.enable_decode_hicache = (
            get_disagg().disaggregation_decode_enable_radix_cache
            and self.enable_hierarchical_cache
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.max_new_tokens_limit = envs.SGLANG_MAX_NEW_TOKENS_LIMIT.get()
        self.enable_hisparse = get_memory().enable_hisparse
        self.enable_dp_attention = get_parallel().enable_dp_attention
        self.enable_unified_memory = get_memory().enable_unified_memory

        # Distributed rank info
        # 把全局 TP 再切成 DP-attn / ATTN_CP / ATTN_TP，供后续 IPC 与集合通信使用。
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                get_parallel().enable_dp_attention,
                tp_rank,
                configured_tp_size(),
                get_parallel().dp_size,
                configured_attn_cp_size(),
            )
        )
        # ParallelState 是本 rank 的并行身份卡，后续 worker / KV / IPC 都只读它，
        # 不再各自从 server_args 重新推导，避免 fork 后出现不一致。
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=configured_tp_size(),
            pp_rank=pp_rank,
            pp_size=configured_pp_size(),
            dp_rank=dp_rank,
            dp_size=get_parallel().dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=configured_attn_cp_size(),
            attn_dcp_rank=tp_rank % configured_dcp_size(),
            attn_dcp_size=configured_dcp_size(),
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=get_parallel().ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=configured_moe_dp_size(),
            gpu_id=gpu_id,
        )

        # Init model configs
        # 先读 HF config，后续 tokenizer / worker / KV 都依赖 model_config。
        self.init_model_config()

        # Init metrics stats
        self.init_metrics_collector(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        # 尽早建 ZMQ，非 rank0 会跳过 tokenizer socket；idle sleeper 依赖这些 socket。
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        # Init ZBAL, switch allocator should before any torch alloc action
        # NPU 必须在任何 torch 分配之前切换 allocator，否则后续权重/KV 走错内存池。
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        # 加载权重、分配 KV、捕获 CUDA Graph；这是启动路径里最重的一步。
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        # KV / radix cache 在 worker 之后构建：需要已经存在的 token pool 与通信组。
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=get_observability().enable_metrics,
            enable_kv_cache_events=bool(
                get_observability().kv_events_config
                and self.ps.pp_rank == 0
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            hicache_draft_plan=(
                self.draft_worker.hicache_draft_plan
                if self.draft_worker is not None
                else None
            ),
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache
        self.emit_metrics_constants()
        self.maybe_init_hccl_dp_prewarm()

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        self.init_hisparse_coordinator()

        if (
            get_disagg().disaggregation_mode == "decode"
            and get_disagg().disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=(
                    self.attn_tp_cpu_group
                    if self.enable_dp_attention
                    else self.tp_cpu_group
                ),
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        # Init running status
        # waiting_queue / running_batch / last_batch 是事件循环的三块核心状态。
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        self.init_metrics_reporter(tp_rank, pp_rank, dp_rank)

        # Init schedule policy and new token estimation
        # FCFS / LPM 等准入策略以及 decode token 占用估计，决定下一拍能塞进多少请求。
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Start the embedded Rust frontend (rank 0). Must precede
        # init_disaggregation: on PD prefill the rust api listener also serves
        # the KV bootstrap registry, and the KVManagers built there register to
        # it synchronously. (The listener is bound synchronously inside launch,
        # so the registry is accepting once this returns.) Must also precede
        # the request receiver, which reads self.recv_from_tokenizer to pick
        # its ingress transport.
        # 中文：Rust 前端必须先于 PD-disagg 和 request receiver。Prefill 侧的
        # KV bootstrap registry 挂在同一个 listener 上，KVManager 会同步注册；
        # receiver 也要根据 recv_from_tokenizer 选择入口传输。
        self.maybe_init_rust_server()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()

        # Init overlap schedule
        # future_map / 双 CUDA stream：让第 N 批 GPU 前向和第 N-1 批 CPU 收尾重叠。
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        self.init_weight_updater()

        # Init request dispatcher
        # 按消息类型分发：生成/embedding 进 waiting_queue，其余是控制 RPC。
        self.init_request_dispatcher()

        # Init LoRA drainer for fair scheduling
        self.init_lora_drainer()

        # Init LoRA overlap loader
        self.init_lora_overlap_loader()

        # Init the grammar backend for constrained generation
        self.init_grammar_manager()

        self.maybe_init_scripted_scheduler_hook()

        self.init_request_receiver()

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()

        self.init_kv_events_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()

        self.init_batch_result_processor()

        self.init_rank_consensus_checker()

        self.is_initializing = False
        self.init_startup_timing_summary()

    def init_startup_timing_begin(self) -> None:
        self.scheduler_startup_begin = time.perf_counter()

    def init_startup_timing_summary(self) -> None:
        self.startup_time = build_scheduler_startup_time(
            target_load_weight=self.tp_worker.weight_load_time,
            draft_load_weight=(
                0.0 if self.draft_worker is None else self.draft_worker.weight_load_time
            ),
            kv_cache_allocation=self.kv_cache_allocation_time,
            scheduler_e2e=time.perf_counter() - self.scheduler_startup_begin,
            target_cuda_graph=self.tp_worker.graph_time_usage,
            draft_cuda_graph=(
                None
                if self.draft_worker is None
                else self.draft_worker.graph_time_usage
            ),
        )

    def maybe_init_hccl_dp_prewarm(self) -> None:
        if not (
            _is_npu
            and is_deepseek_v4(self.tp_worker.model_runner.model_config.hf_config)
        ):
            return

        rank = (
            self.ps.dp_rank
            if self.ps.dp_rank is not None
            else self.tp_group.rank_in_group
        )
        logger.info("HCCL DP prewarm start: rank=%s", rank)
        _prewarm_hccl_group(
            device=self.tp_group.device,
            group=self.tp_group.device_group,
            device_module=self.tp_group.device_module,
        )
        logger.info("HCCL DP prewarm done: rank=%s", rank)

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if self.ps.pp_size > 1:
                logger.error("only zbal mix mode support pp_size > 1!")
            init_zbal(
                self.ps.tp_size, self.ps.gpu_id, self.ps.tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if get_exec().dllm.dllm_algorithm is not None
                else None
            )

    def init_metrics_collector(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

    def init_ipc_channels(self, port_args: PortArgs):
        """建立本 rank 与 Tokenizer / Detokenizer / RPC 之间的 ZMQ 通道。

        只有 PP=0 且 ATTN_TP=0 且 ATTN_CP=0 的 rank 真正 bind/connect tokenizer
        socket。其余 rank 靠进程组广播拿到同一批请求，这样 TP/CP 集合通信才对齐。
        skip_tokenizer_init 必须用构造时快照，不能二次读 bag，否则 IPC 与
        tokenizer 初始化会对不上。
        """
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            # The snapshot taken at construction, not a second bag read: this
            # scheduler gates its tokenizer init on the same value, and the two
            # must not be able to disagree.
            skip_tokenizer_init=self.skip_tokenizer_init,
            metrics_enabled=get_observability().enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or get_observability().enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        self.recv_from_tokenizer = None

        if not is_rank_zero:
            return

        self.recv_from_tokenizer = self.ipc_channels.recv_from_tokenizer
        dp_rank = self.ps.dp_rank if self.ps.dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                port_args,
                self.ps.dp_size,
                dp_rank,
                publish_interval=get_observability().load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        """空闲时阻塞在 ZMQ socket 上，避免热循环空转占满 CPU。

        非入口 rank 没有 tokenizer socket，不创建 sleeper；有请求到达时
        poll 立刻返回，不影响延迟。
        """
        if (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
            and get_device().sleep_on_idle
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        writer = self.load_snapshot_writer
        if writer is None:
            return
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return
        writer.publish_counter = 0
        try:
            writer.write(self.load_inquirer.get_loads())
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)

    def init_tokenizer(self):
        """在 Scheduler 进程内按需加载 tokenizer / processor。

        主路径的分词在 TokenizerManager 完成；这里再加载一份，是为了
        skip_tokenizer_init=False 时做长度校验、多模态 pad、以及把
        detokenize 所需的特殊 token 信息留在 GPU 侧。
        """
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if self.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    image_processor_backend=resolve_image_processor_backend(get_mm()),
                    tokenizer_backend=get_serving().tokenizer_backend,
                    model_name=get_model().model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if (
            self.model_config.is_multimodal
            and self.processor is not None
            and not get_disagg().language_model_only
        ):
            try:
                import_processors("sglang.srt.multimodal.processors")
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        if get_serving().reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=get_serving().reasoning_parser,
                stream_reasoning=False,
                tokenizer=self.tokenizer,
            )
            think_end_ids = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )
            if think_end_ids:
                self.model_config.think_end_ids = think_end_ids
            else:
                logger.warning(
                    "Reasoning parser think_end_token %r could not be encoded; "
                    "grammar-gated reasoning is disabled.",
                    reasoning_parser.detector.think_end_token,
                )

    def init_mamba_backend(self) -> None:
        if initialize_mamba_selective_state_update_backend is not None:
            initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``,
        # LongCat-2.0 uses ``moe_topk``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
            "moe_topk",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)
        initialize_bf16_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        """构造本 rank 的目标模型 worker（权重 + ModelRunner）。

        FIXME 注释仍成立：worker 初始化目前嵌在 Scheduler 里，是因为
        NCCL 端口、gpu_id、ParallelState 都由本进程持有。MLX 走独立
        worker 类，避免把 Apple GPU 路径塞进 CUDA TpModelWorker。
        """
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        """投机解码时再拉一个 draft worker，与 target 共享同一块 GPU。

        draft 不复制一份完整 server_args 差分配置：context length、
        load format、attention backend 都在各自 runner 上解析，避免
        两份配置漂移。无投机算法时 draft_worker 为 None，后续
        model_worker 直接指向 tp_worker。
        """
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding. It builds its draft
        # from this process's own config: what differs for the draft — the
        # target's context length, the draft load format, its attention backend
        # — is resolved per runner, not on a config copy.
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
        )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.ipc_channels.send_to_tokenizer.send_output,
            )
        else:
            self.external_corpus_manager = None

    def init_target_memory_pool(self):
        """Allocate target KV cache pools if they have not been allocated yet."""
        if (
            self.tp_worker.model_runner.memory_pool_config is not None
            and self.tp_worker.model_runner.req_to_token_pool is not None
            and self.tp_worker.model_runner.token_to_kv_pool_allocator is not None
        ):
            return
        self.tp_worker.alloc_memory_pool()

    def init_memory_pools(self):
        """Allocate KV cache pools for target and draft workers."""
        self.init_target_memory_pool()
        # Lands the retraction backend on the disagg bag before the draft
        # worker's HiCache plan reads it.
        kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.tp_worker)
        if self.draft_worker is not None:
            pool, allocator = self.tp_worker.get_memory_pool()
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=self.tp_worker.model_runner.memory_pool_config,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
            )
            self.draft_worker.init_hicache_draft_plan()

    def init_all_attention_backends(self):
        """Initialize attention backends for all workers."""
        self.tp_worker.init_attention_backends()
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_all_cuda_graphs(self):
        """Capture cuda graphs for all workers."""
        self.tp_worker.init_cuda_graphs()
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def init_model_worker(self):
        """加载权重、分配 KV、初始化 attention backend 并捕获 CUDA Graph。

        顺序不能乱：先 target worker，再 draft worker（draft 要挂到
        target 上）；内存池必须在 attention backend / CUDA Graph 之前，
        因为 Graph 捕获会固化 KV buffer 地址。投机解码时对外的
        ``model_worker`` 指向 draft，由它编排 verify + draft extend。
        """
        # Load model weights.
        self.init_tp_model_worker()
        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.start_startup_weight_load()
        self.maybe_init_draft_worker()

        # Prepare KV cache pools for all workers
        tic = time.perf_counter()
        self.init_memory_pools()
        self.kv_cache_allocation_time = time.perf_counter() - tic

        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        model_runner = self.tp_worker.model_runner
        if model_runner.token_to_kv_pool.post_capture_active:
            tic = time.perf_counter()
            model_runner.post_capture_resize_kv_pool()
            self.kv_cache_allocation_time += time.perf_counter() - tic

        if self.server_args.is_startup_weight_load_overlap:
            self.tp_worker.finalize_startup_weight_load()

        if (
            get_exec().moe.elastic_ep_backend is not None
            and get_exec().moe.ep_join_mode == "recover"
        ):
            model_runner.post_capture_elastic_ep_recover()

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
        min_free_slots = resolve_min_free_slots(
            get_schedule().min_free_slots_delay,
            self.max_running_requests,
            is_dflash_family=self.spec_algorithm.is_dflash_family(),
        )
        if min_free_slots is not None:
            self.min_free_slots_delayer = MinFreeSlotsDelayer(
                min_free_slots=min_free_slots
            )
        if not get_parallel().pp_max_micro_batch_size:
            get_context().override(
                "scheduler.pp_max_micro_batch_size_default",
                pp_max_micro_batch_size=max(
                    self.max_running_requests // self.ps.pp_size, 1
                ),
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_parallel().attn_tp_group
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_parallel().attn_cp_group
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group if self.enable_dp_attention else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # TODO(Jialin): Migrate pad_input_ids implementations to return array.
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        self.startup_available_gpu_memory_gb = get_available_gpu_memory(
            self.device, self.ps.gpu_id, empty_cache=False
        )
        if self.ps.tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={get_schedule().chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}="
                f"{self.startup_available_gpu_memory_gb:.2f} GB"
            )

    def emit_metrics_constants(self) -> None:
        if not get_observability().enable_metrics:
            return

        self.metrics_collector.emit_constants(
            max_total_num_tokens=self.max_total_num_tokens,
            max_total_num_tokens_swa=self.swa_tokens_per_layer,
            weight_memory_usage_gb=self.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_memory_usage_gb=(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage
            ),
            graph_memory_usage_gb=combine_graph_memory_usage(
                self.tp_worker.graph_memory_usage,
                (
                    None
                    if self.draft_worker is None
                    else self.draft_worker.graph_memory_usage
                ),
            ),
            # TODO: max_running_requests_under_SLO has no setter — dead chain.
            max_running_requests_under_SLO=None,
            page_size=self.page_size,
            num_pages=self.max_total_num_tokens // self.page_size,
            context_len=self.model_config.context_len,
            startup_available_gpu_memory_gb=self.startup_available_gpu_memory_gb,
        )

    def init_hisparse_coordinator(self) -> None:
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None
        if not self.enable_hisparse:
            return

        # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

    def init_running_status(self):
        """初始化事件循环要用的可变状态。

        ``waiting_queue`` 是准入后尚未 running 的请求；``running_batch`` 是
        当前 decode（或纯 prefill）batch；``last_batch`` 给 overlap 用来
        在下一拍 GPU 前向飞行时处理上一拍结果。``gracefully_exit`` 由
        ShutdownReq 置位，用来干净地跳出循环而不是杀进程。
        """
        # Set by the ShutdownReq handler to break the event loop for graceful shutdown.
        self.gracefully_exit = False
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self._last_logged_elastic_radix_namespace: Optional[str] = None
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        """配置超长 prefill 的切块执行。

        一次把整段 prompt 塞进 GPU 会撑爆 KV 或拖垮 TTFT，所以把长
        prefill 切成多拍。chunked_req 跨 iteration 存活，直到最后一块
        完成才 merge 进 running_batch。多模态 Transformers backend 关掉
        chunked prefill，避免半截图像 token 对不齐。
        """
        self.chunked_prefill_size = get_schedule().chunked_prefill_size
        self.prefill_decode_interval = get_schedule().prefill_decode_interval
        self._prefill_decode_interval_remaining = 0
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and get_schedule().enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            get_schedule().enable_dynamic_chunking and self.ps.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def _should_defer_prefill(self) -> bool:
        if self._prefill_decode_interval_remaining == 0:
            return False

        self._prefill_decode_interval_remaining -= 1
        return True

    def _arm_prefill_decode_interval(self, batch: Optional[ScheduleBatch]) -> None:
        if self.prefill_decode_interval == 0 or batch is None:
            return

        # DP attention synchronizes this flag across ranks. This keeps every
        # rank on the same prefill/decode cadence even when only one rank has
        # local prefill work. Non-DP scheduling can use the local mode directly.
        is_extend = (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        )
        if is_extend:
            self._prefill_decode_interval_remaining = self.prefill_decode_interval

    def init_metrics_reporter(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        # Override point for deployments that need a specialized reporter.
        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

    def init_schedule_policy(self):
        """初始化 waiting_queue 的排序策略和新 token 占用估计。

        SchedulePolicy（FCFS / LPM / priority）决定先服务谁；
        NewTokenRatioTracker 估计 decode 还会再写多少 KV，供 PrefillAdder
        判断还能不能再塞 prefill。PrefillDelayer 是可选的反压阀。
        """
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker(
            window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
        )
        self.max_prefill_bs: int = 0
        if get_schedule().enable_prefill_delayer:
            if get_disagg().disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.ps.dp_size,
                    attn_tp_size=self.ps.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    device_group=self.tp_group.device_group,
                    server_args=self.server_args,
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_reporter.enable_metrics
                        else None
                    ),
                    max_delay_passes=get_schedule().prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=get_schedule().prefill_delayer_token_usage_low_watermark,
                    device=self.tp_group.device,
                    debug_log_enabled=self.ps.attn_tp_rank == 0,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not get_schedule().disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_config()

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := get_device().soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def init_watch_dog_memory_saver_input_blocker(self):
        """启动卡死检测、可选 memory saver，以及同机 batch-gen 的输入门闩。

        watchdog 超时说明 GPU 前向或集合通信卡住，应主动自杀而不是
        让健康检查一直 200。input_blocker 仅在共置 batch-gen 时启用。
        """
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=get_device().watchdog_timeout
        )

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create()
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        """按 PD 角色初始化 bootstrap / 传输队列。

        普通合设模式（NULL）什么都不建。Prefill 实例要能把 KV 发给 Decode
        实例，Decode 实例要先预分配槽位再收 KV。队列实现在 mixin 里，
        这里只接线，避免主循环直接依赖 RDMA/NIXL 细节。
        """
        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        self.disaggregation_mode = DisaggregationMode(get_disagg().disaggregation_mode)
        self.transfer_backend = TransferBackend(
            get_disagg().disaggregation_transfer_backend
        )

        # In rust-server mode the KV bootstrap registry is already serving on
        # the rust api listener (maybe_init_rust_server runs before this
        # method — the PrefillBootstrapQueue's KVManager below registers to it
        # synchronously, and a failed registration only retries ~60s then logs,
        # leaving every PD request unroutable). Only the ascend config store,
        # which start_disagg_service would otherwise create, is left to do.
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and self._hosts_rust_server()
        ):
            maybe_create_ascend_config_store(
                server_args=self.server_args,
                transfer_backend=self.transfer_backend,
            )

        draft_token_to_kv_pool = (
            self.draft_worker.primary_draft_kv_pool
            if self.draft_worker is not None
            else None
        )

        if self.spec_algorithm.carries_draft_hidden_states():
            # `draft_runner` aliases `draft_runner_list[0]` in the multi-layer
            # worker, so a single accessor covers both shapes.
            draft_runner = self.draft_worker.draft_worker.draft_runner
            disagg_hidden_size, disagg_hidden_states_dtype = (
                get_draft_recurrent_hidden_state_spec(draft_runner)
            )
        else:
            disagg_hidden_size = 16  # minimal padding size for RDMA
            disagg_hidden_states_dtype = torch.float32

        # The PD metadata wire schema must match on P and D even when only D
        # enables spec decoding; a seedless prefill writes the invalid sentinel.
        output_dsa_topk_indices_dim = get_dsa_seed_metadata_dim(
            self.model_config.hf_config
        )

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *8 headroom for MiniMax-M3; *2 for other models.
            buffer_multiplier = (
                8 if is_minimax_sparse(self.model_config.hf_config) else 2
            )
            buffer_size = (self.req_to_token_pool.size) * buffer_multiplier
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.ps.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                dp_size=get_parallel().dp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=get_disagg().disaggregation_bootstrap_port,
                max_total_num_tokens=self.max_total_num_tokens,
                pp_rank=self.ps.pp_rank,
                num_reserved_decode_tokens=get_disagg().num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
                output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=get_disagg().disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.ps.pp_rank,
                pp_size=self.ps.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []
            # Requests with a sent chunk that are not yet on the inflight queue.
            self.disagg_prefill_pending_chunk_rids: Set[str] = set()

            self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        if (
            self.enable_unified_memory
            and self.disaggregation_mode != DisaggregationMode.NULL
        ):
            self.token_to_kv_pool_allocator.set_disagg_move_gate(
                unified_memory_disagg_move_gate(self)
            )

        # Init mm receiver for EPD disaggregation mode
        if get_disagg().language_only and get_disagg().encoder_transfer_backend in [
            "zmq_to_scheduler",
            "mooncake",
        ]:
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        """准备 overlap 调度所需的 CUDA stream 和 FutureMap。

        FutureMap 在非 overlap 下也启用，用来在 CPU 与 GPU 之间接力
        input_ids。真正的双 stream 重叠只在 ``enable_overlap`` 时打开：
        ``forward_stream`` 跑模型，``schedule_stream`` 准备下一拍 batch。
        """
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)
        needs_confidence_relay = decide_needs_confidence_relay(self.server_args)
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
            needs_confidence_relay=needs_confidence_relay,
        )

        self._confidence_budget_prepare = None
        if (
            needs_confidence_relay
            and self.enable_overlap
            and self.draft_worker is not None
        ):
            self._confidence_budget_prepare = (
                self.draft_worker.get_confidence_budget_prepare()
            )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        if not self.enable_overlap:
            return

        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.ngram_embedding_manager = (
            self.tp_worker.model_runner.ngram_embedding_manager
        )
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.ngram_embedding_manager.table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        if not get_exec().deterministic.enable_deterministic_inference:
            self.truncation_align_size = None
            return

        backend_sizes = {
            "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
            "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
        }
        # Both entries are prefill knobs (SPLIT_TILE / PREFILL_TRUNCATION):
        # the prefill half decides.
        prefill_backend, _ = attention_backends()
        env_var, default_size = backend_sizes.get(prefill_backend, (None, None))
        self.truncation_align_size = (
            get_int_env_var(env_var, default_size) if env_var else None
        )

    def init_request_dispatcher(self):
        """按 IPC 消息类型分发到对应 handler。

        热路径是 TokenizedGenerate / Embedding；其余是缓存刷新、权重更新、
        LoRA 加载、profile、shutdown 等控制面。用 TypeBasedDispatcher 而
        不是一长串 if-elif，是为了新增消息类型时只加一行映射，并让
        未知类型立刻失败而不是静默丢弃。
        """
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.weight_updater.destroy_weights_update_group,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.weight_updater.update_weights_from_ipc,
                ),
                (
                    UpdateWeightVersionReqInput,
                    self.handle_update_weight_version,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (ScaleElasticEPReqInput, self.handle_scale_elastic_ep),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def _abort_on_running_timeout(self, running_batch: ScheduleBatch):
        # NOTE: this should be called before a batch is launched.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if running_batch.is_empty():
            return

        deadline = time.perf_counter() - timeout_s
        for req in running_batch.reqs:
            if not req.finished() and 0 < req.time_stats.forward_entry_time < deadline:
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.

        中文：通过 Pipe 回给父进程（Engine 或 DP controller）的就绪握手。
        父进程必须等到 ``status=ready`` 才对外接流量；同时带回
        ``max_total_num_tokens`` / ``max_req_input_len``，Tokenizer 用它们
        在 CPU 侧做长度校验，避免超长请求已经进 GPU 才被拒绝。
        ``startup_time`` 用于启动耗时拆解。
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
            "startup_time": self.startup_time,
        }

        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        # 只在优雅退出时调用：pinned host buffer / HiCache 需要显式归还，
        # 否则进程退出后仍可能占着 PINNED 内存直到被 OS 回收。
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()
        self.tree_cache.release_host_resources()
        if self.decode_offload_manager is not None:
            self.decode_offload_manager.release_host_resources()

        rank_consensus_checker.shutdown()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.

        中文：事件循环入口。先装 Triton lazy-load 监视（捕获后首次加载
        kernel 会拖垮 serving），再创建独立的 ``schedule_stream``，让
        CPU 侧准备 batch 与 GPU 前向可以重叠。真正的循环体由
        :func:`dispatch_event_loop` 按 PP / overlap / PD-disagg / MLX
        选出，本方法本身不实现调度算法。
        """
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
        triton_load_watch.install()
        triton_load_watch.mark_serving_started()

        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        elif is_cuda() or _is_hip:
            # CUDA/HIP streams come from a fixed round-robin pool. Redraw if this
            # stream aliases forward_stream, which would eliminate scheduler
            # overlap. Only CUDA/HIP streams expose a ``cuda_stream`` handle;
            # other accelerators (e.g. NPU/XPU) skip the alias check.
            _redraws = 0
            while (
                self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream
                and _redraws < 64
            ):
                self.schedule_stream = self.device_module.Stream(priority=0)
                _redraws += 1
        # The global WAR barrier fences the scheduler's next shared-buffer write
        # on the previous forward's read of the unified memory pool.
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # WAR: keep later schedule_stream writes behind this forward's shared reads.
        # Clearing matters: a phase that skips the publish then falls back to coarse.
        if not self._war_barrier_enabled:
            return
        runner = self.model_worker.last_shared_read_runner
        ev = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            self.schedule_stream.wait_event(ev)
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop.

        中文：最朴素的一拍一拍循环，GPU 前向和 CPU 结果处理串行。

        每一拍：收 IPC → 把请求放进 waiting_queue → 选出下一批 →
        ``run_batch`` 阻塞到 GPU 完成 → ``process_batch_result`` 回收 KV
        并回传 token。没有 overlap 时实现简单、易于推理正确性；代价是
        CPU 收尾会让 GPU 空等。空闲时走 ``on_idle`` 做自检并重置状态。
        """
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            # rank0 从 ZMQ 拉消息，其余 rank 通过进程组拿到同一批。
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            # 可能是新 prefill、继续 decode，或 None（空闲）。
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self._sched_idled = True
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation.

        中文：把第 N 批 GPU 前向和第 N-1 批 CPU 结果处理重叠，提高 GPU 利用率。

        ``result_queue`` 保存尚未处理的 (batch, result)。本拍 ``run_batch``
        只把 kernel 提交到 ``forward_stream`` 就返回，随后立刻处理上一拍
        结果。采样（grammar / structured output）必须等上一拍 FSM 推进完，
        所以 ``launch_batch_sample_if_needed`` 放在 ``pop_and_process`` 之后。
        连续两拍都是 prefill 时可以关掉 overlap，优先保证第一批的 TTFT。
        """
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
                # copy() 快照 batch：后续 waiting_queue 变更不能污染正在飞行的结果。
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None
                self._sched_idled = True

            # Process the last batch
            # 默认处理上一拍；若本拍刚因连续 prefill 关掉 overlap，上面已经处理过了。
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    def is_disable_overlap_for_batch(
        self, batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        # 中文：连续 prefill 关掉 overlap，让第一批的 CPU 收尾立刻完成、尽快出首 token。
        # DP-attn 必须用全局同步后的 is_extend_in_batch，否则各 rank 对 overlap
        # 决策不一致会在 MLP all-reduce 上死锁。
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # Sync so the FSM advance lands before the next batch's bitmask. Permanent
        # path for host-draft algorithms, not a pending migration.
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.grammar_needs_sync()
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        # Algorithms that support grammar overlap advance the FSM inside verify()
        # via the grammar barrier (overlapping the target forward), which resolves
        # whatever result is still pending in the queue — including the
        # extend->decode boundary — so no grammar-specific overlap disable is needed.
        return disable_overlap_for_batch or need_grammar_sync

    def _advance_pending_grammar(self):
        """Grammar barrier (spec-v2 overlap): advance the FSM over any not-yet
        -processed decode result still in the queue, so a following verify()'s
        bitmask sees the previous batch's committed tokens. Invoked mid-worker
        (before generate_token_bitmask) so the CPU advance overlaps the target
        verify forward. Idempotent; no-op when the queue is empty or has no grammar.
        """
        for prev_batch, prev_result in self.result_queue:
            self.batch_result_processor.advance_grammar_fsm(prev_result, prev_batch)

    @scheduler_nvtx_method("scheduler.process_input_requests")
    def process_input_requests(self, recv_reqs: List):
        """把本拍从 ZMQ/RPC 收到的消息交给 TypeBasedDispatcher。

        健康检查请求在服务忙碌时直接记下 IPC 回传地址，避免长 prefill
        堵住 health probe。真正的生成请求在 handler 里变成 ``Req`` 进入
        waiting_queue 或 PD-disagg 队列。
        """
        now = time.monotonic()
        self.session_controller.maybe_reap(now)
        if get_mm().mm_feature_transport == "cuda_vmm":
            for recv_req in recv_reqs:
                self._materialize_cuda_vmm_inputs(recv_req)

        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if self.rust_server is not None:
                    # Embedded Rust server: every control-request response goes
                    # back through the egress ring (the zmq tokenizer socket is
                    # not consumed); the Rust api_server shapes it per-endpoint.
                    self.rust_server.push_control_output(recv_req, output)
                elif isinstance(output, RpcReqOutput):
                    if self.ipc_channels.recv_from_rpc is not None:
                        sock_send(self.ipc_channels.recv_from_rpc, output)
                else:
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()

    def _materialize_cuda_vmm_inputs(self, recv_req):
        """Release VMM slices before request handling can reject the request."""
        if isinstance(
            recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
        ):
            tokenized_reqs = (recv_req,)
        elif isinstance(
            recv_req,
            (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput),
        ):
            tokenized_reqs = recv_req
        else:
            return

        for tokenized_req in tokenized_reqs:
            if tokenized_req.mm_inputs is not None and not isinstance(
                tokenized_req.mm_inputs, MultimodalInputs
            ):
                tokenized_req.mm_inputs = MultimodalInputs.from_processor_output(
                    tokenized_req.mm_inputs
                )

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=self.dp_tp_cpu_group,
            get_forward_ct=lambda: self.forward_ct,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=self.tp_cpu_group,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            scheduler=self,
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if get_lora().lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                get_lora().max_loras_per_batch,
                get_lora().lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(self)

    def maybe_init_scripted_scheduler_hook(self) -> None:
        if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
            from sglang.test.scripted_runtime.scheduler_hook import (
                ScriptedSchedulerHook,
            )

            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                scheduler=self,
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def _hosts_rust_server(self) -> bool:
        """Whether this scheduler rank embeds the Rust server (rank 0 only) —
        and with it the server-process duties a Python ``TokenizerManager``
        would otherwise own (e.g. serving the PD KV bootstrap registry)."""
        return envs.SGLANG_RUST_SERVER.get() and (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )

    def maybe_init_rust_server(self) -> None:
        """Start the embedded Rust server (rank 0) if ``SGLANG_RUST_SERVER`` is
        set, and point the ingress receiver at it. All the plumbing lives in
        ``RustServer`` (scheduler_components/rust_scheduler.py)."""

        if not self._hosts_rust_server():
            # Always define the attribute: init_output_streamer and the
            # process_input_requests hook read self.rust_server unconditionally.
            self.rust_server = None
            return

        rust_server = RustServer.launch(self)
        self.rust_server = rust_server
        # The rust server *is* the ingress source: SchedulerRequestReceiver
        # drains its request ring (rust_server_mode) instead of a zmq socket.
        self.recv_from_tokenizer = rust_server
        # Park the idle loop on the request ring within the rank-0 rust-server
        self.idle_sleeper = RustServerIdleSleeper(rust_server)

    def init_request_receiver(self) -> None:
        """封装「从 tokenizer/RPC/多模态通道收消息并在 TP 组内广播」。

        非 rank0 没有 ZMQ PULL socket，receiver 会走进程组 broadcast，
        保证同一拍所有 rank 看到同一批请求。max_recv_per_poll 限制单拍
        收取数量，避免一次把 waiting_queue 灌满拖垮本拍调度。
        """
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_batch=lambda: self.last_batch,
            scripted_scheduler_hook=self.scripted_scheduler_hook,
        )

    def init_dp_attn_adapter(self) -> None:
        """DP-attention 下让各 rank 对「本拍是 prefill / decode / idle」达成一致。

        MLP all-reduce 要求每个 rank 都进入同一次 forward。某个 DP rank
        没有本地请求时必须发 idle batch 去陪跑，否则集合通信死锁。
        """
        # Spec workers have no .model_runner of their own; the prefill graph
        # runner that votes belongs to the target model.
        target_worker = (
            self.tp_worker.target_worker
            if isinstance(self.tp_worker, BaseSpecWorker)
            else self.tp_worker
        )
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            model_runner=target_worker.model_runner,
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.weight_updater.offload_tags,
            ps=self.ps,
            server_args=self.server_args,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )

    def init_pool_stats_observer(self) -> None:
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens
            * get_parallel().attn_dcp_size,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            disaggregation_mode=self.disaggregation_mode,
            page_size=self.page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_rank_consensus_checker(self) -> None:
        groups = []
        if self.attn_cp_group is not None and self.attn_tp_group is not None:
            groups += [self.attn_cp_group, self.attn_tp_group]
        else:
            groups += [self.tp_group]
        if self.pp_group is not None:
            groups += [self.pp_group]
        rank_consensus_checker.configure(groups)

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=get_observability().kv_events_config,
            ps=self.ps,
            attn_tp_rank=self.ps.attn_tp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            attn_dp_rank=self.ps.attn_dp_rank,
            dp_rank=self.ps.dp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            get_stats=lambda: self.metrics_reporter.stats,
        )

    def init_load_inquirer(self) -> None:
        self.total_prefill_uncached_tokens = 0
        self.total_prefill_busy_us = 0
        self.decode_moment_totals: list[float] = [0.0] * 6
        self._prev_step: Optional[Tuple[int, float, bool]] = None
        self._sched_idled = False
        self.load_inquirer = SchedulerLoadInquirer(
            disaggregation_mode=self.disaggregation_mode,
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            waiting_queue_prefix_matched=lambda: self.policy.waiting_queue_prefix_matched(
                self.waiting_queue
            ),
            get_recent_cache_hit_rate=lambda: self.metrics_reporter.recent_cache_hit_rate,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: self.disagg_prefill_bootstrap_queue,
            get_disagg_prefill_inflight_queue=lambda: self.disagg_prefill_inflight_queue,
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode_prealloc_queue,
            get_disagg_decode_transfer_queue=lambda: self.disagg_decode_transfer_queue,
            get_spec_total_num_accept_tokens=lambda: self.metrics_reporter.spec_total_num_accept_tokens,
            get_spec_total_num_forward_ct=lambda: self.metrics_reporter.spec_total_num_forward_ct,
            get_total_prefill_uncached_tokens=lambda: self.total_prefill_uncached_tokens,
            get_total_prefill_busy_us=lambda: self.total_prefill_busy_us,
            get_decode_moment_totals=lambda: self.decode_moment_totals,
        )

    def init_output_streamer(self) -> None:
        """把 GPU 侧产出的 token id 经 ZMQ 送给 Detokenizer。

        Scheduler 不把 id 直接发回 Tokenizer，是为了解耦「采样」和
        「detokenize + 流式 HTTP」。只有入口 rank 真正 send。
        """
        self.output_streamer = self.get_output_streamer_class()(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=self.tree_cache,
            ps=self.ps,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: self.enable_hicache_storage,
            rust_server=self.rust_server,
        )

    def get_output_streamer_class(self) -> type[SchedulerOutputStreamer]:
        return SchedulerOutputStreamer

    def init_batch_result_processor(self) -> None:
        """把「前向结果 → 更新 Req / 回收 KV / 流式输出」从主循环抽出去。

        decode 与 prefill 的收尾逻辑很长，留在 Scheduler 会让事件循环
        不可读。processor 通过回调拿 tree_cache / streamer，不反向持有
        整个 Scheduler。
        """
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            server_args=self.server_args,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            abort_request=self.abort_request,
        )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        max_new_tokens = (
            req.sampling_params.max_new_tokens
            if req.sampling_params.max_new_tokens is not None
            else 1 << 30
        )
        if self.max_new_tokens_limit is not None and self.max_new_tokens_limit > 0:
            if max_new_tokens > self.max_new_tokens_limit:
                logger.warning(
                    f"Capping max_new_tokens of request {req.rid} to "
                    f"SGLANG_MAX_NEW_TOKENS_LIMIT={self.max_new_tokens_limit} "
                    f"(requested: {req.sampling_params.max_new_tokens})."
                )
            max_new_tokens = min(max_new_tokens, self.max_new_tokens_limit)

        # Keep this bound consistent with PrefillAdder's admission budget:
        # ceil_page(input_len) + max_new_tokens + page_size must be strictly
        # smaller than max_total_num_tokens. Otherwise a request can be accepted
        # into the waiting queue but can never be scheduled, blocking the queue
        # and eventually making health checks fail.
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                max_new_tokens,
                self.max_req_len - input_len - 1,
                self.max_total_num_tokens * get_parallel().attn_dcp_size
                - paged_input_len
                - self.page_size
                - 1,
            ),
        )
        # Clipping above can push max_new_tokens below min_new_tokens, which
        # would suppress EOS for the whole generation. Restore the invariant.
        if req.sampling_params.min_new_tokens > req.sampling_params.max_new_tokens:
            req.sampling_params.min_new_tokens = req.sampling_params.max_new_tokens

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs):
        if isinstance(mm_inputs, MultimodalInputs):
            return mm_inputs

        if get_mm().enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs)
        return MultimodalInputs.from_processor_output(mm_inputs)

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_namespace_elastic_radix_cache(self, req: Req) -> None:
        if (
            get_exec().moe.elastic_ep_backend is None
            or self.disable_radix_cache
            or not self.tree_cache.is_tree_cache()
        ):
            return

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is None:
            return

        namespace = f"elastic_ep_size={ElasticEPStateManager.get_effective_ep_size()}"
        if req.extra_key:
            req.extra_key = f"{req.extra_key}|{namespace}"
        else:
            req.extra_key = namespace

        if self._last_logged_elastic_radix_namespace != namespace:
            self._last_logged_elastic_radix_namespace = namespace
            logger.debug(
                "[Elastic EP][scale] radix cache namespace is now %s",
                namespace,
            )

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        """把 Tokenizer 送来的已分词生成请求变成 ``Req`` 并入队。

        这是热路径入口：校验长度、处理 session / 多模态 / PD bootstrap
        元数据，然后交给 ``_add_request_to_queue``。普通模式进
        ``waiting_queue``；prefill 实例进 bootstrap 队列；decode 实例进
        prealloc 队列。本函数不做 GPU 前向，只做 CPU 侧准入。
        """
        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )
        # Radix-native sessions use only the top-level session_id.
        radix_native_session = (
            recv_req.session_id is not None and self.enable_session_radix_cache
        )

        if session_id is None or radix_native_session:
            # Normal non-session request, or a radix-native session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = get_disagg().disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                return_sampling_mask=recv_req.return_sampling_mask,
                return_flat_raw_top_logprobs=recv_req.return_flat_raw_top_logprobs,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                cache_salt=recv_req.cache_salt,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer

            if radix_native_session:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    recv_req.session_id
                )

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    if not envs.SGLANG_RUST_SERVER.get():
                        recv_req.time_stats.trace_ctx.abort(
                            abort_info={"reason": error_msg}
                        )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            if self.enable_session_radix_cache:
                req.session_generation = self.tree_cache.ensure_session_generation(
                    session_id
                )
            # TODO: set trace context
            if self.metrics_reporter.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        self._maybe_namespace_elastic_radix_cache(req)

        if self.spec_algorithm.is_dflash_family():
            error_msg = validate_dflash_request(req, self.enable_overlap)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        if (
            req.return_sampling_mask
            and self.disaggregation_mode != DisaggregationMode.NULL
            and not self.disagg_metadata_buffers.enable_sampling_mask
        ):
            error_msg = (
                "return_sampling_mask with disaggregation requires "
                "SGLANG_DISAGGREGATION_SAMPLING_MASK_MAX_TOKENS > 0."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and req.sampling_params.top_k == TOP_K_ALL:
            error_msg = (
                "return_sampling_mask requires finite top_k; top_p-only sampling "
                "is valid but can return huge masks in the tail, blowing up "
                "metadata, so we need a safety cap."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            # Spec workers do not emit one sampling support per accepted token, so
            # the returned mask would not align 1:1 with generated tokens. Reject
            # the combination instead of silently returning a misaligned mask.
            error_msg = (
                "return_sampling_mask is not supported with speculative decoding."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if req.return_sampling_mask and get_exec().kernel.sampling_backend == "ascend":
            # The ascend backend samples from logits directly and never builds the
            # top-k/top-p support, so it cannot produce a sampling mask.
            error_msg = (
                "return_sampling_mask is not supported with the ascend "
                "sampling backend."
            )
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if (
            get_device().mlx_enable_sampling
            and req.return_logprob
            and 0 <= req.logprob_start_len < len(req.origin_input_ids)
        ):
            # The MLX sampling path computes output logprobs only; the
            # prefill result carries no input_token_logprobs, so letting
            # this through would crash output processing.
            error_msg = (
                "Prompt input logprobs (logprob_start_len) are not supported "
                "on the MLX sampling path; omit logprob_start_len to get "
                "output logprobs."
            )
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                req.set_finish_with_abort(error_msg)
                self._add_request_to_queue(req)
                return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        if self.enable_hicache_storage:
            req.init_next_round_input(self.tree_cache, cow_mamba=False)
            tree_cache = self.tree_cache
            buffer_mode = self.server_args.hicache_host_memory_mode == "buffer_only"
            last_host_node = req.last_host_node
            # Buffer mode host-backups nothing, so match_prefix anchors at
            # root; re-anchor on the deepest device node. The anchor is only
            # read for hash/extra-key context here (never locked), so a device
            # node serves. Cache mode keeps the is_backuped gate below: its
            # write-through prefix is contiguous from root, so an unbacked
            # anchor means a guaranteed storage miss.
            if (
                buffer_mode
                and tree_cache.is_root(last_host_node)
                and not tree_cache.is_root(req.last_node)
            ):
                last_host_node = req.last_node

            if (
                tree_cache.is_backuped(last_host_node)
                or tree_cache.is_root(last_host_node)
                or (
                    buffer_mode
                    and tree_cache.get_last_hash_value(last_host_node) is not None
                )
            ):
                matched_len = len(req.prefix_indices) + req.host_hit_length
                match_end = req._compute_max_prefix_len(
                    len(req.full_untruncated_fill_ids)
                )
                new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]
                prefix_keys = (
                    tree_cache.get_prefix_hash_values(last_host_node)
                    if tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                tree_cache.prefetch_from_storage(
                    req.rid,
                    last_host_node,
                    new_input_tokens,
                    tree_cache.get_last_hash_value(last_host_node),
                    prefix_keys,
                    matched_prefix_tokens=req.full_untruncated_fill_ids[:matched_len],
                )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        """按当前 disagg 模式把请求放到对应队列。

        NULL：waiting_queue（本实例同时做 prefill+decode）。
        PREFILL：bootstrap 队列，等 KV 传输端点就绪。
        DECODE：prealloc 队列，先占 KV 槽再收远端 KV。
        retracted 请求走同一入口，避免 decode OOM 回退路径另写一套入队。
        """
        if not self._set_or_validate_priority(req):
            return
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = _make_abort_req(
                req,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            _make_abort_req(
                req_to_abort,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        for req in self.waiting_queue:
            entry_time = req.time_stats.wait_queue_entry_time
            if 0 < entry_time < deadline:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(
                        req,
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
        )
        req.tokenizer = self.tokenizer
        self._maybe_namespace_elastic_radix_cache(req)

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                # See companion call site above for the array.array wrap rationale.
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )

            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            get_serving().allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later, so the batch-result processors keep
        ``inflight_middle_chunks`` accounting intact and skip the aborted chunk:
        ``process_batch_result_disagg_prefill`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self._pending_chunked_abort_req
        if req is None:
            return
        if self.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or req.req_pool_idx is None:
                self._pending_chunked_abort_req = None
            return

        prepare_abort(req, "Aborted")
        req.time_stats.trace_ctx.abort(abort_info={"reason": "Aborted"})
        req.to_finish = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.clear_pending_chunk_send(req)
            req.disagg_kv_sender.abort()
            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        release_kv_cache(req, self.tree_cache, is_insert=False)

        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.ipc_channels.send_to_tokenizer.send_output(_make_abort_req(req), req)
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
        device = self.device

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        req_pool_indices = [r.req_pool_idx for r in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # Stash last token into relay; resolve_forward_inputs will gather.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )
        self.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        batch.input_ids = None

        if batch.return_logprob:
            batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        """选出下一拍要送到 GPU 的 batch。这是 continuous batching 的核心。

        逻辑顺序：
        1. 把上一拍已完成的 prefill merge 进 ``running_batch``（chunked
           请求先摘出去，避免半截 prefill 被当成 decode）；
        2. 尝试从 ``waiting_queue`` 组一个新的 prefill batch；
        3. 有新 prefill 就优先跑它（改善 TTFT），否则继续 decode；
        4. DP-attn 下通过 MLP sync 让各 rank 对齐 prefill/decode/idle，
           避免有的 rank 在做 all-reduce 时别的 rank 已经进了下一拍。

        返回 ``NextBatchPlan``：``batch_to_run`` 是本拍 GPU 输入，
        ``running_batch`` 是更新后的 decode 集合。
        """
        self.process_pending_chunked_abort()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout(running_batch)
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        # 上一拍的 extend 完成后，活下来的请求变成 decode，并入 running_batch。
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                if self.dllm_config.first_done_first_out_mode:
                    if not req.dllm_incomplete_ids:
                        self.stash_chunked_request(req)
                        self.req_to_token_pool.free(req)
                    # Otherwise, keep req slot/KV for reuse.
                else:
                    self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # hybrid-SWA early-return) leaves extend_range.end ==
            # len(prefix_indices), so there is nothing new to cache and
            # stashing would be a no-op.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if running_batch.is_empty():
                    running_batch = new_batch
                else:
                    running_batch.merge_batch(new_batch)
                running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and last_batch
            and last_batch.forward_mode.is_extend()
        ):
            if last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            if self.dllm_config is not None and last_batch.reqs:
                chunked_req_to_exclude.update(last_batch.reqs)

            # Filter batch
            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not last_batch.is_empty():
                if running_batch.is_empty():
                    running_batch = last_batch
                else:
                    # Merge running_batch with prefill batch
                    running_batch.merge_batch(last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if running_batch.is_prefill_only:
            running_batch.filter_batch()
            if running_batch.is_empty():
                running_batch.batch_is_full = False

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm(running_batch)
        elif self._should_defer_prefill():
            new_batch = None
        else:
            prefill_plan = self.get_new_batch_prefill(running_batch)
            new_batch = prefill_plan.batch_to_run
            running_batch = prefill_plan.running_batch

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not get_spec().speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        if new_batch is not None:
            # Run prefill first if possible
            # 有新 prefill 就先跑：改善 TTFT，decode 请求仍留在 running_batch 里下一拍继续。
            ret = new_batch
        else:
            # Run decode (skip for prefill-only batches)
            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                running_batch = self.update_running_batch(running_batch)
                ret = running_batch if not running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            ret, need_sync=need_mlp_sync
        )
        self._arm_prefill_decode_interval(ret)

        # Handle ngram embedding
        ret = self.ngram_embedding_manager.prepare_for_forward(
            ret, chunked_req=self.chunked_req
        )

        if ret:
            set_schedule_time_batch(ret)
            if self.enable_fpm:
                ret.fpm_start_time = self._fpm_batch_t0

        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_num_allocatable_reqs(self, running_bs):
        res = get_parallel().pp_max_micro_batch_size - running_bs
        res = min(res, self.req_to_token_pool.available_size())
        return res

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        """从 waiting_queue 尝试组成一个新的 prefill batch。

        PrefillDelayer 在 KV 占用偏高时推迟接纳新 prefill，给正在跑的
        decode 留余量，避免 TTFT 优化把 ITL 打崩。真正的容量估算在
        ``_get_new_batch_prefill_raw`` / ``PrefillAdder`` 里。
        """
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret, running_batch = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            running_batch=running_batch,
        )

        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )

        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def _get_new_batch_prefill_raw(
        self,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
        running_batch: ScheduleBatch,
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        """按 KV 余量、running 上限、LoRA 槽位和调度策略挑选 prefill 请求。

        ``running_batch.batch_is_full`` 为真且没有 chunked_req 时直接放弃，
        避免每个空闲拍都扫描 waiting_queue。chunked 请求必须始终能继续，
        否则 KV 槽位泄漏。grammar 未就绪的请求先不入 waiting_queue。
        """
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache or get_memory().enable_flexkv:
            self.tree_cache.check_hicache_events()

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            running_batch.batch_is_full = False

        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None, running_batch

        running_bs = len(running_batch.reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.get_num_allocatable_reqs(running_bs),
            )
        ):
            return None, running_batch

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return None, running_batch

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None, running_batch

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if self.chunked_req is not None and self.enable_dynamic_chunking:
            history_len = len(self.chunked_req.prefix_indices)
            dynamic_size = self.predict_next_chunk_size(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        # Get BLOCK_M from the backend for tile-budget admission logic
        attn_backend = self.tp_worker.model_runner.attn_backend
        if hasattr(attn_backend, "extend_attention_block_m"):
            prefill_tile_block_m = attn_backend.extend_attention_block_m
        else:
            prefill_tile_block_m = 64  # Fallback for non-Triton backends

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=int(self.max_prefill_bs),
            max_running_requests=self.max_running_requests,
            prefill_max_requests=get_schedule().prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
            prefill_tile_block_m=prefill_tile_block_m,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                continue

            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    running_batch.batch_is_full = True

            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                loaded_tokens = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)
                if loaded_tokens > 0:
                    req.storage_hit_length = loaded_tokens

            req.init_next_round_input(self.tree_cache)
            if (
                self.enable_hicache_storage
                and self.server_args.hicache_host_memory_mode == "buffer_only"
            ):
                # Buffer mode: surface a staged prefetch as the request's host
                # hit (consumed through init_load_back) plus its SWA window,
                # which consumption allocates and the request lock pins —
                # uncharged, the batch alloc can OOM. Set AFTER
                # init_next_round_input (which recomputes host_hit). Mamba
                # (fenced in init_hicache) will need the same charge via
                # mamba_host_hit_length.
                held_tokens = self.tree_cache.staged_prefetch_tokens(req.rid)
                if held_tokens > 0:
                    req.host_hit_length = held_tokens
                    req.swa_host_hit_length = (
                        self.tree_cache.staged_prefetch_swa_tokens(req.rid)
                    )
            res = adder.add_one_req(
                req,
                has_chunked_req=(self.chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                            not running_batch.is_empty()
                        )
                    else:
                        running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if not added:
                    # init_next_round_input() may stage deferred Mamba COW/clear
                    # metadata before add_one_req() rejects the request.
                    req.mamba_cow_src_index = None
                    req.mamba_needs_clear = False
                    if req.mamba_pool_idx is not None and not getattr(
                        req, "session", None
                    ):
                        self.tree_cache.req_to_token_pool.mamba_allocator.free(
                            req.mamba_pool_idx.unsqueeze(-1)
                        )
                        req.mamba_pool_idx = None
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None, running_batch

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )

        new_batch.contains_last_prefill_chunk = (
            self.chunked_req is None or len(can_run_list) != 1
        )

        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        if self.tp_worker.model_runner.prefill_aware_swa:
            for req in can_run_list:
                req.swa_evict_floor = req.extend_range.end

        # Record prefill stats for logging after forward.
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_range.length
                    if self.chunked_req is not None
                    else 0
                ),
            ),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not running_batch.is_empty()
            and not (new_batch.return_logprob or running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            running_batch.filter_batch()
            if not running_batch.is_empty():
                running_batch.prepare_for_decode()
                new_batch.mix_with_running(running_batch)
                new_batch.decoding_reqs = running_batch.reqs
            running_batch = ScheduleBatch(
                reqs=[], batch_is_full=running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch, running_batch

    def _can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if self.enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch.

        中文：准备下一拍 decode：滤掉已完成请求；KV 不够时 retract 部分
        请求回 waiting_queue（释放 page，稍后重新 prefill 前缀）。这是
        continuous batching 在显存压力下的安全阀：宁可拉长个别请求的
        延迟，也不让整批 decode OOM 把服务打挂。
        """
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
            new_available_tokens = self.token_to_kv_pool_allocator.available_size()
            new_token_gained = new_available_tokens - old_available_tokens
            mamba_num_gained = (
                mamba_allocator.available_size() - old_mamba_available
                if mamba_allocator is not None
                else None
            )

            self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
            if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
                self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                    num_retracted_reqs=len(retracted_reqs),
                    num_retracted_input_tokens=sum(
                        len(r.origin_input_ids) for r in retracted_reqs
                    ),
                    num_retracted_output_tokens=sum(
                        len(r.output_ids) for r in retracted_reqs
                    ),
                )
            self.new_token_ratio_tracker.current = new_token_ratio
            for req in reqs_to_abort:
                abort_reason: FINISH_ABORT = req.to_finish
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req, finished_reason=abort_reason.to_json()),
                    req,
                )

            msg_prefix = (
                "KV cache pool is full. Retract requests. "
                if kv_full_retract_flag
                else "Testing retraction. "
            )
            msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
            if mamba_num_gained is not None:
                msg_details += f", #mamba_num_gained: {mamba_num_gained}"
            if kv_full_retract_flag:
                msg_details += (
                    f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
                )
            logger.warning(msg_prefix + msg_details)

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)
        else:
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, batch: ScheduleBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.
        attr_snapshot = [
            getattr(batch, f.name, None) for f in dataclasses.fields(batch)
        ]
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        # List (not tuple) so that workers can register additional refs via
        # GenerationBatchResult.extra_keep_alive_refs after forward returns.
        self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
        """Make SB transactional across one forward (overlap and non-overlap).

        1. Snapshot SB fields so V2's mid-forward mutations (forward_mode /
           input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec
           only need sampling_info restored - V1 carries spec_info forward as
           next-iter draft input.
        2. Substitute sampling_info with a forward-only copy (orchestrator=None,
           shares the pre-accumulated penalty buffer) so V2's multiple init_new
           calls don't double-accumulate penalties.
        3. (overlap=True only) Pin (batch, snapshot) into batch_record_buf
           for 2 iters so GPU tensors in the snapshot survive the caching
           allocator past the forward stream. Must run AFTER the sampling_info
           swap so the forward-only copy gets pinned. The non-overlap (sync) path
           runs on a single stream and doesn't allocate batch_record_buf, so it
           passes overlap=False.
        """
        # 1. snapshot
        snapshot_v2_full = not batch.spec_algorithm.is_none()
        sched_snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if snapshot_v2_full
            else None
        )
        sched_sampling_info = batch.sampling_info

        # 2. sampling_info substitute
        if sched_sampling_info is not None:
            batch.sampling_info = sched_sampling_info.copy_for_forward()

        # 3. pin for 2-iter tensor lifetime (overlap path only)
        if overlap:
            self.record_batch_in_overlap(batch)

        try:
            yield
        finally:
            if snapshot_v2_full:
                for name, value in sched_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sched_sampling_info

    @scheduler_nvtx_method("scheduler.run_batch")
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch.

        中文：把 ``ScheduleBatch`` 交给 ``model_worker`` 做 GPU 前向。

        这是 Scheduler 与 ModelRunner 的边界：本方法负责 profiler、
        overlap stream 同步、PD prefill 提前发送 prefix KV、以及把
        投机解码 / embedding / 普通 generation 路由到对应 worker API。
        真正的 attention / MLP / sampling kernel 都在 worker 内部。
        overlap 模式下这里通常只提交 kernel，不等 GPU 完成。
        """
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct
        batch.launch_ts = time.monotonic()
        batch.after_idle_gap = self._sched_idled
        self._sched_idled = False

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # PD prefill: early-send cached prefix KV, overlapping the suffix forward.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in batch.reqs:
                self.maybe_send_cached_prefix_chunk(req)

        # Run forward
        if self.is_generation:
            if self.enable_overlap:
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                self.future_map.resolve_seq_lens_cpu(batch)
                if self._confidence_budget_prepare is not None:
                    self._confidence_budget_prepare(batch, self.future_map)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        fwd_kwargs = {}
                        if not batch.spec_algorithm.is_none():
                            fwd_kwargs["on_publish"] = partial(
                                self.future_map.publish, future_indices
                            )
                            # Grammar-overlap-capable workers advance the grammar FSM
                            # inside verify() before building the bitmask; hand them the
                            # barrier that resolves the previous batch's committed
                            # tokens (overlapping the target forward).
                            if batch.spec_algorithm.supports_grammar_overlap():
                                fwd_kwargs["grammar_barrier"] = (
                                    self._advance_pending_grammar
                                )

                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        if self.enable_unified_memory:
                            # Record a `forward_done` event after the forward (before
                            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
                            # it. Only the unified pool's allocator exposes these hooks.
                            allocator = self.token_to_kv_pool_allocator
                            forward_done = self.device_module.Event()
                            forward_done.record(stream=self.forward_stream)
                            allocator.set_latest_forward_done_event(forward_done)
                            # Write-set classification: hand the allocator this
                            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
                            allocator.set_inflight_forward(
                                forward_done,
                                batch.out_cache_loc,
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        batch_result.copy_done = self.device_module.Event()
                        if batch_result.delay_sample_func is None:
                            self._relay_forward_payload(future_indices, batch_result)
                            if _is_hip:
                                # Cross-stream sync costs more than the tiny D2H it
                                # overlaps.
                                batch_result.copy_to_cpu(
                                    return_logprob=batch.return_logprob,
                                    return_hidden_states=batch.return_hidden_states,
                                )
                            else:
                                # Result D2H on copy_stream overlaps the next forward
                                # instead of serializing on forward_stream; it's a leaf
                                # gated by copy_done, so nothing on forward_stream waits.
                                self.copy_stream.wait_stream(self.forward_stream)
                                with self.copy_stream_ctx:
                                    batch_result.copy_to_cpu(
                                        return_logprob=batch.return_logprob,
                                        return_hidden_states=batch.return_hidden_states,
                                    )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                batch.input_ids = None

                if not batch.spec_algorithm.is_none():
                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_dsa_topk_indices_available = (
                        batch.spec_info.dsa_topk_indices is not None
                    )
                    batch.spec_info.future_indices = future_indices
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                self._relay_forward_payload(batch.req_pool_indices, batch_result)
                batch.input_ids = None
                self._copy_auxiliary_output_to_cpu(batch, batch_result)
            elif not batch.spec_algorithm.is_none():
                # Non-overlap: drive the V2 worker synchronously (no
                # future_map relay / on_publish).
                resolve_forward_inputs(batch, self.future_map)
                with self._forward_isolation(batch, overlap=False):
                    batch_result = self.model_worker.forward_batch_generation(batch)
                # The isolation restore reverted the worker's in-forward SB edits;
                # re-apply what must carry to the next iter.
                batch.spec_info = batch_result.next_draft_input
                if batch_result.new_seq_lens is not None:
                    batch.seq_lens = batch_result.new_seq_lens
                    if batch.seq_lens_cpu is not None:
                        batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                batch.input_ids = None  # rebuilt next iter from draft_token
                self.update_cache_from_scheduler(batch, batch_result)
                # Sync D2H so the result processor can read CPU tensors.
                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
                if batch_result.has_sampled_token_ids:
                    # Non-spec: relay via future_map, gathered next iter.
                    self._relay_forward_payload(batch.req_pool_indices, batch_result)
                    batch.input_ids = None
                self.update_cache_from_scheduler(batch, batch_result)
                self._copy_auxiliary_output_to_cpu(batch, batch_result)

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob or batch.return_hidden_states:
                batch_result.extend_input_len_per_req = [
                    req.extend_range.length if req.extend_range is not None else 0
                    for req in batch.reqs
                ]
            else:
                batch_result.extend_input_len_per_req = None

            if batch.return_logprob:
                batch_result.extend_logprob_start_len_per_req = (
                    batch.extend_logprob_start_lens
                )
            else:
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            if self.enable_overlap:
                self.record_batch_in_overlap(batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    resolve_forward_inputs(batch, self.future_map)
                    pooler_output, can_run_cuda_graph = (
                        self.tp_worker.forward_batch_embedding(batch)
                    )
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                        can_run_cuda_graph=can_run_cuda_graph,
                    )
                    ret.copy_to_cpu()
            else:
                resolve_forward_inputs(batch, self.future_map)
                pooler_output, can_run_cuda_graph = (
                    self.tp_worker.forward_batch_embedding(batch)
                )
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                    can_run_cuda_graph=can_run_cuda_graph,
                )

        self._maybe_report_active_ranks()

        return ret

    def _maybe_report_active_ranks(self) -> None:
        if not (
            self.enable_dp_attention and get_exec().moe.elastic_ep_backend is not None
        ):
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        inst = ElasticEPStateManager.instance()
        if inst is not None and inst.active_ranks_cpu is not None:
            self.ipc_channels.send_to_tokenizer.send_output(
                ActiveRanksOutput(
                    status=[bool(x) for x in inst.active_ranks_cpu.tolist()]
                )
            )
        else:
            logger.debug("[Elastic EP] active rank state is unavailable")
            return

        model_runner = self.tp_worker.model_runner
        pending = model_runner._pending_elastic_scale_update
        if pending is not None:
            self.ipc_channels.send_to_tokenizer.send_output(pending)
            model_runner._pending_elastic_scale_update = None

    def _relay_forward_payload(
        self, future_indices: torch.Tensor, batch_result: GenerationBatchResult
    ) -> None:
        """Stash this iter's relay payload for next iter's resolve_forward_inputs."""
        if self.spec_algorithm.is_ngram():
            if batch_result.next_draft_input is not None:
                payload = RelayPayload.from_ngram(batch_result.next_draft_input)
                self.future_map.stash(future_indices, payload)
            return
        if batch_result.next_draft_input is not None:
            payload = RelayPayload.from_draft_input(batch_result.next_draft_input)
        elif batch_result.has_sampled_token_ids:
            payload = RelayPayload(bonus_tokens=batch_result.next_token_ids)
        else:
            return
        self.future_map.stash(future_indices, payload)

    def _copy_auxiliary_output_to_cpu(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        logits_output = result.logits_output
        if (
            logits_output is None
            or logits_output.auxiliary_device_output is None
            or result.auxiliary_host_output is not None
        ):
            return
        # PP transports the device output to the first rank before copying it.
        if self.ps.pp_size > 1:
            return
        if result.copy_done is not None:
            raise RuntimeError(
                "generation result has an uncopied auxiliary output after its "
                "device-to-host copy was scheduled"
            )
        result.copy_done = self.device_module.Event()
        result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult, cur_batch: ScheduleBatch
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            # Delay-sample is non-spec only; relays the sampled bonus tokens.
            self._relay_forward_payload(batch_result.future_indices, batch_result)

        # Run device-to-host copy on a separate stream to avoid blocking the
        # forward stream. The copy waits for the sampled result and can overlap
        # with subsequent forward computation.
        self.copy_stream.wait_stream(self.forward_stream)
        with self.copy_stream_ctx:
            batch_result.copy_to_cpu(
                return_logprob=cur_batch.return_logprob,
                return_hidden_states=cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    @scheduler_nvtx_method("scheduler.process_batch_result")
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        """处理一拍 GPU 前向的结果：回传 token、回收 KV、更新指标。

        decode / prefill / PD-prefill / prebuilt / idle 走不同 processor。
        overlap 模式下这段 CPU 工作与下一拍 GPU 前向重叠。trace flush
        放在这里也是为了吃这段空闲 CPU。
        """
        # Flush async trace ops here: in overlap mode this CPU work runs while
        # the next batch's GPU forward is in flight, giving free overlap.
        flush_trace_batch(batch.reqs)
        self.publish_load_snapshot(force=batch.forward_mode.is_extend())

        if batch.forward_mode.is_decode():
            self.batch_result_processor.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.batch_result_processor.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)

        self._record_step_counters(batch, result)

        self.metrics_reporter.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()
        self.metrics_reporter.update_device_timer()

    def _record_step_counters(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        mode = batch.forward_mode
        is_prefill = mode.is_extend_without_speculative()
        if not (is_prefill or mode.is_decode() or mode.is_target_verify()):
            return
        if all(is_health_check_generate_req(req) for req in batch.reqs):
            return
        prev = self._prev_step
        self._prev_step = (batch.forward_iter, batch.launch_ts, is_prefill)
        # An idle pass keeps forward_iter contiguous (forward_ct advances in run_batch).
        if prev is None or batch.after_idle_gap:
            return
        prev_iter, prev_ts, prev_is_prefill = prev
        if prev_iter + 1 != batch.forward_iter or prev_is_prefill != is_prefill:
            return
        step_us = int((batch.launch_ts - prev_ts) * 1e6)
        if not 0 < step_us < STEP_MAX_US:
            return
        if is_prefill:
            self.total_prefill_busy_us += step_us
            self.total_prefill_uncached_tokens += batch.extend_num_tokens
        else:
            batch_size = len(batch.reqs)
            _accumulate_decode_moment(
                self.decode_moment_totals,
                batch_size,
                step_us,
                batch_size + result.num_correct_drafts,
            )

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        # Flush any health-check signal deferred while the engine was busy.
        self.maybe_send_health_check_signal()

        if not self.is_fully_idle():
            return

        if self.enable_unified_memory:
            try:
                self.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp).
        # Also skipped while deferred KV releases are pending: they hold pages out
        # of the allocator by design, so the pool is transiently below `total` and
        # would trip the idle leak invariant. Resumes once the holds resolve.
        deferred_pending = (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_transfer_queue.has_pending_deferred_releases()
        )
        if not self.enable_hisparse and not deferred_pending:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # Publish the idle state so /get_loads and DP balancing do not see stale load.
        self.publish_load_snapshot(force=True)

        # sleep until next event
        self.maybe_sleep_on_idle()

    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if (
            for_health_check
            and not self._engine_paused
            and self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None
        ):
            idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0
                    if self.server_args.hicache_host_memory_mode == "buffer_only":
                        # Queued writes, staged prefetches, and in-flight
                        # storage writes still hold host staging
                        # (buffer-mode unified tree only).
                        idle &= tc.buffer_pipeline.is_idle()

        return idle

    def _pp_microbatches_drained(self) -> bool:
        if self.ps.pp_size == 1:
            return True
        return all(x.is_empty() for x in self.running_mbs) and all(
            mb is None or mb.is_empty() for mb in self.mbs
        )

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=get_serving().served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            hicache_fields = {
                "hicache_storage_backend": recv_req.hicache_storage_backend
            }
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = recv_req.hicache_write_policy
            get_context().override("scheduler.attach_hicache", **hicache_fields)
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            get_context().override(
                "scheduler.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.is_fully_idle():
            self.cur_batch_for_debug = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.req_to_token_pool.reset_aux_cache_allocator()
            self.grammar_manager.clear()
            self.metrics_reporter.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_reporter.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}"
            )
            success = False
        return success

    def get_internal_state(self, recv_req: GetInternalStateReq):
        # Resolved config (pristine server_args + post-publish overrides) so a
        # readback reflects values changed via /set_internal_state, not startup.
        ret = get_context().resolved_server_args_dict()
        ret["world_size"] = compute_world_size(get_server_args())
        ret["last_gen_throughput"] = self.metrics_reporter.last_gen_throughput
        draft_graph_memory_usage = (
            None if self.draft_worker is None else self.draft_worker.graph_memory_usage
        )
        ret["memory_usage"] = build_memory_usage(
            weight_gb=self.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_gb=self.token_to_kv_pool_allocator.get_kvcache().mem_usage,
            startup_available_gb=self.startup_available_gpu_memory_gb,
            token_capacity=self.max_total_num_tokens,
            token_capacity_swa=self.swa_tokens_per_layer,
            target_graph_memory_usage=self.tp_worker.graph_memory_usage,
            draft_graph_memory_usage=draft_graph_memory_usage,
        )
        ret["startup_time"] = self.startup_time
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests

        if get_exec().moe.elastic_ep_backend is not None:
            from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

            ret["is_scaling_elastic_ep"] = ElasticEPStateManager.is_scaling()
            ret["effective_ep_size"] = ElasticEPStateManager.get_effective_ep_size()
            ret["pending_ep_size"] = ElasticEPStateManager.get_pending_ep_size()
            ret["scale_phase"] = ElasticEPStateManager.get_scale_phase()
            ret["elastic_ep_last_error"] = ElasticEPStateManager.get_last_error()

        if (
            not self.spec_algorithm.is_none()
            and self.metrics_reporter.spec_total_num_forward_ct > 0
        ):
            ret["avg_spec_accept_length"] = (
                self.metrics_reporter.spec_total_num_accept_tokens
                / self.metrics_reporter.spec_total_num_forward_ct
            )

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict

        if self.spec_algorithm.is_dspark() and self.draft_worker is not None:
            info_record = self.draft_worker.dump_info_records()
            if info_record is not None:
                ret["dspark_info_record"] = info_record

        if envs.SGLANG_EXPOSE_OWN_ENV_VARS.get():
            ret["env_vars"] = exportable_env_vars()

        # These fields are not msgpack-serializable (a config object and a bound
        # signal handler); no reader consumes them.
        ret.pop("model_config", None)
        ret.pop("custom_sigquit_handler", None)

        return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(ret))

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
                "dspark_force_budget_frac",
                "dspark_clear_info_records",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.ps.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.ps.pp_size}]."
                )
                if_success = False
                break
            elif k == "dspark_force_budget_frac":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "set_dspark_forced_budget_frac"
                ):
                    logging.warning(
                        "dspark_force_budget_frac requires a DSpark draft worker."
                    )
                    if_success = False
                    break
                if v is not None and not (0.0 < float(v) <= 1.0):
                    logging.warning(
                        f"dspark_force_budget_frac must be in (0, 1] or null, got {v}."
                    )
                    if_success = False
                    break
            elif k == "dspark_clear_info_records":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "clear_info_records"
                ):
                    logging.warning(
                        "dspark_clear_info_records requires a DSpark draft worker."
                    )
                    if_success = False
                    break

        if if_success:
            if (
                not self.spec_algorithm.is_none()
                and self.metrics_reporter.spec_total_num_forward_ct > 0
            ):
                avg_spec_accept_length = (
                    self.metrics_reporter.spec_total_num_accept_tokens
                    / self.metrics_reporter.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.metrics_reporter.spec_total_num_accept_tokens = (
                self.metrics_reporter.spec_total_num_forward_ct
            ) = 0
            # DSpark control keys are worker commands, not server args; route
            # them to the draft worker and keep them out of the override.
            remaining = dict(server_args_dict)
            frac = remaining.pop("dspark_force_budget_frac", None)
            if "dspark_force_budget_frac" in server_args_dict:
                self.draft_worker.set_dspark_forced_budget_frac(
                    None if frac is None else float(frac)
                )
            if remaining.pop("dspark_clear_info_records", None):
                self.draft_worker.clear_info_records()
            if remaining:
                get_context().override(source="update_server_args", **remaining)
            logger.info(f"Config updated via context override: {remaining}")

        return SetInternalStateReqOutput(updated=if_success)

    def save_remote_model(self, **kwargs):
        self.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier(group=self.tp_group.cpu_group)
        return RpcReqOutput(success=success, message="" if not exec else str(exec))

    def handle_update_weight_version(
        self, recv_req: UpdateWeightVersionReqInput
    ) -> UpdateWeightVersionReqOutput:
        self.record_weight_version_change(new_version=recv_req.new_version)
        return UpdateWeightVersionReqOutput()

    def record_weight_version_change(self, new_version: Optional[str]) -> None:
        if new_version is None or new_version == get_serving().weight_version:
            return

        old_version = get_serving().weight_version
        get_context().override("scheduler.weight_version", weight_version=new_version)

        live_reqs = {
            *self.collect_inflight_reqs(),
            *self.waiting_queue,
            *([self.chunked_req] if self.chunked_req is not None else []),
        }
        if self.hisparse_coordinator is not None:
            live_reqs.update(
                act.req for act in self.hisparse_coordinator.ack_staging_queue
            )
        num_recorded = record_weight_version_events(live_reqs, old_version=old_version)
        logger.info(
            f"Weight version changed. {old_version=} {new_version=} {num_recorded=}"
        )

    def collect_inflight_reqs(self) -> Set[Req]:
        if self.ps.pp_size == 1:
            inflight_batches = [self.running_batch, self.last_batch]
        else:
            inflight_batches = [*self.running_mbs, *self.mbs]
        return {
            req for batch in inflight_batches if batch is not None for req in batch.reqs
        }

    def abort_request(self, recv_req: AbortReq):
        if (chunked_req := self.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self._pending_chunked_abort_req = chunked_req

        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Abort requests still waiting for encoder embeddings (EPD language-only)
        if self.mm_receiver is not None:
            self.mm_receiver.abort_waiting_requests(recv_req)

        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(_make_abort_req(req), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                bootstrap_pending = req.pending_bootstrap
                maybe_release_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )
                if (
                    bootstrap_pending
                    and hasattr(req, "disagg_kv_sender")
                    and req.disagg_kv_sender is not None
                ):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        if self.dllm_config is not None:
            for req in self.dllm_manager.pop_aborted_reqs(
                recv_req.abort_all, recv_req.rid
            ):
                if self.enable_hicache_storage:
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    _make_abort_req(req), req
                )
                if (
                    req.req_pool_idx is not None
                    or getattr(req, "mamba_pool_idx", None) is not None
                ):
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                logger.debug(f"Abort dLLM queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if self.enable_hicache_storage:
                        self.tree_cache.release_aborted_request(req.rid)

                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()
                    if self.ps.pp_size > 1:
                        prepare_abort(req, "Aborted by AbortReq.")

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()
                    if self.ps.pp_size > 1:
                        prepare_abort(decode_req.req, "Aborted by AbortReq.")

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    receiver = decode_req.kv_receiver
                    receiver.abort()
                    # Arm drain-ack accounting once the ABORT is sent, so acks
                    # arriving before this req is deferred (e.g. during the next
                    # forward step) are captured. A fresh set also drops stale acks
                    # from a prior request that reused this bootstrap_room. A
                    # redundant abort only re-wipes -- holds longer, never releases
                    # early -- so no transition guard is needed.
                    if (
                        receiver.kv_mgr.enable_deferred_decode_kv_release
                        and receiver.abort_notified
                    ):
                        receiver.kv_mgr.register_deferred_abort_room(
                            decode_req.req.bootstrap_room
                        )

            # Abort requests whose KV is already backed up for retraction.
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        retraction_discard(
                            decode_req,
                            self.tree_cache,
                            get_disagg().disaggregation_decode_retraction_backup,
                        )
                        self.ipc_channels.send_to_tokenizer.send_output(
                            _make_abort_req(decode_req), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        for req in self.collect_inflight_reqs():
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        assert recv_req.mode in ("in_place", "retract")
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        retract_reqs = [r for r in self.running_batch.reqs if not r.finished()]
        if (
            self.last_batch is not None
            and self.last_batch.forward_mode.is_extend()
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            retract_reqs += [r for r in self.last_batch.reqs if not r.finished()]

        if (
            self.chunked_req is not None
            and not self.chunked_req.finished()
            and self.chunked_req not in retract_reqs
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            retract_reqs.append(self.chunked_req)

        self.last_batch = None
        self.cur_batch_for_debug = None

        if retract_reqs:
            # Decode-side retract always rebootstraps (recomputes the KV from
            # the prefill), so skip the device->host KV offload that release_req
            # would otherwise do; the offloaded copy would be immediately
            # discarded. Non-decode modes ignore offload_kv (they never offload).
            retract_all(
                reqs=retract_reqs,
                server_args=self.server_args,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                hisparse_coordinator=self.hisparse_coordinator,
                offload_kv=False,
            )
        self.running_batch.reqs = []
        for req in retract_reqs:
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if req.output_ids:
                    req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
                req.pd_rebootstrap_in_progress = True
                req.time_stats.set_retract_time()
                self.disagg_decode_prealloc_queue.hold_rebootstrap(req)
            else:
                self._add_request_to_queue(req)
        self.running_batch.batch_is_full = False
        # In disagg-PREFILL, keep a live mid-chunk chunked_req rather than retract it:
        # freeing its KV under a live disagg KV-sender crashes pop_bootstrapped or
        # sends freed/reused KV to decode. Kept, it resumes prefill after the pause.
        # TODO(disagg-prefill-retract): tear the sender down (abort + release metadata
        # buffer + reset pending_bootstrap) before freeing KV, then retract for real.
        # Until then a weight-update pause leaves stale-weight prefix KV (off-policy).
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            self.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_reporter.last_gen_throughput = 0.0
        if self.metrics_reporter.current_scheduler_metrics_enabled:
            self.metrics_reporter.metrics_collector.last_log_time = 0.0
            self.metrics_reporter._maybe_log_idle_metrics()
        self.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        # Enqueue any rebootstrap requests that were staged during a
        # retract-mode pause. Deferring until resume keeps the preallocation
        # queue empty during the pause window (so an intervening weight update
        # can flush the cache) and recomputes the prefix KV under the updated
        # weights.
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None
        ):
            self.disagg_decode_prealloc_queue.enqueue_held_rebootstrap()
        self._engine_paused = False

    def handle_scale_elastic_ep(
        self, recv_req: ScaleElasticEPReqInput
    ) -> ScaleElasticEPReqOutput:
        """Begin a pending elastic EP scale-up request."""
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        old_ep_size = ElasticEPStateManager.get_effective_ep_size()
        new_ep_size = recv_req.new_ep_size
        max_ep_size = get_parallel().max_ep_size or old_ep_size

        logger.debug(
            "[Elastic EP][scale] request received: new_ep_size=%d "
            "old_ep_size=%d max_ep_size=%d",
            new_ep_size,
            old_ep_size,
            max_ep_size,
        )

        if new_ep_size <= old_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) must be greater than current "
                    f"effective_ep_size ({old_ep_size})."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if new_ep_size > max_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) exceeds --max-ep-size "
                    f"({max_ep_size}). Restart with a larger --max-ep-size."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if ElasticEPStateManager.is_scaling():
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "A previous scale operation has not completed yet. Wait until "
                    "all pending ranks have joined before issuing another scale."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )

        if not ElasticEPStateManager.request_scale(new_ep_size):
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "Failed to queue elastic EP scale: no elastic state or "
                    "scale already pending."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )
        if (eplb_manager := self.tp_worker.model_runner.eplb_manager) is not None:
            eplb_manager.disable_rebalance("elastic EP scale-up is pending")
        logger.debug(
            "[Elastic EP][scale] scale requested: target_ep_size=%d; "
            "waiting for a joining cohort",
            new_ep_size,
        )

        return ScaleElasticEPReqOutput(
            success=True,
            message=f"Scaling initiated from {old_ep_size} to {new_ep_size}",
            old_ep_size=old_ep_size,
            new_ep_size=new_ep_size,
            pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
            scale_phase=ElasticEPStateManager.get_scale_phase(),
        )

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(
            success=success, message=message
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success=success, message=message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.session_controller.open(recv_req)
        if output.success and self.enable_session_radix_cache:
            self.tree_cache.open_radix_session(recv_req.session_id)
        if self.ps.pp_rank == 0 and self.ps.tp_rank == 0 and self.ps.attn_cp_rank == 0:
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        if self.enable_session_radix_cache:
            self.tree_cache.release_radix_session(recv_req.session_id)
        if (
            recv_req.session_id in self.session_controller
            or not self.enable_session_radix_cache
        ):
            self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        # 只置位，不在这里释放 GPU：事件循环还要跑完当前拍，finally 里统一清理。
        self.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


def dispatch_event_loop(scheduler: Scheduler):
    """按并行模式和 PD-disagg 角色选出真正的事件循环实现。

    主循环有多套变体（normal / overlap / PP / PD-prefill / PD-decode /
    pdmux / MLX），是因为它们的「一拍」语义不同：PP 要推进 microbatch，
    PD-prefill 要驱动 KV 传输，overlap 要维护 result_queue。把选择集中
    在这里，``run_event_loop`` 只负责 stream 上下文。
    """
    # The live PP property asserts before torch.distributed init (MLX stub).
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif configured_pp_size() > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if configured_pp_size() > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if configured_pp_size() > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
) -> Optional[int]:
    """Configure scheduler worker logging and process title.

    display_* ranks are cosmetic; runtime ranks stay local.

    中文：子进程落地后、构造 Scheduler 之前的进程级环境配置。

    必须在这里做而不是在父进程做：进程名、logger 前缀、CPU affinity、
    NUMA bind 都是 per-process 的。父进程 fork/spawn 出几十个 rank 后，
    若日志都叫 ``scheduler``，排障几乎不可能。

    ``display_*`` 只影响日志/进程名中展示的 rank（例如 Elastic EP 或
    多节点时对外编号与本地 tp_rank 不同），真正参与集合通信的仍是
    传入的运行时 rank。Router 场景下若环境变量 ``SGLANG_DP_RANK`` 存在，
    会在这里补上 ``dp_rank``。

    返回可能被环境变量改写过的 ``dp_rank``，调用方必须用返回值，
    不能继续用入参。
    """
        kill_itself_when_parent_died()

        # Generate the logger prefix
        # Router 场景下 Engine 不传 dp_rank，靠环境变量把本进程标成某个 DP 副本。
        if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
            # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
            dp_rank = int(os.environ["SGLANG_DP_RANK"])

        shown_dp = display_dp_rank if display_dp_rank is not None else dp_rank
        shown_tp = display_tp_rank if display_tp_rank is not None else tp_rank
        shown_moe_ep = (
            display_moe_ep_rank if display_moe_ep_rank is not None else moe_ep_rank
        )

        prefix = ""
        if shown_dp is not None:
            prefix += f" DP{shown_dp}"
        if configured_pp_size() > 1:
            prefix += f" PP{pp_rank}"
        if configured_attn_cp_size() > 1:
            prefix += f" ATTN_CP{attn_cp_rank}"
        if configured_moe_dp_size() > 1:
            prefix += f" MOE_DP{moe_dp_rank}"
        if configured_tp_size() > 1:
            prefix += f" TP{shown_tp}"
        if get_parallel().ep_size > 1:
            prefix += f" EP{shown_moe_ep}"

        # Config the process
        # 进程名带上 DP/TP/PP 前缀，ps/top 才能区分几十个 scheduler 子进程。
        setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
        faulthandler.enable()

        # Configure the logger
        configure_logger(server_args, prefix=prefix)
        suppress_other_loggers()

        # Set cpu affinity to this gpu process
        # 把本进程绑到靠近这张 GPU 的 CPU/NUMA 节点，降低 PCIe/HBM 拷贝延迟。
        if envs.SGLANG_SET_CPU_AFFINITY.get():
            set_gpu_proc_affinity(
                configured_pp_size(), configured_tp_size(), get_parallel().nnodes, gpu_id
            )
        if not envs.SGLANG_NUMA_BIND_V2.get():
            numa_node = get_numa_node_if_available(server_args, gpu_id)
            if numa_node is not None:
                numa_bind_to_node(numa_node)

        return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
):
    """每个 Scheduler 子进程的 ``multiprocessing.Process`` target。

    为什么需要独立进程
    ------------------
    Tokenizer / HTTP 在 CPU 上跑，模型前向必须钉在一张 GPU 上。把二者
    放进同一进程会互相抢 GIL、共享崩溃域，也无法给每个 TP/PP rank 绑
    不同的 ``CUDA_VISIBLE_DEVICES``。因此 Engine（或 DataParallelController）
    为每个 (DP, TP, PP, CP, EP) 组合 spawn 一个进程，本函数就是该进程
    的入口。Ray 模式下 ``SchedulerActor`` 扮演同一角色，不再走 mp.Process。

    谁调用它
    --------
    * ``dp_size == 1``：``Engine._launch_scheduler_processes`` 直接
      ``mp.Process(target=run_scheduler_process, args=...)``。
    * ``dp_size > 1`` 或 Elastic EP：Engine 只拉起 DataParallelController，
      由控制器再为本节点的每个副本调用本函数。
    * 测试可注入 ``run_scheduler_process_func``，生产路径默认就是本函数。

    启动握手架构::

        Engine / DP Controller          Scheduler 子进程（本函数）
        ----------------------          -------------------------
        mp.Process.start()  ─────────►  load_plugins()
        阻塞读 pipe_reader              publish(server_args)
                                        configure_scheduler_process(...)
                                        Scheduler.__init__  # 加载权重/KV/Graph
        ◄──── pipe_writer.send(ready) ──  get_init_info()
        Tokenizer 开始接流量             run_event_loop()  # 直到 ShutdownReq
                                        finally: 释放 host 资源

    父进程必须等到 ``status=ready`` 才对外服务：模型加载失败时本函数
    向父进程发 SIGQUIT，而不是让 HTTP 端口先亮绿灯再在第一批请求上崩。

    执行顺序
    --------
    1. ``load_plugins()``：让第三方 hook 能替换 Scheduler / worker 类。
    2. ``publish(..., role="scheduler")``：本进程的运行时上下文必须在
       任何 bag getter 之前发布；fork 后不能复用父进程的 publish。
    3. ``configure_scheduler_process``：进程名、日志前缀、affinity、NUMA。
    4. 可选 OTLP tracing，按 PD 角色区分 Prefill/Decode Scheduler。
    5. 构造 ``Scheduler``（同步加载模型，可能耗时数分钟）。
    6. ``pipe_writer.send(get_init_info())``：告诉父进程容量上限和启动耗时。
    7. ``run_event_loop()``：永久循环，直到 ``ShutdownReq`` 置
       ``gracefully_exit``。
    8. 异常：打栈、向父进程 SIGQUIT；可选 ``killpg`` 避免兄弟 rank 在
       NCCL 超时前喷出大量 traceback。
    9. finally：关掉 metrics 后台线程；仅优雅退出时 ``release_host_resources``，
       异常路径 GPU 可能已经卡住，``synchronize()`` 自己也会挂。

    参数含义
    --------
    server_args:
        完整 ``ServerArgs``。子进程通过 ``publish`` 变成进程内全局配置。
        父进程传入的是一份快照，后续 HTTP 改 CLI 不会穿透到已启动的 rank。
    port_args:
        本实例（或本 DP 副本）的 ZMQ IPC 路径和 NCCL 初始化端口。
        多 DP 时每个副本有自己的 ``PortArgs``，否则 tokenizer 无法把
        请求送到正确的 Scheduler。
    gpu_id:
        本进程绑定的 GPU 逻辑编号，已经过 ``maybe_reindex_device_id``
        根据 ``CUDA_VISIBLE_DEVICES`` 重映射。``tp_rank`` 不是设备号：
        多节点 / ``base_gpu_id`` / ``gpu_id_step`` 都会让二者不同。
    tp_rank:
        全局张量并行 rank，范围 ``[0, tp_size)``。也是推导 ATTN_CP /
        MOE_DP / MOE_EP 的主键（见 Engine 的 ``_compute_parallelism_ranks``）。
    attn_cp_rank:
        Attention context-parallel rank。把长序列 KV 切到多个 rank；
        关闭 CP 时为 0。与 ``tp_rank`` 正交：同一 tp_rank 在不同 CP
        组里仍要能对齐 attention。
    moe_dp_rank:
        MoE 数据并行 rank，即 expert mesh 的副本号。和 serving 的
        ``dp_rank`` 不是同一维度：前者复制专家，后者复制整份调度队列。
    moe_ep_rank:
        Expert-parallel rank，专家被切分到哪些卡上。日志里的 ``EP{n}``
        用的就是它（或下面的 ``display_moe_ep_rank``）。
    pp_rank:
        流水线并行阶段。0 收 tokenized 输入并持有 tokenizer ZMQ；
        最后一阶段产出 logits。PP=1 时永远是 0。
    dp_rank:
        数据并行副本 id。Engine 直接 spawn 时传 ``None``（单 DP）；
        DataParallelController spawn 时传入具体副本号。Router 场景
        也可能靠 ``SGLANG_DP_RANK`` 在 ``configure_scheduler_process``
        里补上。
    pipe_writer:
        ``mp.Pipe(duplex=False)`` 的写端。子进程只能 send、父进程只能
        recv。用来回传 ``get_init_info()``，不是业务 token 通道
        （业务走 ZMQ）。握手失败则父进程永远等不到 ready。
    display_tp_rank / display_dp_rank / display_moe_ep_rank:
        仅用于进程名和日志前缀的「对外编号」。Elastic EP、多节点或
        聚合显示时，本地 rank 与用户看到的 rank 可能不同；运行时
        集合通信仍用上面那些真实 rank。默认 ``None`` 表示跟运行时
        rank 一致。
    """
    # Load plugins so hooks can override Scheduler and its dependencies.
    # 必须在 publish / 构造 Scheduler 之前：插件可能替换 Scheduler 类本身。
    load_plugins()
    # Publish before anything in this process reads configuration.
    # fork 后的子进程有自己的 runtime context，不能复用父进程的 publish。
    publish(server_args, role="scheduler")
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )
    # 异常时要主动叫醒还在 pipe.recv() 上阻塞的父进程，否则启动会永久卡住。
    parent_process = psutil.Process().parent()

    # Set up tracing
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            "sglang",
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "Scheduler"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    # 构造函数会加载权重并捕获 CUDA Graph，是启动路径上最容易失败的一步。
    scheduler = None
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # Send initialization info back to the parent process
        # 父进程以此结束握手并决定 Tokenizer 的 max_req_input_len。
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            # 异常路径不做 GPU synchronize：设备可能已经 hang，释放反而死锁。
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()


def _make_abort_req(
    req: Req, finished_reason: Optional[FinishReasonDict] = None
) -> AbortReq:
    """构造回给 Tokenizer 的中止消息，带上 weight version 跨度。

    Scheduler 不直接关 HTTP 连接：它只发 AbortReq，由 TokenizerManager
    把 abort 转成客户端可见的 finish_reason。
    """
    return AbortReq(
        rid=req.rid,
        finished_reason=finished_reason,
        weight_versions=compute_weight_version_spans(
            req.weight_version_events,
            current_version=get_serving().weight_version,
            num_output_tokens=len(req.output_ids),
        ),
    )
