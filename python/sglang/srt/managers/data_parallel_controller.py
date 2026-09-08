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
"""数据并行（DP）请求控制器。

当 ``--dp`` 大于 1 时，SGLang 会启动多组彼此独立的 Scheduler/模型副本。每一组
DP worker 都有自己的调度队列、运行中请求以及 KV Cache；Tokenizer/API 一侧却仍需要
一个统一的入口。因此本模块在二者之间增加一个独立的控制进程，负责把已经 tokenized
的请求稳定地送到 *一组且仅一组* DP worker：

.. code-block:: text

    Tokenizer (node 0) --PULL--> DataParallelController --PUSH--> DP0 Scheduler
                                                        ├--------> DP1 Scheduler
                                                        └--------> ...

控制器不执行模型前向、也不汇聚 token 输出；这些工作仍由 Scheduler、Detokenizer 和
Tokenizer 之间既有的 IPC 通道完成。它存在的主要原因是将“入口只有一个”与“可执行
副本有多个”解耦，并在一个位置集中实现负载均衡、请求亲和性、控制消息广播和 Elastic
EP 的可用 worker 管理。

本文件同时负责启动各 DP Scheduler 子进程。这样 Engine 只需要监督一个控制器进程，
控制器则能够在所有模型副本真正加载完成后，向其父进程一次性回报容量、启动耗时和
子进程 PID。
"""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ElasticScaleUpdateReq,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_reader
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.startup_time import aggregate_scheduler_startup_times
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.runtime_context import (
    configured_attn_cp_size,
    configured_moe_dp_size,
    configured_pp_size,
    get_device,
    get_disagg,
    get_exec,
    get_parallel,
    publish,
)
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)

# 控制器通过启动就绪 Pipe 回传给 Engine 的字段名。DP Scheduler 是控制器的子进程，
# Engine 仍需要这些 PID 来统一监督和退出清理。
SCHEDULER_PIDS_ARG = "scheduler_pids"


class LoadBalanceMethod(Enum):
    """DP 请求到 worker 的选择策略。

    ``ROUND_ROBIN`` 和 ``FOLLOW_BOOTSTRAP_ROOM`` 不读取 Scheduler 的实时负载；
    后两种策略会读取共享内存中的负载快照。这里使用枚举而非在热路径反复比较字符串，
    既在启动时尽早校验配置，也使 ``dispatching`` 可以直接保存对应的方法引用。
    """

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        """将用户配置规范化为枚举，并在启动阶段拒绝未知策略。"""
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    """控制器侧的轻量负载账本。

    Scheduler 会通过共享内存发布当前运行/等待请求数和 token 数；控制器把该快照转换为
    每个 DP rank 的可比较预算。快照天生存在传播延迟，所以每次选中 worker 后还会在
    本地预扣（准确说是预加）本次请求的负载，以避免一个突发中的后续请求都看到同一个
    旧快照并集中落到同一 rank。这不是严格的队列真值，下一次有效快照会覆盖它。
    """

    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.last_timestamp = [0.0] * dp_size

    def update_budget(self, loads):
        """用共享内存快照刷新账本，忽略时间戳未变化的旧读数。

        ``read_all`` 在控制器高频分发时可能多次读到同一份快照。跳过相同时间戳可避免
        无意义地回退本地的推测增量；一旦某个 Scheduler 发布新快照，则以该 rank 的真实
        观测值为准。
        """
        for load in loads:
            if load.timestamp == self.last_timestamp[load.dp_rank]:
                continue
            self.last_timestamp[load.dp_rank] = load.timestamp
            self.total_requests[load.dp_rank] = (
                load.num_running_reqs + load.num_waiting_reqs
            )
            self.total_tokens[load.dp_rank] = load.num_total_tokens

    def dispatch(self, method: LoadBalanceMethod, estimated_tokens: int = 0):
        """选择当前最空闲的 DP rank，并记录本次尚未出现在快照中的请求。

        ``TOTAL_TOKENS`` 以 token 总量为主键、请求数为次键，避免 token 相同的 rank
        始终因列表顺序获选。这里的选择维度刻意仅限于负载账本，不包含 socket/健康状态
        校验；它负责“哪个 rank 的估计负载更低”，而路由端点和 active-rank 的生命周期
        由控制器其他状态管理逻辑负责。
        """
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        self.total_tokens[target_rank] += estimated_tokens
        return target_rank


class DataParallelController:
    """单一入口到多个 DP Scheduler 的路由与生命周期管理者。

    生命周期分两层：本对象在控制器进程内持有 ZMQ socket、负载账本和 worker 状态；
    ``launch_tensor_parallel_group`` 再为每个 TP/PP rank 创建真正执行模型的
    Scheduler 子进程。只有 node 0 绑定 Tokenizer 的输入 socket 并运行事件循环，
    其他节点参与启动本地 rank，避免多节点从同一请求入口重复消费消息。

    路由不应把同一个 ``Req`` 复制给多个 DP 副本。除请求外，暂停、profiling 等控制
    消息需要广播；DP-attention 下广播对象则可能只需要发送给各组 leader，再由组内
    collective 扩散。
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # 将 CLI/运行时上下文固定在控制器中。负载均衡策略只在启动时解析一次；后续请求
        # 路径直接调用已选好的函数，避免热路径解析字符串。
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            get_parallel().load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # 一个控制器对每个 Scheduler 保留一个发送通道，并在 node 0 额外保留一个来自
        # Tokenizer 的接收通道。非 node-0 节点不绑定 PULL socket：全局入口只有一个，
        # 否则同一请求会被多个节点竞争消费。
        self.context = zmq.Context(1 + get_parallel().dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # 将策略映射为可调用对象，并标记是否需要在每批请求前刷新共享内存负载。
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )

        # ``launch_dp_size`` 是初始实际启动的 DP 副本数；Elastic EP 可以预留更多 slot，
        # 但未加入的 slot 必须保持不可路由，直到显式激活。
        self.launch_dp_size: int = get_parallel().dp_size
        self.max_dp_size: int = server_args.max_ep_size or get_parallel().dp_size
        assert self.max_dp_size >= self.launch_dp_size, (
            f"--max-ep-size ({self.max_dp_size}) must be >= "
            f"--dp ({self.launch_dp_size})."
        )

        self.dp_active: List[bool] = [True] * self.launch_dp_size + [False] * (
            self.max_dp_size - self.launch_dp_size
        )

        # 负载快照只服务于 TOTAL_REQUESTS/TOTAL_TOKENS 两种策略；其他策略不读取它。
        self.dp_budget = DPBudget(get_parallel().dp_size)
        self.load_snapshot_reader = create_load_snapshot_reader(
            port_args,
            caller="DataParallelController",
        )
        self._last_refresh_time = 0.0

        # 启动子进程时可能临时改写进程级 CUDA_VISIBLE_DEVICES。多个 DP 组会并发启动，
        # 所以必须串行化这段全局环境变量操作，防止一个 rank 继承另一个 rank 的设备选择。
        self.env_lock = threading.Lock()

        # ``workers`` 是 controller -> scheduler 的 PUSH socket；``scheduler_procs`` 是
        # 真正承载模型的子进程。status 由运行时健康/active-rank 消息更新，dp_active 则
        # 表示 Elastic EP 维度上该 slot 是否已被加入。
        self.scheduler_procs = []
        self.workers: List[Optional[zmq.Socket]] = [None] * self.max_dp_size
        self.status: List[bool] = list(self.dp_active)
        self._active_workers: List[int] = list(range(self.launch_dp_size))
        self._active_count_cache: int = self.launch_dp_size

        if get_parallel().enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # DP attention 将 TP group 再分成多个 attention DP group。若开启 local
            # control broadcast，必须把控制消息发给每组 leader（步长 1），再由各组在
            # 本地广播；否则只发给第一个 leader（步长 tp_size），沿用全 TP group 广播。
            local_ctrl = get_parallel().enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=get_device().soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        """向所有当前状态可用的 worker 广播非普通请求消息。

        Block/Profile 类消息必须在每个模型副本执行，而生成/embedding 请求只能选择一个
        副本，二者不能共用普通 ``dispatching`` 路径。
        """
        for i, worker in enumerate(self.workers):
            if worker is not None and self.status[i]:
                sock_send(worker, obj)

    def send_control_message(self, obj):
        """将未显式注册类型的控制消息发给适当的 group leader。

        普通 DP 时每个活跃 worker 都是独立副本；DP-attention 的 ``control_message_step``
        则控制只挑 leader 还是逐组发送，详见构造函数中的设置。
        """
        for i in self._active_workers[:: self.control_message_step]:
            worker = self.workers[i]
            if worker is not None:
                sock_send(worker, obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        """消费 Scheduler 侧的可用 rank 状态，拒绝长度不匹配的陈旧/错误消息。

        Elastic EP 下最终可路由性是 ``dp_active AND scheduler-reported-status``：一个
        尚未加入的预留 slot 即使错误报告为健康也不能接流量。普通 DP 不会改变 slot
        集合，仅保存 Scheduler 报告的状态。
        """
        if get_exec().moe.elastic_ep_backend is not None:
            if len(ranks.status) != self.max_dp_size:
                logger.warning(
                    "[Elastic EP][DPC] active rank status len=%d != max_dp_size=%d; "
                    "ignoring update",
                    len(ranks.status),
                    self.max_dp_size,
                )
                return
            self.status = [
                self.dp_active[i] and bool(ranks.status[i])
                for i in range(self.max_dp_size)
            ]
            self._refresh_active_workers()
            return
        if len(ranks.status) != self.max_dp_size:
            logger.warning(
                "[DPC] update_active_ranks: status len=%d != max_dp_size=%d; "
                "ignoring update",
                len(ranks.status),
                self.max_dp_size,
            )
            return
        self.status = list(ranks.status)

    def add_elastic_workers(self, slot_offset: int, slot_count: int):
        """激活一段在启动时已预绑定 socket 的 Elastic EP worker slot。

        扩容不会在热路径新建端口或改写 ``workers`` 数组，而是只启用初始为 False 的
        预留槽位；这样已有路由端点保持稳定。越界和“socket 未预绑定”都立即失败，防止
        控制器向不存在的 Scheduler 静默发送请求。
        """
        end = slot_offset + slot_count
        if end > self.max_dp_size:
            raise ValueError(
                f"[Elastic EP] add_elastic_workers: slot_offset={slot_offset} + "
                f"slot_count={slot_count} exceeds max_dp_size={self.max_dp_size}. "
                f"Restart with a larger --max-ep-size."
            )

        for slot in range(slot_offset, end):
            if self.dp_active[slot]:
                logger.debug(
                    "[Elastic EP] add_elastic_workers: slot %d already active; "
                    "skipping",
                    slot,
                )
                continue
            assert self.workers[slot] is not None, (
                f"[Elastic EP] add_elastic_workers: slot {slot} was not "
                f"pre-bound at launch; expected a primary-bound PUSH socket."
            )
            self.dp_active[slot] = True
            self.status[slot] = True

        self._refresh_active_workers()
        logger.debug(
            "[Elastic EP] DataParallelController activated slots %s "
            "(active=%d / max=%d)",
            list(range(slot_offset, end)),
            self._active_count_cache,
            self.max_dp_size,
        )

    def _refresh_active_workers(self) -> None:
        """根据静态加入状态和动态健康状态重建可被轮询的 rank 列表。"""
        self._active_workers = [
            i for i, active in enumerate(self.dp_active) if active and self.status[i]
        ]
        self._active_count_cache = len(self._active_workers)

    def refresh_load_budget(self):
        """限频读取 Scheduler 负载快照，保留突发内的本地推测计数。

        ``dispatching_with_trace`` 会在每个请求前调用此函数，但最多每 20 ms 真正读取
        一次。若不节流，同一突发中的每个请求都会用尚未反映前一请求的旧快照覆盖
        ``DPBudget.dispatch`` 的 ``+1`` 推测量，结果是整批流量连续选中同一 DP rank。
        20 ms 窗口内让本地账本连续决策；窗口结束后再以 Scheduler 的真实观测值校正。
        """
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())

    def dispatching_with_trace(self, req: Req, refresh_load_budget: bool = True):
        """记录控制器路由耗时，并把单个请求交给当前负载均衡策略。

        发送前将 ``time_stats`` 包装为可跨进程传输的对象；路由完成后恢复本地对象并记录
        结束时刻。批量路径会先刷新一次快照，再以 ``False`` 调用本函数，保证一批请求
        看到同一轮负载基线而非逐条重复读取共享内存。
        """
        if refresh_load_budget and self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()

        time_stats = DPControllerReqTimeStats.new_from_obj(
            unwrap_from_pickle(req.time_stats)
        )

        time_stats.set_dp_dispatch_time()
        req.time_stats = wrap_as_pickle(time_stats)
        self.dispatching(req)
        req.time_stats = time_stats
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        """按批分发生成请求；负载型策略每批只刷新一次快照。"""
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        """按批分发 embedding 请求；路由语义与生成请求一致。"""
        if self.refresh_load_budget_on_dispatch:
            self.refresh_load_budget()
        for req in batch_req:
            self.dispatching_with_trace(req, refresh_load_budget=False)

    def init_dispatcher(self):
        """建立消息类型到路由动作的边界。

        Tokenized 请求走“单副本选择”；批量请求逐项选择；需要所有副本一致执行的
        Block/Profile 消息显式广播。其余控制消息使用 fallback，交给 group leader 扩散，
        避免新增控制消息类型时被当成普通用户请求。
        """
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (ActiveRanksOutput, self.update_active_ranks),
                (
                    ElasticScaleUpdateReq,
                    lambda msg: self.add_elastic_workers(
                        msg.slot_offset, msg.slot_count
                    ),
                ),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        """为传统 DP 启动多个完整的 TP/PP Scheduler 副本。

        每个 ``dp_rank`` 都使用独立的 ``PortArgs`` 和连续的一段 GPU；node 0 为其创建
        对应的 PUSH socket。NCCL 端口会先被占住，直到所有 DP rank 计算完各自端口，
        用来消除并发初始化时重复选择“当前空闲端口”的竞态。
        """
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(get_parallel().dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            tmp_port_args.instance_id = port_args.instance_id

            # PortArgs.init_new 只在当时检测端口空闲，不能保证下一个 DP rank 仍不会选中
            # 同一端口。因此先持有 socket，待全部端口编号确定后再统一释放并启动进程。
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # 每组 TP/PP 在独立线程中完成子进程创建；线程间的 CUDA 环境变量临界区由
            # self.env_lock 保护，见 launch_tensor_parallel_group。
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * configured_pp_size() * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # 端口编号已固定，释放临时占用后才能让对应 Scheduler 真正 bind NCCL 端口。
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        """线程包装器：启动一个 DP 的 TP/PP 子进程集合并保持其父线程存活。"""
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # Scheduler 内部通过 ``kill_itself_when_parent_died`` 维护父子存活关系。此线程
        # 不能在启动后立即退出，否则该保护逻辑会把仍正常运行的 Scheduler 误判为失去
        # 父级而杀掉。因此它只负责生命周期占位，实际工作在 Scheduler 子进程中完成。
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """将 node 0 预分配的 DP-attention worker 端口同步给其他节点。

        DP-attention 的同一逻辑 DP group 跨节点启动时，所有节点必须使用同一份
        ``worker_ports`` 来构造 ``PortArgs``；让每个节点自行选端口会使 controller 的
        PUSH 端与 Scheduler 的接收端不匹配。node 0 作为 REP 服务端，其他节点通过
        REQ 握手获取这份只读端口表。

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        is_joiner = server_args.is_ep_scale_joiner
        if server_args.dist_init_addr is None or is_joiner:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """在主节点以 REP/REQ 握手方式同步端口表。

        每个客户端必须先发一次握手，服务端才回复端口列表；这使主节点能确认普通多节点
        启动所需的节点均已取到一致配置。Elastic EP 模式例外：首批客户端完成后仍保留
        REP socket，并转入后台服务 late joiner。
        """
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # REP socket 必须严格遵循“先收一个请求，再回一个响应”的时序。
                client_rank = sock_recv(rep_socket)
                logger.debug(f"Received handshake from node {client_rank}")

                # 这里只发布端口号，不转移 socket 的所有权；socket 始终由主节点持有。
                sock_send(rep_socket, wrap_as_pickle(worker_ports))
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if get_exec().moe.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """为延后加入的 Elastic EP 节点持续重放主节点已经绑定的端口表。

        新 joiner 只能连接既有主节点 socket，不能改变端口表；因此这里永不创建端口、
        也不关闭主节点的 socket。单次握手失败不会终止服务线程，以免一个坏连接阻塞
        后续扩容。
        """
        while True:
            try:
                client_rank = sock_recv(rep_socket)
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            # 仍只发送端口编号，实际连接端点不随 joiner 改变。
            sock_send(rep_socket, wrap_as_pickle(worker_ports))
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """作为非主节点/scale joiner 从主节点取得已绑定的端口表。

        使用 10 分钟收发超时而不是无限等待：分布式启动编排错误时，进程应明确失败并
        交给父进程清理，而不是永久卡在端口握手阶段。
        """
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # node rank 主要用于主节点日志和启动确认，不参与端口重新分配。
            sock_send(req_socket, wrap_as_pickle(str(node_rank)))

            # 所有节点应取得同一列表，随后用于对齐其 scheduler input 端点。
            worker_ports = sock_recv(req_socket)
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def _joiner_local_tp_span(self, server_args: ServerArgs) -> int:
        """返回 Elastic EP joiner 在本机需要覆盖的 TP rank 数。"""
        return server_args.tp_size

    def _joiner_slot_offset(self, server_args: ServerArgs) -> int:
        """返回 joiner 在主节点预留 worker slot 表中的起始偏移。"""
        return server_args.ep_join_rank_offset

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        """启动 DP-attention 的 Scheduler 拓扑并对齐跨节点路由端口。

        与传统 DP“每个 dp_rank 一份完整 TP/PP 副本”不同，DP-attention 的 DP rank
        是由全局 TP rank 推导出来的。主节点为每个可路由 slot 预绑定 PUSH socket，
        再把端口表广播给其他节点；Elastic EP 下还会按 ``max_dp_size`` 预留槽位，供
        后续 joiner 无需改变现有端点即可接入。
        """
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        worker_ports = []
        if server_args.is_ep_scale_joiner:
            # joiner 只取属于自己的连续 slot 切片，并连接主节点预先绑定的端点；不能在
            # joiner 上新建 worker socket，否则入口 controller 无法感知该新端点。
            primary = NetworkAddress.parse(server_args.dist_init_addr)
            primary_endpoint = NetworkAddress(
                primary.host, primary.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA
            ).to_tcp()
            all_ports = self._receive_ports_as_client(
                primary_endpoint, server_args.node_rank
            )
            offset = self._joiner_slot_offset(server_args)
            local_tp_span = self._joiner_local_tp_span(server_args)
            broadcasted_ports = all_ports[offset : offset + local_tp_span]
        elif server_args.node_rank == 0:
            # Elastic primary 为最大 DP 数预先绑定 socket；未激活 slot 暂不接流量，见
            # add_elastic_workers。非 Elastic 场景只需要当前 dp_size 个端点。
            bind_count = (
                self.max_dp_size
                if server_args.elastic_ep_backend is not None
                else get_parallel().dp_size
            )
            for slot in range(bind_count):
                worker_port, worker_socket = get_zmq_socket_on_host(
                    self.context, zmq.PUSH, host=bind_host
                )
                worker_ports.append(worker_port)
                self.workers[slot] = worker_socket
                logger.debug(
                    "Assigned port %s to worker slot %s on host %s",
                    worker_port,
                    slot,
                    bind_host,
                )
            broadcasted_ports = self._broadcast_worker_ports(server_args, worker_ports)
        else:
            broadcasted_ports = self._broadcast_worker_ports(server_args, None)

        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        """按照当前节点的 PP/TP 切片创建一个或多个 Scheduler 子进程。

        本函数是“控制层启动”与“模型执行层”的分界：它不做前向计算，而是枚举本节点
        应承担的 ``(pp_rank, tp_rank)``，为每个 rank 计算 GPU、通信端口及 attention/MoE
        子组 rank，随后调用 ``run_scheduler_process_func`` 启动真正的 Scheduler。

        ``dp_rank`` 在传统 DP 中由外层循环传入；DP-attention 中则必须依据全局 TP rank
        重新计算，因为数据并行与 attention context parallel 的分组交错。所有子进程经
        单向 Pipe 回报“模型已加载”和容量信息，函数只有收齐这些回报才返回，因而上层
        不会在 Scheduler 尚未准备好时宣告服务可用。
        """
        if not get_parallel().enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        # 先将全局 PP/TP 拓扑切到当前节点。下面两个 rank range 精确限定本进程可以
        # spawn 的局部模型 rank，避免每个节点重复启动全量模型。
        pp_size_per_node = max(configured_pp_size() // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // configured_pp_size(), 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        if server_args.is_ep_scale_joiner:
            # Elastic EP joiner 的本机 GPU 排布由 join 语义决定，需要枚举完整 local TP
            # span，而不是按普通 node_rank 分片。
            tp_rank_range = range(server_args.tp_size)
            tp_size_per_node = server_args.tp_size
        else:
            tp_rank_range = range(
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
            )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if get_parallel().enable_dp_attention:
                    # DP-attention 中 DP/ATTN-CP/ATTN-TP rank 由一个全局 TP rank 共同
                    # 推导，不能复用传统 DP 的外层 dp_rank。
                    _, _, dp_rank, _ = compute_dp_attention_world_info(
                        get_parallel().enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        get_parallel().dp_size,
                        configured_attn_cp_size(),
                    )
                    # 每个由公式得到的 DP group 使用自己的 Scheduler 输入端口。各节点
                    # 使用上面同步的 worker_ports，所以 controller 的 PUSH 端保持一致。
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    if server_args.is_ep_scale_joiner:
                        # joiner 不持有独立的 API/Tokenizer：Scheduler 的输出仍回到 primary
                        # 的 tokenizer/detokenizer 地址，保证请求会话从同一前端返回。
                        primary_addr = NetworkAddress.parse(server_args.dist_init_addr)
                        primary_port_base = primary_addr.port + 1
                        rank_port_args.tokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base
                        ).to_tcp()
                        rank_port_args.detokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base + 1
                        ).to_tcp()
                    # DP-attention 的数据并行复用 TP 通信组，故所有 dp_rank 使用同一个
                    # NCCL 端口和 instance_id；各 DP 输入 socket 的区分已由上面完成。
                    rank_port_args.nccl_port = port_args.nccl_port
                    rank_port_args.instance_id = port_args.instance_id

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    get_parallel().dp_size if get_parallel().enable_dp_attention else 1
                )

                # 并行层级（由外至内）。这些 rank 既传给 Scheduler 建立 collective，也用
                # 于把日志中的全局 rank 与进程内部使用的局部 rank 区分开：
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // configured_attn_cp_size()
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % configured_attn_cp_size()
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // configured_moe_dp_size()
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // configured_moe_dp_size())
                    // (
                        server_args.tp_size
                        // configured_moe_dp_size()
                        // get_parallel().ep_size
                    )
                )

                # Scheduler 内部使用 local rank；Elastic join 后日志和可观测性需要展示
                # 不冲突的 global rank，因此单独计算 display_* 值而不污染通信 rank。
                offset = server_args.ep_join_rank_offset
                display_tp_rank = tp_rank + offset
                display_moe_ep_rank = moe_ep_rank + offset
                display_dp_rank = dp_rank + offset if dp_rank is not None else None

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    # multiprocessing 子进程会继承当前环境。锁覆盖“选择/重映射 GPU 到
                    # start()”整个窗口，确保并行启动的不同 DP 组不会交叉继承设备设置。
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                            display_tp_rank,
                            display_dp_rank,
                            display_moe_ep_rank,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        # 两个 context manager 只影响即将 fork/spawn 的子进程：前者处理
                        # 内存保存适配，后者把 CPU/NUMA 亲和性与该 GPU 对齐。
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # 必须等待本节点每个模型 rank 回报加载完成。Pipe 不仅是 ready 信号，也携带
        # capacity/startup-time；这样 controller 对 Engine 的 ready 语义是“可实际调度”，
        # 而非“进程刚被创建”。
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        # 同一 DP/TP/PP 副本按启动配置应得到相同的服务容量，控制器向上层汇报第一份
        # Scheduler 信息；每个子进程仍已通过上面的 recv 被确认完成加载。
        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]
        self.startup_time = aggregate_scheduler_startup_times(
            info.get("startup_time") for info in scheduler_info
        )

    def maybe_external_dp_rank_routing(self, req: Req):
        """优先执行上游指定的 DP-rank 亲和性路由。

        ``routed_dp_rank`` 常用于外部路由器/会话粘性已完成选择的场景。控制器仍校验
        rank 范围、Elastic active 状态和 socket 是否存在，避免将请求发给退出或尚未加入
        的副本；未指定时才交给本地负载均衡策略。
        """
        if req.routed_dp_rank is not None:
            rank = req.routed_dp_rank
            if (
                rank < 0
                or rank >= len(self.workers)
                or rank not in self._active_workers
                or self.workers[rank] is None
            ):
                raise ValueError(f"DP rank {rank} is not active.")
            logger.debug(f"Direct routing to DP rank {rank}")
            sock_send(self.workers[rank], req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        """在当前可用 rank 间轮询，跳过动态状态为不可用的 worker。"""
        if self.maybe_external_dp_rank_routing(req):
            return

        active = self._active_workers
        if not active:
            raise RuntimeError("No active DP workers are available for routing.")
        attempts = 0
        while attempts < len(active):
            slot = active[self.round_robin_counter % len(active)]
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)
            if self.status[slot]:
                logger.debug(f"Choose worker {slot}")
                sock_send(self.workers[slot], req)
                return
            attempts += 1
        raise RuntimeError(
            f"Cannot route request: all {len(active)} active DP workers "
            "are unavailable."
        )

    def follow_bootstrap_room_scheduler(self, req: Req):
        """按 ``bootstrap_room`` 做确定性映射，保持 router 分配的房间亲和性。

        该策略依赖上游 router 填充 ``bootstrap_room``，因此直接向 prefill/decode 实例
        发送缺少该字段的请求会触发断言；此处故意不猜测目标 rank，以免破坏 router 的
        请求归属。
        """
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)

    def total_requests_scheduler(self, req: Req):
        """将请求发送给估计在途请求数最少的 DP worker。"""
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        sock_send(self.workers[target_worker], req)

    def total_tokens_scheduler(self, req: Req):
        """将请求发送给估计 token 负载最低的 DP worker。

        输入 token 数是控制器在分发前唯一已知的规模信号；生成 token 的最终长度未知，
        因而这里只做启发式预估，后续 Scheduler 快照会重新校正。
        """
        if self.maybe_external_dp_rank_routing(req):
            return
        estimated_tokens = len(req.input_ids)
        target_worker = self.dp_budget.dispatch(
            LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=estimated_tokens
        )
        sock_send(self.workers[target_worker], req)

    def event_loop(self):
        """在 node 0 持续抽取 Tokenizer 消息并分发。

        内层循环使用非阻塞 recv 一次性清空当前 socket 中的积压，降低高 QPS 下的排队
        往返；队列暂时为空时跳出内层循环并立刻继续下一轮，同时喂软 watchdog。该函数
        只应由 primary controller 调用，调用方会保证它不是 Elastic EP joiner。
        """
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    """DataParallelController 的独立进程入口，也是 Engine 的启动握手边界。

    为什么需要单独进程
    ------------------
    DP 模式下一个 Engine 要管理多个 Scheduler 子进程、ZMQ 路由 socket 和模型加载过程。
    将这些工作放入独立 controller 进程可保持 API/Tokenizer 父进程轻量，并使 Scheduler
    形成清晰的父子关系；控制器异常时可以显式通知父 Engine 整体失败，而不是留下部分
    已加载但不接流量的模型副本。

    执行顺序
    --------
    1. 设置进程名、故障转储和“父进程死亡即退出”保护；
    2. 发布运行时配置、初始化日志和可选 tracing；
    3. 构造 ``DataParallelController``。构造过程会创建路由端点、启动全部 Scheduler，
       并等待它们通过 Pipe 报告模型加载完成；
    4. 通过 ``pipe_writer`` 向 Engine 发送 ``ready``、容量上限、汇总启动耗时和所有
       Scheduler PID。父进程以此决定服务是否可以进入就绪状态并纳入子进程监督；
    5. 仅 primary（node 0 且非 Elastic EP joiner）进入永久路由循环。其他节点只负责
       本地 Scheduler 生命周期，因为它们没有全局 Tokenizer 输入端点；
    6. 当任一 Scheduler 退出，join 后记录其退出码。若控制器初始化或运行异常，则向
       父 Engine 发送 SIGQUIT，避免父进程继续等待一个不可能完成的启动握手。

    ``run_scheduler_process_func`` 是可注入参数，默认使用生产的
    :func:`run_scheduler_process`；它保留给测试或替代 Scheduler 启动实现，控制器的
    路由与就绪协议不因此改变。
    """
    # 进程名用于 ps/top/trace 中区分控制器与大量 scheduler 子进程。
    setproctitle.setproctitle("sglang::data_parallel_controller")
    # 发生 Python 级崩溃时把所有线程栈输出到日志，分布式启动问题才有可诊断现场。
    faulthandler.enable()
    # 控制器不能在 Engine 已退出后继续持有 GPU/IPC 资源，父进程死亡时一并结束。
    kill_itself_when_parent_died()
    # 异常路径需要主动终止真正等待启动结果的 Engine，而不是只让本子进程静默退出。
    parent_process = psutil.Process().parent()

    # Scheduler 子进程会继承本进程的运行时上下文。因此必须先发布配置命名空间、设置
    # logger/tracing，再构造 controller 并开始 spawn，保证子进程看到一致的配置。
    publish(server_args, role="dp_controller")
    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "DP Controller"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        # 构造函数是同步启动栅栏：其中会创建路由 socket、启动所有本地 Scheduler，并在
        # 每个 Scheduler 通过 Pipe 回报模型加载完成前阻塞。
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        # 这些 PID 不是新的 worker，而是 controller 刚刚创建的模型 Scheduler。回传给
        # Engine 后，Engine 可以把间接子进程也纳入统一的生命周期监督。
        scheduler_pids = [
            proc.pid for proc in controller.scheduler_procs if proc is not None
        ]
        # “ready” 的含义是模型容量已经确定，而非 controller 进程仅仅成功 fork；这是
        # Engine 启动状态机与 DP 子树之间唯一的单向握手。
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
                "startup_time": controller.startup_time,
                SCHEDULER_PIDS_ARG: scheduler_pids,
            }
        )
        # 全局输入 PULL socket 只由 primary 绑定。因此只有它消费 Tokenizer 请求并进行
        # 负载均衡；非 primary/Elastic joiner 继续 join 本地 Scheduler，以维持进程树。
        if server_args.node_rank == 0 and not server_args.is_ep_scale_joiner:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            # 正常运行时这里会一直阻塞；一旦任何 Scheduler 意外退出，记录退出码供
            # 外层进程监督与日志诊断使用。不会在此处重启，以免丢失其请求/KV 状态。
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        # 包含启动握手前失败、端口/拓扑配置错误及 primary 路由循环异常。通知父 Engine
        # 整体失败可避免其继续服务一个不完整的 DP 拓扑。
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
