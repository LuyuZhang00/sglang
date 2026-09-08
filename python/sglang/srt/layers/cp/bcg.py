# Copyright 2023-2026 SGLang Team
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

"""Breakable CUDA graph helpers for context-parallel prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.padding import get_cp_padding_align_size
from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_split_before_forward,
    enable_cp_v2,
    prepare_cp_forward,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.server_args import ServerArgs


def supports_prefill_cp_bcg(server_args: ServerArgs) -> bool:
    """判断当前服务参数能否使用 prefill-CP 的 Breakable CUDA Graph。

    BCG 复用固定地址的 CP 局部输入缓冲区，目前仅覆盖 TP 与 attention-CP 同规模、
    zigzag 布局及 TRT-LLM MHA prefill 后端的组合。条件集中在这里，避免 graph
    runner 捕获不支持布局后才暴露难以定位的静态形状错误。
    """
    # 读取经过 ServerArgs 归一化后的 attention-CP 大小。
    resolved = server_args._resolved()
    # BCG 对 prefill 后端有要求，decode 后端在此无关。
    prefill_attention_backend, _ = server_args._resolved_attention_backends()
    return (
        server_args.enable_prefill_cp
        and resolved.attn_cp_size == server_args.tp_size
        and server_args.cp_strategy == "zigzag"
        and prefill_attention_backend == "trtllm_mha"
    )


def enable_cp_v2_bcg_capture(server_args: ServerArgs) -> bool:
    """仅当 CP-v2 开关与 BCG 配置约束同时满足时允许捕获。

    ``enable_cp_v2`` 决定是否走新 CP 元数据/切分路径，配置检查则保证其局部张量
    几何可被当前 BCG 实现固定下来。两者缺一不可，避免旧 CP 路径误进入 replay。
    """
    return enable_cp_v2() and supports_prefill_cp_bcg(server_args)


def filter_prefill_cp_bcg_capture_num_tokens(
    capture_num_tokens: list[int], server_args: ServerArgs
) -> list[int]:
    """过滤不能容纳 zigzag 最小双端分块的图捕获 token bucket。

    每个 CP rank 需要前、后两个 block，因此全局 bucket 至少为 ``2 * cp_size``。
    若所有候选都被过滤，服务启动时立即报出可操作的配置错误，而不是延迟到 capture
    或 replay 阶段失败。
    """
    # 最小 bucket 必须保证每个 rank 的两个 zigzag block 都可存在。
    min_num_tokens = server_args._resolved().attn_cp_size * 2
    # 保留满足几何限制的原始 bucket 顺序。
    filtered = [size for size in capture_num_tokens if size >= min_num_tokens]
    if not filtered:
        raise ValueError(
            "Prefill CP breakable CUDA graph requires at least one token bucket "
            f">= {min_num_tokens}, but got {capture_num_tokens}."
        )
    return filtered


def _slice_output_rows(output: Any, num_tokens: int) -> Any:
    """递归裁掉 BCG replay 中超过 live batch 的捕获尾行。

    图按 bucket 的固定局部长度运行，而真实请求常更短；只有前 ``num_tokens`` 行是
    当前 batch 的有效 token。模型输出可以是 tensor、pipeline proxy 或嵌套 tuple/
    list，故递归保持容器结构并对每个张量执行相同裁剪。
    """
    # 空辅助输出（例如某些模型的 aux states）无需处理。
    if output is None:
        return None
    # tensor 和 pipeline proxy 共享 token 维位于第 0 维的约定。
    if torch.is_tensor(output) or isinstance(output, PPProxyTensors):
        return output[:num_tokens]
    # 保留模型返回 tuple 的类型和元素顺序。
    if isinstance(output, tuple):
        return tuple(_slice_output_rows(item, num_tokens) for item in output)
    # 部分运行时辅助输出采用 list，按相同规则递归裁剪。
    if isinstance(output, list):
        return [_slice_output_rows(item, num_tokens) for item in output]
    raise TypeError(f"Unsupported prefill CP BCG output: {type(output)}")


@dataclass
class PrefillCPBCGInput:
    """Fixed-address CP-local inputs and per-bucket replay state."""

    input_embeds: torch.Tensor
    positions: torch.Tensor
    bucket_local_tokens: Dict[int, int] = field(default_factory=dict)
    live_local_tokens: int = 0

    @classmethod
    def create(cls, runner: PrefillCudaGraphRunner) -> PrefillCPBCGInput:
        """按最大全局 bucket 分配可复用的固定地址 CP 局部输入缓冲区。

        CUDA graph replay 要求输入地址稳定，因而 embedding 和 positions 一次性分配
        到 runner 的最大 token 容量。实际 capture/replay 只取其前缀，不在 graph
        期间重新分配；dtype 和 hidden size 必须与 model runner 保持一致。
        """
        # 进入 runner device context，确保 buffer 落在 graph 对应 GPU。
        with torch.device(runner.device):
            return cls(
                input_embeds=torch.zeros(
                    (
                        runner.max_num_tokens,
                        runner.model_runner.model_config.hidden_size,
                    ),
                    dtype=runner.model_runner.dtype,
                ),
                positions=torch.zeros(
                    (runner.max_num_tokens,),
                    dtype=torch.int64,
                ),
            )

    def required_local_tokens(self, extend_seq_lens: Any) -> Optional[int]:
        """计算一个 live zigzag batch 在单个 CP rank 上所需的对齐行数。

        每条序列按 ``2 * cp_size`` 分块，当前所有 rank 的前/镜像后 block 数量先在
        CPU 累加，再取最大值并按 CP 对齐粒度向上取整。该值用于判断一个全局 bucket
        已捕获的局部 buffer 是否能安全容纳不均匀长度的后续 replay。
        """
        # BCG 只实现 zigzag；其他策略不应尝试选择 replay bucket。
        strategy = get_cp_strategy()
        if not isinstance(strategy, ZigzagCPStrategy) or extend_seq_lens is None:
            return None

        # 每个 rank 从一条序列获得一个前 block 与一个镜像后 block。
        cp_segment_num = strategy.cp_size * 2
        per_rank_logical_tokens = [0] * strategy.cp_size
        for raw_length in extend_seq_lens:
            # 均分部分使每 rank 获得两个 ``base`` 大小的 block。
            base, remainder = divmod(int(raw_length), cp_segment_num)
            for rank in range(strategy.cp_size):
                # 余数优先分给低 block id；镜像 block 对应相反的 id。
                opposite_rank = cp_segment_num - 1 - rank
                per_rank_logical_tokens[rank] += (
                    base * 2 + int(rank < remainder) + int(opposite_rank < remainder)
                )
        # 与实际 CP padding 相同的规则，保证估算与 capture 几何一致。
        align_size = get_cp_padding_align_size()
        return (
            (max(per_rank_logical_tokens) + align_size - 1) // align_size * align_size
        )

    def select_replay_bucket(
        self,
        *,
        num_tokens: int,
        required_local_tokens: int,
        capture_num_tokens: list[int],
        max_padding_factor: int,
    ) -> Optional[int]:
        """选择可容纳 live batch 的最小已捕获全局 token bucket。

        先用 ``max_padding_factor`` 限制可接受的全局过度 padding，再检查该 bucket
        对应的实际 CP 局部行容量，而不能只比较全局 token 数。原因是 zigzag 对长短
        序列的分块不均衡，两个全局等长 batch 的最大本地行数可能不同。
        """
        # 防止为了复用 graph 选择远大于 live batch 的 bucket。
        max_num_tokens = num_tokens * max_padding_factor
        for bucket in capture_num_tokens:
            # bucket 小于真实全局 token 数肯定无法 replay。
            if bucket < num_tokens:
                continue
            # capture bucket 通常升序，超过 padding 上界即可停止扫描。
            if bucket > max_num_tokens:
                break
            # bucket 必须已被 capture，且其局部固定 buffer 足够大。
            captured_local_tokens = self.bucket_local_tokens.get(bucket)
            if (
                captured_local_tokens is not None
                and required_local_tokens <= captured_local_tokens
            ):
                return bucket
        return None

    def select_replay_bucket_for_batch(
        self,
        *,
        num_tokens: int,
        extend_seq_lens: Any,
        capture_num_tokens: list[int],
        max_padding_factor: int,
    ) -> Optional[int]:
        """为完整 batch 先估算局部容量，再复用通用 bucket 选择逻辑。

        当 strategy 非 zigzag 或缺少长度信息时返回 ``None``，让调用方走 eager/
        capture 选择的安全回退。正常情况下传递 live 的局部对齐长度，避免全局 bucket
        看似足够但固定 CP input buffer 不足的越界写入。
        """
        required_local_tokens = self.required_local_tokens(extend_seq_lens)
        if required_local_tokens is None:
            return None
        return self.select_replay_bucket(
            num_tokens=num_tokens,
            required_local_tokens=required_local_tokens,
            capture_num_tokens=capture_num_tokens,
            max_padding_factor=max_padding_factor,
        )

    def prepare(
        self,
        runner: PrefillCudaGraphRunner,
        forward_batch: ForwardBatch,
        *,
        static_num_tokens: int,
        capture: bool,
    ) -> None:
        """把 live 全局 prefill 输入切分并复制到固定地址的 CP 局部 buffer。

        capture 时记录该 global bucket 的局部行容量；replay 时保持捕获的物理形状，
        仅更新前缀的有效 token。函数在图外完成 embedding、CP 切分和 buffer copy，
        使图内 transformer body 只看到稳定地址、稳定尺寸的输入张量。
        """
        # ForwardBatch 可能来自不同布局的上次 replay，必须重新生成本次 metadata。
        forward_batch.attn_cp_metadata = None
        prepare_cp_forward(forward_batch)

        # replay 必须复用 capture 时记录的局部容量。
        captured_local_tokens = None
        if not capture:
            # 按 global bucket 查找 capture 时确定的固定 local 行数。
            try:
                captured_local_tokens = self.bucket_local_tokens[static_num_tokens]
            except KeyError as exc:
                raise RuntimeError(
                    "Missing CP-local capture capacity for global prefill bucket "
                    f"{static_num_tokens}"
                ) from exc

            # 较短 live batch 仍需伪装为 capture 时的物理行数。
            metadata = forward_batch.attn_cp_metadata
            if hasattr(metadata, "per_rank_actual_token"):
                live_physical_tokens = max(metadata.per_rank_actual_token)
                # 先验证本次不均衡布局没有超过 capture 时最大局部容量。
                if live_physical_tokens > captured_local_tokens:
                    raise RuntimeError(
                        f"Live batch needs {live_physical_tokens} local CP rows, "
                        f"but global prefill bucket {static_num_tokens} has "
                        f"captured capacity {captured_local_tokens}"
                    )
                # 强制 metadata 使用 capture 形状，使后续 graph 输入和 attention buffer 对齐。
                cp_size = len(metadata.per_rank_actual_token)
                metadata.per_rank_actual_token = [captured_local_tokens] * cp_size
                metadata.max_rank_len = [captured_local_tokens] * cp_size

        # 只为真实 token 计算 embedding，bucket 的余量由零 buffer 表示。
        raw_tokens = int(forward_batch.extend_num_tokens)
        global_input_ids = forward_batch.input_ids[:raw_tokens]
        global_positions = forward_batch.positions[:raw_tokens]
        global_input_embeds = runner.model_runner.model.get_input_embeddings()(
            global_input_ids
        )
        # 在图外按 zigzag 切分，得到当前 CP rank 的局部 embedding/position。
        local_input_embeds, local_positions = cp_split_before_forward(
            global_input_embeds,
            global_positions,
            forward_batch,
        )
        live_local_tokens = int(local_input_embeds.shape[0])

        # capture 首次观察到该 bucket 时，把局部长度固化为其容量记录。
        if capture:
            captured_local_tokens = live_local_tokens
            self.bucket_local_tokens[static_num_tokens] = captured_local_tokens
        else:
            # replay 再次核验，避免 live copy 越过固定 buffer 前缀。
            assert captured_local_tokens is not None
            if live_local_tokens > captured_local_tokens:
                raise RuntimeError(
                    f"Live batch needs {live_local_tokens} local CP rows, but global "
                    f"prefill bucket {static_num_tokens} has captured capacity "
                    f"{captured_local_tokens}"
                )

        # 固定总 buffer 也必须能覆盖当前 bucket 的局部 capture 容量。
        if captured_local_tokens > self.input_embeds.shape[0]:
            raise RuntimeError(
                f"CP-local capture needs {captured_local_tokens} rows, but the "
                f"fixed input buffer has capacity {self.input_embeds.shape[0]}"
            )

        # 从固定大 buffer 取 bucket 前缀，地址随 replay 保持不变。
        input_embeds = self.input_embeds[:captured_local_tokens]
        positions = self.positions[:captured_local_tokens]
        # 清除上次更长 batch 的尾部数据，避免其影响本次 attention。
        input_embeds.zero_()
        positions.zero_()
        # 只复制本次有效局部行，其他行保持零 padding。
        input_embeds[:live_local_tokens].copy_(local_input_embeds)
        positions[:live_local_tokens].copy_(local_positions)
        forward_batch.input_embeds = input_embeds
        forward_batch.positions = positions
        self.live_local_tokens = live_local_tokens


def execute_prefill_cp_bcg(
    runner: PrefillCudaGraphRunner,
    forward_batch: ForwardBatch,
    static_forward_batch: ForwardBatch,
    static_num_tokens: int,
    raw_num_tokens: int,
    **kwargs,
):
    """replay CP 局部 transformer body，再在图外收集并计算全局 logits。

    图内只执行固定形状的局部模型主体；live 行数裁剪、CP all-gather 和 logits
    processor 则保留在图外，因其依赖请求的真实 token 布局。pipeline 非末 rank
    直接返回局部输出，只有末 rank 需要恢复全局 hidden states 并生成 logits。
    """
    # 延迟导入避免解析服务参数时经 runner/__init__ 反向导入本模块形成循环。
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

    cp_input = runner.prefill_cp_bcg_input
    assert cp_input is not None
    model = runner.model_runner.model
    # 用静态 batch 建立 graph replay 所需的固定 forward context。
    with runner._prefill_forward_context(
        static_forward_batch,
        num_tokens=static_num_tokens,
        raw_num_tokens=raw_num_tokens,
    ):
        # 以 global bucket 作为 graph shape key 重放局部 transformer body。
        local_output = runner.backend.replay(
            ShapeKey(size=static_num_tokens),
            static_forward_batch,
            **kwargs,
        )
        # 去除 capture 的局部尾行，使图外 gather 只处理 live token。
        local_output = _slice_output_rows(local_output, cp_input.live_local_tokens)

        # 某些模型会捕获额外 hidden states，保持其原返回约定。
        capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
        aux_hidden_states = None
        if capture_aux_hidden_states:
            hidden_states, aux_hidden_states = local_output
        else:
            hidden_states = local_output

        # pipeline 中间 stage 继续传递局部张量，不应提前执行全局 gather。
        if not model.pp_group.is_last_rank:
            return (
                (hidden_states, aux_hidden_states)
                if capture_aux_hidden_states
                else hidden_states
            )

        # 最后 stage 先恢复全局 token 顺序，再交给 logits processor。
        hidden_states = cp_gather_after_forward(
            hidden_states,
            static_forward_batch,
            torch.cuda.current_stream(),
        )
        return model.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            model.lm_head,
            forward_batch,
            aux_hidden_states,
        )
