# Copyright 2025 SGLang Team
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
"""KV cache placement event emission mixin.

The mixin produces the ``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared``
events consumed by KV-aware routers (e.g. dynamo).
"""

from typing import Any

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    hash_str_to_int64,
)

# 本文件实现了 KV 缓存事件追踪机制，供外部消费者（如 KV 感知路由器）使用。
# 通过 Mixin 模式混入缓存类，在缓存块的存储和移除时发出对应事件。
# 支持的事件类型包括：BlockStored（块存储）、BlockRemoved（块移除）、
# AllBlocksCleared（所有块清除）。


class KVCacheEventMixin:
    """KV 缓存事件混入类，为缓存类添加事件记录和导出能力。"""

    def _record_store_event(self, node: Any, medium=None):
        """记录块存储事件：将节点按 page_size 分页，每页生成一个 BlockStored 事件。"""
    def _record_store_event(self, node: Any, medium=None):
        # One BlockStored per ``page_size`` chunk.
        # ``medium`` defaults to StorageMedium.GPU but callers may override
        # for lower-tier insertions (e.g. StorageMedium.CPU for host/L2 cache).
        if self.enable_kv_cache_events:
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            logical_len = len(node.key)
            is_bigram = node.key.is_bigram
            raw = node.key.token_ids
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue
                # Preserve historical event payload: bigram pages expose tuples.
                if is_bigram:
                    page_tokens = [(raw[j], raw[j + 1]) for j in range(start, end)]
                else:
                    page_tokens = list(raw[start:end])

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=medium,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: Any, medium=None):
        """记录块移除事件：将节点的所有分页哈希值合并为一个 BlockRemoved 事件。"""
        # One BlockRemoved per radix node.
        # ``medium`` defaults to StorageMedium.GPU but callers may override for
        # lower-tier removals (e.g. StorageMedium.CPU when evicting from host).
        if self.enable_kv_cache_events:
            if medium is None:
                medium = StorageMedium.GPU

            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            block_hashes = []
            logical_len = len(node.key)
            page_index = 0
            for start in range(0, logical_len, self.page_size):
                end = min(start + self.page_size, logical_len)
                if end <= start:
                    continue

                block_hashes.append(hash_str_to_int64(node.hash_value[page_index]))
                page_index += 1

            if block_hashes:
                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=block_hashes, medium=medium)
                )

    def _record_all_cleared_event(self):
        """记录所有块清除事件，通常在缓存重置时调用。"""
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        # 原子性地取出所有事件并清空队列，供外部消费者（如路由器）轮询
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
