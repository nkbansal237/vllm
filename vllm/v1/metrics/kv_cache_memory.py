# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for HBM KV cache memory.

The values come from `vllm/v1/core/kv_cache_memory_stats.py` through
`SchedulerStats`. Ratios worth charting:

- resident HBM: 1 - free_unhashed / sum(kv_cache_blocks_by_state)
- admissible free blocks:
  cached_evictable + free_unhashed - sum(kv_admission_reserved_blocks)
- tail slack: tail_slack / (filled + tail_slack)
"""

from typing import Any

from vllm.v1.core.kv_cache_memory_stats import (
    BLOCK_STATES,
    REFUSAL_REASONS,
    RESERVED_KINDS,
)
from vllm.v1.metrics.stats import SchedulerStats

# The KV cache groups share one block pool, so block states have one series.
ALL_KV_GROUPS = "all"


class KVCacheMemoryProm:
    def __init__(
        self,
        gauge_cls: Any,
        counter_cls: Any,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.per_engine_labelvalues = per_engine_labelvalues

        self.gauge_blocks_by_state = gauge_cls(
            name="vllm:kv_cache_blocks_by_state",
            documentation=(
                "KV cache blocks in each state; the states sum to the usable "
                "blocks. Refreshed by a periodic scan "
                "(VLLM_KV_BLOCK_STATE_INTERVAL_S)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames + ["kv_group", "state"],
        )
        self.gauge_reserved_blocks = gauge_cls(
            name="vllm:kv_admission_reserved_blocks",
            documentation=(
                "Free KV cache blocks the scheduler holds back from admissions: "
                "the watermark, and blocks reserved for in-flight async KV loads."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames + ["kind"],
        )
        self.counter_refusals = counter_cls(
            name="vllm:kv_admission_refusals",
            documentation=(
                "Waiting requests refused KV cache blocks, once per request and "
                "reason; head_of_line counts requests queued behind a refused "
                "request."
            ),
            labelnames=labelnames + ["reason"],
        )
        self.counter_load_failed_tokens = counter_cls(
            name="vllm:kv_load_failed_tokens",
            documentation=(
                "Claimed external tokens whose KV load failed, by the connector "
                "that claimed them and the failure policy."
            ),
            labelnames=labelnames + ["connector", "policy"],
        )
        self.gauge_token_slots = gauge_cls(
            name="vllm:kv_token_slots",
            documentation=(
                "Token slots in the KV cache blocks of running and loading "
                "requests: filled, unused slots in each request's last block "
                "(tail_slack), and slots allocated ahead of computed tokens "
                "(reserved)."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames + ["kv_group", "kind"],
        )
        self.gauge_orphan_block_info = gauge_cls(
            name="vllm:kv_orphan_block_info",
            documentation=(
                "KV cache blocks referenced more times than any request or "
                "scheduler table accounts for (suspected leaks), with the "
                "request that last held them; at most 16 per engine."
            ),
            multiprocess_mode="mostrecent",
            labelnames=labelnames + ["kv_group", "block_id", "last_owner"],
        )
        self._orphan_labels: dict[int, set[tuple[str, str]]] = {
            idx: set() for idx in per_engine_labelvalues
        }

        # Create the fixed series so they exist at 0 before the first event.
        for labelvalues in per_engine_labelvalues.values():
            for state in BLOCK_STATES:
                self.gauge_blocks_by_state.labels(*labelvalues, ALL_KV_GROUPS, state)
            for kind in RESERVED_KINDS:
                self.gauge_reserved_blocks.labels(*labelvalues, kind)
            for reason in REFUSAL_REASONS:
                self.counter_refusals.labels(*labelvalues, reason)

    def observe(self, stats: SchedulerStats, engine_idx: int = 0) -> None:
        labelvalues = self.per_engine_labelvalues[engine_idx]
        for kind, num_blocks in stats.kv_admission_reserved_blocks.items():
            self.gauge_reserved_blocks.labels(*labelvalues, kind).set(num_blocks)
        for reason, count in stats.kv_admission_refusals.items():
            self.counter_refusals.labels(*labelvalues, reason).inc(count)
        for connector, policy, num_tokens in stats.kv_load_failed_tokens:
            self.counter_load_failed_tokens.labels(*labelvalues, connector, policy).inc(
                num_tokens
            )
        for kv_group, kind, num_slots in stats.kv_token_slots:
            self.gauge_token_slots.labels(*labelvalues, kv_group, kind).set(num_slots)
        if stats.kv_blocks_by_state is not None:
            for state, num_blocks in stats.kv_blocks_by_state.items():
                self.gauge_blocks_by_state.labels(
                    *labelvalues, ALL_KV_GROUPS, state
                ).set(num_blocks)
        if stats.kv_orphan_blocks is not None:
            self._set_orphans(engine_idx, labelvalues, stats.kv_orphan_blocks)

    def _set_orphans(
        self,
        engine_idx: int,
        labelvalues: list[object],
        orphans: list[tuple[int, str]],
    ) -> None:
        current = {(str(block_id), last_owner) for block_id, last_owner in orphans}
        gauge = self.gauge_orphan_block_info
        for block_id, last_owner in self._orphan_labels[engine_idx] - current:
            series = (*labelvalues, ALL_KV_GROUPS, block_id, last_owner)
            if hasattr(gauge, "remove"):
                gauge.remove(*series)
            else:  # Ray metrics cannot drop a series; zero it instead.
                gauge.labels(*series).set(0)
        for block_id, last_owner in current:
            gauge.labels(*labelvalues, ALL_KV_GROUPS, block_id, last_owner).set(1)
        self._orphan_labels[engine_idx] = current
