# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HBM KV cache memory statistics."""

from functools import partial
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge

from vllm.v1.core.kv_cache_memory_stats import (
    ACTIVE_PRIVATE,
    ACTIVE_SHARED,
    BLOCK_STATES,
    CACHED_EVICTABLE,
    EXCEEDS_POOL_CAPACITY,
    FREE_UNHASHED,
    HEAD_OF_LINE,
    HELD_FOR_SEND,
    INFLIGHT_LOAD_RESERVE,
    INSUFFICIENT_FREE_BLOCKS,
    LOADING_KV,
    ORPHAN_SUSPECT,
    TRANSIENT_PIN,
    WATERMARK,
    KVAdmissionRefusalTracker,
    KVLoadFailureTracker,
    classify_refusal,
)
from vllm.v1.metrics.kv_cache_memory import KVCacheMemoryProm
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

BLOCK_SIZE = 16


@pytest.mark.parametrize(
    ("num_required", "num_free", "watermark", "reserved", "reason"),
    [
        (20, 30, 0, 0, EXCEEDS_POOL_CAPACITY),  # capacity is 10
        (8, 5, 0, 0, INSUFFICIENT_FREE_BLOCKS),
        (4, 5, 2, 0, WATERMARK),
        (4, 9, 2, 4, INFLIGHT_LOAD_RESERVE),
    ],
)
def test_classify_refusal(num_required, num_free, watermark, reserved, reason):
    assert (
        classify_refusal(num_required, num_free, watermark, reserved, capacity=10)
        == reason
    )


def test_refusals_count_once_per_episode():
    tracker = KVAdmissionRefusalTracker()
    tracker.record_refusal("a", INSUFFICIENT_FREE_BLOCKS, num_queued_behind=2)
    tracker.record_refusal("a", INSUFFICIENT_FREE_BLOCKS, num_queued_behind=3)
    assert tracker.take_counts() == {INSUFFICIENT_FREE_BLOCKS: 1, HEAD_OF_LINE: 2}

    # A new reason for the same request is a new episode, not a new head.
    tracker.record_refusal("a", WATERMARK, num_queued_behind=3)
    assert tracker.take_counts() == {WATERMARK: 1}

    tracker.record_admission("a")
    tracker.record_refusal("a", WATERMARK, num_queued_behind=1)
    assert tracker.take_counts() == {WATERMARK: 1, HEAD_OF_LINE: 1}


def _request(num_computed_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(num_computed_tokens=num_computed_tokens)


def test_load_failures_count_lost_claimed_tokens():
    tracker = KVLoadFailureTracker("recompute")
    tracker.record_claim("a", "P2PConnector", 64)
    tracker.record_claim("b", "P2PConnector", 32)
    requests = {"a": _request(80), "b": _request(48)}

    before = tracker.snapshot(requests)
    requests["a"].num_computed_tokens = 48  # rolled back 32 claimed tokens
    tracker.record_failures(before, requests)
    assert tracker.take_counts() == [("P2PConnector", "recompute", 32)]

    # A second failure counts only the claim that is left.
    before = tracker.snapshot(requests)
    requests["a"].num_computed_tokens = 0
    tracker.record_failures(before, requests)
    assert tracker.take_counts() == [("P2PConnector", "recompute", 32)]


def test_load_failures_fail_policy_loses_whole_claim():
    tracker = KVLoadFailureTracker("fail")
    tracker.record_claim("a", "OffloadConnector", 64)
    requests = {"a": _request(80)}
    before = tracker.snapshot(requests)
    tracker.record_failures(before, requests, failed_req_ids={"a"})
    assert tracker.take_counts() == [("OffloadConnector", "fail", 64)]

    tracker.forget("a")
    assert tracker.snapshot(requests) == {}


def _scheduler(num_blocks: int = 100):
    scheduler = create_scheduler(
        num_blocks=num_blocks, block_size=BLOCK_SIZE, enable_prefix_caching=True
    )
    scheduler.kv_block_state_scanner.interval_s = 0
    return scheduler


def test_admission_refusals_and_head_of_line():
    # 9 usable blocks; each request takes 4 blocks and needs 5 for its full
    # sequence, so two are admitted and the third is refused.
    scheduler = _scheduler(num_blocks=10)
    for request in create_requests(4, num_tokens=64, max_tokens=16):
        scheduler.add_request(request)

    scheduler.schedule()
    stats = scheduler.make_stats()
    assert stats is not None
    assert len(scheduler.running) == 2
    assert stats.kv_admission_refusals == {
        INSUFFICIENT_FREE_BLOCKS: 1,
        HEAD_OF_LINE: 1,
    }
    assert stats.kv_admission_reserved_blocks == {"watermark": 0, "inflight_load": 0}

    # The same refusal on the next step is the same episode.
    scheduler.schedule()
    stats = scheduler.make_stats()
    assert stats is not None
    assert stats.kv_admission_refusals == {}


def test_token_slots():
    scheduler = _scheduler()
    scheduler.add_request(create_requests(1, num_tokens=70)[0])
    scheduler.schedule()
    stats = scheduler.make_stats()
    assert stats is not None
    slots = {kind: num for _, kind, num in stats.kv_token_slots}
    assert slots == {"filled": 70, "tail_slack": 10, "reserved": 0}


def _states(scheduler) -> dict[str, int]:
    blocks_by_state, _ = scheduler._make_kv_block_state_stats()
    assert blocks_by_state is not None
    usable = scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1
    assert sum(blocks_by_state.values()) == usable
    return {state: n for state, n in blocks_by_state.items() if n}


def test_block_states():
    scheduler = _scheduler(num_blocks=20)
    first, second = create_requests(2, num_tokens=64, same_prompt=True)
    scheduler.add_request(first)
    scheduler.schedule()
    assert _states(scheduler) == {ACTIVE_PRIVATE: 4, FREE_UNHASHED: 15}

    # The second request shares the first one's cached prefix blocks.
    scheduler.add_request(second)
    scheduler.schedule()
    states = _states(scheduler)
    assert states[ACTIVE_SHARED] > 0

    # Finished requests whose blocks are kept, e.g. for a P2P send, hold them.
    first.status = RequestStatus.FINISHED_LENGTH_CAPPED
    second.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert set(_states(scheduler)) == {HELD_FOR_SEND, FREE_UNHASHED}
    second.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    assert LOADING_KV in _states(scheduler)

    # Once freed, full blocks stay cached (the prefix, and blocks the second
    # request computed itself).
    for request in (first, second):
        request.status = RequestStatus.FINISHED_STOPPED
        scheduler._free_blocks(request)
    states = _states(scheduler)
    assert states[CACHED_EVICTABLE] >= 4
    # Freed blocks wait for the step fence before returning to the pool.
    assert set(states) <= {CACHED_EVICTABLE, FREE_UNHASHED, TRANSIENT_PIN}


def test_orphan_reported_after_two_scans():
    scheduler = _scheduler(num_blocks=20)
    leaked = scheduler.kv_cache_manager.block_pool.get_new_blocks(1)[0]

    blocks_by_state, orphans = scheduler._make_kv_block_state_stats()
    assert blocks_by_state is not None and blocks_by_state[ORPHAN_SUSPECT] == 1
    assert orphans == []
    _, orphans = scheduler._make_kv_block_state_stats()
    assert orphans == [(leaked.block_id, "none")]

    scheduler.kv_cache_manager.block_pool.free_blocks([leaked])
    _, orphans = scheduler._make_kv_block_state_stats()
    assert orphans == []


def test_prometheus_metrics():
    registry = CollectorRegistry()
    prom = KVCacheMemoryProm(
        partial(Gauge, registry=registry),
        partial(Counter, registry=registry),
        ["model_name", "engine"],
        {0: ["m", "0"]},
    )
    labels = {"model_name": "m", "engine": "0"}

    def value(name: str, **extra: str) -> float | None:
        return registry.get_sample_value(name, {**labels, **extra})

    # Fixed series exist before any event.
    assert value("vllm:kv_admission_refusals_total", reason=HEAD_OF_LINE) == 0
    assert all(
        value("vllm:kv_cache_blocks_by_state", kv_group="all", state=state) == 0
        for state in BLOCK_STATES
    )

    stats = SchedulerStats(
        kv_admission_reserved_blocks={"watermark": 3, "inflight_load": 5},
        kv_admission_refusals={WATERMARK: 2},
        kv_load_failed_tokens=[("P2PConnector", "recompute", 48)],
        kv_token_slots=[("0", "filled", 70), ("0", "tail_slack", 10)],
        kv_blocks_by_state={ACTIVE_PRIVATE: 4, FREE_UNHASHED: 5},
        kv_orphan_blocks=[(7, "req-1")],
    )
    prom.observe(stats)
    prom.observe(SchedulerStats(kv_admission_refusals={WATERMARK: 1}))

    assert value("vllm:kv_admission_reserved_blocks", kind="inflight_load") == 5
    assert value("vllm:kv_admission_refusals_total", reason=WATERMARK) == 3
    assert (
        value(
            "vllm:kv_load_failed_tokens_total",
            connector="P2PConnector",
            policy="recompute",
        )
        == 48
    )
    assert value("vllm:kv_token_slots", kv_group="0", kind="tail_slack") == 10
    # Block states keep the last scan between scans.
    assert (
        value("vllm:kv_cache_blocks_by_state", kv_group="all", state=ACTIVE_PRIVATE)
        == 4
    )
    orphan = {"kv_group": "all", "block_id": "7", "last_owner": "req-1"}
    assert value("vllm:kv_orphan_block_info", **orphan) == 1

    prom.observe(SchedulerStats(kv_orphan_blocks=[]))
    assert value("vllm:kv_orphan_block_info", **orphan) is None
