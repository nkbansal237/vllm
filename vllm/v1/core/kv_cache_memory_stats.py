# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side statistics about HBM KV cache memory.

Collects, for the Prometheus metrics in `vllm/v1/metrics/loggers.py`:

- why admissions are refused (`classify_refusal`, `KVAdmissionRefusalTracker`),
- claimed external tokens whose KV load failed (`KVLoadFailureTracker`),
- token slots in the blocks of running and loading requests
  (`compute_token_slots`),
- the state of every block and leaked blocks (`KVBlockStateScanner`).
"""

import time
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager,
    )
    from vllm.v1.request import Request

# Admission refusal reasons, from the most to the least fundamental.
EXCEEDS_POOL_CAPACITY = "exceeds_pool_capacity"
INSUFFICIENT_FREE_BLOCKS = "insufficient_free_blocks"
WATERMARK = "watermark"
INFLIGHT_LOAD_RESERVE = "inflight_load_reserve"
HEAD_OF_LINE = "head_of_line"
REFUSAL_REASONS = (
    EXCEEDS_POOL_CAPACITY,
    INSUFFICIENT_FREE_BLOCKS,
    WATERMARK,
    INFLIGHT_LOAD_RESERVE,
    HEAD_OF_LINE,
)

# Kinds of reserved free blocks.
RESERVED_WATERMARK = "watermark"
RESERVED_INFLIGHT_LOAD = "inflight_load"
RESERVED_KINDS = (RESERVED_WATERMARK, RESERVED_INFLIGHT_LOAD)

# Block states, in priority order: a block matching several states counts as
# the first one.
ACTIVE_SHARED = "active_shared"
ACTIVE_PRIVATE = "active_private"
HELD_FOR_SEND = "held_for_send"
LOADING_KV = "loading_kv"
WAITING_HOLD = "waiting_hold"
TRANSIENT_PIN = "transient_pin"
ORPHAN_SUSPECT = "orphan_suspect"
CACHED_EVICTABLE = "cached_evictable"
FREE_UNHASHED = "free_unhashed"
BLOCK_STATES = (
    ACTIVE_SHARED,
    ACTIVE_PRIVATE,
    HELD_FOR_SEND,
    LOADING_KV,
    WAITING_HOLD,
    TRANSIENT_PIN,
    ORPHAN_SUSPECT,
    CACHED_EVICTABLE,
    FREE_UNHASHED,
)
# Holder states by priority; lower wins. Active is split into shared/private
# after counting running referrers.
_ACTIVE = 0
_HOLDER_STATES = (_ACTIVE, HELD_FOR_SEND, LOADING_KV, WAITING_HOLD, TRANSIENT_PIN)
_PRIORITY = {state: i for i, state in enumerate(_HOLDER_STATES)}

# Kinds of token slots.
SLOTS_FILLED = "filled"
SLOTS_TAIL_SLACK = "tail_slack"
SLOTS_RESERVED = "reserved"

MAX_ORPHANS_REPORTED = 16


def classify_refusal(
    num_required: int,
    num_free: int,
    watermark_blocks: int,
    reserved_blocks: int,
    capacity: int,
) -> str:
    """Return why an allocation of `num_required` blocks was refused.

    The reasons are checked from the most to the least fundamental: an
    allocation refused for the watermark would fit without it, and one refused
    for the in-flight load reserve would fit without that reserve.
    """
    if num_required > capacity:
        return EXCEEDS_POOL_CAPACITY
    if num_required > num_free:
        return INSUFFICIENT_FREE_BLOCKS
    if num_required + watermark_blocks > num_free:
        return WATERMARK
    return INFLIGHT_LOAD_RESERVE


class KVAdmissionRefusalTracker:
    """Counts refused admissions of waiting requests, once per episode.

    The head of the waiting queue is retried every step, so a refusal counts
    only when a request starts being refused for a reason. Requests queued
    behind a refused head count once as `head_of_line` when the head's
    episode starts.
    """

    def __init__(self) -> None:
        self._reason_by_req_id: dict[str, str] = {}
        self._blocked_head_req_id: str | None = None
        self._counts: Counter[str] = Counter()

    def record_refusal(self, req_id: str, reason: str, num_queued_behind: int) -> None:
        if self._reason_by_req_id.get(req_id) != reason:
            self._reason_by_req_id[req_id] = reason
            self._counts[reason] += 1
        if self._blocked_head_req_id != req_id:
            self._blocked_head_req_id = req_id
            self._counts[HEAD_OF_LINE] += num_queued_behind

    def record_admission(self, req_id: str) -> None:
        self._reason_by_req_id.pop(req_id, None)
        if self._blocked_head_req_id == req_id:
            self._blocked_head_req_id = None

    def forget(self, req_id: str) -> None:
        self.record_admission(req_id)

    def take_counts(self) -> dict[str, int]:
        counts = dict(self._counts)
        self._counts.clear()
        return counts


class KVLoadFailureTracker:
    """Counts claimed external tokens whose KV load failed.

    A claim is recorded when a connector's external tokens are admitted.
    After a load failure, the tokens a request lost (its drop in
    `num_computed_tokens`) count against the remaining claim, labeled with
    the connector that made it and the failure policy.
    """

    def __init__(self, policy: str) -> None:
        self.policy = policy
        # req_id -> (connector, claimed tokens not yet counted as failed)
        self._claims: dict[str, tuple[str, int]] = {}
        self._failed_tokens: Counter[str] = Counter()

    def record_claim(self, req_id: str, connector: str, num_tokens: int) -> None:
        if num_tokens > 0:
            self._claims[req_id] = (connector, num_tokens)

    def snapshot(self, requests: dict[str, "Request"]) -> dict[str, int]:
        """`num_computed_tokens` of every request holding a claim."""
        return {
            req_id: request.num_computed_tokens
            for req_id in self._claims
            if (request := requests.get(req_id)) is not None
        }

    def record_failures(
        self,
        before: dict[str, int],
        requests: dict[str, "Request"],
        failed_req_ids: Iterable[str] = (),
    ) -> None:
        """Count the tokens lost since `before`.

        Requests in `failed_req_ids` (e.g. failed under the `fail` policy
        without rolling back their computed tokens) lose their whole claim.
        """
        failed = set(failed_req_ids)
        for req_id, num_computed_before in before.items():
            request = requests.get(req_id)
            num_computed_after = (
                request.num_computed_tokens if request is not None else 0
            )
            num_lost = num_computed_before - num_computed_after
            connector, num_claimed = self._claims[req_id]
            if req_id in failed:
                num_lost = num_claimed
            num_failed = min(num_claimed, max(num_lost, 0))
            if num_failed:
                self._failed_tokens[connector] += num_failed
                self._claims[req_id] = (connector, num_claimed - num_failed)

    def forget(self, req_id: str) -> None:
        self._claims.pop(req_id, None)

    def take_counts(self) -> list[tuple[str, str, int]]:
        """(connector, policy, failed tokens) since the last call."""
        counts = [
            (connector, self.policy, num_tokens)
            for connector, num_tokens in self._failed_tokens.items()
        ]
        self._failed_tokens.clear()
        return counts


def _is_token_group(manager: "SingleTypeKVCacheManager") -> bool:
    # A Mamba state block holds one state whatever its size, and
    # cross-attention blocks hold encoder tokens; neither has token slots.
    return not isinstance(manager, (MambaManager, CrossAttentionManager))


def compute_token_slots(
    managers: Iterable["SingleTypeKVCacheManager"],
    requests: Iterable["Request"],
) -> list[tuple[str, str, int]]:
    """Token slots in the blocks held by `requests`, per KV cache group.

    Returns (kv_group, kind, slots) with kinds:
    - filled: computed tokens still held (tokens outside a sliding window,
      whose blocks are released, are excluded),
    - tail_slack: unused slots in each request's last computed block,
    - reserved: slots in blocks allocated past the computed tokens
      (scheduled, lookahead and speculative slots).

    Shared prefix blocks count once per request that references them.
    """
    requests = list(requests)
    slots: list[tuple[str, str, int]] = []
    for group_id, manager in enumerate(managers):
        if not _is_token_group(manager):
            continue
        block_size = manager.block_size
        filled = tail_slack = reserved = 0
        for request in requests:
            blocks = manager.req_to_blocks.get(request.request_id)
            if not blocks:
                continue
            num_computed = request.num_computed_tokens
            num_computed_blocks = -(-num_computed // block_size)
            skipped = manager.get_num_skipped_tokens(num_computed)
            filled += num_computed - skipped // block_size * block_size
            tail_slack += num_computed_blocks * block_size - num_computed
            reserved += max(len(blocks) - num_computed_blocks, 0) * block_size
        group = str(group_id)
        slots += [
            (group, SLOTS_FILLED, filled),
            (group, SLOTS_TAIL_SLACK, tail_slack),
            (group, SLOTS_RESERVED, reserved),
        ]
    return slots


class KVBlockStateScanner:
    """Classifies every block of the HBM pool, at most once per interval.

    A scan walks the block tables of all requests that hold blocks and the
    scheduler's pending frees, then every block of the pool. It costs time
    proportional to the referenced blocks plus the pool size, which is why
    it runs periodically rather than every step.

    A block whose ref_cnt exceeds the references found is an orphan suspect.
    References a connector takes outside the block tables would look like
    leaks, so a block is reported only when the same excess is seen on two
    consecutive scans.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._next_scan = 0.0
        self._prev_owner: dict[int, str] = {}
        self._prev_excess: dict[int, int] = {}
        self._orphans: dict[int, str] = {}

    def maybe_scan(
        self,
        block_pool: "BlockPool",
        managers: Iterable["SingleTypeKVCacheManager"],
        requests: Iterable["Request"],
        finished_loading_req_ids: set[str],
        pinned_blocks: Iterable["KVCacheBlock"],
    ) -> tuple[dict[str, int], list[tuple[int, str]]] | None:
        now = time.monotonic()
        if now < self._next_scan:
            return None
        self._next_scan = now + self.interval_s
        return self.scan(
            block_pool, managers, requests, finished_loading_req_ids, pinned_blocks
        )

    def scan(
        self,
        block_pool: "BlockPool",
        managers: Iterable["SingleTypeKVCacheManager"],
        requests: Iterable["Request"],
        finished_loading_req_ids: set[str],
        pinned_blocks: Iterable["KVCacheBlock"],
    ) -> tuple[dict[str, int], list[tuple[int, str]]]:
        """Return the block count per state and up to 16 orphan suspects.

        `pinned_blocks` are blocks the scheduler holds outside the block
        tables until a step completes (deferred frees, copy-on-write sources).
        """
        managers = list(managers)
        state: dict[int, str | int] = {}
        owner: dict[int, str] = {}
        num_refs: Counter[int] = Counter()
        num_running_refs: Counter[int] = Counter()

        def hold(block: "KVCacheBlock", block_state: str | int, owner_id: str):
            if block.is_null or block.pool is not block_pool:
                return
            block_id = block.block_id
            num_refs[block_id] += 1
            if block_state == _ACTIVE:
                num_running_refs[block_id] += 1
            prev = state.get(block_id)
            if prev is None or _PRIORITY[block_state] < _PRIORITY[prev]:
                state[block_id] = block_state
                owner[block_id] = owner_id

        for request in requests:
            req_id = request.request_id
            if request.status == RequestStatus.RUNNING:
                request_state: str | int = _ACTIVE
            elif request.is_finished():
                request_state = (
                    LOADING_KV if req_id in finished_loading_req_ids else HELD_FOR_SEND
                )
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                request_state = LOADING_KV
            else:
                request_state = WAITING_HOLD
            for manager in managers:
                for block in manager.req_to_blocks.get(req_id, ()):
                    hold(block, request_state, req_id)
        for block in pinned_blocks:
            hold(block, TRANSIENT_PIN, "none")

        counts = dict.fromkeys(BLOCK_STATES, 0)
        excess: dict[int, int] = {}
        for block in block_pool.blocks:
            if block.is_null:
                continue
            block_id = block.block_id
            if block.ref_cnt == 0:
                is_cached = (
                    block.block_hash is not None
                    or block_id in block_pool.cached_block_hashes_by_block
                )
                counts[CACHED_EVICTABLE if is_cached else FREE_UNHASHED] += 1
                continue
            num_excess = block.ref_cnt - num_refs[block_id]
            if num_excess > 0:
                excess[block_id] = num_excess
            block_state = state.get(block_id)
            if block_state is None:
                counts[ORPHAN_SUSPECT] += 1
            elif block_state == _ACTIVE:
                shared = num_running_refs[block_id] > 1
                counts[ACTIVE_SHARED if shared else ACTIVE_PRIVATE] += 1
            else:
                counts[str(block_state)] += 1

        # Report a leak once its excess is stable across two scans.
        for block_id in list(self._orphans):
            if block_id not in excess:
                del self._orphans[block_id]
        for block_id, num_excess in excess.items():
            if (
                self._prev_excess.get(block_id) == num_excess
                and block_id not in self._orphans
            ):
                self._orphans[block_id] = self._prev_owner.get(block_id, "none")
        self._prev_excess = excess
        self._prev_owner = owner
        orphans = list(self._orphans.items())[-MAX_ORPHANS_REPORTED:]
        return counts, orphans
