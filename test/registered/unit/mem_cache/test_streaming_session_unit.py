from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ReqKvInfo,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.srt.utils.common import is_npu
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_npu_ci(est_time=20, suite="nightly-1-npu-a3", nightly=True)


class _FakeAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double. Subclassing the base routes free_full / free_segment /
    free_segments into free(), so a new free API cannot slip past the recorder."""

    def __init__(self, page_size: int = 1):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.free_slots = []

    def free(self, req):
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


class _FakeInnerCache:
    """Stands in for UnifiedRadixCache: owns the session and tries it first."""

    free_kv_row = BasePrefixCache.free_kv_row

    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.disable = False
        self.match_results = list(match_results or [])
        self.session = StreamingSession(self)

    def checkpoint(self, req, *, up_to):
        pass

    def maybe_hand_to_session(self, req):
        self.session.take(req)

    def match_prefix(self, params):
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def unlock(self, lock):
        pass


class _FakeReq:
    release_mm_inputs_on_abort = Req.release_mm_inputs_on_abort
    finished = Req.finished

    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
            _inflight=False,
        )
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
            cache_protected_len=0,
        )
        self.origin_input_ids = list(range(committed))
        self.full_untruncated_fill_ids = self.origin_input_ids.copy()
        self.sampling_params = SimpleNamespace(max_new_tokens=1)
        self.retraction_count = 0
        self.output_ids = []
        self.extra_key = None
        self.cache_salt = None
        self.last_node = None
        self.swa_branching_seqlen = None
        self.lock = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None
        self.multimodal_inputs = None
        self._session_mm_items = []

    def detach_kv(self):
        kv, self.kv = self.kv, ReqKvInfo()
        return kv


def _single_row_cache():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    return _FakeInnerCache(
        _FakeReqToTokenPool(req_to_token), _FakeAllocator(), page_size=1
    )


def _finished_turn(tree_cache, req):
    """The session took the record at allocation; the turn then finished."""
    tree_cache.maybe_hand_to_session(req)
    req.finished_reason = FINISH_LENGTH(length=0)
    assert tree_cache.session.try_cache_finished_req(req)
    return tree_cache.session.slots[req.session.session_id]


def test_finished_turn_leaves_its_record_in_the_slot():
    req = _FakeReq("session-a", req_pool_idx=0, committed=4, allocated=4)
    record = req.kv
    record.mamba_next_track_idx = 1
    record.set_evicted_seqlen(ComponentType.SWA, 2)
    req.swa_branching_seqlen = 8

    slot = _finished_turn(_single_row_cache(), req)

    assert slot.kv is record
    assert slot.kv.mamba_next_track_idx == 1
    assert slot.kv.component_evicted_seqlens == {ComponentType.SWA: 2}
    assert req.kv is not record
    assert req.swa_branching_seqlen is None


def test_preabort_detaches_session_and_preserves_slot():
    """A request aborted before its match is detached from the session; the
    slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator(page_size=16)
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_prefix_len=0,
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )
        ],
    )
    tree_cache = inner
    tree_cache.session.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=48,
            kv_allocated_len=48,
            cache_protected_len=16,
        ),
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    assert req.session is None
    slot = tree_cache.session.slots["session-a"]
    assert slot.kv.req_pool_idx == 0
    assert slot.kv.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert result.device_prefix_len == 0


def test_trim_overshoot_postcondition():
    """Every per-request KV cursor is capped at origin + finished_len and the
    tail past it is freed."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = _FakeInnerCache(req_to_token_pool, allocator, page_size)

    # target = 26 + 12 = 38; the overshoot committed 40 and allocated 44.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.set_evicted_seqlen(ComponentType.SWA, 42)

    tree_cache.session._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == target
    assert len(req.output_ids) == 12
    # [38, 42) already gave its SWA back, so the free splits at 42.
    assert [t.tolist() for t in allocator.freed] == [[38, 39, 40, 41], [42, 43]]


@pytest.mark.parametrize(
    "target, end, freed",
    [(17, 20, []), (17, 32, []), (17, 40, [(32, 40)]), (32, 48, [(32, 48)])],
)
def test_free_kv_aligned_skips_a_range_inside_the_partial_page(target, end, freed):
    """Rounding ``target`` up to a page boundary can move the start of the
    range at or past ``end``; such a rewind frees nothing and must not reach
    the cache at all, while a range that still spans a whole page is freed."""
    calls = []
    session = StreamingSession(
        SimpleNamespace(
            page_size=16, free_kv_row=lambda _kv, ranges: calls.extend(ranges)
        )
    )
    session._free_kv_aligned(None, target, end)
    assert calls == freed


@pytest.mark.parametrize("operation", ["trim", "match"])
@pytest.mark.parametrize("component", [ComponentType.SWA, ComponentType.AUXILIARY_SWA])
def test_session_rewind_keeps_component_cursors_page_aligned(operation, component):
    """Rewinding below any component cursor must free whole pages."""
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator(page_size=page_size)
    tree_cache = _FakeInnerCache(req_to_token_pool, allocator, page_size)

    # origin=26, finished=12 -> raw target 38 (mid-page); cursor 48 > target.
    req = _FakeReq("session-a", req_pool_idx=0, committed=52, allocated=64)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.set_evicted_seqlen(ComponentType.SWA, 16)
    req.kv.set_evicted_seqlen(component, 48)

    if operation == "trim":
        tree_cache.session._trim_overshoot(req, finished_len=12)
        assert len(req.output_ids) == 12
    else:
        tree_cache.session.slots["session-a"] = SessionSlot(kv=req.detach_kv())
        req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
        result = tree_cache.match_prefix(SimpleNamespace(req=req, key=list(range(38))))
        assert result.device_prefix_len == 32

    # Rewound to floor_align(38) = 32; every cursor lands page-aligned.
    assert req.kv.kv_allocated_len == 32
    assert req.kv.kv_committed_len == 32
    assert req.kv.get_evicted_seqlen(component) == 32
    assert req.kv.max_evicted_seqlen == 32
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == (
        32 if component == ComponentType.SWA else 16
    )
    assert [t.tolist() for t in allocator.freed] == (
        [list(range(32, 48)), list(range(48, 64))]
        if component == ComponentType.SWA
        else [list(range(32, 64))]
    )


@pytest.fixture(params=[False, True], ids=["attention", "mamba"])
def npu_private_session(request, monkeypatch):
    if not is_npu():
        pytest.skip("requires NPU runtime")
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.session import streaming_session

    # Only session ownership is under test; device allocation has separate tests.
    allocator = object.__new__(NPUPagedTokenToKVPoolAllocator)
    allocator._kvcache = object.__new__(NPUMHATokenToKVPool)
    pool = _FakeReqToTokenPool(torch.arange(512).reshape(1, 512))
    pool._aux_cache = None
    pool.mamba_allocator = Mock()
    cache = _FakeInnerCache(pool, allocator, page_size=128)
    cache.disable = True
    cache.is_mamba_enabled = request.param
    cache.tree_components = (
        (ComponentType.FULL, ComponentType.MAMBA)
        if request.param
        else (ComponentType.FULL,)
    )
    cache.enable_mamba_extra_buffer = False
    cache.free_kv_row = Mock()
    runtime = SimpleNamespace(
        attn_dcp_size=1,
        disaggregation_mode="null",
        disable_overlap_schedule=True,
        speculative_algorithm=None,
    )
    for getter in ("get_parallel", "get_disagg", "get_schedule", "get_spec"):
        monkeypatch.setattr(streaming_session, getter, lambda: runtime)
    return cache, runtime


@pytest.mark.parametrize(
    "page_size,committed,key_len", [(128, 129, 129), (1, 129, 128)]
)
@pytest.mark.parametrize("already_borrowed", [False, True])
def test_mamba_rewind_drops_state_before_prefix_rematch(
    monkeypatch, page_size, committed, key_len, already_borrowed
):
    """Replaying a KV tail must not start from the state after that tail."""
    from sglang.srt.session import streaming_session

    monkeypatch.setattr(streaming_session, "is_npu", lambda: True)
    pool = _FakeReqToTokenPool(torch.arange(256).reshape(1, 256))
    pool.mamba_allocator = Mock()
    cache = _FakeInnerCache(pool, _FakeAllocator(page_size), page_size)
    previous = _FakeReq("session-a", 0, committed, committed)
    previous.kv.mamba_pool_idx = torch.tensor(1)
    cache.session.take(previous)
    slot = cache.session.slots["session-a"]
    req = _FakeReq("session-a", None, key_len + 1, 0)
    if already_borrowed:
        req.kv = slot.kv

    result = cache.session.try_match_prefix(
        SimpleNamespace(req=req, key=list(range(key_len)))
    )

    assert result is None  # Let the cache rematch a compatible KV + Mamba checkpoint.
    assert not req.kv.holds_kv and not req.kv.holds_mamba
    assert not cache.session.slots
    assert pool.free_slots == [0]
    pool.mamba_allocator.free.assert_called_once()


def test_mamba_finished_overshoot_invalidates_the_retained_state():
    """Finishing at an earlier position must not retain the later Mamba state."""
    cache = _single_row_cache()
    cache.req_to_token_pool.mamba_allocator = Mock()
    req = _FakeReq("session-a", 0, 40, 40)
    req.kv.mamba_pool_idx = torch.tensor(1)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.finished_len = 12
    req.finished_reason = FINISH_LENGTH(length=12)
    req.session.finish_req = Mock()
    cache.session.take(req)

    assert cache.session.try_cache_finished_req(req)

    req.session.finish_req.assert_called_once_with(req)
    assert len(req.output_ids) == 12
    assert not cache.session.slots
    assert not req.kv.holds_mamba and not req.kv.holds_kv
    cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
    freed = torch.cat(cache.token_to_kv_pool_allocator.freed).tolist()
    assert sorted(freed) == list(range(40))


def _npu_finished_turn(cache, length):
    req = _FakeReq("session-a", 0, length, length)
    if cache.is_mamba_enabled:
        req.kv.mamba_pool_idx = torch.tensor(1)
    cache.session.take(req)
    req.output_ids = [length]
    req.finished_reason = FINISH_LENGTH(length=1)
    assert cache.session.try_cache_finished_req(req)
    return cache.session.slots["session-a"]


@pytest.mark.parametrize("start", [57, 127])
def test_npu_private_tail_appends_and_releases(npu_private_session, start):
    """A 129-position session must reuse 129 positions, not rewind to 128."""
    cache, _ = npu_private_session
    slot = _npu_finished_turn(cache, start)
    for length in range(start, start + 4):
        req = _FakeReq("session-a", 0, length + 1, 0)
        result = cache.session.try_match_prefix(
            SimpleNamespace(req=req, key=list(range(length)))
        )
        assert result is not None
        assert result.device_prefix_len == length
        assert req.kv is slot.kv
        assert req.kv.holds_mamba == cache.is_mamba_enabled
        req.kv.kv_committed_len = req.kv.kv_allocated_len = length + 1
        req.output_ids = [length + 1]
        req.finished_reason = FINISH_LENGTH(length=1)
        assert cache.session.try_cache_finished_req(req)
    cache.free_kv_row.assert_not_called()

    cache.session.release_session("session-a")
    cache.session.release_session("session-a")
    cache.free_kv_row.assert_called_once_with(slot.kv, [(0, start + 4)])
    assert cache.req_to_token_pool.free_slots == [0]
    assert cache.req_to_token_pool.mamba_allocator.free.call_count == int(
        cache.is_mamba_enabled
    )

    req = _FakeReq("session-a", 0, 3, 3)
    if cache.is_mamba_enabled:
        req.kv.mamba_pool_idx = torch.tensor(2)
    cache.session.take(req)
    req.finished_reason = FINISH_ABORT("cancelled")
    assert not cache.session.try_cache_finished_req(req)
    assert req.kv.holds_kv
    assert req.kv.holds_mamba == cache.is_mamba_enabled
    assert not cache.session.slots
    assert cache.req_to_token_pool.free_slots == [0]
    assert cache.req_to_token_pool.mamba_allocator.free.call_count == int(
        cache.is_mamba_enabled
    )


@pytest.mark.parametrize(
    "target,attribute,value",
    [
        ("cache", "disable", False),
        ("cache", "enable_mamba_extra_buffer", True),
        ("cache", "token_to_kv_pool_allocator", None),
        ("pool", "_aux_cache", object()),
        ("kv", "cache_protected_len", 1),
        ("kv", "component_evicted_seqlens", {ComponentType.SWA: 1}),
        ("kv", "kv_committed_len", 128),
        ("kv", "kv_allocated_len", 130),
        ("kv", "mamba_needs_clear", True),
        ("kv", "mamba_cow_src_index", 2),
        ("kv", "mamba_ping_pong_track_buffer", torch.tensor([1, 2])),
        ("kv", "retraction_backup", object()),
        ("req", "retraction_count", 1),
        ("req", "full_untruncated_fill_ids", list(range(131))),
        ("sampling", "max_new_tokens", 2),
        ("runtime", "attn_dcp_size", 2),
        ("runtime", "disaggregation_mode", "prefill"),
        ("runtime", "disable_overlap_schedule", False),
        ("runtime", "speculative_algorithm", "EAGLE"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_npu_private_tail_rejects_unsafe_reuse(
    npu_private_session, target, attribute, value
):
    """Fallback may align KV only if it does not leave recurrent state ahead."""
    cache, runtime = npu_private_session
    slot = _npu_finished_turn(cache, 129)
    req = _FakeReq("session-a", 0, 130, 0)
    targets = dict(
        cache=cache,
        pool=cache.req_to_token_pool,
        kv=slot.kv,
        req=req,
        sampling=req.sampling_params,
        runtime=runtime,
    )
    original = getattr(targets[target], attribute)
    setattr(targets[target], attribute, value)
    if attribute == "mamba_ping_pong_track_buffer":
        slot.kv.mamba_ping_pong_empty = [False, False]
    result = cache.session.try_match_prefix(
        SimpleNamespace(req=req, key=list(range(129)))
    )
    if cache.is_mamba_enabled:
        if attribute == "kv_committed_len":
            assert result.device_prefix_len == 128
        else:
            assert result is None
            assert not cache.session.slots
        return
    assert result.device_prefix_len == 128
    # A later successful turn cannot certify recurrent state after a rewind.
    setattr(targets[target], attribute, original)
    req.kv.kv_committed_len = req.kv.kv_allocated_len = 130
    req.output_ids = [130]
    req.finished_reason = FINISH_LENGTH(length=1)
    assert cache.session.try_cache_finished_req(req)
    resumed = _FakeReq("session-a", 0, 131, 0)
    result = cache.session.try_match_prefix(
        SimpleNamespace(req=resumed, key=list(range(130)))
    )
    assert result.device_prefix_len == 128


def test_npu_retracted_first_turn_cannot_certify_tail(npu_private_session):
    """Retraction reaches cache_finished_req before retraction_count advances."""
    cache, _ = npu_private_session
    req = _FakeReq("session-a", 0, 57, 57)
    if cache.is_mamba_enabled:
        req.kv.mamba_pool_idx = torch.tensor(1)
    req.output_ids = [57]
    req.finished_len = 1
    cache.session.take(req)
    assert cache.session.try_cache_finished_req(req)
    resumed = _FakeReq("session-a", 0, 58, 0)
    assert (
        cache.session.try_match_prefix(
            SimpleNamespace(req=resumed, key=list(range(57)))
        )
        is None
    )
    cache.free_kv_row.assert_called_once()
    assert not cache.session.slots


def test_npu_private_tail_requires_matching_cache_state(npu_private_session):
    """A completed prefix is insufficient if its recurrent state is missing."""
    cache, _ = npu_private_session
    slot = _npu_finished_turn(cache, 129)
    slot.kv.mamba_pool_idx = None if cache.is_mamba_enabled else torch.tensor(1)
    req = _FakeReq("session-a", 0, 130, 0)
    result = cache.session.try_match_prefix(
        SimpleNamespace(req=req, key=list(range(129)))
    )
    if cache.is_mamba_enabled:
        assert result.device_prefix_len == 128
    else:
        assert result is None
        assert not cache.session.slots


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
