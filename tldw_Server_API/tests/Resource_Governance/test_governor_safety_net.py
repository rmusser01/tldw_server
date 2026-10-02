"""Safety-net governor behaviour: no configuration can produce a permanent 429.

These tests run against every backend listed in BACKENDS.
"""

import itertools
from typing import Any

import pytest

from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest
from tldw_Server_API.app.core.Resource_Governance.policy_eval import effective_policy, requests_window

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

BACKENDS = ["memory", "redis"]
_ns = itertools.count()


class FakeTime:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class _Loader:
    def __init__(self, policies):
        self.policies = policies

    def get_policy(self, pid):
        return self.policies.get(pid)


def _gov(backend, policies, clock):
    loader = _Loader(policies)
    if backend == "memory":
        return MemoryResourceGovernor(policy_loader=loader, time_source=clock), loader

    from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
    from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor

    gov = RedisResourceGovernor(policy_loader=loader, time_source=clock, ns=f"rg_t_safety_{next(_ns)}")
    # Inject the in-process stub directly so this test never touches a real Redis
    # on 127.0.0.1:6379 if one happens to be running; makes repeated runs
    # deterministic instead of depending on the ambient environment.
    gov._client = InMemoryAsyncRedis()
    return gov, loader


def _req(entity, policy_id, **cats):
    return RGRequest(entity=entity, categories=cats or {"requests": {"units": 1}}, tags={"policy_id": policy_id})


async def _admits(gov, req, n):
    out = []
    for i in range(n):
        # A monotonic counter, not id(req): a short-lived RGRequest can be GC'd and
        # its address reused by the *next* _admits() call's request, which would
        # collide op_ids across logically-unrelated reserve batches and replay a
        # stale cached decision (test flake, not a governor bug).
        dec, _h = await gov.reserve(req, op_id=f"{req.entity}-{req.tags['policy_id']}-{next(_ns)}-{i}")
        out.append(dec.allowed)
    return out


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


async def test_unknown_policy_uses_default_limits(backend):
    clock = FakeTime()
    gov, _ = _gov(backend, {"default": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "no.such.policy"), 3) == [True, True, False]
    clock.advance(61)
    assert await _admits(gov, _req("user:1", "no.such.policy"), 1) == [True]


async def test_missing_default_uses_builtin(backend):
    gov, _ = _gov(backend, {}, FakeTime())
    assert all(await _admits(gov, _req("user:1", "anything"), 50))


async def test_scope_mismatch_charges_per_entity_bucket(backend, monkeypatch):
    if backend == "redis":
        # Without this, Redis's in-process accept-window tracker admits/denies by
        # counting successful admits per (policy, entity) independent of scope
        # buckets, which would pass even if the scope-selection bug were present.
        # Disabling it forces the assertion through the per-entity ZSET window.
        monkeypatch.setenv("RG_TEST_DISABLE_ACCEPT_WINDOW", "1")
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user", "api_key"]}}, FakeTime())
    assert await _admits(gov, _req("ip:1.2.3.4", "p"), 3) == [True, True, False]


async def test_tokens_scope_mismatch_charges_per_entity_bucket(backend):
    # No accept-window override needed here: that tracker only ever gates the
    # "requests" category (see check()'s requests branch), never "tokens".
    gov, _ = _gov(backend, {"p": {"tokens": {"per_min": 100, "burst": 1.0}, "scopes": ["user"]}}, FakeTime())
    req = _req("ip:1.2.3.4", "p", tokens={"units": 100})
    assert await _admits(gov, req, 3) == [True, False, False]


async def test_streams_scope_mismatch_lease_released_on_commit(backend):
    gov, _ = _gov(backend, {"p": {"streams": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"]}}, FakeTime())
    req = _req("ip:9.9.9.9", "p", streams={"units": 1})
    results = []
    for i in range(3):
        dec, handle_id = await gov.reserve(req, op_id=f"lease-{backend}-{next(_ns)}-{i}")
        results.append(dec.allowed)
        if handle_id:
            await gov.commit(handle_id)
    assert results == [True, True, True]


async def test_policy_without_requests_inherits_default(backend):
    policies = {
        "default": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]},
        "tok": {"tokens": {"per_min": 1000}, "scopes": ["user"]},
    }
    gov, _ = _gov(backend, policies, FakeTime())
    assert await _admits(gov, _req("user:1", "tok"), 3) == [True, True, False]


async def test_missing_concurrency_config_is_unbounded(backend):
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 100}, "scopes": ["user"]}}, FakeTime())
    req = _req("user:1", "p", jobs={"units": 1})
    assert all(await _admits(gov, req, 5))


async def test_oversized_token_reservation_admitted_on_full_bucket(backend):
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 1000}, "tokens": {"per_min": 100, "burst": 1.0}, "scopes": ["user"]}}, clock)
    big = _req("user:1", "p", tokens={"units": 10_000})
    assert await _admits(gov, big, 1) == [True]
    assert await _admits(gov, _req("user:1", "p", tokens={"units": 1}), 1) == [False]
    clock.advance(61)
    assert await _admits(gov, big, 1) == [True]


async def test_raised_limit_applies_without_restart(backend):
    clock = FakeTime()
    gov, loader = _gov(backend, {"p": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 3) == [True, True, False]
    loader.policies["p"] = {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]}
    clock.advance(1)  # refills 10 units at the new rate
    assert all(await _admits(gov, _req("user:1", "p"), 5))


async def test_policy_store_failure_is_logged_once_and_default_applies(backend, monkeypatch):
    from loguru import logger

    from tldw_Server_API.app.core.Resource_Governance import policy_eval

    monkeypatch.setattr(policy_eval, "_warned_lookup_errors", set())
    monkeypatch.setattr(policy_eval, "_warned_unknown", set())

    class _BrokenLoader:
        def get_policy(self, pid):
            raise RuntimeError("PolicyLoader not initialized")

    gov, _ = _gov(backend, {}, FakeTime())
    gov._policy_loader = _BrokenLoader()
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="ERROR")
    try:
        admitted = await _admits(gov, _req("user:1", "p"), 2)
    finally:
        logger.remove(sink)
    assert admitted == [True, True]
    assert len([m for m in seen if "'p'" in m and "PolicyLoader not initialized" in m]) == 1, seen


async def test_fractional_rpm_admits_burst_then_refills_after_window(backend: str) -> None:
    # The shipped authnz.magic_link.email policy: 3 up front and 0.3/min long-run on
    # both backends. Redis holds floor(rpm * burst) = 3 per 60 * 3 / rpm = 600 s; the
    # memory bucket refills one unit per 200 s and is full again at 600 s.
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.3, "burst": 10.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 4) == [True, True, True, False]
    clock.advance(600)
    assert await _admits(gov, _req("user:1", "p"), 4) == [True, True, True, False]


async def test_fractional_rpm_bucket_holds_one_unit_per_window(backend: str) -> None:
    # rpm * burst < 1 used to be a bucket that could never hold one request.
    # effective_policy raises burst to 2: one request per 120 s on both backends.
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.5, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 2) == [True, False]
    clock.advance(119)
    assert await _admits(gov, _req("user:1", "p"), 1) == [False]
    clock.advance(1)
    assert await _admits(gov, _req("user:1", "p"), 2) == [True, False]


@pytest.mark.parametrize("burst", [3.0, 2.2])
async def test_fractional_rpm_non_integer_capacity_keeps_burst_and_average(backend: str, burst: float) -> None:
    # rpm * burst = 1.5 / 1.1: the memory bucket admits 1 up front and refills 0.5/min.
    # Redis holds floor(rpm * burst) = 1 per 60 * 1 / rpm = 120 s: the same burst and average.
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.5, "burst": burst}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 2) == [True, False]
    clock.advance(119)
    if backend == "redis":  # the slot frees only when the first request leaves the window
        assert await _admits(gov, _req("user:1", "p"), 1) == [False]
    clock.advance(1)
    assert await _admits(gov, _req("user:1", "p"), 2) == [True, False]


async def test_fractional_rpm_near_integer_capacity_floors_like_memory(backend: str) -> None:
    # rpm * burst = 1.9999996: the memory bucket admits 1; rounding before the floor gave Redis 2.
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.5, "burst": 3.9999992}, "scopes": ["user"]}}, FakeTime())
    assert await _admits(gov, _req("user:1", "p"), 2) == [True, False]


async def test_policy_reload_that_shortens_the_window_applies_mid_window(backend: str) -> None:
    # {0.25, 1} and {0.5, 1} both hold one request; the window shrinks from 240 s to 120 s.
    clock = FakeTime()
    gov, loader = _gov(backend, {"p": {"requests": {"rpm": 0.25, "burst": 1.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 1) == [True]
    clock.advance(10)
    loader.policies["p"] = {"requests": {"rpm": 0.5, "burst": 1.0}, "scopes": ["user"]}
    assert await _admits(gov, _req("user:1", "p"), 1) == [False]
    clock.advance(120)  # past the new 120 s window, still inside the old 240 s one
    assert await _admits(gov, _req("user:1", "p"), 1) == [True]


async def test_redis_retry_in_the_last_second_of_a_window_is_one_second() -> None:
    # A slot frees within the second: report 1, not the 600 s window, and admit right after.
    clock = FakeTime()
    policies = {"p": {"requests": {"rpm": 0.3, "burst": 10.0}, "scopes": ["user"]}}
    worker_a, _ = _gov("redis", policies, clock)
    worker_b, _ = _gov("redis", policies, clock)
    worker_b._keys, worker_b._client = worker_a._keys, worker_a._client  # two workers, one Redis
    req = _req("user:1", "p")
    assert await _admits(worker_b, req, 1) == [True]
    clock.advance(1)
    assert await _admits(worker_a, req, 2) == [True, True]  # the shared window is now full
    clock.advance(598.5)  # the first request leaves the 600 s window in 0.5 s
    assert (await worker_a.check(req)).retry_after == 1  # acceptance-window path
    # Worker B's tracker counts only its own admit, so its denial comes from the ZSET.
    dec, _h = await worker_b.reserve(req, op_id=f"late-{next(_ns)}")
    assert (dec.allowed, dec.retry_after) == (False, 1)
    clock.advance(1)
    assert await _admits(worker_b, req, 1) == [True]


@pytest.mark.parametrize(
    ("requests", "expected"),
    [
        ({"rpm": 600, "burst": 2.0}, (600, 60)),  # integer rpm: Redis does not apply burst
        ({"rpm": 1}, (1, 60)),
        ({"rpm": 1.5}, (2, 60)),  # non-integer rpm >= 1 rounds up
        ({"rpm": 0.3, "burst": 10.0}, (3, 600)),  # rpm < 1, whole capacity
        ({"rpm": 0.5, "burst": 3.0}, (1, 120)),  # rpm < 1, fractional capacity floors
        ({"rpm": 0.5, "burst": 3.9999992}, (1, 120)),  # near-integer capacity floors like memory
        ({"rpm": 0.29, "burst": 100.0}, (28, 5794)),  # 28.999999999999996 floors to 28, as in memory
        ({"rpm": 0.5, "burst": 1.0}, (1, 120)),  # effective_policy raises burst to 2
        ({"rpm": 0.41}, (1, 147)),  # raised burst 2.439...; the window rounds up
    ],
)
async def test_requests_window_boundaries(requests: dict[str, float], expected: tuple[int, int]) -> None:
    # async only because the module-wide asyncio mark warns on sync tests.
    assert requests_window(effective_policy({"p": {"requests": requests}}.get, "p")) == expected


async def test_fractional_rpm_decision_reports_a_limit_of_at_least_one(backend: str) -> None:
    # Rate-limit headers come from this limit; int(0.3) == 0 used to be reported.
    policies = {
        "burst10": {"requests": {"rpm": 0.3, "burst": 10.0}, "scopes": ["user"]},
        "noburst": {"requests": {"rpm": 0.3}, "scopes": ["user"]},
    }
    gov, _ = _gov(backend, policies, FakeTime())
    limits = [(await gov.check(_req("user:1", pid))).details["categories"]["requests"]["limit"] for pid in policies]
    assert limits == [3, 1]


async def test_redis_fractional_rpm_peek_and_retry_after_use_the_same_window() -> None:
    clock = FakeTime()
    gov, _ = _gov("redis", {"p": {"requests": {"rpm": 0.3, "burst": 10.0}, "scopes": ["user"]}}, clock)
    assert await _admits(gov, _req("user:1", "p"), 3) == [True, True, True]
    clock.advance(100)
    dec = await gov.check(_req("user:1", "p"))
    assert (dec.allowed, dec.retry_after) == (False, 500)
    assert await gov.peek_with_policy("user:1", ["requests"], "p") == {"requests": {"remaining": 0, "reset": 500}}


async def test_fractional_rpm_float_rounding_still_holds_one_unit(backend):
    # 0.41 * (1 / 0.41) == 0.9999999999999999, which the memory bucket truncates to 0.
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.41, "burst": 1.0}, "scopes": ["user"]}}, FakeTime())
    assert (await _admits(gov, _req("user:1", "p"), 1))[0] is True


async def test_policy_without_scopes_has_no_server_wide_bucket(backend):
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 1, "burst": 1.0}}}, FakeTime())
    assert await _admits(gov, _req("user:1", "p"), 1) == [True]
    assert await _admits(gov, _req("user:2", "p"), 1) == [True]  # user:1 spent only its own bucket


async def test_memory_refund_without_scopes_creates_no_global_bucket():
    gov, _ = _gov("memory", {"p": {"requests": {"rpm": 100}, "tokens": {"per_min": 100}}}, FakeTime())
    _dec, handle = await gov.reserve(_req("user:1", "p", tokens={"units": 10}), op_id="r")
    await gov.commit(handle, actuals={"tokens": 5})
    assert [k for k in gov._buckets if k[2] == "global"] == []


async def test_memory_evicts_full_idle_buckets_only():
    clock = FakeTime()
    policies = {
        "fast": {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]},
        "slow": {"requests": {"rpm": 0.01, "burst": 100.0}, "scopes": ["user"]},
    }
    gov, _ = _gov("memory", policies, clock)
    await _admits(gov, _req("user:1", "fast"), 1)
    await _admits(gov, _req("user:1", "slow"), 1)
    clock.advance(601)
    await _admits(gov, _req("user:2", "fast"), 1)  # triggers the sweep
    keys = set(gov._buckets)
    assert ("fast", "requests", "user", "1") not in keys  # refilled and idle: evicted
    assert ("slow", "requests", "user", "1") in keys  # still draining: kept


async def test_memory_eviction_sweeps_rotate_through_every_bucket(monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import governor as governor_mod

    monkeypatch.setattr(governor_mod, "_EVICT_BATCH", 2)
    clock = FakeTime()
    policies = {
        "fast": {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]},
        "slow": {"requests": {"rpm": 0.01, "burst": 100.0}, "scopes": ["user"]},
    }
    gov, _ = _gov("memory", policies, clock)
    # Insertion order: kept, kept, evictable, evictable. Batch size 2: without rotation
    # both sweeps would revisit the two kept buckets and never reach the evictable ones.
    for user, pid in ((1, "slow"), (2, "slow"), (3, "fast"), (4, "fast")):
        await _admits(gov, _req(f"user:{user}", pid), 1)
    clock.advance(601)
    gov._maybe_evict_idle(clock())
    clock.advance(61)
    gov._maybe_evict_idle(clock())
    assert {k[3] for k in gov._buckets} == {"1", "2"}


async def test_memory_eviction_batch_scales_with_a_flood(monkeypatch):
    # A fixed batch falls behind a flood of one-shot entities; a sweep covers >= 1/10 of the map.
    from tldw_Server_API.app.core.Resource_Governance import governor as governor_mod

    monkeypatch.setattr(governor_mod, "_EVICT_BATCH", 2)
    clock = FakeTime()
    gov, _ = _gov("memory", {"fast": {"requests": {"rpm": 600, "burst": 1.0}, "scopes": ["user"]}}, clock)
    for user in range(100):
        await _admits(gov, _req(f"user:{user}", "fast"), 1)
        gov._leases[("fast", "streams", "user", str(user))] = {}
    clock.advance(601)
    gov._maybe_evict_idle(clock())
    assert (len(gov._buckets), len(gov._leases)) == (90, 90)


def _redis_maps(gov: Any) -> dict[str, dict]:
    """The Redis governor's in-process maps that the eviction sweep bounds, by name."""
    return {
        "accept_window": gov._requests_accept_window,
        "deny_until": gov._requests_deny_until,
        "backoff_until": gov._stub_backoff_until,
        "leases": gov._stub_leases,
    }


async def test_redis_in_process_maps_drop_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeTime()
    policy = {"requests": {"rpm": 1, "burst": 1.0}, "streams": {"max_concurrent": 1, "ttl_sec": 30}, "scopes": ["user"]}
    gov, _ = _gov("redis", {"p": policy}, clock)

    async def _no_stub_lease_purge(**_kw: Any) -> None:
        """Real Redis never runs the stub-only lease purge."""

    monkeypatch.setattr(gov, "_maybe_test_purge_leases", _no_stub_lease_purge)
    for user in range(50):
        assert await _admits(gov, _req(f"user:{user}", "p"), 2) == [True, False]
        await gov.reserve(_req(f"user:{user}", "p", streams={"units": 1}), op_id=f"lease-{next(_ns)}")
    assert {name: len(m) >= 50 for name, m in _redis_maps(gov).items()} == dict.fromkeys(_redis_maps(gov), True)
    clock.advance(61)
    assert await _admits(gov, _req("user:new", "p"), 1) == [True]  # triggers the sweep
    assert {name: len(m) <= 1 for name, m in _redis_maps(gov).items()} == dict.fromkeys(_redis_maps(gov), True)
    # An evicted entity behaves exactly like a fresh one.
    assert await _admits(gov, _req("user:0", "p"), 2) == [True, False]


async def test_redis_sweep_keeps_live_deny_floors_and_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeTime()
    policy = {"requests": {"rpm": 0.01, "burst": 100.0}, "streams": {"max_concurrent": 2, "ttl_sec": 100}, "scopes": ["user"]}
    gov, _ = _gov("redis", {"p": policy}, clock)

    async def _no_stub_lease_purge(**_kw: Any) -> None:
        """Real Redis never runs the stub-only lease purge."""

    monkeypatch.setattr(gov, "_maybe_test_purge_leases", _no_stub_lease_purge)
    assert await _admits(gov, _req("user:floor", "p"), 1) == [True]  # deny floor until +6000 s
    lease = _req("user:lease", "p", streams={"units": 1})
    await gov.reserve(lease, op_id=f"lease-{next(_ns)}")  # expires at +100 s
    clock.advance(50)
    await gov.reserve(lease, op_id=f"lease-{next(_ns)}")  # expires at +150 s
    clock.advance(60)  # first lease lapsed, second live
    # A floor that ends exactly now has expired: reads deny only while now < until.
    gov._requests_deny_until[(gov._keys.ns, "p", "user:edge", 1, 6000)] = clock()
    gov._maybe_evict_idle(clock())
    assert {k[2] for k in gov._requests_deny_until} == {"user:floor"}
    assert gov._stub_lease_purge_and_count(key=gov._keys.lease("p", "streams", "user", "lease"), now=clock()) == 1
    assert await _admits(gov, _req("user:floor", "p"), 1) == [False]


async def test_redis_eviction_sweeps_rotate_through_every_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    monkeypatch.setattr(governor_redis, "_EVICT_BATCH", 2)
    clock = FakeTime()
    policies = {
        "fast": {"requests": {"rpm": 10, "burst": 1.0}, "scopes": ["user"]},  # 60 s window
        "slow": {"requests": {"rpm": 0.01, "burst": 100.0}, "scopes": ["user"]},  # 6000 s window
    }
    gov, _ = _gov("redis", policies, clock)
    # Insertion order: kept, kept, evictable, evictable. Batch size 2: without rotation
    # both sweeps would revisit the two kept entries and never reach the evictable ones.
    for user, pid in ((1, "slow"), (2, "slow"), (3, "fast"), (4, "fast")):
        await _admits(gov, _req(f"user:{user}", pid), 1)
    clock.advance(61)
    gov._maybe_evict_idle(clock())
    clock.advance(61)
    gov._maybe_evict_idle(clock())
    assert {k[2] for k in gov._requests_accept_window} == {"user:1", "user:2"}


async def test_redis_eviction_batch_scales_with_a_flood(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fixed batch falls behind a flood of one-shot entities; a sweep covers >= 1/10 of the map.
    from tldw_Server_API.app.core.Resource_Governance import governor_redis

    monkeypatch.setattr(governor_redis, "_EVICT_BATCH", 2)
    clock = FakeTime()
    gov, _ = _gov("redis", {"p": {"requests": {"rpm": 10, "burst": 1.0}, "scopes": ["user"]}}, clock)
    for user in range(100):
        await _admits(gov, _req(f"user:{user}", "p"), 1)
    clock.advance(61)
    gov._maybe_evict_idle(clock())
    assert len(gov._requests_accept_window) == 90


async def test_memory_expired_op_id_is_not_replayed_before_the_purge_runs(monkeypatch):
    clock = FakeTime()
    gov, _ = _gov("memory", {"p": {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["user"]}}, clock)
    monkeypatch.setattr(gov, "_purge_expired_ops", lambda _now: None)
    req = _req("user:1", "p")
    assert (await gov.reserve(req, op_id="a"))[0].allowed
    assert not (await gov.reserve(req, op_id="x"))[0].allowed  # denial cached under "x"
    clock.advance(gov._op_ttl + 1)  # "x" has expired and the bucket has refilled
    assert (await gov.reserve(req, op_id="x"))[0].allowed


async def test_memory_ops_purge_runs_at_most_once_per_interval():
    from tldw_Server_API.app.core.Resource_Governance.governor import _OPS_PURGE_INTERVAL_SEC

    clock = FakeTime()
    gov, _ = _gov("memory", {"p": {"requests": {"rpm": 600}, "scopes": ["user"]}}, clock)
    await gov.reserve(_req("user:1", "p"), op_id="old")
    clock.advance(gov._op_ttl - 1)
    await gov.reserve(_req("user:1", "p"), op_id="t1")  # purges; "old" not yet expired
    clock.advance(2)
    await gov.reserve(_req("user:1", "p"), op_id="t2")  # "old" expired, but within the interval
    assert "reserve:old" in gov._ops
    clock.advance(_OPS_PURGE_INTERVAL_SEC)
    await gov.reserve(_req("user:1", "p"), op_id="t3")
    assert "reserve:old" not in gov._ops


# --- Redis tokens window: one ZSET member per quantum of max(1, per_min // 1000) tokens ---

_BIG_TOKENS = {"p": {"tokens": {"per_min": 1_000_000, "burst": 1.5}, "scopes": ["user"]}}
_SMALL_TOKENS = {"p": {"tokens": {"per_min": 500, "burst": 1.0}, "scopes": ["user"]}}


async def _token_members(gov):
    return await gov._client.zcard(gov._keys.win("p", "tokens", "user", "1"))


async def _reserve_tokens(gov, units):
    dec, handle = await gov.reserve(_req("user:1", "p", tokens={"units": units}), op_id=f"tok-{next(_ns)}")
    return dec.allowed, handle


@pytest.mark.parametrize("add_path", ["stub", "python", "lua"])
async def test_redis_tokens_window_holds_one_member_per_quantum(add_path, monkeypatch):
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    lua_argv = []
    if add_path != "stub":
        # Drive reserve() down the real-Redis branches while still on the in-process stub.
        async def _real():
            return True

        monkeypatch.setattr(gov, "_is_real_redis", _real)
    if add_path == "python":

        async def _no_lua():
            return None

        monkeypatch.setattr(gov, "_ensure_multi_reserve_lua", _no_lua)
    if add_path == "lua":

        async def _lua():
            return "sha"

        async def _evalsha(_sha, _nkeys, *args):
            lua_argv.append(args)
            return [1, 0]

        monkeypatch.setattr(gov, "_ensure_multi_reserve_lua", _lua)
        monkeypatch.setattr(gov._client, "evalsha", _evalsha)

    allowed, handle = await _reserve_tokens(gov, 50_000)

    assert allowed and handle
    if add_path == "lua":
        limit, _window, units, csv = lua_argv[0][-4:]
        assert (limit, units, len(csv.split(","))) == (1000, 50, 50)
    else:
        assert await _token_members(gov) == 50


async def test_redis_tokens_quantized_window_admits_per_min_then_denies():
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    assert [(await _reserve_tokens(gov, 50_000))[0] for _ in range(20)] == [True] * 20
    assert not (await gov.check(_req("user:1", "p", tokens={"units": 1}))).allowed
    assert (await _reserve_tokens(gov, 1))[0] is False


async def test_redis_tokens_peek_reports_remaining_in_tokens():
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    await _reserve_tokens(gov, 50_000)
    peek = await gov.peek_with_policy("user:1", ["tokens"], "p")
    assert peek["tokens"]["remaining"] == 950_000


@pytest.mark.parametrize("how", ["refund", "commit"])
async def test_redis_tokens_quantized_refund_restores_capacity(how):
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    handles = [(await _reserve_tokens(gov, 50_000))[1] for _ in range(20)]
    if how == "refund":
        await gov.refund(handles[0], deltas={"tokens": 50_000})
    else:
        await gov.commit(handles[0], actuals={"tokens": 0})
    assert (await _reserve_tokens(gov, 50_000))[0] is True
    assert (await _reserve_tokens(gov, 1))[0] is False


async def test_redis_tokens_partial_commit_removes_unused_quanta():
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    _allowed, handle = await _reserve_tokens(gov, 50_000)  # 50 members at quantum 1000
    await gov.commit(handle, actuals={"tokens": 20_000})  # refunds 30_000 tokens = 30 members
    assert await _token_members(gov) == 20


async def test_redis_tokens_refund_rounds_down_to_whole_members():
    gov, _ = _gov("redis", _BIG_TOKENS, FakeTime())
    _allowed, handle = await _reserve_tokens(gov, 1_500)  # charges ceil(1.5) = 2 members
    await gov.refund(handle, deltas={"tokens": 1_500})  # refunds floor(1.5) = 1 member
    assert await _token_members(gov) == 1


async def test_redis_tokens_small_per_min_keeps_one_member_per_token():
    gov, _ = _gov("redis", _SMALL_TOKENS, FakeTime())
    _allowed, handle = await _reserve_tokens(gov, 200)
    assert await _token_members(gov) == 200
    assert (await _reserve_tokens(gov, 300))[0] is True
    assert (await _reserve_tokens(gov, 1))[0] is False
    await gov.refund(handle, deltas={"tokens": 100})
    assert await _token_members(gov) == 400
    assert (await _reserve_tokens(gov, 100))[0] is True
    assert (await _reserve_tokens(gov, 1))[0] is False
