"""Safety-net governor behaviour: no configuration can produce a permanent 429.

These tests run against every backend listed in BACKENDS.
"""

import itertools

import pytest

from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest

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
    # Insertion order: kept, evictable, evictable, kept. Batch size 2, so two sweeps
    # cover 4 slots; every bucket must be visited even as evictions shrink the dict.
    for user, pid in ((1, "slow"), (2, "fast"), (3, "fast"), (4, "slow")):
        await _admits(gov, _req(f"user:{user}", pid), 1)
    clock.advance(601)
    gov._maybe_evict_idle(clock())
    clock.advance(61)
    gov._maybe_evict_idle(clock())
    assert {k[3] for k in gov._buckets} == {"1", "4"}
