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


async def test_fractional_rpm_admits_then_refills(backend):
    # The shipped authnz.magic_link.email policy. Redis once used int(rpm) == 0 here.
    clock = FakeTime()
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.3, "burst": 10.0}, "scopes": ["user"]}}, clock)
    first = await _admits(gov, _req("user:1", "p"), 5)
    assert first[0] and not first[-1], first
    clock.advance(201)  # one unit refills in 60 / 0.3 = 200 s
    assert await _admits(gov, _req("user:1", "p"), 1) == [True]


async def test_fractional_rpm_bucket_holds_at_least_one_unit(backend):
    # rpm * burst < 1 used to be a bucket that could never hold one request.
    gov, _ = _gov(backend, {"p": {"requests": {"rpm": 0.5, "burst": 1.0}, "scopes": ["user"]}}, FakeTime())
    assert (await _admits(gov, _req("user:1", "p"), 3))[0] is True


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
