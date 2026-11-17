"""A request that carries its own max_concurrent is held to it on both backends (TASK-13435)."""

import asyncio
import uuid

import pytest
from redis.exceptions import RedisError

from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RedisResourceGovernor, RGRequest

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


class FakeTime:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at t=1000 (the Redis governor wipes leases once when the clock reads < 1)."""
        self.t = 1000.0

    def __call__(self) -> float:
        """Now."""
        return self.t


class _Loader:
    """Serve one policy for every id."""

    def __init__(self, policy: dict) -> None:
        """Keep the policy."""
        self.policy = policy

    def get_policy(self, _policy_id: str) -> dict:
        """The same policy for any id."""
        return dict(self.policy)


@pytest.fixture()
async def redis_ns():
    """A Redis namespace of its own, whose keys are deleted after the test, as rg_unique_ns does."""
    ns = f"rg_rcl_{uuid.uuid4().hex[:8]}"
    yield ns
    try:
        client = await create_async_redis_client(fallback_to_fake=False, context="rg_rcl_cleanup")
    except (OSError, RedisError):
        return  # No real Redis: the [redis] cases ran on the in-memory stub.
    async for key in client.scan_iter(match=f"{ns}:*"):
        await client.delete(key)
    await client.aclose()


def _governor(backend: str, policy: dict, clock: FakeTime, ns: str):
    """A fresh governor of the given backend; Redis uses the test's own namespace."""
    if backend == "memory":
        return MemoryResourceGovernor(policy_loader=_Loader(policy), time_source=clock)
    return RedisResourceGovernor(policy_loader=_Loader(policy), time_source=clock, ns=ns)


def _lease(entity: str, limit: int, ttl_sec: int = 120) -> RGRequest:
    """One jobs lease, held to the request's own limit and TTL."""
    return RGRequest(
        entity=entity,
        categories={"jobs": {"units": 1, "max_concurrent": limit, "ttl_sec": ttl_sec}},
        tags={"policy_id": "limits.test_concurrent"},
    )


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_limit_applies_without_a_policy_block(backend: str, redis_ns: str) -> None:
    """A policy with no jobs block still holds a request to the request's own max_concurrent; release frees a slot."""
    rg = _governor(backend, {"scopes": ["user"]}, FakeTime(), redis_ns)
    d1, h1 = await rg.reserve(_lease("user:1", 2))
    d2, h2 = await rg.reserve(_lease("user:1", 2))
    d3, h3 = await rg.reserve(_lease("user:1", 2))
    assert (d1.allowed, d2.allowed, d3.allowed) == (True, True, False)
    assert h3 is None
    await rg.release(h1)
    d4, h4 = await rg.reserve(_lease("user:1", 2))
    assert d4.allowed and h4


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_limit_charges_only_the_callers_bucket(backend: str, redis_ns: str) -> None:
    """A request's own limit is that caller's: another user's leases and the policy's global bucket don't count."""
    rg = _governor(backend, {"jobs": {"max_concurrent": 1}, "scopes": ["global", "user"]}, FakeTime(), redis_ns)
    d1, _h1 = await rg.reserve(_lease("user:1", 1))
    d2, _h2 = await rg.reserve(_lease("user:2", 1))
    assert d1.allowed and d2.allowed


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_renewing_a_request_limited_lease_never_charges_global(backend: str, redis_ns: str) -> None:
    """Renewing user 1's request-limited lease leaves the policy's global bucket free for user 2's policy-limited lease."""
    rg = _governor(backend, {"jobs": {"max_concurrent": 1}, "scopes": ["global", "user"]}, FakeTime(), redis_ns)
    _d1, h1 = await rg.reserve(_lease("user:1", 1))
    await rg.renew(h1, 120)
    policy_limited = RGRequest(entity="user:2", categories={"jobs": {"units": 1}}, tags={"policy_id": "limits.test_concurrent"})
    d2, h2 = await rg.reserve(policy_limited)
    assert d2.allowed and h2


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_ttl_lets_an_unreleased_lease_lapse(backend: str, redis_ns: str) -> None:
    """A lease that is never released lapses after the request's ttl_sec, then the slot is free again."""
    clock = FakeTime()
    rg = _governor(backend, {"scopes": ["user"]}, clock, redis_ns)
    d1, _h1 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    d2, _h2 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    clock.t += 31
    d3, h3 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    assert (d1.allowed, d2.allowed, d3.allowed) == (True, False, True)
    assert h3


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_simultaneous_requests_get_exactly_the_limit(backend: str, redis_ns: str) -> None:
    """Six requests racing for a limit of 2 (a UI firing parallel uploads): exactly two are admitted."""
    rg = _governor(backend, {"scopes": ["user"]}, FakeTime(), redis_ns)
    results = await asyncio.gather(*(rg.reserve(_lease("user:1", 2)) for _ in range(6)))
    assert sum(1 for decision, _handle in results if decision.allowed) == 2


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_renew_keeps_a_lease_past_its_first_ttl(backend: str, redis_ns: str) -> None:
    """Renewing a held lease pushes its expiry out, so the limit still holds after the first ttl_sec."""
    clock = FakeTime()
    rg = _governor(backend, {"scopes": ["user"]}, clock, redis_ns)
    _d1, h1 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    clock.t += 20
    await rg.renew(h1, 30)
    clock.t += 20
    d2, h2 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    assert not d2.allowed and h2 is None
