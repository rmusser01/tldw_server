import time

import pytest
import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest


class _BrokenClient:
    async def evalsha(self, *a, **k):
        raise RuntimeError("boom")
    async def script_load(self, *a, **k):
        raise RuntimeError("boom")
    async def zadd(self, *a, **k):
        raise RuntimeError("boom")
    async def zremrangebyscore(self, *a, **k):
        raise RuntimeError("boom")
    async def zcard(self, *a, **k):
        raise RuntimeError("boom")
    async def set(self, *a, **k):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_fail_open_tokens_allows_on_error(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 1, "fail_mode": "fail_open"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fail_open")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    d, h = await rg.reserve(RGRequest(entity="user:x", categories={"tokens": {"units": 1}}, tags={"policy_id": "p"}))
    assert d.allowed and h


@pytest.mark.asyncio
async def test_fail_closed_requests_denies_on_error(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 1, "fail_mode": "fail_closed"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fail_closed")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    d, h = await rg.reserve(RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert (not d.allowed) and (h is None)


@pytest.mark.asyncio
async def test_fallback_memory_requests_allows_when_redis_broken(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 1, "fail_mode": "fallback_memory"}, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_fallback_mem")

    async def _broken():
        return _BrokenClient()

    monkeypatch.setattr(rg, "_client_get", _broken)
    req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
    d1, h1 = await rg.reserve(req, op_id="f1")
    assert d1.allowed and h1
    d2, h2 = await rg.reserve(req, op_id="f2")
    # Memory fallback should enforce rpm=1
    assert (not d2.allowed) and (h2 is None)


def _redis_down_client() -> aioredis.Redis:
    """A real redis-py client whose server is gone: every command raises redis.exceptions.ConnectionError.

    Nothing listens on port 1, and retries are off, so each command fails at once.
    """
    return aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0), decode_responses=True)


def _governor(policy: dict, ns: str, time_source=time.time) -> RedisResourceGovernor:
    class _Loader:
        def get_policy(self, pid):
            return policy

    return RedisResourceGovernor(policy_loader=_Loader(), ns=ns, time_source=time_source)


@pytest.mark.asyncio
@pytest.mark.parametrize("category, limits", [("requests", {"rpm": 5}), ("tokens", {"per_min": 5})])
@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
async def test_reserve_honours_the_fail_mode_when_redis_raises_connection_error(category, limits, fail_mode, admitted):
    rg = _governor({category: limits, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_{category}_{fail_mode}")
    rg._client = _redis_down_client()  # the worker connected once; Redis has gone away since
    req = RGRequest(entity="user:x", categories={category: {"units": 1}}, tags={"policy_id": "p"})

    decision, handle = await rg.reserve(req)

    assert (decision.allowed, handle is not None) == (admitted, admitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_renew_and_commit_survive_redis_raising_connection_error(fail_mode):
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_lease_{fail_mode}")
    rg._client = InMemoryAsyncRedis()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    decision, handle = await rg.reserve(req)
    assert decision.allowed and handle

    rg._client = _redis_down_client()  # Redis goes away while the lease is held
    await rg.renew(handle, 60)
    await rg.commit(handle)


@pytest.mark.asyncio
async def test_peek_reports_unknown_headroom_when_redis_raises_connection_error():
    rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"]}, ns="rg_down_peek")
    rg._client = _redis_down_client()

    # Ingress headers and the character rate limiter's usage stats read this; it must not raise.
    assert await rg.peek_with_policy("user:x", ["requests"], "p") == {"requests": {"remaining": None, "reset": None}}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_a_lease_committed_while_redis_is_down_frees_its_slot(fail_mode, monkeypatch):
    # The RG conftest's test-only lease purge would drop the in-process lease for us; production never runs it.
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_free_{fail_mode}")
    rg._client = InMemoryAsyncRedis()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    _decision, handle = await rg.reserve(req)

    rg._client = _redis_down_client()
    await rg.commit(handle)
    decision, again = await rg.reserve(req)

    assert decision.allowed and again


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_leases_count_per_worker_while_redis_is_down(fail_mode, monkeypatch):
    # The RG conftest's test-only lease purge drops the in-process leases once Redis is down; production never runs it.
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_count_{fail_mode}")
    rg._client = _redis_down_client()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})

    first, handle = await rg.reserve(req)
    second, _ = await rg.reserve(req)
    await rg.commit(handle)
    third, _ = await rg.reserve(req)

    assert [first.allowed, second.allowed, third.allowed] == [True, False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
async def test_renew_while_redis_is_down_keeps_the_lease(fail_mode, monkeypatch):
    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
    now = [1000.0]
    rg = _governor(
        {"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode},
        ns=f"rg_down_renew_{fail_mode}",
        time_source=lambda: now[0],
    )
    rg._client = _redis_down_client()
    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
    first, handle = await rg.reserve(req)  # the lease runs to 1060

    now[0] = 1050.0
    await rg.renew(handle, 60)  # now to 1110
    now[0] = 1070.0
    second, _ = await rg.reserve(req)

    assert first.allowed and not second.allowed
