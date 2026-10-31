"""Window keys on real Redis expire once their entity goes idle (TASK-13430)."""

import asyncio
import time

import pytest

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest, governor_redis
from tldw_Server_API.app.core.Resource_Governance.governor_redis import _WINDOW_TTL_MARGIN_S, _window_ttl

pytestmark = [pytest.mark.integration, pytest.mark.rate_limit]

_REQ = RGRequest(entity="user:idle", categories={"requests": {"units": 1}, "tokens": {"units": 1}}, tags={"policy_id": "pttl"})


async def _real_governor(ns: str, policy: dict) -> RedisResourceGovernor:
    class _Loader:
        def get_policy(self, pid):
            return {**policy, "scopes": ["global", "user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns=ns)
    if not await rg._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    return rg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "windows"),
    [
        ({"requests": {"rpm": 5}, "tokens": {"per_min": 100}}, {"requests": 60, "tokens": 60}),
        # rpm < 1: 3 requests per 600 s, while tokens keep 60 s in the same reserve.
        ({"requests": {"rpm": 0.3, "burst": 10}, "tokens": {"per_min": 100}}, {"requests": 600, "tokens": 60}),
    ],
)
async def test_real_redis_multi_lua_reserve_sets_window_ttl(real_redis, rg_unique_ns, policy, windows):
    rg = await _real_governor(rg_unique_ns, policy)

    decision, handle_id = await rg.reserve(_REQ)
    assert decision.allowed and handle_id
    assert rg._last_used_multi_lua is True

    client = await rg._client_get()
    for category, window in windows.items():
        for scope in ("global:*", "user:idle"):
            ttl = await client.ttl(f"{rg_unique_ns}:win:pttl:{category}:{scope}")
            assert window <= ttl <= window + _WINDOW_TTL_MARGIN_S


@pytest.mark.asyncio
async def test_real_redis_tokens_lua_add_sets_window_ttl(real_redis, rg_unique_ns):
    rg = await _real_governor(rg_unique_ns, {})
    client = await rg._client_get()
    key = f"{rg_unique_ns}:win:pttl:tokens:user:idle"

    sha = await rg._ensure_tokens_lua()
    assert await client.evalsha(sha, 1, key, 10, 60, time.time(), _window_ttl(60)) == [1, 0]
    assert 60 <= await client.ttl(key) <= _window_ttl(60)


@pytest.mark.asyncio
async def test_real_redis_idle_entity_window_keys_expire(real_redis, rg_unique_ns, monkeypatch):
    # A 1 s TTL lets the test watch Redis drop the keys without waiting out a 60 s window.
    monkeypatch.setattr(governor_redis, "_window_ttl", lambda window: 1)
    rg = await _real_governor(rg_unique_ns, {"requests": {"rpm": 5}, "tokens": {"per_min": 100}})

    decision, _handle_id = await rg.reserve(_REQ)
    assert decision.allowed
    client = await rg._client_get()
    keys = await rg._scan_keys(f"{rg_unique_ns}:win:*")
    assert len(keys) == 4  # requests and tokens, each in global and user scope

    await asyncio.sleep(1.5)
    assert await client.exists(*keys) == 0
