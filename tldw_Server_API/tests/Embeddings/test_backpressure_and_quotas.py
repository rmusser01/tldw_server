import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


class FakeRedisBP:
    def __init__(self, depth=0, age_first_ms=0):
        self.depth = depth
        self.age_first_ms = age_first_ms
        self._kv = {}
        self._incr = {}

    async def xlen(self, name):  # noqa: ARG002
        return self.depth

    async def xrange(self, name, min, max, count=None):  # noqa: ARG002
        if self.age_first_ms <= 0:
            return []
        return [(f"{self.age_first_ms}-0", {})]

    async def close(self):
        return True

    async def get(self, key):
        return self._kv.get(key)

    async def incr(self, key):
        self._incr[key] = self._incr.get(key, 0) + 1
        return self._incr[key]

    async def expire(self, key, ttl):  # noqa: ARG002
        return True


def _override_user(admin=False, uid="u1"):


    async def _f():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=uid, username="admin" if admin else uid, email=f"{uid}@x", is_active=True, is_admin=admin)
    return _f


@pytest.mark.asyncio
@pytest.mark.unit
async def test_backpressure_ignores_default_local_redis_when_redis_disabled(monkeypatch):
    """Disabled Redis config must not let a stray localhost Redis block embeddings."""

    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as ep
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    import redis.asyncio as aioredis

    stale = FakeRedisBP(depth=1, age_first_ms=1000)

    async def fake_from_url(url, decode_responses=True):  # noqa: ARG001
        return stale

    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_REDIS_URL", raising=False)
    monkeypatch.setenv("EMB_BACKPRESSURE_MAX_AGE_SECONDS", "0.1")
    monkeypatch.setitem(ep.settings, "REDIS_ENABLED", False)
    monkeypatch.setitem(ep.settings, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(aioredis, "from_url", fake_from_url)

    user = User(id="u1", username="u1", email="u1@example.test", is_active=True, is_admin=False)
    request = SimpleNamespace(state=SimpleNamespace())

    assert await ep._check_backpressure_and_quotas(request, user) is None


@pytest.mark.unit
def test_backpressure_by_age_returns_429(monkeypatch):
    client = TestClient(app)
    app.dependency_overrides[get_request_user] = _override_user(admin=True)
    # Force age above threshold
    fake = FakeRedisBP(depth=0, age_first_ms=1000)  # 1 second epoch

    import redis.asyncio as aioredis

    async def fake_from_url(url, decode_responses=True):  # noqa: ARG001
        return fake

    monkeypatch.setenv("EMB_BACKPRESSURE_MAX_AGE_SECONDS", "0.1")
    monkeypatch.setenv("REDIS_ENABLED", "true")
    monkeypatch.setattr(aioredis, "from_url", fake_from_url)
    r = client.post("/api/v1/embeddings", json={"input": "hello", "model": "text-embedding-3-small"})
    assert r.status_code == 429
    assert r.headers.get("Retry-After") is not None
    app.dependency_overrides.pop(get_request_user, None)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tenant_quota_429(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as ep
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    fake = FakeRedisBP(depth=0, age_first_ms=0)
    import redis.asyncio as aioredis

    async def fake_from_url(url, decode_responses=True):  # noqa: ARG001
        return fake

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS", "1")

    user = User(id="tenant1", username="tenant1", email="tenant1@example.test", is_active=True, is_admin=False)
    request = SimpleNamespace(state=SimpleNamespace())

    assert await ep._check_backpressure_and_quotas(request, user) is None
    second = await ep._check_backpressure_and_quotas(request, user)

    assert second is not None
    assert second.status_code == 429
    assert second.headers.get("Retry-After") == "1"


@pytest.mark.unit
def test_embeddings_batch_route_has_rbac_rate_limit_parity():
    single_route = None
    batch_route = None
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path == "/api/v1/embeddings" and "POST" in route.methods:
            single_route = route
        if route.path == "/api/v1/embeddings/batch" and "POST" in route.methods:
            batch_route = route

    assert single_route is not None
    assert batch_route is not None

    def _resources(route: APIRoute) -> list[str]:
        resources: list[str] = []
        for dep in route.dependant.dependencies:
            resource = getattr(dep.call, "_tldw_rate_limit_resource", None)
            if resource:
                resources.append(str(resource))
        return resources

    single_resources = _resources(single_route)
    batch_resources = _resources(batch_route)

    assert "embeddings.create" in single_resources
    assert "embeddings.create" in batch_resources
