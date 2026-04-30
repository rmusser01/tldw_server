import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


class FakeRedisKV:
    def __init__(self):
        self.kv = {}

    async def set(self, k, v):
        self.kv[k] = v
        return True

    async def expire(self, k, ttl):  # noqa: ARG002
        return True

    async def close(self):
        return True


class FailingRedisKV:
    async def set(self, k, v):
        raise RuntimeError("redis password leaked")

    async def close(self):
        return True


@pytest.mark.unit
def test_bump_priority_sets_override(monkeypatch, admin_user):
    client = TestClient(app)
    fake = FakeRedisKV()
    import redis.asyncio as aioredis

    async def fake_from_url(url, decode_responses=True):  # noqa: ARG001
        return fake

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)
    r = client.post("/api/v1/embeddings/job/priority/bump", json={"job_id": "j1", "priority": "high", "ttl_seconds": 60})
    assert r.status_code == 200
    assert fake.kv.get("embeddings:priority:override:j1") == "high"
    # Cleanup handled by admin_user fixture


@pytest.mark.unit
def test_bump_priority_sanitizes_backend_failure(monkeypatch, admin_user):
    client = TestClient(app)
    import redis.asyncio as aioredis

    async def fake_from_url(url, decode_responses=True):  # noqa: ARG001
        return FailingRedisKV()

    monkeypatch.setattr(aioredis, "from_url", fake_from_url)
    response = client.post(
        "/api/v1/embeddings/job/priority/bump",
        json={"job_id": "j1", "priority": "high", "ttl_seconds": 60},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to set priority override"
