import os
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import redis
import redis.asyncio as aioredis
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user


class _FakeMediaDB:
    def __init__(self, media_ids=None):
        ids = media_ids or [123]
        self._items = {
            int(media_id): {
                "id": int(media_id),
                "title": f"Doc {media_id}",
                "author": "A",
                "content": {"content": "short text"},
            }
            for media_id in ids
        }

    def get_media_by_id(self, media_id: int, **_kwargs):
        return self._items.get(media_id)


class _FakeRedisClient:
    async def ping(self):
        return True

    async def close(self):
        return None

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_list_media_embedding_jobs_includes_canonical_overfetch_pagination(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

    class _FakeJobsAdapter:
        def list_jobs(self, *, user_id, status=None, limit=50, offset=0):
            assert user_id == "1"
            assert status == "queued"
            assert limit in {2, 3}
            assert offset == 0
            rows = [
                {"id": "job-1", "status": "queued"},
                {"id": "job-2", "status": "queued"},
                {"id": "job-3", "status": "queued"},
            ]
            return rows[:limit]

    monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _FakeJobsAdapter)

    response = await media_embeddings_endpoint.list_media_embedding_jobs(
        current_user=SimpleNamespace(id=1),
        status="queued",
        limit=2,
        offset=0,
    )

    assert [row["id"] for row in response["data"]] == ["job-1", "job-2"]
    assert response["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 0,
        "total": None,
        "has_more": True,
        "next_offset": 2,
        "count": 2,
    }
    assert response["has_more"] is True
    assert response["next_offset"] == 2


@pytest.fixture(autouse=True)
def _stub_redis_clients(monkeypatch):
    def _make_client(*_args, **_kwargs):
        return _FakeRedisClient()

    monkeypatch.setattr(aioredis, "from_url", _make_client)
    monkeypatch.setattr(redis, "from_url", _make_client, raising=False)


from contextlib import contextmanager


@contextmanager
def _client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "test-csrf")
        yield c


def test_media_embedding_job_lifecycle():
    os.environ["TESTING"] = "true"
    try:
        original_allowed_providers = settings.get("ALLOWED_EMBEDDING_PROVIDERS")
        original_allowed_models = settings.get("ALLOWED_EMBEDDING_MODELS")
        original_model_limits = settings.get("EMBEDDING_MODEL_MAX_TOKENS")
        # Keep token limits permissive
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai", "huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2"]
        settings["EMBEDDING_MODEL_MAX_TOKENS"] = {"openai:text-embedding-3-small": 8192}

        # Override media DB dependency
        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB()

        with _client() as client:
            # Start job
            api_key = get_settings().SINGLE_USER_API_KEY
            r = client.post("/api/v1/media/123/embeddings", json={}, headers={"X-API-KEY": api_key})
            assert r.status_code == 200
            body = r.json()
            assert body.get("status") == "accepted"
            job_id = body.get("job_id")
            assert job_id

            # Get job status (may be processing/completed/failed depending on environment)
            r2 = client.get(f"/api/v1/media/embeddings/jobs/{job_id}", headers={"X-API-KEY": api_key})
            assert r2.status_code == 200
            data = r2.json()
            assert data.get("id") == job_id
            assert data.get("media_id") == 123
            assert data.get("status") in ("queued", "processing", "completed", "failed")

            # List jobs
            r3 = client.get("/api/v1/media/embeddings/jobs", headers={"X-API-KEY": api_key})
            assert r3.status_code == 200
            j = r3.json()
            assert isinstance(j.get("data"), list)
            assert any(row.get("id") == job_id for row in j["data"])
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)
        if original_allowed_providers is None:
            settings.pop("ALLOWED_EMBEDDING_PROVIDERS", None)
        else:
            settings["ALLOWED_EMBEDDING_PROVIDERS"] = original_allowed_providers
        if original_allowed_models is None:
            settings.pop("ALLOWED_EMBEDDING_MODELS", None)
        else:
            settings["ALLOWED_EMBEDDING_MODELS"] = original_allowed_models
        if original_model_limits is None:
            settings.pop("EMBEDDING_MODEL_MAX_TOKENS", None)
        else:
            settings["EMBEDDING_MODEL_MAX_TOKENS"] = original_model_limits


def test_media_embedding_job_returns_500_when_job_creation_fails(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _FailingAdapter:
            def create_job(self, **_kwargs):
                raise RuntimeError("queue unavailable")

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB()
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _FailingAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post("/api/v1/media/123/embeddings", json={}, headers={"X-API-KEY": api_key})
            assert resp.status_code == 500
            assert resp.json().get("detail") == "Failed to queue embedding job"
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)


def test_media_embedding_job_returns_500_when_job_id_missing(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _EmptyAdapter:
            def create_job(self, **_kwargs):
                return {}

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB()
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _EmptyAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post("/api/v1/media/123/embeddings", json={}, headers={"X-API-KEY": api_key})
            assert resp.status_code == 500
            assert resp.json().get("detail") == "Failed to queue embedding job"
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)


def test_media_embedding_batch_returns_partial_on_partial_enqueue_failure(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _PartialAdapter:
            def create_job(self, **kwargs):
                media_id = int(kwargs["media_id"])
                if media_id == 456:
                    raise RuntimeError("enqueue failed")
                return {"uuid": f"job-{media_id}"}

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB(media_ids=[123, 456])
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _PartialAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post(
                "/api/v1/media/embeddings/batch",
                json={"media_ids": [123, 456]},
                headers={"X-API-KEY": api_key},
            )
            assert resp.status_code == 202
            body = resp.json()
            assert body.get("status") == "partial"
            assert body.get("job_ids") == ["job-123"]
            assert body.get("submitted") == 1
            assert body.get("failed_media_ids") == [456]
            assert body.get("failure_reasons") == ["media_id=456: RuntimeError"]
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)


def test_media_embedding_batch_returns_accepted_with_empty_failure_lists(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _SuccessfulAdapter:
            def create_job(self, **kwargs):
                media_id = int(kwargs["media_id"])
                return {"uuid": f"job-{media_id}"}

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB(media_ids=[123, 456])
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _SuccessfulAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post(
                "/api/v1/media/embeddings/batch",
                json={"media_ids": [123, 456]},
                headers={"X-API-KEY": api_key},
            )
            assert resp.status_code == 202
            body = resp.json()
            assert body.get("status") == "accepted"
            assert body.get("job_ids") == ["job-123", "job-456"]
            assert body.get("submitted") == 2
            assert body.get("failed_media_ids") == []
            assert body.get("failure_reasons") == []
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)


def test_media_embedding_batch_returns_500_when_nothing_queued(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _FailingAdapter:
            def create_job(self, **_kwargs):
                raise RuntimeError("enqueue failed")

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB(media_ids=[456])
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _FailingAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post(
                "/api/v1/media/embeddings/batch",
                json={"media_ids": [456]},
                headers={"X-API-KEY": api_key},
            )
            assert resp.status_code == 500
            detail = (resp.json() or {}).get("detail") or {}
            assert detail.get("error") == "batch_enqueue_failed"
            assert detail.get("submitted") == 0
            assert detail.get("failed_media_ids") == [456]
            assert detail.get("failure_reasons") == ["media_id=456: RuntimeError"]
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)


def test_media_embedding_batch_returns_500_when_batch_job_id_missing(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _MissingIdAdapter:
            def create_job(self, **_kwargs):
                return {}

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB(media_ids=[456])
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _MissingIdAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            resp = client.post(
                "/api/v1/media/embeddings/batch",
                json={"media_ids": [456]},
                headers={"X-API-KEY": api_key},
            )
            assert resp.status_code == 500
            detail = (resp.json() or {}).get("detail") or {}
            assert detail.get("error") == "batch_enqueue_failed"
            assert detail.get("submitted") == 0
            assert detail.get("failed_media_ids") == [456]
            assert detail.get("failure_reasons") == ["media_id=456: ValueError"]
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)
