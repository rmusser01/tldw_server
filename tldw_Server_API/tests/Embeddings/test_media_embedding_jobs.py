import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
import redis
import redis.asyncio as aioredis
from tldw_Server_API.app.core.config import settings


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


@pytest.mark.asyncio
async def test_media_embedding_job_lifecycle(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _LifecycleJobsAdapter:
            _jobs: list[dict] = []

            def create_job(self, **kwargs):
                row = {
                    "id": "job-123",
                    "uuid": "job-123",
                    "media_id": int(kwargs["media_id"]),
                    "user_id": str(kwargs["user_id"]),
                    "status": "queued",
                    "embedding_model": kwargs["embedding_model"],
                    "embedding_count": None,
                    "chunks_processed": None,
                }
                type(self)._jobs = [row]
                return row

            def get_job(self, job_id, user_id):
                for row in type(self)._jobs:
                    if str(row["id"]) == str(job_id) and str(row["user_id"]) == str(user_id):
                        return dict(row)
                return None

            def list_jobs(self, *, user_id, status=None, limit=50, offset=0):
                rows = [
                    dict(row)
                    for row in type(self)._jobs
                    if str(row["user_id"]) == str(user_id) and (status is None or row["status"] == status)
                ]
                return rows[int(offset) : int(offset) + int(limit)]

        original_allowed_providers = settings.get("ALLOWED_EMBEDDING_PROVIDERS")
        original_allowed_models = settings.get("ALLOWED_EMBEDDING_MODELS")
        original_model_limits = settings.get("EMBEDDING_MODEL_MAX_TOKENS")
        settings["ALLOWED_EMBEDDING_PROVIDERS"] = ["openai", "huggingface"]
        settings["ALLOWED_EMBEDDING_MODELS"] = ["text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2"]
        settings["EMBEDDING_MODEL_MAX_TOKENS"] = {"openai:text-embedding-3-small": 8192}
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _LifecycleJobsAdapter)

        current_user = SimpleNamespace(id=1)
        response = await media_embeddings_endpoint.generate_embeddings(
            media_id=123,
            request=media_embeddings_endpoint.GenerateEmbeddingsRequest(),
            db=_FakeMediaDB(),
            current_user=current_user,
        )
        assert response.status == "accepted"
        job_id = response.job_id
        assert job_id

        data = await media_embeddings_endpoint.get_media_embedding_job(
            job_id,
            current_user=current_user,
        )
        assert data.get("id") == job_id
        assert data.get("media_id") == 123
        assert data.get("status") in ("queued", "processing", "completed", "failed")

        listed = await media_embeddings_endpoint.list_media_embedding_jobs(
            current_user=current_user,
            limit=50,
            offset=0,
        )
        assert isinstance(listed.get("data"), list)
        assert any(row.get("id") == job_id for row in listed["data"])
    finally:
        os.environ.pop("TESTING", None)
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


@pytest.mark.asyncio
async def test_media_embedding_job_returns_500_when_job_creation_fails(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _FailingAdapter:
            def create_job(self, **_kwargs):
                raise RuntimeError("queue unavailable")

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _FailingAdapter)

        with pytest.raises(HTTPException) as exc_info:
            await media_embeddings_endpoint.generate_embeddings(
                media_id=123,
                request=media_embeddings_endpoint.GenerateEmbeddingsRequest(),
                db=_FakeMediaDB(),
                current_user=SimpleNamespace(id=1),
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to queue embedding job"
    finally:
        os.environ.pop("TESTING", None)


@pytest.mark.asyncio
async def test_media_embedding_job_returns_500_when_job_id_missing(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _EmptyAdapter:
            def create_job(self, **_kwargs):
                return {}

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _EmptyAdapter)

        with pytest.raises(HTTPException) as exc_info:
            await media_embeddings_endpoint.generate_embeddings(
                media_id=123,
                request=media_embeddings_endpoint.GenerateEmbeddingsRequest(),
                db=_FakeMediaDB(),
                current_user=SimpleNamespace(id=1),
            )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to queue embedding job"
    finally:
        os.environ.pop("TESTING", None)


@pytest.mark.asyncio
async def test_media_embedding_batch_returns_partial_on_partial_enqueue_failure(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _PartialAdapter:
            def create_job(self, **kwargs):
                media_id = int(kwargs["media_id"])
                if media_id == 456:
                    raise RuntimeError("enqueue failed")
                return {"uuid": f"job-{media_id}"}

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _PartialAdapter)

        body = await media_embeddings_endpoint.generate_embeddings_batch(
            request=media_embeddings_endpoint.BatchMediaEmbeddingsRequest(media_ids=[123, 456]),
            db=_FakeMediaDB(media_ids=[123, 456]),
            current_user=SimpleNamespace(id=1),
        )
        assert body.status == "partial"
        assert body.job_ids == ["job-123"]
        assert body.submitted == 1
        assert body.failed_media_ids == [456]
        assert body.failure_reasons == ["media_id=456: RuntimeError"]
    finally:
        os.environ.pop("TESTING", None)


@pytest.mark.asyncio
async def test_media_embedding_batch_returns_accepted_with_empty_failure_lists(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _SuccessfulAdapter:
            def create_job(self, **kwargs):
                media_id = int(kwargs["media_id"])
                return {"uuid": f"job-{media_id}"}

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _SuccessfulAdapter)

        body = await media_embeddings_endpoint.generate_embeddings_batch(
            request=media_embeddings_endpoint.BatchMediaEmbeddingsRequest(media_ids=[123, 456]),
            db=_FakeMediaDB(media_ids=[123, 456]),
            current_user=SimpleNamespace(id=1),
        )
        assert body.status == "accepted"
        assert body.job_ids == ["job-123", "job-456"]
        assert body.submitted == 2
        assert body.failed_media_ids == []
        assert body.failure_reasons == []
    finally:
        os.environ.pop("TESTING", None)


@pytest.mark.asyncio
async def test_media_embedding_batch_returns_500_when_nothing_queued(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _FailingAdapter:
            def create_job(self, **_kwargs):
                raise RuntimeError("enqueue failed")

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _FailingAdapter)

        with pytest.raises(HTTPException) as exc_info:
            await media_embeddings_endpoint.generate_embeddings_batch(
                request=media_embeddings_endpoint.BatchMediaEmbeddingsRequest(media_ids=[456]),
                db=_FakeMediaDB(media_ids=[456]),
                current_user=SimpleNamespace(id=1),
            )
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail or {}
        assert detail.get("error") == "batch_enqueue_failed"
        assert detail.get("submitted") == 0
        assert detail.get("failed_media_ids") == [456]
        assert detail.get("failure_reasons") == ["media_id=456: RuntimeError"]
    finally:
        os.environ.pop("TESTING", None)


@pytest.mark.asyncio
async def test_media_embedding_batch_returns_500_when_batch_job_id_missing(monkeypatch):
    os.environ["TESTING"] = "true"
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _MissingIdAdapter:
            def create_job(self, **_kwargs):
                return {}

        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _MissingIdAdapter)

        with pytest.raises(HTTPException) as exc_info:
            await media_embeddings_endpoint.generate_embeddings_batch(
                request=media_embeddings_endpoint.BatchMediaEmbeddingsRequest(media_ids=[456]),
                db=_FakeMediaDB(media_ids=[456]),
                current_user=SimpleNamespace(id=1),
            )
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail or {}
        assert detail.get("error") == "batch_enqueue_failed"
        assert detail.get("submitted") == 0
        assert detail.get("failed_media_ids") == [456]
        assert detail.get("failure_reasons") == ["media_id=456: ValueError"]
    finally:
        os.environ.pop("TESTING", None)
