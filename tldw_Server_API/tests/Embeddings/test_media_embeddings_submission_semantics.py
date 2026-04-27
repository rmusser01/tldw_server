import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from tldw_Server_API.app.api.v1.endpoints import media_embeddings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


class _FakeMediaDB:
    def __init__(self, media_ids: list[int]):
        self._ids = set(int(media_id) for media_id in media_ids)

    def get_media_by_id(self, media_id: int, **_kwargs):
        if int(media_id) in self._ids:
            return {"id": int(media_id), "title": "Doc", "author": "A"}
        return None


def _user() -> User:
    return User(id="user-1", username="user-1", email="user-1@example.com", is_active=True, is_admin=True)


@pytest.mark.asyncio
async def test_get_media_content_sanitizes_backend_lookup_error(monkeypatch):
    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.get_media_content(media_id=123, db=object())

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error retrieving media content"


@pytest.mark.asyncio
async def test_get_media_content_fallback_document_content_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api

    logger_stub = MagicMock()

    def _empty_media_item(*_args, **_kwargs):
        return {"id": 123, "title": "Doc", "content": ""}

    def _raise_fallback_error(*_args, **_kwargs):
        raise RuntimeError("fallback document backend exploded at /private/media.db")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "get_media_by_id", _empty_media_item)
    monkeypatch.setattr(media_db_api, "get_document_version", _raise_fallback_error)

    result = await media_embeddings.get_media_content(media_id=123, db=object())

    assert result["media_item"]["content"] == ""
    logger_stub.warning.assert_called_once_with("Failed to load fallback document content")


@pytest.mark.asyncio
async def test_get_embeddings_status_sanitizes_backend_lookup_error(monkeypatch):
    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.get_embeddings_status(
            media_id=123,
            db=object(),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error checking embeddings status"


@pytest.mark.asyncio
async def test_get_embeddings_status_chroma_failure_log_is_sanitized(monkeypatch):
    logger_stub = MagicMock()

    class _FailingChromaDBManager:
        def __init__(self, *_args, **_kwargs):
            return None

        def get_or_create_collection(self, *_args, **_kwargs):
            raise RuntimeError("chroma status backend exploded at /private/chroma")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(
        media_embeddings,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 123, "title": "Doc"},
    )
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", _FailingChromaDBManager)

    response = await media_embeddings.get_embeddings_status(
        media_id=123,
        db=object(),
        current_user=_user(),
    )

    assert response.has_embeddings is False
    logger_stub.warning.assert_called_once_with("ChromaDB status check failed")


@pytest.mark.asyncio
async def test_generate_embeddings_sanitizes_backend_lookup_error(monkeypatch):
    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings(
            media_id=123,
            request=media_embeddings.GenerateEmbeddingsRequest(),
            db=object(),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error generating embeddings"


@pytest.mark.asyncio
async def test_delete_embeddings_sanitizes_backend_lookup_error(monkeypatch):
    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.delete_embeddings(
            media_id=123,
            db=object(),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error deleting embeddings"


@pytest.mark.asyncio
async def test_delete_embeddings_sanitizes_where_delete_fallback_log(monkeypatch):
    logger_stub = MagicMock()

    class _FallbackCollection:
        def __init__(self):
            self._where_delete_attempted = False

        def delete(self, **kwargs):
            if "where" in kwargs:
                self._where_delete_attempted = True
                raise RuntimeError("where delete backend exploded at /private/chroma")
            return None

        def get(self, **_kwargs):
            if self._where_delete_attempted:
                self._where_delete_attempted = False
                return {"ids": ["embedding-123"]}
            return {"ids": []}

    collection = _FallbackCollection()

    class _ChromaDBManager:
        def __init__(self, *_args, **_kwargs):
            return None

        def get_or_create_collection(self, *_args, **_kwargs):
            return collection

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(media_embeddings, "get_media_by_id", lambda *_args, **_kwargs: {"id": 123})
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", _ChromaDBManager)
    monkeypatch.setattr(media_embeddings, "invalidate_rag_caches", lambda *_args, **_kwargs: None)

    response = await media_embeddings.delete_embeddings(
        media_id=123,
        db=object(),
        current_user=_user(),
    )

    assert response["status"] == "success"
    logger_stub.warning.assert_called_once_with(
        "Where-delete failed for media embeddings, falling back to id delete"
    )


@pytest.mark.asyncio
async def test_delete_embeddings_sanitizes_verify_failure_log(monkeypatch):
    logger_stub = MagicMock()

    class _VerifyFailCollection:
        def delete(self, **_kwargs):
            return None

        def get(self, **_kwargs):
            raise RuntimeError("verify backend exploded at /private/chroma")

    class _ChromaDBManager:
        def __init__(self, *_args, **_kwargs):
            return None

        def get_or_create_collection(self, *_args, **_kwargs):
            return _VerifyFailCollection()

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(media_embeddings, "get_media_by_id", lambda *_args, **_kwargs: {"id": 123})
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", _ChromaDBManager)
    monkeypatch.setattr(media_embeddings, "invalidate_rag_caches", lambda *_args, **_kwargs: None)

    response = await media_embeddings.delete_embeddings(
        media_id=123,
        db=object(),
        current_user=_user(),
    )

    assert response["status"] == "success"
    logger_stub.warning.assert_called_once_with("Failed to verify embeddings delete")


@pytest.mark.asyncio
async def test_generate_embeddings_fails_when_job_create_raises(monkeypatch):
    class _FailingAdapter:
        def create_job(self, **_kwargs):
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _FailingAdapter)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings(
            media_id=123,
            request=media_embeddings.GenerateEmbeddingsRequest(),
            db=_FakeMediaDB([123]),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to queue embedding job"


@pytest.mark.asyncio
async def test_generate_embeddings_fails_when_job_id_missing(monkeypatch):
    class _MissingIdAdapter:
        def create_job(self, **_kwargs):
            return {}

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _MissingIdAdapter)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings(
            media_id=123,
            request=media_embeddings.GenerateEmbeddingsRequest(),
            db=_FakeMediaDB([123]),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to queue embedding job"


@pytest.mark.asyncio
async def test_generate_embeddings_batch_returns_partial_response_on_partial_enqueue_error(monkeypatch):
    class _PartialAdapter:
        def create_job(self, **kwargs):
            media_id = int(kwargs["media_id"])
            if media_id == 456:
                raise RuntimeError("enqueue failed")
            return {"uuid": f"job-{media_id}"}

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _PartialAdapter)

    response = await media_embeddings.generate_embeddings_batch(
        request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[123, 456]),
        db=_FakeMediaDB([123, 456]),
        current_user=_user(),
    )

    assert response.status == "partial"
    assert response.job_ids == ["job-123"]
    assert response.submitted == 1
    assert response.failed_media_ids == [456]
    assert response.failure_reasons == ["media_id=456: RuntimeError"]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_returns_partial_when_some_media_ids_missing(monkeypatch):
    class _OkAdapter:
        def create_job(self, **kwargs):
            return {"uuid": f"job-{kwargs['media_id']}"}

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _OkAdapter)

    # _FakeMediaDB only knows about media_id 123; 999 is missing
    response = await media_embeddings.generate_embeddings_batch(
        request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[123, 999]),
        db=_FakeMediaDB([123]),
        current_user=_user(),
    )

    assert response.status == "partial"
    assert response.job_ids == ["job-123"]
    assert response.submitted == 1
    assert response.failed_media_ids == [999]
    assert response.failure_reasons == ["media_id=999: not found"]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_raises_when_all_media_ids_missing(monkeypatch):
    class _OkAdapter:
        def create_job(self, **kwargs):
            return {"uuid": f"job-{kwargs['media_id']}"}

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _OkAdapter)

    # _FakeMediaDB is empty — both IDs are missing
    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings_batch(
            request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[888, 999]),
            db=_FakeMediaDB([]),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert isinstance(excinfo.value.detail, dict)
    assert excinfo.value.detail.get("error") == "batch_enqueue_failed"
    assert excinfo.value.detail.get("failed_media_ids") == [888, 999]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_raises_when_all_enqueues_fail_before_any_success(monkeypatch):
    class _AlwaysFailingAdapter:
        def create_job(self, **kwargs):
            raise RuntimeError("enqueue failed")

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _AlwaysFailingAdapter)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings_batch(
            request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[456]),
            db=_FakeMediaDB([456]),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert isinstance(excinfo.value.detail, dict)
    assert excinfo.value.detail.get("error") == "batch_enqueue_failed"
    assert excinfo.value.detail.get("submitted") == 0
    assert excinfo.value.detail.get("failed_media_ids") == [456]
