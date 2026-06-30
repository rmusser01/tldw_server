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


def test_user_embedding_config_sanitizes_user_db_base_fallback_log(monkeypatch):
    logger_stub = MagicMock()

    def _raise_path_error():
        raise RuntimeError("user db base exploded at /private/user/db")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings.DatabasePaths, "get_user_db_base_dir", _raise_path_error)
    monkeypatch.setitem(media_embeddings.settings, "EMBEDDING_CONFIG", {})
    monkeypatch.setitem(media_embeddings.settings, "USER_DB_BASE_DIR", "/tmp/fallback-user-db")

    config = media_embeddings._user_embedding_config()

    assert config["USER_DB_BASE_DIR"] == "/tmp/fallback-user-db"
    logger_stub.warning.assert_called_once_with(
        "Falling back to USER_DB_BASE_DIR setting after user DB base resolution failed"
    )


def test_embeddings_jobs_backend_sanitizes_ignored_override_log(monkeypatch):
    logger_stub = MagicMock()

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setenv("EMBEDDINGS_JOBS_BACKEND", "legacy:/private/backend")
    monkeypatch.delenv("TLDW_JOBS_BACKEND", raising=False)

    assert media_embeddings._embeddings_jobs_backend() == "jobs"
    logger_stub.warning.assert_called_once_with(
        "Embeddings jobs backend override ignored; core Jobs is the only backend"
    )


@pytest.mark.asyncio
async def test_get_media_content_sanitizes_backend_lookup_error(monkeypatch):
    logger_stub = MagicMock()

    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.get_media_content(media_id=123, db=object())

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error retrieving media content"
    logger_stub.error.assert_called_once_with("Error retrieving media content")


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
    logger_stub = MagicMock()

    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.get_embeddings_status(
            media_id=123,
            db=object(),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error checking embeddings status"
    logger_stub.error.assert_called_once_with("Error checking embeddings status")


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
    logger_stub = MagicMock()

    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
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
    logger_stub.error.assert_called_once_with("Error generating embeddings")


@pytest.mark.asyncio
async def test_delete_embeddings_sanitizes_backend_lookup_error(monkeypatch):
    logger_stub = MagicMock()

    def _raise_backend_error(*_args, **_kwargs):
        raise RuntimeError("media backend exploded")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "get_media_by_id", _raise_backend_error)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.delete_embeddings(
            media_id=123,
            db=object(),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Error deleting embeddings"
    logger_stub.error.assert_called_once_with("Error deleting embeddings")


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
    logger_stub = MagicMock()

    class _FailingAdapter:
        def create_job(self, **_kwargs):
            raise RuntimeError("queue unavailable")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
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
    logger_stub.error.assert_called_once_with("Failed to persist media embedding job")


@pytest.mark.asyncio
async def test_generate_embeddings_fails_when_job_id_missing(monkeypatch):
    logger_stub = MagicMock()

    class _MissingIdAdapter:
        def create_job(self, **_kwargs):
            return {}

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
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
    logger_stub.error.assert_called_once_with("Embeddings job creation returned no job id")


@pytest.mark.asyncio
async def test_generate_embeddings_batch_returns_partial_response_on_partial_enqueue_error(monkeypatch):
    logger_stub = MagicMock()

    class _PartialAdapter:
        def create_job(self, **kwargs):
            media_id = int(kwargs["media_id"])
            if media_id == 456:
                raise RuntimeError("enqueue failed")
            return {"uuid": f"job-{media_id}"}

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
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
    logger_stub.error.assert_called_once_with("Failed to persist batch embedding job")


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
async def test_search_embeddings_sanitizes_embedding_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced

    logger_stub = MagicMock()

    async def _raise_embedding_error(*_args, **_kwargs):
        raise RuntimeError("embedding backend exploded at /private/models")

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        _raise_embedding_error,
    )

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.search_embeddings(
            request=media_embeddings.EmbeddingsSearchRequest(query="needle"),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Embedding service unavailable"
    logger_stub.error.assert_called_once_with("Failed to embed search query")


@pytest.mark.asyncio
async def test_search_embeddings_sanitizes_chroma_query_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced

    logger_stub = MagicMock()

    async def _embedding_result(*_args, **_kwargs):
        return [[0.1, 0.2]]

    class _FailingCollection:
        def query(self, **_kwargs):
            raise RuntimeError("chroma query exploded at /private/chroma")

    class _Client:
        def get_collection(self, **_kwargs):
            return _FailingCollection()

    class _ChromaDBManager:
        def __init__(self, *_args, **_kwargs):
            self.client = _Client()

    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", _ChromaDBManager)
    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        _embedding_result,
    )

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.search_embeddings(
            request=media_embeddings.EmbeddingsSearchRequest(query="needle"),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Search failed"
    logger_stub.error.assert_called_once_with("Chroma query failed")


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
