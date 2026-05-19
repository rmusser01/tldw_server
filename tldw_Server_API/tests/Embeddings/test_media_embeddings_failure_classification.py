import pytest
from unittest.mock import MagicMock

from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced, media_embeddings


@pytest.mark.asyncio
async def test_storage_failure_after_successful_primary_generation_returns_storage_error_and_skips_fallback(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []
    logger_stub = MagicMock()

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        calls.append((provider, model_id))
        return [[0.1, 0.2, 0.3] for _ in texts]

    class FakeChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            self.user_id = user_id
            self.user_embedding_config = user_embedding_config

        def store_in_chroma(self, *args, **kwargs):
            raise RuntimeError("chroma write failed")

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )
    monkeypatch.setattr(media_embeddings, "chunk_media_content", lambda *_args, **_kwargs: [{"text": "hello", "index": 0, "start": 0, "end": 5}])
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", FakeChromaDBManager)
    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": str(tmp_path / "user-db")})

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=9,
        media_content={
            "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
            "content": {"content": "hello"},
        },
        embedding_model="primary-model",
        embedding_provider="primary-provider",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-1",
    )

    assert result["status"] == "error"
    assert "storage" in result["message"].lower() or "storage" in result["error"].lower()
    assert "chroma" in result["message"].lower() or "chroma" in result["error"].lower()
    assert calls == [("primary-provider", "primary-model")]
    logger_stub.error.assert_called_once_with("Error storing primary embeddings")


@pytest.mark.asyncio
async def test_generation_failure_can_fall_back_and_succeed(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []
    stores: list[str] = []
    logger_stub = MagicMock()

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        calls.append((provider, model_id))
        if provider == "primary-provider":
            raise RuntimeError("primary provider failed")
        return [[0.4, 0.5, 0.6] for _ in texts]

    class FakeChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            self.user_id = user_id
            self.user_embedding_config = user_embedding_config

        def store_in_chroma(self, *, collection_name, texts, embeddings, ids, metadatas, embedding_model_id_for_dim_check=None):
            stores.append(embedding_model_id_for_dim_check)

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )
    monkeypatch.setattr(media_embeddings, "chunk_media_content", lambda *_args, **_kwargs: [{"text": "hello", "index": 0, "start": 0, "end": 5}])
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", FakeChromaDBManager)
    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": str(tmp_path / "user-db")})

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=10,
        media_content={
            "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
            "content": {"content": "hello"},
        },
        embedding_model="primary-model",
        embedding_provider="primary-provider",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-2",
    )

    assert result["status"] == "success"
    assert result["embedding_count"] == 1
    assert calls == [
        ("primary-provider", "primary-model"),
        ("huggingface", media_embeddings.FALLBACK_EMBEDDING_MODEL),
    ]
    assert stores == [media_embeddings.FALLBACK_EMBEDDING_MODEL]
    logger_stub.warning.assert_called_once_with("Primary embedding generation failed; trying fallback model")


@pytest.mark.asyncio
async def test_generation_failure_sanitizes_fallback_failure(monkeypatch, tmp_path):
    logger_stub = MagicMock()

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        if provider == "primary-provider":
            raise RuntimeError("primary provider exploded at /private/provider")
        raise RuntimeError("fallback provider exploded at /private/fallback")

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )
    monkeypatch.setattr(media_embeddings, "chunk_media_content", lambda *_args, **_kwargs: [{"text": "hello", "index": 0, "start": 0, "end": 5}])
    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": str(tmp_path / "user-db")})

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=11,
        media_content={
            "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
            "content": {"content": "hello"},
        },
        embedding_model="primary-model",
        embedding_provider="primary-provider",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-3",
    )

    assert result["status"] == "error"
    assert result["message"] == "Failed to generate embeddings"
    assert result["error"] == "Embedding generation failed"
    assert "provider exploded" not in str(result)
    assert "/private/" not in str(result)
    logger_stub.warning.assert_called_once_with("Primary embedding generation failed; trying fallback model")
    logger_stub.error.assert_called_once_with("Fallback embedding generation failed")


@pytest.mark.asyncio
async def test_generation_failure_sanitizes_fallback_storage_log(monkeypatch, tmp_path):
    logger_stub = MagicMock()

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        if provider == "primary-provider":
            raise RuntimeError("primary provider failed")
        return [[0.7, 0.8, 0.9] for _ in texts]

    class FakeChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            self.user_id = user_id
            self.user_embedding_config = user_embedding_config

        def store_in_chroma(self, *args, **kwargs):
            raise RuntimeError("fallback chroma write failed at /private/chroma")

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )
    monkeypatch.setattr(media_embeddings, "chunk_media_content", lambda *_args, **_kwargs: [{"text": "hello", "index": 0, "start": 0, "end": 5}])
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", FakeChromaDBManager)
    monkeypatch.setattr(media_embeddings, "logger", logger_stub)
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": str(tmp_path / "user-db")})

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=13,
        media_content={
            "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
            "content": {"content": "hello"},
        },
        embedding_model="primary-model",
        embedding_provider="primary-provider",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-5",
    )

    assert result["status"] == "error"
    logger_stub.error.assert_called_once_with("Error storing fallback embeddings")


@pytest.mark.asyncio
async def test_generation_failure_sanitizes_outer_error_log(monkeypatch):
    logger_stub = MagicMock()

    def _raise_chunking_error(*_args, **_kwargs):
        raise RuntimeError("chunking exploded at /private/chunks")

    monkeypatch.setattr(media_embeddings, "chunk_media_content", _raise_chunking_error)
    monkeypatch.setattr(media_embeddings, "logger", logger_stub)

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=12,
        media_content={
            "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
            "content": {"content": "hello"},
        },
        embedding_model="primary-model",
        embedding_provider="primary-provider",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-4",
    )

    assert result["status"] == "error"
    assert result["message"] == "Failed to generate embeddings"
    assert result["error"] == "Embedding generation failed"
    logger_stub.error.assert_called_once_with("Error generating embeddings")
