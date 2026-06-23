import base64

import pytest

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Embeddings import redis_pipeline
from tldw_Server_API.app.core.Embeddings.connection_pool import ConnectionPool
from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter
from tldw_Server_API.app.core.Embeddings.services import jobs_worker
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.exceptions import NetworkError


def _setup_jobs_db(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("EMBEDDINGS_JOBS_QUEUE", "default")
    monkeypatch.setenv("EMBEDDINGS_ROOT_JOBS_QUEUE", "low")
    monkeypatch.setenv("EMBEDDINGS_REDIS_ALLOW_STUB", "1")
    return db_path


@pytest.mark.unit
def test_artifact_dir_rejects_path_traversal_identifier(monkeypatch, tmp_path):
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_vector_store_dir",
        lambda user_id: tmp_path / f"user_{user_id}",
    )

    with pytest.raises(jobs_worker.EmbeddingsJobError, match="artifact identifier"):
        jobs_worker._artifact_dir("user-1", "../escape", 42, "job-1")


@pytest.mark.asyncio
async def test_chunking_job_rejects_payload_artifact_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_vector_store_dir",
        lambda user_id: tmp_path / f"user_{user_id}",
    )
    monkeypatch.setattr(
        jobs_worker,
        "_load_media_content",
        lambda media_id, user_id: {
            "media_item": {"id": media_id, "title": "Doc"},
            "content": {"content": "hello embeddings"},
        },
    )

    outside_path = tmp_path / "outside" / "chunks.json"

    with pytest.raises(jobs_worker.EmbeddingsJobError, match="artifact path"):
        await jobs_worker._handle_chunking_job(
            {"id": 1, "uuid": "job-1"},
            {"chunks_path": str(outside_path)},
            media_id=42,
            user_id="user-1",
            chunk_size=100,
            chunk_overlap=0,
            root_uuid="root-1",
        )

    assert not outside_path.exists()


@pytest.mark.unit
def test_embeddings_jobs_adapter_fails_root_when_idempotency_write_fails(monkeypatch, tmp_path):
    db_path = _setup_jobs_db(monkeypatch, tmp_path)

    class _FailingRedis:
        def set(self, *_args, **_kwargs):
            raise RuntimeError("redis unavailable")

        def close(self):
            return None

    monkeypatch.setattr(redis_pipeline, "create_sync_redis_client", lambda **_kwargs: _FailingRedis())
    adapter = EmbeddingsJobsAdapter()

    with pytest.raises(RuntimeError, match="idempotency"):
        adapter.create_job(
            user_id="user1",
            media_id=111,
            embedding_model="model-a",
            embedding_provider="provider-a",
            chunk_size=1000,
            chunk_overlap=200,
            request_source="test",
            force_regenerate=False,
            stage="chunking",
            embedding_priority=50,
        )

    jm = JobManager(db_path)
    jobs = jm.list_jobs(domain="embeddings", owner_user_id="user1", limit=20)
    assert len(jobs) == 1
    assert jobs[0]["job_type"] == "embeddings_pipeline"
    assert jobs[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_connection_pool_http_error_log_omits_response_body(monkeypatch):
    from tldw_Server_API.app.core.Embeddings import connection_pool as pool_mod

    class _DummyResponse:
        status_code = 503
        text = "secret response body sk-test"
        headers = {"content-type": "application/json"}

        def json(self):
            return {"ok": False}

        async def aclose(self):
            return None

    class _Logger:
        def __init__(self):
            self.errors: list[str] = []

        def info(self, *_args, **_kwargs):
            return None

        def error(self, message):
            self.errors.append(str(message))

    logger = _Logger()

    async def _fake_afetch(**_kwargs):
        return _DummyResponse()

    monkeypatch.setattr(pool_mod, "logger", logger)
    monkeypatch.setattr(pool_mod, "afetch", _fake_afetch)

    pool = ConnectionPool(provider="test-provider", retry_attempts=1)
    with pytest.raises(NetworkError):
        await pool.request("POST", "https://example.invalid/embeddings")

    assert logger.errors
    assert all("secret response body" not in message for message in logger.errors)
    assert all("sk-test" not in message for message in logger.errors)


@pytest.mark.unit
def test_dlq_crypto_configured_encryption_rejects_plaintext_fallback(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_DLQ_ENCRYPTION_KEY", "test-passphrase")
    from tldw_Server_API.app.core.Embeddings import dlq_crypto

    monkeypatch.setattr(
        dlq_crypto,
        "_aesgcm_encrypt",
        lambda plaintext, key: {
            "alg": "none",
            "b64": base64.b64encode(plaintext).decode("utf-8"),
        },
    )

    with pytest.raises(RuntimeError, match="DLQ encryption unavailable"):
        dlq_crypto.encrypt_payload_if_configured({"secret": "value"})


@pytest.mark.unit
def test_shard_manager_requires_explicit_experimental_flag(monkeypatch):
    from tldw_Server_API.app.core.Embeddings import sharding

    monkeypatch.delenv("EMBEDDINGS_ENABLE_EXPERIMENTAL_SHARDING", raising=False)
    monkeypatch.setattr(sharding, "_shard_manager", None)
    monkeypatch.setattr(sharding, "EmbeddingShardManager", lambda: object())

    with pytest.raises(RuntimeError, match="experimental sharding"):
        sharding.get_shard_manager()


@pytest.mark.unit
def test_request_signer_singleton_requires_configured_secret(monkeypatch):
    from tldw_Server_API.app.core.Embeddings import request_signing

    monkeypatch.delenv("EMBEDDINGS_REQUEST_SIGNING_SECRET", raising=False)
    monkeypatch.setattr(request_signing, "_request_signer", None)

    with pytest.raises(RuntimeError, match="request signing secret"):
        request_signing.get_request_signer()


@pytest.mark.unit
def test_request_signing_api_key_manager_requires_configured_key_file(monkeypatch):
    from tldw_Server_API.app.core.Embeddings import request_signing

    monkeypatch.delenv("EMBEDDINGS_REQUEST_SIGNING_KEYS_FILE", raising=False)
    monkeypatch.setattr(request_signing, "_api_key_manager", None)

    with pytest.raises(RuntimeError, match="API key file"):
        request_signing.get_api_key_manager()
