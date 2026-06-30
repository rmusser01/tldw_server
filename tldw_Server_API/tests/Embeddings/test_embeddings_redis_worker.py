import pytest

from tldw_Server_API.app.core.Embeddings import redis_pipeline
from tldw_Server_API.app.core.Embeddings.services import redis_worker
from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis


@pytest.mark.asyncio
async def test_embeddings_redis_worker_chunking_enqueues_embedding(monkeypatch):
    client = InMemoryAsyncRedis(decode_responses=True)
    streams = redis_pipeline.RedisEmbeddingsQueues(
        streams={
            "chunking": "embeddings:chunking",
            "embedding": "embeddings:embedding",
            "storage": "embeddings:storage",
        },
        groups={
            "chunking": "chunking-workers",
            "embedding": "embedding-workers",
            "storage": "storage-workers",
        },
        dlq_prefix="embeddings:dlq",
    )

    async def fake_chunking(job, payload, **_kwargs):
        return {
            "chunks_path": "chunks.json",
            "total_chunks": 1,
            "chunks_processed": 1,
        }, False

    monkeypatch.setattr(redis_worker.jobs_worker, "_handle_chunking_job", fake_chunking)
    monkeypatch.setattr(redis_worker.jobs_worker, "_update_root_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(redis_worker.jobs_worker, "_update_root_job", lambda *_a, **_k: None)
    monkeypatch.setattr(redis_worker.jobs_worker, "_update_root_result", lambda *_a, **_k: None)
    monkeypatch.setattr(redis_worker.jobs_worker, "_resolve_model_provider", lambda *_a: ("model", "provider"))

    fields = {
        "root_job_uuid": "root-1",
        "media_id": "1",
        "user_id": "user1",
        "chunk_size": "1000",
        "chunk_overlap": "200",
    }

    await redis_worker._handle_stage_message(
        stage="chunking",
        message_id="1-0",
        fields=fields,
        client=client,
        streams=streams,
    )

    assert await client.xlen(streams.streams["embedding"]) == 1
    entries = await client.xrange(streams.streams["embedding"], "-", "+")
    assert entries
    _, payload = entries[0]
    assert payload.get("root_job_uuid") == "root-1"
    assert payload.get("chunks_path") == "chunks.json"


@pytest.mark.asyncio
async def test_embeddings_redis_worker_storage_marks_media_complete_after_root(monkeypatch):
    client = InMemoryAsyncRedis(decode_responses=True)
    streams = redis_pipeline.RedisEmbeddingsQueues(
        streams={
            "chunking": "embeddings:chunking",
            "embedding": "embeddings:embedding",
            "storage": "embeddings:storage",
        },
        groups={
            "chunking": "chunking-workers",
            "embedding": "embedding-workers",
            "storage": "storage-workers",
        },
        dlq_prefix="embeddings:dlq",
    )
    call_order: list[tuple[object, ...]] = []

    async def fake_storage(job, payload, **_kwargs):
        return {
            "embedding_count": 1,
            "chunks_processed": 1,
            "total_chunks": 1,
            "embedding_model": "model",
            "embedding_provider": "provider",
        }

    monkeypatch.setattr(redis_worker.jobs_worker, "_handle_storage_job", fake_storage)
    monkeypatch.setattr(redis_worker.jobs_worker, "_resolve_model_provider", lambda *_a: ("model", "provider"))
    monkeypatch.setattr(
        redis_worker.jobs_worker,
        "_update_root_job",
        lambda root_uuid, *, status, result=None, error=None: call_order.append(("root", root_uuid, status)),
    )
    monkeypatch.setattr(
        redis_worker.jobs_worker,
        "_mark_media_embeddings_complete",
        lambda *, user_id, media_id: call_order.append(("complete", user_id, media_id)),
    )

    await redis_worker._handle_stage_message(
        stage="storage",
        message_id="1-0",
        fields={
            "root_job_uuid": "root-2",
            "media_id": "2",
            "user_id": "user2",
            "embedding_model": "model",
            "embedding_provider": "provider",
        },
        client=client,
        streams=streams,
    )

    assert call_order == [
        ("root", "root-2", "completed"),
        ("complete", "user2", 2),
    ]
