import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Collections import embedding_queue
from tldw_Server_API.app.core.Collections.embedding_queue import enqueue_embeddings_job_for_item
from tldw_Server_API.app.core.Embeddings import redis_pipeline


def _disable_test_mode(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")


@pytest.mark.asyncio
async def test_enqueue_embeddings_job_uses_manager(monkeypatch):
    captured = {}
    enqueue = {}

    class FakeManager:
        def create_job(self, **kwargs):
            captured["job_kwargs"] = kwargs
            return {"id": 123, "uuid": "root-123"}

    _disable_test_mode(monkeypatch)
    monkeypatch.setattr(embedding_queue, "_jobs_manager", lambda: FakeManager())
    monkeypatch.setattr(redis_pipeline, "enqueue_content_job", lambda **kwargs: enqueue.update(kwargs) or "stream-1")

    await enqueue_embeddings_job_for_item(
        user_id=123,
        item_id=456,
        content="Example content to embed.",
        metadata={"origin": "reading"},
    )

    job_kwargs = captured["job_kwargs"]
    assert job_kwargs["domain"] == "embeddings"
    assert job_kwargs["queue"] == "low"
    assert job_kwargs["job_type"] == "embeddings_pipeline"
    assert job_kwargs["owner_user_id"] == "123"
    assert job_kwargs["payload"]["item_id"] == 456
    assert "content" in job_kwargs["payload"] and "Example content" in job_kwargs["payload"]["content"]
    assert job_kwargs["payload"]["metadata"]["origin"] == "reading"
    assert enqueue["root_job_uuid"] == "root-123"
    assert enqueue["payload"]["root_job_uuid"] == "root-123"
    assert enqueue["payload"]["user_id"] == "123"


@pytest.mark.asyncio
async def test_enqueue_embeddings_job_bounds_payload_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding jobs cap payload content and include truncation metadata."""
    captured: dict[str, Any] = {}
    enqueue: dict[str, Any] = {}

    class FakeManager:
        def create_job(self, **kwargs: Any) -> dict[str, object]:
            captured["job_kwargs"] = kwargs
            return {"id": 123, "uuid": "root-123"}

    _disable_test_mode(monkeypatch)
    monkeypatch.setattr(embedding_queue, "_jobs_manager", lambda: FakeManager())
    monkeypatch.setattr(embedding_queue, "EMBEDDING_JOB_CONTENT_MAX_CHARS", 12)
    monkeypatch.setattr(redis_pipeline, "allow_stub", lambda: True)
    monkeypatch.setattr(redis_pipeline, "enqueue_content_job", lambda **kwargs: enqueue.update(kwargs) or "stream-1")

    await enqueue_embeddings_job_for_item(
        user_id=123,
        item_id=456,
        content="A" * 80,
        metadata={"origin": "reading"},
    )

    payload = captured["job_kwargs"]["payload"]
    assert len(payload["content"]) <= 12
    assert payload["metadata"]["content_truncated"] is True
    assert payload["metadata"]["content_char_count"] == 80
    assert enqueue["payload"]["metadata"]["content_truncated"] is True


@pytest.mark.asyncio
async def test_enqueue_embeddings_skips_empty_content(monkeypatch):
    called = {}

    class FakeManager:
        def create_job(self, **kwargs):
            called["job_kwargs"] = kwargs

    _disable_test_mode(monkeypatch)
    monkeypatch.setattr(embedding_queue, "_jobs_manager", lambda: FakeManager())

    await enqueue_embeddings_job_for_item(
        user_id=1,
        item_id=2,
        content="   ",
        metadata={"origin": "reading"},
    )

    assert "initialized" not in called


@pytest.mark.asyncio
async def test_enqueue_embeddings_best_effort_when_queue_unavailable(monkeypatch):
    class FakeManager:
        def create_job(self, **kwargs):
            raise RuntimeError("queue unavailable")

    _disable_test_mode(monkeypatch)
    monkeypatch.setattr(embedding_queue, "_jobs_manager", lambda: FakeManager())

    await enqueue_embeddings_job_for_item(
        user_id=7,
        item_id=8,
        content="Queue fallback coverage.",
        metadata={"origin": "reading"},
    )
