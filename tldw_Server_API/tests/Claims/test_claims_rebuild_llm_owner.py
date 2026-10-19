"""In-process claims rebuild tasks carry their owner, and their LLM calls are recorded for that owner (TASK-13585)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_rebuild_service
from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import ClaimsRebuildService, ClaimsRebuildTask
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.unit


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, make the test's loop the accounting loop, and skip health writes."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(ClaimsRebuildService, "_persist_health", lambda self, force=False: None)
    return captured


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Yield to the loop until ``predicate`` holds or the timeout passes."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


async def test_a_quota_refusal_fails_the_task_and_the_worker_keeps_going(rows, monkeypatch) -> None:
    """A refused LLM call is one failed task, not a dead worker thread that strands every later task."""
    seen: list[int] = []

    def _fake_rebuild(*, db_path: str, media_id: int) -> dict[str, Any]:
        """Refuse the first media item, rebuild the second."""
        seen.append(media_id)
        if media_id == 1:
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=10, limit=10,
                message="Monthly LLM token limit reached", retry_after=60,
            )
        return {"outcome": "ok", "media_id": media_id}

    monkeypatch.setattr(claims_rebuild_service, "rebuild_claims_for_media", _fake_rebuild)
    svc = ClaimsRebuildService(worker_threads=1)
    svc.start()
    try:
        svc.submit(media_id=1, db_path=":memory:")
        svc.submit(media_id=2, db_path=":memory:")
        await _until(lambda: svc.get_stats()["processed"] == 1)
    finally:
        svc.stop()
    stats = svc.get_stats()
    assert (stats["failed"], stats["processed"], seen) == (1, 1, [1, 2])
    last_failure = svc.get_health()["last_failure"]
    assert (last_failure["media_id"], last_failure["error"]) == (1, "llm_quota_exceeded")


def test_a_refusal_during_extraction_leaves_the_media_claims_in_place(monkeypatch) -> None:
    """Extraction runs before the old claims are replaced, so a refused rebuild deletes nothing."""
    deleted: list[int] = []

    class _FakeDb:
        """A media DB with one item whose claims must survive."""

        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> dict:
            """The media item."""
            return {"id": media_id, "title": "Doc", "content": "A sentence."}

        def soft_delete_claims_for_media(self, media_id: int) -> int:
            """Record a deletion."""
            deleted.append(media_id)
            return 1

    @contextmanager
    def _fake_managed_media_database(*_args: Any, **_kwargs: Any) -> Iterator[_FakeDb]:
        """Hand out the fake DB."""
        yield _FakeDb()

    refusal = LLMQuotaExceededError(
        category="llm_tokens_month", current=10, limit=10,
        message="Monthly LLM token limit reached", retry_after=60,
    )

    def _refuse(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Extraction's LLM call is refused."""
        raise refusal

    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_rebuild_service, "chunk_for_embedding", lambda content, file_name: [{"text": content}])
    monkeypatch.setattr(claims_rebuild_service, "resolve_claims_job_budget", lambda settings: None)
    monkeypatch.setattr(claims_rebuild_service, "extract_claims_for_chunks", _refuse)
    with pytest.raises(LLMQuotaExceededError) as raised:
        ClaimsRebuildService(worker_threads=1)._process_task(ClaimsRebuildTask(media_id=7, db_path="/tmp/m.db"))
    assert raised.value is refusal  # unwrapped, so the worker counts it as a refusal, not a fault
    assert deleted == []
