from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization
from tldw_Server_API.app.core.Watchlists import output_enrichment_handler


pytestmark = pytest.mark.unit


class RecordingScheduler:
    def __init__(self, *, fail_submit: bool = False) -> None:
        self.fail_submit = fail_submit
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def submit(self, *args, **kwargs) -> str:
        self.calls.append((args, kwargs))
        if self.fail_submit:
            raise RuntimeError("scheduler unavailable")
        return "task_enrich_123"


@pytest.mark.asyncio
async def test_schedule_output_enrichment_submits_scheduler_task():
    scheduler = RecordingScheduler()
    grouping = SimpleNamespace(model_dump=lambda: {"group_by": "topic"})

    result = await output_enrichment_handler.schedule_output_enrichment(
        output_id=77,
        user_id=555,
        grouping_config=grouping,
        summary_config={"enabled": True, "llm_provider": "mock"},
        scheduler=scheduler,
    )

    assert result.status == "submitted"
    assert result.task_id == "task_enrich_123"
    assert result.submitted is True
    assert len(scheduler.calls) == 1
    args, kwargs = scheduler.calls[0]
    assert args == ("watchlists_enrich_output",)
    assert kwargs["queue_name"] == "watchlists"
    assert kwargs["idempotency_key"] == "watchlists-output-enrichment:555:77"
    assert kwargs["max_retries"] == 1
    assert kwargs["metadata"] == {
        "source": "watchlists_output_enrichment",
        "watchlist_output_id": 77,
        "user_id": "555",
    }
    assert kwargs["payload"] == {
        "output_id": 77,
        "user_id": 555,
        "grouping_config": {"group_by": "topic"},
        "summary_config": {"enabled": True, "llm_provider": "mock"},
    }


@pytest.mark.asyncio
async def test_schedule_output_enrichment_uses_fallback_when_scheduler_submit_fails():
    scheduler = RecordingScheduler(fail_submit=True)
    fallback_calls: list[tuple[Any, dict[str, Any]]] = []

    def fallback_submitter(func, **kwargs):
        fallback_calls.append((func, kwargs))

    result = await output_enrichment_handler.schedule_output_enrichment(
        output_id=88,
        user_id=777,
        summary_config={"enabled": True},
        scheduler=scheduler,
        fallback_submitter=fallback_submitter,
    )

    assert result.status == "fallback_scheduled"
    assert result.reason == "scheduler_submit_failed"
    assert result.submitted is False
    assert fallback_calls == [
        (
            output_enrichment_handler.enrich_output,
            {
                "output_id": 88,
                "user_id": 777,
                "summary_config": {"enabled": True},
            },
        )
    ]


# --- enrich_output through the real analyze() (TASK-13636) ---


class _Adapter:
    """The provider adapter below analyze(): replies, or raises the quota refusal."""

    def __init__(self) -> None:
        self.error: BaseException | None = None

    def chat(self, request, timeout=None):
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": "Model summary."}}]}


class _CollectionsDB:
    """An output with two items; records each metadata write."""

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def get_output_artifact(self, output_id: int) -> SimpleNamespace:
        return SimpleNamespace(metadata_json=json.dumps({"item_ids": [1, 2]}))

    def get_content_item(self, item_id: int) -> SimpleNamespace:
        return SimpleNamespace(id=item_id, title=f"Story {item_id}", url="", summary=f"About {item_id}", tags=["tech"])

    def update_output_artifact_metadata(self, output_id: int, metadata_json: str) -> None:
        self.saved.append(json.loads(metadata_json))


@pytest.fixture
def enrichment(monkeypatch: pytest.MonkeyPatch) -> tuple[_Adapter, _CollectionsDB]:
    adapter, db = _Adapter(), _CollectionsDB()
    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter))
    monkeypatch.setattr(CollectionsDatabase, "for_user", staticmethod(lambda user_id: db))
    return adapter, db


_SUMMARY = {"llm_provider": "openai", "llm_model": "test-model", "per_group_summaries": True}


async def test_enrich_output_stores_the_model_summaries(enrichment) -> None:
    _adapter, db = enrichment
    await output_enrichment_handler.enrich_output(
        output_id=5, user_id=7, grouping_config={"group_by": "tag"}, summary_config=_SUMMARY
    )
    assert db.saved[-1]["enrichment_status"] == "completed"
    assert db.saved[-1]["briefing_summary_text"] == "Model summary."
    assert db.saved[-1]["enriched_groups"][0]["summary"] == "Model summary."


async def test_enrich_output_records_a_refused_summary_as_failed(enrichment) -> None:
    adapter, db = enrichment
    adapter.error = LLMQuotaExceededError(
        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
    )
    await output_enrichment_handler.enrich_output(
        output_id=5, user_id=7, grouping_config={"group_by": "tag"}, summary_config=_SUMMARY
    )
    assert db.saved[-1]["enrichment_status"] == "failed"
    assert "limit_exceeded" in db.saved[-1]["enrichment_error"]


async def test_enrich_output_records_a_refused_topic_grouping_as_failed(enrichment) -> None:
    adapter, db = enrichment
    adapter.error = LLMQuotaExceededError(
        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
    )
    await output_enrichment_handler.enrich_output(
        output_id=5,
        user_id=7,
        grouping_config={"group_by": "topic", "topic_llm_provider": "openai", "topic_llm_model": "test-model"},
    )
    assert db.saved[-1]["enrichment_status"] == "failed"
    assert "limit_exceeded" in db.saved[-1]["enrichment_error"]
