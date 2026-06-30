from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

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
