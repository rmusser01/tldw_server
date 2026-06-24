from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlist_handler


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_watchlist_run_passes_tenant_id_and_preserves_pipeline_status(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_run_watchlist_job(user_id: int, job_id: int, **kwargs):
        captured["user_id"] = user_id
        captured["job_id"] = job_id
        captured["tenant_id"] = kwargs.get("tenant_id")
        return {"run_id": 42, "status": "cancelled", "items_ingested": 0}

    monkeypatch.setattr(watchlist_handler, "run_watchlist_job", fake_run_watchlist_job)

    result = await watchlist_handler.watchlist_run(
        {
            "inputs": {"watchlist_job_id": 7},
            "user_id": "123",
            "tenant_id": "tenant-acme",
        }
    )

    assert captured == {"user_id": 123, "job_id": 7, "tenant_id": "tenant-acme"}
    assert result == {"run_id": 42, "status": "cancelled", "items_ingested": 0}


@pytest.mark.asyncio
async def test_watchlists_enrich_output_calls_core_handler(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_enrich_output(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(watchlist_handler.output_enrichment_handler, "enrich_output", fake_enrich_output)

    result = await watchlist_handler.watchlists_enrich_output(
        {
            "output_id": "77",
            "user_id": "555",
            "grouping_config": {"group_by": "topic"},
            "summary_config": {"enabled": True, "llm_provider": "mock"},
        }
    )

    assert result == {"output_id": 77, "status": "completed"}
    assert captured == {
        "output_id": 77,
        "user_id": 555,
        "grouping_config": {"group_by": "topic"},
        "summary_config": {"enabled": True, "llm_provider": "mock"},
    }
