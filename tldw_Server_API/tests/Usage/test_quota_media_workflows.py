"""Per-user media-ingest bytes/day and workflow runs/day (spec 2 §4)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.core.Usage import quota_checks

pytestmark = pytest.mark.unit
MB = 1024 * 1024


@pytest.fixture()
def quota(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A per-key limit and today's ledger usage, set by the test; ledger writes recorded."""
    state: dict = {"limits": {}, "used": 0.0, "recorded": []}

    async def _user_quota(_uid: int, key: str) -> object:
        """The test's limit for the key."""
        return state["limits"].get(key)

    async def _used(_uid: str, _category: str) -> float:
        """The test's usage."""
        return state["used"]

    async def _record(**kwargs: object) -> bool:
        """Record a media-bytes ledger write."""
        state["recorded"].append(kwargs)
        return True

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(quota_checks, "ledger_used_today", _used)
    monkeypatch.setattr(persistence, "_record_media_ingestion_bytes_ledger_entry", _record)
    return state


async def test_media_bytes_recorded_even_without_a_limit(quota: dict) -> None:
    """Uploads are always counted (gate the check, never the record)."""
    await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert [(r["entity_scope"], r["entity_value"], r["units"]) for r in quota["recorded"]] == [("user", "7", 3 * MB)]


async def test_media_bytes_429_when_daily_mb_spent(quota: dict) -> None:
    """An upload that would pass the user's daily MB is refused with 429 and not recorded."""
    quota["limits"]["limits.media_ingest_mb_per_day"] = 10
    quota["used"] = 8 * MB
    with pytest.raises(HTTPException) as exc:
        await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert exc.value.status_code == 429 and int(exc.value.headers["Retry-After"]) >= 1
    assert quota["recorded"] == []
    await persistence._enforce_and_record_media_bytes(7, 2 * MB)
    assert len(quota["recorded"]) == 1


async def test_workflows_cap_429_at_allowance(quota: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the user's daily run allowance the endpoint refuses with 429 and rate-limit headers."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.delenv("WORKFLOWS_DISABLE_QUOTAS", raising=False)
    quota["limits"]["limits.workflows_runs_per_day"] = 2
    quota["used"] = 1.0
    await workflows_ep._enforce_workflows_daily_cap(request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None)
    quota["used"] = 2.0
    with pytest.raises(HTTPException) as exc:
        await workflows_ep._enforce_workflows_daily_cap(request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None)
    assert exc.value.status_code == 429


async def test_scheduler_refuses_a_run_past_the_allowance(quota: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduled runs count against the same daily allowance and are refused before the run is created."""
    from tldw_Server_API.app.core.Scheduler.handlers import workflows as sched

    created: list[str] = []
    monkeypatch.setattr(sched, "_get_wf_db", lambda: SimpleNamespace(create_run=lambda **kw: created.append("run")))
    quota["limits"]["limits.workflows_runs_per_day"] = 1
    quota["used"] = 1.0
    with pytest.raises(RuntimeError, match="Daily workflow run quota"):
        await sched.workflow_run({"user_id": 7, "workflow_id": 1, "definition_snapshot": {}})
    assert created == []
