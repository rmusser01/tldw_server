"""Per-user media-ingest bytes/day and workflow runs/day (spec 2 §4)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.core.Usage import quota_checks
from tldw_Server_API.app.core.Workflows import daily_ledger

pytestmark = pytest.mark.unit
MB = 1024 * 1024


@pytest.fixture()
def quota(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A per-key limit and today's (fake) ledger usage; atomic add-if-within-cap calls recorded.

    The fakes mirror ``ResourceDailyLedger.add_if_within_daily_cap``: a write is
    recorded only when it is admitted, and ``state["used"]`` is read fresh on
    every call, so two sequential calls against the same state exercise the
    same atomicity the real ledger provides (Qodo Q16/Q17).
    """
    state: dict = {"limits": {}, "used": 0, "recorded": []}

    async def _user_quota(_uid: int, key: str) -> object:
        """The test's limit for the key."""
        return state["limits"].get(key)

    async def _record(**kwargs: object) -> bool:
        """Record a media-bytes ledger write (the no-limit/unlimited path)."""
        state["recorded"].append(kwargs)
        return True

    class _FakeMediaLedger:
        """An in-memory stand-in for the one atomic media-bytes ledger call used."""

        async def add_if_within_daily_cap(self, entry: object, daily_cap: int) -> tuple[bool, int]:
            """Admit and record only when the write fits the cap, like the real ledger."""
            used = state["used"]
            units = entry.units  # type: ignore[attr-defined]
            if used + units > daily_cap:
                return False, max(0, daily_cap - used)
            state["used"] = used + units
            state["recorded"].append(
                {
                    "entity_scope": entry.entity_scope,  # type: ignore[attr-defined]
                    "entity_value": entry.entity_value,  # type: ignore[attr-defined]
                    "units": units,
                }
            )
            return True, max(0, daily_cap - state["used"])

    async def _get_media_ledger() -> _FakeMediaLedger:
        """Hand back the fake media ledger."""
        return _FakeMediaLedger()

    async def _consume_workflow_run(
        *, entity_scope: str, entity_value: str, run_id: str, daily_cap: int | None
    ) -> tuple[bool, int]:
        """The same add-if-within-cap semantics, for one workflow run (1 unit)."""
        if daily_cap is None:
            state["recorded"].append({"run_id": run_id})
            return True, 0
        used = state["used"]
        if used + 1 > daily_cap:
            return False, max(0, daily_cap - used)
        state["used"] = used + 1
        state["recorded"].append({"run_id": run_id})
        return True, max(0, daily_cap - state["used"])

    monkeypatch.setattr(quota_checks, "user_quota", _user_quota)
    monkeypatch.setattr(persistence, "_get_media_ingestion_daily_ledger", _get_media_ledger)
    monkeypatch.setattr(persistence, "_record_media_ingestion_bytes_ledger_entry", _record)
    monkeypatch.setattr(daily_ledger, "consume_workflow_run_if_within_cap", _consume_workflow_run)
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


async def test_media_bytes_admitted_at_the_boundary(quota: dict) -> None:
    """An upload that lands exactly at the user's daily MB cap is admitted and recorded (Qodo Q15)."""
    quota["limits"]["limits.media_ingest_mb_per_day"] = 10
    quota["used"] = 8 * MB
    await persistence._enforce_and_record_media_bytes(7, 2 * MB)
    assert len(quota["recorded"]) == 1


async def test_media_bytes_skips_check_and_record_for_a_non_numeric_user_id(quota: dict) -> None:
    """A non-numeric user id must not raise; the check and the per-user record are both skipped."""
    await persistence._enforce_and_record_media_bytes("not-a-number", 3 * MB)
    assert quota["recorded"] == []


async def test_media_bytes_second_of_two_adds_at_the_boundary_is_refused(quota: dict) -> None:
    """Two adds whose combined size exceeds the cap: the second is refused and not recorded.

    The admission check and the ledger write are one atomic operation
    (ResourceDailyLedger.add_if_within_daily_cap), so this holds even when the
    two calls race: whichever runs second always sees the first's write (Qodo Q16).
    """
    quota["limits"]["limits.media_ingest_mb_per_day"] = 5
    await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert len(quota["recorded"]) == 1
    with pytest.raises(HTTPException) as exc:
        await persistence._enforce_and_record_media_bytes(7, 3 * MB)
    assert exc.value.status_code == 429
    assert len(quota["recorded"]) == 1


async def test_workflows_cap_429_at_allowance(quota: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs whose combined count exceeds the daily allowance: the second is refused
    with 429 and rate-limit headers (Qodo Q17: the check and the ledger write are one
    atomic operation, so this holds even when the two calls race)."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.delenv("WORKFLOWS_DISABLE_QUOTAS", raising=False)
    quota["limits"]["limits.workflows_runs_per_day"] = 2
    quota["used"] = 1
    await workflows_ep._enforce_workflows_daily_cap(
        request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None, run_id="run-a"
    )
    with pytest.raises(HTTPException) as exc:
        await workflows_ep._enforce_workflows_daily_cap(
            request=SimpleNamespace(), current_user=SimpleNamespace(id=7), db=None, run_id="run-b"
        )
    assert exc.value.status_code == 429


async def test_workflows_runs_decision_allows_when_disabled_via_env_even_at_zero(
    quota: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WORKFLOWS_DISABLE_QUOTAS makes workflows_runs_decision UNLIMITED, even with limit=0.

    The scheduler path (workflow_run) calls workflows_runs_decision directly, with
    no way to honor the env escape hatch unless the function itself does.
    """
    monkeypatch.setenv("WORKFLOWS_DISABLE_QUOTAS", "1")
    quota["limits"]["limits.workflows_runs_per_day"] = 0
    quota["used"] = 0.0
    decision = await quota_checks.workflows_runs_decision(7)
    assert decision.allowed is True


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
