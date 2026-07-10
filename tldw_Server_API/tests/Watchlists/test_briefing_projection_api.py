from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import get_watchlists_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists.briefing_contract import get_briefing_contract
from tldw_Server_API.app.services.outputs_service import _resolve_output_path_for_user

pytestmark = pytest.mark.unit
USER_ID = 7841


@dataclass(frozen=True)
class SeededOccurrence:
    watchlist_id: int
    job_id: int
    run_id: int
    occurrence_id: int
    output_id: int


@pytest.fixture()
def projection_case(monkeypatch, tmp_path):
    base_dir = tmp_path / "projection-users"
    base_dir.mkdir()
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")

    watchlists_db = WatchlistsDatabase.for_user(USER_ID)
    collections_db = CollectionsDatabase.for_user(USER_ID)
    watchlist = watchlists_db.create_watchlist(name="Daily signals")
    contract = get_briefing_contract(
        {
            "briefing_pipeline": {
                "editorial": {"program_format": "concise_briefing", "show_name": "Signal Check"},
                "text": {"enabled": True},
                "delivery": {
                    "reports": {"enabled": True},
                    "email": {"enabled": True, "recipients": ["reader@example.com"]},
                },
            }
        },
        scheduled=True,
    )
    job = watchlists_db.create_job(
        name="Signals job",
        description=None,
        scope_json="{}",
        schedule_expr="0 8 * * *",
        schedule_timezone="UTC",
        active=True,
        max_concurrency=1,
        per_host_delay_ms=0,
        retry_policy_json="{}",
        output_prefs_json=json.dumps({"briefing_pipeline": contract}),
        watchlist_id=int(watchlist.id),
    )
    watchlists_db.set_job_history(int(job.id), next_run_at="2026-07-11T08:00:00+00:00")
    run = watchlists_db.create_run(int(job.id), status="succeeded")
    occurrence = watchlists_db.create_or_get_briefing_occurrence(
        run_id=int(run.id),
        occurrence_key=f"user:{USER_ID}:job:{job.id}:run:{run.id}:v1",
        contract_json=json.dumps(contract),
    )
    output = collections_db.create_output_artifact(
        type_="briefing_markdown",
        title="Signal Check",
        format_="md",
        storage_path="signal-check.md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "occurrence_id": int(occurrence.id),
                "editorial": contract["editorial"],
                "delivery_plan": contract["delivery"],
            }
        ),
        job_id=int(job.id),
        run_id=int(run.id),
    )
    report_path = _resolve_output_path_for_user(USER_ID, "signal-check.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Signal Check\n", encoding="utf-8")
    stages = {
        "collect": {"status": "ready"},
        "select": {"status": "ready", "candidate_count": 5, "selected_count": 3, "omitted_count": 2},
        "render_text": {"status": "ready"},
        "persist_text": {"status": "ready", "artifact_id": int(output.id)},
        "compose_audio_script": {"status": "skipped", "code": "audio_not_selected"},
        "persist_audio_script": {"status": "skipped", "code": "audio_not_selected"},
        "generate_audio": {"status": "skipped", "code": "audio_not_selected"},
        "persist_audio": {"status": "skipped", "code": "audio_not_selected"},
        "deliver": {"status": "failed", "code": "delivery_failed", "retryable": True},
        "deliver:email": {
            "status": "failed",
            "code": "provider_rejected",
            "retryable": True,
            "outcome": "failed",
        },
    }
    watchlists_db.update_briefing_occurrence(
        int(occurrence.id),
        stages=stages,
        artifact_status="ready",
        delivery_status="failed",
        output_id=int(output.id),
        selected_count=3,
        omitted_count=2,
    )

    async def override_user() -> User:
        return User(id=USER_ID, username="projection-user", email="reader@example.com", is_active=True)

    from tldw_Server_API.app.api.v1.endpoints.watchlists import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_watchlists_db_for_user] = lambda: watchlists_db
    app.dependency_overrides[get_collections_db_for_user] = lambda: collections_db

    seeded = SeededOccurrence(
        watchlist_id=int(watchlist.id),
        job_id=int(job.id),
        run_id=int(run.id),
        occurrence_id=int(occurrence.id),
        output_id=int(output.id),
    )
    with TestClient(app) as client:
        yield client, seeded, watchlists_db, collections_db


def test_latest_projection_separates_artifact_and_delivery_state(projection_case):
    client, seeded, _watchlists_db, _collections_db = projection_case

    response = client.get(
        "/api/v1/watchlists/briefings/latest",
        params={"watchlist_id": seeded.watchlist_id},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["artifact_status"] == "ready"
    assert body["delivery_status"] == "failed"
    assert body["output"]["id"] == seeded.output_id
    assert body["selection"] == {"candidate_count": 5, "included_count": 3, "omitted_count": 2}
    assert body["recovery"]["can_retry_delivery"] is True


def test_run_projection_uses_exact_owned_occurrence_after_newer_run(projection_case):
    client, seeded, watchlists_db, _collections_db = projection_case
    newer_run = watchlists_db.create_run(seeded.job_id, status="succeeded")
    watchlists_db.create_or_get_briefing_occurrence(
        run_id=int(newer_run.id),
        occurrence_key=f"user:{USER_ID}:job:{seeded.job_id}:run:{newer_run.id}:v1",
        contract_json=json.dumps(get_briefing_contract({}, scheduled=True)),
    )

    response = client.get(f"/api/v1/watchlists/runs/{seeded.run_id}/briefing")

    assert response.status_code == 200, response.text
    assert response.json()["occurrence_id"] == seeded.occurrence_id


def test_retry_rejects_ready_stage_without_regeneration(projection_case):
    client, seeded, _watchlists_db, _collections_db = projection_case

    response = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "persist_text"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "stage_already_ready"


def test_retry_rejects_reports_as_external_delivery_adapter(projection_case):
    client, seeded, _watchlists_db, _collections_db = projection_case

    response = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "deliver:reports"},
    )

    assert response.status_code == 422


def test_unknown_delivery_retry_requires_duplicate_risk_confirmation(projection_case, monkeypatch):
    client, seeded, watchlists_db, _collections_db = projection_case
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)
    stages = json.loads(occurrence.stages_json)
    stages["deliver:email"] = {
        "status": "failed",
        "code": "delivery_outcome_unknown",
        "retryable": False,
        "outcome": "unknown",
    }
    watchlists_db.update_briefing_occurrence(
        seeded.occurrence_id,
        stages=stages,
        delivery_status="unknown",
    )
    submit = AsyncMock(return_value="delivery-task-2")
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.schedule_briefing_delivery",
        submit,
    )

    blocked = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "deliver:email"},
    )
    confirmed = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "deliver:email", "confirm_unknown_delivery_retry": True},
    )

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "unknown_delivery_confirmation_required"
    assert "duplicate" in blocked.json()["detail"]["message"].lower()
    assert confirmed.status_code == 200, confirmed.text
    assert submit.await_count == 1


def test_sending_delivery_retry_requires_duplicate_risk_confirmation(projection_case, monkeypatch):
    client, seeded, watchlists_db, _collections_db = projection_case
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)
    stages = json.loads(occurrence.stages_json)
    stages["deliver:email"] = {
        "status": "running",
        "code": None,
        "retryable": False,
        "outcome": "sending",
    }
    watchlists_db.update_briefing_occurrence(
        seeded.occurrence_id,
        stages=stages,
        delivery_status="delivering",
    )
    submit = AsyncMock(return_value="delivery-task-sending")
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.schedule_briefing_delivery",
        submit,
    )

    blocked = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "deliver:email"},
    )
    confirmed = client.post(
        f"/api/v1/watchlists/runs/{seeded.run_id}/briefing/retry",
        json={"stage": "deliver:email", "confirm_unknown_delivery_retry": True},
    )

    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "unknown_delivery_confirmation_required"
    assert confirmed.status_code == 200, confirmed.text
    assert submit.await_count == 1


def test_completed_audio_does_not_override_failed_text(projection_case):
    from tldw_Server_API.app.core.Watchlists.briefing_projection import build_briefing_projection

    _client, seeded, watchlists_db, collections_db = projection_case
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)
    contract = json.loads(occurrence.contract_json)
    contract["audio"]["enabled"] = True
    stages = json.loads(occurrence.stages_json)
    stages["persist_text"] = {"status": "failed", "code": "text_persist_failed", "retryable": True}
    watchlists_db.update_briefing_occurrence(
        seeded.occurrence_id,
        stages=stages,
        artifact_status="failed",
    )
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)
    occurrence.contract_json = json.dumps(contract)

    projection = build_briefing_projection(
        occurrence=occurrence,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
        audio={"status": "completed", "final_artifact": {"artifact_id": "audio-final"}},
    )

    assert projection["artifact_status"] == "failed"
    assert projection["stages"]["persist_text"]["status"] == "failed"
    assert projection["stages"]["persist_audio"]["status"] == "ready"


def test_missing_report_file_is_removed_and_persisted_failed(projection_case):
    from tldw_Server_API.app.core.Watchlists.briefing_projection import build_briefing_projection

    _client, seeded, watchlists_db, collections_db = projection_case
    _resolve_output_path_for_user(USER_ID, "signal-check.md").unlink()
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)

    projection = build_briefing_projection(
        occurrence=occurrence,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    persisted = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)
    assert projection["output"] is None
    assert projection["recovery"]["can_open_report"] is False
    assert projection["artifact_status"] == "failed"
    assert json.loads(persisted.stages_json)["persist_text"]["code"] == "briefing_text_artifact_missing"


def test_cross_run_report_is_removed_and_persisted_failed(projection_case):
    from tldw_Server_API.app.core.Watchlists.briefing_projection import build_briefing_projection

    _client, seeded, watchlists_db, collections_db = projection_case
    wrong = collections_db.create_output_artifact(
        type_="briefing_markdown",
        title="Wrong run",
        format_="md",
        storage_path="wrong-run.md",
        metadata_json=json.dumps({"origin": "watchlists", "occurrence_id": seeded.occurrence_id}),
        job_id=seeded.job_id,
        run_id=seeded.run_id + 999,
    )
    wrong_path = _resolve_output_path_for_user(USER_ID, "wrong-run.md")
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_text("wrong", encoding="utf-8")
    watchlists_db.update_briefing_occurrence(seeded.occurrence_id, output_id=int(wrong.id))
    occurrence = watchlists_db.get_briefing_occurrence(seeded.occurrence_id)

    projection = build_briefing_projection(
        occurrence=occurrence,
        watchlists_db=watchlists_db,
        collections_db=collections_db,
    )

    assert projection["output"] is None
    assert projection["artifact_status"] == "failed"
    assert json.loads(watchlists_db.get_briefing_occurrence(seeded.occurrence_id).stages_json)[
        "persist_text"
    ]["code"] == "briefing_output_ownership_mismatch"
