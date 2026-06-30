from __future__ import annotations

import json
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.services.outputs_service import _resolve_output_path_for_user


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=888, username="watchlist-reports", email=None, is_active=True)

    base_dir = tmp_path / "watchlist_reports_user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")
    monkeypatch.setenv("TEST_MODE", "1")

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def client_with_mutable_user(monkeypatch, tmp_path):
    user_state = {"id": 888}

    async def override_user():
        user_id = int(user_state["id"])
        return User(id=user_id, username=f"watchlist-reports-{user_id}", email=None, is_active=True)

    base_dir = tmp_path / "watchlist_reports_mutable_user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")
    monkeypatch.setenv("TEST_MODE", "1")

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, user_state
    app.dependency_overrides.clear()


def _create_watchlist(client: TestClient, *, domain: str = "cti_osint") -> dict:
    response = client.post(
        "/api/v1/watchlists",
        json={
            "name": f"Stage 5B {domain}",
            "description": "Stage 5B report evidence test",
            "objective": "Track defensible report evidence",
            "domain": domain,
            "priority": "critical",
            "tags": ["stage5b"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_source(client: TestClient, *, label: str, watchlist_id: int) -> dict:
    response = client.post(
        "/api/v1/watchlists/sources",
        json={
            "name": f"{label} feed",
            "url": f"https://{label}.example/rss.xml",
            "source_type": "rss",
            "watchlist_id": watchlist_id,
            "tags": ["cti", label],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_job(client: TestClient, *, watchlist_id: int, source_ids: list[int]) -> dict:
    response = client.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Stage 5B monitor",
            "description": "Report evidence monitor",
            "scope": {"sources": source_ids},
            "active": True,
            "watchlist_id": watchlist_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _seed_report_case(client: TestClient) -> dict:
    watchlist = _create_watchlist(client)
    source_a = _create_source(client, label="vendor", watchlist_id=int(watchlist["id"]))
    source_b = _create_source(client, label="local-news", watchlist_id=int(watchlist["id"]))
    job = _create_job(
        client,
        watchlist_id=int(watchlist["id"]),
        source_ids=[int(source_a["id"]), int(source_b["id"])],
    )

    db = WatchlistsDatabase.for_user(888)
    run = db.create_run(int(job["id"]), status="finished")
    item_a = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(source_a["id"]),
        media_id=None,
        media_uuid=None,
        url="https://vendor.example/cve-2026-1234",
        title="Vendor advisory for CVE-2026-1234",
        summary="Active exploitation observed against hospitals.",
        content="Active exploitation details and mitigation guidance.",
        published_at="2026-05-15T10:00:00+00:00",
        tags=["cve", "ransomware"],
        status="ingested",
    )
    item_b = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(source_b["id"]),
        media_id=None,
        media_uuid=None,
        url="https://local-news.example/hospital-ransomware",
        title="Local hospital ransomware disruption",
        summary="Regional reporting confirms operational disruption.",
        content="Local reporting with incident timeline.",
        published_at="2026-05-15T11:00:00+00:00",
        tags=["healthcare"],
        status="ingested",
    )
    item_excluded = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(source_a["id"]),
        media_id=None,
        media_uuid=None,
        url="https://vendor.example/background",
        title="Background item",
        summary="Background item not queued for this report.",
        content="Background content.",
        published_at="2026-05-14T10:00:00+00:00",
        tags=["background"],
        status="ingested",
    )
    db.update_item_flags(int(item_a.id), reviewed=True, queued_for_briefing=True)
    db.update_item_flags(int(item_b.id), reviewed=True, queued_for_briefing=True)
    db.update_item_flags(int(item_excluded.id), reviewed=False, queued_for_briefing=False)

    rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist["id"]),
        name="Active exploitation",
        rule_kind="descriptor",
        match_mode="contains",
        pattern="active exploitation",
        severity="critical",
    )
    alert = db.create_content_alert(
        watchlist_id=int(watchlist["id"]),
        rule_id=int(rule.id),
        item_id=int(item_a.id),
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(source_a["id"]),
        severity="critical",
        title="Active exploitation observed",
        snippet="Active exploitation observed against hospitals.",
        matched_text="active exploitation",
        evidence={"url": "https://vendor.example/cve-2026-1234"},
        dedupe_key=f"stage5b:{item_a.id}",
    )

    return {
        "watchlist": watchlist,
        "job": job,
        "run_id": int(run.id),
        "included_item_ids": [int(item_a.id), int(item_b.id)],
        "excluded_item_id": int(item_excluded.id),
        "alert_id": int(alert.id),
    }


def test_create_output_persists_immutable_report_evidence_and_readiness_endpoints(client_with_user):
    client = client_with_user
    seeded = _seed_report_case(client)

    created = client.post(
        "/api/v1/watchlists/outputs",
        json={
            "run_id": seeded["run_id"],
            "item_ids": seeded["included_item_ids"],
            "title": "CTI evidence report",
            "format": "md",
            "report_preset": "cti_osint",
            "retention_seconds": 0,
        },
    )
    assert created.status_code == 200, created.text
    output = created.json()
    output_id = int(output["id"])
    metadata = output["metadata"]

    assert metadata["report_preset"] == "cti_osint"
    assert metadata["report_schema_version"] == 1
    assert metadata["report_snapshot_path"].endswith(".json")
    assert metadata["report_readiness"]["state"] == "ready"
    assert metadata["included_item_count"] == 2
    assert metadata["excluded_item_count"] == 1
    assert metadata["source_count"] == 2
    assert metadata["alert_count"] == 1
    assert metadata["weak_evidence_warning_count"] == 0

    evidence = client.get(f"/api/v1/watchlists/outputs/{output_id}/evidence")
    assert evidence.status_code == 200, evidence.text
    payload = evidence.json()
    assert payload["immutable_snapshot"] is True
    assert payload["output_id"] == output_id
    assert payload["snapshot"]["output_id"] == output_id
    assert payload["snapshot"]["included_count"] == 2
    assert payload["snapshot"]["excluded_count"] == 1
    assert payload["snapshot"]["alert_count"] == 1
    assert payload["snapshot"]["included_items"][0]["alerts"][0]["id"] == seeded["alert_id"]
    assert payload["snapshot"]["excluded_items"][0]["id"] == seeded["excluded_item_id"]
    assert payload["snapshot"]["excluded_items"][0]["reason"] == "not_queued_for_report"

    readiness = client.get(f"/api/v1/watchlists/outputs/{output_id}/readiness")
    assert readiness.status_code == 200, readiness.text
    assert readiness.json()["state"] == "ready"

    downloaded = client.get(f"/api/v1/watchlists/outputs/{output_id}/download")
    assert downloaded.status_code == 200, downloaded.text
    assert "CTI evidence report" in downloaded.text


def test_create_output_defaults_report_preset_for_existing_clients(client_with_user):
    client = client_with_user
    seeded = _seed_report_case(client)

    created = client.post(
        "/api/v1/watchlists/outputs",
        json={
            "run_id": seeded["run_id"],
            "title": "Default report contract",
            "retention_seconds": 0,
        },
    )
    assert created.status_code == 200, created.text
    metadata = created.json()["metadata"]

    assert metadata["report_preset"] == "cti_osint"
    assert metadata["report_snapshot_path"].endswith(".json")
    assert "report_readiness" in metadata


def test_legacy_output_returns_live_only_readiness(client_with_user):
    client = client_with_user
    seeded = _seed_report_case(client)
    row = CollectionsDatabase.for_user(888).create_output_artifact(
        type_="briefing_markdown",
        title="Legacy report",
        format_="md",
        storage_path="legacy-report.md",
        metadata_json=json.dumps({"origin": "watchlists", "version": 1}),
        job_id=int(seeded["job"]["id"]),
        run_id=seeded["run_id"],
    )

    evidence = client.get(f"/api/v1/watchlists/outputs/{int(row.id)}/evidence")
    assert evidence.status_code == 200, evidence.text
    payload = evidence.json()
    assert payload["immutable_snapshot"] is False
    assert payload["snapshot"] is None
    assert payload["readiness"]["state"] == "legacy_live_only"

    readiness = client.get(f"/api/v1/watchlists/outputs/{int(row.id)}/readiness")
    assert readiness.status_code == 200, readiness.text
    assert readiness.json()["state"] == "legacy_live_only"


def test_output_evidence_reports_missing_sidecar(client_with_user):
    client = client_with_user
    seeded = _seed_report_case(client)
    created = client.post(
        "/api/v1/watchlists/outputs",
        json={
            "run_id": seeded["run_id"],
            "item_ids": seeded["included_item_ids"],
            "title": "Missing sidecar report",
            "retention_seconds": 0,
        },
    )
    assert created.status_code == 200, created.text
    output = created.json()
    sidecar_path = _resolve_output_path_for_user(888, output["metadata"]["report_snapshot_path"])
    sidecar_path.unlink()

    evidence = client.get(f"/api/v1/watchlists/outputs/{int(output['id'])}/evidence")
    assert evidence.status_code == 404, evidence.text
    assert evidence.json()["detail"] == "report_snapshot_missing"


def test_output_evidence_rejects_non_watchlists_and_cross_user_artifacts(client_with_mutable_user):
    client, user_state = client_with_mutable_user
    cdb = CollectionsDatabase.for_user(888)
    non_watchlists = cdb.create_output_artifact(
        type_="briefing_markdown",
        title="Plain output",
        format_="md",
        storage_path="plain-output.md",
        metadata_json=json.dumps({"origin": "outputs"}),
        job_id=1,
        run_id=1,
    )
    non_watchlists_response = client.get(f"/api/v1/watchlists/outputs/{int(non_watchlists.id)}/evidence")
    assert non_watchlists_response.status_code == 404, non_watchlists_response.text
    assert non_watchlists_response.json()["detail"] == "output_not_found"

    user_state["id"] = 889
    foreign = CollectionsDatabase.for_user(889).create_output_artifact(
        type_="briefing_markdown",
        title="Foreign report",
        format_="md",
        storage_path="foreign-report.md",
        metadata_json=json.dumps({"origin": "watchlists", "version": 1}),
        job_id=1,
        run_id=1,
    )

    user_state["id"] = 888
    cross_user_response = client.get(f"/api/v1/watchlists/outputs/{int(foreign.id)}/evidence")
    assert cross_user_response.status_code == 404, cross_user_response.text
