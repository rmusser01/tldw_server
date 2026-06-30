from __future__ import annotations

from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=778, username="content-alert-api", email=None, is_active=True)

    base_dir = tmp_path / "watchlist_content_alert_api_user_dbs"
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


def _create_watchlist(client: TestClient, *, name: str = "CTI Alerts") -> dict:
    response = client.post(
        "/api/v1/watchlists",
        json={
            "name": name,
            "description": f"{name} description",
            "objective": f"Track {name}",
            "domain": "cti_osint",
            "priority": "high",
            "tags": ["cti", "alerts"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_source(client: TestClient, *, watchlist_id: int) -> dict:
    response = client.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Advisory Feed",
            "url": "https://example.com/advisories.xml",
            "source_type": "rss",
            "tags": ["advisory"],
            "watchlist_id": watchlist_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_job(client: TestClient, *, source_id: int, watchlist_id: int) -> dict:
    response = client.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Advisory Monitor",
            "scope": {"sources": [source_id]},
            "active": True,
            "watchlist_id": watchlist_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _seed_alert(
    *,
    watchlist_id: int,
    source_id: int,
    job_id: int,
) -> tuple[int, int]:
    db = WatchlistsDatabase.for_user(778)
    run = db.create_run(int(job_id), status="finished")
    item = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job_id),
        source_id=int(source_id),
        media_id=None,
        media_uuid=None,
        url="https://example.com/advisory/cve-2026-1234",
        title="CVE-2026-1234 exploitation observed",
        summary="Active exploitation of CVE-2026-1234 is reported.",
        content="Emergency mitigations are available.",
        published_at="2026-05-15T10:00:00+00:00",
        tags=["cve", "ransomware"],
        status="ingested",
    )
    rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist_id),
        name="Active exploitation",
        rule_kind="cve",
        match_mode="contains",
        pattern="CVE-2026-1234",
        severity="critical",
        source_constraints={"source_ids": [int(source_id)]},
    )
    alert = db.create_content_alert(
        watchlist_id=int(watchlist_id),
        rule_id=int(rule.id),
        item_id=int(item.id),
        run_id=int(run.id),
        job_id=int(job_id),
        source_id=int(source_id),
        severity="critical",
        title=item.title,
        snippet="Active exploitation of CVE-2026-1234 is reported.",
        matched_text="CVE-2026-1234",
        evidence={
            "url": item.url,
            "source_id": source_id,
            "published_at": item.published_at,
        },
        dedupe_key=f"{watchlist_id}:{rule.id}:{item.id}",
    )
    return int(rule.id), int(alert.id)


def test_content_alert_rule_endpoints_crud_and_validation(client_with_user):
    client = client_with_user
    watchlist = _create_watchlist(client)
    watchlist_id = int(watchlist["id"])

    created = client.post(
        f"/api/v1/watchlists/{watchlist_id}/content-alert-rules",
        json={
            "name": "Critical CVE",
            "rule_kind": "cve",
            "match_mode": "contains",
            "pattern": "CVE-2026-1234",
            "severity": "critical",
            "source_constraints": {"source_tags": ["advisory"]},
            "metadata": {"descriptor": "active exploitation"},
        },
    )
    assert created.status_code == 201, created.text
    created_body = created.json()
    assert created_body["watchlist_id"] == watchlist_id
    assert created_body["pattern"] == "CVE-2026-1234"
    assert created_body["source_constraints"] == {"source_tags": ["advisory"]}

    listed = client.get(f"/api/v1/watchlists/{watchlist_id}/content-alert-rules")
    assert listed.status_code == 200, listed.text
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == created_body["id"]

    updated = client.patch(
        f"/api/v1/watchlists/{watchlist_id}/content-alert-rules/{created_body['id']}",
        json={"enabled": False, "severity": "high"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["enabled"] is False
    assert updated.json()["severity"] == "high"

    invalid = client.post(
        f"/api/v1/watchlists/{watchlist_id}/content-alert-rules",
        json={
            "name": "Bad regex",
            "rule_kind": "regex",
            "match_mode": "regex",
            "pattern": "[",
        },
    )
    assert invalid.status_code == 400
    assert "invalid_content_alert_regex" in invalid.text

    unsafe = client.post(
        f"/api/v1/watchlists/{watchlist_id}/content-alert-rules",
        json={
            "name": "Unsafe regex",
            "rule_kind": "regex",
            "match_mode": "regex",
            "pattern": "(a+)+$",
        },
    )
    assert unsafe.status_code == 400
    assert "unsafe_content_alert_regex" in unsafe.text

    deleted = client.delete(f"/api/v1/watchlists/{watchlist_id}/content-alert-rules/{created_body['id']}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True}


def test_content_alert_inbox_filters_detail_and_review_state(client_with_user):
    client = client_with_user
    watchlist = _create_watchlist(client)
    source = _create_source(client, watchlist_id=int(watchlist["id"]))
    job = _create_job(client, source_id=int(source["id"]), watchlist_id=int(watchlist["id"]))
    rule_id, alert_id = _seed_alert(
        watchlist_id=int(watchlist["id"]),
        source_id=int(source["id"]),
        job_id=int(job["id"]),
    )

    listed = client.get(
        f"/api/v1/watchlists/{watchlist['id']}/alerts",
        params={"status": "unread", "severity": "critical", "rule_id": rule_id, "source_id": source["id"]},
    )
    assert listed.status_code == 200, listed.text
    listed_body = listed.json()
    assert listed_body["total"] == 1
    assert listed_body["items"][0]["id"] == alert_id
    assert listed_body["items"][0]["evidence"]["url"] == "https://example.com/advisory/cve-2026-1234"

    searched = client.get(
        f"/api/v1/watchlists/{watchlist['id']}/alerts",
        params={"q": "exploitation observed"},
    )
    assert searched.status_code == 200, searched.text
    assert searched.json()["total"] == 1

    no_match = client.get(
        f"/api/v1/watchlists/{watchlist['id']}/alerts",
        params={"q": "unrelated topic"},
    )
    assert no_match.status_code == 200, no_match.text
    assert no_match.json()["total"] == 0

    detail = client.get(f"/api/v1/watchlists/{watchlist['id']}/alerts/{alert_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["matched_text"] == "CVE-2026-1234"

    read = client.patch(
        f"/api/v1/watchlists/{watchlist['id']}/alerts/{alert_id}",
        json={"status": "read"},
    )
    assert read.status_code == 200, read.text
    assert read.json()["status"] == "read"
    assert read.json()["read_at"] is not None

    dismissed = client.patch(
        f"/api/v1/watchlists/{watchlist['id']}/alerts/{alert_id}",
        json={"status": "dismissed"},
    )
    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["status"] == "dismissed"
    assert dismissed.json()["dismissed_at"] is not None


def test_content_alert_endpoints_are_watchlist_scoped(client_with_user):
    client = client_with_user
    first = _create_watchlist(client, name="First")
    second = _create_watchlist(client, name="Second")
    source = _create_source(client, watchlist_id=int(first["id"]))
    job = _create_job(client, source_id=int(source["id"]), watchlist_id=int(first["id"]))
    rule_id, alert_id = _seed_alert(
        watchlist_id=int(first["id"]),
        source_id=int(source["id"]),
        job_id=int(job["id"]),
    )

    other_rules = client.get(f"/api/v1/watchlists/{second['id']}/content-alert-rules")
    other_alerts = client.get(f"/api/v1/watchlists/{second['id']}/alerts")
    leaked_rule = client.patch(
        f"/api/v1/watchlists/{second['id']}/content-alert-rules/{rule_id}",
        json={"enabled": False},
    )
    leaked_alert = client.get(f"/api/v1/watchlists/{second['id']}/alerts/{alert_id}")

    assert other_rules.status_code == 200
    assert other_rules.json()["total"] == 0
    assert other_alerts.status_code == 200
    assert other_alerts.json()["total"] == 0
    assert leaked_rule.status_code == 404
    assert leaked_alert.status_code == 404
