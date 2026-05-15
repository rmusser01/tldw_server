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
        return User(id=779, username="items-triage-api", email=None, is_active=True)

    base_dir = tmp_path / "watchlist_items_triage_api_user_dbs"
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


def _create_watchlist(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/watchlists",
        json={
            "name": "Stage 4 API Watchlist",
            "description": "Stage 4 API description",
            "objective": "Track vulnerable products",
            "domain": "cti_osint",
            "priority": "high",
            "tags": ["cti"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_source(client: TestClient, *, watchlist_id: int, label: str) -> dict:
    response = client.post(
        "/api/v1/watchlists/sources",
        json={
            "name": f"{label} Feed",
            "url": f"https://example.com/{label}.xml",
            "source_type": "rss",
            "tags": [label],
            "watchlist_id": watchlist_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_job(client: TestClient, *, watchlist_id: int, source_ids: list[int]) -> dict:
    response = client.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Stage 4 API Monitor",
            "scope": {"sources": source_ids},
            "active": True,
            "watchlist_id": watchlist_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _seed_items_and_alert(client: TestClient) -> dict:
    watchlist = _create_watchlist(client)
    first_source = _create_source(client, watchlist_id=int(watchlist["id"]), label="alpha")
    second_source = _create_source(client, watchlist_id=int(watchlist["id"]), label="beta")
    job = _create_job(
        client,
        watchlist_id=int(watchlist["id"]),
        source_ids=[int(first_source["id"]), int(second_source["id"])],
    )
    db = WatchlistsDatabase.for_user(779)
    run = db.create_run(int(job["id"]), status="finished")
    older = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(second_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/beta/old",
        title="Older beta update",
        summary="Older beta summary.",
        content="Older beta content",
        published_at="2026-05-13T08:00:00+00:00",
        tags=["beta"],
        status="ingested",
    )
    newer = db.record_scraped_item(
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(first_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/alpha/new",
        title="Newer alpha update",
        summary="Active exploitation of CVE-2026-4242.",
        content="Newer alpha content",
        published_at="2026-05-15T08:00:00+00:00",
        tags=["alpha", "cve"],
        status="ingested",
    )
    db.backend.execute(
        "UPDATE scraped_items SET created_at = ? WHERE id = ?",
        ("2026-05-13T09:00:00+00:00", int(older.id)),
    )
    db.backend.execute(
        "UPDATE scraped_items SET created_at = ? WHERE id = ?",
        ("2026-05-15T09:00:00+00:00", int(newer.id)),
    )
    rule = db.create_content_alert_rule(
        watchlist_id=int(watchlist["id"]),
        name="Critical CVE",
        rule_kind="cve",
        pattern="CVE-2026-4242",
        severity="critical",
    )
    alert = db.create_content_alert(
        watchlist_id=int(watchlist["id"]),
        rule_id=int(rule.id),
        item_id=int(newer.id),
        run_id=int(run.id),
        job_id=int(job["id"]),
        source_id=int(first_source["id"]),
        severity="critical",
        title="Newer alpha update",
        snippet="Active exploitation of CVE-2026-4242.",
        matched_text="CVE-2026-4242",
        evidence={"field": "summary"},
        dedupe_key=f"{watchlist['id']}:{rule.id}:{newer.id}",
    )
    return {
        "watchlist": watchlist,
        "older_id": int(older.id),
        "newer_id": int(newer.id),
        "rule_id": int(rule.id),
        "alert_id": int(alert.id),
    }


def test_items_endpoint_sorts_filters_and_returns_optional_alert_summary(client_with_user):
    client = client_with_user
    seeded = _seed_items_and_alert(client)
    watchlist_id = int(seeded["watchlist"]["id"])

    published_asc = client.get(
        "/api/v1/watchlists/items",
        params={"watchlist_id": watchlist_id, "sort": "published_asc"},
    )
    assert published_asc.status_code == 200, published_asc.text
    assert [item["id"] for item in published_asc.json()["items"]] == [
        seeded["older_id"],
        seeded["newer_id"],
    ]

    alert_items = client.get(
        "/api/v1/watchlists/items",
        params={
            "watchlist_id": watchlist_id,
            "has_alert": True,
            "alert_severity": "critical",
            "alert_rule_id": seeded["rule_id"],
            "include_alert_summary": True,
        },
    )
    assert alert_items.status_code == 200, alert_items.text
    body = alert_items.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == seeded["newer_id"]
    assert body["items"][0]["alert_summary"] == {
        "total": 1,
        "unread": 1,
        "read": 0,
        "dismissed": 0,
        "highest_severity": "critical",
        "latest_alert_id": seeded["alert_id"],
        "latest_alert_status": "unread",
        "latest_alert_created_at": body["items"][0]["alert_summary"]["latest_alert_created_at"],
        "latest_matched_text": "CVE-2026-4242",
        "rule_ids": [seeded["rule_id"]],
        "severities": ["critical"],
    }

    quiet_items = client.get(
        "/api/v1/watchlists/items",
        params={"watchlist_id": watchlist_id, "has_alert": False},
    )
    assert quiet_items.status_code == 200, quiet_items.text
    assert [item["id"] for item in quiet_items.json()["items"]] == [seeded["older_id"]]


def test_items_endpoint_rejects_unsupported_triage_sort_and_preserves_static_counts_route(client_with_user):
    client = client_with_user
    seeded = _seed_items_and_alert(client)
    watchlist_id = int(seeded["watchlist"]["id"])

    invalid_sort = client.get(
        "/api/v1/watchlists/items",
        params={"watchlist_id": watchlist_id, "sort": "novelty_desc"},
    )
    assert invalid_sort.status_code == 422

    counts = client.get(
        "/api/v1/watchlists/items/smart-counts",
        params={"watchlist_id": watchlist_id, "alert_severity": "critical"},
    )
    assert counts.status_code == 200, counts.text
    assert counts.json()["all"] == 1
