from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.watchlists import _row_to_output_preset
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
        "first_source_id": int(first_source["id"]),
        "second_source_id": int(second_source["id"]),
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


def test_items_batch_update_endpoint_handles_ids_scope_and_caps(client_with_user):
    client = client_with_user
    seeded = _seed_items_and_alert(client)
    watchlist_id = int(seeded["watchlist"]["id"])
    other_watchlist = _create_watchlist(client)
    outsider_source = _create_source(client, watchlist_id=int(other_watchlist["id"]), label="outside")
    outsider_job = _create_job(
        client,
        watchlist_id=int(other_watchlist["id"]),
        source_ids=[int(outsider_source["id"])],
    )
    db = WatchlistsDatabase.for_user(779)
    outsider_run = db.create_run(int(outsider_job["id"]), status="finished")
    outsider = db.record_scraped_item(
        run_id=int(outsider_run.id),
        job_id=int(outsider_job["id"]),
        source_id=int(outsider_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/outside/item",
        title="Outside item",
        summary="Outside summary",
        content="Outside content",
        published_at="2026-05-15T09:00:00+00:00",
        tags=["outside"],
        status="ingested",
    )

    by_ids = client.post(
        "/api/v1/watchlists/items/batch-update",
        json={
            "watchlist_id": watchlist_id,
            "item_ids": [seeded["older_id"], seeded["newer_id"], int(outsider.id), 999999],
            "reviewed": True,
            "queued_for_briefing": True,
        },
    )

    assert by_ids.status_code == 200, by_ids.text
    body = by_ids.json()
    assert body["matched"] == 2
    assert body["changed"] == 2
    assert body["failed"] == 2
    assert body["changed_ids"] == [seeded["older_id"], seeded["newer_id"]]
    assert body["failed_ids"] == [int(outsider.id), 999999]
    assert body["capped"] is False
    assert body["exhausted"] is True
    assert client.get(f"/api/v1/watchlists/items/{seeded['older_id']}").json()["reviewed"] is True
    assert client.get(f"/api/v1/watchlists/items/{int(outsider.id)}").json()["queued_for_briefing"] is False

    client.patch(f"/api/v1/watchlists/items/{seeded['older_id']}", json={"reviewed": False})
    client.patch(f"/api/v1/watchlists/items/{seeded['newer_id']}", json={"reviewed": False})
    by_scope = client.post(
        "/api/v1/watchlists/items/batch-update",
        json={
            "watchlist_id": watchlist_id,
            "scope": {"status": "ingested", "reviewed": False},
            "reviewed": True,
            "limit": 1,
        },
    )

    assert by_scope.status_code == 200, by_scope.text
    scope_body = by_scope.json()
    assert scope_body["matched"] == 1
    assert scope_body["changed"] == 1
    assert scope_body["capped"] is True
    assert scope_body["exhausted"] is False


def test_item_saved_views_endpoint_crud_and_validation(client_with_user):
    client = client_with_user
    seeded = _seed_items_and_alert(client)
    watchlist_id = int(seeded["watchlist"]["id"])
    other_watchlist = _create_watchlist(client)
    outside_source = _create_source(client, watchlist_id=int(other_watchlist["id"]), label="outside-view")

    created = client.post(
        f"/api/v1/watchlists/{watchlist_id}/item-views",
        json={
            "name": "Critical unread",
            "filters": {
                "source_id": seeded["first_source_id"],
                "smart_filter": "unread",
                "alert_severity": "critical",
            },
            "sort": "unread_first",
            "is_default": True,
        },
    )

    assert created.status_code == 201, created.text
    view = created.json()
    assert view["name"] == "Critical unread"
    assert view["filters"]["smart_filter"] == "unread"
    assert view["sort"] == "unread_first"
    assert view["is_default"] is True

    listed = client.get(f"/api/v1/watchlists/{watchlist_id}/item-views")
    assert listed.status_code == 200, listed.text
    assert [row["id"] for row in listed.json()["items"]] == [view["id"]]

    updated = client.patch(
        f"/api/v1/watchlists/{watchlist_id}/item-views/{view['id']}",
        json={"name": "Critical queue", "sort": "created_asc"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Critical queue"
    assert updated.json()["sort"] == "created_asc"

    bad_sort = client.post(
        f"/api/v1/watchlists/{watchlist_id}/item-views",
        json={"name": "Bad sort", "filters": {}, "sort": "novelty_desc"},
    )
    assert bad_sort.status_code == 422

    bad_source = client.post(
        f"/api/v1/watchlists/{watchlist_id}/item-views",
        json={
            "name": "Bad source",
            "filters": {"source_id": int(outside_source["id"])},
            "sort": "created_desc",
        },
    )
    assert bad_source.status_code == 400
    assert bad_source.json()["detail"] == "source_not_in_watchlist"

    deleted = client.delete(f"/api/v1/watchlists/{watchlist_id}/item-views/{view['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/watchlists/{watchlist_id}/item-views").json()["items"] == []


def test_output_presets_endpoint_crud_and_apply(client_with_user):
    client = client_with_user

    created = client.post(
        "/api/v1/watchlists/job-output-presets",
        json={
            "name": "Daily newsletter",
            "description": "HTML email plus audio",
            "output_prefs": {
                "template": {"default_name": "newsletter_html", "default_format": "html"},
                "deliveries": {"email": {"enabled": True, "recipients": ["ops@example.com"]}},
                "generate_audio": True,
                "audio_voice": "nova",
            },
            "is_default": True,
        },
    )

    assert created.status_code == 201, created.text
    preset = created.json()
    assert preset["name"] == "Daily newsletter"
    assert preset["description"] == "HTML email plus audio"
    assert preset["output_prefs"]["audio_voice"] == "nova"
    assert preset["is_default"] is True

    listed = client.get("/api/v1/watchlists/job-output-presets")
    assert listed.status_code == 200, listed.text
    assert [row["id"] for row in listed.json()["items"]] == [preset["id"]]

    applied = client.post(
        f"/api/v1/watchlists/job-output-presets/{preset['id']}/apply",
        json={
            "base_output_prefs": {
                "template": {"default_name": "old", "experimental_renderer": "keep"},
                "deliveries": {
                    "chatbook": {"enabled": True, "title": "Old"},
                    "webhook": {"url": "https://hooks.example.com/watchlists"},
                },
                "generate_audio": False,
                "raw_advanced": {"preserve": True},
            }
        },
    )
    assert applied.status_code == 200, applied.text
    applied_prefs = applied.json()["output_prefs"]
    assert applied_prefs["template"] == {
        "default_name": "newsletter_html",
        "default_format": "html",
        "experimental_renderer": "keep",
    }
    assert applied_prefs["deliveries"] == {
        "email": {"enabled": True, "recipients": ["ops@example.com"]},
        "webhook": {"url": "https://hooks.example.com/watchlists"},
    }
    assert applied_prefs["audio_voice"] == "nova"
    assert applied_prefs["raw_advanced"] == {"preserve": True}

    updated = client.patch(
        f"/api/v1/watchlists/job-output-presets/{preset['id']}",
        json={
            "name": "Daily executive newsletter",
            "description": None,
            "output_prefs": {"generate_audio": False},
            "is_default": False,
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Daily executive newsletter"
    assert updated.json()["description"] is None
    assert updated.json()["output_prefs"] == {"generate_audio": False}
    assert updated.json()["is_default"] is False

    missing = client.post(
        "/api/v1/watchlists/job-output-presets/999999/apply",
        json={"base_output_prefs": {}},
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "output_preset_not_found"

    deleted = client.delete(f"/api/v1/watchlists/job-output-presets/{preset['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/v1/watchlists/job-output-presets").json()["items"] == []


def test_output_preset_apply_rejects_null_base_output_prefs(client_with_user):
    client = client_with_user

    created = client.post(
        "/api/v1/watchlists/job-output-presets",
        json={
            "name": "Null apply guard",
            "output_prefs": {"generate_audio": True},
        },
    )
    assert created.status_code == 201, created.text

    response = client.post(
        f"/api/v1/watchlists/job-output-presets/{created.json()['id']}/apply",
        json={"base_output_prefs": None},
    )

    assert response.status_code == 422


def test_output_preset_projection_rejects_corrupt_prefs():
    row = SimpleNamespace(
        id=1,
        name="Corrupt preset",
        description=None,
        output_prefs_json="{not-json",
        is_default=0,
        created_at="2026-05-23T00:00:00Z",
        updated_at="2026-05-23T00:00:00Z",
    )

    with pytest.raises(json.JSONDecodeError):
        _row_to_output_preset(row)
