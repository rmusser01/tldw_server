import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from importlib import import_module

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=555, username="wluser", email=None, is_active=True)

    # Route user DB base dir into an isolated temp directory per test.
    base_dir = tmp_path / "test_user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    # Keep this suite focused on watchlists API behavior; seeding behavior is covered separately.
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def client_with_mutable_user(monkeypatch, tmp_path):
    user_state: dict[str, int] = {"id": 555}

    async def override_user():
        current_id = int(user_state["id"])
        return User(id=current_id, username=f"wluser{current_id}", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_mutable"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLISTS_SEED_OUTPUT_TEMPLATES", "false")

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, user_state
    app.dependency_overrides.clear()


def test_sources_crud_and_tags(client_with_user):

    c = client_with_user

    # Create groups for source membership updates
    r = c.post("/api/v1/watchlists/groups", json={"name": "Group A", "description": "A"})
    assert r.status_code == 200, r.text
    group_a = r.json()
    r = c.post("/api/v1/watchlists/groups", json={"name": "Group B", "description": "B"})
    assert r.status_code == 200, r.text
    group_b = r.json()

    # Create source with tags
    body = {
        "name": "Example RSS",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "tags": ["News", "Tech"],
        "settings": {"top_n": 10},
        "group_ids": [group_a["id"]],
    }
    r = c.post("/api/v1/watchlists/sources", json=body)
    assert r.status_code == 200, r.text
    src = r.json()
    sid = src["id"]
    assert set(src.get("tags", [])) == {"news", "tech"}

    # List sources filtered by tag
    r = c.get("/api/v1/watchlists/sources", params={"tags": ["tech"]})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1
    assert data["pagination"]["total"] >= 1
    assert data["pagination"]["limit"] == 50
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["has_more"] is False
    assert data["has_more"] is False
    assert data["next_offset"] is None
    assert any(it["id"] == sid for it in data["items"])

    # Get source
    r = c.get(f"/api/v1/watchlists/sources/{sid}")
    assert r.status_code == 200

    # Update source: replace tags
    r = c.patch(f"/api/v1/watchlists/sources/{sid}", json={"tags": ["updates", "tech"]})
    assert r.status_code == 200
    up = r.json()
    assert set(up.get("tags", [])) == {"updates", "tech"}

    # Update source: replace group membership
    r = c.patch(f"/api/v1/watchlists/sources/{sid}", json={"group_ids": [group_b["id"]]})
    assert r.status_code == 200
    db = WatchlistsDatabase.for_user(555)
    in_group_b = db.list_sources_by_group_ids([int(group_b["id"])])
    assert any(src.id == sid for src in in_group_b)
    in_group_a = db.list_sources_by_group_ids([int(group_a["id"])])
    assert not any(src.id == sid for src in in_group_a)

    # Tags list
    r = c.get("/api/v1/watchlists/tags")
    assert r.status_code == 200
    tags_payload = r.json()
    tags = tags_payload.get("items", [])
    pagination = tags_payload["pagination"]
    assert pagination["total"] >= 2
    assert pagination["limit"] == 50
    assert pagination["offset"] == 0
    assert tags_payload["has_more"] == pagination["has_more"]
    assert tags_payload["next_offset"] == pagination["next_offset"]
    names = {t["name"] for t in tags}
    assert {"news", "tech", "updates"}.issubset(names)

    # Delete source
    r = c.delete(f"/api/v1/watchlists/sources/{sid}")
    assert r.status_code == 200
    delete_payload = r.json()
    assert delete_payload["success"] is True
    assert int(delete_payload["source_id"]) == sid
    assert int(delete_payload["restore_window_seconds"]) >= 1
    assert delete_payload.get("restore_expires_at")

    r = c.get(f"/api/v1/watchlists/sources/{sid}")
    assert r.status_code == 404

    # Restore source
    r = c.post(f"/api/v1/watchlists/sources/{sid}/restore")
    assert r.status_code == 200, r.text
    restored = r.json()
    assert int(restored["id"]) == sid
    assert set(restored.get("tags", [])) == {"updates", "tech"}
    assert restored.get("group_ids") == [group_b["id"]]


def test_sources_test_endpoint(client_with_user, monkeypatch):

    monkeypatch.setenv("TEST_MODE", "1")
    c = client_with_user
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Example Site",
            "url": "https://example.com/",
            "source_type": "site",
            "settings": {
                "scrape_rules": {
                    "list_url": "https://example.com/",
                    "limit": 3,
                }
            },
        },
    )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/sources/{sid}/test", params={"limit": 2})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] == data["total"]
    assert data["filtered"] == 0
    assert all(it["source_id"] == sid for it in data["items"])


def test_sources_test_draft_endpoint_create_mode(client_with_user, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    c = client_with_user

    r = c.post(
        "/api/v1/watchlists/sources/test",
        params={"limit": 3},
        json={
            "url": "https://example.com/draft-source",
            "source_type": "site",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] == data["total"]
    assert data["filtered"] == 0
    assert all(int(it["source_id"]) == 0 for it in data["items"])
    assert all(str(it["source_type"]) == "site" for it in data["items"])


def test_sources_test_draft_endpoint_forum_disabled(client_with_user, monkeypatch):
    monkeypatch.delenv("WATCHLIST_FORUMS_ENABLED", raising=False)
    c = client_with_user

    r = c.post(
        "/api/v1/watchlists/sources/test",
        json={
            "url": "https://forum.example.com/thread/1",
            "source_type": "forum",
        },
    )
    assert r.status_code == 400, r.text
    assert "forum_sources_disabled" in r.text


def test_sources_check_now_endpoint_triggers_runs_and_items(client_with_user, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    c = client_with_user
    source_resp = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Check RSS",
            "url": "https://example.com/check-rss.xml",
            "source_type": "rss",
        },
    )
    assert source_resp.status_code == 200, source_resp.text
    source_id = int(source_resp.json()["id"])

    check_resp = c.post(
        "/api/v1/watchlists/sources/check-now",
        json={"source_ids": [source_id, source_id]},
    )
    assert check_resp.status_code == 200, check_resp.text
    payload = check_resp.json()
    assert payload["total"] == 1
    assert payload["success"] == 1
    assert payload["failed"] == 0
    assert payload["items"][0]["source_id"] == source_id
    assert payload["items"][0]["status"] == "ok"
    assert payload["items"][0].get("last_scraped_at")
    run_id = payload["items"][0].get("run_id")
    assert isinstance(run_id, int) and run_id > 0

    runs_resp = c.get("/api/v1/watchlists/runs", params={"page": 1, "size": 20})
    assert runs_resp.status_code == 200, runs_resp.text
    runs_payload = runs_resp.json()
    assert runs_payload["pagination"]["total"] >= 1
    assert runs_payload["pagination"]["limit"] == 20
    assert runs_payload["pagination"]["offset"] == 0
    assert runs_payload["has_more"] == runs_payload["pagination"]["has_more"]
    assert runs_payload["next_offset"] == runs_payload["pagination"]["next_offset"]
    run_ids = [int(run["id"]) for run in runs_payload.get("items", [])]
    assert run_id in run_ids

    items_resp = c.get("/api/v1/watchlists/items", params={"source_id": source_id, "page": 1, "size": 20})
    assert items_resp.status_code == 200, items_resp.text
    items_payload = items_resp.json()
    assert int(items_payload.get("total", 0)) >= 1
    assert items_payload["has_more"] == items_payload["pagination"]["has_more"]
    assert items_payload["next_offset"] == items_payload["pagination"]["next_offset"]
    assert any(int(item.get("source_id", -1)) == source_id for item in items_payload.get("items", []))


def test_sources_check_now_reports_run_errors_and_missing_sources(client_with_user, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists as watchlists_endpoints

    async def _raise_run_error(*_args, **_kwargs):
        raise RuntimeError("run_failed")

    monkeypatch.setattr(watchlists_endpoints, "run_watchlist_job", _raise_run_error)

    c = client_with_user
    src_resp = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Broken RSS",
            "url": "https://example.com/broken-rss.xml",
            "source_type": "rss",
        },
    )
    assert src_resp.status_code == 200, src_resp.text
    source_id = int(src_resp.json()["id"])

    check_resp = c.post(
        "/api/v1/watchlists/sources/check-now",
        json={"source_ids": [source_id, 999999]},
    )
    assert check_resp.status_code == 200, check_resp.text
    payload = check_resp.json()
    assert payload["total"] == 2
    assert payload["success"] == 0
    assert payload["failed"] == 2

    by_source = {int(item["source_id"]): item for item in payload["items"]}
    assert by_source[source_id]["status"] == "error"
    assert by_source[source_id]["detail"] == "run_trigger_failed"
    assert by_source[999999]["status"] == "not_found"


def test_source_group_validation_and_idempotent_create(client_with_user):

    c = client_with_user

    r = c.post("/api/v1/watchlists/groups", json={"name": "Group C", "description": "C"})
    assert r.status_code == 200, r.text
    group_c = r.json()
    r = c.post("/api/v1/watchlists/groups", json={"name": "Group D", "description": "D"})
    assert r.status_code == 200, r.text
    group_d = r.json()

    # Invalid group id on create should fail
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Bad Group Source",
            "url": "https://example.com/invalid-group.xml",
            "source_type": "rss",
            "group_ids": [999999],
        },
    )
    assert r.status_code == 400
    assert "group_not_found" in r.text

    # Create source in Group C
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Group Source",
            "url": "https://example.com/group-source.xml",
            "source_type": "rss",
            "group_ids": [group_c["id"]],
        },
    )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]

    # Idempotent create should replace group membership when group_ids provided
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Group Source",
            "url": "https://example.com/group-source.xml",
            "source_type": "rss",
            "group_ids": [group_d["id"]],
        },
    )
    assert r.status_code == 200, r.text

    db = WatchlistsDatabase.for_user(555)
    in_group_d = db.list_sources_by_group_ids([int(group_d["id"])])
    assert any(src.id == sid for src in in_group_d)
    in_group_c = db.list_sources_by_group_ids([int(group_c["id"])])
    assert not any(src.id == sid for src in in_group_c)

    # Invalid group id on update should fail
    r = c.patch(f"/api/v1/watchlists/sources/{sid}", json={"group_ids": [999998]})
    assert r.status_code == 400
    assert "group_not_found" in r.text


def test_source_restore_expired_window_returns_gone(client_with_user):
    c = client_with_user

    create_resp = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Expiring Source",
            "url": "https://example.com/expiring-source.xml",
            "source_type": "rss",
            "tags": ["expiring"],
            "group_ids": [],
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    sid = int(create_resp.json()["id"])

    delete_resp = c.delete(f"/api/v1/watchlists/sources/{sid}")
    assert delete_resp.status_code == 200, delete_resp.text

    db = WatchlistsDatabase.for_user(555)
    db.backend.execute(
        "UPDATE deleted_sources SET expires_at = ? WHERE user_id = ? AND source_id = ?",
        ("2000-01-01T00:00:00+00:00", "555", sid),
    )

    restore_resp = c.post(f"/api/v1/watchlists/sources/{sid}/restore")
    assert restore_resp.status_code == 410, restore_resp.text
    assert restore_resp.json().get("detail") == "source_restore_expired"

    # Expired tombstones are purged on first restore attempt.
    restore_again_resp = c.post(f"/api/v1/watchlists/sources/{sid}/restore")
    assert restore_again_resp.status_code == 404, restore_again_resp.text


def test_restore_requires_original_user_context(client_with_mutable_user):
    c, user_state = client_with_mutable_user

    create_resp = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Scoped Source",
            "url": "https://example.com/scoped-source.xml",
            "source_type": "rss",
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    sid = int(create_resp.json()["id"])

    delete_resp = c.delete(f"/api/v1/watchlists/sources/{sid}")
    assert delete_resp.status_code == 200, delete_resp.text

    user_state["id"] = 777
    unauthorized_restore = c.post(f"/api/v1/watchlists/sources/{sid}/restore")
    assert unauthorized_restore.status_code == 404, unauthorized_restore.text
    assert unauthorized_restore.json().get("detail") == "source_restore_not_found"

    user_state["id"] = 555
    authorized_restore = c.post(f"/api/v1/watchlists/sources/{sid}/restore")
    assert authorized_restore.status_code == 200, authorized_restore.text
    assert int(authorized_restore.json()["id"]) == sid


def test_job_delete_and_restore_fidelity(client_with_user):
    c = client_with_user

    source_resp = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Source For Job Restore",
            "url": "https://example.com/job-restore-source.xml",
            "source_type": "rss",
            "tags": ["restore-tag"],
        },
    )
    assert source_resp.status_code == 200, source_resp.text
    source_id = int(source_resp.json()["id"])

    group_resp = c.post(
        "/api/v1/watchlists/groups",
        json={"name": "Restore Group", "description": "for restore"},
    )
    assert group_resp.status_code == 200, group_resp.text
    group_id = int(group_resp.json()["id"])

    job_body = {
        "name": "Restore Fidelity Job",
        "description": "Ensure restored jobs preserve payload fields",
        "scope": {"sources": [source_id], "groups": [group_id], "tags": ["restore-tag"]},
        "schedule_expr": "*/15 * * * *",
        "timezone": "UTC",
        "active": False,
        "max_concurrency": 2,
        "per_host_delay_ms": 500,
        "retry_policy": {"retries": 3, "backoff_seconds": 10},
        "output_prefs": {"template_name": "daily-brief", "delivery_config": {"create_chatbook": True}},
        "job_filters": {"filters": [{"type": "keyword", "action": "include", "value": {"keywords": ["ai"]}}]},
    }
    create_job_resp = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert create_job_resp.status_code == 200, create_job_resp.text
    created_job = create_job_resp.json()
    job_id = int(created_job["id"])

    delete_job_resp = c.delete(f"/api/v1/watchlists/jobs/{job_id}")
    assert delete_job_resp.status_code == 200, delete_job_resp.text
    delete_payload = delete_job_resp.json()
    assert delete_payload["success"] is True
    assert int(delete_payload["job_id"]) == job_id
    assert int(delete_payload["restore_window_seconds"]) >= 1
    assert delete_payload.get("restore_expires_at")

    get_deleted = c.get(f"/api/v1/watchlists/jobs/{job_id}")
    assert get_deleted.status_code == 404, get_deleted.text

    restore_job_resp = c.post(f"/api/v1/watchlists/jobs/{job_id}/restore")
    assert restore_job_resp.status_code == 200, restore_job_resp.text
    restored_job = restore_job_resp.json()

    assert int(restored_job["id"]) == job_id
    assert restored_job["name"] == job_body["name"]
    assert restored_job["description"] == job_body["description"]
    assert restored_job["scope"] == job_body["scope"]
    assert restored_job["schedule_expr"] == job_body["schedule_expr"]
    assert restored_job["timezone"] == job_body["timezone"]
    assert restored_job["active"] is job_body["active"]
    assert restored_job["max_concurrency"] == job_body["max_concurrency"]
    assert restored_job["per_host_delay_ms"] == job_body["per_host_delay_ms"]
    assert restored_job["retry_policy"] == job_body["retry_policy"]
    assert restored_job["output_prefs"] == job_body["output_prefs"]
    restored_filters = restored_job.get("job_filters") or {}
    assert restored_filters.get("require_include") is None
    assert isinstance(restored_filters.get("filters"), list) and len(restored_filters["filters"]) == 1
    assert restored_filters["filters"][0]["type"] == "keyword"
    assert restored_filters["filters"][0]["action"] == "include"
    assert restored_filters["filters"][0]["value"] == {"keywords": ["ai"]}


def test_bulk_sources_and_groups_and_jobs(client_with_user):

    c = client_with_user

    # Bulk create two sources
    payload = {
        "sources": [
            {"name": "Site A", "url": "https://a.example.com/", "source_type": "site", "tags": ["alpha"]},
            {"name": "RSS B", "url": "https://b.example.com/feed", "source_type": "rss", "tags": ["beta"]},
        ]
    }
    r = c.post("/api/v1/watchlists/sources/bulk", json=payload)
    assert r.status_code == 200, r.text
    lst = r.json()
    assert lst["total"] == 2

    # Groups
    r = c.post("/api/v1/watchlists/groups", json={"name": "Top", "description": "Root"})
    assert r.status_code == 200
    g = r.json()
    gid = g["id"]
    r = c.get("/api/v1/watchlists/groups")
    assert r.status_code == 200
    groups_payload = r.json()
    assert groups_payload["pagination"]["total"] >= 1
    assert groups_payload["pagination"]["limit"] == 50
    assert groups_payload["pagination"]["offset"] == 0
    assert groups_payload["has_more"] == groups_payload["pagination"]["has_more"]
    assert groups_payload["next_offset"] == groups_payload["pagination"]["next_offset"]
    assert any(x["id"] == gid for x in groups_payload.get("items", []))
    r = c.patch(f"/api/v1/watchlists/groups/{gid}", json={"description": "Updated"})
    assert r.status_code == 200
    assert r.json()["description"] == "Updated"

    # Jobs
    job_body = {
        "name": "Morning Brief",
        "scope": {"tags": ["alpha", "beta"]},
        "schedule_expr": "0 8 * * *",
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job = r.json()
    jid = job["id"]
    r = c.get("/api/v1/watchlists/jobs")
    assert r.status_code == 200
    jobs_payload = r.json()
    assert jobs_payload["pagination"]["total"] >= 1
    assert jobs_payload["pagination"]["limit"] == 50
    assert jobs_payload["pagination"]["offset"] == 0
    assert jobs_payload["has_more"] == jobs_payload["pagination"]["has_more"]
    assert jobs_payload["next_offset"] == jobs_payload["pagination"]["next_offset"]
    assert any(x["id"] == jid for x in jobs_payload.get("items", []))
    r = c.get(f"/api/v1/watchlists/jobs/{jid}")
    assert r.status_code == 200
    # Update job
    r = c.patch(f"/api/v1/watchlists/jobs/{jid}", json={"active": False})
    assert r.status_code == 200
    assert r.json()["active"] is False

    # Trigger run (stub)
    r = c.post(f"/api/v1/watchlists/jobs/{jid}/run")
    assert r.status_code == 200
    run = r.json()
    rid = run["id"]
    r = c.get(f"/api/v1/watchlists/jobs/{jid}/runs")
    assert r.status_code == 200
    assert any(x["id"] == rid for x in r.json().get("items", []))
    r = c.get(f"/api/v1/watchlists/runs/{rid}")
    assert r.status_code == 200


def _assert_watchlists_validation_error(resp, *, rule: str, message_key: str) -> dict:
    assert resp.status_code == 422, resp.text
    payload = resp.json()
    detail = payload.get("detail")
    assert isinstance(detail, dict), payload
    assert detail.get("code") == "watchlists_validation_error"
    assert detail.get("rule") == rule
    assert detail.get("message_key") == message_key
    assert isinstance(detail.get("message"), str) and detail.get("message")
    assert isinstance(detail.get("remediation"), str) and detail.get("remediation")
    return detail


def test_create_job_scope_validation_returns_structured_detail(client_with_user):
    c = client_with_user
    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Invalid Scope Job",
            "scope": {"sources": [], "groups": [], "tags": []},
            "schedule_expr": None,
            "timezone": "UTC",
            "active": True,
        },
    )
    _assert_watchlists_validation_error(
        r,
        rule="scope_required",
        message_key="watchlists:jobs.form.scopeRequired",
    )


def test_create_job_schedule_validation_returns_structured_detail(client_with_user):
    c = client_with_user
    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Too Frequent Job",
            "scope": {"tags": ["alpha"]},
            "schedule_expr": "* * * * *",
            "timezone": "UTC",
            "active": True,
        },
    )
    detail = _assert_watchlists_validation_error(
        r,
        rule="schedule_too_frequent",
        message_key="watchlists:jobs.form.scheduleTooFrequent",
    )
    meta = detail.get("meta")
    assert isinstance(meta, dict)
    assert int(meta.get("minimum_minutes", 0)) >= 1
    assert float(meta.get("detected_interval_minutes", 0)) <= 1.0


def test_create_job_email_validation_returns_structured_detail(client_with_user):
    c = client_with_user
    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Invalid Email Job",
            "scope": {"tags": ["alpha"]},
            "schedule_expr": None,
            "timezone": "UTC",
            "active": True,
            "output_prefs": {"deliveries": {"email": {"recipients": ["valid@example.com", "bad-email", "also bad"]}}},
        },
    )
    detail = _assert_watchlists_validation_error(
        r,
        rule="invalid_email_recipients",
        message_key="watchlists:jobs.form.emailRecipientsInvalidSubmit",
    )
    meta = detail.get("meta")
    assert isinstance(meta, dict)
    assert int(meta.get("count", 0)) == 2
    assert "bad-email" in (meta.get("invalid_recipients") or [])


def test_update_job_email_validation_returns_structured_detail(client_with_user):
    c = client_with_user
    create_resp = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Updatable Job",
            "scope": {"tags": ["alpha"]},
            "schedule_expr": None,
            "timezone": "UTC",
            "active": True,
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = int(create_resp.json()["id"])

    update_resp = c.patch(
        f"/api/v1/watchlists/jobs/{job_id}",
        json={"output_prefs": {"deliveries": {"email": {"recipients": ["not-an-email"]}}}},
    )
    detail = _assert_watchlists_validation_error(
        update_resp,
        rule="invalid_email_recipients",
        message_key="watchlists:jobs.form.emailRecipientsInvalidSubmit",
    )
    meta = detail.get("meta")
    assert isinstance(meta, dict)
    assert int(meta.get("count", 0)) == 1
    assert "not-an-email" in (meta.get("invalid_recipients") or [])


def test_cancel_run_endpoint_marks_running_run_cancelled(client_with_user):
    c = client_with_user
    db = WatchlistsDatabase.for_user(555)
    job = db.create_job(
        name="Cancelable Job",
        description=None,
        scope_json=json.dumps({"sources": []}),
        schedule_expr=None,
        schedule_timezone=None,
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )
    run = db.create_run(job_id=job.id, status="running")

    cancel_resp = c.post(f"/api/v1/watchlists/runs/{run.id}/cancel")
    assert cancel_resp.status_code == 200, cancel_resp.text
    payload = cancel_resp.json()
    assert payload["run_id"] == run.id
    assert payload["cancelled"] is True
    assert payload["status"] == "cancelled"

    run_resp = c.get(f"/api/v1/watchlists/runs/{run.id}")
    assert run_resp.status_code == 200, run_resp.text
    assert run_resp.json()["status"] == "cancelled"


def test_cancel_run_endpoint_rejects_terminal_runs(client_with_user):
    c = client_with_user
    db = WatchlistsDatabase.for_user(555)
    job = db.create_job(
        name="Terminal Job",
        description=None,
        scope_json=json.dumps({"sources": []}),
        schedule_expr=None,
        schedule_timezone=None,
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )
    run = db.create_run(job_id=job.id, status="completed")

    cancel_resp = c.post(f"/api/v1/watchlists/runs/{run.id}/cancel")
    assert cancel_resp.status_code == 200, cancel_resp.text
    payload = cancel_resp.json()
    assert payload["run_id"] == run.id
    assert payload["cancelled"] is False
    assert payload["status"] == "completed"
    assert payload["message"] == "run_not_cancellable"


def test_forum_sources_feature_flag(client_with_user, monkeypatch):
    c = client_with_user

    monkeypatch.delenv("WATCHLIST_FORUMS_ENABLED", raising=False)
    r = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Forum Source", "url": "https://forum.example.com/", "source_type": "forum"},
    )
    assert r.status_code == 400
    assert "forum_sources_disabled" in r.text

    monkeypatch.setenv("WATCHLIST_FORUMS_ENABLED", "1")
    r = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Forum Source", "url": "https://forum.example.com/", "source_type": "forum"},
    )
    assert r.status_code == 200, r.text


def test_watchlists_run_stream_ws(client_with_user, tmp_path):
    from tldw_Server_API.app.core.AuthNZ.jwt_service import create_access_token

    db = WatchlistsDatabase.for_user(555)
    job = db.create_job(
        name="Job",
        description=None,
        scope_json=json.dumps({"sources": []}),
        schedule_expr=None,
        schedule_timezone=None,
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )
    run = db.create_run(job_id=job.id, status="running")
    log_path = tmp_path / "run.log"
    log_path.write_text("first line\n", encoding="utf-8")
    db.update_run(run.id, stats_json=json.dumps({"items_found": 1}), log_path=str(log_path))

    token = create_access_token(555, "wluser", "admin")
    with client_with_user.websocket_connect(f"/api/v1/watchlists/runs/{run.id}/stream?token={token}") as ws:
        message = ws.receive_json()
        assert message["type"] == "snapshot"
        assert message["run"]["id"] == run.id
        assert "log_tail" in message


def test_items_and_outputs_flow(client_with_user, monkeypatch):

    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WATCHLIST_OUTPUT_DEFAULT_TTL_SECONDS", "0")
    monkeypatch.setenv("WATCHLIST_OUTPUT_TEMP_TTL_SECONDS", "90")

    # Create RSS source
    src_body = {
        "name": "Daily Feed",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "tags": ["daily"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    # Job covering tag
    job_body = {
        "name": "Daily Digest",
        "scope": {"tags": ["daily"]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    # Trigger run
    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run = r.json()
    run_id = run["id"]

    # List items
    r = c.get("/api/v1/watchlists/items", params={"run_id": run_id})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["pagination"]["total"] >= 1
    assert data["pagination"]["limit"] == 50
    assert data["pagination"]["offset"] == 0
    assert data["has_more"] == data["pagination"]["has_more"]
    assert data["next_offset"] == data["pagination"]["next_offset"]
    first_item = data["items"][0]
    item_id = first_item["id"]
    assert first_item["status"] in {"ingested", "error", "duplicate"}
    assert "content" in first_item
    assert first_item.get("queued_for_briefing") is False

    # Item detail should also expose content for feed-reader style rendering.
    r = c.get(f"/api/v1/watchlists/items/{item_id}")
    assert r.status_code == 200, r.text
    item_detail = r.json()
    assert "content" in item_detail

    # Search should match content text as well as title/summary.
    if item_detail.get("content"):
        term = str(item_detail["content"]).split()[0]
        r = c.get("/api/v1/watchlists/items", params={"run_id": run_id, "q": term})
        assert r.status_code == 200, r.text
        matches = r.json().get("items", [])
        assert any(it["id"] == item_id for it in matches)

    # Mark reviewed
    r = c.patch(f"/api/v1/watchlists/items/{item_id}", json={"reviewed": True})
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["reviewed"] is True

    # Queue item for briefing-driven report generation.
    r = c.patch(
        f"/api/v1/watchlists/items/{item_id}",
        json={"queued_for_briefing": True},
    )
    assert r.status_code == 200, r.text
    updated = r.json()
    assert updated["queued_for_briefing"] is True

    # Filter reviewed items
    r = c.get("/api/v1/watchlists/items", params={"run_id": run_id, "reviewed": True})
    assert r.status_code == 200, r.text
    filt = r.json()
    assert any(it["id"] == item_id for it in filt["items"])

    # Filter queued items
    r = c.get(
        "/api/v1/watchlists/items",
        params={"run_id": run_id, "queued_for_briefing": True},
    )
    assert r.status_code == 200, r.text
    queued = r.json()
    assert any(it["id"] == item_id for it in queued["items"])

    # Smart counts endpoint should aggregate the same scoped item set.
    r = c.get("/api/v1/watchlists/items/smart-counts", params={"run_id": run_id})
    assert r.status_code == 200, r.text
    counts = r.json()
    for field in ("all", "today", "today_unread", "unread", "reviewed", "queued"):
        assert field in counts
        assert isinstance(counts[field], int)
        assert counts[field] >= 0
    assert counts["all"] >= 1
    assert counts["reviewed"] >= 1
    assert counts["queued"] >= 1
    assert counts["today"] >= counts["today_unread"]

    # Search-scoped smart counts should return zero for a guaranteed miss.
    r = c.get(
        "/api/v1/watchlists/items/smart-counts",
        params={"run_id": run_id, "q": "__watchlists_unmatched_search_token__"},
    )
    assert r.status_code == 200, r.text
    missing = r.json()
    assert missing["all"] == 0
    assert missing["today"] == 0
    assert missing["today_unread"] == 0
    assert missing["unread"] == 0
    assert missing["reviewed"] == 0
    assert missing["queued"] == 0

    # Generate output
    out_payload = {
        "run_id": run_id,
        "title": "Daily Briefing",
        "format": "md",
        "retention_seconds": 3600,
    }
    r = c.post("/api/v1/watchlists/outputs", json=out_payload)
    assert r.status_code == 200, r.text
    output = r.json()
    output_id = output["id"]
    assert "Daily Briefing" in (output.get("content") or "")
    assert output["version"] == 1
    assert output["expires_at"] is not None
    expires_at = datetime.fromisoformat(output["expires_at"])
    assert expires_at > datetime.now(timezone.utc)

    # Generate output with no expiry (retention_seconds=0)
    r = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "retention_seconds": 0})
    assert r.status_code == 200, r.text
    no_expiry = r.json()
    assert no_expiry["version"] == 2
    assert no_expiry["expires_at"] is None

    # Generate a temporary output without specifying title to trigger default naming
    r = c.post("/api/v1/watchlists/outputs", json={"run_id": run_id, "temporary": True})
    assert r.status_code == 200, r.text
    temp_output = r.json()
    assert temp_output["version"] == 3
    assert temp_output["title"].startswith("Daily Digest-Output-3")
    assert temp_output["expires_at"] is not None

    # Outputs listing (should include both)
    r = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert r.status_code == 200, r.text
    outputs = r.json()
    assert outputs["total"] >= 2
    assert outputs["pagination"]["total"] >= 2
    assert outputs["pagination"]["limit"] == 50
    assert outputs["pagination"]["offset"] == 0
    assert outputs["has_more"] == outputs["pagination"]["has_more"]
    assert outputs["next_offset"] == outputs["pagination"]["next_offset"]
    assert any(o["id"] == output_id for o in outputs["items"])

    # Output metadata
    r = c.get(f"/api/v1/watchlists/outputs/{output_id}")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["version"] == 1

    # Download output
    r = c.get(f"/api/v1/watchlists/outputs/{output_id}/download")
    assert r.status_code == 200, r.text

    # Output templates (DB-backed)
    unique_suffix = uuid.uuid4().hex[:8]
    db_template_name = f"db_daily_md_{unique_suffix}"
    db_template_payload = {
        "name": db_template_name,
        "type": "briefing_markdown",
        "format": "md",
        "body": "DB TEMPLATE {{ title }}",
        "description": "DB-backed template",
    }
    r = c.post("/api/v1/outputs/templates", json=db_template_payload)
    assert r.status_code == 200, r.text
    db_template = r.json()
    assert db_template["name"] == db_template_name

    r = c.post(
        "/api/v1/watchlists/outputs", json={"run_id": run_id, "template_name": db_template_name, "temporary": True}
    )
    assert r.status_code == 200, r.text
    db_output = r.json()
    assert db_output["version"] == 4
    assert db_output["metadata"].get("template_source") == "outputs_templates"
    assert db_output["metadata"].get("template_id") == db_template["id"]
    assert "DB TEMPLATE" in (db_output.get("content") or "")
    r = c.delete(f"/api/v1/outputs/templates/{db_template['id']}")
    assert r.status_code == 200, r.text

    # Template management
    legacy_template_name = f"daily_md_{unique_suffix}"
    template_payload = {
        "name": legacy_template_name,
        "format": "md",
        "content": "{{ title }}\\n{% for item in items %}- {{ loop.index }}. {{ item.title }}{% endfor %}",
        "description": "Markdown summary template",
    }
    r = c.post("/api/v1/watchlists/templates", json=template_payload)
    assert r.status_code == 200, r.text
    # Duplicate without overwrite should fail
    r = c.post("/api/v1/watchlists/templates", json=template_payload)
    assert r.status_code == 409

    # List templates
    r = c.get("/api/v1/watchlists/templates")
    assert r.status_code == 200
    templates = r.json()["items"]
    assert any(t["name"] == legacy_template_name for t in templates)

    # Get template detail
    r = c.get(f"/api/v1/watchlists/templates/{legacy_template_name}")
    assert r.status_code == 200
    template_detail = r.json()
    assert "Markdown summary template" in template_detail["description"]

    # Generate output using stored template
    r = c.post(
        "/api/v1/watchlists/outputs", json={"run_id": run_id, "template_name": legacy_template_name, "temporary": True}
    )
    assert r.status_code == 200, r.text
    templated = r.json()
    assert templated["version"] == 5
    assert templated["format"] == "md"
    assert templated["metadata"].get("template_name") == legacy_template_name
    assert templated["metadata"].get("template_source") == "watchlists_templates"
    assert templated["content"].startswith("Daily Digest-Output-5") or "Daily Digest" in templated["content"]

    # When both stores have the same template name, DB-backed outputs template wins.
    collision_payload = {
        "name": legacy_template_name,
        "type": "briefing_markdown",
        "format": "md",
        "body": "DB OVERRIDE {{ title }}",
        "description": "DB-backed collision template",
    }
    r = c.post("/api/v1/outputs/templates", json=collision_payload)
    assert r.status_code == 200, r.text
    colliding_db_template = r.json()

    r = c.post(
        "/api/v1/watchlists/outputs", json={"run_id": run_id, "template_name": legacy_template_name, "temporary": True}
    )
    assert r.status_code == 200, r.text
    collision_output = r.json()
    assert collision_output["version"] == 6
    assert collision_output["metadata"].get("template_source") == "outputs_templates"
    assert collision_output["metadata"].get("template_id") == colliding_db_template["id"]
    assert "DB OVERRIDE" in (collision_output.get("content") or "")

    r = c.delete(f"/api/v1/outputs/templates/{colliding_db_template['id']}")
    assert r.status_code == 200, r.text

    # Delete template and confirm removal
    r = c.delete(f"/api/v1/watchlists/templates/{legacy_template_name}")
    assert r.status_code == 200
    r = c.get(f"/api/v1/watchlists/templates/{legacy_template_name}")
    assert r.status_code == 404
    r = c.get("/api/v1/watchlists/templates")
    assert all(t["name"] != legacy_template_name for t in r.json().get("items", []))


def test_watchlists_outputs_variants_and_ingest(client_with_user, monkeypatch):

    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")

    class DummyTTS:
        async def generate_speech(self, req):  # noqa: ARG002 - signature used by TTS service
            yield b"FAKEAUDIO"

    async def _fake_get_tts_service_v2(*args, **kwargs):  # noqa: ARG002
        return DummyTTS()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.outputs_service.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )

    # Create RSS source
    src_body = {
        "name": "Variants Feed",
        "url": "https://example.com/variants-feed.xml",
        "source_type": "rss",
        "tags": ["variants"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    # Job scoped to source
    job_body = {
        "name": "Variants Digest",
        "scope": {"sources": [source_id]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    # Trigger run
    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    # Create variant templates
    mece_name = f"mece_watchlists_{run_id}"
    mece_payload = {
        "name": mece_name,
        "type": "mece_markdown",
        "format": "md",
        "body": "# MECE\\n{{ items|length }} items",
        "description": "Watchlists MECE template",
        "is_default": False,
    }
    r = c.post("/api/v1/outputs/templates", json=mece_payload)
    assert r.status_code == 200, r.text
    mece_id = r.json()["id"]

    tts_name = f"tts_watchlists_{run_id}"
    tts_payload = {
        "name": tts_name,
        "type": "tts_audio",
        "format": "mp3",
        "body": "Audio briefing for {{ items|length }} items.",
        "description": "Watchlists TTS template",
        "is_default": False,
    }
    r = c.post("/api/v1/outputs/templates", json=tts_payload)
    assert r.status_code == 200, r.text
    tts_id = r.json()["id"]

    # Generate output variants + ingest
    out_payload = {
        "run_id": run_id,
        "title": "Variants Briefing",
        "format": "md",
        "generate_mece": True,
        "mece_template_name": mece_name,
        "generate_tts": True,
        "tts_template_name": tts_name,
        "ingest_to_media_db": True,
    }
    r = c.post("/api/v1/watchlists/outputs", json=out_payload)
    assert r.status_code == 200, r.text
    base_output = r.json()
    assert base_output["media_item_id"] is not None

    r = c.get("/api/v1/watchlists/outputs", params={"run_id": run_id})
    assert r.status_code == 200, r.text
    outputs = r.json()["items"]
    mece_outputs = [o for o in outputs if o.get("type") == "mece_markdown"]
    tts_outputs = [o for o in outputs if o.get("type") == "tts_audio"]
    assert mece_outputs
    assert tts_outputs
    assert all(o.get("media_item_id") for o in mece_outputs)
    assert all(o.get("media_item_id") for o in tts_outputs)

    tts_output = tts_outputs[0]
    assert tts_output.get("format") == "mp3"
    assert tts_output.get("storage_path")
    r = c.get(f"/api/v1/watchlists/outputs/{tts_output['id']}/download")
    assert r.status_code == 200, r.text

    # Cleanup templates
    r = c.delete(f"/api/v1/outputs/templates/{mece_id}")
    assert r.status_code == 200, r.text
    r = c.delete(f"/api/v1/outputs/templates/{tts_id}")
    assert r.status_code == 200, r.text


def test_watchlists_outputs_pagination_excludes_mixed_origin_rows(client_with_user):
    c = client_with_user
    cdb = CollectionsDatabase.for_user(555)
    suffix = uuid.uuid4().hex[:8]
    job_id = int(f"91{suffix[:6]}", 16) % 1_000_000
    run_id = job_id + 1

    wl_old = cdb.create_output_artifact(
        type_="briefing_markdown",
        title=f"wl-old-{suffix}",
        format_="md",
        storage_path=f"wl-old-{suffix}.md",
        metadata_json=json.dumps({"origin": "watchlists"}),
        job_id=job_id,
        run_id=run_id,
    )
    wl_new = cdb.create_output_artifact(
        type_="briefing_markdown",
        title=f"wl-new-{suffix}",
        format_="md",
        storage_path=f"wl-new-{suffix}.md",
        metadata_json=json.dumps({"origin": "watchlists"}),
        job_id=job_id,
        run_id=run_id,
    )
    nw_1 = cdb.create_output_artifact(
        type_="briefing_markdown",
        title=f"non-watch-1-{suffix}",
        format_="md",
        storage_path=f"non-watch-1-{suffix}.md",
        metadata_json=json.dumps({"origin": "outputs"}),
        job_id=job_id,
        run_id=run_id,
    )
    nw_2 = cdb.create_output_artifact(
        type_="briefing_markdown",
        title=f"non-watch-2-{suffix}",
        format_="md",
        storage_path=f"non-watch-2-{suffix}.md",
        metadata_json=json.dumps({"origin": "outputs"}),
        job_id=job_id,
        run_id=run_id,
    )

    r = c.get("/api/v1/watchlists/outputs", params={"job_id": job_id, "page": 1, "size": 1})
    assert r.status_code == 200, r.text
    page1 = r.json()
    assert page1["total"] == 2
    assert page1["pagination"]["total"] == 2
    assert page1["pagination"]["limit"] == 1
    assert page1["pagination"]["offset"] == 0
    assert page1["pagination"]["has_more"] is True
    assert page1["has_more"] is True
    assert page1["next_offset"] == page1["pagination"]["next_offset"]
    assert len(page1["items"]) == 1
    assert page1["items"][0]["id"] == wl_new.id

    r = c.get("/api/v1/watchlists/outputs", params={"job_id": job_id, "page": 2, "size": 1})
    assert r.status_code == 200, r.text
    page2 = r.json()
    assert page2["total"] == 2
    assert page2["pagination"]["total"] == 2
    assert page2["pagination"]["limit"] == 1
    assert page2["pagination"]["offset"] == 1
    assert page2["pagination"]["has_more"] is False
    assert page2["has_more"] is False
    assert page2["next_offset"] is None
    assert len(page2["items"]) == 1
    assert page2["items"][0]["id"] == wl_old.id

    returned_ids = {page1["items"][0]["id"], page2["items"][0]["id"]}
    assert returned_ids == {wl_old.id, wl_new.id}
    assert nw_1.id not in returned_ids
    assert nw_2.id not in returned_ids


def test_preview_site_sources_returns_items(client_with_user, monkeypatch):

    c = client_with_user
    monkeypatch.delenv("TEST_MODE", raising=False)

    async def _fake_fetch(base_url, rules, *, tenant_id="default", timeout=10.0):
        return [{"title": "Stub Item", "url": "https://example.com/x", "summary": "Stub summary"}]

    from tldw_Server_API.app.api.v1.endpoints import watchlists as watchlists_endpoints

    monkeypatch.setattr(watchlists_endpoints, "fetch_site_items_with_rules", _fake_fetch)

    # Create site source
    src_body = {
        "name": "Preview Site",
        "url": "https://example.com/",
        "source_type": "site",
        "tags": ["preview"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    # Job scoped to source
    job_body = {
        "name": "Preview Job",
        "scope": {"sources": [source_id]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/preview", params={"limit": 10, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1


def test_output_deliveries_email_and_chatbook(client_with_user, monkeypatch, tmp_path):

    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("EMAIL_PROVIDER", "mock")

    src_body = {
        "name": "Daily Feed",
        "url": "https://example.com/feed.xml",
        "source_type": "rss",
        "tags": ["daily"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text

    job_body = {
        "name": "Daily Digest",
        "scope": {"tags": ["daily"]},
        "output_prefs": {
            "retention": {"default_seconds": 7200, "temporary_seconds": 600},
            "deliveries": {
                "email": {"recipients": ["digest@example.com"], "attach_file": False},
                "chatbook": {"enabled": True, "metadata": {"category": "digest"}},
            },
        },
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    payload = {
        "run_id": run_id,
        "title": "Digest",
        "deliveries": {
            "email": {"subject": "Daily Digest", "recipients": ["alt@example.com"], "attach_file": True},
            "chatbook": {"title": "Digest Document", "description": "Auto", "metadata": {"origin": "test"}},
        },
    }
    r = c.post("/api/v1/watchlists/outputs", json=payload)
    assert r.status_code == 200, r.text
    output = r.json()

    deliveries = output.get("metadata", {}).get("deliveries", [])
    assert len(deliveries) == 2
    channels = {d["channel"] for d in deliveries}
    assert {"email", "chatbook"} == channels
    email_result = next(d for d in deliveries if d["channel"] == "email")
    assert email_result["status"] in {"sent", "partial"}
    chat_result = next(d for d in deliveries if d["channel"] == "chatbook")
    assert chat_result["status"] in {"stored", "failed"}

    chatbook_id = output.get("metadata", {}).get("chatbook_document_id")
    if chat_result["status"] == "stored":
        assert isinstance(chatbook_id, int)
        assert output.get("chatbook_path") == f"generated_document:{chatbook_id}"

        db_path = DatabasePaths.get_chacha_db_path(555)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT metadata FROM generated_documents WHERE id = ?", (chatbook_id,)).fetchone()
            assert row is not None
            stored_meta = json.loads(row[0])
            assert stored_meta.get("job_id") == job_id
            assert stored_meta.get("run_id") == run_id
            assert stored_meta.get("origin") == "test"


def test_outputs_generate_audio_payload_triggers_workflow_and_updates_run_stats(client_with_user, monkeypatch):

    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")

    src_body = {
        "name": "Audio Feed",
        "url": "https://example.com/audio.xml",
        "source_type": "rss",
        "tags": ["audio"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text

    job_body = {
        "name": "Audio Digest",
        "scope": {"tags": ["audio"]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]
    audio_cast = {
        "speaker_count": 2,
        "speakers": [
            {
                "id": "host",
                "label": "Host",
                "role": "anchor",
                "voice": "af_bella",
            },
            {
                "id": "analyst",
                "label": "Analyst",
                "voice": "am_adam",
            },
        ],
    }

    with patch(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        new=AsyncMock(return_value="task_output_audio"),
    ) as mock_trigger:
        r = c.post(
            "/api/v1/watchlists/outputs",
            json={
                "run_id": run_id,
                "title": "Audio Briefing Output",
                "generate_audio": True,
                "target_audio_minutes": 5,
                "audio_model": "tts-1",
                "audio_voice": "nova",
                "audio_speed": 1.2,
                "background_audio_uri": "file:///tmp/background.mp3",
                "background_volume": 0.25,
                "background_delay_ms": 750,
                "background_fade_seconds": 2.5,
                "audio_language": "fr",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "persona_summarize": True,
                "persona_id": "analyst",
                "persona_provider": "openai",
                "persona_model": "gpt-4o-mini",
                "voice_map": {"HOST": "af_bella"},
                "audio_cast": audio_cast,
            },
        )
    assert r.status_code == 200, r.text
    output = r.json()
    metadata = output.get("metadata", {})
    assert metadata.get("audio_briefing_requested") is True
    assert metadata.get("audio_briefing_task_id") == "task_output_audio"
    assert metadata.get("audio_briefing_status") == "pending"

    assert mock_trigger.await_count == 1
    kwargs = mock_trigger.await_args.kwargs
    assert kwargs["user_id"] == 555
    assert kwargs["job_id"] == job_id
    assert kwargs["run_id"] == run_id
    assert kwargs["output_prefs"]["generate_audio"] is True
    assert kwargs["output_prefs"]["target_audio_minutes"] == 5
    assert kwargs["output_prefs"]["audio_model"] == "tts-1"
    assert kwargs["output_prefs"]["audio_voice"] == "nova"
    assert kwargs["output_prefs"]["audio_speed"] == 1.2
    assert kwargs["output_prefs"]["background_audio_uri"] == "file:///tmp/background.mp3"
    assert kwargs["output_prefs"]["background_volume"] == 0.25
    assert kwargs["output_prefs"]["background_delay_ms"] == 750
    assert kwargs["output_prefs"]["background_fade_seconds"] == 2.5
    assert kwargs["output_prefs"]["audio_language"] == "fr"
    assert kwargs["output_prefs"]["llm_provider"] == "openai"
    assert kwargs["output_prefs"]["llm_model"] == "gpt-4o-mini"
    assert kwargs["output_prefs"]["persona_summarize"] is True
    assert kwargs["output_prefs"]["persona_id"] == "analyst"
    assert kwargs["output_prefs"]["persona_provider"] == "openai"
    assert kwargs["output_prefs"]["persona_model"] == "gpt-4o-mini"
    assert kwargs["output_prefs"]["voice_map"] == {"HOST": "af_bella"}
    assert kwargs["output_prefs"]["audio_cast"] == audio_cast

    r = c.get(f"/api/v1/watchlists/runs/{run_id}")
    assert r.status_code == 200, r.text
    run_payload = r.json()
    assert run_payload.get("stats", {}).get("audio_briefing_task_id") == "task_output_audio"


def test_outputs_generate_audio_false_does_not_trigger_workflow(client_with_user, monkeypatch):

    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")

    src_body = {
        "name": "No Audio Feed",
        "url": "https://example.com/no-audio.xml",
        "source_type": "rss",
        "tags": ["silent"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text

    job_body = {
        "name": "No Audio Digest",
        "scope": {"tags": ["silent"]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    with patch(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        new=AsyncMock(return_value="task_should_not_exist"),
    ) as mock_trigger:
        r = c.post(
            "/api/v1/watchlists/outputs",
            json={
                "run_id": run_id,
                "title": "No Audio Output",
                "generate_audio": False,
            },
        )
    assert r.status_code == 200, r.text
    output = r.json()
    metadata = output.get("metadata", {})
    assert "audio_briefing_requested" not in metadata
    assert "audio_briefing_task_id" not in metadata
    assert mock_trigger.await_count == 0


def test_outputs_generate_audio_trigger_returns_none_marks_skipped_metadata(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark audio briefing metadata as skipped when enqueue returns no task id."""
    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")

    src_body = {
        "name": "Audio Skip Feed",
        "url": "https://example.com/audio-skip.xml",
        "source_type": "rss",
        "tags": ["audio-skip"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text

    job_body = {
        "name": "Audio Skip Digest",
        "scope": {"tags": ["audio-skip"]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    with patch(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        new=AsyncMock(return_value=None),
    ) as mock_trigger:
        r = c.post(
            "/api/v1/watchlists/outputs",
            json={
                "run_id": run_id,
                "title": "Audio Skipped Output",
                "generate_audio": True,
            },
        )

    assert r.status_code == 200, r.text
    output = r.json()
    metadata = output.get("metadata", {})
    assert metadata.get("audio_briefing_requested") is True
    assert metadata.get("audio_briefing_status") == "skipped"
    assert "audio_briefing_task_id" not in metadata
    assert mock_trigger.await_count == 1

    r = c.get(f"/api/v1/watchlists/runs/{run_id}")
    assert r.status_code == 200, r.text
    run_payload = r.json()
    assert run_payload.get("stats", {}).get("audio_briefing_task_id") is None


def test_outputs_generate_audio_trigger_failure_marks_enqueue_failed_metadata(
    client_with_user: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist enqueue failure metadata when workflow trigger raises an exception."""
    c = client_with_user
    monkeypatch.setenv("TEST_MODE", "1")

    src_body = {
        "name": "Audio Failure Feed",
        "url": "https://example.com/audio-failure.xml",
        "source_type": "rss",
        "tags": ["audio-failure"],
    }
    r = c.post("/api/v1/watchlists/sources", json=src_body)
    assert r.status_code == 200, r.text

    job_body = {
        "name": "Audio Failure Digest",
        "scope": {"tags": ["audio-failure"]},
        "schedule_expr": None,
        "timezone": "UTC",
        "active": True,
    }
    r = c.post("/api/v1/watchlists/jobs", json=job_body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    with patch(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        new=AsyncMock(side_effect=RuntimeError("scheduler unavailable")),
    ) as mock_trigger:
        r = c.post(
            "/api/v1/watchlists/outputs",
            json={
                "run_id": run_id,
                "title": "Audio Failure Output",
                "generate_audio": True,
            },
        )

    assert r.status_code == 200, r.text
    output = r.json()
    metadata = output.get("metadata", {})
    assert metadata.get("audio_briefing_requested") is True
    assert metadata.get("audio_briefing_status") == "enqueue_failed"
    assert "scheduler unavailable" in str(metadata.get("audio_briefing_error", ""))
    assert "audio_briefing_task_id" not in metadata
    assert mock_trigger.await_count == 1

    r = c.get(f"/api/v1/watchlists/runs/{run_id}")
    assert r.status_code == 200, r.text
    run_payload = r.json()
    assert run_payload.get("stats", {}).get("audio_briefing_task_id") is None
