import io
import json
from importlib import import_module, reload
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        # Attach a tenant/organization id if needed in future; not required for this test
        return User(id=902, username="wluser", email=None, is_active=True)

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_run_detail"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    from fastapi import FastAPI
    from tldw_Server_API.app.core.config import API_V1_PREFIX
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_run_detail_includes_filter_totals(client_with_user):


    c = client_with_user

    # Create a source and a job with a 'flag' rule that should match test item
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "active": True,
            "settings": {"limit": 1},
        },
    )
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Flag Totals",
            "scope": {"sources": [source_id]},
            "active": True,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    # Replace filters with a flag keyword that matches 'Test' title
    r = c.patch(
        f"/api/v1/watchlists/jobs/{job_id}/filters",
        json={
            "filters": [
                {"type": "keyword", "action": "flag", "value": {"keywords": ["Test"], "match": "any"}},
            ]
        },
    )
    assert r.status_code == 200, r.text

    # Trigger run and get run id
    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    # Retrieve run detail and verify filter stats included
    r = c.get(f"/api/v1/watchlists/runs/{run_id}/details")
    assert r.status_code == 200, r.text
    detail = r.json()
    stats = detail.get("stats") or {}
    # Base stats
    assert "items_found" in stats and "items_ingested" in stats
    # Filter stats present
    assert "filters_matched" in stats
    assert "filters_flag" in stats or "filters_include" in stats or "filters_exclude" in stats
    assert "filter_tallies" not in stats
    # For our flag rule, ensure at least one flag counted when present
    if "filters_flag" in stats:
        assert stats["filters_flag"] >= 1

    r = c.get(f"/api/v1/watchlists/runs/{run_id}/details", params={"include_tallies": True})
    assert r.status_code == 200, r.text
    detail_with_tallies = r.json()
    assert isinstance(detail_with_tallies.get("filter_tallies"), dict)
    assert "filter_tallies" not in (detail_with_tallies.get("stats") or {})


def test_run_detail_stats_preserve_existing_flat_filter_counters():
    from tldw_Server_API.app.api.v1.endpoints.watchlists import _build_run_detail_stats

    detail_stats = _build_run_detail_stats(
        {
            "items_found": 4,
            "items_ingested": 2,
            "filters_include": 3,
            "filters_exclude": 1,
            "filters_flag": 2,
            "filter_tallies": {"keyword:foo": 6},
        },
        items_found=4,
        items_ingested=2,
    )

    assert detail_stats["filters_include"] == 3
    assert detail_stats["filters_exclude"] == 1
    assert detail_stats["filters_flag"] == 2
    assert "filter_tallies" not in detail_stats


def test_run_detail_includes_include_and_exclude_totals(client_with_user):


    c = client_with_user

    # Create a source
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "active": True,
            "settings": {"limit": 1},
        },
    )
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    # Include path: matching include rule should increment filters_include and ingest
    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Include Totals",
            "scope": {"sources": [source_id]},
            "active": True,
        },
    )
    assert r.status_code == 200, r.text
    include_job_id = r.json()["id"]

    r = c.patch(
        f"/api/v1/watchlists/jobs/{include_job_id}/filters",
        json={
            "filters": [
                {"type": "keyword", "action": "include", "value": {"keywords": ["Test"], "match": "any"}},
            ]
        },
    )
    assert r.status_code == 200, r.text

    r = c.post(f"/api/v1/watchlists/jobs/{include_job_id}/run")
    assert r.status_code == 200, r.text
    include_run_id = r.json()["id"]

    r = c.get(f"/api/v1/watchlists/runs/{include_run_id}/details")
    assert r.status_code == 200, r.text
    stats = (r.json().get("stats") or {})
    assert "filters_matched" in stats
    assert stats.get("filters_include", 0) >= 1
    assert stats.get("items_ingested", 0) >= 1

    # Exclude path: matching exclude rule should increment filters_exclude and block ingestion
    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Exclude Totals",
            "scope": {"sources": [source_id]},
            "active": True,
        },
    )
    assert r.status_code == 200, r.text
    exclude_job_id = r.json()["id"]

    r = c.patch(
        f"/api/v1/watchlists/jobs/{exclude_job_id}/filters",
        json={
            "filters": [
                {"type": "keyword", "action": "exclude", "value": {"keywords": ["Test"], "match": "any"}},
            ]
        },
    )
    assert r.status_code == 200, r.text

    r = c.post(f"/api/v1/watchlists/jobs/{exclude_job_id}/run")
    assert r.status_code == 200, r.text
    exclude_run_id = r.json()["id"]

    r = c.get(f"/api/v1/watchlists/runs/{exclude_run_id}/details")
    assert r.status_code == 200, r.text
    stats = (r.json().get("stats") or {})
    assert "filters_matched" in stats
    assert stats.get("filters_exclude", 0) >= 1
    assert stats.get("items_ingested", 0) == 0


def test_run_stats_exposes_filter_tallies_on_run(client_with_user):


    c = client_with_user

    # Create source and job with a simple include rule that matches
    r = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Feed",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
            "active": True,
            "settings": {"limit": 1},
        },
    )
    assert r.status_code == 200, r.text
    source_id = r.json()["id"]

    r = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Tallies",
            "scope": {"sources": [source_id]},
            "active": True,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    r = c.patch(
        f"/api/v1/watchlists/jobs/{job_id}/filters",
        json={
            "filters": [
                {"type": "keyword", "action": "include", "value": {"keywords": ["Test"], "match": "any"}},
            ]
        },
    )
    assert r.status_code == 200, r.text

    r = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert r.status_code == 200, r.text
    run_id = r.json()["id"]

    # Fetch raw run (not details) and assert filter_tallies presence
    r = c.get(f"/api/v1/watchlists/runs/{run_id}")
    assert r.status_code == 200, r.text
    run = r.json()
    stats = run.get("stats") or {}
    assert isinstance(stats, dict)
    tallies = stats.get("filter_tallies")
    assert isinstance(tallies, dict)
    # Expect at least one key for the matched rule; without ids, key starts with 'idx:'
    keys = list(tallies.keys())
    assert any(k.startswith("idx:") or k.startswith("id:") for k in keys)
