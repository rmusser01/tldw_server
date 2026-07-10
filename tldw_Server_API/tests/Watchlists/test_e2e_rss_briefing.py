"""E2E tests with real RSS feeds for the newsletter briefing workflow.

These tests hit real external feeds and require network access.
Skip by default — enable with RUN_E2E=1 environment variable.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="E2E tests disabled (set RUN_E2E=1)"),
]


def _run_metric(run_payload: dict, key: str) -> int:
    """Read run counters from current (`stats`) and legacy (top-level) payloads."""
    stats = run_payload.get("stats")
    if isinstance(stats, dict):
        value = stats.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    legacy = run_payload.get(key)
    if isinstance(legacy, (int, float)):
        return int(legacy)
    return 0


def _preview_count(preview_payload: dict) -> int:
    """Read source-test preview counts across response variants."""
    direct = preview_payload.get("items_found")
    if isinstance(direct, (int, float)):
        return int(direct)
    total = preview_payload.get("total")
    if isinstance(total, (int, float)):
        return int(total)
    items = preview_payload.get("items")
    if isinstance(items, list):
        return len(items)
    return 0


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=990, username="e2e-user", email="e2e@example.com", is_active=True)

    base_dir = tmp_path / "test_user_dbs_e2e"
    base_dir.mkdir(parents=True, exist_ok=True)
    template_dir = tmp_path / "watchlist_templates_e2e"
    template_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(template_dir))
    monkeypatch.setenv("WATCHLIST_BRIEFING_MAX_ITEMS", "100")
    monkeypatch.setenv("EMAIL_PROVIDER", "mock")

    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Real RSS feed → briefing
# ---------------------------------------------------------------------------


def test_real_xkcd_rss_to_briefing(client_with_user: TestClient):
    """Fetch xkcd RSS (stable, rarely changes format) and generate a briefing."""
    c = client_with_user

    # Create source with real xkcd feed
    src = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "xkcd",
            "url": "https://xkcd.com/rss.xml",
            "source_type": "rss",
        },
    )
    assert src.status_code == 200, src.text
    source_id = src.json()["id"]

    # Create job
    job = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "xkcd Digest",
            "scope": {"sources": [source_id]},
            "output_prefs": {
                "briefing_pipeline": {
                    "version": 1,
                    "text": {"enabled": True, "type": "briefing_markdown", "format": "md"},
                    "audio": {"enabled": False},
                }
            },
        },
    )
    assert job.status_code == 200, job.text
    job_id = job.json()["id"]

    # Run
    run = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert run.status_code == 200, run.text
    run_data = run.json()
    run_id = run_data["id"]
    assert _run_metric(run_data, "items_found") >= 1
    assert _run_metric(run_data, "items_ingested") >= 1
    occurrence = (run_data.get("stats") or {}).get("briefing_occurrence") or {}
    assert occurrence.get("artifact_status") == "ready"
    assert occurrence.get("output_id") is not None

    # Verify items have expected fields
    items_resp = c.get("/api/v1/watchlists/items", params={"run_id": run_id, "limit": 5})
    assert items_resp.status_code == 200, items_resp.text
    items = items_resp.json().get("items", [])
    assert len(items) >= 1
    for item in items:
        assert item.get("title")
        assert item.get("url")

    output_id = int(occurrence["output_id"])
    output = CollectionsDatabase.for_user(990).get_output_artifact(output_id)
    metadata = json.loads(output.metadata_json or "{}")
    content = (DatabasePaths.get_user_outputs_dir(990) / output.storage_path).read_text(encoding="utf-8")
    assert metadata["origin"] == "watchlists"
    assert metadata["generation_mode"] == "auto_output"
    assert metadata["occurrence_id"] == occurrence["id"]
    assert metadata["selected_count"] >= 1
    assert content.startswith("# xkcd Digest")


def test_real_hnrss_best_to_briefing(client_with_user: TestClient):
    """Fetch Hacker News best stories RSS and generate a briefing."""
    c = client_with_user

    src = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "HN Best",
            "url": "https://hnrss.org/best",
            "source_type": "rss",
        },
    )
    assert src.status_code == 200, src.text

    job = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "HN Digest",
            "scope": {"sources": [src.json()["id"]]},
            "job_filters": {
                "filters": [
                    {"type": "date_range", "action": "include", "value": {"max_age_days": 7}},
                ]
            },
        },
    )
    assert job.status_code == 200, job.text

    run = c.post(f"/api/v1/watchlists/jobs/{job.json()['id']}/run")
    assert run.status_code == 200, run.text
    run_data = run.json()
    assert _run_metric(run_data, "items_found") >= 1

    # Generate output only if items were ingested
    if _run_metric(run_data, "items_ingested") > 0:
        output = c.post(
            "/api/v1/watchlists/outputs",
            json={"run_id": run_data["id"], "type": "briefing_markdown", "temporary": True},
        )
        assert output.status_code == 200, output.text


# ---------------------------------------------------------------------------
# 2. Source test endpoint
# ---------------------------------------------------------------------------

def test_source_test_with_real_feed(client_with_user: TestClient):
    """POST /sources/{id}/test should return preview items from a real feed."""
    c = client_with_user

    src = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "xkcd test",
            "url": "https://xkcd.com/rss.xml",
            "source_type": "rss",
        },
    )
    assert src.status_code == 200, src.text
    source_id = src.json()["id"]

    test_resp = c.post(f"/api/v1/watchlists/sources/{source_id}/test")
    assert test_resp.status_code == 200, test_resp.text
    data = test_resp.json()
    # Should return some preview items
    assert _preview_count(data) >= 1


# ---------------------------------------------------------------------------
# 3. Multi-feed aggregation
# ---------------------------------------------------------------------------

def test_multi_feed_aggregation(client_with_user: TestClient):
    """Aggregate items from multiple real feeds into a single briefing."""
    c = client_with_user

    feed_urls = [
        ("xkcd", "https://xkcd.com/rss.xml"),
        ("HN Best", "https://hnrss.org/best"),
    ]

    source_ids = []
    for name, url in feed_urls:
        src = c.post(
            "/api/v1/watchlists/sources",
            json={"name": name, "url": url, "source_type": "rss", "tags": ["multi"]},
        )
        assert src.status_code == 200, src.text
        source_ids.append(src.json()["id"])

    job = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Multi Feed Digest",
            "scope": {"tags": ["multi"]},
        },
    )
    assert job.status_code == 200, job.text

    run = c.post(f"/api/v1/watchlists/jobs/{job.json()['id']}/run")
    assert run.status_code == 200, run.text
    run_data = run.json()
    # Should have items from at least one feed
    assert _run_metric(run_data, "items_found") >= 1

    if _run_metric(run_data, "items_ingested") > 0:
        output = c.post(
            "/api/v1/watchlists/outputs",
            json={"run_id": run_data["id"], "type": "briefing_markdown", "temporary": True},
        )
        assert output.status_code == 200, output.text
        assert len(output.json().get("content", "")) > 0
