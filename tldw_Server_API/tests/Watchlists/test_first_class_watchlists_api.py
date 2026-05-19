from __future__ import annotations

import json
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=777, username="watchlist-api", email=None, is_active=True)

    base_dir = tmp_path / "watchlist_api_user_dbs"
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


def _create_watchlist(client: TestClient, *, name: str, domain: str = "general") -> dict:
    response = client.post(
        "/api/v1/watchlists",
        json={
            "name": name,
            "description": f"{name} description",
            "objective": f"Track {name}",
            "domain": domain,
            "priority": "high",
            "tags": [name.lower().replace(" ", "-")],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_source(client: TestClient, *, label: str, watchlist_id: int | None = None) -> dict:
    payload: dict[str, object] = {
        "name": f"{label} Feed",
        "url": f"https://example.com/{label}/rss.xml",
        "source_type": "rss",
    }
    if watchlist_id is not None:
        payload["watchlist_id"] = watchlist_id
    response = client.post("/api/v1/watchlists/sources", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _create_job(
    client: TestClient,
    *,
    label: str,
    source_id: int,
    watchlist_id: int | None = None,
) -> dict:
    payload: dict[str, object] = {
        "name": f"{label} Monitor",
        "description": f"{label} monitor",
        "scope": {"sources": [source_id]},
        "active": True,
    }
    if watchlist_id is not None:
        payload["watchlist_id"] = watchlist_id
    response = client.post("/api/v1/watchlists/jobs", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _default_watchlist_id(client: TestClient) -> int:
    response = client.get("/api/v1/watchlists")
    assert response.status_code == 200, response.text
    for item in response.json()["items"]:
        if item["name"] == "Imported Watchlist":
            return int(item["id"])
    raise AssertionError("default Watchlist not listed")


def test_watchlist_crud_delete_restore_and_static_routes(client_with_user):
    client = client_with_user

    created = _create_watchlist(client, name="Healthcare Ransomware", domain="cti_osint")
    watchlist_id = int(created["id"])
    assert created["status"] == "active"
    assert created["priority"] == "high"
    assert created["tags"] == ["healthcare-ransomware"]

    listed = client.get("/api/v1/watchlists")
    assert listed.status_code == 200, listed.text
    assert any(int(item["id"]) == watchlist_id for item in listed.json()["items"])

    fetched = client.get(f"/api/v1/watchlists/{watchlist_id}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["objective"] == "Track Healthcare Ransomware"

    patched = client.patch(
        f"/api/v1/watchlists/{watchlist_id}",
        json={"status": "archived", "priority": "critical", "tags": ["cti", "ransomware"]},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["status"] == "archived"
    assert patched.json()["priority"] == "critical"
    assert patched.json()["archived_at"]

    deleted = client.delete(f"/api/v1/watchlists/{watchlist_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["watchlist_id"] == watchlist_id
    assert deleted.json()["restore_expires_at"]

    missing = client.get(f"/api/v1/watchlists/{watchlist_id}")
    assert missing.status_code == 404

    restored = client.post(f"/api/v1/watchlists/{watchlist_id}/restore")
    assert restored.status_code == 200, restored.text
    assert restored.json()["id"] == watchlist_id
    assert restored.json()["deleted_at"] is None

    static_sources = client.get("/api/v1/watchlists/sources")
    assert static_sources.status_code == 200, static_sources.text


def test_watchlist_scopes_sources_jobs_runs_items_and_default_create(client_with_user):
    client = client_with_user

    cti_watchlist = _create_watchlist(client, name="CTI", domain="cti_osint")
    news_watchlist = _create_watchlist(client, name="News", domain="news")

    default_source = _create_source(client, label="default")
    cti_source = _create_source(client, label="cti", watchlist_id=int(cti_watchlist["id"]))
    news_source = _create_source(client, label="news", watchlist_id=int(news_watchlist["id"]))
    default_watchlist_id = _default_watchlist_id(client)

    assert default_watchlist_id in default_source["watchlist_ids"]
    assert cti_source["watchlist_ids"] == [int(cti_watchlist["id"])]

    cti_sources = client.get("/api/v1/watchlists/sources", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_sources.status_code == 200, cti_sources.text
    assert [item["id"] for item in cti_sources.json()["items"]] == [cti_source["id"]]

    default_sources = client.get("/api/v1/watchlists/sources", params={"watchlist_id": default_watchlist_id})
    assert default_sources.status_code == 200, default_sources.text
    assert default_source["id"] in [item["id"] for item in default_sources.json()["items"]]
    assert cti_source["id"] not in [item["id"] for item in default_sources.json()["items"]]

    default_job = _create_job(client, label="default", source_id=int(default_source["id"]))
    cti_job = _create_job(
        client,
        label="cti",
        source_id=int(cti_source["id"]),
        watchlist_id=int(cti_watchlist["id"]),
    )
    news_job = _create_job(
        client,
        label="news",
        source_id=int(news_source["id"]),
        watchlist_id=int(news_watchlist["id"]),
    )

    assert default_job["watchlist_id"] == default_watchlist_id
    assert cti_job["watchlist_id"] == int(cti_watchlist["id"])

    cti_jobs = client.get("/api/v1/watchlists/jobs", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_jobs.status_code == 200, cti_jobs.text
    assert [item["id"] for item in cti_jobs.json()["items"]] == [cti_job["id"]]

    db = WatchlistsDatabase.for_user(777)
    cti_run = db.create_run(int(cti_job["id"]), status="finished")
    cti_item = db.record_scraped_item(
        run_id=int(cti_run.id),
        job_id=int(cti_job["id"]),
        source_id=int(cti_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/cti/story",
        title="CTI story",
        summary="CTI summary",
        published_at=None,
        tags=["cti"],
        status="ingested",
    )
    news_run = db.create_run(int(news_job["id"]), status="finished")
    news_item = db.record_scraped_item(
        run_id=int(news_run.id),
        job_id=int(news_job["id"]),
        source_id=int(news_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/news/story",
        title="News story",
        summary="News summary",
        published_at=None,
        tags=["news"],
        status="ingested",
    )

    cti_runs = client.get("/api/v1/watchlists/runs", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_runs.status_code == 200, cti_runs.text
    assert [item["id"] for item in cti_runs.json()["items"]] == [int(cti_run.id)]

    cti_items = client.get("/api/v1/watchlists/items", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_items.status_code == 200, cti_items.text
    assert [item["id"] for item in cti_items.json()["items"]] == [int(cti_item.id)]
    assert int(news_run.id) not in [item["id"] for item in cti_runs.json()["items"]]
    assert int(news_item.id) not in [item["id"] for item in cti_items.json()["items"]]

    cti_counts = client.get("/api/v1/watchlists/items/smart-counts", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_counts.status_code == 200, cti_counts.text
    assert cti_counts.json()["all"] == 1


def test_watchlist_scopes_outputs_by_job_and_records_output_provenance(client_with_user):
    client = client_with_user

    cti_watchlist = _create_watchlist(client, name="CTI Outputs", domain="cti_osint")
    news_watchlist = _create_watchlist(client, name="News Outputs", domain="news")

    cti_source = _create_source(client, label="cti-outputs", watchlist_id=int(cti_watchlist["id"]))
    news_source = _create_source(client, label="news-outputs", watchlist_id=int(news_watchlist["id"]))
    cti_job = _create_job(
        client,
        label="cti-outputs",
        source_id=int(cti_source["id"]),
        watchlist_id=int(cti_watchlist["id"]),
    )
    news_job = _create_job(
        client,
        label="news-outputs",
        source_id=int(news_source["id"]),
        watchlist_id=int(news_watchlist["id"]),
    )

    db = WatchlistsDatabase.for_user(777)
    cti_run = db.create_run(int(cti_job["id"]), status="finished")
    news_run = db.create_run(int(news_job["id"]), status="finished")
    db.record_scraped_item(
        run_id=int(cti_run.id),
        job_id=int(cti_job["id"]),
        source_id=int(cti_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/cti-output/story",
        title="CTI output story",
        summary="CTI output summary",
        published_at=None,
        tags=["cti"],
        status="ingested",
    )
    db.record_scraped_item(
        run_id=int(news_run.id),
        job_id=int(news_job["id"]),
        source_id=int(news_source["id"]),
        media_id=None,
        media_uuid=None,
        url="https://example.com/news-output/story",
        title="News output story",
        summary="News output summary",
        published_at=None,
        tags=["news"],
        status="ingested",
    )

    cti_output = client.post(
        "/api/v1/watchlists/outputs",
        json={"run_id": int(cti_run.id), "title": "CTI digest", "retention_seconds": 0},
    )
    assert cti_output.status_code == 200, cti_output.text
    news_output = client.post(
        "/api/v1/watchlists/outputs",
        json={"run_id": int(news_run.id), "title": "News digest", "retention_seconds": 0},
    )
    assert news_output.status_code == 200, news_output.text

    cti_payload = cti_output.json()
    news_payload = news_output.json()
    assert cti_payload["metadata"]["watchlist_id"] == int(cti_watchlist["id"])
    assert cti_payload["metadata"]["job_id"] == int(cti_job["id"])
    assert cti_payload["metadata"]["run_id"] == int(cti_run.id)

    legacy_row = CollectionsDatabase.for_user(777).create_output_artifact(
        type_="briefing_markdown",
        title="Legacy CTI digest",
        format_="md",
        storage_path="legacy-cti-digest.md",
        metadata_json=json.dumps({"origin": "watchlists", "version": 1}),
        job_id=int(cti_job["id"]),
        run_id=int(cti_run.id),
    )

    cti_outputs = client.get("/api/v1/watchlists/outputs", params={"watchlist_id": cti_watchlist["id"]})
    assert cti_outputs.status_code == 200, cti_outputs.text
    cti_ids = {item["id"] for item in cti_outputs.json()["items"]}
    assert int(cti_payload["id"]) in cti_ids
    assert int(legacy_row.id) in cti_ids
    assert int(news_payload["id"]) not in cti_ids

    news_outputs = client.get("/api/v1/watchlists/outputs", params={"watchlist_id": news_watchlist["id"]})
    assert news_outputs.status_code == 200, news_outputs.text
    news_ids = {item["id"] for item in news_outputs.json()["items"]}
    assert int(news_payload["id"]) in news_ids
    assert int(cti_payload["id"]) not in news_ids
    assert int(legacy_row.id) not in news_ids


def test_legacy_job_cluster_route_still_uses_job_id(client_with_user):
    client = client_with_user

    source = _create_source(client, label="cluster")
    job = _create_job(client, label="cluster", source_id=int(source["id"]))
    job_id = int(job["id"])

    subscribed = client.post(f"/api/v1/watchlists/{job_id}/clusters", json={"cluster_id": 42})
    assert subscribed.status_code == 200, subscribed.text
    assert subscribed.json()["cluster_id"] == 42

    listed = client.get(f"/api/v1/watchlists/{job_id}/clusters")
    assert listed.status_code == 200, listed.text
    assert any(int(item["cluster_id"]) == 42 for item in listed.json().get("clusters") or [])
