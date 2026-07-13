from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


class _Clock:
    def __init__(self) -> None:
        self.current = datetime.now(timezone.utc)

    def now_utc(self) -> datetime:
        return self.current


class _EmptyMediaDB:
    def get_media_by_urls(self, _urls, **_kwargs):
        return []

    def close_connection(self) -> None:
        return None


@pytest.fixture()
def run_api(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("PLAYLIST_RUN_SSE_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("PLAYLIST_RUN_SSE_TEST_MAX_SECONDS", "0.3")

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
    from tldw_Server_API.app.api.v1.endpoints import media
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_service
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "playlist-run-workflow.db", clock=_Clock())
    monkeypatch.setattr(playlist_ingest_service, "_owner_media_db", lambda _owner: _EmptyMediaDB())
    app = FastAPI()
    app.include_router(media.router, prefix="/api/v1/media", tags=["media"])
    app.include_router(ingest_jobs.router, prefix="/api/v1/media", tags=["media"])
    app.dependency_overrides[get_request_user] = lambda: User(
        id=1,
        username="run-owner",
        email=None,
        is_active=True,
    )
    app.dependency_overrides[get_job_manager] = lambda: manager
    app.dependency_overrides[ingest_jobs.get_job_manager] = lambda: manager
    with TestClient(app, headers={"X-API-KEY": "test-api-key-12345"}) as client:
        yield client, manager, PlaylistIngestStore(manager)


def _create_run(client: TestClient, *occurrence_ids: str) -> dict:
    response = client.post(
        "/api/v1/media/ingest/runs",
        json={
            "inputs": [
                {
                    "input_kind": "direct_url",
                    "occurrence_id": occurrence_id,
                    "url": f"https://example.com/{occurrence_id}",
                    "display_metadata": {"title": occurrence_id},
                }
                for occurrence_id in occurrence_ids
            ],
            "review_overrides": {},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _submit(client: TestClient, run_id: str, *occurrence_ids: str) -> dict:
    response = client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "run_id": run_id,
            "urls": [f"https://example.com/{occurrence_id}" for occurrence_id in occurrence_ids],
            "occurrence_ids": json.dumps(list(occurrence_ids)),
            "attempts": json.dumps([1] * len(occurrence_ids)),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _events(response) -> list[dict]:
    events: list[dict] = []
    current: dict[str, object] = {}
    for line in response.iter_lines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("id:"):
            current["id"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            if payload == "[DONE]":
                current["data"] = payload
            else:
                current["data"] = json.loads(payload)
    if current:
        events.append(current)
    return events


def test_run_sse_replays_from_after_id_query_without_header(run_api):
    client, _manager, store = run_api
    created = _create_run(client, "occ-one", "occ-two")
    persisted = list(store.list_run_events("1", created["run_id"]))
    assert len(persisted) >= 2

    with client.stream(
        "GET",
        created["events_url"],
        params={"after_id": persisted[0].event_id},
    ) as response:
        assert response.status_code == 200, response.text
        events = _events(response)

    occurrence_events = [event for event in events if event.get("event") == "occurrence"]
    assert occurrence_events
    assert all(int(event["id"]) > persisted[0].event_id for event in occurrence_events)
    assert {event["data"]["occurrence_id"] for event in occurrence_events} == {"occ-two"}


def test_run_sse_last_event_id_takes_precedence_over_after_id_query(run_api):
    client, _manager, store = run_api
    created = _create_run(client, "occ-one", "occ-two")
    persisted = list(store.list_run_events("1", created["run_id"]))
    assert len(persisted) >= 2

    with client.stream(
        "GET",
        created["events_url"],
        params={"after_id": persisted[1].event_id},
        headers={"Last-Event-ID": str(persisted[0].event_id)},
    ) as response:
        assert response.status_code == 200, response.text
        events = _events(response)

    assert events[0]["event"] == "snapshot"
    assert events[0]["data"]["run_id"] == created["run_id"]
    occurrence_events = [event for event in events if event.get("event") == "occurrence"]
    assert occurrence_events
    assert all(int(event["id"]) > persisted[0].event_id for event in occurrence_events)
    assert {event["data"]["occurrence_id"] for event in occurrence_events} == {"occ-two"}


def test_run_sse_zero_cursor_replays_all_retained_events_without_resync(run_api):
    client, _manager, store = run_api
    created = _create_run(client, "occ-one", "occ-two")
    persisted = list(store.list_run_events("1", created["run_id"]))
    assert len(persisted) >= 2

    with client.stream("GET", created["events_url"], params={"after_id": 0}) as response:
        assert response.status_code == 200, response.text
        events = _events(response)

    assert not any(event.get("event") == "resync_required" for event in events)
    occurrence_events = [event for event in events if event.get("event") == "occurrence"]
    assert {event["data"]["occurrence_id"] for event in occurrence_events} == {"occ-one", "occ-two"}


@pytest.mark.parametrize("cursor_transport", ["query", "header"])
def test_run_sse_high_cursor_resyncs_then_observes_later_event(run_api, cursor_transport):
    client, manager, store = run_api
    created = _create_run(client, "occ-high")
    submitted = _submit(client, created["run_id"], "occ-high")
    job_id = int(submitted["submissions"][0]["job_id"])
    minimum, maximum = store.run_event_bounds("1", created["run_id"])
    assert minimum is not None and maximum is not None
    high_cursor = maximum + 100

    def update_progress() -> None:
        time.sleep(0.15)
        manager.update_job_progress(job_id, progress_percent=42.0, progress_message="later")

    thread = threading.Thread(target=update_progress, daemon=True)
    thread.start()
    request_kwargs = (
        {"params": {"after_id": high_cursor}}
        if cursor_transport == "query"
        else {"headers": {"Last-Event-ID": str(high_cursor)}}
    )
    with client.stream("GET", created["events_url"], **request_kwargs) as response:
        assert response.status_code == 200, response.text
        events = _events(response)
    thread.join(timeout=1)

    resync = next(event for event in events if event.get("event") == "resync_required")
    assert resync["data"]["min_event_id"] == minimum
    assert resync["data"]["latest_event_id"] == maximum
    assert any(
        event.get("event") == "occurrence"
        and event.get("data", {}).get("progress_percent") == 42.0
        and int(event["id"]) > maximum
        for event in events
    )


def test_run_sse_positive_cursor_resyncs_when_no_events_are_retained(run_api):
    client, manager, store = run_api
    created = _create_run(client, "occ-empty-events")
    _minimum, previous_maximum = store.run_event_bounds("1", created["run_id"])
    assert previous_maximum is not None
    connection = manager._connect()
    try:
        connection.execute(
            "DELETE FROM media_ingest_run_events WHERE owner_user_id = ? AND run_id = ?",
            ("1", created["run_id"]),
        )
        connection.commit()
    finally:
        connection.close()
    assert store.run_event_bounds("1", created["run_id"]) == (None, None)

    appended: dict[str, int] = {}

    def append_later() -> None:
        time.sleep(0.15)
        event = store.append_run_event("1", created["run_id"], event_type="later")
        appended["event_id"] = event.event_id

    thread = threading.Thread(target=append_later, daemon=True)
    thread.start()
    with client.stream(
        "GET",
        created["events_url"],
        params={"after_id": previous_maximum + 100},
    ) as response:
        assert response.status_code == 200, response.text
        events = _events(response)
    thread.join(timeout=1)

    resync = next(event for event in events if event.get("event") == "resync_required")
    assert resync["data"]["min_event_id"] is None
    assert resync["data"]["latest_event_id"] is None
    assert any(
        event.get("event") == "run"
        and event.get("data", {}).get("event_type") == "later"
        and int(event["id"]) == appended["event_id"]
        for event in events
    )


def test_run_sse_expired_replay_emits_resync_required(run_api):
    client, manager, store = run_api
    created = _create_run(client, "occ-one", "occ-two")
    persisted = list(store.list_run_events("1", created["run_id"]))
    stale_id = persisted[0].event_id
    connection = manager._connect()
    try:
        connection.execute(
            "DELETE FROM media_ingest_run_events WHERE owner_user_id = ? AND run_id = ? AND event_id = ?",
            ("1", created["run_id"], stale_id),
        )
        connection.commit()
    finally:
        connection.close()

    with client.stream(
        "GET",
        created["events_url"],
        headers={"Last-Event-ID": str(stale_id)},
    ) as response:
        assert response.status_code == 200, response.text
        events = _events(response)

    resync = next(event for event in events if event.get("event") == "resync_required")
    assert resync["data"]["run_id"] == created["run_id"]
    assert resync["data"]["min_event_id"] == persisted[1].event_id
    assert resync["data"]["latest_event_id"] >= persisted[1].event_id


def test_stream_only_client_observes_job_progress_without_polling(run_api):
    client, manager, _store = run_api
    created = _create_run(client, "occ-progress")
    submitted = _submit(client, created["run_id"], "occ-progress")
    job_id = int(submitted["submissions"][0]["job_id"])

    def update_progress() -> None:
        time.sleep(0.05)
        manager.update_job_progress(job_id, progress_percent=35.0, progress_message="download")

    thread = threading.Thread(target=update_progress, daemon=True)
    thread.start()
    with client.stream("GET", created["events_url"]) as response:
        assert response.status_code == 200, response.text
        events = _events(response)
    thread.join(timeout=1)

    assert any(
        event.get("event") == "occurrence"
        and event.get("data", {}).get("occurrence_id") == "occ-progress"
        and event.get("data", {}).get("progress_percent") == 35.0
        and event.get("data", {}).get("progress_message") == "download"
        for event in events
    )


def test_same_run_stream_observes_job_bound_by_later_submission_chunk(run_api):
    client, _manager, _store = run_api
    created = _create_run(client, "occ-first", "occ-later")
    _submit(client, created["run_id"], "occ-first")
    later: dict[str, object] = {}

    def submit_later() -> None:
        time.sleep(0.05)
        later.update(_submit(client, created["run_id"], "occ-later"))

    thread = threading.Thread(target=submit_later, daemon=True)
    thread.start()
    with client.stream("GET", created["events_url"]) as response:
        assert response.status_code == 200, response.text
        events = _events(response)
    thread.join(timeout=1)

    later_job_id = int(later["submissions"][0]["job_id"])
    assert any(
        event.get("event") == "occurrence"
        and event.get("data", {}).get("occurrence_id") == "occ-later"
        and int(event.get("data", {}).get("job_id") or 0) == later_job_id
        for event in events
    )
