from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit
NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self.current = datetime.now(timezone.utc)

    def now_utc(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


class _EndpointMediaDB:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def get_media_by_urls(self, _urls, **_kwargs):
        return list(self.rows)

    def close_connection(self) -> None:
        return None


@pytest.fixture
def preflight_api(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY", "4")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_OWNER_CAPACITY", "2")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_TTL_SECONDS", "600")

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
    from tldw_Server_API.app.api.v1.endpoints import media
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    clock = _Clock()
    manager = JobManager(db_path=tmp_path / "playlist-api.db", clock=clock)
    owner = {"id": 1}

    def current_user() -> User:
        return User(
            id=owner["id"],
            username=f"owner-{owner['id']}",
            email=None,
            is_active=True,
        )

    app = FastAPI()
    app.include_router(media.router, prefix="/api/v1/media", tags=["media"])
    app.include_router(ingest_jobs.router, prefix="/api/v1/media", tags=["media"])
    app.dependency_overrides[get_request_user] = current_user
    app.dependency_overrides[get_job_manager] = lambda: manager
    app.dependency_overrides[ingest_jobs.get_job_manager] = lambda: manager
    with TestClient(app, headers={"X-API-KEY": "test-api-key-12345"}) as client:
        yield client, manager, clock, owner


def _create_preflight(client: TestClient, playlist_id: str = "PLresource") -> dict:
    response = client.post(
        "/api/v1/media/playlist-preflights",
        json={
            "url": f"https://www.youtube.com/playlist?list={playlist_id}",
            "max_items": 34,
            "timeout_seconds": 12,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def _closure_values(callable_obj) -> list[object]:
    closure = getattr(callable_obj, "__closure__", None) or ()
    return [cell.cell_contents for cell in closure]


def _install_endpoint_media_db(monkeypatch) -> _EndpointMediaDB:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import playlist_ingest_service

    media_db = _EndpointMediaDB()
    monkeypatch.setattr(playlist_ingest_service, "_owner_media_db", lambda _owner: media_db)
    return media_db


def _create_run(client: TestClient, *occurrence_ids: str) -> dict:
    response = client.post(
        "/api/v1/media/ingest/runs",
        json={
            "client_request_id": f"endpoint-run:{':'.join(occurrence_ids)}",
            "inputs": [
                {
                    "input_kind": "direct_url",
                    "occurrence_id": occurrence_id,
                    "url": f"https://example.com/{occurrence_id}",
                    "source_kind": "video",
                    "display_metadata": {"title": occurrence_id},
                }
                for occurrence_id in occurrence_ids
            ],
            "review_overrides": {},
            "processing_options": {"media_type": "video"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize(
    "client_request_id",
    [pytest.param(None, id="missing"), pytest.param(" ", id="blank"), pytest.param("x" * 256, id="too-long")],
)
def test_run_post_requires_bounded_canonical_client_request_id(
    preflight_api,
    monkeypatch,
    client_request_id,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    payload = {
        "inputs": [
            {
                "input_kind": "direct_url",
                "occurrence_id": "occ-request-id",
                "url": "https://example.com/request-id",
            }
        ],
        "review_overrides": {},
    }
    if client_request_id is not None:
        payload["client_request_id"] = client_request_id

    response = client.post("/api/v1/media/ingest/runs", json=payload)

    assert response.status_code == 422
    with manager._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM media_ingest_runs").fetchone()[0] == 0


def _submit_run_occurrences(client: TestClient, run_id: str, *occurrence_ids: str) -> dict:
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


def _ready_snapshot(manager, owner: str, preflight_id: str) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    PlaylistIngestStore(manager).replace_preflight_snapshot(
        owner,
        preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "occ-1",
                "ordinal": 1,
                "occurrence_index_for_source": 1,
                "source_url": "https://www.youtube.com/watch?v=video1",
                "normalized_source_id": "youtube:video:video1",
                "source_kind": "youtube_video",
                "availability": "available",
                "duplicate_status": "new",
                "selected_by_default": True,
                "display_metadata": {
                    "title": "Opening keynote",
                    "channel_or_uploader": "Conference",
                    "playlist_title": "Conference 2010",
                    "internal_note": "must-not-leak",
                },
            },
            {
                "occurrence_id": "occ-2",
                "ordinal": 2,
                "occurrence_index_for_source": 1,
                "source_url": "https://www.youtube.com/watch?v=video2",
                "normalized_source_id": "youtube:video:video2",
                "source_kind": "youtube_video",
                "availability": "available",
                "duplicate_status": "duplicate_existing",
                "selected_by_default": False,
                "display_metadata": {"title": "Closing keynote"},
            },
        ],
        summary={
            "playlist_title": "Conference 2010",
            "total_count": 2,
            "loaded_count": 2,
            "ingestible_count": 2,
            "unavailable_count": 0,
            "duplicate_count": 1,
            "selected_count": 1,
            "warnings": [],
        },
    )


def test_preflight_post_returns_202_and_durably_bound_internal_job(preflight_api):
    client, manager, _clock, _owner = preflight_api

    body = _create_preflight(client)

    assert body["contract_version"] == 2
    assert body["status"] == "pending"
    assert body["status_url"] == f"/api/v1/media/playlist-preflights/{body['preflight_id']}"
    assert body["items_url"] == f"/api/v1/media/playlist-preflights/{body['preflight_id']}/items"
    assert body["limits"] == {"max_items": 34, "global_capacity": 4, "owner_capacity": 2}

    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["job_type"] == "playlist_preflight"
    assert job["owner_user_id"] == "1"
    assert job["payload"] == {
        "preflight_id": body["preflight_id"],
        "max_items": 34,
        "timeout_seconds": 12,
    }

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    stored = PlaylistIngestStore(manager).get_preflight("1", body["preflight_id"])
    assert stored.job_id == int(job["id"])


def test_preflight_publication_is_immediately_acquirable_with_real_sqlite_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "real-clock-publication.db")
    created = PlaylistIngestService(manager).create_preflight(
        "real-clock-owner",
        url="https://www.youtube.com/playlist?list=PLrealclock",
        max_items=25,
        timeout_seconds=15,
    )
    with manager._connect() as connection:
        published_counters = connection.execute(
            """
            SELECT ready_count, scheduled_count, processing_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default' AND job_type = 'playlist_preflight'
            """
        ).fetchone()
    assert dict(published_counters) == {"ready_count": 1, "scheduled_count": 0, "processing_count": 0}

    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="real-clock-worker",
        lease_seconds=120,
        job_type="playlist_preflight",
    )

    assert claimed is not None
    assert int(claimed["id"]) == created.record.job_id
    assert claimed["payload"]["preflight_id"] == created.preflight_id
    with manager._connect() as connection:
        acquired_counters = connection.execute(
            """
            SELECT ready_count, scheduled_count, processing_count FROM job_counters
            WHERE domain = 'media_ingest' AND queue = 'default' AND job_type = 'playlist_preflight'
            """
        ).fetchone()
    assert dict(acquired_counters) == {"ready_count": 0, "scheduled_count": 0, "processing_count": 1}

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    store = PlaylistIngestStore(manager)
    claim = {
        "expected_job_id": int(claimed["id"]),
        "expected_lease_id": str(claimed["lease_id"]),
        "expected_worker_id": str(claimed["worker_id"]),
    }
    store.replace_preflight_snapshot(
        "real-clock-owner",
        created.preflight_id,
        status="running",
        items=[],
        **claim,
    )
    assert PlaylistIngestService(manager).get_preflight("real-clock-owner", created.preflight_id).status == "running"
    store.replace_preflight_snapshot(
        "real-clock-owner",
        created.preflight_id,
        status="ready",
        items=[
            {
                "occurrence_id": "real-clock-occurrence",
                "ordinal": 1,
                "source_url": "https://www.youtube.com/watch?v=realclock",
                "source_kind": "youtube_video",
            }
        ],
        **claim,
    )
    page = PlaylistIngestService(manager).list_preflight_items(
        "real-clock-owner",
        created.preflight_id,
        limit=10,
        cursor=None,
    )
    materialized = PlaylistIngestService(manager).create_materialization(
        "real-clock-owner",
        created.preflight_id,
        ["real-clock-occurrence"],
    )
    assert [item.occurrence_id for item in page] == ["real-clock-occurrence"]
    assert [item.occurrence_id for item in materialized.items] == ["real-clock-occurrence"]


def test_unbound_sentinel_job_is_reconciled_before_next_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY", "1")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_OWNER_CAPACITY", "1")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_ORPHAN_GRACE_SECONDS", "1")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "orphan-reconciliation.db")
    original_bind = PlaylistIngestStore.bind_preflight_job

    def crash_after_create(*_args, **_kwargs):
        raise SystemExit("simulated process death")

    monkeypatch.setattr(PlaylistIngestStore, "bind_preflight_job", crash_after_create)
    with pytest.raises(SystemExit, match="simulated process death"):
        PlaylistIngestService(manager).create_preflight(
            "orphan-owner",
            url="https://www.youtube.com/playlist?list=PLorphan",
            max_items=20,
            timeout_seconds=10,
        )
    monkeypatch.setattr(PlaylistIngestStore, "bind_preflight_job", original_bind)

    orphan = manager.list_jobs(domain="media_ingest", owner_user_id="orphan-owner", limit=10)[0]
    assert str(orphan["available_at"]).startswith("9999-12-31")
    assert (
        manager.acquire_next_job(
            domain="media_ingest",
            queue="default",
            worker_id="must-not-acquire",
            lease_seconds=120,
            job_type="playlist_preflight",
        )
        is None
    )
    with manager._connect() as connection:
        connection.execute(
            "UPDATE jobs SET created_at = DATETIME('now', '-2 minutes') WHERE id = ?",
            (int(orphan["id"]),),
        )
        connection.commit()

    admitted = PlaylistIngestService(manager).create_preflight(
        "orphan-owner",
        url="https://www.youtube.com/playlist?list=PLafterorphan",
        max_items=20,
        timeout_seconds=10,
    )

    assert admitted.preflight_id
    assert manager.get_job(int(orphan["id"]))["status"] == "cancelled"
    with manager._connect() as connection:
        stale_preflight = connection.execute(
            "SELECT status FROM playlist_preflights WHERE owner_user_id = ? AND preflight_id != ?",
            ("orphan-owner", admitted.preflight_id),
        ).fetchone()
    assert stale_preflight["status"] == "blocked"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/playlist?list=PLsecret",
        "https://youtube.com.evil.example/playlist?list=PLsecret",
        "https://www.youtube.com/watch?v=video-only",
        "https://secret@www.youtube.com/playlist?list=PLsecret",
        "https://www.youtube.com/playlist?list=PLsecret&access_token=do-not-echo",
        "https://www.youtube.com/playlist?list=PLsecret#private-fragment",
    ],
)
def test_preflight_post_rejects_untrusted_or_secret_bearing_input_safely(preflight_api, url):
    client, manager, _clock, _owner = preflight_api

    response = client.post(
        "/api/v1/media/playlist-preflights",
        json={"url": url, "max_items": 10},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_playlist_url"
    assert "PLsecret" not in response.text
    assert "do-not-echo" not in response.text
    assert manager.list_jobs(domain="media_ingest", limit=10) == []


def test_preflight_summary_and_signed_item_pages_are_owner_scoped(preflight_api):
    client, manager, _clock, owner = preflight_api
    accepted = _create_preflight(client)
    _ready_snapshot(manager, "1", accepted["preflight_id"])

    summary = client.get(accepted["status_url"])
    first = client.get(accepted["items_url"], params={"limit": 1})

    assert summary.status_code == 200, summary.text
    assert summary.json()["contract_version"] == 2
    assert summary.json()["summary"]["loaded_count"] == 2
    assert first.status_code == 200, first.text
    assert [item["occurrence_id"] for item in first.json()["items"]] == ["occ-1"]
    assert first.json()["next_cursor"]

    second = client.get(
        accepted["items_url"],
        params={"limit": 1, "cursor": first.json()["next_cursor"]},
    )
    assert [item["occurrence_id"] for item in second.json()["items"]] == ["occ-2"]
    assert second.json()["next_cursor"] is None

    tampered = client.get(
        accepted["items_url"],
        params={"limit": 1, "cursor": first.json()["next_cursor"] + "x"},
    )
    assert tampered.status_code == 404
    assert tampered.json()["detail"] == "preflight_not_found"

    owner["id"] = 2
    for response in (
        client.get(accepted["status_url"]),
        client.get(accepted["items_url"]),
        client.delete(accepted["status_url"]),
        client.post(
            f"{accepted['status_url']}/materializations",
            json={"occurrence_ids": ["occ-1"]},
        ),
    ):
        assert response.status_code == 404
        assert response.json()["detail"] == "preflight_not_found"


def test_preflight_get_routes_require_media_read_and_keep_owner_identity_dependency():
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.api.v1.endpoints.media import playlist_ingest
    from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ

    routes = [
        route
        for route in playlist_ingest.router.routes
        if "GET" in (route.methods or set())
        and route.path
        in {
            "/playlist-preflights/{preflight_id}",
            "/playlist-preflights/{preflight_id}/items",
        }
    ]

    assert len(routes) == 2
    for route in routes:
        calls = [dependency.call for dependency in route.dependant.dependencies]
        assert get_request_user in calls
        assert any(MEDIA_READ in value for call in calls for value in _closure_values(call) if isinstance(value, list))
        assert any(getattr(call, "_tldw_rate_limit_resource", None) == "media.read" for call in calls)


@pytest.mark.parametrize("cursor", ["", "x" * 4097, "malformed-cursor"])
def test_preflight_invalid_cursors_are_generic_not_found_without_echo(preflight_api, cursor):
    client, manager, _clock, _owner = preflight_api
    accepted = _create_preflight(client, "PLinvalidcursor")
    _ready_snapshot(manager, "1", accepted["preflight_id"])

    response = client.get(accepted["items_url"], params={"cursor": cursor})

    assert response.status_code == 404
    assert response.json() == {"detail": "preflight_not_found"}
    if cursor:
        assert cursor[:100] not in response.text
    else:
        assert "input" not in response.text


def test_preflight_materialization_is_ready_only_and_returns_compact_authority(preflight_api):
    client, manager, _clock, _owner = preflight_api
    accepted = _create_preflight(client)
    materialize_url = f"{accepted['status_url']}/materializations"

    pending = client.post(materialize_url, json={"occurrence_ids": ["occ-1"]})
    assert pending.status_code == 409
    assert pending.json()["detail"] == "preflight_incomplete"

    _ready_snapshot(manager, "1", accepted["preflight_id"])
    rejected_extra = client.post(
        materialize_url,
        json={
            "occurrence_ids": ["occ-1"],
            "duplicate_policy": "overwrite",
            "source_url": "https://evil.example/client-authority",
        },
    )
    assert rejected_extra.status_code == 422
    assert "client-authority" not in rejected_extra.text

    response = client.post(materialize_url, json={"occurrence_ids": ["occ-1"]})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["contract_version"] == 2
    assert body["preflight_id"] == accepted["preflight_id"]
    assert body["status"] == "ready"
    assert body["items"] == [
        {
            "occurrence_id": "occ-1",
            "ordinal": 1,
            "source_url": "https://www.youtube.com/watch?v=video1",
            "normalized_source_id": "youtube:video:video1",
            "source_kind": "youtube_video",
            "display_metadata": {
                "title": "Opening keynote",
                "channel_or_uploader": "Conference",
                "playlist_title": "Conference 2010",
            },
        }
    ]
    assert "duplicate_status" not in response.text
    assert "internal_note" not in response.text


def test_preflight_delete_cancels_job_expires_resource_and_is_idempotent(preflight_api):
    client, manager, _clock, _owner = preflight_api
    accepted = _create_preflight(client)

    first = client.delete(accepted["status_url"])
    second = client.delete(accepted["status_url"])

    assert first.status_code == 204
    assert second.status_code == 204
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert jobs[0]["status"] == "cancelled"
    assert jobs[0]["cancellation_reason"] == "playlist_preflight_cancelled"
    assert client.get(accepted["status_url"]).status_code == 404

    with manager._connect() as connection:
        row = connection.execute(
            "SELECT status, expires_at FROM playlist_preflights WHERE preflight_id = ?",
            (accepted["preflight_id"],),
        ).fetchone()
    assert row["status"] == "cancelled"


def test_preflight_expiry_is_generic_not_found_and_does_not_leak_source(preflight_api):
    client, _manager, clock, _owner = preflight_api
    accepted = _create_preflight(client, "PLexpiry-secret")

    clock.advance(timedelta(seconds=601))
    response = client.get(accepted["status_url"])

    assert response.status_code == 404
    assert response.json()["detail"] == "preflight_not_found"
    assert "PLexpiry-secret" not in response.text


def test_preflight_capacity_is_reserved_transactionally_under_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY", "1")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_OWNER_CAPACITY", "1")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
        PlaylistPreflightBusyError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "capacity.db", clock=_Clock())
    barrier = threading.Barrier(2)

    def create() -> str:
        service = PlaylistIngestService(manager)
        barrier.wait()
        try:
            return service.create_preflight(
                "same-owner",
                url="https://www.youtube.com/playlist?list=PLcapacity",
                max_items=20,
                timeout_seconds=10,
            ).preflight_id
        except PlaylistPreflightBusyError:
            return "busy"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: create(), range(2)))

    assert results.count("busy") == 1
    assert len([value for value in results if value != "busy"]) == 1
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="same-owner", limit=10)) == 1


def test_preflight_enqueue_failure_expires_reservation_without_orphan(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
        PlaylistPreflightUnavailableError,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "enqueue-failure.db", clock=_Clock())

    def fail_enqueue(**_kwargs):
        raise RuntimeError("sqlite:///secret/path playlist=https://youtube.test/?token=secret")

    monkeypatch.setattr(manager, "create_job", fail_enqueue)
    with pytest.raises(PlaylistPreflightUnavailableError, match="preflight_unavailable"):
        PlaylistIngestService(manager).create_preflight(
            "owner",
            url="https://www.youtube.com/playlist?list=PLfailure",
            max_items=20,
            timeout_seconds=10,
        )

    with manager._connect() as connection:
        active = connection.execute(
            "SELECT COUNT(*) FROM playlist_preflights WHERE status IN ('pending', 'running')"
        ).fetchone()[0]
    assert active == 0
    assert manager.list_jobs(domain="media_ingest", limit=10) == []


def test_preflight_bind_failure_cancels_scheduled_job_and_expires_reservation(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_service import (
        PlaylistIngestService,
        PlaylistPreflightUnavailableError,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "bind-failure.db", clock=_Clock())

    def fail_bind(*_args, **_kwargs):
        raise RuntimeError("postgres://owner:password@db/internal")

    monkeypatch.setattr(PlaylistIngestStore, "bind_preflight_job", fail_bind)
    with pytest.raises(PlaylistPreflightUnavailableError, match="preflight_unavailable"):
        PlaylistIngestService(manager).create_preflight(
            "owner",
            url="https://www.youtube.com/playlist?list=PLbindfailure",
            max_items=20,
            timeout_seconds=10,
        )

    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="owner", limit=10)
    assert len(jobs) == 1
    assert jobs[0]["status"] == "cancelled"
    with manager._connect() as connection:
        active = connection.execute(
            "SELECT COUNT(*) FROM playlist_preflights WHERE status IN ('pending', 'running')"
        ).fetchone()[0]
    assert active == 0


def test_preflight_busy_and_blocked_errors_are_stable_and_sanitized(preflight_api, monkeypatch):
    client, manager, _clock, _owner = preflight_api
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY", "1")
    monkeypatch.setenv("PLAYLIST_PREFLIGHT_OWNER_CAPACITY", "1")
    accepted = _create_preflight(client, "PLbusy")

    busy = client.post(
        "/api/v1/media/playlist-preflights",
        json={"url": "https://www.youtube.com/playlist?list=PLbusy2"},
    )
    assert busy.status_code == 429
    assert busy.json()["detail"] == "preflight_busy"
    assert busy.headers["Retry-After"] == "5"

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    PlaylistIngestStore(manager).replace_preflight_snapshot(
        "1",
        accepted["preflight_id"],
        status="blocked",
        items=[],
        error={
            "code": "playlist_preflight_result_too_large",
            "raw_exception": "yt-dlp token=do-not-return",
            "source_url": "https://www.youtube.com/playlist?list=PLsecret",
        },
    )
    blocked = client.get(accepted["status_url"])
    assert blocked.status_code == 200
    assert blocked.json()["error"] == {"code": "playlist_too_large"}
    assert "do-not-return" not in blocked.text
    assert "PLsecret" not in blocked.text


def test_preflight_internal_route_error_does_not_echo_exception(preflight_api, monkeypatch):
    client, _manager, _clock, _owner = preflight_api

    from tldw_Server_API.app.api.v1.endpoints.media import playlist_ingest

    def fail(*_args, **_kwargs):
        raise RuntimeError("sqlite:///private.db?token=do-not-return")

    monkeypatch.setattr(playlist_ingest.PlaylistIngestService, "create_preflight", fail)
    response = client.post(
        "/api/v1/media/playlist-preflights",
        json={"url": "https://www.youtube.com/playlist?list=PLinternal"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "playlist_preflight_failed"
    assert "do-not-return" not in response.text


def test_preflight_delete_fences_a_claimed_worker_before_cancellation_race(preflight_api):
    client, manager, _clock, _owner = preflight_api
    accepted = _create_preflight(client, "PLcancelrace")
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        worker_id="cancel-race-worker",
        lease_seconds=120,
        job_type="playlist_preflight",
    )
    assert claimed is not None

    assert client.delete(accepted["status_url"]).status_code == 204

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestNotFoundError,
        PlaylistIngestStore,
        PlaylistPreflightLeaseLostError,
    )

    with pytest.raises((PlaylistIngestNotFoundError, PlaylistPreflightLeaseLostError)):
        PlaylistIngestStore(manager).replace_preflight_snapshot(
            "1",
            accepted["preflight_id"],
            status="ready",
            items=[],
            expected_job_id=int(claimed["id"]),
            expected_lease_id=str(claimed["lease_id"]),
            expected_worker_id=str(claimed["worker_id"]),
        )
    assert manager.get_job(int(claimed["id"]))["status"] == "cancelled"


def test_version_two_routes_are_advertised_separately_from_legacy(preflight_api, monkeypatch):
    client, _manager, _clock, _owner = preflight_api
    accepted = _create_preflight(client, "PLversion2")

    from tldw_Server_API.app.api.v1.endpoints.media import playlist_preflight
    from tldw_Server_API.app.api.v1.schemas.media_playlist_preflight import PlaylistPreflightResponse

    async def fake_legacy(_payload):
        return PlaylistPreflightResponse(
            source_url="https://www.youtube.com/playlist?list=PLlegacy",
            source_kind="youtube_playlist",
            playlist_id="PLlegacy",
            item_count=0,
            selected_count=0,
            duplicate_count=0,
            items=[],
        )

    monkeypatch.setattr(playlist_preflight, "_run_preflight_with_timeout", fake_legacy)
    legacy = client.post(
        "/api/v1/media/playlists/preflight",
        json={"url": "https://www.youtube.com/playlist?list=PLlegacy"},
    )

    assert accepted["contract_version"] == 2
    assert legacy.status_code == 200, legacy.text
    assert "contract_version" not in legacy.json()


def test_preflight_post_request_schemas_are_bounded_and_extra_forbid(preflight_api):
    client, _manager, _clock, _owner = preflight_api

    document = client.get("/openapi.json").json()
    preflight_schema = document["paths"]["/api/v1/media/playlist-preflights"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    materialization_schema = document["paths"]["/api/v1/media/playlist-preflights/{preflight_id}/materializations"][
        "post"
    ]["requestBody"]["content"]["application/json"]["schema"]

    assert preflight_schema["title"] == "PlaylistPreflightCreateRequest"
    assert preflight_schema["additionalProperties"] is False
    assert preflight_schema["properties"]["url"] == {
        "maxLength": 8192,
        "minLength": 1,
        "title": "Url",
        "type": "string",
    }
    assert preflight_schema["properties"]["max_items"]["maximum"] == 500
    assert preflight_schema["properties"]["timeout_seconds"]["maximum"] == 60
    assert materialization_schema["title"] == "PlaylistMaterializationCreateRequest"
    assert materialization_schema["additionalProperties"] is False
    assert materialization_schema["properties"]["occurrence_ids"]["minItems"] == 1
    assert materialization_schema["properties"]["occurrence_ids"]["maxItems"] == 500


def test_preflight_response_links_include_deployment_root_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
    from tldw_Server_API.app.api.v1.endpoints import media
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    manager = JobManager(db_path=tmp_path / "root-path.db")
    app = FastAPI(root_path="/deployment")
    app.include_router(media.router, prefix="/api/v1/media", tags=["media"])
    app.dependency_overrides[get_request_user] = lambda: User(
        id=1,
        username="root-path-owner",
        email=None,
        is_active=True,
    )
    app.dependency_overrides[get_job_manager] = lambda: manager

    with TestClient(
        app,
        base_url="http://testserver/deployment",
        headers={"X-API-KEY": "test-api-key-12345"},
    ) as client:
        response = client.post(
            "/api/v1/media/playlist-preflights",
            json={"url": "https://www.youtube.com/playlist?list=PLrootpath"},
        )

    assert response.status_code == 202, response.text
    assert response.json()["status_url"].startswith("/deployment/api/v1/media/playlist-preflights/")
    assert response.json()["items_url"] == f"{response.json()['status_url']}/items"


def test_run_post_resolves_terminal_actions_and_returns_authoritative_processing_occurrences(
    preflight_api,
    monkeypatch,
):
    client, _manager, _clock, _owner = preflight_api
    media_db = _install_endpoint_media_db(monkeypatch)
    media_db.rows = [{"id": 41, "url": "https://example.com/existing"}]

    response = client.post(
        "/api/v1/media/ingest/runs",
        json={
            "client_request_id": "endpoint-terminal-actions",
            "inputs": [
                {
                    "input_kind": "direct_url",
                    "occurrence_id": "occ-existing",
                    "url": "https://example.com/existing",
                    "display_metadata": {"title": "Existing"},
                },
                {
                    "input_kind": "direct_url",
                    "occurrence_id": "occ-new",
                    "url": "https://example.com/new",
                    "display_metadata": {"title": "New"},
                },
            ],
            "review_overrides": {
                "occ-existing": {
                    "duplicate_policy": "skip",
                }
            },
            "processing_options": {"media_type": "video"},
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["contract_version"] == 2
    assert body["status_url"].endswith(f"/ingest/runs/{body['run_id']}")
    assert body["items_url"] == f"{body['status_url']}/items"
    assert body["events_url"] == f"{body['status_url']}/events/stream"
    assert body["processing_occurrences"] == [
        {
            "occurrence_id": "occ-new",
            "ordinal": 2,
            "input_kind": "direct_url",
            "source_url": "https://example.com/new",
            "source_kind": "generic_url",
            "display_metadata": {"title": "New"},
            "state": "staged",
            "outcome": None,
            "job_id": None,
            "batch_id": None,
            "attempt": 1,
            "planned_collection_item_id": None,
        }
    ]

    summary = client.get(body["status_url"])
    assert summary.status_code == 200, summary.text
    assert summary.json()["counts"] == {
        "total": 2,
        "staged": 1,
        "terminal": 1,
        "skipped_existing": 1,
    }


def test_run_post_rejects_an_explicit_stale_duplicate_target(preflight_api, monkeypatch):
    client, _manager, _clock, _owner = preflight_api
    media_db = _install_endpoint_media_db(monkeypatch)
    media_db.rows = [{"id": 41, "url": "https://example.com/existing"}]

    response = client.post(
        "/api/v1/media/ingest/runs",
        json={
            "client_request_id": "endpoint-stale-duplicate-target",
            "inputs": [
                {
                    "input_kind": "direct_url",
                    "occurrence_id": "occ-existing",
                    "url": "https://example.com/existing",
                    "display_metadata": {"title": "Existing"},
                }
            ],
            "review_overrides": {
                "occ-existing": {
                    "duplicate_policy": "skip",
                    "existing_media_id": 999,
                }
            },
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "review_required",
        "items": [
            {
                "occurrence_id": "occ-existing",
                "reason": "duplicate_target_changed",
                "evidence": {
                    "kind": "library",
                    "existing_media_id": 41,
                    "duplicate_of_occurrence_id": None,
                },
                "allowed_actions": [
                    "skip",
                    "include_existing",
                    "update_metadata_only",
                    "overwrite",
                ],
            }
        ],
    }


def test_run_summary_and_items_are_owner_scoped_and_paginated(preflight_api, monkeypatch):
    client, _manager, _clock, owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-one", "occ-two", "occ-three")

    first = client.get(created["items_url"], params={"limit": 2})
    assert first.status_code == 200, first.text
    assert [item["occurrence_id"] for item in first.json()["items"]] == ["occ-one", "occ-two"]
    assert first.json()["version"] >= 2
    assert first.json()["next_cursor"]

    second = client.get(
        created["items_url"],
        params={"limit": 2, "cursor": first.json()["next_cursor"]},
    )
    assert second.status_code == 200, second.text
    assert [item["occurrence_id"] for item in second.json()["items"]] == ["occ-three"]
    assert second.json()["next_cursor"] is None

    owner["id"] = 2
    assert client.get(created["status_url"]).status_code == 404
    assert client.get(created["items_url"]).status_code == 404


def test_run_summary_returns_to_staged_after_only_bound_item_completes(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-bound", "occ-unsent")
    submitted = _submit_run_occurrences(client, created["run_id"], "occ-bound")
    job_id = int(submitted["submissions"][0]["job_id"])
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        worker_id="aggregate-status-worker",
        lease_seconds=120,
    )
    assert claimed is not None
    assert int(claimed["id"]) == job_id
    assert manager.complete_job(
        job_id,
        result={
            "run_id": created["run_id"],
            "occurrence_id": "occ-bound",
            "attempt": 1,
            "media_id": 55,
        },
        enforce=False,
    )

    response = client.get(created["status_url"])

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "staged"
    assert response.json()["counts"]["staged"] == 1
    assert response.json()["counts"]["terminal"] == 1


def test_run_occurrence_cancel_terminalizes_unsent_and_cancels_accepted_job(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-accepted", "occ-unsent", "occ-keep")
    submitted = _submit_run_occurrences(client, created["run_id"], "occ-accepted")
    accepted_job_id = int(submitted["submissions"][0]["job_id"])
    assert client.get(created["status_url"]).json()["status"] == "running"

    response = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-accepted", "occ-unsent"], "reason": "user_removed"},
    )

    assert response.status_code == 200, response.text
    assert manager.get_job(accepted_job_id)["status"] == "cancelled"
    items = {item["occurrence_id"]: item for item in client.get(created["items_url"]).json()["items"]}
    assert items["occ-accepted"]["state"] == "terminal"
    assert items["occ-accepted"]["outcome"] == "cancelled"
    assert items["occ-unsent"]["state"] == "terminal"
    assert items["occ-unsent"]["outcome"] == "cancelled"
    assert items["occ-keep"]["state"] == "staged"

    repeated = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-accepted", "occ-unsent"], "reason": "user_removed"},
    )
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["version"] == response.json()["version"]


def test_run_reconciliation_retries_transient_job_cancellation_failure(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-retry-cancel")
    submitted = _submit_run_occurrences(client, created["run_id"], "occ-retry-cancel")
    job_id = int(submitted["submissions"][0]["job_id"])
    original_cancel_job = manager.cancel_job
    attempts: list[tuple[int, str | None]] = []

    def transient_cancel_failure(cancel_job_id: int, *, reason: str | None = None) -> bool:
        attempts.append((cancel_job_id, reason))
        if len(attempts) <= 2:
            raise RuntimeError("transient Jobs write failure")
        return original_cancel_job(cancel_job_id, reason=reason)

    monkeypatch.setattr(manager, "cancel_job", transient_cancel_failure)

    response = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-retry-cancel"], "reason": "user_removed"},
    )

    assert response.status_code == 200, response.text
    with manager._connect() as connection:
        row = connection.execute(
            """
            SELECT state, outcome FROM media_ingest_run_items
            WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
            """,
            ("1", created["run_id"], "occ-retry-cancel"),
        ).fetchone()
    assert tuple(row) == ("cancellation_requested", None)
    assert manager.get_job(job_id)["status"] == "queued"
    assert attempts == [
        (job_id, "user_removed"),
        (job_id, "playlist_run_cancellation_retry"),
    ]

    summary = client.get(created["status_url"])

    assert summary.status_code == 200, summary.text
    assert attempts[-1] == (job_id, "playlist_run_cancellation_retry")
    assert len(attempts) == 3
    assert manager.get_job(job_id)["status"] == "cancelled"
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["state"] == "terminal"
    assert item["outcome"] == "cancelled"


def test_run_cancel_does_not_cancel_job_with_mismatched_payload_binding(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-mismatch")
    submitted = _submit_run_occurrences(client, created["run_id"], "occ-mismatch")
    job_id = int(submitted["submissions"][0]["job_id"])
    job = manager.get_job(job_id)
    binding = manager.normalize_job_binding_view(job, owner_user_id="1")
    assert binding is not None
    payload = dict(binding["payload"])
    payload["occurrence_id"] = "occ-other"
    connection = manager._connect()
    try:
        connection.execute("UPDATE jobs SET payload = ? WHERE id = ?", (json.dumps(payload), job_id))
        connection.commit()
    finally:
        connection.close()

    response = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-mismatch"]},
    )

    assert response.status_code == 200, response.text
    assert manager.get_job(job_id)["status"] == "queued"
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["state"] == "status_unavailable"
    assert item["outcome"] is None


def test_run_cancel_does_not_cancel_cross_owner_job_from_corrupt_stored_id(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-cross-owner")
    _submit_run_occurrences(client, created["run_id"], "occ-cross-owner")
    item = client.get(created["items_url"]).json()["items"][0]
    other_job = manager.create_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        payload={
            "run_id": created["run_id"],
            "occurrence_id": "occ-cross-owner",
            "attempt": 1,
        },
        owner_user_id="2",
        batch_group=item["batch_id"],
        idempotency_key="other-owner-job",
    )
    other_job_id = int(other_job["id"])
    connection = manager._connect()
    try:
        connection.execute(
            """
            UPDATE media_ingest_run_items SET job_id = ?
            WHERE owner_user_id = ? AND run_id = ? AND occurrence_id = ?
            """,
            (other_job_id, "1", created["run_id"], "occ-cross-owner"),
        )
        connection.commit()
    finally:
        connection.close()

    response = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-cross-owner"]},
    )

    assert response.status_code == 200, response.text
    assert manager.get_job(other_job_id)["status"] == "queued"
    current = client.get(created["items_url"]).json()["items"][0]
    assert current["state"] == "status_unavailable"
    assert current["outcome"] is None


def test_run_cancel_without_occurrences_cancels_whole_run(preflight_api, monkeypatch):
    client, _manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-one", "occ-two")

    response = client.post(f"{created['status_url']}/cancel", json={"reason": "stop_all"})

    assert response.status_code == 200, response.text
    items = client.get(created["items_url"]).json()["items"]
    assert {(item["state"], item["outcome"]) for item in items} == {("terminal", "cancelled")}


def test_run_cancel_without_body_cancels_whole_run(preflight_api, monkeypatch):
    client, _manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-one", "occ-two")

    response = client.post(f"{created['status_url']}/cancel")

    assert response.status_code == 200, response.text
    items = client.get(created["items_url"]).json()["items"]
    assert {(item["state"], item["outcome"]) for item in items} == {("terminal", "cancelled")}


def test_run_cancel_openapi_body_is_an_optional_object(preflight_api):
    client, _manager, _clock, _owner = preflight_api

    operation = client.get("/openapi.json").json()["paths"]["/api/v1/media/ingest/runs/{run_id}/cancel"]["post"]
    request_body = operation["requestBody"]
    schema = request_body["content"]["application/json"]["schema"]

    assert request_body["required"] is False
    assert schema["type"] == "object"


@pytest.mark.parametrize(
    ("request_kwargs", "expected_status", "expected_detail"),
    [
        ({"params": {"after_id": 2**63}}, 422, None),
        ({"headers": {"Last-Event-ID": str(2**63)}}, 400, "invalid_last_event_id"),
    ],
)
def test_run_sse_rejects_cursors_outside_database_integer_range(
    preflight_api,
    monkeypatch,
    request_kwargs,
    expected_status,
    expected_detail,
):
    client, _manager, _clock, _owner = preflight_api
    monkeypatch.setenv("PLAYLIST_RUN_SSE_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("PLAYLIST_RUN_SSE_TEST_MAX_SECONDS", "0.05")
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-cursor-range")

    response = client.get(created["events_url"], **request_kwargs)

    assert response.status_code == expected_status
    if expected_detail is not None:
        assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("", 1.0),
        ("invalid", 1.0),
        ("0", 1.0),
        ("-1", 1.0),
        ("nan", 1.0),
        ("inf", 1.0),
        ("-inf", 1.0),
        ("0.000001", 0.01),
        ("120", 60.0),
        ("0.25", 0.25),
    ],
)
def test_run_sse_poll_seconds_are_finite_positive_and_bounded(
    monkeypatch,
    raw_value,
    expected,
):
    from tldw_Server_API.app.api.v1.endpoints.media import playlist_ingest

    monkeypatch.setenv("TLDW_PLAYLIST_INGEST_SSE_POLL_SECONDS", raw_value)

    assert playlist_ingest._playlist_ingest_sse_poll_seconds() == expected


@pytest.mark.parametrize("payload", [[], False, 0, ""])
def test_run_cancel_rejects_falsy_non_object_body_without_mutation(
    preflight_api,
    monkeypatch,
    payload,
):
    client, _manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-one", "occ-two")

    response = client.post(f"{created['status_url']}/cancel", json=payload)

    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "invalid_run_cancel_request"}
    items = client.get(created["items_url"]).json()["items"]
    assert {(item["state"], item["outcome"]) for item in items} == {("staged", None)}


def test_run_cancel_allows_completed_job_to_win_race(preflight_api, monkeypatch):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-complete")
    _submit_run_occurrences(client, created["run_id"], "occ-complete")
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        worker_id="completion-wins",
        lease_seconds=120,
    )
    assert claimed is not None
    assert manager.complete_job(
        int(claimed["id"]),
        result={
            "run_id": created["run_id"],
            "occurrence_id": "occ-complete",
            "attempt": 1,
            "media_id": 91,
            "status": "Success",
        },
        enforce=False,
    )

    response = client.post(
        f"{created['status_url']}/cancel",
        json={"occurrence_ids": ["occ-complete"]},
    )

    assert response.status_code == 200, response.text
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["state"] == "terminal"
    assert item["outcome"] == "completed"
    assert item["media_id"] == 91


def test_run_summary_reports_status_unavailable_without_leaking_job_error(
    preflight_api,
    monkeypatch,
):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-lost")
    submitted = _submit_run_occurrences(client, created["run_id"], "occ-lost")
    job_id = int(submitted["submissions"][0]["job_id"])
    connection = manager._connect()
    try:
        connection.execute(
            "UPDATE jobs SET last_error = ?, error_message = ? WHERE id = ?",
            ("postgres://secret@internal", "token=do-not-return", job_id),
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(manager, "get_job_or_archived", lambda *_args, **_kwargs: None)
    summary = client.get(created["status_url"])

    assert summary.status_code == 200, summary.text
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["state"] == "status_unavailable"
    assert item["outcome"] is None
    assert "do-not-return" not in summary.text
    assert "secret" not in summary.text


def test_run_retry_resolves_media_before_incrementing_attempt(preflight_api, monkeypatch):
    client, manager, _clock, _owner = preflight_api
    media_db = _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-retry")
    _submit_run_occurrences(client, created["run_id"], "occ-retry")
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        worker_id="retry-failure",
        lease_seconds=120,
    )
    assert claimed is not None
    assert manager.fail_job(int(claimed["id"]), error="private failure", retryable=False, enforce=False)
    assert client.get(created["status_url"]).status_code == 200
    media_db.rows = [{"id": 77, "url": "https://example.com/occ-retry"}]

    response = client.post(
        f"{created['status_url']}/retry",
        json={"occurrence_ids": ["occ-retry"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["processing_occurrences"] == []
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["state"] == "terminal"
    assert item["outcome"] == "completed"
    assert item["media_id"] == 77
    assert item["attempt"] == 1
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)) == 1


def test_run_retry_cas_increments_once_and_clears_prior_job_mapping(preflight_api, monkeypatch):
    client, manager, _clock, _owner = preflight_api
    _install_endpoint_media_db(monkeypatch)
    created = _create_run(client, "occ-retry")
    _submit_run_occurrences(client, created["run_id"], "occ-retry")
    claimed = manager.acquire_next_job(
        domain="media_ingest",
        queue="default",
        job_type="media_ingest_item",
        worker_id="retry-failure",
        lease_seconds=120,
    )
    assert claimed is not None
    assert manager.fail_job(int(claimed["id"]), error="failed", retryable=False, enforce=False)
    assert client.get(created["status_url"]).status_code == 200

    first = client.post(
        f"{created['status_url']}/retry",
        json={"occurrence_ids": ["occ-retry"]},
    )
    second = client.post(
        f"{created['status_url']}/retry",
        json={"occurrence_ids": ["occ-retry"]},
    )

    assert first.status_code == 200, first.text
    assert first.json()["processing_occurrences"][0]["attempt"] == 2
    assert first.json()["processing_occurrences"][0]["job_id"] is None
    assert second.status_code == 200, second.text
    assert second.json()["processing_occurrences"] == []
    item = client.get(created["items_url"]).json()["items"][0]
    assert item["attempt"] == 2
    assert item["state"] == "staged"
    assert item["job_id"] is None
