from __future__ import annotations

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
    app.dependency_overrides[get_request_user] = current_user
    app.dependency_overrides[get_job_manager] = lambda: manager
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
