from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavEvent
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables

pytestmark = pytest.mark.unit


@pytest.fixture
def calendar_db(tmp_path: Path) -> CalendarDatabase:
    db = CalendarDatabase(db_path=tmp_path / "calendar_sync.db")
    db.ensure_schema()
    return db


@pytest.fixture
def jobs_manager(tmp_path: Path) -> JobManager:
    db_path = tmp_path / "calendar_jobs.db"
    ensure_jobs_tables(db_path)
    return JobManager(db_path)


@pytest.fixture(autouse=True)
def calendar_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key = base64.b64encode(b"calendar-secret-store-key-32-bytes").decode("ascii")
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", key)


@dataclass
class _SyncFixture:
    calendar_id: int
    account_id: int
    binding_id: int
    secret_ref: str


class _FakeProvider:
    def __init__(self, *, events: list[CalDavEvent] | None = None, exc: Exception | None = None) -> None:
        self.events = events or []
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def fetch_vevents(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.events


def _create_sync_fixture(calendar_db: CalendarDatabase, *, owner_user_id: int = 1) -> _SyncFixture:
    calendar = calendar_db.create_calendar(
        tenant_id="default",
        owner_user_id=owner_user_id,
        org_id=None,
        name="Imported",
        timezone="UTC",
        color="#2563eb",
    )
    secret_ref = CalendarSecretStore(db=calendar_db, tenant_id="default").create_secret(
        owner_user_id=owner_user_id,
        provider="caldav",
        payload={
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=owner_user_id,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=secret_ref,
        account_metadata_json={
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
        },
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="https://caldav.example.test/calendars/user/work/",
        remote_display_name="Work",
        lookback_days=14,
        lookahead_days=30,
    )
    return _SyncFixture(
        calendar_id=calendar.id,
        account_id=account.id,
        binding_id=binding.id,
        secret_ref=secret_ref,
    )


def test_queue_binding_sync_creates_sanitized_jobs_payload(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    response = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    job = jobs_manager.get_job(response.job_id)
    assert response.queued is True
    assert response.status == "queued"
    assert job is not None
    assert job["domain"] == "calendar"
    assert job["queue"] == "default"
    assert job["job_type"] == "calendar_sync"
    assert job["owner_user_id"] == "1"
    assert job["idempotency_key"] == (
        f"calendar:sync:binding:{fixture.binding_id}:"
        "2026-06-01T00:00:00+00:00:2026-06-08T00:00:00+00:00:manual"
    )
    assert job["payload"] == {
        "binding_id": fixture.binding_id,
        "window_start": "2026-06-01T00:00:00+00:00",
        "window_end": "2026-06-08T00:00:00+00:00",
        "reason": "manual",
    }
    assert "app-secret" not in json.dumps(job["payload"])
    assert "secret_ref" not in json.dumps(job["payload"])


def test_queue_binding_sync_reuses_active_binding_job(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    second = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-07-01T00:00:00+00:00",
        window_end="2026-07-08T00:00:00+00:00",
    )

    assert second.job_id == first.job_id
    assert second.queued is False
    assert second.status == "already_active"
    assert jobs_manager.count_jobs(domain="calendar", queue="default", job_type="calendar_sync") == 1


@pytest.mark.asyncio
async def test_due_calendar_sync_scheduler_queues_sanitized_scheduled_job(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)

    queued = await queue_due_calendar_sync_jobs(
        db=calendar_db,
        job_manager=jobs_manager,
        now=datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc),
    )

    assert len(queued) == 1
    assert queued[0].binding_id == fixture.binding_id
    assert queued[0].queued is True
    job = jobs_manager.get_job(queued[0].job_id)
    assert job["payload"] == {
        "binding_id": fixture.binding_id,
        "window_start": "2026-05-27T12:00:00+00:00",
        "window_end": "2026-07-10T12:00:00+00:00",
        "reason": "scheduled",
    }
    assert "app-secret" not in json.dumps(job["payload"])
    assert "secret_ref" not in json.dumps(job["payload"])


@pytest.mark.asyncio
async def test_worker_resolves_credentials_and_upserts_provider_items_preserving_local_context(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    provider_item = calendar_db.upsert_provider_item(
        calendar_id=fixture.calendar_id,
        external_binding_id=fixture.binding_id,
        source_uid="event-1",
        title="Old title",
        start_at="2026-06-05T16:00:00+00:00",
        end_at="2026-06-05T17:00:00+00:00",
        source_etag="old-etag",
        source_ctag="old-ctag",
    )
    local_tags = CalendarService(db=calendar_db).update_local_tags(
        actor_user_id=1,
        item_id=provider_item.id,
        tags=["important"],
    )
    annotation = calendar_db.create_annotation(
        calendar_item_id=provider_item.id,
        author_user_id=1,
        body="Keep local note",
    )
    link = calendar_db.create_link(
        calendar_item_id=provider_item.id,
        target_type="note",
        target_id="note-1",
        label="Notes",
    )
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )
    job = jobs_manager.get_job(queued.job_id)
    provider = _FakeProvider(
        events=[
            CalDavEvent(
                uid="event-1",
                title="Updated title",
                start_at="2026-06-05T18:00:00+00:00",
                end_at="2026-06-05T19:00:00+00:00",
                location="Room 2",
                description="Remote description",
                source_updated_at="2026-06-04T12:00:00+00:00",
                provider_payload={"etag": "new-etag", "ctag": "new-ctag"},
            )
        ]
    )

    result = await handle_calendar_sync_job(job, db=calendar_db, provider=provider)

    updated = calendar_db.get_item(provider_item.id)
    assert result["items_seen"] == 1
    assert result["items_upserted"] == 1
    assert result["items_tombstoned"] == 0
    assert provider.calls[0]["password"] == "app-secret"
    assert provider.calls[0]["window_start"] == "2026-06-01T00:00:00+00:00"
    assert updated.title == "Updated title"
    assert updated.location == "Room 2"
    assert updated.source_etag == "new-etag"
    assert updated.source_ctag == "new-ctag"
    assert calendar_db.get_annotation(local_tags.id).tags_json == '["important"]'
    assert calendar_db.get_annotation(annotation.id).body == "Keep local note"
    assert calendar_db.get_link(link.id).label == "Notes"


@pytest.mark.asyncio
async def test_worker_tombstones_missing_remote_events_and_keeps_private_scope(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db, owner_user_id=1)
    stale_item = calendar_db.upsert_provider_item(
        calendar_id=fixture.calendar_id,
        external_binding_id=fixture.binding_id,
        source_uid="stale-event",
        title="Stale event",
        start_at="2026-06-05T16:00:00+00:00",
    )
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    await handle_calendar_sync_job(jobs_manager.get_job(queued.job_id), db=calendar_db, provider=_FakeProvider(events=[]))

    tombstoned = calendar_db.get_item(stale_item.id, include_deleted=True)
    assert tombstoned.remote_deleted_at is not None
    other_user_service = CalendarService(db=calendar_db)
    assert other_user_service.list_calendars(actor_user_id=2) == []


@pytest.mark.asyncio
async def test_worker_records_failure_and_updates_binding_error(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await handle_calendar_sync_job(
            jobs_manager.get_job(queued.job_id),
            db=calendar_db,
            provider=_FakeProvider(exc=RuntimeError("provider down")),
        )

    binding = calendar_db.get_external_binding(fixture.binding_id)
    events = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    assert binding.last_error == "provider down"
    assert events[0].status == "failed"
    assert events[0].error_message == "provider down"
