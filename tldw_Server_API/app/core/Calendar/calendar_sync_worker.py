"""Jobs-backed Calendar external sync queueing and worker handlers."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Calendar.errors import CalendarPermissionDenied, CalendarValidationError
from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider, sanitize_provider_metadata
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarDatabase,
    ExternalCalendarAccountRow,
    ExternalCalendarBindingRow,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

CALENDAR_SYNC_DOMAIN = "calendar"
CALENDAR_SYNC_QUEUE = "default"
CALENDAR_SYNC_JOB_TYPE = "calendar_sync"
_ACTIVE_JOB_STATUSES = ("queued", "processing")


@dataclass(frozen=True)
class CalendarSyncJobResponse:
    binding_id: int
    job_id: int
    queued: bool
    status: str
    idempotency_key: str


def build_calendar_sync_payload(
    *,
    binding_id: int,
    window_start: str,
    window_end: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "binding_id": int(binding_id),
        "window_start": str(window_start),
        "window_end": str(window_end),
        "reason": str(reason),
    }


def build_calendar_sync_idempotency_key(
    *,
    binding_id: int,
    window_start: str,
    window_end: str,
    reason: str,
) -> str:
    return f"calendar:sync:binding:{int(binding_id)}:{window_start}:{window_end}:{reason}"


def queue_calendar_binding_sync(
    *,
    db: CalendarDatabase,
    job_manager: JobManager,
    actor_user_id: int,
    tenant_id: str,
    binding_id: int,
    reason: str,
    window_start: str,
    window_end: str,
) -> CalendarSyncJobResponse:
    binding = db.get_external_binding(binding_id)
    account = db.get_external_account(binding.account_id)
    _assert_account_scope(account, actor_user_id=actor_user_id, tenant_id=tenant_id)
    if not binding.sync_enabled or binding.disabled_at:
        raise CalendarValidationError("External calendar binding is not enabled for sync")

    idempotency_key = build_calendar_sync_idempotency_key(
        binding_id=binding.id,
        window_start=window_start,
        window_end=window_end,
        reason=reason,
    )
    existing = _active_job_for_binding(
        job_manager=job_manager,
        owner_user_id=str(actor_user_id),
        binding_id=binding.id,
    )
    if existing is not None:
        return CalendarSyncJobResponse(
            binding_id=binding.id,
            job_id=int(existing["id"]),
            queued=False,
            status="already_active",
            idempotency_key=str(existing.get("idempotency_key") or idempotency_key),
        )

    job = job_manager.create_job(
        domain=CALENDAR_SYNC_DOMAIN,
        queue=CALENDAR_SYNC_QUEUE,
        job_type=CALENDAR_SYNC_JOB_TYPE,
        owner_user_id=str(actor_user_id),
        payload=build_calendar_sync_payload(
            binding_id=binding.id,
            window_start=window_start,
            window_end=window_end,
            reason=reason,
        ),
        idempotency_key=idempotency_key,
    )
    db.record_sync_event(
        binding_id=binding.id,
        event_type="sync_queued",
        status=str(job.get("status") or "queued"),
        metadata_json={"job_id": int(job["id"]), "reason": reason},
    )
    return CalendarSyncJobResponse(
        binding_id=binding.id,
        job_id=int(job["id"]),
        queued=str(job.get("status") or "queued") == "queued",
        status=str(job.get("status") or "queued"),
        idempotency_key=idempotency_key,
    )


async def handle_calendar_sync_job(
    job: dict[str, Any] | None,
    *,
    db: CalendarDatabase | None = None,
    provider: Any | None = None,
) -> dict[str, Any]:
    if job is None:
        raise CalendarValidationError("Calendar sync job is missing")
    if job.get("job_type") != CALENDAR_SYNC_JOB_TYPE:
        raise CalendarValidationError("Unsupported Calendar sync job type")

    payload = _coerce_payload(job.get("payload"))
    binding_id = int(payload["binding_id"])
    window_start = str(payload["window_start"])
    window_end = str(payload["window_end"])
    reason = str(payload.get("reason") or "scheduled")
    calendar_db = db or CalendarDatabase()
    binding = calendar_db.get_external_binding(binding_id)
    account = calendar_db.get_external_account(binding.account_id)
    if account.provider.lower() != "caldav":
        raise CalendarValidationError(f"Unsupported external calendar provider: {account.provider}")

    started_at = _utcnow_iso()
    try:
        if account.status != "active" or account.revoked_at or account.deleted_at:
            raise CalendarValidationError("External calendar account is not active")
        credentials = _resolve_sync_credentials(calendar_db, account=account, binding=binding)
        sync_provider = provider or CalDavProvider()
        events = await _maybe_await(
            sync_provider.fetch_vevents(
                remote_calendar_url=binding.remote_calendar_id,
                username=credentials["username"],
                password=credentials["password"],
                window_start=window_start,
                window_end=window_end,
            )
        )
        result = _upsert_and_tombstone_events(
            calendar_db,
            binding=binding,
            events=list(events or []),
        )
        finished_at = _utcnow_iso()
        calendar_db.update_binding_sync_state(
            binding.id,
            last_sync_at=finished_at,
            next_scan_at=_next_scan_at(binding, finished_at),
            last_error=None,
        )
        calendar_db.record_sync_event(
            binding_id=binding.id,
            event_type="sync",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            items_seen=result["items_seen"],
            items_upserted=result["items_upserted"],
            items_tombstoned=result["items_tombstoned"],
            metadata_json={"reason": reason, "job_id": job.get("id")},
        )
        return result
    except Exception as exc:
        finished_at = _utcnow_iso()
        calendar_db.update_binding_sync_state(binding.id, last_error=str(exc))
        calendar_db.record_sync_event(
            binding_id=binding.id,
            event_type="sync",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            error_message=str(exc),
            metadata_json={"reason": reason, "job_id": job.get("id")},
        )
        raise


async def run_calendar_sync_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("CALENDAR_SYNC_WORKER_ID") or f"calendar-sync-worker-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=CALENDAR_SYNC_DOMAIN,
        queue=CALENDAR_SYNC_QUEUE,
        worker_id=worker_id,
        lease_seconds=int(os.getenv("CALENDAR_SYNC_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"),
        renew_threshold_seconds=int(os.getenv("CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("CALENDAR_SYNC_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    job_manager = JobManager()
    sdk = WorkerSDK(job_manager, cfg)
    stop_waiter = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_waiter = asyncio.create_task(_watch_stop(), name="calendar_sync_worker_stop_waiter")

    logger.info("Calendar sync Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(handler=handle_calendar_sync_job)
    finally:
        if stop_waiter is not None:
            stop_waiter.cancel()


def _active_job_for_binding(
    *,
    job_manager: JobManager,
    owner_user_id: str,
    binding_id: int,
) -> dict[str, Any] | None:
    for job_status in _ACTIVE_JOB_STATUSES:
        rows = job_manager.list_jobs(
            domain=CALENDAR_SYNC_DOMAIN,
            queue=CALENDAR_SYNC_QUEUE,
            job_type=CALENDAR_SYNC_JOB_TYPE,
            owner_user_id=owner_user_id,
            status=job_status,
            limit=100,
        )
        for row in rows:
            try:
                payload = _coerce_payload(row.get("payload"))
            except CalendarValidationError:
                continue
            if int(payload.get("binding_id") or 0) == int(binding_id):
                return row
    return None


def _assert_account_scope(
    account: ExternalCalendarAccountRow,
    *,
    actor_user_id: int,
    tenant_id: str,
) -> None:
    if account.user_id != int(actor_user_id) or account.tenant_id != tenant_id:
        raise CalendarPermissionDenied("External calendar account is outside the current user scope")
    if account.status != "active" or account.revoked_at or account.deleted_at:
        raise CalendarValidationError("External calendar account is not active")


def _coerce_payload(raw_payload: Any) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise CalendarValidationError("Calendar sync job payload is not valid JSON") from exc
        if isinstance(decoded, dict):
            return decoded
    raise CalendarValidationError("Calendar sync job payload must be an object")


def _resolve_sync_credentials(
    db: CalendarDatabase,
    *,
    account: ExternalCalendarAccountRow,
    binding: ExternalCalendarBindingRow,
) -> dict[str, str]:
    metadata = _json_dict(account.account_metadata_json)
    stored_secret: dict[str, Any] = {}
    if account.secret_ref:
        stored_secret = CalendarSecretStore(db=db, tenant_id=account.tenant_id).resolve_secret(
            owner_user_id=account.user_id,
            secret_ref=account.secret_ref,
        )

    username = stored_secret.get("username") or metadata.get("username")
    password = stored_secret.get("password") or stored_secret.get("token")
    server_url = stored_secret.get("server_url") or metadata.get("server_url") or binding.remote_calendar_id
    if not username or not password or not server_url:
        raise CalendarValidationError("CalDAV sync requires server_url, username, and password/token")
    return {
        "server_url": str(server_url),
        "username": str(username),
        "password": str(password),
    }


def _upsert_and_tombstone_events(
    db: CalendarDatabase,
    *,
    binding: ExternalCalendarBindingRow,
    events: list[Any],
) -> dict[str, int]:
    seen_uids: set[str] = set()
    upserted = 0
    for event in events:
        uid = str(_event_value(event, "uid") or "").strip()
        if not uid:
            continue
        seen_uids.add(uid)
        provider_payload = _provider_payload(event)
        db.upsert_provider_item(
            calendar_id=binding.calendar_id,
            external_binding_id=binding.id,
            source_uid=uid,
            title=str(_event_value(event, "title") or "Untitled event"),
            start_at=_none_or_str(_event_value(event, "start_at")),
            end_at=_none_or_str(_event_value(event, "end_at")),
            due_at=_none_or_str(_event_value(event, "due_at")),
            kind=str(_event_value(event, "kind") or "event"),
            description=_none_or_str(_event_value(event, "description")),
            location=_none_or_str(_event_value(event, "location")),
            timezone=_none_or_str(_event_value(event, "timezone")),
            all_day=bool(_event_value(event, "all_day") or False),
            status=str(_event_value(event, "status") or "confirmed"),
            provider_payload_json=provider_payload,
            source_etag=_none_or_str(provider_payload.get("etag")),
            source_ctag=_none_or_str(provider_payload.get("ctag")),
            source_updated_at=_none_or_str(_event_value(event, "source_updated_at")),
        )
        upserted += 1

    tombstoned = 0
    for existing in db.list_provider_items_for_binding(binding.id):
        if existing.source_uid and existing.source_uid not in seen_uids:
            db.mark_provider_item_remote_deleted(
                external_binding_id=binding.id,
                source_uid=existing.source_uid,
            )
            tombstoned += 1
    return {
        "items_seen": len(seen_uids),
        "items_upserted": upserted,
        "items_tombstoned": tombstoned,
    }


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _provider_payload(event: Any) -> dict[str, Any]:
    payload = _event_value(event, "provider_payload")
    if isinstance(payload, dict):
        return sanitize_provider_metadata(payload)
    return {}


def _json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _next_scan_at(binding: ExternalCalendarBindingRow, synced_at: str) -> str | None:
    if not binding.sync_interval_minutes:
        return None
    parsed = datetime.fromisoformat(synced_at.replace("Z", "+00:00"))
    return (parsed + timedelta(minutes=int(binding.sync_interval_minutes))).astimezone(timezone.utc).isoformat()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CALENDAR_SYNC_DOMAIN",
    "CALENDAR_SYNC_JOB_TYPE",
    "CALENDAR_SYNC_QUEUE",
    "CalendarSyncJobResponse",
    "build_calendar_sync_idempotency_key",
    "build_calendar_sync_payload",
    "handle_calendar_sync_job",
    "queue_calendar_binding_sync",
    "run_calendar_sync_worker",
]
