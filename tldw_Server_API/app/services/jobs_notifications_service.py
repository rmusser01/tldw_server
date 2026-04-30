from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, UserNotificationRow
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import env_flag_enabled

_JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_SUPPORTED_JOB_EVENTS = {"job.completed", "job.failed"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_bridge_state_user_id() -> int:
    raw = (os.getenv("JOBS_NOTIFICATIONS_BRIDGE_STATE_USER_ID") or "").strip()
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    try:
        return int(DatabasePaths.get_single_user_id())
    except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
        return 1


def _parse_attrs(attrs_json: Any) -> dict[str, Any]:
    if isinstance(attrs_json, dict):
        return attrs_json
    if isinstance(attrs_json, str):
        try:
            parsed = json.loads(attrs_json)
            return parsed if isinstance(parsed, dict) else {}
        except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
            return {}
    return {}


def _parse_owner_user_id(owner_value: Any) -> int | None:
    if owner_value is None:
        return None
    raw = str(owner_value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class JobsNotificationsService:
    def __init__(
        self,
        *,
        consumer_name: str = "jobs_notifications_bridge",
        lease_owner_id: str | None = None,
        lease_seconds: int = 30,
        poll_batch_size: int = 200,
        poll_interval_seconds: float = 1.0,
        bridge_state_user_id: int | None = None,
    ) -> None:
        self.consumer_name = consumer_name
        self.lease_owner_id = lease_owner_id or f"jobs-notifications-{os.getpid()}"
        self.lease_seconds = max(5, int(lease_seconds))
        self.poll_batch_size = max(1, min(500, int(poll_batch_size)))
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.bridge_state_user_id = int(
            bridge_state_user_id if bridge_state_user_id is not None else _default_bridge_state_user_id()
        )
        self._jobs = JobManager()

    async def process_event(self, event: dict[str, Any]) -> UserNotificationRow | None:
        event_type = str(event.get("event_type") or "")
        if event_type not in _SUPPORTED_JOB_EVENTS:
            return None

        owner_user_id = _parse_owner_user_id(event.get("owner_user_id"))
        if owner_user_id is None:
            logger.warning(
                "jobs_notifications_bridge: unresolved owner, skipping event_id={} type={}",
                event.get("id"),
                event_type,
            )
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "jobs_notifications_bridge", "event": "unresolved_owner"},
                )
            except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            return None

        attrs = _parse_attrs(event.get("attrs_json"))
        source_domain = str(event.get("domain") or "jobs")
        source_job_type = str(event.get("job_type") or "unknown")
        source_job_id = event.get("job_id")
        source_job_id_text = str(source_job_id) if source_job_id is not None else None

        if event_type == "job.completed":
            kind = "job_completed"
            title = "Job completed"
            message = f"{source_domain}/{source_job_type} completed successfully."
            severity = "info"
        else:
            kind = "job_failed"
            title = "Job failed"
            error_code = attrs.get("error_code")
            if error_code:
                message = f"{source_domain}/{source_job_type} failed ({error_code})."
            else:
                message = f"{source_domain}/{source_job_type} failed."
            severity = "error"

        dedupe_key = f"jobs-event:{event.get('id')}"
        with CollectionsDatabase.for_user(user_id=owner_user_id) as cdb:
            prefs = cdb.get_notification_preferences()
            if kind == "job_completed" and not prefs.job_completed_enabled:
                return None
            if kind == "job_failed" and not prefs.job_failed_enabled:
                return None
            return cdb.create_user_notification(
                kind=kind,
                title=title,
                message=message,
                severity=severity,
                source_job_id=source_job_id_text,
                source_domain=source_domain,
                source_job_type=source_job_type,
                dedupe_key=dedupe_key,
            )

    def _fetch_events_after(self, *, after_id: int, limit: int) -> list[dict[str, Any]]:
        return self._jobs.list_job_events_after(
            after_id=after_id,
            limit=limit,
            event_types=("job.completed", "job.failed"),
        )

    async def run_once(self) -> dict[str, int | bool]:
        now_dt = _utcnow()
        now_iso = now_dt.isoformat()
        lease_expires_at = (now_dt + timedelta(seconds=self.lease_seconds)).isoformat()

        with CollectionsDatabase.for_user(user_id=self.bridge_state_user_id) as state_db:
            claimed = state_db.try_claim_notification_bridge_lease(
                consumer_name=self.consumer_name,
                lease_owner_id=self.lease_owner_id,
                lease_expires_at=lease_expires_at,
                now_iso=now_iso,
            )
            if not claimed:
                return {
                    "claimed": False,
                    "processed": 0,
                    "notifications_created": 0,
                    "skipped": 0,
                    "failed": 0,
                    "cursor": 0,
                }
            state = state_db.get_notification_bridge_state(consumer_name=self.consumer_name)
            after_id = max(0, int(state.last_event_id or 0))

        try:
            JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None)
        except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            rows = self._fetch_events_after(after_id=after_id, limit=self.poll_batch_size)
        finally:
            try:
                JobManager.clear_rls_context()
            except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
                pass

        processed = 0
        created = 0
        skipped = 0
        failed = 0
        committed_cursor = after_id

        for row in rows:
            event_id = int(row.get("id") or 0)
            if event_id <= 0:
                continue
            try:
                notif = await self.process_event(row)
            except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS as exc:
                failed += 1
                logger.warning(
                    "jobs_notifications_bridge: failed processing event_id={} type={} exception_type={}",
                    event_id,
                    row.get("event_type"),
                    type(exc).__name__,
                )
                try:
                    get_metrics_registry().increment(
                        "app_exception_events_total",
                        labels={"component": "jobs_notifications_bridge", "event": "process_event_failed"},
                    )
                except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS:
                    pass
                break

            processed += 1
            if notif is None:
                skipped += 1
            else:
                created += 1
            committed_cursor = event_id

        with CollectionsDatabase.for_user(user_id=self.bridge_state_user_id) as state_db:
            state_db.update_notification_bridge_state(
                consumer_name=self.consumer_name,
                last_event_id=committed_cursor,
                lease_owner_id=self.lease_owner_id,
                lease_expires_at=lease_expires_at,
            )

        return {
            "claimed": True,
            "processed": processed,
            "notifications_created": created,
            "skipped": skipped,
            "failed": failed,
            "cursor": committed_cursor,
        }

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        logger.info("Starting Jobs Notifications bridge service")
        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stopping Jobs Notifications bridge service on shutdown signal")
                return
            try:
                summary = await self.run_once()
                sleep_s = self.poll_interval_seconds
                if summary.get("processed", 0):
                    sleep_s = min(0.05, self.poll_interval_seconds)
                await asyncio.sleep(sleep_s)
            except _JOBS_NOTIFICATIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning("jobs_notifications_bridge loop error exception_type={}", type(exc).__name__)
                await asyncio.sleep(self.poll_interval_seconds)


async def start_jobs_notifications_service() -> asyncio.Task | None:
    """Start the jobs-to-notifications bridge service.

    The bridge is ON by default.  It can be disabled by setting
    ``JOBS_NOTIFICATIONS_BRIDGE_DISABLED=true`` (preferred) or by
    explicitly setting ``JOBS_NOTIFICATIONS_BRIDGE_ENABLED=false``
    (legacy, kept for backward compatibility).
    """
    _bridge_var = os.getenv("JOBS_NOTIFICATIONS_BRIDGE_ENABLED")
    _bridge_disabled = env_flag_enabled("JOBS_NOTIFICATIONS_BRIDGE_DISABLED")
    if _bridge_disabled or (
        _bridge_var is not None
        and not env_flag_enabled("JOBS_NOTIFICATIONS_BRIDGE_ENABLED")
    ):
        return None
    service = JobsNotificationsService(
        consumer_name=(os.getenv("JOBS_NOTIFICATIONS_CONSUMER_NAME") or "jobs_notifications_bridge").strip(),
        lease_owner_id=(os.getenv("JOBS_NOTIFICATIONS_LEASE_OWNER_ID") or None),
        lease_seconds=int(os.getenv("JOBS_NOTIFICATIONS_LEASE_SECONDS", "30") or "30"),
        poll_batch_size=int(os.getenv("JOBS_NOTIFICATIONS_BATCH_SIZE", "200") or "200"),
        poll_interval_seconds=float(os.getenv("JOBS_NOTIFICATIONS_POLL_INTERVAL_SEC", "1.0") or "1.0"),
    )
    return asyncio.create_task(service.run_forever(), name="jobs_notifications_bridge")


__all__ = ["JobsNotificationsService", "start_jobs_notifications_service"]
