from __future__ import annotations

import contextlib
import json
import os
import time
from sqlite3 import Error as SQLiteError
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

from .audit_bridge import submit_job_audit_event

try:
    import psycopg  # type: ignore
    _PSYCOPG_EXCEPTIONS: tuple[type[Exception], ...] = (psycopg.Error,)
except ImportError:
    _PSYCOPG_EXCEPTIONS = ()

_EVENT_AUDIT_EXCEPTIONS = (
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_EVENT_LOG_EXCEPTIONS = (OSError, RuntimeError, TypeError, ValueError)
_OUTBOX_NONCRITICAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
) + _PSYCOPG_EXCEPTIONS


def _events_enabled() -> bool:
    return is_truthy(os.getenv("JOBS_EVENTS_ENABLED"))


def observe_job_event(
    event: str,
    *,
    job: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Run the non-durable audit and logging observers for one job event."""

    try:
        submit_job_audit_event(event, job=job, attrs=attrs)
    except _EVENT_AUDIT_EXCEPTIONS:
        # Audit integration is best-effort. Errors should never break job flow.
        pass
    meta = {}
    if job:
        for k in ("id", "uuid", "domain", "queue", "job_type", "status"):
            if k in job:
                meta[k] = job.get(k)
    if attrs:
        meta.update(attrs)
    if _events_enabled():
        with contextlib.suppress(_EVENT_LOG_EXCEPTIONS):
            logger.bind(job_event=True).info(f"job_event event={event} attrs={meta}")


def emit_job_event(
    event: str,
    *,
    job: dict[str, Any] | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Observe an event and append it to the durable outbox when enabled."""

    observe_job_event(event, job=job, attrs=attrs)
    # Admission persists job.created in the same transaction as the job row.
    # This emitter only supplies its best-effort audit/log observers for that event.
    if event == "job.created":
        return
    if not is_truthy(os.getenv("JOBS_EVENTS_OUTBOX")):
        return
    # Outbox write (append-only) when enabled
    # Optional soft rate-limit for extremely high churn
    try:
        if is_truthy(os.getenv("JOBS_EVENTS_OUTBOX")):
            # Basic rate limiter: drop writes if exceeding JOBS_EVENTS_RATE_LIMIT_HZ
            try:
                hz = float(os.getenv("JOBS_EVENTS_RATE_LIMIT_HZ", "0") or "0")
            except (TypeError, ValueError):
                hz = 0.0
            if hz > 0:
                now = time.time()
                last = _rate_state.get("last_ts", 0.0)
                min_interval = 1.0 / max(0.0001, hz)
                # Always allow admin sweep/maintenance events
                is_admin_ev = event.startswith("jobs.")
                if not is_admin_ev and (now - last) < min_interval:
                    return
                _rate_state["last_ts"] = now
            # Fast path: avoid re-running schema DDL while a transaction on jobs table is active
            # Attempt direct connection using JOBS_DB_URL when Postgres is configured.
            _db_url = os.getenv("JOBS_DB_URL", "").strip()
            if _db_url.startswith("postgres"):
                try:
                    from .pg_util import negotiate_pg_dsn
                    _dsn = negotiate_pg_dsn(_db_url)
                    with psycopg.connect(_dsn) as _conn:
                        with _conn.cursor() as _cur:
                            _cur.execute(
                                (
                                    "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at) "
                                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW())"
                                ),
                                (
                                    (job or {}).get("id"),
                                    (job or {}).get("domain"),
                                    (job or {}).get("queue"),
                                    (job or {}).get("job_type"),
                                    event,
                                    json.dumps(attrs or {}),
                                    (job or {}).get("owner_user_id"),
                                    (job or {}).get("request_id"),
                                    (job or {}).get("trace_id"),
                                ),
                            )
                            _conn.commit()
                    return
                except _OUTBOX_NONCRITICAL_EXCEPTIONS:
                    # Fall back to JobManager-based path if direct insert fails
                    pass

            from tldw_Server_API.app.core.Jobs.manager import JobManager
            # Admin context for outbox writes (RLS bypass)
            with contextlib.suppress(_OUTBOX_NONCRITICAL_EXCEPTIONS):
                JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None)
            jm = JobManager()
            conn = jm._connect()
            try:
                if jm.backend == "postgres":
                    with jm._pg_cursor(conn) as cur:
                        cur.execute(
                            (
                                "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at) "
                                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW())"
                            ),
                            (
                                (job or {}).get("id"),
                                (job or {}).get("domain"),
                                (job or {}).get("queue"),
                                (job or {}).get("job_type"),
                                event,
                                json.dumps(attrs or {}),
                                (job or {}).get("owner_user_id"),
                                (job or {}).get("request_id"),
                                (job or {}).get("trace_id"),
                            ),
                        )
                        conn.commit()
                else:
                    conn.execute(
                        (
                            "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))"
                        ),
                        (
                            (job or {}).get("id"),
                            (job or {}).get("domain"),
                            (job or {}).get("queue"),
                            (job or {}).get("job_type"),
                            event,
                            json.dumps(attrs or {}),
                            (job or {}).get("owner_user_id"),
                            (job or {}).get("request_id"),
                            (job or {}).get("trace_id"),
                        ),
                    )
                    conn.commit()
            finally:
                with contextlib.suppress(_OUTBOX_NONCRITICAL_EXCEPTIONS):
                    conn.close()
                with contextlib.suppress(_OUTBOX_NONCRITICAL_EXCEPTIONS):
                    JobManager.clear_rls_context()
    except _OUTBOX_NONCRITICAL_EXCEPTIONS:
        # Swallow outbox errors; logging already occurred
        pass

# Module-level state for soft rate limiter
_rate_state: dict[str, float] = {}
