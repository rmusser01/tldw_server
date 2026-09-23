"""Prompt Studio job queue: leased work items that workers acquire, renew and finish.

Acquisition is the one place the backends need different SQL: PostgreSQL gates each
candidate with a transaction-scoped advisory lock so concurrent processes skip it, while
SQLite's BEGIN IMMEDIATE transaction already serialises writers, so select-then-update is
atomic there. Time arithmetic differs by dialect too; everything else is shared.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_RECLAIMABLE = "(status = 'queued' OR (status = 'processing' AND (leased_until IS NULL OR leased_until <= {now})))"


def lease_seconds() -> int:
    try:
        return max(1, min(3600, int(os.getenv("TLDW_PS_JOB_LEASE_SECONDS", "60"))))
    except ValueError:
        return 60


def _owner(worker_id: Optional[str]) -> Optional[str]:
    owner = str(worker_id).strip()[:128] if worker_id else ""
    return owner or None


def _metrics() -> Any:
    try:
        from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import prompt_studio_metrics
    except ImportError:
        return None
    return prompt_studio_metrics.metrics_manager


def _as_datetime(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


class JobsRepository:
    def __init__(self, session: Any):
        self.session = session

    # --- dialect helpers -------------------------------------------------------

    @property
    def _pg(self) -> bool:
        return self.session.backend_type == BackendType.POSTGRESQL

    def _now(self) -> str:
        return "NOW()" if self._pg else "CURRENT_TIMESTAMP"

    def _plus_seconds(self, base: Optional[str], seconds: int) -> str:
        """SQL for ``base`` (default: now) plus a whole number of seconds."""
        seconds = int(seconds)
        if self._pg:
            return f"{base or 'NOW()'} + INTERVAL '{seconds} seconds'"
        sqlite_base = base or "'now'"
        return f"DATETIME({sqlite_base}, '+{seconds} seconds')"

    def _run(self, fn: Any, failure: str) -> Any:
        try:
            return run_with_contention_retry(fn)
        except DB_ERRORS as exc:
            raise DatabaseError(f"{failure}: {exc}") from exc  # noqa: TRY003

    def _one(self, query: str, params: Any, failure: str) -> Optional[dict[str, Any]]:
        db = self.session

        def _read() -> Optional[dict[str, Any]]:
            cursor = db._execute(query, params)
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        return self._run(_read, failure)

    def _many(self, query: str, params: Any, failure: str) -> list[dict[str, Any]]:
        db = self.session

        def _read() -> list[dict[str, Any]]:
            cursor = db._execute(query, params)
            return [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]

        return self._run(_read, failure)

    # --- writes ----------------------------------------------------------------

    def create(
        self,
        job_type: str,
        entity_id: int,
        payload: Optional[Any],
        *,
        project_id: Optional[int] = None,
        priority: int = 5,
        status: str = "queued",
        max_retries: int = 3,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        params = (
            str(uuid.uuid4()), job_type, entity_id, project_id, priority, status,
            json.dumps(payload if payload is not None else {}), max_retries, client_id or db.client_id,
        )

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(
                    conn,
                    "INSERT INTO prompt_studio_job_queue ("
                    " uuid, job_type, entity_id, project_id, priority, status, payload, max_retries, client_id"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
                    params,
                )
                row = cursor.fetchone()
                if not row:
                    raise DatabaseError("Failed to create prompt studio job queue record")  # noqa: TRY003
                return db._row_to_dict(cursor, row)

        return self._run(_insert, "Failed to create job")

    def acquire_next(self, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Lease the highest-priority queued job, or one whose lease has expired."""
        db = self.session
        owner = _owner(worker_id)
        lease_until = self._plus_seconds(None, lease_seconds())
        metrics = _metrics()

        def _inc(name: str, labels: Optional[dict[str, str]] = None) -> None:
            if metrics is not None:
                with suppress(Exception):
                    metrics.increment(name, labels=labels)

        def _acquire_postgres(conn: Any) -> Optional[dict[str, Any]]:
            _inc("prompt_studio.pg_advisory.lock_attempts_total")
            cursor = db._cursor_exec(
                conn,
                f"""
                WITH candidate AS (
                    SELECT id, (status = 'processing') AS was_reclaim
                    FROM prompt_studio_job_queue
                    WHERE {_RECLAIMABLE.format(now="NOW()")}
                    ORDER BY priority DESC, created_at ASC, id ASC
                    LIMIT 10
                ), locked AS (
                    SELECT id, was_reclaim FROM candidate WHERE pg_try_advisory_xact_lock(id) LIMIT 1
                )
                UPDATE prompt_studio_job_queue AS q
                SET status = 'processing', started_at = CURRENT_TIMESTAMP,
                    leased_until = {lease_until}, lease_owner = COALESCE(?, q.lease_owner)
                FROM locked
                WHERE q.id = locked.id
                  AND (q.status = 'queued' OR (q.status = 'processing' AND (q.leased_until IS NULL OR q.leased_until <= NOW())))
                RETURNING q.*, locked.was_reclaim
                """,  # nosec B608 - lease interval is an int
                (owner,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            # The advisory lock is transaction-scoped and released on commit.
            _inc("prompt_studio.pg_advisory.locks_acquired_total")
            # Released with the transaction; counted to keep the lock/unlock series paired.
            _inc("prompt_studio.pg_advisory.unlocks_total")
            return db._row_to_dict(cursor, row)

        def _acquire_sqlite(conn: Any) -> Optional[dict[str, Any]]:
            # BEGIN IMMEDIATE holds the write lock from the SELECT on, so no other
            # writer can take this job between the two statements.
            candidate = db._cursor_exec(
                conn,
                f"SELECT id, status FROM prompt_studio_job_queue WHERE {_RECLAIMABLE.format(now='CURRENT_TIMESTAMP')}"  # nosec B608
                " ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1",
            ).fetchone()
            if not candidate:
                return None
            cursor = db._cursor_exec(
                conn,
                f"UPDATE prompt_studio_job_queue SET status = 'processing', started_at = CURRENT_TIMESTAMP,"  # nosec B608
                f" leased_until = {lease_until}, lease_owner = COALESCE(?, lease_owner) WHERE id = ?"
                # Re-check under the write lock: defence in depth if a caller's outer
                # transaction did not start with BEGIN IMMEDIATE.
                f" AND {_RECLAIMABLE.format(now='CURRENT_TIMESTAMP')} RETURNING *",
                (owner, candidate[0]),
            )
            row = cursor.fetchone()
            if not row:
                return None
            job = db._row_to_dict(cursor, row)
            job["was_reclaim"] = candidate[1] == "processing"
            return job

        def _acquire() -> Optional[dict[str, Any]]:
            with db._write_lock, db.transaction() as conn:
                return _acquire_postgres(conn) if self._pg else _acquire_sqlite(conn)

        job = self._run(_acquire, "Failed to acquire job")
        if job is None:
            return None
        job_type = str(job.get("job_type", ""))
        if job.pop("was_reclaim", False):
            _inc("jobs.reclaims_total", labels={"job_type": job_type})
        created, started = _as_datetime(job.get("created_at")), _as_datetime(job.get("started_at"))
        if metrics is not None and created and started:
            with suppress(Exception):
                metrics.observe(
                    "jobs.queue_latency_seconds",
                    max(0.0, (started - created).total_seconds()),
                    labels={"job_type": job_type},
                )
        return job

    def update_status(
        self,
        job_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        db = self.session
        updates = ["status = ?"]
        params: list[Any] = [status]
        if status == "processing":
            updates += ["started_at = CURRENT_TIMESTAMP", f"leased_until = {self._plus_seconds(None, lease_seconds())}"]
        elif status in TERMINAL_STATUSES:
            updates += ["completed_at = CURRENT_TIMESTAMP", "leased_until = NULL", "lease_owner = NULL"]
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))
        params.append(job_id)
        query = f"UPDATE prompt_studio_job_queue SET {', '.join(updates)} WHERE id = ? RETURNING *"  # nosec B608

        def _update() -> Optional[dict[str, Any]]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, query, params)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else None

        return self._run(_update, f"Failed to update job {job_id}")

    def renew_lease(self, job_id: int, seconds: int = 60, worker_id: Optional[str] = None) -> bool:
        """Extend a processing job's lease; with ``worker_id``, only if that worker (or no one) holds it."""
        try:
            seconds = max(1, min(3600, int(seconds)))
        except (TypeError, ValueError):
            seconds = 60
        db = self.session
        owner = _owner(worker_id)
        now = self._now()
        set_owner = ", lease_owner = COALESCE(?, lease_owner)" if owner is not None else ""
        owner_guard = " AND (lease_owner IS NULL OR lease_owner = ?)" if owner is not None else ""
        params = (owner, job_id, owner) if owner is not None else (job_id,)
        query = (
            "UPDATE prompt_studio_job_queue SET leased_until = CASE"  # nosec B608 - ints and fixed fragments
            f" WHEN leased_until IS NOT NULL AND leased_until > {now} THEN {self._plus_seconds('leased_until', seconds)}"
            f" ELSE {self._plus_seconds(None, seconds)} END{set_owner}"
            f" WHERE id = ? AND status = 'processing'{owner_guard} RETURNING id"
        )

        def _renew() -> bool:
            with db._write_lock, db.transaction() as conn:
                return db._cursor_exec(conn, query, params).fetchone() is not None

        return self._run(_renew, f"Failed to renew job lease for {job_id}")

    def retry(self, job_id: int) -> bool:
        """Requeue a job for another attempt, clearing its lease and last error."""
        db = self.session

        def _requeue() -> bool:
            with db._write_lock, db.transaction() as conn:
                return db._cursor_exec(
                    conn,
                    "UPDATE prompt_studio_job_queue SET status = 'queued', retry_count = retry_count + 1,"
                    " error_message = NULL, started_at = NULL, completed_at = NULL,"
                    " leased_until = NULL, lease_owner = NULL WHERE id = ? RETURNING id",
                    (job_id,),
                ).fetchone() is not None

        return self._run(_requeue, f"Failed to reschedule job {job_id}")

    def cleanup(self, older_than_days: int = 30) -> int:
        db = self.session
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        # Each backend compares against the format it stores CURRENT_TIMESTAMP in.
        cutoff_value = cutoff.isoformat() if self._pg else cutoff.strftime("%Y-%m-%d %H:%M:%S")

        def _delete() -> int:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(
                    conn,
                    "DELETE FROM prompt_studio_job_queue WHERE status IN ('completed', 'failed', 'cancelled')"
                    " AND completed_at IS NOT NULL AND completed_at < ?",
                    (cutoff_value,),
                )
                return max(int(cursor.rowcount or 0), 0)

        deleted = self._run(_delete, "Failed to clean up old jobs")
        if deleted:
            logger.info("Removed {} finished Prompt Studio jobs older than {} days", deleted, older_than_days)
        return deleted

    # --- reads -----------------------------------------------------------------

    def get(self, job_id: int) -> Optional[dict[str, Any]]:
        return self._one("SELECT * FROM prompt_studio_job_queue WHERE id = ?", (job_id,), f"Failed to fetch job {job_id}")

    def get_by_uuid(self, job_uuid: str) -> Optional[dict[str, Any]]:
        return self._one(
            "SELECT * FROM prompt_studio_job_queue WHERE uuid = ?", (job_uuid,), f"Failed to fetch job {job_uuid}"
        )

    def get_latest_for_entity(self, job_type: str, entity_id: int) -> Optional[dict[str, Any]]:
        return self._one(
            "SELECT * FROM prompt_studio_job_queue WHERE job_type = ? AND entity_id = ?"
            " ORDER BY created_at DESC, id DESC LIMIT 1",
            (job_type, entity_id),
            f"Failed fetching latest job for entity {entity_id}",
        )

    def list(self, *, status: Optional[str] = None, job_type: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return self._many(
            f"SELECT * FROM prompt_studio_job_queue{where} ORDER BY priority DESC, created_at ASC, id ASC LIMIT ?",  # nosec B608
            [*params, limit],
            "Failed to list prompt studio jobs",
        )

    def list_for_entity(
        self, job_type: str, entity_id: int, *, limit: int = 50, ascending: bool = True
    ) -> list[dict[str, Any]]:
        order = "ASC" if ascending else "DESC"
        return self._many(
            "SELECT * FROM prompt_studio_job_queue WHERE job_type = ? AND entity_id = ?"  # nosec B608
            f" ORDER BY created_at {order}, id {order} LIMIT ?",
            (job_type, entity_id, limit),
            f"Failed listing jobs for entity {entity_id}",
        )
