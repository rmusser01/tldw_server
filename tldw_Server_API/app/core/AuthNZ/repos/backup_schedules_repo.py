from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict


_SQLITE_BACKUP_SCHEDULES_DDL = (
    """
    CREATE TABLE IF NOT EXISTS backup_schedules (
        id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        target_user_id INTEGER,
        target_scope_key TEXT NOT NULL,
        frequency TEXT NOT NULL,
        time_of_day TEXT NOT NULL,
        timezone TEXT NOT NULL,
        anchor_day_of_week INTEGER,
        anchor_day_of_month INTEGER,
        retention_count INTEGER NOT NULL,
        is_paused INTEGER NOT NULL DEFAULT 0,
        created_by_user_id INTEGER,
        updated_by_user_id INTEGER,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        next_run_at TIMESTAMP,
        last_run_at TIMESTAMP,
        last_status TEXT,
        last_job_id TEXT,
        last_error TEXT,
        deleted_at TIMESTAMP
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_backup_schedules_target_scope_active
    ON backup_schedules(target_scope_key)
    WHERE deleted_at IS NULL
    """,
    "CREATE INDEX IF NOT EXISTS idx_backup_schedules_next_run_at ON backup_schedules(next_run_at)",
    "CREATE INDEX IF NOT EXISTS idx_backup_schedules_deleted_at ON backup_schedules(deleted_at)",
    """
    CREATE TABLE IF NOT EXISTS backup_schedule_runs (
        id TEXT PRIMARY KEY,
        schedule_id TEXT NOT NULL,
        scheduled_for TIMESTAMP NOT NULL,
        run_slot_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        job_id TEXT,
        error TEXT,
        enqueued_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        FOREIGN KEY (schedule_id) REFERENCES backup_schedules(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_schedule_id ON backup_schedule_runs(schedule_id)",
    "CREATE INDEX IF NOT EXISTS idx_backup_schedule_runs_scheduled_for ON backup_schedule_runs(scheduled_for)",
)


@dataclass
class AuthnzBackupSchedulesRepo:
    """Repository for platform-level backup schedule records and run claims."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema(self) -> None:
        """Ensure backup schedule tables exist for the active backend."""
        try:
            if self._is_postgres_backend():
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_backup_schedules_tables_pg,
                )

                ok = await ensure_backup_schedules_tables_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL backup schedule schema ensure failed")
                return

            for statement in _SQLITE_BACKUP_SCHEDULES_DDL:
                await self.db_pool.execute(statement)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.ensure_schema failed")
            raise

    @staticmethod
    def _target_scope_key(dataset: str, target_user_id: int | None) -> str:
        """Return the unique active-schedule scope key for a dataset target."""
        if target_user_id is None:
            return f"{dataset}::global"
        return f"{dataset}::user::{int(target_user_id)}"

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Normalize backend-specific row objects to plain JSON-friendly dicts."""
        data = row_dict(row)

        for text_id_field in ("id", "schedule_id", "last_job_id", "job_id"):
            if data.get(text_id_field) is not None:
                data[text_id_field] = str(data[text_id_field])

        for int_field in (
            "target_user_id",
            "anchor_day_of_week",
            "anchor_day_of_month",
            "retention_count",
            "created_by_user_id",
            "updated_by_user_id",
        ):
            if data.get(int_field) is not None:
                try:
                    data[int_field] = int(data[int_field])
                except (TypeError, ValueError):
                    pass

        if "is_paused" in data and data["is_paused"] is not None:
            data["is_paused"] = bool(data["is_paused"])

        for ts_field in (
            "created_at",
            "updated_at",
            "next_run_at",
            "last_run_at",
            "deleted_at",
            "scheduled_for",
            "enqueued_at",
            "started_at",
            "completed_at",
        ):
            value = data.get(ts_field)
            if isinstance(value, datetime):
                data[ts_field] = value.isoformat()

        return data

    @staticmethod
    def _command_touched_rows(result: Any) -> bool:
        """Return True when an UPDATE/DELETE command affected at least one row."""
        if isinstance(result, str):
            parts = result.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1]) > 0
            return True
        rowcount = getattr(result, "rowcount", None)
        if isinstance(rowcount, int):
            return rowcount != 0
        return True

    def _normalize_timestamp(self, value: str | None) -> str | datetime | None:
        """Normalize timestamps for the active backend."""
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if self._is_postgres_backend():
            return parsed
        return parsed.isoformat()

    async def create_schedule(
        self,
        *,
        dataset: str,
        target_user_id: int | None,
        frequency: str,
        time_of_day: str,
        timezone: str,
        anchor_day_of_week: int | None,
        anchor_day_of_month: int | None,
        retention_count: int,
        created_by_user_id: int | None,
        updated_by_user_id: int | None,
        next_run_at: str | None,
    ) -> dict[str, Any]:
        """Persist a new backup schedule row and return the stored record."""
        schedule_id = str(uuid4())
        now = datetime.now(dt_timezone.utc)
        created_at = now if self._is_postgres_backend() else now.isoformat()
        next_run_param = self._normalize_timestamp(next_run_at)
        target_scope_key = self._target_scope_key(dataset, target_user_id)
        try:
            await self.db_pool.execute(
                """
                INSERT INTO backup_schedules (
                    id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                    anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                    created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                    last_run_at, last_status, last_job_id, last_error, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    schedule_id,
                    dataset,
                    target_user_id,
                    target_scope_key,
                    frequency,
                    time_of_day,
                    timezone,
                    anchor_day_of_week,
                    anchor_day_of_month,
                    int(retention_count),
                    False if self._is_postgres_backend() else 0,
                    created_by_user_id,
                    updated_by_user_id,
                    created_at,
                    created_at,
                    next_run_param,
                ),
            )
            row = await self.get_schedule(schedule_id, include_deleted=True)
            return row or {}
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.create_schedule failed")
            raise

    async def get_schedule(
        self,
        schedule_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a backup schedule by id."""
        try:
            if include_deleted:
                query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    WHERE id = ?
                    """
            else:
                query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    WHERE id = ? AND deleted_at IS NULL
                    """
            row = await self.db_pool.fetchone(query, (schedule_id,))
            return self._row_to_dict(row) if row else None
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.get_schedule failed")
            raise

    async def list_schedules(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool = False,
        exclude_authnz: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a page of backup schedules and the total count."""
        try:
            if include_deleted and exclude_authnz:
                count_query = "SELECT COUNT(*) as total FROM backup_schedules WHERE dataset != ?"
                list_query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    WHERE dataset != ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """
                count_params: tuple[Any, ...] = ("authnz",)
                list_params: tuple[Any, ...] = ("authnz", int(limit), int(offset))
            elif include_deleted:
                count_query = "SELECT COUNT(*) as total FROM backup_schedules"
                list_query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """
                count_params = ()
                list_params = (int(limit), int(offset))
            elif exclude_authnz:
                count_query = """
                    SELECT COUNT(*) as total
                    FROM backup_schedules
                    WHERE deleted_at IS NULL AND dataset != ?
                    """
                list_query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    WHERE deleted_at IS NULL AND dataset != ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """
                count_params = ("authnz",)
                list_params = ("authnz", int(limit), int(offset))
            else:
                count_query = """
                    SELECT COUNT(*) as total
                    FROM backup_schedules
                    WHERE deleted_at IS NULL
                    """
                list_query = """
                    SELECT id, dataset, target_user_id, target_scope_key, frequency, time_of_day, timezone,
                           anchor_day_of_week, anchor_day_of_month, retention_count, is_paused,
                           created_by_user_id, updated_by_user_id, created_at, updated_at, next_run_at,
                           last_run_at, last_status, last_job_id, last_error, deleted_at
                    FROM backup_schedules
                    WHERE deleted_at IS NULL
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ? OFFSET ?
                    """
                count_params = ()
                list_params = (int(limit), int(offset))

            total = int(await self.db_pool.fetchval(count_query, count_params) or 0)
            rows = await self.db_pool.fetchall(list_query, list_params)
            return [self._row_to_dict(row) for row in rows], total
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.list_schedules failed")
            raise

    async def update_schedule(
        self,
        schedule_id: str,
        *,
        frequency: str | None = None,
        time_of_day: str | None = None,
        timezone: str | None = None,
        anchor_day_of_week: int | None = None,
        anchor_day_of_month: int | None = None,
        retention_count: int | None = None,
        updated_by_user_id: int | None = None,
        next_run_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Update mutable schedule fields and return the stored row."""
        current = await self.get_schedule(schedule_id, include_deleted=True)
        if not current or current.get("deleted_at") is not None:
            return None

        updated_at = datetime.now(dt_timezone.utc)
        updated_at_param = updated_at if self._is_postgres_backend() else updated_at.isoformat()
        next_run_param = self._normalize_timestamp(next_run_at) if next_run_at is not None else current.get("next_run_at")

        try:
            result = await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET frequency = ?,
                    time_of_day = ?,
                    timezone = ?,
                    anchor_day_of_week = ?,
                    anchor_day_of_month = ?,
                    retention_count = ?,
                    updated_by_user_id = ?,
                    updated_at = ?,
                    next_run_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (
                    frequency if frequency is not None else current.get("frequency"),
                    time_of_day if time_of_day is not None else current.get("time_of_day"),
                    timezone if timezone is not None else current.get("timezone"),
                    anchor_day_of_week if anchor_day_of_week is not None else current.get("anchor_day_of_week"),
                    anchor_day_of_month if anchor_day_of_month is not None else current.get("anchor_day_of_month"),
                    int(retention_count) if retention_count is not None else current.get("retention_count"),
                    updated_by_user_id,
                    updated_at_param,
                    next_run_param,
                    schedule_id,
                ),
            )
            if not self._command_touched_rows(result):
                return None
            return await self.get_schedule(schedule_id, include_deleted=True)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.update_schedule failed")
            raise

    async def pause_schedule(self, schedule_id: str, *, updated_by_user_id: int | None) -> dict[str, Any] | None:
        """Pause a schedule and return the stored row."""
        return await self._set_paused_state(schedule_id, is_paused=True, updated_by_user_id=updated_by_user_id)

    async def resume_schedule(self, schedule_id: str, *, updated_by_user_id: int | None) -> dict[str, Any] | None:
        """Resume a paused schedule and return the stored row."""
        return await self._set_paused_state(schedule_id, is_paused=False, updated_by_user_id=updated_by_user_id)

    async def _set_paused_state(
        self,
        schedule_id: str,
        *,
        is_paused: bool,
        updated_by_user_id: int | None,
    ) -> dict[str, Any] | None:
        updated_at = datetime.now(dt_timezone.utc)
        updated_at_param = updated_at if self._is_postgres_backend() else updated_at.isoformat()
        try:
            result = await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET is_paused = ?, updated_by_user_id = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (
                    is_paused if self._is_postgres_backend() else int(is_paused),
                    updated_by_user_id,
                    updated_at_param,
                    schedule_id,
                ),
            )
            if not self._command_touched_rows(result):
                return None
            return await self.get_schedule(schedule_id, include_deleted=True)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo._set_paused_state failed")
            raise

    async def delete_schedule(self, schedule_id: str, *, deleted_at: str) -> bool:
        """Soft-delete a schedule row so active uniqueness no longer applies."""
        deleted_param = self._normalize_timestamp(deleted_at)
        try:
            result = await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET deleted_at = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (deleted_param, deleted_param, schedule_id),
            )
            return self._command_touched_rows(result)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.delete_schedule failed")
            raise

    async def claim_run_slot(
        self,
        *,
        schedule_id: str,
        scheduled_for: str,
        run_slot_key: str,
        enqueued_at: str,
    ) -> dict[str, Any] | None:
        """Claim a unique schedule fire slot for enqueueing."""
        run_id = str(uuid4())
        scheduled_for_param = self._normalize_timestamp(scheduled_for)
        enqueued_at_param = self._normalize_timestamp(enqueued_at)
        try:
            if self._is_postgres_backend():
                row = await self.db_pool.fetchone(
                    """
                    INSERT INTO backup_schedule_runs (
                        id, schedule_id, scheduled_for, run_slot_key, status, job_id, error,
                        enqueued_at, started_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL)
                    ON CONFLICT (run_slot_key) DO NOTHING
                    RETURNING id, schedule_id, scheduled_for, run_slot_key, status, job_id, error,
                              enqueued_at, started_at, completed_at
                    """,
                    (
                        run_id,
                        schedule_id,
                        scheduled_for_param,
                        run_slot_key,
                        "queued",
                        enqueued_at_param,
                    ),
                )
                return self._row_to_dict(row) if row else None

            result = await self.db_pool.execute(
                """
                INSERT OR IGNORE INTO backup_schedule_runs (
                    id, schedule_id, scheduled_for, run_slot_key, status, job_id, error,
                    enqueued_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL)
                """,
                (
                    run_id,
                    schedule_id,
                    scheduled_for_param,
                    run_slot_key,
                    "queued",
                    enqueued_at_param,
                ),
            )
            if not self._command_touched_rows(result):
                return None
            row = await self.db_pool.fetchone(
                """
                SELECT id, schedule_id, scheduled_for, run_slot_key, status, job_id, error,
                       enqueued_at, started_at, completed_at
                FROM backup_schedule_runs
                WHERE run_slot_key = ?
                """,
                (run_slot_key,),
            )
            return self._row_to_dict(row) if row else None
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.claim_run_slot failed")
            raise

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a claimed run row by id."""
        try:
            row = await self.db_pool.fetchone(
                """
                SELECT id, schedule_id, scheduled_for, run_slot_key, status, job_id, error,
                       enqueued_at, started_at, completed_at
                FROM backup_schedule_runs
                WHERE id = ?
                """,
                (run_id,),
            )
            return self._row_to_dict(row) if row else None
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.get_run failed")
            raise

    async def mark_run_queued(
        self,
        *,
        run_id: str,
        job_id: str,
        next_run_at: str | None,
        last_run_at: str,
    ) -> dict[str, Any] | None:
        """Persist queued metadata for a claimed run and its parent schedule."""
        now = datetime.now(dt_timezone.utc)
        now_param = now if self._is_postgres_backend() else now.isoformat()
        next_run_param = self._normalize_timestamp(next_run_at)
        last_run_param = self._normalize_timestamp(last_run_at)
        try:
            run_row = await self.get_run(run_id)
            if not run_row:
                return None
            await self.db_pool.execute(
                """
                UPDATE backup_schedule_runs
                SET status = ?, job_id = ?, enqueued_at = ?
                WHERE id = ?
                """,
                ("queued", str(job_id), now_param, run_id),
            )
            await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET next_run_at = ?,
                    last_run_at = ?,
                    last_status = ?,
                    last_job_id = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    next_run_param,
                    last_run_param,
                    "queued",
                    str(job_id),
                    now_param,
                    str(run_row["schedule_id"]),
                ),
            )
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.mark_run_queued failed")
            raise

    async def mark_run_running(self, *, run_id: str) -> dict[str, Any] | None:
        """Persist running metadata for a claimed run and its parent schedule."""
        now = datetime.now(dt_timezone.utc)
        now_param = now if self._is_postgres_backend() else now.isoformat()
        try:
            run_row = await self.get_run(run_id)
            if not run_row:
                return None
            await self.db_pool.execute(
                """
                UPDATE backup_schedule_runs
                SET status = ?, started_at = ?
                WHERE id = ?
                """,
                ("running", now_param, run_id),
            )
            await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET last_status = ?, updated_at = ?
                WHERE id = ?
                """,
                ("running", now_param, str(run_row["schedule_id"])),
            )
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.mark_run_running failed")
            raise

    async def mark_run_succeeded(self, *, run_id: str) -> dict[str, Any] | None:
        """Persist success metadata for a claimed run and its parent schedule."""
        now = datetime.now(dt_timezone.utc)
        now_param = now if self._is_postgres_backend() else now.isoformat()
        try:
            run_row = await self.get_run(run_id)
            if not run_row:
                return None
            await self.db_pool.execute(
                """
                UPDATE backup_schedule_runs
                SET status = ?, error = NULL, completed_at = ?
                WHERE id = ?
                """,
                ("succeeded", now_param, run_id),
            )
            await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET last_status = ?, last_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                ("succeeded", now_param, str(run_row["schedule_id"])),
            )
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.mark_run_succeeded failed")
            raise

    async def mark_run_failed(
        self,
        *,
        run_id: str,
        error: str,
        last_status: str = "failed",
        next_run_at: str | None = None,
        last_run_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist failure metadata for a claimed run and its parent schedule."""
        now = datetime.now(dt_timezone.utc)
        now_param = now if self._is_postgres_backend() else now.isoformat()
        error_text = str(error or "").strip() or "unknown_error"
        next_run_param = self._normalize_timestamp(next_run_at)
        last_run_param = self._normalize_timestamp(last_run_at)
        try:
            run_row = await self.get_run(run_id)
            if not run_row:
                return None
            await self.db_pool.execute(
                """
                UPDATE backup_schedule_runs
                SET status = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                (last_status, error_text, now_param, run_id),
            )
            await self.db_pool.execute(
                """
                UPDATE backup_schedules
                SET next_run_at = COALESCE(?, next_run_at),
                    last_run_at = COALESCE(?, last_run_at),
                    last_status = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    next_run_param,
                    last_run_param,
                    last_status,
                    error_text,
                    now_param,
                    str(run_row["schedule_id"]),
                ),
            )
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzBackupSchedulesRepo.mark_run_failed failed")
            raise
