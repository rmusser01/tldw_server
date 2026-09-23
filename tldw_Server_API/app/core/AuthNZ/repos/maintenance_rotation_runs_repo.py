from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict


_SQLITE_MAINTENANCE_ROTATION_RUNS_DDL = (
    """
    CREATE TABLE IF NOT EXISTS maintenance_rotation_runs (
        id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        domain TEXT,
        queue TEXT,
        job_type TEXT,
        fields_json TEXT NOT NULL,
        "limit" INTEGER,
        affected_count INTEGER,
        requested_by_user_id INTEGER,
        requested_by_label TEXT,
        confirmation_recorded INTEGER NOT NULL DEFAULT 0,
        job_id TEXT,
        scope_summary TEXT NOT NULL,
        key_source TEXT NOT NULL,
        error_message TEXT,
        created_at TIMESTAMP NOT NULL,
        started_at TIMESTAMP,
        completed_at TIMESTAMP
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_active_execute
    ON maintenance_rotation_runs(mode)
    WHERE mode = 'execute' AND status IN ('queued', 'running')
    """,
    "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_created_at ON maintenance_rotation_runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_maintenance_rotation_runs_status ON maintenance_rotation_runs(status)",
)


@dataclass
class AuthnzMaintenanceRotationRunsRepo:
    """Repository for authoritative maintenance rotation run records."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema(self) -> None:
        """Ensure maintenance rotation run tables exist for the active backend."""
        try:
            if self._is_postgres_backend():
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_maintenance_rotation_runs_table_pg,
                )

                ok = await ensure_maintenance_rotation_runs_table_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL maintenance rotation runs schema ensure failed")
                return

            for statement in _SQLITE_MAINTENANCE_ROTATION_RUNS_DDL:
                await self.db_pool.execute(statement)
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.ensure_schema failed")
            raise

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Normalize backend-specific row objects to plain JSON-friendly dicts."""
        data = row_dict(row)

        for text_id_field in ("id", "job_id"):
            if data.get(text_id_field) is not None:
                data[text_id_field] = str(data[text_id_field])

        for int_field in ("limit", "affected_count", "requested_by_user_id"):
            if data.get(int_field) is not None:
                try:
                    data[int_field] = int(data[int_field])
                except (TypeError, ValueError):
                    pass

        if "confirmation_recorded" in data and data["confirmation_recorded"] is not None:
            data["confirmation_recorded"] = bool(data["confirmation_recorded"])

        for ts_field in ("created_at", "started_at", "completed_at"):
            value = data.get(ts_field)
            if isinstance(value, datetime):
                data[ts_field] = value.isoformat()

        return data

    @staticmethod
    def _command_touched_rows(result: Any) -> bool:
        """Return True when an UPDATE command affected at least one row."""
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

    async def create_run(
        self,
        *,
        mode: str,
        domain: str | None,
        queue: str | None,
        job_type: str | None,
        fields_json: str,
        limit: int | None,
        requested_by_user_id: int | None,
        requested_by_label: str | None,
        confirmation_recorded: bool,
        scope_summary: str,
        key_source: str,
    ) -> dict[str, Any]:
        """Persist a new maintenance rotation run and return the stored record."""
        run_id = str(uuid4())
        now = datetime.now(dt_timezone.utc)
        created_at = now if self._is_postgres_backend() else now.isoformat()
        confirmation_value: bool | int = (
            confirmation_recorded if self._is_postgres_backend() else int(confirmation_recorded)
        )
        try:
            await self.db_pool.execute(
                """
                INSERT INTO maintenance_rotation_runs (
                    id, mode, status, domain, queue, job_type, fields_json, "limit",
                    affected_count, requested_by_user_id, requested_by_label,
                    confirmation_recorded, job_id, scope_summary, key_source,
                    error_message, created_at, started_at, completed_at
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, NULL, ?, NULL, NULL)
                """,
                (
                    run_id,
                    mode,
                    domain,
                    queue,
                    job_type,
                    fields_json,
                    limit,
                    requested_by_user_id,
                    requested_by_label,
                    confirmation_value,
                    scope_summary,
                    key_source,
                    created_at,
                ),
            )
            row = await self.get_run(run_id)
            return row or {}
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.create_run failed")
            raise

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a maintenance rotation run by id."""
        try:
            row = await self.db_pool.fetchone(
                """
                SELECT id, mode, status, domain, queue, job_type, fields_json, "limit",
                       affected_count, requested_by_user_id, requested_by_label,
                       confirmation_recorded, job_id, scope_summary, key_source,
                       error_message, created_at, started_at, completed_at
                FROM maintenance_rotation_runs
                WHERE id = ?
                """,
                (run_id,),
            )
            return self._row_to_dict(row) if row else None
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.get_run failed")
            raise

    async def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        allowed_domains: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List maintenance rotation runs newest-first with total count."""
        if allowed_domains is not None and len(allowed_domains) == 0:
            return [], 0

        try:
            if allowed_domains is not None:
                allowed_domains_token = f",{','.join(allowed_domains)},"
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, mode, status, domain, queue, job_type, fields_json, "limit",
                           affected_count, requested_by_user_id, requested_by_label,
                           confirmation_recorded, job_id, scope_summary, key_source,
                           error_message, created_at, started_at, completed_at
                    FROM maintenance_rotation_runs
                    WHERE domain IS NOT NULL AND ? LIKE ('%,' || domain || ',%')
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (allowed_domains_token, limit, offset),
                )
                count_row = await self.db_pool.fetchone(
                    """
                    SELECT COUNT(*) AS total
                    FROM maintenance_rotation_runs
                    WHERE domain IS NOT NULL AND ? LIKE ('%,' || domain || ',%')
                    """,
                    (allowed_domains_token,),
                )
            else:
                rows = await self.db_pool.fetchall(
                    """
                    SELECT id, mode, status, domain, queue, job_type, fields_json, "limit",
                           affected_count, requested_by_user_id, requested_by_label,
                           confirmation_recorded, job_id, scope_summary, key_source,
                           error_message, created_at, started_at, completed_at
                    FROM maintenance_rotation_runs
                    ORDER BY created_at DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                count_row = await self.db_pool.fetchone(
                    "SELECT COUNT(*) AS total FROM maintenance_rotation_runs"
                )
            total = 0
            if count_row is not None:
                try:
                    total = int(count_row["total"])
                except Exception:
                    total = int(list(count_row)[0])
            return [self._row_to_dict(row) for row in rows], total
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.list_runs failed")
            raise

    async def mark_running(self, run_id: str, *, job_id: str | None) -> bool:
        """Transition a run to running and record the job id."""
        started_at = datetime.now(dt_timezone.utc)
        started_at_value = started_at if self._is_postgres_backend() else started_at.isoformat()
        try:
            result = await self.db_pool.execute(
                """
                UPDATE maintenance_rotation_runs
                SET status = 'running',
                    job_id = ?,
                    started_at = ?
                WHERE id = ?
                """,
                (job_id, started_at_value, run_id),
            )
            return self._command_touched_rows(result)
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.mark_running failed")
            raise

    async def mark_complete(self, run_id: str, *, affected_count: int | None) -> bool:
        """Transition a run to complete and persist its affected count."""
        completed_at = datetime.now(dt_timezone.utc)
        completed_at_value = completed_at if self._is_postgres_backend() else completed_at.isoformat()
        try:
            result = await self.db_pool.execute(
                """
                UPDATE maintenance_rotation_runs
                SET status = 'complete',
                    affected_count = ?,
                    error_message = NULL,
                    completed_at = ?
                WHERE id = ?
                """,
                (affected_count, completed_at_value, run_id),
            )
            return self._command_touched_rows(result)
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.mark_complete failed")
            raise

    async def mark_failed(self, run_id: str, *, error_message: str) -> bool:
        """Transition a run to failed and record a terminal error message."""
        completed_at = datetime.now(dt_timezone.utc)
        completed_at_value = completed_at if self._is_postgres_backend() else completed_at.isoformat()
        try:
            result = await self.db_pool.execute(
                """
                UPDATE maintenance_rotation_runs
                SET status = 'failed',
                    error_message = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (error_message, completed_at_value, run_id),
            )
            return self._command_touched_rows(result)
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.mark_failed failed")
            raise

    async def has_active_execute_run(self) -> bool:
        """Return True when an execute-mode run is queued or running."""
        try:
            row = await self.db_pool.fetchone(
                """
                SELECT COUNT(*) AS active_count
                FROM maintenance_rotation_runs
                WHERE mode = 'execute' AND status IN ('queued', 'running')
                """
            )
            if row is None:
                return False
            try:
                return int(row["active_count"]) > 0
            except Exception:
                return int(list(row)[0]) > 0
        except Exception:
            logger.error("AuthnzMaintenanceRotationRunsRepo.has_active_execute_run failed")
            raise
