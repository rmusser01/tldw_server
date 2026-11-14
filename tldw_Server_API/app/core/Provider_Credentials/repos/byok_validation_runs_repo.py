from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos._dual_backend import row_dict


_SQLITE_BYOK_VALIDATION_RUNS_DDL = (
    """
    CREATE TABLE IF NOT EXISTS byok_validation_runs (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        org_id INTEGER,
        provider TEXT,
        keys_checked INTEGER,
        valid_count INTEGER,
        invalid_count INTEGER,
        error_count INTEGER,
        requested_by_user_id INTEGER,
        requested_by_label TEXT,
        job_id TEXT,
        scope_summary TEXT NOT NULL,
        error_message TEXT,
        created_at TIMESTAMP NOT NULL,
        started_at TIMESTAMP,
        completed_at TIMESTAMP
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_byok_validation_runs_active
    ON byok_validation_runs((1))
    WHERE status IN ('queued', 'running')
    """,
    "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_created_at ON byok_validation_runs(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_byok_validation_runs_status ON byok_validation_runs(status)",
)


@dataclass
class AuthnzByokValidationRunsRepo:
    """Repository for authoritative BYOK validation run records."""

    db_pool: DatabasePool

    def _is_postgres_backend(self) -> bool:
        """Return True when the underlying DatabasePool is using PostgreSQL."""
        return bool(getattr(self.db_pool, "pool", None))

    async def ensure_schema(self) -> None:
        """Ensure BYOK validation run tables exist for the active backend."""
        try:
            if self._is_postgres_backend():
                from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
                    ensure_byok_validation_runs_table_pg,
                )

                ok = await ensure_byok_validation_runs_table_pg(self.db_pool)
                if not ok:
                    raise RuntimeError("PostgreSQL BYOK validation runs schema ensure failed")
                return

            for statement in _SQLITE_BYOK_VALIDATION_RUNS_DDL:
                await self.db_pool.execute(statement)
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.ensure_schema failed")
            raise

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Normalize backend-specific row objects to plain JSON-friendly dicts."""
        data = row_dict(row)

        for text_id_field in ("id", "job_id"):
            if data.get(text_id_field) is not None:
                data[text_id_field] = str(data[text_id_field])

        for int_field in (
            "org_id",
            "keys_checked",
            "valid_count",
            "invalid_count",
            "error_count",
            "requested_by_user_id",
        ):
            if data.get(int_field) is not None:
                try:
                    data[int_field] = int(data[int_field])
                except (TypeError, ValueError):
                    pass

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
        org_id: int | None,
        provider: str | None,
        requested_by_user_id: int | None,
        requested_by_label: str | None,
        scope_summary: str,
    ) -> dict[str, Any]:
        """Persist a new BYOK validation run and return the stored record."""
        run_id = str(uuid4())
        now = datetime.now(dt_timezone.utc)
        created_at = now if self._is_postgres_backend() else now.isoformat()
        try:
            await self.db_pool.execute(
                """
                INSERT INTO byok_validation_runs (
                    id, status, org_id, provider, keys_checked, valid_count, invalid_count,
                    error_count, requested_by_user_id, requested_by_label, job_id,
                    scope_summary, error_message, created_at, started_at, completed_at
                ) VALUES (?, 'queued', ?, ?, NULL, NULL, NULL, NULL, ?, ?, NULL, ?, NULL, ?, NULL, NULL)
                """,
                (
                    run_id,
                    org_id,
                    provider,
                    requested_by_user_id,
                    requested_by_label,
                    scope_summary,
                    created_at,
                ),
            )
            row = await self.get_run(run_id)
            return row or {}
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.create_run failed")
            raise

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a BYOK validation run by id."""
        try:
            row = await self.db_pool.fetchone(
                """
                SELECT id, status, org_id, provider, keys_checked, valid_count, invalid_count,
                       error_count, requested_by_user_id, requested_by_label, job_id,
                       scope_summary, error_message, created_at, started_at, completed_at
                FROM byok_validation_runs
                WHERE id = ?
                """,
                (run_id,),
            )
            return self._row_to_dict(row) if row else None
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.get_run failed")
            raise

    async def list_runs(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """List BYOK validation runs newest-first with total count."""
        try:
            rows = await self.db_pool.fetchall(
                """
                SELECT id, status, org_id, provider, keys_checked, valid_count, invalid_count,
                       error_count, requested_by_user_id, requested_by_label, job_id,
                       scope_summary, error_message, created_at, started_at, completed_at
                FROM byok_validation_runs
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            count_row = await self.db_pool.fetchone(
                "SELECT COUNT(*) AS total FROM byok_validation_runs"
            )
            total = 0
            if count_row is not None:
                try:
                    total = int(count_row["total"])
                except Exception:
                    total = int(list(count_row)[0])
            return [self._row_to_dict(row) for row in rows], total
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.list_runs failed")
            raise

    async def mark_running(self, run_id: str, *, job_id: str | None) -> dict[str, Any] | None:
        """Mark a run as running and attach a job id."""
        started_at = self._normalize_timestamp(datetime.now(dt_timezone.utc).isoformat())
        try:
            result = await self.db_pool.execute(
                """
                UPDATE byok_validation_runs
                SET status = 'running',
                    job_id = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ?
                """,
                (job_id, started_at, run_id),
            )
            if not self._command_touched_rows(result):
                return None
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.mark_running failed")
            raise

    async def mark_complete(
        self,
        run_id: str,
        *,
        keys_checked: int,
        valid_count: int,
        invalid_count: int,
        error_count: int,
    ) -> dict[str, Any] | None:
        """Mark a run complete and persist aggregate counts."""
        completed_at = self._normalize_timestamp(datetime.now(dt_timezone.utc).isoformat())
        try:
            result = await self.db_pool.execute(
                """
                UPDATE byok_validation_runs
                SET status = 'complete',
                    keys_checked = ?,
                    valid_count = ?,
                    invalid_count = ?,
                    error_count = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (keys_checked, valid_count, invalid_count, error_count, completed_at, run_id),
            )
            if not self._command_touched_rows(result):
                return None
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.mark_complete failed")
            raise

    async def mark_failed(self, run_id: str, *, error_message: str) -> dict[str, Any] | None:
        """Mark a run failed and persist a bounded error message."""
        completed_at = self._normalize_timestamp(datetime.now(dt_timezone.utc).isoformat())
        try:
            result = await self.db_pool.execute(
                """
                UPDATE byok_validation_runs
                SET status = 'failed',
                    error_message = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (error_message, completed_at, run_id),
            )
            if not self._command_touched_rows(result):
                return None
            return await self.get_run(run_id)
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.mark_failed failed")
            raise

    async def has_active_run(self) -> bool:
        """Return True when any BYOK validation run is queued or running."""
        try:
            row = await self.db_pool.fetchone(
                """
                SELECT 1
                FROM byok_validation_runs
                WHERE status IN ('queued', 'running')
                LIMIT 1
                """
            )
            return row is not None
        except Exception:
            logger.error("AuthnzByokValidationRunsRepo.has_active_run failed")
            raise
