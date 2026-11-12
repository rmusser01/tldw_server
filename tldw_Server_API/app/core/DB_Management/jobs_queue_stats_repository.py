"""Read-only aggregate queries over one (domain, queue[, owner]) slice of the jobs table.

Owns the SQL and connection handling behind ``core/Jobs/queue_stats.py``; that module
turns the raw values returned here into typed numbers. Both backends are supported: the
dialect differs only in placeholders and time arithmetic, and every caller-supplied value
(including the lease warning window) is bound as a parameter.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Jobs.manager import JobManager


class JobsQueueStatsRepository:
    """Aggregate reads over the ``jobs`` table through a JobManager's connection.

    Rows are returned as plain dicts (both the SQLite ``Row`` and psycopg ``dict_row``
    shapes convert losslessly); values are passed through unconverted so the caller
    decides how to treat NULLs and unexpected types.
    """

    def __init__(self, jm: JobManager) -> None:
        """Bind the repository to ``jm``, whose backend and connection helpers it uses."""
        self._jm = jm

    @property
    def _pg(self) -> bool:
        return self._jm.backend == "postgres"

    def _filters(self, domain: str, queue: str, owner_user_id: str | None) -> tuple[str, list[Any]]:
        token = "%s" if self._pg else "?"
        clauses = [f"domain = {token}", f"queue = {token}"]
        params: list[Any] = [domain, queue]
        if owner_user_id is not None:
            clauses.append(f"owner_user_id = {token}")
            params.append(owner_user_id)
        return " AND ".join(clauses), params

    def _fetch_all(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        conn = self._jm._connect()
        try:
            if self._pg:
                with self._jm._pg_cursor(conn) as cur:
                    cur.execute(sql, params)
                    return [dict(row) for row in cur.fetchall() or []]
            return [dict(row) for row in conn.execute(sql, params).fetchall() or []]
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _fetch_one(self, sql: str, params: list[Any]) -> dict[str, Any]:
        rows = self._fetch_all(sql, params)
        return rows[0] if rows else {}

    def count_by_status(self, *, domain: str, queue: str, owner_user_id: str | None) -> list[dict[str, Any]]:
        """Count jobs per status.

        Returns:
            One ``{"status", "c"}`` row per status present in the slice.
        """
        where_sql, params = self._filters(domain, queue, owner_user_id)
        sql = f"SELECT status, COUNT(*) AS c FROM jobs WHERE {where_sql} GROUP BY status"  # nosec B608
        return self._fetch_all(sql, params)

    def count_by_type_and_status(
        self, *, domain: str, queue: str, owner_user_id: str | None
    ) -> list[dict[str, Any]]:
        """Count jobs per (job_type, status).

        Returns:
            One ``{"job_type", "status", "c"}`` row per pair present in the slice.
        """
        where_sql, params = self._filters(domain, queue, owner_user_id)
        sql = (
            "SELECT job_type, status, COUNT(*) AS c FROM jobs "  # nosec B608
            f"WHERE {where_sql} GROUP BY job_type, status"
        )
        return self._fetch_all(sql, params)

    def avg_processing_seconds(self, *, domain: str, queue: str, owner_user_id: str | None) -> Any:
        """Average ``completed_at - started_at`` in seconds over completed jobs.

        Returns:
            The raw aggregate, or None when no completed job has both timestamps.
        """
        where_sql, params = self._filters(domain, queue, owner_user_id)
        if self._pg:
            expr = "AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))"
        else:
            expr = "AVG((julianday(completed_at) - julianday(started_at)) * 86400.0)"
        sql = (
            f"SELECT {expr} AS avg_seconds FROM jobs WHERE {where_sql} "  # nosec B608
            "AND status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )
        return self._fetch_one(sql, params).get("avg_seconds")

    def success_rate(self, *, domain: str, queue: str, owner_user_id: str | None) -> Any:
        """Percentage of finished (completed or failed) jobs that completed.

        Returns:
            The raw aggregate, or None when no job has finished.
        """
        where_sql, params = self._filters(domain, queue, owner_user_id)
        if self._pg:
            ratio = (
                "COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / "
                "NULLIF(COUNT(*) FILTER (WHERE status IN ('completed', 'failed')), 0)"
            )
        else:
            ratio = (
                "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / "
                "NULLIF(SUM(CASE WHEN status IN ('completed', 'failed') THEN 1 ELSE 0 END), 0)"
            )
        sql = (
            f"SELECT {ratio} AS success_rate FROM jobs "  # nosec B608
            f"WHERE {where_sql} AND status IN ('completed', 'failed')"
        )
        return self._fetch_one(sql, params).get("success_rate")

    def lease_counts(
        self, *, domain: str, queue: str, owner_user_id: str | None, warn_seconds: int
    ) -> dict[str, Any]:
        """Count processing jobs by lease health.

        Args:
            warn_seconds: A lease ending within this many seconds counts as expiring soon.
                Bound as a parameter, never interpolated.

        Returns:
            Raw ``active``, ``expiring_soon`` and ``stale_processing`` aggregates
            (SQLite yields None for an empty slice).
        """
        where_sql, params = self._filters(domain, queue, owner_user_id)
        if self._pg:
            now = "NOW()"
            warn_until = "NOW() + (CAST(%s AS INTEGER) * INTERVAL '1 second')"
            warn_param: Any = int(warn_seconds)
            count = "COUNT(*) FILTER (WHERE {})"
        else:
            now = "DATETIME('now')"
            warn_until = "DATETIME('now', ?)"
            warn_param = f"+{int(warn_seconds)} seconds"
            count = "SUM(CASE WHEN {} THEN 1 ELSE 0 END)"
        leased = f"status = 'processing' AND leased_until IS NOT NULL AND leased_until > {now}"
        expiring = f"{leased} AND leased_until <= {warn_until}"
        stale = f"status = 'processing' AND (leased_until IS NULL OR leased_until <= {now})"
        sql = (
            f"SELECT {count.format(leased)} AS active, "  # nosec B608 - fixed fragments; values are bound
            f"{count.format(expiring)} AS expiring_soon, "
            f"{count.format(stale)} AS stale_processing "
            f"FROM jobs WHERE {where_sql}"
        )
        return self._fetch_one(sql, [warn_param, *params])
