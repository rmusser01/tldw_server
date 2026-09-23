"""Job-queue statistics for one (domain, queue[, owner]) slice of the jobs table.

Moved out of the Prompt Studio status endpoint (TASK-13317): the SQL belongs with the
Jobs owner, which is also the only place allowed to use JobManager's connection
helpers. Both backends are supported; the dialect differs only in placeholders and
time arithmetic.
"""

from __future__ import annotations

import contextlib
from typing import Any, Optional

from tldw_Server_API.app.core.Jobs.manager import JobManager


def build_job_filters(
    *,
    backend: str,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> tuple[str, list[Any]]:
    token = "%s" if backend == "postgres" else "?"
    clauses: list[str] = [f"domain = {token}", f"queue = {token}"]
    params: list[Any] = [domain, queue]
    if owner_user_id is not None:
        clauses.append(f"owner_user_id = {token}")
        params.append(owner_user_id)
    return " AND ".join(clauses), params


def fetch_all(jm: JobManager, sql: str, params: list[Any]) -> list[Any]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall() or [])
        return list(conn.execute(sql, params).fetchall() or [])
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def fetch_one(jm: JobManager, sql: str, params: list[Any]) -> Optional[Any]:
    rows = fetch_all(jm, sql, params)
    return rows[0] if rows else None


def row_value(row: Any, key: str, index: int, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[index]
    except Exception:
        return default


def get_by_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> dict[str, int]:
    where_sql, params = build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    sql = f"SELECT status, COUNT(*) AS c FROM jobs WHERE {where_sql} GROUP BY status"  # nosec B608
    rows = fetch_all(jm, sql, params)
    counts: dict[str, int] = {}
    for row in rows:
        status = row_value(row, "status", 0)
        count = row_value(row, "c", 1, 0)
        if status:
            try:
                counts[str(status)] = int(count or 0)
            except Exception:
                counts[str(status)] = 0
    return counts


def get_by_type_and_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    where_sql, params = build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    sql = """
        SELECT job_type, status, COUNT(*) AS c
        FROM jobs
        WHERE {where_sql}
        GROUP BY job_type, status
    """.format_map(locals())  # nosec B608
    rows = fetch_all(jm, sql, params)
    totals: dict[str, int] = {}
    queued: dict[str, int] = {}
    processing: dict[str, int] = {}
    for row in rows:
        job_type = row_value(row, "job_type", 0)
        status = row_value(row, "status", 1)
        count = row_value(row, "c", 2, 0)
        if not job_type:
            continue
        job_type_str = str(job_type)
        count_int = int(count or 0)
        totals[job_type_str] = totals.get(job_type_str, 0) + count_int
        if status == "queued":
            queued[job_type_str] = queued.get(job_type_str, 0) + count_int
        if status == "processing":
            processing[job_type_str] = processing.get(job_type_str, 0) + count_int
    return totals, queued, processing


def get_avg_processing_time_seconds(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> float:
    where_sql, params = build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    if jm.backend == "postgres":
        sql = (
            "SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS avg_seconds "  # nosec B608
            f"FROM jobs WHERE {where_sql} AND status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )
    else:
        sql = (
            "SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400.0) AS avg_seconds "  # nosec B608
            f"FROM jobs WHERE {where_sql} AND status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )
    row = fetch_one(jm, sql, params)
    value = row_value(row, "avg_seconds", 0)
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def get_success_rate(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> float:
    where_sql, params = build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    if jm.backend == "postgres":
        sql = (
            "SELECT "  # nosec B608
            "COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / "
            "NULLIF(COUNT(*) FILTER (WHERE status IN ('completed', 'failed')), 0) AS success_rate "
            f"FROM jobs WHERE {where_sql} AND status IN ('completed', 'failed')"
        )
    else:
        sql = (
            "SELECT "  # nosec B608
            "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / "
            "NULLIF(SUM(CASE WHEN status IN ('completed', 'failed') THEN 1 ELSE 0 END), 0) AS success_rate "
            f"FROM jobs WHERE {where_sql} AND status IN ('completed', 'failed')"
        )
    row = fetch_one(jm, sql, params)
    value = row_value(row, "success_rate", 0)
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def get_lease_stats(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
    warn_seconds: int,
) -> dict[str, int]:
    where_sql, params = build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    warn_seconds = max(1, min(3600, int(warn_seconds)))
    if jm.backend == "postgres":
        sql = (
            "SELECT "  # nosec B608
            "COUNT(*) FILTER (WHERE status = 'processing' AND leased_until IS NOT NULL AND leased_until > NOW()) AS active, "
            "COUNT(*) FILTER (WHERE status = 'processing' AND leased_until IS NOT NULL AND leased_until > NOW() "
            f"AND leased_until <= NOW() + INTERVAL '{warn_seconds} seconds') AS expiring_soon, "
            "COUNT(*) FILTER (WHERE status = 'processing' AND (leased_until IS NULL OR leased_until <= NOW())) AS stale_processing "
            f"FROM jobs WHERE {where_sql}"
        )
    else:
        sql = (
            "SELECT "  # nosec B608
            "SUM(CASE WHEN status = 'processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now') THEN 1 ELSE 0 END) AS active, "
            "SUM(CASE WHEN status = 'processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now') "
            f"AND leased_until <= DATETIME('now', '+{warn_seconds} seconds') THEN 1 ELSE 0 END) AS expiring_soon, "
            "SUM(CASE WHEN status = 'processing' AND (leased_until IS NULL OR leased_until <= DATETIME('now')) THEN 1 ELSE 0 END) AS stale_processing "
            f"FROM jobs WHERE {where_sql}"
        )
    row = fetch_one(jm, sql, params)
    return {
        "active": int(row_value(row, "active", 0, 0) or 0),
        "expiring_soon": int(row_value(row, "expiring_soon", 1, 0) or 0),
        "stale_processing": int(row_value(row, "stale_processing", 2, 0) or 0),
    }
