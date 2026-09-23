"""Jobs admin operations, moved out of the jobs admin endpoints (TASK-13317).

Endpoints may not reach JobManager's connection helpers; this module belongs to the
Jobs package and may. The bodies are moved verbatim; only request fields became
parameters and response objects became return values. HTTP concerns (confirmation
headers, RLS context, response models) stay in the endpoints.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as _json
from typing import Any

from tldw_Server_API.app.core.Jobs.manager import JobManager, _reconcile_lifecycle_counter_row
from tldw_Server_API.app.core.testing import env_flag_enabled

# The endpoints' best-effort tuple, minus HTTPException.
_BEST_EFFORT_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    _json.JSONDecodeError,
)

def batch_cancel(jm: JobManager, *, domain: str, queue: str | None, job_type: str | None, job_id: int | None, dry_run: bool) -> int:
    """Moved verbatim from the batch_cancel_endpoint admin endpoint; returns the affected row count."""
    conn = jm._connect()
    try:
        where = ["domain = %s"] if jm.backend == "postgres" else ["domain = ?"]
        params: list = [domain]
        if queue:
            where.append("queue = %s" if jm.backend == "postgres" else "queue = ?")
            params.append(queue)
        if job_type:
            where.append("job_type = %s" if jm.backend == "postgres" else "job_type = ?")
            params.append(job_type)
        if job_id is not None:
            where.append("id = %s" if jm.backend == "postgres" else "id = ?")
            params.append(int(job_id))
        if jm.backend == "postgres":
            if dry_run:
                with jm._pg_cursor(conn) as cur:
                    cur.execute(
                        f"SELECT COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) AND status IN ('queued','processing')",  # nosec B608
                        tuple(params),
                    )
                    c = cur.fetchone()
                    count = int(c.get("count") or 0) if isinstance(c, dict) else int(c[0] if c else 0)
                    return count
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:  # noqa: SIM117
                with jm._pg_cursor(conn) as cur:
                    cur.execute(
                        (
                            "WITH candidates AS (SELECT id,domain,queue,job_type,status,available_at "
                            f"FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                            "AND status IN ('queued','processing') FOR UPDATE), changed AS ("
                            "UPDATE jobs AS target SET status='cancelled', cancelled_at=NOW(), "
                            "cancellation_reason='batch_cancel', leased_until=NULL, worker_id=NULL, "
                            "lease_id=NULL FROM candidates WHERE target.id=candidates.id "
                            "AND target.status IN ('queued','processing') "
                            "RETURNING candidates.domain,candidates.queue,candidates.job_type,"
                            "candidates.status AS prior_status,"
                            "candidates.available_at AS prior_available_at) "
                            "SELECT domain,queue,job_type,COUNT(*) AS total_count,"
                            "COUNT(*) FILTER (WHERE prior_status='queued' "
                            "AND prior_available_at IS NULL) AS ready_count,"
                            "COUNT(*) FILTER (WHERE prior_status='queued' "
                            "AND prior_available_at IS NOT NULL) AS scheduled_count,"
                            "COUNT(*) FILTER (WHERE prior_status='processing') AS processing_count "
                            "FROM changed GROUP BY domain,queue,job_type"
                        ),
                        tuple(params),
                    )
                    groups = list(cur.fetchall() or [])
                    affected = sum(int(row.get("total_count") or 0) for row in groups)
                    if counters_enabled:
                        for row in groups:
                            cur.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count=GREATEST(ready_count - %s, 0), "
                                    "scheduled_count=GREATEST(scheduled_count - %s, 0), "
                                    "processing_count=GREATEST(processing_count - %s, 0), "
                                    "updated_at=NOW() WHERE domain=%s AND queue=%s AND job_type=%s"
                                ),
                                (
                                    int(row["ready_count"] or 0),
                                    int(row["scheduled_count"] or 0),
                                    int(row["processing_count"] or 0),
                                    row["domain"],
                                    row["queue"],
                                    row["job_type"],
                                ),
                            )
                            if cur.rowcount == 0:
                                _reconcile_lifecycle_counter_row(
                                    cur,
                                    backend=jm.backend,
                                    domain=row["domain"],
                                    queue=row["queue"],
                                    job_type=row["job_type"],
                                )
        else:
            if dry_run:
                cur = conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) AND status IN ('queued','processing')",  # nosec B608
                    tuple(params),
                )
                r = cur.fetchone()
                return int(r[0] if r else 0)
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                groups = []
                if counters_enabled:
                    groups = list(
                        conn.execute(
                            (
                                "SELECT domain,queue,job_type,COUNT(*) AS total_count,"
                                "SUM(CASE WHEN status='queued' AND available_at IS NULL "
                                "THEN 1 ELSE 0 END) AS ready_count,"
                                "SUM(CASE WHEN status='queued' AND available_at IS NOT NULL "
                                "THEN 1 ELSE 0 END) AS scheduled_count,"
                                "SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) "
                                f"AS processing_count FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                                "AND status IN ('queued','processing') "
                                "GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        ).fetchall()
                        or []
                    )
                changed = conn.execute(
                    (
                        "UPDATE jobs SET status='cancelled', cancelled_at=DATETIME('now'), "
                        "cancellation_reason='batch_cancel', leased_until=NULL, worker_id=NULL, "
                        f"lease_id=NULL WHERE ({' AND '.join(where)}) "  # nosec B608
                        "AND status IN ('queued','processing')"
                    ),
                    tuple(params),
                )
                affected = int(changed.rowcount or 0)
                if counters_enabled:
                    for row in groups:
                        counter_cursor = conn.execute(
                            (
                                "UPDATE job_counters SET "
                                "ready_count=MAX(ready_count - ?, 0), "
                                "scheduled_count=MAX(scheduled_count - ?, 0), "
                                "processing_count=MAX(processing_count - ?, 0), "
                                "updated_at=DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?"
                            ),
                            (
                                int(row[4] or 0),
                                int(row[5] or 0),
                                int(row[6] or 0),
                                row[0],
                                row[1],
                                row[2],
                            ),
                        )
                        if (counter_cursor.rowcount or 0) == 0:
                            _reconcile_lifecycle_counter_row(
                                conn,
                                backend=jm.backend,
                                domain=row[0],
                                queue=row[1],
                                job_type=row[2],
                            )

        try:
            if domain and queue and job_type:
                jm.update_gauges(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                )
        except _BEST_EFFORT_EXCEPTIONS:
            pass
        return int(affected)
    finally:
        with contextlib.suppress(_BEST_EFFORT_EXCEPTIONS):
            conn.close()


def batch_reschedule(jm: JobManager, *, domain: str, queue: str | None, job_type: str | None, delay_seconds: int, dry_run: bool) -> int:
    """Moved verbatim from the batch_reschedule_endpoint admin endpoint; returns the affected row count."""
    conn = jm._connect()
    try:
        where = ["domain = %s", "status = 'queued'"] if jm.backend == "postgres" else ["domain = ?", "status = 'queued'"]
        params: list = [domain]
        if queue:
            where.append("queue = %s" if jm.backend == "postgres" else "queue = ?")
            params.append(queue)
        if job_type:
            where.append("job_type = %s" if jm.backend == "postgres" else "job_type = ?")
            params.append(job_type)
        if jm.backend == "postgres":
            if dry_run:
                with jm._pg_cursor(conn) as cur:
                    cur.execute(
                        f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}",  # nosec B608
                        tuple(params),
                    )
                    r = cur.fetchone()
                    count = int(r.get("count") or 0) if isinstance(r, dict) else int(r[0] if r else 0)
                    return count
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:  # noqa: SIM117
                with jm._pg_cursor(conn) as cur:
                    target_available_at = (
                        "NULL"
                        if delay_seconds == 0
                        else "NOW() + (%s || ' seconds')::interval"
                    )
                    update_params = (
                        tuple(params)
                        if delay_seconds == 0
                        else (*params, int(delay_seconds))
                    )
                    cur.execute(
                        (
                            "WITH candidates AS (SELECT id,domain,queue,job_type,available_at "
                            f"FROM jobs WHERE {' AND '.join(where)} FOR UPDATE), "  # nosec B608
                            "changed AS (UPDATE jobs AS target SET available_at="
                            f"{target_available_at} FROM candidates "  # nosec B608
                            "WHERE target.id=candidates.id AND target.status='queued' "
                            "RETURNING candidates.domain,candidates.queue,candidates.job_type,"
                            "candidates.available_at AS prior_available_at) "
                            "SELECT domain,queue,job_type,COUNT(*) AS total_count,"
                            "COUNT(*) FILTER (WHERE prior_available_at IS NULL) AS ready_count,"
                            "COUNT(*) FILTER (WHERE prior_available_at IS NOT NULL) AS scheduled_count "
                            "FROM changed GROUP BY domain,queue,job_type"
                        ),
                        update_params,
                    )
                    groups = list(cur.fetchall() or [])
                    affected = sum(int(row.get("total_count") or 0) for row in groups)
                    if counters_enabled:
                        for row in groups:
                            moved = int(
                                row["scheduled_count"]
                                if delay_seconds == 0
                                else row["ready_count"]
                            )
                            if moved == 0:
                                continue
                            if delay_seconds == 0:
                                counter_sql = (
                                    "UPDATE job_counters SET "
                                    "scheduled_count=GREATEST(scheduled_count - %s, 0), "
                                    "ready_count=ready_count + %s, updated_at=NOW() "
                                    "WHERE domain=%s AND queue=%s AND job_type=%s"
                                )
                            else:
                                counter_sql = (
                                    "UPDATE job_counters SET "
                                    "ready_count=GREATEST(ready_count - %s, 0), "
                                    "scheduled_count=scheduled_count + %s, updated_at=NOW() "
                                    "WHERE domain=%s AND queue=%s AND job_type=%s"
                                )
                            cur.execute(
                                counter_sql,
                                (
                                    moved,
                                    moved,
                                    row["domain"],
                                    row["queue"],
                                    row["job_type"],
                                ),
                            )
                            if cur.rowcount == 0:
                                _reconcile_lifecycle_counter_row(
                                    cur,
                                    backend=jm.backend,
                                    domain=row["domain"],
                                    queue=row["queue"],
                                    job_type=row["job_type"],
                                )
        else:
            if dry_run:
                cur = conn.execute(
                    f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}",  # nosec B608
                    tuple(params),
                )
                r = cur.fetchone()
                return int(r[0] if r else 0)
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                groups = []
                if counters_enabled:
                    groups = list(
                        conn.execute(
                            (
                                "SELECT domain,queue,job_type,COUNT(*) AS total_count,"
                                "SUM(CASE WHEN available_at IS NULL THEN 1 ELSE 0 END) "
                                "AS ready_count,"
                                "SUM(CASE WHEN available_at IS NOT NULL THEN 1 ELSE 0 END) "
                                f"AS scheduled_count FROM jobs WHERE {' AND '.join(where)} "  # nosec B608
                                "GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        ).fetchall()
                        or []
                    )
                if delay_seconds == 0:
                    changed = conn.execute(
                        f"UPDATE jobs SET available_at=NULL WHERE {' AND '.join(where)}",  # nosec B608
                        tuple(params),
                    )
                else:
                    changed = conn.execute(
                        f"UPDATE jobs SET available_at=DATETIME('now', ?) WHERE {' AND '.join(where)}",  # nosec B608
                        (f"+{int(delay_seconds)} seconds", *params),
                    )
                affected = int(changed.rowcount or 0)
                if counters_enabled:
                    for row in groups:
                        moved = int(
                            row[5]
                            if delay_seconds == 0
                            else row[4]
                        )
                        if moved == 0:
                            continue
                        if delay_seconds == 0:
                            counter_sql = (
                                "UPDATE job_counters SET "
                                "scheduled_count=MAX(scheduled_count - ?, 0), "
                                "ready_count=ready_count + ?, updated_at=DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?"
                            )
                        else:
                            counter_sql = (
                                "UPDATE job_counters SET "
                                "ready_count=MAX(ready_count - ?, 0), "
                                "scheduled_count=scheduled_count + ?, updated_at=DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?"
                            )
                        counter_cursor = conn.execute(
                            counter_sql,
                            (moved, moved, row[0], row[1], row[2]),
                        )
                        if (counter_cursor.rowcount or 0) == 0:
                            _reconcile_lifecycle_counter_row(
                                conn,
                                backend=jm.backend,
                                domain=row[0],
                                queue=row[1],
                                job_type=row[2],
                            )

        try:
            if domain and queue and job_type:
                jm.update_gauges(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                )
        except _BEST_EFFORT_EXCEPTIONS:
            pass
        return int(affected)
    finally:
        with contextlib.suppress(_BEST_EFFORT_EXCEPTIONS):
            conn.close()


def batch_requeue_quarantined(jm: JobManager, *, domain: str, queue: str | None, job_type: str | None, job_id: int | None, dry_run: bool) -> int:
    """Moved verbatim from the batch_requeue_quarantined_endpoint admin endpoint; returns the affected row count."""
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            where = ["domain = %s", "status = 'quarantined'"]
            params: list = [domain]
            if queue:
                where.append("queue = %s")
                params.append(queue)
            if job_type:
                where.append("job_type = %s")
                params.append(job_type)
            if job_id is not None:
                where.append("id = %s")
                params.append(int(job_id))
            if dry_run:
                with jm._pg_cursor(conn) as cur:
                    cur.execute(
                        f"SELECT COUNT(*) AS c FROM jobs WHERE {' AND '.join(where)}",  # nosec B608
                        tuple(params),
                    )
                    row = cur.fetchone()
                    count = int(row.get("c") or 0) if isinstance(row, dict) else int(row[0] if row else 0)
                    return count
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:  # noqa: SIM117
                with jm._pg_cursor(conn) as cur:
                    cur.execute(
                        (
                            "WITH changed AS (UPDATE jobs SET status='queued', "
                            "failure_streak_count=0, failure_streak_code=NULL, "
                            "quarantined_at=NULL, available_at=NULL, leased_until=NULL, "
                            "worker_id=NULL, lease_id=NULL, completion_token=NULL "
                            f"WHERE {' AND '.join(where)} "  # nosec B608
                            "RETURNING domain,queue,job_type) "
                            "SELECT domain,queue,job_type,COUNT(*) AS c FROM changed "
                            "GROUP BY domain,queue,job_type"
                        ),
                        tuple(params),
                    )
                    groups = list(cur.fetchall() or [])
                    affected = sum(int(row.get("c") or 0) for row in groups)
                    if counters_enabled:
                        for row in groups:
                            moved = int(row["c"] or 0)
                            cur.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count=ready_count + %s, "
                                    "quarantined_count=GREATEST(quarantined_count - %s, 0), "
                                    "updated_at=NOW() "
                                    "WHERE domain=%s AND queue=%s AND job_type=%s"
                                ),
                                (
                                    moved,
                                    moved,
                                    row["domain"],
                                    row["queue"],
                                    row["job_type"],
                                ),
                            )
                            if cur.rowcount == 0:
                                _reconcile_lifecycle_counter_row(
                                    cur,
                                    backend=jm.backend,
                                    domain=row["domain"],
                                    queue=row["queue"],
                                    job_type=row["job_type"],
                                )
        else:
            where = ["domain = ?", "status = 'quarantined'"]
            params2: list = [domain]
            if queue:
                where.append("queue = ?")
                params2.append(queue)
            if job_type:
                where.append("job_type = ?")
                params2.append(job_type)
            if job_id is not None:
                where.append("id = ?")
                params2.append(int(job_id))
            if dry_run:
                cur = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}", tuple(params2))  # nosec B608
                r = cur.fetchone()
                return int(r[0] if r else 0)
            counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
            affected = 0
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                groups = []
                if counters_enabled:
                    groups = list(
                        conn.execute(
                            f"SELECT domain,queue,job_type,COUNT(*) FROM jobs WHERE {' AND '.join(where)} GROUP BY domain,queue,job_type",  # nosec B608
                            tuple(params2),
                        ).fetchall()
                        or []
                    )
                changed = conn.execute(
                    (
                        "UPDATE jobs SET status='queued', failure_streak_count=0, "
                        "failure_streak_code=NULL, quarantined_at=NULL, available_at=NULL, "
                        "leased_until=NULL, worker_id=NULL, lease_id=NULL, "
                        f"completion_token=NULL WHERE {' AND '.join(where)}"  # nosec B608
                    ),
                    tuple(params2),
                )
                affected = int(changed.rowcount or 0)
                if counters_enabled:
                    for row in groups:
                        moved = int(row[3] or 0)
                        counter_cursor = conn.execute(
                            (
                                "UPDATE job_counters SET ready_count=ready_count + ?, "
                                "quarantined_count=MAX(quarantined_count - ?, 0), "
                                "updated_at=DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?"
                            ),
                            (moved, moved, row[0], row[1], row[2]),
                        )
                        if (counter_cursor.rowcount or 0) == 0:
                            _reconcile_lifecycle_counter_row(
                                conn,
                                backend=jm.backend,
                                domain=row[0],
                                queue=row[1],
                                job_type=row[2],
                            )

        try:
            if domain and queue and job_type:
                jm.update_gauges(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                )
        except _BEST_EFFORT_EXCEPTIONS:
            pass
        return affected
    finally:
        with contextlib.suppress(_BEST_EFFORT_EXCEPTIONS):
            conn.close()


# --- Read queries ------------------------------------------------------------------
# The caller sets any PostgreSQL RLS context first; JobManager applies it per cursor.


def _filters(jm: JobManager, pairs: list[tuple[str, Any]], base: list[str] | None = None) -> tuple[str, list[Any]]:
    token = "%s" if jm.backend == "postgres" else "?"
    where = list(base or ["1=1"])
    params: list[Any] = []
    for column, value in pairs:
        if value:
            where.append(f"{column} = {token}")
            params.append(value)
    return " AND ".join(where), params


def _rows(jm: JobManager, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[Any]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(sql, tuple(params))
                return list(cur.fetchall() or [])
        return list(conn.execute(sql, tuple(params)).fetchall() or [])
    finally:
        with contextlib.suppress(_BEST_EFFORT_EXCEPTIONS):
            conn.close()


def list_sla_policies(
    jm: JobManager, *, domain: str | None = None, queue: str | None = None, job_type: str | None = None
) -> list[dict[str, Any]]:
    where, params = _filters(jm, [("domain", domain), ("queue", queue), ("job_type", job_type)])
    sql = f"SELECT * FROM job_sla_policies WHERE {where} ORDER BY domain,queue,job_type"  # nosec B608
    return [dict(row) for row in _rows(jm, sql, params)]


def list_enabled_sla_policies(jm: JobManager) -> list[dict[str, Any]]:
    enabled = "true" if jm.backend == "postgres" else "1"
    return [
        dict(row)
        for row in _rows(jm, f"SELECT * FROM job_sla_policies WHERE enabled={enabled} ORDER BY domain,queue,job_type")  # nosec B608
    ]


def list_active_jobs(
    jm: JobManager, *, domain: str | None = None, queue: str | None = None, job_type: str | None = None
) -> list[dict[str, Any]]:
    """Queued and processing jobs, oldest first, with the timestamps SLA checks need."""
    where, params = _filters(
        jm, [("domain", domain), ("queue", queue), ("job_type", job_type)], base=["status IN ('queued', 'processing')"]
    )
    sql = (
        "SELECT id, domain, queue, job_type, status, created_at, acquired_at, started_at "  # nosec B608
        f"FROM jobs WHERE {where} ORDER BY created_at"
    )
    return [dict(row) for row in _rows(jm, sql, params)]


def archived_payload_presence(jm: JobManager, job_id: int) -> dict[str, bool] | None:
    """Which payload/result columns an archived job has, or None if it is not archived."""
    token = "%s" if jm.backend == "postgres" else "?"
    rows = _rows(
        jm,
        f"SELECT payload, result, payload_compressed, result_compressed FROM jobs_archive WHERE id = {token}",  # nosec B608
        (int(job_id),),
    )
    if not rows:
        return None
    row = rows[0]
    values = [row.get(key) for key in ("payload", "result", "payload_compressed", "result_compressed")] if isinstance(
        row, dict
    ) else list(row[:4])
    return {
        "payload_present": values[0] is not None,
        "result_present": values[1] is not None,
        "payload_compressed_present": values[2] is not None,
        "result_compressed_present": values[3] is not None,
    }


def stale_processing_groups(
    jm: JobManager, *, domain: str | None = None, queue: str | None = None
) -> list[tuple[str, str, int]]:
    """(domain, queue, count) of processing jobs whose lease has lapsed."""
    now = "NOW()" if jm.backend == "postgres" else "DATETIME('now')"
    where, params = _filters(
        jm,
        [("domain", domain), ("queue", queue)],
        base=["status='processing'", f"(leased_until IS NULL OR leased_until <= {now})"],
    )
    sql = f"SELECT domain, queue, COUNT(*) FROM jobs WHERE {where} GROUP BY domain, queue"  # nosec B608
    out: list[tuple[str, str, int]] = []
    for row in _rows(jm, sql, params):
        values = list(row.values()) if isinstance(row, dict) else list(row)
        out.append((str(values[0]), str(values[1]), int(values[2])))
    return out
