"""SQL fragment helpers for the Jobs database layer."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    normalize_backend_name,
)

_JOB_EVENT_FILTER_SQL: dict[str, dict[str, str]] = {
    BackendType.POSTGRESQL.value: {
        "domain": "domain = %s",
        "queue": "queue = %s",
        "job_type": "job_type = %s",
        "job_id": "job_id = %s",
        "owner_user_id": "owner_user_id = %s",
    },
    BackendType.SQLITE.value: {
        "domain": "domain = ?",
        "queue": "queue = ?",
        "job_type": "job_type = ?",
        "job_id": "job_id = ?",
        "owner_user_id": "owner_user_id = ?",
    },
}

_POSTGRES_JOB_COUNTER_TRANSITION_SQL = (
    "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,"
    "processing_count,quarantined_count) VALUES(%s,%s,%s,%s,%s,%s,%s) "
    "ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
    "ready_count=GREATEST(job_counters.ready_count + EXCLUDED.ready_count,0), "
    "scheduled_count=GREATEST(job_counters.scheduled_count + EXCLUDED.scheduled_count,0), "
    "processing_count=GREATEST(job_counters.processing_count + EXCLUDED.processing_count,0), "
    "quarantined_count=GREATEST(job_counters.quarantined_count + EXCLUDED.quarantined_count,0), "
    "updated_at=NOW()"
)


def job_event_filter_fragment(column: str, *, backend: str) -> str:
    """Return an allowlisted job-event scalar filter fragment for the SQL backend."""
    backend_name = normalize_backend_name(backend)
    if backend_name not in _JOB_EVENT_FILTER_SQL:
        raise ValueError(f"Unsupported jobs SQL backend: {backend}")

    try:
        return _JOB_EVENT_FILTER_SQL[backend_name][column]
    except KeyError as exc:
        raise ValueError(f"Unsupported job event filter column: {column}") from exc


def apply_postgres_job_counter_transition(
    cursor: Any,
    *,
    domain: Any,
    queue: Any,
    job_type: Any,
    ready_delta: int,
    scheduled_delta: int,
    processing_delta: int,
    quarantined_delta: int,
) -> None:
    """Apply one atomic PostgreSQL Jobs counter transition."""

    cursor.execute(
        _POSTGRES_JOB_COUNTER_TRANSITION_SQL,
        (
            domain,
            queue,
            job_type,
            ready_delta,
            scheduled_delta,
            processing_delta,
            quarantined_delta,
        ),
    )


def fetch_slides_archive_collision_rows(
    connection: Any,
    *,
    backend: str,
    where_clause: str,
    params: tuple[Any, ...],
    cursor: Any | None = None,
) -> list[tuple[Any, list[Any]]]:
    """Load active standalone-generation rows and archived UUID matches."""
    backend_name = normalize_backend_name(backend)
    if backend_name not in {
        BackendType.POSTGRESQL.value,
        BackendType.SQLITE.value,
    }:
        raise ValueError(f"Unsupported jobs SQL backend: {backend}")
    if backend_name == BackendType.POSTGRESQL.value and cursor is None:
        raise RuntimeError("PostgreSQL archive validation requires a cursor")

    executor = cursor if cursor is not None else connection
    scoped_suffix = (
        " AND domain='slides' AND queue='default' "
        "AND job_type='presentation.generate'"
    )
    presence_projection = (
        ", payload IS NOT NULL AS __slides_archive_payload_present, "
        "result IS NOT NULL AS __slides_archive_result_present"
    )
    query = (
        f"SELECT *{presence_projection} FROM jobs{where_clause}{scoped_suffix}"  # nosec B608
    )
    active_cursor = executor.execute(query, params) or executor
    active_rows = list(active_cursor.fetchall() or [])
    placeholder = "%s" if backend_name == BackendType.POSTGRESQL.value else "?"
    collisions: list[tuple[Any, list[Any]]] = []
    for active_row in active_rows:
        job_uuid = str(dict(active_row).get("uuid") or "").strip()
        if not job_uuid:
            continue
        archive_cursor = executor.execute(
            f"SELECT *{presence_projection} FROM jobs_archive "  # nosec B608
            f"WHERE uuid={placeholder} LIMIT 2",  # nosec B608
            (job_uuid,),
        ) or executor
        collisions.append((active_row, list(archive_cursor.fetchall() or [])))
    return collisions
