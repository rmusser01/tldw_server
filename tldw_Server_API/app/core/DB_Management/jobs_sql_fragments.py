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


def job_event_filter_fragment(column: str, *, backend: str) -> str:
    """Return an allowlisted job-event scalar filter fragment for the SQL backend."""
    backend_name = normalize_backend_name(backend)
    if backend_name not in _JOB_EVENT_FILTER_SQL:
        raise ValueError(f"Unsupported jobs SQL backend: {backend}")

    try:
        return _JOB_EVENT_FILTER_SQL[backend_name][column]
    except KeyError as exc:
        raise ValueError(f"Unsupported job event filter column: {column}") from exc


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
    query = f"SELECT * FROM jobs{where_clause}{scoped_suffix}"  # nosec B608
    active_cursor = executor.execute(query, params) or executor
    active_rows = list(active_cursor.fetchall() or [])
    placeholder = "%s" if backend_name == BackendType.POSTGRESQL.value else "?"
    collisions: list[tuple[Any, list[Any]]] = []
    for active_row in active_rows:
        job_uuid = str(dict(active_row).get("uuid") or "").strip()
        if not job_uuid:
            continue
        archive_cursor = executor.execute(
            f"SELECT * FROM jobs_archive WHERE uuid={placeholder} LIMIT 2",  # nosec B608
            (job_uuid,),
        ) or executor
        collisions.append((active_row, list(archive_cursor.fetchall() or [])))
    return collisions
