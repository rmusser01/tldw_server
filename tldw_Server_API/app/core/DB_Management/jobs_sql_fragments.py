"""SQL fragment helpers for the Jobs database layer."""

from __future__ import annotations

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
