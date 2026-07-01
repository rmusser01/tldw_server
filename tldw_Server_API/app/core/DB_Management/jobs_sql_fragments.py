"""SQL fragment helpers for the Jobs database layer."""

from __future__ import annotations

_JOB_EVENT_FILTER_SQL: dict[str, dict[str, str]] = {
    "postgres": {
        "domain": "domain = %s",
        "queue": "queue = %s",
        "job_type": "job_type = %s",
        "job_id": "job_id = %s",
        "owner_user_id": "owner_user_id = %s",
    },
    "sqlite": {
        "domain": "domain = ?",
        "queue": "queue = ?",
        "job_type": "job_type = ?",
        "job_id": "job_id = ?",
        "owner_user_id": "owner_user_id = ?",
    },
}


def job_event_filter_fragment(column: str, *, backend: str) -> str:
    """Return an allowlisted job-event scalar filter fragment for the SQL backend."""
    try:
        return _JOB_EVENT_FILTER_SQL[backend][column]
    except KeyError as exc:
        raise ValueError("Unsupported job event filter column") from exc
