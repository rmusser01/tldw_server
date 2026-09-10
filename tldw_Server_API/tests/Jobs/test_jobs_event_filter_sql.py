from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management import jobs_sql_fragments

job_event_filter_fragment = jobs_sql_fragments.job_event_filter_fragment

pytestmark = pytest.mark.unit


def test_job_event_filter_fragment_rejects_unknown_columns() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported job event filter column: domain OR 1=1",
    ):
        job_event_filter_fragment("domain OR 1=1", backend="sqlite")


def test_job_event_filter_fragment_rejects_unknown_backends() -> None:
    with pytest.raises(ValueError, match="Unsupported jobs SQL backend: mysql"):
        job_event_filter_fragment("domain", backend="mysql")


@pytest.mark.parametrize(
    ("backend", "column", "expected"),
    [
        ("sqlite", "domain", "domain = ?"),
        ("postgresql", "owner_user_id", "owner_user_id = %s"),
        ("postgres", "queue", "queue = %s"),
    ],
)
def test_job_event_filter_fragment_uses_allowlisted_columns(
    backend: str,
    column: str,
    expected: str,
) -> None:
    assert job_event_filter_fragment(column, backend=backend) == expected


def test_postgres_counter_transition_executes_through_db_management() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class Cursor:
        def execute(self, sql: str, params: tuple[object, ...]) -> None:
            calls.append((sql, params))

    assert hasattr(jobs_sql_fragments, "apply_postgres_job_counter_transition")
    jobs_sql_fragments.apply_postgres_job_counter_transition(
        Cursor(),
        domain="admin_webhooks",
        queue="delivery",
        job_type="deliver",
        ready_delta=-1,
        scheduled_delta=1,
        processing_delta=-1,
        quarantined_delta=0,
    )

    assert len(calls) == 1
    sql, params = calls[0]
    assert "INSERT INTO job_counters" in sql
    assert "ON CONFLICT(domain,queue,job_type) DO UPDATE" in sql
    assert params == (
        "admin_webhooks",
        "delivery",
        "deliver",
        -1,
        1,
        -1,
        0,
    )
