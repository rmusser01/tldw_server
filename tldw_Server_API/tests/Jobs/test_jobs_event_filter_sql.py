from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.jobs_sql_fragments import (
    job_event_filter_fragment,
)

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
