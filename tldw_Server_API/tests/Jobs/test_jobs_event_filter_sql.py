from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.jobs_sql_fragments import (
    job_event_filter_fragment,
)

pytestmark = pytest.mark.unit


def test_job_event_filter_fragment_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="Unsupported job event filter column"):
        job_event_filter_fragment("domain OR 1=1", backend="sqlite")


@pytest.mark.parametrize(
    ("backend", "column", "expected"),
    [
        ("sqlite", "domain", "domain = ?"),
        ("postgres", "owner_user_id", "owner_user_id = %s"),
    ],
)
def test_job_event_filter_fragment_uses_allowlisted_columns(
    backend: str,
    column: str,
    expected: str,
) -> None:
    assert job_event_filter_fragment(column, backend=backend) == expected
