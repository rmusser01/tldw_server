"""Property-based tests for MarkdownTableAdapter validation behavior."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.File_Artifacts.adapters.markdown_table_adapter import MarkdownTableAdapter

pytestmark = pytest.mark.unit


# too_slow input generation can fire on loaded CI runners; the strategy is fine,
# the machine is just busy. Suppress the timing-sensitive health check.
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    columns=st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=6, unique=True),
    rows=st.lists(st.lists(st.text(max_size=12), max_size=6), max_size=10),
)
def test_validate_accepts_matching_rows(columns: list[str], rows: list[list[str]]) -> None:
    """Rows normalized to column count should yield no validation issues."""
    adapter = MarkdownTableAdapter()
    normalized_rows = []
    for row in rows:
        if len(row) < len(columns):
            padded = row + [""] * (len(columns) - len(row))
        else:
            padded = row[: len(columns)]
        normalized_rows.append(padded)

    issues = adapter.validate({"columns": columns, "rows": normalized_rows})
    assert issues == []
