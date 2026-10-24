"""Readiness check matching must retain boolean grouping and allowed values."""

import pytest

from tldw_Server_API.app.core.AuthNZ.postgres_runtime_schema import _check_matches, _column_type_allowed


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("((org_id IS NULL) OR (team_id IS NULL))", ("org_id IS NULL OR team_id IS NULL",), True),
        ("org_id IS NULL AND team_id IS NULL", ("org_id IS NULL OR team_id IS NULL",), False),
        ("org_id IS NULL OR team_id IS NULL OR TRUE", ("org_id IS NULL OR team_id IS NULL",), False),
        ("org_id IS NOT NULL OR team_id IS NULL", ("org_id IS NULL OR team_id IS NULL",), False),
        (
            "(purpose = ANY (ARRAY['playback'::text, 'download'::text]))",
            ("purpose = ANY (ARRAY['playback', 'download'])",),
            True,
        ),
        (
            "purpose IN ('playback', 'download', 'other')",
            ("purpose IN ('playback', 'download')",),
            False,
        ),
        ("purpose::integer = 1", ("purpose = '1'",), False),
        ("purpose::varchar(1) IN ('playback', 'download')", ("purpose IN ('playback', 'download')",), False),
        (
            "purpose::text = ANY ((ARRAY['playback'::varchar, 'download'::varchar])::text[])",
            ("purpose = ANY (ARRAY['playback', 'download'])",),
            True,
        ),
        ("broken(", ("org_id IS NULL",), False),
    ],
)
def test_only_declared_check_meaning_is_accepted(actual, expected, matches):
    """Ignore safe deparser decoration without admitting a changed expression."""
    assert _check_matches(actual, expected) is matches  # nosec B101 -- semantic admission control.


@pytest.mark.parametrize(
    ("column", "data_type", "allowed"),
    [
        ("is_superuser", "bool", True),
        ("is_superuser", "text", False),
        ("granted", "text", False),
        ("uuid", "uuid", True),
        ("uuid", "text", True),
        ("uuid", "varchar", True),
        ("uuid", "int4", False),
        ("id", "int4", True),
        ("id", "int8", True),
        ("id", "text", False),
        ("created_at", "timestamp", True),
        ("created_at", "timestamptz", True),
        ("created_at", "text", False),
        ("used_at", "timestamp", True),
        ("used_at", "timestamptz", True),
        ("used_at", "text", False),
        ("metadata", "jsonb", True),
        ("storage_used_mb", "int4", True),
        ("storage_used_mb", "float8", True),
        ("password_hash", "text", True),
        ("password_hash", "bool", False),
    ],
)
def test_serving_types_preserve_legacy_families_without_truthy_text_flags(column, data_type, allowed):
    """Historical representation changes must not reinterpret security fields."""
    assert _column_type_allowed(column, data_type) is allowed  # nosec B101 -- security type boundary.
