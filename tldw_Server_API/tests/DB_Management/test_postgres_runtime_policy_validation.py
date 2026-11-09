"""Policy canonicalization must retain access-control semantics."""

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_runtime_validation import (
    canonical_policy_expression,
)


@pytest.mark.parametrize(
    ("definition", "catalog"),
    [
        ("\"media\".visibility = 'personal'", "((visibility = 'personal'::text))"),
        ("length(media.x) BETWEEN 1 AND 255", "((length(x)>=1) AND (length(x)<=255))"),
        ("a AND (b AND c)", "(a AND b) AND c"),
        (
            "(SELECT current_setting('app.user', true))",
            "(SELECT current_setting('app.user'::text, true) AS current_setting)",
        ),
        ("ARRAY[]::BIGINT[]", "'{}'::bigint[]"),
    ],
)
def test_postgres_deparser_equivalents_match(definition, catalog):
    # PostgreSQL decoration must not require schema writes on runtime open.
    assert canonical_policy_expression(definition, "media") == canonical_policy_expression(catalog, "media")  # nosec B101


@pytest.mark.parametrize(
    ("definition", "changed"),
    [
        ("a OR (b AND c)", "(a OR b) AND c"),
        ("EXISTS(SELECT 1 FROM other WHERE media.id=other.id)", "EXISTS(SELECT 1 FROM other WHERE id=other.id)"),
        ("media.owner::TEXT = '1'", "media.owner = '1'"),
        ("media.x = 'a  b'", "media.x = 'a b'"),
        ("media.x = 'Owner'", "media.x = 'owner'"),
        ("media.x IS NULL", "media.x IS NOT NULL"),
        (
            "(SELECT current_setting('app.user', true) FROM other)",
            "(SELECT current_setting('app.user', true) AS current_setting FROM other)",
        ),
    ],
)
def test_changed_authority_does_not_match(definition, changed):
    # Grouping, correlation, types and literal data are authority, not formatting.
    assert canonical_policy_expression(definition, "media") != canonical_policy_expression(changed, "media")  # nosec B101


def test_unparseable_policy_fails_closed():
    with pytest.raises(SchemaError, match="cannot be parsed"):
        canonical_policy_expression("EXISTS(SELECT", "media")
