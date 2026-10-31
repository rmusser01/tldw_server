"""Real PostgreSQL regressions for idempotent Collections schema setup."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.content_backend import backend_target_key


@pytest.fixture
def collections_backend(pg_database_config):
    """Use the repository's isolated PostgreSQL database, without schema seeds."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    try:
        yield backend
    finally:
        CollectionsDatabase._bootstrapped_backend_targets.discard(
            backend_target_key(backend)
        )
        backend.get_pool().close_all()


def test_fresh_and_repeated_postgres_collections_schema_preserve_content(collections_backend):
    """Fresh and explicit repeated bootstrap must keep usable canonical tables."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    template = db.create_output_template(
        name="PostgreSQL bootstrap control", type_="summary", format_="markdown",
        body="Retain this template across schema initialization.",
        description="Fresh PostgreSQL fixture", is_default=False,
    )
    db.ensure_schema()
    rows, total = db.list_output_templates(
        q="PostgreSQL bootstrap control", limit=10, offset=0,
    )
    assert total == 1
    assert [(row.id, row.body) for row in rows] == [(template.id, template.body)]


def test_postgres_collections_backfills_a_missing_legacy_column(collections_backend):
    """Introspection must still repair an actually missing legacy column."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    collections_backend.execute("ALTER TABLE output_templates DROP COLUMN metadata_json")
    db.ensure_schema()
    assert "metadata_json" in db._table_columns("output_templates")


def test_postgres_collections_does_not_hide_column_inspection_failure(
    collections_backend, monkeypatch,
):
    """An unreadable schema is an error, not an empty column inventory."""
    db = CollectionsDatabase.from_backend("1", collections_backend)
    original = collections_backend.get_table_info

    def fail_output_inspection(table):
        if table == "outputs":
            raise DatabaseError("Cannot inspect output columns")
        return original(table)

    monkeypatch.setattr(collections_backend, "get_table_info", fail_output_inspection)
    with pytest.raises(DatabaseError, match="Cannot inspect output columns"):
        db.ensure_schema()
