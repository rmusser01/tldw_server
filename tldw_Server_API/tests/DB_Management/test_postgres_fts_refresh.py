"""Regression coverage for PostgreSQL FTS bootstrap refresh writes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend


@pytest.mark.unit
def test_fts_refresh_uses_null_safe_difference_predicate() -> None:
    """Repeated setup must not unconditionally rewrite all existing rows."""
    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    connection = MagicMock()

    backend.create_fts_table("docs_fts", "docs", ["title", "body"], connection=connection)

    refresh_sql = next(
        call.args[0]
        for call in connection.cursor.return_value.execute.call_args_list
        if call.args[0].strip().startswith("UPDATE")
    )
    assert 'WHERE "docs_fts_tsv" IS DISTINCT FROM' in refresh_sql


@pytest.mark.integration
def test_fts_bootstrap_repairs_vectors_without_rewriting_unchanged_rows(
    pg_database_config: DatabaseConfig,
) -> None:
    """Backfill null/stale values once, then preserve row versions on rerun."""
    backend = PostgreSQLBackend(pg_database_config)
    try:
        backend.execute(
            "CREATE TABLE fts_refresh_docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT, docs_fts_tsv TSVECTOR)"
        )
        backend.execute(
            "INSERT INTO fts_refresh_docs VALUES "
            "(1, 'hello', 'world', NULL), "
            "(2, 'new', 'content', to_tsvector('english', 'stale'))"
        )
        backend.create_fts_table("docs_fts", "fts_refresh_docs", ["title", "body"])
        repaired = backend.execute(
            "SELECT id, xmin::text AS row_version, "
            "docs_fts_tsv = to_tsvector('english', "
            "coalesce(title, '') || ' ' || coalesce(body, '')) AS correct "
            "FROM fts_refresh_docs ORDER BY id"
        ).rows
        assert all(row["correct"] for row in repaired)

        backend.create_fts_table("docs_fts", "fts_refresh_docs", ["title", "body"])
        repeated = backend.execute("SELECT id, xmin::text AS row_version FROM fts_refresh_docs ORDER BY id").rows
        assert [row["row_version"] for row in repeated] == [row["row_version"] for row in repaired]
    finally:
        backend.get_pool().close_all()
