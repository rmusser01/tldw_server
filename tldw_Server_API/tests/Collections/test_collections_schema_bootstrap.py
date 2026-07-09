"""Regression tests for Collections schema bootstrap SQL."""

import sqlite3
import types
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


pytestmark = pytest.mark.unit


def test_fresh_content_items_bootstrap_does_not_readd_declared_columns(
    tmp_path: Path,
) -> None:
    """Fresh SQLite bootstrap must not re-add columns already in CREATE TABLE."""
    backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "collections.db"),
        )
    )
    original_execute = backend.execute
    executed_sql: list[str] = []

    def spy_execute(
        self: SQLiteBackend,
        query: str,
        params: tuple[object, ...] | dict[str, object] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> QueryResult:
        """Record SQL before delegating to the real backend executor."""
        executed_sql.append(query)
        return original_execute(query, params, connection)

    backend.execute = types.MethodType(spy_execute, backend)
    try:
        CollectionsDatabase.from_backend(user_id="1", backend=backend)
    finally:
        backend.get_pool().close_all()

    assert not [  # nosec B101
        sql
        for sql in executed_sql
        if "ALTER TABLE CONTENT_ITEMS ADD COLUMN" in " ".join(sql.upper().split())
    ]
