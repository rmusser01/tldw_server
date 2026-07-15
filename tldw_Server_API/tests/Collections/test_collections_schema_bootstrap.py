"""Regression tests for Collections schema bootstrap SQL."""

import json
import sqlite3
import types
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

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


def test_fresh_media_collections_schema_has_private_playlist_owner_token(
    tmp_path: Path,
) -> None:
    backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "collections-private-token.db"),
        )
    )
    try:
        CollectionsDatabase.from_backend(user_id="1", backend=backend)
        columns = {row["name"] for row in backend.get_table_info("media_collections")}
    finally:
        backend.get_pool().close_all()

    assert "playlist_ingest_initialization_token" in columns


def test_schema_bootstrap_scrubs_legacy_public_token_without_trusting_it(
    tmp_path: Path,
) -> None:
    backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "collections-legacy-token.db"),
        )
    )
    try:
        db = CollectionsDatabase.from_backend(user_id="1", backend=backend)
        created = db.create_media_collection(
            name="Legacy playlist plan",
            kind="playlist_ingest",
            metadata={"playlist_ingest_run_id": "legacy-run"},
        )
        backend.execute(
            "UPDATE media_collections SET metadata_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "playlist_ingest_run_id": "legacy-run",
                        "playlist_ingest_initialization_token": "untrusted-public-token",
                    }
                ),
                created.id,
            ),
        )

        CollectionsDatabase.from_backend(user_id="1", backend=backend)
        row = backend.execute(
            "SELECT metadata_json, playlist_ingest_initialization_token "
            "FROM media_collections WHERE id = ?",
            (created.id,),
        ).first
    finally:
        backend.get_pool().close_all()

    assert row is not None
    assert row["playlist_ingest_initialization_token"] is None
    assert "playlist_ingest_initialization_token" not in json.loads(row["metadata_json"])
