from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    close_all_backends,
)


def _collections_db(tmp_path: Path) -> tuple[CollectionsDatabase, object]:
    close_all_backends()
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "collections.db"),
        )
    )
    return CollectionsDatabase.from_backend(user_id="91", backend=backend), backend


def test_outputs_schema_adds_nullable_idempotency_key_and_active_unique_index(tmp_path: Path) -> None:
    db, backend = _collections_db(tmp_path)
    try:
        columns = {row["name"] for row in backend.get_table_info("outputs")}
        indexes = {row["name"] for row in backend.execute("PRAGMA index_list(outputs)", ()).rows}

        assert "idempotency_key" in columns
        assert "ux_outputs_user_idempotency_active" in indexes
        assert "ux_outputs_user_idempotency" not in indexes
    finally:
        db.close()
        close_all_backends()


def test_active_idempotency_index_creation_survives_legacy_drop_noop() -> None:
    db = CollectionsDatabase.__new__(CollectionsDatabase)
    statements: list[str] = []

    class Backend:
        backend_type = BackendType.SQLITE

        @staticmethod
        def execute(statement: str, _params: tuple[object, ...]) -> None:
            statements.append(statement)
            if statement.startswith("DROP INDEX"):
                raise RuntimeError("index already exists")

    db._backend = Backend()
    db._local = SimpleNamespace()
    db._uses_shared_content_backend = False

    db._ensure_output_idempotency_index()

    assert any(statement.startswith("CREATE UNIQUE INDEX") for statement in statements)


def test_create_output_artifact_reuses_idempotency_key_but_keeps_legacy_callers(tmp_path: Path) -> None:
    db, _backend = _collections_db(tmp_path)
    try:
        first = db.create_output_artifact(
            type_="watchlist_briefing",
            title="Morning briefing",
            format_="markdown",
            storage_path="briefing.md",
            idempotency_key="occurrence:11:text:v1",
        )
        replay = db.create_output_artifact(
            type_="watchlist_briefing",
            title="Morning briefing",
            format_="markdown",
            storage_path="briefing.md",
            idempotency_key="occurrence:11:text:v1",
        )
        legacy = db.create_output_artifact(
            type_="summary",
            title="Unkeyed output",
            format_="markdown",
            storage_path="legacy.md",
        )

        assert replay.id == first.id
        assert replay.idempotency_key == "occurrence:11:text:v1"
        assert legacy.idempotency_key is None
    finally:
        db.close()
        close_all_backends()


def test_concurrent_create_output_artifact_has_one_logical_key(tmp_path: Path) -> None:
    db, _backend = _collections_db(tmp_path)
    try:

        def create() -> int:
            return db.create_output_artifact(
                type_="watchlist_briefing",
                title="Concurrent briefing",
                format_="markdown",
                storage_path="concurrent.md",
                idempotency_key="occurrence:12:text:v1",
            ).id

        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(lambda _index: create(), range(2)))

        rows, total = db.list_output_artifacts(type_="watchlist_briefing")
        assert ids[0] == ids[1]
        assert total == 1
        assert rows[0].idempotency_key == "occurrence:12:text:v1"
    finally:
        db.close()
        close_all_backends()


def test_soft_deleted_key_can_be_recreated_without_removing_tombstone(tmp_path: Path) -> None:
    db, backend = _collections_db(tmp_path)
    try:
        first = db.create_output_artifact(
            type_="watchlist_briefing",
            title="Recreatable briefing",
            format_="markdown",
            storage_path="recreatable.md",
            idempotency_key="occurrence:13:text:v1",
        )
        assert db.delete_output_artifact(first.id) is True

        recreated = db.create_output_artifact(
            type_="watchlist_briefing",
            title="Recreatable briefing",
            format_="markdown",
            storage_path="recreatable.md",
            idempotency_key="occurrence:13:text:v1",
        )

        assert recreated.id != first.id
        assert db.get_output_artifact_by_idempotency_key("occurrence:13:text:v1").id == recreated.id
        tombstones = backend.execute(
            "SELECT id FROM outputs WHERE user_id = ? AND idempotency_key = ? AND deleted = 1",
            ("91", "occurrence:13:text:v1"),
        ).rows
        assert [row["id"] for row in tombstones] == [first.id]
    finally:
        db.close()
        close_all_backends()
