from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    notes_db = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="folder-test")
    try:
        yield notes_db
    finally:
        notes_db.close_connection()


def test_create_note_folder_path_is_idempotent_and_listed(db: CharactersRAGDB) -> None:
    created = db.create_note_folder_path("Inbox/Captured Articles")
    duplicate = db.create_note_folder_path("inbox/captured articles")

    assert duplicate == created

    folders = db.list_note_folders()
    assert [folder["path"] for folder in folders] == [
        "Inbox",
        "Inbox/Captured Articles",
    ]
    assert folders[0]["parent_id"] is None
    assert folders[1]["parent_id"] == folders[0]["id"]


def test_postgres_note_folder_schema_enforces_case_insensitive_paths() -> None:
    class FakePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str, params: Any = None, connection: Any = None) -> None:
            self.statements.append(statement)

    backend = FakePostgresBackend()
    db_instance = CharactersRAGDB.__new__(CharactersRAGDB)
    db_instance._local = type("Local", (), {})()
    db_instance._backend = backend
    db_instance._uses_shared_content_backend = False

    db_instance._ensure_note_folder_schema_postgres(object())

    assert any(
        "UNIQUE INDEX" in statement
        and "note_folders" in statement
        and "LOWER(path)" in statement
        for statement in backend.statements
    )


def test_postgres_note_folder_schema_deduplicates_paths_before_unique_index() -> None:
    class FakePostgresBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.statements: list[str] = []

        def execute(self, statement: str, params: Any = None, connection: Any = None) -> None:
            self.statements.append(statement)

    backend = FakePostgresBackend()
    db_instance = CharactersRAGDB.__new__(CharactersRAGDB)
    db_instance._local = type("Local", (), {})()
    db_instance._backend = backend
    db_instance._uses_shared_content_backend = False

    db_instance._ensure_note_folder_schema_postgres(object())

    lower_unique_index = next(
        index
        for index, statement in enumerate(backend.statements)
        if "UNIQUE INDEX" in statement and "LOWER(path)" in statement
    )
    dedupe_statements = [
        statement
        for statement in backend.statements[:lower_unique_index]
        if "duplicate_folders" in statement
    ]

    assert any("note_folder_memberships" in statement for statement in dedupe_statements)
    assert any("note_folder_source_memberships" in statement for statement in dedupe_statements)
    assert any("note_folder_source_keys" in statement for statement in dedupe_statements)
    assert any("DELETE FROM note_folders" in statement for statement in dedupe_statements)
