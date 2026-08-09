from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

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


def test_note_folder_sync_id_is_returned_and_stable(db: CharactersRAGDB) -> None:
    created = db.create_note_folder_path("Stable/Child")
    sync_id = created["sync_id"]

    assert uuid.UUID(sync_id).version == 4
    assert db.get_note_folder_by_path("stable/child")["sync_id"] == sync_id
    assert {row["path"]: row["sync_id"] for row in db.list_note_folders()}["Stable/Child"] == sync_id
    assert db.create_note_folder_path("stable/child")["sync_id"] == sync_id


def test_folder_hierarchy_rename_and_move_recalculate_descendants_transactionally(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    parent = db.create_note_folder_path("Parent")
    child = db.create_note_folder_path("Parent/Child")
    grandchild = db.create_note_folder_path("Parent/Child/Grandchild")

    renamed = store.apply_resource(
        domain="notes.folder",
        object_id=parent["sync_id"],
        operation="upsert",
        payload={"name": "Renamed", "parent_sync_id": None},
    )
    assert renamed.sync_id == parent["sync_id"]
    assert db.get_note_folder_by_path("Renamed/Child/Grandchild")["sync_id"] == grandchild["sync_id"]

    moved = store.apply_resource(
        domain="notes.folder",
        object_id=child["sync_id"],
        operation="upsert",
        payload={"name": "Child", "parent_sync_id": None},
    )
    assert moved.parent_sync_id is None
    assert db.get_note_folder_by_path("Child/Grandchild")["sync_id"] == grandchild["sync_id"]


def test_folder_hierarchy_rejects_invalid_parents_cycles_and_long_paths_without_partial_changes(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    root = db.create_note_folder_path("Root")
    child = db.create_note_folder_path("Root/Child")
    before = {row["sync_id"]: row["path"] for row in db.list_note_folders()}

    invalid_mutations = (
        (root["sync_id"], {"name": "Root", "parent_sync_id": root["sync_id"]}),
        (root["sync_id"], {"name": "Root", "parent_sync_id": child["sync_id"]}),
        (child["sync_id"], {"name": "Child", "parent_sync_id": str(uuid.uuid4())}),
        (child["sync_id"], {"name": "x" * 500, "parent_sync_id": root["sync_id"]}),
    )
    for object_id, payload in invalid_mutations:
        with pytest.raises(InputError):
            store.apply_resource(
                domain="notes.folder",
                object_id=object_id,
                operation="upsert",
                payload=payload,
            )
        assert {row["sync_id"]: row["path"] for row in db.list_note_folders()} == before

    store.apply_resource(
        domain="notes.folder",
        object_id=root["sync_id"],
        operation="tombstone",
        payload={},
    )
    with pytest.raises(InputError):
        store.apply_resource(
            domain="notes.folder",
            object_id=child["sync_id"],
            operation="upsert",
            payload={"name": "Child", "parent_sync_id": root["sync_id"]},
        )


def test_folder_soft_delete_preserves_parent_pointer_and_membership_rows(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = db.add_note(title="Folder membership", content="preserved")
    parent = db.create_note_folder_path("Keep")
    child = db.create_note_folder_path("Keep/Linked")
    db.sync_note_folders(note_id, ["Keep/Linked"])

    deleted = store.apply_resource(
        domain="notes.folder",
        object_id=child["sync_id"],
        operation="tombstone",
        payload={},
    )
    assert deleted.deleted is True
    with db.transaction() as conn:
        folder = conn.execute(
            "SELECT parent_id, sync_id FROM note_folders WHERE id = ?",
            (child["id"],),
        ).fetchone()
        membership_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ? AND folder_id = ?",
            (note_id, child["id"]),
        ).fetchone()[0]
    assert folder["parent_id"] == parent["id"]
    assert folder["sync_id"] == child["sync_id"]
    assert membership_count == 1

    restored = store.apply_resource(
        domain="notes.folder",
        object_id=child["sync_id"],
        operation="upsert",
        payload={"name": "Linked", "parent_sync_id": parent["sync_id"]},
    )
    assert restored.deleted is False
    assert db.get_note_folder_by_path("Keep/Linked")["sync_id"] == child["sync_id"]


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
        and "WHERE" in statement
        and "deleted" in statement
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
