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
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id

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


@pytest.mark.parametrize("provenance", ["manual", "source", "mixed"])
def test_folder_link_tombstone_suppresses_effective_membership_without_deleting_source_provenance(
    db: CharactersRAGDB,
    provenance: str,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = db.add_note(title=f"{provenance} folder", content="suppression")
    folder = db.create_note_folder_path(f"Suppressed/{provenance}")
    if provenance in {"manual", "mixed"}:
        db.sync_note_folders(note_id, [folder["path"]])
    if provenance in {"source", "mixed"}:
        db.sync_note_source_folders(note_id, 41, [folder["path"]])

    payload = {"note_id": note_id, "folder_sync_id": folder["sync_id"]}
    object_id = organization_link_id(
        "notes.folder_link", [note_id, folder["sync_id"]]
    )

    for _ in range(2):
        store.apply_relationship(
            domain="notes.folder_link",
            object_id=object_id,
            operation="tombstone",
            payload=payload,
            routing_metadata={},
        )

    assert folder["sync_id"] not in {
        row["sync_id"] for row in db.get_note_folders_for_note(note_id)
    }
    assert folder["sync_id"] not in {
        row["sync_id"]
        for row in db.get_note_folders_for_notes([note_id])[note_id]
    }
    assert all(
        relationship.object_id != object_id
        for relationship in store.snapshot().relationships
    )
    with db.transaction() as conn:
        manual_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0]
        source_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 41, folder["id"]),
        ).fetchone()[0]
        source_key_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_keys WHERE source_id = ? AND folder_id = ?",
            (41, folder["id"]),
        ).fetchone()[0]
        suppression_count = conn.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0]
    assert manual_count == 0
    assert source_count == (1 if provenance in {"source", "mixed"} else 0)
    assert source_key_count == (1 if provenance in {"source", "mixed"} else 0)
    assert suppression_count == 1

    for _ in range(2):
        store.apply_relationship(
            domain="notes.folder_link",
            object_id=object_id,
            operation="upsert",
            payload=payload,
            routing_metadata={},
        )

    visible_sync_ids = [
        row["sync_id"] for row in db.get_note_folders_for_note(note_id)
    ]
    assert visible_sync_ids.count(folder["sync_id"]) == 1
    bulk_visible_sync_ids = [
        row["sync_id"]
        for row in db.get_note_folders_for_notes([note_id])[note_id]
    ]
    assert bulk_visible_sync_ids.count(folder["sync_id"]) == 1
    assert any(
        relationship.object_id == object_id
        for relationship in store.snapshot().relationships
    )
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 41, folder["id"]),
        ).fetchone()[0] == (1 if provenance in {"source", "mixed"} else 0)


@pytest.mark.timeout(2, method="signal", func_only=True)
def test_folder_hierarchy_rejects_preexisting_descendant_cycle_before_mutation(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    root = db.create_note_folder_path("Cycle")
    child = db.create_note_folder_path("Cycle/Child")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_folders SET parent_id = ? WHERE id = ?",
            (child["id"], root["id"]),
        )
        before = [
            tuple(row)
            for row in conn.execute(
                "SELECT id, name, path, parent_id, version FROM note_folders ORDER BY id"
            ).fetchall()
        ]

    with pytest.raises(InputError, match="cycle|invalid"):
        store.apply_resource(
            domain="notes.folder",
            object_id=root["sync_id"],
            operation="upsert",
            payload={"name": "Renamed", "parent_sync_id": None},
        )

    with db.transaction() as conn:
        after = [
            tuple(row)
            for row in conn.execute(
                "SELECT id, name, path, parent_id, version FROM note_folders ORDER BY id"
            ).fetchall()
        ]
    assert after == before


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
    assert any(
        "CREATE TABLE IF NOT EXISTS note_folder_sync_suppressions" in statement
        and "PRIMARY KEY(note_id, folder_id)" in statement
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


def test_effective_sync_union_emits_only_visible_source_transitions(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Union", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Union/Folder")

    assert store.source_folder_transition(
        note_id=note_id,
        source_id=11,
        folder_sync_id=folder["sync_id"],
        present=True,
    ) == "upsert"
    db.sync_note_source_folders(note_id, 11, [folder["path"]])
    assert store.source_folder_transition(
        note_id=note_id,
        source_id=22,
        folder_sync_id=folder["sync_id"],
        present=True,
    ) is None
    db.sync_note_source_folders(note_id, 22, [folder["path"]])
    assert store.source_folder_transition(
        note_id=note_id,
        source_id=11,
        folder_sync_id=folder["sync_id"],
        present=False,
    ) is None
    db.sync_note_source_folders(note_id, 11, [])
    assert store.source_folder_transition(
        note_id=note_id,
        source_id=22,
        folder_sync_id=folder["sync_id"],
        present=False,
    ) == "tombstone"

    db.sync_note_folders(note_id, [folder["path"]])
    assert store.source_folder_transition(
        note_id=note_id,
        source_id=22,
        folder_sync_id=folder["sync_id"],
        present=False,
    ) is None


def test_effective_sync_union_origin_provenance_and_projection_are_atomic(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Origin", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Origin/Folder")
    payload = {"note_id": note_id, "folder_sync_id": folder["sync_id"]}
    object_id = organization_link_id(
        "notes.folder_link", [note_id, folder["sync_id"]]
    )
    provenance = {"operation": "source_upsert", "source_id": 41}

    store.apply_relationship(
        domain="notes.folder_link",
        object_id=object_id,
        operation="upsert",
        payload=payload,
        routing_metadata={},
        origin_provenance=provenance,
    )
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 41, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 0

    monkeypatch.setattr(
        store,
        "_insert_link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rollback")),
    )
    with pytest.raises(RuntimeError, match="rollback"):
        store.apply_relationship(
            domain="notes.folder_link",
            object_id=object_id,
            operation="tombstone",
            payload=payload,
            routing_metadata={},
            origin_provenance={"operation": "source_delete", "source_id": 41},
        )
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 41, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 0


def test_effective_sync_union_trust_filter_is_owner_bound_and_allowlisted(
    db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.materializers import notes_organization

    envelope = type(
        "Envelope",
        (),
        {
            "domain": "notes.folder_link",
            "device_id": "server-origin",
            "routing_metadata": {
                "origin": "server",
                "server_device_id": "server-origin",
                "server_owner_user_id": "folder-test",
                "notes_folder_origin_provenance": {
                    "operation": "source_upsert",
                    "source_id": 51,
                },
            },
        },
    )()
    trusted = notes_organization._trusted_folder_origin_provenance(envelope, db)

    assert trusted == {"operation": "source_upsert", "source_id": 51}
    assert set(envelope.routing_metadata["notes_folder_origin_provenance"]) == {
        "operation",
        "source_id",
    }
    envelope.device_id = "remote-device"
    assert notes_organization._trusted_folder_origin_provenance(envelope, db) is None
    envelope.device_id = "server-origin"
    envelope.routing_metadata["notes_folder_origin_provenance"]["path"] = "/private"
    assert notes_organization._trusted_folder_origin_provenance(envelope, db) is None


def test_effective_sync_union_provenance_only_change_preserves_suppression(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Suppressed origin", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Suppressed/Origin")
    payload = {"note_id": note_id, "folder_sync_id": folder["sync_id"]}
    object_id = organization_link_id(
        "notes.folder_link", [note_id, folder["sync_id"]]
    )
    store.apply_relationship(
        domain="notes.folder_link",
        object_id=object_id,
        operation="tombstone",
        payload=payload,
        routing_metadata={},
    )

    store.apply_source_folder_provenance(
        note_id=note_id,
        folder_sync_id=folder["sync_id"],
        operation="source_upsert",
        source_id=61,
    )

    assert db.get_note_folders_for_note(note_id) == []
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_sync_suppressions "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 1


def test_source_folder_transition_guard_rejects_interleaved_noop_removals(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Interleaved sources", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Interleaved/Sources")
    db.sync_note_source_folders(note_id, 71, [folder["path"]])
    db.sync_note_source_folders(note_id, 72, [folder["path"]])

    remove_71 = store.source_folder_transition_plan(
        note_id=note_id,
        source_id=71,
        folder_sync_id=folder["sync_id"],
        present=False,
        transition_identity="request-remove-71",
    )
    remove_72 = store.source_folder_transition_plan(
        note_id=note_id,
        source_id=72,
        folder_sync_id=folder["sync_id"],
        present=False,
        transition_identity="request-remove-72",
    )

    assert remove_71.operation is None
    assert remove_72.operation is None
    assert store.apply_source_folder_provenance(
        note_id=note_id,
        folder_sync_id=folder["sync_id"],
        operation="source_delete",
        source_id=71,
        pre_state_hash=remove_71.pre_state_hash,
        post_state_hash=remove_71.post_state_hash,
        transition_identity=remove_71.transition_identity,
    ) is True
    with pytest.raises(ConflictError, match="changed after planning"):
        store.apply_source_folder_provenance(
            note_id=note_id,
            folder_sync_id=folder["sync_id"],
            operation="source_delete",
            source_id=72,
            pre_state_hash=remove_72.pre_state_hash,
            post_state_hash=remove_72.post_state_hash,
            transition_identity=remove_72.transition_identity,
        )

    with db.transaction() as conn:
        remaining = conn.execute(
            "SELECT source_id FROM note_folder_source_memberships "
            "WHERE note_id = ? AND folder_id = ? ORDER BY source_id",
            (note_id, folder["id"]),
        ).fetchall()
    assert [int(row["source_id"]) for row in remaining] == [72]
    assert [row["path"] for row in db.get_note_folders_for_note(note_id)] == [
        "Interleaved",
        "Interleaved/Sources",
    ]


def test_source_folder_transition_guard_accepts_provenance_only_post_state_retry(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Provenance retry", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Retry/Provenance")
    db.sync_note_source_folders(note_id, 81, [folder["path"]])
    db.sync_note_source_folders(note_id, 82, [folder["path"]])
    transition = store.source_folder_transition_plan(
        note_id=note_id,
        source_id=81,
        folder_sync_id=folder["sync_id"],
        present=False,
        transition_identity="request-provenance-retry",
    )

    kwargs = {
        "note_id": note_id,
        "folder_sync_id": folder["sync_id"],
        "operation": "source_delete",
        "source_id": 81,
        "pre_state_hash": transition.pre_state_hash,
        "post_state_hash": transition.post_state_hash,
        "transition_identity": transition.transition_identity,
    }
    assert store.apply_source_folder_provenance(**kwargs) is True
    assert store.apply_source_folder_provenance(**kwargs) is False

    with db.transaction() as conn:
        remaining = conn.execute(
            "SELECT source_id FROM note_folder_source_memberships "
            "WHERE note_id = ? AND folder_id = ? ORDER BY source_id",
            (note_id, folder["id"]),
        ).fetchall()
    assert [int(row["source_id"]) for row in remaining] == [82]


def test_source_folder_transition_guard_accepts_visible_post_state_retry(
    db: CharactersRAGDB,
) -> None:
    store = NotesOrganizationSyncStore(db)
    note_id = str(uuid.uuid4())
    db.add_note(title="Visible retry", content="Body", note_id=note_id)
    folder = db.create_note_folder_path("Retry/Visible")
    transition = store.source_folder_transition_plan(
        note_id=note_id,
        source_id=91,
        folder_sync_id=folder["sync_id"],
        present=True,
        transition_identity="group-visible-retry",
    )
    payload = {"note_id": note_id, "folder_sync_id": folder["sync_id"]}
    provenance = {
        "operation": "source_upsert",
        "source_id": 91,
        "pre_state_hash": transition.pre_state_hash,
        "post_state_hash": transition.post_state_hash,
    }
    kwargs = {
        "domain": "notes.folder_link",
        "object_id": organization_link_id(
            "notes.folder_link", [note_id, folder["sync_id"]]
        ),
        "operation": "upsert",
        "payload": payload,
        "routing_metadata": {},
        "origin_provenance": provenance,
        "source_transition_identity": transition.transition_identity,
    }

    assert store.apply_relationship(**kwargs) is True
    assert store.apply_relationship(**kwargs) is False
    with db.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_source_memberships "
            "WHERE note_id = ? AND source_id = ? AND folder_id = ?",
            (note_id, 91, folder["id"]),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM note_folder_memberships "
            "WHERE note_id = ? AND folder_id = ?",
            (note_id, folder["id"]),
        ).fetchone()[0] == 0
