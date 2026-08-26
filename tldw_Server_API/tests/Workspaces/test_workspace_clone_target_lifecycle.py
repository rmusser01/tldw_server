"""Deterministic staged lifecycle contracts for shared Workspace clone targets."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "clone-targets.sqlite"), client_id="user-1")
    yield database
    database.close_all_connections()


def _reserve(
    db: CharactersRAGDB,
    *,
    workspace_id: str = "workspace-target",
    operation_id: str = "operation-1",
    request_fingerprint: str = "fingerprint-1",
    name: str = "Target Workspace",
    description: str | None = "Cloned workspace",
    workspace_profile: str = "research",
) -> dict[str, object]:
    return db.reserve_clone_target(
        workspace_id=workspace_id,
        operation_id=operation_id,
        request_fingerprint=request_fingerprint,
        name=name,
        description=description,
        workspace_profile=workspace_profile,
    )


def _seed_workspace_snapshot(db: CharactersRAGDB, workspace_id: str = "workspace-source") -> None:
    db.upsert_workspace(workspace_id, "Workspace v1", description="Description v1")
    db.add_workspace_source(
        workspace_id,
        {
            "id": "source-1",
            "media_id": 101,
            "title": "Source v1",
            "source_type": "document",
            "position": 0,
        },
    )
    db.add_workspace_resource_membership(
        workspace_id,
        {
            "resource_type": "media",
            "resource_id": "101",
            "role": "source",
            "label": "Membership v1",
            "provenance": {"version": 1},
        },
        user_id="user-1",
    )
    db.add_workspace_note(
        workspace_id,
        {"title": "Note v1", "content": "Note content v1", "keywords": ["v1"]},
    )
    db.add_workspace_artifact(
        workspace_id,
        {
            "id": "artifact-1",
            "artifact_type": "report",
            "title": "Artifact v1",
            "content": "Artifact content v1",
        },
    )


def test_workspace_clone_snapshot_materializes_all_active_collections(
    db: CharactersRAGDB,
) -> None:
    _seed_workspace_snapshot(db)
    db.add_workspace_resource_membership(
        "workspace-source",
        {
            "resource_type": "media",
            "resource_id": "deleted-resource",
            "role": "source",
        },
    )
    db.delete_workspace_resource_membership(
        "workspace-source",
        "media",
        "deleted-resource",
    )
    deleted_note = db.add_workspace_note(
        "workspace-source",
        {"title": "Deleted note", "content": "not cloned"},
    )
    db.delete_workspace_note(
        "workspace-source",
        int(deleted_note["id"]),
    )

    snapshot = db.read_workspace_clone_snapshot("workspace-source")

    assert snapshot.workspace["name"] == "Workspace v1"
    assert [row["title"] for row in snapshot.sources] == ["Source v1"]
    assert [row["label"] for row in snapshot.memberships] == ["Membership v1"]
    assert [row["title"] for row in snapshot.notes] == ["Note v1"]
    assert [row["title"] for row in snapshot.artifacts] == ["Artifact v1"]


def test_workspace_clone_snapshot_does_not_mix_concurrent_collection_versions(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace_snapshot(db)
    writer = CharactersRAGDB(db.db_path_str, client_id="writer-1")
    original_execute = db.backend.execute
    source_row_read = False

    def interleaved_execute(query, params=None, connection=None, **kwargs):
        nonlocal source_row_read
        result = original_execute(query, params, connection=connection, **kwargs)
        if not source_row_read and "FROM workspaces" in query:
            source_row_read = True
            with writer.transaction() as writer_conn:
                writer_conn.execute(
                    "UPDATE workspaces SET name = ?, version = version + 1 WHERE id = ?",
                    ("Workspace v2", "workspace-source"),
                )
                writer_conn.execute(
                    "UPDATE workspace_sources SET title = ?, version = version + 1 WHERE workspace_id = ?",
                    ("Source v2", "workspace-source"),
                )
                writer_conn.execute(
                    "UPDATE workspace_resource_memberships SET label = ?, version = version + 1 "
                    "WHERE workspace_id = ?",
                    ("Membership v2", "workspace-source"),
                )
                writer_conn.execute(
                    "UPDATE workspace_notes SET title = ?, version = version + 1 WHERE workspace_id = ?",
                    ("Note v2", "workspace-source"),
                )
                writer_conn.execute(
                    "UPDATE workspace_artifacts SET title = ?, version = version + 1 WHERE workspace_id = ?",
                    ("Artifact v2", "workspace-source"),
                )
        return result

    monkeypatch.setattr(db.backend, "execute", interleaved_execute)
    try:
        snapshot = db.read_workspace_clone_snapshot("workspace-source")
    finally:
        writer.close_all_connections()

    assert source_row_read is True
    assert snapshot.workspace["name"] == "Workspace v1"
    assert snapshot.sources[0]["title"] == "Source v1"
    assert snapshot.memberships[0]["label"] == "Membership v1"
    assert snapshot.notes[0]["title"] == "Note v1"
    assert snapshot.artifacts[0]["title"] == "Artifact v1"
    assert db.get_workspace("workspace-source")["name"] == "Workspace v2"


@pytest.mark.parametrize("workspace_id", ["missing-workspace", "hidden-workspace"])
def test_workspace_clone_snapshot_rejects_missing_or_hidden_sources(
    db: CharactersRAGDB,
    workspace_id: str,
) -> None:
    if workspace_id == "hidden-workspace":
        _reserve(db, workspace_id=workspace_id)

    with pytest.raises(CloneSnapshotUnavailable) as exc_info:
        db.read_workspace_clone_snapshot(workspace_id)

    assert str(exc_info.value) == "source_snapshot_unavailable"
    assert db.execute_query(
        "SELECT COUNT(*) AS count FROM workspaces WHERE id = ?",
        ("unexpected-target",),
    ).fetchone()["count"] == 0


def test_workspace_clone_snapshot_rejects_private_memory_database() -> None:
    memory_db = CharactersRAGDB(":memory:", client_id="memory-source")
    memory_db.upsert_workspace("workspace-source", "Memory Workspace")
    try:
        with pytest.raises(CloneSnapshotUnavailable):
            memory_db.read_workspace_clone_snapshot("workspace-source")
    finally:
        memory_db.close_all_connections()


def test_workspace_clone_snapshot_reads_named_shared_cache_memory_database() -> None:
    db_uri = f"file:task3-workspace-{uuid.uuid4()}?mode=memory&cache=shared"
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_uri)
    )
    memory_db = CharactersRAGDB(
        ":memory:",
        client_id="shared-memory",
        backend=backend,
    )
    try:
        _seed_workspace_snapshot(memory_db)

        snapshot = memory_db.read_workspace_clone_snapshot("workspace-source")

        assert snapshot.workspace["name"] == "Workspace v1"
        assert snapshot.sources[0]["title"] == "Source v1"
    finally:
        memory_db.close_all_connections()
        backend.get_pool().close_all()


def test_workspace_clone_snapshot_redacts_setup_failure(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace_snapshot(db)
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")

    def fail_connect():
        raise sqlite3.OperationalError("sensitive backend detail")

    monkeypatch.setattr(db.backend, "connect", fail_connect)

    try:
        with pytest.raises(CloneSnapshotUnavailable) as exc_info:
            db.read_workspace_clone_snapshot("workspace-source")
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "source_snapshot_unavailable"
    assert "sensitive backend detail" not in repr(exc_info.value)
    assert "sensitive backend detail" not in "".join(messages)
    assert "Workspace clone snapshot read failed" in "".join(messages)


@pytest.mark.parametrize("workspace_id", ["workspace-source", "missing-workspace"])
def test_workspace_clone_snapshot_closes_dedicated_handle_on_every_path(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    workspace_id: str,
) -> None:
    if workspace_id == "workspace-source":
        _seed_workspace_snapshot(db)
    caller_connection = db.get_connection()
    opened_connections = []
    release_states: list[bool] = []
    original_connect = db.backend.connect
    original_disconnect = db.backend.disconnect

    def tracked_connect():
        connection = original_connect()
        opened_connections.append(connection)
        return connection

    def tracked_disconnect(connection) -> None:
        release_states.append(bool(connection.in_transaction))
        original_disconnect(connection)

    monkeypatch.setattr(db.backend, "connect", tracked_connect)
    monkeypatch.setattr(db.backend, "disconnect", tracked_disconnect)

    if workspace_id == "workspace-source":
        db.read_workspace_clone_snapshot(workspace_id)
    else:
        with pytest.raises(CloneSnapshotUnavailable):
            db.read_workspace_clone_snapshot(workspace_id)

    assert len(opened_connections) == 1
    assert opened_connections[0] is not caller_connection
    assert release_states == [False]
    with pytest.raises(sqlite3.ProgrammingError):
        opened_connections[0].execute("SELECT 1")


def test_first_reservation_is_staged_archived_and_hidden(db: CharactersRAGDB) -> None:
    reserved = _reserve(db, name="  Target   Workspace  ")

    assert reserved["id"] == "workspace-target"
    assert reserved["name"] == "Target Workspace"
    assert bool(reserved["archived"]) is True
    assert reserved["system_operation_id"] == "operation-1"
    assert reserved["system_operation_kind"] == "shared_workspace_clone"
    assert reserved["system_operation_state"] == "staged"
    assert reserved["system_request_fingerprint"] == "fingerprint-1"
    assert db.get_workspace("workspace-target") is None
    assert db.get_workspace("workspace-target", include_deleted=True) is None
    assert db.list_workspaces() == []


def test_identical_reservation_is_idempotent(db: CharactersRAGDB) -> None:
    first = _reserve(db, name="Target   Workspace")
    second = _reserve(db, name="  Target Workspace  ")

    assert second == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_fingerprint", "fingerprint-2"),
        ("name", "Different Workspace"),
        ("description", None),
        ("workspace_profile", "project"),
    ],
)
def test_same_operation_reservation_requires_exact_request_match(
    db: CharactersRAGDB,
    field: str,
    value: object,
) -> None:
    _reserve(db)
    kwargs = {field: value}

    with pytest.raises(ConflictError):
        _reserve(db, **kwargs)


def test_reservation_conflicts_with_ordinary_workspace(db: CharactersRAGDB) -> None:
    db.upsert_workspace("workspace-target", "Ordinary Workspace")

    with pytest.raises(ConflictError):
        _reserve(db)


def test_reservation_conflicts_with_another_operation(db: CharactersRAGDB) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        _reserve(db, operation_id="operation-2")


def test_publish_moves_owned_staged_target_to_hidden_publication_pending(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    published = db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert bool(published["archived"]) is False
    assert published["system_operation_state"] == "publication_pending"
    assert db.get_workspace("workspace-target") is None
    assert db.list_workspaces() == []


def test_identical_reservation_replays_hidden_publication_pending_target(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)
    published = db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    replayed = _reserve(db)

    assert replayed == published
    assert bool(replayed["archived"]) is False
    assert replayed["system_operation_state"] == "publication_pending"
    assert db.get_workspace("workspace-target") is None
    assert db.get_workspace("workspace-target", include_deleted=True) is None
    assert db.list_workspaces() == []


def test_publish_rejects_wrong_operation(db: CharactersRAGDB) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        db.publish_clone_target(
            workspace_id="workspace-target",
            operation_id="operation-2",
        )


def test_confirmation_clears_all_markers_and_exposes_workspace(db: CharactersRAGDB) -> None:
    _reserve(db)
    db.publish_clone_target(workspace_id="workspace-target", operation_id="operation-1")

    confirmed = db.confirm_clone_target_publication(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert bool(confirmed["archived"]) is False
    assert confirmed["system_operation_id"] is None
    assert confirmed["system_operation_kind"] is None
    assert confirmed["system_operation_state"] is None
    assert confirmed["system_request_fingerprint"] is None
    assert db.get_workspace("workspace-target") == confirmed
    assert db.list_workspaces() == [confirmed]


def test_confirmation_requires_exact_owned_publication_pending_row(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    with pytest.raises(ConflictError):
        db.confirm_clone_target_publication(
            workspace_id="workspace-target",
            operation_id="operation-1",
        )


@pytest.mark.parametrize("publish_first", [False, True])
def test_discard_soft_deletes_exact_owned_target(
    db: CharactersRAGDB,
    publish_first: bool,
) -> None:
    _reserve(db)
    if publish_first:
        db.publish_clone_target(workspace_id="workspace-target", operation_id="operation-1")

    assert db.discard_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    ) is True
    row = db.execute_query(
        "SELECT deleted FROM workspaces WHERE id = ?",
        ("workspace-target",),
    ).fetchone()
    assert bool(row["deleted"]) is True


def test_discard_is_operation_fenced(db: CharactersRAGDB) -> None:
    _reserve(db)

    assert db.discard_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-2",
    ) is False
    row = db.execute_query(
        "SELECT deleted, system_operation_state FROM workspaces WHERE id = ?",
        ("workspace-target",),
    ).fetchone()
    assert bool(row["deleted"]) is False
    assert row["system_operation_state"] == "staged"


def test_reconciliation_lookup_is_caller_correlated_and_excludes_deleted_rows(
    db: CharactersRAGDB,
) -> None:
    _reserve(db, workspace_id="workspace-staged", operation_id="operation-staged")
    _reserve(db, workspace_id="workspace-pending", operation_id="operation-pending")
    db.publish_clone_target(workspace_id="workspace-pending", operation_id="operation-pending")
    _reserve(db, workspace_id="workspace-deleted", operation_id="operation-deleted")
    db.discard_clone_target(workspace_id="workspace-deleted", operation_id="operation-deleted")
    _reserve(db, workspace_id="workspace-unrequested", operation_id="operation-unrequested")
    db.upsert_workspace("workspace-ordinary", "Ordinary")

    rows = db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-pending", "operation-staged", "operation-deleted"],
        limit=2,
    )

    assert [(row["system_operation_id"], row["system_operation_state"]) for row in rows] == [
        ("operation-pending", "publication_pending"),
        ("operation-staged", "staged"),
    ]


def test_reconciliation_lookup_accepts_empty_operation_ids_without_scanning(
    db: CharactersRAGDB,
) -> None:
    _reserve(db)

    assert db.list_clone_targets_for_reconciliation(operation_ids=[]) == []


@pytest.mark.parametrize(
    ("operation_ids", "limit"),
    [
        (["operation-1"], 0),
        (["operation-1"], 101),
        (["operation-1"] * 101, 100),
        ("operation-1", 100),
        (["bad operation"], 100),
    ],
)
def test_reconciliation_lookup_rejects_unbounded_or_invalid_inputs(
    db: CharactersRAGDB,
    operation_ids: object,
    limit: int,
) -> None:
    with pytest.raises(InputError):
        db.list_clone_targets_for_reconciliation(
            operation_ids=operation_ids,  # type: ignore[arg-type]
            limit=limit,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_id", "bad workspace"),
        ("operation_id", ""),
        ("request_fingerprint", "fingerprint\nvalue"),
        ("name", "   "),
        ("name", "x" * 256),
        ("workspace_profile", "other"),
    ],
)
def test_reservation_validates_clone_identity_and_workspace_fields(
    db: CharactersRAGDB,
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(InputError):
        _reserve(db, **kwargs)
