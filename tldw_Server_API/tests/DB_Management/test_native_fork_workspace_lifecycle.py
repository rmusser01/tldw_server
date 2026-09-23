"""Workspace deletion closes native admission before its staged cascade."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgres"])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    kwargs = {"db_path": str(tmp_path / "workspace.sqlite"), "client_id": "alice"}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = CharactersRAGDB(**kwargs)
    database.upsert_workspace("workspace-one", "Workspace One")
    yield database
    database.close_connection()
    if request.param == "postgres":
        kwargs["backend"].get_pool().close_all()


def test_soft_delete_closure_survives_cascade_failure_and_retry(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Child", "scope_type": "workspace", "workspace_id": "workspace-one"})
    original = db.soft_delete_conversation

    def fail_once(conversation_id: str, expected_version: int) -> None:
        monkeypatch.setattr(db, "soft_delete_conversation", original)
        raise RuntimeError("injected cascade failure")

    monkeypatch.setattr(db, "soft_delete_conversation", fail_once)
    with pytest.raises(ConflictError, match="workspace_delete_incomplete"):
        db.delete_workspace("workspace-one", expected_version=1)
    workspace = db.get_workspace("workspace-one")
    assert workspace is not None and bool(workspace["native_chat_admission_closed"])
    assert workspace["version"] == 1
    updated = db.update_workspace("workspace-one", {"name": "Renamed before retry"}, expected_version=1)
    assert bool(updated["native_chat_admission_closed"])
    assert db.delete_workspace("workspace-one", expected_version=updated["version"])
    assert db.get_workspace("workspace-one") is None
    assert db.get_conversation_by_id(child, include_deleted=True)["deleted"]


@pytest.mark.parametrize("hard", [False, True])
def test_admission_closure_commits_before_conversation_enumeration(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, hard: bool) -> None:
    enumerating = threading.Event()
    allow_enumeration = threading.Event()
    original = db.execute_query
    results = {}
    enumeration_prefix = (
        "SELECT id FROM conversations WHERE workspace_id"
        if hard else "SELECT id, version FROM conversations WHERE workspace_id"
    )

    def pause_at_enumeration(query: str, params: tuple[Any, ...] = (), **kwargs: Any) -> Any:
        if query.startswith(enumeration_prefix):
            enumerating.set()
            assert allow_enumeration.wait(timeout=10)
        return original(query, params, **kwargs)

    monkeypatch.setattr(db, "execute_query", pause_at_enumeration)

    def delete() -> None:
        try:
            if hard:
                results["deleted"] = db.hard_delete_workspace("workspace-one")
            else:
                results["deleted"] = db.delete_workspace("workspace-one", expected_version=1)
        except Exception as exc:  # noqa: BLE001 - surfaced below
            results["error"] = exc
        finally:
            db.close_connection()

    thread = threading.Thread(target=delete)
    thread.start()
    observer = CharactersRAGDB(db_path=db.db_path_str, client_id="alice", backend=db.backend)
    try:
        assert enumerating.wait(timeout=10)
        workspace = observer.get_workspace("workspace-one")
        assert workspace is not None and bool(workspace["native_chat_admission_closed"])
        assert workspace["version"] == 1
    finally:
        allow_enumeration.set()
        thread.join(timeout=10)
        observer.close_connection()
    assert not thread.is_alive()
    assert "error" not in results
    assert results["deleted"] is (None if hard else True)


@pytest.mark.parametrize("hard", [False, True])
def test_workspace_staged_delete_rejects_enclosing_transaction_without_committing_caller(db: CharactersRAGDB, hard: bool) -> None:
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction() as conn:
            conn.execute("UPDATE workspaces SET name = 'Caller change' WHERE id = 'workspace-one'")
            with pytest.raises(ConflictError, match="outermost"):
                if hard:
                    db.hard_delete_workspace("workspace-one")
                else:
                    db.delete_workspace("workspace-one", expected_version=1)
            raise RuntimeError("caller rollback")
    workspace = db.get_workspace("workspace-one")
    assert workspace is not None and workspace["name"] == "Workspace One"
    assert not bool(workspace["native_chat_admission_closed"])


def test_hard_delete_closure_survives_cascade_failure_and_retry(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    db.add_conversation({"character_id": 1, "title": "Child", "scope_type": "workspace", "workspace_id": "workspace-one"})
    original = db.hard_delete_conversation

    def fail_once(conversation_id: str) -> None:
        monkeypatch.setattr(db, "hard_delete_conversation", original)
        raise RuntimeError("injected cascade failure")

    monkeypatch.setattr(db, "hard_delete_conversation", fail_once)
    with pytest.raises(ConflictError, match="workspace_delete_incomplete"):
        db.hard_delete_workspace("workspace-one")
    workspace = db.get_workspace("workspace-one")
    assert workspace is not None and bool(workspace["native_chat_admission_closed"])
    db.hard_delete_workspace("workspace-one")
    assert db.get_workspace("workspace-one", include_deleted=True) is None


@pytest.mark.parametrize(("hard", "trashed"), [(False, False), (True, False), (True, True)])
def test_final_transition_preserves_workspace_when_protected_chat_was_missed(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, hard: bool, trashed: bool) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    if trashed:
        db.soft_delete_conversation(child, expected_version=1)
    original = db.execute_query

    def miss_enumeration(query: str, params: tuple[Any, ...] = (), **kwargs: Any) -> Any:
        if query.startswith("SELECT id") and "FROM conversations WHERE workspace_id" in query:
            monkeypatch.setattr(db, "execute_query", original)
            return original("SELECT id FROM conversations WHERE 1 = 0", ())
        return original(query, params, **kwargs)

    monkeypatch.setattr(db, "execute_query", miss_enumeration)
    with pytest.raises(ConflictError, match="workspace_delete_incomplete"):
        if hard:
            db.hard_delete_workspace("workspace-one")
        else:
            db.delete_workspace("workspace-one", expected_version=1)
    workspace = db.get_workspace("workspace-one", include_deleted=True)
    assert workspace is not None and bool(workspace["native_chat_admission_closed"])
    assert not workspace["deleted"]
    assert db.get_conversation_by_id(child, include_deleted=True)["workspace_id"] == "workspace-one"


def test_stale_soft_delete_does_not_close_admission(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="version mismatch"):
        db.delete_workspace("workspace-one", expected_version=0)
    assert not bool(db.get_workspace("workspace-one")["native_chat_admission_closed"])


def test_other_owner_cannot_close_or_delete_workspace(db: CharactersRAGDB) -> None:
    other = CharactersRAGDB(db_path=db.db_path_str, client_id="bob", backend=db.backend)
    try:
        with pytest.raises(ConflictError, match="not found"):
            other.delete_workspace("workspace-one", expected_version=1)
        other.hard_delete_workspace("workspace-one")
        workspace = db.get_workspace("workspace-one")
        assert workspace is not None and not bool(workspace["native_chat_admission_closed"])
    finally:
        other.close_connection()


def test_hard_delete_missing_workspace_remains_idempotent(db: CharactersRAGDB) -> None:
    db.hard_delete_workspace("workspace-one")
    db.hard_delete_workspace("workspace-one")
    assert db.get_workspace("workspace-one", include_deleted=True) is None


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("typed_asset", [False, True])
def test_protected_restore_requires_open_workspace(db: CharactersRAGDB, closed: bool, typed_asset: bool) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        if typed_asset:
            conn.execute("UPDATE conversations SET native_bundle_json = '{}' WHERE id = ?", (child,))
        else:
            conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    db.soft_delete_conversation(child, expected_version=1)
    if closed:
        with db.transaction() as conn:
            conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "workspace-one"))
        with pytest.raises(ConflictError, match="workspace_native_unavailable"):
            db.restore_conversation(child, expected_version=2)
        assert db.get_conversation_by_id(child, include_deleted=True)["deleted"]
    else:
        assert db.restore_conversation(child, expected_version=2)
        assert not db.get_conversation_by_id(child, include_deleted=True)["deleted"]


def test_protected_restore_rejects_other_owner(db: CharactersRAGDB) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    db.soft_delete_conversation(child, expected_version=1)
    other = CharactersRAGDB(db_path=db.db_path_str, client_id="bob", backend=db.backend)
    try:
        with pytest.raises(ConflictError):
            other.restore_conversation(child, expected_version=2)
        assert db.get_conversation_by_id(child, include_deleted=True)["deleted"]
    finally:
        other.close_connection()


def test_protected_restore_cannot_reopen_deleted_workspace(db: CharactersRAGDB) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    db.soft_delete_conversation(child, expected_version=1)
    assert db.delete_workspace("workspace-one", expected_version=1)
    with pytest.raises(ConflictError, match="workspace_native_unavailable"):
        db.restore_conversation(child, expected_version=2)
    assert db.get_conversation_by_id(child, include_deleted=True)["deleted"]


def test_resume_state_retains_snapshot_bound_assistant_identity(db: CharactersRAGDB) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Snapshot-bound child"})
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET character_id = NULL, assistant_kind = ?, assistant_id = ? WHERE id = ?",
            ("character", f"snapshot:{child}", child),
        )
    assert db.get_roleplay_resume_state(child)["conversation"]["assistant_binding_mode"] is None
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET required_projection_version = ?, "
            "native_creation_operation_kind = ?, native_creation_operation_id = ? WHERE id = ?",
            ("native-fork-v1", "native_fork_v1", "source-fork", child),
        )
    state = db.get_roleplay_resume_state(child)
    assert state["conversation"]["assistant_binding_mode"] == "snapshot_v1"


def test_postgres_workspace_delete_rejects_caller_owned_driver_transaction(db: CharactersRAGDB) -> None:
    if db.backend_type != BackendType.POSTGRESQL:
        pytest.skip("PostgreSQL driver transaction behavior")
    db.execute_query(
        "UPDATE workspaces SET name = ? WHERE id = ?",
        ("Uncommitted rename", "workspace-one"),
        commit=False,
    )
    raw = db.get_connection()._connection
    assert raw.info.transaction_status.name == "INTRANS"
    try:
        with pytest.raises(ConflictError, match="outermost transaction"):
            db.delete_workspace("workspace-one", expected_version=1)
        assert raw.info.transaction_status.name == "INTRANS"
    finally:
        raw.rollback()
    assert db.get_workspace("workspace-one")["name"] == "Workspace One"


def test_workspace_delete_guard_reads_postgres_driver_transaction_status() -> None:
    raw = SimpleNamespace(info=SimpleNamespace(transaction_status=SimpleNamespace(name="INTRANS")))
    backend = SimpleNamespace(_tx_depth=lambda _connection: 0)
    fake_db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        backend=backend,
        get_connection=lambda: SimpleNamespace(_connection=raw, _backend=backend),
        _connection_state=lambda: SimpleNamespace(tx_depth=0),
    )
    with pytest.raises(ConflictError, match="outermost transaction"):
        CharactersRAGDB._require_outermost_workspace_delete(fake_db, "workspace-one")


def test_active_only_restore_validation_never_reactivates_a_deleted_chat(db: CharactersRAGDB) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Child", "scope_type": "workspace", "workspace_id": "workspace-one"})
    db.soft_delete_conversation(child, expected_version=1)
    with pytest.raises(ConflictError, match="chat_restore_state_changed"):
        db.restore_conversation(child, expected_version=2, require_already_active=True)
    assert db.get_conversation_by_id(child, include_deleted=True)["deleted"]


def test_active_restore_endpoint_rejects_delete_between_read_and_validation(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Racing child"})
    original_get = db.get_conversation_by_id

    def read_then_delete(conversation_id: str, include_deleted: bool = False) -> Any:
        snapshot = original_get(conversation_id, include_deleted=include_deleted)
        if include_deleted and snapshot and not snapshot["deleted"]:
            db.soft_delete_conversation(conversation_id, expected_version=snapshot["version"])
        return snapshot

    monkeypatch.setattr(db, "get_conversation_by_id", read_then_delete)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(character_chat_sessions.restore_chat_session(
            chat_id=child,
            expected_version=2,
            scope_type=None,
            workspace_id=None,
            db=db,
            current_user=SimpleNamespace(id="alice"),
        ))
    assert exc_info.value.status_code == 409
    row = original_get(child, include_deleted=True)
    assert row["deleted"] and row["version"] == 2


@pytest.mark.parametrize("typed_asset", [False, True])
@pytest.mark.parametrize("trashed", [False, True])
def test_sync_upsert_cannot_reparent_or_revive_protected_chat(db: CharactersRAGDB, typed_asset: bool, trashed: bool) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        if typed_asset:
            conn.execute("UPDATE conversations SET native_bundle_json = '{}' WHERE id = ?", (child,))
        else:
            conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    if trashed:
        db.soft_delete_conversation(child, expected_version=1)
    before = db.get_conversation_by_id(child, include_deleted=True)
    with pytest.raises(ConflictError, match="native_conversation_sync_unsupported"):
        db.upsert_conversation_from_sync(
            conversation_id=child,
            title="Reparented",
            sync_client_id="sync-device",
            object_revision=20,
            object_hash="sha256:sync",
            scope_type="global",
        )
    assert db.get_conversation_by_id(child, include_deleted=True) == before


def test_sync_tombstone_cannot_bypass_protected_delete_adapter(db: CharactersRAGDB) -> None:
    child = db.add_conversation({"character_id": 1, "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one"})
    with db.transaction() as conn:
        conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    before = db.get_conversation_by_id(child, include_deleted=True)
    with pytest.raises(ConflictError, match="native_conversation_sync_unsupported"):
        db.tombstone_conversation_from_sync(
            conversation_id=child,
            sync_client_id="sync-device",
            object_revision=20,
            object_hash="sha256:sync-delete",
        )
    assert db.get_conversation_by_id(child, include_deleted=True) == before


@pytest.mark.parametrize("typed_asset", [False, True])
@pytest.mark.parametrize(
    ("identity_field", "new_value"),
    [
        ("assistant_kind", "character"),
        ("assistant_id", "persona-rebound"),
        ("character_id", "character-rebound"),
        ("persona_memory_mode", "none"),
    ],
)
def test_generic_identity_update_cannot_rebind_protected_chat(db: CharactersRAGDB, typed_asset: bool, identity_field: str, new_value: object) -> None:
    child = db.add_conversation({
        "assistant_kind": "persona", "assistant_id": "persona-original",
        "title": "Protected", "scope_type": "workspace", "workspace_id": "workspace-one",
    })
    with db.transaction() as conn:
        if typed_asset:
            conn.execute("UPDATE conversations SET native_bundle_json = '{}' WHERE id = ?", (child,))
        else:
            conn.execute("UPDATE conversations SET required_projection_version = 'native-fork-v1' WHERE id = ?", (child,))
    before = db.get_conversation_by_id(child)
    with pytest.raises(ConflictError, match="native_assistant_identity_locked"):
        db.update_conversation(child, {identity_field: new_value}, expected_version=before["version"])
    assert db.get_conversation_by_id(child) == before
    assert db.update_conversation(child, {"title": "Retitled"}, expected_version=before["version"])
    after = db.get_conversation_by_id(child)
    assert after["title"] == "Retitled" and after["assistant_id"] == "persona-original"
