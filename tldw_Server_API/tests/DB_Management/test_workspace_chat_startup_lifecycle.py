"""Receipt lifecycle follows real conversation mutations, never reconstructed authority."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import WorkspaceStartupError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _start

pytestmark = pytest.mark.integration


def _receipt(db: CharactersRAGDB) -> dict[str, Any]:
    """Read the actual accepted key in an explicitly settled observer transaction."""
    with db.transaction() as conn:
        row = db.workspace_chat_startups.get_receipt(
            db.owner_user_id, hashlib.sha256(b"accepted-1").hexdigest(), conn=conn,
        )
        assert row is not None
        return row


@pytest.mark.parametrize("change", [
    {"assistant_id": "persona-b"},
    {"persona_memory_mode": "read_write"},
    {"assistant_kind": None, "assistant_id": None, "persona_memory_mode": None},
])
def test_binding_mutation_permanently_invalidates_receipt(
    creation_db: CharactersRAGDB, change: dict[str, Any],
) -> None:
    """Actual kind, Persona or memory changes stamp the receipt in the same unit."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    with creation_db.transaction():
        creation_db.update_conversation(cid, change, first.conversation["version"])
    changed = _receipt(creation_db)
    assert changed["invalidated_at"] is not None
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid)
        creation_db.update_conversation(cid, {
            "assistant_kind": "persona", "assistant_id": "persona-a",
            "character_id": None, "persona_memory_mode": "read_only",
        }, row["version"])
    assert _receipt(creation_db)["invalidated_at"] == changed["invalidated_at"]
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert caught.value.code == "workspace_chat_startup_changed"


def test_sync_scope_move_back_cannot_revalidate_accepted_key(creation_db: CharactersRAGDB) -> None:
    """Restoring the exact original binding digest cannot undo durable invalidation."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    for scope, workspace in (("global", None), ("workspace", "ws")):
        creation_db.upsert_conversation_from_sync(
            conversation_id=cid, title=first.conversation["title"],
            sync_client_id="other-device", object_revision=2, object_hash="h",
            assistant_kind="persona", assistant_id="persona-a", persona_memory_mode="read_only",
            scope_type=scope, workspace_id=workspace,
        )
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert caught.value.code == "workspace_chat_startup_changed"
    assert _receipt(creation_db)["invalidated_at"] is not None


@pytest.mark.parametrize("change", [
    {"assistant_id": "persona-b"},
    {"persona_memory_mode": "read_write"},
    {"assistant_kind": None, "assistant_id": None, "persona_memory_mode": None},
])
def test_sync_binding_mutation_permanently_invalidates_receipt(
    creation_db: CharactersRAGDB, change: dict[str, Any],
) -> None:
    """Whole-object Sync identity changes burn authority despite later reversal."""
    first = _start(creation_db)
    original = {"assistant_kind": "persona", "assistant_id": "persona-a", "persona_memory_mode": "read_only"}
    stamp = None
    for revision, binding in enumerate(({**original, **change}, original), start=2):
        creation_db.upsert_conversation_from_sync(
            conversation_id=first.conversation["id"], title="Synced", sync_client_id="another-device",
            object_revision=revision, object_hash="h", scope_type="workspace", workspace_id="ws", **binding,
        )
        receipt = _receipt(creation_db)
        assert receipt["invalidated_at"] is not None
        if stamp is None:
            stamp = receipt["invalidated_at"]
        assert receipt["invalidated_at"] == stamp
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert caught.value.code == "workspace_chat_startup_changed"


def test_sync_failure_after_invalidation_rolls_back_both(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure after the real receipt write cannot commit the Sync replacement."""
    first = _start(creation_db)
    store = creation_db.workspace_chat_startups
    invalidate = store.invalidate_changed_binding

    def fail_after_write(conversation_id: str, before: Any, after: Any, *, conn: Any) -> None:
        invalidate(conversation_id, before, after, conn=conn)
        receipt = store.get_receipt(
            creation_db.owner_user_id, hashlib.sha256(b"accepted-1").hexdigest(), conn=conn,
        )
        assert receipt["invalidated_at"] is not None
        raise RuntimeError("abort Sync invalidation")

    monkeypatch.setattr(store, "invalidate_changed_binding", fail_after_write)
    with pytest.raises(RuntimeError, match="abort Sync invalidation"):
        creation_db.upsert_conversation_from_sync(
            conversation_id=first.conversation["id"], title="Changed", sync_client_id="another-device",
            object_revision=2, object_hash="h", assistant_kind="persona", assistant_id="persona-a",
            persona_memory_mode="read_write", scope_type="workspace", workspace_id="ws",
        )
    assert _receipt(creation_db)["invalidated_at"] is None
    replay = _start(creation_db)
    assert (replay.replayed, replay.conversation["title"], replay.conversation["persona_memory_mode"]) == (
        True, first.conversation["title"], "read_only",
    )


def test_binding_and_receipt_invalidation_rollback_together(creation_db: CharactersRAGDB) -> None:
    """An enclosing rollback restores the binding and uninvalidated accepted authority."""
    first = _start(creation_db)
    with pytest.raises(RuntimeError, match="abort mutation"):
        with creation_db.transaction() as conn:
            creation_db.update_conversation(
                first.conversation["id"], {"persona_memory_mode": "read_write"}, first.conversation["version"],
            )
            row = creation_db.workspace_chat_startups.get_receipt(
                creation_db.owner_user_id, hashlib.sha256(b"accepted-1").hexdigest(), conn=conn,
            )
            assert row["invalidated_at"] is not None
            raise RuntimeError("abort mutation")
    assert _receipt(creation_db)["invalidated_at"] is None
    replay = _start(creation_db)
    assert (replay.replayed, replay.conversation["id"], replay.conversation["persona_memory_mode"]) == (
        True, first.conversation["id"], "read_only",
    )


def test_character_rebinding_keeps_first_invalidation_timestamp(creation_db: CharactersRAGDB) -> None:
    """Character transitions burn the owner's receipt despite changed device attribution."""
    characters = [creation_db.add_character_card({"name": name}) for name in ("One", "Two")]
    first = _start(creation_db)
    creation_db.client_id = "different-device"
    stamp = None
    for version, character in enumerate(characters, start=first.conversation["version"]):
        creation_db.update_conversation(first.conversation["id"], {
            "assistant_kind": "character", "assistant_id": str(character),
            "character_id": character, "persona_memory_mode": None,
        }, version)
        receipt = _receipt(creation_db)
        assert receipt["owner_user_id"] == "user-1"
        assert receipt["invalidated_at"] is not None
        if stamp is None:
            stamp = receipt["invalidated_at"]
        assert receipt["invalidated_at"] == stamp


@pytest.mark.parametrize("writer", ["local", "sync"])
def test_normalized_noop_and_metadata_keep_receipt_replayable(
    creation_db: CharactersRAGDB, writer: str,
) -> None:
    """Normalized identity, title and device changes are not receipt invalidation."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    if writer == "local":
        creation_db.update_conversation(cid, {
            "assistant_kind": " PERSONA ", "assistant_id": " persona-a ",
            "persona_memory_mode": " READ_ONLY ", "title": "Renamed",
        }, first.conversation["version"])
    else:
        creation_db.upsert_conversation_from_sync(
            conversation_id=cid, title="Renamed", sync_client_id="another-device",
            object_revision=2, object_hash="h", assistant_kind=" PERSONA ",
            assistant_id=" persona-a ", persona_memory_mode=" READ_ONLY ",
            scope_type="workspace", workspace_id="ws",
        )
    assert _receipt(creation_db)["invalidated_at"] is None
    replay = _start(creation_db)
    assert (replay.replayed, replay.conversation["title"]) == (True, "Renamed")


def test_settings_and_message_counters_keep_receipt_replayable(creation_db: CharactersRAGDB) -> None:
    """Settings/history revisions do not change the accepted identity or scope."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.upsert_conversation_settings(cid, {"temperature": 0.5})
    creation_db.add_message({"conversation_id": cid, "sender": "user", "content": "Hello"})
    assert _receipt(creation_db)["invalidated_at"] is None
    replay = _start(creation_db)
    assert replay.replayed and replay.conversation["history_version"] > first.conversation["history_version"]


def test_soft_delete_restore_does_not_clear_or_invent_receipt_authority(creation_db: CharactersRAGDB) -> None:
    """An unchanged binding may replay after an admitted restore, never while deleted."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, first.conversation["version"])
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (410, "workspace_chat_deleted")
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid, include_deleted=True)
        creation_db.restore_conversation(cid, row["version"])
    assert _receipt(creation_db)["invalidated_at"] is None
    assert _start(creation_db).replayed


def test_hard_delete_and_id_reuse_cannot_rebind_tombstoned_receipt(creation_db: CharactersRAGDB) -> None:
    """The FK permanently nulls the reference even if an unrelated chat reuses its id."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.hard_delete_conversation(cid)
    creation_db.add_conversation({"id": cid, "title": "Reused id"})
    receipt = _receipt(creation_db)
    assert receipt["conversation_id"] is None
    with pytest.raises(WorkspaceStartupError) as caught:
        _start(creation_db)
    assert (caught.value.status_code, caught.value.code) == (410, "workspace_chat_deleted")


def test_hard_delete_nulls_receipt_before_removing_chat_and_rolls_back_together(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit hard-delete tombstoning follows chat lock and shares its rollback."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    mark = creation_db.workspace_chat_startups.mark_hard_deleted
    seen = False

    def abort_after_mark(conversation_id: str, *, conn: Any) -> None:
        """Inspect real storage after the tombstone write but before DELETE."""
        nonlocal seen
        mark(conversation_id, conn=conn)
        receipt = creation_db.workspace_chat_startups.get_receipt(
            creation_db.owner_user_id, hashlib.sha256(b"accepted-1").hexdigest(), conn=conn,
        )
        assert receipt["conversation_id"] is None
        assert conn.execute("SELECT id FROM conversations WHERE id = ?", (cid,)).fetchone()
        seen = True
        raise RuntimeError("abort hard deletion")

    with monkeypatch.context() as patch:
        patch.setattr(creation_db.workspace_chat_startups, "mark_hard_deleted", abort_after_mark)
        with pytest.raises(RuntimeError, match="abort hard deletion"):
            creation_db.hard_delete_conversation(cid)
    assert seen
    with creation_db.transaction():
        assert creation_db.get_conversation_by_id(cid) is not None
    assert _receipt(creation_db)["conversation_id"] == cid


@pytest.mark.parametrize("writer", ["restore", "sync"])
@pytest.mark.parametrize("gate", ["closed", "system", "deleted", "owner"])
@pytest.mark.parametrize("invalidated", [False, True])
def test_receipt_resurrection_respects_workspace_gate(
    creation_db: CharactersRAGDB, writer: str, gate: str, invalidated: bool,
) -> None:
    """Receipt association fences resurrection even after its retry authority is burned."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    if invalidated:
        creation_db.update_conversation(cid, {"persona_memory_mode": "read_write"}, 1)
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid)
        creation_db.soft_delete_conversation(cid, row["version"])
    with creation_db.transaction() as conn:
        if gate == "closed":
            conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "ws"))
        elif gate == "system":
            conn.execute("UPDATE workspaces SET system_operation_state = 'staged' WHERE id = ?", ("ws",))
        elif gate == "deleted":
            conn.execute("UPDATE workspaces SET deleted = ? WHERE id = ?", (True, "ws"))
        else:
            conn.execute("UPDATE workspaces SET client_id = ? WHERE id = ?", ("another-owner", "ws"))
        before = creation_db.get_conversation_by_id(cid, include_deleted=True)
    with pytest.raises(ConflictError, match="workspace"):
        if writer == "restore":
            creation_db.restore_conversation(cid, before["version"])
        else:
            creation_db.upsert_conversation_from_sync(
                conversation_id=cid, title="Attempted resurrection", sync_client_id="other-device",
                object_revision=before["version"] + 1, object_hash="h", assistant_kind="persona",
                assistant_id="persona-a", persona_memory_mode=before["persona_memory_mode"],
                scope_type="workspace", workspace_id="ws",
            )
    with creation_db.transaction():
        after = creation_db.get_conversation_by_id(cid, include_deleted=True)
    assert (after["deleted"], after["version"], after["title"]) == (
        before["deleted"], before["version"], before["title"],
    )


@pytest.mark.parametrize("writer", ["restore", "sync"])
def test_legacy_nonreceipt_resurrection_retains_closing_workspace_behavior(
    creation_db: CharactersRAGDB, writer: str,
) -> None:
    """The new receipt gate must not broaden legacy nonreceipt admission policy."""
    cid = creation_db.add_conversation({"title": "Legacy", "scope_type": "workspace", "workspace_id": "ws"})
    creation_db.soft_delete_conversation(cid, 1)
    with creation_db.transaction() as conn:
        conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "ws"))
    if writer == "restore":
        assert creation_db.restore_conversation(cid, 2)
    else:
        assert creation_db.upsert_conversation_from_sync(
            conversation_id=cid, title="Legacy restored", sync_client_id="user-1",
            object_revision=3, object_hash="h", scope_type="workspace", workspace_id="ws",
        )
    with creation_db.transaction():
        assert not creation_db.get_conversation_by_id(cid)["deleted"]


@pytest.mark.parametrize("hard", [False, True])
def test_workspace_residual_guard_includes_receipt_chat_with_changed_writer(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, hard: bool,
) -> None:
    """A staged cascade cannot finalize around an uncascaded owner's receipt chat."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.upsert_conversation_from_sync(
        conversation_id=cid, title="Different device", sync_client_id="other-device",
        object_revision=2, object_hash="h", assistant_kind="persona", assistant_id="persona-a",
        persona_memory_mode="read_only", scope_type="workspace", workspace_id="ws",
    )
    if hard:
        monkeypatch.setattr(creation_db, "hard_delete_conversation", lambda *args: True)
    else:
        monkeypatch.setattr(creation_db, "soft_delete_conversation", lambda *args: True)
    with pytest.raises(ConflictError, match="workspace_delete_incomplete"):
        if hard:
            creation_db.hard_delete_workspace("ws")
        else:
            creation_db.delete_workspace("ws", expected_version=2)
    workspace = creation_db.get_workspace("ws", include_deleted=True)
    assert not workspace["deleted"] and workspace["native_chat_admission_closed"]
    with creation_db.transaction():
        assert not creation_db.get_conversation_by_id(cid)["deleted"]


def _resurrect(db: CharactersRAGDB, cid: str, writer: str, version: int, *, workspace: str = "ws") -> None:
    """Invoke the public DB writers with an unchanged accepted assistant binding."""
    if writer == "restore":
        assert db.restore_conversation(cid, version)
    else:
        assert db.upsert_conversation_from_sync(
            conversation_id=cid, title="Resurrected", sync_client_id="other-device",
            object_revision=version + 1, object_hash="h", assistant_kind="persona",
            assistant_id="persona-a", persona_memory_mode="read_only",
            scope_type="workspace", workspace_id=workspace,
        )


@pytest.mark.parametrize("writer", ["restore", "sync"])
@pytest.mark.parametrize("commit", [False, True])
def test_receipt_writer_preserves_driver_open_caller_work(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], writer: str, commit: bool,
) -> None:
    """Successful lifecycle writes join, rather than settle, a raw caller transaction."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    observer = db_factory()
    conn = creation_db.get_connection()
    if creation_db.backend_type.value == "sqlite":
        conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE workspaces SET name = ? WHERE id = ?", ("Caller sentinel", "ws"))
    try:
        _resurrect(creation_db, cid, writer, 2)
        if creation_db.backend_type.value == "postgresql":
            assert conn._connection.info.transaction_status.name == "INTRANS"
        else:
            assert conn.in_transaction
        assert conn.execute("SELECT name FROM workspaces WHERE id = ?", ("ws",)).fetchone()["name"] == "Caller sentinel"
        row = observer.execute_query("SELECT deleted FROM conversations WHERE id = ?", (cid,), read_only=True).fetchone()
        assert row["deleted"]
        if commit:
            conn.commit()
        else:
            conn.rollback()
    finally:
        conn.rollback()
    with observer.transaction() as other:
        row = other.execute("SELECT deleted FROM conversations WHERE id = ?", (cid,)).fetchone()
        assert bool(row["deleted"]) is not commit
        name = other.execute("SELECT name FROM workspaces WHERE id = ?", ("ws",)).fetchone()["name"]
        assert (name == "Caller sentinel") is commit


@pytest.mark.parametrize("writer", ["restore", "sync"])
@pytest.mark.parametrize("boundary", ["managed", "driver"])
@pytest.mark.parametrize("commit", [False, True])
def test_rejected_receipt_writer_preserves_caller_transaction(
    creation_db: CharactersRAGDB, writer: str, boundary: str, commit: bool,
) -> None:
    """A bounded admission error neither mutates the chat nor settles caller work."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    with creation_db.transaction() as conn:
        conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "ws"))
    driver = creation_db.get_connection()
    context = creation_db.transaction() if boundary == "managed" else nullcontext(driver)
    rollback = pytest.raises(RuntimeError, match="caller rollback") if not commit else nullcontext()
    try:
        with rollback, context as conn:
            if boundary == "driver" and creation_db.backend_type.value == "sqlite":
                conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE workspaces SET name = ? WHERE id = ?", ("Caller sentinel", "ws"))
            with pytest.raises(ConflictError, match="workspace"):
                _resurrect(creation_db, cid, writer, 2)
            if creation_db.backend_type.value == "postgresql":
                assert conn._connection.info.transaction_status.name == "INTRANS"
            else:
                assert conn.in_transaction
            assert conn.execute("SELECT name FROM workspaces WHERE id = ?", ("ws",)).fetchone()["name"] == "Caller sentinel"
            if not commit:
                raise RuntimeError("caller rollback")
        if boundary == "driver" and commit:
            driver.commit()
    finally:
        if boundary == "driver":
            driver.rollback()
    with creation_db.transaction() as conn:
        name = conn.execute("SELECT name FROM workspaces WHERE id = ?", ("ws",)).fetchone()["name"]
        assert (name == "Caller sentinel") is commit
        row = creation_db.get_conversation_by_id(cid, include_deleted=True)
        assert (bool(row["deleted"]), row["version"]) == (True, 2)


def test_receipt_sync_reentry_gates_destination_not_origin(creation_db: CharactersRAGDB) -> None:
    """An invalidated global chat cannot re-enter a different closing Workspace."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.upsert_conversation_from_sync(
        conversation_id=cid, title="Global", sync_client_id="other-device",
        object_revision=2, object_hash="h", assistant_kind="persona",
        assistant_id="persona-a", persona_memory_mode="read_only", scope_type="global",
    )
    creation_db.upsert_workspace("destination", "Destination")
    with creation_db.transaction() as conn:
        conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "destination"))
    with pytest.raises(ConflictError, match="workspace"):
        _resurrect(creation_db, cid, "sync", 2, workspace="destination")
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid)
        assert (row["scope_type"], row["workspace_id"], row["version"], row["title"]) == ("global", None, 2, "Global")


@pytest.mark.parametrize("writer", ["restore", "sync"])
def test_receipt_resurrection_allows_archived_workspace_and_changed_device(
    creation_db: CharactersRAGDB, writer: str,
) -> None:
    """Archival blocks new startup, not admitted restoration under immutable ownership."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    with creation_db.transaction() as conn:
        conn.execute("UPDATE workspaces SET archived = ? WHERE id = ?", (True, "ws"))
    creation_db.client_id = "other-device"
    _resurrect(creation_db, cid, writer, 2)
    replay = _start(creation_db)
    assert replay.replayed and replay.conversation["client_id"] == "other-device"
    assert _receipt(creation_db)["invalidated_at"] is None


@pytest.mark.parametrize("writer", ["restore", "sync"])
def test_postgres_receipt_admission_retries_changed_preflight_in_fresh_transaction(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, writer: str,
) -> None:
    """A real competing scope move forces rollback before a fresh admission attempt."""
    if creation_db.backend_type.value != "postgresql":
        pytest.skip("PostgreSQL permits competing writes between unlocked preflight and Workspace lock")
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    contender = db_factory()
    ready, start, preflight, release = Event(), Event(), Event(), Event()
    admit = contender.conversation_store._lock_receipt_admission
    attempts = 0

    def pause_first(*args: Any, **kwargs: Any) -> bool:
        """Pause before real Workspace admission; hold no conversation or receipt lock."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            preflight.set()
            assert release.wait(20), "scope-changing writer was not released"
        return admit(*args, **kwargs)

    def restore_or_sync() -> None:
        """Warm the worker connection outside the measured contention interval."""
        with contender.transaction() as conn:
            conn.execute("SET statement_timeout = '15s'")
        ready.set()
        assert start.wait(20)
        _resurrect(contender, cid, writer, 2)

    monkeypatch.setattr(contender.conversation_store, "_lock_receipt_admission", pause_first)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(restore_or_sync)
        try:
            assert ready.wait(20)
            start.set()
            assert preflight.wait(20)
            creation_db.upsert_conversation_from_sync(
                conversation_id=cid, title="Moved globally", sync_client_id="other-device",
                object_revision=3, object_hash="h", assistant_kind="persona",
                assistant_id="persona-a", persona_memory_mode="read_only", scope_type="global",
            )
        finally:
            release.set()
            start.set()
        future.result(timeout=20)
    assert attempts == 2
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid)
        assert not row["deleted"]
        assert row["scope_type"] == ("global" if writer == "restore" else "workspace")
    assert _receipt(creation_db)["invalidated_at"] is not None


@pytest.mark.parametrize("writer", ["restore", "sync"])
@pytest.mark.parametrize("boundary", ["owned-once", "owned-always", "managed", "driver"])
def test_changed_admission_retry_is_bounded_by_transaction_ownership(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, writer: str, boundary: str,
) -> None:
    """Fault-injected stale preflight retries only owned units; caller units stay intact."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    admit = creation_db.conversation_store._lock_receipt_admission
    attempts = 0

    def change_after_admission(*args: Any, **kwargs: Any) -> bool:
        """Mutate real storage after Workspace admission to exercise rollback, not live races."""
        nonlocal attempts
        result = admit(*args, **kwargs)
        attempts += 1
        if attempts == 1 or boundary == "owned-always":
            kwargs["conn"].execute(
                "UPDATE conversations SET scope_type = 'global', workspace_id = NULL WHERE id = ?", (cid,),
            )
        return result

    monkeypatch.setattr(creation_db.conversation_store, "_lock_receipt_admission", change_after_admission)
    caller_owned = boundary in ("managed", "driver")
    driver = creation_db.get_connection() if boundary == "driver" else None
    context = creation_db.transaction() if boundary == "managed" else nullcontext(driver)
    rollback = pytest.raises(RuntimeError, match="caller rollback") if caller_owned else nullcontext()
    try:
        with rollback, context as conn:
            if caller_owned:
                if boundary == "driver" and creation_db.backend_type.value == "sqlite":
                    conn.execute("BEGIN IMMEDIATE")
                conn.execute("UPDATE workspaces SET name = ? WHERE id = ?", ("Caller sentinel", "ws"))
            if boundary == "owned-once":
                _resurrect(creation_db, cid, writer, 2)
            else:
                with pytest.raises(ConflictError, match="workspace_chat_admission_changed"):
                    _resurrect(creation_db, cid, writer, 2)
            assert attempts == (1 if caller_owned else 2)
            if caller_owned:
                if creation_db.backend_type.value == "postgresql":
                    assert conn._connection.info.transaction_status.name == "INTRANS"
                else:
                    assert conn.in_transaction
                assert conn.execute("SELECT name FROM workspaces WHERE id = ?", ("ws",)).fetchone()["name"] == "Caller sentinel"
                assert conn.execute("SELECT deleted FROM conversations WHERE id = ?", (cid,)).fetchone()["deleted"]
                raise RuntimeError("caller rollback")
    finally:
        if driver is not None:
            driver.rollback()
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid, include_deleted=True)
        assert (row["scope_type"], row["workspace_id"]) == ("workspace", "ws")
        assert bool(row["deleted"]) is (boundary != "owned-once")


@pytest.mark.parametrize("writer", ["restore", "sync"])
def test_failed_rollback_cannot_retry_receipt_admission_in_same_transaction(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, writer: str,
) -> None:
    """A swallowed driver rollback failure must not produce uncommitted success."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    admit = creation_db.conversation_store._lock_receipt_admission
    transaction = creation_db.transaction
    attempts = 0

    class FailedRollback:
        """Fail settlement only; all SQL and driver state still use the real connection."""

        def __init__(self, raw: Any) -> None:
            self.raw = raw

        def __getattr__(self, name: str) -> Any:
            return getattr(self.raw, name)

        def rollback(self) -> None:
            """Leave the real owned unit open as a failed driver rollback would."""
            raise sqlite3.OperationalError("controlled rollback failure")

    @contextmanager
    def fail_rollback(*, preserve_existing: bool = False) -> Iterator[Any]:
        """Replace only the context's settlement reference after its actual entry."""
        context = transaction(preserve_existing=preserve_existing)
        with context as conn:
            attribute = "_raw_conn" if creation_db.backend_type.value == "postgresql" else "conn"
            setattr(context, attribute, FailedRollback(getattr(context, attribute)))
            yield conn

    def change_once(*args: Any, **kwargs: Any) -> bool:
        """Make the first preflight stale using actual storage under its owned locks."""
        nonlocal attempts
        result = admit(*args, **kwargs)
        attempts += 1
        if attempts == 1:
            kwargs["conn"].execute(
                "UPDATE conversations SET scope_type = 'global', workspace_id = NULL WHERE id = ?", (cid,),
            )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(creation_db, "transaction", fail_rollback)
        patch.setattr(creation_db.conversation_store, "_lock_receipt_admission", change_once)
        try:
            with pytest.raises(ConflictError, match="workspace_chat_admission_changed"):
                _resurrect(creation_db, cid, writer, 2)
            assert attempts == 1
            conn = creation_db.get_connection()
            if creation_db.backend_type.value == "postgresql":
                assert conn._connection.info.transaction_status.name == "INTRANS"
            else:
                assert conn.in_transaction
            assert conn.execute("SELECT deleted FROM conversations WHERE id = ?", (cid,)).fetchone()["deleted"]
        finally:
            creation_db.get_connection().rollback()
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid, include_deleted=True)
        assert (row["scope_type"], row["workspace_id"], bool(row["deleted"])) == ("workspace", "ws", True)
