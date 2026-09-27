"""Receipt lifecycle follows real conversation mutations, never reconstructed authority."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import WorkspaceStartupError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
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
