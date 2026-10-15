"""Native leaf deletion composes the existing live parent-graph transaction."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Character_Chat.modules.character_chat import remove_message_from_conversation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import (
    add,
    snapshot,
)
from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import (
    history_db as history_db,
)


def test_native_leaf_delete_keeps_exact_ancestor_and_other_branch(history_db):
    db, cid = history_db
    parent = add(db, cid, "  {{char}} original  ", images=[{"data": b"original", "mime": "image/png"}])
    leaf = add(db, cid, "Delete only this answer", sender="assistant", parent_message_id=parent)
    sibling = add(db, cid, "Other answer", sender="assistant", parent_message_id=parent)
    before = db.get_messages_for_conversation(cid)
    epoch = snapshot(db, cid).fences.history
    assert db.soft_delete_message(leaf, 1, require_leaf=True) is True
    assert db.get_messages_for_conversation(cid) == [row for row in before if row["id"] != leaf]
    assert snapshot(db, cid).interpretation_status["kind"] == "parent_graph_v1"
    assert snapshot(db, cid).fences.history != epoch
    assert db.get_message_by_id(sibling)["parent_message_id"] == parent


def test_native_delete_refuses_hidden_descendants_without_a_write(history_db):
    db, cid = history_db
    parent = add(db, cid, "Original")
    target = add(db, cid, "Visible answer", sender="assistant", parent_message_id=parent)
    add(db, cid, "Hidden later branch", parent_message_id=target)
    before = db.get_messages_for_conversation(cid)
    epoch = snapshot(db, cid).fences.history
    with pytest.raises(ConflictError, match="message_has_descendants"):
        db.soft_delete_message(target, 1, require_leaf=True)
    assert db.get_messages_for_conversation(cid) == before
    assert snapshot(db, cid).fences.history == epoch


def test_deleted_descendant_does_not_block_a_native_leaf_delete(history_db):
    db, cid = history_db
    parent = add(db, cid)
    child = add(db, cid, parent_message_id=parent)
    db.soft_delete_message(child, 1)
    assert db.soft_delete_message(parent, 1, require_leaf=True) is True
    assert db.count_messages_for_conversation(cid) == 0


def test_real_character_wrapper_preserves_target_cas_conflict(history_db):
    db, cid = history_db
    leaf = add(db, cid)
    db.update_message(leaf, {"content": "Newer saved text"}, 1)
    with pytest.raises(ConflictError):
        remove_message_from_conversation(db, leaf, 1)
    assert db.get_message_by_id(leaf)["content"] == "Newer saved text"


@pytest.mark.parametrize("boundary", ["deleted", "other-conversation"])
def test_plain_append_validates_its_live_same_conversation_parent(history_db, boundary):
    db, cid = history_db
    parent = add(db, cid)
    target_cid = cid
    if boundary == "deleted":
        db.soft_delete_message(parent, 1)
    else:
        target_cid = db.add_conversation({"character_id": 1, "title": "Other conversation"})
    with pytest.raises(InputError, match="parent_message_unavailable"):
        add(db, target_cid, "Cannot attach", parent_message_id=parent)
    assert db.count_messages_for_conversation(target_cid) == 0


def test_plain_reparent_cannot_attach_to_a_tombstoned_parent(history_db):
    db, cid = history_db
    parent = add(db, cid)
    leaf = add(db, cid, "Unrelated current row")
    db.soft_delete_message(parent, 1)
    before = db.get_message_by_id(leaf)
    with pytest.raises(InputError, match="parent_message_unavailable"):
        db.update_message(leaf, {"parent_message_id": parent}, 1)
    assert db.get_message_by_id(leaf) == before


def test_legacy_delete_keeps_its_existing_non_leaf_semantics(history_db):
    db, cid = history_db
    parent = add(db, cid)
    child = add(db, cid, parent_message_id=parent)
    assert db.soft_delete_message(parent, 1) is True
    assert db.get_message_by_id(child)["parent_message_id"] == parent


@pytest.mark.parametrize("mutation", ["metadata", "image"])
def test_native_delete_rejects_changed_opaque_state_at_the_same_numeric_version(history_db, mutation):
    db, cid = history_db
    target = add(db, cid)
    revision = snapshot(db, cid).nodes[0]["revision"]
    if mutation == "metadata":
        db.set_message_metadata_extra(target, {"pinned": True})
    else:
        db.append_message_image(target, b"new saved image", "image/png")
    assert db.get_message_by_id(target)["version"] == 1
    with pytest.raises(ConflictError, match="message_changed"):
        db.soft_delete_message(target, 1, require_leaf=True, expected_revision=revision)
    assert db.get_message_by_id(target) is not None


@pytest.mark.parametrize("parent_writer", ["plain", "sync", "reparent"])
@pytest.mark.parametrize("first", ["delete", "child"])
def test_parent_writer_and_native_delete_serialize_without_an_orphan(history_db, monkeypatch, parent_writer, first):
    db, cid = history_db
    parent = add(db, cid)
    existing = add(db, cid, "Unrelated root") if parent_writer == "reparent" else None
    writing, release, second_started = threading.Event(), threading.Event(), threading.Event()
    advance = db.message_store._advance_history_version

    def pause_first(conn, conversation_id):
        if not writing.is_set():
            writing.set()
            assert release.wait(10)
        advance(conn, conversation_id)

    monkeypatch.setattr(db.message_store, "_advance_history_version", pause_first)

    def delete():
        return db.soft_delete_message(parent, 1, require_leaf=True)

    def attach():
        if parent_writer == "reparent":
            return db.update_message(existing, {"parent_message_id": parent}, 1)
        if parent_writer == "sync":
            return db.append_message_from_sync(
                stable_message_id="child",
                conversation_id=cid,
                sender="user",
                content="Child",
                timestamp=None,
                sync_client_id="device",
                object_revision=1,
                payload_hash="child-hash",
                parent_message_id=parent,
            )
        return add(db, cid, "Child", parent_message_id=parent)

    def run(operation, *, second=False):
        if second:
            second_started.set()
        try:
            return operation()
        finally:
            db.close_connection()

    leading, following = (delete, attach) if first == "delete" else (attach, delete)
    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(run, leading)
        assert writing.wait(10)
        later = pool.submit(run, following, second=True)
        try:
            assert second_started.wait(10)
            assert not later.done()
        finally:
            release.set()
        assert winner.result(timeout=15)
        with pytest.raises(
            InputError if first == "delete" else ConflictError,
            match="parent_message_unavailable" if first == "delete" else "message_has_descendants",
        ):
            later.result(timeout=15)
    rows = db.get_messages_for_conversation(cid)
    live = {row["id"] for row in rows}
    assert all(row["parent_message_id"] is None or row["parent_message_id"] in live for row in rows)
