"""D7 P1: the owner refuses a "must extend latest" admission whose selected tip already moved on.

Both admission paths are covered on every supported backend: client admission
(``append_selected_history_input``, behind ``POST /chats/{id}/messages``) and the
server-completion input chain (``append_selected_history_inputs``).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Chat.history_selection import (
    HistoryBranchChangedError,
    HistorySelectionError,
    resolve_history_selection,
    snapshot_to_wire,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

OWNER_KEY = "server:test/account:alice"
OWNER = {"owner_client_id": "alice", "owner_key": OWNER_KEY}


@pytest.fixture(params=["sqlite", "postgres"])
def history_db(request, tmp_path):
    """Use real independent test storage on each supported backend."""
    kwargs = {"db_path": str(tmp_path / "branch.sqlite"), "client_id": "alice"}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(**kwargs)
    conversation_id = db.add_conversation({"character_id": 1, "title": "Branch check"})
    yield db, conversation_id
    db.close_connection()
    if request.param == "postgres":
        db.backend.get_pool().close_all()


def selected(db, cid, after=None, context="client-provenance"):
    snap = snapshot_to_wire(db.get_conversation_history_snapshot(cid, **OWNER))
    cursor = {"kind": "after_message", "message_id": after} if after else {"kind": "empty"}
    result = resolve_history_selection(snap, {
        "owner_key": OWNER_KEY, "conversation_id": cid, "interpretation": {"kind": "parent_graph_v1"},
        "cursor": cursor, "selection_revision": 1,
    }, "send", context)
    assert result["status"] == "ready", result
    return result["selection"]


def admit(db, cid, selection, mid, **kwargs):
    return db.append_selected_history_input(
        cid, selection, {"id": mid, "sender": "user", "content": f"input {mid}"}, **OWNER, **kwargs,
    )


def settle(db, cid, admission, mid):
    reference = {key: admission[key] for key in (
        "version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest",
    )}
    return db.settle_history_admission(
        cid, reference, {"id": mid, "sender": "assistant", "content": f"reply {mid}"}, **OWNER,
    )


def server_turn(db, cid, selection, **kwargs):
    return db.append_selected_history_inputs(
        cid, selection, [{"sender": "user", "content": "server input"}], **OWNER, **kwargs,
    )


def answered_turn(db, cid):
    """u1 -> a1, then capture a selection whose tip is a1."""
    settle(db, cid, admit(db, cid, selected(db, cid), "u1"), "a1")
    return selected(db, cid, after="a1")


def stale_by_other_tab(db, cid):
    """Another tab extends a1 with u2 -> a2; the returned selection still ends at a1."""
    tip = answered_turn(db, cid)
    settle(db, cid, admit(db, cid, tip, "u2", history_branch=False), "a2")
    return tip


def test_stale_tip_refuses_with_current_leaf_and_writes_nothing(history_db):
    db, cid = history_db
    stale = stale_by_other_tab(db, cid)
    before = db.count_messages_for_conversation(cid)
    version = db.get_conversation_by_id(cid)["history_version"]
    with pytest.raises(HistoryBranchChangedError) as caught:
        admit(db, cid, stale, "u3", history_branch=False)
    assert caught.value.code == "history_branch_changed"
    assert caught.value.details == {
        "conversation_id": cid, "parent_message_id": "a1", "leaf_ids": ["a2"], "history_version": version,
    }
    assert db.count_messages_for_conversation(cid) == before
    assert db.get_message_by_id("u3") is None
    assert db.get_conversation_by_id(cid)["history_version"] == version


def test_current_tip_is_admitted_when_extension_is_required(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    admission = admit(db, cid, tip, "u2", history_branch=False)
    assert admission["input_message_id"] == "u2"
    assert db.get_message_by_id("u2")["parent_message_id"] == "a1"


def test_explicit_branch_is_admitted_beside_the_newer_turn(history_db):
    db, cid = history_db
    stale = stale_by_other_tab(db, cid)
    admit(db, cid, stale, "u3", history_branch=True)
    assert db.get_message_by_id("u3")["parent_message_id"] == "a1"
    assert db.get_message_by_id("u2")["parent_message_id"] == "a1"


def test_absent_field_keeps_todays_unchecked_admission(history_db):
    db, cid = history_db
    stale = stale_by_other_tab(db, cid)
    admit(db, cid, stale, "u3")
    assert db.get_message_by_id("u3")["parent_message_id"] == "a1"


def test_idempotent_replay_of_admitted_message_is_unaffected(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    first = admit(db, cid, tip, "u2", history_branch=False)
    settle(db, cid, first, "a2")
    # u2 is now a live child of the tip; replaying u2 itself must still succeed.
    assert admit(db, cid, tip, "u2", history_branch=False) == first
    assert admit(db, cid, tip, "u2") == first
    assert db.count_messages_for_conversation(cid) == 4


def test_deleted_child_does_not_count(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    admit(db, cid, tip, "u2")
    db.soft_delete_message("u2", expected_version=1)
    admit(db, cid, tip, "u3", history_branch=False)
    assert db.get_message_by_id("u3")["parent_message_id"] == "a1"


def test_regenerated_variants_are_siblings_and_never_block_the_chosen_variant(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid), "u1", history_branch=False)
    settle(db, cid, accepted, "a1")
    # Regenerate settles another reply against the same admitted input; no admission runs.
    settle(db, cid, accepted, "a1-variant")
    second = admit(db, cid, selected(db, cid, after="a1"), "u2", history_branch=False)
    # The variant has no child of its own, so continuing from it is still an extension.
    admit(db, cid, selected(db, cid, after="a1-variant"), "u2-variant", history_branch=False)
    # Regenerating under an input admitted by the leaf check stays allowed.
    settle(db, cid, second, "a2")
    settle(db, cid, second, "a2-variant")
    assert db.get_message_by_id("a2-variant")["parent_message_id"] == "u2"
    assert db.get_message_by_id("u2-variant")["parent_message_id"] == "a1-variant"


def test_unanswered_user_tip_with_settled_reply_is_stale(history_db):
    db, cid = history_db
    accepted = admit(db, cid, selected(db, cid), "u1")
    stale = selected(db, cid, after="u1")
    settle(db, cid, accepted, "a1")
    with pytest.raises(HistoryBranchChangedError) as caught:
        admit(db, cid, stale, "u2", history_branch=False)
    assert caught.value.details["leaf_ids"] == ["a1"]


def test_empty_selection_on_started_chat_is_stale(history_db):
    db, cid = history_db
    empty = selected(db, cid)
    settle(db, cid, admit(db, cid, empty, "u1"), "a1")
    with pytest.raises(HistoryBranchChangedError) as caught:
        admit(db, cid, empty, "other-root", history_branch=False)
    assert caught.value.details["parent_message_id"] is None
    assert caught.value.details["leaf_ids"] == ["a1"]
    admit(db, cid, empty, "other-root", history_branch=True)
    assert db.get_message_by_id("other-root")["parent_message_id"] is None


def test_invalid_selection_reports_its_own_code_before_the_leaf_check(history_db):
    db, cid = history_db
    stale = stale_by_other_tab(db, cid)
    db.update_message("a1", {"content": "edited"}, expected_version=1)
    with pytest.raises(HistorySelectionError, match="stale_selection"):
        admit(db, cid, stale, "u3", history_branch=False)


def test_concurrent_extensions_of_one_tip_admit_exactly_one(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    barrier = threading.Barrier(2)

    def run(mid):
        barrier.wait(timeout=10)
        try:
            return admit(db, cid, tip, mid, history_branch=False)
        except HistorySelectionError as exc:
            return exc.code
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["tab-a", "tab-b"]))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert "history_branch_changed" in results
    assert db.count_messages_for_conversation(cid) == 3


def legacy_view(db, cid, after):
    """Reviewed legacy rows have no parent links; the projection order is their ancestry."""
    snap = snapshot_to_wire(db.get_conversation_history_snapshot(cid, projection_id="reviewed", **OWNER))
    return resolve_history_selection(snap, {
        "owner_key": OWNER_KEY, "conversation_id": cid,
        "interpretation": {"kind": "legacy_linear_v1", "projection_id": "reviewed"},
        "cursor": {"kind": "after_message", "message_id": after}, "selection_revision": 1,
    }, "send", "legacy")["selection"]


def test_legacy_projection_successor_counts_as_live_child(history_db):
    db, cid = history_db
    for mid in ("legacy-a", "legacy-b"):
        db.add_message({"id": mid, "conversation_id": cid, "sender": "user", "content": mid, "parent_message_id": None})
    source = db.get_conversation_history_snapshot(cid, **OWNER)
    wire = snapshot_to_wire(source)
    db.confirm_legacy_history_projection({
        "version": 1, "projection_id": "reviewed", "owner_key": OWNER_KEY, "conversation_id": cid,
        "source_digest": source.source_digest, "fences": wire["fences"],
        "source_members": [{"id": row["id"], "revision": row["revision"]} for row in wire["nodes"]],
        "ordered_path_ids": ["legacy-a", "legacy-b"], "cursor": {"kind": "after_message", "message_id": "legacy-b"},
        "selection_revision": 1,
    }, **OWNER)
    with pytest.raises(HistoryBranchChangedError) as caught:
        admit(db, cid, legacy_view(db, cid, "legacy-a"), "mid-path", history_branch=False)
    assert caught.value.details["leaf_ids"] == ["legacy-b"]
    admit(db, cid, legacy_view(db, cid, "legacy-b"), "latest", history_branch=False)
    assert db.get_message_by_id("latest")["parent_message_id"] == "legacy-b"


def test_server_completion_chain_refuses_stale_tip_and_writes_nothing(history_db):
    db, cid = history_db
    stale = stale_by_other_tab(db, cid)
    before = db.count_messages_for_conversation(cid)
    with pytest.raises(HistoryBranchChangedError) as caught:
        server_turn(db, cid, stale, history_branch=False)
    assert caught.value.details["leaf_ids"] == ["a2"]
    assert caught.value.details["parent_message_id"] == "a1"
    assert db.count_messages_for_conversation(cid) == before


def test_server_completion_chain_current_explicit_and_absent(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    current = server_turn(db, cid, tip, history_branch=False)
    assert db.get_message_by_id(current["input_message_id"])["parent_message_id"] == "a1"
    # Distinct request contexts: a server completion consumes its exact selection once.
    branched = server_turn(db, cid, selected(db, cid, after="a1", context="branch"), history_branch=True)
    assert db.get_message_by_id(branched["input_message_id"])["parent_message_id"] == "a1"
    legacy = server_turn(db, cid, selected(db, cid, after="a1", context="legacy"))
    assert db.get_message_by_id(legacy["input_message_id"])["parent_message_id"] == "a1"


def test_server_completion_consumed_selection_keeps_its_code(history_db):
    db, cid = history_db
    tip = answered_turn(db, cid)
    server_turn(db, cid, tip, history_branch=False)
    with pytest.raises(HistorySelectionError, match="selection_already_consumed"):
        server_turn(db, cid, tip, history_branch=False)
