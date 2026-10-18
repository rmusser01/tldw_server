"""D7 regeneration reuses a protected input without changing original graph authority."""

import json

import pytest

from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
from tldw_Server_API.tests.DB_Management.test_history_branch_admission import (
    OWNER,
    selected,
    settle,
)
from tldw_Server_API.tests.DB_Management.test_history_branch_admission import (
    history_db as history_db,
)


def answered(db, cid, source="client"):
    selection = selected(db, cid)
    if source == "server":
        admission = db.append_selected_history_inputs(
            cid, selection, [{"sender": "user", "content": "Original input"}], **OWNER
        )
    else:
        admission = db.append_selected_history_input(
            cid, selection, {"id": "u", "sender": "user", "content": "Original input"}, **OWNER
        )
    settle(db, cid, admission, "a")
    return admission


def regenerate(db, cid, selection, target="a", **extra):
    return db.reuse_selected_history_admission(cid, selection, target, **(OWNER | extra))


@pytest.mark.parametrize("source", ["client", "server"])
def test_reuses_input_authority_without_any_original_write(history_db, source):
    db, cid = history_db
    admission = answered(db, cid, source)
    selection = selected(db, cid, after="a")
    before = db.get_messages_for_conversation(cid)
    fence = db.get_conversation_by_id(cid)["history_version"]
    reused = regenerate(db, cid, selection)
    assert reused == admission
    assert db.get_messages_for_conversation(cid) == before
    assert db.get_conversation_by_id(cid)["history_version"] == fence
    settle(db, cid, reused, "b")
    rows = db.get_messages_for_conversation(cid)
    assert [(row["id"], row["parent_message_id"]) for row in rows if row["sender"] == "assistant"] == [
        ("a", admission["input_message_id"]),
        ("b", admission["input_message_id"]),
    ]
    assert [row["id"] for row in rows if row["sender"] == "user"] == [admission["input_message_id"]]
    assert rows[:2] == before


def test_selected_older_branch_and_freshly_edited_assistant_keep_the_same_user(history_db):
    db, cid = history_db
    admission = answered(db, cid)
    later = db.append_selected_history_input(
        cid, selected(db, cid, after="a"), {"id": "later-user", "sender": "user", "content": "Later input"}, **OWNER
    )
    settle(db, cid, later, "later-answer")
    db.update_message("a", {"content": "Edited original answer"}, 1)
    before = db.get_messages_for_conversation(cid)
    assert regenerate(db, cid, selected(db, cid, after="a")) == admission
    assert db.get_messages_for_conversation(cid) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "stale-user",
        "edited-user",
        "stale-target",
        "deleted-target",
        "wrong-target",
        "wrong-owner",
        "missing-authority",
        "forged-authority",
        "missing-digest",
        "missing-scope",
        "stale-context",
    ],
)
def test_stale_or_unqualified_regeneration_writes_nothing(history_db, boundary):
    db, cid = history_db
    admission = answered(db, cid)
    selection = selected(db, cid, after="a")
    extra = {}
    target = "a"
    if boundary in {"stale-user", "edited-user"}:
        db.update_message(admission["input_message_id"], {"content": "Changed input"}, 1)
        if boundary == "edited-user":
            selection = selected(db, cid, after="a")
    elif boundary == "stale-target":
        db.update_message("a", {"content": "Changed answer"}, 1)
    elif boundary == "deleted-target":
        db.soft_delete_message("a", 1)
    elif boundary == "wrong-target":
        target = "foreign"
    elif boundary == "wrong-owner":
        extra = {"owner_client_id": "bob", "owner_key": "server:test/account:bob"}
    elif boundary in {"missing-authority", "forged-authority", "missing-digest", "missing-scope"}:
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT history_admission_json FROM messages WHERE id = ?", (admission["input_message_id"],)
            ).fetchone()
            authority = json.loads(row["history_admission_json"])
            if boundary == "missing-authority":
                authority.pop("admission")
            elif boundary == "missing-digest":
                authority.pop("input_state_digest")
            elif boundary == "missing-scope":
                authority.pop("scope")
            else:
                authority["admission"]["input_message_id"] = "foreign"
            conn.execute(
                "UPDATE messages SET history_admission_json = ? WHERE id = ?",
                (json.dumps(authority), admission["input_message_id"]),
            )
        selection = selected(db, cid, after="a")
    elif boundary == "stale-context":
        db.upsert_conversation_settings(cid, {"system_prompt": "Changed setup"}, expected_settings_version=0)
    before = db.get_messages_for_conversation(cid)
    fence = db.get_conversation_by_id(cid)["history_version"]
    with pytest.raises(HistorySelectionError) as rejected:
        regenerate(db, cid, selection, target, **extra)
    if boundary == "missing-authority":
        assert rejected.value.code == "unsupported_history_regeneration"
    assert db.get_messages_for_conversation(cid) == before
    assert db.get_conversation_by_id(cid)["history_version"] == fence


def test_selection_must_end_at_the_chosen_assistant(history_db):
    db, cid = history_db
    admission = answered(db, cid)
    with pytest.raises(HistorySelectionError):
        regenerate(db, cid, selected(db, cid, after=admission["input_message_id"]))


def test_input_deletion_after_reuse_still_refuses_late_settlement(history_db):
    db, cid = history_db
    admission = answered(db, cid)
    reused = regenerate(db, cid, selected(db, cid, after="a"))
    db.soft_delete_message(admission["input_message_id"], 1)
    with pytest.raises(HistorySelectionError):
        settle(db, cid, reused, "b")
    assert db.get_message_by_id("b") is None


def test_tool_parent_is_refused_without_guessing_the_original_user(history_db):
    db, cid = history_db
    original = answered(db, cid)
    tool = db.append_selected_history_input(
        cid,
        selected(db, cid, after=original["input_message_id"]),
        {"id": "tool", "sender": "tool", "content": "Tool result"},
        **OWNER,
    )
    settle(db, cid, tool, "tool-answer")
    before = db.get_messages_for_conversation(cid)
    with pytest.raises(HistorySelectionError, match="unsupported_history_regeneration"):
        regenerate(db, cid, selected(db, cid, after="tool-answer"), "tool-answer")
    assert db.get_messages_for_conversation(cid) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_state_digest", None),
        ("scope", None),
        ("interpretation", None),
        ("interpretation", {"kind": []}),
        ("input_chain", None),
        ("input_chain", [{}]),
    ],
)
def test_malformed_protected_authority_refuses_all_late_settlement(history_db, field, value):
    db, cid = history_db
    admission = answered(db, cid)
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT history_admission_json FROM messages WHERE id = ?", (admission["input_message_id"],)
        ).fetchone()
        authority = json.loads(row["history_admission_json"])
        authority[field] = value
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?",
            (json.dumps(authority), admission["input_message_id"]),
        )
    before = db.get_messages_for_conversation(cid)
    with pytest.raises(HistorySelectionError):
        settle(db, cid, admission, "late-answer")
    assert db.get_messages_for_conversation(cid) == before


def test_malformed_authority_json_has_a_declared_settlement_refusal(history_db):
    db, cid = history_db
    admission = answered(db, cid)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?",
            ("{", admission["input_message_id"]),
        )
    before = db.get_messages_for_conversation(cid)
    with pytest.raises(HistorySelectionError, match="invalid_admission"):
        settle(db, cid, admission, "late-answer")
    assert db.get_messages_for_conversation(cid) == before
