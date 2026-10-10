"""Strict Resume requests and atomic caller-ID admission on real owner storage."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError


@pytest.fixture(params=["sqlite", "postgres"])
def resume_db(request, tmp_path):
    kwargs = {"db_path": str(tmp_path / "roleplay-resume.sqlite"), "client_id": "alice"}
    if request.param == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(**kwargs)
    card = db.add_character_card(
        {"name": "Ari", "description": "The original archivist", "system_prompt": "Preserve continuity."}
    )
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Exact Resume"},
        provider="local-llm",
        model="local-test",
        initial_messages=[
            {"id": "ancestor", "sender": "user", "content": "Earlier question"},
            {"id": "prior", "sender": "assistant", "content": "Earlier answer", "parent_message_id": "ancestor"},
        ],
    )
    assert db.get_roleplay_resume_state(cid)["resume_eligible"] is True
    yield db, cid
    db.close_connection()
    if request.param == "postgres":
        db.backend.get_pool().close_all()


def fence(db, cid):
    state = db.get_roleplay_resume_state(cid)
    return {
        "version": 1,
        "snapshot_digest": state["behavior_snapshot"]["digest"],
        "settings_version": state["settings_version"],
        "history_version": state["history_version"],
        "tail": state["tail"],
    }


def append(db, cid, request):
    return db.append_idempotent_user_message(cid, request, owner_client_id="alice")


def test_unchanged_append_and_identical_replay_return_original_receipt(resume_db):
    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Exact proposal?\n  Keep spaces.  ", "expected": fence(db, cid)}
    original = deepcopy(request)
    accepted = append(db, cid, request)
    db.add_message(
        {
            "id": "later",
            "conversation_id": cid,
            "sender": "assistant",
            "content": "A later response",
            "parent_message_id": "chosen-user",
        }
    )
    replay = append(db, cid, request)
    assert replay == accepted
    assert request == original
    assert db.count_messages_for_conversation(cid) == 4
    assert accepted["parent_message_id"] == "prior"
    assert accepted["message_version"] == 1
    assert accepted["fence"]["tail"] == {"message_id": "chosen-user", "message_version": 1}
    assert db.get_message_by_id("chosen-user")["content"] == original["content"]


@pytest.mark.parametrize("change", ["content", "expected", "legacy-id"])
def test_conflicting_reuse_never_changes_the_accepted_input(resume_db, change):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    if change == "legacy-id":
        db.add_message({"id": "chosen-user", "conversation_id": cid, "sender": "user", "content": "Legacy row"})
    else:
        append(db, cid, request)
    altered = deepcopy(request)
    if change == "content":
        altered["content"] = "Replacement"
    if change == "expected":
        altered["expected"] = fence(db, cid)
    before = db.get_roleplay_resume_state(cid)["history_version"]
    with pytest.raises(RoleplayResumeConflictError) as caught:
        append(db, cid, altered)
    assert caught.value.code == "idempotency_conflict"
    assert db.get_roleplay_resume_state(cid)["history_version"] == before


@pytest.mark.parametrize(
    "mutation",
    [
        "ancestor-edit",
        "ancestor-delete",
        "append",
        "settings",
        "metadata",
        "snapshot-missing",
        "snapshot-invalid",
        "tail",
    ],
)
def test_original_fence_rejects_drift_without_a_user_or_branch(resume_db, mutation):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original proposal", "expected": fence(db, cid)}
    if mutation == "ancestor-edit":
        db.update_message("ancestor", {"content": "Changed earlier message"}, 1)
    elif mutation == "ancestor-delete":
        db.soft_delete_message("ancestor", expected_version=1)
    elif mutation == "append":
        db.add_message(
            {
                "id": "other-user",
                "conversation_id": cid,
                "sender": "user",
                "content": "Another tab",
                "parent_message_id": "prior",
            }
        )
    elif mutation == "settings":
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    elif mutation == "metadata":
        db.add_message_metadata("ancestor", extra={"changed": True})
    elif mutation.startswith("snapshot-"):
        with db.transaction() as conn:
            if mutation == "snapshot-missing":
                conn.execute("DELETE FROM conversation_behavior_snapshots WHERE conversation_id = ?", (cid,))
            else:
                conn.execute(
                    "UPDATE conversation_behavior_snapshots SET canonical_json = ?, size_bytes = 2 WHERE conversation_id = ?",
                    ("{}", cid),
                )
    elif mutation == "tail":
        request["expected"]["tail"] = {"message_id": "ancestor", "message_version": 1}
    before = (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"])
    with pytest.raises(RoleplayResumeConflictError):
        append(db, cid, request)
    assert db.get_message_by_id("chosen-user") is None
    assert (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"]) == before


def test_retired_owner_cannot_append_or_replay(resume_db):
    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    append(db, cid, request)
    for mid in ("chosen-user", "new-user"):
        with pytest.raises(NotFoundError):
            db.append_idempotent_user_message(cid, {**request, "message_id": mid}, owner_client_id="bob")
    assert db.count_messages_for_conversation(cid) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1.0),
        ("version", "1"),
        ("settings_version", 0),
        ("history_version", True),
        ("snapshot_digest", "bad"),
    ],
)
def test_request_rejects_coercion_and_incomplete_fences(field, value):
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import RoleplayResumeUserAppendRequest

    expected = {
        "version": 1,
        "snapshot_digest": "sha256:" + "a" * 64,
        "settings_version": 1,
        "history_version": 1,
        "tail": {"message_id": None, "message_version": None},
    }
    expected[field] = value
    with pytest.raises(ValidationError):
        RoleplayResumeUserAppendRequest.model_validate(
            {"message_id": "id", "content": "Original", "expected": expected}
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"role": "assistant"},
        {"parent_message_id": "other"},
        {"provider": "changed"},
        {"image_base64": "unqualified"},
        {"unknown": True},
    ],
)
def test_request_rejects_unqualified_append_controls(extra):
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import RoleplayResumeUserAppendRequest

    expected = {
        "version": 1,
        "snapshot_digest": "sha256:" + "a" * 64,
        "settings_version": 1,
        "history_version": 1,
        "tail": {"message_id": None, "message_version": None},
    }
    with pytest.raises(ValidationError):
        RoleplayResumeUserAppendRequest.model_validate(
            {"message_id": "id", "content": "Original", "expected": expected, **extra}
        )


@pytest.mark.parametrize("mutation", ["edited", "deleted", "snapshot-missing"])
def test_original_replay_does_not_rewrite_or_recheck_later_source_changes(resume_db, mutation):
    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    receipt = append(db, cid, request)
    if mutation == "edited":
        db.update_message("chosen-user", {"content": "Later edit"}, 1)
    elif mutation == "deleted":
        db.soft_delete_message("chosen-user", expected_version=1)
    else:
        with db.transaction() as conn:
            conn.execute("DELETE FROM conversation_behavior_snapshots WHERE conversation_id = ?", (cid,))
    before = (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"])
    assert append(db, cid, request) == receipt
    assert (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"]) == before


@pytest.mark.parametrize(
    "tail",
    [
        {"message_id": "id", "message_version": None},
        {"message_id": None, "message_version": 1},
        {"message_id": "", "message_version": 1},
    ],
)
def test_request_requires_complete_prior_tail_identity(tail):
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import RoleplayResumeUserAppendRequest

    expected = {
        "version": 1,
        "snapshot_digest": "sha256:" + "a" * 64,
        "settings_version": 1,
        "history_version": 1,
        "tail": tail,
    }
    with pytest.raises(ValidationError):
        RoleplayResumeUserAppendRequest.model_validate(
            {"message_id": "id", "content": "Original", "expected": expected}
        )


@pytest.mark.parametrize(
    "field",
    [
        "conversation_id",
        "message_id",
        "parent_message_id",
        "message_version",
        "snapshot_digest",
        "settings_version",
        "history_version",
        "tail",
    ],
)
def test_corrupted_original_receipt_cannot_claim_another_acknowledgment(resume_db, field):
    import json

    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    append(db, cid, request)
    with db.transaction() as conn:
        authority = json.loads(
            conn.execute("SELECT history_admission_json FROM messages WHERE id = ?", ("chosen-user",)).fetchone()[0]
        )
        receipt = authority["roleplay_resume_v1"]["receipt"]
        if field in {"conversation_id", "message_id", "parent_message_id"}:
            receipt[field] = "other-identity"
        elif field == "message_version":
            receipt[field] = 2
        elif field == "snapshot_digest":
            receipt["fence"][field] = "sha256:" + "b" * 64
        elif field in {"settings_version", "history_version"}:
            receipt["fence"][field] += 1
        else:
            receipt["fence"]["tail"] = {"message_id": "other-identity", "message_version": 1}
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?", (json.dumps(authority), "chosen-user")
        )
    before = db.get_roleplay_resume_state(cid)["history_version"]
    with pytest.raises(RoleplayResumeConflictError) as caught:
        append(db, cid, request)
    assert caught.value.code == "idempotency_conflict"
    assert db.get_roleplay_resume_state(cid)["history_version"] == before
    assert db.count_messages_for_conversation(cid) == 3


def test_simultaneous_identical_input_ids_return_one_original_receipt(resume_db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    db, cid = resume_db
    second = CharactersRAGDB(db_path=db.db_path_str, client_id="alice", backend=db.backend)
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    gate = Barrier(2)

    def send(connection_owner):
        gate.wait(timeout=10)
        return append(connection_owner, cid, request)

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs = [workers.submit(send, owner) for owner in (db, second)]
            receipts = [job.result(timeout=20) for job in jobs]
        assert receipts[0] == receipts[1]
        assert db.count_messages_for_conversation(cid) == 3
    finally:
        second.close_connection()


def test_identical_id_in_another_owned_conversation_conflicts_without_writing(resume_db):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)}
    append(db, cid, request)
    card = db.add_character_card({"name": "Other character"})
    other = create_character_conversation(
        db, conversation_data={"character_id": card}, provider="local-llm", model="local-test"
    )
    other_request = {**request, "expected": fence(db, other)}
    with pytest.raises(RoleplayResumeConflictError) as caught:
        append(db, other, other_request)
    assert caught.value.code == "idempotency_conflict"
    assert db.count_messages_for_conversation(other) == 0
    assert db.count_messages_for_conversation(cid) == 3


def test_protected_input_remains_readable_without_granting_h1_admission_authority(resume_db):
    db, cid = resume_db
    append(db, cid, {"message_id": "chosen-user", "content": "Original", "expected": fence(db, cid)})
    snapshot = db.get_conversation_history_snapshot(cid, owner_client_id="alice")
    assert snapshot.interpretation_status == {"kind": "parent_graph_v1"}
    chosen = next(node for node in snapshot.nodes if node["id"] == "chosen-user")
    assert chosen["settled"] is True
    content = db.get_conversation_history_selected_content(
        cid, ["ancestor", "prior", "chosen-user"], snapshot=snapshot, owner_client_id="alice"
    )
    assert [row["message"] for row in content] == ["Earlier question", "Earlier answer", "Original"]
