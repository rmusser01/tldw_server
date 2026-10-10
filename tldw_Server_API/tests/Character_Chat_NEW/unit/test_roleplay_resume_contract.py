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


def completion_request(db, cid):
    receipt = append(
        db,
        cid,
        {
            "message_id": "chosen-user",
            "content": "Original proposal?",
            "expected": fence(db, cid),
        },
    )
    return {
        "version": 1,
        "input_message_id": "chosen-user",
        "input_message_version": 1,
        "assistant_message_id": "chosen-assistant",
        "expected": receipt["fence"],
    }


def test_prepare_generation_uses_frozen_prompt_exact_branch_and_saved_model(resume_db, monkeypatch):
    db, cid = resume_db
    request = completion_request(db, cid)
    state = db.get_roleplay_resume_state(cid)
    card_id = int(state["behavior_snapshot"]["payload"]["participants"][0]["source"]["id"])
    with db.transaction() as conn:
        conn.execute(
            "UPDATE character_cards SET description = ?, system_prompt = ?, deleted = TRUE WHERE id = ?",
            ("Unrelated current behavior", "Forget everything", card_id),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Generation loaded a mutable source")

    for name in (
        "get_character_card_by_id",
        "get_prompt_preset",
        "list_character_exemplars",
        "list_persona_memory_entries",
    ):
        monkeypatch.setattr(db, name, forbidden)
    prepared = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert prepared["fence"] == request["expected"]
    assert prepared["effective_completion"] == state["effective_completion"]
    assert prepared["effective_completion"]["provider"] == "local-llm"
    assert prepared["effective_completion"]["model"] == "local-test"
    assert prepared["messages"][0]["role"] == "system"
    assert "original archivist" in prepared["messages"][0]["content"]
    assert "Forget everything" not in str(prepared["messages"])
    assert prepared["messages"][-3:] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Original proposal?"},
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "ancestor-edit",
        "ancestor-delete",
        "append",
        "settings",
        "input-edit",
        "input-delete",
        "input-id",
        "input-version",
        "snapshot-missing",
    ],
)
def test_prepare_generation_refuses_every_original_fence_drift(resume_db, mutation):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = completion_request(db, cid)
    if mutation == "ancestor-edit":
        db.update_message("ancestor", {"content": "Changed before dispatch"}, 1)
    elif mutation == "ancestor-delete":
        db.soft_delete_message("ancestor", expected_version=1)
    elif mutation == "append":
        db.add_message(
            {
                "id": "other-tab",
                "conversation_id": cid,
                "sender": "user",
                "content": "Other tab",
                "parent_message_id": "chosen-user",
            }
        )
    elif mutation == "settings":
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    elif mutation == "input-edit":
        db.update_message("chosen-user", {"content": "Different input"}, 1)
    elif mutation == "input-delete":
        db.soft_delete_message("chosen-user", expected_version=1)
    elif mutation == "input-id":
        request["input_message_id"] = "prior"
    elif mutation == "input-version":
        request["input_message_version"] = 2
    else:
        with db.transaction() as conn:
            conn.execute("DELETE FROM conversation_behavior_snapshots WHERE conversation_id = ?", (cid,))
    before = (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"])
    with pytest.raises(RoleplayResumeConflictError):
        db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert (db.count_messages_for_conversation(cid), db.get_roleplay_resume_state(cid)["history_version"]) == before
    assert db.get_message_by_id("chosen-assistant") is None


@pytest.mark.parametrize(
    "extra",
    [
        {"provider": "openai"},
        {"provider": None},
        {"model": "other"},
        {"temperature": 0.9},
        {"top_p": 0.9},
        {"repetition_penalty": 1.1},
        {"stop": ["halt"]},
        {"prompt_preset": "other"},
        {"directed_character_id": 1},
        {"continue_as_user": True},
        {"force_narrate": True},
        {"impersonate_user": True},
        {"tools": []},
        {"tool_choice": {}},
        {"max_tokens": 5},
        {"limit": 5},
        {"offset": 1},
        {"save_to_db": False},
        {"append_user_message": "Unacknowledged input"},
        {"include_character_context": False},
        {"unknown": True},
    ],
)
def test_protected_completion_rejects_request_time_behavior_and_unknown_controls(extra):
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
        CharacterChatCompletionV2Request,
    )

    expected = {
        "version": 1,
        "snapshot_digest": "sha256:" + "a" * 64,
        "settings_version": 1,
        "history_version": 2,
        "tail": {"message_id": "input", "message_version": 1},
    }
    contract = {
        "version": 1,
        "input_message_id": "input",
        "input_message_version": 1,
        "assistant_message_id": "reply",
        "expected": expected,
    }
    with pytest.raises(ValidationError):
        CharacterChatCompletionV2Request.model_validate({"tldw_roleplay_resume_v1": contract, **extra})


def test_protected_completion_does_not_change_legacy_request_validation():
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
        CharacterChatCompletionV2Request,
    )

    legacy = CharacterChatCompletionV2Request.model_validate(
        {
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "Legacy input",
            "save_to_db": False,
        }
    )
    assert legacy.append_user_message == "Legacy input"
    assert legacy.save_to_db is False


@pytest.mark.parametrize("contract", [None, False, {}, "legacy"])
def test_present_resume_contract_cannot_fall_through_to_legacy(contract):
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
        CharacterChatCompletionV2Request,
    )

    with pytest.raises(ValidationError):
        CharacterChatCompletionV2Request.model_validate({"tldw_roleplay_resume_v1": contract})


@pytest.mark.parametrize("corruption", ["version", "settled", "interpretation", "parent", "assistant-id"])
def test_prepare_generation_requires_original_input_authority(resume_db, corruption):
    import json

    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = completion_request(db, cid)
    if corruption == "assistant-id":
        request["assistant_message_id"] = "chosen-user"
    else:
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT history_admission_json FROM messages WHERE id = ?",
                ("chosen-user",),
            ).fetchone()
            authority = json.loads(row[0])
            if corruption == "version":
                authority["version"] = True
            elif corruption == "settled":
                authority["settled"] = False
            elif corruption == "interpretation":
                authority["interpretation"] = {"kind": "legacy"}
            else:
                authority["roleplay_resume_v1"]["receipt"]["parent_message_id"] = "ancestor"
            conn.execute(
                "UPDATE messages SET history_admission_json = ? WHERE id = ?",
                (json.dumps(authority), "chosen-user"),
            )
    with pytest.raises(RoleplayResumeConflictError):
        db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert db.get_message_by_id("chosen-assistant") is None


def test_saved_lore_memory_scene_and_examples_do_not_load_changed_sources(resume_db, monkeypatch):
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
        WorldBookService,
    )

    db, _ = resume_db
    card = db.add_character_card(
        {
            "name": "Lio",
            "description": "Original traveller",
            "first_message": "Welcome, {{user}}.",
        }
    )
    service = WorldBookService(db)
    book = service.create_world_book("Frozen harbor")
    service.add_entry(book, ["harbor"], "The harbor bell rings twice.")
    service.attach_to_character(book, card)
    db.add_character_exemplar(
        card,
        {"text": "The harbor welcomes careful travellers.", "rhetorical": ["opener"]},
    )
    db.create_persona_profile(
        {
            "id": f"char:{card}",
            "user_id": "alice",
            "name": f"char_memory:{card}",
            "origin_character_id": card,
            "origin_character_name": "Lio",
            "mode": "persistent_scoped",
            "is_active": True,
        }
    )
    db.add_persona_memory_entry(
        {
            "persona_id": f"char:{card}",
            "user_id": "alice",
            "memory_type": "fact",
            "content": "The user collects sea maps.",
        }
    )
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Resources"},
        provider="local-llm",
        model="local-test",
        memory_by_character_id={str(card): "Remember the lighthouse."},
        conversation_settings={
            "authorNote": "The tide is rising for {{char}}.",
            "authorNotePosition": {"mode": "depth", "depth": 1},
            "memoryScope": "both",
        },
    )
    receipt = append(
        db,
        cid,
        {
            "message_id": "resource-user",
            "content": "Explore the harbor.",
            "expected": fence(db, cid),
        },
    )
    request = {
        "version": 1,
        "input_message_id": "resource-user",
        "input_message_version": 1,
        "assistant_message_id": "resource-reply",
        "expected": receipt["fence"],
    }
    with db.transaction() as conn:
        conn.execute(
            "UPDATE character_cards SET deleted = TRUE, first_message = ? WHERE id = ?",
            ("Changed greeting", card),
        )
        conn.execute(
            "UPDATE world_book_entries SET content = ? WHERE world_book_id = ?",
            ("Changed lore", book),
        )
        conn.execute(
            "UPDATE persona_memory_entries SET content = ? WHERE persona_id = ?",
            ("Changed memory", f"char:{card}"),
        )
        conn.execute(
            "UPDATE character_exemplars SET text = ? WHERE character_id = ?",
            ("Changed example", card),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Loaded a mutable source")

    for name in (
        "get_character_card_by_id",
        "get_prompt_preset",
        "list_character_exemplars",
        "list_persona_memory_entries",
    ):
        monkeypatch.setattr(db, name, forbidden)
    monkeypatch.setattr(WorldBookService, "__init__", forbidden)
    prepared = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    text = str(prepared["messages"])
    for saved in (
        "harbor bell rings twice",
        "collects sea maps",
        "Remember the lighthouse",
        "tide is rising for Lio",
        "harbor welcomes careful travellers",
        "Welcome, User",
    ):
        assert saved in text
    assert "Changed" not in text
    roles = [row["role"] for row in prepared["messages"]]
    assert roles.index("assistant") < roles.index("user")


@pytest.mark.parametrize("snapshot_bound", [False, True])
def test_saved_context_source_identity_is_distinct_from_fork_address(resume_db, snapshot_bound):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_prompt import (
        build_roleplay_resume_messages,
    )

    db, cid = resume_db
    state = db.get_roleplay_resume_state(cid)
    original_source = state["behavior_snapshot"]["payload"]["participants"][0]["source"]["id"]
    if snapshot_bound:
        state["conversation"].update(
            {
                "id": "fork-address",
                "character_id": None,
                "assistant_id": "snapshot:fork-address",
                "assistant_binding_mode": "snapshot_v1",
            }
        )
    context, messages = build_roleplay_resume_messages(state, [{"role": "user", "content": "Continue."}])
    assert str(context["character_id"]) == original_source
    assert messages[-1] == {"role": "user", "content": "Continue."}


def test_acknowledged_literal_input_is_identical_in_admission_and_generation(resume_db):
    db, cid = resume_db
    content = "Keep {{char}} and {{user}} literal.\n  Exact spaces.  "
    receipt = append(
        db,
        cid,
        {"message_id": "literal-input", "content": content, "expected": fence(db, cid)},
    )
    prepared = db.prepare_roleplay_resume_generation(
        cid,
        {
            "version": 1,
            "input_message_id": "literal-input",
            "input_message_version": 1,
            "assistant_message_id": "literal-reply",
            "expected": receipt["fence"],
        },
        owner_client_id="alice",
    )
    assert prepared["messages"][-1] == {"role": "user", "content": content}


def test_resume_commit_saves_one_exact_reply_and_replays_original_receipt(resume_db):
    db, cid = resume_db
    request = completion_request(db, cid)
    prepared = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    content = "Original generated answer.\n  Keep spaces.  "
    result = db.commit_roleplay_resume_generation(cid, prepared, content, owner_client_id="alice")
    assert result["saved"] is True
    assert result["assistant_content"] == content
    assert result["assistant_message_id"] == "chosen-assistant"
    assert db.get_message_by_id("chosen-assistant")["parent_message_id"] == "chosen-user"
    assert db.count_messages_for_conversation(cid) == 4
    db.update_message("chosen-assistant", {"content": "Later edited answer"}, 1)
    assert db.commit_roleplay_resume_generation(cid, prepared, content, owner_client_id="alice") == result
    replay = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert replay["completed"] == result
    assert db.count_messages_for_conversation(cid) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "ancestor-edit",
        "ancestor-delete",
        "append",
        "settings",
        "input-edit",
        "input-delete",
        "metadata",
        "snapshot-missing",
    ],
)
def test_changes_during_generation_refuse_assistant_commit_without_a_branch(resume_db, mutation):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = completion_request(db, cid)
    prepared = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    if mutation == "ancestor-edit":
        db.update_message("ancestor", {"content": "Another writer"}, 1)
    elif mutation == "ancestor-delete":
        db.soft_delete_message("ancestor", expected_version=1)
    elif mutation == "append":
        db.add_message(
            {
                "id": "other-writer",
                "conversation_id": cid,
                "sender": "user",
                "content": "Another writer",
                "parent_message_id": "chosen-user",
            }
        )
    elif mutation == "settings":
        db.upsert_conversation_settings(cid, db.get_roleplay_resume_state(cid)["settings"])
    elif mutation == "input-edit":
        db.update_message("chosen-user", {"content": "Another input"}, 1)
    elif mutation == "input-delete":
        db.soft_delete_message("chosen-user", expected_version=1)
    elif mutation == "metadata":
        db.add_message_metadata("ancestor", extra={"sender_name": "Changed context"})
    else:
        with db.transaction() as conn:
            conn.execute("DELETE FROM conversation_behavior_snapshots WHERE conversation_id = ?", (cid,))
    before = db.get_roleplay_resume_state(cid)["history_version"]
    with pytest.raises(RoleplayResumeConflictError):
        db.commit_roleplay_resume_generation(cid, prepared, "Retained generated reply", owner_client_id="alice")
    assert db.get_message_by_id("chosen-assistant") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before


@pytest.mark.parametrize("recursive", [False, True])
def test_regex_timeout_rejects_resume_before_user_admission(resume_db, monkeypatch, recursive):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
        WorldBookEntry,
        WorldBookService,
    )

    db, _ = resume_db
    card = db.add_character_card({"name": "Tess", "description": "Frozen lore timeout"})
    service = WorldBookService(db)
    book = service.create_world_book("Timed-out frozen lore", recursive_scanning=recursive)
    if recursive:
        service.add_entry(book, ["harbor"], "chain", metadata={"recursive_scanning": True})
    trigger = "chain" if recursive else "harbor"
    danger = service.add_entry(book, [trigger], "Required lore", regex_match=True)
    service.attach_to_character(book, card)
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Timeout refusal"},
        provider="local-llm",
        model="local-test",
    )
    original = WorldBookEntry._search_pattern

    def search(self, pattern, text, *, strict=False):
        if self.entry_id == danger and trigger in text:
            if strict:
                raise TimeoutError("Owned evaluation failed")
            return None
        return original(self, pattern, text)

    monkeypatch.setattr(WorldBookEntry, "_search_pattern", search)
    before = db.get_roleplay_resume_state(cid)["history_version"]
    with pytest.raises(RoleplayResumeConflictError) as caught:
        append(
            db,
            cid,
            {
                "message_id": "timeout-user",
                "content": "harbor",
                "expected": fence(db, cid),
            },
        )
    assert caught.value.code == "resume_context_unsupported"
    assert db.get_message_by_id("timeout-user") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before


@pytest.mark.parametrize(
    "key,value,accepted",
    [
        ("enabled", 0, False),
        ("gm_only", 1, False),
        ("exclude_from_prompt", 1, False),
        ("enabled", False, True),
        ("gm_only", True, True),
        ("exclude_from_prompt", True, True),
    ],
)
def test_materialized_note_controls_do_not_coerce_suppression(resume_db, key, value, accepted):
    import json

    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError
    from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
        build_materialized_behavior_settings,
    )

    db, cid = resume_db
    state = db.get_roleplay_resume_state(cid)
    settings = deepcopy(state["settings"])
    values = deepcopy(state["materialized_settings"]["values"])
    values["behavior_controls"]["author_note"][key] = value
    values["behavior_controls"]["applied_overrides"]["authorNote"] = "Suppressed frozen note"
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversation_settings SET settings_json = ? WHERE conversation_id = ?",
            (json.dumps(settings), cid),
        )
    request = {
        "message_id": "control-input",
        "content": "Continue.",
        "expected": fence(db, cid),
    }
    if accepted:
        receipt = append(db, cid, request)
        prepared = db.prepare_roleplay_resume_generation(
            cid,
            {
                "version": 1,
                "input_message_id": "control-input",
                "input_message_version": 1,
                "assistant_message_id": "control-reply",
                "expected": receipt["fence"],
            },
            owner_client_id="alice",
        )
        assert "Suppressed frozen note" not in str(prepared["messages"])
    else:
        before = db.get_roleplay_resume_state(cid)["history_version"]
        with pytest.raises(RoleplayResumeConflictError):
            append(db, cid, request)
        assert db.get_message_by_id("control-input") is None
        assert db.get_roleplay_resume_state(cid)["history_version"] == before


@pytest.mark.parametrize(
    "path,numeric_kind",
    [
        (("version",), "bool"),
        (("input_message_version",), "bool"),
        (("input_message_version",), "float"),
        (("expected", "settings_version"), "float"),
        (("expected", "history_version"), "float"),
        (("expected", "tail", "message_version"), "float"),
    ],
)
def test_original_completion_request_rejects_stored_numeric_coercion(resume_db, path, numeric_kind):
    import json

    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    request = completion_request(db, cid)
    prepared = db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    db.commit_roleplay_resume_generation(cid, prepared, "Original reply", owner_client_id="alice")
    with db.transaction() as conn:
        raw = conn.execute(
            "SELECT history_admission_json FROM messages WHERE id = ?",
            ("chosen-assistant",),
        ).fetchone()[0]
        authority = json.loads(raw)
        stored = authority["roleplay_completion_v1"]["request"]
        for key in path[:-1]:
            stored = stored[key]
        value = True if numeric_kind == "bool" else float(stored[path[-1]])
        # The corruption must compare equal under Python's ordinary numeric equality.
        assert stored[path[-1]] == value
        stored[path[-1]] = value
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?",
            (json.dumps(authority), "chosen-assistant"),
        )
    with pytest.raises(RoleplayResumeConflictError, match="idempotency_conflict"):
        db.prepare_roleplay_resume_generation(cid, request, owner_client_id="alice")
    assert db.count_messages_for_conversation(cid) == 4


def test_unsupported_saved_sampling_refuses_before_input_admission(resume_db):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, _ = resume_db
    card = db.add_character_card(
        {
            "name": "Sampling Ari",
            "extensions": {"tldw": {"generation": {"repetition_penalty": 1.2}}},
        }
    )
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Unsupported saved sampling"},
        provider="local-llm",
        model="local-test",
    )
    before = db.get_roleplay_resume_state(cid)
    assert before["resume_eligible"] is True
    assert before["effective_completion"]["sampling"]["repetition_penalty"] == 1.2
    with pytest.raises(RoleplayResumeConflictError, match="resume_sampling_unsupported"):
        append(
            db,
            cid,
            {
                "message_id": "sampling-user",
                "content": "Original input",
                "expected": fence(db, cid),
            },
        )
    assert db.get_message_by_id("sampling-user") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]


@pytest.mark.parametrize("position", [None, {"mode": "depth", "depth": 1}])
def test_saved_note_position_preserves_first_present_alias(resume_db, position):
    from tldw_Server_API.app.core.Character_Chat.roleplay_prompt_effects import (
        _inject_author_note_from_settings,
    )
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_prompt import (
        build_roleplay_resume_messages,
    )

    db, _ = resume_db
    card = {"name": "Note Ava", "description": "An archivist."}
    card_id = db.add_character_card(card)
    settings = {
        "authorNote": "Keep the note.",
        "authorNotePosition": position,
        "authorNotePlacement": {"mode": "depth", "depth": 1},
    }
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card_id, "title": "Note position"},
        provider="local-llm",
        model="local-test",
        conversation_settings=settings,
    )
    context, messages = build_roleplay_resume_messages(
        db.get_roleplay_resume_state(cid), [{"role": "user", "content": "Continue."}]
    )
    legacy = _inject_author_note_from_settings(
        [
            {"role": "system", "content": context["card"]["system_prompt"]},
            {"role": "user", "content": "Continue."},
        ],
        {"settings": settings},
        card,
        "Note Ava",
        "User",
    )
    assert messages == legacy


@pytest.mark.parametrize("preset", ["default", "st_default"])
@pytest.mark.parametrize("fields", ["empty", "literal", "ordinary"])
def test_saved_builtin_presets_preserve_live_semantics_without_live_definitions(resume_db, monkeypatch, preset, fields):
    from tldw_Server_API.app.core.Character_Chat.modules import character_prompt_presets
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_prompt import (
        build_roleplay_resume_messages,
    )
    from tldw_Server_API.app.core.Chat import history_context

    db, _ = resume_db
    card = {"name": "Frozen Ava", "extensions": {"tldw": {"prompt_preset": preset}}}
    if fields != "empty":
        card.update(
            {
                "description": (
                    "Keep {{scenario}} literal." if fields == "literal" else "  Guide {{user}} as {{char}}.  "
                ),
                "personality": "Patient.",
                "scenario": "A harbor.",
                "system_prompt": "Stay curious.",
                "message_example": "{{char}}: Hello {{user}}.",
                "post_history_instructions": "Ask about the map.",
            }
        )
    expected = character_prompt_presets.build_character_system_prompt(card, "Frozen Ava", "User", preset=preset)
    card_id = db.add_character_card(card)
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card_id, "title": "Frozen builtin"},
        provider="local-llm",
        model="local-test",
    )
    state = db.get_roleplay_resume_state(cid)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reloaded current builtin behavior")

    monkeypatch.setattr(character_prompt_presets, "get_builtin_presets", forbidden)
    monkeypatch.setattr(history_context, "build_character_system_prompt", forbidden)
    monkeypatch.setattr(db, "get_character_card_by_id", forbidden)
    context, messages = build_roleplay_resume_messages(state, [{"role": "user", "content": "Continue."}])
    assert context["card"]["system_prompt"] == expected
    assert messages[0] == {"role": "system", "content": expected}


@pytest.mark.parametrize("memory_only", [False, True])
def test_old_note_alias_materialization_refuses_before_append(resume_db, monkeypatch, memory_only):
    from tldw_Server_API.app.core.Character_Chat import (
        character_conversation_factory as factory,
    )
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, _ = resume_db
    card = db.add_character_card({"name": "Historical Note Ava"})
    settings = {
        "authorNote": "Original note",
        "authorNotePosition": None,
        "authorNotePlacement": {"mode": "depth", "depth": 1},
    }
    with monkeypatch.context() as old_writer:
        old_writer.setattr(factory, "_author_note_position", lambda row: row["authorNotePlacement"])
        cid = factory.create_character_conversation(
            db,
            conversation_data={"character_id": card, "title": "Old note aliases"},
            provider="local-llm",
            model="local-test",
            conversation_settings=settings,
        )
        with db.transaction() as conn:
            state = db.get_roleplay_resume_state(cid, conn=conn)
            rebuilt = factory.materialize_roleplay_behavior_settings(
                conn,
                conversation=state["conversation"],
                resume_state=state,
                merged_settings=state["settings"],
                owner_user_id="alice",
                changed_keys={"authorNotePosition"},
            )
            db.upsert_conversation_settings(cid, {**state["settings"], "roleplayBehaviorV1": rebuilt}, conn=conn)
    if memory_only:
        with db.transaction() as conn:
            state = db.get_roleplay_resume_state(cid, conn=conn)
            merged = {**state["settings"], "promptPreset": "st_default"}
            rebuilt = factory.materialize_roleplay_behavior_settings(
                conn,
                conversation=state["conversation"],
                resume_state=state,
                merged_settings=merged,
                owner_user_id="alice",
                changed_keys={"promptPreset"},
            )
            db.upsert_conversation_settings(cid, {**merged, "roleplayBehaviorV1": rebuilt}, conn=conn)
    before = db.get_roleplay_resume_state(cid)
    assert before["resume_eligible"] is True
    values = before["materialized_settings"]["values"]
    assert values["memory"]["author_note_position"] == settings["authorNotePlacement"]
    if memory_only:
        assert values["behavior_controls"]["author_note"]["position"] is None
    with pytest.raises(RoleplayResumeConflictError, match="resume_context_unsupported"):
        append(
            db,
            cid,
            {
                "message_id": "historical-input",
                "content": "Continue.",
                "expected": fence(db, cid),
            },
        )
    assert db.get_message_by_id("historical-input") is None
    assert db.get_roleplay_resume_state(cid)["settings"] == before["settings"]


@pytest.mark.parametrize(
    "provider,model,sampling,settings,reason",
    [
        ("openai", "gpt-5", {"top_p": 0.4}, {}, "resume_sampling_unsupported"),
        ("mlx", "saved-model", {"stop": []}, {}, "resume_provider_unsupported"),
        ("mlx", "saved-model", {"stop": ["END"]}, {}, "resume_provider_unsupported"),
        (
            "anthropic",
            None,
            {},
            {
                "authorNote": "Depth note",
                "authorNotePosition": {"mode": "depth", "depth": 1},
            },
            "resume_context_unsupported",
        ),
    ],
)
def test_unrepresentable_provider_behavior_refuses_before_input(resume_db, provider, model, sampling, settings, reason):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, _ = resume_db
    if model is None:
        from tldw_Server_API.app.core.Chat.chat_service import (
            known_models_for_provider_cached,
        )

        model = known_models_for_provider_cached(provider)[0]
    card = db.add_character_card({"name": "Provider Control Ava"})
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Unrepresentable behavior"},
        provider=provider,
        model=model,
        sampling=sampling,
        conversation_settings=settings,
    )
    before = db.get_roleplay_resume_state(cid)
    assert before["resume_eligible"] is True
    with pytest.raises(RoleplayResumeConflictError, match=reason):
        append(
            db,
            cid,
            {
                "message_id": "unsupported-provider-input",
                "content": "Continue.",
                "expected": fence(db, cid),
            },
        )
    assert db.get_message_by_id("unsupported-provider-input") is None
    assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]


@pytest.mark.parametrize(
    "provider,legacy,accepted",
    [
        ("google", False, False),
        ("cohere", False, False),
        ("anthropic", True, False),
        ("anthropic", False, True),
    ],
)
def test_vendor_history_tool_mapping_is_admitted_only_when_representable(resume_db, provider, legacy, accepted):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError
    from tldw_Server_API.app.core.Chat.chat_service import (
        known_models_for_provider_cached,
    )

    db, _ = resume_db
    card = db.add_character_card({"name": "Tool Vendor Ava"})
    if legacy:
        call = {"extra_metadata": {"function_call": {"name": "archive", "arguments": "{}"}}}
        result = {
            "sender": "function",
            "extra_metadata": {"sender_role": "function", "sender_name": "archive"},
        }
    else:
        call = {
            "tool_calls": [
                {
                    "id": "historical-call",
                    "type": "function",
                    "function": {"name": "archive", "arguments": "{}"},
                }
            ]
        }
        result = {
            "sender": "tool",
            "extra_metadata": {
                "sender_role": "tool",
                "tool_call_id": "historical-call",
            },
        }
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Vendor tool history"},
        provider=provider,
        model=known_models_for_provider_cached(provider)[0],
        initial_messages=[
            {"id": "tool-question", "sender": "user", "content": "Search the archive."},
            {
                "id": "tool-answer",
                "sender": "assistant",
                "content": "I searched.",
                "parent_message_id": "tool-question",
                **call,
            },
            {
                "id": "tool-result",
                "content": "Found a map.",
                "parent_message_id": "tool-answer",
                **result,
            },
        ],
    )
    assert (
        db.get_message_metadata("tool-answer")["extra" if legacy else "tool_calls"]
        == call["extra_metadata" if legacy else "tool_calls"]
    )
    assert db.get_message_metadata("tool-result")["extra"] == result["extra_metadata"]
    before = db.get_roleplay_resume_state(cid)
    assert before["resume_eligible"] is True
    request = {
        "message_id": "tool-next-input",
        "content": "Continue.",
        "expected": fence(db, cid),
    }
    if accepted:
        receipt = append(db, cid, request)
        prepared = db.prepare_roleplay_resume_generation(
            cid,
            {
                "version": 1,
                "input_message_id": "tool-next-input",
                "input_message_version": 1,
                "assistant_message_id": "tool-next-reply",
                "expected": receipt["fence"],
            },
            owner_client_id="alice",
        )
        messages = prepared["context"]["provider_arguments"]["messages_payload"]
        assert any(message.get("tool_calls") for message in messages)
        assert any(message.get("tool_call_id") == "historical-call" for message in messages)
    else:
        with pytest.raises(RoleplayResumeConflictError, match="resume_context_unsupported"):
            append(db, cid, request)
        assert db.get_message_by_id("tool-next-input") is None
        assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing-result",
        "unmatched-result",
        "duplicate-call",
        "nonfinite-arguments",
        "credential-arguments",
    ],
)
def test_resume_history_requires_closed_inert_tool_replay(resume_db, damage):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    arguments = (
        '{"value":NaN}'
        if damage == "nonfinite-arguments"
        else '{"api_key":"test-only"}' if damage == "credential-arguments" else "{}"
    )
    calls = [
        {
            "id": "history-call",
            "type": "function",
            "function": {"name": "archive", "arguments": arguments},
        }
    ]
    if damage == "duplicate-call":
        calls.append(deepcopy(calls[0]))
    db.add_message(
        {
            "id": "history-tool-assistant",
            "conversation_id": cid,
            "sender": "assistant",
            "content": "I searched.",
            "parent_message_id": "prior",
        }
    )
    assert db.add_message_metadata("history-tool-assistant", tool_calls=calls)
    if damage != "missing-result":
        db.add_message(
            {
                "id": "history-tool-result",
                "conversation_id": cid,
                "sender": "tool",
                "content": "Found a map.",
                "parent_message_id": "history-tool-assistant",
            }
        )
        assert db.add_message_metadata(
            "history-tool-result",
            extra={
                "sender_role": "tool",
                "tool_call_id": ("other-call" if damage == "unmatched-result" else "history-call"),
            },
        )
    before = db.get_roleplay_resume_state(cid)
    request = {
        "message_id": "closed-tool-input",
        "content": "Continue.",
        "expected": fence(db, cid),
    }
    if damage is None:
        receipt = append(db, cid, request)
        assert receipt["parent_message_id"] == "history-tool-result"
    else:
        with pytest.raises(RoleplayResumeConflictError, match="resume_context_unsupported"):
            append(db, cid, request)
        assert db.get_message_by_id("closed-tool-input") is None
        assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]


def test_character_creation_metadata_failure_rolls_back_initial_messages(resume_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    db, _ = resume_db
    card = db.add_character_card({"name": "Metadata Atomic Ava"})
    monkeypatch.setattr(db, "add_message_metadata", lambda *args, **kwargs: False)
    with pytest.raises(InputError, match="initial message metadata"):
        create_character_conversation(
            db,
            conversation_data={
                "character_id": card,
                "title": "Atomic metadata failure",
            },
            provider="local-llm",
            model="local-test",
            initial_messages=[
                {
                    "id": "metadata-failed-initial",
                    "sender": "user",
                    "content": "Keep me.",
                    "extra_metadata": {"sender_role": "user"},
                }
            ],
        )
    assert db.get_message_by_id("metadata-failed-initial") is None
    assert (
        db.execute_query(
            "SELECT COUNT(*) FROM conversations WHERE title = ?",
            ("Atomic metadata failure",),
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("damage", [None, "blank-name", "scalar-arguments", "image-result"])
def test_anthropic_tool_history_must_retain_its_actual_provider_shape(resume_db, damage):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError
    from tldw_Server_API.app.core.Chat.chat_service import (
        known_models_for_provider_cached,
    )

    db, _ = resume_db
    card = db.add_character_card({"name": "Anthropic Shape Ava"})
    calls = [
        {
            "id": "shape-call",
            "type": "function",
            "function": {
                "name": " " if damage == "blank-name" else "archive",
                "arguments": "3" if damage == "scalar-arguments" else "{}",
            },
        }
    ]
    result = {
        "id": "shape-result",
        "sender": "tool",
        "content": "Found a map.",
        "parent_message_id": "shape-answer",
        "extra_metadata": {"sender_role": "tool", "tool_call_id": "shape-call"},
    }
    if damage == "image-result":
        result["images"] = [{"data": b"synthetic-image-bytes", "mime": "image/png"}]
    cid = create_character_conversation(
        db,
        conversation_data={"character_id": card, "title": "Anthropic shape"},
        provider="anthropic",
        model=known_models_for_provider_cached("anthropic")[0],
        initial_messages=[
            {
                "id": "shape-question",
                "sender": "user",
                "content": "Search the archive.",
            },
            {
                "id": "shape-answer",
                "sender": "assistant",
                "content": "I searched.",
                "parent_message_id": "shape-question",
                "tool_calls": calls,
            },
            result,
        ],
    )
    before = db.get_roleplay_resume_state(cid)
    request = {
        "message_id": "shape-next-input",
        "content": "Continue.",
        "expected": fence(db, cid),
    }
    if damage is None:
        receipt = append(db, cid, request)
        assert receipt["parent_message_id"] == "shape-result"
    else:
        with pytest.raises(RoleplayResumeConflictError, match="resume_context_unsupported"):
            append(db, cid, request)
        assert db.get_message_by_id("shape-next-input") is None
        assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]


def test_character_creation_failed_initial_insertion_rolls_back(resume_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    db, _ = resume_db
    card = db.add_character_card({"name": "Insertion Atomic Ava"})
    monkeypatch.setattr(db, "add_message", lambda *args, **kwargs: None)
    with pytest.raises(InputError, match="initial message"):
        create_character_conversation(
            db,
            conversation_data={
                "character_id": card,
                "title": "Atomic insertion failure",
            },
            provider="local-llm",
            model="local-test",
            initial_messages=[
                {
                    "id": "insertion-failed-initial",
                    "sender": "user",
                    "content": "Keep me.",
                }
            ],
        )
    assert (
        db.execute_query(
            "SELECT COUNT(*) FROM conversations WHERE title = ?",
            ("Atomic insertion failure",),
        ).fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_actual_empty_tool_result_is_retained_as_empty_text(resume_db):
    from tldw_Server_API.app.api.v1.endpoints.chat import (
        _process_content_for_db_sync,
        _save_message_turn_to_db,
    )
    from tldw_Server_API.app.core.Chat.persistence_service import (
        prepare_native_history_message,
    )

    db, cid = resume_db
    call = await _save_message_turn_to_db(
        db,
        cid,
        {
            "role": "assistant",
            "content": None,
            "parent_message_id": "prior",
            "tool_calls": [
                {
                    "id": "empty-call",
                    "type": "function",
                    "function": {"name": "archive", "arguments": "{}"},
                }
            ],
        },
        use_transaction=True,
    )
    result_body = {
        "role": "tool",
        "content": "",
        "tool_call_id": "empty-call",
        "parent_message_id": call,
    }
    result = await _save_message_turn_to_db(db, cid, result_body, use_transaction=True)
    assert result is not None
    assert db.get_message_metadata(result)["extra"]["content_placeholder_reason"] == "empty_tool_result"
    prepared_row = await prepare_native_history_message(result_body, cid, _process_content_for_db_sync)
    assert prepared_row["extra_metadata"]["content_placeholder_reason"] == "empty_tool_result"
    receipt = append(
        db,
        cid,
        {
            "message_id": "empty-result-input",
            "content": "Continue.",
            "expected": fence(db, cid),
        },
    )
    prepared = db.prepare_roleplay_resume_generation(
        cid,
        {
            "version": 1,
            "input_message_id": "empty-result-input",
            "input_message_version": 1,
            "assistant_message_id": "empty-result-reply",
            "expected": receipt["fence"],
        },
        owner_client_id="alice",
    )
    assert next(message for message in prepared["messages"] if message.get("role") == "tool")["content"] == ""


@pytest.mark.parametrize("malformed", [False, True])
def test_unbacked_null_tool_result_refuses_before_new_input(resume_db, malformed):
    from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import RoleplayResumeConflictError

    db, cid = resume_db
    db.add_message(
        {
            "id": "null-call",
            "conversation_id": cid,
            "sender": "assistant",
            "content": "I searched.",
            "parent_message_id": "prior",
        }
    )
    db.add_message_metadata(
        "null-call",
        tool_calls=[
            {
                "id": "null-call-id",
                "type": "function",
                "function": {"name": "archive", "arguments": "{}"},
            }
        ],
    )
    db.add_message(
        {
            "id": "null-result",
            "conversation_id": cid,
            "sender": "tool",
            "content": "[function_call]" if malformed else "Found a map.",
            "parent_message_id": "null-call",
        }
    )
    extra = {"sender_role": "tool", "tool_call_id": "null-call-id"}
    if malformed:
        extra["content_placeholder_reason"] = "function_call"
    db.add_message_metadata("null-result", extra=extra)
    before = db.get_roleplay_resume_state(cid)
    request = {
        "message_id": "null-next-input",
        "content": "Continue.",
        "expected": fence(db, cid),
    }
    if malformed:
        with pytest.raises(RoleplayResumeConflictError, match="resume_context_unsupported"):
            append(db, cid, request)
        assert db.get_message_by_id("null-next-input") is None
        assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]
    else:
        assert append(db, cid, request)["parent_message_id"] == "null-result"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [False, True])
async def test_actual_saved_placeholder_is_checked_before_h1_input_admission(resume_db, invalid):
    import asyncio

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.chat import (
        _process_content_for_db_sync,
        _save_message_turn_to_db,
    )
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
        ChatCompletionRequest,
    )
    from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
    from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages
    from tldw_Server_API.app.core.Chat.history_selection import (
        resolve_history_selection,
        snapshot_to_wire,
    )
    from tldw_Server_API.app.core.Chat.persistence_service import (
        prepare_native_history_message,
    )

    db, cid = resume_db
    saved = await _save_message_turn_to_db(
        db,
        cid,
        {
            "role": "user",
            "content": None if invalid else "Literal saved row",
            "parent_message_id": "prior",
        },
        use_transaction=True,
    )
    assert saved is not None
    owner = "server:test/account:alice"
    snap = snapshot_to_wire(db.get_conversation_history_snapshot(cid, owner_client_id="alice", owner_key=owner))
    resolved = resolve_history_selection(
        snap,
        {
            "owner_key": owner,
            "conversation_id": cid,
            "interpretation": {"kind": "parent_graph_v1"},
            "cursor": {"kind": "after_message", "message_id": saved},
            "selection_revision": 1,
        },
        "send",
        "actual-h1-test",
    )
    assert resolved["status"] == "ready"
    message = {"role": "user", "content": "Next H1 input"}
    req = ChatCompletionRequest(
        model="local-test",
        conversation_id=cid,
        messages=[message],
        save_to_db=True,
        tldw_history_selection_v1=resolved["selection"],
    )
    runtime = {
        "history_owner_key": owner,
        "history_owner_client_id": "alice",
        "history_inputs": [await prepare_native_history_message(message, cid, _process_content_for_db_sync)],
    }
    before = db.get_roleplay_resume_state(cid)
    before_count = db.count_messages_for_conversation(cid)

    async def build():
        return await build_context_and_messages(
            chat_db=db,
            request_data=req,
            loop=asyncio.get_running_loop(),
            metrics=get_chat_metrics(),
            default_save_to_db=True,
            final_conversation_id=cid,
            save_message_fn=_save_message_turn_to_db,
            runtime_state=runtime,
        )

    if invalid:
        with pytest.raises(HTTPException) as failure:
            await build()
        assert failure.value.status_code == 409
        assert db.count_messages_for_conversation(cid) == before_count
        assert db.get_roleplay_resume_state(cid)["history_version"] == before["history_version"]
    else:
        assert (await build())[4][-1]["content"] == "Next H1 input"
        assert db.count_messages_for_conversation(cid) == before_count + 1
