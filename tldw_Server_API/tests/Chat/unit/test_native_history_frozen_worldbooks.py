"""Native selected history uses owner-bound frozen lore without live sources."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.chat import _process_content_for_db_sync, _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import build_behavior_snapshot
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import apply_prompt_templating, build_context_and_messages
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection, snapshot_to_wire
from tldw_Server_API.app.core.Chat.persistence_service import native_history_owner_key, prepare_native_history_message
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import build_materialized_behavior_settings
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

pytestmark = pytest.mark.integration
QUESTION = "What is the ORBIT-742 access phrase?"
LORE = "ORBIT-742 requires the access phrase amber kestrel."
OWNER_KEY = native_history_owner_key(Request({
    "type": "http", "scheme": "http", "server": ("testserver", 80),
    "path": "/api/v1/chat/completions", "headers": [],
}), "1")


@pytest.fixture
def frozen_chat(tmp_path, request) -> Iterator[SimpleNamespace]:
    """Create actual scoped storage and capture source data through the factory."""
    backend = (
        request.getfixturevalue("pg_restricted_backend") if getattr(request, "param", "sqlite") == "postgres" else None
    )
    db = CharactersRAGDB(tmp_path / "native-lore.db", client_id="1", **({"backend": backend} if backend else {}))
    books = WorldBookService(db)
    character_id = db.add_character_card({"name": "LoreGuide", "system_prompt": "Preserve the character instructions."})
    book_id = books.create_world_book("ORBIT canon")
    entry_id = books.add_world_book_entry(book_id, ["ORBIT-742"], LORE)
    books.attach_to_character(book_id, character_id)
    conversation_id = create_character_conversation(
        db,
        conversation_data={"character_id": character_id, "title": "Native frozen lore"},
        provider="local-llm",
        model="local-test",
    )
    db.close_connection()
    try:
        yield SimpleNamespace(
            db=db,
            books=books,
            character_id=character_id,
            book_id=book_id,
            entry_id=entry_id,
            conversation_id=conversation_id,
        )
    finally:
        db.close_all_connections()


def capture_selection(chat, *, after=None):
    """Capture the real owned path and bind its canonical send selection."""
    snapshot = snapshot_to_wire(chat.db.get_conversation_history_snapshot(
        chat.conversation_id, owner_client_id="1", owner_key=OWNER_KEY,
    ))
    result = resolve_history_selection(snapshot, {
        "owner_key": OWNER_KEY, "conversation_id": chat.conversation_id,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "after_message", "message_id": after} if after else {"kind": "empty"},
        "selection_revision": 1,
    }, "send", "uat-frozen-lore")
    assert result["status"] == "ready", result  # nosec B101 - pytest behavior assertion, not a runtime security guard
    chat.db.close_connection()
    return result["selection"]


async def build_native(chat, *, text=QUESTION, system=None, selection=None, owner_client="1", owner_key=OWNER_KEY):
    """Prepare/admit through real native history and render its provider payload."""
    selection = selection or capture_selection(chat)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    request = ChatCompletionRequest(
        model="local-test", api_provider="local-llm", conversation_id=chat.conversation_id,
        character_id=str(chat.character_id), save_to_db=True, messages=messages,
        tldw_history_selection_v1=selection,
    )
    prepared_input = await prepare_native_history_message(messages[-1], chat.conversation_id, _process_content_for_db_sync)
    runtime = {"history_owner_key": owner_key, "history_owner_client_id": owner_client,
               "history_inputs": [prepared_input]}
    loop = asyncio.get_running_loop()
    try:
        with chacha_operation(independent=True):
            prepared = await build_context_and_messages(
                chat_db=chat.db, request_data=request, loop=loop, metrics=get_chat_metrics(),
                default_save_to_db=True, final_conversation_id=chat.conversation_id,
                save_message_fn=_save_message_turn_to_db, runtime_state=runtime,
            )
        instructions, payload = apply_prompt_templating(request, prepared[0], prepared[4])
        return instructions, payload, runtime
    finally:
        await loop.run_in_executor(None, chat.db.close_connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [None, "Preserve the custom user instructions."])
async def test_native_owner_snapshot_build_includes_frozen_lore_after_base_instructions(frozen_chat, monkeypatch, system):
    """Removing the frozen projector or replacing base instructions breaks this."""
    # Editing current sources must not change the accepted conversation's lore.
    frozen_chat.books.update_entry(frozen_chat.entry_id, content="LIVE replacement must never reach the prompt.")
    frozen_chat.db.close_connection()

    def no_live_reader(*args, **kwargs):
        raise AssertionError("Native context must use its accepted snapshot, not the live catalog")

    monkeypatch.setattr(WorldBookService, "get_world_book", no_live_reader)
    monkeypatch.setattr(WorldBookService, "get_entries", no_live_reader)
    monkeypatch.setattr(frozen_chat.db, "get_character_card_by_id", no_live_reader)
    instructions, payload, runtime = await build_native(frozen_chat, system=system)
    expected_base = system or "Preserve the character instructions."
    assert expected_base in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert instructions.endswith("World info:\n" + LORE)  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert "LIVE replacement" not in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert payload[-1]["content"] == [{"type": "text", "text": QUESTION}]  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert frozen_chat.db.get_message_by_id(runtime["user_message_id"])["content"] == QUESTION  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", ["client", "key"])
async def test_foreign_native_owner_rejected_before_any_input_append(frozen_chat, foreign):
    """Relaxing the existing owner fence must never expose frozen lore or write."""
    selection = capture_selection(frozen_chat)
    with pytest.raises((NotFoundError, HTTPException)):
        await build_native(frozen_chat, selection=selection,
                           owner_client="2" if foreign == "client" else "1",
                           owner_key="foreign-owner" if foreign == "key" else OWNER_KEY)
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 0  # nosec B101 - pytest behavior assertion, not a runtime security guard


def captured_payload(chat):
    """Read the actual accepted canonical snapshot through the owner boundary."""
    state = chat.db.get_roleplay_resume_state(chat.conversation_id, owner_client_id="1")
    payload = deepcopy(state["behavior_snapshot"]["payload"])
    chat.db.close_connection()
    return payload


def clone_captured_chat(chat, payload):
    """Store a canonical historical snapshot with normal DB writer abstractions.

    Historical V1 snapshots permit opaque book objects. This exercises the
    projector's stricter matching-input guard without bypassing owner/digest
    validation or manually mutating database rows.
    """
    snapshot = build_behavior_snapshot(payload)
    with chat.db.transaction() as conn:
        conversation_id = chat.db.add_conversation({
            "character_id": chat.character_id, "title": "Historical frozen lore",
        }, conn=conn)
        chat.db.put_behavior_snapshot(conversation_id, snapshot, conn=conn)
    chat.db.close_connection()
    copied = vars(chat).copy()
    copied["conversation_id"] = conversation_id
    return SimpleNamespace(**copied)


@pytest.mark.asyncio
async def test_nonmatching_native_lore_keeps_base_and_admits_exact_current_input(frozen_chat):
    text = "Describe a quiet garden."
    instructions, payload, runtime = await build_native(frozen_chat, text=text)
    assert "Preserve the character instructions." in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert "World info:" not in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert payload[-1]["content"] == [{"type": "text", "text": text}]  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert frozen_chat.db.get_message_by_id(runtime["user_message_id"])["content"] == text  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", ["book", "attachment", "entry"])
async def test_disabled_captured_lore_is_absent_without_rejecting_native_send(frozen_chat, disabled):
    if disabled == "book":
        frozen_chat.books.update_world_book(frozen_chat.book_id, enabled=False)
    elif disabled == "attachment":
        frozen_chat.books.attach_to_character(frozen_chat.book_id, frozen_chat.character_id, enabled=False)
    else:
        frozen_chat.books.update_entry(frozen_chat.entry_id, enabled=False)
    frozen_chat.conversation_id = create_character_conversation(
        frozen_chat.db, conversation_data={"character_id": frozen_chat.character_id, "title": "Disabled lore"},
        provider="local-llm", model="local-test",
    )
    frozen_chat.db.close_connection()
    instructions, _payload, _runtime = await build_native(frozen_chat)
    assert "Preserve the character instructions." in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert "World info:" not in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 1  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
async def test_lore_scan_includes_only_the_retained_selected_history(frozen_chat):
    prior = frozen_chat.db.add_message({
        "conversation_id": frozen_chat.conversation_id, "sender": "user", "content": QUESTION,
    })
    reply = frozen_chat.db.add_message({
        "conversation_id": frozen_chat.conversation_id, "sender": "assistant", "content": "A prior reply.",
        "parent_message_id": prior,
    })
    frozen_chat.db.close_connection()
    selection = capture_selection(frozen_chat, after=reply)
    instructions, payload, _runtime = await build_native(frozen_chat, text="Continue the explanation.", selection=selection)
    assert instructions.endswith("World info:\n" + LORE)  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert payload[0]["content"] == [{"type": "text", "text": QUESTION}]  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert payload[-1]["content"] == [{"type": "text", "text": "Continue the explanation."}]  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
async def test_changed_frozen_storage_digest_is_rejected_before_input_append(frozen_chat):
    selection = capture_selection(frozen_chat)
    stored = frozen_chat.db.get_conversation_settings(frozen_chat.conversation_id)
    settings = deepcopy(stored["settings"])
    values = deepcopy(settings["roleplayBehaviorV1"]["values"])
    participants = captured_payload(frozen_chat)["participants"]
    participants[0]["world_books"][0]["entries"][0]["content"] = "Updated owned frozen lore."
    values["participants"] = participants
    settings["roleplayBehaviorV1"] = build_materialized_behavior_settings(values)
    assert frozen_chat.db.upsert_conversation_settings(  # nosec B101 - pytest behavior assertion, not a runtime security guard
        frozen_chat.conversation_id, settings, expected_settings_version=stored["settings_version"],
    )
    frozen_chat.db.close_connection()
    with pytest.raises(HTTPException) as failure:
        await build_native(frozen_chat, selection=selection)
    assert failure.value.status_code == 409  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert failure.value.detail["status"] == "stale_selection"  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 0  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [
    "missing_entries", "book_version", "book_enabled", "entry_owner", "duplicate_entry",
    "entry_content", "entry_keywords", "entry_metadata", "entry_policy", "unknown_entry_field",
])
async def test_canonical_but_malformed_saved_lore_rejected_before_native_admission(frozen_chat, malformed):
    payload = captured_payload(frozen_chat)
    book = payload["participants"][0]["world_books"][0]
    entry = book["entries"][0]
    if malformed == "missing_entries":
        book.pop("entries")
    elif malformed == "book_version":
        book["version"] = 0
    elif malformed == "book_enabled":
        book["enabled"] = "true"
    elif malformed == "entry_owner":
        entry["world_book_id"] = book["id"] + 1
    elif malformed == "duplicate_entry":
        book["entries"].append(deepcopy(entry))
    elif malformed == "entry_content":
        entry["content"] = 42
    elif malformed == "entry_keywords":
        entry["keywords"] = [42]
    elif malformed == "entry_metadata":
        entry["metadata"] = []
    elif malformed == "entry_policy":
        entry["regex_match"] = "false"
    else:
        entry["future_prompt_effect"] = "Must not be silently ignored."
    historical = clone_captured_chat(frozen_chat, payload)
    with pytest.raises(HTTPException) as failure:
        await build_native(historical)
    assert failure.value.status_code == 409  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert failure.value.detail["code"] == "unsupported_history_context_world_books"  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert historical.db.count_messages_for_conversation(historical.conversation_id) == 0  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
async def test_frozen_native_budget_skips_oversized_high_priority_entry_and_keeps_small_entry(frozen_chat):
    payload = captured_payload(frozen_chat)
    book = payload["participants"][0]["world_books"][0]
    prototype = book["entries"][0]
    book["entries"] = [
        {**prototype, "id": 101, "priority": 100, "content": "oversized " * 8000},
        {**prototype, "id": 102, "priority": 50, "content": "Fits the frozen budget."},
    ]
    historical = clone_captured_chat(frozen_chat, payload)
    instructions, _payload, _runtime = await build_native(historical)
    assert instructions.endswith("World info:\nFits the frozen budget.")  # nosec B101 - pytest behavior assertion, not a runtime security guard
    assert "oversized" not in instructions  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
async def test_frozen_native_recursive_policy_matches_saved_chain(frozen_chat):
    payload = captured_payload(frozen_chat)
    book = payload["participants"][0]["world_books"][0]
    book["recursive_scanning"] = True
    prototype = book["entries"][0]
    book["entries"] = [
        {**prototype, "id": 201, "priority": 100, "content": "BRIDGE-913"},
        {**prototype, "id": 202, "priority": 50, "keywords": ["BRIDGE-913"], "content": "Recursive frozen fact."},
    ]
    historical = clone_captured_chat(frozen_chat, payload)
    instructions, _payload, _runtime = await build_native(historical)
    assert instructions.endswith("World info:\nBRIDGE-913\n\nRecursive frozen fact.")  # nosec B101 - pytest behavior assertion, not a runtime security guard


@pytest.mark.asyncio
@pytest.mark.parametrize("live_change", ["edit", "disable", "detach"])
async def test_native_prompt_preview_preserves_frozen_lore_after_live_catalog_change(
    frozen_chat,
    live_change,
):
    """Preview the same saved lore that native dispatch actually assembles."""
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import prompt_assembly_preview
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import CharacterChatCompletionPrepRequest

    if live_change == "edit":
        frozen_chat.books.update_entry(frozen_chat.entry_id, content="LIVE replacement must not reach saved lore.")
    elif live_change == "disable":
        frozen_chat.books.update_world_book(frozen_chat.book_id, enabled=False)
    else:
        frozen_chat.books.detach_from_character(frozen_chat.book_id, frozen_chat.character_id)
    frozen_chat.db.close_connection()

    preview = await prompt_assembly_preview(
        chat_id=frozen_chat.conversation_id,
        body=CharacterChatCompletionPrepRequest(append_user_message=QUESTION, saved_native_lore=True),
        db=frozen_chat.db,
        current_user=SimpleNamespace(id=1),
    )
    instructions, _payload, _runtime = await build_native(frozen_chat)
    expected = "World info:\n" + LORE
    assert instructions.endswith(expected)  # nosec B101 - pytest checks actual native assembly against captured lore.
    section = next(item for item in preview["sections"] if item["name"] == "lorebook")
    assert section["content"] == expected  # nosec B101 - pytest checks preview parity with the same saved native owner.


async def saved_lore_preview(chat, **overrides):
    """Exercise the normal endpoint over the fixture's real owned storage."""
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import prompt_assembly_preview
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import CharacterChatCompletionPrepRequest

    return await prompt_assembly_preview(
        chat_id=chat.conversation_id,
        body=CharacterChatCompletionPrepRequest(
            **{"saved_native_lore": True, "append_user_message": QUESTION, **overrides},
        ),
        db=chat.db,
        current_user=SimpleNamespace(id=1),
    )


@pytest.mark.asyncio
async def test_saved_lore_preview_acknowledges_scope_and_never_appends_input(frozen_chat):
    preview = await saved_lore_preview(frozen_chat)
    assert [item["name"] for item in preview["sections"]] == ["lorebook"]  # nosec B101 - pytest checks positive native-mode acknowledgement.
    assert preview["warnings"] == ["Saved native lore only; other prompt sections are not included."]  # nosec B101 - old servers that ignore the opt-in must not pass.
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 0  # nosec B101 - preview must remain read-only.


@pytest.mark.asyncio
async def test_legacy_default_preview_keeps_its_live_completion_contract(frozen_chat):
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import prompt_assembly_preview
    from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import CharacterChatCompletionPrepRequest

    frozen_chat.books.update_entry(frozen_chat.entry_id, content="Legacy live content.")
    preview = await prompt_assembly_preview(
        chat_id=frozen_chat.conversation_id,
        body=CharacterChatCompletionPrepRequest(append_user_message=QUESTION),
        db=frozen_chat.db,
        current_user=SimpleNamespace(id=1),
    )
    section = next(item for item in preview["sections"] if item["name"] == "lorebook")
    assert section["content"] == "World info:\nLegacy live content."  # nosec B101 - pytest preserves the default legacy API contract.


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["nonmatching", "disabled", "unattached"])
async def test_saved_lore_preview_reports_real_absence_for_saved_scope(frozen_chat, case):
    text = "Describe a quiet garden."
    if case != "nonmatching":
        text = QUESTION
        if case == "disabled":
            frozen_chat.books.update_world_book(frozen_chat.book_id, enabled=False)
        else:
            frozen_chat.books.detach_from_character(frozen_chat.book_id, frozen_chat.character_id)
        frozen_chat.conversation_id = create_character_conversation(
            frozen_chat.db,
            conversation_data={"character_id": frozen_chat.character_id},
            provider="local-llm",
            model="local-test",
        )
    preview = await saved_lore_preview(frozen_chat, append_user_message=text)
    instructions, _payload, _runtime = await build_native(frozen_chat, text=text)
    assert "World info:" not in instructions  # nosec B101 - actual native assembly proves the negative scope.
    assert preview["sections"][0]["content"] == ""  # nosec B101 - preview must not substitute current or unrelated lore.
    assert preview["sections"][0]["diagnostics"] == []  # nosec B101 - explicit empty diagnostics distinguish absence from a failed read.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "controls",
    [
        {"offset": 1},
        {"continue_as_user": True},
        {"force_narrate": True},
        {"impersonate_user": True},
        {"prompt_preset": "custom"},
        {"include_character_context": False},
    ],
)
async def test_saved_lore_preview_refuses_unrepresented_controls(frozen_chat, controls):
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat, **controls)
    assert failure.value.status_code == 409  # nosec B101 - unsupported native controls must not silently fall back to live lore.
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 0  # nosec B101 - rejected preview cannot append input.


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["partial", "branched"])
async def test_saved_lore_preview_refuses_incomplete_or_ambiguous_history(frozen_chat, case):
    root = frozen_chat.db.add_message(
        {"conversation_id": frozen_chat.conversation_id, "sender": "user", "content": QUESTION}
    )
    frozen_chat.db.add_message(
        {
            "conversation_id": frozen_chat.conversation_id,
            "sender": "assistant",
            "content": "Reply.",
            "parent_message_id": root,
        }
    )
    if case == "branched":
        frozen_chat.db.add_message(
            {
                "conversation_id": frozen_chat.conversation_id,
                "sender": "assistant",
                "content": "Other reply.",
                "parent_message_id": root,
            }
        )
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat, **({"limit": 1} if case == "partial" else {}))
    assert failure.value.status_code == 409  # nosec B101 - a partial or ambiguous path is not an exact native lore preview.


@pytest.mark.asyncio
async def test_saved_lore_preview_rechecks_source_fences_after_owner_read(frozen_chat, monkeypatch):
    original = frozen_chat.db.get_conversation_history_selected_content

    def concurrent_append(*args, **kwargs):
        frozen_chat.db.add_message(
            {"conversation_id": frozen_chat.conversation_id, "sender": "user", "content": "Concurrent input."}
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(frozen_chat.db, "get_conversation_history_selected_content", concurrent_append)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 409  # nosec B101 - real storage drift must reject a mixed-source preview.
    assert failure.value.detail["code"] == "stale_source"  # nosec B101 - the existing canonical fence, not a mock result, detects the write.


def assign_preview_dictionary(chat, *, probability=1.0, timed_effects=None):
    """Capture a real owned dictionary using the normal settings materializer."""
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
        materialize_roleplay_behavior_settings,
    )
    from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService

    service = ChatDictionaryService(chat.db)
    dictionary_id = service.create_dictionary("Preview dictionary")
    service.add_entry(
        dictionary_id, key="ALIAS742", content="ORBIT-742", probability=probability, timed_effects=timed_effects
    )
    with chat.db.transaction() as conn:
        state = chat.db.get_roleplay_resume_state(chat.conversation_id, conn=conn, owner_client_id="1")
        settings = deepcopy(state["settings"])
        settings["chat_dictionary_ids"] = [dictionary_id]
        settings["roleplayBehaviorV1"] = materialize_roleplay_behavior_settings(
            conn,
            conversation=state["conversation"],
            resume_state=state,
            merged_settings=settings,
            owner_user_id="1",
            changed_keys={"chat_dictionary_ids"},
        )
        assert chat.db.upsert_conversation_settings(  # nosec B101 - fixture uses the actual versioned settings writer.
            chat.conversation_id,
            settings,
            conn=conn,
            expected_settings_version=state["settings_version"],
        )
    chat.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["current", "historical"])
async def test_saved_lore_preview_transforms_only_current_dictionary_input(frozen_chat, scope):
    assign_preview_dictionary(frozen_chat)
    text = "What is the ALIAS742 access phrase?"
    after = None
    if scope == "historical":
        after = frozen_chat.db.add_message(
            {"conversation_id": frozen_chat.conversation_id, "sender": "user", "content": text}
        )
        text = "Describe a quiet garden."
    preview = await saved_lore_preview(frozen_chat, append_user_message=text)
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == (1 if after else 0)  # nosec B101 - dictionary preview cannot persist transformed input.
    selection = capture_selection(frozen_chat, after=after)
    instructions, _payload, runtime = await build_native(frozen_chat, text=text, selection=selection)
    expected = "World info:\n" + LORE if scope == "current" else ""
    assert preview["sections"][0]["content"] == expected  # nosec B101 - preview and actual native input transformation must agree.
    if expected:
        assert instructions.endswith(expected)  # nosec B101 - current dictionary input must match actual native assembly.
    else:
        assert "World info:" not in instructions  # nosec B101 - historical user text must not be transformed again.
    assert frozen_chat.db.get_message_by_id(runtime["user_message_id"])["content"] == text  # nosec B101 - canonical input retains the authored text.


@pytest.mark.asyncio
@pytest.mark.parametrize("effects", [{"probability": 0.5}, {"timed_effects": {"delay": 1}}])
async def test_saved_lore_preview_refuses_nondeterministic_dictionary_prediction(frozen_chat, effects):
    assign_preview_dictionary(frozen_chat, **effects)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat, append_user_message="ALIAS742")
    assert failure.value.status_code == 409  # nosec B101 - preview must not promise a random or time-dependent future match.


@pytest.mark.asyncio
async def test_saved_lore_preview_matches_native_historical_placeholders(frozen_chat):
    frozen_chat.books.update_entry(frozen_chat.entry_id, keywords=["LoreGuide"])
    frozen_chat.conversation_id = create_character_conversation(
        frozen_chat.db,
        conversation_data={"character_id": frozen_chat.character_id},
        provider="local-llm",
        model="local-test",
    )
    prior = frozen_chat.db.add_message(
        {"conversation_id": frozen_chat.conversation_id, "sender": "user", "content": "Ask {{char}}."}
    )
    text = "Continue."
    preview = await saved_lore_preview(frozen_chat, append_user_message=text)
    instructions, _payload, _runtime = await build_native(
        frozen_chat, text=text, selection=capture_selection(frozen_chat, after=prior)
    )
    assert instructions.endswith("World info:\n" + LORE)  # nosec B101 - actual native historical placeholder replacement triggers captured lore.
    assert preview["sections"][0]["content"] == "World info:\n" + LORE  # nosec B101 - preview must use the same historical replacement, not legacy current-card names.


@pytest.mark.asyncio
async def test_saved_lore_preview_rechecks_locked_scope_after_sync_move(frozen_chat, monkeypatch):
    """The initial global-route read must not authorize later workspace lore."""
    frozen_chat.db.upsert_workspace(
        "preview-scope-workspace", "Preview scope workspace", initial_assistant_opt_out=True
    )
    original = frozen_chat.db.get_conversation_history_snapshot

    def move_before_snapshot(*args, **kwargs):
        current = frozen_chat.db.get_conversation_by_id(frozen_chat.conversation_id)
        assert frozen_chat.db.upsert_conversation_from_sync(  # nosec B101 - real supported Sync writer models the concurrent scope change.
            conversation_id=frozen_chat.conversation_id,
            title=current["title"],
            sync_client_id="1",
            object_revision=current["version"] + 1,
            object_hash="scope-move",
            character_id=frozen_chat.character_id,
            scope_type="workspace",
            workspace_id="preview-scope-workspace",
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(frozen_chat.db, "get_conversation_history_snapshot", move_before_snapshot)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 404  # nosec B101 - global preview cannot return lore from the authoritative workspace binding.


@pytest.mark.asyncio
@pytest.mark.parametrize("ordering", ["backdated", "tied"])
async def test_saved_lore_preview_uses_graph_leaf_despite_timestamp_order(frozen_chat, ordering):
    """Imported message order must follow ancestry, as native selection does."""
    root = frozen_chat.db.add_message(
        {
            "id": "z-preview-root",
            "conversation_id": frozen_chat.conversation_id,
            "sender": "user",
            "content": "Describe a quiet garden.",
            "timestamp": "2026-01-02T00:00:00Z",
        }
    )
    leaf = frozen_chat.db.add_message(
        {
            "id": "a-preview-leaf",
            "conversation_id": frozen_chat.conversation_id,
            "sender": "assistant",
            "content": QUESTION,
            "parent_message_id": root,
            "timestamp": "2026-01-01T00:00:00Z" if ordering == "backdated" else "2026-01-02T00:00:00Z",
        }
    )
    text = "Continue."
    preview = await saved_lore_preview(frozen_chat, append_user_message=text)
    instructions, _payload, _runtime = await build_native(
        frozen_chat,
        text=text,
        selection=capture_selection(frozen_chat, after=leaf),
    )
    assert instructions.endswith("World info:\n" + LORE)  # nosec B101 - native follows the explicit canonical graph leaf.
    assert preview["sections"][0]["content"] == "World info:\n" + LORE  # nosec B101 - timestamps cannot change complete-path lore parity.


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "tool"])
async def test_saved_lore_preview_refuses_media_and_tools(frozen_chat, kind):
    import base64

    data = {"conversation_id": frozen_chat.conversation_id, "sender": "assistant", "content": QUESTION}
    if kind == "image":
        data.update(
            image_data=base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jG4sAAAAASUVORK5CYII="
            ),
            image_mime_type="image/png",
        )
    mid = frozen_chat.db.add_message(data)
    if kind == "tool":
        frozen_chat.db.add_message_metadata(
            mid, tool_calls=[{"id": "tool1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        )
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 409  # nosec B101 - a text-only native preview cannot certify media/tool context.
    assert failure.value.detail["code"] == "unsupported_history_context_preview_media_or_tools"  # nosec B101 - unsupported content must remain an explicit refusal.


@pytest.mark.asyncio
async def test_saved_lore_preview_refuses_unsettled_manifest(frozen_chat, monkeypatch):
    """A controlled unsettled canonical manifest must retain the ready-selection guard."""
    from dataclasses import replace

    frozen_chat.db.add_message({"conversation_id": frozen_chat.conversation_id, "sender": "user", "content": QUESTION})
    original = frozen_chat.db.get_conversation_history_snapshot

    def unsettled(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        return replace(snapshot, nodes=tuple({**row, "settled": False} for row in snapshot.nodes))

    monkeypatch.setattr(frozen_chat.db, "get_conversation_history_snapshot", unsettled)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 409  # nosec B101 - unsettled history cannot be previewed as a complete future turn.
    assert failure.value.detail["code"] == "unsettled_message"  # nosec B101 - retain the shared ready-selection refusal.


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["missing", "malformed"])
async def test_saved_lore_preview_refuses_unusable_saved_context(frozen_chat, kind):
    if kind == "missing":
        frozen_chat.conversation_id = frozen_chat.db.add_conversation({"character_id": frozen_chat.character_id})
    else:
        payload = captured_payload(frozen_chat)
        payload["participants"][0]["world_books"][0].pop("entries")
        frozen_chat = clone_captured_chat(frozen_chat, payload)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 409  # nosec B101 - missing/invalid saved authority must never become successful empty lore.


@pytest.mark.asyncio
async def test_saved_lore_preview_preserves_storage_read_failure(frozen_chat, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    def unavailable(*args, **kwargs):
        raise CharactersRAGDBError("Controlled unavailable selected-content read")

    monkeypatch.setattr(frozen_chat.db, "get_conversation_history_selected_content", unavailable)
    with pytest.raises(HTTPException) as failure:
        await saved_lore_preview(frozen_chat)
    assert failure.value.status_code == 500  # nosec B101 - a failed read cannot masquerade as a successful empty context.
    assert failure.value.detail == "An unexpected error occurred while generating prompt preview"  # nosec B101 - response must retain the existing generic error contract.


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen_chat", ["sqlite", "postgres"], indirect=True)
@pytest.mark.parametrize("mode", ["saved", "foreign", "string", "integer", "null"])
async def test_saved_lore_preview_http_boundary(frozen_chat, mode, monkeypatch):
    import httpx
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/chats")
    monkeypatch.setitem(app.dependency_overrides, get_chacha_db_for_user, lambda: frozen_chat.db)
    monkeypatch.setitem(app.dependency_overrides, get_request_user, lambda: SimpleNamespace(id=2 if mode == "foreign" else 1))
    value = {"string": "true", "integer": 1, "null": None}.get(mode, True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/chats/{frozen_chat.conversation_id}/prompt-preview",
            json={"saved_native_lore": value, "append_user_message": QUESTION},
        )
    expected = 200 if mode == "saved" else 403 if mode == "foreign" else 422
    assert response.status_code == expected, response.text  # nosec B101 - real route parsing/owner guard must reject coercion and foreign access.
    if mode == "saved":
        assert response.json()["warnings"] == ["Saved native lore only; other prompt sections are not included."]  # nosec B101 - positive acknowledgement identifies the opt-in contract through response serialization.
        assert response.json()["sections"][0]["content"] == "World info:\n" + LORE  # nosec B101 - actual owner storage is used on both supported engines.
    assert frozen_chat.db.count_messages_for_conversation(frozen_chat.conversation_id) == 0  # nosec B101 - successful and rejected API previews stay read-only.


@pytest.mark.asyncio
@pytest.mark.parametrize("frozen_chat", ["sqlite", "postgres"], indirect=True)
async def test_world_book_catalog_db_read_failure_never_becomes_empty(frozen_chat, monkeypatch):
    """A controlled DB read failure must remain HTTP500 and recover without data loss."""
    import httpx
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/characters")
    monkeypatch.setitem(app.dependency_overrides, get_chacha_db_for_user, lambda: frozen_chat.db)
    monkeypatch.setitem(app.dependency_overrides, get_request_user, lambda: SimpleNamespace(id=1))
    original = frozen_chat.db.execute_query
    failures = []

    def unavailable(query, *args, **kwargs):
        if "FROM world_books " in query and kwargs.get("read_only") is True:
            failures.append(True)
            raise CharactersRAGDBError("Controlled catalog database read unavailable")
        return original(query, *args, **kwargs)

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with monkeypatch.context() as fault:
                fault.setattr(frozen_chat.db, "execute_query", unavailable)
                failed = await client.get("/api/v1/characters/world-books")
            assert failures == [True]  # nosec B101 - The actual catalog database-read boundary was exercised.
            assert failed.status_code == 500  # nosec B101 - A DB failure cannot become a successful empty catalog.
            assert "world_books" not in failed.json()  # nosec B101 - Never return an empty catalog payload for a failed read.
            recovered = await client.get("/api/v1/characters/world-books")
            assert recovered.status_code == 200  # nosec B101 - The same real store recovers after the fault is removed.
            payload = recovered.json()
            assert payload["total"] == 1  # nosec B101 - Failed reads must not mutate persisted catalog contents.
            assert payload["world_books"][0]["id"] == frozen_chat.book_id  # nosec B101 - The original owner-bound book is retained.
            assert payload["world_books"][0]["entry_count"] == 1  # nosec B101 - The original enabled entry remains counted.
    finally:
        app.dependency_overrides.clear()
        frozen_chat.db.close_all_connections()
