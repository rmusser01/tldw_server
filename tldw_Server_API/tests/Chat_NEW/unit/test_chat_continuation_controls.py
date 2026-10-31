import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return ""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_continuation_branch_uses_anchor_chain(populated_chacha_db) -> None:
    char = populated_chacha_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
    assert char
    conv_id = populated_chacha_db.add_conversation(
        {"character_id": char["id"], "title": "Continuation Branch Conversation"}
    )

    root_id = populated_chacha_db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "root-msg"}
    )
    anchor_id = populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "anchor-msg",
            "parent_message_id": root_id,
        }
    )
    populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "tip-msg",
            "parent_message_id": anchor_id,
        }
    )
    populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "sibling-msg",
            "parent_message_id": root_id,
        }
    )

    request_data = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        conversation_id=conv_id,
        history_message_limit=100,
        history_message_order="asc",
        save_to_db=False,
        messages=[{"role": "user", "content": "continue-from-anchor"}],
        tldw_continuation={
            "from_message_id": anchor_id,
            "mode": "branch",
        },
    )

    loop = asyncio.get_running_loop()
    metrics = get_chat_metrics()
    _, _, _, _, llm_payload_messages, _ = await build_context_and_messages(
        chat_db=populated_chacha_db,
        request_data=request_data,
        loop=loop,
        metrics=metrics,
        default_save_to_db=False,
        final_conversation_id=conv_id,
        save_message_fn=_save_message_turn_to_db,
    )

    text_messages = [_message_text(msg) for msg in llm_payload_messages]
    assert "root-msg" in text_messages
    assert "anchor-msg" in text_messages
    assert "tip-msg" not in text_messages
    assert "sibling-msg" not in text_messages


@pytest.mark.asyncio
@pytest.mark.unit
async def test_continuation_append_requires_tip(populated_chacha_db) -> None:
    char = populated_chacha_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
    assert char
    conv_id = populated_chacha_db.add_conversation(
        {"character_id": char["id"], "title": "Continuation Append Conversation"}
    )

    root_id = populated_chacha_db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "append-root"}
    )
    anchor_id = populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "append-anchor",
            "parent_message_id": root_id,
        }
    )
    populated_chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "append-latest",
            "parent_message_id": root_id,
        }
    )

    request_data = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        conversation_id=conv_id,
        history_message_limit=100,
        history_message_order="asc",
        save_to_db=False,
        messages=[{"role": "user", "content": "append-request"}],
        tldw_continuation={
            "from_message_id": anchor_id,
            "mode": "append",
        },
    )

    loop = asyncio.get_running_loop()
    metrics = get_chat_metrics()
    with pytest.raises(HTTPException) as exc_info:
        await build_context_and_messages(
            chat_db=populated_chacha_db,
            request_data=request_data,
            loop=loop,
            metrics=metrics,
            default_save_to_db=False,
            final_conversation_id=conv_id,
            save_message_fn=_save_message_turn_to_db,
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_versioned_selected_history_keeps_images_tools_and_current_identity(populated_chacha_db, monkeypatch):
    """Selected content replaces timestamp reload without losing normal composer fields."""
    from unittest.mock import AsyncMock, MagicMock

    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection, snapshot_to_wire
    db = populated_chacha_db
    cid = db.add_conversation({"character_id": None, "title": "Bound selected content"})
    first = db.add_message({"conversation_id": cid, "sender": "user", "content": "same",
        "images": [{"data": b"first", "mime": "image/png"}, {"data": b"second", "mime": "image/png"}]})
    second = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "[tool_calls]", "parent_message_id": first})
    db.add_message_metadata(second, tool_calls=[{"id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
                            extra={"sender_role": "assistant", "content_placeholder_reason": "tool_calls"})
    third = db.add_message({"conversation_id": cid, "sender": "tool", "content": "evidence", "parent_message_id": second})
    db.add_message_metadata(third, extra={"sender_role": "tool", "tool_call_id": "call", "tool_name": "lookup"})
    snap = snapshot_to_wire(db.get_conversation_history_snapshot(cid, owner_client_id=db.client_id, owner_key="owner"))
    selection = resolve_history_selection(snap, {"owner_key": "owner", "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": third},
        "selection_revision": 1}, "send", "prepared")["selection"]
    request = ChatCompletionRequest(model="gpt-4o-mini", conversation_id=cid, save_to_db=True,
        messages=[{"role": "user", "content": "same"}], tldw_history_selection_v1=selection)
    runtime = {"history_owner_key": "owner", "history_owner_client_id": db.client_id,
               "history_inputs": [{"sender": "user", "content": "same"}]}
    def timestamp_reload(*args, **kwargs):
        raise AssertionError("versioned history must not use timestamp loader")
    monkeypatch.setattr(db, "get_messages_for_conversation", timestamp_reload)
    save = AsyncMock()
    result = await build_context_and_messages(db, request, asyncio.get_running_loop(), MagicMock(), True, cid, save, runtime)
    payload = result[4]
    assert [part["image_url"]["url"] for part in payload[0]["content"] if part["type"] == "image_url"] == [
        "data:image/png;base64,Zmlyc3Q=", "data:image/png;base64,c2Vjb25k"]
    assert payload[1]["content"] is None and payload[1]["tool_calls"][0]["id"] == "call"
    assert payload[2]["tool_call_id"] == "call" and payload[2]["name"] == "lookup"
    assert payload[3]["content"] == "same"
    accepted = db.get_message_by_id(runtime["assistant_parent_message_id"])
    assert accepted["id"] != first and accepted["parent_message_id"] == third
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_versioned_saved_character_uses_snapshot_after_live_card_edit(populated_chacha_db, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
    from tldw_Server_API.app.core.Chat.chat_service import apply_prompt_templating
    from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection, snapshot_to_wire
    db = populated_chacha_db
    with db.transaction() as conn:
        conn.execute("UPDATE character_cards SET system_prompt = ?, description = ? WHERE id = 1", ("Saved {{char}} instruction.", "A saved description."))
    cid = create_character_conversation(db, conversation_data={"character_id": 1, "title": "Saved behavior"},
        prompt_preset_id="st_default", provider="openai", model="gpt-4o-mini", sampling={"temperature": 0.23})
    with db.transaction() as conn:
        conn.execute("UPDATE character_cards SET system_prompt = 'live override', deleted = TRUE WHERE id = 1")
    snap = snapshot_to_wire(db.get_conversation_history_snapshot(cid, owner_client_id=db.client_id, owner_key="owner"))
    selection = resolve_history_selection(snap, {"owner_key": "owner", "conversation_id": cid,
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "empty"}, "selection_revision": 1}, "send", "saved")["selection"]
    request = ChatCompletionRequest(model="gpt-4o-mini", conversation_id=cid, character_id="1", save_to_db=True,
        messages=[{"role": "user", "content": "hello"}], tldw_history_selection_v1=selection)
    def no_live_lookup(*args, **kwargs):
        raise AssertionError("live character/profile source cannot supply saved history context")
    for name in ("get_character_card_by_id", "get_character_card_by_name", "get_persona_profile"):
        monkeypatch.setattr(db, name, no_live_lookup)
    runtime = {"history_owner_key": "owner", "history_owner_client_id": db.client_id,
               "history_inputs": [{"sender": "user", "content": "hello"}]}
    card, _, returned_cid, _, messages, persist = await build_context_and_messages(
        db, request, asyncio.get_running_loop(), MagicMock(), True, cid, AsyncMock(), runtime)
    system, _ = apply_prompt_templating(request_data=request, character_card=card, llm_payload_messages=messages)
    assert "Saved " in system and " instruction." in system and "A saved description." in system
    assert request.temperature == 0.23
    assert "live override" not in system
    assert returned_cid == cid and persist
    assert db.count_messages_for_conversation(cid) == 1
