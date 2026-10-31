"""Successful regeneration preserves canonical turn identity on both engines."""

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages
from tldw_Server_API.app.core.Chat.persistence_service import build_assistant_message_payload
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

QUESTION = "Describe this literal {{char}} request."
with io.BytesIO() as image_buffer:
    Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
    PNG = base64.b64encode(image_buffer.getvalue()).decode("ascii")


@pytest.fixture(params=["sqlite", "postgres"])
def saved_chat(request, tmp_path):
    """Use the official PostgreSQL fixture, with the same real DB API as SQLite."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "regeneration.db", client_id="1", **({"backend": backend} if backend else {}))
    conversation = db.add_conversation({"title": "Saved regeneration"})
    try:
        yield SimpleNamespace(db=db, conversation=conversation)
    finally:
        db.close_all_connections()
        if backend:
            backend.get_pool().close_all()


def seed_answer(chat, *, image=False, parent=True):
    """Seed a real legacy or linked reply, optionally with a literal attachment."""
    user = chat.db.add_message({
        "conversation_id": chat.conversation, "sender": "user", "content": QUESTION,
        "timestamp": "2026-01-01T00:00:00+00:00",
        **({"images": [{"data": base64.b64decode(PNG), "mime": "image/png"}]} if image else {}),
    })
    reply = chat.db.add_message({
        "conversation_id": chat.conversation, "sender": "assistant", "content": "Original answer",
        "timestamp": "2026-01-01T00:00:01+00:00", **({"parent_message_id": user} if parent else {}),
    })
    return user, reply


def complete(chat, reply=None, *, image=False, text=QUESTION, order="asc", limit=100):
    """Exercise real context preparation and assistant persistence, without an LLM."""
    content = [{"type": "text", "text": text}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}})
    request = ChatCompletionRequest(
        model="unused", conversation_id=chat.conversation, save_to_db=True,
        messages=[{"role": "user", "content": content}], history_message_order=order,
        metadata={"tldw_regenerate_from_message_id": reply} if reply else None,
    ).model_copy(update={"history_message_limit": limit})

    async def run():
        with chacha_operation(independent=True):
            state = {}
            result = await build_context_and_messages(
                chat_db=chat.db, request_data=request, loop=asyncio.get_running_loop(),
                metrics=get_chat_metrics(), default_save_to_db=True,
                final_conversation_id=chat.conversation,
                save_message_fn=_save_message_turn_to_db, runtime_state=state,
            )
            payload = build_assistant_message_payload(
                character_card_for_context=None,
                assistant_parent_message_id=state.get("assistant_parent_message_id"),
                content="Regenerated answer", tool_calls=None, function_call=None,
            )
            saved = await _save_message_turn_to_db(chat.db, chat.conversation, payload, use_transaction=True)
            return result[4], state, saved

    return asyncio.run(run())


@pytest.mark.parametrize("image,parent,order,limit", [
    (False, True, "asc", 100), (True, False, "desc", 100), (True, True, "asc", 0),
])
def test_regeneration_reuses_saved_user_and_links_both_answers(saved_chat, image, parent, order, limit):
    user, reply = seed_answer(saved_chat, image=image, parent=parent)
    payload, state, regenerated = complete(saved_chat, reply, image=image, order=order, limit=limit)
    rows = saved_chat.db.get_messages_for_conversation(saved_chat.conversation, 100, 0, "ASC", strict_images=True)
    assert [row["id"] for row in rows if row["sender"] == "user"] == [user]
    assert [(row["id"], row["parent_message_id"]) for row in rows if row["sender"] == "assistant"] == [
        (reply, user), (regenerated, user),
    ]
    assert state["user_message_id"] == user
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"][0] == {"type": "text", "text": QUESTION}
    assert [part["image_url"]["url"] for part in payload[-1]["content"] if part["type"] == "image_url"] == (
        [f"data:image/png;base64,{PNG}"] if image else []
    )
    assert not any("Original answer" in str(message) for message in payload)
    assert sum(message["role"] == "user" for message in payload) == 1
    # An older selected variant remains a valid regeneration target for this turn.
    complete(saved_chat, reply, image=image)
    assert len(saved_chat.db.get_messages_for_conversation(saved_chat.conversation)) == 4


def test_identical_ordinary_send_still_creates_new_user(saved_chat):
    user, _ = seed_answer(saved_chat)
    _, state, reply = complete(saved_chat)
    rows = saved_chat.db.get_messages_for_conversation(saved_chat.conversation)
    assert len([row for row in rows if row["sender"] == "user"]) == 2
    assert state["user_message_id"] != user
    assert saved_chat.db.get_message_by_id(reply)["parent_message_id"] == state["user_message_id"]


@pytest.mark.parametrize("invalid", ["changed-text", "missing-image", "foreign-reply", "newer-turn"])
def test_regeneration_rejects_changed_or_unrelated_turn_without_writes(saved_chat, invalid):
    _, reply = seed_answer(saved_chat, image=True)
    if invalid == "foreign-reply":
        other = saved_chat.db.add_conversation({"title": "Other conversation"})
        reply = saved_chat.db.add_message({"conversation_id": other, "sender": "assistant", "content": "Unrelated"})
    elif invalid == "newer-turn":
        complete(saved_chat, image=True, text="An intentional new question")
    before = saved_chat.db.get_messages_for_conversation(saved_chat.conversation, 100, 0, "ASC")
    with pytest.raises(HTTPException) as rejected:
        complete(saved_chat, reply, image=invalid != "missing-image", text="Changed" if invalid == "changed-text" else QUESTION)
    assert rejected.value.status_code in {404, 409}
    assert saved_chat.db.get_messages_for_conversation(saved_chat.conversation, 100, 0, "ASC") == before


def test_regeneration_fails_closed_when_saved_images_cannot_be_read(saved_chat, monkeypatch):
    _, reply = seed_answer(saved_chat, image=True)
    original_query = saved_chat.db.execute_query

    def unavailable_images(query, *args, **kwargs):
        if "FROM message_images WHERE" in query:
            raise CharactersRAGDBError("Attachment storage unavailable")
        return original_query(query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(saved_chat.db, "execute_query", unavailable_images)
        with pytest.raises(HTTPException) as rejected:
            complete(saved_chat, reply, image=True)
        assert rejected.value.status_code == 409
    assert len(saved_chat.db.get_messages_for_conversation(saved_chat.conversation)) == 2


@pytest.mark.parametrize("order,limit", [("asc", 100), ("desc", 100), ("desc", 1)])
def test_regeneration_retains_prior_turns_but_omits_answer_variants(saved_chat, order, limit):
    first_user, first_reply = seed_answer(saved_chat)
    _, state, latest_reply = complete(saved_chat, text="An intentional new question")
    payload, regenerated_state, _ = complete(saved_chat, latest_reply, text="An intentional new question", order=order, limit=limit)
    assert regenerated_state["user_message_id"] == state["user_message_id"]
    assert payload[-1]["role"] == "user"
    assert "An intentional new question" in str(payload[-1]["content"])
    assert not any("Regenerated answer" in str(message) for message in payload)
    if limit == 100:
        assert len(payload) == 3
        assert any("Original answer" in str(message) for message in payload[:-1])
    else:
        assert len(payload) == 1
    rows = saved_chat.db.get_messages_for_conversation(saved_chat.conversation)
    assert [row["id"] for row in rows if row["sender"] == "user"] == [first_user, state["user_message_id"]]
    assert saved_chat.db.get_message_by_id(first_reply)["content"] == "Original answer"


@pytest.mark.parametrize("case", ["missing-conversation", "malformed-marker", "changed-character"])
def test_regeneration_rejection_never_creates_an_empty_conversation(saved_chat, case):
    _, reply = seed_answer(saved_chat)
    character = saved_chat.db.add_character_card({"name": "Different assistant"})
    conversation = saved_chat.conversation if case == "changed-character" else None
    request = ChatCompletionRequest(
        model="unused", conversation_id=conversation, character_id=str(character), save_to_db=True,
        messages=[{"role": "user", "content": QUESTION}],
        metadata={"tldw_regenerate_from_message_id": "invalid id!" if case == "malformed-marker" else reply},
    )
    before = {row["id"] for row in saved_chat.db.get_conversations_for_user("1")}

    async def run():
        with chacha_operation(independent=True):
            with pytest.raises(HTTPException) as rejected:
                await build_context_and_messages(
                    chat_db=saved_chat.db, request_data=request, loop=asyncio.get_running_loop(),
                    metrics=get_chat_metrics(), default_save_to_db=True,
                    final_conversation_id=conversation, save_message_fn=_save_message_turn_to_db,
                )
            assert rejected.value.status_code in {409, 422}

    asyncio.run(run())
    assert {row["id"] for row in saved_chat.db.get_conversations_for_user("1")} == before


def test_regeneration_rejects_extra_history_before_any_write(saved_chat):
    seed_answer(saved_chat)
    _, _, reply = complete(saved_chat, text="Second question")
    request = ChatCompletionRequest(
        model="unused", conversation_id=saved_chat.conversation, save_to_db=True,
        history_message_order="asc", history_message_limit=1,
        messages=[{"role": "user", "content": QUESTION},
                  {"role": "assistant", "content": "Original answer"},
                  {"role": "user", "content": "Second question"}],
        metadata={"tldw_regenerate_from_message_id": reply},
    )
    before = saved_chat.db.get_messages_for_conversation(saved_chat.conversation)

    async def run():
        with chacha_operation(independent=True):
            with pytest.raises(HTTPException) as rejected:
                await build_context_and_messages(
                    chat_db=saved_chat.db, request_data=request, loop=asyncio.get_running_loop(),
                    metrics=get_chat_metrics(), default_save_to_db=True,
                    final_conversation_id=saved_chat.conversation, save_message_fn=_save_message_turn_to_db,
                )
            assert rejected.value.status_code == 409

    asyncio.run(run())
    assert saved_chat.db.get_messages_for_conversation(saved_chat.conversation) == before
