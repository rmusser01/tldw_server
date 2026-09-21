"""Failed-turn Retry preserves canonical instruction blocks on both backends."""

from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def retry_chat(request, tmp_path):
    """Use the official temporary PostgreSQL database or an isolated SQLite DB."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "retry.db", client_id="1", backend=backend)
    db.add_character_card({"name": chat_service.DEFAULT_CHARACTER_NAME})
    conversation_id = db.add_conversation({"client_id": "1", "title": "Retry instructions"})
    try:
        yield SimpleNamespace(db=db, conversation_id=conversation_id)
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


async def prepare(chat, instructions, *, retry=False, history_limit=20, history_order="asc", **overrides):
    """Exercise production preparation and persistence without calling a provider."""
    body = {
        "model": "unused-provider",
        "conversation_id": chat.conversation_id,
        "save_to_db": True,
        "history_message_order": history_order,
        "messages": [{"role": "system", "content": text} for text in instructions]
        + [{"role": "user", "content": "Original question"}],
        "metadata": {"tldw_client_message_id": "original-client", "tldw_retry_failed_turn": retry},
        **overrides,
    }
    request = ChatCompletionRequest(**body).model_copy(update={"history_message_limit": history_limit})
    state = {}
    loop = asyncio.get_running_loop()
    try:
        with chacha_operation(independent=True):
            result = await chat_service.build_context_and_messages(
                chat_db=chat.db,
                request_data=request,
                loop=loop,
                metrics=get_chat_metrics(),
                default_save_to_db=True,
                final_conversation_id=chat.conversation_id,
                save_message_fn=_save_message_turn_to_db,
                runtime_state=state,
            )
        system, payload = chat_service.apply_prompt_templating(request, result[0], result[4])
        return state["user_message_id"], system, payload
    finally:
        await loop.run_in_executor(None, chat.db.close_connection)


def saved_rows(chat):
    """Read canonical rows after completed operations, releasing the read handle."""
    try:
        return chat.db.get_messages_for_conversation(chat.conversation_id)
    finally:
        chat.db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("history_limit,history_order", [(0, "asc"), (1, "asc"), (20, "asc"), (20, "desc")])
async def test_unchanged_system_retry_preserves_rows_and_stays_retryable(retry_chat, history_limit, history_order):
    """Re-inserting the unchanged system row breaks both history and the next Retry."""
    user_id, _, _ = await prepare(retry_chat, ["Pirate", "Say ARRR"])
    original = saved_rows(retry_chat)
    for _ in range(2):
        reused_id, system, payload = await prepare(
            retry_chat, ["Pirate", "Say ARRR"], retry=True,
            history_limit=history_limit, history_order=history_order,
        )
        assert saved_rows(retry_chat) == original
        assert reused_id == user_id
        assert system == "Pirate\n\nSay ARRR"
        assert [message["role"] for message in payload].count("user") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [["Plain", "Be concise"], ["Say ARRR", "Pirate"], ["Pirate"]])
async def test_changed_ordered_instruction_block_is_saved_once_and_can_return_to_original(retry_chat, changed):
    """A repeated update is reused, but A to B to A still records both intentional changes."""
    user_id, _, _ = await prepare(retry_chat, ["Pirate", "Say ARRR"])
    for block in [changed, changed, ["Pirate", "Say ARRR"], ["Pirate", "Say ARRR"]]:
        reused_id, system, _ = await prepare(retry_chat, block, retry=True, history_limit=0)
        assert reused_id == user_id
        assert system == "\n\n".join(block)
    rows = saved_rows(retry_chat)
    assert [row["content"] for row in rows if row["sender"] == "system"] == [
        "Pirate", "Say ARRR", *changed, "Pirate", "Say ARRR",
    ]
    assert [row["id"] for row in rows if row["sender"] == "user"] == [user_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["answered", "text", "client"])
async def test_system_updates_do_not_hide_answered_or_mismatched_retry(retry_chat, mismatch):
    """An intervening system row must not bypass Retry's existing conflict checks."""
    user_id, _, _ = await prepare(retry_chat, ["Pirate"])
    await prepare(retry_chat, ["Plain"], retry=True)
    if mismatch == "answered":
        retry_chat.db.add_message({
            "conversation_id": retry_chat.conversation_id, "sender": "assistant",
            "content": "Saved answer", "parent_message_id": user_id,
        })
    before = saved_rows(retry_chat)
    overrides = {}
    if mismatch == "text":
        overrides["messages"] = [{"role": "system", "content": "New"}, {"role": "user", "content": "Changed"}]
    elif mismatch == "client":
        overrides["metadata"] = {"tldw_client_message_id": "different", "tldw_retry_failed_turn": True}
    with pytest.raises(HTTPException) as error:
        await prepare(retry_chat, ["New"], retry=True, **overrides)
    assert error.value.status_code == 409
    assert saved_rows(retry_chat) == before


@pytest.mark.asyncio
async def test_legacy_instruction_block_is_reused_without_backfilling_rows(retry_chat):
    """Existing unmarked system rows remain valid retry context."""
    for sender, content in [("system", "Pirate"), ("system", "Say ARRR"), ("user", "Original question")]:
        retry_chat.db.add_message({"conversation_id": retry_chat.conversation_id, "sender": sender, "content": content})
    before = saved_rows(retry_chat)
    await prepare(retry_chat, ["Pirate", "Say ARRR"], retry=True, history_limit=0)
    assert saved_rows(retry_chat) == before


@pytest.mark.asyncio
async def test_changed_retry_instruction_survives_overlapping_client_history(retry_chat):
    """Trimming a matching user suffix must not discard the changed system prefix."""
    for sender, content in [("user", "Earlier question"), ("assistant", "Earlier answer")]:
        retry_chat.db.add_message({"conversation_id": retry_chat.conversation_id, "sender": sender, "content": content})
    user_id, _, _ = await prepare(retry_chat, ["Pirate"])
    reused_id, system, _ = await prepare(retry_chat, ["Plain"], retry=True, messages=[
        {"role": "system", "content": "Plain"},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Original question"},
    ])
    assert reused_id == user_id
    assert system == "Plain"
    assert [row["content"] for row in saved_rows(retry_chat) if row["sender"] == "system"] == ["Pirate", "Plain"]


def test_retry_context_does_not_expose_another_owners_instructions(retry_chat):
    """The new canonical lookup keeps conversation ownership on both backends."""
    foreign = retry_chat.db.add_conversation({"client_id": "2", "title": "Private instructions"})
    retry_chat.db.add_message({"conversation_id": foreign, "sender": "system", "content": "Private prompt"})
    assert retry_chat.db.get_retry_message_context(foreign) == ([], [])


@pytest.mark.asyncio
async def test_image_retry_keeps_original_attachment_and_saves_one_successful_answer(retry_chat):
    """The same canonical prompt/image turn survives repeated preparation on real DBs."""
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    image_bytes = output.getvalue()
    url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    messages = [
        {"role": "system", "content": "Count the dots."},
        {"role": "user", "content": [
            {"type": "text", "text": "Original question"},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    user_id, _, _ = await prepare(retry_chat, [], messages=messages)
    original = saved_rows(retry_chat)
    for _ in range(2):
        reused_id, system, _ = await prepare(retry_chat, [], retry=True, messages=messages)
        assert reused_id == user_id
        assert system == "Count the dots."
        assert saved_rows(retry_chat) == original
    with chacha_operation(independent=True):
        await _save_message_turn_to_db(retry_chat.db, retry_chat.conversation_id, {
            "role": "assistant", "content": "Three dots", "parent_message_id": user_id,
        }, use_transaction=True)
    rows = saved_rows(retry_chat)
    assert [row["sender"] for row in rows] == ["system", "user", "assistant"]
    assert rows[1]["id"] == user_id
    assert rows[1]["images"][0]["image_data"] == image_bytes
    assert rows[2]["parent_message_id"] == user_id
