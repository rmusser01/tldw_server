"""Queued resume trims proven history without losing the new or interrupted turn."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

PREFIX = [
    {"role": "system", "content": "Owned instructions"},
    {"role": "assistant", "content": "Greeting"},
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
]
INTERRUPTED = {"role": "user", "content": "Interrupted question"}


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def queued_chat(request, tmp_path):
    """Use real adapters and the official per-test PostgreSQL fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "queued.db", client_id="1", backend=backend)
    db.add_character_card({"name": chat_service.DEFAULT_CHARACTER_NAME})
    conversation_id = db.add_conversation({"client_id": "1", "title": "Queued resume"})
    for message in [*PREFIX, INTERRUPTED]:
        db.add_message({"conversation_id": conversation_id, "sender": message["role"], "content": message["content"]})
    try:
        yield SimpleNamespace(db=db, conversation_id=conversation_id)
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def rows(chat):
    """Release canonical read transactions before another worker saves."""
    try:
        return chat.db.get_messages_for_conversation(chat.conversation_id)
    finally:
        chat.db.close_connection()


async def resume(chat, messages, *, limit=20, order="asc"):
    """Call production request preparation and persistence without an LLM."""
    request = ChatCompletionRequest(
        model="unused-provider", conversation_id=chat.conversation_id, save_to_db=True,
        messages=messages, history_message_order=order,
        metadata={"tldw_client_message_id": "queued-original-client"},
    ).model_copy(update={"history_message_limit": limit})
    state = {}
    loop = asyncio.get_running_loop()
    try:
        with chacha_operation(independent=True):
            prepared = await chat_service.build_context_and_messages(
                chat_db=chat.db, request_data=request, loop=loop, metrics=get_chat_metrics(),
                default_save_to_db=True, final_conversation_id=chat.conversation_id,
                save_message_fn=_save_message_turn_to_db, runtime_state=state,
            )
        return state, prepared[4]
    finally:
        await loop.run_in_executor(None, chat.db.close_connection)


def text_content(message):
    """Normalize persisted text parts only for assertions on the provider payload."""
    content = message.get("content")
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content or [])


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("prefix_start", [0, 2])
async def test_stale_prefix_preserves_canonical_gap_and_new_client_identity(queued_chat, order, prefix_start):
    """Native Character cancellation leaves client history behind the canonical user."""
    before = rows(queued_chat)
    state, payload = await resume(
        queued_chat, [*PREFIX[prefix_start:], {"role": "user", "content": "Queued question"}], order=order,
    )
    saved = rows(queued_chat)
    assert saved[:-1] == before
    assert saved[-1]["content"] == "Queued question"
    assert saved[-1]["id"] == state["user_message_id"]
    metadata = queued_chat.db.get_message_metadata(saved[-1]["id"])
    assert metadata["extra"]["client_message_id"] == "queued-original-client"
    # The gap is already in authoritative model context; the client need not replay it.
    assert [text_content(message) for message in payload].count("Interrupted question") == 1
    assert [text_content(message) for message in payload].count("First answer") == 1
    assert [text_content(message) for message in payload].count("Queued question") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,order,prefix_start,expected_gap", [(3, "desc", 2, 1), (4, "asc", 0, 0)])
async def test_proven_overlap_respects_the_configured_history_window(queued_chat, limit, order, prefix_start, expected_gap):
    before = rows(queued_chat)
    _, payload = await resume(
        queued_chat, [*PREFIX[prefix_start:], {"role": "user", "content": "Queued question"}],
        limit=limit, order=order,
    )
    assert rows(queued_chat)[:-1] == before
    assert [text_content(message) for message in payload].count("First answer") == 1
    assert [text_content(message) for message in payload].count("Interrupted question") == expected_gap


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Interrupted question", "First question"])
async def test_deliberately_repeated_final_user_is_a_new_turn(queued_chat, text):
    before = rows(queued_chat)
    state, _ = await resume(queued_chat, [*PREFIX, {"role": "user", "content": text}])
    saved = rows(queued_chat)
    assert saved[:-1] == before
    assert saved[-1]["content"] == text
    assert saved[-1]["id"] == state["user_message_id"]
    assert saved[-1]["id"] not in {row["id"] for row in before}


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,order", [(0, "asc"), (1, "asc"), (1, "desc")])
async def test_insufficient_history_window_does_not_guess_a_match(queued_chat, limit, order):
    before = rows(queued_chat)
    await resume(queued_chat, [*PREFIX, {"role": "user", "content": "Queued question"}], limit=limit, order=order)
    saved = rows(queued_chat)
    assert saved[:len(before)] == before
    # The existing suffix rule can trim the first system row in an ASC window.
    # A new historical-prefix rule must not infer the missing assistant rows.
    prefix = PREFIX[1:] if limit == 1 and order == "asc" else PREFIX
    assert [row["content"] for row in saved[len(before):]] == [message["content"] for message in prefix] + ["Queued question"]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [
    [{"role": "system", "content": "Changed instructions"}, *PREFIX[1:]],
    [{"role": "user", "content": "Deliberately new input"}, *PREFIX],
    [PREFIX[0]],
    [{"role": "user", "content": "First question"}],
])
async def test_unmatched_start_or_no_assistant_is_not_a_proven_history_prefix(queued_chat, prefix):
    before = rows(queued_chat)
    await resume(queued_chat, [*prefix, {"role": "user", "content": "Queued question"}])
    saved = rows(queued_chat)
    assert saved[:len(before)] == before
    assert [row["content"] for row in saved[len(before):]] == [message["content"] for message in prefix] + ["Queued question"]


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["asc", "desc"])
async def test_existing_complete_history_overlap_still_saves_one_new_turn(queued_chat, order):
    before = rows(queued_chat)
    await resume(queued_chat, [*PREFIX, INTERRUPTED, {"role": "user", "content": "Queued question"}], order=order)
    assert rows(queued_chat)[:-1] == before


@pytest.mark.asyncio
async def test_foreign_conversation_is_not_used_as_history_evidence(queued_chat):
    foreign = queued_chat.db.add_conversation({"client_id": "2", "title": "Other owner"})
    for message in PREFIX:
        queued_chat.db.add_message({"conversation_id": foreign, "sender": message["role"], "content": message["content"]})
    foreign_before = queued_chat.db.get_messages_for_conversation(foreign)
    queued_chat.conversation_id = queued_chat.db.add_conversation({"client_id": "1", "title": "Empty owned chat"})
    queued_chat.db.close_connection()
    await resume(queued_chat, [*PREFIX, {"role": "user", "content": "Queued question"}])
    assert [row["content"] for row in rows(queued_chat)] == [message["content"] for message in PREFIX] + ["Queued question"]
    assert queued_chat.db.get_messages_for_conversation(foreign) == foreign_before
