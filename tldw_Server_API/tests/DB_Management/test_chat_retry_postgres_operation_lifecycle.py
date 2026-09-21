"""An answered-turn Retry must release PostgreSQL reads when its operation ends."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

QUESTION = "Repeat the saved answer."
CLIENT_MESSAGE_ID = "answered-retry-lifecycle"


@pytest.fixture
def retry_database(pg_database_config, tmp_path, monkeypatch):
    """Seed actual persisted history using the official per-test PostgreSQL DB."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "retry.db", client_id="1", backend=backend)
    db.add_character_card({"name": chat_service.DEFAULT_CHARACTER_NAME})
    conversation_id = db.add_conversation({"title": "Answered Retry lifecycle"})
    user_id = db.add_message({
        "conversation_id": conversation_id,
        "sender": "user",
        "content": QUESTION,
        "timestamp": "2026-01-01T00:00:00+00:00",
    })
    db.add_message_metadata(user_id, extra={"client_message_id": CLIENT_MESSAGE_ID})
    assistant_id = db.add_message({
        "conversation_id": conversation_id,
        "sender": "assistant",
        "content": "Previously saved answer.",
        "timestamp": "2026-01-01T00:00:01+00:00",
    })
    db.close_connection()

    pool = backend.get_pool()
    original_get = pool.get_connection
    original_return = pool.return_connection
    active = {}
    acquired = []

    def observe_get():
        raw = original_get()
        active[id(raw)] = raw
        acquired.append(raw)
        return raw

    def observe_return(raw):
        original_return(raw)
        active.pop(id(raw), None)

    # Observe real loans without replacing connections, SQL or cleanup behavior.
    monkeypatch.setattr(pool, "get_connection", observe_get)
    monkeypatch.setattr(pool, "return_connection", observe_return)
    try:
        yield SimpleNamespace(
            db=db,
            backend=backend,
            conversation_id=conversation_id,
            message_ids=[user_id, assistant_id],
            active=active,
            acquired=acquired,
        )
    finally:
        db.close_all_connections()
        pool.close_all()


async def _unexpected_save(*_args, **_kwargs):
    """The answered-turn rejection must happen before any persistence attempt."""
    raise AssertionError("An already answered Retry attempted to persist a new message")


def _assert_restart_lock_available(fixture):
    """A separate real connection must be able to take the schema-change lock."""
    try:
        with fixture.backend.transaction() as connection:
            fixture.backend.execute("SET LOCAL lock_timeout = '100ms'", connection=connection)
            fixture.backend.execute(
                "LOCK TABLE messages IN ACCESS EXCLUSIVE MODE", connection=connection
            )
    except DatabaseError:
        pytest.fail("The finished Retry still blocks a separate connection's messages schema lock")


@pytest.mark.parametrize("outcome", ["checkout_returned", "restart_lock", "saved_history_unchanged"])
def test_answered_retry_conflict_finishes_its_postgres_operation(retry_database, outcome):
    """Exercise actual Retry preparation and its 409 before inspecting cleanup."""
    fixture = retry_database
    # Establish lock permission/availability before the Retry introduces reads.
    _assert_restart_lock_available(fixture)
    fixture.acquired.clear()
    request = ChatCompletionRequest(
        model="unused-no-provider",
        conversation_id=fixture.conversation_id,
        save_to_db=True,
        messages=[{"role": "user", "content": QUESTION}],
        metadata={
            "tldw_client_message_id": CLIENT_MESSAGE_ID,
            "tldw_retry_failed_turn": True,
        },
    ).model_copy(update={"history_message_limit": 0})
    # Zero is supported by the service; it isolates the new unconditional tail
    # reads from the optional history-window reads. No model call is reached.

    async def exercise():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loop.set_default_executor(executor)
            try:
                with chacha_operation(independent=True):
                    with pytest.raises(HTTPException) as rejection:
                        await chat_service.build_context_and_messages(
                            chat_db=fixture.db,
                            request_data=request,
                            loop=loop,
                            metrics=get_chat_metrics(),
                            default_save_to_db=True,
                            final_conversation_id=fixture.conversation_id,
                            save_message_fn=_unexpected_save,
                        )
                    assert rejection.value.status_code == 409
                    assert "already has an answer" in rejection.value.detail

                assert fixture.acquired, "The actual Retry must reach PostgreSQL"
                if outcome == "checkout_returned":
                    states = [raw.info.transaction_status.name for raw in fixture.active.values()]
                    active_count = len(fixture.active)
                    assert active_count == 0, f"Finished Retry retains PostgreSQL checkouts: {states}"
                elif outcome == "restart_lock":
                    _assert_restart_lock_available(fixture)
                else:
                    rows = fixture.backend.execute(
                        "SELECT id, content FROM messages WHERE conversation_id = %s ORDER BY timestamp",
                        (fixture.conversation_id,),
                    ).rows
                    assert [(row["id"], row["content"]) for row in rows] == [
                        (fixture.message_ids[0], QUESTION),
                        (fixture.message_ids[1], "Previously saved answer."),
                    ]
            finally:
                # Cleanup happens only after the observed outcome, on the same
                # worker that owns any leaked legacy connection in the RED case.
                await loop.run_in_executor(None, fixture.db.close_connection)

    asyncio.run(exercise())
