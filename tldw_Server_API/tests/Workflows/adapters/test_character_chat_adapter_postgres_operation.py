"""A workflow's character_chat step owns its ChaCha checkouts on PostgreSQL (TASK-13861).

A voice command runs a SYNC workflow inside its own operation. The engine's detached
tasks (a human step's timeout, the orphan reaper's continue_run) inherit that operation,
and run character_chat steps after it has closed.
"""

import asyncio

import pytest

import tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB as chacha_db
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Workflows.adapters.integration import messaging

pytestmark = pytest.mark.integration


async def test_character_chat_in_a_task_that_outlived_its_creators_operation(pg_database_config, tmp_path, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    real = chacha_db.CharactersRAGDB

    def per_call_handle(db_path=None, client_id=None):  # noqa: ARG001 - the adapter's per-call handle
        return real(tmp_path / "workflow.db", client_id="1", backend=backend)

    monkeypatch.setattr(chacha_db, "CharactersRAGDB", per_call_handle)
    monkeypatch.setattr(messaging, "is_test_mode", lambda: False)
    with chacha_operation(independent=True):
        character_id = per_call_handle().add_character_card({"name": "Ada", "first_message": "Hello."})

    async def steps(gate):
        await gate.wait()  # e.g. a human step's timeout, or a reaped orphan's continue_run
        started = await messaging.run_character_chat_adapter(
            {"action": "start", "character_id": character_id}, {"user_id": "1"}
        )
        loaded = await messaging.run_character_chat_adapter(
            {"action": "load", "conversation_id": started.get("conversation_id")}, {"user_id": "1"}
        )
        missing = await messaging.run_character_chat_adapter(
            {"action": "load", "conversation_id": "no-such-conversation"}, {"user_id": "1"}
        )
        return started, loaded, missing

    gate = asyncio.Event()
    with chacha_operation(independent=True):  # the voice command's operation
        task = asyncio.create_task(steps(gate))
    gate.set()  # the command has returned, and its operation is closed
    try:
        started, loaded, missing = await task
    finally:
        backend.get_pool().close_all()

    assert started.get("character_name") == "Ada", started
    assert (loaded.get("conversation_id"), loaded.get("character_name")) == (started["conversation_id"], "Ada"), loaded
    assert missing.get("error") == "conversation_not_found", missing
