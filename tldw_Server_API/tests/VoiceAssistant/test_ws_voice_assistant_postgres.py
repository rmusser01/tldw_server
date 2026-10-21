"""Voice assistant WebSocket over a real PostgreSQL ChaCha database (TASK-13861 AC1, AC2)."""

import threading
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.VoiceAssistant import test_ws_integration as ws_integration

pytestmark = pytest.mark.integration
DummyWebSocket = ws_integration.DummyWebSocket
mock_ws_dependencies = ws_integration.mock_ws_dependencies  # the shared fixture, registered in this module


@pytest.fixture
def pg_voice_db(pg_database_config, tmp_path, monkeypatch):
    """The user's cached handle, served to the WebSocket, on a pool that records its holders."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    db = CharactersRAGDB(tmp_path / "voice.db", client_id="1", backend=backend)

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001
        return db

    monkeypatch.setattr(voice_assistant, "get_chacha_db_for_user_id", _db_for_user)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield db, outstanding
    finally:
        db.close_connection()
        pool.close_all()


async def test_session_start_returns_its_checkout(mock_ws_dependencies, pg_voice_db):
    _db, outstanding = pg_voice_db
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])

    await voice_assistant.websocket_voice_assistant(ws, token=None)

    assert [m for m in ws.sent_json if m.get("type") == "auth_ok"]
    assert outstanding() == []


async def test_a_text_command_returns_its_checkout(mock_ws_dependencies, pg_voice_db, monkeypatch):
    _db, outstanding = pg_voice_db
    _sessions, command_router = mock_ws_dependencies
    original = command_router.process_command

    async def process_with_db(text, user_id, session_id, db=None, persona_id=None):
        # The real router reads and saves through the handle on the event-loop thread.
        db.get_character_card_by_name("nobody")
        return await original(text, user_id, session_id, db=db, persona_id=persona_id)

    monkeypatch.setattr(command_router, "process_command", process_with_db)
    ws = DummyWebSocket(
        [{"type": "auth", "token": "valid-token"}, {"type": "config"}, {"type": "text", "text": "hello there"}]
    )

    await voice_assistant.websocket_voice_assistant(ws, token=None)

    assert command_router.processed_commands == ["hello there"]
    assert outstanding() == []
