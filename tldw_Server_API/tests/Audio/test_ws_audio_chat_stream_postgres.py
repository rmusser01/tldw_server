"""/chat/stream persistence over a real PostgreSQL ChaCha database (TASK-13861 AC1, AC2)."""

import asyncio
import threading
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError
from tldw_Server_API.tests.Audio.test_ws_audio_chat_stream import (  # noqa: F401 - autouse fixture
    _AUDIO_LABEL_TO_SAMPLE,
    DummyWebSocket,
    _DummyTTSService,
    _pcm16_audio,
    mock_audio_ws_dependencies,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_owners(pg_database_config, tmp_path, monkeypatch):
    """Three users' cached handles on one pool that records who holds each checkout."""
    pg_database_config.pool_timeout = 5.0  # a pinned pool fails in 5 s, not 30
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
    # New users: their concurrent first sessions race to create the default character.
    owners = {
        user_id: CharactersRAGDB(tmp_path / f"user{user_id}.db", client_id=str(user_id), backend=backend)
        for user_id in (1, 2, 3)
    }
    for db in owners.values():
        db.close_connection()

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield owners, outstanding
    finally:
        for db in owners.values():
            db.close_connection()
        pool.close_all()


def _persisting_session():
    payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    return DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )


@pytest.fixture
def authenticated_sockets(pg_owners, monkeypatch):
    """Give each socket a stable owner, independent of concurrent DB lookup order."""
    owners, _ = pg_owners
    sessions_per_user = 3
    socket_owners = {_persisting_session(): user_id for user_id in owners for _ in range(sessions_per_user)}

    async def _auth(websocket: DummyWebSocket, **_kwargs) -> tuple[bool, int]:
        user_id = socket_owners[websocket]
        websocket.state.auth_principal = AuthPrincipal(kind="user", user_id=user_id, subject=f"user:{user_id}")
        return True, user_id

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.settings.is_multi_user_mode", lambda: True)
    monkeypatch.setattr("tldw_Server_API.app.core.Audio.streaming_service._audio_ws_authenticate", _auth)
    return socket_owners


async def test_chat_stream_sessions_return_every_checkout_while_the_server_runs(
    pg_owners, authenticated_sockets, monkeypatch
):
    owners, outstanding = pg_owners
    socket_owners = authenticated_sockets
    sockets = list(socket_owners)

    async def _tts():
        return _DummyTTSService([b"tts"])

    async def _db_for_user(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
        assert client_id == str(user_id)
        return owners[user_id]

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.audio.audio_tts.get_tts_service", _tts)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user_id",
        _db_for_user,
        raising=False,
    )

    await asyncio.gather(*(audio.websocket_audio_chat_stream(ws, token=None) for ws in sockets))

    # The event loop and its default executor threads are still alive here, as in a running server.
    assert outstanding() == []
    warnings = [m for ws in sockets for m in ws.sent_json if "persist" in str(m.get("message", "")).lower()]
    assert warnings == []
    conversation_ids = set()
    for ws, user_id in socket_owners.items():
        session_ids = [m["session_id"] for m in ws.sent_json if m.get("type") == "session"]
        assert len(session_ids) == 1
        conversation_id = session_ids[0]
        conversation_ids.add(conversation_id)
        with chacha_operation(independent=True):
            conversation = owners[user_id].get_conversation_by_id(conversation_id)
            assert conversation is not None
            assert conversation["client_id"] == str(user_id)
            messages = owners[user_id].get_messages_for_conversation(conversation_id)
            assert Counter(m["sender"] for m in messages) == {"user": 1, "assistant": 1}
            assert {m["client_id"] for m in messages} == {str(user_id)}
        # The admin fixture proves the application guard, not restricted-role RLS.
        for other_id, db in owners.items():
            if other_id == user_id:
                continue
            with chacha_operation(independent=True):
                with pytest.raises(InputError, match="not found or deleted") as rejected:
                    db.add_message(
                        {
                            "conversation_id": conversation_id,
                            "sender": "user",
                            "content": "cross-owner write must fail",
                            "client_id": str(user_id),
                        }
                    )
                assert isinstance(rejected.value.__cause__, NotFoundError)
    assert len(conversation_ids) == len(sockets)
    with chacha_operation(independent=True):
        rows = owners[1].execute_query("SELECT sender FROM messages").fetchall()  # one database for all three
    assert Counter(row["sender"] for row in rows) == {"user": len(sockets), "assistant": len(sockets)}
    assert outstanding() == []
