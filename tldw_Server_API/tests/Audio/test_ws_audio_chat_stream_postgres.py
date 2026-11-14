"""/chat/stream persistence over a real PostgreSQL ChaCha database (TASK-13861 AC1, AC2)."""

import asyncio
import itertools
import threading
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
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


async def test_chat_stream_sessions_return_every_checkout_while_the_server_runs(pg_owners, monkeypatch):
    owners, outstanding = pg_owners
    sessions_per_user = 3
    order = itertools.cycle([user_id for user_id in owners for _ in range(sessions_per_user)])

    async def _tts():
        return _DummyTTSService([b"tts"])

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001 - one call per session
        return owners[next(order)]

    monkeypatch.setattr(audio, "get_tts_service", _tts)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _db_for_user, raising=False)
    sockets = [_persisting_session() for _ in range(len(owners) * sessions_per_user)]

    await asyncio.gather(*(audio.websocket_audio_chat_stream(ws, token=None) for ws in sockets))

    # The event loop and its default executor threads are still alive here, as in a running server.
    assert outstanding() == []
    warnings = [m for ws in sockets for m in ws.sent_json if "persist" in str(m.get("message", "")).lower()]
    assert warnings == []
    with chacha_operation(independent=True):
        rows = owners[1].execute_query("SELECT sender FROM messages").fetchall()  # one database for all three
    assert Counter(row["sender"] for row in rows) == {"user": len(sockets), "assistant": len(sockets)}
