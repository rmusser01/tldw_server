"""Voice assistant WebSocket over a real PostgreSQL ChaCha database (TASK-13861, TASK-13896).

Every ChaCha checkout the WebSocket makes is taken off the event loop: on the loop, a
full pool would stall the whole server for up to pool_timeout.
"""

import asyncio
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VoiceAssistant.registry import VoiceCommandRegistry
from tldw_Server_API.app.core.VoiceAssistant.router import VoiceCommandRouter
from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionResult, ActionType, ParsedIntent, VoiceIntent
from tldw_Server_API.app.core.VoiceAssistant.session import VoiceSessionManager
from tldw_Server_API.tests.VoiceAssistant import test_ws_integration as ws_integration

pytestmark = pytest.mark.integration
DummyWebSocket = ws_integration.DummyWebSocket
mock_ws_dependencies = ws_integration.mock_ws_dependencies  # the shared fixture, registered in this module
_POOL_SIZE = 3


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def _heartbeat(gaps: list[float]) -> None:
    """Record how long the event loop goes between 20 ms ticks."""
    last = time.monotonic()
    while True:
        await asyncio.sleep(0.02)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


@pytest.fixture
def pg_voice_db(pg_database_config, tmp_path, monkeypatch):
    """The user's cached handle, served to the WebSocket, on a small pool that records each checkout."""
    pg_database_config.pool_size = _POOL_SIZE
    pg_database_config.pool_timeout = 1.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned, on_loop = {}, Counter(), Counter(), []
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        on_loop.append(_on_event_loop())
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
    db.close_connection()
    on_loop.clear()

    async def _db_for_user(_user_id, client_id=None):  # noqa: ARG001
        return db

    monkeypatch.setattr(voice_assistant, "get_chacha_db_for_user_id", _db_for_user)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield SimpleNamespace(
            db=db, outstanding=outstanding, on_loop=on_loop, hold=original_get, give_back=original_return
        )
    finally:
        db.close_connection()
        pool.close_all()


async def test_session_start_takes_no_checkout_on_the_event_loop(mock_ws_dependencies, pg_voice_db):
    # The shared fixture's registry loads no commands, so the session's save is the ChaCha work here.
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])

    await voice_assistant.websocket_voice_assistant(ws)

    sent = [m.get("type") for m in ws.sent_json]
    assert ("auth_ok" in sent, len(pg_voice_db.on_loop) > 0, sum(pg_voice_db.on_loop)) == (True, True, 0)
    assert pg_voice_db.outstanding() == []


async def test_a_full_pool_at_session_start_degrades_the_session(mock_ws_dependencies, pg_voice_db):
    held = [pg_voice_db.hold() for _ in range(_POOL_SIZE)]  # other work holds the whole pool
    gaps: list[float] = []
    beat = asyncio.create_task(_heartbeat(gaps))
    await asyncio.sleep(0.05)  # the heartbeat is ticking before the session starts
    ws = DummyWebSocket([{"type": "auth", "token": "valid-token"}, {"type": "config"}])
    escaped = None
    try:
        await voice_assistant.websocket_voice_assistant(ws)
    except Exception as exc:  # noqa: BLE001 - the RED shape: PoolTimeout used to escape the socket
        escaped = type(exc).__name__
    finally:
        await asyncio.sleep(0.05)  # one more tick records any stall the session caused
        beat.cancel()
        for conn in held:
            pg_voice_db.give_back(conn)

    sent = [m.get("type") for m in ws.sent_json]
    assert (escaped, "auth_ok" in sent, "config_ack" in sent, max(gaps) < 0.5) == (None, True, True, True), (
        escaped,
        sent,
        max(gaps),
    )


class _StubParser:
    """Answers every utterance with an LLM chat intent, as the real parser's fallback does."""

    async def parse(self, text, user_id, context, persona_id=None):  # noqa: ARG002
        return ParsedIntent(intent=VoiceIntent(action_type=ActionType.LLM_CHAT, raw_text=text), match_method="stub")


async def test_a_command_takes_no_checkout_on_the_event_loop_and_holds_none_while_it_acts(
    mock_ws_dependencies, pg_voice_db, monkeypatch
):
    sessions = VoiceSessionManager()
    router = VoiceCommandRouter(
        registry=VoiceCommandRegistry(), parser=_StubParser(), session_manager=sessions, workflow_handler=object()
    )
    held_while_acting = []

    async def act(intent, session, *, db=None, persona_id=None):  # noqa: ARG001
        """Stands in for the LLM call, MCP action or SYNC workflow the command runs."""
        held_while_acting.append(pg_voice_db.outstanding())
        return ActionResult(success=True, action_type=ActionType.LLM_CHAT, response_text="done")

    monkeypatch.setattr(router, "_execute_intent", act)
    monkeypatch.setattr(voice_assistant, "get_voice_session_manager", lambda: sessions)
    monkeypatch.setattr(voice_assistant, "get_voice_command_router", lambda: router)
    ws = DummyWebSocket(
        [{"type": "auth", "token": "valid-token"}, {"type": "config"}, {"type": "text", "text": "hello there"}]
    )
    try:
        await voice_assistant.websocket_voice_assistant(ws)
    finally:
        await sessions.stop()

    loop_takes, outstanding = sum(pg_voice_db.on_loop), pg_voice_db.outstanding()
    with chacha_operation(independent=True):  # the command's analytics row was committed
        events = pg_voice_db.db.execute_query("SELECT count(*) AS n FROM voice_command_events").fetchone()["n"]
    assert (held_while_acting, loop_takes, outstanding, events) == ([[]], 0, [], 1)


async def test_a_command_s_analytics_row_survives_a_cancel_while_its_write_is_queued(pg_voice_db, monkeypatch):
    """The command's analytics-and-session save must finish once queued, even if the command's task is cancelled.

    The default executor gets one worker, which the action takes, so the save's job is queued when the
    task is cancelled, as when a WebSocket handler is cancelled while the executor is saturated.
    """
    from concurrent.futures import ThreadPoolExecutor

    sessions = VoiceSessionManager()
    router = VoiceCommandRouter(
        registry=VoiceCommandRegistry(), parser=_StubParser(), session_manager=sessions, workflow_handler=object()
    )
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release, acted, busy = threading.Event(), asyncio.Event(), []

    async def act(intent, session, *, db=None, persona_id=None):  # noqa: ARG001
        busy.append(loop.run_in_executor(None, release.wait, 5))  # the only worker is taken
        acted.set()
        return ActionResult(success=True, action_type=ActionType.LLM_CHAT, response_text="done")

    monkeypatch.setattr(router, "_execute_intent", act)
    task = asyncio.create_task(router.process_command(text="hello there", user_id=1, db=pg_voice_db.db))
    try:
        await asyncio.wait_for(acted.wait(), timeout=10)
        await asyncio.sleep(0.1)  # the save's job is queued behind the busy worker
        task.cancel()
        await asyncio.sleep(0)  # the cancel reaches the queued job before the worker frees up
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await busy[0]
        await asyncio.to_thread(time.sleep, 0)  # anything still queued has run
    finally:
        release.set()
        await sessions.stop()

    with chacha_operation(independent=True):
        events = pg_voice_db.db.execute_query("SELECT count(*) AS n FROM voice_command_events").fetchone()["n"]
    assert events == 1


async def test_an_external_action_reads_its_connection_off_the_event_loop(pg_voice_db):
    router = VoiceCommandRouter(
        registry=VoiceCommandRegistry(), parser=_StubParser(), session_manager=VoiceSessionManager(), workflow_handler=object()
    )

    result = await router._execute_external_connection_action(
        intent=VoiceIntent(action_type=ActionType.CUSTOM, raw_text="open the door"),
        session=SimpleNamespace(user_id=1, session_id="voice-1"),
        db=pg_voice_db.db,
        persona_id="persona-1",
        command=SimpleNamespace(connection_id="conn-1", persona_id="persona-1"),
    )

    assert (result.error_message, len(pg_voice_db.on_loop) > 0, sum(pg_voice_db.on_loop)) == (
        "Connection 'conn-1' was not found for persona 'persona-1'.",
        True,
        0,
    )
    assert pg_voice_db.outstanding() == []
