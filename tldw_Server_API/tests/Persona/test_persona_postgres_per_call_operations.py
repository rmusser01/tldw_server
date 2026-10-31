"""The persona live WebSocket's per-session handle returns each checkout per call (TASK-13861), off the event loop (TASK-13896)."""

import ast
import asyncio
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.exemplar_runtime import resolve_persona_exemplar_runtime_context
from tldw_Server_API.app.core.Persona.exemplar_turn_classifier import PersonaTurnClassification

pytestmark = pytest.mark.integration

_APP = Path(__file__).resolve().parents[2] / "app"
# Off-loop uses of a persona session's handle, and the wrapper each must use: the WebSocket owns
# every call, and the two helpers its turns call join an HTTP request's operation when there is one.
_HANDLE_SITES = {
    "api/v1/endpoints/persona.py": ("persona_scope_db", "chacha_operation(independent=True)("),
    "core/Persona/live_conversation.py": ("profile_store", "chacha_operation()("),
    "core/Persona/exemplar_runtime.py": ("persona_scope_db", "chacha_operation()("),
}


def test_every_off_loop_persona_handle_call_runs_in_its_own_operation():
    unwrapped, wrapped = [], 0
    for rel, (handle, wrapper) in _HANDLE_SITES.items():
        for node in ast.walk(ast.parse((_APP / rel).read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread"):
                continue
            if handle not in ast.unparse(node):
                continue
            callable_text = ast.unparse(node.args[0])
            if callable_text.startswith(wrapper):
                wrapped += 1
            else:
                unwrapped.append(f"{rel}:{node.lineno} {callable_text}")
    assert unwrapped == [], f"wrap these in the file's chacha_operation(...)(callable): {unwrapped}"
    assert wrapped == 21  # persona.py's 19 (TASK-13896 moved its 13 on-loop calls here) and the helpers' 2


# Calls that receive the session's handle but own their ChaCha work themselves, or end the session.
_SELF_OWNING = {
    "asyncio.to_thread",
    "resolve_persona_exemplar_runtime_context",
    "live_conversation.complete_persona_turn",
    "handle_skill_tool_call",
    "persona_scope_db.close_all_connections",
    "persona_scope_db.close_connection",
}
# The session's analytics helpers close over the handle; they are async, so every call is awaited.
_HANDLE_HELPERS = {
    "_resolve_persona_live_event_persona_id",
    "_upsert_persona_live_session_summary_safe",
    "_record_persona_live_voice_event_safe",
}


def test_no_persona_handle_call_runs_on_the_event_loop():
    """On the loop, a call's checkout would stall the whole server for pool_timeout when the pool is full."""
    tree = ast.parse((_APP / "api/v1/endpoints/persona.py").read_text(encoding="utf-8"))
    awaited = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Await)}
    on_loop, unawaited = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if ast.unparse(target) in _HANDLE_HELPERS and id(node) not in awaited:
            unawaited.append(f"persona.py:{node.lineno} {ast.unparse(target)}")
        per_call = isinstance(target, ast.Call) and ast.unparse(target.func).startswith("chacha_operation(")
        callee = target.args[0] if per_call else target
        passes_handle = "persona_scope_db" in {ast.unparse(v) for v in [*node.args, *(k.value for k in node.keywords)]}
        calls_handle = isinstance(callee, ast.Attribute) and ast.unparse(callee.value) == "persona_scope_db"
        if (passes_handle or calls_handle) and ast.unparse(callee) not in _SELF_OWNING:
            on_loop.append(f"persona.py:{node.lineno} {ast.unparse(target)}")
    assert on_loop == [], f"run these as await asyncio.to_thread(chacha_operation(independent=True)(fn), ...): {on_loop}"
    assert unawaited == [], f"await these: {unawaited}"


# The session's writes. Each must finish once queued, even if the session's task is cancelled meanwhile.
_WRITES = {
    "persona_scope_db.upsert_persona_live_voice_session_summary",
    "record_persona_live_voice_event",
    "_persist_persona_session_preferences",
}


def test_every_persona_write_is_shielded_once_queued():
    """A cancel that lands while a write's executor job is still queued would drop the write."""
    tree = ast.parse((_APP / "api/v1/endpoints/persona.py").read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    unshielded, shielded = [], 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread" and node.args):
            continue
        wrapped = node.args[0]
        inner = ast.unparse(wrapped.args[0]) if isinstance(wrapped, ast.Call) and wrapped.args else ""
        if inner not in _WRITES:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and ast.unparse(parent.func) == "run_quiescent_operation":
            shielded += 1
        else:
            unshielded.append(f"persona.py:{node.lineno} {inner}")
    assert (unshielded, shielded) == ([], 4), "wrap these as run_quiescent_operation(asyncio.to_thread(...))"


@pytest.fixture
def session_handles(pg_database_config, tmp_path, monkeypatch):
    """Four persona sessions' own handles on one ten-connection pool that records its holders."""
    pg_database_config.pool_timeout = 5.0
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
    handles = [CharactersRAGDB(tmp_path / f"session{n}.db", client_id="1", backend=backend) for n in range(4)]
    for handle in handles:
        handle.close_connection()

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    try:
        yield handles, outstanding
    finally:
        for handle in handles:
            handle.close_connection()
        pool.close_all()


async def test_concurrent_persona_sessions_hold_no_checkout_between_calls(session_handles):
    handles, outstanding = session_handles

    async def session(handle):
        for _turn in range(12):
            await resolve_persona_exemplar_runtime_context(
                persona_scope_db=handle,
                user_id="1",
                persona_id="persona-1",
                classifier=PersonaTurnClassification(),
                current_turn_text="hello",
            )

    await asyncio.gather(*(session(handle) for handle in handles))

    # The loop and its default executor threads are alive, as during live sessions.
    assert outstanding() == []


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.fixture
def live_session_on_a_small_pool(pg_database_config, tmp_path, monkeypatch):
    """A live persona WebSocket over a PostgreSQL handle whose two-connection pool the test can fill.

    It records, for each checkout, whether it was taken on an event-loop thread and how long
    it waited, and the server loop's gaps between 20 ms heartbeat ticks.
    """
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
    from tldw_Server_API.tests.Persona.test_persona_ws import fastapi_app

    pg_database_config.pool_size = 2
    pg_database_config.pool_timeout = 10.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    original_get = pool.get_connection
    takes, gaps, stop = [], [], threading.Event()

    def get_connection():
        started = time.monotonic()
        raw = original_get()
        takes.append((_on_event_loop(), round(time.monotonic() - started, 1)))
        return raw

    monkeypatch.setattr(pool, "get_connection", get_connection)
    db = CharactersRAGDB(tmp_path / "persona.db", client_id="1", backend=backend)
    persona_id = db.create_persona_profile(
        {"id": "research_assistant", "user_id": "1", "name": "Research Assistant", "mode": "session_scoped",
         "system_prompt": "Helper", "is_active": True}
    )
    db.create_persona_session(
        {"id": "pg-live", "persona_id": persona_id, "user_id": "1", "mode": "session_scoped", "status": "active"}
    )
    db.close_connection()

    async def heartbeat():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beats = []

    async def authenticate(*_args, **_kwargs):
        beats.append(asyncio.get_running_loop().create_task(heartbeat()))  # on the server's loop
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", authenticate)
    monkeypatch.setattr(persona_ep, "_open_persona_ws_db", lambda _user_id: db)
    try:
        with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
            ws.receive_json()
            takes.clear()
            gaps.clear()
            yield SimpleNamespace(ws=ws, pool=pool, hold=original_get, takes=takes, gaps=gaps)
            stop.set()
    finally:
        stop.set()
        db.close_connection()
        pool.close_all()


def test_a_persona_call_waits_for_a_full_pool_without_stalling_the_event_loop(live_session_on_a_small_pool):
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    live = live_session_on_a_small_pool
    held = [live.hold() for _ in range(2)]  # other work holds the whole pool for 1.5 s
    release = threading.Timer(1.5, lambda: [live.pool.return_connection(conn) for conn in held])
    release.start()
    try:
        live.ws.send_json({"type": "voice_config", "session_id": "pg-live", "stt": {"model": "tiny"}})
        _recv_until(live.ws, lambda event: event.get("reason_code") == "VOICE_CONFIG_UPDATED", timeout=20)
    finally:
        release.join()

    on_loop = [waited for loop, waited in live.takes if loop]
    longest_wait = max((waited for loop, waited in live.takes if not loop), default=0.0)
    assert (on_loop, longest_wait >= 1.0, max(live.gaps) < 0.75) == ([], True, True), (live.takes, max(live.gaps))


@pytest.mark.parametrize("shielded", [True, False], ids=["shielded", "plain-to-thread"])
async def test_a_queued_preferences_write_survives_its_task_being_cancelled(pg_database_config, tmp_path, shielded):
    """The shield persona.py puts around its writes: a cancel while the write is still queued must not drop it.

    The default executor gets one worker, held busy, so the write's job is queued when its task is cancelled,
    as when a session ends while the executor is saturated. A plain to_thread drops the queued job.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
    from tldw_Server_API.app.core.Notes_Graph.semantic_publication import run_quiescent_operation

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "persona.db", client_id="1", backend=backend)
    persona_id = db.create_persona_profile(
        {"id": "research_assistant", "user_id": "1", "name": "Research Assistant", "mode": "session_scoped",
         "system_prompt": "Helper", "is_active": True}
    )
    db.create_persona_session(
        {"id": "pg-live", "persona_id": persona_id, "user_id": "1", "mode": "session_scoped", "status": "active"}
    )
    db.close_connection()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    busy = loop.run_in_executor(None, release.wait, 5)  # the only worker is taken
    write = asyncio.to_thread(
        chacha_operation(independent=True)(persona_ep._persist_persona_session_preferences),
        db, session_id="pg-live", user_id="1", base_preferences={}, patch_preferences={"use_memory_context": False},
    )
    task = asyncio.create_task(run_quiescent_operation(write) if shielded else write)
    try:
        await asyncio.sleep(0.1)  # the write's job is queued behind the busy worker
        task.cancel()
        await asyncio.sleep(0)  # the cancel reaches the queued job before the worker frees up
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await busy
        await asyncio.to_thread(time.sleep, 0)  # anything still queued has run
        with chacha_operation(independent=True):
            saved = db.get_persona_session("pg-live", user_id="1")["preferences"]
        assert saved.get("use_memory_context") is (False if shielded else None)
    finally:
        release.set()
        db.close_connection()
        backend.get_pool().close_all()
