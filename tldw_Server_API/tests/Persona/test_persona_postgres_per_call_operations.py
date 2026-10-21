"""The persona live WebSocket's per-session handle returns each checkout per call (TASK-13861)."""

import ast
import asyncio
import threading
from collections import Counter
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.exemplar_runtime import resolve_persona_exemplar_runtime_context
from tldw_Server_API.app.core.Persona.exemplar_turn_classifier import PersonaTurnClassification

pytestmark = pytest.mark.integration

_APP = Path(__file__).resolve().parents[2] / "app"
# Off-loop uses of a persona session's handle: the WebSocket and the two helpers its turns call.
_HANDLE_SITES = {
    "api/v1/endpoints/persona.py": "persona_scope_db",
    "core/Persona/live_conversation.py": "profile_store",
    "core/Persona/exemplar_runtime.py": "persona_scope_db",
}


def test_every_off_loop_persona_handle_call_runs_in_its_own_operation():
    unwrapped, wrapped = [], 0
    for rel, handle in _HANDLE_SITES.items():
        for node in ast.walk(ast.parse((_APP / rel).read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "asyncio.to_thread"):
                continue
            if handle not in ast.unparse(node):
                continue
            callable_text = ast.unparse(node.args[0])
            if callable_text.startswith("chacha_operation("):
                wrapped += 1
            else:
                unwrapped.append(f"{rel}:{node.lineno} {callable_text}")
    assert unwrapped == [], f"wrap these in chacha_operation(...)(callable): {unwrapped}"
    assert wrapped == 8


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
