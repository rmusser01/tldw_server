"""Strict startup contention through independent real SQLite/PostgreSQL handles."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import WorkspaceChatStartupRequest
from tldw_Server_API.app.core.Chat.assistant_startup import decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import (
    WorkspaceStartupError,
    WorkspaceStartupResult,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Workspaces.chat_startup import start_workspace_chat
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import _wait_for_blocked_writer
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import _mutate
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db

pytestmark = pytest.mark.integration


def _request(**overrides: Any) -> WorkspaceChatStartupRequest:
    """Use the fixture's saved whole-Workspace version, not a racing resolver."""
    return WorkspaceChatStartupRequest.model_validate({
        "scope_type": "workspace", "workspace_id": "ws",
        "workspace_assistant_selection": "inherit", "workspace_assistant_default_version": 2,
        **overrides,
    })


def _start(
    db: CharactersRAGDB, request: WorkspaceChatStartupRequest, key: str, stamp: str,
) -> WorkspaceStartupResult:
    """Keep capacity one so a duplicate must replay before capacity decisions."""
    return start_workspace_chat(
        db, owner_id="user-1", request=request, idempotency_key=key,
        receipt_limit=1, chat_limit=None, title_timestamp=stamp,
    )


def _warm_worker(db: CharactersRAGDB) -> int | None:
    """Initialize the worker's actual thread connection before accepting transactions."""
    with db.transaction() as conn:
        if db.backend_type == BackendType.POSTGRESQL:
            conn.execute("SET statement_timeout = '15s'")
            return conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
        conn.execute("SELECT 1").fetchone()
    return None


def _observe_rows(db: CharactersRAGDB) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read committed chats and receipt targets together on a preinitialized observer."""
    with db.transaction() as conn:
        chats = [dict(row) for row in conn.execute(
            "SELECT id, workspace_id, title, assistant_id FROM conversations ORDER BY id"
        ).fetchall()]
        receipts = [dict(row) for row in conn.execute(
            "SELECT owner_user_id, workspace_id, conversation_id FROM workspace_chat_startup_receipts"
        ).fetchall()]
    return chats, receipts


def _assert_pg_blocker(
    conn: Any, pid: int, blocker_pid: int, future: Future[Any], *, owner_lock: bool = False,
) -> None:
    """Require the known writer in PostgreSQL's lock graph, not an elapsed-time proxy."""
    _wait_for_blocked_writer(conn, pid, future)
    blockers = conn.execute("SELECT pg_blocking_pids(?) AS blockers", (pid,)).fetchone()["blockers"]
    assert blocker_pid in blockers, "contender was not blocked by the controlled transaction"
    if owner_lock:
        assert conn.execute(
            "SELECT 1 FROM pg_locks WHERE pid = ? AND locktype = 'advisory' AND NOT granted", (pid,),
        ).fetchone() is not None, "startup must wait on owner admission, not a Workspace row"


def _contend(
    winner_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
    *, second_request: WorkspaceChatStartupRequest, second_key: str = "same-key",
) -> tuple[WorkspaceStartupResult, WorkspaceStartupResult | WorkspaceStartupError, CharactersRAGDB]:
    """Hold the real winner receipt uncommitted until the actual contender reaches admission."""
    contender_db, observer = db_factory(), db_factory()
    _warm_worker(observer)
    postgres = winner_db.backend_type == BackendType.POSTGRESQL
    ready = [Event(), Event()]
    start = [Event(), Event()]
    inserted, release, attempting = Event(), Event(), Event()
    pids: list[int | None] = [None, None]
    insert = winner_db.workspace_chat_startups.insert_receipt
    lock_owner = contender_db.workspace_chat_startups.lock_owner

    def pause_insert(*args: Any, **kwargs: Any) -> None:
        """Preserve the real INSERT and its locks; only control the commit boundary."""
        insert(*args, **kwargs)
        inserted.set()
        assert release.wait(15), "winner acceptance was not released"

    def observe_owner(*args: Any, **kwargs: Any) -> None:
        """Signal entry into real PostgreSQL owner admission before its blocking SQL."""
        attempting.set()
        lock_owner(*args, **kwargs)

    def trace_begin(statement: str) -> None:
        """SQLite blocks at BEGIN IMMEDIATE, before the owner helper is called."""
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            attempting.set()

    def run(index: int) -> WorkspaceStartupResult:
        """Warm both worker handles before either is permitted to accept a request."""
        db = winner_db if index == 0 else contender_db
        pids[index] = _warm_worker(db)
        if not postgres and index == 1:
            db.get_connection().set_trace_callback(trace_begin)
        ready[index].set()
        try:
            assert start[index].wait(20), "startup worker was not released"
            return _start(
                db, _request() if index == 0 else second_request,
                "same-key" if index == 0 else second_key, "winner" if index == 0 else "contender",
            )
        finally:
            if not postgres and index == 1:
                db.get_connection().set_trace_callback(None)

    with monkeypatch.context() as patch:
        patch.setattr(winner_db.workspace_chat_startups, "insert_receipt", pause_insert)
        if postgres:
            patch.setattr(contender_db.workspace_chat_startups, "lock_owner", observe_owner)
        with ThreadPoolExecutor(max_workers=2) as executor:
            winner, contender = executor.submit(run, 0), executor.submit(run, 1)
            try:
                assert all(event.wait(10) for event in ready), "worker connections were not initialized"
                start[0].set()
                assert inserted.wait(10), "winner never inserted its real receipt"
                start[1].set()
                assert attempting.wait(10), "contender never attempted database admission"
                if postgres:
                    with observer.transaction() as conn:
                        _assert_pg_blocker(conn, pids[1], pids[0], contender, owner_lock=True)
                else:
                    # A zero-timeout third handle confirms the live SQLite write lock.
                    # The contender's traced BEGIN establishes overlapping attempts;
                    # SQLite has no per-writer lock graph, so do not claim one here.
                    conn = observer.get_connection()
                    conn.execute("PRAGMA busy_timeout = 0")
                    with pytest.raises(sqlite3.OperationalError) as blocked:
                        conn.execute("BEGIN IMMEDIATE")
                    assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
                    assert not contender.done(), "contender completed while winner still held admission"
            finally:
                release.set()
                for event in start:
                    event.set()
            first = winner.result(timeout=20)
            try:
                second = contender.result(timeout=20)
            except WorkspaceStartupError as error:
                second = error
    return first, second, observer


def _assert_one_acceptance(observer: CharactersRAGDB, first: WorkspaceStartupResult) -> None:
    """A response match alone cannot hide duplicate chat or receipt rows."""
    assert first.replayed is False
    assert _observe_rows(observer) == (
        [{"id": first.conversation["id"], "workspace_id": "ws", "title": "First Chat (winner)",
          "assistant_id": "persona-a"}],
        [{"owner_user_id": "user-1", "workspace_id": "ws", "conversation_id": first.conversation["id"]}],
    )


def test_concurrent_same_key_request_replays_original_id(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing receipt rechecks after owner admission would create or reject this duplicate."""
    first, second, observer = _contend(creation_db, db_factory, monkeypatch, second_request=_request())
    assert isinstance(second, WorkspaceStartupResult)
    assert (second.replayed, second.conversation["id"], second.conversation["title"]) == (
        True, first.conversation["id"], "First Chat (winner)",
    )
    assert second.conversation["assistant_id"] == "persona-a"
    assert decode_assistant_startup(second.conversation["assistant_startup_json"]).workspace_version == 2
    _assert_one_acceptance(observer, first)


def test_concurrent_same_key_different_request_conflicts(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring the fingerprint must not turn competing requested metadata into replay."""
    first, second, observer = _contend(
        creation_db, db_factory, monkeypatch, second_request=_request(title="Other requested title"),
    )
    assert isinstance(second, WorkspaceStartupError)
    assert (second.status_code, second.code, second.reason) == (409, "idempotency_key_conflict", None)
    _assert_one_acceptance(observer, first)


def test_concurrent_owner_capacity_is_shared_across_workspaces(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-Workspace admission lock would let both different keys consume owner capacity one."""
    with creation_db.transaction():
        creation_db.upsert_workspace("other", "Other Workspace")
        creation_db.update_workspace("other", {"assistant_defaults_json": {
            "assistant_kind": "persona", "assistant_id": "persona-b", "persona_memory_mode": "read_write",
        }}, 1)
    first, second, observer = _contend(
        creation_db, db_factory, monkeypatch, second_request=_request(workspace_id="other"), second_key="other-key",
    )
    assert isinstance(second, WorkspaceStartupError)
    assert (second.status_code, second.code, second.reason) == (
        409, "workspace_chat_receipt_capacity_exceeded", None,
    )
    _assert_one_acceptance(observer, first)


@pytest.mark.postgres
@pytest.mark.parametrize("db_factory", ["postgres"], indirect=True)
@pytest.mark.parametrize("mutation", ["clear", "deactivate"])
def test_postgres_mutation_between_preflight_and_selection_rejects_without_acceptance(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], mutation: str,
) -> None:
    """Stale preflight version or availability cannot survive a real blocking selection read."""
    creator, observer = db_factory(), db_factory()
    _warm_worker(observer)
    request = _request()
    with creation_db.transaction() as conn:
        assert creation_db.get_workspace("ws", conn=conn)["version"] == 2
        assert creation_db.get_persona_profile("persona-a", user_id="user-1", conn=conn)["is_active"]
    ready, start = Event(), Event()
    pid: list[int] = []

    def accept() -> WorkspaceStartupResult:
        """Use an idle worker connection so strict startup owns the outer transaction."""
        pid.append(_warm_worker(creator))
        ready.set()
        assert start.wait(20), "mutation contender was not released"
        return _start(creator, request, "mutation-key", "after-preflight")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(accept)
        try:
            assert ready.wait(10), "creator connection was not initialized"
            with creation_db.transaction() as conn:
                _mutate(creation_db, mutation)
                blocker_pid = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
                start.set()
                _assert_pg_blocker(conn, pid[0], blocker_pid, future)
        finally:
            start.set()
        with pytest.raises(WorkspaceStartupError) as caught:
            future.result(timeout=20)
    expected = (
        (409, "workspace_assistant_version_conflict", None) if mutation == "clear"
        else (409, "workspace_assistant_unavailable", "persona_unavailable")
    )
    assert (caught.value.status_code, caught.value.code, caught.value.reason) == expected
    assert _observe_rows(observer) == ([], [])
    with observer.transaction() as conn:
        if mutation == "clear":
            row = observer.get_workspace("ws", conn=conn)
            assert (row["version"], row["assistant_defaults_json"]) == (3, None)
        else:
            assert not observer.get_persona_profile("persona-a", user_id="user-1", conn=conn)["is_active"]
