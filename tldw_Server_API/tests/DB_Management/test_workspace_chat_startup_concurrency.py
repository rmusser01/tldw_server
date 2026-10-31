"""Strict startup contention through independent real SQLite/PostgreSQL handles."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event
from time import monotonic, sleep
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import WorkspaceChatStartupRequest
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Chat.assistant_startup import decode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
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


def _process_startup(
    db_path: str, config: Any, channel: Any, payload: dict[str, Any], key: str,
    receipt_limit: int, chat_limit: int | None, boundary: str,
) -> None:
    """Open fresh process-local drivers; pause only at real admission/commit boundaries."""
    feature_flags.is_persona_enabled = lambda: True
    backend = DatabaseBackendFactory.create_backend(config) if config is not None else None
    db = CharactersRAGDB(db_path, client_id="user-1", backend=backend)
    try:
        channel.send(("ready", _warm_worker(db)))
        assert channel.recv() == "start"
        if db.backend_type == BackendType.POSTGRESQL:
            lock_owner = db.workspace_chat_startups.lock_owner

            def observe_owner(*args: Any, **kwargs: Any) -> None:
                """Signal immediately before entering the real owner lock."""
                channel.send(("attempting",))
                lock_owner(*args, **kwargs)

            db.workspace_chat_startups.lock_owner = observe_owner
        else:
            def trace_begin(statement: str) -> None:
                """SQLite's writer lock is taken before owner admission."""
                if statement.strip().upper() == "BEGIN IMMEDIATE":
                    channel.send(("attempting",))

            db.get_connection().set_trace_callback(trace_begin)
        if boundary in {"hold", "crash-before"}:
            insert = db.workspace_chat_startups.insert_receipt

            def pause_insert(*args: Any, **kwargs: Any) -> None:
                """Keep the real chat and receipt uncommitted until the parent observes them."""
                insert(*args, **kwargs)
                channel.send(("inserted",))
                assert channel.recv() == "release"
                if boundary == "crash-before":
                    os._exit(87)

            db.workspace_chat_startups.insert_receipt = pause_insert
        try:
            result = start_workspace_chat(
                db, owner_id="user-1", request=WorkspaceChatStartupRequest.model_validate(payload),
                idempotency_key=key, receipt_limit=receipt_limit, chat_limit=chat_limit, title_timestamp="process",
            )
        except WorkspaceStartupError as error:
            channel.send(("error", error.status_code, error.code))
        else:
            if boundary == "crash-after":
                channel.send(("committed",))
                assert channel.recv() == "release"
                os._exit(87)
            channel.send(("result", result.replayed, result.conversation["id"]))
    finally:
        db.close_all_connections()
        channel.close()


def _spawn_startup(
    db: CharactersRAGDB, *, boundary: str, payload: dict[str, Any], key: str,
    receipt_limit: int = 1, chat_limit: int | None = None,
) -> tuple[Any, Any]:
    """Pass only the official fixture's configuration, never a live inherited handle."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    config = db.backend.config if db.backend_type == BackendType.POSTGRESQL else None
    process = context.Process(target=_process_startup, args=(
        db.db_path_str, config, child, payload, key, receipt_limit, chat_limit, boundary,
    ))
    process.start()
    child.close()
    return process, parent


def _process_message(channel: Any, expected: str) -> tuple[Any, ...]:
    """Timeout failed children without substituting sleeps for positive boundary evidence."""
    assert channel.poll(60), f"startup process did not reach {expected}"
    message = channel.recv()
    assert message[0] == expected, message
    return message


def _close_process(process: Any, channel: Any) -> None:
    """Clean up only this test's child, including failures at controlled pause points."""
    try:
        if process.is_alive():
            process.terminate()
        process.join(10)
        assert not process.is_alive(), "test-owned startup process did not stop"
    finally:
        channel.close()
        process.close()


@pytest.mark.parametrize("race", ["same-key", "owner-capacity", "workspace-quota"])
def test_spawned_processes_serialize_startup_acceptance(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], race: str,
) -> None:
    """Separate OS processes share key authority, owner capacity and Workspace quota."""
    if race == "owner-capacity":
        with creation_db.transaction():
            creation_db.upsert_workspace("other", "Other Workspace")
    observer = db_factory()
    payload = _request().model_dump(exclude_unset=True)
    second_payload = (
        {"scope_type": "workspace", "workspace_id": "other", "workspace_assistant_selection": "none"}
        if race == "owner-capacity" else payload
    )
    limits = {"receipt_limit": 2, "chat_limit": 1} if race == "workspace-quota" else {}
    with ExitStack() as cleanup:
        first, first_channel = _spawn_startup(creation_db, boundary="hold", payload=payload, key="process-key", **limits)
        cleanup.callback(_close_process, first, first_channel)
        first_pid = _process_message(first_channel, "ready")[1]
        second, second_channel = _spawn_startup(
            creation_db, boundary="run", payload=second_payload,
            key="process-key" if race == "same-key" else "other-key", **limits,
        )
        cleanup.callback(_close_process, second, second_channel)
        second_pid = _process_message(second_channel, "ready")[1]
        first_channel.send("start")
        _process_message(first_channel, "attempting")
        _process_message(first_channel, "inserted")
        second_channel.send("start")
        _process_message(second_channel, "attempting")
        if creation_db.backend_type == BackendType.POSTGRESQL:
            with observer.transaction() as conn:
                deadline = monotonic() + 15
                while first_pid not in conn.execute(
                    "SELECT pg_blocking_pids(?) AS blockers", (second_pid,),
                ).fetchone()["blockers"]:
                    assert second.is_alive() and monotonic() < deadline, "contender did not block on winner"
                    sleep(0.02)
                assert conn.execute(
                    "SELECT 1 FROM pg_locks WHERE pid = ? AND locktype = 'advisory' AND NOT granted", (second_pid,),
                ).fetchone() is not None
        else:
            conn = observer.get_connection()
            conn.execute("PRAGMA busy_timeout = 0")
            with pytest.raises(sqlite3.OperationalError) as blocked:
                conn.execute("BEGIN IMMEDIATE")
            assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert not second_channel.poll(), "contender returned while winner's acceptance was uncommitted"
        first_channel.send("release")
        accepted = _process_message(first_channel, "result")
        assert accepted[1] is False
        if race == "same-key":
            assert _process_message(second_channel, "result") == ("result", True, accepted[2])
        else:
            expected = (
                (409, "workspace_chat_receipt_capacity_exceeded") if race == "owner-capacity"
                else (429, "workspace_chat_quota_exceeded")
            )
            assert _process_message(second_channel, "error") == ("error", *expected)
        for process in (first, second):
            process.join(30)
            assert process.exitcode == 0
        chats, receipts = _observe_rows(observer)
        assert len(chats) == len(receipts) == 1
        assert chats[0]["id"] == receipts[0]["conversation_id"] == accepted[2]


@pytest.mark.parametrize("boundary", ["crash-before", "crash-after"])
def test_process_exit_preserves_atomic_acceptance_and_response_loss_replay(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], boundary: str,
) -> None:
    """An abrupt exit rolls back both rows or preserves both, never a half acceptance."""
    payload = _request().model_dump(exclude_unset=True)
    process, channel = _spawn_startup(creation_db, boundary=boundary, payload=payload, key="crash-key")
    try:
        _process_message(channel, "ready")
        channel.send("start")
        _process_message(channel, "attempting")
        _process_message(channel, "inserted" if boundary == "crash-before" else "committed")
        channel.send("release")
        process.join(30)
        assert process.exitcode == 87
        observer = db_factory()
        chats, receipts = _observe_rows(observer)
        assert len(chats) == len(receipts) == int(boundary == "crash-after")
        result = _start(observer, _request(), "crash-key", "retry")
        assert result.replayed is (boundary == "crash-after")
        after_chats, after_receipts = _observe_rows(observer)
        assert len(after_chats) == len(after_receipts) == 1
        assert after_chats[0]["id"] == after_receipts[0]["conversation_id"] == result.conversation["id"]
        if chats:
            assert result.conversation["id"] == chats[0]["id"]
    finally:
        _close_process(process, channel)


def test_failed_second_spawn_cleans_first_process(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure launching a contender cannot leak the already-started process/driver."""
    spawn, close = _spawn_startup, _close_process
    started, closed = [], []

    def fail_second_spawn(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Fail only after the first real process has started."""
        if started:
            raise RuntimeError("controlled contender launch failure")
        worker = spawn(*args, **kwargs)
        started.append(worker)
        return worker

    def record_cleanup(process: Any, channel: Any) -> None:
        """Retain real termination/join and observe cleanup registration."""
        close(process, channel)
        closed.append(process)

    monkeypatch.setattr(f"{__name__}._spawn_startup", fail_second_spawn)
    monkeypatch.setattr(f"{__name__}._close_process", record_cleanup)
    try:
        with pytest.raises(RuntimeError, match="controlled contender launch failure"):
            test_spawned_processes_serialize_startup_acceptance(creation_db, db_factory, "same-key")
        assert closed == [started[0][0]]
    finally:
        for process, channel in started:
            if process not in closed:
                close(process, channel)
