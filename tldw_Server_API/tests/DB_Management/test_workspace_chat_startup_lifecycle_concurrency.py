"""Actual Workspace closure and replay contention against receipt-bound writers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.native_fork_schemas import NativeScopeV1
from tldw_Server_API.app.core.Chat.native_fork_projection import AuthorizedNativeOwner
from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import WorkspaceStartupError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Workspaces import chat_startup
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory
from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_acceptance import _start
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_concurrency import _assert_pg_blocker, _warm_worker
from tldw_Server_API.tests.DB_Management.test_workspace_chat_startup_lifecycle import _receipt, _resurrect

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("writer", ["restore", "sync"])
@pytest.mark.parametrize("hard", [False, True])
def test_receipt_admission_committing_before_closure_is_cascaded(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, writer: str, hard: bool,
) -> None:
    """Real admission locks delay closure; enumeration then includes the committed chat."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    closer, observer = db_factory(), db_factory()
    _warm_worker(observer)
    postgres = creation_db.backend_type.value == "postgresql"
    ready = [Event(), Event()]
    start = [Event(), Event()]
    admitted, closing, release = Event(), Event(), Event()
    pids: list[int | None] = [None, None]
    recheck = creation_db.conversation_store._recheck_receipt_admission
    lock_close = closer._lock_native_workspace_delete

    def pause_admitted(*args: Any, **kwargs: Any) -> None:
        """Keep actual Workspace/chat locks until the closure contender is observed."""
        recheck(*args, **kwargs)
        admitted.set()
        assert release.wait(20), "admission was not released"

    def observe_close(*args: Any, **kwargs: Any) -> Any:
        """Observe the real PostgreSQL closure lock, without changing its strength."""
        closing.set()
        return lock_close(*args, **kwargs)

    def trace_begin(statement: str) -> None:
        """SQLite's closure contender blocks before it reaches the Workspace helper."""
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            closing.set()

    def run(index: int) -> None:
        """Initialize both thread-local connections before entering the critical interval."""
        db = creation_db if index == 0 else closer
        pids[index] = _warm_worker(db)
        if index == 1 and not postgres:
            db.get_connection().set_trace_callback(trace_begin)
        ready[index].set()
        try:
            assert start[index].wait(20)
            if index == 0:
                _resurrect(db, cid, writer, 2)
            elif hard:
                db.hard_delete_workspace("ws")
            else:
                assert db.delete_workspace("ws", expected_version=2)
        finally:
            if index == 1 and not postgres:
                db.get_connection().set_trace_callback(None)

    monkeypatch.setattr(creation_db.conversation_store, "_recheck_receipt_admission", pause_admitted)
    monkeypatch.setattr(closer, "_lock_native_workspace_delete", observe_close)
    with ThreadPoolExecutor(max_workers=2) as executor:
        accepting, deleting = executor.submit(run, 0), executor.submit(run, 1)
        try:
            assert all(event.wait(20) for event in ready)
            start[0].set()
            assert admitted.wait(20)
            start[1].set()
            assert closing.wait(20)
            if postgres:
                with observer.transaction() as conn:
                    _assert_pg_blocker(conn, pids[1], pids[0], deleting)
            else:
                conn = observer.get_connection()
                conn.execute("PRAGMA busy_timeout = 0")
                with pytest.raises(sqlite3.OperationalError) as blocked:
                    conn.execute("BEGIN IMMEDIATE")
                assert blocked.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
                assert not deleting.done()
        finally:
            release.set()
            for event in start:
                event.set()
        accepting.result(timeout=20)
        deleting.result(timeout=20)
    with observer.transaction():
        row = observer.get_conversation_by_id(cid, include_deleted=True)
        assert row is None if hard else bool(row["deleted"])
    receipt = _receipt(observer)
    assert receipt["conversation_id"] == (None if hard else cid)
    assert receipt["workspace_id"] == "ws"
    with observer.transaction() as conn:
        assert observer.workspace_chat_startups.count_receipts("user-1", conn=conn) == 1
    with pytest.raises(WorkspaceStartupError) as gone:
        _start(observer)
    assert gone.value.status_code == 404


@pytest.mark.parametrize("writer", ["restore", "sync"])
def test_committed_workspace_closure_blocks_receipt_resurrection_before_insert(
    creation_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, writer: str,
) -> None:
    """H2's real durable closure rejects admission before Sync can lock via INSERT."""
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.soft_delete_conversation(cid, 1)
    assert creation_db._begin_native_workspace_delete("ws", expected_version=2, hard=False)
    conn_type = type(creation_db.get_connection())
    inserted = False
    if creation_db.backend_type.value == "postgresql":
        execute = conn_type.execute

        def observe_execute(conn: Any, query: str, *args: Any, **kwargs: Any) -> Any:
            """Keep real execution; flag only attempts to insert the conversation."""
            nonlocal inserted
            if query.strip().startswith("INSERT INTO conversations"):
                inserted = True
            return execute(conn, query, *args, **kwargs)

        monkeypatch.setattr(conn_type, "execute", observe_execute)
    else:
        def trace_insert(statement: str) -> None:
            """Trace the real SQLite statement rather than a mocked writer."""
            nonlocal inserted
            if statement.strip().startswith("INSERT INTO conversations"):
                inserted = True

        creation_db.get_connection().set_trace_callback(trace_insert)
    try:
        with pytest.raises(ConflictError, match="workspace_chat_admission_unavailable"):
            _resurrect(creation_db, cid, writer, 2)
    finally:
        if creation_db.backend_type.value == "sqlite":
            creation_db.get_connection().set_trace_callback(None)
    assert not inserted
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid, include_deleted=True)
        assert bool(row["deleted"]) and row["version"] == 2


def test_postgres_sync_transfer_back_to_origin_and_replay_do_not_deadlock(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay waits on Sync's admission before chat lock, then rejects the burned key."""
    if creation_db.backend_type.value != "postgresql":
        pytest.skip("PostgreSQL Workspace/FK row-lock compatibility")
    first = _start(creation_db)
    cid = first.conversation["id"]
    creation_db.upsert_conversation_from_sync(
        conversation_id=cid, title="Moved", sync_client_id="other-device", object_revision=2,
        object_hash="h", assistant_kind="persona", assistant_id="persona-a",
        persona_memory_mode="read_only", scope_type="global",
    )
    replay_db, observer = db_factory(), db_factory()
    _warm_worker(observer)
    ready = [Event(), Event()]
    start = [Event(), Event()]
    admitted, replaying, release = Event(), Event(), Event()
    pids: list[int | None] = [None, None]
    recheck = creation_db.conversation_store._recheck_receipt_admission
    lock_workspace = replay_db.workspace_chat_startups.lock_receipt_workspace

    def pause_sync(*args: Any, **kwargs: Any) -> None:
        """Hold real destination/chat locks before the FK-bearing update."""
        recheck(*args, **kwargs)
        admitted.set()
        assert release.wait(20)

    def observe_replay(*args: Any, **kwargs: Any) -> Any:
        """Observe replay's actual NO KEY UPDATE Workspace admission."""
        replaying.set()
        return lock_workspace(*args, **kwargs)

    def run(index: int) -> Any:
        """Warm before overlap and use the actual orchestrator/Sync writer."""
        db = creation_db if index == 0 else replay_db
        pids[index] = _warm_worker(db)
        ready[index].set()
        assert start[index].wait(20)
        if index == 0:
            return _resurrect(db, cid, "sync", 2)
        return _start(db)

    monkeypatch.setattr(creation_db.conversation_store, "_recheck_receipt_admission", pause_sync)
    monkeypatch.setattr(replay_db.workspace_chat_startups, "lock_receipt_workspace", observe_replay)
    with ThreadPoolExecutor(max_workers=2) as executor:
        syncing, replay = executor.submit(run, 0), executor.submit(run, 1)
        try:
            assert all(event.wait(20) for event in ready)
            start[0].set()
            assert admitted.wait(20)
            start[1].set()
            assert replaying.wait(20)
            with observer.transaction() as conn:
                _assert_pg_blocker(conn, pids[1], pids[0], replay)
        finally:
            release.set()
            for event in start:
                event.set()
        syncing.result(timeout=20)
        with pytest.raises(WorkspaceStartupError) as changed:
            replay.result(timeout=20)
    assert changed.value.code == "workspace_chat_startup_changed"
    with observer.transaction():
        assert observer.get_conversation_by_id(cid)["workspace_id"] == "ws"


@pytest.mark.parametrize("closed", [False, True])
def test_postgres_receipt_appearing_after_sync_preflight_requires_fresh_admission(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch, closed: bool,
) -> None:
    """A real accepted receipt appearing before INSERT cannot bypass its new gate."""
    if creation_db.backend_type.value != "postgresql":
        pytest.skip("SQLite serializes writers before preflight")
    cid = "bdc16e67-8d63-4318-8958-4e8c22365f14"
    monkeypatch.setattr(chat_startup, "uuid4", lambda: cid)
    syncing = db_factory()
    ready, start, preflight, release = Event(), Event(), Event(), Event()
    admit = syncing.conversation_store._lock_receipt_admission
    attempts = 0

    def pause_unassociated(*args: Any, **kwargs: Any) -> bool:
        """Observe actual association, then let strict startup commit before INSERT."""
        nonlocal attempts
        associated = admit(*args, **kwargs)
        attempts += 1
        if attempts == 1:
            assert not associated
            preflight.set()
            assert release.wait(20)
        return associated

    def run_sync() -> None:
        """Warm this independent writer before allowing the preflight race."""
        _warm_worker(syncing)
        ready.set()
        assert start.wait(20)
        _resurrect(syncing, cid, "sync", 2)

    monkeypatch.setattr(syncing.conversation_store, "_lock_receipt_admission", pause_unassociated)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_sync)
        try:
            assert ready.wait(20)
            start.set()
            assert preflight.wait(20)
            first = _start(creation_db)
            assert first.conversation["id"] == cid
            if closed:
                assert creation_db._begin_native_workspace_delete("ws", expected_version=2, hard=False)
        finally:
            release.set()
            start.set()
        if closed:
            with pytest.raises(ConflictError, match="workspace_chat_admission_unavailable"):
                future.result(timeout=20)
        else:
            future.result(timeout=20)
    assert attempts == (1 if closed else 2)
    with creation_db.transaction():
        row = creation_db.get_conversation_by_id(cid)
        assert (row["version"], row["title"]) == (
            (1, first.conversation["title"]) if closed else (3, "Resurrected")
        )
    assert _receipt(creation_db)["invalidated_at"] is None


def test_postgres_hard_delete_waits_for_native_operation_before_chat_lock(
    creation_db: CharactersRAGDB, db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native soft and hard deletion cannot invert operation/chat row locks."""
    if creation_db.backend_type.value != "postgresql":
        pytest.skip("PostgreSQL row-lock ordering")
    cid = creation_db.add_conversation({"title": "Native child", "character_id": 1})
    owner = AuthorizedNativeOwner("user-1", "native:user-1", NativeScopeV1(kind="global"))
    with creation_db.transaction() as conn:
        creation_db.native_forks.reserve_operation(
            owner, "native_fork_v1", "operation-1", "sha256:one", {"title": "Native child"}, conn=conn,
        )
        conn.execute(
            "UPDATE native_chat_operations SET state = 'committed', child_conversation_id = ?, result_json = '{}' "
            "WHERE client_id = ? AND operation_id = ?", (cid, "user-1", "operation-1"),
        )
        conn.execute(
            "UPDATE conversations SET native_creation_operation_kind = 'native_fork_v1', "
            "native_creation_operation_id = 'operation-1', required_projection_version = 'native-fork-v1' "
            "WHERE id = ?", (cid,),
        )
    soft, hard, observer = db_factory(), db_factory(), db_factory()
    _warm_worker(observer)
    ready = [Event(), Event()]
    start = [Event(), Event()]
    soft_operation_locked, hard_operation_attempted, hard_chat_locked, release = Event(), Event(), Event(), Event()
    pids: list[int | None] = [None, None]
    soft_mark = soft.native_forks.mark_child_gone
    hard_mark = hard.native_forks.mark_child_gone
    hard_lock = hard.workspace_chat_startups.lock_conversation

    def hold_soft_operation(*args: Any, **kwargs: Any) -> None:
        """Hold the actual committed native operation lock before soft chat update."""
        soft_mark(*args, **kwargs)
        soft_operation_locked.set()
        assert release.wait(20)

    def observe_hard_operation(*args: Any, **kwargs: Any) -> None:
        """Signal immediately before the real native operation FOR UPDATE."""
        hard_operation_attempted.set()
        hard_mark(*args, **kwargs)

    def observe_hard_chat(*args: Any, **kwargs: Any) -> Any:
        """A hard delete must not hold this lock while blocked on operation."""
        row = hard_lock(*args, **kwargs)
        hard_chat_locked.set()
        return row

    def run(index: int) -> None:
        """Use independent warm driver connections and the public delete writers."""
        db = soft if index == 0 else hard
        pids[index] = _warm_worker(db)
        ready[index].set()
        assert start[index].wait(20)
        if index == 0:
            assert db.soft_delete_conversation(cid, 1)
        else:
            assert db.hard_delete_conversation(cid)

    monkeypatch.setattr(soft.native_forks, "mark_child_gone", hold_soft_operation)
    monkeypatch.setattr(hard.native_forks, "mark_child_gone", observe_hard_operation)
    monkeypatch.setattr(hard.workspace_chat_startups, "lock_conversation", observe_hard_chat)
    with ThreadPoolExecutor(max_workers=2) as executor:
        deleting_soft, deleting_hard = executor.submit(run, 0), executor.submit(run, 1)
        try:
            assert all(event.wait(20) for event in ready)
            start[0].set()
            assert soft_operation_locked.wait(20)
            start[1].set()
            assert hard_operation_attempted.wait(20)
            with observer.transaction() as conn:
                _assert_pg_blocker(conn, pids[1], pids[0], deleting_hard)
            assert not hard_chat_locked.is_set(), "hard delete locked the chat before the native operation"
        finally:
            release.set()
            for event in start:
                event.set()
        deleting_soft.result(timeout=20)
        deleting_hard.result(timeout=20)
    assert hard_chat_locked.is_set()
    assert observer.get_conversation_by_id(cid, include_deleted=True) is None
