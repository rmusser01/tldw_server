"""Legacy thread-local PostgreSQL checkouts go back to the pool (TASK-13861 AC2).

The thread-exit cases need no garbage collection: CPython clears a dead thread's
thread-locals at once. A dropped handle is different: its stores hold it, so only the
cyclic collector frees it, and that case is a backstop that the test drives with
gc.collect(). Returns are queued by the finalizer and made by the next checkout or
close_connection() on any handle, so each test ends with one of those.
"""

import asyncio
import gc
import os
import queue
import sys
import threading
import time
from collections import Counter
from types import SimpleNamespace

import anyio
import psycopg
import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def ledger_backend(pg_database_config, monkeypatch):
    """A three-connection backend whose pool records every checkout and return."""
    pg_database_config.pool_size = 3
    pg_database_config.pool_timeout = 3.0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    taken, returned = Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    backend.taken, backend.returned = taken, returned
    try:
        yield backend
    finally:
        pool.close_all()


def _outstanding(backend):
    return sum((backend.taken - backend.returned).values())


def _session_state(config, backend_pid, settle_seconds=3.0):
    """The server's state for one session: None once it has ended, waiting up to settle_seconds for that."""
    deadline = time.monotonic() + settle_seconds
    with psycopg.connect(
        host=config.pg_host,
        port=config.pg_port,
        dbname=config.pg_database,
        user=config.pg_user,
        password=config.pg_password or None,
        autocommit=True,
    ) as admin:
        while True:
            row = admin.execute("SELECT state FROM pg_stat_activity WHERE pid = %s", (backend_pid,)).fetchone()
            state = row[0] if row else None
            if state is None or time.monotonic() >= deadline:
                return state
            time.sleep(0.1)


def _run_in_plain_thread(work):
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()


def _run_on_anyio_worker(work):
    async def main():
        await anyio.to_thread.run_sync(work)

    anyio.run(main)


def _run_on_default_executor(work):
    async def main():
        await asyncio.get_running_loop().run_in_executor(None, work)

    asyncio.run(main())


@pytest.mark.parametrize(
    "run", [_run_in_plain_thread, _run_on_anyio_worker, _run_on_default_executor], ids=["thread", "anyio", "executor"]
)
def test_a_legacy_checkout_returns_to_the_pool_when_its_thread_exits(ledger_backend, tmp_path, run):
    db = CharactersRAGDB(tmp_path / "exit.db", client_id="1", backend=ledger_backend)
    db.close_connection()
    before = _outstanding(ledger_backend)

    run(lambda: db.get_character_card_by_name("nobody"))
    db.close_connection()  # no checkout on this thread: it only makes the queued return

    assert _outstanding(ledger_backend) == before


def test_a_dropped_handle_returns_its_legacy_checkout_when_collected(ledger_backend, tmp_path, monkeypatch):
    """Backstop for per-call handles that are never closed; production must close or own them."""
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: ledger_backend)
    for call in range(5):  # more calls than the pool has connections
        handle = CharactersRAGDB(tmp_path / f"call{call}.db", client_id="1")
        handle.execute_query("SELECT count(*) AS n FROM character_cards")
        handle = None
        gc.collect()  # the handle is cyclic garbage; nothing but the collector frees it
    CharactersRAGDB(tmp_path / "last.db", client_id="1").close_connection()

    assert _outstanding(ledger_backend) == 0


def test_a_collected_guard_never_calls_the_pool_itself(ledger_backend, tmp_path):
    """A collection can run the finalizer on a thread that holds the pool's own lock.

    The finalizer must only queue the return; calling into the pool there can deadlock.
    """
    db = CharactersRAGDB(tmp_path / "queued.db", client_id="1", backend=ledger_backend)
    db.get_character_card_by_name("nobody")  # this thread's legacy checkout, now armed
    db = None
    returns_before = sum(ledger_backend.returned.values())

    gc.collect()

    assert sum(ledger_backend.returned.values()) == returns_before, "the finalizer called into the pool"
    CharactersRAGDB(tmp_path / "drain.db", client_id="1", backend=ledger_backend).close_connection()
    assert _outstanding(ledger_backend) == 0


def test_an_explicitly_returned_checkout_is_returned_once(ledger_backend, tmp_path):
    """Every explicit return disarms the guard, so a later thread exit returns nothing twice."""
    db = CharactersRAGDB(tmp_path / "once.db", client_id="1", backend=ledger_backend)
    db.close_connection()

    def work():
        db.get_character_card_by_name("nobody")
        db._local.conn.close()  # the closed-connection branch releases and replaces it
        db.get_character_card_by_name("nobody")
        db.close_connection()

    _run_in_plain_thread(work)
    db.close_connection()

    assert _outstanding(ledger_backend) == 0
    # The pool hands the same connection out again, so compare per checkout, not per object.
    assert not ledger_backend.returned - ledger_backend.taken, "a checkout was returned twice"


@pytest.mark.parametrize(
    "caller_holds_checkout", [False, True], ids=["caller-holds-none", "close-all-while-caller-holds-one"]
)
def test_a_dead_threads_session_does_not_stay_idle_in_a_transaction(
    ledger_backend, pg_database_config, tmp_path, caller_holds_checkout
):
    """A read leaves the session idle in a transaction, with its locks and its snapshot.

    That session must end with its thread, not wait for the next ChaCha call in the process.
    """
    db = CharactersRAGDB(tmp_path / "idle.db", client_id="1", backend=ledger_backend)
    db.close_connection()
    if caller_holds_checkout:
        db.execute_query("SELECT count(*) AS n FROM character_cards")
    sessions = []

    def work():
        db.execute_query("SELECT count(*) AS n FROM character_cards")
        info = db._local.conn.info
        sessions.append((info.backend_pid, info.transaction_status.name))

    _run_in_plain_thread(work)
    if caller_holds_checkout:
        db.close_all_connections()  # discards the caller's own checkout and closes the pool
    backend_pid, status = sessions[0]

    assert status == "INTRANS"
    assert _session_state(pg_database_config, backend_pid) is None


def test_a_guard_armed_in_another_process_neither_closes_nor_queues(monkeypatch):
    """After fork() a child frees the parent's thread-local slots; those sessions are the parent's."""
    dropped = queue.SimpleQueue()
    monkeypatch.setattr(chacha_module, "_DROPPED_CHECKOUTS", dropped)
    closed = []
    parent_pid = os.getpid() + 1
    with monkeypatch.context() as parent:
        parent.setattr(os, "getpid", lambda: parent_pid)
        guard = chacha_module._LegacyCheckout(SimpleNamespace(close=lambda: closed.append(True)), object())

    del guard  # the child drops the slot

    assert (closed, dropped.qsize()) == ([], 0)


def test_a_guard_finalized_at_interpreter_teardown_never_raises(monkeypatch):
    """Module globals can already be gone when a late finalizer runs."""
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    guard = chacha_module._LegacyCheckout(SimpleNamespace(close=lambda: None), object())
    monkeypatch.setattr(chacha_module, "_DROPPED_CHECKOUTS", None)

    del guard

    assert unraisable == []


def test_a_guard_dropped_off_its_owning_thread_queues_without_closing(monkeypatch):
    """The owner may be inside libpq on that connection; finishing it from here would free it under the owner."""
    dropped = queue.SimpleQueue()
    monkeypatch.setattr(chacha_module, "_DROPPED_CHECKOUTS", dropped)
    conn = SimpleNamespace(closed=False)
    conn.close = lambda: setattr(conn, "closed", True)
    armed = []
    _run_in_plain_thread(lambda: armed.append(chacha_module._LegacyCheckout(conn, "backend")))

    armed.clear()  # this thread, not the owner, drops the guard

    assert [(backend, queued.closed) for backend, queued, _owner in [dropped.get_nowait()]] == [("backend", False)]


def test_close_all_connections_leaves_a_live_threads_checkout_to_that_thread(ledger_backend, tmp_path, monkeypatch):
    """close_all_connections() on a caller's pool must not drop another live thread's slot.

    That thread may be inside libpq on its checkout: psycopg's C generators keep the raw PGconn
    across yields and call libpq without the GIL, so returning or closing the connection from here
    would free it under them. The thread's own guard closes it when the thread exits.
    """
    dropped = queue.SimpleQueue()
    monkeypatch.setattr(chacha_module, "_DROPPED_CHECKOUTS", dropped)
    db = CharactersRAGDB(tmp_path / "live.db", client_id="1", backend=ledger_backend)
    db.close_connection()
    ready, release, seen = threading.Event(), threading.Event(), []

    def owner():
        db.execute_query("SELECT count(*) AS n FROM character_cards")
        conn = db._local.conn
        ready.set()
        release.wait(timeout=30)
        seen.append((conn, conn.closed))

    worker = threading.Thread(target=owner)
    worker.start()
    ready.wait(timeout=30)
    db.close_all_connections()  # a caller-owned pool: it is closed, but no other thread's slot is dropped
    queued_while_owner_lives = dropped.qsize()
    release.set()
    worker.join()

    assert (queued_while_owner_lives, [(was_closed, conn.closed) for conn, was_closed in seen]) == (0, [(False, True)])


def test_a_drain_leaves_a_checkout_queued_while_its_owner_thread_lives(monkeypatch):
    """A checkout dropped off its owning thread goes back to the pool only once that thread has exited."""
    dropped = queue.SimpleQueue()
    monkeypatch.setattr(chacha_module, "_DROPPED_CHECKOUTS", dropped)
    returned = []
    backend = SimpleNamespace(get_pool=lambda: SimpleNamespace(return_connection=returned.append))
    conn = SimpleNamespace(closed=False)
    conn.close = lambda: setattr(conn, "closed", True)
    armed, ready, release = [], threading.Event(), threading.Event()

    def owner():
        armed.append(chacha_module._LegacyCheckout(conn, backend))
        ready.set()
        release.wait(timeout=30)

    worker = threading.Thread(target=owner)
    worker.start()
    ready.wait(timeout=30)
    armed.clear()  # dropped here, off the owner's thread, while the owner lives
    chacha_module._return_dropped_checkouts()
    while_owner_lives = (list(returned), conn.closed, dropped.qsize())
    release.set()
    worker.join()
    chacha_module._return_dropped_checkouts()

    assert (while_owner_lives, returned) == (([], False, 1), [conn])
