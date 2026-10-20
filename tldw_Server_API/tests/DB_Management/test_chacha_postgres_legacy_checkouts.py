"""Legacy thread-local PostgreSQL checkouts go back to the pool (TASK-13861 AC2).

The thread-exit cases need no garbage collection: CPython clears a dead thread's
thread-locals at once. A dropped handle is different: its stores hold it, so only the
cyclic collector frees it, and that case is a backstop that the test drives with
gc.collect(). Returns are queued by the finalizer and made by the next checkout or
close_connection() on any handle, so each test ends with one of those.
"""

import asyncio
import gc
import threading
from collections import Counter

import anyio
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
