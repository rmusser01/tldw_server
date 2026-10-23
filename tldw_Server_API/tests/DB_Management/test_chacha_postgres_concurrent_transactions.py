"""Two managed transactions open at once on one operation's connection state (TASK-13899).

Inside one chacha_operation every thread and task shares one ConnectionState, so one
connection and one transaction depth. A second db.transaction() that enters while
another thread's or task's is open must raise, not join it: joined, it leaves the
depth wrong, and every later transaction in the operation is silently rolled back.
"""

import asyncio
import threading
from contextvars import copy_context

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration
_INSERT = "INSERT INTO tx_probe (name) VALUES (?)"


@pytest.fixture
def tx_db(pg_database_config, tmp_path):
    """A handle on its own pool, and a reader of what was committed."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    backend.execute("CREATE TABLE tx_probe (name TEXT NOT NULL)")
    db = CharactersRAGDB(tmp_path / "tx.db", client_id="1", backend=backend)
    db.close_connection()

    def committed():
        return [row["name"] for row in backend.execute("SELECT name FROM tx_probe ORDER BY name").rows]

    try:
        yield db, committed
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _overlap_on_two_threads(db):
    """The second transaction enters while the first is open, and leaves after it."""
    first_open, second_entered, first_left = threading.Event(), threading.Event(), threading.Event()
    outcome = []

    def first():
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r1",))
            first_open.set()
            second_entered.wait(5)
        first_left.set()

    def second():
        first_open.wait(5)
        try:
            with db.transaction() as conn:
                second_entered.set()
                first_left.wait(5)
                conn.execute(_INSERT, ("r2",))
        except Exception as exc:  # noqa: BLE001 - the guard's error class does not exist before the fix
            outcome.append(type(exc).__name__)
        finally:
            second_entered.set()

    threads = [threading.Thread(target=copy_context().run, args=(fn,)) for fn in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return outcome


async def _overlap_on_two_tasks(db):
    """The same interleaving between two tasks on the event-loop thread."""
    first_open, second_entered, first_left = asyncio.Event(), asyncio.Event(), asyncio.Event()
    outcome = []

    async def first():
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r1",))
            first_open.set()
            await second_entered.wait()
        first_left.set()

    async def second():
        await first_open.wait()
        try:
            with db.transaction() as conn:
                second_entered.set()
                await first_left.wait()
                conn.execute(_INSERT, ("r2",))
        except Exception as exc:  # noqa: BLE001 - the guard's error class does not exist before the fix
            outcome.append(type(exc).__name__)
        finally:
            second_entered.set()

    await asyncio.wait_for(asyncio.gather(first(), second()), timeout=10)
    return outcome


@pytest.mark.parametrize("overlap", ["two-threads", "two-tasks-on-one-thread"])
def test_an_overlapping_transaction_raises_and_no_write_is_lost(tx_db, overlap):
    db, committed = tx_db
    with chacha_operation(independent=True):
        if overlap == "two-threads":
            outcome = _overlap_on_two_threads(db)
        else:
            outcome = asyncio.run(_overlap_on_two_tasks(db))
        # A later transaction in the same operation must still commit.
        with db.transaction() as conn:
            conn.execute(_INSERT, ("r3",))

    assert (outcome, committed()) == (["ConcurrentChaChaTransactionError"], ["r1", "r3"])
