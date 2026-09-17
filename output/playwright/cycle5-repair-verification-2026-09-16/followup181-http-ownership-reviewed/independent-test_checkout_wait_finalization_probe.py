"""Actual async owner finalization while fake pool acquisition is pending."""

import asyncio
import threading
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def test_owner_close_does_not_wait_for_an_active_worker_to_acquire_a_checkout():
    entered = threading.Event()
    closing = threading.Event()
    closed = threading.Event()
    release = threading.Event()
    blocked = []
    returns = []
    raw = SimpleNamespace(closed=False)

    def acquire():
        entered.set()
        assert release.wait(3)
        return raw

    pool = SimpleNamespace(get_connection=acquire, return_connection=returns.append)
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, get_pool=lambda: pool)
    db._local = threading.local()
    db._uses_shared_content_backend = False
    db.client_id = ""  # Fake raw connection has no session/config driver I/O.

    def controller():
        assert closing.wait(3)
        blocked.append(not closed.wait(0.3))
        release.set()

    control = threading.Thread(target=controller)
    control.start()

    async def run():
        with chacha_operation(independent=True) as owner:
            worker = asyncio.create_task(asyncio.to_thread(db.get_connection))
            assert await asyncio.to_thread(entered.wait, 3)
            closing.set()
            owner.close()
            closed.set()
            await worker

    try:
        asyncio.run(run())
    finally:
        release.set()
        control.join(3)
    assert not control.is_alive()
    assert returns == [raw]
    assert blocked == [False], "Owner close waited for the worker's blocked pool acquisition"
