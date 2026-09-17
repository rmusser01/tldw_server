"""Actual owner/state/get-connection lock ordering with no database I/O."""

import threading
from contextvars import copy_context
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
    ClosedChaChaOperationError,
    chacha_operation,
    current_connection_state,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class BoundedOwnerLock:
    """Keep real RLock semantics; bound an otherwise deadlocked test wait."""

    def __init__(self):
        self.lock = threading.RLock()

    def __enter__(self):
        if not self.lock.acquire(timeout=1):
            raise RuntimeError("owner/state lock order deadlock")
        return self

    def __exit__(self, *_):
        self.lock.release()


def test_active_connection_acquisition_and_late_closed_owner_lookup_do_not_deadlock(monkeypatch):
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    db._local = threading.local()
    raw = SimpleNamespace(closed=False)
    db._backend.get_pool = lambda: SimpleNamespace(return_connection=lambda _: None)
    active = threading.Event()
    proceed = threading.Event()
    holding_state = threading.Event()
    checking_late_state = threading.Event()
    errors = []
    original_acquire = db._get_connection_for_state

    with chacha_operation(independent=True) as owner:
        state = current_connection_state(db)
        state.conn, state.backend_ref = raw, db._backend
        owner._lock = BoundedOwnerLock()
        original_permits = state.permits_current_use

        def acquire(captured):
            holding_state.set()  # Actual _get_thread_connection holds state.lock.
            assert checking_late_state.wait(2)
            return original_acquire(captured)

        def permits():
            if threading.current_thread().name == "late-owner-probe":
                checking_late_state.set()
            return original_permits()

        monkeypatch.setattr(db, "_get_connection_for_state", acquire)
        monkeypatch.setattr(state, "permits_current_use", permits)

        def work():
            try:
                with state.use():
                    active.set()
                    assert proceed.wait(2)
                    assert db._get_thread_connection() is raw
            except BaseException as exc:
                errors.append(f"active:{type(exc).__name__}:{exc}")

        def late():
            try:
                current_connection_state(db)
                errors.append("late lookup unexpectedly admitted")
            except ClosedChaChaOperationError:
                pass
            except BaseException as exc:
                errors.append(f"late:{type(exc).__name__}:{exc}")

        active_context, late_context = copy_context(), copy_context()
        worker = threading.Thread(target=lambda: active_context.run(work), name="active-owner-probe")
        rejected = threading.Thread(target=lambda: late_context.run(late), name="late-owner-probe")
        worker.start()
        assert active.wait(2)
        owner.close()
        proceed.set()
        assert holding_state.wait(2)
        rejected.start()
        worker.join(3)
        rejected.join(3)
        assert not worker.is_alive() and not rejected.is_alive()
        assert errors == []
