"""Operation lifetime and ownership controls that do not require PostgreSQL."""

import asyncio
import threading
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import (
    ChaChaOperationMiddleware,
    ClosedChaChaOperationError,
    ExternalConnection,
    chacha_operation,
    current_connection_state,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _checkout(database):
    state = current_connection_state(database)
    connection = Mock()
    pool = Mock()
    state.conn = connection
    state.backend_ref = SimpleNamespace(get_pool=lambda: pool)
    return state, connection, pool


@pytest.mark.parametrize("cancel", [False, True])
def test_repeated_or_cancelled_finalization_returns_each_checkout_once(cancel):
    db = object()
    with chacha_operation(independent=True) as owner:
        state, raw, pool = _checkout(db)
        with chacha_operation() as nested:
            assert nested is owner and current_connection_state(db) is state
        if cancel:

            async def cancelled():
                with chacha_operation():
                    raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                asyncio.run(cancelled())
        owner.close()
        owner.close()
        with pytest.raises(ClosedChaChaOperationError):
            current_connection_state(db)
    pool.return_connection.assert_called_once_with(raw)
    raw.commit.assert_not_called()
    assert current_connection_state(db) is None


@pytest.mark.parametrize("primary", [False, True])
def test_cleanup_attempts_all_checkouts_and_preserves_primary_exception(primary):
    one, two = object(), object()
    expected = ValueError if primary else RuntimeError
    with pytest.raises(expected, match="primary failure" if primary else "cleanup failed"):
        with chacha_operation(independent=True):
            _, first, first_pool = _checkout(one)
            _, second, second_pool = _checkout(two)
            first_pool.return_connection.side_effect = RuntimeError("Private pool failure")
            if primary:
                raise ValueError("primary failure")
    first_pool.return_connection.assert_called_once_with(first)
    second_pool.return_connection.assert_called_once_with(second)
    assert current_connection_state(one) is None


def test_independent_scope_does_not_adopt_an_existing_owner_or_external_binding():
    db = object()
    with chacha_operation(independent=True):
        original, raw, pool = _checkout(db)
        with chacha_operation(independent=True):
            assert current_connection_state(db).conn is None
        assert current_connection_state(db) is original
        with pytest.raises(ValueError, match="independent"):
            with chacha_operation(bindings=(ExternalConnection(db, raw, original.backend_ref),)):
                pass
        pool.return_connection.assert_not_called()
    pool.return_connection.assert_called_once_with(raw)


@pytest.mark.parametrize("scope_type", ["websocket", "lifespan"])
def test_non_http_asgi_work_does_not_gain_an_implicit_owner(scope_type):
    db = object()
    seen = []

    async def downstream(scope, receive, send):
        seen.append(current_connection_state(db))

    asyncio.run(ChaChaOperationMiddleware(downstream)({"type": scope_type}, None, None))
    assert seen == [None]


@pytest.mark.parametrize("minimal", [False, True], ids=["normal", "minimal"])
def test_actual_main_registers_one_operation_middleware_in_both_modes(monkeypatch, minimal):
    from tldw_Server_API.tests.helpers.app_main_state import (
        clear_app_main,
        import_app_main,
        restore_app_main,
        snapshot_app_main,
    )

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1" if minimal else "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    previous = snapshot_app_main()
    clear_app_main()
    try:
        main = import_app_main()
        registered = [entry for entry in main.app.user_middleware if entry.cls is ChaChaOperationMiddleware]
        assert len(registered) == 1
        assert any(route.path == "/health" for route in main.app.routes)
    finally:
        restore_app_main(previous)


def test_sqlite_explicit_transaction_retains_legacy_owner_after_operation(tmp_path):
    db = CharactersRAGDB(tmp_path / "sqlite-owner.db", client_id="1")
    try:
        note = db.add_note(title="Committed", content="SQLite control")
        raw = db.get_connection()
        raw.execute("BEGIN")
        with chacha_operation(independent=True):
            db.execute_query("UPDATE notes SET title=? WHERE id=?", ("Pending", note))
            assert db.get_connection() is raw
        assert raw.in_transaction
        assert db.get_note_by_id(note)["title"] == "Pending"
        raw.rollback()
        assert db.get_note_by_id(note)["title"] == "Committed"
    finally:
        db.close_all_connections()


def test_inherited_task_and_thread_cannot_adopt_an_active_parent_use_after_close():
    db = object()

    async def run():
        with chacha_operation(independent=True) as owner:
            state, raw, pool = _checkout(db)
            with state.use():
                owner.close()
                assert current_connection_state(db) is state
                pool.return_connection.assert_not_called()

                def rejected():
                    with pytest.raises(ClosedChaChaOperationError):
                        current_connection_state(db)

                async def child():
                    rejected()

                await asyncio.create_task(child())
                await asyncio.to_thread(rejected)
                assert current_connection_state(db) is state
            pool.return_connection.assert_called_once_with(raw)

    asyncio.run(run())


@pytest.mark.parametrize("primary", [False, True])
def test_deferred_last_use_cleanup_preserves_primary_exception_and_returns_once(primary):
    db = object()
    expected = ValueError if primary else RuntimeError
    with pytest.raises(expected, match="primary" if primary else "pool failure"):
        with chacha_operation(independent=True) as owner:
            state, raw, pool = _checkout(db)
            pool.return_connection.side_effect = RuntimeError("pool failure")
            with state.use():
                owner.close()
                owner.close()
                pool.return_connection.assert_not_called()
                if primary:
                    raise ValueError("primary")
    pool.return_connection.assert_called_once_with(raw)
    assert current_connection_state(db) is None


@pytest.mark.parametrize("failure_at", ["checkout", "wrapper"])
def test_failed_transaction_entry_does_not_leave_an_active_checkout_use(failure_at):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendManagedTransaction

    database = object()
    with chacha_operation(independent=True):
        state, raw, pool = _checkout(database)

        def fail():
            raise RuntimeError("entry failure")

        db = SimpleNamespace(
            _connection_state=lambda: state,
            _get_thread_connection=fail if failure_at == "checkout" else lambda: raw,
            _get_pinned_backend=fail,
        )
        transaction = BackendManagedTransaction(db)
        with pytest.raises(RuntimeError, match="entry failure"):
            with transaction:
                pytest.fail("Entry should fail")
    pool.return_connection.assert_called_once_with(raw)


@pytest.mark.parametrize("missing", ["connection", "backend"])
def test_incomplete_external_binding_fails_before_creating_a_borrowed_checkout(missing):
    db = object()
    connection = None if missing == "connection" else object()
    backend = None if missing == "backend" else object()
    with pytest.raises(ValueError, match="External"):
        with chacha_operation(independent=True, bindings=(ExternalConnection(db, connection, backend),)):
            pass


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
            holding_state.set()  # Pause actual acquisition before its pinned-backend lookup.
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
            except BaseException as exc:  # noqa: BLE001 - assert worker failures on the parent thread
                errors.append(f"active:{type(exc).__name__}:{exc}")

        def late():
            try:
                current_connection_state(db)
                errors.append("late lookup unexpectedly admitted")
            except ClosedChaChaOperationError:
                pass
            except BaseException as exc:  # noqa: BLE001 - assert worker failures on the parent thread
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
