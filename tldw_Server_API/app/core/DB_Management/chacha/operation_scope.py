"""Explicit PostgreSQL checkout ownership across a complete HTTP operation.

These scopes own checkouts, not transaction decisions. Returning an owned
checkout discards unfinished work using the backend pool's existing policy.
Legacy thread-local and explicitly borrowed connections are never adopted.
"""

import asyncio
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import Any, TypeVar

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send


class ClosedChaChaOperationError(RuntimeError):
    """An inherited operation no longer admits new commands."""


class ConcurrentChaChaTransactionError(RuntimeError):
    """Another thread or task already holds this operation's managed transaction (ADR-080)."""


@dataclass
class ConnectionState:
    """Captured state for one database within an explicit owner."""

    conn: Any = None
    backend_ref: Any = None
    tx_depth: int = 0
    borrowed: bool = False
    lock: Any = field(default_factory=RLock)
    allocation_lock: Any = field(default_factory=RLock)
    retiring: bool = False
    tx_owner: tuple[int, Any] | None = None
    _uses: dict[tuple[int, Any], int] = field(default_factory=dict)

    @staticmethod
    def _execution_key() -> tuple[int, Any]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return get_ident(), task

    def enter_transaction(self) -> bool:
        """Claim the managed transaction for this thread and task; True if this call claimed it.

        Every thread and task of an operation shares this one connection and one
        ``tx_depth``. A transaction entered while another thread's or task's is open
        would join it, and the depth each writes back on exit would leave every later
        transaction in the operation uncommitted. So that raises instead (ADR-080).
        Nesting on the owning thread and task joins as before.
        """
        key = self._execution_key()
        with self.lock:
            if self.tx_owner is None:
                self.tx_owner = key
                return True
            if self.tx_owner == key:
                return False
        # depth=2 reports the caller's `with db.transaction()` line, past BackendManagedTransaction.__enter__,
        # so a refusal a caller swallows as RuntimeError can still be traced to the overlapping code.
        logger.opt(depth=2).error("A ChaCha transaction overlapped another thread's or task's in one operation")
        raise ConcurrentChaChaTransactionError(
            "Another thread or task holds this ChaCha operation's transaction; "
            "give concurrent work its own chacha_operation(independent=True)"
        )

    def leave_transaction(self) -> None:
        """Release the claim taken by the outermost ``enter_transaction()``."""
        with self.lock:
            self.tx_owner = None

    def permits_current_use(self) -> bool:
        """Only already-entered work in this task/thread may finish after close."""
        with self.lock:
            return self._uses.get(self._execution_key(), 0) > 0

    @contextmanager
    def use(self) -> Iterator[None]:
        """Keep a captured checkout out of the pool until this command finishes."""
        key = self._execution_key()
        with self.lock:
            if self.retiring and not self._uses.get(key):
                raise ClosedChaChaOperationError("The ChaCha operation has finished")
            self._uses[key] = self._uses.get(key, 0) + 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            with self.lock:
                self._uses[key] -= 1
                if not self._uses[key]:
                    del self._uses[key]
                try:
                    if self.retiring and not self._uses:
                        self._return_checkout()
                except Exception:
                    if not failed:
                        raise
                    logger.error("ChaCha last-use cleanup failed while handling an earlier error")

    def _return_checkout(self) -> None:
        if self.borrowed or self.conn is None:
            return
        connection, backend = self.conn, self.backend_ref
        self.conn = self.backend_ref = None
        self.tx_depth = 0
        backend.get_pool().return_connection(connection)

    def release(self) -> None:
        """Return an owned checkout at most once, leaving borrowed state intact."""
        with self.lock:
            self.retiring = True
            if not self._uses:
                self._return_checkout()


@dataclass(frozen=True)
class ExternalConnection:
    """Explicitly lend an existing connection; its caller retains all decisions."""

    database: Any
    connection: Any
    backend: Any


class ChaChaOperation:
    """Capture only this operation's per-database PostgreSQL checkouts."""

    def __init__(self, bindings: tuple[ExternalConnection, ...] = ()) -> None:
        self._lock = RLock()
        self._closed = False
        self._states: dict[Any, ConnectionState] = {}
        for binding in bindings:
            if binding.connection is None or binding.backend is None:
                raise ValueError("External bindings require a connection and backend")
            if binding.database in self._states:
                raise ValueError("A database can only have one external binding")
            # A borrowed outer transaction is never owned by db.transaction(),
            # including when the caller has not issued its first statement yet.
            self._states[binding.database] = ConnectionState(
                conn=binding.connection, backend_ref=binding.backend, tx_depth=1, borrowed=True
            )

    def state_for(self, database: Any) -> ConnectionState:
        """Resolve a live state, including after context propagation to a worker."""
        with self._lock:
            if not self._closed:
                return self._states.setdefault(database, ConnectionState())
            state = self._states.get(database)
        # Check active use without nesting the owner and state locks.
        if state is None or not state.permits_current_use():
            raise ClosedChaChaOperationError("The ChaCha operation has finished")
        return state

    def close(self) -> None:
        """Finish once without a success commit or another owner's cleanup."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._states.values())
        first_error = None
        for state in states:
            try:
                state.release()
            except Exception as exc:  # noqa: BLE001 - finish every captured checkout before propagating
                # Finish other owned checkouts too; never log connection details.
                logger.error("ChaCha operation cleanup failed ({})", type(exc).__name__)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise RuntimeError("ChaCha operation cleanup failed") from first_error


_current_operation: ContextVar[ChaChaOperation | None] = ContextVar("chacha_operation", default=None)


def current_connection_state(database: Any) -> ConnectionState | None:
    """Return explicit state, or let legacy callers keep their current behavior."""
    owner = _current_operation.get()
    return owner.state_for(database) if owner is not None else None


def current_operation_holds_connection(database: Any) -> bool:
    """Whether the open current operation already holds a connection to ``database``.

    Read-only: unlike ``current_connection_state`` it never creates state.

    Args:
        database: The ChaCha database instance to look up in the current operation.

    Returns:
        True only when an operation is open in this context, is not closed, and has
        checked out a connection for ``database``; False otherwise.
    """
    owner = _current_operation.get()
    if owner is None:
        return False
    with owner._lock:
        state = None if owner._closed else owner._states.get(database)
    return state is not None and state.conn is not None


@contextmanager
def chacha_operation(
    *, independent: bool = False, bindings: tuple[ExternalConnection, ...] = ()
) -> Iterator[ChaChaOperation]:
    """Join the current owner, or explicitly create an independent operation.

    External bindings require a new scope so a nested call cannot replace a live
    owner's state. Detached tasks must start an independent scope after their
    inherited owner has finished.
    """
    existing = _current_operation.get()
    if existing is not None and not independent:
        if bindings:
            raise ValueError("External bindings require an independent operation")
        if existing._closed:
            raise ClosedChaChaOperationError("The ChaCha operation has finished")
        yield existing
        return
    owner = ChaChaOperation(bindings)
    token = _current_operation.set(owner)
    failed = False
    try:
        yield owner
    except BaseException:
        failed = True
        raise
    finally:
        try:
            owner.close()
        except Exception:
            if not failed:
                raise
        finally:
            _current_operation.reset(token)


@contextmanager
def outside_chacha_operation() -> Iterator[None]:
    """Run without the caller's operation, for work that may outlive it, such as a detached task.

    ChaCha calls inside take the thread-local fallback or open their own operations. Never
    wrap such work in one operation of its own instead: concurrent branches would share
    one connection and lose writes (ADR-080).
    """
    token = _current_operation.set(None)
    try:
        yield
    finally:
        _current_operation.reset(token)


_F = TypeVar("_F", bound=Callable[..., Any])


def current_operation_is_open() -> bool:
    """Whether an open ChaCha operation owns this context's PostgreSQL checkouts."""
    owner = _current_operation.get()
    return owner is not None and not owner._closed


def owns_chacha_checkouts(func: _F | None = None, *, independent: bool = False) -> Any:
    """Run ``func`` inside a ChaCha operation: the caller's, or its own outside one.

    For a unit of work outside HTTP (a job handler, a cancel check, a call that opens
    a handle of its own), so its PostgreSQL checkouts go back to the pool when it
    returns, on whichever thread they were taken. Decorate the ``async def`` itself:
    a lambda that returns a coroutine would leave the operation before the coroutine
    runs, and so would ``@chacha_operation()`` on an ``async def``.

    ``independent=True`` always opens a new operation. Use it for work that a detached
    task can reach after the request that started it has closed its operation.
    """
    if func is None:
        return functools.partial(owns_chacha_checkouts, independent=independent)
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def owned(*args: Any, **kwargs: Any) -> Any:
            with chacha_operation(independent=independent):
                return await func(*args, **kwargs)

        return owned
    return chacha_operation(independent=independent)(func)


class ChaChaOperationMiddleware:
    """Own the complete HTTP ASGI response, including stream/background work."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with chacha_operation(independent=True):
            await self.app(scope, receive, send)
