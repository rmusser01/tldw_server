"""Explicit PostgreSQL checkout ownership across a complete HTTP operation.

These scopes own checkouts, not transaction decisions. Returning an owned
checkout discards unfinished work using the backend pool's existing policy.
Legacy thread-local and explicitly borrowed connections are never adopted.
"""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import Any

from loguru import logger
from starlette.types import ASGIApp, Receive, Scope, Send


class ClosedChaChaOperationError(RuntimeError):
    """An inherited operation no longer admits new commands."""


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
    _uses: dict[tuple[int, Any], int] = field(default_factory=dict)

    @staticmethod
    def _execution_key() -> tuple[int, Any]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return get_ident(), task

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
