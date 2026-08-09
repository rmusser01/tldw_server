"""Thread-safe cancellation for strict MCP gateway requests."""

from __future__ import annotations

import asyncio
import threading

_MAX_REASON_CODE_POINTS = 128


class GatewayCancellationToken:
    """Share one cancellation signal between dispatch and runtime work."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._reason: str | None = None
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []

    @property
    def cancelled(self) -> bool:
        """Return whether the request has been cancelled."""

        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str | None:
        """Return the first bounded diagnostic cancellation reason."""

        with self._lock:
            return self._reason

    def cancel(self, reason: str | None = None) -> bool:
        """Cancel once and wake all waiters, including from a worker thread."""

        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or None")
        bounded_reason = reason[:_MAX_REASON_CODE_POINTS] if reason is not None else None
        with self._lock:
            if self._cancelled:
                return False
            self._cancelled = True
            self._reason = bounded_reason
            waiters = self._waiters
            self._waiters = []

        for loop, waiter in waiters:
            try:
                loop.call_soon_threadsafe(self._wake_waiter, waiter)
            except RuntimeError:
                continue
        return True

    def is_cancelled(self) -> bool:
        """Return whether the request has been cancelled."""

        return self.cancelled

    def raise_if_cancelled(self) -> None:
        """Raise ``CancelledError`` when cancellation has been requested."""

        if self.cancelled:
            raise asyncio.CancelledError

    async def wait(self) -> None:
        """Wait until cancellation is requested."""

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        with self._lock:
            if self._cancelled:
                return
            self._waiters.append((loop, waiter))
        try:
            await waiter
        finally:
            with self._lock:
                try:
                    self._waiters.remove((loop, waiter))
                except ValueError:
                    pass

    @staticmethod
    def _wake_waiter(waiter: asyncio.Future[None]) -> None:
        """Resolve one waiter on its owning event-loop thread."""

        if not waiter.done():
            waiter.set_result(None)


__all__ = ["GatewayCancellationToken"]
