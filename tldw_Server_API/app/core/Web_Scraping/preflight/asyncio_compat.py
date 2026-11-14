"""Compatibility boundary for preflight timeout contexts."""

from __future__ import annotations

import asyncio
import math
from typing import Any


class _TimeoutContext:
    """Normalize the stdlib and async-timeout public timeout interfaces."""

    def __init__(self, native_timeout: Any) -> None:
        self._native_timeout = native_timeout

    async def __aenter__(self) -> _TimeoutContext:
        await self._native_timeout.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self._native_timeout.__aexit__(*args)

    def expired(self) -> bool:
        native_expired = self._native_timeout.expired
        return bool(native_expired() if callable(native_expired) else native_expired)

    def reschedule(self, deadline: float) -> None:
        """Set a finite absolute deadline in event-loop time coordinates."""
        if not math.isfinite(deadline):
            raise ValueError("timeout deadline must be finite")
        if hasattr(self._native_timeout, "reschedule"):
            self._native_timeout.reschedule(deadline)
        else:
            self._native_timeout.update(deadline)


def timeout(delay: float | None) -> _TimeoutContext:
    """Return a timeout context on Python 3.10 through current Python."""
    stdlib_timeout = getattr(asyncio, "timeout", None)
    if callable(stdlib_timeout):
        return _TimeoutContext(stdlib_timeout(delay))

    from async_timeout import timeout as async_timeout

    return _TimeoutContext(async_timeout(delay))


__all__ = ["timeout"]
