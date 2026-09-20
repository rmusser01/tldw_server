"""Compatibility boundary for preflight timeout contexts."""

from __future__ import annotations

import asyncio
from typing import Any


class _TimeoutContext:
    """Normalize the stdlib and async-timeout expiration interfaces."""

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


def timeout(delay: float | None) -> _TimeoutContext:
    """Return a timeout context on Python 3.10 through current Python."""
    stdlib_timeout = getattr(asyncio, "timeout", None)
    if callable(stdlib_timeout):
        return _TimeoutContext(stdlib_timeout(delay))

    from async_timeout import timeout as async_timeout

    return _TimeoutContext(async_timeout(delay))


__all__ = ["timeout"]
