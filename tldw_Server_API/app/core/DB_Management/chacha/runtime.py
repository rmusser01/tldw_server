from __future__ import annotations

import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar


T = TypeVar("T")


class ChaChaRuntimeUnavailableError(RuntimeError):
    """Raised when ChaChaNotes resources are unavailable for new work."""


class ChaChaRuntimeManager:
    """Small lifecycle gate around ChaChaNotes runtime resource creation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._shutting_down = False

    async def get_or_create(
        self,
        factory: Callable[..., T | Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a DB factory unless the runtime is shutting down."""
        with self._lock:
            if self._shutting_down:
                raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

        result = factory(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def shutdown(self) -> None:
        """Stop accepting new ChaChaNotes initialization work."""
        with self._lock:
            self._shutting_down = True

    def snapshot(self) -> dict[str, Any]:
        """Return the runtime lifecycle state for diagnostics."""
        with self._lock:
            return {"shutting_down": self._shutting_down}

    def reset_for_tests(self) -> None:
        """Reset lifecycle state between tests that reuse the same process."""
        with self._lock:
            self._shutting_down = False
