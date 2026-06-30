"""
Heavy-startup deferral policy extracted from the application lifespan.
"""

from __future__ import annotations

import os
from typing import Any, Callable


def resolve_deferred_heavy_startup(
    *,
    shared_is_truthy: Callable[[Any], bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> bool:
    try:
        disable_heavy_startup = shared_is_truthy(_getenv("DISABLE_HEAVY_STARTUP"))
        if disable_heavy_startup:
            return False
        return bool(shared_is_truthy(_getenv("DEFER_HEAVY_STARTUP")))
    except startup_guard_exceptions:
        return False


def _getenv(key: str) -> str | None:
    return os.getenv(key)
