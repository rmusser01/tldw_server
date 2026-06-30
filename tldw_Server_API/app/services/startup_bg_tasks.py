"""
Startup background-task container preparation extracted from the application lifespan.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any


def prepare_startup_bg_tasks(
    *,
    app: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    with suppress(*startup_guard_exceptions):
        app.state.bg_tasks = {}
