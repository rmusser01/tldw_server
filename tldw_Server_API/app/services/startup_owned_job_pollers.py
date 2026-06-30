"""
Startup owned job-poller preparation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def prepare_startup_owned_job_pollers(
    *,
    app: Any,
    publish_shutdown_job_poller_inventory: Callable[[Any, list[Any]], None],
) -> list[Any]:
    handles: list[Any] = []
    publish_shutdown_job_poller_inventory(app, handles)
    return handles
