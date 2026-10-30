"""App-scoped service container.

Holds the process's long-lived services in explicit slots. ``main.lifespan``
publishes it as ``app.state.services``; module getters such as
``get_db_pool()`` are the fallback for code with no request or constructor
argument and read the same slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


@dataclass
class AppServices:
    """One slot per service; each slot is ``None`` until its getter fills it."""

    db_pool: DatabasePool | None = None


_SERVICES = AppServices()


def app_services() -> AppServices:
    """Return the process-wide service container."""
    return _SERVICES
