"""
Worker lifecycle runtime state used by the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LifespanWorkerRuntimeState:
    """Mutable lifecycle state shared across startup and shutdown."""

    worker_lifecycle_session: Any | None = None

    def apply_startup_worker_bootstrap_handles(self, handles: Any) -> None:
        self.worker_lifecycle_session = getattr(handles, "worker_lifecycle_session", None)
