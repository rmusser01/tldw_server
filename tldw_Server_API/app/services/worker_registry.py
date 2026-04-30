"""Compatibility exports for current lifecycle worker registry support."""

from __future__ import annotations

from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase,
    WorkerRegistry,
)

__all__ = ["ManagedWorker", "ShutdownPhase", "WorkerRegistry"]
