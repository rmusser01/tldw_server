"""
Background worker lifecycle registry.

Extracted from main.py lifespan to centralize the repetitive pattern of:
  1. Check if worker is enabled (env flag + route policy)
  2. Import the worker function lazily
  3. Create asyncio.Event + asyncio.create_task
  4. Track for shutdown

Usage in main.py lifespan::

    from tldw_Server_API.app.services.worker_registry import WorkerRegistry

    registry = WorkerRegistry(sidecar_mode=_sidecar_mode)
    registry.register("files", "FILES_JOBS_WORKER_ENABLED", "files",
                       "tldw_Server_API.app.core.File_Artifacts.jobs_worker",
                       "run_file_artifacts_jobs_worker")
    await registry.start_all()
    ...
    yield
    ...
    await registry.stop_all()
"""

from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger


@dataclass
class WorkerEntry:
    """A registered background worker."""
    name: str
    env_flag: str
    route_key: str
    module_path: str
    function_name: str
    default_stable: bool = True
    task: asyncio.Task | None = None
    stop_event: asyncio.Event | None = None


class WorkerRegistry:
    """Manages background worker lifecycle (start/stop).

    Args:
        sidecar_mode: If True, all workers are disabled (external sidecar handles them).
        test_mode: If True, workers are disabled by default.
    """

    def __init__(self, *, sidecar_mode: bool = False, test_mode: bool = False):
        self._entries: list[WorkerEntry] = []
        self._sidecar_mode = sidecar_mode
        self._test_mode = test_mode

    def register(
        self,
        name: str,
        env_flag: str,
        route_key: str,
        module_path: str,
        function_name: str,
        *,
        default_stable: bool = True,
    ) -> None:
        """Register a worker for later startup."""
        self._entries.append(WorkerEntry(
            name=name,
            env_flag=env_flag,
            route_key=route_key,
            module_path=module_path,
            function_name=function_name,
            default_stable=default_stable,
        ))

    def _is_enabled(self, entry: WorkerEntry) -> bool:
        """Check if a worker should start based on flags and policy."""
        if self._sidecar_mode:
            return False
        if self._test_mode:
            return False

        # Check env flag
        raw = os.getenv(entry.env_flag)
        if raw is not None and raw.strip():
            return raw.strip().lower() in {"true", "1", "yes", "y", "on"}

        # Fall back to route policy
        try:
            from tldw_Server_API.app.core.config import route_enabled
            return bool(route_enabled(entry.route_key, default_stable=entry.default_stable))
        except Exception:  # noqa: BLE001
            return entry.default_stable

    async def start_all(self) -> int:
        """Start all enabled workers. Returns count of workers started."""
        started = 0
        for entry in self._entries:
            if not self._is_enabled(entry):
                logger.info(f"Worker '{entry.name}' disabled by policy/flag")
                continue
            try:
                mod = importlib.import_module(entry.module_path)
                run_fn = getattr(mod, entry.function_name)
                entry.stop_event = asyncio.Event()
                entry.task = asyncio.create_task(run_fn(entry.stop_event))
                started += 1
                logger.info(f"Worker '{entry.name}' started")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to start worker '{entry.name}': {e}")
        return started

    async def stop_all(self, timeout: float = 10.0) -> int:
        """Stop all running workers gracefully. Returns count stopped."""
        stopped = 0
        for entry in reversed(self._entries):
            if entry.task is None:
                continue
            try:
                # Signal stop
                if entry.stop_event is not None:
                    entry.stop_event.set()
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(entry.task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Worker '{entry.name}' did not stop within {timeout}s, cancelling")
                    entry.task.cancel()
                    try:
                        await entry.task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                stopped += 1
                logger.info(f"Worker '{entry.name}' stopped")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error stopping worker '{entry.name}': {e}")
                stopped += 1  # Count as stopped even if error
        return stopped

    @property
    def running_workers(self) -> list[str]:
        """Names of workers with active tasks."""
        return [e.name for e in self._entries if e.task is not None and not e.task.done()]

    def get_task(self, name: str) -> asyncio.Task | None:
        """Get the task for a named worker (for legacy shutdown integration)."""
        for entry in self._entries:
            if entry.name == name:
                return entry.task
        return None

    def get_stop_event(self, name: str) -> asyncio.Event | None:
        """Get the stop event for a named worker (for legacy shutdown integration)."""
        for entry in self._entries:
            if entry.name == name:
                return entry.stop_event
        return None
