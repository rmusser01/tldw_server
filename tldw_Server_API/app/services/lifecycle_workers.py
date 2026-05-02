"""Lifecycle worker registration, inventory publication, and shutdown helpers.

The helpers in this module centralize the repeated FastAPI lifespan pattern of
creating a stop event, starting a named task, publishing diagnostic inventory,
and later stopping registered workers with bounded cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS


class ShutdownPhase(str, Enum):
    """Shutdown phase that owns a registered worker."""

    JOB_POLLER_QUIESCE = "job_poller_quiesce"
    BACKGROUND_WORKER_SHUTDOWN = "background_worker_shutdown"


@dataclass
class ManagedWorker:
    """Runtime handle for one lifecycle-managed worker task."""

    name: str
    task: asyncio.Task[Any] | None
    stop_event: asyncio.Event | None = None
    shutdown_callback: Callable[[], Awaitable[None]] | None = None
    timeout_sec: float = 5.0
    category: str | None = None
    shutdown_phase: ShutdownPhase = ShutdownPhase.JOB_POLLER_QUIESCE


class WorkerInventory:
    """Owns lifecycle-managed worker handles and app-state inventory publication."""

    def __init__(
        self,
        app: Any,
        handles: Sequence[ManagedWorker] | None = None,
    ) -> None:
        self._app = app
        self._handles = list(handles or [])
        self.publish()

    @property
    def handles(self) -> list[ManagedWorker]:
        """Return the mutable handle list shared with legacy startup paths."""

        return self._handles

    def register(self, worker: ManagedWorker | None = None, **worker_kwargs: Any) -> ManagedWorker:
        """Register an already-started worker and republish app-state inventory."""

        if worker is None:
            worker = ManagedWorker(**worker_kwargs)
        self._handles.append(worker)
        self.publish()
        return worker

    def replace_phase(
        self,
        shutdown_phase: ShutdownPhase | str,
        handles: Sequence[ManagedWorker],
    ) -> None:
        """Replace all workers owned by one shutdown phase."""

        target_phase = _normalize_shutdown_phase(shutdown_phase)
        self._handles[:] = [
            handle
            for handle in self._handles
            if _normalize_shutdown_phase(handle.shutdown_phase) is not target_phase
        ]
        self._handles.extend(handles)
        self.publish()

    def handles_for_phase(self, shutdown_phase: ShutdownPhase | str) -> list[ManagedWorker]:
        """Return workers currently owned by a shutdown phase."""

        target_phase = _normalize_shutdown_phase(shutdown_phase)
        return [
            handle
            for handle in self._handles
            if _normalize_shutdown_phase(handle.shutdown_phase) is target_phase
        ]

    def publish(self) -> None:
        """Publish full and compatibility inventories to app state."""

        publish_worker_inventory(self._app, self._handles)


class WorkerRegistry(WorkerInventory):
    """Current-code registry facade for Phase 2.1 worker migrations.

    The original issue references a standalone ``worker_registry.py`` module.
    Current startup/shutdown ownership uses the worker inventory model instead,
    so this facade exposes the WorkerRegistry naming and ``register_custom``
    hook without reintroducing the removed stale implementation.
    """

    async def register_custom(
        self,
        *,
        name: str,
        task_name: str,
        coroutine_factory: Callable[[asyncio.Event], Awaitable[Any]],
        timeout_sec: float = 5.0,
        category: str | None = None,
        shutdown_phase: ShutdownPhase = ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    ) -> tuple[asyncio.Task[Any], asyncio.Event]:
        """Start and register a custom stop-event worker."""

        return await start_stop_event_worker(
            self,
            name=name,
            task_name=task_name,
            coroutine_factory=coroutine_factory,
            timeout_sec=timeout_sec,
            category=category,
            shutdown_phase=shutdown_phase,
        )


def publish_worker_inventory(app: Any, handles: Sequence[ManagedWorker]) -> None:
    """Publish full worker metadata and the job-poller compatibility view."""

    full_inventory = [
        {
            "name": handle.name,
            "task_name": _task_name(handle.task),
            "has_stop_event": handle.stop_event is not None,
            "timeout_sec": handle.timeout_sec,
            "category": handle.category,
            "shutdown_phase": _normalize_shutdown_phase(handle.shutdown_phase).value,
        }
        for handle in handles
    ]
    job_poller_inventory = [
        {
            "name": handle.name,
            "task_name": _task_name(handle.task),
            "has_stop_event": handle.stop_event is not None,
            "timeout_sec": handle.timeout_sec,
        }
        for handle in handles
        if _normalize_shutdown_phase(handle.shutdown_phase) is ShutdownPhase.JOB_POLLER_QUIESCE
    ]

    try:
        app.state._tldw_shutdown_worker_inventory = full_inventory
    except LIFECYCLE_GUARD_EXCEPTIONS as exc:
        _log_state_publication_failure("_tldw_shutdown_worker_inventory", exc)

    try:
        app.state._tldw_shutdown_job_poller_inventory = job_poller_inventory
    except LIFECYCLE_GUARD_EXCEPTIONS as exc:
        _log_state_publication_failure("_tldw_shutdown_job_poller_inventory", exc)


async def stop_registered_workers(
    app: Any,
    handles: Sequence[ManagedWorker],
    *,
    stopped_names_attr: str,
    log_label: str,
) -> None:
    """Stop registered workers concurrently and publish stopped worker names."""

    async def _request_worker_stop(handle: ManagedWorker) -> bool | None:
        if handle.stop_event is not None:
            handle.stop_event.set()
            if handle.task is None:
                return True
            return None
        if handle.shutdown_callback is not None:
            try:
                await asyncio.wait_for(handle.shutdown_callback(), timeout=handle.timeout_sec)
                if handle.task is None:
                    return True
                return None
            except asyncio.TimeoutError:
                logger.warning(
                    "App Shutdown: Timed out waiting for {} {} shutdown callback after {}s; cancelling",
                    log_label,
                    handle.name,
                    handle.timeout_sec,
                )
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                logger.warning(
                    "App Shutdown: {} {} shutdown callback was cancelled",
                    log_label,
                    handle.name,
                )
            except Exception as exc:  # noqa: BLE001 - shutdown hooks must not block teardown.
                logger.warning(
                    "App Shutdown: {} {} shutdown callback failed: {}",
                    log_label,
                    handle.name,
                    exc,
                )
        if handle.task is None:
            return False
        try:
            handle.task.cancel()
        except Exception as exc:  # noqa: BLE001 - cancel hooks can raise arbitrary errors.
            logger.warning(
                "App Shutdown: {} {} cancel request failed: {}",
                log_label,
                handle.name,
                exc,
            )
        return None

    async def _await_worker_shutdown(handle: ManagedWorker, stop_result: bool | None) -> bool:
        if handle.task is None:
            return bool(stop_result)
        try:
            await asyncio.wait_for(asyncio.shield(handle.task), timeout=handle.timeout_sec)
        except asyncio.CancelledError:
            return bool(handle.task.done())
        except asyncio.TimeoutError:
            logger.warning(
                "App Shutdown: Timed out waiting for {} {} after {}s; cancelling",
                log_label,
                handle.name,
                handle.timeout_sec,
            )
            handle.task.cancel()
            try:
                await asyncio.wait_for(handle.task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "App Shutdown: {} {} did not cancel within 1.0s after timeout",
                    log_label,
                    handle.name,
                )
            except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
                logger.warning(
                    "App Shutdown: {} {} raised after cancellation: {}",
                    log_label,
                    handle.name,
                    exc,
                )
        except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
            logger.warning(
                "App Shutdown: {} {} exited during shutdown: {}",
                log_label,
                handle.name,
                exc,
            )
        return bool(handle.task.done())

    stop_results = await asyncio.gather(
        *(_request_worker_stop(handle) for handle in handles),
        return_exceptions=False,
    )

    stopped_results = await asyncio.gather(
        *(
            _await_worker_shutdown(handle, stop_result)
            for handle, stop_result in zip(handles, stop_results)
        ),
        return_exceptions=False,
    )
    try:
        setattr(
            app.state,
            stopped_names_attr,
            [
                handle.name
                for handle, stopped in zip(handles, stopped_results)
                if stopped
            ],
        )
    except LIFECYCLE_GUARD_EXCEPTIONS as exc:
        _log_state_publication_failure(stopped_names_attr, exc)


async def start_stop_event_worker(
    inventory: WorkerInventory,
    *,
    name: str,
    task_name: str,
    coroutine_factory: Callable[[asyncio.Event], Awaitable[Any]],
    timeout_sec: float = 5.0,
    category: str | None = None,
    shutdown_phase: ShutdownPhase = ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
) -> tuple[asyncio.Task[Any], asyncio.Event]:
    """Create a stop event, start a task, register it, and return both handles."""

    stop_event = asyncio.Event()
    task = asyncio.create_task(coroutine_factory(stop_event), name=task_name)
    inventory.register(
        ManagedWorker(
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
            category=category,
            shutdown_phase=shutdown_phase,
        )
    )
    return task, stop_event


def _task_name(task: asyncio.Task[Any] | None) -> str | None:
    if task is None:
        return None
    get_name = getattr(task, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return task.__class__.__name__


def _normalize_shutdown_phase(shutdown_phase: ShutdownPhase | str) -> ShutdownPhase:
    if isinstance(shutdown_phase, ShutdownPhase):
        return shutdown_phase
    return ShutdownPhase(str(shutdown_phase))


def _log_state_publication_failure(attr_name: str, exc: BaseException) -> None:
    logger.debug("Lifecycle worker metadata publication skipped for {}: {}", attr_name, exc)
