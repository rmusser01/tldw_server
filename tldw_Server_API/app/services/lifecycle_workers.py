from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from loguru import logger

from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS


class ShutdownPhase(str, Enum):
    JOB_POLLER_QUIESCE = "job_poller_quiesce"
    BACKGROUND_WORKER_SHUTDOWN = "background_worker_shutdown"


@dataclass
class ManagedWorker:
    name: str
    task: asyncio.Task[Any]
    stop_event: asyncio.Event | None = None
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
        return self._handles

    def register(self, worker: ManagedWorker | None = None, **worker_kwargs: Any) -> ManagedWorker:
        if worker is None:
            worker = ManagedWorker(**worker_kwargs)
        self._handles.append(worker)
        self.publish()
        return worker

    def replace_phase(
        self,
        shutdown_phase: ShutdownPhase,
        handles: Sequence[ManagedWorker],
    ) -> None:
        self._handles[:] = [
            handle
            for handle in self._handles
            if _normalize_shutdown_phase(handle.shutdown_phase) is not shutdown_phase
        ]
        self._handles.extend(handles)
        self.publish()

    def handles_for_phase(self, shutdown_phase: ShutdownPhase) -> list[ManagedWorker]:
        return [
            handle
            for handle in self._handles
            if _normalize_shutdown_phase(handle.shutdown_phase) is shutdown_phase
        ]

    def publish(self) -> None:
        publish_worker_inventory(self._app, self._handles)


def publish_worker_inventory(app: Any, handles: Sequence[ManagedWorker]) -> None:
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
    except LIFECYCLE_GUARD_EXCEPTIONS:
        pass

    try:
        app.state._tldw_shutdown_job_poller_inventory = job_poller_inventory
    except LIFECYCLE_GUARD_EXCEPTIONS:
        pass


async def stop_registered_workers(
    app: Any,
    handles: Sequence[ManagedWorker],
    *,
    stopped_names_attr: str,
    log_label: str,
) -> None:
    async def _await_worker_shutdown(handle: ManagedWorker) -> bool:
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
            except LIFECYCLE_GUARD_EXCEPTIONS as exc:
                logger.debug(
                    "App Shutdown: {} cancel guard triggered for {}: {}",
                    log_label,
                    handle.name,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "App Shutdown: {} {} raised after cancellation: {}",
                    log_label,
                    handle.name,
                    exc,
                )
        except LIFECYCLE_GUARD_EXCEPTIONS as exc:
            logger.debug(
                "App Shutdown: {} stop guard triggered for {}: {}",
                log_label,
                handle.name,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "App Shutdown: {} {} exited during shutdown: {}",
                log_label,
                handle.name,
                exc,
            )
        return bool(handle.task.done())

    for handle in handles:
        if handle.stop_event is not None:
            handle.stop_event.set()
        else:
            with suppress(*LIFECYCLE_GUARD_EXCEPTIONS):
                handle.task.cancel()

    stopped_results = await asyncio.gather(
        *(_await_worker_shutdown(handle) for handle in handles),
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
    except LIFECYCLE_GUARD_EXCEPTIONS:
        pass


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


def _task_name(task: asyncio.Task[Any]) -> str:
    get_name = getattr(task, "get_name", None)
    if callable(get_name):
        return str(get_name())
    return task.__class__.__name__


def _normalize_shutdown_phase(shutdown_phase: ShutdownPhase | str) -> ShutdownPhase:
    if isinstance(shutdown_phase, ShutdownPhase):
        return shutdown_phase
    return ShutdownPhase(str(shutdown_phase))
