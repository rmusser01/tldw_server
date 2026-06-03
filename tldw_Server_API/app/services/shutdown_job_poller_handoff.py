"""
Job-poller shutdown handoff helper extracted from the application lifespan.
"""

from __future__ import annotations

import os as _env_os
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase


@dataclass
class JobPollerShutdownHandoffHandles:
    """Outputs produced by the job-poller shutdown handoff."""

    early_quiesced_job_poller_names: set[str] = field(default_factory=set)
    should_run_late_stop: Callable[[str, Any], bool] = lambda *_args, **_kwargs: False


async def shutdown_job_poller_handoff(
    *,
    app: Any,
    worker_lifecycle_session: Any | None,
    lifecycle_worker_engine: Any | None,
    quiesce_owned_job_pollers_for_shutdown: Callable[..., Awaitable[None]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> JobPollerShutdownHandoffHandles:
    """Quiesce owned job pollers and return the late-stop gating predicate."""
    if worker_lifecycle_session is None or lifecycle_worker_engine is None:
        return JobPollerShutdownHandoffHandles()

    wait_for_leases_sec = _resolve_wait_for_leases_sec(startup_guard_exceptions)
    count_active_processing = _build_count_active_processing(import_exceptions)
    job_poller_handles = _filter_job_poller_quiesce_handles(
        list(worker_lifecycle_session.handles_for_phase(ShutdownPhase.JOB_POLLER_QUIESCE))
    )

    async def _stop_job_pollers_with_lifecycle_engine(
        _app: Any,
        _poller_handles: list[Any],
    ) -> None:
        await lifecycle_worker_engine.stop_phase(
            worker_lifecycle_session,
            ShutdownPhase.JOB_POLLER_QUIESCE,
        )

    await quiesce_owned_job_pollers_for_shutdown(
        app,
        job_poller_handles,
        wait_for_leases_sec=wait_for_leases_sec,
        count_active_processing=count_active_processing,
        stop_registered_job_pollers=_stop_job_pollers_with_lifecycle_engine,
    )

    early_quiesced_job_poller_names = set(
        getattr(app.state, "_tldw_shutdown_quiesced_job_poller_names", [])
    )
    return JobPollerShutdownHandoffHandles(
        early_quiesced_job_poller_names=early_quiesced_job_poller_names,
        should_run_late_stop=_build_should_run_late_stop(early_quiesced_job_poller_names),
    )


def _filter_job_poller_quiesce_handles(handles: list[Any]) -> list[Any]:
    """Return task-backed workers owned by the job-poller quiesce phase."""

    return [
        handle
        for handle in handles
        if getattr(handle, "task", None) is not None
        and _is_job_poller_quiesce_handle(handle)
    ]


def _is_job_poller_quiesce_handle(handle: Any) -> bool:
    try:
        shutdown_phase = ShutdownPhase(
            getattr(handle, "shutdown_phase", ShutdownPhase.JOB_POLLER_QUIESCE)
        )
    except (TypeError, ValueError):
        return False
    return shutdown_phase is ShutdownPhase.JOB_POLLER_QUIESCE


async def run_shutdown_job_poller_handoff(
    *,
    app: Any,
    worker_lifecycle_session: Any | None,
    lifecycle_worker_engine: Any | None,
    quiesce_owned_job_pollers_for_shutdown: Callable[..., Awaitable[None]],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> JobPollerShutdownHandoffHandles:
    """Run the shutdown job-poller handoff with main-lifespan fallback behavior."""
    try:
        return await shutdown_job_poller_handoff(
            app=app,
            worker_lifecycle_session=worker_lifecycle_session,
            lifecycle_worker_engine=lifecycle_worker_engine,
            quiesce_owned_job_pollers_for_shutdown=quiesce_owned_job_pollers_for_shutdown,
            startup_guard_exceptions=startup_guard_exceptions,
            import_exceptions=import_exceptions,
        )
    except (startup_guard_exceptions + import_exceptions) as exc:
        logger.debug(f"Job-poller shutdown handoff skipped: {exc}")
        return JobPollerShutdownHandoffHandles()


def _resolve_wait_for_leases_sec(
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> int:
    try:
        return int(_env_os.getenv("JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC", "0") or "0")
    except startup_guard_exceptions:
        return 0


def _build_count_active_processing(
    import_exceptions: tuple[type[BaseException], ...],
) -> Callable[[], int]:
    try:
        return _load_shutdown_job_manager()().count_active_processing
    except import_exceptions:
        return lambda: 0


def _load_shutdown_job_manager() -> type[Any]:
    from tldw_Server_API.app.core.Jobs.manager import JobManager as _ShutdownJM

    return _ShutdownJM


def _build_should_run_late_stop(
    early_quiesced_job_poller_names: set[str],
) -> Callable[[str, Any], bool]:
    def _should_run_late_stop(task_name: str, task: Any) -> bool:
        return bool(task) and task_name not in early_quiesced_job_poller_names

    return _should_run_late_stop
