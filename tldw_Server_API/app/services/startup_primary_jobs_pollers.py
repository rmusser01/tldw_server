"""
Primary jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    route_enabled_predicate,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import WorkerRegistry

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_TRUTHY_ENV_VALUES = {"true", "1", "yes", "y", "on"}


@dataclass
class PrimaryJobsPollerHandles:
    """Startup-owned job poller handles used later in shutdown flow."""

    core_jobs_stop_event: Any | None = None
    core_jobs_task: Any | None = None
    files_jobs_stop_event: Any | None = None
    files_jobs_task: Any | None = None
    data_tables_jobs_stop_event: Any | None = None
    data_tables_jobs_task: Any | None = None
    prompt_studio_jobs_stop_event: Any | None = None
    prompt_studio_jobs_task: Any | None = None
    workspace_file_inventory_jobs_stop_event: Any | None = None
    workspace_file_inventory_jobs_task: Any | None = None


def provide_primary_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for primary jobs pollers.

    Core jobs preserve the legacy sidecar gate through
    ``context.sidecar_mode``. Omitted settings default to the
    existing non-sidecar behavior.
    """

    return (
        stop_event_worker_spec(
            name="core_jobs_task",
            worker_service=_run_chatbooks_core_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=_core_jobs_worker_enabled,
        ),
        stop_event_worker_spec(
            name="files_jobs_task",
            worker_service=_run_file_artifacts_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate("FILES_JOBS_WORKER_ENABLED", "files"),
        ),
        stop_event_worker_spec(
            name="data_tables_jobs_task",
            worker_service=_run_data_tables_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "DATA_TABLES_JOBS_WORKER_ENABLED",
                "data-tables",
            ),
        ),
        stop_event_worker_spec(
            name="prompt_studio_jobs_task",
            worker_service=_run_prompt_studio_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "PROMPT_STUDIO_JOBS_WORKER_ENABLED",
                "prompt-studio",
            ),
        ),
        stop_event_worker_spec(
            name="workspace_file_inventory_jobs_task",
            worker_service=_run_workspace_file_inventory_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED",
                "workspaces",
            ),
        ),
    )


def _core_jobs_worker_enabled(context: WorkerLifecycleContext) -> bool:
    backend = (
        os.getenv("CHATBOOKS_JOBS_BACKEND")
        or os.getenv("TLDW_JOBS_BACKEND")
        or ""
    ).lower()
    is_core = backend == "core" or not backend
    core_worker_enabled = (
        os.getenv("CHATBOOKS_CORE_WORKER_ENABLED", "true").lower()
        in _TRUTHY_ENV_VALUES
    )
    return is_core and core_worker_enabled and not context.sidecar_mode


async def start_primary_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[[str, str], bool],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> PrimaryJobsPollerHandles:
    """Start the first owned jobs pollers and return their handles."""

    core_jobs_stop_event, core_jobs_task = await _start_core_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    files_jobs_stop_event, files_jobs_task = await _start_files_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    data_tables_jobs_stop_event, data_tables_jobs_task = await _start_data_tables_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    prompt_studio_jobs_stop_event, prompt_studio_jobs_task = await _start_prompt_studio_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    workspace_file_inventory_jobs_stop_event, workspace_file_inventory_jobs_task = (
        await _start_workspace_file_inventory_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
            worker_inventory=worker_inventory,
        )
    )
    return PrimaryJobsPollerHandles(
        core_jobs_stop_event=core_jobs_stop_event,
        core_jobs_task=core_jobs_task,
        files_jobs_stop_event=files_jobs_stop_event,
        files_jobs_task=files_jobs_task,
        data_tables_jobs_stop_event=data_tables_jobs_stop_event,
        data_tables_jobs_task=data_tables_jobs_task,
        prompt_studio_jobs_stop_event=prompt_studio_jobs_stop_event,
        prompt_studio_jobs_task=prompt_studio_jobs_task,
        workspace_file_inventory_jobs_stop_event=workspace_file_inventory_jobs_stop_event,
        workspace_file_inventory_jobs_task=workspace_file_inventory_jobs_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _start_core_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        backend = (os.getenv("CHATBOOKS_JOBS_BACKEND") or os.getenv("TLDW_JOBS_BACKEND") or "").lower()
        is_core = (backend == "core") or (not backend)
        core_worker_enabled = os.getenv("CHATBOOKS_CORE_WORKER_ENABLED", "true").lower() in _TRUTHY_ENV_VALUES
        if sidecar_mode:
            core_worker_enabled = False
        if not is_core or not core_worker_enabled:
            logger.info("Core Jobs worker (Chatbooks) disabled by backend selection or flag")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="core_jobs_task",
                task_name="core_jobs_task",
                coroutine_factory=_run_chatbooks_core_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Core Jobs worker (Chatbooks) started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_chatbooks_core_jobs_worker_service(stop_event))
        logger.info("Core Jobs worker (Chatbooks) started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="core_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start core Jobs worker (Chatbooks): {exc}")
        return None, None


async def _start_files_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[[str, str], bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the File Artifacts jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("FILES_JOBS_WORKER_ENABLED", "files")
        if not enabled:
            logger.info("File Artifacts Jobs worker disabled by flag (FILES_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="files_jobs_task",
                task_name="files_jobs_task",
                coroutine_factory=_run_file_artifacts_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("File Artifacts Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_file_artifacts_jobs_worker_service(stop_event))
        logger.info("File Artifacts Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="files_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start File Artifacts Jobs worker: {exc}")
        return None, None


async def _start_data_tables_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[[str, str], bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Data Tables jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("DATA_TABLES_JOBS_WORKER_ENABLED", "data-tables")
        if not enabled:
            logger.info("Data Tables Jobs worker disabled by flag (DATA_TABLES_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="data_tables_jobs_task",
                task_name="data_tables_jobs_task",
                coroutine_factory=_run_data_tables_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Data Tables Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_data_tables_jobs_worker_service(stop_event))
        logger.info("Data Tables Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="data_tables_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Data Tables Jobs worker: {exc}")
        return None, None


async def _start_prompt_studio_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[[str, str], bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Prompt Studio jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("PROMPT_STUDIO_JOBS_WORKER_ENABLED", "prompt-studio")
        if not enabled:
            logger.info("Prompt Studio Jobs worker disabled by flag (PROMPT_STUDIO_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="prompt_studio_jobs_task",
                task_name="prompt_studio_jobs_task",
                coroutine_factory=_run_prompt_studio_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Prompt Studio Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_prompt_studio_jobs_worker_service(stop_event))
        logger.info("Prompt Studio Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="prompt_studio_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Prompt Studio Jobs worker: {exc}")
        return None, None


async def _start_workspace_file_inventory_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[[str, str], bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Workspace file inventory jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker(
            "WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED",
            "workspace_file_inventory_jobs_task",
        )
        if not enabled:
            logger.info(
                "Workspace file inventory Jobs worker disabled by flag "
                "(WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="workspace_file_inventory_jobs_task",
                task_name="workspace_file_inventory_jobs_task",
                coroutine_factory=_run_workspace_file_inventory_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Workspace file inventory Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_workspace_file_inventory_jobs_worker_service(stop_event))
        logger.info("Workspace file inventory Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="workspace_file_inventory_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Workspace file inventory Jobs worker: {exc}")
        return None, None


def _run_chatbooks_core_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.core_jobs_worker import (
        run_chatbooks_core_jobs_worker as _run_chatbooks_core_jobs_worker,
    )

    return _run_chatbooks_core_jobs_worker(stop_event)


def _run_file_artifacts_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.File_Artifacts.jobs_worker import (
        run_file_artifacts_jobs_worker as _run_file_artifacts_jobs_worker,
    )

    return _run_file_artifacts_jobs_worker(stop_event)


def _run_data_tables_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Data_Tables.jobs_worker import (
        run_data_tables_jobs_worker as _run_data_tables_jobs_worker,
    )

    return _run_data_tables_jobs_worker(stop_event)


def _run_prompt_studio_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services.jobs_worker import (
        run_prompt_studio_jobs_worker as _run_prompt_studio_jobs_worker,
    )

    return _run_prompt_studio_jobs_worker(stop_event)


def _run_workspace_file_inventory_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.workspace_file_inventory_jobs_worker import (
        run_workspace_file_inventory_jobs_worker as _run_workspace_file_inventory_jobs_worker,
    )

    return _run_workspace_file_inventory_jobs_worker(stop_event)
