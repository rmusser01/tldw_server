"""
Startup worker-bootstrap helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
from tldw_Server_API.app.services.lifecycle_worker_session import WorkerLifecycleSession
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)


@dataclass
class StartupWorkerBootstrapHandles:
    """Combined handles returned from the worker-bootstrap startup block."""

    app_settings: Any
    owned_job_pollers: list[Any]
    startup_worker_group_handles: Any
    startup_service_tail_handles: Any
    worker_inventory: Any | None = None
    worker_lifecycle_session: WorkerLifecycleSession | None = None


async def initialize_startup_worker_bootstrap(
    *,
    app: Any,
    test_mode: bool,
    route_enabled: Callable[..., bool],
    run_pg_rls_auto_ensure: Callable[..., Any],
    register_owned_job_poller: Callable[..., None],
    replace_owned_job_poller_inventory: Callable[..., None],
    logger: Any,
    startup_api_key_log_value: Any,
    shared_is_truthy: Callable[..., bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> StartupWorkerBootstrapHandles:
    """Start lifecycle workers and run non-worker startup tail setup."""
    app_settings = _load_app_settings()
    context = WorkerLifecycleContext(
        app=app,
        test_mode=test_mode,
        settings=app_settings,
        route_enabled=route_enabled,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    worker_specs = _collect_startup_worker_specs(context)
    worker_lifecycle_session = await _start_lifecycle_workers(context, worker_specs)
    owned_job_pollers = list(worker_lifecycle_session.handles_by_name.values())
    await _run_startup_non_worker_tail(
        app=app,
        app_settings=app_settings,
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        logger=logger,
        startup_api_key_log_value=startup_api_key_log_value,
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    return StartupWorkerBootstrapHandles(
        app_settings=app_settings,
        owned_job_pollers=owned_job_pollers,
        startup_worker_group_handles=None,
        startup_service_tail_handles=None,
        worker_inventory=worker_lifecycle_session,
        worker_lifecycle_session=worker_lifecycle_session,
    )


def _load_app_settings():
    from tldw_Server_API.app.core.config import settings as app_settings

    return app_settings


def _collect_startup_worker_specs(
    context: WorkerLifecycleContext,
) -> tuple[WorkerSpec, ...]:
    from tldw_Server_API.app.services.startup_worker_groups import (
        collect_startup_worker_specs,
    )

    return collect_startup_worker_specs(context)


async def _start_lifecycle_workers(
    context: WorkerLifecycleContext,
    specs: tuple[WorkerSpec, ...],
) -> WorkerLifecycleSession:
    return await LifecycleWorkerEngine().start(context, specs)


async def _run_startup_non_worker_tail(
    *,
    app: Any,
    app_settings: Any,
    run_pg_rls_auto_ensure: Callable[..., Any],
    logger: Any,
    startup_api_key_log_value: Any,
    shared_is_truthy: Callable[..., bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _run_startup_infra_non_worker_setup(
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
    )
    await _report_startup_environment(
        app=app,
        logger=logger,
        startup_api_key_log_value=startup_api_key_log_value,
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )


async def _run_startup_infra_non_worker_setup(**kwargs: Any) -> None:
    from tldw_Server_API.app.services.startup_infra_services import (
        run_startup_infra_non_worker_setup,
    )

    await run_startup_infra_non_worker_setup(**kwargs)


async def _report_startup_environment(**kwargs: Any) -> None:
    from tldw_Server_API.app.services.startup_environment_reporting import (
        report_startup_environment,
    )

    await report_startup_environment(**kwargs)
