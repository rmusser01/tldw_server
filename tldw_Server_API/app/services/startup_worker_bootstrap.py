"""
Startup worker-bootstrap helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tldw_Server_API.app.services.worker_registry import WorkerRegistry


@dataclass
class StartupWorkerBootstrapHandles:
    """Combined handles returned from the worker-bootstrap startup block."""

    app_settings: Any
    owned_job_pollers: list[Any]
    startup_worker_group_handles: Any
    startup_service_tail_handles: Any
    worker_inventory: WorkerRegistry | None = None


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
    """Run the owned-poller, worker-group, and service-tail bootstrap in legacy order."""
    worker_inventory = WorkerRegistry(app)
    owned_job_pollers = worker_inventory.handles
    app_settings = _load_app_settings()
    startup_worker_group_handles = await _start_worker_groups(
        app=app,
        app_settings=app_settings,
        test_mode=test_mode,
        route_enabled=route_enabled,
        startup_guard_exceptions=startup_guard_exceptions,
        owned_job_pollers=owned_job_pollers,
        worker_inventory=worker_inventory,
        register_owned_job_poller=register_owned_job_poller,
    )
    startup_service_tail_handles = await _initialize_startup_service_tail(
        app=app,
        app_settings=app_settings,
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        owned_job_pollers=owned_job_pollers,
        worker_inventory=worker_inventory,
        register_owned_job_poller=register_owned_job_poller,
        startup_worker_group_handles=startup_worker_group_handles,
        replace_owned_job_poller_inventory=replace_owned_job_poller_inventory,
        test_mode=test_mode,
        logger=logger,
        startup_api_key_log_value=startup_api_key_log_value,
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    return StartupWorkerBootstrapHandles(
        app_settings=app_settings,
        owned_job_pollers=owned_job_pollers,
        startup_worker_group_handles=startup_worker_group_handles,
        startup_service_tail_handles=startup_service_tail_handles,
        worker_inventory=worker_inventory,
    )


def _load_app_settings():
    from tldw_Server_API.app.core.config import settings as app_settings

    return app_settings


async def _start_worker_groups(**kwargs):
    from tldw_Server_API.app.services.startup_worker_groups import start_worker_groups

    return await start_worker_groups(**kwargs)


async def _initialize_startup_service_tail(**kwargs):
    from tldw_Server_API.app.services.startup_service_tail import (
        initialize_startup_service_tail,
    )

    return await initialize_startup_service_tail(**kwargs)
