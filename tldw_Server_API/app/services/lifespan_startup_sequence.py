"""
Startup sequence orchestration extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
    LifespanWorkerRuntimeState,
)


@dataclass
class LifespanStartupSequenceHandles:
    """Startup handles that remain needed after the helper returns."""

    db_pool: Any | None = None
    session_manager: Any | None = None
    heavy_startup_handles: Any | None = None


async def run_lifespan_startup_sequence(
    *,
    app: Any,
    worker_runtime: LifespanWorkerRuntimeState,
    module_file: str,
    logger: Any,
    readiness_state: dict[str, Any],
    shared_is_truthy: Any,
    route_enabled: Any,
    get_mcp_config: Any,
    validate_mcp_config: Any,
    test_mode: bool,
    run_pg_rls_auto_ensure: Any,
    register_owned_job_poller: Any,
    replace_owned_job_poller_inventory: Any,
    startup_api_key_log_value: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> LifespanStartupSequenceHandles:
    """Run the pre-core, core-init, and worker-bootstrap startup sequence."""
    from tldw_Server_API.app.services.startup_pre_core import prepare_startup_pre_core

    defer_heavy = await prepare_startup_pre_core(
        app=app,
        logger=logger,
        readiness_state=readiness_state,
        shared_is_truthy=shared_is_truthy,
        route_enabled=route_enabled,
        get_mcp_config=get_mcp_config,
        validate_mcp_config=validate_mcp_config,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
        test_mode=test_mode,
    )

    from tldw_Server_API.app.services.startup_core_initialization import (
        initialize_startup_core_components,
    )

    startup_core_handles = await initialize_startup_core_components(
        app=app,
        module_file=module_file,
        logger=logger,
        route_enabled=route_enabled,
        defer_heavy=defer_heavy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )

    from tldw_Server_API.app.services.startup_worker_bootstrap import (
        initialize_startup_worker_bootstrap,
    )

    startup_worker_bootstrap_handles = await initialize_startup_worker_bootstrap(
        app=app,
        test_mode=test_mode,
        route_enabled=route_enabled,
        run_pg_rls_auto_ensure=run_pg_rls_auto_ensure,
        register_owned_job_poller=register_owned_job_poller,
        replace_owned_job_poller_inventory=replace_owned_job_poller_inventory,
        logger=logger,
        startup_api_key_log_value=startup_api_key_log_value,
        shared_is_truthy=shared_is_truthy,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )
    worker_runtime.apply_startup_worker_bootstrap_handles(
        startup_worker_bootstrap_handles,
    )

    return LifespanStartupSequenceHandles(
        db_pool=startup_core_handles.db_pool,
        session_manager=startup_core_handles.session_manager,
        heavy_startup_handles=startup_core_handles.heavy_startup_handles,
    )
