"""
Startup sequence orchestration extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
    LifespanWorkerRuntimeState,
)
from tldw_Server_API.app.services.startup_warning_registry import StartupWarningRegistry


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
    startup_warning_registry = StartupWarningRegistry(startup_id=uuid4().hex)
    app.state.startup_warning_registry = startup_warning_registry

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
    app.state.startup_sandbox_orchestrator = getattr(
        startup_core_handles,
        "startup_sandbox_orchestrator",
        None,
    )
    _run_startup_warning_producers(
        app=app,
        startup_core_handles=startup_core_handles,
    )
    _raise_if_startup_blocked(startup_warning_registry)

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


def _run_startup_warning_producers(*, app: Any, startup_core_handles: Any) -> None:
    from tldw_Server_API.app.services.startup_warning_sandbox import (
        produce_sandbox_startup_warnings,
    )

    registry = app.state.startup_warning_registry
    produce_sandbox_startup_warnings(
        orchestrator=getattr(startup_core_handles, "startup_sandbox_orchestrator", None),
        registry=registry,
    )


def _raise_if_startup_blocked(registry: StartupWarningRegistry) -> None:
    if not registry.should_block_startup():
        return
    blocking_warning = next(
        (
            item
            for item in registry.list_warnings()
            if item.startup_action == "block_startup"
        ),
        None,
    )
    if blocking_warning is None:
        raise RuntimeError("startup blocked by unknown warning")
    raise RuntimeError(
        f"startup blocked by {blocking_warning.code}: {blocking_warning.summary}"
    )
