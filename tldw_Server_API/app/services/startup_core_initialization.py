"""
Startup core initialization helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class StartupCoreInitializationHandles:
    """Handles returned from the core startup initialization block."""

    db_pool: Any | None = None
    session_manager: Any | None = None
    heavy_startup_handles: Any | None = None


async def initialize_startup_core_components(
    *,
    app: Any,
    module_file: str,
    logger: Any,
    route_enabled: Callable[..., bool],
    defer_heavy: bool,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> StartupCoreInitializationHandles:
    """Run the remaining startup core initialization block in the legacy order."""
    await _run_startup_validations()

    logger.info("App Startup: Initializing authentication services...")
    auth_runtime_handles = await _initialize_auth_runtime_services(
        app=app,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )

    await _warm_chacha_notes_on_startup(
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )

    _validate_startup_privilege_metadata(
        app=app,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
    )

    _load_startup_catalogs(
        module_file=module_file,
        logger=logger,
        startup_guard_exceptions=startup_guard_exceptions,
        import_exceptions=import_exceptions,
    )

    logger.info("App Startup: Initializing Chat module components...")
    heavy_startup_handles = await _start_heavy_initializations(
        app,
        route_enabled=route_enabled,
        defer_heavy=defer_heavy,
    )

    return StartupCoreInitializationHandles(
        db_pool=auth_runtime_handles.db_pool,
        session_manager=auth_runtime_handles.session_manager,
        heavy_startup_handles=heavy_startup_handles,
    )


async def _run_startup_validations() -> None:
    from tldw_Server_API.app.services.startup_validation import run_startup_validations

    await run_startup_validations()


async def _initialize_auth_runtime_services(**kwargs):
    from tldw_Server_API.app.services.startup_auth_runtime import (
        initialize_auth_runtime_services,
    )

    return await initialize_auth_runtime_services(**kwargs)


async def _warm_chacha_notes_on_startup(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_chacha_warmup import (
        warm_chacha_notes_on_startup,
    )

    await warm_chacha_notes_on_startup(**kwargs)


def _validate_startup_privilege_metadata(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_privilege_validation import (
        validate_startup_privilege_metadata,
    )

    validate_startup_privilege_metadata(**kwargs)


def _load_startup_catalogs(**kwargs) -> None:
    from tldw_Server_API.app.services.startup_catalog_loading import load_startup_catalogs

    load_startup_catalogs(**kwargs)


async def _start_heavy_initializations(app_arg, **kwargs):
    from tldw_Server_API.app.services.startup_heavy_init import (
        start_heavy_initializations,
    )

    return await start_heavy_initializations(app_arg, **kwargs)
