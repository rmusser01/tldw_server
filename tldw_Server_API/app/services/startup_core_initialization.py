"""
Startup core initialization helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from typing import Any, Callable


@dataclass
class StartupCoreInitializationHandles:
    """Handles returned from the core startup initialization block."""

    db_pool: Any | None = None
    session_manager: Any | None = None
    heavy_startup_handles: Any | None = None
    startup_sandbox_orchestrator: Any | None = None


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
    startup_sandbox_orchestrator = None
    if _sandbox_startup_warning_configured():
        startup_sandbox_orchestrator = _build_startup_sandbox_orchestrator(
            logger=logger,
            startup_guard_exceptions=startup_guard_exceptions,
        )

    return StartupCoreInitializationHandles(
        db_pool=auth_runtime_handles.db_pool,
        session_manager=auth_runtime_handles.session_manager,
        heavy_startup_handles=heavy_startup_handles,
        startup_sandbox_orchestrator=startup_sandbox_orchestrator,
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


def _sandbox_startup_warning_configured() -> bool:
    """Return True when the macOS VZ helper path is configured enough to make startup warnings actionable."""
    helper_socket = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET") or "").strip()
    return bool(helper_socket) and platform.system() == "Darwin"


def _build_startup_sandbox_orchestrator(
    *,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> Any | None:
    """Best-effort startup-owned sandbox orchestrator factory for startup warning producers."""
    try:
        from tldw_Server_API.app.core.Sandbox.service import SandboxService

        return SandboxService(enable_background_tasks=False)._orch
    except startup_guard_exceptions as exc:
        logger.warning(
            "Startup sandbox orchestrator unavailable; continuing without startup sandbox warnings: {}",
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Startup sandbox orchestrator unavailable; continuing without startup sandbox warnings: {}",
            exc,
        )
        return None
