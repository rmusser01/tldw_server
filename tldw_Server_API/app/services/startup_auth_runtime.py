"""
Post-auth startup runtime orchestration extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.services.startup_auth import AuthStartupError


@dataclass
class AuthRuntimeHandles:
    db_pool: Any | None = None
    session_manager: Any | None = None


async def initialize_auth_runtime_services(
    *,
    app: Any,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> AuthRuntimeHandles:
    handles = AuthRuntimeHandles()

    try:
        db_pool = await _init_auth_services()
        if db_pool is None:
            raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_RETURNED_NONE")
        handles.db_pool = db_pool
        await _init_resource_governor(app)
        _validate_auth_rg_startup_guards(app)
        handles.session_manager = await _get_session_manager()
        logger.info("App Startup: Session manager initialized")

        try:
            dispatcher = _get_security_alert_dispatcher()
            dispatcher.validate_configuration()
            if dispatcher.enabled:
                logger.info("App Startup: Security alert configuration validated")
        except ValueError as config_error:
            logger.exception(f"App Startup: Security alert configuration invalid: {config_error}")
            raise
    except AuthStartupError:
        raise
    except ValueError:
        raise
    except startup_guard_exceptions as exc:
        logger.exception(f"App Startup: Security alert validation / auth services init failed: {exc}")

    return handles


async def _init_auth_services() -> Any:
    from tldw_Server_API.app.services.startup_auth import init_auth_services

    return await init_auth_services()


async def _init_resource_governor(app: Any) -> None:
    from tldw_Server_API.app.services.startup_resource_governor import (
        init_resource_governor,
    )

    await init_resource_governor(app)


def _validate_auth_rg_startup_guards(app: Any) -> None:
    from tldw_Server_API.app.core.AuthNZ.rg_startup_guard import (
        validate_auth_rg_startup_guards,
    )

    validate_auth_rg_startup_guards(app)


async def _get_session_manager() -> Any:
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

    return await get_session_manager()


def _get_security_alert_dispatcher() -> Any:
    from tldw_Server_API.app.core.AuthNZ.alerting import get_security_alert_dispatcher

    return get_security_alert_dispatcher()
