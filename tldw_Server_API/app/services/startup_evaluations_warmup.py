"""
Startup Evaluations lazy-manager warmup extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def warm_lazy_evaluations_managers(
    *,
    route_enabled: Callable[[str], bool],
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    test_mode: bool,
) -> None:
    try:
        if test_mode or not route_enabled("evaluations"):
            return

        _warm_evaluations_connection_manager()
        _warm_evaluations_webhook_manager()
        logger.info("App Startup: Warmed lazy Evaluations managers (fail-fast enabled)")
    except startup_guard_exceptions as exc:
        logger.exception(f"Startup aborted: lazy subsystem warmup failed: {exc}")
        raise


def _warm_evaluations_connection_manager() -> None:
    from tldw_Server_API.app.core.Evaluations.connection_pool import (
        get_connection_manager,
    )

    get_connection_manager()


def _warm_evaluations_webhook_manager() -> None:
    from tldw_Server_API.app.core.Evaluations.webhook_manager import (
        get_webhook_manager,
    )

    get_webhook_manager()
