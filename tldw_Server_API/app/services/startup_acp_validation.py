"""
ACP startup validation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def validate_startup_acp_configuration(
    *,
    route_enabled: Callable[..., bool],
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
    load_acp_runner_config: Callable[[], Any] | None = None,
    validate_acp_config: Callable[[Any], list[str]] | None = None,
) -> None:
    try:
        if not route_enabled("acp", default_stable=False):
            return

        load_config = load_acp_runner_config or _load_acp_runner_config
        validate_config = validate_acp_config or _validate_acp_config

        acp_cfg = load_config()
        acp_warnings = validate_config(acp_cfg)
        for acp_warning in acp_warnings:
            logger.warning("ACP config: {}", acp_warning)
        if not acp_warnings:
            logger.info("App Startup: ACP runner configuration validated")
    except startup_guard_exceptions as exc:
        logger.debug("App Startup: ACP config validation skipped: {}", exc)


def _load_acp_runner_config() -> Any:
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
        load_acp_runner_config,
    )

    return load_acp_runner_config()


def _validate_acp_config(config: Any) -> list[str]:
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
        validate_acp_config,
    )

    return validate_acp_config(config)
