"""
MCP startup validation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def validate_startup_mcp_configuration(
    *,
    get_mcp_config: Callable[[], Any] | None,
    validate_mcp_config: Callable[[], bool] | None,
    logger: Any,
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        if get_mcp_config is None or validate_mcp_config is None:
            return

        mcp_cfg = get_mcp_config()
        debug_mode = (
            bool(mcp_cfg.get("debug_mode", False))
            if isinstance(mcp_cfg, dict)
            else bool(getattr(mcp_cfg, "debug_mode", False))
        )
        if not debug_mode:
            ok = validate_mcp_config()
            if not ok:
                raise RuntimeError(
                    "MCP configuration validation failed; refusing to start in production"
                )
    except startup_guard_exceptions as exc:
        logger.exception(f"Startup aborted due to insecure MCP configuration: {exc}")
        raise
