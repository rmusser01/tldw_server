"""
Startup environment reporting extracted from the application lifespan.
"""

from __future__ import annotations

import os
from typing import Any, Callable


async def report_startup_environment(
    *,
    app: Any,
    logger: Any,
    startup_api_key_log_value: Callable[[str], str],
    shared_is_truthy: Callable[[Any], bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    import_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        await _log_startup_banner(
            logger=logger,
            startup_api_key_log_value=startup_api_key_log_value,
            startup_guard_exceptions=startup_guard_exceptions,
        )
    except import_exceptions as exc:
        logger.exception(f"Failed to display startup info: {exc}")

    try:
        await _log_preflight_environment_report(
            app=app,
            logger=logger,
            shared_is_truthy=shared_is_truthy,
            startup_guard_exceptions=startup_guard_exceptions,
        )
    except startup_guard_exceptions + import_exceptions as exc:
        logger.warning(f"Preflight report could not be generated: {exc}")


async def _log_startup_banner(
    *,
    logger: Any,
    startup_api_key_log_value: Callable[[str], str],
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    settings = _get_auth_settings()

    logger.info("=" * 60)
    logger.info("🚀 TLDW Server Started Successfully")
    logger.info("=" * 60)

    if _is_single_user_mode():
        logger.info("🔐 Authentication Mode: SINGLE USER")
        display_key = startup_api_key_log_value(settings.SINGLE_USER_API_KEY)
        masked_note = ""
        if display_key != settings.SINGLE_USER_API_KEY:
            masked_note = " (masked; set SHOW_API_KEY_ON_STARTUP=true to display once)"
        logger.info(f"🔑 API Key: {display_key}{masked_note}")
        logger.info("=" * 60)
        logger.info("Use this API key in the X-API-KEY header for requests")
    else:
        logger.info("🔐 Authentication Mode: MULTI USER")
        try:
            pool = await _get_db_pool()
            is_pg = bool(getattr(pool, "pool", None) is not None)
            if is_pg:
                logger.info("JWT Bearer tokens required for authentication")
            else:
                logger.info("JWT Bearer tokens or X-API-KEY (per-user) supported for SQLite setups")
        except startup_guard_exceptions:
            logger.info("JWT Bearer tokens required for authentication")
        logger.info("=" * 60)

    logger.info("📍 API Documentation: http://127.0.0.1:8000/docs")
    logger.info("🧭 Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart")
    logger.info("🛠 Setup UI: http://127.0.0.1:8000/setup (if required)")
    logger.info("=" * 60)


async def _log_preflight_environment_report(
    *,
    app: Any,
    logger: Any,
    shared_is_truthy: Callable[[Any], bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    settings = _get_auth_settings()
    prod = os.getenv("tldw_production", "false").lower() in {"true", "1", "yes", "y", "on"}
    auth_mode = settings.AUTH_MODE
    db_url = settings.DATABASE_URL

    try:
        pool = await _get_db_pool()
        is_pg = bool(getattr(pool, "pool", None) is not None)
        db_engine = "postgresql" if is_pg else ("sqlite" if str(db_url).startswith("sqlite") else "other")
    except startup_guard_exceptions:
        db_engine = "sqlite" if str(db_url).startswith("sqlite") else "other"

    redis_enabled = bool(settings.REDIS_URL) or bool(
        os.getenv("REDIS_ENABLED", "false").lower() in {"true", "1", "yes", "y", "on"}
    )
    csrf_enabled = (auth_mode == "multi_user") or (_get_csrf_global_settings().get("CSRF_ENABLED", None) is True)
    cors_diagnostics = _get_cors_runtime_diagnostics()
    cors_disable = bool(cors_diagnostics.get("disable_cors", False))
    cors_disable_source = str(cors_diagnostics.get("disable_cors_source", "unknown"))
    cors_allow_credentials = bool(cors_diagnostics.get("allow_credentials", False))
    cors_allow_credentials_source = str(cors_diagnostics.get("allow_credentials_source", "unknown"))
    cors_count = int(cors_diagnostics.get("allowed_origins_count", 0))
    cors_allowed_origins_source = str(cors_diagnostics.get("allowed_origins_source", "unknown"))
    cors_allowed_origins = cors_diagnostics.get("allowed_origins", [])
    if not isinstance(cors_allowed_origins, list):
        cors_allowed_origins = []
    cors_config_path = cors_diagnostics.get("config_path")
    cors_config_loaded = bool(cors_diagnostics.get("config_loaded", False))
    has_limiter = hasattr(app.state, "limiter")
    provider_manager = _get_provider_manager()
    providers = len(provider_manager.providers) if provider_manager and hasattr(provider_manager, "providers") else 0

    logger.info("Preflight Environment Report ─────────────────────────────────────────")
    logger.info(f"• Mode: {auth_mode} | Production: {prod}")
    logger.info(f"• Database: engine={db_engine}")
    if db_engine == "sqlite" and auth_mode == "multi_user":
        if prod:
            logger.error("• Database check: FAIL (SQLite in multi-user prod not supported)")
        else:
            logger.warning("• Database check: WARN (SQLite in multi-user; prefer PostgreSQL)")
    else:
        logger.info("• Database check: OK")
    logger.info(f"• Redis: enabled={redis_enabled}")
    logger.info(f"• CSRF: enabled={csrf_enabled}")
    if cors_disable:
        logger.info("• CORS: disabled")
    else:
        logger.info(f"• CORS: allowed_origins={cors_count} | allow_credentials={cors_allow_credentials}")
    logger.info(
        "• CORS effective settings: "
        f"disable={cors_disable} (source={cors_disable_source}) | "
        f"allow_credentials={cors_allow_credentials} (source={cors_allow_credentials_source}) | "
        f"origins={cors_count} (source={cors_allowed_origins_source})"
    )
    logger.info(f"• CORS config file: path={cors_config_path or '(unknown)'} | loaded={cors_config_loaded}")
    if cors_allowed_origins:
        origin_preview_max = 6
        origin_preview = ", ".join(str(origin) for origin in cors_allowed_origins[:origin_preview_max])
        if len(cors_allowed_origins) > origin_preview_max:
            origin_preview += f", ... (+{len(cors_allowed_origins) - origin_preview_max} more)"
        logger.info(f"• CORS origins preview: {origin_preview}")
    logger.info(f"• Global rate limiter: {has_limiter}")
    logger.info(f"• Providers configured: {providers}")
    logger.info(f"• OpenTelemetry available: {bool(_otel_available())}")
    logger.info("──────────────────────────────────────────────────────────────────────")

    try:
        if prod:
            test_flags = {
                "TEST_MODE": os.getenv("TEST_MODE", ""),
                "TLDW_TEST_MODE": os.getenv("TLDW_TEST_MODE", ""),
            }
            enabled = [key for key, value in test_flags.items() if shared_is_truthy(value)]
            if enabled:
                logger.warning(
                    f"Test-mode toggles enabled in production: {', '.join(enabled)} - disable these for secure deployments"
                )
    except startup_guard_exceptions:
        pass


def _get_auth_settings() -> Any:
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    return get_settings()


def _is_single_user_mode() -> bool:
    from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode

    return bool(is_single_user_mode())


async def _get_db_pool() -> Any:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    return await get_db_pool()


def _get_csrf_global_settings() -> dict[str, Any]:
    from tldw_Server_API.app.core.AuthNZ.csrf_protection import global_settings

    return global_settings


def _get_provider_manager() -> Any:
    from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager

    return get_provider_manager()


def _get_cors_runtime_diagnostics() -> dict[str, Any]:
    from tldw_Server_API.app.core.config import get_cors_runtime_diagnostics

    return get_cors_runtime_diagnostics()


def _otel_available() -> bool:
    from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE

    return bool(OTEL_AVAILABLE)
