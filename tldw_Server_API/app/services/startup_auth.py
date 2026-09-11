"""
Auth services initialization extracted from the application lifespan startup.
"""

from __future__ import annotations

from importlib import import_module

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    DatabaseError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_IMPORT_EXCEPTIONS = (
    AssertionError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class AuthStartupError(RuntimeError):
    """Raised when AuthNZ startup cannot safely continue."""


async def init_auth_services() -> object:
    """Initialize AuthNZ database access, migrations, seed data, and overrides."""
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        if db_pool is None:
            raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_RETURNED_NONE")
        logger.info("App Startup: Database pool initialized")
    except AuthStartupError:
        raise
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.bind(exception_type=type(exc).__name__).error(
            "App Startup: Failed to initialize database pool"
        )
        raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_FAILED") from None

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once
    except _IMPORT_EXCEPTIONS:
        logger.error("App Startup: Required AuthNZ schema readiness is unavailable")
        raise AuthStartupError("AUTHNZ_SCHEMA_READINESS_UNAVAILABLE") from None
    else:
        try:
            await ensure_authnz_schema_ready_once()
        except _STARTUP_GUARD_EXCEPTIONS as exc:
            logger.bind(exception_type=type(exc).__name__).error(
                "App Startup: AuthNZ schema is not ready"
            )
            raise AuthStartupError("AUTHNZ_SCHEMA_NOT_READY") from None

    await _ensure_pg_extras(db_pool)

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed

        await ensure_single_user_rbac_seed_if_needed()
        logger.info("App Startup: Ensured single-user RBAC seed (baseline roles/permissions)")
    except _IMPORT_EXCEPTIONS as exc:
        logger.bind(exception_type=type(exc).__name__).debug(
            "App Startup: RBAC single-user seed ensure skipped"
        )

    override_store = None
    try:
        override_store = import_module(
            "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
        )

        await override_store.refresh_llm_provider_overrides(db_pool)
        logger.info("App Startup: Loaded LLM provider overrides")
    except _IMPORT_EXCEPTIONS:
        logger.warning(
            "App Startup: LLM provider overrides unavailable; server fallback disabled"
        )
    if override_store is not None:
        try:
            override_store.start_llm_provider_override_refresh_service()
        except _IMPORT_EXCEPTIONS:
            logger.warning("App Startup: LLM provider override refresh service unavailable")

    return db_pool


async def _ensure_pg_extras(db_pool: object) -> None:
    """Ensure additive PostgreSQL-only AuthNZ tables when a PG pool is active."""
    try:
        if not getattr(db_pool, "pool", None):
            return

        from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
            ensure_admin_webhook_canonical_tables_pg,
            ensure_api_keys_tables_pg,
            ensure_authnz_core_tables_pg,
            ensure_generated_files_table_pg,
            ensure_llm_provider_overrides_pg,
            ensure_notification_permissions_pg,
            ensure_privilege_snapshots_table_pg,
            ensure_sharing_tables_pg,
            ensure_tool_catalogs_tables_pg,
            ensure_usage_tables_pg,
            ensure_user_timestamp_timezones_pg,
            ensure_virtual_key_counters_pg,
        )

        pg_ensures = [
            (
                "users timestamp time zones",
                ensure_user_timestamp_timezones_pg,
                "AUTHNZ_USER_TIMESTAMPS_NOT_READY",
            ),
            (
                "AuthNZ core tables",
                ensure_authnz_core_tables_pg,
                "AUTHNZ_CORE_SCHEMA_NOT_READY",
            ),
            (
                "canonical admin webhook tables",
                ensure_admin_webhook_canonical_tables_pg,
                None,
            ),
            (
                "sharing tables",
                ensure_sharing_tables_pg,
                "AUTHNZ_PG_SHARING_SCHEMA_NOT_READY",
            ),
            ("notification permissions", ensure_notification_permissions_pg, None),
            ("generated_files table", ensure_generated_files_table_pg, None),
            ("tool catalogs tables", ensure_tool_catalogs_tables_pg, None),
            ("privilege_snapshots table", ensure_privilege_snapshots_table_pg, None),
            ("api_keys tables", ensure_api_keys_tables_pg, None),
            ("usage tables", ensure_usage_tables_pg, None),
            ("virtual-key counters tables", ensure_virtual_key_counters_pg, None),
            ("llm_provider_overrides table", ensure_llm_provider_overrides_pg, None),
        ]

        for label, ensure_fn, readiness_error in pg_ensures:
            try:
                ok = await ensure_fn(db_pool)
            except _STARTUP_GUARD_EXCEPTIONS:
                if readiness_error is not None:
                    raise AuthStartupError(readiness_error) from None
                raise
            if ok:
                logger.info(f"App Startup: Ensured PG {label}")
            elif readiness_error is not None:
                raise AuthStartupError(readiness_error)
            else:
                logger.warning(
                    f"App Startup: PG {label} ensure returned False; "
                    "canonical database state may be incomplete"
                )
    except AuthStartupError:
        raise
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.bind(exception_type=type(exc).__name__).debug(
            "App Startup: PG extras ensure failed/skipped"
        )
