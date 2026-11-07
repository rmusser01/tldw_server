"""
Auth services initialization extracted from the application lifespan startup.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

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


async def _get_startup_db_pool() -> DatabasePool:
    """Acquire the current AuthNZ pool or abort startup with a sanitized error."""
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        if db_pool is None:
            raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_RETURNED_NONE")
        return db_pool
    except AuthStartupError:
        raise
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.bind(exception_type=type(exc).__name__).error(
            "App Startup: Failed to initialize database pool"
        )
        raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_FAILED") from None


async def init_auth_services() -> object:
    """Initialize AuthNZ database access, migrations, seed data, and overrides."""
    db_pool = await _get_startup_db_pool()
    logger.info("App Startup: Database pool initialized")

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

    # Seeding may replace the registered pool; keep refresh and teardown on its owner.
    db_pool = await _get_startup_db_pool()

    override_store = await refresh_provider_override_snapshot(db_pool)
    if override_store is not None:
        try:
            override_store.start_llm_provider_override_refresh_service()
        except _IMPORT_EXCEPTIONS:
            logger.warning("App Startup: LLM provider override refresh service unavailable")

    return db_pool


async def refresh_provider_override_snapshot(db_pool: object) -> ModuleType | None:
    """Verify provider policy against the actual store; leave failures unavailable."""
    override_store = None
    try:
        override_store = import_module(
            "tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides"
        )

        await override_store.refresh_llm_provider_overrides(db_pool)
        logger.info("App Startup: Loaded LLM provider overrides")
    except _IMPORT_EXCEPTIONS:
        logger.warning(
            "App Startup: LLM provider overrides unavailable; server fallback disabled"
        )
    return override_store


async def _validate_pg_serving_schema(db_pool: object) -> bool:
    """Admit a nonowner only after required schema and backfill checks succeed."""
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_lockout_tables_pg,
        ensure_authnz_token_blacklist_pg,
        ensure_resource_daily_ledger_pg,
    )
    from tldw_Server_API.app.core.AuthNZ.postgres_runtime_schema import (
        POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL,
        validate_postgres_org_stt_runtime,
        validate_postgres_startup_core,
        validate_postgres_startup_timestamps,
    )
    from tldw_Server_API.app.core.DB_Management.backends.pg_sharing_schema import postgres_sharing_schema_issues

    readiness_error = "AUTHNZ_CORE_SCHEMA_NOT_READY"
    try:
        async with db_pool.transaction() as conn:
            if await conn.fetchval(POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL):
                return False
            for validator, failure_reason in (
                (validate_postgres_startup_timestamps, "AUTHNZ_USER_TIMESTAMPS_NOT_READY"),
                (validate_postgres_startup_core, "AUTHNZ_CORE_SCHEMA_NOT_READY"),
                (validate_postgres_org_stt_runtime, "AUTHNZ_PG_STT_SCHEMA_NOT_READY"),
            ):
                readiness_error = failure_reason
                await validator(conn)
        # This validator acquires its own connection; release ours even for a one-slot pool.
        readiness_error = "AUTHNZ_PG_SHARING_SCHEMA_NOT_READY"
        if await postgres_sharing_schema_issues(db_pool):
            raise AuthStartupError(readiness_error)
        readiness_error = "AUTHNZ_PG_LOCKOUT_SCHEMA_NOT_READY"
        if not await ensure_authnz_lockout_tables_pg(db_pool):
            raise AuthStartupError(readiness_error)
        readiness_error = "AUTHNZ_PG_TOKEN_BLACKLIST_NOT_READY"
        if not await ensure_authnz_token_blacklist_pg(db_pool):
            raise AuthStartupError(readiness_error)
        readiness_error = "AUTHNZ_PG_DAILY_LEDGER_NOT_READY"
        if not await ensure_resource_daily_ledger_pg(db_pool):
            raise AuthStartupError(readiness_error)
    except _STARTUP_GUARD_EXCEPTIONS:
        raise AuthStartupError(readiness_error) from None
    logger.info("App Startup: Validated required PG serving schema; optional owner migrations were not run")
    return True


async def _ensure_pg_extras(db_pool: DatabasePool) -> None:
    """Ensure additive PostgreSQL-only AuthNZ tables when a PG pool is active."""
    try:
        if not getattr(db_pool, "pool", None):
            return
        if await _validate_pg_serving_schema(db_pool):
            return

        from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
            PGUsageSchemaError,
            ensure_admin_monitoring_tables_pg,
            ensure_admin_webhook_canonical_tables_pg,
            ensure_api_keys_tables_pg,
            ensure_authnz_core_tables_pg,
            ensure_authnz_lockout_tables_pg,
            ensure_authnz_token_blacklist_pg,
            ensure_generated_files_table_pg,
            ensure_llm_provider_overrides_pg,
            ensure_notification_permissions_pg,
            ensure_org_stt_settings_pg,
            ensure_privilege_snapshots_table_pg,
            ensure_resource_daily_ledger_pg,
            ensure_sharing_tables_pg,
            ensure_storage_quota_overrides_backfill_pg,
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
                "AuthNZ lockout tables",
                ensure_authnz_lockout_tables_pg,
                "AUTHNZ_PG_LOCKOUT_SCHEMA_NOT_READY",
            ),
            (
                "AuthNZ token blacklist",
                ensure_authnz_token_blacklist_pg,
                "AUTHNZ_PG_TOKEN_BLACKLIST_NOT_READY",
            ),
            (
                "resource daily ledger",
                ensure_resource_daily_ledger_pg,
                "AUTHNZ_PG_DAILY_LEDGER_NOT_READY",
            ),
            (
                "org STT settings",
                ensure_org_stt_settings_pg,
                "AUTHNZ_PG_STT_SCHEMA_NOT_READY",
            ),
            (
                "admin monitoring tables",
                ensure_admin_monitoring_tables_pg,
                "AUTHNZ_PG_MONITORING_SCHEMA_NOT_READY",
            ),
            ("storage quota overrides backfill", ensure_storage_quota_overrides_backfill_pg, None),
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
            except PGUsageSchemaError:
                raise AuthStartupError("AUTHNZ_PG_USAGE_SCHEMA_NOT_READY") from None
            except _STARTUP_GUARD_EXCEPTIONS as exc:
                if readiness_error is not None:
                    raise AuthStartupError(readiness_error) from None
                logger.bind(exception_type=type(exc).__name__).warning(
                    "App Startup: PG {} ensure failed; continuing independent schema ensures",
                    label,
                )
                continue
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
