"""
Auth services initialization extracted from the application lifespan startup.
"""

from __future__ import annotations

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
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
        logger.error(f"App Startup: Failed to initialize database pool: {exc}")
        raise AuthStartupError("AUTHNZ_DB_POOL_STARTUP_FAILED") from exc

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except _IMPORT_EXCEPTIONS as exc:
        logger.warning(f"App Startup: Skipped AuthNZ SQLite migration ensure: {exc}")

    await _ensure_pg_extras(db_pool)

    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed

        await ensure_single_user_rbac_seed_if_needed()
        logger.info("App Startup: Ensured single-user RBAC seed (baseline roles/permissions)")
    except _IMPORT_EXCEPTIONS as exc:
        logger.debug(f"App Startup: RBAC single-user seed ensure skipped: {exc}")

    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            refresh_llm_provider_overrides,
        )

        await refresh_llm_provider_overrides(db_pool)
        logger.info("App Startup: Loaded LLM provider overrides")
    except _IMPORT_EXCEPTIONS as exc:
        logger.debug(f"App Startup: LLM provider overrides load skipped: {exc}")

    return db_pool


async def _ensure_pg_extras(db_pool: object) -> None:
    """Ensure additive PostgreSQL-only AuthNZ tables when a PG pool is active."""
    try:
        if not getattr(db_pool, "pool", None):
            return

        from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
            ensure_api_keys_tables_pg,
            ensure_authnz_core_tables_pg,
            ensure_generated_files_table_pg,
            ensure_llm_provider_overrides_pg,
            ensure_privilege_snapshots_table_pg,
            ensure_tool_catalogs_tables_pg,
            ensure_usage_tables_pg,
            ensure_virtual_key_counters_pg,
        )

        pg_ensures = [
            ("AuthNZ core tables", ensure_authnz_core_tables_pg),
            ("generated_files table", ensure_generated_files_table_pg),
            ("tool catalogs tables", ensure_tool_catalogs_tables_pg),
            ("privilege_snapshots table", ensure_privilege_snapshots_table_pg),
            ("api_keys tables", ensure_api_keys_tables_pg),
            ("usage tables", ensure_usage_tables_pg),
            ("virtual-key counters tables", ensure_virtual_key_counters_pg),
            ("llm_provider_overrides table", ensure_llm_provider_overrides_pg),
        ]

        for label, ensure_fn in pg_ensures:
            ok = await ensure_fn(db_pool)
            if ok:
                logger.info(f"App Startup: Ensured PG {label}")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"App Startup: PG extras ensure failed/skipped: {exc}")
