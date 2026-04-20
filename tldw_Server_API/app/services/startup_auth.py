"""
Auth services initialization extracted from main.py lifespan.

Handles: database pool creation, schema/migration ensures,
PostgreSQL extras, RBAC seed, and LLM provider overrides.
"""

from __future__ import annotations

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (Exception,)
_IMPORT_EXCEPTIONS = (ImportError, ModuleNotFoundError)


async def init_auth_services() -> object | None:
    """Initialize auth database pool, schema, and RBAC seed.

    Returns the database pool object (or None on failure).
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        logger.info("App Startup: Database pool initialized")
    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.error(f"App Startup: Failed to initialize database pool: {e}")
        return None

    # Ensure AuthNZ schema/migrations (SQLite)
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except _IMPORT_EXCEPTIONS as e:
        logger.debug(f"App Startup: Skipped AuthNZ SQLite migration ensure: {e}")

    # Postgres-only: ensure additive extras
    await _ensure_pg_extras(db_pool)

    # Ensure RBAC seed exists in single-user mode (idempotent)
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed

        await ensure_single_user_rbac_seed_if_needed()
        logger.info("App Startup: Ensured single-user RBAC seed (baseline roles/permissions)")
    except _IMPORT_EXCEPTIONS as e:
        logger.debug(f"App Startup: RBAC single-user seed ensure skipped: {e}")

    # Load LLM provider overrides into memory
    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            refresh_llm_provider_overrides,
        )

        await refresh_llm_provider_overrides(db_pool)
        logger.info("App Startup: Loaded LLM provider overrides")
    except _IMPORT_EXCEPTIONS as e:
        logger.debug(f"App Startup: LLM provider overrides load skipped: {e}")

    return db_pool


async def _ensure_pg_extras(db_pool: object) -> None:
    """Ensure PostgreSQL-only additive extras if using PG backend."""
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

        _pg_ensures = [
            ("AuthNZ core tables", ensure_authnz_core_tables_pg),
            ("generated_files table", ensure_generated_files_table_pg),
            ("tool catalogs tables", ensure_tool_catalogs_tables_pg),
            ("privilege_snapshots table", ensure_privilege_snapshots_table_pg),
            ("api_keys tables", ensure_api_keys_tables_pg),
            ("usage tables", ensure_usage_tables_pg),
            ("virtual-key counters", ensure_virtual_key_counters_pg),
            ("llm_provider_overrides table", ensure_llm_provider_overrides_pg),
        ]

        for label, ensure_fn in _pg_ensures:
            ok = await ensure_fn(db_pool)
            if ok:
                logger.info(f"App Startup: Ensured PG {label}")

    except _STARTUP_GUARD_EXCEPTIONS as e:
        logger.debug(f"App Startup: PG extras ensure failed/skipped: {e}")
