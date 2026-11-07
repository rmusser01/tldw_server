"""Async AuthNZ admission using shared read-only PostgreSQL catalog checks."""

from typing import Any

from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _API_KEY_COLUMNS as _API_KEY_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _API_KEY_DEFAULTS as _API_KEY_DEFAULTS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _API_KEY_INDEXES as _API_KEY_INDEXES,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _AUDIO_COLUMNS as _AUDIO_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _CHECKS as _CHECKS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _COLUMNS_SQL as _COLUMNS_SQL,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _CONSTRAINTS_SQL as _CONSTRAINTS_SQL,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _CORE_COLUMNS as _CORE_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _CORE_INDEXES as _CORE_INDEXES,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _FOREIGN_KEYS as _FOREIGN_KEYS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _INDEX_COLUMNS as _INDEX_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _INDEXES_SQL as _INDEXES_SQL,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _NATIVE_QUOTA_CHECKS as _NATIVE_QUOTA_CHECKS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _NATIVE_QUOTA_COLUMNS as _NATIVE_QUOTA_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _NATIVE_QUOTA_INTEGERS as _NATIVE_QUOTA_INTEGERS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _NATIVE_QUOTA_NULLABLE as _NATIVE_QUOTA_NULLABLE,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _PRIMARY_KEYS as _PRIMARY_KEYS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _PROVIDER_SECRET_COLUMNS as _PROVIDER_SECRET_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _PROVIDER_SECRET_INDEXES as _PROVIDER_SECRET_INDEXES,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _RBAC_COLUMNS as _RBAC_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _SHARED_INDEXES as _SHARED_INDEXES,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STARTUP_COLUMNS as _STARTUP_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STARTUP_DEFAULTS as _STARTUP_DEFAULTS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STARTUP_GENERATED_IDS as _STARTUP_GENERATED_IDS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STARTUP_INDEXES as _STARTUP_INDEXES,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STARTUP_NOT_NULL as _STARTUP_NOT_NULL,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _STT_COLUMNS as _STT_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _UNIQUE_KEYS as _UNIQUE_KEYS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _USERS_COLUMNS as _USERS_COLUMNS,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL as POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _check_matches as _check_matches,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _column_type_allowed as _column_type_allowed,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _validate_metadata as _validate_metadata,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    _validate_required_metadata as _validate_required_metadata,
)
from tldw_Server_API.app.core.DB_Management.postgres_auth_catalog import (
    validate_postgres_rbac_runtime as validate_postgres_rbac_runtime,
)
from tldw_Server_API.app.core.exceptions import SchemaReadinessError


async def validate_postgres_api_keys_runtime(conn: Any) -> None:
    """Admit API-key reads/writes only against the complete owner-migrated schema."""
    from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import _postgres_generated_id_matches

    columns = await conn.fetch(_COLUMNS_SQL)
    indexes = await conn.fetch(_INDEXES_SQL)
    constraints = await conn.fetch(_CONSTRAINTS_SQL)
    _validate_required_metadata(
        columns, indexes, [row for row in constraints if not row["deferrable"]], _API_KEY_COLUMNS, _API_KEY_INDEXES
    )
    for key in (("key_hash",), ("key_id",)):
        if not any(
            row["table_name"] == "api_keys"
            and row["ready"]
            and row["is_unique"]
            and row["immediate"]
            and row["predicate"] is None
            and tuple(row["columns"]) == key
            for row in indexes
        ):
            raise SchemaReadinessError("PostgreSQL API key uniqueness requires owner migration")
    required_not_null = {
        "api_keys": {"id", "user_id", "key_hash", "key_prefix"},
        "api_key_audit_log": {"id", "api_key_id", "action"},
    }
    for row in columns:
        table, column = row["table_name"], row["column_name"]
        if table not in _API_KEY_COLUMNS or column not in _API_KEY_COLUMNS[table]:
            continue
        if column in required_not_null[table] and not row["not_null"]:
            raise SchemaReadinessError("PostgreSQL API key nullability requires owner migration")
        expected = _API_KEY_DEFAULTS[table].get(column)
        if expected and not _check_matches(row["default_expression"], expected):
            raise SchemaReadinessError("PostgreSQL API key defaults require owner migration")
        if column == "id" and not _postgres_generated_id_matches(
            table,
            {
                "default": row["default_expression"],
                "is_identity": row["is_identity"],
                "identity_generation": row["identity_generation"],
            },
        ):
            raise SchemaReadinessError("PostgreSQL API key generated ID requires owner migration")


async def validate_postgres_user_provider_secrets_runtime(conn: Any) -> None:
    """Validate BYOK persistence without running owner DDL on a serving request."""
    from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import _postgres_generated_id_matches

    columns = await conn.fetch(_COLUMNS_SQL)
    indexes = await conn.fetch(_INDEXES_SQL)
    constraints = await conn.fetch(_CONSTRAINTS_SQL)
    _validate_required_metadata(
        columns,
        indexes,
        [row for row in constraints if not row["deferrable"]],
        _PROVIDER_SECRET_COLUMNS,
        _PROVIDER_SECRET_INDEXES,
    )
    if not any(
        row["table_name"] == "user_provider_secrets"
        and row["ready"]
        and row["is_unique"]
        and row["immediate"]
        and row["predicate"] is None
        and tuple(row["columns"]) == ("user_id", "provider")
        for row in indexes
    ):
        raise SchemaReadinessError("PostgreSQL provider credential uniqueness requires owner migration")
    if any(
        row["table_name"] == "user_provider_secrets"
        and row["ready"]
        and row["is_unique"]
        and not row["immediate"]
        and row["predicate"] is None
        and len(row["columns"]) == 2
        and set(row["columns"]) == {"user_id", "provider"}
        for row in indexes
    ):
        raise SchemaReadinessError("PostgreSQL provider credential conflict arbiter requires owner migration")
    for row in columns:
        if row["table_name"] != "user_provider_secrets":
            continue
        column = row["column_name"]
        if column in {"id", "user_id", "provider", "encrypted_blob"} and not row["not_null"]:
            raise SchemaReadinessError("PostgreSQL provider credential nullability requires owner migration")
        if column in {"created_at", "updated_at"} and not _check_matches(
            row["default_expression"], ("CURRENT_TIMESTAMP", "now()")
        ):
            raise SchemaReadinessError("PostgreSQL provider credential defaults require owner migration")
        if column == "id" and not _postgres_generated_id_matches(
            "user_provider_secrets",
            {
                "default": row["default_expression"],
                "is_identity": row["is_identity"],
                "identity_generation": row["identity_generation"],
            },
        ):
            raise SchemaReadinessError("PostgreSQL provider credential generated ID requires owner migration")


async def validate_postgres_startup_timestamps(conn: Any) -> None:
    """Require all six owner-normalized user timestamp columns without ALTER."""
    from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import _USER_TIMESTAMP_COLUMNS

    columns = {
        row["column_name"]: row["data_type"] for row in await conn.fetch(_COLUMNS_SQL) if row["table_name"] == "users"
    }
    if any(columns.get(name) != "timestamptz" for name in _USER_TIMESTAMP_COLUMNS):
        raise SchemaReadinessError("PostgreSQL users timestamps require owner migration")


async def validate_postgres_startup_core(conn: Any) -> None:
    """Read the required startup schema, completed backfills and default MCP grants."""
    from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
        validate_postgres_profile_version_on_connection,
    )
    from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import (
        _postgres_generated_id_matches,
        validate_postgres_profile_candidate_schema,
    )

    columns = await conn.fetch(_COLUMNS_SQL)
    constraints = await conn.fetch(_CONSTRAINTS_SQL)
    _validate_required_metadata(
        columns, await conn.fetch(_INDEXES_SQL), constraints, _STARTUP_COLUMNS, _STARTUP_INDEXES
    )
    for row in columns:
        table, column = row["table_name"], row["column_name"]
        if column in _STARTUP_NOT_NULL.get(table, set()) and not row["not_null"]:
            raise SchemaReadinessError("PostgreSQL startup nullability requires owner migration")
        expected_default = _STARTUP_DEFAULTS.get(table, {}).get(column)
        if expected_default and not _check_matches(row["default_expression"], expected_default):
            raise SchemaReadinessError("PostgreSQL startup defaults require owner migration")
        if (
            table in _STARTUP_GENERATED_IDS
            and column == "id"
            and not _postgres_generated_id_matches(
                table,
                {
                    "default": row["default_expression"],
                    "is_identity": row["is_identity"],
                    "identity_generation": row["identity_generation"],
                },
            )
        ):
            raise SchemaReadinessError("PostgreSQL startup generated ID requires owner migration")
    registration_links = {
        (tuple(row["columns"]), row["foreign_table"], tuple(row["foreign_columns"]), row["delete_action"])
        for row in constraints
        if row["table_name"] == "registration_codes"
        and row["kind"] == "f"
        and row["validated"]
        and row["foreign_schema"] == "public"
        and row["update_action"] == "a"
    }
    if not {(("org_id",), "organizations", ("id",), "n"), (("team_id",), "teams", ("id",), "n")} <= registration_links:
        raise SchemaReadinessError("PostgreSQL registration organization links require owner migration")
    if await conn.fetchval("SELECT EXISTS (SELECT 1 FROM public.sessions WHERE last_activity IS NULL)"):
        raise SchemaReadinessError("PostgreSQL session activity requires owner migration")
    if any(
        row["table_name"] == "registration_codes" and row["column_name"] == "uses" for row in columns
    ) and await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM public.registration_codes "
        "WHERE (times_used IS NULL OR times_used = 0) AND times_used IS DISTINCT FROM uses)"
    ):
        raise SchemaReadinessError("PostgreSQL registration usage requires owner migration")
    if not await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM public.permissions WHERE name = 'prompts.read')"
    ) or await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM public.roles r WHERE r.name IN ('admin', 'user') AND NOT EXISTS ("
        "SELECT 1 FROM public.role_permissions rp JOIN public.permissions p ON p.id = rp.permission_id "
        "WHERE rp.role_id = r.id AND p.name = 'prompts.read'))"
    ):
        raise SchemaReadinessError("PostgreSQL MCP prompt grants require owner migration")
    await validate_postgres_profile_version_on_connection(conn)
    await validate_postgres_profile_candidate_schema(conn)


async def validate_postgres_org_stt_runtime(conn: Any) -> None:
    """Require owner-provisioned STT privacy settings, never repair them while serving."""
    columns = await conn.fetch(_COLUMNS_SQL)
    _validate_required_metadata(
        columns,
        await conn.fetch(_INDEXES_SQL),
        await conn.fetch(_CONSTRAINTS_SQL),
        _STT_COLUMNS,
        {"idx_org_stt_settings_updated_by": "org_stt_settings"},
    )
    defaults = {
        "delete_audio_after_success": ("TRUE",),
        "audio_retention_hours": ("0", "0.0"),
        "redact_pii": ("FALSE",),
        "allow_unredacted_partials": ("FALSE",),
        "redact_categories_json": ("'[]'::jsonb",),
    }
    for row in columns:
        if row["table_name"] != "org_stt_settings":
            continue
        name = row["column_name"]
        if name == "audio_retention_hours" and row["data_type"] != "float8":
            raise SchemaReadinessError("PostgreSQL STT retention type requires owner migration")
        if name in defaults and (not row["not_null"] or not _check_matches(row["default_expression"], defaults[name])):
            raise SchemaReadinessError("PostgreSQL STT defaults require owner migration")
        if name == "updated_at" and (
            row["data_type"] != "timestamptz"
            or not _check_matches(row["default_expression"], ("CURRENT_TIMESTAMP", "now()"))
        ):
            raise SchemaReadinessError("PostgreSQL STT timestamp requires owner migration")


async def validate_postgres_auth_runtime(conn: Any) -> None:
    """Validate the async pool's bootstrap objects without DDL or writes."""
    _validate_metadata(
        await conn.fetch(_COLUMNS_SQL), await conn.fetch(_INDEXES_SQL), await conn.fetch(_CONSTRAINTS_SQL), rbac=False
    )
