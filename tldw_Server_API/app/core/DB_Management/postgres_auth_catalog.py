"""Read-only admission for owner-migrated PostgreSQL AuthNZ stores."""

from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from tldw_Server_API.app.core.exceptions import SchemaReadinessError

POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL = """
SELECT CASE WHEN to_regclass('public.users') IS NULL
    THEN has_schema_privilege(current_user, 'public', 'CREATE')
    ELSE pg_has_role(current_user,
        (SELECT relowner FROM pg_class WHERE oid = to_regclass('public.users')),
        'USAGE')
END AS can_migrate
"""

_USERS_COLUMNS = {
    "id",
    "uuid",
    "username",
    "email",
    "password_hash",
    "metadata",
    "role",
    "is_active",
    "is_verified",
    "is_superuser",
    "failed_login_attempts",
    "locked_until",
    "storage_quota_mb",
    "storage_used_mb",
    "created_at",
    "updated_at",
    "profile_version",
    "last_login",
    "email_verified",
    "email_verified_at",
    "two_factor_enabled",
    "two_factor_secret",
    "totp_secret",
    "backup_codes",
    "created_by",
    "password_changed_at",
}
_AUDIO_COLUMNS = {
    "id",
    "token_hash",
    "user_id",
    "project_id",
    "artifact_id",
    "purpose",
    "expires_at",
    "consumed_at",
    "revoked_at",
    "created_at",
    "created_by_auth_mode",
    "last_redeemed_at",
}
_NATIVE_QUOTA_COLUMNS = {
    "user_id",
    "user_uuid",
    "owner_key",
    "candidate_key",
    "scope_type",
    "workspace_id",
    "org_id",
    "team_id",
    "size_bytes",
    "state",
}
_NATIVE_QUOTA_INTEGERS = {"user_id", "org_id", "team_id", "size_bytes"}
_NATIVE_QUOTA_NULLABLE = {"workspace_id", "org_id", "team_id"}
_NATIVE_QUOTA_CHECKS = (
    ("scope_type IN ('global', 'workspace')", "scope_type = ANY (ARRAY['global', 'workspace'])"),
    ("state IN ('active', 'released')", "state = ANY (ARRAY['active', 'released'])"),
    ("size_bytes >= 0",),
    ("team_id IS NULL OR org_id IS NOT NULL",),
    ("(scope_type = 'global' AND workspace_id IS NULL) OR (scope_type = 'workspace' AND workspace_id IS NOT NULL)",),
)
_API_KEY_COLUMNS = {
    "api_keys": {
        "id",
        "user_id",
        "key_hash",
        "key_id",
        "key_prefix",
        "name",
        "description",
        "scope",
        "status",
        "created_at",
        "expires_at",
        "last_used_at",
        "last_used_ip",
        "usage_count",
        "rate_limit",
        "allowed_ips",
        "metadata",
        "rotated_from",
        "rotated_to",
        "revoked_at",
        "revoked_by",
        "revoke_reason",
        "is_virtual",
        "parent_key_id",
        "org_id",
        "team_id",
        "llm_budget_day_tokens",
        "llm_budget_month_tokens",
        "llm_budget_day_usd",
        "llm_budget_month_usd",
        "llm_allowed_endpoints",
        "llm_allowed_providers",
        "llm_allowed_models",
    },
    "api_key_audit_log": {"id", "api_key_id", "action", "user_id", "ip_address", "user_agent", "details", "created_at"},
}
_API_KEY_INDEXES = {
    "idx_api_keys_user_id": "api_keys",
    "idx_api_keys_key_hash": "api_keys",
    "idx_api_keys_key_id": "api_keys",
    "idx_api_keys_status": "api_keys",
    "idx_api_keys_expires_at": "api_keys",
    "idx_api_keys_virtual": "api_keys",
    "idx_api_keys_org": "api_keys",
    "idx_api_keys_team": "api_keys",
    "idx_api_key_audit_log_api_key_id": "api_key_audit_log",
    "idx_api_key_audit_log_created_at": "api_key_audit_log",
}
_API_KEY_DEFAULTS = {
    "api_keys": {
        "status": ("'active'",),
        "is_virtual": ("FALSE",),
        "usage_count": ("0",),
        "created_at": ("CURRENT_TIMESTAMP", "now()"),
    },
    "api_key_audit_log": {"created_at": ("CURRENT_TIMESTAMP", "now()")},
}

_PROVIDER_SECRET_COLUMNS = {
    "user_provider_secrets": {
        "id",
        "user_id",
        "provider",
        "encrypted_blob",
        "key_hint",
        "metadata",
        "created_by",
        "updated_by",
        "revoked_by",
        "revoked_at",
        "created_at",
        "updated_at",
        "last_used_at",
    }
}
_PROVIDER_SECRET_INDEXES = {
    "idx_user_provider_secrets_user_id": "user_provider_secrets",
    "idx_user_provider_secrets_provider": "user_provider_secrets",
}

_CORE_COLUMNS = {
    "native_chat_quota_reservations": _NATIVE_QUOTA_COLUMNS,
    "users": _USERS_COLUMNS,
    "storage_quotas": {
        "id",
        "org_id",
        "team_id",
        "quota_mb",
        "used_mb",
        "soft_limit_pct",
        "hard_limit_pct",
        "created_at",
        "updated_at",
    },
    "audio_studio_media_tickets": _AUDIO_COLUMNS,
}
_RBAC_COLUMNS = {
    "users": _USERS_COLUMNS,
    "audio_studio_media_tickets": _AUDIO_COLUMNS,
    "roles": {"id", "name", "description", "is_system"},
    "permissions": {"id", "name", "description", "category"},
    "role_permissions": {"role_id", "permission_id"},
    "user_roles": {"user_id", "role_id", "granted_by", "expires_at"},
    "user_permissions": {"user_id", "permission_id", "granted", "expires_at"},
    "registration_codes": {
        "id",
        "code",
        "created_by",
        "created_at",
        "expires_at",
        "max_uses",
        "times_used",
        "is_active",
        "role_id",
    },
    "registration_code_usage": {"id", "code_id", "user_id", "ip_address", "user_agent", "used_at"},
    "auth_audit_log": {"id", "event_type", "user_id", "target_user_id", "details", "created_at"},
}
_SHARED_INDEXES = {
    "idx_users_username": "users",
    "idx_users_email": "users",
    "idx_audio_studio_media_tickets_hash": "audio_studio_media_tickets",
    "idx_audio_studio_media_tickets_expiry": "audio_studio_media_tickets",
    "idx_audio_studio_media_tickets_artifact": "audio_studio_media_tickets",
}
_CORE_INDEXES = {
    "native_chat_quota_user": "native_chat_quota_reservations",
    "native_chat_quota_org": "native_chat_quota_reservations",
    "native_chat_quota_team": "native_chat_quota_reservations",
    **_SHARED_INDEXES,
    "idx_users_uuid": "users",
    "idx_users_role": "users",
    "idx_users_is_active": "users",
    "idx_storage_quotas_org_unique": "storage_quotas",
    "idx_storage_quotas_team_unique": "storage_quotas",
    "idx_orgs_owner": "organizations",
    "idx_teams_org": "teams",
}
_STARTUP_COLUMNS = {
    **_CORE_COLUMNS,
    **_RBAC_COLUMNS,
    "sessions": {
        "id",
        "user_id",
        "token_hash",
        "refresh_token_hash",
        "encrypted_token",
        "encrypted_refresh",
        "expires_at",
        "refresh_expires_at",
        "ip_address",
        "user_agent",
        "device_id",
        "is_active",
        "is_revoked",
        "revoked_at",
        "revoked_by",
        "revoke_reason",
        "access_jti",
        "refresh_jti",
        "created_at",
        "updated_at",
        "last_activity",
    },
    "audit_logs": {
        "id",
        "user_id",
        "action",
        "resource_type",
        "resource_id",
        "ip_address",
        "user_agent",
        "status",
        "details",
        "created_at",
    },
    "retention_policy_overrides": {"policy_key", "days", "updated_at"},
    "password_history": {"id", "user_id", "password_hash", "created_at"},
    "registration_codes": _RBAC_COLUMNS["registration_codes"]
    | {"description", "allowed_email_domain", "org_id", "org_role", "team_id"},
    "rbac_role_rate_limits": {"role_id", "resource", "limit_per_min", "burst"},
    "rbac_user_rate_limits": {"user_id", "resource", "limit_per_min", "burst"},
    "org_invites": {
        "id",
        "code",
        "org_id",
        "team_id",
        "role_to_grant",
        "created_by",
        "created_at",
        "expires_at",
        "max_uses",
        "uses_count",
        "is_active",
        "allowed_email_domain",
        "description",
        "metadata",
    },
    "org_invite_redemptions": {"id", "invite_id", "user_id", "redeemed_at", "ip_address", "user_agent"},
}
_STT_COLUMNS = {
    "org_stt_settings": {
        "org_id",
        "delete_audio_after_success",
        "audio_retention_hours",
        "redact_pii",
        "allow_unredacted_partials",
        "redact_categories_json",
        "updated_by",
        "updated_at",
    }
}
_STARTUP_INDEXES = {
    **_CORE_INDEXES,
    "idx_audit_logs_user_id": "audit_logs",
    "idx_audit_logs_action": "audit_logs",
    "idx_audit_logs_created_at": "audit_logs",
    "idx_sessions_token_hash": "sessions",  # nosec B105 -- canonical table name, not a credential.
    "idx_sessions_user_id": "sessions",
    "idx_sessions_expires_at": "sessions",
    "idx_sessions_access_jti": "sessions",
    "idx_password_history_user_created_at": "password_history",  # nosec B105 -- canonical table name, not a credential.
    "idx_org_invites_code": "org_invites",
    "idx_org_invites_org_active": "org_invites",
    "idx_org_invites_expires": "org_invites",
    "idx_invite_redemptions_invite": "org_invite_redemptions",
    "idx_invite_redemptions_user": "org_invite_redemptions",
}
_STARTUP_NOT_NULL = {
    "sessions": {"user_id", "token_hash", "expires_at"},
    "audit_logs": {"action"},
    "retention_policy_overrides": {"days"},
    "password_history": {"user_id", "password_hash"},
    "org_invites": {"code", "org_id", "expires_at"},
    "org_invite_redemptions": {"invite_id", "user_id"},
}
_STARTUP_DEFAULTS = {
    "sessions": {
        "is_active": ("TRUE",),
        "is_revoked": ("FALSE",),
        "created_at": ("CURRENT_TIMESTAMP", "now()"),
        "updated_at": ("CURRENT_TIMESTAMP", "now()"),
        "last_activity": ("CURRENT_TIMESTAMP", "now()"),
    },
    "audit_logs": {"created_at": ("CURRENT_TIMESTAMP", "now()")},
    "password_history": {"created_at": ("CURRENT_TIMESTAMP", "now()")},
    "org_invites": {
        "is_active": ("TRUE",),
        "uses_count": ("0",),
        "created_at": ("CURRENT_TIMESTAMP", "now()"),
    },
    "org_invite_redemptions": {"redeemed_at": ("CURRENT_TIMESTAMP", "now()")},
}
_STARTUP_GENERATED_IDS = {"sessions", "audit_logs", "password_history", "org_invites", "org_invite_redemptions"}
_COLUMNS_SQL = """
SELECT relation.relname AS table_name, attribute.attname AS column_name, column_type.typname AS data_type,
       attribute.attnotnull AS not_null, pg_get_expr(default_info.adbin, default_info.adrelid) AS default_expression,
       attribute.attidentity IN ('a', 'd') AS is_identity,
       CASE attribute.attidentity WHEN 'a' THEN 'ALWAYS' WHEN 'd' THEN 'BY DEFAULT' END AS identity_generation
FROM pg_class AS relation
JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
JOIN pg_attribute AS attribute ON attribute.attrelid = relation.oid
JOIN pg_type AS column_type ON column_type.oid = attribute.atttypid
LEFT JOIN pg_attrdef AS default_info ON default_info.adrelid = relation.oid AND default_info.adnum = attribute.attnum
WHERE namespace.nspname = 'public' AND relation.relkind IN ('r', 'p')
  AND attribute.attnum > 0 AND NOT attribute.attisdropped
"""
_INDEXES_SQL = """
SELECT index_relation.relname AS index_name, relation.relname AS table_name,
       indexes.indisvalid AND indexes.indisready AS ready,
       indexes.indisunique AS is_unique, indexes.indimmediate AS immediate,
       ARRAY(SELECT attribute.attname
             FROM unnest(indexes.indkey) WITH ORDINALITY AS key_part(number, position)
             LEFT JOIN pg_attribute AS attribute
               ON attribute.attrelid = relation.oid AND attribute.attnum = key_part.number
             WHERE key_part.position <= indexes.indnkeyatts ORDER BY key_part.position) AS columns,
       pg_get_expr(indexes.indpred, indexes.indrelid) AS predicate
FROM pg_index AS indexes
JOIN pg_class AS index_relation ON index_relation.oid = indexes.indexrelid
JOIN pg_class AS relation ON relation.oid = indexes.indrelid
JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
"""
_CONSTRAINTS_SQL = """
SELECT relation.relname AS table_name, constraint_info.contype::text AS kind,
       constraint_info.convalidated AS validated, constraint_info.condeferrable AS deferrable,
       COALESCE(indexes.indisvalid AND indexes.indisready, TRUE) AS index_ready,
       ARRAY(SELECT attribute.attname
             FROM unnest(constraint_info.conkey) WITH ORDINALITY AS key_part(number, position)
             JOIN pg_attribute AS attribute
               ON attribute.attrelid = relation.oid AND attribute.attnum = key_part.number
             ORDER BY key_part.position) AS columns,
       foreign_namespace.nspname AS foreign_schema, foreign_relation.relname AS foreign_table,
       ARRAY(SELECT attribute.attname
             FROM unnest(constraint_info.confkey) WITH ORDINALITY AS key_part(number, position)
             JOIN pg_attribute AS attribute
               ON attribute.attrelid = foreign_relation.oid AND attribute.attnum = key_part.number
             ORDER BY key_part.position) AS foreign_columns,
       constraint_info.confdeltype::text AS delete_action, constraint_info.confupdtype::text AS update_action,
       pg_get_expr(constraint_info.conbin, constraint_info.conrelid) AS check_expression
FROM pg_constraint AS constraint_info
JOIN pg_class AS relation ON relation.oid = constraint_info.conrelid
JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
LEFT JOIN pg_class AS foreign_relation ON foreign_relation.oid = constraint_info.confrelid
LEFT JOIN pg_namespace AS foreign_namespace ON foreign_namespace.oid = foreign_relation.relnamespace
LEFT JOIN pg_index AS indexes ON indexes.indexrelid = constraint_info.conindid
WHERE namespace.nspname = 'public'
"""
_PRIMARY_KEYS = {
    "native_chat_quota_reservations": ("user_id", "candidate_key"),
    "org_stt_settings": ("org_id",),
    "retention_policy_overrides": ("policy_key",),
    "rbac_role_rate_limits": ("role_id", "resource"),
    "rbac_user_rate_limits": ("user_id", "resource"),
    "role_permissions": ("role_id", "permission_id"),
    "user_roles": ("user_id", "role_id"),
    "user_permissions": ("user_id", "permission_id"),
}
_UNIQUE_KEYS = {
    "user_provider_secrets": {("user_id", "provider")},
    "api_keys": {("key_hash",), ("key_id",)},
    "users": {("uuid",), ("username",), ("email",)},
    "org_invites": {("code",)},
    "org_invite_redemptions": {("invite_id", "user_id")},
    "roles": {("name",)},
    "permissions": {("name",)},
    "registration_codes": {("code",)},
    "audio_studio_media_tickets": {("token_hash",)},
}
_FOREIGN_KEYS = {
    "user_provider_secrets": {(("user_id",), "users", ("id",), "c")},
    "api_keys": {
        (("user_id",), "users", ("id",), "c"),
        (("parent_key_id",), "api_keys", ("id",), "a"),
        (("rotated_from",), "api_keys", ("id",), "a"),
        (("rotated_to",), "api_keys", ("id",), "a"),
        (("org_id",), "organizations", ("id",), "n"),
        (("team_id",), "teams", ("id",), "n"),
    },
    "api_key_audit_log": {(("api_key_id",), "api_keys", ("id",), "c"), (("user_id",), "users", ("id",), "n")},
    "sessions": {(("user_id",), "users", ("id",), "c")},
    "audit_logs": {(("user_id",), "users", ("id",), "n")},
    "password_history": {(("user_id",), "users", ("id",), "c")},
    "rbac_role_rate_limits": {(("role_id",), "roles", ("id",), "c")},
    "rbac_user_rate_limits": {(("user_id",), "users", ("id",), "c")},
    "org_invites": {
        (("org_id",), "organizations", ("id",), "c"),
        (("team_id",), "teams", ("id",), "c"),
        (("created_by",), "users", ("id",), "n"),
    },
    "org_invite_redemptions": {(("invite_id",), "org_invites", ("id",), "c"), (("user_id",), "users", ("id",), "c")},
    "org_stt_settings": {(("org_id",), "organizations", ("id",), "c"), (("updated_by",), "users", ("id",), "n")},
    "storage_quotas": {(("org_id",), "organizations", ("id",), "c"), (("team_id",), "teams", ("id",), "c")},
    "role_permissions": {(("role_id",), "roles", ("id",), "c"), (("permission_id",), "permissions", ("id",), "c")},
    "user_roles": {(("user_id",), "users", ("id",), "c"), (("role_id",), "roles", ("id",), "c")},
    "user_permissions": {(("user_id",), "users", ("id",), "c"), (("permission_id",), "permissions", ("id",), "c")},
    "registration_codes": {(("role_id",), "roles", ("id",), "n")},
    "registration_code_usage": {
        (("code_id",), "registration_codes", ("id",), "c"),
        (("user_id",), "users", ("id",), "c"),
    },
    "auth_audit_log": {(("user_id",), "users", ("id",), "n"), (("target_user_id",), "users", ("id",), "n")},
}
_CHECKS = {
    "storage_quotas": (
        "org_id IS NULL OR team_id IS NULL",
        "(org_id IS NOT NULL AND team_id IS NULL) OR (org_id IS NULL AND team_id IS NOT NULL) "
        "OR (org_id IS NULL AND team_id IS NULL)",
    ),
    "audio_studio_media_tickets": (
        "purpose IN ('playback', 'download')",
        "purpose = ANY (ARRAY['playback', 'download'])",
    ),
}
_INDEX_COLUMNS = {
    "idx_user_provider_secrets_user_id": ("user_id",),
    "idx_user_provider_secrets_provider": ("provider",),
    "idx_api_keys_user_id": ("user_id",),
    "idx_api_keys_key_hash": ("key_hash",),
    "idx_api_keys_key_id": ("key_id",),
    "idx_api_keys_status": ("status",),
    "idx_api_keys_expires_at": ("expires_at",),
    "idx_api_keys_virtual": ("is_virtual",),
    "idx_api_keys_org": ("org_id",),
    "idx_api_keys_team": ("team_id",),
    "idx_api_key_audit_log_api_key_id": ("api_key_id",),
    "idx_api_key_audit_log_created_at": ("created_at",),
    "native_chat_quota_user": ("user_id", "user_uuid", "state"),
    "native_chat_quota_org": ("org_id", "state"),
    "native_chat_quota_team": ("team_id", "state"),
    "idx_audit_logs_user_id": ("user_id",),
    "idx_audit_logs_action": ("action",),
    "idx_audit_logs_created_at": ("created_at",),
    "idx_sessions_token_hash": ("token_hash",),
    "idx_sessions_user_id": ("user_id",),
    "idx_sessions_expires_at": ("expires_at",),
    "idx_sessions_access_jti": ("access_jti",),
    "idx_password_history_user_created_at": ("user_id", "created_at", "id"),
    "idx_org_invites_code": ("code",),
    "idx_org_invites_org_active": ("org_id", "is_active"),
    "idx_org_invites_expires": ("expires_at",),
    "idx_invite_redemptions_invite": ("invite_id",),
    "idx_invite_redemptions_user": ("user_id",),
    "idx_org_stt_settings_updated_by": ("updated_by",),
    "idx_users_username": ("username",),
    "idx_users_email": ("email",),
    "idx_users_uuid": ("uuid",),
    "idx_users_role": ("role",),
    "idx_users_is_active": ("is_active",),
    "idx_orgs_owner": ("owner_user_id",),
    "idx_teams_org": ("org_id",),
    "idx_storage_quotas_org_unique": ("org_id",),
    "idx_storage_quotas_team_unique": ("team_id",),
    "idx_audio_studio_media_tickets_hash": ("token_hash",),
    "idx_audio_studio_media_tickets_expiry": ("expires_at",),
    "idx_audio_studio_media_tickets_artifact": ("user_id", "project_id", "artifact_id"),
}


def _check_matches(actual: str | None, expected: tuple[str, ...]) -> bool:
    """Compare fixed check forms without executing catalog-provided expressions."""
    if not isinstance(actual, str):
        return False

    def canonical(expression: str) -> exp.Expression:
        """Remove only nontruncating text casts and redundant parentheses."""

        def decoration(node: exp.Expression) -> exp.Expression:
            """Normalize descendants even when the current wrapper is replaced."""
            if isinstance(node, exp.Paren):
                return node.this.transform(decoration, copy=False)
            if isinstance(node, exp.Cast):
                target = node.to
                scalar_text = (
                    target.this in {exp.DataType.Type.TEXT, exp.DataType.Type.VARCHAR} and not target.expressions
                )
                cast_value = node.this
                while isinstance(cast_value, exp.Paren):
                    cast_value = cast_value.this
                text_array = (
                    target.this == exp.DataType.Type.ARRAY
                    and len(target.expressions) == 1
                    and target.expressions[0].this == exp.DataType.Type.TEXT
                    and not target.expressions[0].expressions
                    and isinstance(cast_value, exp.Array)
                )
                if scalar_text or text_array:
                    return node.this.transform(decoration, copy=False)
            return node

        return parse_one(expression, read="postgres").transform(decoration)

    try:
        parsed = canonical(actual)
        return any(parsed == canonical(candidate) for candidate in expected)
    except (ParseError, ValueError, TypeError):
        return False


def _column_type_allowed(column: str, data_type: str) -> bool:
    """Accept declared legacy families while refusing incompatible serving values."""
    if column in {
        "is_active",
        "is_verified",
        "is_superuser",
        "email_verified",
        "two_factor_enabled",
        "is_system",
        "granted",
        "is_revoked",
        "is_virtual",
        "delete_audio_after_success",
        "redact_pii",
        "allow_unredacted_partials",
    }:
        return data_type == "bool"
    if column in {
        "id",
        "user_id",
        "org_id",
        "team_id",
        "role_id",
        "permission_id",
        "code_id",
        "created_by",
        "target_user_id",
        "granted_by",
        "max_uses",
        "times_used",
        "storage_quota_mb",
        "quota_mb",
        "soft_limit_pct",
        "hard_limit_pct",
        "failed_login_attempts",
        "resource_id",
        "revoked_by",
        "updated_by",
        "invite_id",
        "uses_count",
        "days",
        "limit_per_min",
        "burst",
        "api_key_id",
        "parent_key_id",
        "rotated_from",
        "rotated_to",
        "rate_limit",
        "usage_count",
        "llm_budget_day_tokens",
        "llm_budget_month_tokens",
    }:
        return data_type in {"int2", "int4", "int8"}
    if column in {"llm_budget_day_usd", "llm_budget_month_usd"}:
        return data_type in {"float4", "float8", "numeric"}
    if column in {"storage_used_mb", "used_mb", "audio_retention_hours"}:
        return data_type in {"int2", "int4", "int8", "float4", "float8", "numeric"}
    if column in {
        "created_at",
        "updated_at",
        "profile_version",
        "last_login",
        "locked_until",
        "email_verified_at",
        "password_changed_at",
        "expires_at",
        "consumed_at",
        "revoked_at",
        "last_redeemed_at",
        "last_used_at",
        "used_at",
        "last_activity",
        "refresh_expires_at",
        "redeemed_at",
    }:
        return data_type in {"timestamp", "timestamptz"}
    if column == "size_bytes":
        return data_type == "int8"
    if column == "redact_categories_json":
        return data_type == "jsonb"
    if column == "uuid":
        return data_type in {"uuid", "text", "varchar"}
    if column in {"metadata", "details", "llm_allowed_endpoints", "llm_allowed_providers", "llm_allowed_models"}:
        return data_type in {"json", "jsonb", "text", "varchar"}
    return data_type in {"text", "varchar"}


def _validate_required_metadata(
    columns: Any, indexes: Any, constraints: Any, required: dict[str, set[str]], required_indexes: dict[str, str]
) -> None:
    """Require fixed bootstrap fields without rejecting supported legacy types."""
    actual: dict[str, set[str]] = {}
    for row in columns:
        actual.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
    if any(not names <= actual.get(table, set()) for table, names in required.items()):
        raise SchemaReadinessError("PostgreSQL AuthNZ runtime schema requires owner migration")
    if any(
        row["table_name"] in required
        and row["column_name"] in required[row["table_name"]]
        and not _column_type_allowed(str(row["column_name"]), str(row["data_type"]))
        for row in columns
    ):
        raise SchemaReadinessError("PostgreSQL AuthNZ runtime column type requires owner migration")
    ready = {str(row["index_name"]): row for row in indexes if row["ready"]}
    for name, table in required_indexes.items():
        row = ready.get(name)
        if not row or row["table_name"] != table or tuple(row["columns"]) != _INDEX_COLUMNS[name]:
            raise SchemaReadinessError("PostgreSQL AuthNZ runtime indexes require owner migration")
        if table == "native_chat_quota_reservations" and row["is_unique"] is not False:
            raise SchemaReadinessError("PostgreSQL native quota indexes require owner migration")
        if table == "storage_quotas":
            if not row["is_unique"] or not _check_matches(row["predicate"], (f"{row['columns'][0]} IS NOT NULL",)):
                raise SchemaReadinessError("PostgreSQL quota uniqueness requires owner migration")
        elif row["predicate"] is not None:
            raise SchemaReadinessError("PostgreSQL AuthNZ runtime indexes require owner migration")

    valid = [row for row in constraints if row["validated"] and row["index_ready"]]
    for table in required:
        if table == "native_chat_quota_reservations":
            # Serving cannot skip the retained-byte authority installed by the owner.
            if any(
                row["data_type"] != ("int8" if row["column_name"] in _NATIVE_QUOTA_INTEGERS else "text")
                or row["not_null"] != (row["column_name"] not in _NATIVE_QUOTA_NULLABLE)
                for row in columns
                if row["table_name"] == table and row["column_name"] in _NATIVE_QUOTA_COLUMNS
            ):
                raise SchemaReadinessError("PostgreSQL native quota columns require owner migration")
            if any(row.get("deferrable", True) for row in valid if row["table_name"] == table and row["kind"] == "p"):
                raise SchemaReadinessError("PostgreSQL native quota primary key requires owner migration")
            if not all(
                any(
                    row["table_name"] == table
                    and row["kind"] == "c"
                    and _check_matches(row["check_expression"], expected)
                    for row in valid
                )
                for expected in _NATIVE_QUOTA_CHECKS
            ):
                raise SchemaReadinessError("PostgreSQL native quota checks require owner migration")
        primary = _PRIMARY_KEYS.get(table, ("id",))
        if not any(
            row["table_name"] == table and row["kind"] == "p" and tuple(row["columns"]) == primary for row in valid
        ):
            raise SchemaReadinessError("PostgreSQL AuthNZ primary key requires owner migration")
        unique = {
            tuple(row["columns"])
            for row in indexes
            if row["table_name"] == table and row["ready"] and row["is_unique"] and row["predicate"] is None
        }
        if not _UNIQUE_KEYS.get(table, set()) <= unique:
            raise SchemaReadinessError("PostgreSQL AuthNZ unique key requires owner migration")
        foreign = {
            (tuple(row["columns"]), row["foreign_table"], tuple(row["foreign_columns"]), row["delete_action"])
            for row in valid
            if row["table_name"] == table
            and row["kind"] == "f"
            and row["foreign_schema"] == "public"
            and row["update_action"] == "a"
        }
        if not _FOREIGN_KEYS.get(table, set()) <= foreign:
            raise SchemaReadinessError("PostgreSQL AuthNZ foreign key requires owner migration")
        if table in _CHECKS and not any(
            row["table_name"] == table
            and row["kind"] == "c"
            and _check_matches(row["check_expression"], _CHECKS[table])
            for row in valid
        ):
            raise SchemaReadinessError("PostgreSQL AuthNZ check requires owner migration")


def _validate_metadata(columns: Any, indexes: Any, constraints: Any, *, rbac: bool) -> None:
    """Keep the pool and synchronous RBAC admission contracts distinct."""
    _validate_required_metadata(
        columns,
        indexes,
        constraints,
        _RBAC_COLUMNS if rbac else _CORE_COLUMNS,
        _SHARED_INDEXES if rbac else _CORE_INDEXES,
    )


def validate_postgres_rbac_runtime(executor: Any, *, connection: Any) -> None:
    """Validate the configured synchronous RBAC store without seeding or DDL."""
    _validate_metadata(
        executor.execute(_COLUMNS_SQL, connection=connection).rows,
        executor.execute(_INDEXES_SQL, connection=connection).rows,
        executor.execute(_CONSTRAINTS_SQL, connection=connection).rows,
        rbac=True,
    )
