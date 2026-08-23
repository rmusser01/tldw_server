"""PostgreSQL schema ownership for canonical workspace sharing tables."""

from __future__ import annotations

from typing import Any

POSTGRES_SHARING_SCHEMA_STATEMENTS = [
    (
        """
        CREATE TABLE IF NOT EXISTS shared_workspaces (
            id SERIAL PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            owner_user_id INTEGER NOT NULL REFERENCES users(id),
            share_scope_type TEXT NOT NULL DEFAULT 'team',
            share_scope_id INTEGER NOT NULL,
            access_level TEXT NOT NULL DEFAULT 'view_chat',
            allow_clone BOOLEAN NOT NULL DEFAULT TRUE,
            created_by INTEGER NOT NULL REFERENCES users(id),
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            revoked_at TIMESTAMPTZ,
            CONSTRAINT ck_shared_workspaces_scope_type
                CHECK (share_scope_type IN ('team', 'org')),
            CONSTRAINT ck_shared_workspaces_access_level
                CHECK (access_level IN ('view_chat', 'view_chat_add', 'full_edit')),
            CONSTRAINT uq_shared_workspaces_scope
                UNIQUE (workspace_id, owner_user_id, share_scope_type, share_scope_id)
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_shared_ws_owner "
        "ON shared_workspaces(owner_user_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_shared_ws_scope "
        "ON shared_workspaces(share_scope_type, share_scope_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS share_tokens (
            id SERIAL PRIMARY KEY,
            token_hash TEXT UNIQUE NOT NULL,
            token_prefix TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            owner_user_id INTEGER NOT NULL REFERENCES users(id),
            access_level TEXT NOT NULL DEFAULT 'view_chat',
            allow_clone BOOLEAN NOT NULL DEFAULT TRUE,
            password_hash TEXT,
            max_uses INTEGER,
            use_count INTEGER NOT NULL DEFAULT 0,
            expires_at TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            revoked_at TIMESTAMPTZ,
            CONSTRAINT ck_share_tokens_resource_type
                CHECK (resource_type IN ('chatbook', 'workspace', 'prototype_workspace'))
        )
        """,
        (),
    ),
    (
        """
        DO $$
        DECLARE
            canonical_constraint_exists BOOLEAN;
            stale_constraint_name TEXT;
        BEGIN
            FOR stale_constraint_name IN
                SELECT c.conname
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = current_schema()
                  AND t.relname = 'share_tokens'
                  AND c.contype = 'c'
                  AND replace(
                      regexp_replace(
                          lower(pg_get_constraintdef(c.oid)),
                          '[[:space:]"()]',
                          '',
                          'g'
                      ),
                      '::text',
                      ''
                  ) IN (
                      'checkresource_type=anyarray[''chatbook'',''workspace'']',
                      'checkresource_type=anyarray[''workspace'',''chatbook'']'
                  )
            LOOP
                EXECUTE format(
                    'ALTER TABLE share_tokens DROP CONSTRAINT %I',
                    stale_constraint_name
                );
            END LOOP;

            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = current_schema()
                  AND t.relname = 'share_tokens'
                  AND c.conname = 'ck_share_tokens_resource_type'
            )
            INTO canonical_constraint_exists;

            IF NOT canonical_constraint_exists
            THEN
                ALTER TABLE share_tokens
                    ADD CONSTRAINT ck_share_tokens_resource_type
                    CHECK (
                        resource_type IN (
                            'chatbook', 'workspace', 'prototype_workspace'
                        )
                    ) NOT VALID;
                ALTER TABLE share_tokens
                    VALIDATE CONSTRAINT ck_share_tokens_resource_type;
            END IF;
        END
        $$
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_share_tokens_prefix ON share_tokens(token_prefix)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_share_tokens_owner ON share_tokens(owner_user_id)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_share_tokens_resource "
        "ON share_tokens(resource_type, resource_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS share_audit_log (
            id SERIAL PRIMARY KEY,
            event_type TEXT NOT NULL,
            actor_user_id INTEGER,
            resource_type TEXT NOT NULL,
            resource_id TEXT NOT NULL,
            owner_user_id INTEGER NOT NULL,
            share_id INTEGER,
            token_id INTEGER,
            metadata_json TEXT DEFAULT '{}',
            ip_address TEXT,
            user_agent TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_share_audit_created ON share_audit_log(created_at)",
        (),
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_share_audit_owner ON share_audit_log(owner_user_id)",
        (),
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS sharing_config (
            id SERIAL PRIMARY KEY,
            scope_type TEXT NOT NULL DEFAULT 'global',
            scope_id INTEGER,
            config_key TEXT NOT NULL,
            config_value TEXT NOT NULL,
            updated_by INTEGER,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT ck_sharing_config_scope_type
                CHECK (scope_type IN ('global', 'org', 'team')),
            CONSTRAINT uq_sharing_config_scope_key
                UNIQUE (scope_type, scope_id, config_key)
        )
        """,
        (),
    ),
    (
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_sharing_config_global_key
        ON sharing_config(scope_type, config_key)
        WHERE scope_id IS NULL
        """,
        (),
    ),
]


POSTGRES_SHARING_SCHEMA_ISSUES_SQL = """
WITH required_columns(table_name, column_name, data_type, is_nullable) AS (
    VALUES
        ('shared_workspaces', 'id', 'integer', 'NO'),
        ('shared_workspaces', 'workspace_id', 'text', 'NO'),
        ('shared_workspaces', 'owner_user_id', 'integer', 'NO'),
        ('shared_workspaces', 'share_scope_type', 'text', 'NO'),
        ('shared_workspaces', 'share_scope_id', 'integer', 'NO'),
        ('shared_workspaces', 'access_level', 'text', 'NO'),
        ('shared_workspaces', 'allow_clone', 'boolean', 'NO'),
        ('shared_workspaces', 'created_by', 'integer', 'NO'),
        ('shared_workspaces', 'created_at', 'timestamp with time zone', 'YES'),
        ('shared_workspaces', 'updated_at', 'timestamp with time zone', 'YES'),
        ('shared_workspaces', 'revoked_at', 'timestamp with time zone', 'YES'),
        ('share_tokens', 'id', 'integer', 'NO'),
        ('share_tokens', 'token_hash', 'text', 'NO'),
        ('share_tokens', 'token_prefix', 'text', 'NO'),
        ('share_tokens', 'resource_type', 'text', 'NO'),
        ('share_tokens', 'resource_id', 'text', 'NO'),
        ('share_tokens', 'owner_user_id', 'integer', 'NO'),
        ('share_tokens', 'access_level', 'text', 'NO'),
        ('share_tokens', 'allow_clone', 'boolean', 'NO'),
        ('share_tokens', 'password_hash', 'text', 'YES'),
        ('share_tokens', 'max_uses', 'integer', 'YES'),
        ('share_tokens', 'use_count', 'integer', 'NO'),
        ('share_tokens', 'expires_at', 'text', 'YES'),
        ('share_tokens', 'created_at', 'timestamp with time zone', 'YES'),
        ('share_tokens', 'revoked_at', 'timestamp with time zone', 'YES'),
        ('share_audit_log', 'id', 'integer', 'NO'),
        ('share_audit_log', 'event_type', 'text', 'NO'),
        ('share_audit_log', 'actor_user_id', 'integer', 'YES'),
        ('share_audit_log', 'resource_type', 'text', 'NO'),
        ('share_audit_log', 'resource_id', 'text', 'NO'),
        ('share_audit_log', 'owner_user_id', 'integer', 'NO'),
        ('share_audit_log', 'share_id', 'integer', 'YES'),
        ('share_audit_log', 'token_id', 'integer', 'YES'),
        ('share_audit_log', 'metadata_json', 'text', 'YES'),
        ('share_audit_log', 'ip_address', 'text', 'YES'),
        ('share_audit_log', 'user_agent', 'text', 'YES'),
        ('share_audit_log', 'created_at', 'timestamp with time zone', 'YES'),
        ('sharing_config', 'id', 'integer', 'NO'),
        ('sharing_config', 'scope_type', 'text', 'NO'),
        ('sharing_config', 'scope_id', 'integer', 'YES'),
        ('sharing_config', 'config_key', 'text', 'NO'),
        ('sharing_config', 'config_value', 'text', 'NO'),
        ('sharing_config', 'updated_by', 'integer', 'YES'),
        ('sharing_config', 'updated_at', 'timestamp with time zone', 'YES')
),
required_defaults(table_name, column_name, default_fragment) AS (
    VALUES
        ('shared_workspaces', 'id', 'nextval('),
        ('shared_workspaces', 'share_scope_type', 'team'),
        ('shared_workspaces', 'access_level', 'view_chat'),
        ('shared_workspaces', 'allow_clone', 'true'),
        ('shared_workspaces', 'created_at', 'current_timestamp'),
        ('shared_workspaces', 'updated_at', 'current_timestamp'),
        ('share_tokens', 'id', 'nextval('),
        ('share_tokens', 'access_level', 'view_chat'),
        ('share_tokens', 'allow_clone', 'true'),
        ('share_tokens', 'use_count', '0'),
        ('share_tokens', 'created_at', 'current_timestamp'),
        ('share_audit_log', 'id', 'nextval('),
        ('share_audit_log', 'metadata_json', '{}'),
        ('share_audit_log', 'created_at', 'current_timestamp'),
        ('sharing_config', 'id', 'nextval('),
        ('sharing_config', 'scope_type', 'global'),
        ('sharing_config', 'updated_at', 'current_timestamp')
),
required_constraints(table_name, constraint_name, definition_fragment) AS (
    VALUES
        ('shared_workspaces', 'shared_workspaces_pkey', 'primarykey(id)'),
        ('shared_workspaces', 'shared_workspaces_owner_user_id_fkey', 'foreignkey(owner_user_id)'),
        ('shared_workspaces', 'shared_workspaces_created_by_fkey', 'foreignkey(created_by)'),
        ('shared_workspaces', 'ck_shared_workspaces_scope_type', NULL),
        ('shared_workspaces', 'ck_shared_workspaces_access_level', NULL),
        (
            'shared_workspaces',
            'uq_shared_workspaces_scope',
            'unique(workspace_id,owner_user_id,share_scope_type,share_scope_id)'
        ),
        ('share_tokens', 'share_tokens_pkey', 'primarykey(id)'),
        ('share_tokens', 'share_tokens_token_hash_key', 'unique(token_hash)'),
        ('share_tokens', 'share_tokens_owner_user_id_fkey', 'foreignkey(owner_user_id)'),
        ('share_tokens', 'ck_share_tokens_resource_type', NULL),
        ('share_audit_log', 'share_audit_log_pkey', 'primarykey(id)'),
        ('sharing_config', 'sharing_config_pkey', 'primarykey(id)'),
        ('sharing_config', 'ck_sharing_config_scope_type', NULL),
        (
            'sharing_config',
            'uq_sharing_config_scope_key',
            'unique(scope_type,scope_id,config_key)'
        )
),
required_check_constraints(table_name, constraint_name, normalized_definition) AS (
    VALUES
        (
            'shared_workspaces',
            'ck_shared_workspaces_scope_type',
            'checkshare_scope_type=anyarray[''team'',''org'']'
        ),
        (
            'shared_workspaces',
            'ck_shared_workspaces_access_level',
            'checkaccess_level=anyarray[''view_chat'',''view_chat_add'',''full_edit'']'
        ),
        (
            'share_tokens',
            'ck_share_tokens_resource_type',
            'checkresource_type=anyarray[''chatbook'',''workspace'',''prototype_workspace'']'
        ),
        (
            'sharing_config',
            'ck_sharing_config_scope_type',
            'checkscope_type=anyarray[''global'',''org'',''team'']'
        )
),
required_indexes(
    table_name,
    index_name,
    key_fragment,
    is_unique,
    predicate_fragment
) AS (
    VALUES
        ('shared_workspaces', 'idx_shared_ws_owner', 'owner_user_id', FALSE, NULL),
        (
            'shared_workspaces',
            'idx_shared_ws_scope',
            'share_scope_type,share_scope_id',
            FALSE,
            NULL
        ),
        ('share_tokens', 'idx_share_tokens_prefix', 'token_prefix', FALSE, NULL),
        ('share_tokens', 'idx_share_tokens_owner', 'owner_user_id', FALSE, NULL),
        (
            'share_tokens',
            'idx_share_tokens_resource',
            'resource_type,resource_id',
            FALSE,
            NULL
        ),
        ('share_audit_log', 'idx_share_audit_created', 'created_at', FALSE, NULL),
        ('share_audit_log', 'idx_share_audit_owner', 'owner_user_id', FALSE, NULL),
        (
            'sharing_config',
            'uq_sharing_config_global_key',
            'scope_type,config_key',
            TRUE,
            'where(scope_idisnull)'
        )
),
index_catalog AS (
    SELECT
        i.tablename,
        i.indexname,
        regexp_replace(lower(i.indexdef), '[[:space:]"]', '', 'g') AS definition,
        x.indisvalid,
        x.indisready
    FROM pg_indexes i
    JOIN pg_class ic ON ic.relname = i.indexname
    JOIN pg_namespace n ON n.oid = ic.relnamespace
    JOIN pg_index x ON x.indexrelid = ic.oid
    WHERE i.schemaname = current_schema()
      AND n.nspname = current_schema()
),
sharing_schema_issues(issue) AS (
    SELECT 'missing column ' || r.table_name || '.' || r.column_name
    FROM required_columns r
    WHERE NOT EXISTS (
        SELECT 1
        FROM information_schema.columns c
        WHERE c.table_schema = current_schema()
          AND c.table_name = r.table_name
          AND c.column_name = r.column_name
    )
    UNION ALL
    SELECT 'invalid column ' || r.table_name || '.' || r.column_name
    FROM required_columns r
    WHERE EXISTS (
        SELECT 1
        FROM information_schema.columns c
        WHERE c.table_schema = current_schema()
          AND c.table_name = r.table_name
          AND c.column_name = r.column_name
    )
      AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns c
        WHERE c.table_schema = current_schema()
          AND c.table_name = r.table_name
          AND c.column_name = r.column_name
          AND c.data_type = r.data_type
          AND c.is_nullable = r.is_nullable
    )
    UNION ALL
    SELECT 'invalid default ' || r.table_name || '.' || r.column_name
    FROM required_defaults r
    WHERE NOT EXISTS (
        SELECT 1
        FROM information_schema.columns c
        WHERE c.table_schema = current_schema()
          AND c.table_name = r.table_name
          AND c.column_name = r.column_name
          AND lower(COALESCE(c.column_default, '')) LIKE
              '%' || r.default_fragment || '%'
    )
    UNION ALL
    SELECT 'invalid constraint ' || r.table_name || '.' || r.constraint_name
    FROM required_constraints r
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = current_schema()
          AND t.relname = r.table_name
          AND c.conname = r.constraint_name
          AND c.convalidated
          AND (
              r.definition_fragment IS NULL
              OR regexp_replace(
                  lower(pg_get_constraintdef(c.oid)),
                  '[[:space:]"]',
                  '',
                  'g'
              ) LIKE '%' || r.definition_fragment || '%'
          )
    )
    UNION ALL
    SELECT 'invalid constraint ' || r.table_name || '.' || r.constraint_name
    FROM required_check_constraints r
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = current_schema()
          AND t.relname = r.table_name
          AND c.conname = r.constraint_name
          AND c.convalidated
          AND replace(
              regexp_replace(
                  lower(pg_get_constraintdef(c.oid)),
                  '[[:space:]"()]',
                  '',
                  'g'
              ),
              '::text',
              ''
          ) = r.normalized_definition
    )
    UNION ALL
    SELECT 'invalid constraint share_tokens.' || c.conname
    FROM pg_constraint c
    JOIN pg_class t ON t.oid = c.conrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = current_schema()
      AND t.relname = 'share_tokens'
      AND c.contype = 'c'
      AND c.conname <> 'ck_share_tokens_resource_type'
      AND lower(pg_get_constraintdef(c.oid)) LIKE '%resource_type%'
    UNION ALL
    SELECT 'missing or invalid index ' || r.table_name || '.' || r.index_name
    FROM required_indexes r
    WHERE NOT EXISTS (
        SELECT 1
        FROM index_catalog i
        WHERE i.tablename = r.table_name
          AND i.indexname = r.index_name
          AND i.indisvalid
          AND i.indisready
          AND i.definition LIKE '%(' || r.key_fragment || ')%'
          AND (NOT r.is_unique OR i.definition LIKE 'createuniqueindex%')
          AND (
              (
                  r.predicate_fragment IS NULL
                  AND i.definition NOT LIKE '%where%'
              )
              OR i.definition LIKE '%' || r.predicate_fragment || '%'
          )
    )
)
SELECT DISTINCT issue
FROM sharing_schema_issues
ORDER BY issue
"""

async def apply_postgres_sharing_schema(pool: Any) -> None:
    """Apply every idempotent canonical sharing schema statement."""
    for sql, params in POSTGRES_SHARING_SCHEMA_STATEMENTS:
        await pool.execute(sql, *params)


async def postgres_sharing_schema_issues(pool: Any) -> list[str]:
    """Return canonical sharing schema contract violations from PostgreSQL."""
    rows = await pool.fetchall(POSTGRES_SHARING_SCHEMA_ISSUES_SQL, ())
    return [str(row["issue"]) for row in rows if row.get("issue")]


__all__ = [
    "apply_postgres_sharing_schema",
    "postgres_sharing_schema_issues",
]
