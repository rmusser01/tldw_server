"""Registered PostgreSQL Skill registry owner isolation and drift verification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OWNER_DEFAULT = "NULLIF(current_setting('app.current_user_id', true), '')"
OWNER_PREDICATE = "client_id = NULLIF(current_setting('app.current_user_id', true), '')"


def postgres_skill_registry_statements() -> tuple[str, ...]:
    """Preserve unowned legacy rows; only future writes get the session owner."""
    return (
        """CREATE TABLE IF NOT EXISTS skill_registry (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            argument_hint TEXT,
            disable_model_invocation BOOLEAN DEFAULT FALSE,
            user_invocable BOOLEAN DEFAULT TRUE,
            allowed_tools TEXT,
            model TEXT,
            context TEXT DEFAULT 'inline',
            directory_path TEXT NOT NULL,
            last_modified TIMESTAMP NOT NULL DEFAULT NOW(),
            file_hash TEXT,
            uuid TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            deleted BOOLEAN DEFAULT FALSE,
            version INTEGER NOT NULL DEFAULT 1
        )""",
        "ALTER TABLE skill_registry ADD COLUMN IF NOT EXISTS client_id TEXT",
        "ALTER TABLE skill_registry ALTER COLUMN client_id SET DEFAULT "
        "NULLIF(current_setting('app.current_user_id', true), '')",
        "ALTER TABLE skill_registry DROP CONSTRAINT IF EXISTS skill_registry_name_key",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_skill_registry_owner_name ON skill_registry(client_id, name)",
        "CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_registry(name)",
        "CREATE INDEX IF NOT EXISTS idx_skill_user_invocable ON skill_registry(user_invocable)",
        "CREATE INDEX IF NOT EXISTS idx_skill_deleted ON skill_registry(deleted)",
        "ALTER TABLE skill_registry ENABLE ROW LEVEL SECURITY",
        "ALTER TABLE skill_registry FORCE ROW LEVEL SECURITY",
        "DROP POLICY IF EXISTS skill_registry_tenant_isolation ON skill_registry",
        "CREATE POLICY skill_registry_tenant_isolation ON skill_registry "
        "USING (client_id = NULLIF(current_setting('app.current_user_id', true), '')) "
        "WITH CHECK (client_id = NULLIF(current_setting('app.current_user_id', true), ''))",
    )


def verify_postgres_skill_registry(backend: Any, conn: Any, normalize: Callable[[Any], str]) -> None:
    """Fail closed on owner/default/uniqueness/policy drift without request DDL."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    backend.execute("LOCK TABLE skill_registry IN ACCESS SHARE MODE", connection=conn)
    tables = backend.execute(
        "SELECT c.relrowsecurity,c.relforcerowsecurity FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') AND c.relname='skill_registry'",
        connection=conn,
    ).rows
    if len(tables) != 1 or not tables[0]["relrowsecurity"] or not tables[0]["relforcerowsecurity"]:
        raise SchemaError("Skill registry PostgreSQL tenant schema is missing or unsafe.")
    columns = backend.execute(
        "SELECT data_type,is_nullable,column_default FROM information_schema.columns "
        "WHERE table_schema=current_schema() AND table_name='skill_registry' AND column_name='client_id'",
        connection=conn,
    ).rows
    if (
        len(columns) != 1
        or columns[0]["data_type"] != "text"
        or columns[0]["is_nullable"] != "YES"
        or normalize(columns[0]["column_default"]) != normalize(OWNER_DEFAULT)
    ):
        raise SchemaError("Skill registry PostgreSQL owner column drifted.")
    indexes = backend.execute(
        "SELECT i.indisvalid,i.indisready,i.indpred IS NULL AS complete, "
        "i.indexprs IS NULL AS expression_free,i.indnkeyatts AS key_count, "
        "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum,ord) "
        "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum ORDER BY k.ord) AS columns "
        "FROM pg_index i WHERE i.indrelid='skill_registry'::regclass AND i.indisunique",
        connection=conn,
    ).rows
    allowed_unique_keys = {("id",), ("uuid",), ("client_id", "name")}
    if {tuple(row["columns"]) for row in indexes} != allowed_unique_keys or any(
        tuple(row["columns"]) not in allowed_unique_keys
        or not row["indisvalid"]
        or not row["indisready"]
        or not row["complete"]
        or not row["expression_free"]
        or row["key_count"] != len(row["columns"])
        for row in indexes
    ):
        raise SchemaError("Skill registry PostgreSQL owner-name uniqueness drifted.")
    policies = backend.execute(
        "SELECT policyname,permissive,roles::text,cmd,qual,with_check FROM pg_policies "
        "WHERE schemaname=current_schema() AND tablename='skill_registry'",
        connection=conn,
    ).rows
    expected = normalize(OWNER_PREDICATE)
    if len(policies) != 1:
        raise SchemaError("Skill registry PostgreSQL policy inventory drifted.")
    row = policies[0]
    if (
        row["policyname"] != "skill_registry_tenant_isolation"
        or row["permissive"] != "PERMISSIVE"
        or row["roles"] != "{public}"
        or row["cmd"] != "ALL"
        or normalize(row["qual"]).replace("skill_registry.", "") != expected
        or normalize(row["with_check"]).replace("skill_registry.", "") != expected
    ):
        raise SchemaError("Skill registry PostgreSQL policy authority drifted.")
