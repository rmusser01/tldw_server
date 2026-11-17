"""Registered PostgreSQL dictionary catalog schema and tenant verification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DICTIONARY_TABLES = ("chat_dictionaries", "dictionary_entries", "dictionary_transform_activity", "dictionary_version_history")
_OWNER_DEFAULT = "NULLIF(current_setting('app.current_user_id', true), '')"


def dictionary_owner_predicates() -> dict[str, str]:
    """Bind definitions directly and every child through its owned definition."""
    return {
        "chat_dictionaries": f"client_id = {_OWNER_DEFAULT}",
        **{table: f"EXISTS (SELECT 1 FROM chat_dictionaries AS dictionary WHERE "  # nosec B608 - fixed table inventory and tenant expression
           f"dictionary.id = {table}.dictionary_id AND dictionary.client_id = {_OWNER_DEFAULT})"
           for table in DICTIONARY_TABLES[1:]},
    }


def postgres_dictionary_schema_statements() -> tuple[str, ...]:
    """Create known tables and upgrade legacy columns without assigning old rows."""
    return (
        """CREATE TABLE IF NOT EXISTS chat_dictionaries (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT,
                            category TEXT,
                            tags TEXT,
                            included_dictionary_ids TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            default_token_budget INTEGER NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            version INTEGER DEFAULT 1,
                            deleted BOOLEAN DEFAULT FALSE
                        )""",
        """CREATE TABLE IF NOT EXISTS dictionary_entries (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            key TEXT NOT NULL,
                            content TEXT NOT NULL,
                            is_regex BOOLEAN DEFAULT FALSE,
                            probability REAL DEFAULT 1.0,
                            max_replacements INTEGER DEFAULT 1,
                            group_name TEXT,
                            timed_effects TEXT DEFAULT '{}',
                            enabled BOOLEAN DEFAULT TRUE,
                            case_sensitive BOOLEAN DEFAULT TRUE,
                            sort_order INTEGER,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )""",
        """CREATE TABLE IF NOT EXISTS dictionary_transform_activity (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            chat_id TEXT NULL,
                            entries_used TEXT NOT NULL,
                            replacements INTEGER DEFAULT 0,
                            iterations INTEGER DEFAULT 0,
                            token_budget_used INTEGER NULL,
                            original_text_preview TEXT NOT NULL,
                            processed_text_preview TEXT NOT NULL,
                            processing_time_ms REAL NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )""",
        """CREATE TABLE IF NOT EXISTS dictionary_version_history (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            revision INTEGER NOT NULL,
                            source_dictionary_version INTEGER NULL,
                            change_type TEXT NOT NULL,
                            summary TEXT NULL,
                            entry_count INTEGER DEFAULT 0,
                            dictionary_snapshot TEXT NOT NULL,
                            entries_snapshot TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(dictionary_id, revision),
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )""",
        # ADD has no default: every existing row stays NULL/unowned.
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS client_id TEXT",
        "ALTER TABLE chat_dictionaries ALTER COLUMN client_id SET DEFAULT " + _OWNER_DEFAULT,
        "ALTER TABLE chat_dictionaries DROP CONSTRAINT IF EXISTS chat_dictionaries_name_key",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_dictionaries_owner_name ON chat_dictionaries(client_id,name)",
        "ALTER TABLE dictionary_entries ADD COLUMN IF NOT EXISTS sort_order INTEGER",
        "ALTER TABLE dictionary_entries ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0",
        "ALTER TABLE dictionary_entries ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMP",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS default_token_budget INTEGER",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS category TEXT",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS tags TEXT",
        "ALTER TABLE chat_dictionaries ADD COLUMN IF NOT EXISTS included_dictionary_ids TEXT",
        "CREATE INDEX IF NOT EXISTS idx_dict_entries_dict_id ON dictionary_entries(dictionary_id)",
        "CREATE INDEX IF NOT EXISTS idx_dict_entries_group ON dictionary_entries(group_name)",
        "CREATE INDEX IF NOT EXISTS idx_dict_entries_sort_order ON dictionary_entries(dictionary_id,sort_order)",
        "CREATE INDEX IF NOT EXISTS idx_dict_transform_activity_dictionary ON dictionary_transform_activity(dictionary_id,created_at)",
        "CREATE INDEX IF NOT EXISTS idx_dict_version_history_lookup ON dictionary_version_history(dictionary_id,revision DESC)",
    )


def verify_postgres_dictionary_schema(
    backend: Any, conn: Any, normalize: Callable[[Any], str],
) -> None:
    """Refuse missing/changed ownership authority instead of repairing on requests."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    backend.execute(
        "LOCK TABLE chat_dictionaries, dictionary_entries, dictionary_transform_activity, "
        "dictionary_version_history IN ACCESS SHARE MODE", connection=conn,
    )
    tables = backend.execute(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relkind IN ('r','p') AND c.relname = ANY(%s)",
        (list(DICTIONARY_TABLES),), connection=conn,
    ).rows
    if ({row["relname"] for row in tables} != set(DICTIONARY_TABLES)
            or any(not row["relrowsecurity"] or not row["relforcerowsecurity"] for row in tables)):
        raise SchemaError("Dictionary PostgreSQL tenant schema is missing or unsafe.")
    owner = backend.execute(
        "SELECT data_type,is_nullable,column_default FROM information_schema.columns "
        "WHERE table_schema=current_schema() AND table_name='chat_dictionaries' AND column_name='client_id'",
        connection=conn,
    ).rows
    if (len(owner) != 1 or owner[0]["data_type"] != "text" or owner[0]["is_nullable"] != "YES"
            or normalize(owner[0]["column_default"]) != normalize(_OWNER_DEFAULT)):
        raise SchemaError("Dictionary PostgreSQL owner column drifted.")
    indexes = backend.execute(
        "SELECT i.indisunique,i.indisvalid,i.indisready,i.indpred IS NULL AS complete, "
        "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum,ord) "
        "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum ORDER BY k.ord) AS columns "
        "FROM pg_index i WHERE i.indrelid='chat_dictionaries'::regclass AND i.indisunique",
        connection=conn,
    ).rows
    if (any(list(row["columns"]) == ["name"] for row in indexes)
            or not any(list(row["columns"]) == ["client_id", "name"] and row["indisvalid"]
                       and row["indisready"] and row["complete"] for row in indexes)):
        raise SchemaError("Dictionary PostgreSQL owner-name uniqueness drifted.")
    foreign = backend.execute(
        "SELECT c.relname, con.confrelid='chat_dictionaries'::regclass AS correct_parent, "
        "con.confdeltype,con.convalidated, "
        "ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum,ord) "
        "JOIN pg_attribute a ON a.attrelid=con.conrelid AND a.attnum=k.attnum ORDER BY k.ord) AS columns, "
        "ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum,ord) "
        "JOIN pg_attribute a ON a.attrelid=con.confrelid AND a.attnum=k.attnum ORDER BY k.ord) AS parent_columns "
        "FROM pg_constraint con JOIN pg_class c ON c.oid=con.conrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND con.contype='f' AND c.relname = ANY(%s)",
        (list(DICTIONARY_TABLES[1:]),), connection=conn,
    ).rows
    if any(not any(row["relname"] == table and row["correct_parent"] and row["confdeltype"] == "c"
                   and row["convalidated"] and list(row["columns"]) == ["dictionary_id"]
                   and list(row["parent_columns"]) == ["id"] for row in foreign) for table in DICTIONARY_TABLES[1:]):
        raise SchemaError("Dictionary PostgreSQL parent binding drifted.")
    policies = backend.execute(
        "SELECT tablename,policyname,permissive,roles::text,cmd,qual,with_check FROM pg_policies "
        "WHERE schemaname=current_schema() AND tablename = ANY(%s)",
        (list(DICTIONARY_TABLES),), connection=conn,
    ).rows
    expected = dictionary_owner_predicates()
    if len(policies) != len(expected) or {row["tablename"] for row in policies} != set(expected):
        raise SchemaError("Dictionary PostgreSQL policies drifted.")
    for row in policies:
        table = row["tablename"]
        expression = normalize(expected[table]).replace(table + ".", "")
        if (row["policyname"] != table + "_tenant_isolation" or row["permissive"] != "PERMISSIVE"
                or row["roles"] != "{public}" or row["cmd"] != "ALL"
                or normalize(row["qual"]).replace(table + ".", "") != expression
                or normalize(row["with_check"]).replace(table + ".", "") != expression):
            raise SchemaError("Dictionary PostgreSQL policy authority drifted.")
