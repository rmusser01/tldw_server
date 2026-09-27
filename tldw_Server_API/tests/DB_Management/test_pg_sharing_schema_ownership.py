"""Architecture contract for PostgreSQL sharing schema ownership."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHNZ_MIGRATIONS = (
    REPO_ROOT / "tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py"
)
SHARING_SCHEMA = (
    REPO_ROOT
    / "tldw_Server_API/app/core/DB_Management/backends/pg_sharing_schema.py"
)


def test_postgres_sharing_ddl_is_owned_by_db_management() -> None:
    """AuthNZ orchestration must reference DB-owned sharing DDL."""
    assert SHARING_SCHEMA.exists(), "DB_Management sharing schema module is missing"

    authnz_source = AUTHNZ_MIGRATIONS.read_text(encoding="utf-8")
    schema_source = SHARING_SCHEMA.read_text(encoding="utf-8")
    for statement in (
        "CREATE TABLE IF NOT EXISTS shared_workspaces",
        "CREATE TABLE IF NOT EXISTS share_tokens",
        "CREATE TABLE IF NOT EXISTS share_audit_log",
        "CREATE TABLE IF NOT EXISTS sharing_config",
    ):
        assert statement not in authnz_source
        assert statement in schema_source
