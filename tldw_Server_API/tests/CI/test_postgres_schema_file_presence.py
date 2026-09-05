from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_postgres_users_schema_file_exists_with_core_columns() -> None:
    schema_path = Path("tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql")
    assert schema_path.exists()

    sql = schema_path.read_text(encoding="utf-8").lower()
    assert "create table if not exists public.users" in sql
    assert "create table if not exists public.organizations" in sql
    assert "create table if not exists public.teams" in sql
    for required in ("username", "email", "password_hash", "is_active", "is_verified", "role"):
        assert required in sql
