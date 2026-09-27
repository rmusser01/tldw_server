import re
from pathlib import Path


def test_postgres_users_schema_file_exists_with_core_columns() -> None:
    schema_path = Path("tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql")
    assert schema_path.exists()

    sql = schema_path.read_text(encoding="utf-8").lower()
    # Tables are schema-qualified (public.) since 5f31630280; accept either spelling.
    for table in ("users", "organizations", "teams"):
        assert re.search(rf"create table if not exists (public\.)?{table} \(", sql), table
    for required in ("username", "email", "password_hash", "is_active", "is_verified", "role"):
        assert required in sql
