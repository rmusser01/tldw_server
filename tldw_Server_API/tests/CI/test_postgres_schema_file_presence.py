from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.unit


def test_postgres_users_schema_file_exists_with_core_columns() -> None:
    schema_path = Path("tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql")
    assert schema_path.exists()

    sql = schema_path.read_text(encoding="utf-8").lower()
    for table in ("users", "organizations", "teams"):
        assert re.search(rf"create\s+table\s+if\s+not\s+exists\s+(?:public\.)?{table}\s*\(", sql)
    for required in ("username", "email", "password_hash", "is_active", "is_verified", "role"):
        assert required in sql
