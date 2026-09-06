import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.query_utils import (
    convert_sqlite_placeholders_to_postgres,
    prepare_backend_many_statement,
    prepare_backend_statement,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CASE ? WHEN ? THEN ? ELSE ? END",
        "SELECT CASE WHEN ? THEN ? ELSE ? END",
        "SELECT CASE WHEN n > ? THEN n ELSE ? END FROM items",
        "SELECT CASE WHEN ? THEN CASE ? WHEN ? THEN ? ELSE ? END ELSE ? END",
        "SELECT case when ? then ? else ? end",
        "UPDATE state SET version = ?, seq = CASE WHEN seq > ? THEN seq ELSE ? END, updated = ? WHERE id = ? AND domain = ?",
    ],
)
def test_case_placeholders_preserve_parameter_order(sql: str) -> None:
    """CASE binds preserve their order on PostgreSQL and remain unchanged on SQLite."""
    params = tuple(range(sql.count("?")))
    converted, prepared = prepare_backend_statement(BackendType.POSTGRESQL, sql, params)
    assert converted == sql.replace("?", "%s")
    assert prepared == params
    assert prepare_backend_statement(BackendType.SQLITE, sql, params) == (sql, params)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT CASE WHEN payload ? 'key' THEN ? ELSE ? END", "SELECT CASE WHEN payload ? 'key' THEN %s ELSE %s END"),
        ("SELECT CASE WHEN payload ? ? THEN ? ELSE ? END", "SELECT CASE WHEN payload ? %s THEN %s ELSE %s END"),
        (
            "SELECT CASE WHEN flag THEN payload ELSE payload END ? 'key'",
            "SELECT CASE WHEN flag THEN payload ELSE payload END ? 'key'",
        ),
        (
            "SELECT payload ? CASE WHEN flag THEN 'a' ELSE 'b' END",
            "SELECT payload ? CASE WHEN flag THEN 'a' ELSE 'b' END",
        ),
        (
            "SELECT CASE WHEN \"END\" ? 'key' THEN '?' ELSE ? END",
            "SELECT CASE WHEN \"END\" ? 'key' THEN '?' ELSE %s END",
        ),
        (
            "SELECT CASE WHEN payload ?| array['a'] THEN ? ELSE ? END",
            "SELECT CASE WHEN payload ?| array['a'] THEN %s ELSE %s END",
        ),
        (
            "SELECT CASE WHEN payload ?& array['a'] THEN ? ELSE ? END",
            "SELECT CASE WHEN payload ?& array['a'] THEN %s ELSE %s END",
        ),
    ],
)
def test_case_preserves_jsonb_operators_and_quoted_text(sql: str, expected: str) -> None:
    """CASE conversion leaves JSONB operators and quoted question marks intact."""
    assert convert_sqlite_placeholders_to_postgres(sql) == expected


def test_convert_placeholders_ignores_single_quoted_literals():

    sql = "SELECT '? literal ?' as txt, id FROM table WHERE id = ? AND note = '?keep?'"
    converted = convert_sqlite_placeholders_to_postgres(sql)
    # Only the WHERE id = ? should be converted
    assert "'? literal ?'" in converted
    assert "'?keep?'" in converted
    assert converted.count("%s") == 1


def test_convert_placeholders_ignores_double_quoted_identifiers_or_literals():

    sql = 'SELECT id, "weird?col" FROM "my?table" WHERE id = ?'
    converted = convert_sqlite_placeholders_to_postgres(sql)
    assert '"weird?col"' in converted
    assert '"my?table"' in converted
    assert converted.endswith("%s") or "%s" in converted


def test_prepare_backend_statement_positional_params():

    sql = "UPDATE users SET name = ? WHERE id = ?"
    params = ("Alice", 7)
    converted, prepared = prepare_backend_statement(BackendType.POSTGRESQL, sql, params)
    assert converted == "UPDATE users SET name = %s WHERE id = %s"
    assert prepared == params


def test_prepare_backend_many_statement_batch_params():

    sql = "INSERT INTO items (sku, qty) VALUES (?, ?)"
    params_list = [("A", 1), ("B", 2)]
    converted, prepared_list = prepare_backend_many_statement(BackendType.POSTGRESQL, sql, params_list)
    assert converted == "INSERT INTO items (sku, qty) VALUES (%s, %s)"
    assert prepared_list == params_list


@pytest.mark.skipif(
    pytest.importorskip(
        "tldw_Server_API.app.core.DB_Management.backends.postgresql_backend",
        reason="psycopg not available",
    )
    is None,
    reason="psycopg not available",
)
def test_postgres_backend_prepare_query_no_replace_inside_literals():
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import (
        PostgreSQLBackend,
    )

    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    sql = "SELECT '? literal ?' as txt, id FROM table WHERE id = ? AND note = 'x?y'"
    converted, params = backend._prepare_query(sql, (42,))
    assert converted.count("%s") == 1
    assert "'? literal ?'" in converted and "'x?y'" in converted
    assert params == (42,)
