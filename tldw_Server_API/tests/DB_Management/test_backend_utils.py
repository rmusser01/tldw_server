from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.backends import query_utils
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.query_utils import (
    convert_sqlite_placeholders_to_postgres,
    normalise_params,
    prepare_backend_many_statement,
    prepare_backend_statement,
    transform_sqlite_query_for_postgres,
)


def test_normalise_params_handles_sequences_and_scalars():

    assert normalise_params(None) is None
    assert normalise_params((1, 2)) == (1, 2)
    assert normalise_params([1, 2]) == (1, 2)
    assert normalise_params({"a": 1}) == {"a": 1}
    assert normalise_params(5) == (5,)


def test_repeated_sql_preparation_reuses_rewrite_work_but_keeps_fresh_parameters(monkeypatch):
    original = query_utils._replace_boolean_comparisons
    rewrites = []

    def observe(sql):
        rewrites.append(sql)
        return original(sql)

    monkeypatch.setattr(query_utils, '_replace_boolean_comparisons', observe)
    sql = 'SELECT uuid FROM Media WHERE deleted = 0 AND id = ? /* repeated email preparation */'
    for identity in range(20):
        prepared, params = prepare_backend_statement(
            BackendType.POSTGRESQL, sql, [identity], apply_default_transform=True,
        )
        assert prepared == 'SELECT uuid FROM Media WHERE deleted = FALSE AND id = %s /* repeated email preparation */'
        assert params == (identity,)
    assert len(rewrites) == 1


def test_convert_sqlite_placeholders_to_postgres_preserves_literals():

    query = "SELECT '?' as literal, col FROM table WHERE id = ?"
    converted = convert_sqlite_placeholders_to_postgres(query)
    assert converted == "SELECT '?' as literal, col FROM table WHERE id = %s"


def test_convert_sqlite_placeholders_to_postgres_preserves_jsonb_operators():

    query = "SELECT * FROM demo WHERE payload ? 'key' AND id = ?"
    converted = convert_sqlite_placeholders_to_postgres(query)
    assert "payload ? 'key'" in converted
    assert converted.count("%s") == 1

    query_param = "SELECT * FROM demo WHERE payload ? ? AND id = ?"
    converted_param = convert_sqlite_placeholders_to_postgres(query_param)
    assert "payload ? %s" in converted_param
    assert converted_param.count("%s") == 2


def test_transform_sqlite_query_for_postgres_rewrites_conflicts_and_collation():

    source = "INSERT OR IGNORE INTO demo(name) VALUES (?) COLLATE NOCASE"
    transformed = transform_sqlite_query_for_postgres(source)
    assert "INSERT OR IGNORE" not in transformed.upper()
    assert "COLLATE" not in transformed.upper()
    assert "ON CONFLICT DO NOTHING" in transformed.upper()


def test_prepare_backend_statement_noop_for_sqlite():

    query, params = prepare_backend_statement(BackendType.SQLITE, "SELECT 1", (1,))
    assert query == "SELECT 1"
    assert params == (1,)


def test_prepare_backend_statement_applies_placeholder_conversion_and_transform():

    query, params = prepare_backend_statement(
        BackendType.POSTGRESQL,
        "INSERT OR IGNORE INTO demo VALUES (?)",
        ["value"],
        apply_default_transform=True,
        ensure_returning=True,
    )
    assert query == "INSERT INTO demo VALUES (%s) ON CONFLICT DO NOTHING RETURNING id"
    assert params == ("value",)


def test_prepare_backend_many_statement_handles_lists():

    query, params = prepare_backend_many_statement(
        BackendType.POSTGRESQL,
        "UPDATE demo SET name = ? WHERE id = ?",
        [["alice", 1], ["bob", 2]],
    )
    assert query == "UPDATE demo SET name = %s WHERE id = %s"
    assert params == [("alice", 1), ("bob", 2)]


def test_media_db_runtime_prepare_backend_statement_forwards_expected_defaults(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        backend_prepare_ops as backend_prepare_ops_module,
    )

    calls: list[tuple[object, str, object, bool, bool]] = []

    def fake_prepare_backend_statement(
        backend_type,
        query,
        params,
        *,
        apply_default_transform,
        ensure_returning,
    ):
        calls.append(
            (
                backend_type,
                query,
                params,
                apply_default_transform,
                ensure_returning,
            )
        )
        return ("prepared", ("params",))

    monkeypatch.setattr(
        backend_prepare_ops_module,
        "prepare_backend_statement",
        fake_prepare_backend_statement,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)

    result = backend_prepare_ops_module._prepare_backend_statement(db, "SELECT ?", [1])

    assert calls == [(BackendType.POSTGRESQL, "SELECT ?", [1], True, False)]
    assert result == ("prepared", ("params",))


def test_media_db_runtime_prepare_backend_many_statement_forwards_expected_defaults(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        backend_prepare_ops as backend_prepare_ops_module,
    )

    calls: list[tuple[object, str, object, bool, bool]] = []

    def fake_prepare_backend_many_statement(
        backend_type,
        query,
        params_list,
        *,
        apply_default_transform,
        ensure_returning,
    ):
        calls.append(
            (
                backend_type,
                query,
                params_list,
                apply_default_transform,
                ensure_returning,
            )
        )
        return ("prepared-many", [("params",)])

    monkeypatch.setattr(
        backend_prepare_ops_module,
        "prepare_backend_many_statement",
        fake_prepare_backend_many_statement,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)

    result = backend_prepare_ops_module._prepare_backend_many_statement(
        db,
        "UPDATE demo SET value = ?",
        [[1]],
    )

    assert calls == [(BackendType.POSTGRESQL, "UPDATE demo SET value = ?", [[1]], True, False)]
    assert result == ("prepared-many", [("params",)])


def test_media_db_runtime_normalise_params_delegates_to_query_utils(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        backend_prepare_ops as backend_prepare_ops_module,
    )

    calls: list[object] = []

    def fake_normalise_params(params):
        calls.append(params)
        return ("normalised",)

    monkeypatch.setattr(
        backend_prepare_ops_module,
        "normalise_params",
        fake_normalise_params,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)

    result = backend_prepare_ops_module._normalise_params(db, [1])

    assert calls == [[1]]
    assert result == ("normalised",)


def test_media_db_runtime_keyword_order_expression_uses_sqlite_collation():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        query_utility_ops as query_utility_ops_module,
    )

    db = SimpleNamespace(backend_type=BackendType.SQLITE)

    result = query_utility_ops_module._keyword_order_expression(db, "keyword")

    assert result == "keyword COLLATE NOCASE"


def test_media_db_runtime_keyword_order_expression_uses_postgres_lower_sort():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        query_utility_ops as query_utility_ops_module,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)

    result = query_utility_ops_module._keyword_order_expression(db, "keyword")

    assert result == "LOWER(keyword), keyword"


def test_media_db_runtime_append_case_insensitive_like_uses_sqlite_clause():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        query_utility_ops as query_utility_ops_module,
    )

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    clauses: list[str] = []
    params: list[object] = []

    query_utility_ops_module._append_case_insensitive_like(db, clauses, params, "title", "%deep%")

    assert clauses == ["title LIKE ? COLLATE NOCASE"]
    assert params == ["%deep%"]


def test_media_db_runtime_append_case_insensitive_like_uses_postgres_ilike():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        query_utility_ops as query_utility_ops_module,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    clauses: list[str] = []
    params: list[object] = []

    query_utility_ops_module._append_case_insensitive_like(db, clauses, params, "title", "%deep%")

    assert clauses == ["title ILIKE ?"]
    assert params == ["%deep%"]


def test_media_db_runtime_convert_sqlite_placeholders_to_postgres_delegates_to_query_utils(monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        query_utility_ops as query_utility_ops_module,
    )

    calls: list[str] = []

    def fake_convert(query: str) -> str:
        calls.append(query)
        return "SELECT %s"

    monkeypatch.setattr(
        query_utility_ops_module,
        "convert_sqlite_placeholders_to_postgres",
        fake_convert,
    )

    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)

    result = query_utility_ops_module._convert_sqlite_placeholders_to_postgres(db, "SELECT ?")

    assert calls == ["SELECT ?"]
    assert result == "SELECT %s"


def test_transform_sqlite_query_for_postgres_rewrites_randomblob_and_json_extract():

    source = (
        "INSERT INTO prompt_studio_job_queue (uuid, payload) "
        "VALUES (lower(hex(randomblob(16))), json_extract(scores, '$.quality'))"
    )
    transformed = transform_sqlite_query_for_postgres(source, ensure_returning=True)
    assert "gen_random_bytes" in transformed
    assert "->> 'quality'" in transformed
    assert "RETURNING id" in transformed


def test_transform_sqlite_query_for_postgres_converts_boolean_columns():

    source = "SELECT * FROM table WHERE deleted = 0 AND is_active = 1 AND priority = 0"
    transformed = transform_sqlite_query_for_postgres(source)
    assert "deleted = FALSE" in transformed
    assert "is_active = TRUE" in transformed
    # Non-boolean column should remain unchanged
    assert "priority = 0" in transformed


def test_transform_sqlite_query_for_postgres_converts_boolean_literals_in_multi_row_insert():

    source = "INSERT INTO demo (is_active, name) VALUES (1, 'a'), (0, 'b')"
    transformed = transform_sqlite_query_for_postgres(source)
    assert "VALUES (TRUE, 'a'), (FALSE, 'b')" in transformed


def test_transform_sqlite_query_for_postgres_preserves_literals_and_comments():

    source = (
        "SELECT * FROM demo WHERE deleted = 0 "
        "AND note LIKE '%deleted = 0%' -- deleted = 0\n"
        "AND enabled = 1 /* deleted = 1 */"
    )
    transformed = transform_sqlite_query_for_postgres(source)
    assert "deleted = FALSE" in transformed
    assert "enabled = TRUE" in transformed
    assert "LIKE '%deleted = 0%'" in transformed
    assert "-- deleted = 0" in transformed
    assert "/* deleted = 1 */" in transformed
