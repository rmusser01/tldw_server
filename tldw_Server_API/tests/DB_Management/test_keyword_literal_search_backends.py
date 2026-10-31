"""Literal keyword searches bind their limit and match punctuation on both backends."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def keyword_db(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "keywords.db", client_id="3", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize(
    "literal,decoy",
    [
        ("C++", "C--"),
        ("foo-bar", "foo_bar"),
        ("100%", "1000"),
        ("under_score", "underXscore"),
        ("bang!mark", "bangmark"),
        (r"path\leaf", "pathleaf"),
        ("%_!\\", "other"),
    ],
)
def test_literal_keyword_search_matches_only_bound_literal(keyword_db, literal, decoy):
    match = keyword_db.add_keyword(literal)
    keyword_db.add_keyword(decoy)
    assert [row["id"] for row in keyword_db.search_keywords(literal, limit=1)] == [match]


@pytest.mark.parametrize("invalid", ["", '"', "'"])
def test_literal_search_preserves_input_rejection(keyword_db, invalid):
    with pytest.raises(InputError):
        keyword_db.search_keywords(invalid)


def test_literal_search_orders_before_applying_bound_limit(keyword_db):
    first = keyword_db.add_keyword(r"path\!leaf-first")
    last = keyword_db.add_keyword(r"path\!leaf-last")
    with keyword_db.transaction() as conn:
        table = keyword_db._map_table_for_backend("keywords")
        conn.execute(f"UPDATE {table} SET last_modified=? WHERE id=?", ("2026-09-16T01:00:00+00:00", first))  # nosec B608 - fixed backend table mapping; values stay bound.
        conn.execute(f"UPDATE {table} SET last_modified=? WHERE id=?", ("2026-09-16T02:00:00+00:00", last))  # nosec B608 - fixed backend table mapping; values stay bound.
    assert [row["id"] for row in keyword_db.search_keywords(r"path\!leaf", limit=2)] == [last, first]
    assert [row["id"] for row in keyword_db.search_keywords(r"path\!leaf", limit=1)] == [last]
