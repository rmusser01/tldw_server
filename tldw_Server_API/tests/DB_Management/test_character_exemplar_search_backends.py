"""Real exemplar search keeps optional filters typed and search results intact."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def exemplar_db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "exemplars.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("filters", [{}, {"emotion": "other"}, {"scenario": "other"}, {"emotion": "other", "scenario": "other"}, {"emotion": "", "scenario": ""}], ids=["absent", "emotion-only", "scenario-only", "both", "empty"])
@pytest.mark.parametrize("query", [None, "Citrine"], ids=["browse", "text-search"])
def test_owned_search_optional_filters_return_matching_exemplar(exemplar_db, filters, query):
    character = exemplar_db.add_character_card({"name": "Search owner"})
    item = exemplar_db.add_character_exemplar(character, {"text": "Citrine archive", "emotion": "other", "scenario": "other"})
    rows, total = exemplar_db.search_character_exemplars(character, query=query, **filters)
    assert total == 1
    assert [row["id"] for row in rows] == [item["id"]]


def test_search_filters_deleted_rows_and_preserves_total_before_pagination(exemplar_db):
    character = exemplar_db.add_character_card({"name": "Paged search owner"})
    one = exemplar_db.add_character_exemplar(character, {"text": "Citrine first", "rhetorical": ["metaphor"]})
    two = exemplar_db.add_character_exemplar(character, {"text": "Citrine second", "rhetorical": ["metaphor"]})
    removed = exemplar_db.add_character_exemplar(character, {"text": "Citrine removed", "rhetorical": ["metaphor"]})
    assert exemplar_db.soft_delete_character_exemplar(character, removed["id"])
    rows, total = exemplar_db.search_character_exemplars(character, query="Citrine", rhetorical=["metaphor"], limit=1, offset=1)
    assert total == 2
    assert len(rows) == 1 and rows[0]["id"] in {one["id"], two["id"]}
    assert exemplar_db.search_character_exemplars(character, query="NoMatchingWord") == ([], 0)
