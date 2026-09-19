"""Real exemplar search keeps optional filters typed and search results intact."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)

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


@pytest.mark.parametrize(
    "query",
    [
        "media: Rowan Observatory",
        "Booking starts at 18:00.",
        'Mira said "Cedar Ridge".',
        "Mira's archive (ORBIT-742)",
        "Citrine & cedar | quartz ! amber",
    ],
)
def test_exemplar_search_treats_source_prose_as_literal_text(exemplar_db, query):
    character = exemplar_db.add_character_card({"name": "Prose owner"})
    matching = exemplar_db.add_character_exemplar(character, {"text": query})
    exemplar_db.add_character_exemplar(character, {"text": "Unrelated ocean diary"})

    rows, total = exemplar_db.search_character_exemplars(character, query=query)

    assert total == 1
    assert [row["id"] for row in rows] == [matching["id"]]
    assert exemplar_db.get_character_card_by_id(character)["name"] == "Prose owner"


@pytest.mark.parametrize("query", ["::: !!! ()", "the and of"])
def test_exemplar_search_without_searchable_words_keeps_connection_usable(
    exemplar_db, query
):
    character = exemplar_db.add_character_card({"name": "Empty query owner"})
    exemplar_db.add_character_exemplar(character, {"text": "Citrine archive"})

    assert exemplar_db.search_character_exemplars(character, query=query) == ([], 0)
    assert exemplar_db.get_character_card_by_id(character)["name"] == "Empty query owner"


@pytest.mark.parametrize("pending_write", [False, True], ids=["idle", "pending-write"])
def test_failed_exemplar_read_preserves_connection_and_caller_transaction(
    exemplar_db, monkeypatch, pending_write
):
    character = exemplar_db.add_character_card({"name": "Recovery owner"})
    item = exemplar_db.add_character_exemplar(character, {"text": "Citrine archive"})
    connection = exemplar_db.get_connection()
    connection.commit()
    if pending_write:
        # Deliberately leave caller-owned work uncommitted around the lookup.
        # SQLite connections are otherwise in autocommit mode.
        connection.execute("BEGIN")
        exemplar_db.execute_query(
            "UPDATE character_cards SET description = ? WHERE id = ?",
            ("pending caller write", character),
        )

    original = exemplar_db._prepare_backend_statement
    fail_next_search = True

    def fail_search_once(sql, params):
        nonlocal fail_next_search
        if fail_next_search and "character_exemplars" in sql:
            fail_next_search = False
            return original("SELECT missing_uat296_lookup_column", ())
        return original(sql, params)

    monkeypatch.setattr(exemplar_db, "_prepare_backend_statement", fail_search_once)
    with pytest.raises(CharactersRAGDBError):
        exemplar_db.search_character_exemplars(character, query="Citrine")

    assert [row["id"] for row in exemplar_db.list_character_exemplars(character)] == [
        item["id"]
    ]
    if pending_write:
        assert exemplar_db.get_character_card_by_id(character)["description"] == (
            "pending caller write"
        )
    connection.rollback()
    assert exemplar_db.get_character_card_by_id(character)["description"] != (
        "pending caller write"
    )


@pytest.mark.parametrize("failed_lookup", [False, True], ids=["success", "failure"])
def test_exemplar_lookup_preserves_owned_outer_rollback(
    exemplar_db, monkeypatch, failed_lookup
):
    character = exemplar_db.add_character_card({"name": "Managed transaction owner"})
    exemplar_db.add_character_exemplar(character, {"text": "Citrine archive"})
    exemplar_db.close_connection()

    class CallerRollback(Exception):
        pass

    with pytest.raises(CallerRollback):
        with chacha_operation(independent=True), exemplar_db.transaction():
            # A managed transaction may not yet have sent BEGIN to PostgreSQL.
            if failed_lookup:
                original = exemplar_db._prepare_backend_statement

                def fail_search(sql, params):
                    if "character_exemplars" in sql:
                        return original("SELECT missing_uat296_lookup_column", ())
                    return original(sql, params)

                with monkeypatch.context() as patch:
                    patch.setattr(exemplar_db, "_prepare_backend_statement", fail_search)
                    with pytest.raises(CharactersRAGDBError):
                        exemplar_db.search_character_exemplars(character, query="Citrine")
            else:
                assert exemplar_db.search_character_exemplars(character, query="Citrine")[1] == 1

            connection = exemplar_db.get_connection()
            if exemplar_db.backend_type.value == "postgresql":
                assert connection._connection.info.transaction_status.name == "INTRANS"
            else:
                assert connection.in_transaction
            exemplar_db.execute_query(
                "UPDATE character_cards SET description = ? WHERE id = ?",
                ("uncommitted managed write", character),
            )
            assert exemplar_db.search_character_exemplars(character, query="Citrine")[1] == 1
            assert exemplar_db.get_character_card_by_id(character)["description"] == (
                "uncommitted managed write"
            )
            raise CallerRollback

    with chacha_operation(independent=True):
        assert exemplar_db.get_character_card_by_id(character)["description"] != (
            "uncommitted managed write"
        )
