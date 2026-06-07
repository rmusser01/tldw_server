import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


def test_list_generic_items_rejects_invalid_order_by(monkeypatch, tmp_path) -> None:
    # Avoid full schema setup; validation should run before any SQL execution.
    monkeypatch.setattr(CharactersRAGDB, "_initialize_schema", lambda self: None)

    db_path = tmp_path / "ChaChaNotes.db"
    db = CharactersRAGDB(db_path, client_id="test_client")
    try:
        with pytest.raises(InputError):
            db._list_generic_items("keywords", "name; DROP TABLE keywords; --", limit=10, offset=0)
    finally:
        db.close_all_connections()
