"""Read-only diagnostic of the observed production transaction logging branch."""
import pytest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

@pytest.mark.parametrize('message', ['ordinary synthetic failure', 'synthetic {missing_key} failure'])
def test_original_error_and_rollback_survive_diagnostic_text(tmp_path, message):
    db = CharactersRAGDB(tmp_path / 'probe.db', client_id='probe-device')
    try:
        caught = None
        try:
            with db.transaction():
                db.add_keyword('must rollback')
                raise ValueError(message)
        except (ValueError, KeyError) as error:
            caught = error
        connection = db.get_connection()
        observed = (type(caught).__name__, connection.in_transaction, db.get_keyword_by_text('must rollback') is not None)
        if connection.in_transaction:
            connection.rollback()
        assert observed == ('ValueError', False, False)
    finally:
        db.close_all_connections()
