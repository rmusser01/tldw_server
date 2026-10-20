"""Locking reads inside a ChaCha operation see the operation's transaction (TASK-13861)."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_db(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "lock.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()


def test_a_workspace_locking_read_inside_an_operation_sees_its_transaction(pg_db):
    pg_db.upsert_workspace("ws-lock", "Locked workspace")

    with chacha_operation(independent=True):  # every production HTTP request runs inside one
        with pg_db.transaction() as conn:
            workspace = pg_db.get_workspace("ws-lock", conn=conn, for_update=True)

    assert workspace["id"] == "ws-lock"


def test_a_persona_locking_read_inside_an_operation_sees_its_transaction(pg_db):
    persona_id = pg_db.create_persona_profile({"user_id": "1", "name": "Locked persona"})

    with chacha_operation(independent=True):
        with pg_db.transaction() as conn:
            persona = pg_db.get_persona_profile(persona_id, user_id="1", conn=conn, for_update=True)

    assert persona["id"] == persona_id
