"""Reconcile the relevant released catalog shapes without reconstructing history."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("catalog", ["pr3071", "incoming"])
def test_published_catalog_shapes_reconcile_without_changing_existing_messages(db_factory, monkeypatch, catalog):
    # The fixture supplies the real isolated backend; remove only the features
    # that the other released migration catalog had not installed.
    with monkeypatch.context() as previous:
        previous.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 75 if catalog == "pr3071" else 76)
        previous.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 79 if catalog == "pr3071" else 80)
        old = db_factory()
    postgres = old.backend_type.value == "postgresql"
    conversation = old.add_conversation({"title": "Preserved catalog"})
    message = old.add_message({"conversation_id": conversation, "sender": "user", "content": "Preserved question"})
    before = old.get_message_by_id(message)
    with old.transaction() as connection:
        if catalog == "pr3071":
            old.backend.execute("ALTER TABLE conversations DROP COLUMN create_request_fingerprint", connection=connection)
        else:
            statements = (
                "DROP TRIGGER IF EXISTS messages_record_insertion_order ON messages",
                "DROP TRIGGER IF EXISTS messages_lock_insertion_order ON messages",
            ) if postgres else (
                "DROP TRIGGER IF EXISTS messages_record_insertion_order",
                "DROP TRIGGER IF EXISTS messages_lock_insertion_order",
            )
            for sql in statements:
                old.backend.execute(sql, connection=connection)
            old.backend.execute("DROP TABLE message_insertion_order", connection=connection)
            if postgres:
                for sql in (
                    "DROP FUNCTION IF EXISTS messages_record_insertion_order()",
                    "DROP FUNCTION IF EXISTS messages_lock_insertion_order()",
                    "DROP FUNCTION IF EXISTS message_insertion_order_immutable()",
                ):
                    old.backend.execute(sql, connection=connection)
    old.close_all_connections()

    upgraded = db_factory()
    with upgraded.transaction() as connection:
        columns = {row["name"] for row in upgraded.backend.get_table_info("conversations", connection=connection)}
        assert "create_request_fingerprint" in columns
        assert upgraded.backend.table_exists("workspace_chat_startup_receipts", connection=connection)
        assert upgraded.backend.table_exists("message_insertion_order", connection=connection)
        count = upgraded.backend.execute("SELECT COUNT(*) AS total FROM message_insertion_order", connection=connection).rows[0]["total"]
        assert count == (1 if catalog == "pr3071" else 0)
    assert upgraded.get_message_by_id(message) == before
    added = upgraded.add_message({"conversation_id": conversation, "sender": "assistant", "content": "New answer"})
    with upgraded.transaction() as connection:
        rows = upgraded.backend.execute("SELECT message_id FROM message_insertion_order ORDER BY sequence", connection=connection).rows
        assert [row["message_id"] for row in rows] == ([message, added] if catalog == "pr3071" else [added])
    upgraded.close_all_connections()
    reopened = db_factory()
    assert reopened.get_message_by_id(message) == before
    assert reopened.get_message_by_id(added)["content"] == "New answer"
