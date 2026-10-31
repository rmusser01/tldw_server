"""Native graph writes preserve replacement behavior with bounded round trips."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context


def _media(db: MediaDatabase) -> int:
    media_id, _, _ = db.add_media_with_keywords(
        url="email://synthetic-roundtrips", title="Synthetic subject", media_type="email",
        content="Synthetic body", keywords=[],
    )
    return int(media_id)


def _graph(db: MediaDatabase, media_id: int, *, updated: bool = False) -> dict:
    return db.upsert_email_message_graph(
        media_id=media_id, tenant_id="42", source_key="synthetic-roundtrips",
        source_message_id="provider-synthetic-1", body_text="Synthetic body",
        metadata={"email": {
            "message_id": "<synthetic-roundtrips@example.test>",
            "from": "sender@example.test" if updated else "Sender <sender@example.test>",
            "to": "replacement@example.test" if updated else "recipient@example.test",
            "labels": ["Updated" if updated else "Inbox"],
            "attachments": [] if updated else [{"filename": "synthetic.txt", "size_bytes": 5}],
        }},
    )


@pytest.mark.unit
def test_new_graph_does_not_delete_empty_relations(monkeypatch):
    db = MediaDatabase(":memory:", client_id="42")
    try:
        media_id = _media(db)
        queries = []
        execute = db._execute_with_connection

        def observe(conn, query, params=None):
            queries.append(query)
            return execute(conn, query, params)

        monkeypatch.setattr(db, "_execute_with_connection", observe)
        result = _graph(db, media_id)
        assert result["match_strategy"] == "new"
        assert not any(query.lstrip().upper().startswith("DELETE") for query in queries)
    finally:
        db.close_connection()


@pytest.mark.integration
def test_postgres_graph_roundtrips_preserve_ids_and_replace_relations(pg_database_config, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(":memory:", client_id="42", backend=backend)
    try:
        with scoped_context(user_id=42):
            media_id = _media(db)
            queries = []
            execute = db._execute_with_connection

            def observe(conn, query, params=None):
                queries.append(query)
                return execute(conn, query, params)

            monkeypatch.setattr(db, "_execute_with_connection", observe)
            first = _graph(db, media_id)
            assert first["match_strategy"] == "new"
            # Includes one attachment, two participants, and one label.
            assert len(queries) <= 13
            message_id = first["email_message_id"]
            for table, expected in (
                ("email_message_participants", 2), ("email_message_labels", 1), ("email_attachments", 1),
            ):
                assert db.execute_query(
                    f"SELECT COUNT(*) FROM {table} WHERE email_message_id = ?", (message_id,),
                ).fetchone()[0] == expected
            queries.clear()
            second = _graph(db, media_id, updated=True)
            assert second["email_message_id"] == first["email_message_id"]
            assert second["source_id"] == first["source_id"]
            assert second["match_strategy"] == "source_message_id"
            assert sum(query.lstrip().upper().startswith("DELETE") for query in queries) == 3
            participants = db.execute_query(
                "SELECT p.email_normalized, p.display_name FROM email_message_participants mp "
                "JOIN email_participants p ON p.id = mp.participant_id "
                "WHERE mp.email_message_id = ? ORDER BY mp.role", (message_id,),
            ).fetchall()
            assert [(row["email_normalized"], row["display_name"]) for row in participants] == [
                ("sender@example.test", "Sender"), ("replacement@example.test", None),
            ]
            assert db.execute_query(
                "SELECT l.label_name FROM email_message_labels ml "
                "JOIN email_labels l ON l.id = ml.label_id WHERE ml.email_message_id = ?",
                (message_id,),
            ).fetchone()["label_name"] == "Updated"
            assert db.execute_query(
                "SELECT COUNT(*) FROM email_attachments WHERE email_message_id = ?", (message_id,),
            ).fetchone()[0] == 0
    finally:
        db.close_connection()
        backend.get_pool().close_all()
