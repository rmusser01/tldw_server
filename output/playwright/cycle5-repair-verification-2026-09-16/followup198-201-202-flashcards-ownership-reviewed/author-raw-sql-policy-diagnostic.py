"""Retained failing diagnostic; raw SQL isolation is outside approved application repair.

Executed before repair via the original permanent-test path/command retained in the
packet. Requires the official DB_Management pg_database_config fixture.
"""
import json
from pathlib import Path
from uuid import uuid4
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

def test_postgres_restricted_role_private_deck_card_policy(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "rls.db", client_id="2", backend=backend)
    role = f"flashcard_owner_{uuid4().hex[:12]}"
    quoted = backend.escape_identifier(role)
    created_role = False
    receipt = {}
    try:
        deck = db.add_deck("Role-private deck")
        card = db.add_flashcard({"deck_id": deck, "front": "Role-private card", "back": "Owned"})
        db.close_connection()
        receipt["tables"] = backend.execute("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid IN ('decks'::regclass, 'flashcards'::regclass) ORDER BY relname").rows
        receipt["policies"] = backend.execute("SELECT tablename, policyname FROM pg_policies WHERE schemaname=current_schema() AND tablename IN ('decks','flashcards') ORDER BY tablename,policyname").rows
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {quoted} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}", connection=conn)
            backend.execute(f"GRANT SELECT ON decks, flashcards TO {quoted}", connection=conn)
            backend.execute(f"GRANT {quoted} TO CURRENT_USER", connection=conn)
        created_role = True
        with backend.transaction() as conn:
            backend.execute(f"SET LOCAL ROLE {quoted}", connection=conn)
            backend.execute("SELECT set_config('app.current_user_id', %s, true)", ("3",), connection=conn)
            receipt["role_flags"] = backend.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user", connection=conn).rows[0]
            receipt["foreign_decks"] = backend.execute("SELECT id, client_id FROM decks WHERE id=%s", (deck,), connection=conn).rows
            receipt["foreign_cards"] = backend.execute("SELECT uuid, client_id FROM flashcards WHERE uuid=%s", (card,), connection=conn).rows
            backend.execute("SELECT set_config('app.current_user_id', %s, true)", ("2",), connection=conn)
            receipt["owned_cards"] = backend.execute("SELECT uuid, client_id FROM flashcards WHERE uuid=%s", (card,), connection=conn).rows
        output = Path(".tmp/uat198-diagnosis-20260917/restricted-role-catalog.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        assert receipt["role_flags"] == {"rolsuper": False, "rolbypassrls": False}
        assert len(receipt["owned_cards"]) == 1
        assert receipt["foreign_decks"] == [] and receipt["foreign_cards"] == []
    finally:
        db.close_connection()
        if created_role:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {quoted}", connection=conn)
                backend.execute(f"DROP ROLE {quoted}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()
