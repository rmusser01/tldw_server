"""PostgreSQL parity and forced-RLS contracts for the recipient chat store."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)

NOW = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)


def _thread(db: CharactersRAGDB, share_id: int = 51):
    return db.shared_workspace_chat_store.get_or_create_thread(
        share_id=share_id,
        owner_user_id="owner-a",
        workspace_id="workspace-a",
        workspace_name="PostgreSQL workspace",
    )


def _claim(db: CharactersRAGDB, conversation_id: str, share_id: int = 51):
    return db.shared_workspace_chat_store.claim_request(
        share_id=share_id,
        request_id=uuid4(),
        request_fingerprint=f"fingerprint-{uuid4()}",
        conversation_id=conversation_id,
        lease_seconds=600,
        now=NOW,
    )


def _close(db: CharactersRAGDB) -> None:
    db.close_all_connections()


def _reset_restricted_session(db: CharactersRAGDB) -> None:
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("RESET ROLE")
    cursor.execute("RESET row_security")
    cursor.execute("RESET app.current_user_id")
    conn.commit()


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_postgres_store_round_trips_thread_claim_freeze_complete_and_history(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    try:
        thread = _thread(db)
        claim = _claim(db, thread.conversation_id)
        assert db.shared_workspace_chat_store.freeze_sources(
            claim=claim,
            source_mode="include",
            source_ids=("source-a",),
            snapshot_hash="snapshot-a",
            provider="llama",
            model="model-a",
        )
        stored = db.shared_workspace_chat_store.complete_turn(
            claim=claim,
            query="Question",
            answer="Answer",
            citations=[
                {
                    "citation_id": "citation-a",
                    "source_id": "source-a",
                    "source_title": "Source A",
                    "locator": {"chunk": 1},
                    "quote": "Evidence",
                    "score": 0.9,
                }
            ],
            provider="llama",
            model="model-a",
            source_mode="include",
            effective_source_count=1,
        )

        assert db.shared_workspace_chat_store.load_completed_turn(
            share_id=51, request_id=claim.request_id
        ) == stored
        page = db.shared_workspace_chat_store.list_messages(
            share_id=51, before=None, limit=30
        )
        assert [message.content for message in page.messages] == ["Question", "Answer"]
        assert page.messages[-1].citations == stored.citations
    finally:
        _close(db)
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_postgres_store_blocks_cross_recipient_select_claim_update_and_delete(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    db_b = CharactersRAGDB(":memory:", client_id="recipient-b", backend=backend)
    role_name = f"shared_chat_store_{uuid4().hex[:12]}"
    ident = backend.escape_identifier
    role_created = False
    try:
        bypasses_rls = bool(
            backend.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).scalar
        )
        assert bypasses_rls, "The PostgreSQL store isolation test requires fixture admin seeding"

        thread_a = _thread(db_a, share_id=61)
        with backend.transaction() as conn:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}", connection=conn)
            backend.execute(
                f"GRANT SELECT ON conversations, messages TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON "
                "shared_workspace_chat_threads, shared_workspace_chat_requests "
                f"TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        for database, recipient in ((db_a, "recipient-a"), (db_b, "recipient-b")):
            database.close_connection()
            conn = database.get_connection()
            cursor = conn.cursor()
            cursor.execute(f"SET ROLE {ident(role_name)}")
            cursor.execute("SET row_security = on")
            cursor.execute(
                "SELECT set_config('app.current_user_id', %s, false)",
                (recipient,),
            )
            conn.commit()

        assert db_a.shared_workspace_chat_store.get_thread(share_id=61) is not None
        assert db_b.shared_workspace_chat_store.get_thread(share_id=61) is None

        claim_a = _claim(db_a, thread_a.conversation_id, share_id=61)
        with pytest.raises(CharactersRAGDBError):
            db_b.shared_workspace_chat_store.claim_request(
                share_id=61,
                request_id=uuid4(),
                request_fingerprint="recipient-b-forged",
                conversation_id=thread_a.conversation_id,
                lease_seconds=600,
                now=NOW,
            )

        forged = replace(claim_a, lease_token="recipient-b-forged")
        assert not db_b.shared_workspace_chat_store.freeze_sources(
            claim=forged,
            source_mode="include",
            source_ids=("source-a",),
            snapshot_hash="snapshot-forged",
            provider="llama",
            model="model-forged",
        )
        assert not db_b.shared_workspace_chat_store.mark_retryable(
            claim=forged, error_code="forged"
        )

        with db_b.transaction() as conn:
            hidden_threads = conn.execute(
                "SELECT share_id FROM shared_workspace_chat_threads "
                "WHERE share_id = ? AND conversation_id = ?",
                (61, thread_a.conversation_id),
            ).fetchall()
            hidden_receipts = conn.execute(
                "SELECT request_id FROM shared_workspace_chat_requests "
                "WHERE share_id = ? AND request_id = ?",
                (61, str(claim_a.request_id)),
            ).fetchall()
            direct_update = conn.execute(
                "UPDATE shared_workspace_chat_requests SET error_code = ? "
                "WHERE share_id = ? AND request_id = ?",
                ("recipient-b-direct-update", 61, str(claim_a.request_id)),
            )
            direct_delete = conn.execute(
                "DELETE FROM shared_workspace_chat_requests "
                "WHERE share_id = ? AND request_id = ?",
                (61, str(claim_a.request_id)),
            )
        assert hidden_threads == []
        assert hidden_receipts == []
        assert direct_update.rowcount == 0
        assert direct_delete.rowcount == 0
        assert db_a.execute_query(
            "SELECT error_code FROM shared_workspace_chat_requests WHERE request_id = ?",
            (str(claim_a.request_id),),
        ).fetchone()[0] is None

        with db_a.transaction() as conn:
            conn.execute(
                "UPDATE shared_workspace_chat_requests SET status = 'conflicted', "
                "updated_at = ? WHERE request_id = ?",
                ("2026-08-19T00:00:00+00:00", str(claim_a.request_id)),
            )
        assert db_b.shared_workspace_chat_store.purge_expired_conflicts(now=NOW) == 0
        assert db_a.execute_query(
            "SELECT count(*) FROM shared_workspace_chat_requests WHERE request_id = ?",
            (str(claim_a.request_id),),
        ).fetchone()[0] == 1
    finally:
        try:
            try:
                _reset_restricted_session(db_a)
            finally:
                try:
                    _reset_restricted_session(db_b)
                finally:
                    if role_created:
                        with backend.transaction() as conn:
                            backend.execute(
                                f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                                connection=conn,
                            )
                            backend.execute(
                                f"DROP OWNED BY {ident(role_name)}", connection=conn
                            )
                            backend.execute(
                                f"DROP ROLE {ident(role_name)}", connection=conn
                            )
        finally:
            try:
                db_a.close_connection()
            finally:
                try:
                    db_b.close_connection()
                finally:
                    backend.get_pool().close_all()
