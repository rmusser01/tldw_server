from __future__ import annotations

import os
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

_LIVE_POSTGRES_DSN = os.getenv("TEST_DATABASE_URL") or os.getenv("POSTGRES_TEST_DSN")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not str(_LIVE_POSTGRES_DSN or "").lower().startswith("postgres"),
        reason="A PostgreSQL TEST_DATABASE_URL or POSTGRES_TEST_DSN is required",
    ),
]


def _create_owner_clip(
    db: CharactersRAGDB,
    *,
    clip_id: str,
    note_id: str,
    workspace_id: str,
) -> None:
    assert db.add_note(title=f"Clip for {db.client_id}", content="Body", note_id=note_id)
    db.upsert_workspace(workspace_id, f"Workspace for {db.client_id}")
    db.upsert_note_clipper_document(
        clip_id=clip_id,
        note_id=note_id,
        clip_type="article",
        source_url="https://example.com/shared",
        source_title=f"Title for {db.client_id}",
        capture_metadata={"captured_at": "2026-08-10T00:00:00+00:00"},
        enrichments={},
        content_budget={},
        source_note_version=1,
    )
    db.upsert_note_clipper_workspace_placement(
        clip_id=clip_id,
        workspace_id=workspace_id,
        source_note_id=note_id,
        source_note_version=1,
    )


def test_postgres_web_clipper_same_clip_is_owner_isolated_by_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "910001"
    owner_b = "910002"
    clip_id = "shared-public-clip"
    note_a = str(uuid4())
    note_b = str(uuid4())
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(db_path=":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(db_path=":memory:", client_id=owner_b, backend=backend_b)

    try:
        _create_owner_clip(
            db_a,
            clip_id=clip_id,
            note_id=note_a,
            workspace_id="workspace-a",
        )
        _create_owner_clip(
            db_b,
            clip_id=clip_id,
            note_id=note_b,
            workspace_id="workspace-b",
        )

        document_a = db_a.get_note_clipper_document_by_clip_id(clip_id)
        document_b = db_b.get_note_clipper_document_by_clip_id(clip_id)
        assert document_a is not None and document_a["note_id"] == note_a
        assert document_b is not None and document_b["note_id"] == note_b
        assert db_a.get_note_clipper_document_by_note_id(note_b) is None
        assert db_b.get_note_clipper_document_by_note_id(note_a) is None
        assert [row["workspace_id"] for row in db_a.list_note_clipper_workspace_placements(clip_id)] == ["workspace-a"]
        assert [row["workspace_id"] for row in db_b.list_note_clipper_workspace_placements(clip_id)] == ["workspace-b"]

        with db_a.transaction() as conn:
            policy_rows = conn.execute(
                """
                SELECT relname, relrowsecurity, relforcerowsecurity
                  FROM pg_class
                 WHERE relname IN (
                   'note_clipper_documents',
                   'note_clipper_workspace_placements'
                 )
                """
            ).fetchall()
            assert {row["relname"] for row in policy_rows} == {
                "note_clipper_documents",
                "note_clipper_workspace_placements",
            }
            assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in policy_rows)
            hidden_update = conn.execute(
                """
                UPDATE note_clipper_documents
                   SET source_title = 'cross-owner overwrite'
                 WHERE client_id = ? AND clip_id = ?
                """,
                (owner_b, clip_id),
            )
            assert hidden_update.rowcount == 0

        assert db_b.get_note_clipper_document_by_clip_id(clip_id)["source_title"] == (f"Title for {owner_b}")

        with pytest.raises(DatabaseError):
            with db_a.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO note_clipper_documents(
                      client_id, clip_id, note_id, clip_type, source_url,
                      source_title, capture_metadata_json, analysis_json,
                      content_budget_json, source_note_version, deleted
                    ) VALUES (?, ?, ?, 'article', '', '', '{}', '{}', '{}', 1, FALSE)
                    """,
                    (owner_b, "cross-owner-insert", note_b),
                )

        with pytest.raises(DatabaseError):
            with db_a.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO note_clipper_documents(
                      client_id, clip_id, note_id, clip_type, source_url,
                      source_title, capture_metadata_json, analysis_json,
                      content_budget_json, source_note_version, deleted
                    ) VALUES (?, ?, ?, 'article', '', '', '{}', '{}', '{}', 1, FALSE)
                    """,
                    (owner_a, "cross-owner-note-endpoint", note_b),
                )

        with pytest.raises(DatabaseError):
            with db_a.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO note_clipper_workspace_placements(
                      client_id, clip_id, workspace_id, source_note_id,
                      source_note_version, deleted
                    ) VALUES (?, ?, ?, ?, 1, FALSE)
                    """,
                    (owner_a, clip_id, "workspace-b", note_b),
                )
    finally:
        db_a.close_connection()
        db_b.close_connection()
