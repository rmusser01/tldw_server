from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.workspaces import get_source_preview
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseBackend,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema
from tldw_Server_API.app.core.DB_Management.chacha.migrations import postgres_steps
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, SchemaError

# Runs under `pg_restricted_backend`, whose role is NOSUPERUSER NOBYPASSRLS.
# The plain `pg_database_config` role is a superuser, and a superuser is exempt
# from row-level security even under FORCE -- under it the cross-owner UPDATE
# below succeeds and this test fails, which is what it did while it was gated
# behind a hand-set DSN and therefore never ran.
pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.owner_isolation]


def _create_owner_clip(
    db: CharactersRAGDB,
    *,
    clip_id: str,
    note_id: str,
    workspace_id: str,
) -> None:
    assert db.add_note(title=f"Clip for {db.client_id}", content="Body", note_id=note_id)
    db.workspace_store.upsert_workspace(workspace_id, f"Workspace for {db.client_id}")
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
    db.workspace_store.upsert_note_clipper_workspace_placement(
        clip_id=clip_id,
        workspace_id=workspace_id,
        source_note_id=note_id,
        source_note_version=1,
    )


@pytest.mark.parametrize("with_clips", [False, True], ids=["empty", "shared-clip"])
def test_postgres_web_clipper_v56_upgrades_owner_scoped_layout(
    pg_restricted_backend: DatabaseBackend,
    with_clips: bool,
) -> None:
    backend = pg_restricted_backend
    owners = ("910001", "910002") if with_clips else ("910001",)
    databases = [CharactersRAGDB(db_path=":memory:", client_id=owner, backend=backend) for owner in owners]
    clip_id = "shared-migration-clip"
    note_ids = (clip_id, str(uuid4())) if with_clips else ()
    try:
        if with_clips:
            db_a, db_b = databases
            assert db_a.add_note(title="Legacy capture", content="Body", note_id=clip_id)
            db_a.workspace_store.upsert_workspace("ws-0", "Legacy workspace")
            with db_a.transaction() as conn:
                conn.execute(
                    "INSERT INTO note_clipper_documents(client_id, clip_id, note_id, clip_type) "
                    "VALUES (?, ?, ?, 'article')",
                    (db_a.client_id, clip_id, clip_id),
                )
                conn.execute(
                    "INSERT INTO note_clipper_workspace_placements(client_id, clip_id, workspace_id, source_note_id) "
                    "VALUES (?, ?, 'ws-0', ?)",
                    (db_a.client_id, clip_id, clip_id),
                )
            _create_owner_clip(db_b, clip_id=clip_id, note_id=note_ids[1], workspace_id="ws-1")

        ctx = databases[0].ctx
        with backend.transaction() as conn:
            if with_clips:
                conn.execute(
                    "ALTER TABLE note_clipper_documents "
                    "RENAME CONSTRAINT note_clipper_documents_pkey TO fixture_document_pk"
                )
            postgres_schema._set_schema_version_postgres(ctx, conn, 55)

        with backend.transaction() as conn:
            postgres_steps._migrate_from_v55_to_v56_postgres(ctx, conn)
            assert postgres_schema._get_schema_version_postgres(ctx, conn) == 56
            constraints = backend.execute(
                """
                SELECT conname, convalidated
                  FROM pg_constraint
                 WHERE (conrelid = 'note_clipper_documents'::regclass AND contype IN ('p', 'u'))
                    OR (conrelid = 'note_clipper_workspace_placements'::regclass
                        AND (contype = 'p' OR confrelid = 'note_clipper_documents'::regclass))
                """,
                connection=conn,
            ).rows
            assert {row["conname"]: row["convalidated"] for row in constraints} == {
                "note_clipper_documents_pkey": True,
                "note_clipper_documents_client_note_key": True,
                "note_clipper_workspace_placements_pkey": True,
                "note_clipper_workspace_placements_document_fkey": True,
            }
            forced_tables = backend.execute(
                """
                SELECT relname FROM pg_class
                 WHERE relrowsecurity AND relforcerowsecurity
                   AND relname IN (
                       'notes', 'workspaces', 'sync_log',
                       'note_clipper_documents', 'note_clipper_workspace_placements'
                   )
                """,
                connection=conn,
            ).rows
            assert {row["relname"] for row in forced_tables} == {
                "notes", "workspaces", "sync_log", "note_clipper_documents", "note_clipper_workspace_placements",
            }

        if with_clips:
            for index, (db, note_id) in enumerate(zip(databases, note_ids)):
                document = db.get_note_clipper_document_by_clip_id(clip_id)
                assert document is not None
                migrated_id = document["note_id"]
                if note_id == clip_id:
                    assert migrated_id != note_id and UUID(migrated_id).version == 4
                else:
                    assert migrated_id == note_id
                placements = db.workspace_store.list_note_clipper_workspace_placements(clip_id)
                assert [(row["workspace_id"], row["source_note_id"]) for row in placements] == [
                    (f"ws-{index}", migrated_id)
                ]
    finally:
        for db in databases:
            db.close_connection()


def test_postgres_web_clipper_v56_upgrades_legacy_single_owner_keys(
    pg_restricted_backend: DatabaseBackend,
) -> None:
    backend = pg_restricted_backend
    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=backend)
    clip_id = "legacy-clip-3001"
    try:
        assert db.add_note(title="Legacy capture", content="Body", note_id=clip_id)
        db.workspace_store.upsert_workspace("legacy-ws", "Legacy workspace")
        with backend.transaction() as conn:
            conn.execute("DROP TABLE note_clipper_workspace_placements")
            conn.execute("DROP TABLE note_clipper_documents")
            conn.execute(
                """
                CREATE TABLE note_clipper_documents (
                    clip_id TEXT PRIMARY KEY REFERENCES notes(id) ON UPDATE CASCADE ON DELETE CASCADE,
                    note_id TEXT NOT NULL UNIQUE REFERENCES notes(id) ON UPDATE CASCADE ON DELETE CASCADE,
                    CHECK (clip_id = note_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE note_clipper_workspace_placements (
                    clip_id TEXT NOT NULL REFERENCES note_clipper_documents(clip_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
                    source_note_id TEXT NOT NULL REFERENCES notes(id) ON UPDATE CASCADE ON DELETE CASCADE,
                    PRIMARY KEY (clip_id, workspace_id)
                )
                """
            )
            conn.execute("INSERT INTO note_clipper_documents VALUES (%s, %s)", (clip_id, clip_id))
            conn.execute(
                "INSERT INTO note_clipper_workspace_placements VALUES (%s, 'legacy-ws', %s)", (clip_id, clip_id)
            )
            postgres_schema._set_schema_version_postgres(db.ctx, conn, 55)

        with backend.transaction() as conn:
            postgres_steps._migrate_from_v55_to_v56_postgres(db.ctx, conn)
            assert postgres_schema._get_schema_version_postgres(db.ctx, conn) == 56
            conn.execute("SELECT set_config('app.current_user_id', %s, true)", (db.client_id,))
            document = conn.execute(
                "SELECT client_id, note_id FROM note_clipper_documents WHERE clip_id = %s", (clip_id,)
            ).fetchone()
            assert document is not None
            assert document["client_id"] == db.client_id
            assert document["note_id"] != clip_id and UUID(document["note_id"]).version == 4
            placement = conn.execute(
                "SELECT client_id, source_note_id FROM note_clipper_workspace_placements WHERE clip_id = %s",
                (clip_id,),
            ).fetchone()
            assert placement == {"client_id": db.client_id, "source_note_id": document["note_id"]}
    finally:
        db.close_connection()


def test_postgres_web_clipper_v56_preserves_unrecognized_foreign_key_actions(
    pg_restricted_backend: DatabaseBackend,
) -> None:
    backend = pg_restricted_backend
    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=backend)
    try:
        with backend.transaction() as conn:
            conn.execute(
                "ALTER TABLE note_clipper_workspace_placements "
                "DROP CONSTRAINT note_clipper_workspace_placements_client_id_clip_id_fkey"
            )
            conn.execute(
                """
                ALTER TABLE note_clipper_workspace_placements ADD CONSTRAINT clip_restricts_delete
                FOREIGN KEY (client_id, clip_id) REFERENCES note_clipper_documents(client_id, clip_id)
                ON UPDATE CASCADE ON DELETE RESTRICT
                """
            )
            postgres_schema._set_schema_version_postgres(db.ctx, conn, 55)

        with pytest.raises(SchemaError, match="placement constraints"), backend.transaction() as conn:
            postgres_steps._migrate_from_v55_to_v56_postgres(db.ctx, conn)

        with backend.transaction() as conn:
            assert postgres_schema._get_schema_version_postgres(db.ctx, conn) == 55
            key = conn.execute(
                "SELECT confdeltype FROM pg_constraint WHERE conname = 'clip_restricts_delete'"
            ).fetchone()
            assert key == {"confdeltype": "r"}
    finally:
        db.close_connection()


def test_postgres_web_clipper_v56_rolls_back_mismatched_placement(
    pg_restricted_backend: DatabaseBackend,
) -> None:
    backend = pg_restricted_backend
    db_a = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=backend)
    db_b = CharactersRAGDB(db_path=":memory:", client_id="910002", backend=backend)
    try:
        _create_owner_clip(db_a, clip_id="legacy-mapping", note_id=str(uuid4()), workspace_id="owned-ws")
        db_b.workspace_store.upsert_workspace("other-ws", "Other workspace")
        with backend.transaction() as conn:
            conn.execute("ALTER TABLE note_clipper_workspace_placements NO FORCE ROW LEVEL SECURITY")
            updated = conn.execute(
                "UPDATE note_clipper_workspace_placements SET workspace_id = 'other-ws' "
                "WHERE clip_id = 'legacy-mapping'"
            )
            assert updated.rowcount == 1
            conn.execute("ALTER TABLE note_clipper_workspace_placements FORCE ROW LEVEL SECURITY")
            postgres_schema._set_schema_version_postgres(db_a.ctx, conn, 55)

        with pytest.raises(SchemaError, match="placement.*owner relationships"), backend.transaction() as conn:
            postgres_steps._migrate_from_v55_to_v56_postgres(db_a.ctx, conn)

        with backend.transaction() as conn:
            assert postgres_schema._get_schema_version_postgres(db_a.ctx, conn) == 55
            state = backend.execute(
                """
                SELECT relrowsecurity, relforcerowsecurity
                  FROM pg_class
                 WHERE oid = 'note_clipper_workspace_placements'::regclass
                """,
                connection=conn,
            ).first
            assert state == {"relrowsecurity": True, "relforcerowsecurity": True}
    finally:
        db_a.close_connection()
        db_b.close_connection()


def test_postgres_web_clipper_same_clip_is_owner_isolated_by_rls(
    pg_restricted_backend: DatabaseBackend,
) -> None:
    owner_a = "910001"
    owner_b = "910002"
    clip_id = "shared-public-clip"
    note_a = str(uuid4())
    note_b = str(uuid4())
    db_a = CharactersRAGDB(db_path=":memory:", client_id=owner_a, backend=pg_restricted_backend)
    db_b = CharactersRAGDB(db_path=":memory:", client_id=owner_b, backend=pg_restricted_backend)

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
        assert [row["workspace_id"] for row in db_a.workspace_store.list_note_clipper_workspace_placements(clip_id)] == ["workspace-a"]
        assert [row["workspace_id"] for row in db_b.workspace_store.list_note_clipper_workspace_placements(clip_id)] == ["workspace-b"]

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


@pytest.mark.asyncio
async def test_postgres_pinned_preview_foreign_workspace_denied_before_media_read(pg_restricted_backend):
    db_a = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=pg_restricted_backend)
    db_b = CharactersRAGDB(db_path=":memory:", client_id="910002", backend=pg_restricted_backend)
    workspace = f"foreign-{uuid4()}"
    try:
        db_b.workspace_store.upsert_workspace(workspace, "Foreign workspace")
        assert db_b.workspace_store.get_workspace(workspace) is not None
        assert db_a.workspace_store.get_workspace(workspace) is None

        class UnreadableMedia:
            def get_media_by_id(self, *_args, **_kwargs):
                pytest.fail("foreign workspace must be rejected before Media read")

            def get_document_version(self, *_args, **_kwargs):
                pytest.fail("foreign workspace must be rejected before exact version read")

        with pytest.raises(HTTPException) as error:
            await get_source_preview(
                workspace_id=workspace,
                source_id="source",
                max_chars=3000,
                chunk_limit=3,
                version_number=1,
                db=db_a,
                media_db=UnreadableMedia(),
                jm=None,
                current_user=SimpleNamespace(id=910001),
            )
        assert error.value.status_code == 404
    finally:
        db_a.close_connection()
        db_b.close_connection()


def test_postgres_capture_promotes_owned_source_and_retries(pg_restricted_backend, tmp_path):
    from loguru import logger

    from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.WebClipper.schemas import WebClipperSaveRequest
    from tldw_Server_API.app.core.WebClipper.service import WebClipperService

    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=pg_restricted_backend)
    media = MediaDatabase(db_path=str(tmp_path / "capture-media.db"), client_id="910001")
    diagnostics = []
    sink = logger.add(
        lambda message: (
            diagnostics.append(message.record["extra"])
            if "PostgreSQL query execution failed" in message.record["message"]
            else None
        )
    )
    workspace = f"owned-{uuid4()}"
    clip = str(uuid4())
    text = "accepted café article"
    request = WebClipperSaveRequest(
        clip_id=clip,
        clip_type="article",
        source_url="https://example.com/article",
        source_title="Article",
        destination_mode="workspace",
        workspace={"workspace_id": workspace, "default_review_state": "needs_review"},
        content={"full_extract": text},
        capture_metadata={
            "web_capture_v1": {
                "mode": "server_article",
                "requested_url": "https://example.com/article",
                "captured_at": "2026-10-07T18:00:00Z",
                "refresh_of": None,
                "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        },
    )
    try:
        db.workspace_store.upsert_workspace(workspace, "Owned capture")
        service = WebClipperService(db=db, media_db=media, user_id=910001, promote_workspace_sources=True)
        try:
            first = service.save_clip(request)
            retry = service.save_clip(request)
        except DatabaseError:
            pytest.fail(f"PostgreSQL capture write failed: {diagnostics}")
        assert first.status == retry.status == "saved", (first.warnings, retry.warnings, diagnostics)
        assert first.note.id == retry.note.id
        sources = db.workspace_store.list_workspace_sources(workspace)
        assert len(sources) == 1
        assert sources[0]["id"] == f"web-clipper:{clip}"
        assert sources[0]["selected"] is True
        original_source = sources[0]
        original_media_id = int(original_source["media_id"])
        original_version = media_db_api.get_document_version(media, original_media_id, 1)
        original_chunk = media_db_api.get_unvectorized_chunks_in_range(media, original_media_id, 0, 0)
        media_ids = {original_media_id}
        media_uuids = {media_db_api.get_media_by_id(media, original_media_id)["uuid"]}
        previous_clip = clip
        for refreshed_text in (text, "changed accepted café article"):
            payload = request.model_dump()
            payload["clip_id"] = str(uuid4())
            payload["content"]["full_extract"] = refreshed_text
            descriptor = payload["capture_metadata"]["web_capture_v1"]
            descriptor["refresh_of"] = previous_clip
            descriptor["content_sha256"] = hashlib.sha256(refreshed_text.encode("utf-8")).hexdigest()
            refreshed_request = WebClipperSaveRequest.model_validate(payload)
            refreshed = service.save_clip(refreshed_request)
            retried = service.save_clip(refreshed_request)
            assert refreshed.status == retried.status == "saved"
            assert refreshed.note.id == retried.note.id != first.note.id
            sources = db.workspace_store.list_workspace_sources(workspace)
            source = next(row for row in sources if row["id"] == f"web-clipper:{refreshed_request.clip_id}")
            media_id = int(source["media_id"])
            assert media_id not in media_ids
            media_uuid = media_db_api.get_media_by_id(media, media_id)["uuid"]
            assert media_uuid not in media_uuids
            version = media_db_api.get_document_version(media, media_id, 1)
            assert version["uuid"] != original_version["uuid"]
            assert version["content"] == refreshed_text
            assert json.loads(version["safe_metadata"])["capture_metadata"]["web_capture_v1"] == descriptor
            assert len(media_db_api.list_document_versions(media, media_id)) == 1
            assert original_source in sources
            assert media_db_api.get_document_version(media, original_media_id, 1) == original_version
            assert media_db_api.get_unvectorized_chunks_in_range(media, original_media_id, 0, 0) == original_chunk
            media_ids.add(media_id)
            media_uuids.add(media_uuid)
            previous_clip = refreshed_request.clip_id
        assert len(db.workspace_store.list_workspace_sources(workspace)) == 3
        assert service.save_clip(request).note.id == first.note.id
        retried_version = media_db_api.get_document_version(media, original_media_id, 1)
        assert (retried_version["uuid"], retried_version["content"], retried_version["safe_metadata"]) == (
            original_version["uuid"],
            original_version["content"],
            original_version["safe_metadata"],
        )
        assert len(db.workspace_store.list_workspace_sources(workspace)) == 3
    finally:
        logger.remove(sink)
        db.close_connection()
        media.close_connection()


def test_postgres_source_duplicate_retains_first_snapshot(pg_restricted_backend):
    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=pg_restricted_backend)
    workspace = f"duplicate-{uuid4()}"
    try:
        db.workspace_store.upsert_workspace(workspace, "Duplicate sources")
        first = db.workspace_store.add_workspace_source(
            workspace, {"id": "same", "media_id": 1, "title": "First", "source_type": "web", "selected": False}
        )
        retry = db.workspace_store.add_workspace_source(
            workspace, {"id": "same", "media_id": 2, "title": "Replacement", "source_type": "pdf", "selected": True}
        )
        assert retry == first
        assert len(db.workspace_store.list_workspace_sources(workspace)) == 1
    finally:
        db.close_connection()


def test_postgres_source_selection_uses_native_booleans_and_versions(pg_restricted_backend):
    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=pg_restricted_backend)
    workspace = f"selection-{uuid4()}"
    try:
        db.workspace_store.upsert_workspace(workspace, "Select sources")
        first = db.workspace_store.add_workspace_source(workspace, {"id": "a", "media_id": 1, "title": "A", "source_type": "web"})
        second = db.workspace_store.add_workspace_source(
            workspace, {"id": "b", "media_id": 2, "title": "B", "source_type": "web", "selected": False}
        )
        updated = db.workspace_store.update_workspace_source(workspace, "b", {"selected": True}, expected_version=second["version"])
        assert updated["selected"] is True
        db.workspace_store.update_workspace_source_selection(workspace, selected_ids=["a"])
        rows = {row["id"]: row for row in db.workspace_store.list_workspace_sources(workspace)}
        assert rows["a"]["selected"] is True
        assert rows["a"]["version"] == first["version"] + 2
        assert rows["b"]["selected"] is False
        assert rows["b"]["version"] == updated["version"] + 1
        db.workspace_store.update_workspace_source_selection(workspace, selected_ids=[])
        assert all(row["selected"] is False for row in db.workspace_store.list_workspace_sources(workspace))
    finally:
        db.close_connection()


def test_postgres_source_constraints_and_foreign_selection_remain_enforced(pg_restricted_backend):
    db_a = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=pg_restricted_backend)
    db_b = CharactersRAGDB(db_path=":memory:", client_id="910002", backend=pg_restricted_backend)
    workspace = f"constraints-{uuid4()}"
    try:
        db_a.workspace_store.upsert_workspace(workspace, "Owned source")
        first = db_a.workspace_store.add_workspace_source(workspace, {"id": "a", "media_id": 1, "title": "A", "source_type": "web"})
        with pytest.raises(DatabaseError):
            db_a.workspace_store.add_workspace_source(workspace, {"id": "invalid", "media_id": 2, "title": None, "source_type": "web"})
        with pytest.raises(DatabaseError):
            db_a.workspace_store.add_workspace_source(
                "nonexistent-workspace", {"id": "missing", "media_id": 2, "title": "B", "source_type": "web"}
            )
        with pytest.raises(DatabaseError):
            db_b.workspace_store.add_workspace_source(workspace, {"id": "foreign", "media_id": 2, "title": "B", "source_type": "web"})
        db_b.workspace_store.update_workspace_source_selection(workspace, selected_ids=[])
        assert db_a.workspace_store.list_workspace_sources(workspace) == [first]
        assert db_b.workspace_store.list_workspace_sources(workspace) == []
    finally:
        db_a.close_connection()
        db_b.close_connection()
