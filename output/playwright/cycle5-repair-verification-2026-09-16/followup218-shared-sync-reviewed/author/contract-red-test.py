"""UAT218: real reverse initialization must preserve the shared content stores."""

import json
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


@contextmanager
def stores(request, tmp_path, kind, order):
    """Use normal constructors and official PG fixtures; SQLite stores stay separate."""
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if kind == "postgresql" else None
    )
    db = media = None
    try:
        if order == "media-first":
            media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
        with chacha_operation(independent=True):
            note = db.add_note(title="Preserved owner source", content="Synthetic evidence")
            before = db.get_sync_log_entries()
        if media is None:
            media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        yield db, media, backend, note, before
    finally:
        if db is not None:
            db.close_connection()
        if media is not None:
            media.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("kind", ["postgresql", "sqlite"])
@pytest.mark.parametrize("order", ["media-first", "chacha-first"])
def test_both_orders_preserve_history_and_normal_media_cascade(request, tmp_path, kind, order):
    with stores(request, tmp_path, kind, order) as (db, media, backend, note, before):
        with chacha_operation(independent=True):
            after = db.get_sync_log_entries()
            # Added nullable scope columns are allowed; every original value is preserved.
            assert [{key: row[key] for key in old} for row, old in zip(after, before, strict=True)] == before
            assert db.get_note_by_id(note)["content"] == "Synthetic evidence"
        media_id, media_uuid, _ = media.add_media_with_keywords(
            title="Synthetic media", media_type="document", content="Synthetic media evidence", keywords=["fixture"]
        )
        assert media.soft_delete_media(media_id, cascade=True)
        rows = media.get_sync_log_entries()
        assert any(row["entity_uuid"] == media_uuid and row["operation"] == "delete" for row in rows)
        assert any(row["entity"] == "MediaKeywords" and row["operation"] == "unlink" for row in rows)
        assert all(row["client_id"] == "2" for row in rows)
        if backend is not None:
            names = {column["name"] for column in backend.get_table_info("sync_log")}
            assert names & {"entity_id", "entity_uuid"} == {"entity_id" if order == "chacha-first" else "entity_uuid"}
        assert media.get_sync_log_entries(since_change_id=rows[-2]["change_id"], limit=1) == [rows[-1]]


@pytest.mark.parametrize("order", ["media-first", "chacha-first"])
def test_optional_and_invalid_payloads_keep_both_readers_compatible(request, tmp_path, order):
    with stores(request, tmp_path, "postgresql", order) as (db, media, backend, _note, _before):
        with media.transaction() as conn:
            media._log_sync_event(conn, "fixture_optional", "optional-id", "create", 1)
        rows = media.get_sync_log_entries()
        optional = next(row for row in rows if row["entity"] == "fixture_optional")
        assert optional["payload"] is None
        with chacha_operation(independent=True):
            assert db.get_sync_log_entries(entity_type="fixture_optional")[0]["payload"] is None
        backend.execute("UPDATE sync_log SET payload=%s WHERE change_id=%s", ("not-json", optional["change_id"]))
        assert media.get_sync_log_entries(since_change_id=optional["change_id"] - 1)[0]["payload"] is None
        with chacha_operation(independent=True):
            assert db.get_sync_log_entries(entity_type="fixture_optional")[0]["payload"] is None


def _sync_shape(backend):
    with backend.transaction() as conn:
        return {
            "columns": backend.get_table_info("sync_log", connection=conn),
            "table": backend.execute(
                "SELECT oid, relowner, relacl, relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid='sync_log'::regclass",
                connection=conn,
            ).rows,
            "checks": backend.execute(
                "SELECT oid, conname, pg_get_constraintdef(oid) AS definition FROM pg_constraint "
                "WHERE conrelid='sync_log'::regclass AND contype='c' ORDER BY conname", connection=conn,
            ).rows,
            "rows": backend.execute("SELECT * FROM sync_log ORDER BY change_id", connection=conn).rows,
            "policies": backend.execute(
                "SELECT * FROM pg_policies WHERE schemaname='public' AND tablename='sync_log' ORDER BY policyname",
                connection=conn,
            ).rows,
        }


@pytest.mark.parametrize("order", ["media-first", "chacha-first"])
def test_reopen_preserves_rows_constraints_scope_and_policies(request, tmp_path, order):
    with stores(request, tmp_path, "postgresql", order) as (_db, media, backend, _note, _before):
        with media.transaction() as conn:
            media._log_sync_event(conn, "fixture_scope", "scope-id", "create", 1, {"owned": True})
        backend.execute("UPDATE sync_log SET org_id=7, team_id=8 WHERE entity=%s", ("fixture_scope",))
        media.close_connection()
        before = _sync_shape(backend)
        reopened = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        try:
            assert _sync_shape(backend) == before
            scoped = next(row for row in reopened.get_sync_log_entries() if row["entity"] == "fixture_scope")
            assert (scoped["org_id"], scoped["team_id"], scoped["payload"]) == (7, 8, {"owned": True})
        finally:
            reopened.close_connection()


def test_late_initializer_failure_rolls_back_normalization(pg_database_config, tmp_path, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    try:
        with chacha_operation(independent=True):
            db.add_note(title="Committed source", content="Preserve through rollback")
        before = _sync_shape(backend)
        def fail_late(*_args):
            raise SchemaError("controlled late failure")
        with monkeypatch.context() as scoped:
            scoped.setattr(MediaDatabase, "_ensure_postgres_email_schema", fail_late)
            with pytest.raises(SchemaError, match="controlled late failure"):
                MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        assert _sync_shape(backend) == before
        assert not backend.table_exists("media")
        media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        media.close_connection()
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("shape", ["both-identifiers", "missing-identifier", "custom-operation-check"])
def test_unknown_existing_sync_shape_rejects_reopen_atomically(pg_database_config, tmp_path, shape):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
    try:
        media.close_connection()
        statements = {
            "both-identifiers": "ALTER TABLE sync_log ADD COLUMN entity_id TEXT",
            "missing-identifier": "ALTER TABLE sync_log RENAME COLUMN entity_uuid TO custom_identity",
            "custom-operation-check": "ALTER TABLE sync_log ADD CONSTRAINT fixture_operation CHECK(operation <> 'unlink')",
        }
        backend.execute(statements[shape])
        before = _sync_shape(backend)
        with pytest.raises(SchemaError, match="sync.log"):
            MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        assert _sync_shape(backend) == before
    finally:
        media.close_connection()
        backend.get_pool().close_all()


def test_chacha_then_media_can_initialize_shared_content(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    chacha = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    media = None
    try:
        media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        assert backend.table_exists("media")
        assert backend.table_exists("study_packs")
    finally:
        chacha.close_connection()
        if media is not None:
            media.close_connection()
        backend.get_pool().close_all()
