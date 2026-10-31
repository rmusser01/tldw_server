"""UAT218: real reverse initialization must preserve the shared content stores."""

import asyncio
import json
from contextlib import contextmanager
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.StudyPacks import generation_service
from tldw_Server_API.app.services import study_pack_jobs_worker as worker

pytestmark = pytest.mark.integration


@contextmanager
def stores(request, tmp_path, kind, order):
    """Use normal constructors and official PG fixtures; SQLite stores stay separate."""
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if kind == "postgresql"
        else None
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
                "WHERE conrelid='sync_log'::regclass AND contype='c' ORDER BY conname",
                connection=conn,
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
            with pytest.raises(DatabaseError, match="controlled late failure") as error:
                MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
            assert isinstance(error.value.__cause__, SchemaError)
        assert _sync_shape(backend) == before
        assert not backend.table_exists("media")
        media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        media.close_connection()
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize(
    "shape",
    [
        "both-identifiers",
        "missing-identifier",
        "custom-operation-check",
        "scope-nullability",
        "identifier-nullable",
        "operation-nullable",
    ],
)
def test_unknown_existing_sync_shape_rejects_reopen_atomically(pg_database_config, tmp_path, shape):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
    try:
        media.close_connection()
        statements = {
            "both-identifiers": "ALTER TABLE sync_log ADD COLUMN entity_id TEXT",
            "missing-identifier": "ALTER TABLE sync_log RENAME COLUMN entity_uuid TO custom_identity",
            "custom-operation-check": "ALTER TABLE sync_log ADD CONSTRAINT fixture_operation CHECK(operation <> 'unlink')",
            "scope-nullability": "ALTER TABLE sync_log ALTER COLUMN org_id SET NOT NULL",
            "identifier-nullable": "ALTER TABLE sync_log ALTER COLUMN entity_uuid DROP NOT NULL",
            "operation-nullable": "ALTER TABLE sync_log ALTER COLUMN operation DROP NOT NULL",
        }
        backend.execute(statements[shape])
        before = _sync_shape(backend)
        with pytest.raises(DatabaseError, match="sync.log") as error:
            MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        assert isinstance(error.value.__cause__, SchemaError)
        assert _sync_shape(backend) == before
    finally:
        media.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["cre ate", "create "])
def test_unknown_operation_literal_is_preserved_and_reopen_rejected(pg_database_config, tmp_path, operation):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
    try:
        media.close_connection()
        before = _sync_shape(backend)
        with backend.transaction() as conn:
            constraint = backend.escape_identifier(before["checks"][0]["conname"])
            backend.execute(f"ALTER TABLE sync_log DROP CONSTRAINT {constraint}", connection=conn)
            # Negative fixture DDL is fixed, never interpolated from an unchecked value.
            definitions = {
                "cre ate": "ALTER TABLE sync_log ADD CONSTRAINT fixture_operation CHECK(operation IN ('cre ate','update','delete'))",
                "create ": "ALTER TABLE sync_log ADD CONSTRAINT fixture_operation CHECK(operation IN ('create ','update','delete'))",
            }
            backend.execute(definitions[operation], connection=conn)
        before = _sync_shape(backend)
        with pytest.raises(DatabaseError, match="sync_log operation constraint") as error:
            MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        assert isinstance(error.value.__cause__, SchemaError)
        assert _sync_shape(backend) == before
    finally:
        media.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("kind", ["postgresql", "sqlite"])
@pytest.mark.parametrize("order", ["media-first", "chacha-first"])
def test_actual_worker_keeps_pack_membership_citation_and_sync_owner(request, tmp_path, monkeypatch, kind, order):
    with stores(request, tmp_path, kind, order) as (db, media, _backend, note, _before):

        async def acquire(owner):
            assert owner == "2"
            return db, media

        async def model(self, *, system_prompt, user_prompt):
            assert "Synthetic evidence" in user_prompt
            return json.dumps(
                {
                    "cards": [
                        {
                            "front": "Synthetic question",
                            "back": "Synthetic answer",
                            "citations": [
                                {"source_type": "note", "source_id": note, "citation_text": "Synthetic evidence"}
                            ],
                        }
                    ]
                }
            )

        monkeypatch.setattr(worker, "_get_databases_for_user", acquire)
        monkeypatch.setattr(
            generation_service, "_resolve_generation_provider_and_model", lambda *_: ("fixture", "fixture", {})
        )
        monkeypatch.setattr(generation_service.StudyPackGenerationService, "_call_generation_model", model)
        result = asyncio.run(
            worker.handle_study_pack_job(
                {
                    "id": 2,
                    "owner_user_id": "2",
                    "job_type": "study_pack_generate",
                    "payload": {
                        "title": "Synthetic pack",
                        "deck_mode": "new",
                        "source_items": [{"source_type": "note", "source_id": note}],
                    },
                }
            )
        )
        with chacha_operation(independent=True):
            pack = db.get_study_pack(result["pack_id"])
            assert pack["client_id"] == "2" and pack["deck_id"] == result["deck_id"]
            members = db.list_study_pack_cards(pack["id"])
            assert len(members) == 1
            card = db.get_flashcard(members[0]["flashcard_uuid"])
            assert (card["client_id"], card["front"], card["back"]) == ("2", "Synthetic question", "Synthetic answer")
            citation = db.list_flashcard_citations(card["uuid"])[0]
            assert (citation["client_id"], citation["source_id"]) == ("2", note)
            for entity, expected_id in (
                ("study_packs", pack["id"]),
                ("study_pack_cards", members[0]["id"]),
                ("flashcard_citations", citation["id"]),
            ):
                row = db.get_sync_log_entries(entity_type=entity)[-1]
                assert row.get("entity_id", row.get("entity_uuid")) == str(expected_id)
                assert row["client_id"] == row["payload"]["client_id"] == "2"
                assert row["payload"]["id"] == expected_id


def test_legacy_pack_rows_survive_media_normalization(pg_database_config, tmp_path):
    from tldw_Server_API.tests.DB_Management.test_study_pack_shared_sync_schema import _create_graph

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="2", backend=backend)
    media = None
    try:
        with chacha_operation(independent=True):
            pack, card = _create_graph(db)
            before = db.get_sync_log_entries()
        original_table = _sync_shape(backend)["table"][0]
        media = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        with chacha_operation(independent=True):
            after = db.get_sync_log_entries()
            assert [{key: row[key] for key in old} for row, old in zip(after, before, strict=True)] == before
            assert db.get_study_pack(pack)["client_id"] == "2"
            assert db.list_flashcard_citations(card)[0]["citation_text"] == "Synthetic"
            assert db.soft_delete_study_pack(pack, expected_version=1)
            assert db.get_sync_log_entries(entity_type="study_packs")[-1]["operation"] == "delete"
        current_table = _sync_shape(backend)["table"][0]
        assert {key: current_table[key] for key in ("oid", "relowner", "relacl")} == {
            key: original_table[key] for key in ("oid", "relowner", "relacl")
        }
    finally:
        db.close_connection()
        if media is not None:
            media.close_connection()
        backend.get_pool().close_all()


def test_helper_does_not_commit_caller_work_or_repeat_compatible_ddl(request, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
        ensure_postgres_sync_log_contract,
    )

    with stores(request, tmp_path, "postgresql", "chacha-first") as (_db, media, backend, _note, _before):
        before = _sync_shape(backend)
        statements = []
        execute = backend.execute

        def record(query, *args, **kwargs):
            statements.append(query)
            return execute(query, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(backend, "execute", record)
            with pytest.raises(RuntimeError, match="caller rollback"):
                with backend.transaction() as conn:
                    backend.execute("UPDATE sync_log SET client_id=%s", ("pending",), connection=conn)
                    ensure_postgres_sync_log_contract(media, conn)
                    raise RuntimeError("caller rollback")
        assert _sync_shape(backend) == before
        assert not any(query.lstrip().upper().startswith("ALTER TABLE") for query in statements)


@pytest.mark.parametrize("order", ["media-first", "chacha-first"])
def test_personal_org_team_rls_remains_enforced_after_reopen(request, tmp_path, order):
    with stores(request, tmp_path, "postgresql", order) as (_db, media, backend, _note, _before):
        with media.transaction() as conn:
            for entity in ("fixture_personal", "fixture_org", "fixture_team"):
                media._log_sync_event(conn, entity, entity, "create", 1, {"owner": "2"})
        backend.execute("UPDATE sync_log SET org_id=7 WHERE entity=%s", ("fixture_org",))
        backend.execute("UPDATE sync_log SET team_id=8 WHERE entity=%s", ("fixture_team",))
        media.close_connection()
        reopened = MediaDatabase(str(tmp_path / "media.db"), client_id="2", backend=backend)
        role = backend.escape_identifier(f"sync_contract_{uuid4().hex[:12]}")
        created = False
        try:
            with backend.transaction() as conn:
                backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
                backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
                backend.execute(f"GRANT SELECT ON sync_log TO {role}", connection=conn)
                backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
            created = True
            for owner, org, team, expected in (
                ("2", "", "", {"fixture_personal", "fixture_org", "fixture_team"}),
                ("3", "", "", set()),
                ("3", "7", "", {"fixture_org"}),
                ("3", "", "8", {"fixture_team"}),
            ):
                with backend.transaction() as conn:
                    backend.execute(f"SET LOCAL ROLE {role}", connection=conn)
                    assert (
                        backend.execute(
                            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname=current_user", connection=conn
                        ).scalar
                        is False
                    )
                    for key, value in (
                        ("app.current_user_id", owner),
                        ("app.org_ids", org),
                        ("app.team_ids", team),
                        ("app.is_admin", "0"),
                    ):
                        backend.execute("SELECT set_config(%s,%s,true)", (key, value), connection=conn)
                    rows = backend.execute(
                        "SELECT entity FROM sync_log WHERE entity LIKE 'fixture_%'", connection=conn
                    ).rows
                    assert {row["entity"] for row in rows} == expected
        finally:
            if created:
                with backend.transaction() as conn:
                    backend.execute(f"DROP OWNED BY {role}", connection=conn)
                    backend.execute(f"DROP ROLE {role}", connection=conn)
            reopened.close_connection()


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
