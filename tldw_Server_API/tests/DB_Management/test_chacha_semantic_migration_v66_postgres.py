"""PostgreSQL schema-v66 contracts for durable semantic cleanup authority."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticChunkRecord,
    SemanticDimensionState,
    SemanticSnapshotSeed,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    build_semantic_chunks,
    semantic_content_fingerprint,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

_TABLE = "note_semantic_obsolete_vectors"
_NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


class _FakeTransaction:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    def table_exists(self, _name: str, connection: object = None) -> bool:
        return True


def _fake_postgres_initializer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_version: int,
    target_version: int,
) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", target_version)
    monkeypatch.setattr(
        db,
        "_get_schema_version_postgres",
        lambda _conn, lock=False: current_version,
    )
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_semantic_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )
    return db


def _seed_complete_semantic_generation(db: CharactersRAGDB) -> tuple[str, str, str, str]:
    note_id = "11111111-1111-4111-8111-111111111111"
    dataset_id = "dataset-a"
    root_job_id = "migration-job-fence"
    db.note_store.add_note("Title", "Body", note_id=note_id)
    config = db.note_semantic_store.create_configuration(
        dataset_id=dataset_id,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="openai",
        model="embedding-model-v1",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local semantic vectors",
        normalization_version="notes-semantic-normalization-v1",
        chunker_version="notes-semantic-chunker-v1",
        now=_NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=dataset_id,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=_NOW,
    )
    assert enabled is not None
    generation = db.note_semantic_store.create_generation(
        dataset_id=dataset_id,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id=root_job_id,
        now=_NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=dataset_id,
        generation_id=generation.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=2,
        compatibility_hash="compatibility-v1",
        now=_NOW,
    )
    assert resolved is not None
    chunks = build_semantic_chunks(
        generation_id=generation.id,
        note_id=note_id,
        title="Title",
        content="Body",
        content_version=1,
    )
    assert db.note_semantic_store.seed_generation_snapshot(
        dataset_id=dataset_id,
        generation_id=generation.id,
        expected_configuration_revision=resolved.configuration_revision,
        generation_fencing_token=root_job_id,
        seeds=(
            SemanticSnapshotSeed(
                note_id=note_id,
                content_version=1,
                content_fingerprint=semantic_content_fingerprint("Title", "Body", 1),
                state="pending",
                planned_chunk_count=len(chunks),
                error_code=None,
            ),
        ),
        now=_NOW,
    )
    work = db.note_semantic_store.claim_work_batch(
        dataset_id=dataset_id,
        generation_id=generation.id,
        kind="index_note",
        limit=1,
        now=_NOW,
    )[0]
    publication = db.note_semantic_store.publish_indexed_manifest(
        owner_user_id="owner-a",
        dataset_id=dataset_id,
        generation_id=generation.id,
        generation_fencing_token=root_job_id,
        expected_configuration_revision=resolved.configuration_revision,
        work_id=work.id,
        claim_token=work.claim_token or "",
        work_fencing_token=work.fencing_token,
        claimed_dirty_generation=work.dirty_generation or 0,
        content_version=1,
        content_fingerprint=chunks[0].content_fingerprint,
        chunks=tuple(
            SemanticChunkRecord(
                chunk_id=chunk.vector_id,
                generation_id=chunk.generation_id,
                note_id=chunk.note_id,
                content_version=chunk.content_version,
                ordinal=chunk.ordinal,
                field=chunk.field,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                chunk_fingerprint=chunk.chunk_fingerprint,
                normalization_version=chunk.normalization_version,
                chunker_version=chunk.chunker_version,
            )
            for chunk in chunks
        ),
        now=_NOW,
    )
    assert publication is not None
    return dataset_id, generation.id, note_id, work.id


def test_postgres_v66_ddl_is_owner_scoped_forced_rls_and_has_no_cascading_authority() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V65_TO_V66_POSTGRES.split())

    assert f"CREATE TABLE IF NOT EXISTS {_TABLE}" in sql
    assert f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY" in sql
    assert f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY" in sql
    assert f"CREATE POLICY {_TABLE}_tenant_isolation" in sql
    assert "idx_note_semantic_obsolete_vectors_claimable" in sql
    assert "idx_note_semantic_obsolete_vectors_generation" in sql
    assert "REFERENCES notes" not in sql
    assert "REFERENCES note_semantic_generations" not in sql


def test_postgres_initializer_routes_schema_v65_through_v66(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _fake_postgres_initializer(
        monkeypatch,
        current_version=65,
        target_version=66,
    )

    def _reached_v66(_conn: object) -> None:
        raise RuntimeError("reached-v66")

    monkeypatch.setattr(db, "_migrate_from_v65_to_v66_postgres", _reached_v66)

    with pytest.raises(RuntimeError, match="^reached-v66$"):
        db._initialize_schema_postgres()


def test_postgres_initializer_honors_target_v65_without_dispatching_v66(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _fake_postgres_initializer(
        monkeypatch,
        current_version=65,
        target_version=65,
    )

    def _unexpected_v66(_conn: object) -> None:
        pytest.fail("v66 migration dispatched for a v65 target")

    def _stayed_at_v65(_conn: object) -> None:
        raise RuntimeError("stayed-v65")

    monkeypatch.setattr(db, "_migrate_from_v65_to_v66_postgres", _unexpected_v66)
    monkeypatch.setattr(db, "_ensure_flashcard_asset_schema_postgres", _stayed_at_v65)

    with pytest.raises(RuntimeError, match="^stayed-v65$"):
        db._initialize_schema_postgres()


def test_postgres_v66_live_schema_has_cleanup_constraints_indexes_and_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        relation = backend.execute(
            """
            SELECT relrowsecurity,relforcerowsecurity
              FROM pg_class
             WHERE oid = to_regclass(%s)
            """,
            (_TABLE,),
        ).rows[0]
        columns = {
            str(row["column_name"])
            for row in backend.execute(
                """
                SELECT column_name FROM information_schema.columns
                 WHERE table_schema=current_schema() AND table_name=%s
                """,
                (_TABLE,),
            ).rows
        }
        indexes = {
            str(row["indexname"])
            for row in backend.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() AND tablename=%s",
                (_TABLE,),
            ).rows
        }
        foreign_keys = backend.execute(
            """
            SELECT confrelid::regclass::text AS referenced_table
              FROM pg_constraint
             WHERE conrelid=to_regclass(%s) AND contype='f'
            """,
            (_TABLE,),
        ).rows

        assert int(version) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
        assert relation == {"relrowsecurity": True, "relforcerowsecurity": True}
        assert {"owner_user_id", "dataset_id", "generation_id", "vector_id"} <= columns
        assert {
            "idx_note_semantic_obsolete_vectors_claimable",
            "idx_note_semantic_obsolete_vectors_generation",
        } <= indexes
        assert foreign_keys == []
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_postgres_v65_to_v66_upgrade_preserves_all_semantic_rows(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    db_after: CharactersRAGDB | None = None
    try:
        dataset_id, generation_id, note_id, work_id = _seed_complete_semantic_generation(db)
        with backend.transaction() as conn:
            backend.execute(f"DROP TABLE {_TABLE}", connection=conn)  # nosec B608
            backend.execute(
                "UPDATE db_schema_version SET version=65 WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            )
        db.close_connection()

        db_after = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
        with backend.transaction() as conn:
            backend.execute(
                "SELECT set_config('app.current_dataset_id', %s, true)",
                (dataset_id,),
                connection=conn,
            )
            preserved = {
                table: int(
                    backend.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE owner_user_id=%s "  # nosec B608
                        "AND dataset_id=%s",
                        ("owner-a", dataset_id),
                        connection=conn,
                    ).scalar
                )
                for table in (
                    "note_semantic_index_configs",
                    "note_semantic_generations",
                    "note_semantic_note_state",
                    "note_semantic_chunks",
                    "note_semantic_work",
                )
            }
            assert int(
                backend.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name=%s",
                    (CharactersRAGDB._SCHEMA_NAME,),
                    connection=conn,
                ).scalar
            ) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
            assert preserved == {
                "note_semantic_index_configs": 1,
                "note_semantic_generations": 1,
                "note_semantic_note_state": 1,
                "note_semantic_chunks": 1,
                "note_semantic_work": 1,
            }
            assert backend.execute(
                "SELECT id FROM note_semantic_generations WHERE id=%s",
                (generation_id,),
                connection=conn,
            ).scalar == generation_id
            assert backend.execute(
                "SELECT note_id FROM note_semantic_note_state WHERE note_id=%s",
                (note_id,),
                connection=conn,
            ).scalar == note_id
            assert backend.execute(
                "SELECT id FROM note_semantic_work WHERE id=%s",
                (work_id,),
                connection=conn,
            ).scalar == work_id
    finally:
        if db_after is not None:
            db_after.close_all_connections()
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v66_migration_failure_rolls_back_table_and_version(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            backend.execute(f"DROP TABLE {_TABLE}", connection=conn)  # nosec B608
            backend.execute(
                "UPDATE db_schema_version SET version=65 WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            )
        monkeypatch.setattr(
            db,
            "_convert_sqlite_schema_to_postgres_statements",
            lambda _script: [
                f"CREATE TABLE {_TABLE}(id TEXT PRIMARY KEY)",
                "SELECT value FROM missing_v66_relation",
            ],
        )

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                db._migrate_from_v65_to_v66_postgres(conn)

        assert int(
            backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).scalar
        ) == 65
        assert backend.execute("SELECT to_regclass(%s)", (_TABLE,)).scalar is None
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v66_ledger_enforces_exact_tenant_scope_and_fails_closed(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    db_b = CharactersRAGDB(":memory:", client_id="owner-b", backend=backend)
    role_name = f"semantic_cleanup_rls_{uuid4().hex[:12]}"
    ident = backend.escape_identifier
    role_created = False
    try:
        assert db_a.note_semantic_store.stage_obsolete_vector_cleanup(
            dataset_id="dataset-a",
            generation_id="generation-a",
            vector_ids=("vector-a",),
            source_kind="hard_delete",
            note_id=None,
            dirty_generation=None,
            now=_NOW,
        ) == 1
        assert db_b.note_semantic_store.stage_obsolete_vector_cleanup(
            dataset_id="dataset-b",
            generation_id="generation-b",
            vector_ids=("vector-b",),
            source_kind="hard_delete",
            note_id=None,
            dirty_generation=None,
            now=_NOW,
        ) == 1
        with backend.transaction() as conn:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT,INSERT,UPDATE,DELETE ON {_TABLE} TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        def execute_as_scope(
            owner_user_id: str,
            dataset_id: str | None,
            sql: str,
            params: tuple[object, ...] = (),
        ) -> QueryResult:
            with backend.transaction() as conn:
                backend.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend.execute("SET LOCAL row_security=on", connection=conn)
                backend.execute(
                    "SELECT set_config('app.current_user_id', %s, true)",
                    (owner_user_id,),
                    connection=conn,
                )
                if dataset_id is None:
                    backend.execute(
                        "SET LOCAL app.current_dataset_id TO DEFAULT",
                        connection=conn,
                    )
                else:
                    backend.execute(
                        "SELECT set_config('app.current_dataset_id', %s, true)",
                        (dataset_id,),
                        connection=conn,
                    )
                return backend.execute(sql, params, connection=conn)

        select_sql = f"SELECT vector_id FROM {_TABLE} ORDER BY vector_id"  # nosec B608
        assert execute_as_scope("owner-a", "dataset-a", select_sql).rows == [
            {"vector_id": "vector-a"}
        ]

        mismatched_scopes = (
            ("owner-a", "dataset-b", "wrong-dataset"),
            ("owner-b", "dataset-a", "wrong-owner"),
        )
        insert_sql = (
            f"INSERT INTO {_TABLE}(id,owner_user_id,dataset_id,generation_id,"  # nosec B608
            "vector_id,source_kind,claim_state,attempt_count,next_eligible_at,"
            "created_at,updated_at) VALUES (%s,%s,%s,%s,%s,'hard_delete',"
            "'pending',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
        )
        update_sql = (
            f"UPDATE {_TABLE} SET error_code='rls-forged' WHERE vector_id=%s "  # nosec B608
            "RETURNING vector_id"
        )
        delete_sql = (
            f"DELETE FROM {_TABLE} WHERE vector_id=%s RETURNING vector_id"  # nosec B608
        )
        for scoped_owner, scoped_dataset, suffix in mismatched_scopes:
            assert execute_as_scope(scoped_owner, scoped_dataset, select_sql).rows == []
            with pytest.raises(BackendDatabaseError):
                execute_as_scope(
                    scoped_owner,
                    scoped_dataset,
                    insert_sql,
                    (
                        f"forged-cleanup-{suffix}",
                        "owner-a",
                        "dataset-a",
                        "generation-a",
                        f"forged-vector-{suffix}",
                    ),
                )
            assert execute_as_scope(
                scoped_owner,
                scoped_dataset,
                update_sql,
                ("vector-a",),
            ).rows == []
            assert execute_as_scope(
                scoped_owner,
                scoped_dataset,
                delete_sql,
                ("vector-a",),
            ).rows == []

        assert execute_as_scope("owner-a", None, select_sql).rows == []
        assert execute_as_scope(
            "owner-a",
            "dataset-a",
            update_sql,
            ("vector-a",),
        ).rows == [{"vector_id": "vector-a"}]
        assert execute_as_scope(
            "owner-a",
            "dataset-a",
            delete_sql,
            ("vector-a",),
        ).rows == [{"vector_id": "vector-a"}]
        assert execute_as_scope("owner-b", "dataset-b", select_sql).rows == [
            {"vector_id": "vector-b"}
        ]
    finally:
        db_a.close_connection()
        db_b.close_connection()
        if role_created:
            with backend.transaction() as conn:
                backend.execute(
                    f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                    connection=conn,
                )
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                backend.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        backend.get_pool().close_all()
