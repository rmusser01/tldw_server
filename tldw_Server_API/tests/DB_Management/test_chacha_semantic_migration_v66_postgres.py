"""PostgreSQL schema-v87 contracts for durable semantic cleanup authority."""

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
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

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

    def execute(self, *_args: object, **_kwargs: object) -> None:
        return None

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
    for name in vars(CharactersRAGDB):
        if name.startswith("_ensure_"):
            monkeypatch.setattr(db, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(db, "_postgres_schema_is_current", lambda _conn: False)
    monkeypatch.setattr(db, "_repair_conversation_assistant_identity", lambda _conn: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.chacha.schema_bootstrap.postgres_schema_migration",
        lambda backend, _timeout: backend.transaction(),
    )

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", target_version)
    monkeypatch.setattr(
        db,
        "_get_schema_version_postgres",
        lambda _conn, lock=False: current_version,
    )
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: True)
    monkeypatch.setattr(db, "_postgres_v61_progress_row", lambda _conn, _phase: {"status": "complete"})
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_semantic_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )
    return db


def _seed_complete_semantic_generation(db: CharactersRAGDB) -> tuple[str, str, str, str]:
    """Seed the genuine v86 columns without invoking a newer store contract."""
    note_id = "11111111-1111-4111-8111-111111111111"
    dataset_id, generation_id, work_id = "dataset-a", "generation-a", "work-a"
    digest = "sha256:" + "a" * 64
    db.note_store.add_note("Title", "Body", note_id=note_id)
    with db.backend.transaction() as conn:
        db.backend.execute("SELECT set_config('app.current_user_id', %s, true)", ("owner-a",), connection=conn)
        db.backend.execute("SELECT set_config('app.current_dataset_id', %s, true)", (dataset_id,), connection=conn)
        db.backend.execute(
            """INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,active_generation_id,configuration_revision,
                semantic_index_revision,metric,dimension_state,dimensions,compatibility_hash,
                normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a',%s,'enabled',%s,1,1,'cosine','resolved',2,
                      'compatibility-v1','v1','v1',%s)""",
            (dataset_id, generation_id, _NOW), connection=conn,
        )
        db.backend.execute(
            """INSERT INTO note_semantic_generations(
                id,owner_user_id,dataset_id,configuration_revision,state,compatibility_hash,
                dimension_state,dimensions,expected_note_count,expected_chunk_count,
                published_note_count,published_chunk_count,created_at
            ) VALUES (%s,'owner-a',%s,1,'active','compatibility-v1','resolved',2,1,1,1,1,%s)""",
            (generation_id, dataset_id, _NOW), connection=conn,
        )
        db.backend.execute(
            """INSERT INTO note_semantic_note_state(
                owner_user_id,dataset_id,generation_id,note_id,content_version,
                content_fingerprint,dirty_generation,state,chunk_count
            ) VALUES ('owner-a',%s,%s,%s,1,%s,1,'indexed',1)""",
            (dataset_id, generation_id, note_id, digest), connection=conn,
        )
        db.backend.execute(
            """INSERT INTO note_semantic_chunks(
                chunk_id,owner_user_id,dataset_id,generation_id,note_id,content_version,
                ordinal,field,start_offset,end_offset,chunk_fingerprint,
                normalization_version,chunker_version
            ) VALUES ('chunk-a','owner-a',%s,%s,%s,1,0,'content',0,4,%s,'v1','v1')""",
            (dataset_id, generation_id, note_id, digest), connection=conn,
        )
        db.backend.execute(
            """INSERT INTO note_semantic_work(
                id,owner_user_id,dataset_id,kind,note_id,generation_id,dirty_generation,
                fencing_token,claim_state,attempt_count,next_eligible_at,created_at,updated_at
            ) VALUES (%s,'owner-a',%s,'index_note',%s,%s,1,'migration-job-fence',
                      'completed',1,%s,%s,%s)""",
            (work_id, dataset_id, note_id, generation_id, _NOW, _NOW, _NOW), connection=conn,
        )
    return dataset_id, generation_id, note_id, work_id


def test_postgres_v87_ddl_is_owner_scoped_forced_rls_and_has_no_cascading_authority() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V86_TO_V87_POSTGRES.split())

    assert f"CREATE TABLE IF NOT EXISTS {_TABLE}" in sql
    assert f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY" in sql
    assert f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY" in sql
    assert f"CREATE POLICY {_TABLE}_tenant_isolation" in sql
    assert "idx_note_semantic_obsolete_vectors_claimable" in sql
    assert "idx_note_semantic_obsolete_vectors_generation" in sql
    assert "REFERENCES notes" not in sql
    assert "REFERENCES note_semantic_generations" not in sql


def test_postgres_initializer_routes_schema_v86_through_v87(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _fake_postgres_initializer(
        monkeypatch,
        current_version=86,
        target_version=87,
    )

    def _reached_v87(_conn: object) -> None:
        raise RuntimeError("reached-v87")

    monkeypatch.setattr(db, "_migrate_from_v86_to_v87_postgres", _reached_v87)

    with pytest.raises(RuntimeError, match="^reached-v87$"):
        db._initialize_schema_postgres()


def test_postgres_initializer_honors_target_v86_without_dispatching_v87(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _fake_postgres_initializer(
        monkeypatch,
        current_version=86,
        target_version=86,
    )

    def _unexpected_v87(_conn: object) -> None:
        pytest.fail("v87 migration dispatched for a v86 target")

    monkeypatch.setattr(db, "_migrate_from_v86_to_v87_postgres", _unexpected_v87)

    db._initialize_schema_postgres()


def test_postgres_v87_live_schema_has_cleanup_constraints_indexes_and_forced_rls(
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


def test_postgres_v86_to_v87_upgrade_preserves_all_semantic_rows(
    pg_database_config: DatabaseConfig,
) -> None:
    class LegacyV86(CharactersRAGDB):
        _POSTGRES_SCHEMA_VERSION = 86

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = LegacyV86(":memory:", client_id="owner-a", backend=backend)
    db_after: CharactersRAGDB | None = None
    try:
        dataset_id, generation_id, note_id, work_id = _seed_complete_semantic_generation(db)
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


def test_postgres_v87_migration_failure_rolls_back_table_and_version(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            backend.execute(f"DROP TABLE {_TABLE}", connection=conn)  # nosec B608
            backend.execute(
                "UPDATE db_schema_version SET version=86 WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            )
        monkeypatch.setattr(
            db,
            "_convert_sqlite_schema_to_postgres_statements",
            lambda _script: [
                f"CREATE TABLE {_TABLE}(id TEXT PRIMARY KEY)",
                "SELECT value FROM missing_v87_relation",
            ],
        )

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                db._migrate_from_v86_to_v87_postgres(conn)

        assert int(
            backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).scalar
        ) == 86
        assert backend.execute("SELECT to_regclass(%s)", (_TABLE,)).scalar is None
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v87_ledger_enforces_exact_tenant_scope_and_fails_closed(
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
