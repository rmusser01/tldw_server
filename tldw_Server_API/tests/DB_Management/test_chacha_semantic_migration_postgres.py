"""PostgreSQL schema-v65 contracts for Notes semantic-index persistence."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)

pytestmark = pytest.mark.unit


_TABLES = (
    "note_semantic_index_configs",
    "note_semantic_generations",
    "note_semantic_note_state",
    "note_semantic_chunks",
    "note_semantic_work",
)
_DIGEST = f"sha256:{'a' * 64}"
_NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


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


def _set_tenant_scope(backend: object, conn: object) -> None:
    backend.execute(  # type: ignore[attr-defined]
        "SELECT set_config('app.current_user_id', %s, true)",
        ("owner-a",),
        connection=conn,
    )
    backend.execute(  # type: ignore[attr-defined]
        "SELECT set_config('app.current_dataset_id', %s, true)",
        ("dataset-a",),
        connection=conn,
    )


def _prepare_live_v64(pg_database_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            for table in reversed(_TABLES):
                backend.execute(f"DROP TABLE {table} CASCADE", connection=conn)  # nosec B608
            backend.execute(
                "UPDATE db_schema_version SET version = 64 WHERE schema_name = %s",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            )
    finally:
        db.close_all_connections()


def _create_store_configuration(db: CharactersRAGDB):
    return db.note_semantic_store.create_configuration(
        dataset_id="dataset-a",
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local semantic vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=_NOW,
    )


def _create_resolved_store_generation(db: CharactersRAGDB):
    config = _create_store_configuration(db)
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id="dataset-a",
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=_NOW,
    )
    assert enabled is not None
    pending = db.note_semantic_store.create_generation(
        dataset_id="dataset-a",
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="job-probe",
        now=_NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id="dataset-a",
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=768,
        compatibility_hash="compatibility-v1",
        now=_NOW,
    )
    assert resolved is not None
    resolved_config = db.note_semantic_store.get_configuration("dataset-a")
    assert resolved_config is not None
    return resolved_config, resolved


def test_postgres_v65_ddl_has_owner_keys_forced_rls_and_no_dimension_tables() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V64_TO_V65_POSTGRES.split())

    for table in _TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in sql
        assert (
            f"CREATE POLICY {table}_tenant_isolation ON {table} USING "
            "(owner_user_id = current_setting('app.current_user_id', true) "
            "AND dataset_id = current_setting('app.current_dataset_id', true))"
        ) in sql

    assert "idx_note_semantic_generations_one_active" in sql
    assert "idx_note_semantic_generations_one_staging" in sql
    assert "idx_note_semantic_work_claimable" in sql
    assert "vector(" not in sql.lower()
    assert "note_semantic_vectors_" not in sql


def test_postgres_initializer_routes_schema_v64_through_v65(monkeypatch: pytest.MonkeyPatch) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 65)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, lock=False: 64)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )

    def _reached_v65(_conn: object) -> None:
        raise RuntimeError("reached-v65")

    monkeypatch.setattr(db, "_migrate_from_v64_to_v65_postgres", _reached_v65)

    with pytest.raises(RuntimeError, match="^reached-v65$"):
        db._initialize_schema_postgres()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v65_live_schema_has_forced_owner_dataset_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    _prepare_live_v64(pg_database_config)
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        columns = backend.execute(
            """
            SELECT table_name, column_name
              FROM information_schema.columns
             WHERE table_schema = current_schema()
               AND table_name = ANY(%s)
               AND column_name IN ('owner_user_id', 'dataset_id')
            """,
            (list(_TABLES),),
        ).rows
        relations = backend.execute(
            """
            SELECT relname, relrowsecurity, relforcerowsecurity
              FROM pg_class AS relation
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relname = ANY(%s)
            """,
            (list(_TABLES),),
        ).rows
        policies = backend.execute(
            """
            SELECT tablename, qual, with_check
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND tablename = ANY(%s)
            """,
            (list(_TABLES),),
        ).rows
        vector_tables = backend.execute(
            """
            SELECT tablename FROM pg_tables
             WHERE schemaname = current_schema()
               AND tablename LIKE 'note_semantic_vectors_%'
            """,
        ).rows
        constraints = backend.execute(
            """
            SELECT relation.relname AS table_name,
                   pg_get_constraintdef(constraint_row.oid) AS definition
              FROM pg_constraint AS constraint_row
              JOIN pg_class AS relation ON relation.oid = constraint_row.conrelid
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname = ANY(%s)
            """,
            (list(_TABLES),),
        ).rows

        assert int(version) == 65
        assert {(str(row["table_name"]), str(row["column_name"])) for row in columns} >= {
            (table, column) for table in _TABLES for column in ("owner_user_id", "dataset_id")
        }
        assert {
            (str(row["relname"]), bool(row["relrowsecurity"]), bool(row["relforcerowsecurity"]))
            for row in relations
        } >= {(table, True, True) for table in _TABLES}
        assert len(policies) == len(_TABLES)
        for row in policies:
            predicate = f"{row['qual']} {row['with_check']}"
            assert "owner_user_id" in predicate
            assert "dataset_id" in predicate
            assert "app.current_user_id" in predicate
            assert "app.current_dataset_id" in predicate
        definitions = " ".join(str(row["definition"]) for row in constraints)
        assert "content_fingerprint" in definitions
        assert "chunk_fingerprint" in definitions
        assert "generation_id IS NOT NULL" in definitions
        assert vector_tables == []
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v65_live_dimension_identity_constraints_preserve_disabled_states(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn)
            backend.execute(
                """
                INSERT INTO note_semantic_index_configs(
                    owner_user_id,dataset_id,desired_state,configuration_revision,
                    semantic_index_revision,metric,dimension_state,dimensions,
                    compatibility_hash,normalization_version,chunker_version,updated_at
                ) VALUES ('owner-a','dataset-a','disabled',1,0,'cosine','pending',
                          NULL,NULL,'v1','v1',CURRENT_TIMESTAMP)
                """,
                connection=conn,
            )

        for dimension_state, dimensions, compatibility_hash in (
            ("pending", 768, None),
            ("pending", None, "compatibility-v1"),
            ("resolved", None, "compatibility-v1"),
            ("resolved", 768, None),
            ("resolved", 768, ""),
        ):
            with pytest.raises(BackendDatabaseError):
                with backend.transaction() as conn:
                    _set_tenant_scope(backend, conn)
                    backend.execute(
                        "UPDATE note_semantic_index_configs SET dimension_state=%s, dimensions=%s, "
                        "compatibility_hash=%s WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'",
                        (dimension_state, dimensions, compatibility_hash),
                        connection=conn,
                    )

        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn)
            backend.execute(
                "UPDATE note_semantic_index_configs SET dimension_state='resolved', dimensions=768, "
                "compatibility_hash='compatibility-v1' "
                "WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'",
                connection=conn,
            )
            row = backend.execute(
                "SELECT desired_state,dimension_state,dimensions,compatibility_hash "
                "FROM note_semantic_index_configs "
                "WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'",
                connection=conn,
            ).rows[0]
        assert (
            str(row["desired_state"]),
            str(row["dimension_state"]),
            int(row["dimensions"]),
            str(row["compatibility_hash"]),
        ) == ("disabled", "resolved", 768, "compatibility-v1")

        for index, (dimension_state, dimensions, compatibility_hash) in enumerate(
            (
                ("pending", 768, None),
                ("pending", None, "compatibility-v1"),
                ("resolved", None, "compatibility-v1"),
                ("resolved", 768, None),
                ("resolved", 768, ""),
            )
        ):
            with pytest.raises(BackendDatabaseError):
                with backend.transaction() as conn:
                    _set_tenant_scope(backend, conn)
                    backend.execute(
                        """
                        INSERT INTO note_semantic_generations(
                            id,owner_user_id,dataset_id,configuration_revision,state,
                            compatibility_hash,dimension_state,dimensions,created_at
                        ) VALUES (%s,'owner-a','dataset-a',1,'staging',%s,%s,%s,CURRENT_TIMESTAMP)
                        """,
                        (f"generation-{index}", compatibility_hash, dimension_state, dimensions),
                        connection=conn,
                    )
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_semantic_store_generation_creation_matches_configuration_identity(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        config = _create_store_configuration(db)
        enabled = db.note_semantic_store.enable_configuration(
            dataset_id="dataset-a",
            expected_configuration_revision=config.configuration_revision,
            capability_revision="capability-v1",
            now=_NOW,
        )
        assert enabled is not None

        with pytest.raises(ValueError, match="notes_semantic_generation_identity_mismatch"):
            db.note_semantic_store.create_generation(
                dataset_id="dataset-a",
                configuration_revision=enabled.configuration_revision,
                compatibility_hash="compatibility-v1",
                dimension_state=SemanticDimensionState.RESOLVED,
                dimensions=768,
                root_job_id="job-bypass-resolved",
                now=_NOW,
            )

        pending = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id="job-pending",
            now=_NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=pending.id,
            expected_configuration_revision=enabled.configuration_revision,
            dimensions=768,
            compatibility_hash="compatibility-v1",
            now=_NOW,
        )
        assert resolved is not None
        resolved_config = db.note_semantic_store.get_configuration("dataset-a")
        assert resolved_config is not None

        with db.transaction() as conn:
            conn.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                ("dataset-a",),
            )
            conn.execute(
                "UPDATE note_semantic_generations SET state='failed' "
                "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                ("owner-a", "dataset-a", resolved.id),
            )

        with pytest.raises(ValueError, match="notes_semantic_generation_identity_mismatch"):
            db.note_semantic_store.create_generation(
                dataset_id="dataset-a",
                configuration_revision=resolved_config.configuration_revision,
                compatibility_hash=None,
                dimension_state=SemanticDimensionState.PENDING,
                dimensions=None,
                root_job_id="job-bypass-pending",
                now=_NOW,
            )

        matched = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=resolved_config.configuration_revision,
            compatibility_hash="compatibility-v1",
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=768,
            root_job_id="job-resolved",
            now=_NOW,
        )
        assert matched.state is SemanticGenerationState.STAGING
        assert matched.dimensions == 768
        assert matched.compatibility_hash == "compatibility-v1"
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("dimensions", "compatibility_hash"),
    ((384, "compatibility-v1"), (768, "compatibility-v2")),
)
def test_postgres_semantic_store_activation_identity_mismatch_has_no_side_effects(
    pg_database_config: DatabaseConfig,
    dimensions: int,
    compatibility_hash: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        resolved_config, first = _create_resolved_store_generation(db)
        active = db.note_semantic_store.activate_generation(
            dataset_id="dataset-a",
            generation_id=first.id,
            expected_configuration_revision=resolved_config.configuration_revision,
            publication_receipt="receipt-1",
            now=_NOW,
        )
        assert active is not None
        replacement = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=active.configuration_revision,
            compatibility_hash="compatibility-v1",
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=768,
            root_job_id="job-replacement",
            now=_NOW,
        )
        with db.transaction() as conn:
            conn.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                ("dataset-a",),
            )
            conn.execute(
                "UPDATE note_semantic_generations SET dimensions=?, compatibility_hash=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (dimensions, compatibility_hash, "owner-a", "dataset-a", replacement.id),
            )

        assert db.note_semantic_store.activate_generation(
            dataset_id="dataset-a",
            generation_id=replacement.id,
            expected_configuration_revision=active.configuration_revision,
            publication_receipt="receipt-2",
            now=_NOW,
        ) is None
        unchanged = db.note_semantic_store.get_configuration("dataset-a")
        assert unchanged is not None
        assert unchanged.configuration_revision == active.configuration_revision
        assert unchanged.semantic_index_revision == active.semantic_index_revision
        assert unchanged.active_generation_id == first.id
        assert (
            db.note_semantic_store.get_generation("dataset-a", first.id).state
            is SemanticGenerationState.ACTIVE
        )
        assert (
            db.note_semantic_store.get_generation("dataset-a", replacement.id).state
            is SemanticGenerationState.STAGING
        )
        with db.transaction() as conn:
            conn.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                ("dataset-a",),
            )
            cleanup_count = conn.execute(
                "SELECT COUNT(*) AS total FROM note_semantic_work "
                "WHERE owner_user_id=? AND dataset_id=? AND kind='delete_generation'",
                ("owner-a", "dataset-a"),
            ).fetchone()["total"]
        assert int(cleanup_count) == 0
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_create_generation_fences_concurrent_configuration_revision(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    resolved_config, old_generation = _create_resolved_store_generation(db)
    with db.transaction() as conn:
        conn.execute(
            "SELECT set_config('app.current_dataset_id', ?, true)",
            ("dataset-a",),
        )
        conn.execute(
            "UPDATE note_semantic_generations SET state='failed' "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            ("owner-a", "dataset-a", old_generation.id),
        )

    generation_inserted = threading.Event()
    release_generation_commit = threading.Event()
    configuration_update_started = threading.Event()
    configuration_update_finished = threading.Event()
    results: dict[str, object] = {}

    def create_generation() -> None:
        try:
            with db.transaction():
                results["generation"] = db.note_semantic_store.create_generation(
                    dataset_id="dataset-a",
                    configuration_revision=resolved_config.configuration_revision,
                    compatibility_hash="compatibility-v1",
                    dimension_state=SemanticDimensionState.RESOLVED,
                    dimensions=768,
                    root_job_id="job-racing-create",
                    now=_NOW,
                )
                generation_inserted.set()
                assert release_generation_commit.wait(timeout=10)
        except Exception as exc:  # noqa: BLE001 - surface thread failures in the test
            results["generation_error"] = exc
            generation_inserted.set()
        finally:
            db.close_connection()

    def update_configuration() -> None:
        try:
            with db.transaction():
                configuration_update_started.set()
                results["disabled"] = db.note_semantic_store.disable_configuration(
                    dataset_id="dataset-a",
                    expected_configuration_revision=resolved_config.configuration_revision,
                    now=_NOW,
                )
        except Exception as exc:  # noqa: BLE001 - surface thread failures in the test
            results["configuration_error"] = exc
        finally:
            configuration_update_finished.set()
            db.close_connection()

    generation_thread = threading.Thread(target=create_generation)
    configuration_thread = threading.Thread(target=update_configuration)
    try:
        generation_thread.start()
        assert generation_inserted.wait(timeout=10)
        assert "generation_error" not in results
        configuration_thread.start()
        assert configuration_update_started.wait(timeout=10)
        update_was_serialized = not configuration_update_finished.wait(timeout=1)
    finally:
        release_generation_commit.set()
        generation_thread.join(timeout=10)
        if configuration_thread.ident is not None:
            configuration_thread.join(timeout=10)

    try:
        assert update_was_serialized
        assert not generation_thread.is_alive()
        assert not configuration_thread.is_alive()
        assert "generation_error" not in results
        assert "configuration_error" not in results
        generation = results["generation"]
        disabled = results["disabled"]
        assert generation.configuration_revision == resolved_config.configuration_revision
        assert disabled.configuration_revision == resolved_config.configuration_revision + 1

        reenabled = db.note_semantic_store.enable_configuration(
            dataset_id="dataset-a",
            expected_configuration_revision=disabled.configuration_revision,
            capability_revision="capability-v1",
            now=_NOW,
        )
        assert reenabled is not None
        assert db.note_semantic_store.activate_generation(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=reenabled.configuration_revision,
            publication_receipt="receipt-stale",
            now=_NOW,
        ) is None
        assert (
            db.note_semantic_store.get_generation("dataset-a", generation.id).state
            is SemanticGenerationState.STAGING
        )

        with db.transaction() as conn:
            conn.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                ("dataset-a",),
            )
            failed = conn.execute(
                "UPDATE note_semantic_generations SET state='failed' "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='staging'",
                ("owner-a", "dataset-a", generation.id),
            )
        assert failed.rowcount == 1
        replacement = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=reenabled.configuration_revision,
            compatibility_hash="compatibility-v1",
            dimension_state=SemanticDimensionState.RESOLVED,
            dimensions=768,
            root_job_id="job-after-stale-cleanup",
            now=_NOW,
        )
        assert replacement.state is SemanticGenerationState.STAGING
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v65_live_constraints_reject_raw_fingerprints_and_unbound_work(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn)
            backend.execute(
                "INSERT INTO notes(id,title,content,client_id) "
                "VALUES ('note-a','title','body','owner-a')",
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_semantic_index_configs(
                    owner_user_id,dataset_id,desired_state,configuration_revision,
                    semantic_index_revision,metric,dimension_state,dimensions,
                    compatibility_hash,normalization_version,chunker_version,updated_at
                ) VALUES ('owner-a','dataset-a','enabled',1,0,'cosine','resolved',
                          768,'compatibility-v1','v1','v1',CURRENT_TIMESTAMP)
                """,
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_semantic_generations(
                    id,owner_user_id,dataset_id,configuration_revision,state,
                    compatibility_hash,dimension_state,dimensions,created_at
                ) VALUES ('generation-a','owner-a','dataset-a',1,'staging',
                          'compatibility-v1','resolved',768,CURRENT_TIMESTAMP)
                """,
                connection=conn,
            )

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                _set_tenant_scope(backend, conn)
                backend.execute(
                    """
                    INSERT INTO note_semantic_note_state(
                        owner_user_id,dataset_id,generation_id,note_id,content_version,
                        content_fingerprint,dirty_generation,state
                    ) VALUES ('owner-a','dataset-a','generation-a','note-a',1,
                              'raw Note body',1,'pending')
                    """,
                    connection=conn,
                )

        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn)
            backend.execute(
                """
                INSERT INTO note_semantic_note_state(
                    owner_user_id,dataset_id,generation_id,note_id,content_version,
                    content_fingerprint,dirty_generation,state
                ) VALUES ('owner-a','dataset-a','generation-a','note-a',1,%s,1,'pending')
                """,
                (_DIGEST,),
                connection=conn,
            )

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                _set_tenant_scope(backend, conn)
                backend.execute(
                    """
                    INSERT INTO note_semantic_chunks(
                        chunk_id,owner_user_id,dataset_id,generation_id,note_id,
                        content_version,ordinal,field,start_offset,end_offset,
                        chunk_fingerprint,normalization_version,chunker_version
                    ) VALUES ('chunk-a','owner-a','dataset-a','generation-a','note-a',
                              1,0,'content',0,5,'raw Note body','v1','v1')
                    """,
                    connection=conn,
                )

        for index, (kind, note_id, dirty_generation) in enumerate(
            (
                ("index_note", "note-a", 1),
                ("delete_note_vectors", "note-a", 1),
                ("delete_generation", None, None),
            )
        ):
            with pytest.raises(BackendDatabaseError):
                with backend.transaction() as conn:
                    _set_tenant_scope(backend, conn)
                    backend.execute(
                        """
                        INSERT INTO note_semantic_work(
                            id,owner_user_id,dataset_id,kind,note_id,generation_id,
                            dirty_generation,fencing_token,claim_state,attempt_count,
                            next_eligible_at,created_at,updated_at
                        ) VALUES (%s,'owner-a','dataset-a',%s,%s,NULL,%s,'fence',
                                  'pending',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                        """,
                        (f"work-{index}", kind, note_id, dirty_generation),
                        connection=conn,
                    )
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v65_live_migration_failure_rolls_back_to_v64(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_live_v64(pg_database_config)
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    monkeypatch.setattr(
        CharactersRAGDB,
        "_MIGRATION_SQL_V64_TO_V65_POSTGRES",
        CharactersRAGDB._MIGRATION_SQL_V64_TO_V65_POSTGRES
        + "\nTHIS IS AN INJECTED MIGRATION FAILURE;",
    )

    try:
        with pytest.raises(CharactersRAGDBError, match="Unexpected database initialization error"):
            CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)

        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        tables = backend.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=current_schema() "
            "AND tablename=ANY(%s)",
            (list(_TABLES),),
        ).rows
        assert int(version) == 64
        assert tables == []
    finally:
        backend.get_pool().close_all()
