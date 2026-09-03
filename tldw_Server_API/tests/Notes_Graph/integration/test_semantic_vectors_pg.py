"""Live PostgreSQL contracts for Notes pgvector semantic storage."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import SemanticDimensionState
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorCapabilityError,
    create_semantic_vector_store,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors_pg import (
    PGVECTOR_TABLES,
    SEMANTIC_VECTOR_METRIC_LABELS,
    PostgresSemanticVectorBackend,
)
from tldw_Server_API.tests.Notes_Graph.vector_contract import (
    assert_vector_isolation_contract,
    assert_vector_lifecycle_contract,
    assert_vector_validation_contract,
    axis_vector,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]

DIMENSIONS = 384
DATASET_ID = "dataset-a"
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
_PGVECTOR_REQUIRED_ENV = "TLDW_TEST_PGVECTOR_REQUIRED"


def _skip_or_fail_unavailable_pgvector(exc: SemanticVectorCapabilityError) -> None:
    assert exc.code in {
        "notes_semantic_pgvector_extension_unavailable",
        "notes_semantic_pgvector_extension_version_unsupported",
    }
    if os.getenv(_PGVECTOR_REQUIRED_ENV) == "1":
        pytest.fail(f"required pgvector capability unavailable: {exc.code}")
    pytest.skip(f"pgvector capability unavailable: {exc.code}")


def _generation(db: CharactersRAGDB, dataset_id: str = DATASET_ID) -> str:
    config = db.note_semantic_store.create_configuration(
        dataset_id=dataset_id,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="pgvector",
        storage_boundary="server_local",
        storage_label="pgvector",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id=dataset_id,
        expected_configuration_revision=config.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    pending = db.note_semantic_store.create_generation(
        dataset_id=dataset_id,
        configuration_revision=enabled.configuration_revision,
        compatibility_hash=None,
        dimension_state=SemanticDimensionState.PENDING,
        dimensions=None,
        root_job_id="job-vector",
        now=NOW,
    )
    resolved = db.note_semantic_store.resolve_generation_dimensions(
        dataset_id=dataset_id,
        generation_id=pending.id,
        expected_configuration_revision=enabled.configuration_revision,
        dimensions=DIMENSIONS,
        compatibility_hash="compatibility-v1",
        now=NOW,
    )
    assert resolved is not None
    return pending.id


def _additional_generation(db: CharactersRAGDB, dataset_id: str) -> str:
    config = db.note_semantic_store.get_configuration(dataset_id)
    assert config is not None
    assert config.active_generation_id is None
    with db.transaction() as connection:
        if db.note_semantic_store.is_postgres:
            connection.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (dataset_id,),
            )
        row = connection.execute(
            "SELECT id FROM note_semantic_generations WHERE owner_user_id=? "
            "AND dataset_id=? AND state='staging'",
            (db.note_semantic_store.owner_user_id, dataset_id),
        ).fetchone()
    assert row is not None
    current = db.note_semantic_store.get_generation(dataset_id, row["id"])
    assert current is not None
    activated = db.note_semantic_store.activate_generation(
        dataset_id=dataset_id,
        generation_id=current.id,
        expected_configuration_revision=current.configuration_revision,
        publication_receipt="receipt-isolation",
        now=NOW,
    )
    assert activated is not None
    generation = db.note_semantic_store.create_generation(
        dataset_id=dataset_id,
        configuration_revision=activated.configuration_revision,
        compatibility_hash="compatibility-v1",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=DIMENSIONS,
        root_job_id="job-vector-additional",
        now=NOW,
    )
    return generation.id


@pytest.mark.asyncio
async def test_pgvector_schema_and_reusable_contract(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    generation_id = _generation(db)
    other_owner_db = None
    role_name = f"notes_semantic_vectors_{uuid4().hex[:8]}"
    role_created = False
    settings = SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({DIMENSIONS}))
    try:
        try:
            store = await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
        except SemanticVectorCapabilityError as exc:
            _skip_or_fail_unavailable_pgvector(exc)

        other_generation = _additional_generation(db, DATASET_ID)
        other_dataset_generation = _generation(db, "dataset-b")
        other_owner_db = CharactersRAGDB(
            ":memory:",
            client_id="owner-b",
            backend=backend,
        )
        other_owner_generation = _generation(other_owner_db)
        other_owner_store = await create_semantic_vector_store(
            "pgvector",
            authority=other_owner_db.note_semantic_store,
            postgres_backend=backend,
            settings=settings,
        )
        await assert_vector_isolation_contract(
            (store, DATASET_ID, generation_id),
            (
                (store, DATASET_ID, other_generation),
                (store, "dataset-b", other_dataset_generation),
                (other_owner_store, DATASET_ID, other_owner_generation),
            ),
            dimensions=DIMENSIONS,
        )

        await assert_vector_validation_contract(
            store,
            dataset_id=DATASET_ID,
            generation_id=generation_id,
            dimensions=DIMENSIONS,
        )
        await assert_vector_lifecycle_contract(
            store,
            dataset_id=DATASET_ID,
            generation_id=generation_id,
            dimensions=DIMENSIONS,
        )

        table = PGVECTOR_TABLES[DIMENSIONS]
        schema = backend.execute(
            "SELECT n.nspname AS schemaname,c.relname AS tablename,"
            "c.relrowsecurity,c.relforcerowsecurity,"
            "format_type(a.atttypid,a.atttypmod) AS vector_type "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid=c.oid "
            "WHERE n.nspname=current_schema() AND c.relname=? "
            "AND a.attname='embedding'",
            (table,),
        ).one
        assert schema == {
            "schemaname": "public",
            "tablename": table,
            "relrowsecurity": True,
            "relforcerowsecurity": True,
            "vector_type": f"vector({DIMENSIONS})",
        }
        primary_key = backend.execute(
            "SELECT pg_get_constraintdef(pc.oid) AS definition "
            "FROM pg_constraint pc JOIN pg_class c ON c.oid=pc.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=? AND pc.contype='p'",
            (table,),
        ).one["definition"]
        assert primary_key == "PRIMARY KEY (owner_user_id, dataset_id, generation_id, vector_id)"
        index_definitions = backend.execute(
            "SELECT i.schemaname,i.tablename,i.indexname,i.indexdef,"
            "x.indisvalid,x.indisready FROM pg_indexes i "
            "JOIN pg_class ic ON ic.relname=i.indexname "
            "JOIN pg_namespace ni ON ni.oid=ic.relnamespace AND ni.nspname=i.schemaname "
            "JOIN pg_index x ON x.indexrelid=ic.oid "
            "WHERE i.schemaname=current_schema() AND i.tablename=? AND i.indexname=?",
            (table, f"{table}_embedding_hnsw"),
        ).rows
        assert len(index_definitions) == 1
        assert index_definitions[0]["schemaname"] == "public"
        assert index_definitions[0]["tablename"] == table
        assert index_definitions[0]["indexname"] == f"{table}_embedding_hnsw"
        assert "USING hnsw (embedding vector_cosine_ops)" in index_definitions[0]["indexdef"]
        assert index_definitions[0]["indisvalid"] is True
        assert index_definitions[0]["indisready"] is True
        policies = backend.execute(
            "SELECT schemaname,tablename,policyname,permissive,roles,cmd,qual,with_check "
            "FROM pg_policies WHERE schemaname=current_schema() AND tablename=?",
            (table,),
        ).rows
        assert len(policies) == 1
        assert policies[0]["schemaname"] == "public"
        assert policies[0]["tablename"] == table
        assert policies[0]["policyname"] == f"{table}_tenant_isolation"
        assert policies[0]["permissive"] == "PERMISSIVE"
        assert policies[0]["roles"] == ["public"]
        assert policies[0]["cmd"] == "ALL"
        assert "app.current_user_id" in policies[0]["qual"]
        assert "app.current_dataset_id" in policies[0]["qual"]
        assert "app.current_user_id" in policies[0]["with_check"]
        assert "app.current_dataset_id" in policies[0]["with_check"]

        ident = backend.escape_identifier
        policy_name = f"{table}_tenant_isolation"
        with backend.transaction() as connection:
            backend.execute(
                f"DROP POLICY {ident(policy_name)} ON {ident(table)}",  # nosec B608 - allowlisted identifiers.
                connection=connection,
            )
            backend.execute(
                f"CREATE POLICY {ident(policy_name)} ON {ident(table)} "  # nosec B608 - allowlisted identifiers.
                "USING (owner_user_id = current_setting('app.current_user_id', true) "
                "OR dataset_id = current_setting('app.current_dataset_id', true)) "
                "WITH CHECK (owner_user_id = current_setting('app.current_user_id', true) "
                "OR dataset_id = current_setting('app.current_dataset_id', true))",
                connection=connection,
            )
        store = await create_semantic_vector_store(
            "pgvector",
            authority=db.note_semantic_store,
            postgres_backend=backend,
            settings=settings,
        )
        repaired_policy = backend.execute(
            "SELECT qual,with_check FROM pg_policies WHERE tablename=? AND policyname=?",
            (table, policy_name),
        ).one
        assert " OR " not in repaired_policy["qual"]
        assert " AND " in repaired_policy["qual"]
        assert " OR " not in repaired_policy["with_check"]
        assert " AND " in repaired_policy["with_check"]

        await store.create_generation_storage(DATASET_ID, generation_id)
        await store.upsert(
            DATASET_ID,
            generation_id,
            (SemanticVector("rls-vector", axis_vector(DIMENSIONS, 0)),),
        )
        ident = backend.escape_identifier
        with backend.transaction() as connection:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=connection,
            )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                f"GRANT SELECT ON {ident(table)} TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                f"GRANT {ident(role_name)} TO CURRENT_USER",
                connection=connection,
            )
        role_created = True
        with backend.transaction() as connection:
            backend.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=connection)
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                ("owner-a",),
                connection=connection,
            )
            backend.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (DATASET_ID,),
                connection=connection,
            )
            assert backend.execute(
                f"SELECT vector_id FROM {ident(table)}",  # nosec B608 - allowlisted table.
                connection=connection,
            ).rows == [{"vector_id": "rls-vector"}]
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                ("owner-b",),
                connection=connection,
            )
            assert backend.execute(
                f"SELECT vector_id FROM {ident(table)}",  # nosec B608 - allowlisted table.
                connection=connection,
            ).rows == []
        assert backend.table_exists(table)
    finally:
        if role_created:
            ident = backend.escape_identifier
            with backend.transaction() as connection:
                backend.execute(
                    f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                    connection=connection,
                )
                backend.execute(
                    f"DROP OWNED BY {ident(role_name)}",
                    connection=connection,
                )
                backend.execute(
                    f"DROP ROLE {ident(role_name)}",
                    connection=connection,
                )
        if other_owner_db is not None:
            other_owner_db.close_all_connections()
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_pgvector_query_is_bounded_and_never_returns_other_generation_decoys(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    generation_id = _generation(db)
    settings = SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({DIMENSIONS}))
    try:
        try:
            store = await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
        except SemanticVectorCapabilityError as exc:
            _skip_or_fail_unavailable_pgvector(exc)

        decoy_generation = _additional_generation(db, DATASET_ID)
        await store.create_generation_storage(DATASET_ID, generation_id)
        await store.upsert(
            DATASET_ID,
            generation_id,
            (SemanticVector("target", axis_vector(DIMENSIONS, 1)),),
        )
        await store.upsert(
            DATASET_ID,
            decoy_generation,
            tuple(
                SemanticVector(f"decoy-{index:04d}", axis_vector(DIMENSIONS, 0))
                for index in range(256)
            ),
        )

        matches = await store.query(
            DATASET_ID,
            generation_id,
            (axis_vector(DIMENSIONS, 0),),
            limit=1,
        )

        assert len(matches) == 1
        assert len(matches[0]) <= 1
        assert all(match.vector_id == "target" for match in matches[0])

        table = PGVECTOR_TABLES[DIMENSIONS]
        with backend.transaction() as connection:
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                ("owner-a",),
                connection=connection,
            )
            backend.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (DATASET_ID,),
                connection=connection,
            )
            backend.execute(
                "SELECT set_config('hnsw.iterative_scan', ?, true)",
                ("strict_order",),
                connection=connection,
            )
            backend.execute(
                "SELECT set_config('hnsw.max_scan_tuples', ?, true)",
                (str(settings.pgvector_hnsw_max_scan_tuples),),
                connection=connection,
            )
            backend.execute("SET LOCAL enable_seqscan=off", connection=connection)
            backend.execute("SET LOCAL enable_sort=off", connection=connection)
            plan_rows = backend.execute(
                "EXPLAIN (COSTS OFF) SELECT vector_id,distance FROM ("
                f"SELECT vector_id,(embedding <=> ?::vector) AS distance FROM {backend.escape_identifier(table)} "  # nosec B608 - allowlisted table.
                "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? "
                "ORDER BY embedding <=> ?::vector LIMIT ?) AS candidates "
                "ORDER BY distance,vector_id LIMIT ?",
                (
                    "[1" + ",0" * (DIMENSIONS - 1) + "]",
                    "owner-a",
                    DATASET_ID,
                    generation_id,
                    "[1" + ",0" * (DIMENSIONS - 1) + "]",
                    settings.query_candidate_oversampling_factor,
                    1,
                ),
                connection=connection,
            ).rows
        plan = "\n".join(str(row["QUERY PLAN"]) for row in plan_rows)
        assert f"{table}_embedding_hnsw" in plan
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_pgvector_repairs_policy_modes_and_concurrent_factory_calls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    settings = SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({DIMENSIONS}))
    try:
        try:
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
        except SemanticVectorCapabilityError as exc:
            _skip_or_fail_unavailable_pgvector(exc)

        table = PGVECTOR_TABLES[DIMENSIONS]
        policy = f"{table}_tenant_isolation"
        ident = backend.escape_identifier
        predicate = (
            "owner_user_id = current_setting('app.current_user_id', true) "
            "AND dataset_id = current_setting('app.current_dataset_id', true)"
        )
        variants = (
            f"AS PERMISSIVE FOR SELECT TO PUBLIC USING ({predicate})",
            f"AS RESTRICTIVE FOR ALL TO PUBLIC USING ({predicate}) "
            f"WITH CHECK ({predicate})",
            f"AS PERMISSIVE FOR ALL TO CURRENT_USER USING ({predicate}) "
            f"WITH CHECK ({predicate})",
        )
        for variant in variants:
            with backend.transaction() as connection:
                backend.execute(
                    f"DROP POLICY {ident(policy)} ON {ident(table)}",  # nosec B608
                    connection=connection,
                )
                backend.execute(
                    f"CREATE POLICY {ident(policy)} ON {ident(table)} {variant}",  # nosec B608
                    connection=connection,
                )
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
            repaired = backend.execute(
                "SELECT permissive,roles,cmd FROM pg_policies "
                "WHERE schemaname=current_schema() AND tablename=? AND policyname=?",
                (table, policy),
            ).one
            assert repaired == {
                "permissive": "PERMISSIVE",
                "roles": ["public"],
                "cmd": "ALL",
            }

        stores = await asyncio.gather(
            *(
                create_semantic_vector_store(
                    "pgvector",
                    authority=db.note_semantic_store,
                    postgres_backend=backend,
                    settings=settings,
                )
                for _ in range(4)
            )
        )
        assert len(stores) == 4
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_pgvector_verification_is_schema_bound_and_rejects_wrong_index(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    settings = SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({DIMENSIONS}))
    schema_name = f"semantic_decoy_{uuid4().hex[:8]}"
    table = PGVECTOR_TABLES[DIMENSIONS]
    index = f"{table}_embedding_hnsw"
    ident = backend.escape_identifier
    try:
        try:
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
        except SemanticVectorCapabilityError as exc:
            _skip_or_fail_unavailable_pgvector(exc)

        with backend.transaction() as connection:
            backend.execute(
                f"CREATE SCHEMA {ident(schema_name)}",  # nosec B608
                connection=connection,
            )
            backend.execute(
                f"CREATE TABLE {ident(schema_name)}.{ident(table)} "  # nosec B608
                "(embedding vector(384))",
                connection=connection,
            )
        await create_semantic_vector_store(
            "pgvector",
            authority=db.note_semantic_store,
            postgres_backend=backend,
            settings=settings,
        )

        with backend.transaction() as connection:
            backend.execute(
                f"DROP INDEX {ident(index)}",  # nosec B608
                connection=connection,
            )
            backend.execute(
                f"CREATE INDEX {ident(index)} ON {ident(table)} (vector_id)",  # nosec B608
                connection=connection,
            )
        with pytest.raises(SemanticVectorCapabilityError) as exc_info:
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=settings,
            )
        assert exc_info.value.code == "notes_semantic_pgvector_schema_unavailable"

        with backend.transaction() as connection:
            backend.execute(
                f"DROP INDEX {ident(index)}",  # nosec B608
                connection=connection,
            )
        await create_semantic_vector_store(
            "pgvector",
            authority=db.note_semantic_store,
            postgres_backend=backend,
            settings=settings,
        )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_pgvector_identifiers_and_metric_labels_are_operator_bounded() -> None:
    assert set(PGVECTOR_TABLES) == {384, 768, 1024, 1536}
    assert all(name.startswith("note_semantic_vectors_d") for name in PGVECTOR_TABLES.values())
    assert frozenset({"backend", "operation", "outcome"}) == SEMANTIC_VECTOR_METRIC_LABELS
    assert SEMANTIC_VECTOR_METRIC_LABELS.isdisjoint(
        {"owner", "dataset", "generation", "table", "dimensions"}
    )


def test_pgvector_required_flag_converts_capability_skip_to_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SemanticVectorCapabilityError(
        "notes_semantic_pgvector_extension_unavailable"
    )
    monkeypatch.delenv(_PGVECTOR_REQUIRED_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        _skip_or_fail_unavailable_pgvector(error)

    monkeypatch.setenv(_PGVECTOR_REQUIRED_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        _skip_or_fail_unavailable_pgvector(error)


def test_notes_graph_ci_shard_requires_pgvector_service_and_failure_flag() -> None:
    workflow_path = Path(__file__).resolve().parents[4] / ".github/workflows/ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    job = workflow["jobs"]["full-suite-linux-312-shards"]
    notes_shard = next(
        shard
        for shard in job["strategy"]["matrix"]["shard"]
        if shard["name"] == "gap-verified-5"
    )

    assert "tldw_Server_API/tests/Notes_Graph" in notes_shard["paths"]
    assert notes_shard["postgres_image"] == "pgvector/pgvector:pg18"
    assert notes_shard["pgvector_required"] == "1"
    assert job["services"]["postgres"]["image"] == (
        "${{ matrix.shard.postgres_image || "
        "'mirror.gcr.io/library/postgres:18-bookworm' }}"
    )
    assert job["env"][_PGVECTOR_REQUIRED_ENV] == (
        "${{ matrix.shard.pgvector_required || '0' }}"
    )


@pytest.mark.asyncio
async def test_pgvector_rejects_non_allowlisted_dimension_before_schema_io(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with pytest.raises(ValueError, match="pgvector dimensions"):
            SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({3_072}))
        assert not backend.table_exists("note_semantic_vectors_d3072")
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_pgvector_default_settings_have_live_capability_without_3072(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        try:
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
            )
        except SemanticVectorCapabilityError as exc:
            _skip_or_fail_unavailable_pgvector(exc)

        rows = backend.execute(
            "SELECT c.relname FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname LIKE 'note_semantic_vectors_d%' "
            "AND c.relkind='r' "
            "ORDER BY c.relname"
        ).rows
        assert [row["relname"] for row in rows] == sorted(PGVECTOR_TABLES.values())
        assert "note_semantic_vectors_d3072" not in {
            row["relname"] for row in rows
        }
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_pgvector_storage_stays_bound_to_resolved_schema_when_search_path_changes(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    target_schema = f"semantic_target_{uuid4().hex[:8]}"
    other_schema = f"semantic_other_{uuid4().hex[:8]}"
    table = PGVECTOR_TABLES[DIMENSIONS]
    ident = setup_backend.escape_identifier
    def qualified(schema: str) -> str:
        return f"{ident(schema)}.{ident(table)}"
    try:
        try:
            setup_backend.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except DatabaseError:
            _skip_or_fail_unavailable_pgvector(
                SemanticVectorCapabilityError(
                    "notes_semantic_pgvector_extension_unavailable"
                )
            )
        with setup_backend.transaction() as connection:
            setup_backend.execute(
                f"CREATE SCHEMA {ident(target_schema)}",  # nosec B608
                connection=connection,
            )
            setup_backend.execute(
                f"CREATE SCHEMA {ident(other_schema)}",  # nosec B608
                connection=connection,
            )
            for schema in ("public", other_schema):
                setup_backend.execute(
                    f"CREATE TABLE {qualified(schema)} ("  # nosec B608
                    "owner_user_id TEXT NOT NULL,dataset_id TEXT NOT NULL,"
                    "generation_id TEXT NOT NULL,vector_id TEXT NOT NULL,"
                    f"embedding vector({DIMENSIONS}) NOT NULL)",
                    connection=connection,
                )
    finally:
        setup_backend.get_pool().close_all()

    monkeypatch.setenv("PGOPTIONS", f"-c search_path={target_schema},public")
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    original_transaction = backend.transaction
    db = CharactersRAGDB(str(tmp_path / "schema-authority.sqlite"), client_id="owner-a")
    generation_id = _generation(db)
    settings = SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({DIMENSIONS}))
    try:
        store = await create_semantic_vector_store(
            "pgvector",
            authority=db.note_semantic_store,
            postgres_backend=backend,
            settings=settings,
        )
        await store.upsert(
            DATASET_ID,
            generation_id,
            (SemanticVector("target-row", axis_vector(DIMENSIONS, 0)),),
        )
        with backend.transaction() as connection:
            for schema in ("public", other_schema):
                backend.execute(
                    f"INSERT INTO {qualified(schema)} VALUES (?,?,?,?,?::vector)",  # nosec B608
                    (
                        "owner-a",
                        DATASET_ID,
                        generation_id,
                        "target-row",
                        "[1" + ",0" * (DIMENSIONS - 1) + "]",
                    ),
                    connection=connection,
                )

        pg_backend = store._backend
        assert isinstance(pg_backend, PostgresSemanticVectorBackend)
        assert pg_backend._schema_name == target_schema

        @contextmanager
        def changed_search_path_transaction():
            with original_transaction() as connection:
                backend.execute(
                    f"SET LOCAL search_path={ident('public')},{ident(other_schema)}",  # nosec B608
                    connection=connection,
                )
                yield connection

        backend.transaction = changed_search_path_transaction  # type: ignore[method-assign]
        await pg_backend.check_capability()
        assert pg_backend._schema_name == target_schema
        cleanup = await store.delete_generation(DATASET_ID, generation_id)
        backend.transaction = original_transaction  # type: ignore[method-assign]

        assert cleanup.confirmed_absent is True
        target_count = backend.execute(
            f"SELECT COUNT(*) AS count FROM {qualified(target_schema)} "  # nosec B608
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=?",
            ("owner-a", DATASET_ID, generation_id),
        ).scalar
        public_count = backend.execute(
            f"SELECT COUNT(*) AS count FROM {qualified('public')} "  # nosec B608
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND vector_id=?",
            ("owner-a", DATASET_ID, generation_id, "target-row"),
        ).scalar
        other_count = backend.execute(
            f"SELECT COUNT(*) AS count FROM {qualified(other_schema)} "  # nosec B608
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND vector_id=?",
            ("owner-a", DATASET_ID, generation_id, "target-row"),
        ).scalar
        assert (target_count, public_count, other_count) == (0, 1, 1)

        backend.transaction = changed_search_path_transaction  # type: ignore[method-assign]
        repeated = await store.delete_generation(DATASET_ID, generation_id)
        backend.transaction = original_transaction  # type: ignore[method-assign]
        assert repeated.confirmed_absent is True
    finally:
        backend.transaction = original_transaction  # type: ignore[method-assign]
        db.close_all_connections()
        backend.get_pool().close_all()
