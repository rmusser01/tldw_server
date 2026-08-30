"""Live PostgreSQL contracts for Notes pgvector semantic storage."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
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
            assert exc.code == "notes_semantic_pgvector_extension_unavailable"
            pytest.skip("pgvector extension is unavailable in the PostgreSQL fixture")

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
            "SELECT c.relrowsecurity,c.relforcerowsecurity,"
            "format_type(a.atttypid,a.atttypmod) AS vector_type "
            "FROM pg_class c JOIN pg_attribute a ON a.attrelid=c.oid "
            "WHERE c.relname=? AND a.attname='embedding'",
            (table,),
        ).one
        assert schema == {
            "relrowsecurity": True,
            "relforcerowsecurity": True,
            "vector_type": f"vector({DIMENSIONS})",
        }
        primary_key = backend.execute(
            "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid=?::regclass AND contype='p'",
            (table,),
        ).one["definition"]
        assert primary_key == "PRIMARY KEY (owner_user_id, dataset_id, generation_id, vector_id)"
        index_definitions = backend.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename=?",
            (table,),
        ).rows
        assert any(
            "USING hnsw (embedding vector_cosine_ops)" in row["indexdef"]
            for row in index_definitions
        )
        policies = backend.execute(
            "SELECT policyname,qual,with_check FROM pg_policies WHERE tablename=?",
            (table,),
        ).rows
        assert len(policies) == 1
        assert policies[0]["policyname"] == f"{table}_tenant_isolation"
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
async def test_pgvector_query_is_not_underfilled_by_other_generation_decoys(
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
            assert exc.code == "notes_semantic_pgvector_extension_unavailable"
            pytest.skip("pgvector extension is unavailable in the PostgreSQL fixture")

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

        assert [[match.vector_id for match in batch] for batch in matches] == [["target"]]
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_pgvector_identifiers_and_metric_labels_are_operator_bounded() -> None:
    assert set(PGVECTOR_TABLES) == {384, 768, 1024, 1536, 3072}
    assert all(name.startswith("note_semantic_vectors_d") for name in PGVECTOR_TABLES.values())
    assert frozenset({"backend", "operation", "outcome"}) == SEMANTIC_VECTOR_METRIC_LABELS
    assert SEMANTIC_VECTOR_METRIC_LABELS.isdisjoint(
        {"owner", "dataset", "generation", "table", "dimensions"}
    )


@pytest.mark.asyncio
async def test_pgvector_rejects_non_allowlisted_dimension_before_schema_io(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with pytest.raises(SemanticVectorCapabilityError) as exc_info:
            await create_semantic_vector_store(
                "pgvector",
                authority=db.note_semantic_store,
                postgres_backend=backend,
                settings=SemanticIndexSettings(
                    pgvector_allowed_dimensions=frozenset({32_768})
                ),
            )
        assert exc_info.value.code == "notes_semantic_pgvector_dimensions_unsupported"
        assert not backend.table_exists("note_semantic_vectors_d32768")
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
