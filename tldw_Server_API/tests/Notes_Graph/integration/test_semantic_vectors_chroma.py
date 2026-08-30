"""ChromaDB integration contracts for Notes vector-only semantic storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import chromadb
import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import SemanticDimensionState
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorError,
    create_semantic_vector_store,
)
from tldw_Server_API.tests.Notes_Graph.vector_contract import (
    assert_vector_isolation_contract,
    assert_vector_lifecycle_contract,
    assert_vector_validation_contract,
    axis_vector,
)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

DIMENSIONS = 384
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def _generation(db: CharactersRAGDB, dataset_id: str) -> str:
    config = db.note_semantic_store.create_configuration(
        dataset_id=dataset_id,
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="provider",
        vector_backend="chromadb",
        storage_boundary="server_local",
        storage_label="local vectors",
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


def _manager(owner: str, base_dir: Path, client) -> ChromaDBManager:
    return ChromaDBManager(
        user_id=owner,
        user_embedding_config={
            "USER_DB_BASE_DIR": str(base_dir),
            "embedding_config": {"default_model_id": "unused", "models": {}},
        },
        client=client,
    )


def _additional_generation(db: CharactersRAGDB, dataset_id: str) -> str:
    config = db.note_semantic_store.get_configuration(dataset_id)
    assert config is not None
    assert config.active_generation_id is None
    with db.transaction() as connection:
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
async def test_chroma_satisfies_reusable_vector_contract(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "authority.sqlite"), client_id="owner-a")
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    manager = _manager("owner-a", tmp_path, client)
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )
        await assert_vector_validation_contract(
            store,
            dataset_id="dataset-a",
            generation_id=generation_id,
            dimensions=DIMENSIONS,
        )
        await assert_vector_lifecycle_contract(
            store,
            dataset_id="dataset-a",
            generation_id=generation_id,
            dimensions=DIMENSIONS,
        )
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_first_write_is_vector_only_and_namespace_is_opaque(tmp_path: Path) -> None:
    collection = Mock()
    collection.metadata = {"hnsw:space": "cosine"}
    client = Mock()
    client.get_or_create_collection.return_value = collection
    client.get_collection.return_value = collection
    manager = _manager("owner-a", tmp_path, client)
    manager.store_in_chroma = Mock(side_effect=AssertionError("document path used"))
    db = CharactersRAGDB(str(tmp_path / "authority-spy.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "private-dataset-name")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )
        await store.create_generation_storage("private-dataset-name", generation_id)
        await store.upsert(
            "private-dataset-name",
            generation_id,
            (SemanticVector("opaque-vector", axis_vector(DIMENSIONS, 0)),),
        )

        create_kwargs = client.get_or_create_collection.call_args.kwargs
        assert create_kwargs["metadata"] == {"hnsw:space": "cosine"}
        namespace = create_kwargs["name"]
        assert namespace.startswith("nsv_")
        assert "owner-a" not in namespace
        assert "private-dataset-name" not in namespace
        assert generation_id not in namespace
        assert collection.upsert.call_args.kwargs == {
            "ids": ["opaque-vector"],
            "embeddings": [list(axis_vector(DIMENSIONS, 0))],
        }
        manager.store_in_chroma.assert_not_called()
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_fetch_and_delete_do_not_create_missing_storage(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "authority-missing.sqlite"), client_id="owner-a")
    client = chromadb.PersistentClient(path=str(tmp_path / "missing-chroma"))
    manager = _manager("owner-a", tmp_path, client)
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )
        assert await store.fetch("dataset-a", generation_id, ("missing",)) == ()
        assert (
            await store.delete_ids("dataset-a", generation_id, ("missing",))
        ).confirmed_absent is True
        assert (
            await store.delete_generation("dataset-a", generation_id)
        ).confirmed_absent is True
        assert client.list_collections() == []
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_maps_malformed_distance_to_stable_error(tmp_path: Path) -> None:
    collection = Mock()
    collection.metadata = {"hnsw:space": "cosine"}
    collection.query.return_value = {
        "ids": [["opaque-vector"]],
        "distances": [["not-a-distance"]],
    }
    client = Mock()
    client.get_or_create_collection.return_value = collection
    client.get_collection.return_value = collection
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-malformed.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )
        with pytest.raises(SemanticVectorError) as exc_info:
            await store.query(
                "dataset-a",
                generation_id,
                (axis_vector(DIMENSIONS, 0),),
                limit=1,
            )
        assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_isolates_owner_dataset_and_generation_namespaces(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path / "isolated-chroma"))
    owner_a_db = CharactersRAGDB(str(tmp_path / "owner-a.sqlite"), client_id="owner-a")
    owner_b_db = CharactersRAGDB(str(tmp_path / "owner-b.sqlite"), client_id="owner-b")
    owner_a_manager = _manager("owner-a", tmp_path, client)
    owner_b_manager = _manager("owner-b", tmp_path, client)
    dataset_generation = _generation(owner_a_db, "dataset-a")
    other_generation = _additional_generation(owner_a_db, "dataset-a")
    other_dataset_generation = _generation(owner_a_db, "dataset-b")
    other_owner_generation = _generation(owner_b_db, "dataset-a")
    try:
        owner_a_store = await create_semantic_vector_store(
            "chromadb",
            authority=owner_a_db.note_semantic_store,
            chroma_manager=owner_a_manager,
        )
        owner_b_store = await create_semantic_vector_store(
            "chromadb",
            authority=owner_b_db.note_semantic_store,
            chroma_manager=owner_b_manager,
        )
        await assert_vector_isolation_contract(
            (owner_a_store, "dataset-a", dataset_generation),
            (
                (owner_a_store, "dataset-a", other_generation),
                (owner_a_store, "dataset-b", other_dataset_generation),
                (owner_b_store, "dataset-a", other_owner_generation),
            ),
            dimensions=DIMENSIONS,
        )
    finally:
        owner_b_manager.client = None
        owner_a_manager.close()
        owner_a_db.close_all_connections()
        owner_b_db.close_all_connections()
