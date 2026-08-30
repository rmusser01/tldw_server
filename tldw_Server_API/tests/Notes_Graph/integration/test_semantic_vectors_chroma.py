"""ChromaDB integration contracts for Notes vector-only semantic storage."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import chromadb
import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import SemanticDimensionState
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    SemanticVector,
    SemanticVectorBinding,
    SemanticVectorCapabilityError,
    SemanticVectorError,
    create_semantic_vector_store,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors_chroma import (
    ChromaSemanticVectorBackend,
    _namespace,
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
    client.get_collection.side_effect = [ValueError("legacy missing"), collection]
    client.list_collections.return_value = []
    client.create_collection.return_value = collection
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

        create_kwargs = client.create_collection.call_args.kwargs
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
        client.get_or_create_collection.assert_not_called()
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
async def test_chroma_internal_keyerror_cannot_confirm_generation_absence(
    tmp_path: Path,
) -> None:
    client = Mock()
    client.get_collection.side_effect = KeyError("internal lookup failure")
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-keyerror.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        with pytest.raises(SemanticVectorError) as exc_info:
            await store.delete_generation("dataset-a", generation_id)

        assert exc_info.value.code == "notes_semantic_chroma_operation_failed"
        client.delete_collection.assert_not_called()
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "listed_collections",
    ([], ["unrelated"], [SimpleNamespace(name="unrelated")]),
)
async def test_chroma_legacy_valueerror_confirms_absence_from_collection_listing(
    tmp_path: Path,
    listed_collections: list[object],
) -> None:
    client = Mock()
    client.get_collection.side_effect = ValueError("legacy missing collection")
    client.list_collections.return_value = listed_collections
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-legacy.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        cleanup = await store.delete_generation("dataset-a", generation_id)

        assert cleanup.confirmed_absent is True
        client.list_collections.assert_called_once_with()
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listing", "code"),
    (
        ("target", "notes_semantic_chroma_operation_failed"),
        ([object()], "notes_semantic_vector_backend_result_invalid"),
        (None, "notes_semantic_vector_backend_result_invalid"),
        (["duplicate", "duplicate"], "notes_semantic_vector_backend_result_invalid"),
        ([""], "notes_semantic_vector_backend_result_invalid"),
        (["ab"], "notes_semantic_vector_backend_result_invalid"),
        (["bad\x00name"], "notes_semantic_vector_backend_result_invalid"),
        (["bad..name"], "notes_semantic_vector_backend_result_invalid"),
        (["\ud800xx"], "notes_semantic_vector_backend_result_invalid"),
        ({"valid-name"}, "notes_semantic_vector_backend_result_invalid"),
    ),
)
async def test_chroma_legacy_valueerror_fails_closed_without_proven_absence(
    tmp_path: Path,
    listing: object,
    code: str,
) -> None:
    client = Mock()
    client.get_collection.side_effect = ValueError("legacy ambiguous lookup")

    def list_collections():
        if listing == "target":
            return [client.get_collection.call_args.kwargs["name"]]
        return listing

    client.list_collections.side_effect = list_collections
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-legacy-fail.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        with pytest.raises(SemanticVectorError) as exc_info:
            await store.delete_generation("dataset-a", generation_id)

        assert exc_info.value.code == code
        client.delete_collection.assert_not_called()
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_existing_legacy_collection_is_validated_without_metadata_mutation(
    tmp_path: Path,
) -> None:
    collection = Mock()
    collection.metadata = {"hnsw:space": "l2", "legacy": "preserve"}

    class MutationSensitiveClient:
        def __init__(self) -> None:
            self.get_or_create_calls = 0
            self.create_calls = 0

        def get_collection(self, *, name: str):
            return collection

        def get_or_create_collection(self, **kwargs):
            self.get_or_create_calls += 1
            collection.metadata = kwargs["metadata"]
            return collection

        def create_collection(self, **_kwargs):
            self.create_calls += 1
            raise AssertionError("existing collection must not be recreated")

        def list_collections(self):
            return ["unrelated"]

        def delete_collection(self, **_kwargs):
            return None

    client = MutationSensitiveClient()
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-legacy-l2.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        with pytest.raises(SemanticVectorCapabilityError) as exc_info:
            await store.create_generation_storage("dataset-a", generation_id)

        assert exc_info.value.code == "notes_semantic_chroma_cosine_unavailable"
        assert collection.metadata == {"hnsw:space": "l2", "legacy": "preserve"}
        assert client.get_or_create_calls == 0
        assert client.create_calls == 0
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_concurrent_create_rereads_and_validates_winner(tmp_path: Path) -> None:
    collection = Mock()
    collection.metadata = {"hnsw:space": "cosine"}
    client = Mock()
    client.get_collection.side_effect = [ValueError("missing"), collection]
    client.list_collections.return_value = []
    client.create_collection.side_effect = ValueError("already exists")
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-race.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        await store.create_generation_storage("dataset-a", generation_id)

        assert client.get_collection.call_count == 2
        client.get_or_create_collection.assert_not_called()
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
async def test_chroma_capability_requires_collection_listing(tmp_path: Path) -> None:
    client = SimpleNamespace(
        create_collection=lambda **_kwargs: None,
        delete_collection=lambda **_kwargs: None,
        get_collection=lambda **_kwargs: None,
    )
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-capability.sqlite"), client_id="owner-a")
    try:
        with pytest.raises(SemanticVectorCapabilityError) as exc_info:
            await create_semantic_vector_store(
                "chromadb",
                authority=db.note_semantic_store,
                chroma_manager=manager,
            )
        assert exc_info.value.code == "notes_semantic_chroma_unavailable"
    finally:
        manager.close()
        db.close_all_connections()


def test_chroma_backend_imports_without_modern_not_found_error() -> None:
    module_name = "tldw_Server_API.app.core.Notes_Graph.semantic_vectors_chroma"
    errors = importlib.import_module("chromadb.errors")
    missing = object()
    modern_not_found = getattr(errors, "NotFoundError", missing)
    if modern_not_found is not missing:
        delattr(errors, "NotFoundError")
    sys.modules.pop(module_name, None)
    try:
        imported = importlib.import_module(module_name)
        assert imported.ChromaSemanticVectorBackend is not None
    finally:
        if modern_not_found is not missing:
            errors.NotFoundError = modern_not_found
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    (
        None,
        [],
        {},
        {"ids": None},
        {"ids": "opaque-vector"},
        {"ids": object()},
        {"ids": ["foreign-vector"]},
        {"ids": [1]},
    ),
)
async def test_chroma_delete_ids_rejects_ambiguous_cleanup_results(
    tmp_path: Path,
    result: object,
) -> None:
    collection = Mock()
    collection.get.return_value = result
    client = Mock()
    client.get_collection.return_value = collection
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-delete-result.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        with pytest.raises(SemanticVectorError) as exc_info:
            await store.delete_ids("dataset-a", generation_id, ("opaque-vector",))

        assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"
    finally:
        manager.close()
        db.close_all_connections()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remaining_ids", "confirmed_absent"),
    (([], True), (["opaque-vector"], False)),
)
async def test_chroma_delete_ids_accepts_explicit_requested_id_subset(
    tmp_path: Path,
    remaining_ids: list[str],
    confirmed_absent: bool,
) -> None:
    collection = Mock()
    collection.get.return_value = {"ids": remaining_ids}
    client = Mock()
    client.get_collection.return_value = collection
    manager = _manager("owner-a", tmp_path, client)
    db = CharactersRAGDB(str(tmp_path / "authority-delete-control.sqlite"), client_id="owner-a")
    generation_id = _generation(db, "dataset-a")
    try:
        store = await create_semantic_vector_store(
            "chromadb",
            authority=db.note_semantic_store,
            chroma_manager=manager,
        )

        cleanup = await store.delete_ids(
            "dataset-a",
            generation_id,
            ("opaque-vector",),
        )

        assert cleanup.confirmed_absent is confirmed_absent
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


class _FloatLike:
    def __float__(self) -> float:
        return 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    (
        {},
        {"ids": [[]]},
        {"ids": "opaque-vector", "distances": [[0.0]]},
        {"ids": [["opaque-vector"]], "distances": "0.0"},
        {"ids": ["opaque-vector"], "distances": [[0.0]]},
        {"ids": [["opaque-vector"]], "distances": ["0.0"]},
        {"ids": [[object()]], "distances": [[0.0]]},
        {"ids": [["opaque-vector"]], "distances": [[_FloatLike()]]},
        {"ids": [["opaque-vector"]], "distances": [[True]]},
        {"ids": [["opaque-vector"]], "distances": [[float("nan")]]},
        {"ids": [["opaque-vector"]], "distances": [[float("inf")]]},
        {"ids": [], "distances": []},
        {"ids": [["a", "b"]], "distances": [[0.0]]},
    ),
)
async def test_chroma_query_rejects_malformed_result_shapes_and_values(
    result: object,
) -> None:
    collection = Mock()
    collection.query.return_value = result
    client = Mock()
    client.get_collection.return_value = collection
    backend = ChromaSemanticVectorBackend(SimpleNamespace(client=client))
    await backend.check_capability()
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", DIMENSIONS)

    with pytest.raises(SemanticVectorError) as exc_info:
        await backend.query(
            binding,
            (axis_vector(DIMENSIONS, 0),),
            limit=1,
            candidate_limit=2,
        )

    assert exc_info.value.code == "notes_semantic_vector_backend_result_invalid"


@pytest.mark.asyncio
async def test_current_chroma_existing_l2_collection_remains_unmodified(tmp_path: Path) -> None:
    client = chromadb.PersistentClient(path=str(tmp_path / "existing-l2-chroma"))
    manager = _manager("owner-a", tmp_path, client)
    backend = ChromaSemanticVectorBackend(manager)
    binding = SemanticVectorBinding("owner-a", "dataset-a", "generation-a", DIMENSIONS)
    collection_name = _namespace(binding)
    client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "l2", "legacy": "preserve"},
    )
    try:
        await backend.check_capability()

        with pytest.raises(SemanticVectorCapabilityError) as exc_info:
            await backend.create_generation_storage(binding)

        assert exc_info.value.code == "notes_semantic_chroma_cosine_unavailable"
        assert client.get_collection(name=collection_name).metadata == {
            "hnsw:space": "l2",
            "legacy": "preserve",
        }
    finally:
        manager.close()


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
