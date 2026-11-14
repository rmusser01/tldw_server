"""Reusable behavioral contract for Notes semantic vector stores."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import (
    NotesSemanticVectorStore,
    SemanticVector,
    SemanticVectorValidationError,
)


def axis_vector(dimensions: int, axis: int) -> tuple[float, ...]:
    """Return one deterministic non-zero vector for contract tests."""

    values = [0.0] * dimensions
    values[axis] = 1.0
    return tuple(values)


async def assert_vector_lifecycle_contract(
    store: NotesSemanticVectorStore,
    *,
    dataset_id: str,
    generation_id: str,
    dimensions: int,
) -> None:
    """Exercise the backend-independent vector lifecycle and result shapes."""

    first = axis_vector(dimensions, 0)
    second = axis_vector(dimensions, 1)

    await store.create_generation_storage(dataset_id, generation_id)
    assert await store.upsert(
        dataset_id,
        generation_id,
        (
            SemanticVector(vector_id="vector-b", embedding=second),
            SemanticVector(vector_id="vector-a", embedding=first),
        ),
    ) == 2

    fetched = await store.fetch(
        dataset_id,
        generation_id,
        ("vector-a", "missing-vector", "vector-b"),
    )
    assert fetched == (
        SemanticVector(vector_id="vector-a", embedding=first),
        SemanticVector(vector_id="vector-b", embedding=second),
    )

    results = await store.query(
        dataset_id,
        generation_id,
        (first, second),
        limit=2,
    )
    assert len(results) == 2
    assert [match.vector_id for match in results[0]] == ["vector-a", "vector-b"], results
    assert [match.vector_id for match in results[1]] == ["vector-b", "vector-a"], results
    assert results[0][0].distance == pytest.approx(0.0, abs=1e-6)
    assert results[0][1].distance == pytest.approx(1.0, abs=1e-6)
    assert max(0.0, min(1.0, 1.0 - results[0][1].distance)) == pytest.approx(0.0)

    deleted = await store.delete_ids(
        dataset_id,
        generation_id,
        ("vector-a", "already-absent"),
    )
    assert deleted.confirmed_absent is True
    assert await store.fetch(dataset_id, generation_id, ("vector-a",)) == ()
    assert (
        await store.delete_ids(dataset_id, generation_id, ("vector-a",))
    ).confirmed_absent is True

    cleanup = await store.delete_generation(dataset_id, generation_id)
    assert cleanup.confirmed_absent is True
    assert (
        await store.delete_generation(dataset_id, generation_id)
    ).confirmed_absent is True


async def assert_vector_validation_contract(
    store: NotesSemanticVectorStore,
    *,
    dataset_id: str,
    generation_id: str,
    dimensions: int,
) -> None:
    """Assert invalid vectors fail before reaching either physical backend."""

    await store.create_generation_storage(dataset_id, generation_id)
    invalid_embeddings: Sequence[tuple[float, ...]] = (
        (1.0,),
        tuple(float("nan") if index == 0 else 0.0 for index in range(dimensions)),
        tuple(float("inf") if index == 0 else 0.0 for index in range(dimensions)),
        tuple(0.0 for _ in range(dimensions)),
    )
    for embedding in invalid_embeddings:
        with pytest.raises(SemanticVectorValidationError):
            await store.upsert(
                dataset_id,
                generation_id,
                (SemanticVector(vector_id="invalid-vector", embedding=embedding),),
            )
        with pytest.raises(SemanticVectorValidationError):
            await store.query(
                dataset_id,
                generation_id,
                (embedding,),
                limit=1,
            )


async def assert_vector_isolation_contract(
    primary: tuple[NotesSemanticVectorStore, str, str],
    isolated: Sequence[tuple[NotesSemanticVectorStore, str, str]],
    *,
    dimensions: int,
) -> None:
    """Assert owner, dataset, and generation namespaces cannot see each other."""

    store, dataset_id, generation_id = primary
    vector = SemanticVector("shared-opaque-id", axis_vector(dimensions, 0))
    await store.create_generation_storage(dataset_id, generation_id)
    await store.upsert(dataset_id, generation_id, (vector,))

    for isolated_store, isolated_dataset, isolated_generation in isolated:
        await isolated_store.create_generation_storage(
            isolated_dataset,
            isolated_generation,
        )
        assert await isolated_store.fetch(
            isolated_dataset,
            isolated_generation,
            (vector.vector_id,),
        ) == ()

    assert await store.fetch(dataset_id, generation_id, (vector.vector_id,)) == (vector,)
    assert (
        await store.delete_ids(dataset_id, generation_id, (vector.vector_id,))
    ).confirmed_absent is True
