"""Task 6 pure calculation contracts, independent of publication and Jobs."""

import importlib
import math
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import (
    ClusterRebuildLimits,
    ClusterRebuildSettings,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterRebuildStorageError,
    ClusterSourceRow,
    ClusterSourceSnapshot,
)

LIMITS = ClusterRebuildLimits(50000, 5000, 33554432, 4096, 2000000, 50000000, 50000)


def calculate(rows, *, method="exact", vectors=None, limits=LIMITS, threshold=0.85, min_size=2):
    module = importlib.import_module("tldw_Server_API.app.core.Claims_Extraction.claims_cluster_rebuild")
    return module.calculate_claim_cluster_rebuild(
        ClusterSourceSnapshot("1", tuple(rows), "digest"),
        settings=ClusterRebuildSettings(1, method, min_size, threshold, 1),
        limits=limits,
        read_vectors=vectors,
    )


def row(claim_id, text, *, media_id=1, chunk_index=0):
    return ClusterSourceRow(claim_id, media_id, chunk_index, text)


def vector_reader(rows, vectors):
    mapping = {claim_embedding_id(r.media_id, r.chunk_index, r.claim_text): v for r, v in zip(rows, vectors)}
    return lambda ids: {key: mapping[key] for key in ids if key in mapping}


def test_exact_normalization_preserves_first_source_representative():
    result = calculate([row(1, "  MiXeD\tcase  "), row(2, "mixed case"), row(3, "other")])
    (assignment,) = result.assignments
    assert (assignment.canonical_claim_text, assignment.representative_claim_id) == ("MiXeD\tcase", 1)
    assert [(m.claim_id, m.similarity) for m in assignment.members] == [(1, 1.0), (2, 1.0)]


def test_empty_exact_replaces_with_empty_output():
    result = calculate([])
    assert (result.outcome, result.reason, result.assignments, result.claims_skipped) == ("ok", None, (), 0)


def test_missing_vectors_are_a_preserving_skip():
    result = calculate([row(1, "a")], method="embeddings", vectors=lambda ids: {})
    assert (result.outcome, result.reason, result.assignments, result.claims_skipped) == (
        "skipped",
        "no_embeddings",
        (),
        1,
    )


def test_partial_vectors_keep_legacy_missing_count():
    rows = [row(1, "a"), row(2, "b"), row(3, "c")]
    result = calculate(rows, method="embeddings", vectors=vector_reader(rows[:2], [[1, 0], [1, 0]]))
    assert result.claims_skipped == 1
    assert [m.claim_id for m in result.assignments[0].members] == [1, 2]


def test_embedding_threshold_equality_does_not_join():
    rows = [row(1, "a"), row(2, "b")]
    result = calculate(rows, method="embeddings", vectors=vector_reader(rows, [[1, 0], [1, 0]]), threshold=1)
    assert result.assignments == ()
    assert result.outcome == "ok"


def test_embedding_greedy_order_and_final_centroid_similarity():
    rows = [row(1, "a"), row(2, "b"), row(3, "c")]
    result = calculate(rows, method="embeddings", vectors=vector_reader(rows, [[1, 0], [1, 1], [0, 1]]), threshold=0.5)
    (assignment,) = result.assignments
    assert assignment.representative_claim_id == 1
    assert [m.claim_id for m in assignment.members] == [1, 2]
    assert [m.similarity for m in assignment.members] == pytest.approx([2 / math.sqrt(5), 3 / math.sqrt(10)])


def test_colliding_embedding_ids_keep_last_claim_mapping_without_expanding_coverage():
    rows = [row(1, "same"), row(2, "same")]
    result = calculate(rows, method="embeddings", vectors=vector_reader(rows, [[1, 0], [1, 0]]), min_size=1)
    assert [m.claim_id for a in result.assignments for m in a.members] == [2]
    assert result.claims_skipped == 1


@pytest.mark.parametrize("vectors", [[[0, 0]], [[math.nan, 1]], [[math.inf, 1]], [[1e308, 1]], [[]], [[1e-300, 0]]])
def test_invalid_vectors_fail_closed(vectors):
    rows = [row(1, "a")]
    with pytest.raises(ClusterRebuildStorageError) as caught:
        calculate(rows, method="embeddings", vectors=vector_reader(rows, vectors))
    assert not caught.value.retryable


def test_mixed_dimensions_fail_closed():
    rows = [row(1, "a"), row(2, "b")]
    with pytest.raises(ClusterRebuildStorageError):
        calculate(rows, method="embeddings", vectors=vector_reader(rows, [[1], [1, 0]]))


@pytest.mark.parametrize("field,value", [("exact_claims", 1), ("text_bytes", 1)])
def test_exact_resource_limits(field, value):
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        calculate([row(1, "aa"), row(2, "aa")], limits=replace(LIMITS, **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_claims", 1),
        ("vector_width", 1),
        ("embedding_components", 3),
        ("similarity_component_comparisons", 1),
    ],
)
def test_embedding_resource_limits(field, value):
    rows = [row(1, "a"), row(2, "b")]
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        calculate(
            rows,
            method="embeddings",
            vectors=vector_reader(rows, [[1, 0], [1, 0]]),
            limits=replace(LIMITS, **{field: value}),
        )


def test_vector_access_error_is_not_missing_embeddings():
    def broken(ids):
        raise ConnectionError("private path or credentials")

    with pytest.raises(ClusterRebuildStorageError) as caught:
        calculate([row(1, "a")], method="embeddings", vectors=broken)
    assert caught.value.retryable
    assert "private" not in str(caught.value)
