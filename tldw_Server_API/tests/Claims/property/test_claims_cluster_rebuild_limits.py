"""Property checks for version-1 semantics and input/work ceilings."""

import math
from dataclasses import replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import ClusterRebuildLimits, ClusterRebuildSettings
from tldw_Server_API.app.core.Claims_Extraction.claims_cluster_rebuild import calculate_claim_cluster_rebuild
from tldw_Server_API.app.core.Claims_Extraction.claims_clustering import ClaimEmbedding, _cluster_embeddings
from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
    ClusterRebuildStorageError,
    ClusterSourceRow,
    ClusterSourceSnapshot,
)

LIMITS = ClusterRebuildLimits(50000, 5000, 33554432, 4096, 2000000, 50000000, 50000)


@given(st.lists(st.sampled_from(["A", " a ", "B", " b\t", "C"]), max_size=40), st.integers(1, 5))
def test_exact_partition_is_stable_and_source_is_unchanged(texts, min_size):
    rows = tuple(ClusterSourceRow(i + 1, 1, 0, text) for i, text in enumerate(texts))
    source = ClusterSourceSnapshot("1", rows, "digest")
    result = calculate_claim_cluster_rebuild(
        source, settings=ClusterRebuildSettings(1, "exact", min_size, None, 1), limits=LIMITS, read_vectors=None
    )
    assigned = [member.claim_id for cluster in result.assignments for member in cluster.members]
    assert len(assigned) == len(set(assigned))
    assert source.rows == rows
    for cluster in result.assignments:
        assert len(cluster.members) >= min_size
        assert cluster.representative_claim_id == min(member.claim_id for member in cluster.members)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(st.integers(1, 10), st.integers(-10, 10)), min_size=1, max_size=15),
    st.sampled_from([0.0, 0.5, 0.85, 1.0]),
    st.integers(1, 3),
)
def test_embedding_matches_legacy_greedy_order_and_member_scores(vectors, threshold, min_size):
    rows = tuple(ClusterSourceRow(i + 1, 1, i, str(i)) for i in range(len(vectors)))
    mapping = {
        claim_embedding_id(r.media_id, r.chunk_index, r.claim_text): tuple(v)
        for r, v in zip(rows, vectors, strict=True)
    }
    result = calculate_claim_cluster_rebuild(
        ClusterSourceSnapshot("1", rows, "digest"),
        settings=ClusterRebuildSettings(1, "embeddings", min_size, threshold, 1),
        limits=LIMITS,
        read_vectors=lambda ids: {key: mapping[key] for key in reversed(ids)},
    )
    legacy = _cluster_embeddings(
        [
            ClaimEmbedding(
                r.claim_id, r.media_id, r.chunk_index, r.claim_text, list(v), math.sqrt(sum(x * x for x in v))
            )
            for r, v in zip(rows, vectors, strict=True)
        ],
        threshold,
        min_size,
    )
    assert [a.representative_claim_id for a in result.assignments] == [a["representative_claim_id"] for a in legacy]
    for actual, expected in zip(result.assignments, legacy, strict=True):
        assert [m.claim_id for m in actual.members] == [m["claim_id"] for m in expected["members"]]
        assert [m.similarity for m in actual.members] == pytest.approx([m["similarity"] for m in expected["members"]])


@given(st.integers(2, 30), st.sampled_from(["exact", "embeddings"]))
def test_count_ceiling_rejects_before_vector_access(count, method):
    rows = tuple(ClusterSourceRow(i + 1, 1, i, "a") for i in range(count))

    def forbidden(ids):
        pytest.fail("vectors were loaded before checking the claim ceiling")

    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        calculate_claim_cluster_rebuild(
            ClusterSourceSnapshot("1", rows, "digest"),
            settings=ClusterRebuildSettings(1, method, 1, 0.85, 1),
            limits=replace(LIMITS, **{f"{method[:-1] if method == 'embeddings' else method}_claims": count - 1}),
            read_vectors=forbidden,
        )


@given(st.text(alphabet=st.sampled_from(["a", "\u00e9", "\U0001f680"]), min_size=2, max_size=40))
def test_utf8_byte_ceiling_is_not_a_character_limit(text):
    size = len(text.encode("utf-8"))
    rows = (ClusterSourceRow(1, 1, 0, text),)
    with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_resource_limit"):
        calculate_claim_cluster_rebuild(
            ClusterSourceSnapshot("1", rows, "digest"),
            settings=ClusterRebuildSettings(1, "exact", 1, None, 1),
            limits=replace(LIMITS, text_bytes=size - 1),
            read_vectors=None,
        )
