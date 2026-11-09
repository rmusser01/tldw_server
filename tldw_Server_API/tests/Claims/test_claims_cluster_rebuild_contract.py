"""Neutral cluster settings reject unsafe inputs before Jobs admission."""

from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_Server_API.app.core import claims_cluster_rebuild_contract as contract


def test_exact_threshold_is_not_a_semantic_setting():
    first = contract.ClusterRebuildSettings(1, "exact", 2, None, 1)
    explicit = contract.ClusterRebuildSettings(1, "exact", 2, 0.85, 1)
    assert contract.canonical_cluster_settings(first) == {
        "version": 1,
        "method": "exact",
        "min_size": 2,
        "algorithm_version": 1,
    }
    assert contract.canonical_cluster_settings(explicit) == contract.canonical_cluster_settings(first)
    assert explicit.similarity_threshold is None
    with pytest.raises(FrozenInstanceError):
        first.min_size = 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("version", 1.0),
        ("algorithm_version", False),
        ("algorithm_version", 2),
        ("min_size", True),
        ("min_size", 0),
        ("min_size", 1001),
        ("min_size", 2.0),
        ("method", "EXACT"),
        ("method", "unknown"),
        ("similarity_threshold", None),
        ("similarity_threshold", True),
        ("similarity_threshold", float("nan")),
        ("similarity_threshold", float("inf")),
        ("similarity_threshold", -0.1),
        ("similarity_threshold", 1.1),
    ],
)
def test_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(contract.ClusterRebuildSettings(1, "embeddings", 2, 0.85, 1), **{field: value})


@pytest.mark.parametrize("threshold", [0, -0.0, 1, 0.85])
def test_embedding_settings_preserve_effective_threshold(threshold):
    settings = contract.ClusterRebuildSettings(1, "embeddings", 1000, threshold, 1)
    assert contract.canonical_cluster_settings(settings)["similarity_threshold"] == threshold


@pytest.mark.parametrize("owner", [None, True, 1, "0", "01", " 1", "1 ", "u1", "../1", "9223372036854775808"])
def test_payload_owner_uses_existing_routable_owner_contract(owner):
    with pytest.raises(ValueError):
        contract.validate_cluster_rebuild_payload(
            {
                "version": 1,
                "owner_user_id": owner,
                "method": "exact",
                "min_size": 2,
                "similarity_threshold": None,
                "algorithm_version": 1,
            }
        )


def test_payload_is_bounded_and_normalized_without_domain_dependencies():
    payload = {
        "version": 1,
        "owner_user_id": "1",
        "method": "exact",
        "min_size": 2,
        "similarity_threshold": 0.85,
        "algorithm_version": 1,
    }
    validated = contract.validate_cluster_rebuild_payload(payload)
    assert validated == {**payload, "similarity_threshold": None}
    for extra in ("claim_text", "vectors", "path", "queue", "resource_generation"):
        with pytest.raises(ValueError):
            contract.validate_cluster_rebuild_payload({**payload, extra: "forbidden"})
    with pytest.raises(ValueError):
        contract.validate_cluster_rebuild_payload({key: value for key, value in payload.items() if key != "version"})


@pytest.mark.parametrize(
    "field,maximum",
    [
        ("exact_claims", 50_000),
        ("embedding_claims", 5_000),
        ("text_bytes", 33_554_432),
        ("vector_width", 4_096),
        ("embedding_components", 2_000_000),
        ("similarity_component_comparisons", 50_000_000),
        ("cleanup_references", 50_000),
    ],
)
def test_limits_are_exact_integers_with_hard_ceilings(field, maximum):
    limits = contract.ClusterRebuildLimits(50_000, 5_000, 33_554_432, 4_096, 2_000_000, 50_000_000, 50_000)
    assert getattr(replace(limits, **{field: 1}), field) == 1
    for invalid in (True, 0, -1, maximum + 1, 1.0):
        with pytest.raises(ValueError):
            replace(limits, **{field: invalid})
