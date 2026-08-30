"""Unit contracts for fenced Notes semantic indexing and observability."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_observability import (
    SemanticObservationError,
    build_semantic_audit_event,
    build_semantic_metric_event,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    SemanticAuthorityState,
    SemanticExecutionFence,
    SemanticIndexingError,
    validate_execution_fence,
)

pytestmark = pytest.mark.unit
GENERATION_FENCE = "-".join(("job", "fence", "a"))
STALE_GENERATION_FENCE = "-".join(("stale", "fence"))


def _fence() -> SemanticExecutionFence:
    return SemanticExecutionFence(
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        generation_id="generation-a",
        generation_fencing_token=GENERATION_FENCE,
        configuration_revision=4,
        capability_revision="capability-a",
        disclosure_hash="disclosure-a",
        provider="openai",
        model="embedding-model-a",
        model_revision="revision-a",
        endpoint_origin_revision="origin-a",
        compatibility_hash="compatibility-a",
        dimensions=2,
        vector_backend="chromadb",
    )


def _authority() -> SemanticAuthorityState:
    fence = _fence()
    return SemanticAuthorityState(
        user_exists=True,
        owner_authorized=True,
        semantic_manage_allowed=True,
        desired_enabled=True,
        owner_user_id=fence.owner_user_id,
        dataset_id=fence.dataset_id,
        generation_id=fence.generation_id,
        generation_fencing_token=fence.generation_fencing_token,
        configuration_revision=fence.configuration_revision,
        capability_revision=fence.capability_revision,
        disclosure_hash=fence.disclosure_hash,
        provider=fence.provider,
        model=fence.model,
        model_revision=fence.model_revision,
        endpoint_origin_revision=fence.endpoint_origin_revision,
        endpoint_policy_allowed=True,
        compatibility_hash=fence.compatibility_hash,
        dimensions=fence.dimensions,
        vector_backend=fence.vector_backend,
        vector_capable=True,
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"user_exists": False}, "notes_semantic_user_missing"),
        ({"owner_authorized": False}, "notes_semantic_owner_authority_revoked"),
        (
            {"semantic_manage_allowed": False},
            "notes_semantic_manage_permission_revoked",
        ),
        ({"desired_enabled": False}, "notes_semantic_index_disabled"),
        ({"owner_user_id": "other-owner"}, "notes_semantic_owner_authority_revoked"),
        ({"dataset_id": "other-dataset"}, "notes_semantic_owner_authority_revoked"),
        ({"capability_revision": "drift"}, "notes_semantic_capability_drift"),
        ({"disclosure_hash": "drift"}, "notes_semantic_disclosure_drift"),
        ({"configuration_revision": 5}, "notes_semantic_configuration_drift"),
        (
            {"generation_fencing_token": STALE_GENERATION_FENCE},
            "notes_semantic_generation_fence_mismatch",
        ),
        ({"generation_id": "other-generation"}, "notes_semantic_generation_fence_mismatch"),
        ({"provider": "other"}, "notes_semantic_provider_model_drift"),
        ({"model": "other"}, "notes_semantic_provider_model_drift"),
        ({"model_revision": "other"}, "notes_semantic_model_revision_drift"),
        ({"endpoint_policy_allowed": False}, "notes_semantic_endpoint_policy_denied"),
        ({"endpoint_origin_revision": "other"}, "notes_semantic_endpoint_drift"),
        ({"compatibility_hash": "other"}, "notes_semantic_compatibility_drift"),
        ({"dimensions": 3}, "notes_semantic_dimension_drift"),
        ({"vector_backend": "pgvector"}, "notes_semantic_vector_capability_drift"),
        ({"vector_capable": False}, "notes_semantic_vector_capability_drift"),
    ],
)
def test_complete_execution_fence_fails_closed_with_stable_codes(
    change: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(SemanticIndexingError) as exc_info:
        validate_execution_fence(_fence(), replace(_authority(), **change))

    assert exc_info.value.code == code
    assert str(exc_info.value) == code
    assert "owner-a" not in str(exc_info.value)
    assert "generation-a" not in str(exc_info.value)


def test_complete_execution_fence_accepts_the_exact_authoritative_identity() -> None:
    assert validate_execution_fence(_fence(), _authority()) == _authority()


def test_observability_exposes_only_allowlisted_low_cardinality_fields() -> None:
    metric = build_semantic_metric_event(
        operation="initial_build",
        status="degraded",
        backend="chromadb",
        error_code="note_failed",
        value=2,
    )
    audit = build_semantic_audit_event(
        event="generation_publication",
        status="degraded",
        reason="note_failed",
        counts={"indexed": 2, "excluded": 1, "failed": 1, "pending": 0},
    )

    assert metric.labels == {
        "operation": "initial_build",
        "status": "degraded",
        "backend": "chromadb",
        "error_code": "note_failed",
    }
    assert audit.fields == {
        "status": "degraded",
        "reason": "note_failed",
        "indexed": 2,
        "excluded": 1,
        "failed": 1,
        "pending": 0,
    }
    serialized = f"{metric!r}{audit!r}"
    for forbidden in (
        "owner-a",
        "dataset-a",
        "generation-a",
        "embedding-model-a",
        "https://",
        "collection",
        "table",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (build_semantic_metric_event, {"operation": "owner-a", "status": "success", "value": 1}),
        (
            build_semantic_audit_event,
            {"event": "generation_publication", "status": "success", "owner_id": "owner-a"},
        ),
    ],
)
def test_observability_rejects_unbounded_or_identifier_fields(builder, kwargs) -> None:
    with pytest.raises(SemanticObservationError):
        builder(**kwargs)
