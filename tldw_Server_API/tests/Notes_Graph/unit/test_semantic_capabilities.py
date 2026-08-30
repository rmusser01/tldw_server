from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Notes_Graph.semantic_capabilities import (
    SemanticCapabilityContract,
    build_semantic_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings

pytestmark = pytest.mark.unit


def _contract(**overrides: object) -> SemanticCapabilityContract:
    values: dict[str, object] = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "model_revision": "2026-08-01",
        "endpoint_url": "https://embed.example.test/v1/embeddings?api_key=secret",
        "execution_boundary": "external",
        "vector_backend": "chromadb",
        "storage_boundary": "local",
        "resolved_dimensions": 768,
        "normalization_version": "unicode-nfc-v1",
        "chunker_version": "notes-semantic-v1",
        "credential_source": "durable",
        "provider_healthy": True,
        "vector_storage_available": True,
        "active_note_count": 3,
    }
    values.update(overrides)
    return SemanticCapabilityContract(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "cohere"},
        {"model": "text-embedding-3-large"},
        {"model_revision": "2026-08-02"},
        {"vector_backend": "pgvector"},
        {"metric": "dot"},
        {"resolved_dimensions": 384},
        {"normalization_version": "unicode-nfc-v2"},
        {"chunker_version": "notes-semantic-v2"},
    ],
)
def test_compatibility_hash_changes_only_for_compatibility_identity(
    change: dict[str, object],
) -> None:
    baseline = build_semantic_capabilities(_contract())
    changed = build_semantic_capabilities(_contract(**change))

    assert baseline.compatibility_hash != changed.compatibility_hash


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint_url": "https://other.example.test/v1/embeddings"},
        {"execution_boundary": "local"},
        {"storage_boundary": "external"},
        {"outbound_data_categories": ("note_title",)},
    ],
)
def test_disclosure_hash_changes_for_disclosure_identity(
    change: dict[str, object],
) -> None:
    baseline = build_semantic_capabilities(_contract())
    changed = build_semantic_capabilities(_contract(**change))

    assert baseline.disclosure_hash != changed.disclosure_hash


def test_credential_rotation_does_not_change_semantic_identity() -> None:
    baseline = build_semantic_capabilities(
        _contract(credential_rotation_revision="rotation-2026-08-01")
    )
    rotated = build_semantic_capabilities(
        _contract(credential_rotation_revision="rotation-2026-08-29")
    )

    assert baseline.compatibility_hash == rotated.compatibility_hash
    assert baseline.disclosure_hash == rotated.disclosure_hash
    assert baseline.capability_revision == rotated.capability_revision


def test_credential_rotation_revision_rejects_secret_shaped_values() -> None:
    with pytest.raises(
        ValueError,
        match="credential_rotation_revision is invalid",
    ) as exc_info:
        _contract(credential_rotation_revision="sk-secret-credential-value")
    assert "sk-secret-credential-value" not in str(exc_info.value)


def test_model_and_revision_are_independent_identity_fields() -> None:
    baseline = build_semantic_capabilities(_contract())
    changed_model = build_semantic_capabilities(
        _contract(model="text-embedding-3-large", model_revision="2026-08-01")
    )
    changed_revision = build_semantic_capabilities(
        _contract(model="text-embedding-3-small", model_revision="2026-08-02")
    )

    assert baseline.compatibility_hash != changed_model.compatibility_hash
    assert baseline.disclosure_hash != changed_model.disclosure_hash
    assert baseline.compatibility_hash != changed_revision.compatibility_hash
    assert baseline.disclosure_hash != changed_revision.disclosure_hash


@pytest.mark.parametrize(
    "model",
    ["text-embedding-3-small", "org/model", "model:v1.5"],
)
def test_model_label_policy_accepts_common_stable_identifiers(model: str) -> None:
    capabilities = build_semantic_capabilities(_contract(model=model))

    assert capabilities.model == model
    assert capabilities.model_revision == "2026-08-01"
    assert capabilities.indexing_available is True


@pytest.mark.parametrize(
    "model",
    [
        "model name",
        "model\tname",
        "https://embed.example.test/model",
        "user:password@example.test/model",
        "org/model?api_key=credential",
        "org/model#revision",
        "/private/models/model",
        r"C:\\private\\models\\model",
        "sk-secret-credential-value",
        "org/sk-secret-credential-value",
        "model$untrusted",
        "m" * 257,
    ],
)
def test_model_label_policy_rejects_untrusted_disclosure(model: str) -> None:
    contract = _contract(model=model)
    rejected = build_semantic_capabilities(contract)
    canonical = build_semantic_capabilities(_contract(model="unconfigured"))

    assert rejected.model == "unconfigured"
    assert rejected.compatibility_hash == canonical.compatibility_hash
    assert rejected.disclosure_hash == canonical.disclosure_hash
    assert rejected.capability_revision == canonical.capability_revision
    assert rejected.indexing_available is False
    assert rejected.unavailable_reason == "notes_semantic_provider_unavailable"
    assert model not in repr(rejected)
    assert model not in repr(contract)


def test_model_revision_policy_rejects_untrusted_disclosure() -> None:
    revision = "https://user:credential@example.test/model?token=credential"
    contract = _contract(model_revision=revision)
    rejected = build_semantic_capabilities(contract)
    canonical = build_semantic_capabilities(_contract(model_revision=None))

    assert rejected.model_revision is None
    assert rejected.compatibility_hash == canonical.compatibility_hash
    assert rejected.disclosure_hash == canonical.disclosure_hash
    assert rejected.capability_revision == canonical.capability_revision
    assert rejected.indexing_available is False
    assert rejected.unavailable_reason == "notes_semantic_provider_unavailable"
    assert revision not in repr(rejected)
    assert revision not in repr(contract)


def test_capabilities_sanitize_endpoint_and_storage_provider_labels() -> None:
    capabilities = build_semantic_capabilities(
        _contract(
            endpoint_url="https://user:secret@embed.example.test:8443/path?token=secret#fragment",
            vector_backend="pgvector",
        ),
        settings=SemanticIndexSettings(pgvector_allowed_dimensions=frozenset({768})),
    )

    assert capabilities.endpoint_display == "https://embed.example.test:8443"
    assert capabilities.provider_label == "OpenAI"
    assert capabilities.storage_label == "pgvector"
    assert "secret" not in repr(capabilities)
    assert "/path" not in capabilities.endpoint_display


def test_unknown_boundaries_fail_closed_and_request_credentials_are_not_durable() -> None:
    capabilities = build_semantic_capabilities(
        _contract(
            execution_boundary="unknown",
            storage_boundary="unknown",
            credential_source="request",
        )
    )

    assert capabilities.execution_boundary == "external"
    assert capabilities.storage_boundary == "unavailable"
    assert capabilities.indexing_available is False
    assert capabilities.unavailable_reason == "notes_semantic_durable_credentials_unavailable"


def test_capabilities_defer_compatibility_hash_until_dimensions_resolve() -> None:
    capabilities = build_semantic_capabilities(_contract(resolved_dimensions=None))

    assert capabilities.compatibility_hash is None
    assert capabilities.indexing_available is False
    assert capabilities.unavailable_reason == "notes_semantic_dimensions_pending"
