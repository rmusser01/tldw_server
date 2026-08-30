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
    baseline = build_semantic_capabilities(_contract())
    rotated = build_semantic_capabilities(_contract(credential_source="durable"))

    assert baseline.compatibility_hash == rotated.compatibility_hash
    assert baseline.disclosure_hash == rotated.disclosure_hash
    assert baseline.capability_revision == rotated.capability_revision


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
