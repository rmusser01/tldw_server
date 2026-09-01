from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.notes_semantic_index import (
    SemanticCapabilitiesResponse,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_capabilities import (
    SemanticCapabilityContract,
    build_semantic_capabilities,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    PendingSemanticConfig,
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


def _public_capability(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "active_note_count": 1,
        "estimated_chunk_count": 2,
        "estimated_run_count": 1,
        "provider_label": "OpenAI compatible",
        "model": "embedding-model",
        "endpoint_display": "https://embed.example.test:8443",
        "execution_boundary": "external",
        "storage_boundary": "local",
        "storage_label": "ChromaDB",
        "outbound_data_categories": ("note_content_chunks", "note_title"),
        "capability_revision": "capability-v1",
        "indexing_available": True,
        "unavailable_reason": None,
        "metric": "cosine",
        "resolved_dimensions": 768,
        "dimension_probe_required": False,
        "renewal_requires_delete": False,
        "manage_authorized": True,
    }
    values.update(overrides)
    return values


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


def test_capabilities_preserve_brackets_in_sanitized_ipv6_origin() -> None:
    capabilities = build_semantic_capabilities(
        _contract(
            endpoint_url="https://user:secret@[2001:db8::1]:8443/path?token=secret"
        )
    )

    assert capabilities.endpoint_display == "https://[2001:db8::1]:8443"
    response = SemanticCapabilitiesResponse.model_validate(
        _public_capability(endpoint_display=capabilities.endpoint_display)
    )
    assert response.endpoint_display == "https://[2001:db8::1]:8443"


@pytest.mark.parametrize(
    ("endpoint_url", "expected_origin"),
    [
        (
            "https://user:secret@[2001:0db8::1]:8443/v1?token=secret",
            "https://[2001:db8::1]:8443",
        ),
        (
            "HTTPS://user:secret@BÜCHER.Example:8443/v1?token=secret",
            "https://xn--bcher-kva.example:8443",
        ),
    ],
    ids=["ipv6", "idn"],
)
def test_capability_origin_passes_public_schema_and_pending_worker_authority(
    endpoint_url: str,
    expected_origin: str,
) -> None:
    capabilities = build_semantic_capabilities(
        _contract(endpoint_url=endpoint_url)
    )
    response = SemanticCapabilitiesResponse.model_validate(
        _public_capability(endpoint_display=capabilities.endpoint_display)
    )
    pending = PendingSemanticConfig(
        provider="openai",
        model=response.model,
        model_revision=None,
        endpoint_origin=response.endpoint_display,
        credential_source="server_default",
        consented=True,
        dimensions=response.resolved_dimensions,
    )

    assert capabilities.endpoint_display == expected_origin
    assert response.endpoint_display == expected_origin
    assert pending.endpoint_origin == expected_origin


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
    assert capabilities.vector_backend == "chromadb"
    assert capabilities.dimension_probe_required is True
    assert capabilities.indexing_available is True
    assert capabilities.unavailable_reason is None


def test_pending_dimension_capability_revision_binds_vector_storage_identity() -> None:
    chromadb = build_semantic_capabilities(
        _contract(vector_backend="chromadb", resolved_dimensions=None)
    )
    pgvector = build_semantic_capabilities(
        _contract(vector_backend="pgvector", resolved_dimensions=None)
    )

    assert chromadb.compatibility_hash is None
    assert pgvector.compatibility_hash is None
    assert chromadb.disclosure_hash != pgvector.disclosure_hash
    assert chromadb.capability_revision != pgvector.capability_revision


def test_known_unsupported_pgvector_dimensions_remain_unavailable() -> None:
    capabilities = build_semantic_capabilities(
        _contract(vector_backend="pgvector", resolved_dimensions=3_072)
    )

    assert capabilities.vector_backend == "pgvector"
    assert capabilities.dimension_probe_required is False
    assert capabilities.indexing_available is False
    assert (
        capabilities.unavailable_reason
        == "notes_semantic_pgvector_dimensions_unsupported"
    )


@pytest.mark.parametrize(
    "field",
    ["provider_label", "model", "storage_label", "endpoint_display"],
)
def test_public_capability_schema_rejects_blank_consent_identity(field: str) -> None:
    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(_public_capability(**{field: "  "}))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:secret@embed.example.test",
        "https://embed.example.test/v1/embeddings",
        "https://embed.example.test?token=secret",
        "https://embed.example.test#secret",
    ],
)
def test_public_capability_schema_rejects_unsanitized_endpoint(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(
            _public_capability(endpoint_display=endpoint)
        )


def test_public_capability_schema_requires_exact_outbound_and_dimension_identity() -> None:
    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(
            _public_capability(outbound_data_categories=("note_title",))
        )
    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(
            _public_capability(
                resolved_dimensions=None,
                dimension_probe_required=False,
            )
        )

    pending = SemanticCapabilitiesResponse.model_validate(
        _public_capability(
            resolved_dimensions=None,
            dimension_probe_required=True,
        )
    )
    assert pending.endpoint_display == "https://embed.example.test:8443"


def test_public_capability_schema_accepts_unresolved_dimensions_when_unavailable() -> None:
    unavailable = SemanticCapabilitiesResponse.model_validate(
        _public_capability(
            indexing_available=False,
            unavailable_reason="notes_semantic_provider_unavailable",
            resolved_dimensions=None,
            dimension_probe_required=False,
        )
    )

    assert unavailable.indexing_available is False
    assert unavailable.dimension_probe_required is False


def test_public_capability_schema_accepts_missing_endpoint_only_when_unavailable() -> None:
    unavailable = SemanticCapabilitiesResponse.model_validate(
        _public_capability(
            endpoint_display=None,
            indexing_available=False,
            unavailable_reason="notes_semantic_endpoint_unavailable",
            resolved_dimensions=None,
            dimension_probe_required=False,
        )
    )

    assert unavailable.endpoint_display is None

    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(
            _public_capability(endpoint_display=None)
        )
    with pytest.raises(ValidationError):
        SemanticCapabilitiesResponse.model_validate(
            _public_capability(
                endpoint_display=None,
                indexing_available=False,
                unavailable_reason=None,
                resolved_dimensions=None,
                dimension_probe_required=False,
            )
        )
