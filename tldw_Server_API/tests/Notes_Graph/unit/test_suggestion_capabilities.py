from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.Notes_Graph.suggestion_capabilities import (
    DEFAULT_OUTBOUND_DATA_CATEGORIES,
    PROMPT_CONTRACT_VERSION,
    ProviderCapabilityContract,
    SuggestionCapabilityLimits,
    build_suggestion_capabilities,
    canonical_endpoint_origin_digest,
)

pytestmark = pytest.mark.unit


def _contract(**overrides: object) -> ProviderCapabilityContract:
    values: dict[str, object] = {
        "adapter": "openai",
        "model": "gpt-test",
        "endpoint_url": "https://API.Example.test:443/v1/chat/completions",
        "call_policy": ProviderCallPolicy(
            max_transport_attempts=1,
            allow_streaming=False,
            allow_tools=False,
            allow_stop=False,
            allow_response_format=True,
            candidate_count=1,
            privacy_safe_errors=True,
        ),
        "data_boundary": "remote",
        "outbound_data_categories": DEFAULT_OUTBOUND_DATA_CATEGORIES,
        "limits": SuggestionCapabilityLimits(),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "supports_one_attempt": True,
        "enforces_same_origin_redirects": True,
        "credentials_available": True,
        "provider_healthy": True,
        "health_heartbeat": "heartbeat-a",
    }
    values.update(overrides)
    return ProviderCapabilityContract(**values)  # type: ignore[arg-type]


def test_revision_covers_only_approved_stable_fields() -> None:
    baseline = build_suggestion_capabilities(_contract())
    stable_changes = (
        {"adapter": "anthropic"},
        {"model": "model-b"},
        {"endpoint_url": "https://other.example.test/v1"},
        {
            "call_policy": replace(
                _contract().call_policy,
                allow_response_format=False,
            )
        },
        {"data_boundary": "local"},
        {
            "outbound_data_categories": (
                *DEFAULT_OUTBOUND_DATA_CATEGORIES,
                "future_approved_category",
            )
        },
        {"limits": replace(SuggestionCapabilityLimits(), max_candidates=29)},
        {"prompt_contract_version": "notes-graph-suggestion-prompt-v2"},
    )

    for change in stable_changes:
        changed = build_suggestion_capabilities(_contract(**change))
        assert changed.revision != baseline.revision

    assert build_suggestion_capabilities(_contract(credentials_available=False)).revision == baseline.revision
    assert (
        build_suggestion_capabilities(_contract(provider_healthy=False, health_heartbeat="heartbeat-b")).revision
        == baseline.revision
    )


def test_endpoint_revision_uses_canonical_origin_without_exposing_url() -> None:
    first = canonical_endpoint_origin_digest("HTTPS://API.ExAmPlE.test:443/v1/chat/completions?secret=query")
    equivalent = canonical_endpoint_origin_digest("https://api.example.test/other")
    changed_port = canonical_endpoint_origin_digest("https://api.example.test:8443/other")

    assert first == equivalent
    assert first != changed_port
    assert "example" not in first
    assert "secret" not in first

    capability = build_suggestion_capabilities(_contract())
    rendered = repr(capability)
    assert "API.Example" not in rendered
    assert capability.endpoint_origin_revision == first


@pytest.mark.parametrize(
    ("data_boundary", "disclosure_external"),
    [("local", False), ("remote", True), ("unknown", True)],
)
def test_unknown_data_boundary_is_disclosed_as_external(
    data_boundary: str,
    disclosure_external: bool,
) -> None:
    capability = build_suggestion_capabilities(_contract(data_boundary=data_boundary))

    assert capability.data_boundary == data_boundary
    assert capability.disclosure_external is disclosure_external


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"supports_one_attempt": False},
            "notes_graph_provider_retry_policy_unsupported",
        ),
        (
            {"enforces_same_origin_redirects": False},
            "notes_graph_provider_redirect_policy_unsupported",
        ),
        (
            {"credentials_available": False},
            "notes_graph_provider_not_configured",
        ),
        (
            {"provider_healthy": False, "health_heartbeat": "private-health"},
            "notes_graph_provider_unavailable",
        ),
    ],
)
def test_unavailable_reasons_are_stable_and_safe(
    changes: dict[str, object],
    reason: str,
) -> None:
    capability = build_suggestion_capabilities(_contract(**changes))

    assert capability.generation_available is False
    assert capability.unavailable_reason == reason
    assert "private" not in repr(capability)
    assert "http" not in repr(capability)


def test_default_disclosure_and_effective_limits_are_exact() -> None:
    capability = build_suggestion_capabilities(_contract())

    assert capability.generation_available is True
    assert capability.unavailable_reason is None
    assert capability.outbound_data_categories == (
        "selected_note_title",
        "selected_note_excerpts",
        "candidate_note_titles",
        "candidate_note_excerpts",
        "existing_tag_labels",
    )
    assert capability.limits == SuggestionCapabilityLimits(
        max_candidates=30,
        max_relationships=5,
        max_tags=5,
        max_new_tags=2,
        max_tag_catalog=100,
        max_estimated_input_tokens=24_000,
        max_output_tokens=2_000,
        provider_timeout_seconds=120,
        response_candidates=1,
    )
    assert capability.allowed_actions == (
        "generate",
        "cancel",
        "accept",
        "reject",
        "reset_rejections",
    )


def test_transport_and_readiness_facts_must_be_explicit() -> None:
    with pytest.raises(TypeError):
        ProviderCapabilityContract(
            adapter="openai",
            model="gpt-test",
            endpoint_url="https://example.test/v1",
            call_policy=_contract().call_policy,
            data_boundary="remote",
        )


@pytest.mark.parametrize(
    ("policy_change", "reason"),
    [
        (
            {"max_transport_attempts": 2},
            "notes_graph_provider_retry_policy_unsupported",
        ),
        (
            {"allow_streaming": True},
            "notes_graph_provider_call_policy_unsupported",
        ),
        (
            {"allow_tools": True},
            "notes_graph_provider_call_policy_unsupported",
        ),
        (
            {"allow_stop": True},
            "notes_graph_provider_call_policy_unsupported",
        ),
        (
            {"candidate_count": 2},
            "notes_graph_provider_call_policy_unsupported",
        ),
        (
            {"privacy_safe_errors": False},
            "notes_graph_provider_call_policy_unsupported",
        ),
    ],
)
def test_capability_preflight_verifies_effective_call_policy(
    policy_change: dict[str, object],
    reason: str,
) -> None:
    policy = replace(_contract().call_policy, **policy_change)
    capability = build_suggestion_capabilities(_contract(call_policy=policy))

    assert capability.generation_available is False
    assert capability.unavailable_reason == reason
