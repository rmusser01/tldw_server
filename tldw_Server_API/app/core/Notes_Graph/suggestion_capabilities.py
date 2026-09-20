"""Stable provider disclosure contract for Notes graph suggestions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

PROMPT_CONTRACT_VERSION = "notes-graph-suggestion-prompt-v1"
DEFAULT_OUTBOUND_DATA_CATEGORIES = (
    "selected_note_title",
    "selected_note_excerpts",
    "candidate_note_titles",
    "candidate_note_excerpts",
    "existing_tag_labels",
)
DEFAULT_ALLOWED_ACTIONS = (
    "generate",
    "cancel",
    "accept",
    "reject",
    "reset_rejections",
)

DataBoundary = Literal["local", "remote", "unknown"]

_HARD_LIMIT_VALUES = {
    "max_candidates": 30,
    "max_relationships": 5,
    "max_tags": 5,
    "max_new_tags": 2,
    "max_tag_catalog": 100,
    "max_estimated_input_tokens": 24_000,
    "max_output_tokens": 2_000,
    "provider_timeout_seconds": 120,
    "response_candidates": 1,
}
_UNAVAILABLE_ENDPOINT_FACTS = b'{"configured":false}'


@dataclass(frozen=True, slots=True)
class SuggestionCapabilityLimits:
    """Hard generation limits disclosed to the client and bound by revision."""

    max_candidates: int = _HARD_LIMIT_VALUES["max_candidates"]
    max_relationships: int = _HARD_LIMIT_VALUES["max_relationships"]
    max_tags: int = _HARD_LIMIT_VALUES["max_tags"]
    max_new_tags: int = _HARD_LIMIT_VALUES["max_new_tags"]
    max_tag_catalog: int = _HARD_LIMIT_VALUES["max_tag_catalog"]
    max_estimated_input_tokens: int = _HARD_LIMIT_VALUES["max_estimated_input_tokens"]
    max_output_tokens: int = _HARD_LIMIT_VALUES["max_output_tokens"]
    provider_timeout_seconds: int = _HARD_LIMIT_VALUES["provider_timeout_seconds"]
    response_candidates: int = _HARD_LIMIT_VALUES["response_candidates"]

    def __post_init__(self) -> None:
        for field_name, hard_maximum in _HARD_LIMIT_VALUES.items():
            value = getattr(self, field_name)
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            if value > hard_maximum:
                raise ValueError(f"{field_name} exceeds its hard maximum")
        if self.response_candidates != 1:
            raise ValueError("response_candidates must equal 1")


HARD_SUGGESTION_CAPABILITY_LIMITS = SuggestionCapabilityLimits()


@dataclass(frozen=True, slots=True)
class ProviderCapabilityContract:
    """Resolved in-memory provider facts used for disclosure preflight."""

    adapter: str
    model: str
    endpoint_url: str | None = field(repr=False)
    call_policy: ProviderCallPolicy
    data_boundary: DataBoundary
    credentials_available: bool
    provider_healthy: bool
    outbound_data_categories: tuple[str, ...] = DEFAULT_OUTBOUND_DATA_CATEGORIES
    limits: SuggestionCapabilityLimits = SuggestionCapabilityLimits()
    prompt_contract_version: str = PROMPT_CONTRACT_VERSION
    health_heartbeat: str | None = field(default=None, repr=False)
    allowed_actions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.data_boundary not in {"local", "remote", "unknown"}:
            raise ValueError("data_boundary must be local, remote, or unknown")


@dataclass(frozen=True, slots=True)
class SuggestionCapabilities:
    """Sanitized capability disclosure without endpoint or credential material."""

    provider: str
    model: str
    endpoint_origin_revision: str
    data_boundary: DataBoundary
    disclosure_external: bool
    outbound_data_categories: tuple[str, ...]
    generation_available: bool
    unavailable_reason: str | None
    limits: SuggestionCapabilityLimits
    allowed_actions: tuple[str, ...]
    revision: str


def canonical_endpoint_origin_digest(endpoint_url: str) -> str:
    """Hash the canonical configured origin without retaining URL material."""

    scope = ConfiguredEndpointScope.from_url(endpoint_url)
    canonical = f"{scope.scheme}://{scope.host}:{scope.port}".encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _endpoint_origin_revision(endpoint_url: str | None) -> str:
    if endpoint_url is None:
        return f"sha256:{hashlib.sha256(_UNAVAILABLE_ENDPOINT_FACTS).hexdigest()}"
    return canonical_endpoint_origin_digest(endpoint_url)


def _scope_origin_digest(scope: ConfiguredEndpointScope | None) -> str | None:
    if scope is None:
        return None
    canonical = f"{scope.scheme}://{scope.host}:{scope.port}".encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _policy_revision_fields(policy: ProviderCallPolicy) -> dict[str, object]:
    return {
        "max_transport_attempts": policy.max_transport_attempts,
        "allow_streaming": policy.allow_streaming,
        "allow_tools": policy.allow_tools,
        "allow_stop": policy.allow_stop,
        "allow_response_format": policy.allow_response_format,
        "candidate_count": policy.candidate_count,
        "temperature": policy.temperature,
        "top_p": policy.top_p,
        "privacy_safe_errors": policy.privacy_safe_errors,
        "maximum_timeout_seconds": policy.maximum_timeout_seconds,
        "required_endpoint_origin_revision": _scope_origin_digest(policy.required_endpoint_scope),
    }


def _availability(contract: ProviderCapabilityContract) -> tuple[bool, str | None]:
    if contract.unavailable_reason is not None:
        return False, contract.unavailable_reason
    policy = contract.call_policy
    if policy.max_transport_attempts != 1:
        return False, "notes_graph_provider_retry_policy_unsupported"
    transport = get_registry().get_audited_call_policy_transport(contract.adapter)
    if (
        transport is None
        or transport.maximum_transport_attempts != 1
        or not transport.enforces_configured_endpoint_scope
        or not transport.enforces_maximum_timeout
    ):
        return False, "notes_graph_provider_call_policy_unsupported"
    if contract.endpoint_url is None:
        return False, "notes_graph_provider_call_policy_unsupported"
    try:
        endpoint_scope = ConfiguredEndpointScope.from_url(contract.endpoint_url)
    except ValueError:
        return False, "notes_graph_provider_call_policy_unsupported"
    if (
        policy.allow_streaming is not False
        or policy.allow_tools is not False
        or policy.allow_stop is not False
        or policy.allow_response_format not in {False, True}
        or policy.candidate_count != 1
        or not policy.privacy_safe_errors
        or policy.required_endpoint_scope != endpoint_scope
        or policy.maximum_timeout_seconds != contract.limits.provider_timeout_seconds
    ):
        return False, "notes_graph_provider_call_policy_unsupported"
    if not contract.credentials_available:
        return False, "notes_graph_provider_not_configured"
    if not contract.provider_healthy:
        return False, "notes_graph_provider_unavailable"
    return True, None


def build_suggestion_capabilities(
    contract: ProviderCapabilityContract,
) -> SuggestionCapabilities:
    """Build one deterministic sanitized provider disclosure snapshot."""

    endpoint_revision = _endpoint_origin_revision(contract.endpoint_url)
    revision_payload = {
        "adapter": contract.adapter,
        "model": contract.model,
        "endpoint_configured": contract.endpoint_url is not None,
        "endpoint_origin_revision": endpoint_revision,
        "call_policy_configured": contract.call_policy.required_endpoint_scope is not None,
        "call_policy": _policy_revision_fields(contract.call_policy),
        "data_boundary": contract.data_boundary,
        "outbound_data_categories": list(contract.outbound_data_categories),
        "limits": asdict(contract.limits),
        "prompt_contract_version": contract.prompt_contract_version,
    }
    revision_bytes = json.dumps(
        revision_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    generation_available, unavailable_reason = _availability(contract)
    return SuggestionCapabilities(
        provider=contract.adapter,
        model=contract.model,
        endpoint_origin_revision=endpoint_revision,
        data_boundary=contract.data_boundary,
        disclosure_external=contract.data_boundary != "local",
        outbound_data_categories=contract.outbound_data_categories,
        generation_available=generation_available,
        unavailable_reason=unavailable_reason,
        limits=contract.limits,
        allowed_actions=contract.allowed_actions,
        revision=f"sha256:{hashlib.sha256(revision_bytes).hexdigest()}",
    )


def build_unavailable_suggestion_capabilities(
    *,
    provider: str | None,
    model: str | None,
    reason: str = "notes_graph_provider_disallowed",
) -> SuggestionCapabilities:
    """Build a canonical content-free disclosure for unresolved provider facts."""

    safe_provider = provider.strip() if isinstance(provider, str) and provider.strip() else "unconfigured"
    safe_model = model.strip() if isinstance(model, str) and model.strip() else "unconfigured"
    return build_suggestion_capabilities(
        ProviderCapabilityContract(
            adapter=safe_provider,
            model=safe_model,
            endpoint_url=None,
            call_policy=ProviderCallPolicy(),
            data_boundary="unknown",
            credentials_available=False,
            provider_healthy=False,
            unavailable_reason=reason,
        )
    )


__all__ = [
    "DEFAULT_ALLOWED_ACTIONS",
    "DEFAULT_OUTBOUND_DATA_CATEGORIES",
    "HARD_SUGGESTION_CAPABILITY_LIMITS",
    "PROMPT_CONTRACT_VERSION",
    "ProviderCapabilityContract",
    "SuggestionCapabilities",
    "SuggestionCapabilityLimits",
    "build_suggestion_capabilities",
    "build_unavailable_suggestion_capabilities",
    "canonical_endpoint_origin_digest",
]
