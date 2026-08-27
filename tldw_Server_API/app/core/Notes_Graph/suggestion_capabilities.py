"""Stable provider disclosure contract for Notes graph suggestions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

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


@dataclass(frozen=True, slots=True)
class SuggestionCapabilityLimits:
    """Hard generation limits disclosed to the client and bound by revision."""

    max_candidates: int = 30
    max_relationships: int = 5
    max_tags: int = 5
    max_new_tags: int = 2
    max_tag_catalog: int = 100
    max_estimated_input_tokens: int = 24_000
    max_output_tokens: int = 2_000
    provider_timeout_seconds: int = 120
    response_candidates: int = 1


@dataclass(frozen=True, slots=True)
class ProviderCapabilityContract:
    """Resolved in-memory provider facts used for disclosure preflight."""

    adapter: str
    model: str
    endpoint_url: str = field(repr=False)
    call_policy: ProviderCallPolicy
    data_boundary: DataBoundary
    supports_one_attempt: bool
    enforces_same_origin_redirects: bool
    credentials_available: bool
    provider_healthy: bool
    outbound_data_categories: tuple[str, ...] = DEFAULT_OUTBOUND_DATA_CATEGORIES
    limits: SuggestionCapabilityLimits = SuggestionCapabilityLimits()
    prompt_contract_version: str = PROMPT_CONTRACT_VERSION
    health_heartbeat: str | None = field(default=None, repr=False)
    allowed_actions: tuple[str, ...] = DEFAULT_ALLOWED_ACTIONS

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
    }


def _availability(contract: ProviderCapabilityContract) -> tuple[bool, str | None]:
    if not contract.supports_one_attempt:
        return False, "notes_graph_provider_retry_policy_unsupported"
    policy = contract.call_policy
    if policy.max_transport_attempts != 1:
        return False, "notes_graph_provider_retry_policy_unsupported"
    if not contract.enforces_same_origin_redirects:
        return False, "notes_graph_provider_redirect_policy_unsupported"
    if (
        policy.allow_streaming is not False
        or policy.allow_tools is not False
        or policy.allow_stop is not False
        or policy.allow_response_format not in {False, True}
        or policy.candidate_count != 1
        or not policy.privacy_safe_errors
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

    endpoint_revision = canonical_endpoint_origin_digest(contract.endpoint_url)
    revision_payload = {
        "adapter": contract.adapter,
        "model": contract.model,
        "endpoint_origin_revision": endpoint_revision,
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


__all__ = [
    "DEFAULT_ALLOWED_ACTIONS",
    "DEFAULT_OUTBOUND_DATA_CATEGORIES",
    "PROMPT_CONTRACT_VERSION",
    "ProviderCapabilityContract",
    "SuggestionCapabilities",
    "SuggestionCapabilityLimits",
    "build_suggestion_capabilities",
    "canonical_endpoint_origin_digest",
]
