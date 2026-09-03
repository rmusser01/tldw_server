"""Pure, deterministic Notes semantic-index capability policy."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import ceil
from typing import Any, Literal

from .semantic_endpoint import canonical_semantic_endpoint_origin
from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings

ExecutionBoundary = Literal["local", "external", "unknown"]
StorageBoundary = Literal["local", "external", "unavailable", "unknown"]
CredentialSource = Literal["durable", "request", "none"]


@dataclass(frozen=True, slots=True)
class SemanticProviderPolicy:
    """Executable Notes adapter and endpoint policy."""

    label: str
    config_section: str
    default_base_url: str


_PROVIDER_CATALOG = {
    "google": SemanticProviderPolicy(
        label="Google",
        config_section="google_api",
        default_base_url="https://generativelanguage.googleapis.com/v1",
    ),
    "huggingface": SemanticProviderPolicy(
        label="HuggingFace",
        config_section="huggingface_api",
        default_base_url="https://api-inference.huggingface.co/models",
    ),
    "openai": SemanticProviderPolicy(
        label="OpenAI",
        config_section="openai_api",
        default_base_url="https://api.openai.com/v1",
    ),
}
_PROVIDER_LABELS = {provider: policy.label for provider, policy in _PROVIDER_CATALOG.items()}
_ENDPOINT_FIELDS = (
    "base_url",
    "api_base_url",
    "api_base",
    "api_url",
    "api_ip",
    "endpoint",
    "runtime_endpoint",
)
_STORAGE_LABELS = {"chromadb": "ChromaDB", "pgvector": "pgvector"}
_OUTBOUND_DATA_CATEGORIES = frozenset({"note_content_chunks", "note_title"})
_UNAVAILABLE_ENDPOINT_FACTS = b'{"configured":false}'
_MODEL_LABEL_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?\Z"
)
_ROTATION_REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CREDENTIAL_MARKER_PATTERN = re.compile(
    r"(?:^|[._:/-])"
    r"(?:api[-_]?key|apikey|bearer|password|secret|sk|token)"
    r"(?:[._:/=-]|$)",
    re.IGNORECASE,
)


def _looks_credential_bearing(value: str) -> bool:
    return _CREDENTIAL_MARKER_PATTERN.search(value) is not None


def _safe_model_label(value: str | None) -> str | None:
    if not isinstance(value, str) or len(value) > 256:
        return None
    if (
        value.lower() == "unconfigured"
        or _looks_credential_bearing(value)
        or _MODEL_LABEL_PATTERN.fullmatch(value) is None
    ):
        return None
    return value


def _valid_rotation_revision(value: str | None) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, str)
        and not _looks_credential_bearing(value)
        and _ROTATION_REVISION_PATTERN.fullmatch(value) is not None
    )


@dataclass(frozen=True, slots=True)
class SemanticCapabilityContract:
    """Already-resolved, non-secret facts used to disclose semantic capability."""

    provider: str
    model: str = field(repr=False)
    model_revision: str | None = field(default=None, repr=False)
    endpoint_url: str | None = field(default=None, repr=False)
    execution_boundary: ExecutionBoundary = "unknown"
    vector_backend: str = ""
    storage_boundary: StorageBoundary = "unknown"
    metric: str = "cosine"
    resolved_dimensions: int | None = None
    normalization_version: str = ""
    chunker_version: str = ""
    credential_source: CredentialSource = "none"
    credential_rotation_revision: str | None = field(default=None, repr=False)
    provider_healthy: bool = False
    vector_storage_available: bool = False
    active_note_count: int = 0
    outbound_data_categories: tuple[str, ...] = ("note_title", "note_content_chunks")

    def __post_init__(self) -> None:
        if self.execution_boundary not in {"local", "external", "unknown"}:
            raise ValueError("execution_boundary is invalid")
        if self.storage_boundary not in {"local", "external", "unavailable", "unknown"}:
            raise ValueError("storage_boundary is invalid")
        if self.credential_source not in {"durable", "request", "none"}:
            raise ValueError("credential_source is invalid")
        if not _valid_rotation_revision(self.credential_rotation_revision):
            raise ValueError("credential_rotation_revision is invalid")
        if type(self.active_note_count) is not int or self.active_note_count < 0:
            raise ValueError("active_note_count must be a non-negative integer")
        if self.resolved_dimensions is not None and (
            type(self.resolved_dimensions) is not int or self.resolved_dimensions <= 0
        ):
            raise ValueError("resolved_dimensions must be a positive integer or None")
        if not set(self.outbound_data_categories) <= _OUTBOUND_DATA_CATEGORIES:
            raise ValueError("outbound_data_categories contains an unapproved category")


@dataclass(frozen=True, slots=True)
class SemanticCapabilities:
    """Sanitized deterministic capability disclosure for Notes semantic indexing."""

    active_note_count: int
    estimated_chunk_count: int
    estimated_run_count: int
    provider_label: str
    model: str
    model_revision: str | None
    endpoint_display: str | None
    endpoint_origin_revision: str
    execution_boundary: Literal["local", "external"]
    storage_boundary: Literal["local", "external", "unavailable"]
    storage_label: str
    vector_backend: str
    outbound_data_categories: tuple[str, ...]
    durable_credential_available: bool
    compatibility_hash: str | None
    disclosure_hash: str
    capability_revision: str
    metric: str
    resolved_dimensions: int | None
    dimension_probe_required: bool
    effective_limits: SemanticIndexSettings
    indexing_available: bool
    unavailable_reason: str | None


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _safe_identifier(value: str | None, *, allowed: dict[str, str]) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    return normalized if normalized in allowed else "unavailable"


def semantic_provider_policy(provider: str | None) -> SemanticProviderPolicy | None:
    """Return the closed Notes policy for one normalized provider."""

    normalized = provider.strip().lower() if isinstance(provider, str) else ""
    return _PROVIDER_CATALOG.get(normalized)


def semantic_provider_label(provider: str | None) -> str:
    """Return the public label for persisted provider authority."""

    policy = semantic_provider_policy(provider)
    return policy.label if policy is not None else "unavailable"


def semantic_provider_is_registered(
    provider: str | None,
    registry: Any,
) -> bool:
    """Bind Notes admission to the executable production adapter registry."""

    policy = semantic_provider_policy(provider)
    if policy is None:
        return False
    normalized = str(provider).strip().lower()
    registered = getattr(registry, "has_adapter", None)
    if callable(registered):
        try:
            return bool(registered(normalized))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    specs = getattr(registry, "_adapter_specs", None)
    return isinstance(specs, Mapping) and normalized in specs


def resolve_semantic_provider_endpoint(
    provider: str | None,
    *,
    configured_url: object = None,
    app_config: Mapping[str, object] | None = None,
) -> str | None:
    """Resolve the exact adapter base URL under the closed endpoint policy."""

    policy = semantic_provider_policy(provider)
    if policy is None:
        return None
    candidates: list[object] = [configured_url]
    section = app_config.get(policy.config_section) if app_config is not None else None
    if isinstance(section, Mapping):
        candidates.extend(section.get(field_name) for field_name in _ENDPOINT_FIELDS)
    candidates.append(policy.default_base_url)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().rstrip("/")
    return None


def _endpoint_display(endpoint_url: str | None) -> str | None:
    return canonical_semantic_endpoint_origin(endpoint_url)


def semantic_capability_binding_matches(
    persisted_hash: str | None,
    current_hash: str | None,
) -> bool:
    """Match projector fail-safe semantics for unresolved live capabilities."""

    return current_hash is None or current_hash == persisted_hash


def _endpoint_origin_revision(endpoint_display: str | None) -> str:
    if endpoint_display is None:
        return f"sha256:{hashlib.sha256(_UNAVAILABLE_ENDPOINT_FACTS).hexdigest()}"
    return _canonical_hash({"origin": endpoint_display})


def _execution_boundary(value: ExecutionBoundary) -> Literal["local", "external"]:
    return "local" if value == "local" else "external"


def _storage_boundary(value: StorageBoundary) -> Literal["local", "external", "unavailable"]:
    if value in {"local", "external", "unavailable"}:
        return value
    return "unavailable"


def build_semantic_capabilities(
    contract: SemanticCapabilityContract,
    *,
    settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
) -> SemanticCapabilities:
    """Build semantic capability facts without provider or vector I/O."""

    provider_key = _safe_identifier(contract.provider, allowed=_PROVIDER_LABELS)
    backend_key = _safe_identifier(contract.vector_backend, allowed=_STORAGE_LABELS)
    provider_label = _PROVIDER_LABELS.get(provider_key, "unavailable")
    storage_label = _STORAGE_LABELS.get(backend_key, "unavailable")
    safe_model = _safe_model_label(contract.model)
    model = safe_model or "unconfigured"
    model_revision = _safe_model_label(contract.model_revision)
    model_revision_invalid = contract.model_revision is not None and model_revision is None
    endpoint_display = _endpoint_display(contract.endpoint_url)
    endpoint_origin_revision = _endpoint_origin_revision(endpoint_display)
    execution_boundary = _execution_boundary(contract.execution_boundary)
    storage_boundary = _storage_boundary(contract.storage_boundary)
    outbound_categories = tuple(sorted(contract.outbound_data_categories))
    compatibility_hash = None
    if contract.resolved_dimensions is not None:
        compatibility_hash = _canonical_hash(
            {
                "provider": provider_key,
                "model": model,
                "model_revision": model_revision,
                "vector_backend": backend_key,
                "metric": contract.metric,
                "resolved_dimensions": contract.resolved_dimensions,
                "normalization_version": contract.normalization_version,
                "chunker_version": contract.chunker_version,
            }
        )
    disclosure_hash = _canonical_hash(
        {
            "provider": provider_key,
            "model": model,
            "model_revision": model_revision,
            "endpoint_origin_revision": endpoint_origin_revision,
            "execution_boundary": execution_boundary,
            "vector_backend": backend_key,
            "storage_boundary": storage_boundary,
            "storage_label": storage_label,
            "outbound_data_categories": outbound_categories,
        }
    )
    estimated_chunks = min(contract.active_note_count, settings.max_active_notes) * settings.max_chunks_per_note
    estimated_runs = ceil(estimated_chunks / settings.max_chunks_per_run) if estimated_chunks else 0
    unavailable_reason: str | None = None
    if not settings.indexing_enabled:
        unavailable_reason = "notes_semantic_indexing_disabled"
    elif contract.active_note_count > settings.max_active_notes:
        unavailable_reason = "notes_semantic_active_note_limit_exceeded"
    elif contract.metric != "cosine":
        unavailable_reason = "notes_semantic_metric_unsupported"
    elif provider_key == "unavailable" or safe_model is None or model_revision_invalid:
        unavailable_reason = "notes_semantic_provider_unavailable"
    elif contract.credential_source != "durable":
        unavailable_reason = "notes_semantic_durable_credentials_unavailable"
    elif endpoint_display is None:
        unavailable_reason = "notes_semantic_endpoint_unavailable"
    elif not contract.provider_healthy:
        unavailable_reason = "notes_semantic_provider_unavailable"
    elif backend_key == "unavailable" or storage_boundary == "unavailable" or not contract.vector_storage_available:
        unavailable_reason = "notes_semantic_vector_storage_unavailable"
    elif (
        backend_key == "pgvector"
        and contract.resolved_dimensions is not None
        and contract.resolved_dimensions not in settings.pgvector_allowed_dimensions
    ):
        unavailable_reason = "notes_semantic_pgvector_dimensions_unsupported"
    dimension_probe_required = contract.resolved_dimensions is None and unavailable_reason is None
    limits_payload = asdict(settings)
    limits_payload["pgvector_allowed_dimensions"] = sorted(settings.pgvector_allowed_dimensions)
    capability_revision = _canonical_hash(
        {
            "compatibility_hash": compatibility_hash,
            "disclosure_hash": disclosure_hash,
            "effective_limits": limits_payload,
            "credential_source": contract.credential_source,
        }
    )
    return SemanticCapabilities(
        active_note_count=contract.active_note_count,
        estimated_chunk_count=estimated_chunks,
        estimated_run_count=estimated_runs,
        provider_label=provider_label,
        model=model,
        model_revision=model_revision,
        endpoint_display=endpoint_display,
        endpoint_origin_revision=endpoint_origin_revision,
        execution_boundary=execution_boundary,
        storage_boundary=storage_boundary,
        storage_label=storage_label,
        vector_backend=backend_key,
        outbound_data_categories=outbound_categories,
        durable_credential_available=contract.credential_source == "durable",
        compatibility_hash=compatibility_hash,
        disclosure_hash=disclosure_hash,
        capability_revision=capability_revision,
        metric=contract.metric,
        resolved_dimensions=contract.resolved_dimensions,
        dimension_probe_required=dimension_probe_required,
        effective_limits=settings,
        indexing_available=unavailable_reason is None,
        unavailable_reason=unavailable_reason,
    )


__all__ = [
    "SemanticCapabilities",
    "SemanticCapabilityContract",
    "build_semantic_capabilities",
    "resolve_semantic_provider_endpoint",
    "semantic_capability_binding_matches",
    "semantic_provider_is_registered",
    "semantic_provider_label",
    "semantic_provider_policy",
]
