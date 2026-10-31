"""Closed configuration boundary for standalone HTML presentation generation."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Callable, Mapping
from configparser import Error as ConfigParserError
from configparser import NoSectionError
from dataclasses import asdict, dataclass, field
from typing import Any

from tldw_Server_API.app.core.config_sections.slides import SlidesConfig, load_slides_config
from tldw_Server_API.app.core.Utils.prompt_loader import (
    PromptAssetUnavailableError,
    load_prompt_strict,
)

PROMPT_MAX_BYTES = 131_072
PROMPT_CONTRACT_VERSION = "slides.standalone_html.v1"
ALLOWED_TARGETS_JSON_MAX_BYTES = 65_536
ALLOWED_TARGETS_MAX_ENTRIES = 64

MAX_REQUEST_BYTES = 4_194_304
MAX_SOURCE_CHARS = 200_000
MAX_SOURCE_TOKENS = 50_000
MAX_AUDIENCE_CHARS = 500
MAX_SOURCE_IDENTIFIER_BYTES = 256
MAX_NOTE_IDS = 100
MAX_RAG_QUERY_CHARS = 20_000
MAX_RAG_TOP_K = 100
MAX_PROVIDER_RESPONSE_BYTES = 8_388_608
MAX_DOCUMENT_BYTES = 1_048_576

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_OVERALL_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
MAX_OUTPUT_TOKENS = 32_768

_FORBIDDEN_OPTIONS = (
    "base_url",
    "endpoint",
    "endpoint_override",
    "proxy",
    "router",
    "fallback_provider",
    "fallback_model",
    "custom_adapter",
    "verify_tls",
)
_ALLOWED_OPTIONS = {
    "enabled",
    "egress_enabled",
    "default_provider",
    "default_model",
    "default_adapter_id",
    "allowed_targets_json",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "overall_timeout_seconds",
    "max_output_tokens",
    "max_source_chars",
    "max_source_tokens",
    "max_provider_response_bytes",
}
_FORBIDDEN_ENV_KEYS = (
    "SLIDES_STANDALONE_BASE_URL",
    "SLIDES_STANDALONE_ENDPOINT",
    "SLIDES_STANDALONE_ENDPOINT_OVERRIDE",
    "SLIDES_STANDALONE_PROXY",
    "SLIDES_STANDALONE_ROUTER",
    "SLIDES_STANDALONE_FALLBACK_PROVIDER",
    "SLIDES_STANDALONE_FALLBACK_MODEL",
    "SLIDES_STANDALONE_CUSTOM_ADAPTER",
    "SLIDES_STANDALONE_VERIFY_TLS",
)
_PROVIDER_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "llama.cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "ollama": "ollama",
}


class StandaloneHtmlConfigError(ValueError):
    """Source-free standalone configuration error."""

    def __init__(self) -> None:
        super().__init__("standalone_html_config_invalid")


@dataclass(frozen=True, slots=True)
class StandaloneHtmlAdapter:
    """One application-owned request/response adapter identity."""

    adapter_id: str
    provider: str
    endpoint_identity: str
    verified_https: bool
    fixed_endpoint: bool = True


CLOSED_ADAPTER_CATALOG = (
    StandaloneHtmlAdapter(
        "openai_official_chat_v1",
        "openai",
        "https://api.openai.com:443/v1/chat/completions",
        True,
    ),
    StandaloneHtmlAdapter(
        "anthropic_official_messages_v1",
        "anthropic",
        "https://api.anthropic.com:443/v1/messages",
        True,
    ),
    StandaloneHtmlAdapter(
        "llamacpp_loopback_chat_v1_ipv4",
        "llama.cpp",
        "http://127.0.0.1:8080/v1/chat/completions",
        False,
    ),
    StandaloneHtmlAdapter(
        "llamacpp_loopback_chat_v1_ipv6",
        "llama.cpp",
        "http://[::1]:8080/v1/chat/completions",
        False,
    ),
    StandaloneHtmlAdapter(
        "ollama_loopback_chat_v1_ipv4",
        "ollama",
        "http://127.0.0.1:11434/v1/chat/completions",
        False,
    ),
    StandaloneHtmlAdapter(
        "ollama_loopback_chat_v1_ipv6",
        "ollama",
        "http://[::1]:11434/v1/chat/completions",
        False,
    ),
)
_ADAPTERS_BY_ID = {adapter.adapter_id: adapter for adapter in CLOSED_ADAPTER_CATALOG}


@dataclass(frozen=True, slots=True)
class ResolvedExecutionTarget:
    """Exact provider/model/adapter/endpoint tuple used for one call."""

    provider: str
    model: str
    adapter_id: str
    endpoint_identity: str


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """Bounded application prompt and its nonsecret audit metadata."""

    text: str = field(repr=False)
    sha256: str
    contract_version: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class StandaloneHtmlInputLimits:
    """Effective closed input limits advertised to clients."""

    max_request_bytes: int
    max_source_chars: int
    max_source_tokens: int
    max_audience_chars: int
    max_source_identifier_bytes: int
    max_note_ids: int
    max_rag_query_chars: int
    max_rag_top_k: int


@dataclass(frozen=True, slots=True)
class StandaloneHtmlOutputLimits:
    """Effective provider-envelope and document limits."""

    max_provider_response_bytes: int
    max_document_bytes: int


@dataclass(frozen=True, slots=True)
class StandaloneHtmlProviderLimits:
    """Server-owned provider timeouts and output-token budget."""

    connect_timeout_seconds: float
    read_timeout_seconds: float
    overall_timeout_seconds: float
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class StandaloneHtmlGenerationAvailability:
    """Pure source-free dynamic availability inputs."""

    digest_key_available: bool
    worker_handler_registered: bool
    reconciler_admission_ready: bool
    validator_available: bool

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.digest_key_available,
                self.worker_handler_registered,
                self.reconciler_admission_ready,
                self.validator_available,
            )
        ):
            raise StandaloneHtmlConfigError() from None


@dataclass(frozen=True, slots=True)
class SlidesStandaloneHtmlConfig:
    """Immutable effective snapshot consumed by capabilities and workers."""

    feature_enabled: bool
    egress_enabled: bool
    enabled: bool
    disabled_reason: str | None
    target: ResolvedExecutionTarget | None
    prompt: ResolvedPrompt | None
    allowed_targets: tuple[ResolvedExecutionTarget, ...]
    input_limits: StandaloneHtmlInputLimits
    output_limits: StandaloneHtmlOutputLimits
    provider_limits: StandaloneHtmlProviderLimits
    generation_config_revision: str | None
    _revision_manifest: str = field(repr=False)

    @property
    def revision_manifest(self) -> str:
        """Return the canonical nonsecret manifest used for the revision."""

        return self._revision_manifest


def _raise_invalid() -> None:
    raise StandaloneHtmlConfigError() from None


def _canonical_provider(raw: str) -> str:
    provider = _PROVIDER_ALIASES.get(raw.strip().casefold())
    if provider is None:
        _raise_invalid()
    return provider


def _validate_model(model: str) -> None:
    if not model or len(model) > 256:
        _raise_invalid()
    if model != model.strip() or "*" in model:
        _raise_invalid()
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in model):
        _raise_invalid()
    try:
        encoded = model.encode("utf-8")
    except UnicodeEncodeError:
        _raise_invalid()
    if len(encoded) > 256:
        _raise_invalid()


def _strict_json_loads(raw: str) -> object:
    def reject_constant(_value: str) -> None:
        _raise_invalid()

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _raise_invalid()
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError):
        _raise_invalid()


def _parse_allowed_targets(raw: str) -> tuple[ResolvedExecutionTarget, ...]:
    if not raw or len(raw) > ALLOWED_TARGETS_JSON_MAX_BYTES:
        _raise_invalid()
    try:
        raw_bytes = raw.encode("utf-8")
    except UnicodeEncodeError:
        _raise_invalid()
    if len(raw_bytes) > ALLOWED_TARGETS_JSON_MAX_BYTES:
        _raise_invalid()

    decoded = _strict_json_loads(raw)
    if not isinstance(decoded, list) or len(decoded) > ALLOWED_TARGETS_MAX_ENTRIES:
        _raise_invalid()

    targets: list[ResolvedExecutionTarget] = []
    seen: set[ResolvedExecutionTarget] = set()
    expected_keys = {"provider", "model", "adapter_id"}
    for entry in decoded:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            _raise_invalid()
        provider_raw = entry["provider"]
        model = entry["model"]
        adapter_id = entry["adapter_id"]
        if not all(isinstance(value, str) for value in (provider_raw, model, adapter_id)):
            _raise_invalid()
        _validate_model(model)
        if not adapter_id or adapter_id != adapter_id.strip() or "*" in adapter_id:
            _raise_invalid()
        if not provider_raw or provider_raw != provider_raw.strip():
            _raise_invalid()
        provider = _canonical_provider(provider_raw)
        if "*" in provider_raw:
            _raise_invalid()
        adapter = _ADAPTERS_BY_ID.get(adapter_id)
        if adapter is None or adapter.provider != provider:
            _raise_invalid()
        target = ResolvedExecutionTarget(
            provider=provider,
            model=model,
            adapter_id=adapter.adapter_id,
            endpoint_identity=adapter.endpoint_identity,
        )
        if target in seen:
            _raise_invalid()
        seen.add(target)
        targets.append(target)
    return tuple(targets)


def _forbidden_override_is_configured(
    config_parser: Any,
    env: Mapping[str, str],
) -> bool:
    if any(key in env and str(env[key]).strip() for key in _FORBIDDEN_ENV_KEYS):
        return True
    options = getattr(config_parser, "options", None)
    if callable(options):
        try:
            if set(options("SlidesStandaloneHtml")) - _ALLOWED_OPTIONS:
                return True
        except NoSectionError:
            return False
    for option in _FORBIDDEN_OPTIONS:
        try:
            value = config_parser.get("SlidesStandaloneHtml", option, fallback="")
        except NoSectionError:
            return False
        if str(value).strip():
            return True
    return False


def _effective_limits(settings: SlidesConfig) -> tuple[
    StandaloneHtmlInputLimits,
    StandaloneHtmlOutputLimits,
    StandaloneHtmlProviderLimits,
]:
    input_limits = StandaloneHtmlInputLimits(
        max_request_bytes=MAX_REQUEST_BYTES,
        max_source_chars=min(settings.max_source_chars, MAX_SOURCE_CHARS),
        max_source_tokens=min(settings.max_source_tokens, MAX_SOURCE_TOKENS),
        max_audience_chars=MAX_AUDIENCE_CHARS,
        max_source_identifier_bytes=MAX_SOURCE_IDENTIFIER_BYTES,
        max_note_ids=MAX_NOTE_IDS,
        max_rag_query_chars=MAX_RAG_QUERY_CHARS,
        max_rag_top_k=MAX_RAG_TOP_K,
    )
    output_limits = StandaloneHtmlOutputLimits(
        max_provider_response_bytes=min(settings.max_provider_response_bytes, MAX_PROVIDER_RESPONSE_BYTES),
        max_document_bytes=MAX_DOCUMENT_BYTES,
    )
    provider_limits = StandaloneHtmlProviderLimits(
        connect_timeout_seconds=min(settings.connect_timeout_seconds, DEFAULT_CONNECT_TIMEOUT_SECONDS),
        read_timeout_seconds=min(settings.read_timeout_seconds, DEFAULT_READ_TIMEOUT_SECONDS),
        overall_timeout_seconds=min(settings.overall_timeout_seconds, DEFAULT_OVERALL_TIMEOUT_SECONDS),
        max_output_tokens=min(settings.max_output_tokens, MAX_OUTPUT_TOKENS),
    )
    return input_limits, output_limits, provider_limits


def _resolve_default_target(
    settings: SlidesConfig,
    allowed_targets: tuple[ResolvedExecutionTarget, ...],
) -> tuple[ResolvedExecutionTarget | None, str | None]:
    provider_raw = settings.default_provider.strip()
    model = settings.default_model.strip()
    adapter_id = settings.default_adapter_id.strip()
    if not provider_raw or not model or not adapter_id:
        return None, "default_model_not_configured"
    _validate_model(model)
    adapter = _ADAPTERS_BY_ID.get(adapter_id)
    if adapter is None:
        return None, "default_endpoint_not_allowed"
    try:
        provider = _canonical_provider(provider_raw)
    except StandaloneHtmlConfigError:
        return None, "default_endpoint_not_allowed"
    if adapter.provider != provider:
        return None, "default_endpoint_not_allowed"
    target = ResolvedExecutionTarget(
        provider=provider,
        model=model,
        adapter_id=adapter.adapter_id,
        endpoint_identity=adapter.endpoint_identity,
    )
    if target in allowed_targets:
        return target, None
    if any(item.provider == provider and item.adapter_id == adapter_id for item in allowed_targets):
        return None, "default_model_not_allowed"
    return None, "default_endpoint_not_allowed"


def _resolve_prompt(prompt_loader: Callable[[str, str, int], str]) -> ResolvedPrompt:
    prompt = prompt_loader("slides", "standalone_html_system", PROMPT_MAX_BYTES)
    if not isinstance(prompt, str):
        raise PromptAssetUnavailableError()
    prompt = prompt.strip()
    if not prompt or len(prompt) > PROMPT_MAX_BYTES:
        raise PromptAssetUnavailableError()
    if "\x00" in prompt or any(0xD800 <= ord(character) <= 0xDFFF for character in prompt):
        raise PromptAssetUnavailableError()
    try:
        raw = prompt.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PromptAssetUnavailableError() from exc
    if len(raw) > PROMPT_MAX_BYTES:
        raise PromptAssetUnavailableError()
    return ResolvedPrompt(
        text=prompt,
        sha256=hashlib.sha256(raw).hexdigest(),
        contract_version=PROMPT_CONTRACT_VERSION,
        byte_count=len(raw),
    )


def _revision_manifest(
    *,
    settings: SlidesConfig,
    target: ResolvedExecutionTarget,
    prompt: ResolvedPrompt,
    availability: StandaloneHtmlGenerationAvailability,
    input_limits: StandaloneHtmlInputLimits,
    output_limits: StandaloneHtmlOutputLimits,
    provider_limits: StandaloneHtmlProviderLimits,
) -> str:
    payload = {
        "schema_version": 1,
        "feature_enabled": settings.enabled,
        "egress_enabled": settings.egress_enabled,
        "target": {
            "provider": target.provider,
            "model": target.model,
            "adapter_id": target.adapter_id,
            "endpoint_identity": target.endpoint_identity,
        },
        "prompt": {
            "sha256": prompt.sha256,
            "contract_version": prompt.contract_version,
        },
        "availability": {
            "digest_key_available": availability.digest_key_available,
            "worker_handler_registered": availability.worker_handler_registered,
            "reconciler_admission_ready": availability.reconciler_admission_ready,
            "validator_available": availability.validator_available,
        },
        "input_limits": asdict(input_limits),
        "output_limits": asdict(output_limits),
        "provider_limits": asdict(provider_limits),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _snapshot(
    *,
    settings: SlidesConfig,
    allowed_targets: tuple[ResolvedExecutionTarget, ...],
    input_limits: StandaloneHtmlInputLimits,
    output_limits: StandaloneHtmlOutputLimits,
    provider_limits: StandaloneHtmlProviderLimits,
    reason: str | None,
    target: ResolvedExecutionTarget | None = None,
    prompt: ResolvedPrompt | None = None,
    revision_manifest: str = "",
) -> SlidesStandaloneHtmlConfig:
    enabled = reason is None
    revision = None
    if enabled:
        revision = "sha256:" + hashlib.sha256(revision_manifest.encode("utf-8")).hexdigest()
    return SlidesStandaloneHtmlConfig(
        feature_enabled=settings.enabled,
        egress_enabled=settings.egress_enabled,
        enabled=enabled,
        disabled_reason=reason,
        target=target if enabled else None,
        prompt=prompt if enabled else None,
        allowed_targets=allowed_targets,
        input_limits=input_limits,
        output_limits=output_limits,
        provider_limits=provider_limits,
        generation_config_revision=revision,
        _revision_manifest=revision_manifest if enabled else "",
    )


def load_standalone_html_config(
    config_parser: Any,
    *,
    env: Mapping[str, str] | None = None,
    availability: StandaloneHtmlGenerationAvailability,
    prompt_loader: Callable[[str, str, int], str] = load_prompt_strict,
) -> SlidesStandaloneHtmlConfig:
    """Resolve one fail-closed immutable generation configuration snapshot."""

    if not isinstance(availability, StandaloneHtmlGenerationAvailability):
        _raise_invalid()
    env_map: Mapping[str, str] = env if env is not None else os.environ
    try:
        forbidden_override = _forbidden_override_is_configured(config_parser, env_map)
    except (ConfigParserError, ValueError):
        _raise_invalid()
    if forbidden_override:
        _raise_invalid()
    try:
        settings = load_slides_config(config_parser, env=env_map)
    except (ConfigParserError, ValueError):
        _raise_invalid()
    allowed_targets = _parse_allowed_targets(settings.allowed_targets_json)
    input_limits, output_limits, provider_limits = _effective_limits(settings)

    common = {
        "settings": settings,
        "allowed_targets": allowed_targets,
        "input_limits": input_limits,
        "output_limits": output_limits,
        "provider_limits": provider_limits,
    }
    if not settings.enabled:
        return _snapshot(reason="feature_disabled", **common)
    if not settings.egress_enabled:
        return _snapshot(reason="egress_disabled", **common)

    target, target_reason = _resolve_default_target(settings, allowed_targets)
    if target_reason is not None or target is None:
        return _snapshot(reason=target_reason or "default_endpoint_not_allowed", **common)
    try:
        prompt = _resolve_prompt(prompt_loader)
    except PromptAssetUnavailableError:
        return _snapshot(reason="prompt_asset_unavailable", **common)

    dynamic_reasons = (
        (availability.validator_available, "validator_unavailable"),
        (availability.digest_key_available, "digest_key_unavailable"),
        (availability.worker_handler_registered, "generation_worker_unavailable"),
        (availability.reconciler_admission_ready, "generation_reconciler_overloaded"),
    )
    for available, reason in dynamic_reasons:
        if not available:
            return _snapshot(reason=reason, **common)

    manifest = _revision_manifest(
        settings=settings,
        target=target,
        prompt=prompt,
        availability=availability,
        input_limits=input_limits,
        output_limits=output_limits,
        provider_limits=provider_limits,
    )
    return _snapshot(
        reason=None,
        target=target,
        prompt=prompt,
        revision_manifest=manifest,
        **common,
    )


__all__ = [
    "ALLOWED_TARGETS_JSON_MAX_BYTES",
    "ALLOWED_TARGETS_MAX_ENTRIES",
    "CLOSED_ADAPTER_CATALOG",
    "PROMPT_CONTRACT_VERSION",
    "PROMPT_MAX_BYTES",
    "ResolvedExecutionTarget",
    "ResolvedPrompt",
    "SlidesStandaloneHtmlConfig",
    "StandaloneHtmlAdapter",
    "StandaloneHtmlConfigError",
    "StandaloneHtmlGenerationAvailability",
    "StandaloneHtmlInputLimits",
    "StandaloneHtmlOutputLimits",
    "StandaloneHtmlProviderLimits",
    "load_standalone_html_config",
]
