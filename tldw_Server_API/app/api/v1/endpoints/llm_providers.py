# llm_providers.py
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from time import monotonic as _vision_cache_time
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from loguru import logger

import tldw_Server_API.app.core.LLM_Calls.adapter_registry as llm_adapter_registry
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import users_own_key_credentials
from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager
from tldw_Server_API.app.core.config_llm_providers import (
    ALL_SUPPORTED_PROVIDER_NAMES_LIST,
)
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_config_option_names,
    custom_openai_provider_name,
    iter_custom_openai_provider_names,
    iter_custom_openai_provider_numbers,
)
from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import fetch as _http_fetch
from tldw_Server_API.app.core.LLM_Calls.capability_registry import get_allowed_fields
from tldw_Server_API.app.core.LLM_Calls.extra_body_compat_catalog import (
    get_model_extra_body_compat,
    get_provider_extra_body_compat,
)
from tldw_Server_API.app.core.LLM_Calls.llamacpp_request_extensions import resolve_llamacpp_runtime_caps
from tldw_Server_API.app.core.LLM_Calls.model_metadata import get_model_metadata
from tldw_Server_API.app.core.LLM_Calls.openrouter_model_inventory import (
    discover_openrouter_models as _discover_openrouter_models_shared,
)
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import (
    configured_provider_generation_metadata,
    has_custom_openai_env_configuration,
    resolve_provider_api_key_value,
    resolve_provider_endpoint_url,
    resolve_provider_model_value,
    valid_provider_api_key,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    PROVIDER_CAPABILITIES,
    provider_requires_api_key,
)
from tldw_Server_API.app.core.LLM_Calls.provider_model_inventory import (
    CLOUD_MODEL_PROVIDERS,
    has_model_listing,
    inventory_endpoint_scope,
)
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import (
    ModelDiscoveryResult,
)
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import (
    normalize_catalog_provider_for_chat as _normalize_catalog_provider_for_chat,
)
from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
    strict_token_counting_enabled as _strict_token_counting_enabled_shared,
)
from tldw_Server_API.app.core.Local_LLM import llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import (
    managed_profile_model_metadata,
)
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import LLMInferenceLibError
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import (
    ByokResolutionError,
    ServerFallbackCredentials,
)
from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
    ProviderOverrideCallSnapshot,
)
from tldw_Server_API.app.core.Security.egress import (
    ConfiguredEndpointScope,
    URLPolicyResult,
)
import tldw_Server_API.app.api.v1.API_Deps.auth_deps as _auth_deps
import tldw_Server_API.app.core.Image_Generation.listing as _listing
import tldw_Server_API.app.core.LLM_Calls.provider_model_inventory as _provider_model_inventory
import tldw_Server_API.app.core.LLM_Calls.provider_readiness as __provider_readiness
import tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver as _tokenizer_resolver
import tldw_Server_API.app.core.Provider_Credentials.byok_runtime as _byok_runtime
import tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides as _llm_provider_overrides
import tldw_Server_API.app.core.Usage.pricing_catalog as _pricing_catalog
import tldw_Server_API.app.core.Utils.egress as _egress
import tldw_Server_API.app.core.config as _config
import tldw_Server_API.app.core.config_llm_providers as _config_llm_providers

#######################################################################################################################
#
# Functions:

router = APIRouter()

_CATALOG_CREDENTIAL_STORE_UNAVAILABLE = "credential_store_unavailable"

_LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)
_LLAMACPP_METADATA_EXCEPTIONS = _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS + (
    LLMInferenceLibError,
)


@router.get("/llm/health", summary="LLM inference health", response_model=dict[str, Any])
async def llm_health():
    """Health endpoint for the LLM inference subsystem (providers, queue, rate limiter)."""
    from datetime import datetime

    from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager
    from tldw_Server_API.app.core.Chat.rate_limiter import get_rate_limiter
    from tldw_Server_API.app.core.Chat.request_queue import get_request_queue

    health: dict[str, Any] = {
        "service": "llm_inference",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    try:
        # Provider manager
        pm = get_provider_manager()
        if pm is None:
            health["components"]["providers"] = {"initialized": False}
            health["status"] = "degraded"
        else:
            report = pm.get_health_report()
            any_unhealthy = any(v.get("status") in ["unhealthy", "circuit_open"] for v in report.values())
            health["components"]["providers"] = {
                "initialized": True,
                "count": len(report),
                "report": report
            }
            if any_unhealthy and health["status"] == "healthy":
                health["status"] = "degraded"

        # Request queue
        rq = get_request_queue()
        if rq is None:
            health["components"]["queue"] = {"initialized": False}
            health["status"] = "degraded"
        else:
            q_status = rq.get_queue_status()
            health["components"]["queue"] = {"initialized": True, **q_status}

        # Rate limiter
        rl = get_rate_limiter()
        if rl is None:
            health["components"]["rate_limiter"] = {"initialized": False}
            # Not critical, keep status as-is
        else:
            cfg = rl.config
            health["components"]["rate_limiter"] = {
                "initialized": True,
                "limits": {
                    "global_rpm": cfg.global_rpm,
                    "per_user_rpm": cfg.per_user_rpm,
                    "per_conversation_rpm": cfg.per_conversation_rpm,
                    "per_user_tokens_per_minute": cfg.per_user_tokens_per_minute
                }
            }
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        logger.error("LLM health check failed")
        health["status"] = "unhealthy"
        health["error"] = "LLM health check failed"

    return health

def parse_model_string(model_value: str) -> list[str]:
    """
    Parse a model string which could be a single model or comma-separated list.

    Args:
        model_value: Model string from config

    Returns:
        List of model names
    """
    if not model_value:
        return []

    # Handle comma-separated lists
    if ',' in model_value:
        return [m.strip() for m in model_value.split(',') if m.strip()]

    # Single model
    return [model_value.strip()] if model_value.strip() else []


# Local model discovery helpers (best-effort; cached to avoid hammering endpoints)
LOCAL_MODEL_DISCOVERY_TIMEOUT = 1.5  # seconds
# Local/LAN endpoints answer in milliseconds when present. This bound is only
# ever paid in full by an endpoint that is configured but not listening, so it
# is kept short: it is discovery, and a miss degrades to "no models found".
LOCAL_MODEL_DISCOVERY_TTL = 300  # seconds
_LOCAL_MODEL_CACHE: dict[str, tuple[float, ModelDiscoveryResult]] = {}
_LLAMACPP_VISION_CACHE: dict[str, tuple[float, dict[str, bool]]] = {}
# Upper bound on simultaneous discovery probes for one endpoint.
_MODEL_DISCOVERY_MAX_PARALLEL_PROBES = 6
OPENROUTER_MODEL_DISCOVERY_TIMEOUT = 5.0  # seconds
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _discover_llamacpp_vision(
    endpoint_url: str,
    api_key: Optional[str],
    *,
    configured_endpoint: ConfiguredEndpointScope,
) -> dict[str, bool]:
    """Confirm vision only for identities reported by the configured llama.cpp server.

    Compatible servers without /props may explicitly declare vision in /v1/models.
    Keep one short-lived snapshot, including failures, to avoid delaying every
    catalog request when /props is unavailable. Endpoint or credential changes
    invalidate it. This optional capability probe does not change readiness.
    """
    parsed = urlparse(endpoint_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/completion", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    props_url = parsed._replace(path=f"{path}/props", params="", query="", fragment="").geturl()
    cache_key = hashlib.sha256(json.dumps([props_url, api_key]).encode()).hexdigest()
    now = _vision_cache_time()
    cached = _LLAMACPP_VISION_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < 30:
        return cached[1]

    confirmed: dict[str, bool] = {}
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = _http_fetch(
            method="GET",
            url=props_url,
            headers=headers,
            timeout=LOCAL_MODEL_DISCOVERY_TIMEOUT,
            retry=_RetryPolicy(attempts=1),
            allow_redirects=False,
            configured_endpoint=configured_endpoint,
            sensitive_observability=True,
        )
        unsupported_props = response.status_code in (404, 405)
        try:
            if response.status_code == 200:
                payload = response.json()
                modalities = payload.get("modalities") if isinstance(payload, dict) else None
                vision = modalities.get("vision") if isinstance(modalities, dict) else None
                if isinstance(vision, bool):
                    for field in ("model_alias", "model_path"):
                        identity = payload.get(field)
                        if isinstance(identity, str) and identity.strip():
                            confirmed[identity.strip()] = vision
        finally:
            response.close()
        if unsupported_props:
            models_url = parsed._replace(path=f"{path}/v1/models", params="", query="", fragment="").geturl()
            models_response = _http_fetch(
                method="GET",
                url=models_url,
                headers=headers,
                timeout=LOCAL_MODEL_DISCOVERY_TIMEOUT,
                retry=_RetryPolicy(attempts=1),
                allow_redirects=False,
                configured_endpoint=configured_endpoint,
                sensitive_observability=True,
            )
            try:
                if models_response.status_code == 200:
                    payload = models_response.json()
                    models = payload.get("data") if isinstance(payload, dict) else None
                    if isinstance(models, list):
                        for model in models:
                            if not isinstance(model, dict):
                                continue
                            identity, vision = model.get("id"), model.get("vision")
                            if isinstance(identity, str) and identity.strip() and isinstance(vision, bool):
                                identity = identity.strip()
                                confirmed[identity] = confirmed.get(identity, True) and vision
            finally:
                models_response.close()
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        logger.debug("External llama.cpp capability discovery unavailable")
    _LLAMACPP_VISION_CACHE.clear()
    _LLAMACPP_VISION_CACHE[cache_key] = (now, confirmed)
    return confirmed


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for v in values:
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _candidate_model_urls_for_openai_endpoint(endpoint_url: str) -> list[str]:
    """Return likely /models endpoints for OpenAI-compatible servers."""
    try:
        parsed = urlparse(endpoint_url.strip())
        if not parsed.scheme or not parsed.netloc:
            return []

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").rstrip("/")

        candidates = [
            urljoin(base, "/v1/models"),
        ]

        if path:
            candidates.append(urljoin(base, f"{path}/models"))
            # Remove common suffixes like /chat/completions
            for suffix in ["/chat/completions", "/completions", "/v1/chat/completions", "/v1/completions"]:
                if path.endswith(suffix):
                    prefix = path[: -len(suffix)]
                    candidates.append(urljoin(base, f"{prefix}/models"))
                    candidates.append(urljoin(base, f"{prefix}/v1/models"))

        return _dedupe_preserve_order([c.rstrip("/") for c in candidates if c])
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        return []


def _candidate_model_urls_for_ollama_endpoint(endpoint_url: str) -> list[str]:
    """Return likely endpoints for Ollama model listings."""
    try:
        parsed = urlparse(endpoint_url.strip())
        if not parsed.scheme or not parsed.netloc:
            return []

        base = f"{parsed.scheme}://{parsed.netloc}"
        path = (parsed.path or "").rstrip("/")
        candidates = [
            urljoin(base, "/api/tags"),
            urljoin(base, "/v1/models"),
        ]
        if path and path not in {"/"}:
            candidates.append(urljoin(base, f"{path}/api/tags"))
            candidates.append(urljoin(base, f"{path}/models"))

        return _dedupe_preserve_order([c.rstrip("/") for c in candidates if c])
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        return []


def _extract_models_from_response(payload: Any) -> list[str]:
    """Normalize various model-list responses into a flat list of ids."""
    models: list[str] = []
    if isinstance(payload, dict):
        data_section = payload.get("data")
        if isinstance(data_section, list):
            for item in data_section:
                candidate = item.get("id") or item.get("model") or item.get("name") if isinstance(item, dict) else item
                if isinstance(candidate, str) and candidate.strip():
                    models.append(candidate.strip())

        models_section = payload.get("models")
        if isinstance(models_section, list):
            for item in models_section:
                candidate = item.get("name") or item.get("model") or item.get("id") if isinstance(item, dict) else item
                if isinstance(candidate, str) and candidate.strip():
                    models.append(candidate.strip())

    return _dedupe_preserve_order(models)


def discover_openrouter_models(
    api_key: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Best-effort discovery of OpenRouter model ids from /models."""
    return _discover_openrouter_models_shared(
        api_key,
        force_refresh=force_refresh,
        include_extended_aliases=False,
        timeout_seconds=OPENROUTER_MODEL_DISCOVERY_TIMEOUT,
        log_prefix="[OpenRouter model discovery]",
        fetch_fn=_http_fetch,
    )


def discover_models_from_endpoint(
    provider: str,
    endpoint_url: str,
    discovery_type: str = "openai",
    api_key: Optional[str] = None,
    *,
    configured_endpoint: ConfiguredEndpointScope,
) -> ModelDiscoveryResult:
    """
    Best-effort discovery of models from a configured local endpoint.

    Tries common OpenAI-compatible /models endpoints (or Ollama tags),
    uses a short timeout, and caches results to avoid noisy retries.
    """
    endpoint_url = (endpoint_url or "").strip()
    if not endpoint_url:
        return ModelDiscoveryResult("unreachable")

    cache_key = f"{provider}:{endpoint_url}"
    now = time.time()
    cached = _LOCAL_MODEL_CACHE.get(cache_key)
    if cached and (now - cached[0] < LOCAL_MODEL_DISCOVERY_TTL):
        return cached[1]

    discovery_type = (discovery_type or "openai").lower()
    if discovery_type == "ollama":
        candidates = _candidate_model_urls_for_ollama_endpoint(endpoint_url)
    else:
        candidates = _candidate_model_urls_for_openai_endpoint(endpoint_url)

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    best_result = ModelDiscoveryResult("unreachable")
    precedence = {
        "unreachable": 0,
        "unsupported": 1,
        "server_error": 2,
        "auth_failed": 3,
        "ready": 4,
    }
    def _probe_candidate(url: str) -> ModelDiscoveryResult:
        """Probe one candidate URL and classify the outcome."""
        candidate_result = ModelDiscoveryResult("unreachable")
        try:
            resp = _http_fetch(
                method="GET",
                url=url,
                headers=headers,
                timeout=LOCAL_MODEL_DISCOVERY_TIMEOUT,
                retry=_RetryPolicy(attempts=1),
                configured_endpoint=configured_endpoint,
            )
            try:
                if resp.status_code in {401, 403}:
                    candidate_result = ModelDiscoveryResult("auth_failed")
                elif resp.status_code == 429 or resp.status_code >= 500:
                    logger.debug("Model discovery endpoint returned an error status")
                    candidate_result = ModelDiscoveryResult("server_error")
                elif resp.status_code >= 400:
                    logger.debug("Model discovery endpoint returned an error status")
                    candidate_result = ModelDiscoveryResult("unsupported")
                else:
                    try:
                        payload = resp.json()
                    except (json.JSONDecodeError, TypeError, ValueError):
                        candidate_result = ModelDiscoveryResult("unsupported")
                    else:
                        has_collection = isinstance(payload, dict) and (
                            isinstance(payload.get("data"), list)
                            or isinstance(payload.get("models"), list)
                        )
                        if not has_collection:
                            candidate_result = ModelDiscoveryResult("unsupported")
                        else:
                            discovered = tuple(_extract_models_from_response(payload))
                            candidate_result = ModelDiscoveryResult("ready", discovered)
                            if discovered:
                                logger.info(
                                    f"[Model discovery] {provider}: found {len(discovered)} models via {url}"
                                )
            finally:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Model discovery endpoint query failed")
            candidate_result = ModelDiscoveryResult("unreachable")
        except Exception:  # noqa: BLE001 - best-effort local discovery must fail open
            logger.debug("Model discovery endpoint query failed unexpectedly")
            candidate_result = ModelDiscoveryResult("unreachable")
        return candidate_result

    def _merge(result: ModelDiscoveryResult) -> None:
        nonlocal best_result
        if precedence[result.status] > precedence[best_result.status]:
            best_result = result

    # Candidates are probed concurrently. An endpoint that is not listening burns
    # LOCAL_MODEL_DISCOVERY_TIMEOUT per candidate, so probing them in series cost
    # timeout x len(candidates) on every call -- and failures are deliberately
    # never cached, so that cost repeated on every request.
    #
    # Cancelling on the first "ready" only skips candidates that have not started
    # yet, which is why it is worth doing at all: with more candidates than
    # workers, the queued ones never run. Probes already in flight are still
    # waited for -- the executor shuts down normally rather than abandoning
    # threads -- so the floor is one probe's worth of time, bounded by
    # LOCAL_MODEL_DISCOVERY_TIMEOUT, not the sum across candidates.
    #
    # A healthy endpoint does receive the extra candidate requests, but a "ready"
    # result is cached for LOCAL_MODEL_DISCOVERY_TTL, so that happens at most
    # once per TTL rather than per request.
    if candidates:
        with ThreadPoolExecutor(
            max_workers=min(len(candidates), _MODEL_DISCOVERY_MAX_PARALLEL_PROBES),
            thread_name_prefix="model-discovery",
        ) as pool:
            futures = [pool.submit(_probe_candidate, url) for url in candidates]
            for future in as_completed(futures):
                _merge(future.result())
                if best_result.status == "ready":
                    for pending in futures:
                        pending.cancel()
                    break

    if best_result.status in {"ready", "unsupported"}:
        _LOCAL_MODEL_CACHE[cache_key] = (now, best_result)
    return best_result


def _coerce_model_discovery_result(value: Any) -> ModelDiscoveryResult:
    """Keep test/plugin list stubs compatible with the typed discovery boundary."""
    if isinstance(value, ModelDiscoveryResult):
        return value
    if isinstance(value, (list, tuple)):
        models = tuple(str(item).strip() for item in value if str(item).strip())
        return ModelDiscoveryResult("ready", models) if models else ModelDiscoveryResult("unreachable")
    return ModelDiscoveryResult("unreachable")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _resolve_model_tokenizer_support(
    provider_name: str,
    model_name: str,
    config_parser: Any,
) -> dict[str, Any]:
    strict_mode_effective = _strict_token_counting_enabled_shared(default=False)
    if _truthy(os.getenv("TEST_MODE")) and _truthy(os.getenv("E2E_INPROCESS")):
        return _skipped_model_tokenizer_support(
            "Tokenizer probe skipped in in-process test mode to avoid heavyweight runtime imports",
            strict_mode_effective=strict_mode_effective,
        )
    try:
        return _tokenizer_resolver.resolve_tokenizer_metadata(
            provider_name,
            model_name,
            strict_mode_effective=strict_mode_effective,
            config_parser=config_parser,
            runtime_probe_exact=True,
            runtime_probe_timeout_seconds=2.0,
            runtime_probe_text="",
        )
    except Exception:  # noqa: BLE001 - metadata path should fail open
        return {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": strict_mode_effective,
        }


def _skipped_model_tokenizer_support(
    reason: str,
    *,
    strict_mode_effective: bool,
) -> dict[str, Any]:
    return {
        "available": False,
        "tokenizer": None,
        "kind": None,
        "source": None,
        "detokenize": False,
        "count_accuracy": "unavailable",
        "strict_mode_effective": strict_mode_effective,
        "error": reason,
    }


def _should_probe_model_tokenizer(
    *,
    provider_type: str,
    model_info: dict[str, Any],
    model_name: str,
    configured_model_names: set[str],
    model_index: int,
) -> tuple[bool, str | None]:
    normalized_name = str(model_name or "").strip()
    if not normalized_name:
        return False, "Tokenizer probe skipped: model name is missing"

    model_type = _infer_model_type(model_info)
    if model_type in {"image", "video", "embedding"}:
        return False, f"Tokenizer probe skipped: unsupported model type '{model_type}'"

    modalities = model_info.get("modalities")
    output_modalities: list[str] = []
    if isinstance(modalities, dict):
        output_modalities = _normalize_modalities(modalities.get("output"))
    if output_modalities and "text" not in output_modalities:
        return False, "Tokenizer probe skipped: model does not output text"

    if provider_type == "commercial":
        if configured_model_names and normalized_name not in configured_model_names:
            return False, "Tokenizer probe skipped: catalog model is not explicitly configured"
        if not configured_model_names and model_index > 0:
            return False, "Tokenizer probe skipped: additional catalog models are not probed by default"

    return True, None


def _build_runtime_context(config_parser: Any) -> dict[str, Any]:
    strict_openai_compat = False
    thinking_budget_request_key = ""
    try:
        if config_parser.has_section("Local-API") and config_parser.has_option("Local-API", "strict_openai_compat"):
            strict_openai_compat = _truthy(config_parser.get("Local-API", "strict_openai_compat", fallback="false"))
        if config_parser.has_section("Local-API") and config_parser.has_option("Local-API", "llama_cpp_thinking_budget_param"):
            thinking_budget_request_key = (
                config_parser.get("Local-API", "llama_cpp_thinking_budget_param", fallback="") or ""
            ).strip()
    except Exception:  # noqa: BLE001 - capability metadata should fail open
        strict_openai_compat = False
        thinking_budget_request_key = ""

    env_override = os.getenv("LOCAL_LLM_STRICT_OPENAI_COMPAT")
    if env_override is not None:
        strict_openai_compat = _truthy(env_override)
    thinking_budget_env = os.getenv("LLAMA_CPP_THINKING_BUDGET_PARAM")
    if thinking_budget_env is not None:
        thinking_budget_request_key = thinking_budget_env.strip()
    return {
        "strict_openai_compat": strict_openai_compat,
        "thinking_budget_request_key": thinking_budget_request_key or None,
    }


def _resolve_llama_cpp_control_caps(runtime_context: dict[str, Any]) -> dict[str, Any]:
    runtime_caps = resolve_llamacpp_runtime_caps(runtime_context=runtime_context)
    strict_mode = bool(runtime_caps.get("strict_openai_compat"))
    thinking_budget = runtime_caps.get("thinking_budget") or {}
    request_key = str(thinking_budget.get("request_key") or "").strip()
    return {
        "grammar": {
            "supported": not strict_mode,
            "effective_reason": (
                "disabled by strict_openai_compat runtime setting"
                if strict_mode else "supported in current deployment"
            ),
            "source": "first_class+extra_body",
        },
        "thinking_budget": {
            "supported": (not strict_mode) and bool(request_key),
            "request_key": request_key or None,
            "effective_reason": (
                "disabled by strict_openai_compat runtime setting"
                if strict_mode
                else "no configured thinking-budget mapping for this deployment"
                if not request_key else "supported in current deployment"
            ),
        },
        "reserved_extra_body_keys": list(runtime_caps.get("reserved_extra_body_keys") or ["grammar"]),
    }


def _fallback_extra_body_compat() -> dict[str, Any]:
    return {
        "supported": False,
        "effective_reason": "unsupported for deployment/runtime configuration",
        "known_params": [],
        "param_groups": [],
        "notes": "No extra_body compatibility metadata registered for this provider/model.",
        "example": {"extra_body": {}},
        "source": "catalog+runtime",
    }


def _safe_provider_extra_body_compat(provider: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    try:
        return get_provider_extra_body_compat(provider, runtime_context=runtime_context)
    except Exception:  # noqa: BLE001 - capability metadata should fail open
        return _fallback_extra_body_compat()


def _safe_model_extra_body_compat(provider: str, model: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    try:
        return get_model_extra_body_compat(provider, model, runtime_context=runtime_context)
    except Exception:  # noqa: BLE001 - capability metadata should fail open
        return _fallback_extra_body_compat()


_LISTING_DISCOVERY_WORKERS = 4
_CommercialListing = tuple[Optional[str], bool, Optional[ModelDiscoveryResult]]


def _listing_on_key(
    provider_name: str,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    refresh_openrouter: bool,
    *,
    endpoint_provenance: str | None = None,
) -> _CommercialListing:
    """(API key, configured, live inventory) for a commercial provider listed on ``api_key``.

    The key is the server's, or the requesting user's own (TASK-15583); discovery runs as chat's does on it (the
    gateway's ``_provider_inventory_endpoint``). Its cache is keyed by the key's SHA-256 digest, so a listing only
    ever shows a model list fetched with the key it is listed on.
    """
    if not api_key:
        return None, False, None
    if not has_model_listing(provider_name):
        return api_key, True, None  # no listing: the configured models are shown, as before
    base_url = _provider_model_inventory.resolve_provider_models_base_url(
        provider_name, app_config, credentials_resolved=True,
    )
    if base_url is None:
        return api_key, True, ModelDiscoveryResult("unsupported")
    scope = inventory_endpoint_scope(provider_name, base_url, endpoint_provenance=endpoint_provenance)
    scope_option: dict[str, Any] = {"configured_endpoint": scope} if scope is not None else {}
    return api_key, True, _provider_model_inventory.discover_provider_models(
        provider_name, api_key, base_url=base_url,
        force_refresh=refresh_openrouter and provider_name == "openrouter", **scope_option,
    )


def _resolve_commercial_listing(
    provider_name: str,
    provider_info: dict[str, Any],
    config_parser: Any,
    api_keys_by_provider: dict[str, Any],
    override: Any,
    refresh_openrouter: bool,
    own_key: Any = None,
) -> _CommercialListing:
    """(API key, configured, live inventory) for one commercial listing entry.

    With ``own_key`` (the requesting user's own-key handle, TASK-15583) the provider is listed on that key, unless
    the admin disabled it.

    Raises ByokResolutionError when the credential store itself is unavailable.
    """
    if own_key is not None and not (override is not None and override.is_enabled is False):
        return _listing_on_key(
            provider_name, valid_provider_api_key(own_key.api_key), own_key.app_config, refresh_openrouter,
            endpoint_provenance=own_key.endpoint_provenance,
        )
    section_name = provider_info.get('section')
    api_key_field = provider_info.get('api_key_field')
    config_key = None
    if (
        api_key_field and section_name and config_parser.has_section(section_name)
        and config_parser.has_option(section_name, api_key_field)
    ):
        config_key = config_parser.get(section_name, api_key_field, fallback='')
    api_key_value = (
        valid_provider_api_key(api_keys_by_provider.get(provider_name)) or valid_provider_api_key(config_key)
    )
    if override is not None and (override.is_enabled is False or override.credentials_invalid):
        if override.credentials_invalid:
            return None, False, ModelDiscoveryResult("auth_failed")
        return api_key_value, bool(api_key_value), ModelDiscoveryResult("auth_failed")
    try:
        static = _byok_runtime.resolve_static_server_fallback(provider_name)
        fallback = ProviderOverrideCallSnapshot(provider_name, override).server_fallback(
            ServerFallbackCredentials(api_key=api_key_value, credential_fields={}, app_config=static.app_config)
        )
    except ByokResolutionError as error:
        if error.code != "invalid_provider_credentials":
            raise
        return None, False, ModelDiscoveryResult("auth_failed")
    if fallback is not None:
        api_key_value = valid_provider_api_key(fallback.api_key)
    if not api_key_value:
        return None, False, None
    fallback_config = fallback.app_config if fallback is not None else static.app_config
    return _listing_on_key(
        provider_name, api_key_value, dict(fallback_config) if fallback_config is not None else None,
        refresh_openrouter,
    )


def _with_pricing_models(provider_name: str, models: list[str]) -> list[str]:
    """Configured models, then the pricing catalog's chat models: dev's listing for a provider with no model list."""
    try:
        pricing_models = [m for m in _pricing_catalog.list_provider_models(provider_name) if 'embed' not in m.lower()]
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        pricing_models = []
    seen = {m.strip() for m in models}
    return models + [m for m in pricing_models if m not in seen]


def _resolve_commercial_listings(
    provider_mappings: dict[str, Any],
    config_parser: Any,
    api_keys_by_provider: dict[str, Any],
    provider_overrides: Any,
    refresh_openrouter: bool,
    own_keys: Mapping[str, Any] | None = None,
) -> dict[str, _CommercialListing] | None:
    """Resolve every listed commercial provider, discovering concurrently (review I5).

    One slow or failing provider then costs about one discovery budget instead of delaying every provider after
    it. Returns None when the credential store is unavailable: commercial entries are left out, and the local
    providers, which never read the store, are still listed (review M6).
    """
    jobs = []
    for provider_name, provider_info in provider_mappings.items():
        if provider_info.get('type') != 'commercial':
            continue
        section_name = provider_info.get('section')
        override = provider_overrides.get(_normalize_catalog_provider_for_chat(provider_name))
        listed = (
            bool(section_name and config_parser.has_section(section_name))
            or has_custom_openai_env_configuration(provider_name)
            or override is not None
        )
        if listed:
            jobs.append((provider_name, provider_info, override))
    if not jobs:
        return {}

    def resolve(job: tuple[str, dict[str, Any], Any]) -> _CommercialListing:
        provider_name, provider_info, override = job
        return _resolve_commercial_listing(
            provider_name, provider_info, config_parser, api_keys_by_provider, override, refresh_openrouter,
            (own_keys or {}).get(provider_name),
        )

    try:
        with ThreadPoolExecutor(max_workers=min(_LISTING_DISCOVERY_WORKERS, len(jobs))) as pool:
            return dict(zip([job[0] for job in jobs], pool.map(resolve, jobs)))
    except ByokResolutionError:
        return None


def _local_only_provider_listing(listing: dict[str, Any]) -> dict[str, Any]:
    """Keep local discovery available when cloud credential policy cannot be verified."""
    providers = [
        entry
        for entry in listing.get("providers", [])
        if isinstance(entry, dict) and entry.get("type") == "local"
    ]
    return {
        **listing,
        "providers": providers,
        "total_configured": len(providers),
        "error": "Provider credential store is unavailable.",
        "error_code": _CATALOG_CREDENTIAL_STORE_UNAVAILABLE,
    }


@dataclass(frozen=True, repr=False)
class _ProviderListingCredentials:
    """Credential state for one provider discovery call."""

    own_keys: Mapping[str, Any] | None = None
    local_only: bool = False


def get_configured_providers(
    include_deprecated: bool = False,
    refresh_openrouter: bool = False,
    credentials: _ProviderListingCredentials = _ProviderListingCredentials(),
) -> dict[str, Any]:
    """
    Get list of configured LLM providers with their models from the config file.

    List commercial providers on the requesting user's own key where available.
    Use local-only discovery when the caller already knows the credential store is unavailable.
    """
    try:
        config_parser = _config.load_comprehensive_config()
        provider_overrides = {}
        local_only = credentials.local_only
        if not local_only:
            try:
                provider_overrides = _llm_provider_overrides.get_llm_provider_overrides_snapshot()
            except ByokResolutionError:
                local_only = True
        runtime_context = _build_runtime_context(config_parser)
        providers = []

        # Check if we have the required sections or env-only custom OpenAI config
        has_env_custom_openai = any(
            has_custom_openai_env_configuration(provider_name)
            for provider_name in iter_custom_openai_provider_names()
        )
        if (
            not config_parser.has_section('API')
            and not config_parser.has_section('Local-API')
            and not has_env_custom_openai
            and not provider_overrides
        ):
            logger.warning("No API or Local-API sections found in config")
            empty_listing: dict[str, Any] = {
                'providers': [],
                'default_provider': None,
                'total_configured': 0,
                'message': 'No API configuration sections found in config.txt'
            }
            return _local_only_provider_listing(empty_listing) if local_only else empty_listing

        try:
            api_keys_by_provider = _config_llm_providers.get_api_keys()
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            api_keys_by_provider = {}

        # Define provider mappings with their config keys
        provider_mappings: dict[str, Any] = {
            # Commercial APIs (from API section)
            'openai': {
                'display_name': 'OpenAI',
                'api_key_field': 'openai_api_key',
                'model_field': 'openai_model',
                'type': 'commercial',
                'section': 'API'
            },
            'bedrock': {
                'display_name': 'AWS Bedrock',
                'api_key_field': 'bedrock_api_key',
                'model_field': 'bedrock_model',
                'type': 'commercial',
                'section': 'API'
            },
            'anthropic': {
                'display_name': 'Anthropic',
                'api_key_field': 'anthropic_api_key',
                'model_field': 'anthropic_model',
                'type': 'commercial',
                'section': 'API'
            },
            'cohere': {
                'display_name': 'Cohere',
                'api_key_field': 'cohere_api_key',
                'model_field': 'cohere_model',
                'type': 'commercial',
                'section': 'API'
            },
            'deepseek': {
                'display_name': 'DeepSeek',
                'api_key_field': 'deepseek_api_key',
                'model_field': 'deepseek_model',
                'type': 'commercial',
                'section': 'API'
            },
            'qwen': {
                'display_name': 'Qwen',
                'api_key_field': 'qwen_api_key',
                'model_field': 'qwen_model',
                'type': 'commercial',
                'section': 'API'
            },
            'google': {
                'display_name': 'Google',
                'api_key_field': 'google_api_key',
                'model_field': 'google_model',
                'type': 'commercial',
                'section': 'API'
            },
            'groq': {
                'display_name': 'Groq',
                'api_key_field': 'groq_api_key',
                'model_field': 'groq_model',
                'type': 'commercial',
                'section': 'API'
            },
            'mistral': {
                'display_name': 'Mistral',
                'api_key_field': 'mistral_api_key',
                'model_field': 'mistral_model',
                'type': 'commercial',
                'section': 'API'
            },
            'huggingface': {
                'display_name': 'HuggingFace',
                'api_key_field': 'huggingface_api_key',
                'model_field': 'huggingface_model',
                'type': 'commercial',
                'section': 'API'
            },
            'openrouter': {
                'display_name': 'OpenRouter',
                'api_key_field': 'openrouter_api_key',
                'model_field': 'openrouter_model',
                'type': 'commercial',
                'section': 'API'
            },
            'novita': {
                'display_name': 'Novita',
                'api_key_field': 'novita_api_key',
                'model_field': 'novita_model',
                'type': 'commercial',
                'section': 'API'
            },
            'poe': {
                'display_name': 'Poe',
                'api_key_field': 'poe_api_key',
                'model_field': 'poe_model',
                'type': 'commercial',
                'section': 'API'
            },
            'together': {
                'display_name': 'Together',
                'api_key_field': 'together_api_key',
                'model_field': 'together_model',
                'type': 'commercial',
                'section': 'API'
            },
            'moonshot': {
                'display_name': 'Moonshot',
                'api_key_field': 'moonshot_api_key',
                'model_field': 'moonshot_model',
                'type': 'commercial',
                'section': 'API'
            },
            'zai': {
                'display_name': 'Z.AI',
                'api_key_field': 'zai_api_key',
                'model_field': 'zai_model',
                'type': 'commercial',
                'section': 'API'
            },
            'minimax': {
                'display_name': 'MiniMax',
                'api_key_field': 'minimax_api_key',
                'model_field': 'minimax_model',
                'type': 'commercial',
                'section': 'API'
            },
            # Local APIs (from Local-API section)
            'llama': {
                'display_name': 'Llama.cpp',
                'endpoint_field': 'llama_api_IP',
                'model_field': 'llama_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'kobold': {
                'display_name': 'Kobold.cpp',
                'endpoint_field': 'kobold_api_IP',
                'model_field': 'kobold_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'ooba': {
                'display_name': 'Oobabooga',
                'endpoint_field': 'ooba_api_IP',
                'model_field': 'ooba_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'tabby': {
                'display_name': 'TabbyAPI',
                'endpoint_field': 'tabby_api_IP',
                'model_field': 'tabby_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'vllm': {
                'display_name': 'vLLM',
                'endpoint_field': 'vllm_api_IP',
                'model_field': 'vllm_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'ollama': {
                'display_name': 'Ollama',
                'endpoint_field': 'ollama_api_IP',
                'model_field': 'ollama_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'ollama',
            },
            'aphrodite': {
                'display_name': 'Aphrodite',
                'endpoint_field': 'aphrodite_api_IP',
                'model_field': 'aphrodite_model',
                'type': 'local',
                'section': 'Local-API',
                'model_discovery': 'openai',
            },
            'mlx': {
                'display_name': 'MLX',
                'endpoint_field': None,
                'model_field': 'mlx_model_path',
                'type': 'local',
                'section': 'MLX',
                'model_discovery': None,
            },
            'custom_openai_api': {
                'display_name': 'Custom OpenAI API',
                'endpoint_field': 'custom_openai_api_ip',
                'api_key_field': 'custom_openai_api_key',
                'model_field': 'custom_openai_api_model',
                'type': 'local',
                'section': 'API',
                'model_discovery': 'openai',
            }
        }

        def _configured_option_name(option_names: tuple[str, ...]) -> str:
            """Select the configured alias while retaining deterministic fallback."""
            if config_parser.has_section("API"):
                for option_name in option_names:
                    if config_parser.has_option("API", option_name):
                        return option_name
            return option_names[0]

        for provider_number in iter_custom_openai_provider_numbers(start=2):
            provider_name = custom_openai_provider_name(provider_number)
            endpoint_field = _configured_option_name(
                custom_openai_config_option_names(provider_number, "ip")
            )
            model_field = _configured_option_name(
                custom_openai_config_option_names(provider_number, "model")
            )
            api_key_field = _configured_option_name(
                custom_openai_config_option_names(provider_number, "key")
            )
            if not any(
                (
                    resolve_provider_endpoint_url(
                        provider_name,
                        config_parser,
                        "API",
                        endpoint_field,
                    ),
                    resolve_provider_model_value(
                        provider_name,
                        config_parser,
                        "API",
                        model_field,
                    ),
                    resolve_provider_api_key_value(
                        provider_name,
                        config_parser,
                        "API",
                        api_key_field,
                    ),
                )
            ):
                continue
            provider_mappings[provider_name] = {
                "display_name": f"Custom OpenAI API {provider_number}",
                "endpoint_field": endpoint_field,
                "api_key_field": api_key_field,
                "model_field": model_field,
                "type": "local",
                "section": "API",
                "model_discovery": "openai",
            }

        # Optional: live health report
        health_report = {}
        try:
            pm = get_provider_manager()
            if pm:
                health_report = pm.get_health_report()
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            health_report = {}
        registry_capability_envelopes = _llm_registry_capability_envelopes()
        supported_chat_providers = set(_config_llm_providers.ALL_SUPPORTED_PROVIDER_NAMES_LIST)
        endpoint_probe_enabled = __provider_readiness.configured_endpoint_probe_enabled()

        commercial_listings = (
            _resolve_commercial_listings(
                provider_mappings, config_parser, api_keys_by_provider, provider_overrides, refresh_openrouter,
                credentials.own_keys,
            )
            if not local_only else None
        )
        if commercial_listings is None:
            logger.error("Provider credential store is unavailable; listing local providers only")

        # Process each provider
        for provider_name, provider_info in provider_mappings.items():
            if provider_info['type'] == 'commercial' and commercial_listings is None:
                continue
            section_name = provider_info.get('section')
            override = provider_overrides.get(_normalize_catalog_provider_for_chat(provider_name))

            config_section_exists = bool(section_name and config_parser.has_section(section_name))
            custom_env_configured = has_custom_openai_env_configuration(provider_name)
            # Allow MLX to surface even when only env-based config exists
            if not config_section_exists and provider_name != "mlx" and not custom_env_configured and override is None:
                continue
            # For MLX, proceed even if the [MLX] section is absent to allow env-only configs or to show as disabled
            section_exists = config_section_exists or provider_name == "mlx" or custom_env_configured or override is not None
            if not section_exists:
                continue

            # Check if provider is configured
            is_configured = False
            endpoint_url: Optional[str] = None
            api_key_value: Optional[str] = None

            commercial_discovery: ModelDiscoveryResult | None = None
            if provider_info['type'] == 'commercial':
                api_key_value, is_configured, commercial_discovery = (commercial_listings or {}).get(
                    provider_name, (None, False, None),
                )
            else:
                # Check for endpoint URL for local providers
                endpoint_field = provider_info.get('endpoint_field')
                endpoint_url = resolve_provider_endpoint_url(
                    provider_name,
                    config_parser,
                    section_name,
                    endpoint_field,
                )
                if endpoint_url:
                    is_configured = True
                # Optional API key support for local endpoints that require it
                api_key_field = provider_info.get('api_key_field')
                api_key_value = resolve_provider_api_key_value(
                    provider_name,
                    config_parser,
                    section_name,
                    api_key_field,
                )
                api_key_value = (
                    valid_provider_api_key(api_key_value)
                    or valid_provider_api_key(api_keys_by_provider.get(provider_name))
                    or valid_provider_api_key(api_keys_by_provider.get(_normalize_catalog_provider_for_chat(provider_name)))
                )

            # Always include the provider, but mark if it's configured
            # Get the models from config
            model_field = provider_info.get('model_field')
            models = []

            model_value = resolve_provider_model_value(
                provider_name,
                config_parser,
                section_name,
                model_field,
            )
            if model_value:
                models = parse_model_string(model_value)
            # Env-based model path for MLX when no config section is present
            if provider_name == "mlx" and not models:
                env_model = os.getenv("MLX_MODEL_PATH", "")
                if env_model:
                    models = parse_model_string(env_model)
                    is_configured = True
            if provider_name == "mlx" and models and not is_configured:
                is_configured = True
            configured_model_names = {str(m).strip() for m in models if str(m).strip()}
            provider_envelope = registry_capability_envelopes.get(provider_name)

            endpoint_scope: ConfiguredEndpointScope | None = None
            endpoint_policy: URLPolicyResult | None = None
            if endpoint_url:
                try:
                    endpoint_scope = ConfiguredEndpointScope.from_url(endpoint_url)
                    endpoint_policy = _egress.evaluate_url_policy(
                        endpoint_url,
                        configured_endpoint=endpoint_scope,
                    )
                except (TypeError, ValueError):
                    endpoint_policy = URLPolicyResult(
                        False,
                        "Invalid configured endpoint",
                        reason_code="invalid_url",
                    )

            discovery_result: ModelDiscoveryResult | None = commercial_discovery
            listed_inventory = provider_info['type'] == 'commercial' and has_model_listing(provider_name)
            if listed_inventory:
                models = (
                    list(discovery_result.models)
                    if discovery_result is not None and discovery_result.status == "ready" else []
                )
            elif provider_info['type'] == 'commercial':
                models = _with_pricing_models(provider_name, models)
            should_discover = (
                provider_info['type'] == 'local'
                and is_configured
                and endpoint_url is not None
                and endpoint_scope is not None
                and endpoint_policy is not None
                and endpoint_policy.allowed
                and bool(provider_info.get('model_discovery'))
                and (not models or endpoint_probe_enabled)
            )
            if should_discover:
                discovery_result = _coerce_model_discovery_result(
                    discover_models_from_endpoint(
                        provider_name,
                        endpoint_url,
                        provider_info.get('model_discovery', 'openai'),
                        api_key_value,
                        configured_endpoint=endpoint_scope,
                    )
                )
                if discovery_result.status == "ready":
                    models = _dedupe_preserve_order(
                        [*models, *discovery_result.models]
                    )

            provider_readiness = __provider_readiness.provider_readiness(
                provider_name=provider_name,
                provider_info=provider_info,
                is_configured=is_configured,
                endpoint_url=endpoint_url,
                api_key_value=api_key_value,
                current_availability=(
                    "disabled"
                    if override is not None and override.is_enabled is False
                    else (
                        provider_envelope.get("availability")
                        if isinstance(provider_envelope, dict)
                        else None
                    )
                ),
                health_entry=health_report.get(provider_name),
                supported_chat_providers=supported_chat_providers,
                endpoint_policy=endpoint_policy,
                discovery_result=discovery_result,
                has_explicit_models=bool(configured_model_names),
                endpoint_probe_enabled=endpoint_probe_enabled,
            )

            # Build models and metadata
            models_info = [get_model_metadata(provider_name, m) for m in models]
            if (
                provider_name == "llama"
                and models_info
                and endpoint_url
                and endpoint_scope is not None
                and endpoint_policy is not None
                and endpoint_policy.allowed
                and provider_readiness.get("provider_enabled") is not False
            ):
                external_vision = _discover_llamacpp_vision(
                    endpoint_url, api_key_value, configured_endpoint=endpoint_scope,
                )
                for model_info in models_info:
                    vision = external_vision.get(model_info["name"])
                    if vision is not None:
                        model_info["capabilities"] = {**model_info["capabilities"], "vision": vision}
                        model_info["vision_support"] = vision
                        inputs = [m for m in model_info["modalities"]["input"] if m != "image"]
                        model_info["modalities"] = {
                            **model_info["modalities"],
                            "input": [*inputs, "image"] if vision else inputs,
                        }
            if not include_deprecated and not listed_inventory:
                # Filter out deprecated models by default
                filtered = [mi for mi in models_info if not mi.get('deprecated', False)]
                models_info = filtered
                models = [mi['name'] for mi in models_info]

            llama_cpp_controls = _resolve_llama_cpp_control_caps(runtime_context) if provider_name == "llama" else None
            strict_mode_effective = _strict_token_counting_enabled_shared(default=False)
            for model_index, model_info in enumerate(models_info):
                model_name = str(model_info.get("name") or model_info.get("id") or "").strip()
                model_info["extra_body_compat"] = _safe_model_extra_body_compat(
                    provider_name,
                    model_name,
                    runtime_context,
                )
                if llama_cpp_controls is not None:
                    model_info["llama_cpp_controls"] = dict(llama_cpp_controls)
                should_probe_tokenizer, skip_reason = _should_probe_model_tokenizer(
                    provider_type=provider_info["type"],
                    model_info=model_info,
                    model_name=model_name,
                    configured_model_names=configured_model_names,
                    model_index=model_index,
                )
                if provider_readiness.get("provider_enabled") is False:
                    should_probe_tokenizer = False
                    skip_reason = (
                        str(provider_readiness.get("readiness_message") or "").strip()
                        or "Provider readiness check failed"
                    )
                if should_probe_tokenizer:
                    tokenizer_support = _resolve_model_tokenizer_support(provider_name, model_name, config_parser)
                else:
                    tokenizer_support = _skipped_model_tokenizer_support(
                        skip_reason or "Tokenizer probe skipped",
                        strict_mode_effective=strict_mode_effective,
                    )
                model_info["tokenizer_available"] = bool(tokenizer_support.get("available"))
                model_info["tokenizer"] = tokenizer_support.get("tokenizer")
                model_info["tokenizer_kind"] = tokenizer_support.get("kind")
                model_info["tokenizer_source"] = tokenizer_support.get("source")
                model_info["detokenize_available"] = bool(tokenizer_support.get("detokenize"))
                model_info["count_accuracy"] = tokenizer_support.get("count_accuracy", "unavailable")
                model_info["strict_mode_effective"] = bool(
                    tokenizer_support.get("strict_mode_effective", strict_mode_effective)
                )
                if tokenizer_support.get("error"):
                    model_info["tokenization_error"] = tokenizer_support.get("error")

            tokenizers_by_model = {
                str(model_info.get("name") or model_info.get("id") or "").strip(): {
                    "available": bool(model_info.get("tokenizer_available")),
                    "tokenizer": model_info.get("tokenizer"),
                    "kind": model_info.get("tokenizer_kind"),
                    "source": model_info.get("tokenizer_source"),
                    "detokenize": bool(model_info.get("detokenize_available")),
                    "count_accuracy": model_info.get("count_accuracy", "unavailable"),
                    "strict_mode_effective": bool(model_info.get("strict_mode_effective", False)),
                    "error": model_info.get("tokenization_error"),
                }
                for model_info in models_info
                if str(model_info.get("name") or model_info.get("id") or "").strip()
            }

            endpoint_only = provider_info['type'] == 'local' and is_configured and not models

            provider_data = {
                'name': provider_name,
                'display_name': provider_info['display_name'],
                'models': models,
                'model_inventory_source': 'provider' if listed_inventory else 'configured',
                # New: detailed metadata per model
                'models_info': models_info,
                'type': provider_info['type'],
                'default_model': models[0] if models else None,
                'is_configured': (
                    False
                    if provider_readiness.get("readiness_reason_code") == "missing_credentials"
                    else is_configured
                ),
                'provider_enabled': provider_readiness.get("provider_enabled"),
                'availability': provider_readiness.get("availability"),
                'readiness_reason_code': provider_readiness.get("readiness_reason_code"),
                'readiness_message': provider_readiness.get("readiness_message"),
                'chat_provider': provider_readiness.get("chat_provider"),
                'endpoint_only': endpoint_only,
                'extra_body_compat': _safe_provider_extra_body_compat(provider_name, runtime_context),
                'tokenizers': tokenizers_by_model or None,
            }
            if llama_cpp_controls is not None:
                provider_data['llama_cpp_controls'] = llama_cpp_controls

            # Add endpoint for local providers
            if provider_info['type'] == 'local':
                if endpoint_url:
                    provider_data['endpoint'] = endpoint_url
                else:
                    endpoint_field = provider_info.get('endpoint_field')
                    if (
                        endpoint_field
                        and section_name
                        and config_parser.has_section(section_name)
                        and config_parser.has_option(section_name, endpoint_field)
                    ):
                        provider_data['endpoint'] = config_parser.get(section_name, endpoint_field, fallback='')

            provider_data.update(configured_provider_generation_metadata(config_parser, section_name, provider_name))

            # Centralized capability diagnostics
            try:
                provider_data['requires_api_key'] = provider_requires_api_key(
                    _normalize_catalog_provider_for_chat(provider_name)
                )
                # Start with defaults from static map
                capabilities = dict(PROVIDER_CAPABILITIES.get(provider_name, {}))
                envelope = registry_capability_envelopes.get(provider_name)
                # Merge adapter-reported capabilities if available
                if envelope:
                    env_caps = envelope.get("capabilities")
                    if isinstance(env_caps, dict):
                        capabilities.update(env_caps)
                    provider_data["capability_envelope"] = {
                        "provider": provider_name,
                        "availability": envelope.get("availability", "unknown"),
                        "capabilities": env_caps if isinstance(env_caps, dict) else None,
                    }
                # Merge config-indicated streaming support as an override if provided
                if 'supports_streaming' not in capabilities and 'supports_streaming' in provider_data:
                    capabilities['supports_streaming'] = provider_data['supports_streaming']
                provider_data['capabilities'] = capabilities
            except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
                provider_data['requires_api_key'] = provider_info['type'] == 'commercial'

            # Attach live health if available
            try:
                if provider_name in health_report:
                    provider_data['health'] = health_report[provider_name]
            except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
                pass

            providers.append(provider_data)

        # Get the default provider from config
        default_api = 'openai'
        if config_parser.has_section('API') and config_parser.has_option('API', 'default_api'):
            default_api = config_parser.get('API', 'default_api', fallback='openai')

        # Strict policy: do not pull models from other sections.

        listing: dict[str, Any] = {
            'providers': providers,
            'default_provider': default_api,
            'total_configured': len(providers)
        }
        return _local_only_provider_listing(listing) if commercial_listings is None else listing

    except ByokResolutionError:
        logger.error("Provider credential store is unavailable")
        return _local_only_provider_listing({"default_provider": "openai"})
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        logger.error("Error getting configured providers")
        return {
            'providers': [],
            'default_provider': 'openai',
            'total_configured': 0,
            'error': 'An internal error occurred getting available providers. Please check your config.txt file for errors and try again. If the problem persists, please contact support for assistance.'
        }


async def get_configured_providers_async(
    include_deprecated: bool = False,
    refresh_openrouter: bool = False,
    credentials: _ProviderListingCredentials = _ProviderListingCredentials(),
) -> dict[str, Any]:
    """Run provider discovery in a worker thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        partial(
            get_configured_providers,
            include_deprecated=include_deprecated,
            refresh_openrouter=refresh_openrouter,
            credentials=credentials,
        ),
    )


def get_all_available_models() -> list[str]:
    """
    Get a flat list of all available models across all configured providers.

    Returns:
        List of all available model names
    """
    result = _llm_provider_overrides.apply_llm_provider_overrides_to_listing(get_configured_providers())
    models = []

    for provider in result.get('providers', []):
        for model in provider.get('models', []):
            # Add provider prefix to make models unique
            models.append(f"{provider['name']}/{model}")

    return models


def _normalize_filter_values(values: Optional[list[str]]) -> Optional[set[str]]:
    if not values:
        return None
    normalized = {str(v).strip().lower() for v in values if v and str(v).strip()}
    return normalized or None


_IMAGE_MODEL_HINTS = (
    "image",
    "dall-e",
    "dalle",
    "flux",
    "stable-diffusion",
    "sdxl",
    "midjourney",
    "recraft",
    "pixart",
    "playground",
    "kolors",
    "imagen",
)
_VIDEO_MODEL_HINTS = (
    "video",
    "veo",
    "sora",
    "kling",
    "hunyuan-video",
)
_AUDIO_MODEL_HINTS = (
    "whisper",
    "transcribe",
    "asr",
    "tts",
    "speech",
    "audio",
    "voice",
)
_NON_CHAT_MODEL_HINTS = (
    "rerank",
    "moderation",
    "safety",
)


def _contains_any_hint(value: str, hints: tuple[str, ...]) -> bool:
    return any(hint in value for hint in hints)


def _infer_model_type(model_info: dict[str, Any]) -> str:
    declared = model_info.get("type")
    if declared:
        return str(declared).strip().lower()
    name = str(model_info.get("name") or model_info.get("id") or "").lower()
    if "embed" in name or "embedding" in name:
        return "embedding"
    caps = model_info.get("capabilities")
    if isinstance(caps, dict) and caps.get("embedding"):
        return "embedding"
    if isinstance(caps, dict) and (
        caps.get("image_generation") or caps.get("image_output")
    ):
        return "image"
    if _contains_any_hint(name, _IMAGE_MODEL_HINTS):
        return "image"
    if isinstance(caps, dict) and caps.get("video_generation"):
        return "video"
    if _contains_any_hint(name, _VIDEO_MODEL_HINTS):
        return "video"
    if _contains_any_hint(name, _AUDIO_MODEL_HINTS):
        return "audio"
    if _contains_any_hint(name, _NON_CHAT_MODEL_HINTS):
        return "other"
    return "chat"


def _normalize_modalities(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip().lower() for v in value if v and str(v).strip()]
    return [str(value).strip().lower()]


def _llm_registry_capability_envelopes() -> dict[str, dict[str, Any]]:
    """
    Return provider capability envelopes from the LLM adapter registry.

    Preferred format is wrapper-level `list_capabilities()` entries:
      {"provider": str, "availability": str, "capabilities": dict|None}

    Falls back to legacy `get_all_capabilities()` (treated as enabled).
    """
    envelopes: dict[str, dict[str, Any]] = {}

    try:
        registry = llm_adapter_registry.get_registry()
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        return envelopes

    try:
        list_caps = getattr(registry, "list_capabilities", None)
        if callable(list_caps):
            try:
                entries = list_caps(include_disabled=True)
            except TypeError:
                entries = list_caps()
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    provider_name = str(entry.get("provider") or "").strip()
                    if not provider_name:
                        continue
                    availability = str(entry.get("availability") or "unknown").strip().lower() or "unknown"
                    raw_caps = entry.get("capabilities")
                    envelopes[provider_name] = {
                        "provider": provider_name,
                        "availability": availability,
                        "capabilities": raw_caps if isinstance(raw_caps, dict) else None,
                    }
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        pass

    if envelopes:
        return envelopes

    # Backward-compatible fallback for stubs exposing only get_all_capabilities().
    try:
        get_all = getattr(registry, "get_all_capabilities", None)
        legacy_caps = get_all() if callable(get_all) else None
        if isinstance(legacy_caps, dict):
            for provider_name, caps in legacy_caps.items():
                normalized_name = str(provider_name or "").strip()
                if not normalized_name:
                    continue
                envelopes[normalized_name] = {
                    "provider": normalized_name,
                    "availability": "enabled",
                    "capabilities": caps if isinstance(caps, dict) else None,
                }
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
        pass

    return envelopes


def _model_matches_filters(
    model_info: dict[str, Any],
    *,
    type_filters: Optional[set[str]],
    input_filters: Optional[set[str]],
    output_filters: Optional[set[str]],
) -> bool:
    model_type = _infer_model_type(model_info)
    if type_filters and model_type not in type_filters:
        return False

    modalities = model_info.get("modalities")
    input_mods: list[str] = []
    output_mods: list[str] = []
    if isinstance(modalities, dict):
        input_mods = _normalize_modalities(modalities.get("input"))
        output_mods = _normalize_modalities(modalities.get("output"))

    if not input_mods:
        input_mods = ["text"]
    if not output_mods:
        if model_type == "image":
            output_mods = ["image"]
        elif model_type == "embedding":
            output_mods = ["embedding"]
        else:
            output_mods = ["text"]

    if input_filters and not set(input_mods).intersection(input_filters):
        return False
    return not (output_filters and not set(output_mods).intersection(output_filters))


def _managed_llamacpp_profile_metadata_entries(supervisor: Any | None) -> list[dict[str, Any]]:
    """Return bounded model metadata entries for managed llama.cpp profiles."""
    if supervisor is None:
        return []
    try:
        profiles = supervisor.list_profiles()
    except _LLAMACPP_METADATA_EXCEPTIONS:
        logger.debug("Failed to list managed llama.cpp profiles for model metadata", exc_info=True)
        return []
    if not profiles:
        return []

    asset_scan_warnings: list[str] = []
    try:
        asset_scan = llamacpp_inventory_service.scan_assets()
        assets = asset_scan.assets
        if asset_scan.scan_limited:
            asset_scan_warnings.append(
                "Managed llama.cpp asset inventory scan was truncated; capability metadata may be incomplete."
            )
    except _LLAMACPP_METADATA_EXCEPTIONS:
        logger.debug("Failed to scan llama.cpp assets for model metadata", exc_info=True)
        assets = []
        asset_scan_warnings.append("Managed llama.cpp asset scan failed; capability metadata may be incomplete.")

    entries: list[dict[str, Any]] = []
    for profile in profiles:
        try:
            entry = managed_profile_model_metadata(profile, assets=assets)
        except _LLAMACPP_METADATA_EXCEPTIONS:
            logger.debug("Failed to build managed llama.cpp profile metadata", exc_info=True)
            continue
        if asset_scan_warnings:
            warnings = entry.setdefault("capability_warnings", [])
            if isinstance(warnings, list):
                warnings.extend(asset_scan_warnings)
            else:
                entry["capability_warnings"] = list(asset_scan_warnings)
        entries.append(entry)
    return entries


def _llamacpp_supervisor_from_request(request: Request | None) -> Any | None:
    """Return the managed llama.cpp supervisor attached to the app, if present."""
    if request is None:
        return None
    manager = getattr(request.app.state, "llm_manager", None)
    return getattr(manager, "llamacpp_supervisor", None)


async def listing_user_id(request: Request, response: Response) -> int | None:
    """The requesting user's id for a provider listing (TASK-15583), or None.

    The listings stay public: a request with no credentials, or one the auth layer refuses (any HTTPException),
    gets the server's listing, never an error. A principal with no user (a service) is no user. Every listing
    varies by credential, and a user's is private, so a shared cache neither stores a user's listing nor serves one
    caller's to another.
    """
    response.headers["Vary"] = "Authorization, X-API-KEY, Cookie"
    try:
        principal = await _auth_deps.get_auth_principal(request)
    except HTTPException:
        return None
    user_id = getattr(principal, "user_id", None)
    if not isinstance(user_id, int):
        return None
    response.headers["Cache-Control"] = "private, no-store"
    return user_id


def _own_key_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A provider entry listed on the requesting user's own key, held only to what every key refuses (TASK-15583).

    The owner's 2026-11-21 decision (TASK-15582): a disabled provider and a model on ``denied_models`` are refused
    on every key, a model off ``allowed_models`` only on the server's, so it stays. The override's ``config`` (its
    model list, default and ranking) shapes the server key's offer and is not applied.
    """
    snapshot = _llm_provider_overrides.capture_provider_override_call_snapshot(str(entry.get("name") or ""))
    models = [m for m in entry.get("models") or [] if snapshot.policy_error(m, allowlist=False) is None]
    default_model = entry.get("default_model")
    return {
        **entry,
        "enabled": snapshot.policy_error(None, allowlist=False) is None,
        "models": models,
        "models_info": [
            info for info in entry.get("models_info") or [] if isinstance(info, dict) and info.get("name") in models
        ],
        "default_model": default_model if default_model in models else (models[0] if models else None),
    }


async def _users_listing(user_id: Any, **options: Any) -> dict[str, Any]:
    """The provider listing for one requester, with the admin provider override applied (TASK-15583).

    Each commercial provider the user's own stored key serves (``users_own_key_credentials``: BYOK on, the provider
    allowlisted, an active key, the vendor's host) is listed on that key: its readiness and models come from that
    key's own model list. The handle is asked for without the override check (``policy=False``): this listing applies
    the override itself (``_resolve_commercial_listing`` skips a disabled provider; ``_own_key_entry`` refuses what
    every key refuses), so a denied default model cannot hide the provider. Every other entry, and the whole listing
    for a request with no user (anonymous, or a direct call), is the server's, exactly as before.
    """
    own_keys: dict[str, Any] = {}
    if isinstance(user_id, int):
        # ponytail: one own-key lookup per cloud provider (a store read each when BYOK is on); list the user's
        # stored providers once if listings show up in profiles.
        for provider in sorted(CLOUD_MODEL_PROVIDERS):
            try:
                snapshot = _llm_provider_overrides.capture_provider_override_call_snapshot(provider)
                disabled = snapshot.policy_error(None, allowlist=False) is not None
            except ByokResolutionError as error:
                if error.code != _CATALOG_CREDENTIAL_STORE_UNAVAILABLE:
                    raise
                return await get_configured_providers_async(
                    **options, credentials=_ProviderListingCredentials(local_only=True),
                )
            except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
                disabled = True
            if disabled:
                continue  # a disabled provider's stored key is never touched (a lookup can refresh an OAuth token)
            handle = await users_own_key_credentials(user_id, provider, policy=False)
            if handle is not None:
                own_keys[provider] = handle
    listing = await get_configured_providers_async(
        **options, credentials=_ProviderListingCredentials(own_keys=own_keys),
    )
    try:
        if not own_keys:
            return _llm_provider_overrides.apply_llm_provider_overrides_to_listing(listing)
        providers: list[Any] = []
        for entry in listing.get("providers", []):
            if isinstance(entry, dict) and entry.get("name") in own_keys:
                providers.append(_own_key_entry(entry))
            else:
                applied = _llm_provider_overrides.apply_llm_provider_overrides_to_listing({"providers": [entry]})
                providers += applied["providers"]
        return {**listing, "providers": providers}
    except ByokResolutionError as error:
        if error.code != _CATALOG_CREDENTIAL_STORE_UNAVAILABLE:
            raise
        logger.warning("Provider credential store became unavailable while publishing model discovery")
        return _local_only_provider_listing(listing)


#######################################################################################################################
#
# Endpoints:

@router.get("/llm/providers",
    summary="Get configured LLM providers",
    description="Returns a list of all configured LLM providers with their models from config",
    response_model=dict[str, Any])
async def get_llm_providers(
    include_deprecated: bool = False,
    user_id: int | None = Depends(listing_user_id),
):
    """
    Get all configured LLM providers and their models.

    Returns:
        Dictionary containing:
        - providers: List of provider configurations
        - default_provider: The default provider name
        - total_configured: Number of configured providers

    A signed-in user's listing puts each provider the user's own key serves on that key (TASK-15583).
    """
    try:
        result = await _users_listing(user_id, include_deprecated=include_deprecated)

        # Inject Diagnostics UI interval bounds from server config if available
        try:
            cfg = _config.load_comprehensive_config()
            section = 'LLM_Diagnostics'
            def _getint(key: str, fallback: int) -> int:
                try:
                    return cfg.getint(section, key, fallback=fallback)
                except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
                    return fallback

            qs_min = _getint('queue_status_auto_min_secs', 1)
            qs_max = _getint('queue_status_auto_max_secs', 60)
            qa_min = _getint('queue_activity_auto_min_secs', 1)
            qa_max = _getint('queue_activity_auto_max_secs', 60)
            # Normalize if misconfigured
            if qs_min > qs_max:
                qs_min, qs_max = qs_max, qs_min
            if qa_min > qa_max:
                qa_min, qa_max = qa_max, qa_min
            result['diagnostics_ui'] = {
                'queue_status_auto': {'min': int(max(1, qs_min)), 'max': int(max(1, qs_max))},
                'queue_activity_auto': {'min': int(max(1, qa_min)), 'max': int(max(1, qa_max))},
            }
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            # Best-effort; omit diagnostics_ui on failure
            pass

        if result['total_configured'] == 0:
            logger.warning("No LLM providers are configured")
            return {
                'providers': [],
                'default_provider': None,
                'total_configured': 0,
                'message': 'No LLM providers are configured. Please check your config.txt file.'
            }

        logger.info(f"Found {result['total_configured']} configured LLM providers")
        return result

    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error in get_llm_providers endpoint")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve LLM providers"
        ) from e


# Optional sampling settings the chat UI offers that only some providers accept.
_OPTIONAL_SAMPLING_FIELDS = ("top_k", "min_p")


def _optional_sampling_fields(provider: Any) -> list[str]:
    """Return the optional sampling fields a provider accepts (#3283).

    The chat UI sends Top K and Min P only when these are listed, because a
    provider's adapter rejects request fields it does not support
    (``capability_registry.validate_payload``).

    Args:
        provider: The provider key of a catalog entry.

    Returns:
        The supported fields among ``top_k`` and ``min_p``, in that order.
    """
    if not isinstance(provider, str) or not provider.strip():
        return []
    # Catalog names ("llama", "tabby") differ from the registry keys.
    allowed = get_allowed_fields(_normalize_catalog_provider_for_chat(provider))
    return [field for field in _OPTIONAL_SAMPLING_FIELDS if field in allowed]


@router.get("/llm/models/metadata",
    summary="Get model metadata across providers",
    description=(
        "Returns flattened model metadata for all providers (chat, embeddings, image). "
        "Image backends appear with type=image and modalities output=image."
    ),
    response_model=dict[str, Any])
async def get_models_metadata(
    request: Request,
    include_deprecated: bool = False,
    refresh_openrouter: bool = Query(
        False,
        description=(
            "When true, refresh OpenRouter model IDs from OpenRouter /models before"
            " building the catalog response."
        ),
    ),
    model_type: Optional[list[str]] = Query(
        None,
        alias="type",
        description="Optional model type filter (repeatable).",
    ),
    input_modality: Optional[list[str]] = Query(
        None,
        description="Optional input modality filter (repeatable).",
    ),
    output_modality: Optional[list[str]] = Query(
        None,
        description="Optional output modality filter (repeatable).",
    ),
    user_id: int | None = Depends(listing_user_id),
):
    try:
        type_filters = _normalize_filter_values(model_type)
        input_filters = _normalize_filter_values(input_modality)
        output_filters = _normalize_filter_values(output_modality)
        result = await _users_listing(
            user_id,
            include_deprecated=include_deprecated,
            refresh_openrouter=refresh_openrouter,
        )
        flattened: list[dict[str, Any]] = []
        for provider in result.get('providers', []):
            provider_is_configured = bool(provider.get('is_configured'))
            provider_enabled = provider.get('provider_enabled')
            for mi in provider.get('models_info', []):
                entry = {
                    'provider': provider.get('name'),
                    **mi,
                }
                entry['provider_is_configured'] = provider_is_configured
                entry['is_configured'] = provider_is_configured
                entry['catalog_only'] = not provider_is_configured
                entry['sampling_fields'] = _optional_sampling_fields(entry.get('provider'))
                if isinstance(provider_enabled, bool):
                    entry['provider_enabled'] = provider_enabled
                for key in (
                    'availability',
                    'readiness_reason_code',
                    'readiness_message',
                    'chat_provider',
                ):
                    if provider.get(key) is not None:
                        entry[key] = provider.get(key)
                if not _model_matches_filters(
                    entry,
                    type_filters=type_filters,
                    input_filters=input_filters,
                    output_filters=output_filters,
                ):
                    continue
                flattened.append(entry)
        managed_entries = await asyncio.to_thread(
            _managed_llamacpp_profile_metadata_entries,
            _llamacpp_supervisor_from_request(request),
        )
        for entry in managed_entries:
            entry.setdefault('sampling_fields', _optional_sampling_fields(entry.get('provider')))
            if not _model_matches_filters(
                entry,
                type_filters=type_filters,
                input_filters=input_filters,
                output_filters=output_filters,
            ):
                continue
            flattened.append(entry)
        # Append image generation backends to the catalog
        try:
            image_models = _listing.list_image_models_for_catalog()
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to list image generation models")
            image_models = []
        for entry in image_models:
            if not _model_matches_filters(
                entry,
                type_filters=type_filters,
                input_filters=input_filters,
                output_filters=output_filters,
            ):
                continue
            flattened.append(entry)
        metadata: dict[str, Any] = {'models': flattened, 'total': len(flattened)}
        if 'error_code' in result:
            metadata['error_code'] = result['error_code']
        return metadata
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting models metadata")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model metadata"
        ) from e

@router.get("/llm/providers/{provider_name}",
    summary="Get specific provider details",
    description="Returns details for a specific LLM provider",
    response_model=dict[str, Any])
async def get_provider_details(
    provider_name: str,
    include_deprecated: bool = False,
    user_id: int | None = Depends(listing_user_id),
):
    """
    Get details for a specific LLM provider.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')

    Returns:
        Provider details including models and configuration
    """
    try:
        result = await _users_listing(user_id, include_deprecated=include_deprecated)

        # Find the specific provider
        for provider in result['providers']:
            if provider['name'] == provider_name:
                logger.info(f"Retrieved details for provider: {provider_name}")
                return provider

        # Provider not found or not configured
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_name}' is not configured or not found"
        )

    except HTTPException:
        raise
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting provider details")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve provider details"
        ) from e

@router.get("/llm/models",
    summary="Get all available models",
    description=(
        "Returns a flat list of all available models across all providers. "
        "Includes image backends as image/<backend> entries."
    ),
    response_model=list[str])
async def get_all_models(
    include_deprecated: bool = False,
    model_type: Optional[list[str]] = Query(
        None,
        alias="type",
        description="Optional model type filter (repeatable).",
    ),
    input_modality: Optional[list[str]] = Query(
        None,
        description="Optional input modality filter (repeatable).",
    ),
    output_modality: Optional[list[str]] = Query(
        None,
        description="Optional output modality filter (repeatable).",
    ),
    user_id: int | None = Depends(listing_user_id),
):
    """
    Get all available models from all configured providers.

    Returns:
        List of model names with provider prefix
    """
    try:
        type_filters = _normalize_filter_values(model_type)
        input_filters = _normalize_filter_values(input_modality)
        output_filters = _normalize_filter_values(output_modality)
        result = await _users_listing(user_id, include_deprecated=include_deprecated)
        models: list[str] = []
        for provider in result.get('providers', []):
            provider_name = provider.get('name') or "unknown"
            for model in provider.get('models', []):
                entry = {
                    "provider": provider_name,
                    **get_model_metadata(provider_name, model),
                }
                if not _model_matches_filters(
                    entry,
                    type_filters=type_filters,
                    input_filters=input_filters,
                    output_filters=output_filters,
                ):
                    continue
                models.append(f"{provider_name}/{model}")
        # Append image generation backends to the flat list
        try:
            image_models = _listing.list_image_models_for_catalog()
        except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to list image generation models")
            image_models = []
        for entry in image_models:
            if not _model_matches_filters(
                entry,
                type_filters=type_filters,
                input_filters=input_filters,
                output_filters=output_filters,
            ):
                continue
            model_id = entry.get("id") or f"image/{entry.get('name') or ''}"
            if model_id and model_id != "image/":
                models.append(str(model_id))
        logger.info(f"Found {len(models)} total models across all providers")
        return models
    except _LLM_PROVIDERS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting all models")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve models"
        ) from e

# End of llm_providers.py
