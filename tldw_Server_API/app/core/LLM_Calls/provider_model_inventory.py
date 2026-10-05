"""Bounded, provider-authoritative cloud model discovery.

Qwen uses its documented native model-list API on the generation origin.
Hugging Face discovery is limited to the global chat router, not Hub listings or
model-specific inference routes. Bedrock control-plane discovery is not yet
implemented; Z.AI and MiniMax lack a verified listing contract in this module.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit

from tldw_Server_API.app.core.http_client import RetryPolicy
from tldw_Server_API.app.core.http_client import fetch as _http_fetch
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

if TYPE_CHECKING:
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy

CLOUD_MODEL_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "bedrock",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "mistral",
        "openrouter",
        "qwen",
        "moonshot",
        "zai",
        "novita",
        "poe",
        "together",
        "minimax",
    }
)
_SUPPORTED = frozenset(
    {
        "openai",
        "anthropic",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "mistral",
        "moonshot",
        "openrouter",
        "qwen",
        "novita",
        "poe",
        "together",
    }
)
_ENDPOINT_ADAPTERS = {
    "openai": ("openai_adapter", "OpenAIAdapter"),
    "anthropic": ("anthropic_adapter", "AnthropicAdapter"),
    "deepseek": ("deepseek_adapter", "DeepSeekAdapter"),
    "google": ("google_adapter", "GoogleAdapter"),
    "groq": ("groq_adapter", "GroqAdapter"),
    "mistral": ("mistral_adapter", "MistralAdapter"),
    "openrouter": ("openrouter_adapter", "OpenRouterAdapter"),
    "qwen": ("qwen_adapter", "QwenAdapter"),
    "huggingface": ("huggingface_adapter", "HuggingFaceAdapter"),
    "novita": ("custom_openai_adapter", "NovitaAdapter"),
    "poe": ("custom_openai_adapter", "PoeAdapter"),
    "together": ("custom_openai_adapter", "TogetherAdapter"),
}
_TIMEOUT_SECONDS = 5.0
_TTL_SECONDS = 300.0
_MAX_PAGES = 10
_MAX_MODELS = 1000
_MAX_CACHE_ENTRIES = 128
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_CACHE: OrderedDict[tuple[str, str, str], tuple[float, ModelDiscoveryResult]] = OrderedDict()
_IN_FLIGHT: dict[tuple[str, str, str], object] = {}
_CACHE_LOCK = threading.Lock()


def _safe_base_url(value: Any) -> str | None:
    """Reject unsafe URL syntax before credential-bearing HTTP dispatch."""
    if (
        not isinstance(value, str)
        or not value
        or any(char.isspace() or ord(char) < 32 for char in value)
        or "\\" in value
    ):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "?" in value
            or "#" in value
            or parsed.port == 0
        ):
            return None
    except ValueError:
        return None
    return value.rstrip("/")


def resolve_provider_models_base_url(
    provider: str,
    app_config: dict[str, Any] | None = None,
    *,
    credentials_resolved: bool = False,
    base_url: str | None = None,
) -> str | None:
    """Resolve the actual chat adapter base without widening egress policy.

    Imports are lazy because adapter_utils imports chat_service, which can itself
    consume this inventory. Runtime credentials suppress adapter environment
    fallback exactly as they do during generation. Invalid bases never fall back.
    """
    provider = (provider or "").strip().lower()
    if provider not in CLOUD_MODEL_PROVIDERS:
        return None
    if base_url is not None:
        return _safe_base_url(base_url)
    try:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

        config = ensure_app_config(app_config)
        if not isinstance(config, dict):
            return None
        request = {"app_config": config, "credentials_resolved": credentials_resolved}
        if provider in {"cohere", "moonshot", "zai"}:
            # These functional adapters use the same config field directly.
            defaults = {
                "cohere": "https://api.cohere.ai",
                "moonshot": "https://api.moonshot.cn/v1",
                "zai": "https://api.z.ai/api/paas/v4",
            }
            base = (config.get(f"{provider}_api") or {}).get("api_base_url", defaults[provider])
            return _safe_base_url(base)
        endpoint_adapter = _ENDPOINT_ADAPTERS.get(provider)
        if endpoint_adapter is None:
            return None
        # Endpoint-only resolution must not depend on generation enablement or
        # startup credentials. These adapters have no SDK-initializing constructor.
        module_name, class_name = endpoint_adapter
        adapter_type = getattr(import_module(f"tldw_Server_API.app.core.LLM_Calls.providers.{module_name}"), class_name)
        adapter = adapter_type()
        if provider in {"openai", "anthropic", "groq", "mistral", "openrouter"}:
            base = adapter._resolve_base_url(request)
        elif provider in {"deepseek", "qwen"}:
            base = adapter._base_url(config, request)
        elif provider == "google":
            base = adapter._base_url(request)
        elif provider == "huggingface":
            resolved = adapter._resolve_url_and_headers(request)
            url = resolved["url"]
            if not url.endswith("/chat/completions"):
                return None
            base = url.removesuffix("/chat/completions")
            if not _is_huggingface_router_base(base):
                return None
        elif provider in {"novita", "poe", "together"}:
            base = adapter._resolve_base(request)
        else:
            return None
        return _safe_base_url(base)
    except Exception:  # noqa: BLE001 - configuration failure must not reveal credentials
        return None


def _is_huggingface_router_base(base: str) -> bool:
    parsed = urlsplit(base)
    return (
        parsed.hostname == "router.huggingface.co"
        and parsed.port in {None, 443}
        and parsed.path in {"/v1", "/v1/models"}
    )


def _models_url(provider: str, base: str) -> str | None:
    if provider == "huggingface" and not _is_huggingface_router_base(base):
        return None
    if provider == "openrouter":
        return base + ("/user" if base.endswith("/models") else "/models/user")
    if base.endswith("/models"):
        return base
    if provider == "qwen":
        if base.endswith("/compatible-mode/v1"):
            return base.removesuffix("/compatible-mode/v1") + "/api/v1/models"
        if base.endswith("/api/v1"):
            return base + "/models"
        return None
    if provider in {"novita", "poe", "together"}:
        if base.endswith("/chat/completions"):
            return base.removesuffix("/chat/completions") + "/models"
        return base + ("/models" if base.endswith("/v1") else "/v1/models")
    if provider in {"anthropic", "cohere"} and not base.endswith("/v1"):
        return base + "/v1/models"
    return base + "/models"


def _identifier(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(char.isspace() or ord(char) < 32 for char in value)
    ):
        raise ValueError("Invalid model identifier")
    return value


def _parse_page(provider: str, payload: Any, *, page_number: int = 1) -> tuple[list[str], str | None, int]:
    """Extract exact IDs and one documented cursor; never invent aliases."""
    cursor = None
    if provider == "qwen":
        output = payload.get("output") if isinstance(payload, dict) else None
        if not isinstance(output, dict) or payload.get("success") is not True:
            raise ValueError("Unsupported inventory shape")
        total, page, size = output.get("total"), output.get("page_no"), output.get("page_size")
        if (
            type(total) is not int
            or not 0 <= total <= _MAX_MODELS
            or type(page) is not int
            or page != page_number
            or type(size) is not int
            or not 1 <= size <= _MAX_MODELS
        ):
            raise ValueError("Invalid pagination")
        rows = output.get("models")
        remaining = total - (page - 1) * size
        if not isinstance(rows, list) or remaining < 0 or len(rows) != min(size, remaining):
            raise ValueError("Incomplete model page")
        if remaining > size:
            cursor = str(page + 1)
    elif provider == "together":
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("models" if provider in {"google", "cohere"} else "data")
    else:
        raise ValueError("Unsupported inventory shape")
    if not isinstance(rows, list) or len(rows) > _MAX_MODELS:
        raise ValueError("Unsupported inventory collection")
    models = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Unsupported inventory entry")
        if provider == "qwen":
            metadata = row.get("inference_metadata")
            if metadata is not None:
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid model capabilities")
                modalities = metadata.get("response_modality")
                if not isinstance(modalities, list):
                    raise ValueError("Invalid output modalities")
                if "Text" not in modalities:
                    continue
            elif "capabilities" in row:
                capabilities = row["capabilities"]
                if not isinstance(capabilities, list):
                    raise ValueError("Invalid model capabilities")
                if not {"TG", "Reasoning"}.intersection(capabilities):
                    continue
            model = _identifier(row.get("model"))
        elif provider == "google":
            methods = row.get("supportedGenerationMethods")
            if not isinstance(methods, list) or "generateContent" not in methods:
                continue
            name = _identifier(row.get("name"))
            if not name.startswith("models/"):
                raise ValueError("Invalid model resource")
            model = _identifier(name.removeprefix("models/"))
        elif provider == "cohere":
            endpoints = row.get("endpoints")
            if row.get("is_deprecated") is True or not isinstance(endpoints, list) or "chat" not in endpoints:
                continue
            if not isinstance(row.get("is_deprecated", False), bool):
                raise ValueError("Invalid model lifecycle")
            model = _identifier(row.get("name"))
        else:
            if provider == "together" and "type" in row and row["type"] != "chat":
                continue
            if provider == "mistral" and "capabilities" in row:
                capabilities = row["capabilities"]
                if not isinstance(capabilities, dict):
                    raise ValueError("Invalid model capabilities")
                if "completion_chat" in capabilities:
                    if not isinstance(capabilities["completion_chat"], bool):
                        raise ValueError("Invalid chat capability")
                    if not capabilities["completion_chat"]:
                        continue
            if provider in {"openrouter", "huggingface"} and "architecture" in row:
                architecture = row["architecture"]
                if not isinstance(architecture, dict):
                    raise ValueError("Invalid model architecture")
                if "output_modalities" in architecture:
                    modalities = architecture["output_modalities"]
                    if not isinstance(modalities, list):
                        raise ValueError("Invalid output modalities")
                    if "text" not in modalities:
                        continue
            model = _identifier(row.get("id"))
        models.append(model)

    if provider == "anthropic":
        has_more = payload.get("has_more")
        if not isinstance(has_more, bool):
            raise ValueError("Invalid pagination")
        if has_more:
            cursor = _identifier(payload.get("last_id"))
            if not rows or cursor != rows[-1].get("id"):
                raise ValueError("Invalid pagination cursor")
    elif provider in {"google", "cohere"}:
        cursor = payload.get("nextPageToken" if provider == "google" else "next_page_token")
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 2048):
            raise ValueError("Invalid pagination cursor")
        cursor = cursor or None
    elif isinstance(payload, dict) and payload.get("has_more"):
        raise ValueError("Unsupported pagination")
    return models, cursor, len(rows)


def discover_provider_models(
    provider: str,
    api_key: str | None,
    *,
    base_url: str | None = None,
    force_refresh: bool = False,
    fetch_fn: Callable[..., Any] | None = None,
    call_policy: ProviderCallPolicy | None = None,
) -> ModelDiscoveryResult:
    """Return a complete authoritative inventory or a sanitized failure.

    All pages share a five-second monotonic budget and one-attempt HTTP timeouts.
    Limits abort discovery rather than publishing a partial catalog. Only ready
    results (including empty inventories) are cached, scoped by provider, endpoint
    and SHA-256 credential digest. A failed refresh evicts prior cached success.
    """
    provider = (provider or "").strip().lower()
    if provider not in _SUPPORTED:
        return ModelDiscoveryResult("unsupported")
    if not isinstance(api_key, str) or not api_key.strip() or any(ord(char) < 32 for char in api_key):
        return ModelDiscoveryResult("auth_failed")
    key = api_key.strip()
    budget = _TIMEOUT_SECONDS
    scope = None
    if call_policy is not None:
        from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy

        if not isinstance(call_policy, ProviderCallPolicy):
            return ModelDiscoveryResult("unsupported")
        if call_policy.maximum_timeout_seconds is not None:
            budget = min(budget, float(call_policy.maximum_timeout_seconds))
        scope = call_policy.required_endpoint_scope
    deadline = time.monotonic() + budget
    base = _safe_base_url(base_url) if base_url is not None else resolve_provider_models_base_url(provider)
    if base is None:
        return ModelDiscoveryResult("unsupported")
    endpoint = _models_url(provider, base)
    if endpoint is None:
        return ModelDiscoveryResult("unsupported")
    if scope is not None and not scope.matches(endpoint):
        return ModelDiscoveryResult("unsupported")
    cache_key = (provider, endpoint, hashlib.sha256(key.encode("utf-8")).hexdigest())
    refresh_token = object()
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if not force_refresh and cached and time.monotonic() - cached[0] < _TTL_SECONDS:
            return cached[1]
        _CACHE.pop(cache_key, None)
        _IN_FLIGHT[cache_key] = refresh_token

    headers = {"Accept": "application/json"}
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    elif provider == "google":
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    cursor_parameter = {"anthropic": "after_id", "google": "pageToken", "cohere": "page_token"}.get(provider)
    fetcher = fetch_fn or _http_fetch
    models: dict[str, None] = {}
    cursors: set[str] = set()
    cursor = None
    count = 0
    qwen_total = None
    try:
        for _page in range(_MAX_PAGES):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ModelDiscoveryResult("unreachable")
            if provider == "qwen":
                url = endpoint + "?" + urlencode({"page_no": _page + 1, "page_size": 100})
            else:
                url = endpoint + ("?" + urlencode({cursor_parameter: cursor}) if cursor else "")
            response = fetcher(
                method="GET",
                url=url,
                headers=headers,
                timeout=remaining,
                deadline=deadline,
                retry=RetryPolicy(attempts=1),
                allow_redirects=False,
                sensitive_observability=True,
                configured_endpoint=scope,
                max_response_bytes=_MAX_RESPONSE_BYTES,
            )
            try:
                status = response.status_code
                if status in {401, 403, 498}:
                    return ModelDiscoveryResult("auth_failed")
                if status == 429 or status >= 500:
                    return ModelDiscoveryResult("server_error")
                if status != 200:
                    return ModelDiscoveryResult("unsupported")
                if time.monotonic() >= deadline:
                    return ModelDiscoveryResult("unreachable")
                try:
                    payload = response.json()
                    page_models, cursor, page_count = _parse_page(provider, payload, page_number=_page + 1)
                    if provider == "qwen":
                        total = payload["output"]["total"]
                        if qwen_total is not None and qwen_total != total:
                            return ModelDiscoveryResult("unsupported")
                        qwen_total = total
                except (ValueError, TypeError, KeyError):
                    return ModelDiscoveryResult("unsupported")
            finally:
                response.close()
            count += page_count
            if count > _MAX_MODELS:
                return ModelDiscoveryResult("unsupported")
            models.update(dict.fromkeys(page_models))
            if time.monotonic() >= deadline:
                return ModelDiscoveryResult("unreachable")
            if not cursor:
                result = ModelDiscoveryResult("ready", tuple(models))
                with _CACHE_LOCK:
                    # Only the latest refresh may publish, even if it failed
                    # while an older request was still in flight.
                    if _IN_FLIGHT.get(cache_key) is refresh_token:
                        _CACHE[cache_key] = (time.monotonic(), result)
                        while len(_CACHE) > _MAX_CACHE_ENTRIES:
                            _CACHE.popitem(last=False)
                return result
            if cursor in cursors:
                return ModelDiscoveryResult("unsupported")
            cursors.add(cursor)
        return ModelDiscoveryResult("unsupported")
    except Exception:  # noqa: BLE001 - discovery failures expose no provider response or secrets
        return ModelDiscoveryResult("unreachable")
    finally:
        with _CACHE_LOCK:
            if _IN_FLIGHT.get(cache_key) is refresh_token:
                _IN_FLIGHT.pop(cache_key, None)
