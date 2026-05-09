from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, quote, quote_plus, urlparse

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_config_option_names,
    custom_openai_provider_name,
    custom_openai_provider_number,
    custom_openai_section_name,
    iter_custom_openai_provider_numbers,
)
from tldw_Server_API.app.core.exceptions import TokenizerUnavailable

_TRUE_VALUES = {"1", "true", "yes", "on"}

TOKENIZER_PROVIDER_ALIASES: dict[str, str] = {
    "llama.cpp": "llama",
    "llama_cpp": "llama",
    "custom-openai-api": "custom_openai_api",
    "custom_openai": "custom_openai_api",
    "custom-openai": "custom_openai_api",
    "gemini": "google",
}

PROVIDER_NATIVE_TOKENIZER_CONFIG: dict[str, dict[str, str]] = {
    "llama": {
        "section": "Local-API",
        "endpoint_field": "llama_api_IP",
        "api_key_field": "llama_api_key",
        "label": "llama.cpp",
    },
    "kobold": {
        "section": "Local-API",
        "endpoint_field": "kobold_api_IP",
        "api_key_field": "kobold_api_key",
        "label": "kobold.cpp",
    },
    "ooba": {
        "section": "Local-API",
        "endpoint_field": "ooba_api_IP",
        "api_key_field": "ooba_api_key",
        "label": "oobabooga",
    },
    "tabby": {
        "section": "Local-API",
        "endpoint_field": "tabby_api_IP",
        "api_key_field": "tabby_api_key",
        "label": "tabbyapi",
    },
    "vllm": {
        "section": "Local-API",
        "endpoint_field": "vllm_api_IP",
        "api_key_field": "vllm_api_key",
        "label": "vllm",
    },
    "ollama": {
        "section": "Local-API",
        "endpoint_field": "ollama_api_IP",
        "api_key_field": "ollama_api_key",
        "label": "ollama",
    },
    "aphrodite": {
        "section": "Local-API",
        "endpoint_field": "aphrodite_api_IP",
        "api_key_field": "aphrodite_api_key",
        "label": "aphrodite",
    },
    "custom_openai_api": {
        "section": "API",
        "endpoint_field": "custom_openai_api_ip",
        "api_key_field": "custom_openai_api_key",
        "label": "custom-openai-api",
    },
}
PROVIDER_NATIVE_TOKENIZER_CONFIG.update(
    {
        custom_openai_section_name(provider_number): {
            "section": "API",
            "endpoint_field": custom_openai_config_option_names(provider_number, "ip")[0],
            "api_key_field": custom_openai_config_option_names(provider_number, "key")[0],
            "label": custom_openai_provider_name(provider_number),
        }
        for provider_number in iter_custom_openai_provider_numbers(start=3)
    }
)

COMMERCIAL_EXACT_TOKENIZER_CONFIG: dict[str, dict[str, Any]] = {
    "bedrock": {
        "section": "API",
        "label": "bedrock",
        "mode": "bedrock-count-only",
    },
    "anthropic": {
        "section": "API",
        "api_key_field": "anthropic_api_key",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_fields": ("anthropic_api_base_url", "anthropic_api_base"),
        "base_url_env": "ANTHROPIC_BASE_URL",
        "base_url_default": "https://api.anthropic.com/v1",
        "label": "anthropic",
        "mode": "count-only",
    },
    "google": {
        "section": "API",
        "api_key_field": "google_api_key",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url_fields": ("google_api_base_url", "google_api_base"),
        "base_url_env": "GOOGLE_GEMINI_BASE_URL",
        "base_url_default": "https://generativelanguage.googleapis.com/v1beta",
        "label": "google",
        "mode": "count-only",
    },
    "cohere": {
        "section": "API",
        "api_key_field": "cohere_api_key",
        "api_key_env": "COHERE_API_KEY",
        "base_url_fields": ("cohere_api_base_url", "cohere_api_base"),
        "base_url_env": "COHERE_BASE_URL",
        "base_url_default": "https://api.cohere.ai",
        "label": "cohere",
        "mode": "tokenize",
    },
}

NATIVE_TOKENIZER_STRIP_SUFFIXES = (
    "/api/v1/generate",
    "/v1/generate",
    "/api/generate",
    "/api/chat",
    "/v1/chat",
    "/api/v1/chat/completions",
    "/v1/messages/count_tokens",
    "/v1/messages",
    "/messages/count_tokens",
    "/messages",
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
    "/completions",
    "/completion",
)


@dataclass
class TokenizerResolution:
    available: bool
    tokenizer: str | None
    kind: str | None
    source: str | None
    detokenize_available: bool
    count_accuracy: str
    strict_mode_effective: bool
    encoding: Any = None
    error: str | None = None

    def as_support_dict(self) -> dict[str, Any]:
        payload = {
            "available": bool(self.available),
            "tokenizer": self.tokenizer,
            "kind": self.kind,
            "source": self.source,
            "detokenize": bool(self.detokenize_available),
            "count_accuracy": self.count_accuracy,
            "strict_mode_effective": bool(self.strict_mode_effective),
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class BedrockSigV4Credentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


class ProviderNativeTokenizerHTTPAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        model: str | None,
        api_key: str | None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _request_json(self, paths: tuple[str, ...], payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for path in paths:
            url = f"{self.base_url}{path}"
            try:
                response = _http_post(url=url, payload=payload, headers=headers, timeout=self.timeout_seconds)
            except Exception as exc:
                last_error = exc
                continue

            if response.status_code == 404:
                continue
            if response.status_code >= 400:
                raise TokenizerUnavailable(
                    f"Provider-native tokenizer endpoint error ({response.status_code})"
                )
            try:
                data = response.json()
            except Exception as exc:
                raise TokenizerUnavailable("Provider-native tokenizer returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise TokenizerUnavailable("Provider-native tokenizer returned invalid payload")
            return data

        if last_error is not None:
            raise TokenizerUnavailable("Provider-native tokenizer endpoint unavailable") from last_error
        raise TokenizerUnavailable("Provider-native tokenizer endpoint not found")

    def encode(self, text: str) -> list[int]:
        payload: dict[str, Any] = {
            "content": text,
            "prompt": text,
            "add_special": False,
            "with_pieces": False,
        }
        if self.model:
            payload["model"] = self.model
        data = self._request_json(("/api/tokenize", "/tokenize", "/v1/tokenize"), payload)
        raw_tokens = data.get("tokens")
        if not isinstance(raw_tokens, list):
            raw_tokens = data.get("token_ids")
        if not isinstance(raw_tokens, list):
            raise TokenizerUnavailable("Provider-native tokenizer response missing token ids")
        try:
            return [int(token_id) for token_id in raw_tokens]
        except Exception as exc:
            raise TokenizerUnavailable("Provider-native tokenizer returned invalid token ids") from exc

    def decode(self, token_ids: list[int]) -> str:
        payload: dict[str, Any] = {"tokens": [int(token_id) for token_id in token_ids]}
        if self.model:
            payload["model"] = self.model
        data = self._request_json(("/api/detokenize", "/detokenize", "/v1/detokenize"), payload)
        for key in ("content", "text", "decoded", "value"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        raise TokenizerUnavailable("Provider-native detokenize response missing text payload")


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            try:
                return int(stripped)
            except Exception:
                return None
    return None


def _extract_int_value(payload: Any, keys: tuple[str, ...]) -> int | None:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                parsed = _coerce_int(payload.get(key))
                if parsed is not None:
                    return parsed
        for value in payload.values():
            nested = _extract_int_value(value, keys)
            if nested is not None:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _extract_int_value(item, keys)
            if nested is not None:
                return nested
    return None


class _HTTPJSONAdapterBase:
    def __init__(self, *, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _request_json(self, paths: tuple[str, ...], payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()
        last_error: Exception | None = None
        for path in paths:
            url = f"{self.base_url}{path}"
            try:
                response = _http_post(url=url, payload=payload, headers=headers, timeout=self.timeout_seconds)
            except Exception as exc:
                last_error = exc
                continue

            if response.status_code == 404:
                continue
            if response.status_code >= 400:
                raise TokenizerUnavailable(
                    f"Provider tokenizer endpoint error ({response.status_code})"
                )
            try:
                data = response.json()
            except Exception as exc:
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid payload")
            return data

        if last_error is not None:
            raise TokenizerUnavailable("Provider tokenizer endpoint unavailable") from last_error
        raise TokenizerUnavailable("Provider tokenizer endpoint not found")


class AnthropicCountOnlyHTTPAdapter(_HTTPJSONAdapterBase):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["x-api-key"] = self.api_key
        headers["anthropic-version"] = "2023-06-01"
        return headers

    def count_tokens(self, text: str) -> int:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }
        data = self._request_json(("/messages/count_tokens", "/v1/messages/count_tokens"), payload)
        parsed = _extract_int_value(data, ("input_tokens", "total_tokens", "totalTokens", "token_count", "count"))
        if parsed is None or parsed < 0:
            raise TokenizerUnavailable("Anthropic count_tokens response missing token count")
        return int(parsed)


class GoogleCountOnlyHTTPAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    def _candidate_urls(self) -> tuple[str, ...]:
        model_segment = self.model
        if not model_segment.startswith("models/"):
            model_segment = f"models/{model_segment}"

        base = self.base_url
        if "/v1beta" in base or base.endswith("/v1beta") or "/v1/" in base or base.endswith("/v1"):
            return (f"{base}/{model_segment}:countTokens",)
        return (
            f"{base}/v1beta/{model_segment}:countTokens",
            f"{base}/v1/{model_segment}:countTokens",
        )

    def count_tokens(self, text: str) -> int:
        headers = {"Content-Type": "application/json"}
        header_auth = dict(headers)
        if self.api_key:
            header_auth["x-goog-api-key"] = self.api_key
        allow_query_key_fallback = _truthy(
            str(os.getenv("GOOGLE_COUNTTOKENS_ALLOW_QUERY_KEY_FALLBACK", "")).strip().lower()
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            ]
        }

        last_error: Exception | None = None
        for url in self._candidate_urls():
            try:
                response = _http_post(
                    url=url,
                    payload=payload,
                    headers=header_auth,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                continue

            if response.status_code == 404:
                continue

            # Optional compatibility mode for deployments that require key in query params.
            # Disabled by default to avoid leaking secrets in URL logs.
            if (
                response.status_code in {401, 403}
                and allow_query_key_fallback
                and self.api_key
            ):
                separator = "&" if "?" in url else "?"
                query_url = f"{url}{separator}key={quote_plus(self.api_key)}"
                try:
                    response = _http_post(
                        url=query_url,
                        payload=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                except Exception as exc:
                    last_error = exc
                    continue
                if response.status_code == 404:
                    continue

            if response.status_code >= 400:
                raise TokenizerUnavailable(
                    f"Provider tokenizer endpoint error ({response.status_code})"
                )
            try:
                data = response.json()
            except Exception as exc:
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid payload")

            parsed = _extract_int_value(
                data,
                ("totalTokens", "total_tokens", "token_count", "count", "promptTokenCount"),
            )
            if parsed is None or parsed < 0:
                raise TokenizerUnavailable("Google countTokens response missing token count")
            return int(parsed)

        if last_error is not None:
            raise TokenizerUnavailable("Provider tokenizer endpoint unavailable") from last_error
        raise TokenizerUnavailable("Provider tokenizer endpoint not found")


class CohereTokenizerHTTPAdapter(_HTTPJSONAdapterBase):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds)
        self.model = model
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def encode(self, text: str) -> list[int]:
        payload: dict[str, Any] = {"text": text}
        if self.model:
            payload["model"] = self.model
        data = self._request_json(("/v1/tokenize", "/tokenize"), payload)
        raw_tokens = data.get("tokens")
        if not isinstance(raw_tokens, list):
            raw_tokens = data.get("token_ids")
        if not isinstance(raw_tokens, list):
            raise TokenizerUnavailable("Cohere tokenize response missing token ids")
        converted: list[int] = []
        for token_id in raw_tokens:
            parsed = _coerce_int(token_id)
            if parsed is None:
                raise TokenizerUnavailable("Cohere tokenize response returned invalid token ids")
            converted.append(int(parsed))
        return converted

    def decode(self, token_ids: list[int]) -> str:
        payload: dict[str, Any] = {"tokens": [int(token_id) for token_id in token_ids]}
        if self.model:
            payload["model"] = self.model
        data = self._request_json(("/v1/detokenize", "/detokenize"), payload)
        text = data.get("text")
        if not isinstance(text, str):
            text = data.get("content")
        if not isinstance(text, str):
            raise TokenizerUnavailable("Cohere detokenize response missing text payload")
        return text


class BedrockCountOnlyHTTPAdapter(_HTTPJSONAdapterBase):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        region: str | None,
        aws_credentials: BedrockSigV4Credentials | None,
        timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__(base_url=base_url, timeout_seconds=timeout_seconds)
        self.model = model.strip()
        self.api_key = api_key
        self.region = str(region or "").strip() or None
        self.aws_credentials = aws_credentials

    def _candidate_paths(self) -> tuple[str, ...]:
        model_id = quote(self.model, safe="")
        return (
            f"/model/{model_id}/count-tokens",
            f"/bedrock-runtime/model/{model_id}/count-tokens",
        )

    def _should_use_local_proxy_auth(self) -> bool:
        host = _url_hostname(self.base_url)
        return _is_local_or_private_host(host)

    def _request_json(self, paths: tuple[str, ...], payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for path in paths:
            url = f"{self.base_url}{path}"
            headers = {"Content-Type": "application/json"}
            try:
                if self._should_use_local_proxy_auth():
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                else:
                    credentials = self.aws_credentials or _resolve_bedrock_sigv4_credentials(parser=None)
                    if credentials is None:
                        raise TokenizerUnavailable("Bedrock SigV4 credentials are not configured")
                    self.aws_credentials = credentials
                    signed_headers = _build_bedrock_sigv4_headers(
                        url=url,
                        payload=payload,
                        region=self.region,
                        credentials=credentials,
                    )
                    headers.update(signed_headers)
                response = _http_post(url=url, payload=payload, headers=headers, timeout=self.timeout_seconds)
            except Exception as exc:
                last_error = exc
                continue

            if response.status_code == 404:
                continue
            if response.status_code >= 400:
                raise TokenizerUnavailable(
                    f"Provider tokenizer endpoint error ({response.status_code})"
                )
            try:
                data = response.json()
            except Exception as exc:
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise TokenizerUnavailable("Provider tokenizer endpoint returned invalid payload")
            return data

        if last_error is not None:
            raise TokenizerUnavailable("Provider tokenizer endpoint unavailable") from last_error
        raise TokenizerUnavailable("Provider tokenizer endpoint not found")

    def count_tokens(self, text: str) -> int:
        payload = {
            "input": {
                "converse": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": text}],
                        }
                    ]
                }
            }
        }
        data = self._request_json(self._candidate_paths(), payload)
        parsed = _extract_int_value(
            data,
            ("inputTokens", "input_tokens", "totalTokens", "total_tokens", "token_count", "count"),
        )
        if parsed is None or parsed < 0:
            raise TokenizerUnavailable("Bedrock count-tokens response missing token count")
        return int(parsed)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def strict_token_counting_enabled(default: bool = False) -> bool:
    env_val = os.getenv("STRICT_TOKEN_COUNTING")
    if env_val is None:
        return bool(default)
    return _truthy(env_val)


def _http_post(*, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise TokenizerUnavailable("Provider tokenizer HTTP client unavailable") from exc
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


def normalize_provider_for_tokenizer(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if not raw:
        return ""
    custom_number = custom_openai_provider_number(raw)
    if custom_number is not None:
        return custom_openai_section_name(custom_number)
    return TOKENIZER_PROVIDER_ALIASES.get(raw, raw)


def normalize_native_tokenizer_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in NATIVE_TOKENIZER_STRIP_SUFFIXES:
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _url_hostname(url: str) -> str:
    try:
        parsed = urlparse(str(url or "").strip())
        return str(parsed.hostname or "").strip().lower()
    except Exception:
        return ""


def _is_openai_host_url(url: str) -> bool:
    host = _url_hostname(url)
    if not host:
        return False
    return host == "api.openai.com" or host.endswith(".openai.com") or host.endswith(".openai.azure.com")


def _is_local_or_private_host(host: str) -> bool:
    lowered = str(host or "").strip().lower()
    if not lowered:
        return False
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        return True
    if lowered.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
    except Exception:
        return False
    return bool(addr.is_loopback or addr.is_private)


def _is_bedrock_runtime_base_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    path = str(parsed.path or "").strip()
    if path and path != "/":
        return False
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    if _is_local_or_private_host(host):
        return True
    if "bedrock-runtime" not in host:
        return False
    return host.endswith(".amazonaws.com") or host.endswith(".amazonaws.com.cn")


def _sigv4_hash_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sigv4_hmac(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _canonical_query_string(raw_query: str) -> str:
    if not raw_query:
        return ""
    pairs = parse_qsl(raw_query, keep_blank_values=True)
    encoded: list[tuple[str, str]] = []
    for key, value in pairs:
        encoded.append(
            (
                quote(str(key), safe="-_.~"),
                quote(str(value), safe="-_.~"),
            )
        )
    encoded.sort()
    return "&".join(f"{key}={value}" for key, value in encoded)


def _canonical_header_value(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _build_bedrock_sigv4_headers(
    *,
    url: str,
    payload: dict[str, Any],
    region: str | None,
    credentials: BedrockSigV4Credentials,
) -> dict[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.netloc or "").strip()
    if not host:
        raise TokenizerUnavailable("Bedrock SigV4 signing requires an absolute URL")
    canonical_uri = quote(str(parsed.path or "/"), safe="/-_.~%")
    canonical_query = _canonical_query_string(str(parsed.query or ""))
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    payload_hash = _sigv4_hash_hex(payload_json)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp[:8]
    signing_region = (
        str(region or "").strip()
        or _infer_bedrock_region_from_base_url(url)
        or "us-west-2"
    )
    canonical_headers_map = {
        "content-type": "application/json",
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": timestamp,
    }
    if credentials.session_token:
        canonical_headers_map["x-amz-security-token"] = credentials.session_token
    signed_header_keys = sorted(canonical_headers_map.keys())
    canonical_headers = "".join(
        f"{key}:{_canonical_header_value(canonical_headers_map[key])}\n"
        for key in signed_header_keys
    )
    signed_headers = ";".join(signed_header_keys)
    canonical_request = "\n".join(
        (
            "POST",
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        )
    )
    credential_scope = f"{date_stamp}/{signing_region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            timestamp,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    signing_key = _sigv4_hmac(("AWS4" + credentials.secret_access_key).encode("utf-8"), date_stamp)
    signing_key = _sigv4_hmac(signing_key, signing_region)
    signing_key = _sigv4_hmac(signing_key, "bedrock")
    signing_key = _sigv4_hmac(signing_key, "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={credentials.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": timestamp,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": authorization,
    }
    if credentials.session_token:
        headers["X-Amz-Security-Token"] = credentials.session_token
    return headers


def _runtime_probe_exact_resolution(
    resolution: TokenizerResolution,
    *,
    timeout_seconds: float = 2.0,
    probe_text: str = "",
) -> TokenizerResolution:
    if not resolution.available or resolution.encoding is None:
        return resolution
    if str(resolution.count_accuracy or "").strip().lower() != "exact":
        return resolution
    if str(resolution.kind or "").strip() not in {"provider-native", "provider-native-count"}:
        return resolution

    encoding = resolution.encoding
    original_timeout: float | None = None
    try:
        current_timeout = getattr(encoding, "timeout_seconds", None)
        if isinstance(current_timeout, (int, float)) and math.isfinite(float(current_timeout)):
            original_timeout = float(current_timeout)
            bounded_timeout = max(0.5, float(timeout_seconds))
            if original_timeout > bounded_timeout:
                setattr(encoding, "timeout_seconds", bounded_timeout)
    except Exception:
        original_timeout = None

    try:
        count_fn = getattr(encoding, "count_tokens", None)
        if callable(count_fn):
            counted = count_fn(str(probe_text or ""))
            parsed = _coerce_int(counted)
            if parsed is None or parsed < 0:
                raise TokenizerUnavailable("Provider tokenizer runtime probe returned invalid count payload")
        else:
            encode_fn = getattr(encoding, "encode", None)
            if not callable(encode_fn):
                raise TokenizerUnavailable("Provider tokenizer runtime probe missing encode/count method")
            try:
                encoded = encode_fn(str(probe_text or ""), disallowed_special=())
            except TypeError:
                encoded = encode_fn(str(probe_text or ""))
            if not isinstance(encoded, list):
                encoded = list(encoded)
            for token_id in encoded[:16]:
                if _coerce_int(token_id) is None:
                    raise TokenizerUnavailable("Provider tokenizer runtime probe returned invalid token ids")
    except Exception as exc:
        return _unavailable_resolution(
            strict_mode_effective=bool(resolution.strict_mode_effective),
            error=f"Provider tokenizer runtime probe failed: {exc}",
        )
    finally:
        if original_timeout is not None:
            try:
                setattr(encoding, "timeout_seconds", original_timeout)
            except Exception:
                pass

    return resolution


@lru_cache(maxsize=128)
def _resolve_tiktoken_encoding_cached(model: str) -> Any:
    try:
        import tiktoken  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TokenizerUnavailable("Tokenizer library unavailable") from exc
    try:
        return tiktoken.encoding_for_model(model)
    except Exception as exc:
        raise TokenizerUnavailable("Tokenizer not available for provider/model") from exc


def resolve_tiktoken_encoding(model: str) -> Any:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        raise TokenizerUnavailable("Model is required")
    return _resolve_tiktoken_encoding_cached(normalized_model)


def _resolve_tiktoken_tokenizer_name(model: str) -> str | None:
    try:
        encoding = resolve_tiktoken_encoding(model)
    except TokenizerUnavailable:
        return None
    name = getattr(encoding, "name", "unknown")
    return str(name) if name else "unknown"


def _load_config(config_parser: Any | None, config_loader: Callable[[], Any] | None) -> Any:
    if config_parser is not None:
        return config_parser
    loader = config_loader or load_comprehensive_config
    return loader()


def _safe_config_string(parser: Any, section: str, option: str) -> str | None:
    try:
        if not parser.has_section(section):
            return None
        if not parser.has_option(section, option):
            return None
        value = str(parser.get(section, option, fallback="") or "").strip()
        if not value or value.startswith("<"):
            return None
        return value
    except Exception:
        return None


def _resolve_commercial_api_key(parser: Any, *, section: str, field: str, env_name: str) -> str | None:
    env_val = str(os.getenv(env_name, "") or "").strip()
    if env_val and not env_val.startswith("<"):
        return env_val
    return _safe_config_string(parser, section, field)


def _resolve_commercial_base_url(
    parser: Any,
    *,
    section: str,
    fields: tuple[str, ...],
    env_name: str,
    default_url: str,
) -> str:
    env_val = str(os.getenv(env_name, "") or "").strip()
    if env_val:
        return env_val.rstrip("/")

    for field in fields:
        value = _safe_config_string(parser, section, field)
        if value:
            return value.rstrip("/")

    return str(default_url or "").strip().rstrip("/")


def _resolve_bedrock_api_key(parser: Any) -> str | None:
    for env_name in ("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        env_val = str(os.getenv(env_name, "") or "").strip()
        if env_val and not env_val.startswith("<"):
            return env_val
    return _safe_config_string(parser, "API", "bedrock_api_key")


def _resolve_bedrock_region(parser: Any) -> str:
    return (
        str(os.getenv("BEDROCK_REGION", "") or "").strip()
        or str(_safe_config_string(parser, "API", "bedrock_region") or "").strip()
        or "us-west-2"
    )


def _normalize_bedrock_runtime_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    lowered = normalized.lower()
    for suffix in ("/openai/v1", "/openai", "/v1/chat/completions", "/chat/completions"):
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.rstrip("/")


def _resolve_bedrock_runtime_base_url(parser: Any) -> str:
    candidates = (
        os.getenv("BEDROCK_RUNTIME_ENDPOINT"),
        _safe_config_string(parser, "API", "bedrock_runtime_endpoint"),
        os.getenv("BEDROCK_API_BASE_URL"),
        os.getenv("BEDROCK_OPENAI_BASE_URL"),
        _safe_config_string(parser, "API", "bedrock_api_base_url"),
        _safe_config_string(parser, "API", "bedrock_openai_base_url"),
    )
    for candidate in candidates:
        if candidate:
            normalized = _normalize_bedrock_runtime_base_url(candidate)
            if normalized:
                return normalized

    region = _resolve_bedrock_region(parser)
    return f"https://bedrock-runtime.{region}.amazonaws.com"


def _infer_bedrock_region_from_base_url(base_url: str) -> str | None:
    host = _url_hostname(base_url)
    if not host:
        return None
    marker = "bedrock-runtime."
    if marker not in host:
        return None
    suffixes = (".amazonaws.com", ".amazonaws.com.cn")
    for suffix in suffixes:
        if host.endswith(suffix):
            prefix = host.split(marker, 1)[1]
            region = prefix[: -len(suffix)] if suffix else prefix
            region = str(region or "").strip().strip(".")
            if region:
                return region
    return None


def _resolve_bedrock_sigv4_credentials(parser: Any) -> BedrockSigV4Credentials | None:
    def _first_nonempty(*values: str | None) -> str | None:
        for value in values:
            normalized = str(value or "").strip()
            if normalized and not normalized.startswith("<"):
                return normalized
        return None

    access_key_id = _first_nonempty(
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_ACCESS_KEY"),
        _safe_config_string(parser, "API", "bedrock_aws_access_key_id"),
        _safe_config_string(parser, "API", "aws_access_key_id"),
    )
    secret_access_key = _first_nonempty(
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_SECRET_KEY"),
        _safe_config_string(parser, "API", "bedrock_aws_secret_access_key"),
        _safe_config_string(parser, "API", "aws_secret_access_key"),
    )
    session_token = _first_nonempty(
        os.getenv("AWS_SESSION_TOKEN"),
        os.getenv("AWS_SECURITY_TOKEN"),
        _safe_config_string(parser, "API", "bedrock_aws_session_token"),
        _safe_config_string(parser, "API", "aws_session_token"),
    )
    if access_key_id and secret_access_key:
        return BedrockSigV4Credentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
        )

    try:
        from botocore.session import Session as _BotocoreSession  # type: ignore
    except Exception:
        return None

    try:
        creds = _BotocoreSession().get_credentials()
        if creds is None:
            return None
        frozen = creds.get_frozen_credentials()
        access = str(getattr(frozen, "access_key", "") or "").strip()
        secret = str(getattr(frozen, "secret_key", "") or "").strip()
        if not access or not secret:
            return None
        token = str(getattr(frozen, "token", "") or "").strip() or None
        return BedrockSigV4Credentials(
            access_key_id=access,
            secret_access_key=secret,
            session_token=token,
        )
    except Exception:
        return None


def _bedrock_model_supports_exact_count(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith("anthropic.claude-"):
        return True
    if normalized.startswith("arn:") and "anthropic.claude-" in normalized:
        return True
    return False


def _unavailable_resolution(
    *,
    strict_mode_effective: bool,
    error: str,
) -> TokenizerResolution:
    return TokenizerResolution(
        available=False,
        tokenizer=None,
        kind=None,
        source=None,
        detokenize_available=False,
        count_accuracy="unavailable",
        strict_mode_effective=bool(strict_mode_effective),
        encoding=None,
        error=error,
    )


def resolve_provider_native_tokenizer(
    provider: str,
    model: str,
    *,
    strict_mode_effective: bool | None = None,
    config_parser: Any | None = None,
    config_loader: Callable[[], Any] | None = None,
    adapter_cls: type[ProviderNativeTokenizerHTTPAdapter] = ProviderNativeTokenizerHTTPAdapter,
    runtime_probe_exact: bool = False,
    runtime_probe_timeout_seconds: float = 2.0,
    runtime_probe_text: str = "",
) -> TokenizerResolution:
    strict_flag = strict_token_counting_enabled() if strict_mode_effective is None else bool(strict_mode_effective)

    provider_key = normalize_provider_for_tokenizer(provider)
    mapping = PROVIDER_NATIVE_TOKENIZER_CONFIG.get(provider_key)
    if not mapping:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider-native tokenizer is not configured for provider",
        )

    section = mapping.get("section") or ""
    endpoint_field = mapping.get("endpoint_field") or ""
    api_key_field = mapping.get("api_key_field") or ""
    label = mapping.get("label") or provider_key

    try:
        parser = _load_config(config_parser, config_loader)
    except Exception as exc:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error=f"Provider-native tokenizer config unavailable: {exc}",
        )

    if not parser.has_section(section) or not parser.has_option(section, endpoint_field):
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider-native tokenizer endpoint is not configured",
        )

    endpoint = parser.get(section, endpoint_field, fallback="").strip()
    if not endpoint or endpoint.startswith("<"):
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider-native tokenizer endpoint is not configured",
        )

    api_key = None
    if api_key_field and parser.has_option(section, api_key_field):
        raw_api_key = parser.get(section, api_key_field, fallback="").strip()
        if raw_api_key and not raw_api_key.startswith("<"):
            api_key = raw_api_key

    base_url = normalize_native_tokenizer_base_url(endpoint)
    if not base_url:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider-native tokenizer endpoint is invalid",
        )

    if custom_openai_provider_number(provider_key) is not None and _is_openai_host_url(base_url):
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error=f"{provider_key} endpoint points to OpenAI-hosted API; provider-native tokenizer endpoint is unavailable",
        )

    adapter = adapter_cls(
        base_url=base_url,
        model=str(model or "").strip() or None,
        api_key=api_key,
    )

    resolution = TokenizerResolution(
        available=True,
        tokenizer=f"{label}:remote",
        kind="provider-native",
        source=f"{label}.http.tokenize",
        detokenize_available=True,
        count_accuracy="exact",
        strict_mode_effective=bool(strict_flag),
        encoding=adapter,
        error=None,
    )
    if runtime_probe_exact:
        return _runtime_probe_exact_resolution(
            resolution,
            timeout_seconds=runtime_probe_timeout_seconds,
            probe_text=runtime_probe_text,
        )
    return resolution


def resolve_commercial_exact_tokenizer(
    provider: str,
    model: str,
    *,
    strict_mode_effective: bool | None = None,
    config_parser: Any | None = None,
    config_loader: Callable[[], Any] | None = None,
    runtime_probe_exact: bool = False,
    runtime_probe_timeout_seconds: float = 2.0,
    runtime_probe_text: str = "",
) -> TokenizerResolution:
    strict_flag = strict_token_counting_enabled() if strict_mode_effective is None else bool(strict_mode_effective)

    provider_key = normalize_provider_for_tokenizer(provider)
    mapping = COMMERCIAL_EXACT_TOKENIZER_CONFIG.get(provider_key)
    if not mapping:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Commercial exact tokenizer is not configured for provider",
        )

    section = str(mapping.get("section") or "API")
    api_key_field = str(mapping.get("api_key_field") or "")
    api_key_env = str(mapping.get("api_key_env") or "")
    base_url_fields = tuple(mapping.get("base_url_fields") or ())
    base_url_env = str(mapping.get("base_url_env") or "")
    base_url_default = str(mapping.get("base_url_default") or "")
    label = str(mapping.get("label") or provider_key)
    mode = str(mapping.get("mode") or "count-only").strip().lower()

    try:
        parser = _load_config(config_parser, config_loader)
    except Exception as exc:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error=f"Commercial tokenizer config unavailable: {exc}",
        )

    normalized_model = str(model or "").strip()
    if provider_key == "bedrock":
        if not _bedrock_model_supports_exact_count(normalized_model):
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Bedrock exact tokenizer unavailable for provider/model",
            )
        base_url = _resolve_bedrock_runtime_base_url(parser)
        if not base_url:
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Provider tokenizer endpoint is not configured",
            )
        if _is_openai_host_url(base_url):
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Bedrock runtime endpoint points to OpenAI-hosted API",
            )
        if not _is_bedrock_runtime_base_url(base_url):
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Bedrock runtime endpoint must target a bedrock-runtime host and must not include a path",
            )
        host = _url_hostname(base_url)
        region = _infer_bedrock_region_from_base_url(base_url) or _resolve_bedrock_region(parser)
        aws_credentials = _resolve_bedrock_sigv4_credentials(parser)
        api_key = _resolve_bedrock_api_key(parser)
        if not _is_local_or_private_host(host) and aws_credentials is None:
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Bedrock SigV4 credentials are not configured",
            )
        adapter = BedrockCountOnlyHTTPAdapter(
            base_url=base_url,
            model=normalized_model,
            api_key=api_key,
            region=region,
            aws_credentials=aws_credentials,
        )
        resolution = TokenizerResolution(
            available=True,
            tokenizer=f"{label}:remote-count",
            kind="provider-native-count",
            source=f"{label}.http.count_tokens",
            detokenize_available=False,
            count_accuracy="exact",
            strict_mode_effective=bool(strict_flag),
            encoding=adapter,
            error=None,
        )
        if runtime_probe_exact:
            return _runtime_probe_exact_resolution(
                resolution,
                timeout_seconds=runtime_probe_timeout_seconds,
                probe_text=runtime_probe_text,
            )
        return resolution

    api_key = _resolve_commercial_api_key(
        parser,
        section=section,
        field=api_key_field,
        env_name=api_key_env,
    )
    if not api_key:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider tokenizer API key is not configured",
        )

    base_url = _resolve_commercial_base_url(
        parser,
        section=section,
        fields=base_url_fields,
        env_name=base_url_env,
        default_url=base_url_default,
    )
    if not base_url:
        return _unavailable_resolution(
            strict_mode_effective=strict_flag,
            error="Provider tokenizer endpoint is not configured",
        )

    if mode == "count-only":
        if provider_key == "anthropic":
            adapter: Any = AnthropicCountOnlyHTTPAdapter(
                base_url=base_url,
                model=normalized_model,
                api_key=api_key,
            )
        elif provider_key == "google":
            adapter = GoogleCountOnlyHTTPAdapter(
                base_url=base_url,
                model=normalized_model,
                api_key=api_key,
            )
        else:
            return _unavailable_resolution(
                strict_mode_effective=strict_flag,
                error="Commercial tokenizer mode is not supported",
            )
        resolution = TokenizerResolution(
            available=True,
            tokenizer=f"{label}:remote-count",
            kind="provider-native-count",
            source=f"{label}.http.count_tokens",
            detokenize_available=False,
            count_accuracy="exact",
            strict_mode_effective=bool(strict_flag),
            encoding=adapter,
            error=None,
        )
        if runtime_probe_exact:
            return _runtime_probe_exact_resolution(
                resolution,
                timeout_seconds=runtime_probe_timeout_seconds,
                probe_text=runtime_probe_text,
            )
        return resolution

    if mode == "tokenize" and provider_key == "cohere":
        adapter = CohereTokenizerHTTPAdapter(
            base_url=base_url,
            model=normalized_model,
            api_key=api_key,
        )
        resolution = TokenizerResolution(
            available=True,
            tokenizer=f"{label}:remote",
            kind="provider-native",
            source=f"{label}.http.tokenize",
            detokenize_available=True,
            count_accuracy="exact",
            strict_mode_effective=bool(strict_flag),
            encoding=adapter,
            error=None,
        )
        if runtime_probe_exact:
            return _runtime_probe_exact_resolution(
                resolution,
                timeout_seconds=runtime_probe_timeout_seconds,
                probe_text=runtime_probe_text,
            )
        return resolution

    return _unavailable_resolution(
        strict_mode_effective=strict_flag,
        error="Commercial tokenizer mode is not supported",
    )


def _openrouter_canonical_model(model: str) -> str | None:
    normalized = str(model or "").strip()
    if not normalized:
        return None
    if "/" not in normalized:
        return normalized
    provider_hint, _, model_id = normalized.partition("/")
    if provider_hint.strip().lower() != "openai":
        return None
    canonical = model_id.strip()
    return canonical or None


def _groq_canonical_model(model: str) -> str | None:
    normalized = str(model or "").strip()
    if not normalized:
        return None
    if "/" not in normalized:
        return None
    provider_hint, _, model_id = normalized.partition("/")
    if provider_hint.strip().lower() != "openai":
        return None
    canonical = model_id.strip()
    return canonical or None


def _exact_tiktoken_model_for_provider(provider_key: str, model: str) -> str | None:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return None
    if provider_key == "openai":
        return normalized_model
    if provider_key == "openrouter":
        return _openrouter_canonical_model(normalized_model)
    if provider_key == "groq":
        return _groq_canonical_model(normalized_model)
    return None


def _build_tiktoken_resolution(
    *,
    strict_mode_effective: bool,
    encoding: Any,
    count_accuracy: str,
    source: str,
) -> TokenizerResolution:
    tokenizer_name = getattr(encoding, "name", "unknown")
    return TokenizerResolution(
        available=True,
        tokenizer=f"tiktoken:{tokenizer_name}",
        kind="tiktoken",
        source=source,
        detokenize_available=True,
        count_accuracy=count_accuracy,
        strict_mode_effective=bool(strict_mode_effective),
        encoding=encoding,
        error=None,
    )


def _get_active_mlx_tokenizer(model: str) -> tuple[Any, str] | None:
    try:
        from tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider import get_active_mlx_tokenizer

        return get_active_mlx_tokenizer(model)
    except Exception:
        return None


def _trusted_mlx_model_root() -> Path:
    model_root = os.getenv("MLX_MODEL_DIR", "").strip()
    root = Path(model_root).expanduser() if model_root else Path.cwd()
    return root.resolve(strict=False)


def _normalize_mlx_model_path(model: str) -> Path | None:
    raw = str(model or "").strip()
    if not raw:
        return None

    # Disallow absolute paths entirely; all MLX models must reside under the trusted root.
    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        return None

    # Normalize the path string to eliminate redundant separators.
    normalized_str = os.path.normpath(raw)
    # Reject no-op / current-directory paths.
    if normalized_str in ("", "."):
        return None
    # Prevent leading separators that could escape the trusted root when joined.
    if normalized_str.startswith(os.path.sep) or normalized_str.startswith("/"):
        return None
    # Construct a Path object from the normalized string.
    normalized = Path(normalized_str)
    # Reject any path components that indicate traversal or current directory.
    if any(part in ("", ".", "..") for part in normalized.parts):
        return None

    return normalized


def _mlx_candidate_paths(model: str) -> list[Path]:
    trusted_root = _trusted_mlx_model_root()
    normalized = _normalize_mlx_model_path(model)
    if normalized is None:
        return []

    # At this point, normalized is guaranteed to be a safe, relative path segment.
    candidate = (trusted_root / normalized).resolve(strict=False)

    try:
        # Ensure the resolved candidate remains under the trusted root directory.
        candidate.relative_to(trusted_root)
    except ValueError:
        return []

    return [candidate]


@lru_cache(maxsize=32)
def _load_mlx_artifact_tokenizer(model: str) -> Any:
    candidates = _mlx_candidate_paths(model)
    if not candidates:
        raise TokenizerUnavailable("MLX model path is required")

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        raise TokenizerUnavailable("MLX tokenizer loader unavailable") from exc

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            resolved = candidate.expanduser()
            if not resolved.exists() or not resolved.is_dir():
                continue
            return AutoTokenizer.from_pretrained(
                str(resolved),
                local_files_only=True,
            )  # nosec B615
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise TokenizerUnavailable("MLX tokenizer unavailable for model artifact") from last_error
    raise TokenizerUnavailable("MLX model artifact path not found")


def _resolve_mlx_tokenizer(
    model: str,
    *,
    strict_mode_effective: bool,
) -> TokenizerResolution:
    active = _get_active_mlx_tokenizer(model)
    if active is not None:
        tokenizer_obj, model_id = active
        detok_available = callable(getattr(tokenizer_obj, "decode", None))
        return TokenizerResolution(
            available=True,
            tokenizer=f"mlx:active:{model_id}",
            kind="provider-native",
            source="mlx.registry.active",
            detokenize_available=detok_available,
            count_accuracy="exact",
            strict_mode_effective=bool(strict_mode_effective),
            encoding=tokenizer_obj,
            error=None,
        )

    try:
        tokenizer_obj = _load_mlx_artifact_tokenizer(model)
    except TokenizerUnavailable as exc:
        return _unavailable_resolution(
            strict_mode_effective=strict_mode_effective,
            error=str(exc),
        )

    detok_available = callable(getattr(tokenizer_obj, "decode", None))
    return TokenizerResolution(
        available=True,
        tokenizer="mlx:artifact",
        kind="provider-native",
        source="mlx.artifact.tokenizer",
        detokenize_available=detok_available,
        count_accuracy="exact",
        strict_mode_effective=bool(strict_mode_effective),
        encoding=tokenizer_obj,
        error=None,
    )


def resolve_tokenizer(
    provider: str,
    model: str,
    *,
    strict_mode_effective: bool | None = None,
    config_parser: Any | None = None,
    config_loader: Callable[[], Any] | None = None,
    adapter_cls: type[ProviderNativeTokenizerHTTPAdapter] = ProviderNativeTokenizerHTTPAdapter,
    runtime_probe_exact: bool = False,
    runtime_probe_timeout_seconds: float = 2.0,
    runtime_probe_text: str = "",
) -> TokenizerResolution:
    strict_flag = strict_token_counting_enabled() if strict_mode_effective is None else bool(strict_mode_effective)

    normalized_provider = normalize_provider_for_tokenizer(provider)
    normalized_model = str(model or "").strip()

    if not normalized_provider:
        return _unavailable_resolution(strict_mode_effective=strict_flag, error="Provider is required")
    if not normalized_model:
        return _unavailable_resolution(strict_mode_effective=strict_flag, error="Model is required")

    native_resolution = None
    if normalized_provider in PROVIDER_NATIVE_TOKENIZER_CONFIG:
        native_resolution = resolve_provider_native_tokenizer(
            normalized_provider,
            normalized_model,
            strict_mode_effective=strict_flag,
            config_parser=config_parser,
            config_loader=config_loader,
            adapter_cls=adapter_cls,
            runtime_probe_exact=runtime_probe_exact,
            runtime_probe_timeout_seconds=runtime_probe_timeout_seconds,
            runtime_probe_text=runtime_probe_text,
        )
        if native_resolution.available:
            return native_resolution

    if normalized_provider == "mlx":
        mlx_resolution = _resolve_mlx_tokenizer(normalized_model, strict_mode_effective=strict_flag)
        if mlx_resolution.available:
            return mlx_resolution

    commercial_resolution = None
    if normalized_provider in COMMERCIAL_EXACT_TOKENIZER_CONFIG:
        commercial_resolution = resolve_commercial_exact_tokenizer(
            normalized_provider,
            normalized_model,
            strict_mode_effective=strict_flag,
            config_parser=config_parser,
            config_loader=config_loader,
            runtime_probe_exact=runtime_probe_exact,
            runtime_probe_timeout_seconds=runtime_probe_timeout_seconds,
            runtime_probe_text=runtime_probe_text,
        )
        if commercial_resolution.available:
            return commercial_resolution

    exact_model = _exact_tiktoken_model_for_provider(normalized_provider, normalized_model)
    if exact_model:
        try:
            exact_encoding = resolve_tiktoken_encoding(exact_model)
            source = "tiktoken.encoding_for_model"
            if normalized_provider == "openrouter" and exact_model != normalized_model:
                source = "tiktoken.encoding_for_model(openrouter-canonical)"
            return _build_tiktoken_resolution(
                strict_mode_effective=strict_flag,
                encoding=exact_encoding,
                count_accuracy="exact",
                source=source,
            )
        except TokenizerUnavailable:
            pass

    try:
        fallback_encoding = resolve_tiktoken_encoding(normalized_model)
    except TokenizerUnavailable as exc:
        fallback_error = str(exc)
    else:
        return _build_tiktoken_resolution(
            strict_mode_effective=strict_flag,
            encoding=fallback_encoding,
            count_accuracy="unavailable",
            source="tiktoken.encoding_for_model(best-effort)",
        )

    native_error = native_resolution.error if native_resolution is not None else None
    commercial_error = commercial_resolution.error if commercial_resolution is not None else None
    error_message = native_error or commercial_error or fallback_error
    if not error_message:
        error_message = "Tokenizer not available for provider/model"
    return _unavailable_resolution(strict_mode_effective=strict_flag, error=error_message)


def resolve_tokenizer_metadata(
    provider: str,
    model: str,
    *,
    strict_mode_effective: bool | None = None,
    config_parser: Any | None = None,
    config_loader: Callable[[], Any] | None = None,
    adapter_cls: type[ProviderNativeTokenizerHTTPAdapter] = ProviderNativeTokenizerHTTPAdapter,
    runtime_probe_exact: bool = False,
    runtime_probe_timeout_seconds: float = 2.0,
    runtime_probe_text: str = "",
) -> dict[str, Any]:
    resolution = resolve_tokenizer(
        provider,
        model,
        strict_mode_effective=strict_mode_effective,
        config_parser=config_parser,
        config_loader=config_loader,
        adapter_cls=adapter_cls,
        runtime_probe_exact=runtime_probe_exact,
        runtime_probe_timeout_seconds=runtime_probe_timeout_seconds,
        runtime_probe_text=runtime_probe_text,
    )
    return resolution.as_support_dict()
