from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, quote, urlparse

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body, merge_extra_headers
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream

from .base import ChatProvider, apply_tool_choice

# Patchable client factory (mirrors other adapters)
http_client_factory = _hc_create_client

_BEDROCK_OPENAI_MODEL_MAP = {
    "gpt-oss-20b-1": "openai.gpt-oss-20b-1:0",
}
_OPENAI_STYLE_PREFIXES = ("gpt-", "gpt_", "o1", "o3", "text-", "chatgpt")


@dataclass(frozen=True)
class _BedrockAWSCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        normalized = str(value or "").strip()
        if normalized and not normalized.startswith("<"):
            return normalized
    return None


def _url_hostname(url: str) -> str:
    try:
        parsed = urlparse(str(url or "").strip())
        return str(parsed.hostname or "").strip().lower()
    except Exception:
        return ""


def _is_local_or_private_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    if normalized.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(normalized)
    except Exception:
        return False
    return bool(addr.is_private or addr.is_loopback)


def _is_aws_bedrock_runtime_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return False
    if "bedrock-runtime" not in normalized:
        return False
    return (
        normalized.endswith(".amazonaws.com")
        or normalized.endswith(".amazonaws.com.cn")
        or normalized.endswith(".api.aws")
    )


def _is_aws_bedrock_mantle_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return False
    if "bedrock-mantle" not in normalized:
        return False
    return (
        normalized.endswith(".amazonaws.com")
        or normalized.endswith(".amazonaws.com.cn")
        or normalized.endswith(".api.aws")
    )


def _infer_region_from_bedrock_host(host: str) -> str | None:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return None
    markers = ("bedrock-runtime.", "bedrock-runtime-fips.", "bedrock-mantle.")
    for marker in markers:
        if marker not in normalized:
            continue
        for suffix in (".amazonaws.com", ".amazonaws.com.cn", ".api.aws"):
            if not normalized.endswith(suffix):
                continue
            segment = normalized.split(marker, 1)[1]
            region = segment[: -len(suffix)] if suffix else segment
            region = str(region or "").strip().strip(".")
            if region:
                return region
    return None


def _normalize_bedrock_base_url(raw_url: str, *, runtime_endpoint: bool = False) -> str:
    normalized = str(raw_url or "").strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    host = str(parsed.hostname or "").strip().lower()
    is_aws_runtime_host = _is_aws_bedrock_runtime_host(host)
    is_aws_mantle_host = _is_aws_bedrock_mantle_host(host)

    strip_suffixes = ["/v1/chat/completions", "/chat/completions"]
    if runtime_endpoint or is_aws_runtime_host or is_aws_mantle_host:
        strip_suffixes.extend(("/openai/v1", "/openai"))

    lowered = normalized.lower()
    for suffix in strip_suffixes:
        if lowered.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            lowered = normalized.lower()
            break

    parsed = urlparse(normalized)
    host = str(parsed.hostname or "").strip().lower()
    path = str(parsed.path or "").strip()
    if runtime_endpoint or (_is_aws_bedrock_runtime_host(host) and (not path or path == "/")):
        return normalized.rstrip("/") + "/openai"
    return normalized


def _build_chat_completions_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    lowered = normalized.lower()
    if lowered.endswith("/v1/chat/completions") or lowered.endswith("/chat/completions"):
        return normalized
    if lowered.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _canonical_query(raw_query: str) -> str:
    if not raw_query:
        return ""
    pairs = parse_qsl(raw_query, keep_blank_values=True)
    encoded = [
        (
            quote(str(key), safe="-_.~"),
            quote(str(value), safe="-_.~"),
        )
        for key, value in pairs
    ]
    encoded.sort()
    return "&".join(f"{key}={value}" for key, value in encoded)


def _canonical_header_value(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _hmac_sha256(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _build_sigv4_headers(
    *,
    url: str,
    payload: dict[str, Any],
    region: str,
    credentials: _BedrockAWSCredentials,
) -> dict[str, str]:
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.netloc or "").strip()
    if not host:
        raise ChatConfigurationError(provider="bedrock", message="Bedrock URL must be absolute for SigV4 signing")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp[:8]
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

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
            quote(str(parsed.path or "/"), safe="/-_.~%"),
            _canonical_query(str(parsed.query or "")),
            canonical_headers,
            signed_headers,
            payload_hash,
        )
    )
    scope = f"{date_stamp}/{region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            timestamp,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    signing_key = _hmac_sha256(("AWS4" + credentials.secret_access_key).encode("utf-8"), date_stamp)
    signing_key = _hmac_sha256(signing_key, region)
    signing_key = _hmac_sha256(signing_key, "bedrock")
    signing_key = _hmac_sha256(signing_key, "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": timestamp,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": (
            "AWS4-HMAC-SHA256 "
            f"Credential={credentials.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, "
            f"Signature={signature}"
        ),
    }
    if credentials.session_token:
        headers["X-Amz-Security-Token"] = credentials.session_token
    return headers


class BedrockAdapter(ChatProvider):
    """AWS Bedrock chat adapter.

    Targets the Bedrock Runtime OpenAI compatibility surface:
      https://bedrock-runtime.<region>.amazonaws.com/openai/v1/chat/completions

    AWS Bedrock runtime hosts are signed with SigV4. Local/private/custom proxy
    hosts keep Bearer-token compatibility to avoid breaking existing deployments.
    """

    name = "bedrock"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": None,
        }

    def _use_native_http(self) -> bool:
        # Always use native HTTP unless explicitly disabled
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_BEDROCK") or "").lower()
        return v not in {"0", "false", "no", "off"}

    def _base_url(self, request: dict[str, Any] | None = None) -> str:
        # Allow explicit base override; otherwise derive from runtime endpoint or region
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return _normalize_bedrock_base_url(override, runtime_endpoint=False)
        app_config = (request or {}).get("app_config") or {}
        provider_config = (
            app_config.get("bedrock_api") or {}
            if (request or {}).get("credentials_resolved") is True and isinstance(app_config, dict)
            else {}
        )
        if not isinstance(provider_config, dict):
            provider_config = {}
        configured_runtime = provider_config.get("runtime_endpoint")
        if isinstance(configured_runtime, str) and configured_runtime.strip():
            return _normalize_bedrock_base_url(configured_runtime, runtime_endpoint=True)
        configured_base = (
            provider_config.get("api_base_url")
            or provider_config.get("base_url")
            or provider_config.get("api_url")
            or provider_config.get("endpoint")
        )
        if isinstance(configured_base, str) and configured_base.strip():
            return _normalize_bedrock_base_url(configured_base, runtime_endpoint=False)
        runtime = os.getenv("BEDROCK_RUNTIME_ENDPOINT")
        if runtime:
            # Expect a hostname like https://bedrock-runtime.us-west-2.amazonaws.com
            return _normalize_bedrock_base_url(runtime, runtime_endpoint=True)
        base = (
            os.getenv("BEDROCK_API_BASE_URL")
            or os.getenv("BEDROCK_OPENAI_BASE_URL")
        )
        if base:
            return _normalize_bedrock_base_url(base, runtime_endpoint=False)

        region = os.getenv("BEDROCK_REGION") or "us-west-2"
        return _normalize_bedrock_base_url(f"https://bedrock-runtime.{region}.amazonaws.com", runtime_endpoint=True)

    def _resolve_sigv4_credentials(self, request: dict[str, Any]) -> _BedrockAWSCredentials | None:
        access_key_id = _first_nonempty(
            request.get("aws_access_key_id"),
            request.get("bedrock_aws_access_key_id"),
            os.getenv("AWS_ACCESS_KEY_ID"),
            os.getenv("AWS_ACCESS_KEY"),
        )
        secret_access_key = _first_nonempty(
            request.get("aws_secret_access_key"),
            request.get("bedrock_aws_secret_access_key"),
            os.getenv("AWS_SECRET_ACCESS_KEY"),
            os.getenv("AWS_SECRET_KEY"),
        )
        session_token = _first_nonempty(
            request.get("aws_session_token"),
            request.get("bedrock_aws_session_token"),
            os.getenv("AWS_SESSION_TOKEN"),
            os.getenv("AWS_SECURITY_TOKEN"),
        )
        if access_key_id and secret_access_key:
            return _BedrockAWSCredentials(
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
            return _BedrockAWSCredentials(
                access_key_id=access,
                secret_access_key=secret,
                session_token=token,
            )
        except Exception:
            return None

    def _resolve_region(self, request: dict[str, Any], base_url: str) -> str:
        host = _url_hostname(base_url)
        return (
            _first_nonempty(
                request.get("bedrock_region"),
                request.get("region"),
                _infer_region_from_bedrock_host(host),
                os.getenv("BEDROCK_REGION"),
            )
            or "us-west-2"
        )

    def _build_headers(self, *, request: dict[str, Any], url: str, payload: dict[str, Any]) -> dict[str, str]:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved

        headers = merge_extra_headers({"Content-Type": "application/json"}, request)
        host = _url_hostname(url)
        is_runtime_host = _is_aws_bedrock_runtime_host(host)
        is_mantle_host = _is_aws_bedrock_mantle_host(host)
        credentials_resolved = request.get("credentials_resolved") is True
        if credentials_resolved:
            api_key = _first_nonempty(request.get("api_key"))
            if not provider_auth_is_resolved(
                self.name,
                api_key=api_key,
                app_config=request.get("app_config"),
                credentials_resolved=True,
            ):
                raise ChatConfigurationError(
                    provider=self.name,
                    message="Bedrock runtime authentication was not resolved for this call.",
                )
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                return headers
            if not (is_runtime_host or is_mantle_host):
                raise ChatConfigurationError(
                    provider=self.name,
                    message="AWS default-chain authentication requires an AWS Bedrock endpoint.",
                )
            credentials = self._resolve_sigv4_credentials(request)
            if credentials is None:
                raise ChatConfigurationError(
                    provider=self.name,
                    message="AWS default-chain credentials are unavailable for Bedrock.",
                )
            headers.update(
                _build_sigv4_headers(
                    url=url,
                    payload=payload,
                    region=self._resolve_region(request, url),
                    credentials=credentials,
                )
            )
            return headers

        key = _first_nonempty(
            request.get("api_key"),
            os.getenv("BEDROCK_API_KEY"),
            os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
        )
        # Keep existing compatibility for local/private/custom proxy endpoints.
        if _is_local_or_private_host(host) or not (is_runtime_host or is_mantle_host):
            if key:
                headers["Authorization"] = f"Bearer {key}"
            return headers

        if is_mantle_host:
            # Bedrock Mantle is API-key-first for OpenAI-compatible flows; keep
            # SigV4 fallback for direct HTTP integrations when AWS creds exist.
            if key:
                headers["Authorization"] = f"Bearer {key}"
                return headers
            credentials = self._resolve_sigv4_credentials(request)
            if credentials is not None:
                headers.update(
                    _build_sigv4_headers(
                        url=url,
                        payload=payload,
                        region=self._resolve_region(request, url),
                        credentials=credentials,
                    )
                )
                return headers
            raise ChatConfigurationError(
                provider=self.name,
                message=(
                    "Bedrock Mantle endpoint authentication is required "
                    "(provide api_key/BEDROCK_API_KEY, or configure AWS SigV4 credentials via "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or an AWS credential provider)."
                ),
            )

        credentials = self._resolve_sigv4_credentials(request)
        if credentials is not None:
            headers.update(
                _build_sigv4_headers(
                    url=url,
                    payload=payload,
                    region=self._resolve_region(request, url),
                    credentials=credentials,
                )
            )
            return headers

        # AWS runtime OpenAI compatibility also supports API key auth for direct HTTP calls.
        if key:
            headers["Authorization"] = f"Bearer {key}"
            return headers

        if credentials is None:
            raise ChatConfigurationError(
                provider=self.name,
                message=(
                    "Bedrock runtime authentication is required for AWS Bedrock Runtime endpoints "
                    "(provide api_key/BEDROCK_API_KEY for Bearer auth, or configure AWS SigV4 credentials via "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or an AWS credential provider)."
                ),
            )
        return headers

    def _normalize_model(self, model: str | None) -> str | None:
        if model is None:
            return None
        normalized = model.strip()
        if not normalized:
            return normalized
        mapped = _BEDROCK_OPENAI_MODEL_MAP.get(normalized)
        if mapped:
            return mapped
        lowered = normalized.lower()
        if lowered.startswith(_OPENAI_STYLE_PREFIXES):
            raise ChatConfigurationError(
                provider=self.name,
                message=(
                    "Invalid Bedrock model ID. Use a Bedrock model identifier like "
                    "'anthropic.claude-3-5-sonnet-20241022-v2:0' or "
                    "'meta.llama3-8b-instruct', not an OpenAI-style ID."
                ),
            )
        if "." not in normalized:
            raise ChatConfigurationError(
                provider=self.name,
                message=(
                    "Invalid Bedrock model ID. Expected a provider-qualified model "
                    "like 'anthropic.claude-3-5-sonnet-20241022-v2:0'."
                ),
            )
        return normalized

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        model = self._normalize_model(request.get("model"))
        payload: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
        }
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "seed",
        ):
            value = request.get(key)
            if value is not None:
                payload[key] = value
        logprobs = request.get("logprobs")
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if logprobs and request.get("top_logprobs") is not None:
            payload["top_logprobs"] = request.get("top_logprobs")
        # Optional fields
        if request.get("response_format") is not None:
            payload["response_format"] = request.get("response_format")
        if request.get("stop") is not None:
            payload["stop"] = request.get("stop")
        tools = request.get("tools")
        if tools is not None:
            payload["tools"] = tools
        apply_tool_choice(payload, tools, request.get("tool_choice"))
        return payload

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        if not self._use_native_http():
            raise RuntimeError("BedrockAdapter native HTTP disabled by configuration")

        url = _build_chat_completions_url(self._base_url(request))
        payload = self._build_payload(request)
        payload["stream"] = False
        payload = merge_extra_body(payload, request)
        headers = self._build_headers(request=request, url=url, payload=payload)
        try:
            with http_client_factory(timeout=timeout or 90.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise self.normalize_error(e) from e

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        if not self._use_native_http():
            raise RuntimeError("BedrockAdapter native HTTP disabled by configuration")

        url = _build_chat_completions_url(self._base_url(request))
        payload = self._build_payload(request)
        payload["stream"] = True
        payload = merge_extra_body(payload, request)
        headers = self._build_headers(request=request, url=url, payload=payload)
        try:
            with http_client_factory(timeout=timeout or 90.0) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    seen_done = False
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        # Normalize to str for helper functions
                        try:
                            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                        except Exception:
                            line = str(raw)
                        if is_done_line(line):
                            if not seen_done:
                                seen_done = True
                                yield sse_done()
                            continue
                        normalized = normalize_provider_line(line)
                        if normalized is not None:
                            yield normalized
                    # Ensure a single terminal DONE marker
                    yield from finalize_stream(response=resp, done_already=seen_done)
            return
        except Exception as e:
            raise self.normalize_error(e) from e

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        # Reuse Groq/OpenAI-style mapping which inspects httpx/requests error payloads
        from tldw_Server_API.app.core.LLM_Calls.error_utils import (
            get_http_error_text,
            get_http_status_from_exception,
            is_http_status_error,
            log_http_400_body,
        )
        if is_http_status_error(exc):
            from tldw_Server_API.app.core.Chat.Chat_Deps import (
                ChatAPIError,
                ChatAuthenticationError,
                ChatBadRequestError,
                ChatProviderError,
                ChatRateLimitError,
            )
            resp = getattr(exc, "response", None)
            status = get_http_status_from_exception(exc)
            body = None
            try:
                body = resp.json()
            except Exception:
                body = None
            log_http_400_body(self.name, exc, body)
            detail = None
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                eobj = body["error"]
                msg = (eobj.get("message") or "").strip()
                typ = (eobj.get("type") or "").strip()
                detail = (f"{typ} {msg}" if typ else msg) or str(exc)
            else:
                detail = get_http_error_text(exc)
            if status in (400, 404, 422):
                return ChatBadRequestError(provider=self.name, message=str(detail))
            if status in (401, 403):
                return ChatAuthenticationError(provider=self.name, message=str(detail))
            if status == 429:
                return ChatRateLimitError(provider=self.name, message=str(detail))
            if status and 500 <= status < 600:
                return ChatProviderError(provider=self.name, message=str(detail), status_code=status)
            return ChatAPIError(provider=self.name, message=str(detail), status_code=status or 500)
        return super().normalize_error(exc)
