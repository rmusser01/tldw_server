"""Local provider endpoint validation for first-run setup."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderValidationResponse

VALIDATION_STATUS_READY = "ready"
VALIDATION_STATUS_ACCEPTED = "accepted"
VALIDATION_STATUS_FAILED = "failed"
FAILURE_LOCAL_PROVIDER_UNREACHABLE = "local_provider_unreachable"
FAILURE_AUTH_FAILED = "auth_failed"
FAILURE_UNSUPPORTED_API_SHAPE = "unsupported_api_shape"
FAILURE_MODEL_DISCOVERY_UNAVAILABLE = "model_discovery_unavailable"
FAILURE_PROVIDER_API_KEY_REQUIRED = "provider_api_key_required"
FAILURE_PROVIDER_API_KEY_INVALID = "provider_api_key_invalid"
FAILURE_LOCAL_PROVIDER_ENDPOINT_NOT_ALLOWED = "local_provider_endpoint_not_allowed"
_ALLOWED_PRIVATE_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_ALLOWED_PRIVATE_IPV6_NETWORKS = (ipaddress.ip_network("fc00::/7"),)
_ALLOWED_LOCAL_HOST_SUFFIXES = (".home", ".internal", ".lan", ".local")


class LocalEndpointValidationRequest(BaseModel):
    provider_key: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model: str | None = None
    api_key: str | None = None


class HostedProviderValidationRequest(BaseModel):
    provider_key: str = Field(..., min_length=1)
    api_key: str | None = None


def _failed_response(
    payload: LocalEndpointValidationRequest,
    *,
    failure_category: str,
    message: str,
) -> SetupProviderValidationResponse:
    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status=VALIDATION_STATUS_FAILED,
        failure_category=failure_category,
        message=message,
    )


def _manual_model_fallback_response(
    payload: LocalEndpointValidationRequest,
) -> SetupProviderValidationResponse:
    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status=VALIDATION_STATUS_ACCEPTED,
        failure_category=FAILURE_MODEL_DISCOVERY_UNAVAILABLE,
        message="Model discovery is unavailable. Enter the model name manually; first chat will verify it.",
        validation_level="live_endpoint_shape",
        can_gate_first_chat=True,
    )


def _extract_model_ids(body: Any) -> list[str] | None:
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None

    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            return None
        model_ids.append(model_id)
    return model_ids


def _has_kobold_native_result(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    results = body.get("results")
    if not isinstance(results, list) or not results:
        return False
    first_result = results[0]
    if not isinstance(first_result, dict):
        return False
    return isinstance(first_result.get("text"), str)


def _is_allowed_local_provider_host(hostname: str) -> bool:
    normalized_host = hostname.strip().lower()
    if normalized_host == "localhost":
        return True
    if any(normalized_host.endswith(suffix) for suffix in _ALLOWED_LOCAL_HOST_SUFFIXES):
        return True

    if "%" in normalized_host:
        normalized_host = normalized_host.split("%", 1)[0]

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return False

    if address.is_multicast or address.is_unspecified or address.is_link_local:
        return False

    if address.version == 4:
        return address.is_loopback or any(address in network for network in _ALLOWED_PRIVATE_IPV4_NETWORKS)

    if address.is_loopback:
        return True
    return any(address in network for network in _ALLOWED_PRIVATE_IPV6_NETWORKS)


def _validate_local_provider_target(
    payload: LocalEndpointValidationRequest,
) -> SetupProviderValidationResponse | None:
    try:
        parsed = urlsplit(payload.base_url)
    except ValueError:
        return _failed_response(
            payload,
            failure_category=FAILURE_LOCAL_PROVIDER_UNREACHABLE,
            message="Local provider endpoint is unreachable.",
        )

    if parsed.scheme.lower() not in {"http", "https"}:
        return _failed_response(
            payload,
            failure_category=FAILURE_LOCAL_PROVIDER_ENDPOINT_NOT_ALLOWED,
            message="Local provider endpoint target is not allowed.",
        )
    if not parsed.hostname or not _is_allowed_local_provider_host(parsed.hostname):
        return _failed_response(
            payload,
            failure_category=FAILURE_LOCAL_PROVIDER_ENDPOINT_NOT_ALLOWED,
            message="Local provider endpoint target is not allowed.",
        )
    return None


def validate_hosted_provider_credentials(
    payload: HostedProviderValidationRequest,
) -> SetupProviderValidationResponse:
    """Validate hosted credentials with local-only syntax and presence checks."""
    provider_key = payload.provider_key.strip().lower()
    api_key = payload.api_key.strip() if payload.api_key is not None else ""

    if not api_key:
        return SetupProviderValidationResponse(
            provider_key=provider_key,
            status=VALIDATION_STATUS_FAILED,
            failure_category=FAILURE_PROVIDER_API_KEY_REQUIRED,
            message="Provider API key is required.",
        )

    if provider_key == "openai" and not api_key.startswith("sk-"):
        return SetupProviderValidationResponse(
            provider_key=provider_key,
            status=VALIDATION_STATUS_FAILED,
            failure_category=FAILURE_PROVIDER_API_KEY_INVALID,
            message="OpenAI API key format is not recognized.",
        )

    return SetupProviderValidationResponse(
        provider_key=provider_key,
        status=VALIDATION_STATUS_ACCEPTED,
        message="Provider credentials passed local syntax checks.",
        validation_level="local_syntax",
        can_gate_first_chat=True,
    )


def _create_validation_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=5.0)


def _openai_compatible_models_url(base_url: str) -> str:
    """Return the OpenAI-compatible models URL matching runtime URL normalization."""
    base = base_url.rstrip("/")
    lower = base.lower()
    if lower.endswith("/models"):
        return base
    if lower.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


async def validate_local_openai_endpoint(
    payload: LocalEndpointValidationRequest,
) -> SetupProviderValidationResponse:
    """Validate a local OpenAI-compatible endpoint using its ``/models`` shape."""
    if rejected_response := _validate_local_provider_target(payload):
        return rejected_response

    models_url = _openai_compatible_models_url(payload.base_url)
    headers: dict[str, str] = {}
    if payload.api_key:
        headers["Authorization"] = f"Bearer {payload.api_key}"

    try:
        async with _create_validation_client() as client:
            response = await client.get(models_url, headers=headers)
    except (httpx.InvalidURL, httpx.HTTPError, TimeoutError, OSError):
        return _failed_response(
            payload,
            failure_category=FAILURE_LOCAL_PROVIDER_UNREACHABLE,
            message="Local provider endpoint is unreachable.",
        )

    if response.status_code in {401, 403}:
        return _failed_response(
            payload,
            failure_category=FAILURE_AUTH_FAILED,
            message="Local provider rejected the supplied credentials.",
        )
    if response.status_code >= 400:
        return _manual_model_fallback_response(payload)

    try:
        body = response.json()
    except ValueError:
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not return valid JSON.",
        )

    model_ids = _extract_model_ids(body)
    if model_ids is None or not model_ids:
        return _manual_model_fallback_response(payload)

    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status=VALIDATION_STATUS_READY,
        models=model_ids,
        validation_level="live_non_generative",
        can_gate_first_chat=True,
    )


async def validate_native_kobold_endpoint(
    payload: LocalEndpointValidationRequest,
) -> SetupProviderValidationResponse:
    """Validate a Kobold.cpp native ``/api/v1/generate`` endpoint shape."""
    if rejected_response := _validate_local_provider_target(payload):
        return rejected_response

    headers = {"Content-Type": "application/json"}
    if payload.api_key:
        headers["X-Api-Key"] = payload.api_key
    request_body = {
        "prompt": "ping",
        "max_context_length": 128,
        "max_length": 1,
        "temperature": 0.0,
    }

    try:
        async with _create_validation_client() as client:
            response = await client.post(
                payload.base_url,
                headers=headers,
                json=request_body,
            )
    except (httpx.InvalidURL, httpx.HTTPError, TimeoutError, OSError):
        return _failed_response(
            payload,
            failure_category=FAILURE_LOCAL_PROVIDER_UNREACHABLE,
            message="Local provider endpoint is unreachable.",
        )

    if response.status_code in {401, 403}:
        return _failed_response(
            payload,
            failure_category=FAILURE_AUTH_FAILED,
            message="Local provider rejected the supplied credentials.",
        )
    if response.status_code >= 400:
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not return a supported Kobold-compatible response.",
        )

    try:
        body = response.json()
    except ValueError:
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not return valid JSON.",
        )

    if not _has_kobold_native_result(body):
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not expose a Kobold-compatible generate response.",
        )

    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status=VALIDATION_STATUS_READY,
        validation_level="live_endpoint_shape",
        can_gate_first_chat=True,
    )
