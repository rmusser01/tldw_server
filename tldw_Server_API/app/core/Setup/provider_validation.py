"""Local provider endpoint validation for first-run setup."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderValidationResponse

VALIDATION_STATUS_READY = "ready"
VALIDATION_STATUS_FAILED = "failed"
FAILURE_LOCAL_PROVIDER_UNREACHABLE = "local_provider_unreachable"
FAILURE_AUTH_FAILED = "auth_failed"
FAILURE_UNSUPPORTED_API_SHAPE = "unsupported_api_shape"


class LocalEndpointValidationRequest(BaseModel):
    provider_key: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model: str | None = None
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


def _create_validation_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=5.0)


async def validate_local_openai_endpoint(
    payload: LocalEndpointValidationRequest,
) -> SetupProviderValidationResponse:
    """Validate a local OpenAI-compatible endpoint using its ``/models`` shape."""
    models_url = f"{payload.base_url.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if payload.api_key:
        headers["Authorization"] = f"Bearer {payload.api_key}"

    try:
        async with _create_validation_client() as client:
            response = await client.get(models_url, headers=headers)
    except (httpx.HTTPError, TimeoutError, OSError):
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
            message="Local provider did not return a supported OpenAI-compatible response.",
        )

    try:
        body = response.json()
    except ValueError:
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not return valid JSON.",
        )

    model_ids = _extract_model_ids(body)
    if model_ids is None:
        return _failed_response(
            payload,
            failure_category=FAILURE_UNSUPPORTED_API_SHAPE,
            message="Local provider did not expose an OpenAI-compatible models list.",
        )

    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status=VALIDATION_STATUS_READY,
        models=model_ids,
    )
