"""Admin endpoints for managing LLM provider overrides and tests."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import APIRouter, Depends, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
    get_auth_principal,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    LLMProviderOverrideListResponse,
    LLMProviderOverrideRequest,
    LLMProviderOverrideResponse,
    LLMProviderTestRequest,
    LLMProviderTestResponse,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    canonical_builtin_llm_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import (
    normalize_catalog_provider_for_chat,
)
from tldw_Server_API.app.services import admin_llm_providers_service

router = APIRouter()


class AdminLLMProvidersService(Protocol):
    async def list_overrides(
        self,
        provider: str | None,
    ) -> LLMProviderOverrideListResponse: ...

    async def get_override(
        self,
        provider: str,
    ) -> LLMProviderOverrideResponse: ...

    async def upsert_override(
        self,
        provider: str,
        payload: LLMProviderOverrideRequest,
    ) -> LLMProviderOverrideResponse: ...

    async def delete_override(self, provider: str) -> None: ...

    async def test_provider(
        self,
        payload: LLMProviderTestRequest,
        *,
        refresh_overrides: bool = True,
        timeout_seconds: float | None = None,
    ) -> LLMProviderTestResponse: ...


def get_admin_llm_providers_service() -> AdminLLMProvidersService:
    """Return the admin LLM providers service for DI overrides."""
    return admin_llm_providers_service


def _get_ensure_sqlite_authnz_ready_if_test_mode() -> Callable[[], Awaitable[None]]:
    """Return the AuthNZ test-mode readiness hook."""
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._ensure_sqlite_authnz_ready_if_test_mode


async def _get_configured_provider_names() -> list[str]:
    """Return providers configured through environment or config sources."""
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import (
            get_configured_providers_async,
        )

        response = await get_configured_providers_async(include_deprecated=False)
    except Exception:
        return []
    providers = response.get("providers") if isinstance(response, dict) else None
    if not isinstance(providers, list):
        return []
    return [
        str(item.get("name") or "")
        for item in providers
        if isinstance(item, dict)
        and item.get("is_configured") is True
        and item.get("provider_enabled") is not False
    ]


def _dedupe_provider_names(names: list[str]) -> list[str]:
    """Canonicalize supported chat providers and preserve first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw_name in names:
        try:
            name = canonical_builtin_llm_provider_name(
                normalize_catalog_provider_for_chat(raw_name)
            )
        except ValueError:
            continue
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _provider_health_timeout_seconds() -> float:
    """Return a short bounded timeout for each provider health check."""
    default = 5.0
    try:
        timeout = float(
            str(
                os.getenv(
                    "ADMIN_LLM_PROVIDER_HEALTH_TIMEOUT_SECONDS",
                    default,
                )
            ).strip()
        )
    except (TypeError, ValueError):
        return default
    return timeout if math.isfinite(timeout) and 0.05 <= timeout <= 30.0 else default


@router.get(
    "/llm/providers/overrides",
    response_model=LLMProviderOverrideListResponse,
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_list_llm_provider_overrides(
    provider: str | None = Query(None),
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> LLMProviderOverrideListResponse:
    """List LLM provider overrides (admin scope)."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_llm_providers_service.list_overrides(provider)


@router.get(
    "/llm/providers/overrides/{provider}",
    response_model=LLMProviderOverrideResponse,
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_get_llm_provider_override(
    provider: str,
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> LLMProviderOverrideResponse:
    """Get an LLM provider override (admin scope)."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_llm_providers_service.get_override(provider)


@router.put(
    "/llm/providers/overrides/{provider}",
    response_model=LLMProviderOverrideResponse,
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_upsert_llm_provider_override(
    provider: str,
    payload: LLMProviderOverrideRequest,
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> LLMProviderOverrideResponse:
    """Create or update an LLM provider override (admin scope)."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_llm_providers_service.upsert_override(provider, payload)


@router.delete(
    "/llm/providers/overrides/{provider}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_delete_llm_provider_override(
    provider: str,
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> Response:
    """Delete an LLM provider override (admin scope)."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    await admin_llm_providers_service.delete_override(provider)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/llm/providers/test",
    response_model=LLMProviderTestResponse,
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_test_llm_provider(
    payload: LLMProviderTestRequest,
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> LLMProviderTestResponse:
    """Test an LLM provider configuration (admin scope)."""
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()
    return await admin_llm_providers_service.test_provider(payload)


@router.get(
    "/llm/providers/health",
    dependencies=[Depends(get_auth_principal), Depends(check_rate_limit)],
)
async def admin_llm_providers_health(
    admin_llm_providers_service: AdminLLMProvidersService = Depends(
        get_admin_llm_providers_service,
    ),
) -> dict[str, Any]:
    """Batch health check for all configured LLM providers.

    Pings each provider with a minimal test request and returns
    per-provider health status, latency, and error details.
    """
    await _get_ensure_sqlite_authnz_ready_if_test_mode()()

    configured_providers = await _get_configured_provider_names()
    # Listing once refreshes the shared override snapshot for the whole batch.
    overrides = await admin_llm_providers_service.list_overrides(None)
    disabled_providers = set(
        _dedupe_provider_names(
            [
                override.provider
                for override in overrides.items
                if override.is_enabled is False
            ]
        )
    )
    providers = [
        provider
        for provider in _dedupe_provider_names(
            [
                *(
                    override.provider
                    for override in overrides.items
                    if override.is_enabled is not False
                ),
                *configured_providers,
            ]
        )
        if provider not in disabled_providers
    ]
    timeout_seconds = _provider_health_timeout_seconds()

    results: list[dict[str, Any]] = []

    async def _check_provider(provider_name: str) -> dict[str, Any]:
        start = time.monotonic()
        try:
            test_result = await asyncio.wait_for(
                admin_llm_providers_service.test_provider(
                    LLMProviderTestRequest(provider=provider_name),
                    refresh_overrides=False,
                    timeout_seconds=timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
            latency_ms = round((time.monotonic() - start) * 1000)
            healthy = test_result.status == "valid"
            return {
                "provider": provider_name,
                "status": "healthy" if healthy else "unhealthy",
                "latency_ms": latency_ms,
                "model": test_result.model,
            }
        except Exception:
            latency_ms = round((time.monotonic() - start) * 1000)
            logger.warning("LLM provider health check failed")
            return {
                "provider": provider_name,
                "status": "error",
                "latency_ms": latency_ms,
                "error": "Provider health check failed",
            }

    if providers:
        check_tasks = [_check_provider(p) for p in providers]
        results = await asyncio.gather(*check_tasks)

    healthy_count = sum(1 for r in results if r["status"] == "healthy")
    return {
        "providers": results,
        "total": len(results),
        "healthy": healthy_count,
        "unhealthy": len(results) - healthy_count,
    }
