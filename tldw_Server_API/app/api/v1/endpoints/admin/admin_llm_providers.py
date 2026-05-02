"""Admin endpoints for managing LLM provider overrides and tests."""

from __future__ import annotations

import asyncio
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
    ) -> LLMProviderTestResponse: ...


def get_admin_llm_providers_service() -> AdminLLMProvidersService:
    """Return the admin LLM providers service for DI overrides."""
    return admin_llm_providers_service


def _get_ensure_sqlite_authnz_ready_if_test_mode() -> Callable[[], Awaitable[None]]:
    """Return the AuthNZ test-mode readiness hook."""
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._ensure_sqlite_authnz_ready_if_test_mode


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

    # Get configured providers
    try:
        from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import get_available_providers

        providers = get_available_providers()
    except Exception:
        providers = []

    if not providers:
        # Fallback: list from overrides
        try:
            overrides = await admin_llm_providers_service.list_overrides(None)
            providers = [o.provider for o in (overrides.overrides or [])]
        except Exception:
            providers = []

    results: list[dict[str, Any]] = []

    async def _check_provider(provider_name: str) -> dict[str, Any]:
        start = time.monotonic()
        try:
            test_result = await admin_llm_providers_service.test_provider(
                LLMProviderTestRequest(provider=provider_name)
            )
            latency_ms = round((time.monotonic() - start) * 1000)
            return {
                "provider": provider_name,
                "status": "healthy" if test_result.success else "unhealthy",
                "latency_ms": latency_ms,
                "message": test_result.message if hasattr(test_result, "message") else None,
                "error": test_result.error if hasattr(test_result, "error") and not test_result.success else None,
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
        check_tasks = [_check_provider(p) for p in providers[:20]]  # Cap at 20 providers
        results = await asyncio.gather(*check_tasks)

    healthy_count = sum(1 for r in results if r["status"] == "healthy")
    return {
        "providers": results,
        "total": len(results),
        "healthy": healthy_count,
        "unhealthy": len(results) - healthy_count,
    }
