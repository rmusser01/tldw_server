from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import llm_providers as llm_endpoints


@pytest.mark.asyncio
async def test_get_llm_providers_sanitizes_generic_failure(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded")

    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_llm_providers()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve LLM providers"


@pytest.mark.asyncio
async def test_get_models_metadata_sanitizes_generic_failure(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded")

    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_models_metadata()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve model metadata"


@pytest.mark.asyncio
async def test_get_provider_details_sanitizes_generic_failure(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded")

    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_provider_details("openai")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve provider details"


@pytest.mark.asyncio
async def test_get_all_models_sanitizes_generic_failure(monkeypatch):
    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded")

    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_all_models()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve models"


@pytest.mark.asyncio
async def test_llm_health_sanitizes_provider_manager_failure(monkeypatch):
    from tldw_Server_API.app.core.Chat import provider_manager

    def boom():
        raise RuntimeError("llm provider manager exploded at /private/provider.db")

    monkeypatch.setattr(provider_manager, "get_provider_manager", boom)

    response = await llm_endpoints.llm_health()

    assert response["status"] == "unhealthy"
    assert response["error"] == "LLM health check failed"
