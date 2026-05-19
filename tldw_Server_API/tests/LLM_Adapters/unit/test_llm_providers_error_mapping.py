from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import llm_providers as llm_endpoints


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.errors: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.warnings: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append((str(message), args, kwargs))

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append((str(message), args, kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append((str(message), args, kwargs))

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _assert_sanitized_log(
    records: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    expected_message: str,
) -> None:
    assert records == [(expected_message, (), {})]
    rendered = " ".join(record[0] for record in records)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert "private-provider" not in rendered
    assert "127.0.0.1" not in rendered


class _FakeModelResponse:
    status_code = 503

    def close(self) -> None:
        return None


class _UnexpectedModelDiscoveryError(Exception):
    pass


def test_discover_models_from_endpoint_sanitizes_http_status_log(monkeypatch):
    logger_stub = _LoggerStub()

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(
        llm_endpoints,
        "_http_fetch",
        lambda **_kwargs: _FakeModelResponse(),
    )
    llm_endpoints._LOCAL_MODEL_CACHE.clear()

    models = llm_endpoints.discover_models_from_endpoint(
        "private-provider",
        "http://127.0.0.1:1234/v1",
    )

    assert models == []
    _assert_sanitized_log(
        logger_stub.debugs,
        "Model discovery endpoint returned an error status",
    )


def test_discover_models_from_endpoint_sanitizes_noncritical_error_log(monkeypatch):
    logger_stub = _LoggerStub()

    def boom(**_kwargs):
        raise RuntimeError("model discovery backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "_http_fetch", boom)
    llm_endpoints._LOCAL_MODEL_CACHE.clear()

    models = llm_endpoints.discover_models_from_endpoint(
        "private-provider",
        "http://127.0.0.1:1234/v1",
    )

    assert models == []
    _assert_sanitized_log(
        logger_stub.debugs,
        "Model discovery endpoint query failed",
    )


def test_discover_models_from_endpoint_sanitizes_unexpected_error_log(monkeypatch):
    logger_stub = _LoggerStub()

    def boom(**_kwargs):
        raise _UnexpectedModelDiscoveryError(
            "unexpected model discovery exploded at /private/llm-providers.db"
        )

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "_http_fetch", boom)
    llm_endpoints._LOCAL_MODEL_CACHE.clear()

    models = llm_endpoints.discover_models_from_endpoint(
        "private-provider",
        "http://127.0.0.1:1234/v1",
    )

    assert models == []
    _assert_sanitized_log(
        logger_stub.debugs,
        "Model discovery endpoint query failed unexpectedly",
    )


def test_get_configured_providers_sanitizes_generic_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def boom():
        raise RuntimeError("llm config backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "load_comprehensive_config", boom)

    result = llm_endpoints.get_configured_providers()

    assert result["providers"] == []
    assert result["error"]
    _assert_sanitized_log(logger_stub.errors, "Error getting configured providers")


@pytest.mark.asyncio
async def test_get_llm_providers_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_llm_providers()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve LLM providers"
    _assert_sanitized_log(logger_stub.errors, "Error in get_llm_providers endpoint")


@pytest.mark.asyncio
async def test_get_models_metadata_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_models_metadata(
            request=None,
            refresh_openrouter=False,
            model_type=None,
            input_modality=None,
            output_modality=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve model metadata"
    _assert_sanitized_log(logger_stub.errors, "Error getting models metadata")


@pytest.mark.asyncio
async def test_get_models_metadata_sanitizes_image_model_warning(monkeypatch):
    logger_stub = _LoggerStub()

    async def configured(**_kwargs):
        return {"providers": []}

    def boom():
        raise RuntimeError("image catalog backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", configured)
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", boom)

    result = await llm_endpoints.get_models_metadata(
        request=None,
        refresh_openrouter=False,
        model_type=None,
        input_modality=None,
        output_modality=None,
    )

    assert result == {"models": [], "total": 0}
    _assert_sanitized_log(logger_stub.warnings, "Failed to list image generation models")


@pytest.mark.asyncio
async def test_get_provider_details_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_provider_details("openai")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve provider details"
    _assert_sanitized_log(logger_stub.errors, "Error getting provider details")


@pytest.mark.asyncio
async def test_get_all_models_sanitizes_generic_failure(monkeypatch):
    logger_stub = _LoggerStub()

    async def boom(**_kwargs):
        raise RuntimeError("llm providers backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", boom)

    with pytest.raises(HTTPException) as exc_info:
        await llm_endpoints.get_all_models(
            model_type=None,
            input_modality=None,
            output_modality=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve models"
    _assert_sanitized_log(logger_stub.errors, "Error getting all models")


@pytest.mark.asyncio
async def test_get_all_models_sanitizes_image_model_warning(monkeypatch):
    logger_stub = _LoggerStub()

    async def configured(**_kwargs):
        return {"providers": []}

    def boom():
        raise RuntimeError("image catalog backend exploded at /private/llm-providers.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(llm_endpoints, "get_configured_providers_async", configured)
    monkeypatch.setattr(llm_endpoints, "list_image_models_for_catalog", boom)

    result = await llm_endpoints.get_all_models(
        model_type=None,
        input_modality=None,
        output_modality=None,
    )

    assert result == []
    _assert_sanitized_log(logger_stub.warnings, "Failed to list image generation models")


@pytest.mark.asyncio
async def test_llm_health_sanitizes_provider_manager_failure(monkeypatch):
    from tldw_Server_API.app.core.Chat import provider_manager

    logger_stub = _LoggerStub()

    def boom():
        raise RuntimeError("llm provider manager exploded at /private/provider.db")

    monkeypatch.setattr(llm_endpoints, "logger", logger_stub)
    monkeypatch.setattr(provider_manager, "get_provider_manager", boom)

    response = await llm_endpoints.llm_health()

    assert response["status"] == "unhealthy"
    assert response["error"] == "LLM health check failed"
    _assert_sanitized_log(logger_stub.errors, "LLM health check failed")
