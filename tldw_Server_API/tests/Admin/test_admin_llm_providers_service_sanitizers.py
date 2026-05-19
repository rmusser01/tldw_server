import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    LLMProviderOverrideRequest,
    LLMProviderTestRequest,
)
from tldw_Server_API.app.services import admin_llm_providers_service as service


pytestmark = pytest.mark.unit


class _NoExistingOverrideRepo:
    async def fetch_override(self, _provider: str):
        return None


class _ExplodingOverrideRepo:
    async def fetch_override(self, _provider: str):
        return None

    async def upsert_override(self, **_kwargs):
        raise RuntimeError("provider override upsert failed at /private/provider-overrides.db")

    async def delete_override(self, _provider: str):
        raise RuntimeError("provider override delete failed at /private/provider-overrides.db")


async def _noop_refresh() -> None:
    return None


async def _assert_provider_override_log_sanitized(
    call,
    *,
    expected_detail: str,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_get_llm_provider_overrides_repo_sanitizes_infrastructure_failure_log(monkeypatch):
    async def fail_get_db_pool():
        raise RuntimeError("provider overrides DB failed at /private/provider-overrides.db")

    monkeypatch.setattr(service, "get_db_pool", fail_get_db_pool)

    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await service.get_llm_provider_overrides_repo()
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Provider overrides infrastructure is not available"
    assert "Failed to initialize LLM provider overrides repository" in joined
    assert "provider overrides DB failed" not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_upsert_override_sanitizes_credential_field_validation(monkeypatch):
    async def get_repo():
        return _NoExistingOverrideRepo()

    def fail_normalize(_provider, _fields):
        raise ValueError("provider credential token at /private/provider-config.json")

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "_normalize_credential_fields", fail_normalize)

    with pytest.raises(HTTPException) as exc_info:
        await service.upsert_override(
            "openai",
            LLMProviderOverrideRequest(credential_fields={"base_url": "https://api.example.test"}),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "provider credential token" not in exc_info.value.detail
    assert "/private/provider-config.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_test_provider_sanitizes_credential_field_validation(monkeypatch):
    def fail_normalize(_provider, _fields):
        raise ValueError("provider test token at /private/provider-test.json")

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(service, "_normalize_credential_fields", fail_normalize)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_provider(
            LLMProviderTestRequest(
                provider="openai",
                api_key="sk-test",
                credential_fields={"base_url": "https://api.example.test"},
                use_override=False,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "provider test token" not in exc_info.value.detail
    assert "/private/provider-test.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_test_provider_sanitizes_provider_validation_failures(monkeypatch):
    async def fail_provider_test(**_kwargs):
        raise ValueError("provider validation token at /private/provider-validation.json")

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _noop_refresh)
    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_provider(
            LLMProviderTestRequest(provider="openai", api_key="sk-test", use_override=False)
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "provider validation token" not in exc_info.value.detail
    assert "/private/provider-validation.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_upsert_override_sanitizes_backend_failure_log(monkeypatch):
    async def get_repo():
        return _ExplodingOverrideRepo()

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)

    await _assert_provider_override_log_sanitized(
        lambda: service.upsert_override("openai", LLMProviderOverrideRequest(is_enabled=True)),
        expected_detail="Failed to store provider override",
        expected_log="Failed to store provider override",
        raw_marker="provider override upsert failed",
    )


@pytest.mark.asyncio
async def test_delete_override_sanitizes_backend_failure_log(monkeypatch):
    async def get_repo():
        return _ExplodingOverrideRepo()

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)

    await _assert_provider_override_log_sanitized(
        lambda: service.delete_override("openai"),
        expected_detail="Failed to delete provider override",
        expected_log="Failed to delete provider override",
        raw_marker="provider override delete failed",
    )
