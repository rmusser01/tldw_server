import asyncio

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    LLMProviderOverrideRequest,
    LLMProviderTestRequest,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverridesRefreshError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
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

    async def patch_override(self, **_kwargs):
        raise RuntimeError("provider override upsert failed at /private/provider-overrides.db")

    async def delete_override(self, _provider: str):
        raise RuntimeError("provider override delete failed at /private/provider-overrides.db")


class _ExplodingFetchOverrideRepo:
    async def fetch_override(self, _provider: str):
        raise RuntimeError(
            "provider override read exposed sk-fetch-secret at /private/provider-overrides.db"
        )


async def _noop_refresh(*, force: bool | None = None) -> None:
    return None


async def _failed_refresh(*, force: bool | None = None) -> None:
    raise LLMProviderOverridesRefreshError()


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
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
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
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            ChatAuthenticationError(
                message="hostile auth body sk-admin-auth-/private/admin-auth.json",
                provider="openai",
                status_code=403,
            ),
            502,
            "The selected provider credentials could not be authenticated.",
        ),
        (
            ChatBadRequestError(
                message="hostile request body sk-admin-request-/private/admin-request.json",
                provider="openai",
            ),
            400,
            "The selected provider configuration is invalid.",
        ),
        (
            ChatRateLimitError(
                message="hostile rate body sk-admin-rate-/private/admin-rate.json",
                provider="openai",
            ),
            429,
            "The chat service provider is currently unavailable.",
        ),
        (
            ChatProviderError(
                message="hostile timeout body sk-admin-timeout-/private/admin-timeout.json",
                provider="openai",
                status_code=504,
            ),
            504,
            "The chat service provider is currently unavailable.",
        ),
    ],
)
async def test_test_provider_preserves_bounded_provider_validation_status(
    monkeypatch,
    error,
    expected_status: int,
    expected_detail: str,
) -> None:
    async def fail_provider_test(**_kwargs):
        raise error

    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_provider(
            LLMProviderTestRequest(
                provider="openai",
                api_key="sk-test",
                model="gpt-test",
                use_override=False,
            )
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "hostile" not in repr(exc_info.value)
    assert "/private/" not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["", "   ", "voyage", "elevenlabs", "unknown-provider"],
)
async def test_test_provider_rejects_unsupported_identity_before_refresh_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    refresh_called = False
    dispatch_called = False

    async def refresh(*, force: bool = False) -> None:
        nonlocal refresh_called
        refresh_called = True

    async def dispatch(**_kwargs) -> None:
        nonlocal dispatch_called
        dispatch_called = True

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", refresh)
    monkeypatch.setattr(service, "test_provider_credentials", dispatch)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_provider(
            LLMProviderTestRequest(
                provider=provider,
                api_key="sk-test",
                use_override=False,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unsupported LLM provider"
    assert refresh_called is False
    assert dispatch_called is False


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
async def test_upsert_override_sanitizes_backend_read_failure(monkeypatch):
    async def get_repo():
        return _ExplodingFetchOverrideRepo()

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)

    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await service.upsert_override(
                "openai",
                LLMProviderOverrideRequest(is_enabled=True),
            )
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to load provider override"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "Failed to load provider override" in joined
    assert "sk-fetch-secret" not in joined
    assert "/private/provider-overrides.db" not in joined


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["", "   ", "voyage", "elevenlabs", "unknown-provider"])
async def test_upsert_override_rejects_unsupported_identity_before_repo_access(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    repo_accessed = False

    async def get_repo():
        nonlocal repo_accessed
        repo_accessed = True
        return _NoExistingOverrideRepo()

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)

    with pytest.raises(HTTPException) as exc_info:
        await service.upsert_override(
            provider,
            LLMProviderOverrideRequest(is_enabled=True),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unsupported LLM provider"
    assert repo_accessed is False


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


@pytest.mark.asyncio
async def test_list_overrides_surfaces_store_refresh_failure_as_503(monkeypatch):
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", _failed_refresh)

    with pytest.raises(HTTPException) as exc_info:
        await service.list_overrides(None)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Provider credential storage is temporarily unavailable"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_successful_upsert_with_failed_refresh_never_returns_stale_success(monkeypatch):
    writes: list[dict] = []
    refresh_forces: list[bool | None] = []

    class SuccessfulRepo:
        async def fetch_override(self, _provider: str):
            return None

        async def patch_override(self, **kwargs):
            writes.append(kwargs)
            return {}

    async def get_repo():
        return SuccessfulRepo()

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)

    async def failed_refresh(*, force: bool | None = None) -> None:
        refresh_forces.append(force)
        raise LLMProviderOverridesRefreshError()

    monkeypatch.setattr(service, "refresh_llm_provider_overrides", failed_refresh)

    with pytest.raises(HTTPException) as exc_info:
        await service.upsert_override(
            "openai",
            LLMProviderOverrideRequest(is_enabled=True),
        )

    assert len(writes) == 1
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Provider credential storage is temporarily unavailable"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert refresh_forces == [True]


@pytest.mark.asyncio
async def test_successful_delete_preserves_cancelled_forced_refresh(monkeypatch):
    refresh_forces: list[bool | None] = []

    class SuccessfulRepo:
        async def delete_override(self, _provider: str) -> bool:
            return True

    async def get_repo():
        return SuccessfulRepo()

    async def cancelled_refresh(*, force: bool | None = None) -> None:
        refresh_forces.append(force)
        raise asyncio.CancelledError("refresh cancelled with /private/provider.key")

    monkeypatch.setattr(service, "get_llm_provider_overrides_repo", get_repo)
    monkeypatch.setattr(service, "refresh_llm_provider_overrides", cancelled_refresh)

    with pytest.raises(asyncio.CancelledError):
        await service.delete_override("openai")

    assert refresh_forces == [True]
