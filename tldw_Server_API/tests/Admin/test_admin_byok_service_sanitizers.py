import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.user_keys import (
    SharedProviderKeyTestRequest,
    SharedProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_byok_service as service


pytestmark = pytest.mark.unit


class _SharedRepo:
    def __init__(self, *, list_error: Exception | None = None) -> None:
        self.list_error = list_error

    async def fetch_secret(self, *_args):
        return {"encrypted_blob": "encrypted-provider-secret"}

    async def list_secrets(self, **_kwargs):
        if self.list_error is not None:
            raise self.list_error
        return []


class _ExplodingUserRepo:
    async def list_secrets_for_user(self, _user_id: int):
        raise RuntimeError("user BYOK list failed at /private/byok-user.db")

    async def delete_secret(self, *_args, **_kwargs):
        raise RuntimeError("user BYOK revoke failed at /private/byok-user.db")


class _ExplodingSharedRepo:
    async def upsert_secret(self, **_kwargs):
        raise RuntimeError("shared BYOK upsert failed at /private/byok-shared.db")

    async def list_secrets(self, **_kwargs):
        raise RuntimeError("shared BYOK list failed at /private/byok-shared.db")

    async def delete_secret(self, *_args, **_kwargs):
        raise RuntimeError("shared BYOK delete failed at /private/byok-shared.db")


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


def _allow_byok(monkeypatch) -> None:
    monkeypatch.setattr(service, "require_byok_enabled", lambda: None)
    monkeypatch.setattr(service, "is_provider_allowlisted", lambda _provider: True)


async def _repo() -> _SharedRepo:
    return _SharedRepo()


def _stored_payload(*_args, **_kwargs):
    return {"api_key": "sk-test", "credential_fields": {"base_url": "https://api.example.test"}}


async def _assert_byok_infrastructure_log_sanitized(call, expected_log: str, raw_marker: str) -> None:
    messages: list[str] = []
    sink_id = service.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(HTTPException) as exc_info:
            await call()
    finally:
        service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK infrastructure is not available"
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


async def _assert_byok_operation_log_sanitized(
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
async def test_get_user_byok_repo_sanitizes_infrastructure_failure_log(monkeypatch):
    async def fail_get_db_pool():
        raise RuntimeError("user BYOK DB failed at /private/byok-user.db")

    monkeypatch.setattr(service, "get_db_pool", fail_get_db_pool)

    await _assert_byok_infrastructure_log_sanitized(
        service.get_user_byok_repo,
        "Failed to initialize user BYOK repository",
        "user BYOK DB failed",
    )


@pytest.mark.asyncio
async def test_get_shared_byok_repo_sanitizes_infrastructure_failure_log(monkeypatch):
    async def fail_get_db_pool():
        raise RuntimeError("shared BYOK DB failed at /private/byok-shared.db")

    monkeypatch.setattr(service, "get_db_pool", fail_get_db_pool)

    await _assert_byok_infrastructure_log_sanitized(
        service.get_shared_byok_repo,
        "Failed to initialize shared BYOK repository",
        "shared BYOK DB failed",
    )


@pytest.mark.asyncio
async def test_upsert_shared_key_sanitizes_credential_field_validation(monkeypatch):
    def fail_normalize(_provider, _fields):
        raise ValueError("shared credential token at /private/shared-byok.json")

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "normalize_credential_fields", fail_normalize)

    with pytest.raises(HTTPException) as exc_info:
        await service.upsert_shared_key(
            _principal(),
            SharedProviderKeyUpsertRequest(
                scope_type="org",
                scope_id=42,
                provider="openai",
                api_key="sk-test",
                credential_fields={"base_url": "https://api.example.test"},
            ),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "shared credential token" not in exc_info.value.detail
    assert "/private/shared-byok.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_upsert_shared_key_sanitizes_provider_validation_failures(monkeypatch):
    async def fail_provider_test(**_kwargs):
        raise ValueError("shared provider token at /private/shared-provider.json")

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await service.upsert_shared_key(
            _principal(),
            SharedProviderKeyUpsertRequest(
                scope_type="org",
                scope_id=42,
                provider="openai",
                api_key="sk-test",
            ),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "shared provider token" not in exc_info.value.detail
    assert "/private/shared-provider.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_test_shared_key_sanitizes_stored_credential_field_validation(monkeypatch):
    async def get_repo():
        return _SharedRepo()

    def fail_normalize(_provider, _fields):
        raise ValueError("stored credential token at /private/stored-byok.json")

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_repo)
    monkeypatch.setattr(service, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(service, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(service, "normalize_credential_fields", fail_normalize)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_shared_key(
            _principal(),
            SharedProviderKeyTestRequest(scope_type="org", scope_id=42, provider="openai"),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "stored credential token" not in exc_info.value.detail
    assert "/private/stored-byok.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_test_shared_key_sanitizes_provider_validation_failures(monkeypatch):
    async def get_repo():
        return _SharedRepo()

    async def fail_provider_test(**_kwargs):
        raise ValueError("stored provider token at /private/stored-provider.json")

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_repo)
    monkeypatch.setattr(service, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(service, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await service.test_shared_key(
            _principal(),
            SharedProviderKeyTestRequest(scope_type="org", scope_id=42, provider="openai"),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "stored provider token" not in exc_info.value.detail
    assert "/private/stored-provider.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_shared_keys_sanitizes_filter_validation(monkeypatch):
    async def get_repo():
        return _SharedRepo(list_error=ValueError("shared list token at /private/byok-list.json"))

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_repo)

    with pytest.raises(HTTPException) as exc_info:
        await service.list_shared_keys(
            _principal(),
            scope_type="org",
            scope_id=42,
            provider="openai",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid shared BYOK key query"
    assert "shared list token" not in exc_info.value.detail
    assert "/private/byok-list.json" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_user_keys_sanitizes_backend_failure_log(monkeypatch):
    async def get_user_repo():
        return _ExplodingUserRepo()

    async def allow_scope(*_args, **_kwargs):
        return None

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_user_byok_repo", get_user_repo)
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", allow_scope)

    await _assert_byok_operation_log_sanitized(
        lambda: service.list_user_keys(_principal(), 42),
        expected_detail="Failed to list user BYOK keys",
        expected_log="Failed to list user BYOK keys",
        raw_marker="user BYOK list failed",
    )


@pytest.mark.asyncio
async def test_revoke_user_key_sanitizes_backend_failure_log(monkeypatch):
    async def get_user_repo():
        return _ExplodingUserRepo()

    async def allow_scope(*_args, **_kwargs):
        return None

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_user_byok_repo", get_user_repo)
    monkeypatch.setattr(service.admin_scope_service, "enforce_admin_user_scope", allow_scope)

    await _assert_byok_operation_log_sanitized(
        lambda: service.revoke_user_key(_principal(), 42, "openai"),
        expected_detail="Failed to revoke user BYOK key",
        expected_log="Failed to revoke user BYOK key",
        raw_marker="user BYOK revoke failed",
    )


@pytest.mark.asyncio
async def test_upsert_shared_key_sanitizes_backend_failure_log(monkeypatch):
    async def get_shared_repo():
        return _ExplodingSharedRepo()

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(service, "encrypt_byok_payload", lambda _payload: {"ciphertext": "sealed"})
    monkeypatch.setattr(service, "dumps_envelope", lambda _envelope: "sealed-envelope")

    await _assert_byok_operation_log_sanitized(
        lambda: service.upsert_shared_key(
            _principal(),
            SharedProviderKeyUpsertRequest(
                scope_type="org",
                scope_id=42,
                provider="openai",
                api_key="sk-test",
            ),
        ),
        expected_detail="Failed to store shared BYOK key",
        expected_log="Failed to store shared BYOK key",
        raw_marker="shared BYOK upsert failed",
    )


@pytest.mark.asyncio
async def test_list_shared_keys_sanitizes_backend_failure_log(monkeypatch):
    async def get_shared_repo():
        return _ExplodingSharedRepo()

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)

    await _assert_byok_operation_log_sanitized(
        lambda: service.list_shared_keys(
            _principal(),
            scope_type="org",
            scope_id=42,
            provider="openai",
        ),
        expected_detail="Failed to list shared BYOK keys",
        expected_log="Failed to list shared BYOK keys",
        raw_marker="shared BYOK list failed",
    )


@pytest.mark.asyncio
async def test_delete_shared_key_sanitizes_backend_failure_log(monkeypatch):
    async def get_shared_repo():
        return _ExplodingSharedRepo()

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)

    await _assert_byok_operation_log_sanitized(
        lambda: service.delete_shared_key(_principal(), "org", 42, "openai"),
        expected_detail="Failed to delete shared BYOK key",
        expected_log="Failed to delete shared BYOK key",
        raw_marker="shared BYOK delete failed",
    )
