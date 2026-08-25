import asyncio

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.user_keys import (
    SharedProviderKeyTestRequest,
    SharedProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipScopeNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatProviderError,
)
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


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


def _allow_byok(monkeypatch) -> None:
    monkeypatch.setattr(service, "require_byok_enabled", lambda: None)
    monkeypatch.setattr(service, "is_provider_allowlisted", lambda _provider: True)


def _assert_detached_validation_error(exc: HTTPException, sentinel: str) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert sentinel not in repr(exc)


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
async def test_touch_shared_last_used_sanitizes_decryption_failure_log(monkeypatch) -> None:
    sentinel = "shared-decrypt-failure-sk-secret-/private/shared-credential.json"
    logger_stub = _LoggerStub()

    def fail_decrypt(_envelope):
        raise ValueError(sentinel)

    monkeypatch.setattr(service, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(service, "decrypt_byok_payload", fail_decrypt)
    monkeypatch.setattr(service, "logger", logger_stub)

    await service.touch_shared_last_used_if_match(
        _SharedRepo(),
        scope_type="org",
        scope_id=42,
        provider="openai",
        api_key="sk-test",
    )

    assert logger_stub.debugs == ["BYOK: failed to decrypt shared secret"]
    assert sentinel not in repr(logger_stub.debugs)
    assert "/private/" not in repr(logger_stub.debugs)


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
    _assert_detached_validation_error(
        exc_info.value,
        "shared credential token at /private/shared-byok.json",
    )


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
    _assert_detached_validation_error(
        exc_info.value,
        "shared provider token at /private/shared-provider.json",
    )


@pytest.mark.asyncio
async def test_upsert_shared_key_detaches_encryption_failure(monkeypatch) -> None:
    sentinel = "shared-encrypt-failure-sk-secret-/private/shared-credential.json"

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(service, "encrypt_byok_payload", fail_encrypt)

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

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_upsert_shared_key_detaches_unexpected_provider_validation_failure(
    monkeypatch,
) -> None:
    sentinel = "sk-admin-runtime-error-/private/admin-provider-runtime.json"

    async def fail_provider_test(**_kwargs):
        raise RuntimeError(sentinel)

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

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Provider test call failed"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_admin_shared_key_upsert_sanitizes_chat_provider_error_context(monkeypatch):
    sentinel = "sk-admin-upstream-secret-/private/admin-provider-body.json"

    async def fail_provider_test(**_kwargs):
        raise ChatProviderError(
            message=f"hostile upstream body {sentinel}",
            status_code=502,
            provider="openai",
            details={"endpoint": f"https://provider.invalid/{sentinel}"},
        )

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

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_admin_byok_auth_failures_are_detached_upstream_502s(
    monkeypatch,
) -> None:
    sentinels = {
        "sk-admin-auth-401": "admin-byok-401-secret-/private/byok-401.json",
        "sk-admin-auth-403": "admin-byok-403-secret-/private/byok-403.json",
    }
    statuses = {"sk-admin-auth-401": 401, "sk-admin-auth-403": 403}
    entered: set[str] = set()
    all_entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_provider_test(**kwargs):
        api_key = kwargs["api_key"]
        entered.add(api_key)
        if entered == set(sentinels):
            all_entered.set()
        await release.wait()
        raise ChatAuthenticationError(
            message=sentinels[api_key],
            provider="openai",
            status_code=statuses[api_key],
        )

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", fail_provider_test)

    async def invoke(api_key: str):
        return await service.upsert_shared_key(
            _principal(),
            SharedProviderKeyUpsertRequest(
                scope_type="org",
                scope_id=42,
                provider="openai",
                api_key=api_key,
            ),
        )

    tasks = [asyncio.create_task(invoke(api_key)) for api_key in sentinels]
    try:
        await asyncio.wait_for(all_entered.wait(), timeout=1.0)
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert entered == set(sentinels)
    for result in results:
        assert isinstance(result, HTTPException)
        assert result.status_code == 502
        assert result.detail == (
            "The selected provider credentials could not be authenticated."
        )
        assert result.__cause__ is None
        assert result.__context__ is None
        assert all(sentinel not in repr(result) for sentinel in sentinels.values())


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
    _assert_detached_validation_error(
        exc_info.value,
        "stored credential token at /private/stored-byok.json",
    )


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
    _assert_detached_validation_error(
        exc_info.value,
        "stored provider token at /private/stored-provider.json",
    )


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
        lambda: service.revoke_user_key(_principal(), 42, "anthropic"),
        expected_detail="Failed to revoke user BYOK key",
        expected_log="Failed to revoke user BYOK key",
        raw_marker="user BYOK revoke failed",
    )


@pytest.mark.asyncio
async def test_upsert_shared_key_maps_alias_conflict_to_409(monkeypatch):
    class _AliasConflictRepo:
        async def upsert_secret(self, **_kwargs):
            raise ProviderCredentialAliasConflictError("sensitive alias detail")

    async def get_shared_repo():
        return _AliasConflictRepo()

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(service, "encrypt_byok_payload", lambda _payload: {"ciphertext": "sealed"})
    monkeypatch.setattr(service, "dumps_envelope", lambda _envelope: "sealed-envelope")

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

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Conflicting provider credential aliases"


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


@pytest.mark.asyncio
async def test_upsert_shared_key_canonicalizes_registered_alias(monkeypatch):
    captured: dict = {}

    class CapturingRepo:
        async def upsert_secret(self, **kwargs):
            captured.update(kwargs)
            return {"provider": kwargs["provider"], "key_hint": "test"}

    async def get_shared_repo():
        return CapturingRepo()

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda _provider, _fields: {})
    monkeypatch.setattr(service, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(service, "encrypt_byok_payload", lambda _payload: {"ciphertext": "sealed"})
    monkeypatch.setattr(service, "dumps_envelope", lambda _envelope: "sealed-envelope")

    response = await service.upsert_shared_key(
        _principal(),
        SharedProviderKeyUpsertRequest(
            scope_type="org",
            scope_id=42,
            provider="oai",
            api_key="sk-test",
        ),
    )

    assert captured["provider"] == "openai"
    assert captured["authorization_context"].actor_user_id == 7
    assert (
        captured["authorization_context"].required_authority
        is MembershipAuthority.PLATFORM_ADMIN
    )
    assert response.provider == "openai"


@pytest.mark.asyncio
async def test_delete_shared_key_supplies_platform_admin_authorization_context(
    monkeypatch,
) -> None:
    captured: dict = {}

    class CapturingRepo:
        async def delete_secret(self, *_args, **kwargs):
            captured.update(kwargs)
            return True

    async def get_shared_repo():
        return CapturingRepo()

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)

    await service.delete_shared_key(_principal(), "org", 42, "openai")

    assert captured["authorization_context"].actor_user_id == 7
    assert (
        captured["authorization_context"].required_authority
        is MembershipAuthority.PLATFORM_ADMIN
    )


@pytest.mark.asyncio
async def test_delete_shared_key_preserves_invalid_actor_as_forbidden(
    monkeypatch,
) -> None:
    class UnusedRepo:
        async def delete_secret(self, *_args, **_kwargs):
            pytest.fail("storage must not run for an invalid actor")

    async def get_shared_repo():
        return UnusedRepo()

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)

    invalid_principal = AuthPrincipal(
        kind="user",
        user_id=0,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )
    with pytest.raises(HTTPException) as exc_info:
        await service.delete_shared_key(
            invalid_principal,
            "org",
            42,
            "openai",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_type", "expected_status", "expected_detail"),
    (
        (
            MembershipAuthorizationError,
            403,
            "Not authorized to manage shared BYOK keys",
        ),
        (MembershipScopeNotFound, 404, "Shared BYOK scope not found"),
        (
            DatabaseLockError,
            503,
            "Authentication database is busy. Please retry shortly.",
        ),
        (
            ConnectionPoolExhaustedError,
            503,
            "Authentication database is busy. Please retry shortly.",
        ),
        (
            TimeoutError,
            503,
            "Authentication database is busy. Please retry shortly.",
        ),
    ),
)
async def test_shared_key_mutations_preserve_bounded_control_failures(
    monkeypatch,
    failure_type: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    class FailingRepo:
        async def upsert_secret(self, **_kwargs):
            raise failure_type()

        async def delete_secret(self, *_args, **_kwargs):
            raise failure_type()

    async def get_shared_repo():
        return FailingRepo()

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_shared_repo)
    monkeypatch.setattr(service, "normalize_credential_fields", lambda *_args: {})
    monkeypatch.setattr(service, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(
        service,
        "encrypt_byok_payload",
        lambda _payload: {"ciphertext": "sealed"},
    )
    monkeypatch.setattr(service, "dumps_envelope", lambda _envelope: "sealed")

    operations = (
        lambda: service.upsert_shared_key(
            _principal(),
            SharedProviderKeyUpsertRequest(
                scope_type="org",
                scope_id=42,
                provider="openai",
                api_key="sk-test",
            ),
        ),
        lambda: service.delete_shared_key(_principal(), "org", 42, "openai"),
    )
    for operation in operations:
        with pytest.raises(HTTPException) as exc_info:
            await operation()
        assert exc_info.value.status_code == expected_status
        assert exc_info.value.detail == expected_detail
        if expected_status == 503:
            from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
                get_authnz_transaction_policy,
            )

            assert exc_info.value.headers == {
                "Retry-After": str(
                    get_authnz_transaction_policy().busy_retry_after_seconds
                ),
            }


@pytest.mark.asyncio
async def test_list_shared_keys_maps_alias_conflict_to_bounded_409(monkeypatch):
    async def get_repo():
        return _SharedRepo(
            list_error=ProviderCredentialAliasConflictError("raw alias details"),
        )

    _allow_byok(monkeypatch)
    monkeypatch.setattr(service, "get_shared_byok_repo", get_repo)

    with pytest.raises(HTTPException) as exc_info:
        await service.list_shared_keys(
            _principal(),
            scope_type="org",
            scope_id=42,
            provider="custom-openai-api",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Conflicting provider credential aliases"
    assert "raw alias details" not in exc_info.value.detail
