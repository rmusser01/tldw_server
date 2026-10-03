from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import shared_keys_scoped as routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    ProviderKeyTestRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthority,
    MembershipScopeNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(str(message))


class _SharedRepo:
    async def fetch_secret_for_manager(self, **_kwargs):
        return {"encrypted_blob": "encrypted-shared-provider-secret"}

    async def touch_last_used(self, *_args):
        return None


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


def _non_admin_principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["user"], permissions=[], is_admin=False)


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [expected_message]
    rendered = " ".join(logger_stub.debugs)
    assert "exploded" not in rendered
    assert "/private/" not in rendered


def _assert_detached_validation_error(exc: HTTPException, sentinel: str) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert sentinel not in repr(exc)


async def _allow_scope(*_args, **_kwargs) -> None:
    return None


def _stored_payload(*_args, **_kwargs):
    return {"api_key": "sk-test", "credential_fields": {"base_url": "https://api.example.test"}}


def _install_common_patches(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(routes, "_require_org_manager", _allow_scope)
    monkeypatch.setattr(routes, "_require_team_manager", _allow_scope)
    monkeypatch.setattr(routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(routes, "is_trusted_base_url_request", lambda *_args, **_kwargs: True)


async def _repo() -> _SharedRepo:
    return _SharedRepo()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected_detail"),
    (
        ("org", "Organization not found"),
        ("team", "Team not found"),
    ),
)
async def test_scoped_shared_key_upsert_maps_stale_scope_to_sanitized_404(
    monkeypatch,
    scope: str,
    expected_detail: str,
) -> None:
    sentinel = f"deleted {scope} leaked from repository"

    class _MissingScopeRepo:
        async def upsert_secret(self, **_kwargs):
            failure = MembershipScopeNotFound()
            failure.args = (sentinel,)
            raise failure

    async def _missing_scope_repo():
        return _MissingScopeRepo()

    async def _provider_ok(**_kwargs):
        return "validated"

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _provider_ok)
    monkeypatch.setattr(routes, "encrypt_byok_payload", lambda _payload: {})
    monkeypatch.setattr(routes, "dumps_envelope", lambda _envelope: "encrypted")
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _missing_scope_repo)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(
                provider="openai",
                api_key="sk-test",
            ),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == expected_detail
    assert sentinel not in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
async def test_scoped_shared_key_upsert_maps_alias_conflict_to_409(
    monkeypatch,
    scope: str,
) -> None:
    class _AliasConflictRepo:
        async def upsert_secret(self, **_kwargs):
            raise ProviderCredentialAliasConflictError("sensitive alias detail")

    async def _alias_conflict_repo():
        return _AliasConflictRepo()

    async def _provider_ok(**_kwargs):
        return "validated"

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _provider_ok)
    monkeypatch.setattr(routes, "encrypt_byok_payload", lambda _payload: {})
    monkeypatch.setattr(routes, "dumps_envelope", lambda _envelope: "encrypted")
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _alias_conflict_repo)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}
    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Conflicting provider credential aliases"
    assert "sensitive alias detail" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_require_org_manager_backend_failure_log_is_sanitized(monkeypatch) -> None:
    logger_stub = _LoggerStub()

    async def fail_membership_lookup(**_kwargs):
        raise RuntimeError("org membership backend exploded at /private/shared-keys.db")

    monkeypatch.setattr(routes, "logger", logger_stub)
    monkeypatch.setattr(routes, "list_org_members", fail_membership_lookup)

    with pytest.raises(HTTPException) as exc_info:
        await routes._require_org_manager(_non_admin_principal(), org_id=42)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Org manager role required"
    _assert_sanitized_debug_log(logger_stub, "Org manager check failed")


@pytest.mark.asyncio
async def test_require_team_manager_backend_failure_log_is_sanitized(monkeypatch) -> None:
    logger_stub = _LoggerStub()

    async def fail_membership_lookup(_team_id):
        raise RuntimeError("team membership backend exploded at /private/shared-keys.db")

    monkeypatch.setattr(routes, "logger", logger_stub)
    monkeypatch.setattr(routes, "list_team_members", fail_membership_lookup)

    with pytest.raises(HTTPException) as exc_info:
        await routes._require_team_manager(_non_admin_principal(), team_id=42)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Team manager role required"
    _assert_sanitized_debug_log(logger_stub, "Team manager check failed")


@pytest.mark.parametrize(
    ("checker_name", "list_name", "identifier_name"),
    (
        ("_require_org_manager", "list_org_members", "org_id"),
        ("_require_team_manager", "list_team_members", "team_id"),
    ),
)
@pytest.mark.asyncio
async def test_inactive_scope_manager_is_rejected(
    monkeypatch,
    checker_name: str,
    list_name: str,
    identifier_name: str,
) -> None:
    async def _inactive_members(*_args, **_kwargs):
        return [{"user_id": 7, "role": "admin", "status": "inactive"}]

    monkeypatch.setattr(routes, list_name, _inactive_members)

    with pytest.raises(HTTPException) as exc_info:
        await getattr(routes, checker_name)(
            _non_admin_principal(),
            **{identifier_name: 42},
        )

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize("scope", ("org", "team"))
@pytest.mark.asyncio
async def test_scoped_shared_key_upsert_passes_persisted_authorization_context(
    monkeypatch,
    scope: str,
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingRepo:
        async def upsert_secret(self, **kwargs):
            captured.update(kwargs)
            return {"key_hint": "test", "updated_at": datetime.now(timezone.utc)}

    async def _capturing_repo():
        return _CapturingRepo()

    async def _provider_ok(**_kwargs):
        return "validated"

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _provider_ok)
    monkeypatch.setattr(routes, "encrypt_byok_payload", lambda _payload: {})
    monkeypatch.setattr(routes, "dumps_envelope", lambda _envelope: "encrypted")
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _capturing_repo)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}
    await endpoint(
        **kwargs,
        payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
        request=SimpleNamespace(),
        principal=_principal(),
    )

    context = captured["authorization_context"]
    assert context.actor_user_id == 7
    assert context.required_authority is MembershipAuthority.PLATFORM_ADMIN


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
async def test_scoped_shared_key_list_uses_lock_bound_manager_read(
    monkeypatch,
    scope: str,
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingRepo:
        async def list_secrets_for_manager(self, **kwargs):
            captured.update(kwargs)
            return []

    async def _capturing_repo():
        return _CapturingRepo()

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _capturing_repo)
    endpoint = routes.list_org_shared_keys if scope == "org" else routes.list_team_shared_keys
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    response = await endpoint(**kwargs, principal=_principal())

    assert response.items == []
    assert captured["scope_type"] == scope
    assert captured["scope_id"] == 42
    assert captured["authorization_context"].actor_user_id == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-shared-credentials.json"),
        ("team", "/private/team-shared-credentials.json"),
    ],
)
async def test_scoped_shared_key_upsert_sanitizes_credential_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError(f"shared credential token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(
                provider="openai",
                api_key="sk-test",
                credential_fields={"base_url": "https://api.example.test"},
            ),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert raw_token not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        f"shared credential token at {raw_token}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-provider-test.json"),
        ("team", "/private/team-provider-test.json"),
    ],
)
async def test_scoped_shared_key_upsert_sanitizes_provider_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError(f"shared provider token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert raw_token not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        f"shared provider token at {raw_token}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["org", "team"])
async def test_scoped_shared_key_upsert_detaches_unexpected_provider_validation_failure(
    monkeypatch,
    scope: str,
) -> None:
    sentinel = f"sk-{scope}-runtime-error-/private/{scope}-provider-runtime.json"

    async def fail_provider_test(**_kwargs):
        raise RuntimeError(sentinel)

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}
    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Provider test call failed"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["org", "team"])
async def test_scoped_shared_key_upsert_sanitizes_chat_provider_error_context(
    monkeypatch,
    scope: str,
) -> None:
    sentinel = f"sk-{scope}-upstream-secret-/private/{scope}-provider-body.json"

    async def fail_provider_test(**_kwargs):
        raise ChatProviderError(
            message=f"hostile upstream body {sentinel}",
            status_code=502,
            provider="openai",
            details={"endpoint": f"https://provider.invalid/{sentinel}"},
        )

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}
    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["org", "team"])
async def test_scoped_shared_key_upsert_detaches_encryption_failure(
    monkeypatch,
    scope: str,
) -> None:
    sentinel = f"sk-{scope}-encrypt-failure-/private/{scope}-credential.json"

    async def pass_provider_test(**_kwargs):
        return "gpt-test"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", pass_provider_test)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}
    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-stored-credentials.json"),
        ("team", "/private/team-stored-credentials.json"),
    ],
)
async def test_scoped_shared_key_test_sanitizes_stored_credential_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError(f"stored credential token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    endpoint = routes.test_org_shared_key if scope == "org" else routes.test_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert raw_token not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        f"stored credential token at {raw_token}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-stored-provider.json"),
        ("team", "/private/team-stored-provider.json"),
    ],
)
async def test_scoped_shared_key_test_sanitizes_provider_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError(f"stored provider token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.test_org_shared_key if scope == "org" else routes.test_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert raw_token not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        f"stored provider token at {raw_token}",
    )
