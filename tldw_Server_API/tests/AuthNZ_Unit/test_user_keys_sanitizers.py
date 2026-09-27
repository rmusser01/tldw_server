import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import user_keys as routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    OpenAICredentialSourceSwitchRequest,
    OpenAIOAuthAuthorizeRequest,
    ProviderKeyTestRequest,
    SharedProviderKeyTestRequest,
    SharedProviderKeyUpsertRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError

pytestmark = pytest.mark.unit


class _UserRepo:
    async def fetch_secret_for_user(self, *_args):
        return {"encrypted_blob": "encrypted-user-provider-secret"}


class _OAuthStateRepo:
    async def consume_state(self, **_kwargs):
        return {
            "user_id": 7,
            "redirect_uri": "https://app.example.test/api/v1/users/keys/openai/oauth/callback",
            "pkce_verifier_encrypted": "encrypted-pkce-verifier",
            "return_path": "/settings",
        }


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["user"], permissions=[], is_admin=False)


def _text(*parts: str) -> str:
    return "".join(parts)


def _assert_detached_validation_error(exc: HTTPException, sentinel: str) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert sentinel not in repr(exc)


def _oauth_settings() -> SimpleNamespace:
    return SimpleNamespace(
        OPENAI_OAUTH_ENABLED=True,
        OPENAI_OAUTH_CLIENT_ID="client-id",
        OPENAI_OAUTH_CLIENT_SECRET=_text("client", "-secret"),
        OPENAI_OAUTH_AUTH_URL="https://auth.example.test/oauth/authorize",
        OPENAI_OAUTH_TOKEN_URL=_text("https://auth.example.test/oauth/", "token"),
        OPENAI_OAUTH_REDIRECT_URI="https://app.example.test/api/v1/users/keys/openai/oauth/callback",
        OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES=["/"],
    )


def _stored_payload(*_args, **_kwargs):
    return {"api_key": "sk-test", "credential_fields": {"base_url": "https://api.example.test"}}


async def _user_repo() -> _UserRepo:
    return _UserRepo()


async def _oauth_state_repo() -> _OAuthStateRepo:
    return _OAuthStateRepo()


def _openai_credential_payload() -> dict[str, object]:
    return {
        "credential_version": 2,
        "credentials": {
            "api_key": {"api_key": "sk-existing-api-key"},
            "oauth": {
                "access_token": "existing-access-token",
                "refresh_token": "existing-refresh-token",
                "token_type": "Bearer",
            },
        },
        "active_auth_source": "oauth",
    }


def _install_openai_repo_patches(monkeypatch) -> None:
    repo = _UserRepo()

    async def get_repo() -> _UserRepo:
        return repo

    @asynccontextmanager
    async def mutation_repo(**_kwargs):
        yield repo

    monkeypatch.setattr(routes, "_get_user_repo", get_repo)
    monkeypatch.setattr(routes, "_openai_mutation_repo", mutation_repo)
    monkeypatch.setattr(routes, "_extract_payload_from_row", lambda _row: _openai_credential_payload())


async def _pass_provider_test(**_kwargs) -> str:
    return "gpt-test"


async def _skip_oauth_audit(**_kwargs) -> None:
    return None


def _install_user_key_patches(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(routes, "is_trusted_base_url_request", lambda *_args, **_kwargs: True)


def _install_oauth_patches(monkeypatch) -> None:
    async def token_exchange(**_kwargs):
        return {
            "access_token": _text("oauth", "-access", "-token"),
            "refresh_token": _text("oauth", "-refresh", "-token"),
            "token_type": _text("Bear", "er"),
        }

    monkeypatch.setattr(routes, "_require_openai_oauth_settings", _oauth_settings)
    monkeypatch.setattr(routes, "_get_oauth_state_repo", _oauth_state_repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(
        routes,
        "decrypt_byok_payload",
        lambda *_args, **_kwargs: {
            "pkce_verifier": "verifier",
            "credential_fields": {"project_id": "proj_test"},
        },
    )
    monkeypatch.setattr(routes, "_openai_oauth_token_exchange", token_exchange)


@pytest.mark.parametrize(
    ("model_type", "payload"),
    [
        (
            UserProviderKeyUpsertRequest,
            {
                "provider": "openai",
                "api_key": "sk-test",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            ProviderKeyTestRequest,
            {"provider": "openai", "headers": {"X-Attacker": "yes"}},
        ),
        (
            SharedProviderKeyUpsertRequest,
            {
                "scope_type": "org",
                "scope_id": 7,
                "provider": "openai",
                "api_key": "sk-test",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            SharedProviderKeyTestRequest,
            {
                "scope_type": "org",
                "scope_id": 7,
                "provider": "openai",
                "modle": "misspelled-model",
            },
        ),
    ],
)
def test_provider_key_mutation_and_test_models_forbid_unknown_fields(
    model_type,
    payload,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_type.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_name",
    [
        "upsert",
        "list",
        "test",
        "oauth_callback",
        "oauth_status",
        "oauth_refresh",
        "oauth_disconnect",
        "source_switch",
        "delete",
    ],
)
async def test_user_key_routes_map_alias_conflicts_to_detached_409(
    monkeypatch,
    route_name: str,
) -> None:
    """Every user credential read/mutation exposes one bounded conflict contract."""
    sentinel = f"sk-alias-conflict-{route_name}-/private/provider-row.json"

    async def conflict_repo():
        raise ProviderCredentialAliasConflictError(sentinel)

    _install_user_key_patches(monkeypatch)
    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_user_repo", conflict_repo)
    monkeypatch.setattr(routes, "resolve_byok_allowlist", lambda: {"openai"})
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _pass_provider_test)
    monkeypatch.setattr(routes, "_emit_openai_oauth_audit_event", _skip_oauth_audit)
    principal = _principal()
    request = SimpleNamespace()

    async def invoke():
        if route_name == "upsert":
            return await routes.upsert_user_provider_key(
                UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
                request=request,
                principal=principal,
            )
        if route_name == "list":
            return await routes.list_user_provider_keys(request=request, principal=principal)
        if route_name == "test":
            return await routes.test_user_provider_key(
                ProviderKeyTestRequest(provider="openai"),
                request=request,
                principal=principal,
            )
        if route_name == "oauth_callback":
            return await routes.callback_openai_oauth(
                request=request,
                code="oauth-code",
                state="oauth-state",
            )
        if route_name == "oauth_status":
            return await routes.openai_oauth_status(principal=principal)
        if route_name == "oauth_refresh":
            return await routes.refresh_openai_oauth(request=request, principal=principal)
        if route_name == "oauth_disconnect":
            return await routes.disconnect_openai_oauth(request=request, principal=principal)
        if route_name == "source_switch":
            return await routes.switch_openai_credential_source(
                OpenAICredentialSourceSwitchRequest(auth_source="api_key"),
                request=request,
                principal=principal,
            )
        return await routes.delete_user_provider_key("openai", principal=principal)

    with pytest.raises(HTTPException) as exc_info:
        await invoke()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Conflicting provider credential aliases"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_user_key_alias_conflicts_share_bounded_409_contract(
    monkeypatch,
) -> None:
    """Concurrent reads cannot bypass or cross-contaminate the conflict boundary."""
    sentinel = "sk-concurrent-alias-conflict-/private/provider-row.json"
    both_waiting = asyncio.Event()
    release = asyncio.Event()

    class ConflictRepo:
        calls = 0

        async def fetch_secret_for_user(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                both_waiting.set()
            await release.wait()
            raise ProviderCredentialAliasConflictError(sentinel)

    repo = ConflictRepo()

    async def get_repo():
        return repo

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_user_repo", get_repo)
    request = SimpleNamespace()
    principal = _principal()
    tasks = [
        asyncio.create_task(
            routes.test_user_provider_key(
                ProviderKeyTestRequest(provider="openai"),
                request=request,
                principal=principal,
            )
        )
        for _ in range(2)
    ]
    try:
        await asyncio.wait_for(both_waiting.wait(), timeout=1.0)
        assert all(not task.done() for task in tasks)
    finally:
        release.set()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert repo.calls == 2
    assert all(isinstance(result, HTTPException) for result in results)
    for result in results:
        assert isinstance(result, HTTPException)
        assert result.status_code == 409
        assert result.detail == "Conflicting provider credential aliases"
        _assert_detached_validation_error(result, sentinel)


@pytest.mark.asyncio
async def test_user_key_upsert_sanitizes_credential_validation(monkeypatch) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError("user credential token at /private/user-key-credentials.json")

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    with pytest.raises(HTTPException) as exc_info:
        await routes.upsert_user_provider_key(
            UserProviderKeyUpsertRequest(
                provider="openai",
                api_key="sk-test",
                credential_fields={"base_url": "https://api.example.test"},
            ),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "/private/user-key-credentials.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "user credential token at /private/user-key-credentials.json",
    )


@pytest.mark.asyncio
async def test_user_key_upsert_sanitizes_provider_validation(monkeypatch) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError("user provider token at /private/user-key-provider.json")

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await routes.upsert_user_provider_key(
            UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "/private/user-key-provider.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "user provider token at /private/user-key-provider.json",
    )


@pytest.mark.asyncio
async def test_user_key_upsert_detaches_unexpected_provider_validation_failure(
    monkeypatch,
) -> None:
    sentinel = "sk-user-runtime-error-/private/user-provider-runtime.json"

    async def fail_provider_test(**_kwargs):
        raise RuntimeError(sentinel)

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await routes.upsert_user_provider_key(
            UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Provider test call failed"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_user_key_upsert_sanitizes_chat_provider_error_context(monkeypatch) -> None:
    sentinel = "sk-user-upstream-secret-/private/user-provider-body.json"

    async def fail_provider_test(**_kwargs):
        raise ChatProviderError(
            message=f"hostile upstream body {sentinel}",
            status_code=502,
            provider="openai",
            details={"endpoint": f"https://provider.invalid/{sentinel}"},
        )

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await routes.upsert_user_provider_key(
            UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "The chat service provider is currently unavailable."
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_user_key_test_sanitizes_stored_credential_validation(monkeypatch) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError("stored credential token at /private/user-key-stored-credentials.json")

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_user_repo", _user_repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    with pytest.raises(HTTPException) as exc_info:
        await routes.test_user_provider_key(
            ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert "/private/user-key-stored-credentials.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "stored credential token at /private/user-key-stored-credentials.json",
    )


@pytest.mark.asyncio
async def test_user_key_test_sanitizes_provider_validation(monkeypatch) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError("stored provider token at /private/user-key-stored-provider.json")

    _install_user_key_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_user_repo", _user_repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await routes.test_user_provider_key(
            ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "/private/user-key-stored-provider.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "stored provider token at /private/user-key-stored-provider.json",
    )


@pytest.mark.asyncio
async def test_openai_oauth_authorize_sanitizes_credential_validation(monkeypatch) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError("oauth authorize token at /private/oauth-authorize.json")

    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    with pytest.raises(HTTPException) as exc_info:
        await routes.authorize_openai_oauth(
            request=SimpleNamespace(),
            payload=OpenAIOAuthAuthorizeRequest(credential_fields={"project_id": "proj_test"}),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid OpenAI OAuth credential fields"
    assert "/private/oauth-authorize.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "oauth authorize token at /private/oauth-authorize.json",
    )


@pytest.mark.asyncio
async def test_openai_oauth_callback_sanitizes_state_credential_validation(monkeypatch) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError("oauth state token at /private/oauth-state.json")

    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    with pytest.raises(HTTPException) as exc_info:
        await routes.callback_openai_oauth(
            request=SimpleNamespace(),
            code="oauth-code",
            state="oauth-state",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid OpenAI OAuth credential fields"
    assert "/private/oauth-state.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "oauth state token at /private/oauth-state.json",
    )


@pytest.mark.asyncio
async def test_openai_oauth_callback_sanitizes_provider_validation(monkeypatch) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError("oauth provider token at /private/oauth-provider.json")

    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    with pytest.raises(HTTPException) as exc_info:
        await routes.callback_openai_oauth(
            request=SimpleNamespace(),
            code="oauth-code",
            state="oauth-state",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert "/private/oauth-provider.json" not in exc_info.value.detail
    _assert_detached_validation_error(
        exc_info.value,
        "oauth provider token at /private/oauth-provider.json",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_user_key_upsert_detaches_encryption_failure(monkeypatch, provider: str) -> None:
    sentinel = f"sk-{provider}-encrypt-failure-/private/{provider}-credential.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_user_key_patches(monkeypatch)
    _install_openai_repo_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _pass_provider_test)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.upsert_user_provider_key(
            UserProviderKeyUpsertRequest(provider=provider, api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_oauth_authorize_detaches_state_encryption_failure(monkeypatch) -> None:
    sentinel = "pkce-verifier-encrypt-failure-/private/oauth-state.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.authorize_openai_oauth(
            request=SimpleNamespace(),
            payload=OpenAIOAuthAuthorizeRequest(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_oauth_callback_detaches_state_decryption_failure(monkeypatch) -> None:
    sentinel = "pkce-verifier-decrypt-failure-/private/oauth-state.json"

    def fail_decrypt(_envelope):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    monkeypatch.setattr(routes, "decrypt_byok_payload", fail_decrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.callback_openai_oauth(
            request=SimpleNamespace(),
            code="oauth-code",
            state="oauth-state",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "OAuth state verifier could not be decrypted"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_oauth_callback_detaches_token_encryption_failure(monkeypatch) -> None:
    sentinel = "oauth-token-encrypt-failure-/private/oauth-token.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    _install_openai_repo_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", _pass_provider_test)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.callback_openai_oauth(
            request=SimpleNamespace(),
            code="oauth-code",
            state="oauth-state",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_oauth_refresh_detaches_token_encryption_failure(monkeypatch) -> None:
    sentinel = "oauth-refresh-encrypt-failure-/private/oauth-refresh-token.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    _install_openai_repo_patches(monkeypatch)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)
    monkeypatch.setattr(routes, "_emit_openai_oauth_audit_event", _skip_oauth_audit)

    with pytest.raises(HTTPException) as exc_info:
        await routes.refresh_openai_oauth(
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_oauth_disconnect_detaches_fallback_encryption_failure(monkeypatch) -> None:
    sentinel = "oauth-disconnect-encrypt-failure-/private/oauth-fallback-key.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    _install_openai_repo_patches(monkeypatch)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.disconnect_openai_oauth(
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_openai_source_switch_detaches_encryption_failure(monkeypatch) -> None:
    sentinel = "oauth-source-switch-encrypt-failure-/private/oauth-credentials.json"

    def fail_encrypt(_payload):
        raise ValueError(sentinel)

    _install_oauth_patches(monkeypatch)
    _install_openai_repo_patches(monkeypatch)
    monkeypatch.setattr(routes, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(HTTPException) as exc_info:
        await routes.switch_openai_credential_source(
            OpenAICredentialSourceSwitchRequest(auth_source="api_key"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "BYOK encryption is not configured"
    _assert_detached_validation_error(exc_info.value, sentinel)


def test_openai_oauth_counter_failure_log_is_sanitized(monkeypatch) -> None:
    def fail_increment_counter(*_args, **_kwargs):
        raise RuntimeError("oauth counter failed at /private/openai-oauth-metrics.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "increment_counter", fail_increment_counter)
    monkeypatch.setattr(routes, "logger", logger_stub)

    routes._record_openai_oauth_counter(
        "openai.oauth.secret.metric",
        labels={"reason": "secret-reason"},
    )

    assert logger_stub.debugs == ["OpenAI OAuth metric emission failed"]
    assert "openai.oauth.secret.metric" not in str(logger_stub.debugs)
    assert "secret-reason" not in str(logger_stub.debugs)
    assert "oauth counter failed" not in str(logger_stub.debugs)
    assert "/private/openai-oauth-metrics.db" not in str(logger_stub.debugs)


def test_openai_oauth_histogram_failure_log_is_sanitized(monkeypatch) -> None:
    def fail_observe_histogram(*_args, **_kwargs):
        raise RuntimeError("oauth histogram failed at /private/openai-oauth-metrics.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "observe_histogram", fail_observe_histogram)
    monkeypatch.setattr(routes, "logger", logger_stub)

    routes._record_openai_oauth_histogram(
        "openai.oauth.secret.duration",
        value=1.5,
        labels={"outcome": "secret-outcome"},
    )

    assert logger_stub.debugs == ["OpenAI OAuth histogram emission failed"]
    assert "openai.oauth.secret.duration" not in str(logger_stub.debugs)
    assert "secret-outcome" not in str(logger_stub.debugs)
    assert "oauth histogram failed" not in str(logger_stub.debugs)
    assert "/private/openai-oauth-metrics.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_openai_oauth_audit_failure_log_is_sanitized(monkeypatch) -> None:
    async def fail_audit_service(_user_id):
        raise RuntimeError("oauth audit failed at /private/openai-oauth-audit.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "get_or_create_audit_service_for_user_id_optional", fail_audit_service)
    monkeypatch.setattr(routes, "logger", logger_stub)

    await routes._emit_openai_oauth_audit_event(
        user_id=7,
        action="openai.oauth.secret.action",
    )

    assert logger_stub.debugs == ["OpenAI OAuth audit emission skipped"]
    assert "openai.oauth.secret.action" not in str(logger_stub.debugs)
    assert "oauth audit failed" not in str(logger_stub.debugs)
    assert "/private/openai-oauth-audit.db" not in str(logger_stub.debugs)


def test_extract_payload_from_row_decrypt_failure_log_is_sanitized(monkeypatch) -> None:
    def fail_decrypt_byok_payload(_envelope):
        raise RuntimeError("BYOK decrypt failed at /private/user-provider-secret.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {"ciphertext": "encrypted-secret"})
    monkeypatch.setattr(routes, "decrypt_byok_payload", fail_decrypt_byok_payload)
    monkeypatch.setattr(routes, "logger", logger_stub)

    payload = routes._extract_payload_from_row({"encrypted_blob": "encrypted-secret"})

    assert payload is None
    assert logger_stub.warnings == ["Failed to decrypt BYOK payload for provider row"]
    assert "encrypted-secret" not in str(logger_stub.warnings)
    assert "BYOK decrypt failed" not in str(logger_stub.warnings)
    assert "/private/user-provider-secret.db" not in str(logger_stub.warnings)
