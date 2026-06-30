from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import user_keys as routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    OpenAIOAuthAuthorizeRequest,
    ProviderKeyTestRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

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
