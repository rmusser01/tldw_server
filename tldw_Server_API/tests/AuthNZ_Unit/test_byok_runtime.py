from __future__ import annotations

import base64
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    loads_envelope,
)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _encrypted_row(payload: dict) -> dict:
    envelope = encrypt_byok_payload(payload)
    return {"encrypted_blob": dumps_envelope(envelope), "last_used_at": None}


def _decrypted_payload_from_row(row: dict) -> dict:
    return decrypt_byok_payload(loads_envelope(row["encrypted_blob"]))


@pytest.mark.asyncio
async def test_resolve_byok_credentials_invalid_fields_raise_typed_failure(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    payload = build_secret_payload("sk-test", credential_fields={"bad_field": "nope"})
    envelope = encrypt_byok_payload(payload)
    row = {"encrypted_blob": dumps_envelope(envelope), "last_used_at": None}

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials("OpenAI", user_id=1)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert vars(exc_info.value) == {
        "code": "invalid_provider_credentials",
        "provider": "openai",
    }
    assert "bad_field" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_resolve_byok_credentials_v2_oauth_active_uses_access_token(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {"access_token": "oauth-access-token-123"},
            "api_key": {"api_key": "sk-api-fallback-123"},
        },
    }
    row = _encrypted_row(payload)

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key == "oauth-access-token-123"
    assert resolved.auth_source == "oauth"


@pytest.mark.asyncio
async def test_resolve_byok_credentials_v2_missing_oauth_token_falls_back_to_api_key(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {"access_token": ""},
            "api_key": {"api_key": "sk-api-key-usable-456"},
        },
    }
    row = _encrypted_row(payload)

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key == "sk-api-key-usable-456"
    assert resolved.auth_source == "api_key"


@pytest.mark.asyncio
async def test_resolve_byok_credentials_v2_oauth_refresh_success_updates_payload(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "stale-access-token",
                "refresh_token": "refresh-token-123",
                "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
            },
            "api_key": {"api_key": "sk-api-fallback-123"},
        },
    }
    row = _encrypted_row(payload)
    row["metadata"] = None
    row["key_hint"] = "oauth"

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

        async def upsert_secret(
            self,
            *,
            user_id: int,
            provider: str,
            encrypted_blob: str,
            key_hint: str | None,
            metadata,
            updated_at: datetime,
            created_by: int | None = None,
            updated_by: int | None = None,
        ):
            row["encrypted_blob"] = encrypted_blob
            row["key_hint"] = key_hint
            row["metadata"] = metadata
            row["updated_at"] = updated_at
            return {"updated_at": updated_at}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "inference",
            }

        async def aclose(self):
            return None

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _fake_afetch(*_args, **_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _fake_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key == "new-access-token"
    assert resolved.auth_source == "oauth"

    stored_payload = _decrypted_payload_from_row(row)
    assert stored_payload["active_auth_source"] == "oauth"
    assert stored_payload["credentials"]["oauth"]["access_token"] == "new-access-token"
    assert stored_payload["credentials"]["oauth"]["refresh_token"] == "new-refresh-token"


@pytest.mark.asyncio
async def test_resolve_byok_credentials_v2_oauth_refresh_failure_falls_back_to_api_key(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "expired-access-token",
                "refresh_token": "refresh-token-123",
                "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
            },
            "api_key": {"api_key": "sk-api-fallback-xyz"},
        },
    }
    row = _encrypted_row(payload)
    row["metadata"] = None
    row["key_hint"] = "oauth"

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

        async def upsert_secret(
            self,
            *,
            user_id: int,
            provider: str,
            encrypted_blob: str,
            key_hint: str | None,
            metadata,
            updated_at: datetime,
            created_by: int | None = None,
            updated_by: int | None = None,
        ):
            row["encrypted_blob"] = encrypted_blob
            row["key_hint"] = key_hint
            row["metadata"] = metadata
            row["updated_at"] = updated_at
            return {"updated_at": updated_at}

    class _FakeResponse:
        status_code = 400

        def json(self):
            return {"error": "invalid_grant"}

        async def aclose(self):
            return None

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _fake_afetch(*_args, **_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _fake_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key == "sk-api-fallback-xyz"
    assert resolved.auth_source == "api_key"

    stored_payload = _decrypted_payload_from_row(row)
    assert stored_payload["active_auth_source"] == "api_key"
    assert stored_payload["credentials"]["api_key"]["api_key"] == "sk-api-fallback-xyz"


@pytest.mark.asyncio
async def test_resolve_byok_credentials_v2_oauth_refresh_failure_without_api_key_raises_typed_failure(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "expired-access-token",
                "refresh_token": "refresh-token-123",
                "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
            },
        },
    }
    row = _encrypted_row(payload)
    row["metadata"] = None
    row["key_hint"] = "oauth"

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

    class _FakeResponse:
        status_code = 400

        def json(self):
            return {"error": "invalid_grant"}

        async def aclose(self):
            return None

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _fake_afetch(*_args, **_kwargs):
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _fake_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert "refresh-token-123" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_decrypt_failure_does_not_advance_to_server_default(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return {"encrypted_blob": "not-an-envelope", "last_used_at": None}

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    fallback_calls: list[str] = []

    def _fallback(provider: str) -> str:
        fallback_calls.append(provider)
        return "server-secret-must-not-be-used"

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            fallback_resolver=_fallback,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert fallback_calls == []


@pytest.mark.asyncio
async def test_user_repository_outage_raises_sanitized_typed_failure(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    async def _fake_get_user_repo():
        raise OSError("database unavailable with secret=sk-do-not-leak")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials("OpenAI", user_id=1)

    assert exc_info.value.code == "credential_store_unavailable"
    assert exc_info.value.provider == "openai"
    assert vars(exc_info.value) == {
        "code": "credential_store_unavailable",
        "provider": "openai",
    }
    assert "sk-do-not-leak" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope_type", "team_ids", "org_ids"),
    [
        ("team", [7], []),
        ("org", [], [8]),
    ],
)
async def test_shared_repository_outage_does_not_advance_precedence(
    monkeypatch,
    scope_type,
    team_ids,
    org_ids,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    class _FakeSharedRepo:
        async def fetch_secret(self, requested_scope: str, scope_id: int, provider: str):
            assert requested_scope == scope_type
            raise ConnectionError("shared store outage with token=do-not-leak")

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _fake_get_org_repo():
        return _FakeSharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _fake_get_org_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            team_ids=team_ids,
            org_ids=org_ids,
            fallback_resolver=lambda _provider: "server-secret-must-not-be-used",
        )

    assert exc_info.value.code == "credential_store_unavailable"
    assert exc_info.value.provider == "openai"
    assert "do-not-leak" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_membership_lookup_outage_raises_sanitized_typed_failure(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _fail_membership_lookup(user_id: int):
        raise TimeoutError("membership backend timed out with cookie=do-not-leak")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "list_memberships_for_user", _fail_membership_lookup)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert exc_info.value.code == "credential_store_unavailable"
    assert exc_info.value.provider == "openai"
    assert "do-not-leak" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("active_field", "active_id", "team_ids", "org_ids"),
    [
        ("active_team_id", 99, [7], []),
        ("active_org_id", 99, [], [8]),
    ],
)
async def test_invalid_active_scope_raises_revoked_failure(
    monkeypatch,
    active_field,
    active_id,
    team_ids,
    org_ids,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    state = SimpleNamespace(active_team_id=None, active_org_id=None)
    setattr(state, active_field, active_id)
    request = SimpleNamespace(state=state)

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    error_type = getattr(byok_runtime, "ByokResolutionError", RuntimeError)
    with pytest.raises(error_type) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            request=request,
            team_ids=team_ids,
            org_ids=org_ids,
        )

    assert exc_info.value.code == "credential_scope_revoked"
    assert exc_info.value.provider == "openai"


def test_resolved_credentials_repr_redacts_all_sensitive_fields():
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    async def _touch_secret():
        return None

    resolved = byok_runtime.ResolvedByokCredentials(
        provider="openai",
        api_key="sk-repr-secret",
        app_config={
            "openai_api": {
                "api_key": "config-secret",
                "model": "secret-looking-model",
            }
        },
        credential_fields={"base_url": "https://credential-field.example"},
        source="user",
        allowlisted=True,
        auth_source="oauth",
        _touch_cb=_touch_secret,
    )

    rendered = repr(resolved)

    for hidden in (
        "sk-repr-secret",
        "config-secret",
        "secret-looking-model",
        "credential-field.example",
        "_touch_secret",
        "api_key",
        "app_config",
        "credential_fields",
        "_touch_cb",
    ):
        assert hidden not in rendered
    assert "provider='openai'" in rendered
    assert "source='user'" in rendered


def test_build_app_config_is_provider_scoped_and_scrubs_secrets(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    source_proxy_allowlist = ["proxy.example"]
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "openai_api": {
                "api_key": "server-openai-secret",
                "access_token": "server-openai-token",
                "client_secret": "oauth-client-secret",
                "Authorization": "Bearer server-openai-secret",
                "model": "gpt-safe",
                "api_timeout": 12,
                "api_retries": 2,
                "api_retry_delay": 0.25,
                "api_base_url": "https://server-openai.example/v1",
                "organization_id": "org-safe",
                "project_id": "project-safe",
                "temperature": 0.2,
            },
            "anthropic_api": {
                "api_key": "unrelated-secret",
                "model": "unrelated-model",
            },
            "HTTP": {
                "connect_timeout": 5,
                "read_timeout": 30,
                "proxy_allowlist": source_proxy_allowlist,
                "authorization": "Basic do-not-copy",
                "cookie": "do-not-copy",
            },
            "Egress": {
                "egress_allowlist": ["api.openai.com"],
                "allowed_ports": [443],
                "block_private": True,
                "client_secret": "do-not-copy",
            },
            "database": {"password": "do-not-copy"},
        },
    )

    app_config = byok_runtime._build_app_config(
        "openai",
        {
            "base_url": "https://byok-openai.example/v1",
            "org_id": "byok-org",
            "project_id": "byok-project",
        },
    )

    assert app_config == {
        "openai_api": {
            "model": "gpt-safe",
            "api_timeout": 12,
            "api_retries": 2,
            "api_retry_delay": 0.25,
            "api_base_url": "https://byok-openai.example/v1",
            "organization_id": "org-safe",
            "org_id": "byok-org",
            "project_id": "byok-project",
        },
        "HTTP": {
            "connect_timeout": 5,
            "read_timeout": 30,
            "proxy_allowlist": ["proxy.example"],
        },
        "Egress": {
            "egress_allowlist": ["api.openai.com"],
            "allowed_ports": [443],
            "block_private": True,
        },
    }
    app_config["HTTP"]["proxy_allowlist"].append("mutated.example")
    assert source_proxy_allowlist == ["proxy.example"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "server_key", "expected_source", "expected_status", "section", "endpoint_key"),
    [
        ("openai", "sk-server-only", "server_default", "RESOLVED", "openai_api", "api_base_url"),
        ("ollama", None, "none", "ABSENT", "ollama_api", "api_url"),
        (
            "custom-openai-api-3",
            "sk-custom-server",
            "server_default",
            "RESOLVED",
            "custom_openai_api_3",
            "api_base_url",
        ),
    ],
)
async def test_fallback_results_keep_only_selected_provider_config(
    monkeypatch,
    provider,
    server_key,
    expected_source,
    expected_status,
    section,
    endpoint_key,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            section: {
                "api_key": "config-secret",
                "model": "selected-model",
                endpoint_key: "http://selected-provider.example/v1",
                "api_timeout": 10,
            },
            "anthropic_api": {"api_key": "unrelated-secret", "model": "unrelated-model"},
        },
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        provider,
        user_id=1,
        fallback_resolver=lambda _provider: server_key,
    )

    assert resolved.source == expected_source
    assert resolved.status == expected_status
    assert resolved.api_key == server_key
    assert resolved.app_config == {
        section: {
            "model": "selected-model",
            endpoint_key: "http://selected-provider.example/v1",
            "api_timeout": 10,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "store_unavailable"),
    [
        (sqlite3.OperationalError, True),
        (sqlite3.InterfaceError, True),
        (sqlite3.ProgrammingError, False),
    ],
)
@pytest.mark.parametrize("failure_site", ["user_repository", "membership"])
async def test_sqlite_errors_respect_operational_boundary(
    monkeypatch,
    error_type,
    store_unavailable,
    failure_site,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _AbsentUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    async def _get_user_repo():
        if failure_site == "user_repository":
            raise error_type("sqlite failure with api_key=do-not-leak")
        return _AbsentUserRepo()

    async def _list_memberships(user_id: int):
        raise error_type("sqlite membership failure with token=do-not-leak")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "list_memberships_for_user", _list_memberships)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    if store_unavailable:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials("OpenAI", user_id=1)
        assert exc_info.value.code == "credential_store_unavailable"
        assert exc_info.value.provider == "openai"
        assert "do-not-leak" not in str(exc_info.value)
    else:
        with pytest.raises(error_type, match="do-not-leak"):
            await byok_runtime.resolve_byok_credentials("OpenAI", user_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "store_unavailable"),
    [
        pytest.param("ConnectionPoolExhaustedError", True, id="pool-exhausted"),
        pytest.param("DatabaseLockError", True, id="database-locked"),
        pytest.param("DatabaseError", False, id="unbounded-database-error"),
    ],
)
@pytest.mark.parametrize("failure_site", ["user_repository", "membership"])
async def test_authnz_database_errors_respect_operational_boundary(
    monkeypatch,
    error_type,
    store_unavailable,
    failure_site,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ import exceptions as authnz_exceptions

    exception_type = getattr(authnz_exceptions, error_type)

    def _error():
        if error_type == "DatabaseError":
            return exception_type("unbounded database error with secret=do-not-leak")
        return exception_type()

    class _AbsentUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    async def _get_user_repo():
        if failure_site == "user_repository":
            raise _error()
        return _AbsentUserRepo()

    async def _list_memberships(user_id: int):
        raise _error()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "list_memberships_for_user", _list_memberships)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    if store_unavailable:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)
        assert exc_info.value.code == "credential_store_unavailable"
        assert exc_info.value.provider == "openai"
    else:
        with pytest.raises(exception_type, match="do-not-leak"):
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_name", "store_unavailable"),
    [
        ("OperationalError", True),
        ("InterfaceError", True),
        ("ProgrammingError", False),
    ],
)
async def test_aiosqlite_errors_respect_operational_boundary(
    monkeypatch,
    error_name,
    store_unavailable,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    module = pytest.importorskip("aiosqlite")
    error_type = getattr(module, error_name, None)
    if error_type is None:
        pytest.skip(f"installed aiosqlite has no {error_name}")
    error = error_type("aiosqlite failure with secret=do-not-leak")

    async def _get_user_repo():
        raise error

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    if store_unavailable:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)
        assert exc_info.value.code == "credential_store_unavailable"
        assert "do-not-leak" not in str(exc_info.value)
    else:
        with pytest.raises(error_type, match="do-not-leak"):
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_name", "store_unavailable"),
    [
        ("InterfaceError", True),
        ("PostgresConnectionError", True),
        ("CannotConnectNowError", True),
        ("TooManyConnectionsError", True),
        ("PostgresSyntaxError", False),
    ],
)
async def test_asyncpg_errors_respect_operational_boundary(
    monkeypatch,
    error_name,
    store_unavailable,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    module = pytest.importorskip("asyncpg")
    error_type = getattr(module, error_name)
    error = error_type("asyncpg failure with secret=do-not-leak")

    async def _get_user_repo():
        raise error

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    if store_unavailable:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)
        assert exc_info.value.code == "credential_store_unavailable"
        assert "do-not-leak" not in str(exc_info.value)
    else:
        with pytest.raises(error_type, match="do-not-leak"):
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)


@pytest.mark.asyncio
async def test_programmer_error_is_not_misclassified_as_store_outage(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    async def _get_user_repo():
        raise AssertionError("programmer error")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    with pytest.raises(AssertionError, match="programmer error"):
        await byok_runtime.resolve_byok_credentials("openai", user_id=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["user", "team", "org"])
@pytest.mark.parametrize("openai_v2", [False, True])
@pytest.mark.parametrize("invalid_fields", [[], ["bad"], "", "bad"])
async def test_present_non_dict_credential_fields_fail_closed(
    monkeypatch,
    scope,
    openai_v2,
    invalid_fields,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    if openai_v2:
        payload = {
            "credential_version": 2,
            "active_auth_source": "api_key",
            "credentials": {"api_key": {"api_key": "sk-v2-test"}},
            "credential_fields": invalid_fields,
        }
    else:
        payload = {
            "api_key": "sk-legacy-test",
            "credential_fields": invalid_fields,
        }
    row = _encrypted_row(payload)

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row if scope == "user" else None

    class _FakeSharedRepo:
        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            assert scope_type == scope
            return row

    async def _get_user_repo():
        return _FakeUserRepo()

    async def _get_org_repo():
        return _FakeSharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _get_org_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    team_ids = [7] if scope == "team" else []
    org_ids = [8] if scope == "org" else []
    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            team_ids=team_ids,
            org_ids=org_ids,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error_type",
    [
        pytest.param("RetryExhaustedError", id="retry-exhausted"),
        pytest.param("NetworkError", id="network"),
        pytest.param("EgressPolicyError", id="egress-policy"),
    ],
)
@pytest.mark.parametrize("has_api_key_fallback", [True, False])
async def test_openai_oauth_transport_errors_use_api_key_or_fail_typed(
    monkeypatch,
    transport_error_type,
    has_api_key_fallback,
):
    from tldw_Server_API.app.core import exceptions as core_exceptions
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    credentials = {
        "oauth": {
            "access_token": "expired-access-token",
            "refresh_token": "refresh-token-transport-test",
            "expires_at": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        }
    }
    if has_api_key_fallback:
        credentials["api_key"] = {"api_key": "sk-api-transport-fallback"}
    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": credentials,
        }
    )
    row["metadata"] = None
    row["key_hint"] = "oauth"

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return row

        async def upsert_secret(self, **kwargs):
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            return {"updated_at": kwargs["updated_at"]}

    async def _get_user_repo():
        return _FakeUserRepo()

    async def _fail_transport(*args, **kwargs):
        error_type = getattr(core_exceptions, transport_error_type)
        raise error_type("transport failure with token=do-not-leak")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _fail_transport)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    if has_api_key_fallback:
        resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)
        assert resolved.api_key == "sk-api-transport-fallback"
        assert resolved.auth_source == "api_key"
    else:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials("openai", user_id=1)
        assert exc_info.value.code == "invalid_provider_credentials"
        assert "do-not-leak" not in str(exc_info.value)
