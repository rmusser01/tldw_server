from __future__ import annotations

import base64
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


def _gateway_spec(
    backend_id: str = "gateway:voice-lab",
    *,
    enabled: bool = True,
    allow_user_api_key: bool = True,
    api_key: str | None = "admin-secret",
    config_generation: str = "generation-one",
):
    return SimpleNamespace(
        backend_id=backend_id,
        enabled=enabled,
        allow_user_api_key=allow_user_api_key,
        api_key=api_key,
        config_generation=config_generation,
    )


@pytest.fixture
def gateway_byok_encryption(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"g"))
    reset_settings()


@pytest.mark.asyncio
async def test_resolve_byok_credentials_invalid_fields_returns_invalid(monkeypatch):
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

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key is None
    assert resolved.credential_fields == {}


@pytest.mark.asyncio
async def test_gateway_resolution_uses_only_user_key_and_opaque_scope(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    row = _encrypted_row(
        build_secret_payload(
            "user-secret",
            credential_fields={
                "base_url": "https://attacker.example/v1",
                "headers": {"X-Attacker": "yes"},
            },
        )
    )
    row.update(
        {
            "id": 17,
            "user_id": 404,
            "provider": "gateway:voice-lab",
            "metadata": {"base_url": "https://attacker.example/v1"},
            "updated_at": "2026-07-16T12:00:00+00:00",
        }
    )

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str):
            assert user_id == 404
            assert provider == "gateway:voice-lab"
            return row

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    resolved = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=404,
        gateway_spec=_gateway_spec(),
    )

    assert resolved.source == "user"
    assert resolved.api_key == "user-secret"
    assert resolved.credential_fields == {}
    assert resolved.app_config is None
    assert resolved.credential_scope_token
    assert "user-secret" not in resolved.credential_scope_token
    assert "404" not in resolved.credential_scope_token
    assert "voice-lab" not in resolved.credential_scope_token
    assert resolved.credential_scope_token not in repr(resolved)


@pytest.mark.asyncio
async def test_gateway_user_record_is_authoritative_and_never_falls_through_to_admin(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    row = _encrypted_row({"credential_fields": {"base_url": "https://legacy.invalid"}})
    row.update({"id": 18, "updated_at": "2026-07-16T12:00:00+00:00"})

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, _user_id: int, _provider: str):
            return row

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    resolved = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=5,
        gateway_spec=_gateway_spec(api_key="admin-must-not-be-used"),
    )

    assert resolved.source == "user"
    assert resolved.api_key is None
    assert resolved.credential_scope_token is None


@pytest.mark.asyncio
async def test_gateway_admin_scope_uses_backend_and_config_generation_only(
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    first = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(config_generation="generation-one"),
    )
    second = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(config_generation="generation-two"),
    )

    assert first.source == "server_default"
    assert first.api_key == "admin-secret"
    assert first.credential_scope_token
    assert first.credential_scope_token != second.credential_scope_token
    assert "admin-secret" not in first.credential_scope_token
    assert "voice-lab" not in first.credential_scope_token


@pytest.mark.asyncio
async def test_gateway_scope_changes_on_rotation_and_is_distinct_between_records(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    rows = {
        1: {
            **_encrypted_row(build_secret_payload("same-key")),
            "id": 101,
            "updated_at": "2026-07-16T12:00:00+00:00",
        },
        2: {
            **_encrypted_row(build_secret_payload("same-key")),
            "id": 202,
            "updated_at": "2026-07-16T12:00:00+00:00",
        },
    }

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, _provider: str):
            return rows[user_id]

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    first = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=1,
        gateway_spec=_gateway_spec(),
    )
    other_user = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=2,
        gateway_spec=_gateway_spec(),
    )
    rows[1]["encrypted_blob"] = _encrypted_row(
        build_secret_payload("rotated-key")
    )["encrypted_blob"]
    rows[1]["updated_at"] = "2026-07-16T12:01:00+00:00"
    rotated = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=1,
        gateway_spec=_gateway_spec(),
    )

    assert first.credential_scope_token != other_user.credential_scope_token
    assert first.credential_scope_token != rotated.credential_scope_token
    assert "101" not in first.credential_scope_token
    assert "202" not in other_user.credential_scope_token


@pytest.mark.asyncio
async def test_gateway_scope_ignores_usage_timestamps_but_changes_with_ciphertext(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    row = {
        **_encrypted_row(build_secret_payload("first-key")),
        "id": 707,
        "updated_at": "2026-07-16T12:00:00+00:00",
        "last_used_at": None,
    }

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, _user_id: int, _provider: str):
            return row

        async def touch_last_used(self, _user_id: int, _provider: str, used_at):
            row["last_used_at"] = used_at.isoformat()
            row["updated_at"] = used_at.isoformat()

    repo = _FakeUserRepo()

    async def _fake_get_user_repo():
        return repo

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    first = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=44,
        gateway_spec=_gateway_spec(),
    )
    await repo.touch_last_used(
        44,
        "gateway:voice-lab",
        datetime.now(timezone.utc),
    )
    after_touch = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=44,
        gateway_spec=_gateway_spec(),
    )

    row["encrypted_blob"] = _encrypted_row(build_secret_payload("rotated-key"))[
        "encrypted_blob"
    ]
    row["updated_at"] = "2026-07-16T12:05:00+00:00"
    after_rotation = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=44,
        gateway_spec=_gateway_spec(),
    )

    assert first.credential_scope_token == after_touch.credential_scope_token
    assert first.credential_scope_token != after_rotation.credential_scope_token
    assert "first-key" not in first.credential_scope_token
    assert "rotated-key" not in after_rotation.credential_scope_token
    assert "44" not in first.credential_scope_token


@pytest.mark.asyncio
async def test_disabled_or_removed_gateway_cannot_resolve_stored_or_admin_key(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _FailingRepo:
        async def fetch_secret_for_user(self, _user_id: int, _provider: str):
            raise AssertionError("disabled gateway must not read stored credentials")

    async def _fake_get_user_repo():
        return _FailingRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    disabled = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=1,
        gateway_spec=_gateway_spec(enabled=False),
    )
    monkeypatch.setattr(
        byok_runtime,
        "get_byok_gateway_spec",
        lambda _backend: None,
        raising=False,
    )
    removed = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:removed",
        user_id=1,
    )

    for resolved in (disabled, removed):
        assert resolved.source == "none"
        assert resolved.api_key is None
        assert resolved.credential_scope_token is None


@pytest.mark.asyncio
async def test_each_gateway_target_resolves_its_own_fresh_credential(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    calls: list[str] = []
    rows = {
        "gateway:first": {
            **_encrypted_row(build_secret_payload("first-key")),
            "id": 301,
            "updated_at": "2026-07-16T12:00:00+00:00",
        },
        "gateway:second": {
            **_encrypted_row(build_secret_payload("second-key")),
            "id": 302,
            "updated_at": "2026-07-16T12:00:00+00:00",
        },
    }

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, _user_id: int, provider: str):
            calls.append(provider)
            return rows[provider]

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    first = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:first",
        user_id=7,
        gateway_spec=_gateway_spec("gateway:first"),
    )
    second = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:second",
        user_id=7,
        gateway_spec=_gateway_spec("gateway:second"),
    )

    assert calls == ["gateway:first", "gateway:second"]
    assert first.api_key == "first-key"
    assert second.api_key == "second-key"


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
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=30)
                ).isoformat(),
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
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                ).isoformat(),
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
async def test_resolve_byok_credentials_v2_oauth_refresh_failure_without_api_key_fails_closed(monkeypatch):
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
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                ).isoformat(),
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

    resolved = await byok_runtime.resolve_byok_credentials("openai", user_id=1)

    assert resolved.source == "user"
    assert resolved.api_key is None
    assert resolved.auth_source is None
