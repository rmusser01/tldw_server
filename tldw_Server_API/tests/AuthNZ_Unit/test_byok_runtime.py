from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from loguru import logger

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


def _server_fallback_value(**kwargs):
    """Construct one production structured server fallback."""
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ServerFallbackCredentials

    return ServerFallbackCredentials(**kwargs)


def _capture_real_openai_adapter_headers(monkeypatch):
    """Install a fake transport beneath the real OpenAI adapter."""
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter as adapter_module

    captured_headers: list[dict[str, str]] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _url, *, headers, json):
            del json
            captured_headers.append(dict(headers))
            return FakeResponse()

    monkeypatch.setattr(adapter_module, "http_client_factory", lambda **_kwargs: FakeClient())
    return adapter_module.OpenAIAdapter(), captured_headers


def test_extract_payload_detaches_secret_bearing_decrypt_failure(monkeypatch):
    """Stored credential decryption failures cannot remain on typed errors."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    secret = "sk-decrypt-cause-/private/provider-envelope.json"
    monkeypatch.setattr(byok_runtime, "loads_envelope", lambda _blob: {})

    def fail_decrypt(_envelope):
        raise ValueError(secret)

    monkeypatch.setattr(byok_runtime, "decrypt_byok_payload", fail_decrypt)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        byok_runtime._extract_payload(
            {"encrypted_blob": "encrypted-provider-credential"},
            "openai",
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)


def _assert_opaque_scope_token(token: str | None) -> None:
    assert token is not None
    assert len(token) == 64
    assert token == token.lower()
    int(token, 16)


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
async def test_persist_payload_detaches_secret_bearing_encrypt_failure(monkeypatch):
    """Credential re-encryption failures cannot remain on typed errors."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    secret = "sk-encrypt-cause-/private/provider-payload.json"

    def fail_encrypt(_payload):
        raise ValueError(secret)

    monkeypatch.setattr(byok_runtime, "encrypt_byok_payload", fail_encrypt)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime._persist_user_payload_update(
            repo=object(),  # type: ignore[arg-type]
            provider="openai",
            user_id=7,
            row={"encrypted_blob": "old-encrypted-provider-credential"},
            payload={"api_key": "sk-provider-value"},
            updated_at=datetime.now(timezone.utc),
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)


def test_server_fallback_repr_redacts_all_credential_material():
    """Fallback diagnostics cannot expose keys, endpoints, or auth metadata."""
    fallback = _server_fallback_value(
        api_key="server-key-canary",
        credential_fields={"base_url": "https://credential-endpoint-canary.example/v1"},
        auth_source="auth-source-canary",
        app_config={"openai_api": {"project_id": "project-canary"}},
    )

    rendered = repr(fallback)

    assert rendered == "ServerFallbackCredentials(credentials=[REDACTED])"
    assert "canary" not in rendered


def test_openai_user_resolution_repr_redacts_decrypted_tokens():
    """Internal OAuth resolution diagnostics cannot render decrypted secrets."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    resolution = byok_runtime._OpenAIUserResolution(
        payload={"credentials": {"oauth": {"access_token": "access-canary"}}},
        api_key="api-key-canary",
        auth_source="oauth",
        fail_closed=False,
        credential_generation="generation-canary",
    )

    rendered = repr(resolution)

    assert rendered == "_OpenAIUserResolution(credentials=[REDACTED])"
    assert "canary" not in rendered


@pytest.mark.asyncio
async def test_explicit_server_snapshot_rejects_dynamic_fallback_resolver():
    """A caller cannot reintroduce a second live config generation."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    fallback_called = False

    def dynamic_fallback(_provider):
        nonlocal fallback_called
        fallback_called = True
        return "sk-dynamic-generation-b"

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=None,
            server_config_snapshot={
                "openai_api": {"api_key": "sk-snapshot-generation-a"}
            },
            fallback_resolver=dynamic_fallback,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert fallback_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "team", "org", "absent"])
async def test_byok_resolution_freezes_one_server_config_generation(
    monkeypatch,
    source,
):
    """A→B rotation cannot pair credentials with B's endpoint/model config."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()
    row = _encrypted_row(build_secret_payload("sk-byok-generation-a"))
    row.update({"revoked_at": None, "last_used_at": None})
    live_config = {
        "anthropic_api": {
            "api_key": "sk-static-generation-a",
            "api_url": "https://generation-a.example/v1",
            "model": "model-a",
        }
    }
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    class UserRepo:
        calls = 0

        async def fetch_secret_for_user(
            self,
            _user_id,
            _provider,
            *,
            include_revoked=False,
        ):
            self.calls += 1
            if self.calls == 1:
                lookup_started.set()
                await release_lookup.wait()
            if source == "user":
                return row
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

    class SharedRepo:
        async def fetch_authorized_secret_for_user(
            self,
            scope_type,
            _scope_id,
            user_id,
            _provider,
        ):
            assert user_id == 7
            return row if scope_type == source else None

    async def get_user_repo():
        return UserRepo()

    async def get_shared_repo():
        return SharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", get_shared_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", live_config)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"anthropic_api": dict(live_config["anthropic_api"])},
    )
    task = asyncio.create_task(
        byok_runtime.resolve_byok_credentials(
            "anthropic",
            user_id=7,
            team_ids=[11] if source == "team" else [],
            org_ids=[13] if source == "org" else [],
        )
    )
    try:
        await asyncio.wait_for(lookup_started.wait(), timeout=1.0)
        live_config["anthropic_api"] = {
            "api_key": "sk-static-generation-b",
            "api_url": "https://generation-b.example/v1",
            "model": "model-b",
        }
        release_lookup.set()
        resolution = await task
        expected_key = (
            "sk-static-generation-a"
            if source == "absent"
            else "sk-byok-generation-a"
        )
        assert resolution.api_key == expected_key
        assert resolution.app_config == {
            "anthropic_api": {
                "api_url": "https://generation-a.example/v1",
                "model": "model-a",
            }
        }
    finally:
        release_lookup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture(autouse=True)
def _use_in_process_oauth_refresh_lock(monkeypatch):
    """Keep unrelated unit tests isolated from the production DB lock backend."""
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")


@pytest.mark.asyncio
async def test_structured_server_fallback_keeps_key_fields_and_auth_source_atomic(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    fields = {
        "base_url": "https://override-openai.example/v1",
        "org_id": "override-org",
        "project_id": "override-project",
    }
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {
            "openai_api": {"model": "gpt-4o-mini", "api_key": "config-secret"},
            "anthropic_api": {"api_key": "unrelated-secret"},
        },
    )
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not mix fallback sources")
        ),
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=None,
        fallback_resolver=lambda _provider: _server_fallback_value(
            api_key="override-key",
            credential_fields=fields,
            auth_source="api_key",
        ),
    )
    fields["org_id"] = "mutated-after-resolution"

    assert resolved.api_key == "override-key"
    assert resolved.auth_source == "api_key"
    assert resolved.credential_fields == {
        "base_url": "https://override-openai.example/v1",
        "org_id": "override-org",
        "project_id": "override-project",
    }
    assert resolved.app_config == {
        "openai_api": {
            "model": "gpt-4o-mini",
            "api_base_url": "https://override-openai.example/v1",
            "org_id": "override-org",
            "project_id": "override-project",
            "_runtime_auth_source": "api_key",
        }
    }
    assert "anthropic_api" not in resolved.app_config
    assert "config-secret" not in repr(resolved.app_config)
    assert "unrelated-secret" not in repr(resolved.app_config)


@pytest.mark.asyncio
async def test_static_server_fallback_keeps_key_and_endpoint_from_one_config_snapshot(monkeypatch):
    """A config reload cannot cross a key from one generation with another endpoint."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    snapshot = {
        "openai_api": {
            "api_key": "static-key-a",
            "api_base_url": "https://static-a.example/v1",
        }
    }
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        byok_runtime,
        "loaded_config_data",
        {
            "openai_api": {
                "api_key": "static-key-b",
                "api_base_url": "https://static-b.example/v1",
            }
        },
    )

    fallback = byok_runtime.resolve_static_server_fallback("openai")
    snapshot["openai_api"]["api_base_url"] = "https://mutated.example/v1"
    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=None,
        fallback_resolver=lambda _provider: fallback,
    )

    assert resolved.api_key == "static-key-a"
    assert resolved.app_config == {
        "openai_api": {"api_base_url": "https://static-a.example/v1"}
    }


def test_static_server_fallback_from_snapshot_never_reloads_or_retains_mutable_config(
    monkeypatch,
):
    """A caller-owned snapshot remains the sole immutable fallback generation."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    snapshot = {
        "openai_api": {
            "api_key": "captured-key",
            "api_base_url": "https://captured.example/v1",
        }
    }
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected config reload")),
    )

    fallback = byok_runtime.resolve_static_server_fallback_from_snapshot(
        "openai",
        snapshot,
    )
    snapshot["openai_api"]["api_key"] = "rotated-key"
    snapshot["openai_api"]["api_base_url"] = "https://rotated.example/v1"

    assert fallback.api_key == "captured-key"
    assert fallback.app_config == {
        "openai_api": {"api_base_url": "https://captured.example/v1"}
    }


def test_static_server_fallback_does_not_reread_environment_after_snapshot(monkeypatch):
    """An env rotation after config capture cannot cross credential generations."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    def capture_then_rotate_environment():
        snapshot = {
            "qwen_api": {
                "api_key": "qwen-static-key-a",
                "api_base_url": "https://qwen-static-a.example/v1",
            }
        }
        monkeypatch.setenv("QWEN_API_KEY", "qwen-static-key-b")
        return snapshot

    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        capture_then_rotate_environment,
    )

    fallback = byok_runtime.resolve_static_server_fallback("qwen")

    assert fallback.api_key == "qwen-static-key-a"
    assert fallback.app_config == {
        "qwen_api": {"api_base_url": "https://qwen-static-a.example/v1"}
    }


@pytest.mark.parametrize(
    ("provider", "env_name", "section"),
    (
        ("novita", "NOVITA_API_KEY", "novita_api"),
        ("poe", "POE_API_KEY", "poe_api"),
        ("together", "TOGETHER_API_KEY", "together_api"),
        ("voyage", "VOYAGE_API_KEY", "voyage_api"),
        ("llama.cpp", "LLAMA_API_KEY", "llama_api"),
        ("custom-openai-api", "CUSTOM_OPENAI_API_KEY", "custom_openai_api"),
        ("custom-openai-api-2", "CUSTOM_OPENAI2_API_KEY", "custom_openai_api_2"),
        ("custom-openai-api-3", "CUSTOM_OPENAI_API_KEY_3", "custom_openai_api_3"),
    ),
)
def test_server_config_snapshot_preserves_env_only_provider_keys(
    monkeypatch,
    provider: str,
    env_name: str,
    section: str,
):
    """Providers absent from the legacy loader retain environment credentials."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setenv(env_name, f"{provider}-env-key")
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})

    snapshot = byok_helpers.load_server_config_snapshot()

    assert snapshot[section]["api_key"] == f"{provider}-env-key"
    assert (
        byok_helpers.resolve_server_default_key_from_snapshot(provider, snapshot)
        == f"{provider}-env-key"
    )


def test_server_config_snapshot_captures_key_and_endpoint_before_loader_rotation(
    monkeypatch,
):
    """The loader cannot cross environment generations while it is running."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key-a")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "https://openai-env-a.example/v1")

    def _rotate_during_load(**_kwargs):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key-b")
        monkeypatch.setenv(
            "OPENAI_API_BASE_URL",
            "https://openai-env-b.example/v1",
        )
        return {
            "openai_api": {
                "api_key": "loader-key-b",
                "api_base_url": "https://loader-b.example/v1",
            }
        }

    monkeypatch.setattr(byok_helpers, "load_and_log_configs", _rotate_during_load)

    snapshot = byok_helpers.load_server_config_snapshot()

    assert snapshot["openai_api"] == {
        "api_key": "openai-env-key-a",
        "api_base_url": "https://openai-env-a.example/v1",
    }


def test_server_config_snapshot_rejects_environment_value_absent_at_capture(
    monkeypatch,
):
    """A provider value appearing during load cannot enter the old generation."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key-a")
    for name in (
        "OPENAI_API_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "MOCK_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    observed_environment: dict[str, str] = {}

    def _rotate_during_load(*, environment):
        observed_environment.update(environment)
        monkeypatch.setenv(
            "OPENAI_API_BASE_URL",
            "https://openai-env-b.example/v1",
        )
        return {
            "openai_api": {
                "api_key": environment.get("OPENAI_API_KEY"),
                "api_base_url": environment.get("OPENAI_API_BASE_URL"),
            }
        }

    monkeypatch.setattr(byok_helpers, "load_and_log_configs", _rotate_during_load)

    snapshot = byok_helpers.load_server_config_snapshot()

    assert observed_environment["OPENAI_API_KEY"] == "openai-env-key-a"
    assert "OPENAI_API_BASE_URL" not in observed_environment
    assert snapshot["openai_api"]["api_key"] == "openai-env-key-a"
    assert snapshot["openai_api"]["api_base_url"] is None


def test_server_config_snapshot_keeps_env_credentials_when_loader_fails(monkeypatch):
    """An unrelated config parse error cannot erase a valid environment key."""
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime

    monkeypatch.setenv("NOVITA_API_KEY", "novita-loader-failure-canary")

    def _raise_loader_error(**_kwargs):
        raise RuntimeError("unrelated config parse failure")

    monkeypatch.setattr(byok_helpers, "load_and_log_configs", _raise_loader_error)

    snapshot = byok_helpers.load_server_config_snapshot()
    fallback = byok_runtime.resolve_static_server_fallback("novita")

    assert snapshot["novita_api"]["api_key"] == "novita-loader-failure-canary"
    assert fallback.api_key == "novita-loader-failure-canary"
    assert "loader-failure-canary" not in repr(fallback.app_config)


@pytest.mark.asyncio
async def test_static_bedrock_fallback_certifies_default_chain_without_bearer_key(monkeypatch):
    """Server IAM/role auth remains explicit when no Bedrock bearer key exists."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"bedrock_api": {"region": "us-west-2"}},
    )

    fallback = byok_runtime.resolve_static_server_fallback("bedrock")
    resolved = await byok_runtime.resolve_byok_credentials(
        "bedrock",
        user_id=None,
        fallback_resolver=lambda _provider: fallback,
    )

    assert fallback.api_key is None
    assert fallback.auth_source == "aws_default_chain"
    assert resolved.status is byok_runtime.ByokResolutionStatus.RESOLVED
    assert resolved.auth_source == "aws_default_chain"
    assert resolved.app_config == {
        "bedrock_api": {
            "region": "us-west-2",
            "_runtime_auth_source": "aws_default_chain",
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "section"),
    (("llama.cpp", "llama_api"), ("tabbyapi", "tabby_api")),
)
async def test_static_local_server_fallback_uses_canonical_config_section(
    monkeypatch,
    provider: str,
    section: str,
):
    """Local gateway keys and endpoints use the same shared provider-section map."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {
            section: {
                "api_key": "local-static-key",
                "api_ip": "https://local-static.example",
            }
        },
    )

    fallback = byok_runtime.resolve_static_server_fallback(provider)
    resolved = await byok_runtime.resolve_byok_credentials(
        provider,
        user_id=None,
        fallback_resolver=lambda _provider: fallback,
    )

    assert resolved.api_key == "local-static-key"
    assert resolved.app_config == {
        section: {"api_ip": "https://local-static.example"}
    }


@pytest.mark.asyncio
async def test_malformed_structured_server_fallback_fails_without_secondary_key(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    secondary_calls: list[str] = []
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        lambda provider, _snapshot: secondary_calls.append(provider)
        or _server_fallback_value(api_key="secondary-key", credential_fields={}),
    )

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=None,
            fallback_resolver=lambda _provider: _server_fallback_value(
                api_key=None,
                credential_fields={"org_id": "must-not-pair"},
                auth_source="api_key",
            ),
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert secondary_calls == []


@pytest.mark.asyncio
async def test_legacy_string_server_fallback_remains_supported(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=None,
        fallback_resolver=lambda _provider: "legacy-server-key",
    )

    assert resolved.api_key == "legacy-server-key"
    assert resolved.source == "server_default"


@pytest.mark.asyncio
async def test_authoritative_empty_fallback_excludes_late_override_from_static_lookup(
    monkeypatch,
):
    """A structured absence must not be followed by a second override-cache read."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    static_calls: list[tuple[str, dict[str, Any]]] = []
    real_static_fallback = byok_runtime.resolve_static_server_fallback_from_snapshot

    def resolve_static_fallback(
        provider: str,
        snapshot: dict[str, Any],
    ):
        static_calls.append((provider, snapshot))
        return real_static_fallback(provider, snapshot)

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "configured-static-key"}},
    )
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        resolve_static_fallback,
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=None,
        fallback_resolver=lambda _provider: None,
    )

    assert resolved.api_key == "configured-static-key"
    assert static_calls == [
        ("openai", {"openai_api": {"api_key": "configured-static-key"}})
    ]


@pytest.mark.asyncio
async def test_omitted_fallback_resolver_uses_frozen_server_snapshot(monkeypatch):
    """Callers without a structured resolver use one frozen server snapshot."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    static_calls: list[tuple[str, dict[str, Any]]] = []
    real_static_fallback = byok_runtime.resolve_static_server_fallback_from_snapshot

    def resolve_static_fallback(
        provider: str,
        snapshot: dict[str, Any],
    ):
        static_calls.append((provider, snapshot))
        return real_static_fallback(provider, snapshot)

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "server-default-key"}},
    )
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        resolve_static_fallback,
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=None,
    )

    assert resolved.api_key == "server-default-key"
    assert static_calls == [
        ("openai", {"openai_api": {"api_key": "server-default-key"}})
    ]


@pytest.mark.asyncio
async def test_typed_fallback_store_outage_is_not_swallowed(monkeypatch):
    """The fallback boundary must propagate typed credential-store outages."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    outage = byok_runtime.ByokResolutionError(
        "credential_store_unavailable",
        "openai",
    )
    static_calls: list[str] = []

    def raise_outage(_provider: str) -> None:
        raise outage

    def static_fallback(provider: str, _snapshot: dict[str, Any]):
        static_calls.append(provider)
        return _server_fallback_value(
            api_key="must-not-fail-open",
            credential_fields={},
        )

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        static_fallback,
    )

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=None,
            fallback_resolver=raise_outage,
        )

    assert exc_info.value is outage
    assert static_calls == []


@pytest.mark.asyncio
async def test_generic_fallback_exception_log_redacts_hostile_details(monkeypatch):
    """Generic fallback failures keep behavior while logging bounded metadata."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    secret = "sk-hostile-fallback-canary"
    private_path = "/private/provider-credentials.json"

    def raise_generic_failure(_provider: str) -> None:
        raise RuntimeError(f"failure with {secret} at {private_path}")

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "configured-static-key"}},
    )
    messages: list[str] = []
    sink_id = byok_runtime.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )

    try:
        resolved = await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=None,
            fallback_resolver=raise_generic_failure,
        )
    finally:
        byok_runtime.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert resolved.api_key == "configured-static-key"
    assert "BYOK fallback resolver failed" in joined
    assert "provider=openai" in joined
    assert "error_type=RuntimeError" in joined
    assert secret not in joined
    assert private_path not in joined


@pytest.mark.asyncio
async def test_structured_bedrock_default_chain_is_certified_without_api_key(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"bedrock_api": {"model": "meta.model"}},
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        "bedrock",
        user_id=None,
        fallback_resolver=lambda _provider: byok_runtime.ServerFallbackCredentials(
            api_key=None,
            credential_fields={},
            auth_source="aws_default_chain",
        ),
    )

    assert resolved.api_key is None
    assert resolved.status is byok_runtime.ByokResolutionStatus.RESOLVED
    assert resolved.auth_source == "aws_default_chain"
    assert resolved.app_config == {
        "bedrock_api": {
            "model": "meta.model",
            "_runtime_auth_source": "aws_default_chain",
        }
    }


@pytest.mark.asyncio
async def test_admin_override_fallback_keeps_key_and_fields_atomic(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime, llm_provider_overrides

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "validate_base_url_override", lambda value: value)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", {})
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": llm_provider_overrides.LLMProviderOverride(
                provider="openai",
                api_key="admin-key",
                credential_fields={
                    "base_url": "https://admin-openai.example/v1",
                    "org_id": "admin-org",
                },
                config={"auth_source": "api_key"},
            )
        }
    )
    try:
        resolved = await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=None,
            fallback_resolver=llm_provider_overrides.get_override_server_fallback,
        )
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert resolved.api_key == "admin-key"
    assert resolved.credential_fields == {
        "base_url": "https://admin-openai.example/v1",
        "org_id": "admin-org",
    }
    assert resolved.app_config == {
        "openai_api": {
            "api_base_url": "https://admin-openai.example/v1",
            "org_id": "admin-org",
            "_runtime_auth_source": "api_key",
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "decrypted_payload"),
    [
        (
            {"provider": "openai", "secret_blob": "opaque"},
            {
                "api_key": "stored-key",
                "credential_fields": ["not", "an", "object"],
            },
        ),
        ({"provider": "openai", "secret_blob": "opaque"}, {}),
        ({"provider": "openai", "api_key_hint": "stored-hint"}, {}),
    ],
)
async def test_corrupt_stored_admin_override_fails_without_secondary_key(
    monkeypatch,
    row,
    decrypted_payload,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime, llm_provider_overrides

    secondary_calls: list[str] = []
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        lambda provider, _snapshot: secondary_calls.append(provider)
        or _server_fallback_value(api_key="secondary-key", credential_fields={}),
    )
    monkeypatch.setattr(llm_provider_overrides, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(
        llm_provider_overrides,
        "decrypt_byok_payload",
        lambda _envelope: decrypted_payload,
    )
    override = llm_provider_overrides._parse_override_row(row)
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": override}
    )

    try:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials(
                "openai",
                user_id=None,
                fallback_resolver=llm_provider_overrides.get_override_server_fallback,
            )
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert exc_info.value.code == "invalid_provider_credentials"
    assert secondary_calls == []


@pytest.mark.asyncio
async def test_malformed_admin_override_helper_state_fails_without_secondary_key(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime, llm_provider_overrides

    secondary_calls: list[str] = []
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "resolve_static_server_fallback_from_snapshot",
        lambda provider, _snapshot: secondary_calls.append(provider)
        or _server_fallback_value(api_key="secondary-key", credential_fields={}),
    )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": llm_provider_overrides.LLMProviderOverride(
                provider="openai",
                api_key="stored-key",
                credential_fields=["not", "a", "mapping"],  # type: ignore[arg-type]
            )
        }
    )

    try:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await byok_runtime.resolve_byok_credentials(
                "openai",
                user_id=None,
                fallback_resolver=llm_provider_overrides.get_override_server_fallback,
            )
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})

    assert exc_info.value.code == "invalid_provider_credentials"
    assert secondary_calls == []


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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

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
async def test_resolve_byok_credentials_alias_conflict_fails_closed(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            raise ProviderCredentialAliasConflictError("legacy alias rows conflict")

        fetch_secret_for_active_user = fetch_secret_for_user

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    fallback_calls: list[str] = []
    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "oai",
            user_id=1,
            fallback_resolver=lambda provider: fallback_calls.append(provider) or "server-key",
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert fallback_calls == []
    assert "legacy alias rows conflict" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_stage",
    ["locked_reload", "cas_persist"],
    ids=["locked-reload", "cas-persist"],
)
async def test_late_openai_oauth_alias_conflict_fails_before_adapter(
    monkeypatch,
    failure_stage,
):
    """Late repository alias conflicts stay inside the public error contract."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    reset_settings()

    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        }
    )
    row.update({"metadata": None, "key_hint": "oauth", "revoked_at": None})
    conflict_sites: list[str] = []
    refresh_calls = 0

    class InitialRepo:
        async def fetch_secret_for_active_user(self, *_args, **_kwargs):
            return dict(row)

    class LockedRepo:
        async def _reload(self):
            if failure_stage == "locked_reload":
                conflict_sites.append(failure_stage)
                raise ProviderCredentialAliasConflictError(
                    "late alias conflict must not escape"
                )
            return dict(row)

        async def fetch_secret_for_user(self, *_args, **_kwargs):
            return await self._reload()

        async def fetch_secret_for_active_user(self, *_args, **_kwargs):
            return await self._reload()

        async def update_secret_if_active_and_unchanged(self, **_kwargs):
            assert failure_stage == "cas_persist"
            conflict_sites.append(failure_stage)
            raise ProviderCredentialAliasConflictError(
                "late alias conflict must not escape"
            )

    initial_repo = InitialRepo()
    locked_repo = LockedRepo()

    async def get_initial_repo():
        return initial_repo

    @contextlib.asynccontextmanager
    async def locked_refresh(**_kwargs):
        yield locked_repo

    async def refresh_token(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        return {
            "access_token": "must-not-reach-adapter",
            "refresh_token": "rotated-refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_initial_repo)
    monkeypatch.setattr(byok_runtime, "_openai_oauth_refresh_lock", locked_refresh)
    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    adapter_calls = 0

    async def resolve_and_dispatch():
        nonlocal adapter_calls
        resolved = await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            team_ids=[],
            org_ids=[],
            required_source="user",
            server_config_snapshot={},
        )
        adapter_calls += 1
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    try:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await resolve_and_dispatch()

        assert vars(exc_info.value) == {
            "code": "invalid_provider_credentials",
            "provider": "openai",
        }
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert conflict_sites == [failure_stage]
        assert refresh_calls == (1 if failure_stage == "cas_persist" else 0)
        assert adapter_calls == 0
        assert captured_headers == []
        assert "late alias conflict" not in repr(exc_info.value)
    finally:
        reset_settings()


@pytest.mark.asyncio
async def test_revoked_user_credential_blocks_shared_and_server_fallback(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    revoked_row = {
        "provider": "openai",
        "encrypted_blob": "revoked-blob",
        "revoked_at": datetime.now(timezone.utc),
    }

    class _FakeUserRepo:
        async def fetch_secret_for_user(
            self,
            user_id: int,
            provider: str,
            *,
            include_revoked: bool = False,
        ):
            return revoked_row if include_revoked else None

        fetch_secret_for_active_user = fetch_secret_for_user

    async def _fake_get_user_repo():
        return _FakeUserRepo()

    async def _shared_lookup_must_not_run():
        raise AssertionError("revoked user credential must block shared fallback")

    fallback_calls: list[str] = []
    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _shared_lookup_must_not_run)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "oai",
            user_id=1,
            team_ids=[7],
            org_ids=[9],
            fallback_resolver=lambda provider: fallback_calls.append(provider) or "server-key",
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert fallback_calls == []


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
    _assert_opaque_scope_token(resolved.credential_scope_token)
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
async def test_gateway_admin_scope_tracks_config_and_key_rotation(
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    first = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(
            api_key="admin-secret-one",
            config_generation="generation-one",
        ),
    )
    same = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(
            api_key="admin-secret-one",
            config_generation="generation-one",
        ),
    )
    rotated_key = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(
            api_key="admin-secret-two",
            config_generation="generation-one",
        ),
    )
    changed_config = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=None,
        gateway_spec=_gateway_spec(
            api_key="admin-secret-one",
            config_generation="generation-two",
        ),
    )

    assert first.source == "server_default"
    assert first.api_key == "admin-secret-one"
    assert first.credential_scope_token
    assert first.credential_scope_token == same.credential_scope_token
    assert first.credential_scope_token != rotated_key.credential_scope_token
    assert first.credential_scope_token != changed_config.credential_scope_token
    key_fingerprint = hashlib.sha256(
        b"admin-secret-one",
        usedforsecurity=True,
    ).hexdigest()
    assert key_fingerprint not in first.credential_scope_token
    assert "admin-secret-one" not in first.credential_scope_token
    assert "voice-lab" not in first.credential_scope_token
    assert first.credential_scope_token not in repr(first)


@pytest.mark.asyncio
async def test_gateway_repository_driver_error_fails_closed_without_admin_fallback(
    monkeypatch,
    gateway_byok_encryption,
):
    import aiosqlite

    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    log_calls: list[tuple[str, tuple[object, ...]]] = []

    class _Logger:
        def debug(self, message: str, *args: object) -> None:
            log_calls.append((message, args))

    class _FailingRepo:
        async def fetch_secret_for_user(self, _user_id: int, _provider: str):
            raise aiosqlite.Error("sensitive database detail")

    async def _fake_get_user_repo():
        return _FailingRepo()

    monkeypatch.setattr(byok_runtime, "logger", _Logger())
    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)

    resolved = await byok_runtime.resolve_gateway_byok_credentials(
        "gateway:voice-lab",
        user_id=9,
        gateway_spec=_gateway_spec(api_key="admin-must-not-be-used"),
    )

    assert resolved.source == "none"
    assert resolved.api_key is None
    assert resolved.credential_scope_token is None
    assert log_calls
    assert log_calls[-1][1] == ("gateway:voice-lab", "Error")
    assert "sensitive database detail" not in repr(log_calls)


@pytest.mark.asyncio
async def test_gateway_repository_cancellation_propagates(
    monkeypatch,
    gateway_byok_encryption,
):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _CancelledRepo:
        async def fetch_secret_for_user(self, _user_id: int, _provider: str):
            raise asyncio.CancelledError

    async def _fake_get_user_repo():
        return _CancelledRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _fake_get_user_repo)

    with pytest.raises(asyncio.CancelledError):
        await byok_runtime.resolve_gateway_byok_credentials(
            "gateway:voice-lab",
            user_id=9,
            gateway_spec=_gateway_spec(api_key="admin-must-not-be-used"),
        )


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
    _assert_opaque_scope_token(first.credential_scope_token)
    _assert_opaque_scope_token(other_user.credential_scope_token)


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
    _assert_opaque_scope_token(first.credential_scope_token)


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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(
            self,
            *,
            user_id: int,
            provider: str,
            encrypted_blob: str,
            expected_encrypted_blob: str,
            key_hint: str | None,
            metadata,
            updated_at: datetime,
            updated_by: int | None = None,
        ):
            assert row["encrypted_blob"] == expected_encrypted_blob
            row["encrypted_blob"] = encrypted_blob
            row["key_hint"] = key_hint
            row["metadata"] = metadata
            row["updated_at"] = updated_at
            return True

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(
            self,
            *,
            user_id: int,
            provider: str,
            encrypted_blob: str,
            expected_encrypted_blob: str,
            key_hint: str | None,
            metadata,
            updated_at: datetime,
            updated_by: int | None = None,
        ):
            assert row["encrypted_blob"] == expected_encrypted_blob
            row["encrypted_blob"] = encrypted_blob
            row["key_hint"] = key_hint
            row["metadata"] = metadata
            row["updated_at"] = updated_at
            return True

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return {"encrypted_blob": "not-an-envelope", "last_used_at": None}

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

    class _FakeSharedRepo:
        async def fetch_authorized_secret_for_user(
            self,
            requested_scope: str,
            scope_id: int,
            user_id: int,
            provider: str,
        ):
            assert requested_scope == scope_type
            assert user_id == 1
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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

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
        replace_credential_metadata=True,
    )

    assert app_config == {
        "openai_api": {
            "model": "gpt-safe",
            "api_timeout": 12,
            "api_retries": 2,
            "api_retry_delay": 0.25,
            "api_base_url": "https://byok-openai.example/v1",
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
@pytest.mark.concurrent
async def test_concurrent_openai_user_keys_do_not_inherit_server_credential_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each user key carries only its own OpenAI organization/project metadata."""

    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        ProviderCredentialRuntime,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    reset_settings()
    rows = {
        1: _encrypted_row(build_secret_payload("user-key-a")),
        2: _encrypted_row(
            build_secret_payload(
                "user-key-b",
                credential_fields={
                    "org_id": "user-org-b",
                    "project_id": "user-project-b",
                },
            )
        ),
    }

    class UserRepo:
        async def fetch_secret_for_active_user(
            self,
            user_id: int,
            _provider: str,
            **_kwargs: Any,
        ) -> dict[str, Any] | None:
            return rows.get(user_id)

    repo = UserRepo()

    async def get_user_repo() -> UserRepo:
        return repo

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_user_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    server_config = {
        "openai_api": {
            "api_base_url": "https://server-openai.example/v1",
            "organization": "server-org",
            "organization_id": "server-org-id",
            "org_id": "server-org-short",
            "project": "server-project",
            "project_id": "server-project-id",
        }
    }
    runtimes = [
        ProviderCredentialRuntime(
            user_id=user_id,
            team_ids=(),
            org_ids=(),
            trusted_base_url_override=False,
            server_config_snapshot=server_config,
        )
        for user_id in (1, 2)
    ]
    handles = await asyncio.gather(
        *(runtime.resolve("openai", model="gpt-4o-mini") for runtime in runtimes)
    )

    both_entered = threading.Event()
    release = threading.Event()
    capture_lock = threading.Lock()
    captured: list[tuple[str, str | None, str | None, float | None]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

    class Client:
        def __init__(self, timeout: float | None) -> None:
            self.timeout = timeout

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def post(
            self,
            _url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> Response:
            del json
            with capture_lock:
                captured.append(
                    (
                        headers["Authorization"],
                        headers.get("OpenAI-Organization"),
                        headers.get("OpenAI-Project"),
                        self.timeout,
                    )
                )
                if len(captured) == 2:
                    both_entered.set()
            assert release.wait(timeout=5)
            return Response()

    monkeypatch.setattr(
        openai_adapter,
        "http_client_factory",
        lambda *, timeout=None: Client(timeout),
    )
    adapter = openai_adapter.OpenAIAdapter()

    async def invoke(handle: Any, timeout: float) -> dict[str, Any]:
        return await adapter.achat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-4o-mini",
                "credentials_resolved": True,
                PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
            },
            timeout=timeout,
        )

    tasks = [
        asyncio.create_task(invoke(handles[0], 11.0)),
        asyncio.create_task(invoke(handles[1], 23.0)),
    ]
    try:
        assert await asyncio.to_thread(both_entered.wait, 5)
    finally:
        release.set()
    try:
        await asyncio.gather(*tasks)
    finally:
        await asyncio.gather(*(runtime.close() for runtime in runtimes))

    assert set(captured) == {
        ("Bearer user-key-a", None, None, 11.0),
        (
            "Bearer user-key-b",
            "user-org-b",
            "user-project-b",
            23.0,
        ),
    }


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
    server_config = {
        section: {
            "api_key": "config-secret",
            "model": "selected-model",
            endpoint_key: "http://selected-provider.example/v1",
            "api_timeout": 10,
        },
        "anthropic_api": {
            "api_key": "unrelated-secret",
            "model": "unrelated-model",
        },
    }
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: server_config,
    )

    resolved = await byok_runtime.resolve_byok_credentials(
        provider,
        user_id=1,
        fallback_resolver=lambda _provider: _server_fallback_value(
            api_key=server_key,
            credential_fields={},
            app_config=server_config,
        ),
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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row if scope == "user" else None

        fetch_secret_for_active_user = fetch_secret_for_user

    class _FakeSharedRepo:
        async def fetch_authorized_secret_for_user(
            self,
            scope_type: str,
            scope_id: int,
            user_id: int,
            provider: str,
        ):
            assert scope_type == scope
            assert user_id == 1
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
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            assert row["encrypted_blob"] == kwargs["expected_encrypted_blob"]
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            return True

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


@pytest.mark.asyncio
async def test_openai_oauth_reload_missing_after_lock_fails_without_resurrection(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    stale_token = "stale-token-must-not-leak"
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": stale_token,
                    "refresh_token": "refresh-token-race",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
                }
            },
        }
    )
    row["metadata"] = None
    row["key_hint"] = "oauth"

    initial_fetch_complete = asyncio.Event()
    reload_waiting = asyncio.Event()
    credential_revoked = asyncio.Event()
    calls = {"fetch": 0, "http": 0, "upsert": 0, "touch": 0, "fallback": 0}

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            calls["fetch"] += 1
            if calls["fetch"] == 1:
                initial_fetch_complete.set()
                return row
            reload_waiting.set()
            await credential_revoked.wait()
            return None

        fetch_secret_for_active_user = fetch_secret_for_user

        async def upsert_secret(self, **kwargs):
            calls["upsert"] += 1
            return {"updated_at": kwargs["updated_at"]}

        async def touch_last_used(self, user_id: int, provider: str, updated_at: datetime):
            calls["touch"] += 1

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": "resurrected-token", "expires_in": 3600}

        async def aclose(self):
            return None

    async def _get_user_repo():
        return _FakeUserRepo()

    async def _http_afetch(*args, **kwargs):
        calls["http"] += 1
        return _FakeResponse()

    def _fallback(provider: str):
        calls["fallback"] += 1
        return "server-key-must-not-be-used"

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    captured_logs: list[str] = []
    sink_id = logger.add(lambda message: captured_logs.append(str(message)), level="DEBUG")
    resolution_task = asyncio.create_task(
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            fallback_resolver=_fallback,
        )
    )
    await initial_fetch_complete.wait()
    await reload_waiting.wait()
    credential_revoked.set()
    try:
        with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
            await resolution_task
    finally:
        logger.remove(sink_id)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.provider == "openai"
    assert stale_token not in str(exc_info.value)
    assert stale_token not in repr(exc_info.value)
    assert stale_token not in "".join(captured_logs)
    assert calls == {
        "fetch": 2,
        "http": 0,
        "upsert": 0,
        "touch": 0,
        "fallback": 0,
    }


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_openai_oauth_refresh_cas_discards_token_after_revocation(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    issued_token = "issued-token-must-not-be-stored"
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    reset_settings()

    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "stale-access-token",
                    "refresh_token": "refresh-token-race",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
                }
            },
        }
    )
    row.update({"metadata": None, "key_hint": "oauth", "revoked_at": None})
    original_blob = row["encrypted_blob"]
    token_request_started = asyncio.Event()
    release_token_response = asyncio.Event()

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            return row if row["revoked_at"] is None else None

        fetch_secret_for_active_user = fetch_secret_for_user

        async def delete_secret(self, user_id: int, provider: str):
            row["revoked_at"] = datetime.now(timezone.utc)
            return True

        async def upsert_secret(self, **kwargs):
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            row["revoked_at"] = None
            return {"updated_at": kwargs["updated_at"]}

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            if row["revoked_at"] is not None:
                return False
            if row["encrypted_blob"] != kwargs["expected_encrypted_blob"]:
                return False
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            return True

    repo = _FakeUserRepo()

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"access_token": issued_token, "expires_in": 3600}

        async def aclose(self):
            return None

    async def _get_user_repo():
        return repo

    async def _http_afetch(*_args, **_kwargs):
        token_request_started.set()
        await release_token_response.wait()
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolution_task = asyncio.create_task(byok_runtime.resolve_byok_credentials("openai", user_id=1))
    await token_request_started.wait()
    await repo.delete_secret(1, "openai")
    release_token_response.set()

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await resolution_task

    assert exc_info.value.code == "invalid_provider_credentials"
    assert row["revoked_at"] is not None
    assert row["encrypted_blob"] == original_blob
    assert issued_token not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "winning_access_token",
    ["winner-access-token-1", "stale-access-token"],
    ids=["rotated-access-token", "same-access-new-refresh-token"],
)
async def test_concurrent_forced_openai_oauth_refreshes_share_the_winning_update(
    monkeypatch,
    winning_access_token,
):
    """Two overlapping forced resolutions must rotate a refresh token only once."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    reset_settings()

    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "stale-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                }
            },
        }
    )
    row.update({"metadata": None, "key_hint": "oauth", "revoked_at": None})
    initial_fetches_ready = asyncio.Event()
    task_fetch_counts: dict[asyncio.Task, int] = {}
    initial_fetch_count = 0
    token_requests: list[str] = []

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            nonlocal initial_fetch_count
            task = asyncio.current_task()
            assert task is not None
            prior = task_fetch_counts.get(task, 0)
            task_fetch_counts[task] = prior + 1
            # Capture the DB row at fetch time. Returning a live copy only
            # after the barrier would let the winner update it first and no
            # longer model two overlapping stale reads.
            snapshot = dict(row)
            if prior == 0:
                initial_fetch_count += 1
                if initial_fetch_count == 2:
                    initial_fetches_ready.set()
                await initial_fetches_ready.wait()
            return snapshot

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            if row["encrypted_blob"] != kwargs["expected_encrypted_blob"]:
                return False
            assert kwargs["encrypted_blob"] != kwargs["expected_encrypted_blob"]
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            row["key_hint"] = kwargs["key_hint"]
            row["metadata"] = kwargs["metadata"]
            row["updated_at"] = kwargs["updated_at"]
            return True

    repo = _FakeUserRepo()

    class _FakeResponse:
        status_code = 200

        def __init__(self, token_number: int) -> None:
            self._token_number = token_number

        def json(self):
            return {
                "access_token": winning_access_token,
                "refresh_token": f"winner-refresh-token-{self._token_number}",
                # Stay beyond even the maximum configured refresh skew so the
                # second waiter can prove it adopted the winner.
                "expires_in": 7200,
            }

        async def aclose(self):
            return None

    async def _get_user_repo():
        return repo

    async def _http_afetch(*_args, **_kwargs):
        token_requests.append(str(_kwargs["data"]["refresh_token"]))
        return _FakeResponse(len(token_requests))

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    first, second = await asyncio.gather(
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            force_oauth_refresh=True,
        ),
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            force_oauth_refresh=True,
        ),
    )

    assert token_requests == ["single-use-refresh-token"]
    assert first.api_key == winning_access_token
    assert second.api_key == winning_access_token
    assert _decrypted_payload_from_row(row)["credentials"]["oauth"]["refresh_token"] == (
        "winner-refresh-token-1"
    )


@pytest.mark.asyncio
async def test_forced_openai_oauth_refresh_ignores_unrelated_secret_row_update(monkeypatch):
    """An unrelated credential-field edit must not masquerade as token rotation."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"u"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    reset_settings()

    oauth_payload = {
        "access_token": "still-current-access-token",
        "refresh_token": "force-refresh-token",
        "issued_at": "2026-07-14T00:00:00+00:00",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
    }
    initial_row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {"oauth": dict(oauth_payload)},
        }
    )
    latest_row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    **oauth_payload,
                    "scope": "scope-edited-while-waiting",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=8)
                    ).isoformat(),
                }
            },
            "credential_fields": {"project_id": "edited-while-waiting"},
        }
    )
    latest_row.update({"metadata": None, "key_hint": "oauth", "revoked_at": None})
    fetch_count = 0
    token_requests: list[str] = []

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            nonlocal fetch_count
            fetch_count += 1
            return dict(initial_row if fetch_count == 1 else latest_row)

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            assert kwargs["expected_encrypted_blob"] == latest_row["encrypted_blob"]
            latest_row["encrypted_blob"] = kwargs["encrypted_blob"]
            return True

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "access_token": "forced-access-token",
                "refresh_token": "forced-refresh-token",
                "expires_in": 7200,
            }

        async def aclose(self):
            return None

    repo = _FakeUserRepo()

    async def _get_user_repo():
        return repo

    async def _http_afetch(*_args, **kwargs):
        token_requests.append(str(kwargs["data"]["refresh_token"]))
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolved = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=1,
        force_oauth_refresh=True,
    )

    assert token_requests == ["force-refresh-token"]
    assert resolved.api_key == "forced-access-token"
    assert resolved.credential_fields == {"project_id": "edited-while-waiting"}


@pytest.mark.asyncio
async def test_forced_openai_oauth_refresh_ignores_refresh_token_only_rotation(
    monkeypatch,
):
    """A new refresh token cannot prove the rejected access token was repaired."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"v"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    reset_settings()

    def _row(refresh_token: str) -> dict[str, Any]:
        row = _encrypted_row(
            {
                "credential_version": 2,
                "active_auth_source": "oauth",
                "credentials": {
                    "oauth": {
                        "access_token": "rejected-access-token",
                        "refresh_token": refresh_token,
                        "expires_at": (
                            datetime.now(timezone.utc) + timedelta(hours=4)
                        ).isoformat(),
                    }
                },
            }
        )
        row.update({"metadata": None, "key_hint": "oauth", "revoked_at": None})
        return row

    initial_row = _row("refresh-token-1")
    latest_row = _row("refresh-token-2")
    fetch_count = 0
    token_requests: list[str] = []

    class _FakeUserRepo:
        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            nonlocal fetch_count
            fetch_count += 1
            return dict(initial_row if fetch_count == 1 else latest_row)

        fetch_secret_for_active_user = fetch_secret_for_user

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            assert kwargs["expected_encrypted_blob"] == latest_row["encrypted_blob"]
            latest_row["encrypted_blob"] = kwargs["encrypted_blob"]
            return True

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "access_token": "repaired-access-token",
                "refresh_token": "refresh-token-3",
                "expires_in": 7200,
            }

        async def aclose(self):
            return None

    repo = _FakeUserRepo()

    async def _get_user_repo():
        return repo

    async def _http_afetch(*_args, **kwargs):
        token_requests.append(str(kwargs["data"]["refresh_token"]))
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    rejected = await byok_runtime.resolve_byok_credentials("openai", user_id=1)
    repaired = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=1,
        force_oauth_refresh=True,
        rejected_credential_generation=rejected._credential_generation,
    )

    assert token_requests == ["refresh-token-2"]
    assert repaired.api_key == "repaired-access-token"


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_postgres_oauth_refresh_does_not_deadlock_at_pool_capacity(monkeypatch):
    """The advisory-lock owner must reload and CAS on its already-held connection."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"p"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    reset_settings()

    row = _encrypted_row(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "stale-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                }
            },
        }
    )
    row.update(
        {
            "id": 1,
            "user_id": 1,
            "provider": "openai",
            "metadata": None,
            "key_hint": "oauth",
            "revoked_at": None,
        }
    )
    capacity = asyncio.Semaphore(2)
    initial_fetches_ready = asyncio.Event()
    advisory_attempts_ready = asyncio.Event()
    task_fetch_counts: dict[asyncio.Task, int] = {}
    initial_fetch_count = 0
    advisory_attempts = 0
    advisory_locked = False
    active_connections = 0
    max_active_connections = 0
    token_requests: list[str] = []
    refresh_started = asyncio.Event()
    allow_refresh = asyncio.Event()

    class _FakeConnection:
        @contextlib.asynccontextmanager
        async def transaction(self):
            yield self

        async def execute(self, query: str, *_args):
            if "pg_advisory_xact_lock" in query:
                return "SELECT 1"
            if "DELETE FROM user_provider_secrets" in query:
                return "DELETE 0"
            raise AssertionError(f"Unexpected provider-secret mutation: {query}")

        async def fetchval(self, query: str, *_args):
            nonlocal advisory_attempts, advisory_locked
            if "pg_try_advisory_lock" in query:
                advisory_attempts += 1
                won = not advisory_locked
                if won:
                    advisory_locked = True
                if advisory_attempts >= 2:
                    advisory_attempts_ready.set()
                if won:
                    await advisory_attempts_ready.wait()
                return won
            if "pg_advisory_unlock" in query:
                advisory_locked = False
                return True
            raise AssertionError(f"Unexpected advisory-lock query: {query}")

        async def fetch(self, query: str, *args):
            if "SELECT provider, revoked_at" in query:
                return [
                    {
                        "provider": row["provider"],
                        "revoked_at": row["revoked_at"],
                    }
                ]
            result = await self.fetchrow(query, *args)
            return [result] if result else []

        async def fetchrow(self, query: str, *args):
            nonlocal initial_fetch_count
            if "UPDATE user_provider_secrets" in query:
                if row["encrypted_blob"] != args[-1]:
                    return None
                row["encrypted_blob"] = args[0]
                row["key_hint"] = args[1]
                row["metadata"] = args[2]
                row["updated_at"] = args[3]
                return {"id": 1}
            if "SELECT" in query and "user_provider_secrets" in query:
                task = asyncio.current_task()
                assert task is not None
                prior = task_fetch_counts.get(task, 0)
                task_fetch_counts[task] = prior + 1
                snapshot = dict(row)
                if prior == 0:
                    initial_fetch_count += 1
                    if initial_fetch_count == 2:
                        initial_fetches_ready.set()
                    await initial_fetches_ready.wait()
                return snapshot
            raise AssertionError(f"Unexpected provider-secret query: {query}")

    class _BoundedFakePool:
        backend_type = "postgres"
        pool = object()

        @contextlib.asynccontextmanager
        async def acquire(self, *, timeout=None):
            nonlocal active_connections, max_active_connections
            await capacity.acquire()
            active_connections += 1
            max_active_connections = max(max_active_connections, active_connections)
            try:
                yield _FakeConnection()
            finally:
                active_connections -= 1
                capacity.release()

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            async with self.acquire(timeout=timeout) as connection:
                yield connection

        async def fetchone(self, query: str, *args):
            async with self.acquire() as connection:
                result = await connection.fetchrow(query, *args)
                return dict(result) if result else None

        async def fetchall(self, query: str, *args):
            async with self.acquire() as connection:
                results = await connection.fetch(query, *args)
                return [dict(result) for result in results]

    pool = _BoundedFakePool()
    repo = AuthnzUserProviderSecretsRepo(pool)

    async def _get_db_pool():
        return pool

    async def _get_user_repo():
        return repo

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "access_token": "winner-access-token",
                "refresh_token": "winner-refresh-token",
                "expires_in": 7200,
            }

        async def aclose(self):
            return None

    async def _http_afetch(*_args, **kwargs):
        token_requests.append(str(kwargs["data"]["refresh_token"]))
        refresh_started.set()
        await allow_refresh.wait()
        return _FakeResponse()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)
    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_http_afetch", _http_afetch)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolutions = asyncio.gather(
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            force_oauth_refresh=True,
        ),
        byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=1,
            force_oauth_refresh=True,
        ),
    )
    await refresh_started.wait()

    async def _unrelated_pool_user() -> None:
        async with pool.acquire():
            return None

    try:
        # A losing advisory-lock waiter must release its pool connection while
        # sleeping so unrelated AuthNZ traffic can continue.
        await asyncio.wait_for(_unrelated_pool_user(), timeout=0.25)
    finally:
        allow_refresh.set()
        first, second = await asyncio.wait_for(resolutions, timeout=2)

    assert max_active_connections == 2
    assert token_requests == ["single-use-refresh-token"]
    assert first.api_key == "winner-access-token"
    assert second.api_key == "winner-access-token"


@pytest.mark.asyncio
async def test_postgres_oauth_refresh_lock_bounds_pool_acquisition(monkeypatch):
    """The refresh-lock timeout must include waiting for a pool connection."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_TIMEOUT_SECONDS",
        0.02,
    )
    acquisition_started = asyncio.Event()
    acquisition_cancelled = asyncio.Event()

    class _BlockedPool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            acquisition_started.set()
            try:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=timeout)
            except asyncio.TimeoutError:
                acquisition_cancelled.set()
                raise
            yield  # pragma: no cover - a blocked pool never yields

    async def _get_db_pool():
        return _BlockedPool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(
            user_id=24,
            provider="openai",
        ):
            raise AssertionError("exhausted pool entered protected body")

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await asyncio.wait_for(_worker(), timeout=0.5)

    assert acquisition_started.is_set()
    assert acquisition_cancelled.is_set()
    assert exc_info.value.code == "credential_store_unavailable"


def test_db_oauth_refresh_lock_serializes_independent_event_loops(monkeypatch):
    """The DB backend must serialize process-like workers, not fall back to loop locks."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    advisory_lock = threading.Lock()
    state_lock = threading.Lock()
    start_barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    lock_attempts = 0
    unlocks = 0

    class _FakeConnection:
        async def fetchval(self, query: str, *_args):
            nonlocal lock_attempts, unlocks
            if "pg_try_advisory_lock" in query:
                with state_lock:
                    lock_attempts += 1
                return advisory_lock.acquire(blocking=False)
            if "pg_advisory_unlock" in query:
                advisory_lock.release()
                with state_lock:
                    unlocks += 1
                return True
            raise AssertionError(f"Unexpected advisory-lock query: {query}")

    class _FakePool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            yield _FakeConnection()

    pool = _FakePool()

    async def _get_db_pool():
        return pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async def _worker() -> None:
        nonlocal active, max_active
        async with byok_runtime._openai_oauth_refresh_lock(user_id=17, provider="openai"):
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            with state_lock:
                active -= 1

    def _run_worker() -> None:
        start_barrier.wait(timeout=5)
        asyncio.run(_worker())

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_worker) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)

    assert max_active == 1
    assert lock_attempts >= 2
    assert unlocks == 2


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_redis_oauth_refresh_lock_serializes_waiters_without_memory_fallback(monkeypatch):
    """An explicitly configured Redis backend must own the concurrency boundary."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    locked = False
    set_calls = 0
    release_calls = 0

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            nonlocal locked, set_calls
            set_calls += 1
            if locked:
                return False
            locked = True
            return True

        async def eval(self, _script, _count, _key, _token):
            nonlocal locked, release_calls
            locked = False
            release_calls += 1
            return 1

        async def aclose(self):
            return None

    fake_redis = _FakeRedis()
    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", lambda: fake_redis)
    monkeypatch.setattr(
        byok_runtime,
        "_get_openai_refresh_lock",
        lambda _key: (_ for _ in ()).throw(AssertionError("memory fallback used")),
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    entries: list[str] = []

    async def _first() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(user_id=23, provider="openai"):
            entries.append("first")
            first_entered.set()
            await release_first.wait()

    async def _second() -> None:
        await first_entered.wait()
        async with byok_runtime._openai_oauth_refresh_lock(user_id=23, provider="openai"):
            entries.append("second")

    first_task = asyncio.create_task(_first())
    second_task = asyncio.create_task(_second())
    await first_entered.wait()
    await asyncio.sleep(0.06)
    assert entries == ["first"]
    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert entries == ["first", "second"]
    assert set_calls >= 3
    assert release_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_error",
    [
        ValueError("protected mutation rejected"),
        HTTPException(status_code=404, detail="credential not found"),
    ],
    ids=("value-error", "http-exception"),
)
async def test_redis_mutation_lock_preserves_protected_body_errors(
    monkeypatch,
    body_error: Exception,
):
    """Redis transport sanitization must not rewrite endpoint/repository errors."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        3600,
        raising=False,
    )

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            return 1

        async def aclose(self):
            return None

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    with pytest.raises(type(body_error)) as exc_info:
        async with byok_runtime.openai_credential_mutation_lock(
            user_id=24,
            provider="openai",
        ):
            raise body_error

    assert exc_info.value is body_error


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_redis_oauth_refresh_lock_renews_lease_until_owner_exits(monkeypatch):
    """A refresh longer than the initial lease must remain single-owner."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        0.01,
        raising=False,
    )
    token: str | None = None
    renew_calls = 0
    active = 0
    max_active = 0
    second_attempted = asyncio.Event()
    renewed_twice = asyncio.Event()
    lease_window_elapsed = asyncio.Event()
    allow_first_exit = asyncio.Event()

    class _FakeRedis:
        async def set(self, _key, candidate, **_kwargs):
            nonlocal token
            if token is not None:
                second_attempted.set()
                if lease_window_elapsed.is_set() and renew_calls < 2:
                    token = candidate
                    return True
                return False
            token = candidate
            return True

        async def eval(self, script, _count, _key, candidate, *_args):
            nonlocal token, renew_calls
            if "expire" in script:
                if token != candidate:
                    return 0
                renew_calls += 1
                if renew_calls >= 2:
                    renewed_twice.set()
                return 1
            if token == candidate:
                token = None
                return 1
            return 0

        async def aclose(self):
            return None

    fake_redis = _FakeRedis()
    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", lambda: fake_redis)
    first_entered = asyncio.Event()

    async def _worker(name: str) -> None:
        nonlocal active, max_active
        if name == "second":
            await first_entered.wait()
        async with byok_runtime._openai_oauth_refresh_lock(user_id=29, provider="openai"):
            active += 1
            max_active = max(max_active, active)
            if name == "first":
                first_entered.set()
                await allow_first_exit.wait()
            active -= 1

    tasks = [
        asyncio.create_task(_worker("first")),
        asyncio.create_task(_worker("second")),
    ]
    try:
        await asyncio.wait_for(second_attempted.wait(), timeout=1)
        await asyncio.wait_for(renewed_twice.wait(), timeout=1)
        lease_window_elapsed.set()
        await asyncio.sleep(0.06)
        assert max_active == 1
    finally:
        allow_first_exit.set()
        await asyncio.gather(*tasks)

    assert renew_calls >= 2
    assert max_active == 1


@pytest.mark.asyncio
async def test_redis_oauth_refresh_lock_fails_closed_when_lease_is_lost(monkeypatch):
    """A worker must stop protected refresh work after losing Redis ownership."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        0.005,
        raising=False,
    )
    body_entered = asyncio.Event()
    body_completed = False

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            if "expire" in script:
                return 0
            return 0

        async def aclose(self):
            return None

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    async def _worker() -> None:
        nonlocal body_completed
        async with byok_runtime._openai_oauth_refresh_lock(user_id=30, provider="openai"):
            body_entered.set()
            await asyncio.sleep(1)
            body_completed = True

    task = asyncio.create_task(_worker())
    await body_entered.wait()
    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await asyncio.wait_for(task, timeout=1)

    assert exc_info.value.code == "credential_store_unavailable"
    assert not body_completed


@pytest.mark.asyncio
async def test_redis_oauth_refresh_lock_fails_after_body_suppresses_lease_cancellation(
    monkeypatch,
):
    """Suppressing cancellation in protected work cannot hide lost ownership."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        0.005,
        raising=False,
    )
    cancellation_suppressed = asyncio.Event()

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            return 0 if "expire" in script else 1

        async def aclose(self):
            return None

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(user_id=33, provider="openai"):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancellation_suppressed.set()

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await asyncio.wait_for(_worker(), timeout=1)

    assert cancellation_suppressed.is_set()
    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("release_mode", ("not_owner", "transport_error"))
async def test_redis_oauth_refresh_lock_requires_confirmed_final_release(
    monkeypatch,
    release_mode: str,
):
    """A normal body result is unsafe unless Redis confirms final ownership."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        3600,
        raising=False,
    )
    closed = False

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            if "expire" in script:
                return 1
            if release_mode == "transport_error":
                raise ConnectionError("release transport failed")
            return 0

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime._openai_oauth_refresh_lock(
            user_id=38,
            provider="openai",
        ):
            pass

    assert closed
    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
async def test_redis_oauth_refresh_lock_fails_if_ownership_is_lost_after_body(
    monkeypatch,
):
    """Ownership loss between body return and final release must fail closed."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    monkeypatch.setattr(
        byok_runtime,
        "OPENAI_OAUTH_REFRESH_LOCK_RENEW_INTERVAL_SECONDS",
        3600,
        raising=False,
    )
    body_returned = asyncio.Event()
    release_started = asyncio.Event()
    allow_release_result = asyncio.Event()

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, script, _count, _key, _token, *_args):
            if "expire" in script:
                return 1
            release_started.set()
            await allow_release_result.wait()
            return 0

        async def aclose(self):
            return None

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(
            user_id=39,
            provider="openai",
        ):
            body_returned.set()

    task = asyncio.create_task(_worker())
    await body_returned.wait()
    await release_started.wait()
    allow_release_result.set()

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await task

    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
async def test_redis_oauth_refresh_lock_releases_uncertain_cancelled_acquisition(monkeypatch):
    """Cancellation after Redis accepted SET must still release by ownership token."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    ownership_written = asyncio.Event()
    release_called = asyncio.Event()
    closed = False

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            ownership_written.set()
            await asyncio.Event().wait()

        async def eval(self, _script, _count, _key, _token, *_args):
            release_called.set()
            return 1

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(byok_runtime, "_openai_oauth_redis_client", _FakeRedis)

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(user_id=32, provider="openai"):
            raise AssertionError("cancelled acquisition entered protected body")

    task = asyncio.create_task(_worker())
    await ownership_written.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert release_called.is_set()
    assert closed


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_postgres_oauth_refresh_lock_finishes_unlock_before_cancellation_returns(
    monkeypatch,
):
    """Cancellation cannot return a PostgreSQL session while its lock is held."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    entered = asyncio.Event()
    leave_body = asyncio.Event()
    unlock_started = asyncio.Event()
    allow_unlock = asyncio.Event()
    connection_returned = False
    unlocked = False

    class _FakeConnection:
        async def fetchval(self, query: str, *_args):
            nonlocal unlocked
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                unlock_started.set()
                await allow_unlock.wait()
                unlocked = True
                return True
            raise AssertionError(f"Unexpected advisory-lock query: {query}")

    class _FakePool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            nonlocal connection_returned
            try:
                yield _FakeConnection()
            finally:
                connection_returned = True

    async def _get_db_pool():
        return _FakePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(user_id=31, provider="openai"):
            entered.set()
            await leave_body.wait()

    task = asyncio.create_task(_worker())
    await entered.wait()
    leave_body.set()
    await unlock_started.wait()
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert not connection_returned
    finally:
        allow_unlock.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert unlocked
    assert connection_returned


@pytest.mark.asyncio
async def test_postgres_oauth_refresh_lock_returns_connection_when_unlock_fails(monkeypatch):
    """A broken unlock cannot skip returning the session to its pool context."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    connection_returned = False

    class _FakeConnection:
        async def fetchval(self, query: str, *_args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                raise RuntimeError("unlock transport failed")
            raise AssertionError(f"Unexpected advisory-lock query: {query}")

    class _FakePool:
        backend_type = "postgres"

        @contextlib.asynccontextmanager
        async def acquire_openai_credential_lock_connection(self, *, timeout=None):
            nonlocal connection_returned
            try:
                yield _FakeConnection()
            finally:
                connection_returned = True

    async def _get_db_pool():
        return _FakePool()

    monkeypatch.setattr(byok_runtime, "get_db_pool", _get_db_pool)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        async with byok_runtime._openai_oauth_refresh_lock(
            user_id=34,
            provider="openai",
        ):
            pass

    assert connection_returned
    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.asyncio
async def test_database_pool_release_finishes_before_cancellation_returns():
    """Cancelling a holder cannot interrupt return of its PostgreSQL session."""
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    released = False
    connection = object()

    class _FakeAsyncpgPool:
        async def acquire(self, *, timeout=None):
            return connection

        async def release(self, candidate, *, timeout=None):
            nonlocal released
            assert candidate is connection
            release_started.set()
            await allow_release.wait()
            released = True

    database_pool = object.__new__(DatabasePool)
    database_pool._initialized = True
    database_pool.pool = _FakeAsyncpgPool()

    async def _worker() -> None:
        async with database_pool.acquire(timeout=1):
            pass

    task = asyncio.create_task(_worker())
    await release_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not released
    allow_release.set()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert released


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_redis_oauth_refresh_lock_finishes_release_before_cancellation_returns(
    monkeypatch,
):
    """Cancellation cannot close Redis before its ownership token is released."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "redis")
    entered = asyncio.Event()
    leave_body = asyncio.Event()
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    released = False
    closed = False

    class _FakeRedis:
        async def set(self, _key, _token, **_kwargs):
            return True

        async def eval(self, _script, _count, _key, _token):
            nonlocal released
            release_started.set()
            await allow_release.wait()
            released = True
            return 1

        async def aclose(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        byok_runtime,
        "_openai_oauth_redis_client",
        lambda: _FakeRedis(),
    )

    async def _worker() -> None:
        async with byok_runtime._openai_oauth_refresh_lock(user_id=37, provider="openai"):
            entered.set()
            await leave_body.wait()

    task = asyncio.create_task(_worker())
    await entered.wait()
    leave_body.set()
    await release_started.wait()
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert not closed
    finally:
        allow_release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert released
    assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope_type", "team_ids", "org_ids", "scope_id"),
    [
        ("team", [41], [], 41),
        ("org", [], [43], 43),
    ],
)
async def test_default_shared_resolution_reauthorizes_active_membership(
    monkeypatch,
    scope_type,
    team_ids,
    org_ids,
    scope_id,
) -> None:
    """Candidate scope IDs never authorize a shared credential by themselves."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    class _UserRepo:
        async def fetch_secret_for_active_user(self, *_args, **_kwargs):
            return None

        async def fetch_secret_for_user(self, *_args, **_kwargs):
            return None

    class _SharedRepo:
        async def fetch_secret(self, *_args, **_kwargs):
            raise AssertionError("default resolution must use the authorized read")

        async def fetch_authorized_secret_for_user(
            self,
            actual_scope_type,
            actual_scope_id,
            user_id,
            provider,
        ):
            assert (actual_scope_type, actual_scope_id) == (scope_type, scope_id)
            assert user_id == 7
            assert provider == "openai"
            return {"revoked_at": None, "last_used_at": None}

    async def _get_user_repo():
        return _UserRepo()

    async def _get_shared_repo():
        return _SharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _get_shared_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "_extract_payload",
        lambda _row, _provider: {"api_key": "current-shared-key"},
    )

    resolution = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=7,
        team_ids=team_ids,
        org_ids=org_ids,
        server_config_snapshot={},
    )

    assert resolution.source == scope_type
    assert resolution.api_key == "current-shared-key"


@pytest.mark.asyncio
async def test_required_team_source_skips_user_credentials_before_shared_lookup(
    monkeypatch,
) -> None:
    """Source-bound resolution cannot refresh or otherwise touch a new user key."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    user_lookup_called = False

    async def _unexpected_user_repo():
        nonlocal user_lookup_called
        user_lookup_called = True
        raise AssertionError("required team source must skip the user repository")

    class _SharedRepo:
        async def fetch_authorized_secret_for_user(
            self,
            scope_type,
            scope_id,
            user_id,
            provider,
        ):
            assert (scope_type, scope_id, provider) == ("team", 41, "openai")
            assert user_id == 7
            return {"revoked_at": None, "last_used_at": None}

    async def _get_shared_repo():
        return _SharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _unexpected_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _get_shared_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(
        byok_runtime,
        "_extract_payload",
        lambda _row, _provider: {"api_key": "current-team-key"},
    )

    resolution = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=7,
        team_ids=[41],
        org_ids=[],
        required_source="team",
        server_config_snapshot={},
    )

    assert user_lookup_called is False
    assert resolution.source == "team"
    assert resolution.api_key == "current-team-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("required_source", "team_ids", "org_ids", "scope_type", "scope_id"),
    [
        ("team", [41], [], "team", 41),
        ("org", [], [43], "org", 43),
    ],
)
async def test_required_shared_source_revocation_never_uses_legacy_or_server_fallback(
    monkeypatch,
    required_source,
    team_ids,
    org_ids,
    scope_type,
    scope_id,
) -> None:
    """A revoked authorized row is terminal for its bound shared source."""

    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    extracted = False
    fallback_called = False

    class _SharedRepo:
        async def fetch_authorized_secret_for_user(
            self,
            actual_scope_type,
            actual_scope_id,
            user_id,
            provider,
        ):
            assert (actual_scope_type, actual_scope_id) == (scope_type, scope_id)
            assert user_id == 7
            assert provider == "custom-openai-api"
            return {
                "provider": "custom-openai-api",
                "encrypted_blob": "revoked-canonical-must-not-decrypt",
                "revoked_at": datetime.now(timezone.utc),
                "last_used_at": None,
            }

    async def _get_shared_repo():
        return _SharedRepo()

    def _unexpected_extract(*_args, **_kwargs):
        nonlocal extracted
        extracted = True
        raise AssertionError("revoked shared credentials must not be decrypted")

    def _unexpected_fallback(_provider):
        nonlocal fallback_called
        fallback_called = True
        return "server-key-must-not-be-used"

    monkeypatch.setattr(byok_runtime, "_get_org_repo", _get_shared_repo)
    monkeypatch.setattr(byok_runtime, "_extract_payload", _unexpected_extract)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai-compatible",
            user_id=7,
            team_ids=team_ids,
            org_ids=org_ids,
            required_source=required_source,
            fallback_resolver=_unexpected_fallback,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert extracted is False
    assert fallback_called is False


@pytest.mark.asyncio
async def test_required_user_source_never_falls_through_to_shared_credentials(
    monkeypatch,
) -> None:
    """A missing bound user key uses the active-user read and skips shared scopes."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    unrestricted_lookups = 0

    class _UserRepo:
        async def fetch_secret_for_active_user(
            self,
            _user_id,
            _provider,
            *,
            include_revoked=False,
        ):
            assert include_revoked is True
            return None

        async def fetch_secret_for_user(
            self,
            _user_id,
            _provider,
            *,
            include_revoked=False,
        ):
            nonlocal unrestricted_lookups
            assert include_revoked is True
            unrestricted_lookups += 1
            return None

    async def _get_user_repo():
        return _UserRepo()

    async def _unexpected_shared_repo():
        raise AssertionError("required user source must skip shared repositories")

    monkeypatch.setattr(byok_runtime, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", _unexpected_shared_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    resolution = await byok_runtime.resolve_byok_credentials(
        "openai",
        user_id=7,
        team_ids=[],
        org_ids=[],
        required_source="user",
        fallback_override=_server_fallback_value(
            api_key=None,
            credential_fields={},
            app_config={},
        ),
        server_config_snapshot={},
    )

    assert resolution.source == "none"
    assert resolution.api_key is None
    assert unrestricted_lookups == 1


@pytest.mark.asyncio
async def test_invalid_required_credential_source_fails_closed(monkeypatch) -> None:
    """The source constraint is exact and cannot silently become unbound."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)

    with pytest.raises(byok_runtime.ByokResolutionError) as exc_info:
        await byok_runtime.resolve_byok_credentials(
            "openai",
            user_id=7,
            required_source="invalid",
            server_config_snapshot={},
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
