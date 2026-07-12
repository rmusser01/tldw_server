from datetime import datetime, timezone

import pytest
from fastapi import Request

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    encrypt_byok_payload,
    dumps_envelope,
    key_hint_for_api_key,
)

from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import _setup_byok_sqlite


async def _upsert_user_key(repo, user_id, provider, api_key, credential_fields=None):
    payload = build_secret_payload(api_key, credential_fields=credential_fields)
    envelope = encrypt_byok_payload(payload)
    encrypted_blob = dumps_envelope(envelope)
    await repo.upsert_secret(
        user_id=user_id,
        provider=provider,
        encrypted_blob=encrypted_blob,
        key_hint=key_hint_for_api_key(api_key),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )


async def _upsert_shared_key(repo, scope_type, scope_id, provider, api_key, credential_fields=None):
    payload = build_secret_payload(api_key, credential_fields=credential_fields)
    envelope = encrypt_byok_payload(payload)
    encrypted_blob = dumps_envelope(envelope)
    await repo.upsert_secret(
        scope_type=scope_type,
        scope_id=scope_id,
        provider=provider,
        encrypted_blob=encrypted_blob,
        key_hint=key_hint_for_api_key(api_key),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )


def _make_request(principal: AuthPrincipal) -> Request:
    scope = {"type": "http", "method": "GET", "path": "/"}
    request = Request(scope)
    request.state.auth = AuthContext(
        principal=principal,
        ip="127.0.0.1",
        user_agent="pytest",
        request_id="byok-test",
    )
    return request


@pytest.mark.asyncio
async def test_byok_resolution_precedence(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    pool = state["pool"]

    user_repo = AuthnzUserProviderSecretsRepo(pool)
    org_repo = AuthnzOrgProviderSecretsRepo(pool)

    await _upsert_user_key(
        user_repo,
        user_id,
        "openai",
        "sk-user-openai-1111",
        credential_fields={"base_url": "https://example.com/v1"},
    )
    await _upsert_shared_key(
        org_repo,
        "team",
        team_id,
        "openai",
        "sk-team-openai-2222",
    )
    await _upsert_shared_key(
        org_repo,
        "org",
        org_id,
        "openai",
        "sk-org-openai-3333",
    )

    request = _make_request(
        AuthPrincipal(
            kind="user",
            user_id=user_id,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=[],
            is_admin=True,
            org_ids=[org_id],
            team_ids=[team_id],
            active_org_id=org_id,
            active_team_id=team_id,
        )
    )

    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
    )
    assert resolved.source == "user"
    assert resolved.status == "RESOLVED"
    assert resolved.api_key == "sk-user-openai-1111"
    assert resolved.app_config
    assert resolved.app_config["openai_api"]["api_base_url"] == "https://example.com/v1"

    # Remove user key to validate shared precedence (team before org)
    await user_repo.delete_secret(user_id, "openai")
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
    )
    assert resolved.source == "team"
    assert resolved.status == "RESOLVED"
    assert resolved.api_key == "sk-team-openai-2222"

    # Remove team key to fall back to org shared key
    await org_repo.delete_secret("team", team_id, "openai")
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
    )
    assert resolved.source == "org"
    assert resolved.status == "RESOLVED"
    assert resolved.api_key == "sk-org-openai-3333"

    # Remove the org key to validate server precedence.
    await org_repo.delete_secret("org", org_id, "openai")
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
        fallback_resolver=lambda _provider: "sk-server-openai-4444",
    )
    assert resolved.source == "server_default"
    assert resolved.status == "RESOLVED"
    assert resolved.api_key == "sk-server-openai-4444"

    # Successful not-found queries at every scope produce explicit absence.
    monkeypatch.setattr(byok_runtime, "resolve_server_default_key", lambda _provider: None)
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
        fallback_resolver=lambda _provider: None,
    )
    assert resolved.source == "none"
    assert resolved.status == "ABSENT"
    assert resolved.api_key is None


@pytest.mark.asyncio
async def test_byok_resolution_base_url_requires_trusted_request(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    pool = state["pool"]

    user_repo = AuthnzUserProviderSecretsRepo(pool)
    await _upsert_user_key(
        user_repo,
        user_id,
        "openai",
        "sk-user-openai-1111",
        credential_fields={"base_url": "https://example.com/v1"},
    )

    request = _make_request(
        AuthPrincipal(
            kind="user",
            user_id=user_id,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=[],
            is_admin=False,
            org_ids=[org_id],
            team_ids=[team_id],
            active_org_id=org_id,
            active_team_id=team_id,
        )
    )

    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
    )

    base_url = ((resolved.app_config or {}).get("openai_api") or {}).get("api_base_url")
    assert base_url != "https://example.com/v1"

    # A supplied authority decision is authoritative over the legacy request path.
    monkeypatch.setattr(byok_runtime, "is_trusted_base_url_request", lambda _request: True)
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
        trusted_base_url_override=False,
    )
    base_url = ((resolved.app_config or {}).get("openai_api") or {}).get("api_base_url")
    assert base_url != "https://example.com/v1"

    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
        trusted_base_url_override=True,
    )
    base_url = ((resolved.app_config or {}).get("openai_api") or {}).get("api_base_url")
    assert base_url == "https://example.com/v1"


@pytest.mark.asyncio
async def test_byok_resolution_respects_allowlist(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    pool = state["pool"]

    user_repo = AuthnzUserProviderSecretsRepo(pool)
    await _upsert_user_key(user_repo, user_id, "openai", "sk-user-openai-9999")

    monkeypatch.setenv("BYOK_ALLOWED_PROVIDERS", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-openai-0000")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()

    resolved = await resolve_byok_credentials("openai", user_id=user_id)
    assert resolved.source == "server_default"
    assert resolved.api_key == "sk-server-openai-0000"


@pytest.mark.asyncio
async def test_byok_resolution_emits_metrics(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    pool = state["pool"]

    user_repo = AuthnzUserProviderSecretsRepo(pool)
    await _upsert_user_key(user_repo, user_id, "openai", "sk-user-openai-1111")

    reg = get_metrics_registry()
    labels = {
        "provider": "openai",
        "source": "user",
        "allowlisted": "true",
        "byok_enabled": "true",
    }
    before = reg.get_metric_stats("byok_resolution_total", labels=labels).get("count", 0)

    resolved = await resolve_byok_credentials("openai", user_id=user_id)
    assert resolved.source == "user"

    after = reg.get_metric_stats("byok_resolution_total", labels=labels).get("count", 0)
    assert after >= before + 1
