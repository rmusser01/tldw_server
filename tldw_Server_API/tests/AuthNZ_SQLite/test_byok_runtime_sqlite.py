import asyncio
from datetime import datetime, timezone

import pytest
from fastapi import Request

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
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

from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import (
    _insert_raw_shared_key,
    _setup_byok_sqlite,
)


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

    # Remove the row to validate absence precedence (team before org).
    await pool.execute(
        "DELETE FROM user_provider_secrets WHERE user_id = ? AND provider = ?",
        (user_id, "openai"),
    )
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

    # Remove the row to fall back to the org shared key.
    await pool.execute(
        "DELETE FROM org_provider_secrets WHERE scope_type = ? AND scope_id = ? AND provider = ?",
        ("team", team_id, "openai"),
    )
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

    # Remove the row to validate server precedence.
    await pool.execute(
        "DELETE FROM org_provider_secrets WHERE scope_type = ? AND scope_id = ? AND provider = ?",
        ("org", org_id, "openai"),
    )
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


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_revoked_selected_shared_scope_blocks_alias_and_lower_fallback(
    tmp_path,
    monkeypatch,
    scope_type,
):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    scope_id = team_id if scope_type == "team" else org_id
    revoked_at = datetime.now(timezone.utc).isoformat()
    await _insert_raw_shared_key(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        api_key=f"sk-{scope_type}-revoked-1111",
        revoked_at=revoked_at,
    )
    await _insert_raw_shared_key(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        api_key=f"sk-{scope_type}-legacy-2222",
    )
    if scope_type == "team":
        await _upsert_shared_key(
            AuthnzOrgProviderSecretsRepo(state["pool"]),
            "org",
            org_id,
            "openai",
            "sk-org-lower-3333",
        )

    fallback_calls: list[str] = []

    def fallback(provider: str) -> str:
        fallback_calls.append(provider)
        return "sk-server-lower-4444"

    with pytest.raises(ByokResolutionError) as exc_info:
        await resolve_byok_credentials(
            "oai",
            user_id=user_id,
            team_ids=[team_id] if scope_type == "team" else [],
            org_ids=[org_id],
            fallback_resolver=fallback,
        )

    assert exc_info.value.code == "invalid_provider_credentials"
    assert fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_runtime_coalesces_accepted_underscore_alias_and_canonical_resolution(
    tmp_path,
    monkeypatch,
):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    await _insert_raw_shared_key(
        state["pool"],
        scope_type="team",
        scope_id=team_id,
        provider="aws_bedrock",
        api_key="sk-team-alias-5555",
    )
    await _upsert_shared_key(
        AuthnzOrgProviderSecretsRepo(state["pool"]),
        "org",
        org_id,
        "bedrock",
        "sk-org-lower-6666",
    )

    entered = asyncio.Event()
    release = asyncio.Event()
    second_started = asyncio.Event()
    resolver_calls: list[str] = []
    fallback_calls: list[str] = []

    async def gated_resolver(provider: str, **kwargs):
        resolver_calls.append(provider)
        entered.set()
        await release.wait()
        return await resolve_byok_credentials(provider, **kwargs)

    def fallback(provider: str) -> str:
        fallback_calls.append(provider)
        return "sk-server-lower-7777"

    runtime = ProviderCredentialRuntime(
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        trusted_base_url_override=False,
        fallback_resolver=fallback,
        resolver=gated_resolver,
    )

    async def resolve_canonical():
        second_started.set()
        return await runtime.resolve("bedrock")

    try:
        alias_task = asyncio.create_task(runtime.resolve("aws_bedrock"))
        await entered.wait()
        canonical_task = asyncio.create_task(resolve_canonical())
        await second_started.wait()
        release.set()
        alias_handle, canonical_handle = await asyncio.gather(alias_task, canonical_task)

        assert resolver_calls == ["bedrock"]
        assert alias_handle.api_key == "sk-team-alias-5555"
        assert canonical_handle.api_key == "sk-team-alias-5555"
        assert fallback_calls == []
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_revoked_shared_tombstone_check_wins_before_lower_scope(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    await _insert_raw_shared_key(
        state["pool"],
        scope_type="team",
        scope_id=team_id,
        provider="openai",
        api_key="sk-team-revoked-8888",
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )
    await _upsert_shared_key(
        AuthnzOrgProviderSecretsRepo(state["pool"]),
        "org",
        org_id,
        "openai",
        "sk-org-lower-9999",
    )

    real_repo = AuthnzOrgProviderSecretsRepo(state["pool"])
    tombstone_check_entered = asyncio.Event()
    allow_tombstone_check = asyncio.Event()
    lower_scope_reached = asyncio.Event()

    class GatedSharedRepo:
        async def fetch_secret(self, scope_type, scope_id, provider, *, include_revoked=False):
            if scope_type == "team" and include_revoked:
                tombstone_check_entered.set()
                await allow_tombstone_check.wait()
            if scope_type == "org":
                lower_scope_reached.set()
            return await real_repo.fetch_secret(
                scope_type,
                scope_id,
                provider,
                include_revoked=include_revoked,
            )

    async def get_gated_repo():
        return GatedSharedRepo()

    monkeypatch.setattr(byok_runtime, "_get_org_repo", get_gated_repo)
    resolution_task = asyncio.create_task(
        resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[team_id],
            org_ids=[org_id],
            fallback_resolver=lambda _provider: "sk-server-lower-0000",
        )
    )
    tombstone_wait = asyncio.create_task(tombstone_check_entered.wait())
    lower_wait = asyncio.create_task(lower_scope_reached.wait())
    done, pending = await asyncio.wait(
        {tombstone_wait, lower_wait},
        timeout=5,
        return_when=asyncio.FIRST_COMPLETED,
    )
    tombstone_won = tombstone_wait in done and lower_wait not in done
    allow_tombstone_check.set()
    resolution_error = None
    try:
        await resolution_task
    except ByokResolutionError as exc:
        resolution_error = exc
    finally:
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    assert tombstone_won
    assert resolution_error is not None
    assert resolution_error.code == "invalid_provider_credentials"
    assert not lower_scope_reached.is_set()


@pytest.mark.asyncio
async def test_team_alias_credentials_cross_real_openai_adapter_boundary(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    team_key = "sk-team-boundary-1212"
    org_key = "sk-org-sentinel-3434"
    server_key = "sk-server-sentinel-5656"
    await _insert_raw_shared_key(
        state["pool"],
        scope_type="team",
        scope_id=team_id,
        provider="oai",
        api_key=team_key,
    )
    await _upsert_shared_key(
        AuthnzOrgProviderSecretsRepo(state["pool"]),
        "org",
        org_id,
        "openai",
        org_key,
    )

    captured_headers: list[dict[str, str]] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class FakeStreamResponse(FakeResponse):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            return [
                b'data: {"choices":[{"delta":{"content":"ok"}}]}',
                b"data: [DONE]",
            ]

        def close(self):
            return None

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _url, *, headers, json):
            del json
            captured_headers.append(dict(headers))
            return FakeResponse()

        def stream(self, _method, _url, *, headers, json):
            del json
            captured_headers.append(dict(headers))
            return FakeStreamResponse()

    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter as adapter_module

    monkeypatch.setattr(adapter_module, "http_client_factory", lambda **_kwargs: FakeClient())
    fallback_calls: list[str] = []

    def fallback(provider: str) -> str:
        fallback_calls.append(provider)
        return server_key

    runtime = ProviderCredentialRuntime(
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        trusted_base_url_override=False,
        fallback_resolver=fallback,
    )
    try:
        handle = await runtime.resolve("oai")
        request = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-test",
            "api_key": handle.api_key,
            "app_config": handle.app_config,
        }
        adapter = adapter_module.OpenAIAdapter()
        response = adapter.chat(dict(request))
        stream_items = list(adapter.stream(dict(request)))
    finally:
        await runtime.close()

    assert response["choices"][0]["message"]["content"] == "ok"
    assert stream_items[-1] == "data: [DONE]\n\n"
    assert [headers["Authorization"] for headers in captured_headers] == [
        f"Bearer {team_key}",
        f"Bearer {team_key}",
    ]
    assert all(org_key not in str(headers) for headers in captured_headers)
    assert all(server_key not in str(headers) for headers in captured_headers)
    assert fallback_calls == []
