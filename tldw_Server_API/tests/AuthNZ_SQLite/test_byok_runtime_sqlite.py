import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Request

from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
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


async def _set_user_active(pool, *, user_id: int, is_active: bool) -> None:
    async with pool.transaction() as conn:
        await VersionedUserWriteGateway("sqlite").execute_update(
            conn,
            user_id=user_id,
            profile_visible_fields=("is_active",),
            statement="UPDATE users SET is_active = ? WHERE id = ?",
            parameters=(is_active, user_id),
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


class _SQLiteAliasReadGatePool:
    """Place a revocation exactly between legacy split reads, if they exist."""

    pool = None

    def __init__(self, delegate, *, table: str) -> None:
        self.delegate = delegate
        self.table = table
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()
        self.alias_statement_count = 0
        self.last_alias_query = ""
        self._gated = False

    def _is_alias_read(self, query: str) -> bool:
        normalized = " ".join(query.split()).lower()
        return (
            normalized.startswith("select")
            and self.table in normalized
            and "provider" in normalized
        )

    async def fetchone(self, query: str, params=()):
        if not self._is_alias_read(query):
            return await self.delegate.fetchone(query, params)

        self.alias_statement_count += 1
        self.last_alias_query = query
        row = await self.delegate.fetchone(query, params)
        if not self._gated:
            # The legacy implementation has already observed the canonical
            # spelling as absent when the writer is released here.
            self._gated = True
            self.read_started.set()
            await self.release_read.wait()
        return row

    async def fetchall(self, query: str, params=()):
        if not self._is_alias_read(query):
            return await self.delegate.fetchall(query, params)

        self.alias_statement_count += 1
        self.last_alias_query = query
        if not self._gated:
            # A set-based implementation waits before its one snapshot, so it
            # must observe the committed canonical tombstone.
            self._gated = True
            self.read_started.set()
            await self.release_read.wait()
        return await self.delegate.fetchall(query, params)

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


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


async def _insert_raw_user_payload(pool, *, user_id: int, provider: str, payload: dict) -> None:
    """Insert a legacy provider spelling without repository canonicalization."""
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        """
        INSERT INTO user_provider_secrets (
            user_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            provider,
            dumps_envelope(encrypt_byok_payload(payload)),
            "oauth" if payload.get("credential_version") == 2 else key_hint_for_api_key(
                str(payload.get("api_key") or "")
            ),
            now,
            now,
        ),
    )


@pytest.mark.asyncio
async def test_byok_resolution_precedence(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    monkeypatch.setenv("BYOK_LAST_USED_THROTTLE_SECONDS", "0")
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
    await resolved.touch_last_used()
    user_row = await user_repo.fetch_secret_for_user(user_id, "openai")
    assert user_row is not None
    assert user_row["last_used_at"] is not None

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
    await resolved.touch_last_used()
    team_row = await org_repo.fetch_secret("team", team_id, "openai")
    assert team_row is not None
    assert team_row["last_used_at"] is not None

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
    await resolved.touch_last_used()
    org_row = await org_repo.fetch_secret("org", org_id, "openai")
    assert org_row is not None
    assert org_row["last_used_at"] is not None

    # Remove the final BYOK key to prove the complete precedence chain reaches
    # the server-secret boundary without manufacturing a repository fallback.
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
    assert resolved.api_key == "sk-server-openai-4444"

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
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        request=request,
        server_config_snapshot={},
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
    async def gated_resolver(provider: str, **kwargs):
        resolver_calls.append(provider)
        entered.set()
        await release.wait()
        return await resolve_byok_credentials(provider, **kwargs)

    runtime = ProviderCredentialRuntime(
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        trusted_base_url_override=False,
        server_config_snapshot={
            "bedrock_api": {"api_key": "sk-server-lower-7777"}
        },
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
        async def fetch_authorized_secret_for_user(
            self,
            scope_type,
            scope_id,
            authorized_user_id,
            provider,
        ):
            assert authorized_user_id == user_id
            if scope_type == "team":
                tombstone_check_entered.set()
                await allow_tombstone_check.wait()
            if scope_type == "org":
                lower_scope_reached.set()
            return await real_repo.fetch_authorized_secret_for_user(
                scope_type,
                scope_id,
                authorized_user_id,
                provider,
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
    runtime = ProviderCredentialRuntime(
        user_id=user_id,
        team_ids=[team_id],
        org_ids=[org_id],
        trusted_base_url_override=False,
        server_config_snapshot={"openai_api": {"api_key": server_key}},
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


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("owner_kind", ["user", "team"])
async def test_alias_revoke_interleaving_fails_closed_before_openai_adapter_sqlite(
    tmp_path,
    monkeypatch,
    owner_kind,
):
    """A canonical tombstone created mid-lookup cannot expose a lower key."""
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])
    legacy_key = f"sk-{owner_kind}-legacy-1111"
    lower_key = f"sk-{owner_kind}-lower-sentinel-2222"
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)

    if owner_kind == "user":
        await _insert_raw_user_payload(
            pool,
            user_id=user_id,
            provider="oai",
            payload=build_secret_payload(legacy_key),
        )
        gate_pool = _SQLiteAliasReadGatePool(
            pool,
            table="user_provider_secrets",
        )
        gated_repo = AuthnzUserProviderSecretsRepo(gate_pool)
        writer_repo = AuthnzUserProviderSecretsRepo(pool)

        async def get_gated_user_repo():
            return gated_repo

        monkeypatch.setattr(byok_runtime, "_get_user_repo", get_gated_user_repo)
        resolve_kwargs = {
            "team_ids": [],
            "org_ids": [],
            "required_source": "user",
            "server_config_snapshot": {"openai_api": {"api_key": lower_key}},
        }

        async def revoke():
            return await writer_repo.delete_secret(
                user_id,
                "openai",
                revoked_by=user_id,
            )

        final_rows_sql = (
            "SELECT provider, revoked_at FROM user_provider_secrets "
            "WHERE user_id = ? ORDER BY provider"
        )
        final_rows_params = (user_id,)
    else:
        await _insert_raw_shared_key(
            pool,
            scope_type="team",
            scope_id=team_id,
            provider="oai",
            api_key=legacy_key,
        )
        await _upsert_shared_key(
            AuthnzOrgProviderSecretsRepo(pool),
            "org",
            org_id,
            "openai",
            lower_key,
        )
        gate_pool = _SQLiteAliasReadGatePool(
            pool,
            table="org_provider_secrets",
        )
        gated_repo = AuthnzOrgProviderSecretsRepo(gate_pool)
        writer_repo = AuthnzOrgProviderSecretsRepo(pool)

        async def get_gated_org_repo():
            return gated_repo

        monkeypatch.setattr(byok_runtime, "_get_org_repo", get_gated_org_repo)
        resolve_kwargs = {
            "team_ids": [team_id],
            "org_ids": [org_id],
            "server_config_snapshot": {"openai_api": {"api_key": lower_key}},
        }

        async def revoke():
            return await writer_repo.delete_secret(
                "team",
                team_id,
                "openai",
                revoked_by=user_id,
            )

        final_rows_sql = (
            "SELECT provider, revoked_at FROM org_provider_secrets "
            "WHERE scope_type = ? AND scope_id = ? ORDER BY provider"
        )
        final_rows_params = ("team", team_id)

    async def resolve_and_dispatch():
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            **resolve_kwargs,
        )
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    resolution_task = asyncio.create_task(resolve_and_dispatch())
    await asyncio.wait_for(gate_pool.read_started.wait(), timeout=5)
    try:
        assert await revoke()
    finally:
        gate_pool.release_read.set()

    with pytest.raises(ByokResolutionError) as exc_info:
        await asyncio.wait_for(resolution_task, timeout=5)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert gate_pool.alias_statement_count == 1
    assert "provider IN (" in gate_pool.last_alias_query
    assert captured_headers == []
    rows = await pool.fetchall(final_rows_sql, final_rows_params)
    assert [(row["provider"], row["revoked_at"] is not None) for row in rows] == [
        ("openai", True)
    ]


@pytest.mark.asyncio
async def test_legacy_oai_oauth_refresh_migrates_before_openai_adapter_sqlite(
    tmp_path,
    monkeypatch,
):
    """A legacy OAuth row refreshes once and persists under the canonical name."""
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path / "locks"))
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    original_payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "expired-access-token",
                "refresh_token": "legacy-refresh-token",
                "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            }
        },
    }
    await _insert_raw_user_payload(
        pool,
        user_id=user_id,
        provider="oai",
        payload=original_payload,
    )
    refresh_tokens: list[str] = []

    async def refresh_token(**kwargs):
        refresh_tokens.append(str(kwargs["refresh_token"]))
        return {
            "access_token": "fresh-access-token",
            "refresh_token": "fresh-refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[],
        org_ids=[],
        server_config_snapshot={"openai_api": {"api_key": "sk-server-sentinel"}},
    )
    response = adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-test",
            "api_key": resolved.api_key,
            "app_config": resolved.app_config,
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert refresh_tokens == ["legacy-refresh-token"]
    assert [headers["Authorization"] for headers in captured_headers] == [
        "Bearer fresh-access-token"
    ]
    rows = await pool.fetchall(
        """
        SELECT provider, encrypted_blob, revoked_at
        FROM user_provider_secrets
        WHERE user_id = ?
        ORDER BY provider
        """,
        (user_id,),
    )
    assert [row["provider"] for row in rows] == ["openai"]
    assert rows[0]["revoked_at"] is None
    stored_payload = decrypt_byok_payload(loads_envelope(rows[0]["encrypted_blob"]))
    assert stored_payload["credentials"]["oauth"]["access_token"] == "fresh-access-token"
    assert stored_payload["credentials"]["oauth"]["refresh_token"] == "fresh-refresh-token"


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_legacy_oai_oauth_refresh_cannot_resurrect_revoke_sqlite(
    tmp_path,
    monkeypatch,
):
    """A revoke committed during refresh wins over canonicalizing CAS."""
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_DIR", str(tmp_path / "locks"))
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    await _insert_raw_user_payload(
        pool,
        user_id=user_id,
        provider="oai",
        payload={
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        },
    )
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_calls = 0

    async def refresh_token(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        refresh_started.set()
        await release_refresh.wait()
        return {
            "access_token": "must-not-be-persisted",
            "refresh_token": "must-not-resurrect",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)

    async def resolve_and_dispatch():
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[],
            org_ids=[],
            server_config_snapshot={},
        )
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    resolution_task = asyncio.create_task(resolve_and_dispatch())
    await asyncio.wait_for(refresh_started.wait(), timeout=5)
    repo = AuthnzUserProviderSecretsRepo(pool)
    try:
        assert await repo.delete_secret(user_id, "openai", revoked_by=user_id)
    finally:
        release_refresh.set()

    with pytest.raises(ByokResolutionError) as exc_info:
        await asyncio.wait_for(resolution_task, timeout=5)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert refresh_calls == 1
    assert captured_headers == []
    rows = await pool.fetchall(
        """
        SELECT provider, encrypted_blob, revoked_at
        FROM user_provider_secrets
        WHERE user_id = ?
        ORDER BY provider
        """,
        (user_id,),
    )
    assert [row["provider"] for row in rows] == ["openai"]
    assert rows[0]["revoked_at"] is not None
    stored_payload = decrypt_byok_payload(loads_envelope(rows[0]["encrypted_blob"]))
    assert stored_payload["credentials"]["oauth"]["access_token"] == "expired-access-token"


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_inactive_user_blocks_overlapping_oauth_refresh_before_openai_adapter_sqlite(
    tmp_path,
    monkeypatch,
):
    """Deactivation wins against the refresher and its credential-lock waiter."""
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    await _insert_raw_user_payload(
        pool,
        user_id=user_id,
        provider="openai",
        payload={
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        },
    )
    original_row = await pool.fetchone(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = ? AND provider = ?",
        (user_id, "openai"),
    )
    assert original_row is not None
    original_blob = original_row["encrypted_blob"]

    real_repo = AuthnzUserProviderSecretsRepo(pool)
    initial_reads_ready = asyncio.Event()
    task_active_reads: dict[asyncio.Task, int] = {}
    initial_read_count = 0

    class CountingRepo:
        async def fetch_secret_for_active_user(self, *args, **kwargs):
            nonlocal initial_read_count
            task = asyncio.current_task()
            assert task is not None
            prior = task_active_reads.get(task, 0)
            task_active_reads[task] = prior + 1
            row = await real_repo.fetch_secret_for_active_user(*args, **kwargs)
            if prior == 0:
                assert row is not None
                initial_read_count += 1
                if initial_read_count == 2:
                    initial_reads_ready.set()
            return row

        def __getattr__(self, name):
            return getattr(real_repo, name)

    counting_repo = CountingRepo()

    async def get_counting_repo():
        return counting_repo

    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_calls = 0

    async def refresh_token(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        refresh_started.set()
        await release_refresh.wait()
        return {
            "access_token": "must-not-be-persisted",
            "refresh_token": "must-not-be-used",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_counting_repo)
    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    adapter_calls = 0

    async def resolve_and_dispatch():
        nonlocal adapter_calls
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
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

    first_task = asyncio.create_task(resolve_and_dispatch())
    second_task = None
    try:
        await asyncio.wait_for(refresh_started.wait(), timeout=5)
        second_task = asyncio.create_task(resolve_and_dispatch())
        await asyncio.wait_for(initial_reads_ready.wait(), timeout=5)
        await asyncio.sleep(0)
        assert not second_task.done()
        await _set_user_active(
            pool,
            user_id=user_id,
            is_active=False,
        )
    finally:
        release_refresh.set()

    assert second_task is not None
    results = await asyncio.wait_for(
        asyncio.gather(first_task, second_task, return_exceptions=True),
        timeout=5,
    )
    errors = [result for result in results if isinstance(result, ByokResolutionError)]
    assert len(errors) == 2
    assert all(error.code == "invalid_provider_credentials" for error in errors)
    assert all(error.provider == "openai" for error in errors)
    assert all(error.__cause__ is None and error.__context__ is None for error in errors)
    assert initial_read_count == 2
    assert refresh_calls == 1
    assert adapter_calls == 0
    assert captured_headers == []

    stored_row = await pool.fetchone(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = ? AND provider = ?",
        (user_id, "openai"),
    )
    assert stored_row is not None
    assert stored_row["encrypted_blob"] == original_blob


@pytest.mark.asyncio
async def test_inactive_user_static_openai_key_fails_before_adapter_sqlite(
    tmp_path,
    monkeypatch,
):
    """Default-source lookup cannot dispatch a deactivated owner's API key."""
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    await _upsert_user_key(
        AuthnzUserProviderSecretsRepo(pool),
        user_id,
        "openai",
        "sk-inactive-owner-must-not-dispatch",
    )
    await _set_user_active(
        pool,
        user_id=user_id,
        is_active=False,
    )
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    adapter_calls = 0

    async def resolve_and_dispatch():
        nonlocal adapter_calls
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[],
            org_ids=[],
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

    with pytest.raises(ByokResolutionError) as exc_info:
        await resolve_and_dispatch()

    assert vars(exc_info.value) == {
        "code": "invalid_provider_credentials",
        "provider": "openai",
    }
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert adapter_calls == 0
    assert captured_headers == []
