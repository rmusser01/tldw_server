from __future__ import annotations

import asyncio
import base64
import copy
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pydantic import BaseModel, TypeAdapter

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
    ServerFallbackCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    capture_provider_override_call_snapshot,
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    reject_provider_call_credentials,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.RAG.rag_service.checkpoint import (
    CheckpointData,
    CheckpointManager,
)

SECRET = "sk-runtime-sentinel-secret-value"


def _resolution(
    provider: str,
    api_key: str | None = SECRET,
    *,
    app_config: dict[str, Any] | None = None,
    auth_source: str | None = "api_key",
    touch=None,
    credential_generation: str | None = None,
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config=app_config,
        credential_fields={},
        source="user",
        allowlisted=True,
        status=(ByokResolutionStatus.RESOLVED if api_key is not None else ByokResolutionStatus.ABSENT),
        auth_source=auth_source,
        _touch_cb=touch,
        _credential_generation=credential_generation,
    )


def _runtime(resolver, **overrides) -> ProviderCredentialRuntime:
    fallback = overrides.pop("fallback_resolver", lambda _provider: None)
    server_config_snapshot = overrides.pop("server_config_snapshot", {})
    return ProviderCredentialRuntime(
        user_id=41,
        team_ids=[7],
        org_ids=[9],
        trusted_base_url_override=True,
        fallback_resolver=fallback,
        server_config_snapshot=server_config_snapshot,
        resolver=resolver,
        **overrides,
    )


def test_default_runtime_rejects_dynamic_fallback_resolver() -> None:
    """Production resolution cannot combine a frozen snapshot with live fallback reads."""
    with pytest.raises(
        ValueError,
        match="requires a frozen server config snapshot",
    ):
        ProviderCredentialRuntime(
            user_id=41,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            fallback_resolver=lambda _provider: "dynamic-key",
            server_config_snapshot={"openai_api": {"api_key": "snapshot-key"}},
        )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("oai", "openai"),
        ("openai-compatible", "custom-openai-api"),
    ],
)
@pytest.mark.asyncio
async def test_aliases_share_one_lookup_and_forward_only_trusted_scope(
    alias: str,
    canonical: str,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    fallback = lambda _provider: "server-key"  # noqa: E731

    async def resolver(provider: str, **kwargs) -> ResolvedByokCredentials:
        calls.append((provider, kwargs))
        return _resolution(provider)

    runtime = _runtime(resolver, fallback_resolver=fallback)
    first = await runtime.resolve(alias)
    second = await runtime.resolve(canonical)

    assert first.provider == canonical
    assert second.provider == canonical
    assert len(calls) == 1
    assert calls[0][0] == canonical
    forwarded_kwargs = dict(calls[0][1])
    forwarded_fallback = forwarded_kwargs.pop("fallback_resolver")
    assert forwarded_fallback(canonical) == "server-key"
    assert forwarded_kwargs == {
        "user_id": 41,
        "team_ids": [7],
        "org_ids": [9],
        "server_config_snapshot": {},
        "force_oauth_refresh": False,
        "trusted_base_url_override": True,
    }
    await runtime.close()


@pytest.mark.asyncio
async def test_override_config_preserves_legacy_string_server_fallback() -> None:
    """A config-only override must not suppress a legacy static key fallback."""

    async def resolver(provider: str, **kwargs) -> ResolvedByokCredentials:
        fallback = kwargs["fallback_resolver"](provider)
        assert isinstance(fallback, ServerFallbackCredentials)
        return _resolution(
            provider,
            fallback.api_key,
            app_config=dict(fallback.app_config or {}),
        )

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={"default_model": "gpt-override"},
            )
        }
    )
    runtime = _runtime(
        resolver,
        fallback_resolver=lambda _provider: "legacy-static-key",
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        handle = await runtime.resolve("openai")

        assert handle.api_key == "legacy-static-key"
        assert handle.app_config == {"openai_api": {"model": "gpt-override"}}
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})


class _ExactSecretLookupPool:
    pool = None

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.lookups: list[tuple[str, ...]] = []

    async def fetchall(self, _query: str, params: tuple[object, ...]):
        providers = tuple(str(provider) for provider in params[1:])
        self.lookups.append(providers)
        return [self.rows[provider] for provider in providers if provider in self.rows]


@pytest.mark.asyncio
async def test_user_secret_lookup_prefers_canonical_row_over_legacy_aliases() -> None:
    pool = _ExactSecretLookupPool(
        {
            "custom-openai-api": {"provider": "custom-openai-api", "encrypted_blob": "canonical"},
            "openai-compatible": {"provider": "openai-compatible", "encrypted_blob": "legacy"},
        }
    )
    repo = AuthnzUserProviderSecretsRepo(pool)

    row = await repo.fetch_secret_for_user(41, "openai-compatible")

    assert row is not None
    assert row["encrypted_blob"] == "canonical"
    assert len(pool.lookups) == 1
    assert pool.lookups[0][0] == "custom-openai-api"
    assert "openai-compatible" in pool.lookups[0]


@pytest.mark.asyncio
async def test_user_secret_lookup_reads_one_legacy_alias_row() -> None:
    pool = _ExactSecretLookupPool({"openai-compatible": {"provider": "openai-compatible", "encrypted_blob": "legacy"}})
    repo = AuthnzUserProviderSecretsRepo(pool)

    row = await repo.fetch_secret_for_user(41, "custom-openai-api")

    assert row is not None
    assert row["provider"] == "openai-compatible"
    assert row["encrypted_blob"] == "legacy"


@pytest.mark.asyncio
async def test_user_secret_lookup_rejects_multiple_legacy_alias_rows() -> None:
    pool = _ExactSecretLookupPool(
        {
            "custom-openai": {"provider": "custom-openai", "encrypted_blob": "legacy-one"},
            "openai-compatible": {"provider": "openai-compatible", "encrypted_blob": "legacy-two"},
        }
    )
    repo = AuthnzUserProviderSecretsRepo(pool)

    with pytest.raises(ValueError, match="conflicting legacy provider credentials"):
        await repo.fetch_secret_for_user(41, "custom-openai-api")


@pytest.mark.asyncio
async def test_revoked_canonical_user_secret_blocks_active_legacy_alias() -> None:
    pool = _ExactSecretLookupPool(
        {
            "openai": {
                "provider": "openai",
                "encrypted_blob": "revoked-canonical",
                "revoked_at": "2026-07-13T00:00:00+00:00",
            },
            "oai": {
                "provider": "oai",
                "encrypted_blob": "active-legacy",
                "revoked_at": None,
            },
        }
    )
    repo = AuthnzUserProviderSecretsRepo(pool)

    assert await repo.fetch_secret_for_user(41, "openai") is None
    revoked_row = await repo.fetch_secret_for_user(41, "openai", include_revoked=True)

    assert revoked_row is not None
    assert revoked_row["encrypted_blob"] == "revoked-canonical"
    assert len(pool.lookups) == 2
    assert all(batch[0] == "openai" and "oai" in batch for batch in pool.lookups)


@pytest.mark.asyncio
async def test_active_user_secret_postgres_lookup_binds_owner_and_secret_in_one_query() -> None:
    """The PostgreSQL adapter must enforce active ownership in the credential read."""

    class _Pool:
        pool = object()

        def __init__(self) -> None:
            self.query = ""
            self.args: tuple[object, ...] = ()

        async def fetchall(self, query: str, *args: object):
            self.query = query
            self.args = args
            return [
                {
                    "user_id": 41,
                    "provider": "openai",
                    "encrypted_blob": "encrypted",
                    "revoked_at": None,
                }
            ]

    pool = _Pool()
    repo = AuthnzUserProviderSecretsRepo(pool)

    row = await repo.fetch_secret_for_active_user(
        41,
        "openai",
        include_revoked=True,
    )

    assert row is not None
    assert "JOIN users u" in pool.query
    assert "u.is_active = TRUE" in pool.query
    assert "COALESCE(u.is_active" not in pool.query
    assert "s.user_id = $1" in pool.query
    assert "s.provider = ANY($2::text[])" in pool.query
    assert pool.args[0] == 41
    assert isinstance(pool.args[1], list)
    assert pool.args[1][0] == "openai"
    assert "oai" in pool.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope_type", "expected_membership_join"),
    (("team", "JOIN team_members tm"), ("org", "JOIN org_members om")),
)
async def test_authorized_shared_secret_postgres_lookup_uses_portable_bindings(
    scope_type: str,
    expected_membership_join: str,
) -> None:
    """Shared authorization SQL must use PostgreSQL placeholders and active joins."""

    class _Pool:
        pool = object()

        def __init__(self) -> None:
            self.query = ""
            self.args: tuple[object, ...] = ()
            self.calls = 0

        async def fetchall(self, query: str, *args: object):
            self.calls += 1
            self.query = query
            self.args = args
            return [
                {
                    "scope_type": scope_type,
                    "scope_id": 73,
                    "provider": "openai",
                    "encrypted_blob": "encrypted",
                    "revoked_at": None,
                }
            ]

        async def fetchone(self, *_args: object):
            pytest.fail("authorized alias lookup must use one set-based statement")

    pool = _Pool()
    repo = AuthnzOrgProviderSecretsRepo(pool)

    row = await repo.fetch_authorized_secret_for_user(
        scope_type,
        73,
        41,
        "openai",
    )

    assert row is not None
    assert pool.calls == 1
    assert expected_membership_join in pool.query
    assert "JOIN users u" in pool.query
    assert "COALESCE(" not in pool.query
    assert "u.is_active = TRUE" in pool.query
    assert "o.is_active = TRUE" in pool.query
    if scope_type == "team":
        assert "tm.status = 'active'" in pool.query
        assert "t.is_active = TRUE" in pool.query
    else:
        assert "om.status = 'active'" in pool.query
    assert "s.provider IN ($3, $4)" in pool.query
    assert "$5" in pool.query
    assert "?" not in pool.query
    assert "s.revoked_at IS NULL" not in pool.query
    assert pool.args == (scope_type, 73, "openai", "oai", 41)


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_resolution() -> None:
    started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        started.set()
        await second_started.wait()
        await release.wait()
        return _resolution(provider)

    runtime = _runtime(resolver)
    first = asyncio.create_task(runtime.resolve("openai"))
    await started.wait()

    async def second_caller() -> ProviderCallCredentials:
        second_started.set()
        return await runtime.resolve("OpenAI")

    second = asyncio.create_task(second_caller())
    await second_started.wait()
    release.set()
    handles = await asyncio.gather(first, second)

    assert calls == 1
    assert handles[0].api_key == handles[1].api_key == SECRET
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    ("configured_model", "expected_code"),
    [
        ("gpt-allowed", None),
        ("gpt-blocked", "model_not_allowed"),
    ],
)
async def test_concurrent_implicit_configured_model_policy_is_enforced_once(
    configured_model: str,
    expected_code: str | None,
) -> None:
    """A provider default model must cross the same policy boundary as an explicit model."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return _resolution(
            provider,
            app_config={"openai_api": {"model": configured_model}},
        )

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-allowed"],
            )
        }
    )
    runtime = _runtime(
        resolver,
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        first = asyncio.create_task(runtime.resolve("openai"))
        await started.wait()
        second = asyncio.create_task(runtime.resolve("openai"))
        release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        assert calls == 1
        if expected_code is None:
            assert all(isinstance(item, ProviderCallCredentials) for item in results)
        else:
            assert all(isinstance(item, ByokResolutionError) for item in results)
            errors = [item for item in results if isinstance(item, ByokResolutionError)]
            assert {item.code for item in errors} == {"invalid_provider_credentials"}
            assert {getattr(item, "policy_code", None) for item in errors} == {
                expected_code,
            }
    finally:
        release.set()
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_explicit_and_implicit_models_enforce_per_caller_policy() -> None:
    """Joining an explicit-model lookup cannot authorize a blocked provider default."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return _resolution(
            provider,
            app_config={"openai_api": {"model": "gpt-blocked-default"}},
        )

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-explicit-allowed"],
            )
        }
    )
    runtime = _runtime(
        resolver,
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        explicit = asyncio.create_task(
            runtime.resolve("openai", model="gpt-explicit-allowed")
        )
        await started.wait()
        implicit = asyncio.create_task(runtime.resolve("openai"))
        release.set()

        explicit_result = await explicit
        implicit_result = await asyncio.gather(implicit, return_exceptions=True)

        assert isinstance(explicit_result, ProviderCallCredentials)
        assert len(implicit_result) == 1
        assert isinstance(implicit_result[0], ByokResolutionError)
        assert implicit_result[0].code == "invalid_provider_credentials"
        assert getattr(implicit_result[0], "policy_code", None) == "model_not_allowed"
        assert calls == 1
    finally:
        release.set()
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
async def test_implicit_model_policy_uses_numbered_custom_openai_section() -> None:
    """Numbered custom adapters must enforce the model from their canonical config section."""

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(
            provider,
            app_config={"custom_openai_api_3": {"model": "blocked-custom-model"}},
        )

    provider = "custom-openai-api-3"
    set_llm_provider_overrides_cache_for_tests(
        {
            provider: LLMProviderOverride(
                provider=provider,
                allowed_models=["allowed-custom-model"],
            )
        }
    )
    runtime = _runtime(
        resolver,
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        with pytest.raises(ByokResolutionError) as exc_info:
            await runtime.resolve(provider)

        assert exc_info.value.code == "invalid_provider_credentials"
        assert getattr(exc_info.value, "policy_code", None) == "model_not_allowed"
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
async def test_rag_critique_implicit_override_model_is_enforced_at_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-less RAG critique call cannot bypass an override allowlist."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: False)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {
            "openai_api": {
                "api_key": "static-key",
                "model": "gpt-static",
            }
        },
    )
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-allowed"],
                config={"default_model": "gpt-blocked-critique"},
                api_key="override-key",
            )
        }
    )
    runtime = ProviderCredentialRuntime(
        user_id=41,
        team_ids=[],
        org_ids=[],
        trusted_base_url_override=True,
        server_config_snapshot={
            "openai_api": {
                "api_key": "static-key",
                "model": "gpt-static",
            }
        },
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        with pytest.raises(ByokResolutionError) as exc_info:
            await runtime.resolve("openai")

        assert exc_info.value.code == "invalid_provider_credentials"
        assert getattr(exc_info.value, "policy_code", None) == "model_not_allowed"
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "force_refresh",
    (False, True),
    ids=("normal", "forced-refresh"),
)
async def test_concurrent_callers_do_not_share_resolution_across_override_snapshots(
    force_refresh: bool,
) -> None:
    """A caller starting after rotation must resolve from the rotated snapshot."""
    first_started = asyncio.Event()
    second_captured = asyncio.Event()
    release_first = asyncio.Event()
    calls = 0
    rotated = False

    async def resolver(
        provider: str,
        *,
        fallback_resolver,
        **_kwargs,
    ) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        fallback = fallback_resolver(provider)
        assert isinstance(fallback, ServerFallbackCredentials)
        return _resolution(provider, fallback.api_key)

    def capture_snapshot(provider: str):
        snapshot = capture_provider_override_call_snapshot(provider)
        if rotated:
            second_captured.set()
        return snapshot

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="original-key")}
    )
    runtime = _runtime(
        resolver,
        override_snapshot_resolver=capture_snapshot,
    )
    try:
        first = asyncio.create_task(
            runtime.resolve(
                "openai",
                model="gpt-4o",
                force_refresh=force_refresh,
            )
        )
        await asyncio.wait_for(first_started.wait(), timeout=1.0)

        rotated = True
        set_llm_provider_overrides_cache_for_tests(
            {"openai": LLMProviderOverride(provider="openai", api_key="rotated-key")}
        )
        second = asyncio.create_task(
            runtime.resolve(
                "openai",
                model="gpt-4o",
                force_refresh=force_refresh,
            )
        )
        await asyncio.wait_for(second_captured.wait(), timeout=1.0)
        release_first.set()

        first_handle, second_handle = await asyncio.gather(first, second)

        assert first_handle.api_key == "original-key"
        assert second_handle.api_key == "rotated-key"
        assert calls == 2
    finally:
        release_first.set()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_successful_inflight_call_can_mark_used_after_override_rotation() -> None:
    """Rotation must not invalidate a genuine handle already dispatched in flight."""

    touches: list[str] = []
    rotated = False

    async def resolver(
        provider: str,
        *,
        fallback_resolver,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        fallback = fallback_resolver(provider)
        assert isinstance(fallback, ServerFallbackCredentials)
        selected_key = str(fallback.api_key)

        async def touch() -> None:
            await asyncio.sleep(0)
            touches.append(selected_key)

        return _resolution(provider, selected_key, touch=touch)

    def capture_snapshot(provider: str):
        assert provider == "openai"
        return capture_provider_override_call_snapshot(provider)

    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="original-key")}
    )
    runtime = _runtime(
        resolver,
        override_snapshot_resolver=capture_snapshot,
    )
    try:
        original_handle = await runtime.resolve("openai", model="gpt-4o")

        rotated = True
        set_llm_provider_overrides_cache_for_tests(
            {"openai": LLMProviderOverride(provider="openai", api_key="rotated-key")}
        )
        rotated_handle = await runtime.resolve("openai", model="gpt-4o")

        assert rotated is True
        assert original_handle.api_key == "original-key"
        assert rotated_handle.api_key == "rotated-key"
        persisted = await asyncio.gather(
            runtime.mark_used(original_handle),
            runtime.mark_used(rotated_handle),
        )

        assert persisted == [True, True]
        assert set(touches) == {"original-key", "rotated-key"}
    finally:
        await runtime.close()
        set_llm_provider_overrides_cache_for_tests({})
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_resolution() -> None:
    started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    cancelled_inside_resolver = False

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal cancelled_inside_resolver
        started.set()
        try:
            await second_started.wait()
            await release.wait()
        except asyncio.CancelledError:
            cancelled_inside_resolver = True
            raise
        return _resolution(provider)

    runtime = _runtime(resolver)
    cancelled_waiter = asyncio.create_task(runtime.resolve("openai"))
    await started.wait()

    async def surviving_caller() -> ProviderCallCredentials:
        second_started.set()
        return await runtime.resolve("openai")

    survivor = asyncio.create_task(surviving_caller())
    await second_started.wait()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    release.set()
    handle = await survivor
    assert handle.api_key == SECRET
    assert cancelled_inside_resolver is False
    await runtime.close()


@pytest.mark.asyncio
async def test_different_providers_resolve_independently() -> None:
    openai_started = asyncio.Event()
    release_openai = asyncio.Event()
    anthropic_finished = asyncio.Event()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        if provider == "openai":
            openai_started.set()
            await release_openai.wait()
        else:
            anthropic_finished.set()
        return _resolution(provider, f"key-for-{provider}")

    runtime = _runtime(resolver)
    openai = asyncio.create_task(runtime.resolve("openai"))
    await openai_started.wait()
    anthropic = await runtime.resolve("anthropic")

    assert anthropic_finished.is_set()
    assert anthropic.api_key == "key-for-anthropic"
    release_openai.set()
    assert (await openai).api_key == "key-for-openai"
    await runtime.close()


@pytest.mark.asyncio
async def test_forced_refresh_wins_over_slower_original_resolution() -> None:
    original_started = asyncio.Event()
    release_original = asyncio.Event()
    calls: list[bool] = []

    async def resolver(provider: str, *, force_oauth_refresh: bool, **_kwargs) -> ResolvedByokCredentials:
        calls.append(force_oauth_refresh)
        if not force_oauth_refresh:
            original_started.set()
            await release_original.wait()
            return _resolution(provider, "stale-key")
        return _resolution(provider, "refreshed-key")

    runtime = _runtime(resolver)
    original = asyncio.create_task(runtime.resolve("openai"))
    await original_started.wait()

    refreshed = await runtime.resolve("openai", force_refresh=True)
    release_original.set()

    assert refreshed.api_key == "refreshed-key"
    assert (await original).api_key == "refreshed-key"
    assert (await runtime.resolve("openai")).api_key == "refreshed-key"
    assert calls == [False, True]
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_forced_refresh_callers_share_one_refresh() -> None:
    refresh_started = asyncio.Event()
    second_started = asyncio.Event()
    release_refresh = asyncio.Event()
    calls: list[bool] = []

    async def resolver(provider: str, *, force_oauth_refresh: bool, **_kwargs) -> ResolvedByokCredentials:
        calls.append(force_oauth_refresh)
        if force_oauth_refresh:
            refresh_started.set()
            await second_started.wait()
            await release_refresh.wait()
            return _resolution(provider, "refreshed-key")
        return _resolution(provider, "original-key")

    runtime = _runtime(resolver)
    await runtime.resolve("openai")
    first = asyncio.create_task(runtime.resolve("openai", force_refresh=True))
    await refresh_started.wait()

    async def second_caller() -> ProviderCallCredentials:
        second_started.set()
        return await runtime.resolve("openai", force_refresh=True)

    second = asyncio.create_task(second_caller())
    await second_started.wait()
    release_refresh.set()
    handles = await asyncio.gather(first, second)

    assert [handle.api_key for handle in handles] == [
        "refreshed-key",
        "refreshed-key",
    ]
    assert calls == [False, True]
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_sequential_runtimes_coalesce_real_encrypted_oauth_refresh(
    monkeypatch,
) -> None:
    """A stale sibling runtime adopts the OAuth rotation already persisted by its peer."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv(
        "BYOK_ENCRYPTION_KEY",
        base64.b64encode(b"k" * 32).decode("ascii"),
    )
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_SKEW_SECONDS", "120")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "memory")
    reset_settings()

    envelope = encrypt_byok_payload(
        {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "stale-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                }
            },
        }
    )
    row = {
        "encrypted_blob": dumps_envelope(envelope),
        "last_used_at": None,
        "metadata": None,
        "key_hint": "oauth",
        "revoked_at": None,
    }
    token_requests: list[str] = []

    class _FakeUserRepo:
        async def fetch_secret_for_active_user(self, user_id: int, provider: str, **_kwargs):
            assert user_id == 41
            assert provider == "openai"
            return dict(row)

        async def fetch_secret_for_user(self, user_id: int, provider: str, **_kwargs):
            assert user_id == 41
            assert provider == "openai"
            return dict(row)

        async def update_secret_if_active_and_unchanged(self, **kwargs):
            if row["encrypted_blob"] != kwargs["expected_encrypted_blob"]:
                return False
            row["encrypted_blob"] = kwargs["encrypted_blob"]
            row["key_hint"] = kwargs["key_hint"]
            row["metadata"] = kwargs["metadata"]
            row["updated_at"] = kwargs["updated_at"]
            return True

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "access_token": "refreshed-access-token",
                "refresh_token": "rotated-refresh-token",
                "expires_in": 7200,
                "token_type": "Bearer",
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

    runtime_a = ProviderCredentialRuntime(
        user_id=41,
        team_ids=[7],
        org_ids=[9],
        trusted_base_url_override=True,
        server_config_snapshot={},
    )
    runtime_b = ProviderCredentialRuntime(
        user_id=41,
        team_ids=[7],
        org_ids=[9],
        trusted_base_url_override=True,
        server_config_snapshot={},
    )
    try:
        assert (await runtime_a.resolve("openai")).api_key == "stale-access-token"
        assert (await runtime_b.resolve("openai")).api_key == "stale-access-token"

        refreshed_a = await runtime_a.resolve("openai", force_refresh=True)
        refreshed_b = await runtime_b.resolve("openai", force_refresh=True)

        assert refreshed_a.api_key == "refreshed-access-token"
        assert refreshed_b.api_key == "refreshed-access-token"
        assert token_requests == ["single-use-refresh-token"]
        stored_payload = decrypt_byok_payload(loads_envelope(row["encrypted_blob"]))
        assert stored_payload["credentials"]["oauth"]["access_token"] == (
            "refreshed-access-token"
        )
        assert stored_payload["credentials"]["oauth"]["refresh_token"] == (
            "rotated-refresh-token"
        )
    finally:
        await runtime_a.close()
        await runtime_b.close()
        reset_settings()


@pytest.mark.asyncio
async def test_late_failure_from_sibling_stale_handle_reuses_completed_refresh() -> None:
    calls: list[bool] = []

    async def resolver(provider: str, *, force_oauth_refresh: bool, **_kwargs) -> ResolvedByokCredentials:
        calls.append(force_oauth_refresh)
        key = "refreshed-key" if force_oauth_refresh else "stale-key"
        return _resolution(provider, key)

    runtime = _runtime(resolver)
    first_stale = await runtime.resolve("openai")
    second_stale = await runtime.resolve("openai")

    first_retry = await runtime.resolve("openai", force_refresh=True)
    late_retry = await runtime.resolve("openai", force_refresh=True)

    assert first_stale.api_key == second_stale.api_key == "stale-key"
    assert first_retry.api_key == late_retry.api_key == "refreshed-key"
    assert calls == [False, True]
    await runtime.close()


@pytest.mark.asyncio
async def test_only_current_generation_marks_use_and_marks_once() -> None:
    original_touches = 0
    refreshed_touches = 0

    async def touch_original() -> None:
        nonlocal original_touches
        original_touches += 1

    async def touch_refreshed() -> None:
        nonlocal refreshed_touches
        refreshed_touches += 1

    async def resolver(provider: str, *, force_oauth_refresh: bool, **_kwargs) -> ResolvedByokCredentials:
        if force_oauth_refresh:
            return _resolution(provider, "new-key", touch=touch_refreshed)
        return _resolution(provider, "old-key", touch=touch_original)

    runtime = _runtime(resolver)
    stale = await runtime.resolve("openai")
    current = await runtime.resolve("openai", force_refresh=True)

    stale_persisted = await runtime.mark_used(stale)
    await runtime.mark_used(current)
    await runtime.mark_used(current)

    assert stale_persisted is False
    assert original_touches == 0
    assert refreshed_touches == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_concurrent_mark_used_callers_share_persistence_and_wait_for_completion() -> None:
    touch_started = asyncio.Event()
    release_touch = asyncio.Event()
    touches = 0

    async def touch() -> None:
        nonlocal touches
        touches += 1
        touch_started.set()
        await release_touch.wait()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    first = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()
    second = asyncio.create_task(runtime.mark_used(handle))
    await asyncio.sleep(0)

    assert touches == 1
    assert first.done() is False
    assert second.done() is False
    assert runtime._cache["openai"].used is False

    release_touch.set()
    await asyncio.gather(first, second)
    assert runtime._cache["openai"].used is True
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_mark_used_normal_wait_is_bounded_and_keeps_exactly_one_owned_write(
    monkeypatch,
) -> None:
    """A hostile usage callback cannot hold request completion indefinitely."""
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "USAGE_TASK_DRAIN_TIMEOUT_SECONDS", 0.01, raising=False)
    touch_started = asyncio.Event()
    release_touch = asyncio.Event()
    touches = 0

    async def touch() -> None:
        nonlocal touches
        touches += 1
        touch_started.set()
        await release_touch.wait()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    first = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()
    completed, _pending = await asyncio.wait({first}, timeout=0.2)

    try:
        assert first in completed
        await first
        assert touches == 1
        assert runtime._cache["openai"].used is False

        second = asyncio.create_task(runtime.mark_used(handle))
        completed, _pending = await asyncio.wait({second}, timeout=0.2)
        assert second in completed
        await second
        assert touches == 1
    finally:
        release_touch.set()
        await asyncio.gather(*tuple(runtime._usage_tasks.values()), return_exceptions=True)
        await runtime.close()

    assert runtime._usage_tasks == {}


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_cancelled_mark_used_waiter_drains_persistence_once() -> None:
    touch_started = asyncio.Event()
    release_touch = asyncio.Event()
    touches = 0
    completions = 0

    async def touch() -> None:
        nonlocal touches, completions
        touches += 1
        touch_started.set()
        await release_touch.wait()
        completions += 1

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    waiter = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()

    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    assert waiter.done() is False
    assert touches == 1
    assert completions == 0

    release_touch.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await runtime.mark_used(handle)

    assert touches == 1
    assert completions == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_close_owns_and_drains_inflight_usage_persistence() -> None:
    touch_started = asyncio.Event()
    touch_cancelled = asyncio.Event()
    touch_completed = asyncio.Event()
    release_touch = asyncio.Event()

    async def touch() -> None:
        touch_started.set()
        try:
            await release_touch.wait()
        except asyncio.CancelledError:
            touch_cancelled.set()
            raise
        touch_completed.set()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    entry = runtime._cache["openai"]
    usage_waiter = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()
    close_waiter = asyncio.create_task(runtime.close())
    try:
        await asyncio.sleep(0)
        assert touch_cancelled.is_set() is False
        assert close_waiter.done() is False
    finally:
        release_touch.set()
        await asyncio.gather(usage_waiter, close_waiter, return_exceptions=True)

    assert touch_completed.is_set()
    assert entry.used is True
    assert runtime._usage_tasks == {}


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_repeated_cancellation_abandons_noncooperative_usage_task(monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "USAGE_TASK_DRAIN_TIMEOUT_SECONDS", 0.01, raising=False)
    touch_started = asyncio.Event()
    touch_cancelled = asyncio.Event()
    release_touch = asyncio.Event()

    async def touch() -> None:
        touch_started.set()
        try:
            await release_touch.wait()
        except asyncio.CancelledError:
            touch_cancelled.set()
            await release_touch.wait()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    waiter = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()
    usage_task = next(iter(runtime._usage_tasks.values()))

    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    completed, _pending = await asyncio.wait({waiter}, timeout=0.2)
    try:
        assert waiter in completed
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert touch_cancelled.is_set()
        assert runtime._usage_tasks == {}
    finally:
        release_touch.set()
        await asyncio.gather(usage_task, waiter, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_close_abandons_noncooperative_usage_and_scrubs_references(monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "USAGE_TASK_DRAIN_TIMEOUT_SECONDS", 0.01, raising=False)
    touch_started = asyncio.Event()
    touch_cancelled = asyncio.Event()
    release_touch = asyncio.Event()

    async def touch() -> None:
        touch_started.set()
        try:
            await release_touch.wait()
        except asyncio.CancelledError:
            touch_cancelled.set()
            await release_touch.wait()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    original_identity = runtime._identity
    handle = await runtime.resolve("openai")
    usage_waiter = asyncio.create_task(runtime.mark_used(handle))
    await touch_started.wait()
    usage_task = next(iter(runtime._usage_tasks.values()))
    close_waiter = asyncio.create_task(runtime.close())

    completed, _pending = await asyncio.wait({close_waiter}, timeout=0.2)
    try:
        assert close_waiter in completed
        await close_waiter
        assert touch_cancelled.is_set()
        assert runtime._usage_tasks == {}
        assert runtime._cache == {}
        assert runtime._identity is not original_identity
        assert runtime._user_id is None
        assert runtime._team_ids == []
        assert runtime._org_ids == []
        assert runtime._fallback_resolver is None
        assert runtime._server_config_snapshot == {}
    finally:
        release_touch.set()
        await asyncio.gather(usage_task, usage_waiter, close_waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_touch_failure_does_not_log_credential_derived_exception_text() -> None:
    async def failing_touch() -> None:
        raise RuntimeError(f"{SECRET}: user=41 team=7")

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=failing_touch)

    messages: list[str] = []
    sink_id = logger.add(messages.append)
    runtime = _runtime(resolver)
    try:
        handle = await runtime.resolve("openai")
        await runtime.mark_used(handle)
    finally:
        await runtime.close()
        logger.remove(sink_id)

    assert SECRET not in "".join(messages)
    assert "user=41" not in "".join(messages)
    assert "team=7" not in "".join(messages)


@pytest.mark.asyncio
async def test_failed_usage_persistence_remains_retryable() -> None:
    touches = 0

    async def touch() -> None:
        nonlocal touches
        touches += 1
        if touches == 1:
            raise RuntimeError(f"{SECRET}: private durable touch detail")

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider, touch=touch)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")

    first_persisted = await runtime.mark_used(handle)
    assert first_persisted is False
    assert touches == 1
    assert runtime._cache["openai"].used is False

    second_persisted = await runtime.mark_used(handle)
    assert second_persisted is True
    assert touches == 2
    assert runtime._cache["openai"].used is True
    await runtime.close()


@pytest.mark.asyncio
async def test_app_config_is_copied_from_resolution_and_between_handles() -> None:
    resolved = _resolution(
        "openai",
        app_config={"openai_api": {"base_url": "https://example.test"}},
    )

    async def resolver(_provider: str, **_kwargs) -> ResolvedByokCredentials:
        return resolved

    runtime = _runtime(resolver)
    first = await runtime.resolve("openai")
    second = await runtime.resolve("openai")
    first.app_config["openai_api"]["base_url"] = SECRET

    assert resolved.app_config == {"openai_api": {"base_url": "https://example.test"}}
    assert second.app_config == {"openai_api": {"base_url": "https://example.test"}}
    await runtime.close()


@pytest.mark.asyncio
async def test_explicit_absence_is_cached_without_secondary_fallback() -> None:
    resolver_calls = 0
    fallback_calls = 0

    def fallback(_provider: str) -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return SECRET

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal resolver_calls
        resolver_calls += 1
        return _resolution(provider, None)

    runtime = _runtime(resolver, fallback_resolver=fallback)
    first = await runtime.resolve("openai")
    second = await runtime.resolve("OpenAI")

    assert first.api_key is None
    assert second.api_key is None
    assert first.credentials_resolved is True
    assert resolver_calls == 1
    assert fallback_calls == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_handle_rejects_serialization_and_persistence_without_leaking(
    tmp_path: Path,
) -> None:
    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(
            provider,
            app_config={"openai_api": {"organization": SECRET}},
        )

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")

    assert not hasattr(handle, "__dict__")
    assert not hasattr(handle, "source")
    assert SECRET not in repr(handle)
    assert "REDACTED" in repr(handle)

    operations = (
        lambda: pickle.dumps(handle),
        lambda: copy.copy(handle),
        lambda: copy.deepcopy(handle),
        lambda: json.dumps(handle),
        lambda: handle.model_dump(),
        lambda: handle.model_dump_json(),
    )
    for operation in operations:
        with pytest.raises(Exception) as exc_info:
            operation()
        assert SECRET not in str(exc_info.value)
        assert SECRET not in repr(exc_info.value)

    class AnyEnvelope(BaseModel):
        credential: Any

    any_envelope = AnyEnvelope(credential=handle)
    python_payload = any_envelope.model_dump(mode="python")
    assert python_payload["credential"] is handle
    with pytest.raises(TypeError) as exc_info:
        reject_provider_call_credentials(python_payload)
    assert str(exc_info.value) == "ProviderCallCredentials cannot be serialized"

    class TypedEnvelope(BaseModel):
        credential: ProviderCallCredentials

    typed_envelope = TypedEnvelope(credential=handle)
    adapter = TypeAdapter(ProviderCallCredentials)
    assert adapter.validate_python(handle) is handle
    pydantic_operations = (
        lambda: any_envelope.model_dump(mode="json"),
        lambda: any_envelope.model_dump_json(),
        lambda: typed_envelope.model_dump(mode="json"),
        lambda: typed_envelope.model_dump_json(),
        lambda: adapter.dump_python(handle, mode="json"),
        lambda: adapter.dump_json(handle),
    )
    for operation in pydantic_operations:
        with pytest.raises(Exception) as exc_info:
            operation()
        assert SECRET not in str(exc_info.value)
        assert SECRET not in repr(exc_info.value)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = CheckpointData(
        checkpoint_id="credential-test",
        task_type="credential-test",
        created_at="2026-07-12T00:00:00+00:00",
        updated_at="2026-07-12T00:00:00+00:00",
        total_items=1,
        config={"credential": handle},
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    checkpoint_path = checkpoint_manager.get_checkpoint_path("credential-test")
    original_checkpoint = b'{"existing":true}\n'
    checkpoint_path.write_bytes(original_checkpoint)
    with pytest.raises(Exception) as exc_info:
        checkpoint_manager._save_atomic(checkpoint)
    assert SECRET not in str(exc_info.value)
    assert checkpoint_path.read_bytes() == original_checkpoint

    manager = JobManager(db_path=tmp_path / "jobs.db")
    with pytest.raises(Exception) as exc_info:
        manager.create_job(
            domain="credential-test",
            queue="default",
            job_type="credential-test",
            payload={"credential": handle},
            owner_user_id=None,
        )
    assert SECRET not in str(exc_info.value)

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_bytes().decode("utf-8", errors="ignore")
    await runtime.close()


@pytest.mark.asyncio
async def test_close_cancels_owned_work_is_idempotent_and_closes_public_methods() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return _resolution(provider)

    runtime = _runtime(blocked_resolver)
    pending = asyncio.create_task(runtime.resolve("openai"))
    await started.wait()
    owned_resolution = runtime._inflight["openai"]
    await asyncio.gather(runtime.close(), runtime.close())
    assert cancelled.is_set()
    assert owned_resolution.done()

    with pytest.raises(RuntimeError, match="runtime is closed"):
        await pending
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await runtime.resolve("openai")

    async def immediate_resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        return _resolution(provider)

    other_runtime = _runtime(immediate_resolver)
    handle = await other_runtime.resolve("openai")
    await other_runtime.close()
    await other_runtime.close()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await other_runtime.mark_used(handle)


@pytest.mark.asyncio
async def test_cancelled_close_waiter_bounds_noncooperative_resolution_and_scrubs_runtime(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "RESOLUTION_TASK_CANCEL_DRAIN_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    resolution_started = asyncio.Event()
    cleanup_active = asyncio.Event()
    release_resolution = asyncio.Event()

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        if provider == "openai":
            return _resolution(provider)
        resolution_started.set()
        try:
            await release_resolution.wait()
        except asyncio.CancelledError:
            cleanup_active.set()
            await release_resolution.wait()
        return _resolution(provider)

    runtime = _runtime(resolver)
    handle = await runtime.resolve("openai")
    pending_resolution = asyncio.create_task(runtime.resolve("anthropic"))
    await asyncio.wait_for(resolution_started.wait(), timeout=1.0)
    owned_resolution = runtime._inflight["anthropic"]
    resolution_finished = asyncio.Event()
    owned_resolution.add_done_callback(lambda _task: resolution_finished.set())

    close_waiter = asyncio.create_task(runtime.close())
    await asyncio.wait_for(cleanup_active.wait(), timeout=1.0)
    owned_cleanup = runtime._close_task
    assert owned_cleanup is not None
    cleanup_finished = asyncio.Event()
    owned_cleanup.add_done_callback(lambda _task: cleanup_finished.set())

    close_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_waiter
    assert owned_cleanup.cancelled() is False

    try:
        completed, _pending = await asyncio.wait(
            {owned_cleanup},
            timeout=0.2,
        )
        assert owned_cleanup in completed
        await asyncio.wait_for(cleanup_finished.wait(), timeout=0.2)
        assert owned_resolution.done() is False
        assert runtime._close_task is None
        assert runtime._cache == {}
        assert runtime._issued_entries == {}
        assert runtime._inflight == {}
        assert runtime._refresh_tasks == {}
        assert runtime._usage_tasks == {}
        assert runtime._user_id is None
        assert runtime._team_ids == []
        assert runtime._org_ids == []
        assert runtime._fallback_resolver is None
        assert runtime._server_config_snapshot == {}
        waiter_done, _pending = await asyncio.wait(
            {pending_resolution},
            timeout=0.2,
        )
        assert pending_resolution in waiter_done
        with pytest.raises(RuntimeError, match="runtime is closed"):
            await pending_resolution

        release_resolution.set()
        await asyncio.wait_for(resolution_finished.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert owned_resolution.done()
        assert owned_resolution.get_coro().cr_frame is None
        assert owned_resolution._log_traceback is False

        await runtime.close()
        await runtime.close()
        with pytest.raises(RuntimeError, match="runtime is closed"):
            await runtime.resolve("openai")
        with pytest.raises(RuntimeError, match="runtime is closed"):
            await runtime.mark_used(handle)
    finally:
        release_resolution.set()
        await asyncio.gather(
            owned_resolution,
            pending_resolution,
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_resolver_failure_is_sanitized_and_not_cached() -> None:
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"{SECRET}: user=41 team=7 api_key={SECRET}")

    runtime = _runtime(resolver)
    for _ in range(2):
        with pytest.raises(RuntimeError) as exc_info:
            await runtime.resolve("openai")
        assert str(exc_info.value) == "Provider credential resolution failed"
        assert SECRET not in repr(exc_info.value)
        assert exc_info.value.__context__ is None

    assert calls == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_typed_resolver_failure_remains_typed_sanitized_and_uncached() -> None:
    calls = 0

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        nonlocal calls
        calls += 1
        raise ByokResolutionError("credential_store_unavailable", provider)

    runtime = _runtime(resolver)
    for _ in range(2):
        with pytest.raises(ByokResolutionError) as exc_info:
            await runtime.resolve("openai")
        assert exc_info.value.code == "credential_store_unavailable"
        assert str(exc_info.value) == "credential_store_unavailable: openai"
        assert SECRET not in repr(exc_info.value)
        assert exc_info.value.__context__ is None

    assert calls == 2
    await runtime.close()


@pytest.mark.asyncio
async def test_typed_resolver_failure_detaches_private_exception_graph() -> None:
    """The runtime boundary must sever a resolver's secret-bearing cause graph."""
    secret = "sk-private-resolver-cause-/private/provider-credential.json"

    async def resolver(provider: str, **_kwargs) -> ResolvedByokCredentials:
        try:
            raise ValueError(secret)
        except ValueError as exc:
            raise ByokResolutionError("invalid_provider_credentials", provider) from exc

    runtime = _runtime(resolver)
    try:
        with pytest.raises(ByokResolutionError) as exc_info:
            await runtime.resolve("openai")

        assert exc_info.value.code == "invalid_provider_credentials"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert secret not in repr(exc_info.value)
    finally:
        await runtime.close()
