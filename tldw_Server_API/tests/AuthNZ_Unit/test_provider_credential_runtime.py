from __future__ import annotations

import asyncio
import copy
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pydantic import BaseModel, TypeAdapter

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
    reject_provider_call_credentials,
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
    )


def _runtime(resolver, **overrides) -> ProviderCredentialRuntime:
    fallback = overrides.pop("fallback_resolver", lambda _provider: None)
    return ProviderCredentialRuntime(
        user_id=41,
        team_ids=[7],
        org_ids=[9],
        trusted_base_url_override=True,
        fallback_resolver=fallback,
        resolver=resolver,
        **overrides,
    )


@pytest.mark.asyncio
async def test_aliases_share_one_lookup_and_forward_only_trusted_scope() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    fallback = lambda _provider: "server-key"  # noqa: E731

    async def resolver(provider: str, **kwargs) -> ResolvedByokCredentials:
        calls.append((provider, kwargs))
        return _resolution(provider)

    runtime = _runtime(resolver, fallback_resolver=fallback)
    first = await runtime.resolve("  OpenAI  ")
    second = await runtime.resolve("openai")

    assert first.provider == "openai"
    assert second.provider == "openai"
    assert len(calls) == 1
    assert calls[0][0] == "openai"
    assert calls[0][1] == {
        "user_id": 41,
        "team_ids": [7],
        "org_ids": [9],
        "fallback_resolver": fallback,
        "force_oauth_refresh": False,
        "trusted_base_url_override": True,
    }
    await runtime.close()


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

    await runtime.mark_used(stale)
    await runtime.mark_used(current)
    await runtime.mark_used(current)

    assert original_touches == 0
    assert refreshed_touches == 1
    await runtime.close()


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
    await asyncio.gather(runtime.close(), runtime.close())
    await cancelled.wait()

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
async def test_cancelled_close_waiter_does_not_retain_completed_cleanup_task() -> None:
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
    await resolution_started.wait()

    close_waiter = asyncio.create_task(runtime.close())
    await cleanup_active.wait()
    owned_cleanup = runtime._close_task
    assert owned_cleanup is not None
    cleanup_finished = asyncio.Event()
    owned_cleanup.add_done_callback(lambda _task: cleanup_finished.set())

    close_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_waiter
    assert owned_cleanup.cancelled() is False

    release_resolution.set()
    await cleanup_finished.wait()

    assert runtime._close_task is None
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await pending_resolution
    await runtime.close()
    await runtime.close()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await runtime.resolve("openai")
    with pytest.raises(RuntimeError, match="runtime is closed"):
        await runtime.mark_used(handle)


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
