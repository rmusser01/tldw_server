import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.tests.Chat_NEW import conftest as chat_new_conftest

_STATE_FIELDS = (
    "_OVERRIDE_CACHE_HEALTHY",
    "_OVERRIDE_CACHE_REFRESHED_AT",
    "_OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS",
    "_OVERRIDE_REFRESH_GENERATION",
    "_OVERRIDE_COMPLETED_GENERATION",
    "_OVERRIDE_RECOVERY_IN_FLIGHT",
    "_OVERRIDE_RECOVERY_TASK",
    "_OVERRIDE_REFRESH_SERVICE_TASK",
    "_OVERRIDE_RECOVERY_FAILURES",
    "_OVERRIDE_RECOVERY_NEXT_RETRY_AT",
)


def _snapshot_override_state():
    with llm_provider_overrides._OVERRIDE_LOCK:
        return (
            llm_provider_overrides._OVERRIDE_CACHE,
            dict(llm_provider_overrides._OVERRIDE_CACHE),
            {
                name: getattr(llm_provider_overrides, name)
                for name in _STATE_FIELDS
            },
        )


def _restore_override_state(snapshot) -> None:
    cache, entries, fields = snapshot
    with llm_provider_overrides._OVERRIDE_LOCK:
        llm_provider_overrides._OVERRIDE_CACHE.clear()
        llm_provider_overrides._OVERRIDE_CACHE.update(entries)
        for name, value in fields.items():
            setattr(llm_provider_overrides, name, value)
    assert llm_provider_overrides._OVERRIDE_CACHE is cache


def test_test_credential_restores_unhealthy_ttl_enabled_snapshot():
    baseline = _snapshot_override_state()
    original_override = LLMProviderOverride(
        provider="anthropic",
        api_key="original-key",
    )

    try:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            {"anthropic": original_override},
            healthy=False,
            ttl_enabled=True,
        )

        with chat_new_conftest._test_openai_server_credential():
            assert llm_provider_overrides._OVERRIDE_CACHE["openai"].api_key == (
                "test-openai-key"
            )
            assert llm_provider_overrides._OVERRIDE_CACHE_HEALTHY is True
            assert (
                llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
                is True
            )

        with llm_provider_overrides._OVERRIDE_LOCK:
            assert set(llm_provider_overrides._OVERRIDE_CACHE) == {"anthropic"}
            assert (
                llm_provider_overrides._OVERRIDE_CACHE["anthropic"].api_key
                == "original-key"
            )
            assert llm_provider_overrides._OVERRIDE_CACHE_HEALTHY is False
            assert (
                llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
                is False
            )
            assert llm_provider_overrides._OVERRIDE_RECOVERY_TASK is None
            assert llm_provider_overrides._OVERRIDE_REFRESH_SERVICE_TASK is None
    finally:
        _restore_override_state(baseline)


@pytest.mark.asyncio
async def test_test_credential_retires_active_refresh_service_before_seeding():
    """A late periodic refresh cannot erase the fixture's test credential."""
    baseline = _snapshot_override_state()
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_finished = asyncio.Event()

    async def publish_empty_snapshot() -> None:
        refresh_started.set()
        try:
            await release_refresh.wait()
            with llm_provider_overrides._OVERRIDE_LOCK:
                llm_provider_overrides._OVERRIDE_CACHE.clear()
        finally:
            refresh_finished.set()

    refresh_task = asyncio.create_task(publish_empty_snapshot())
    await asyncio.wait_for(refresh_started.wait(), timeout=1)
    try:
        with llm_provider_overrides._OVERRIDE_LOCK:
            llm_provider_overrides._OVERRIDE_REFRESH_SERVICE_TASK = refresh_task

        with chat_new_conftest._test_openai_server_credential():
            release_refresh.set()
            await asyncio.wait_for(refresh_finished.wait(), timeout=1)
            snapshot = llm_provider_overrides.get_llm_provider_overrides_snapshot()
            assert snapshot["openai"].api_key == "test-openai-key"
    finally:
        release_refresh.set()
        if not refresh_task.done():
            refresh_task.cancel()
        await asyncio.gather(refresh_task, return_exceptions=True)
        _restore_override_state(baseline)


def test_credentialed_client_seeds_after_startup_and_restores_before_shutdown():
    baseline = _snapshot_override_state()
    lifecycle_keys: list[tuple[str, str | None]] = []

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        with llm_provider_overrides._OVERRIDE_LOCK:
            llm_provider_overrides._OVERRIDE_CACHE.clear()
            llm_provider_overrides._OVERRIDE_CACHE["openai"] = LLMProviderOverride(
                provider="openai",
                api_key="startup-key",
            )
            llm_provider_overrides._OVERRIDE_CACHE_HEALTHY = True
            llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS = True
        lifecycle_keys.append(
            ("startup", llm_provider_overrides._OVERRIDE_CACHE["openai"].api_key)
        )
        try:
            yield
        finally:
            lifecycle_keys.append(
                ("shutdown", llm_provider_overrides._OVERRIDE_CACHE["openai"].api_key)
            )
            _restore_override_state(baseline)

    test_app = FastAPI(lifespan=lifespan)

    with chat_new_conftest._credentialed_test_client_context(test_app):
        assert llm_provider_overrides._OVERRIDE_CACHE["openai"].api_key == (
            "test-openai-key"
        )

    assert lifecycle_keys == [
        ("startup", "startup-key"),
        ("shutdown", "startup-key"),
    ]
