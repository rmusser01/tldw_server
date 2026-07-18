from contextlib import asynccontextmanager

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


def test_test_credential_restores_unhealthy_ttl_enabled_state_exactly():
    baseline = _snapshot_override_state()
    recovery_task = object()
    refresh_service_task = object()
    original_override = LLMProviderOverride(
        provider="anthropic",
        api_key="original-key",
    )

    try:
        with llm_provider_overrides._OVERRIDE_LOCK:
            llm_provider_overrides._OVERRIDE_CACHE.clear()
            llm_provider_overrides._OVERRIDE_CACHE["anthropic"] = original_override
            llm_provider_overrides._OVERRIDE_CACHE_HEALTHY = False
            llm_provider_overrides._OVERRIDE_CACHE_REFRESHED_AT = 123.5
            llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS = False
            llm_provider_overrides._OVERRIDE_REFRESH_GENERATION = 41
            llm_provider_overrides._OVERRIDE_COMPLETED_GENERATION = 37
            llm_provider_overrides._OVERRIDE_RECOVERY_IN_FLIGHT = True
            llm_provider_overrides._OVERRIDE_RECOVERY_TASK = recovery_task
            llm_provider_overrides._OVERRIDE_REFRESH_SERVICE_TASK = refresh_service_task
            llm_provider_overrides._OVERRIDE_RECOVERY_FAILURES = 5
            llm_provider_overrides._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 789.25
        expected = _snapshot_override_state()

        with chat_new_conftest._test_openai_server_credential():
            assert llm_provider_overrides._OVERRIDE_CACHE["openai"].api_key == (
                "test-openai-key"
            )
            assert llm_provider_overrides._OVERRIDE_CACHE_HEALTHY is True
            assert (
                llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
                is True
            )

        restored = _snapshot_override_state()
        assert restored == expected
        assert restored[0] is expected[0]
        assert restored[2]["_OVERRIDE_RECOVERY_TASK"] is recovery_task
        assert restored[2]["_OVERRIDE_REFRESH_SERVICE_TASK"] is refresh_service_task
    finally:
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
