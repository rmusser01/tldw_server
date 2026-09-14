"""Public regressions for provider-override credential-store outages."""

from __future__ import annotations

import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    LLMProviderOverridesRefreshError,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat import chat_service

pytestmark = pytest.mark.integration


def _chat_request() -> dict[str, Any]:
    return {
        "api_provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "save_to_db": False,
    }


def _configure_no_dispatch_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    static_calls: list[str],
    dispatch_calls: list[str],
) -> None:
    def static_fallback(provider: str, **_kwargs: Any):
        static_calls.append(provider)
        return "must-not-be-used", {"source": "static"}

    def dispatch(**kwargs: Any):
        dispatch_calls.append(str(kwargs.get("api_endpoint")))
        raise AssertionError("provider dispatch must not run while the override store is unhealthy")

    monkeypatch.setenv("CHAT_FORCE_MOCK", "false")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(chat_endpoint, "resolve_provider_api_key", static_fallback)
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", dispatch)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: None)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False)
    monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False)


@pytest.fixture(autouse=True)
def healthy_override_cache_between_tests():
    # Capture raw cache state so an intentionally unhealthy prior test cannot
    # make fixture setup fail, and restore health/TTL semantics exactly.
    with overrides_module._OVERRIDE_LOCK:
        original = copy.deepcopy(overrides_module._OVERRIDE_CACHE)
        original_healthy = overrides_module._OVERRIDE_CACHE_HEALTHY
        original_ttl_enabled = not overrides_module._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    overrides_module.set_llm_provider_overrides_cache_for_tests(original)
    try:
        yield
    finally:
        overrides_module.set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=original_healthy,
            ttl_enabled=original_ttl_enabled,
        )


def test_public_chat_fails_closed_before_static_fallback_or_provider_dispatch(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    static_calls: list[str] = []
    dispatch_calls: list[str] = []
    _configure_no_dispatch_boundary(
        monkeypatch,
        static_calls=static_calls,
        dispatch_calls=dispatch_calls,
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )

    response = authenticated_client.post("/api/v1/chat/completions", json=_chat_request())

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == {
        "error_code": "credential_store_unavailable",
        "message": "Provider credential storage is temporarily unavailable.",
    }
    assert static_calls == []
    assert dispatch_calls == []


def test_legacy_chat_key_resolver_cannot_bypass_unhealthy_override_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy adapters share the same fail-closed credential boundary."""
    # This regression exercises the dynamic/config fallback used in production,
    # not the explicit module-key seam installed by the Chat test harness.
    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {})
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {})
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys",
        lambda: {"openai": "configured-static-key"},
    )
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="stale-key")},
        healthy=False,
    )

    with pytest.raises(ByokResolutionError) as exc_info:
        chat_service.resolve_provider_api_key("openai")

    assert exc_info.value.code == "credential_store_unavailable"


@pytest.mark.concurrent
def test_failed_refresh_racing_two_public_resolutions_never_reaches_static_fallback(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_arrived = threading.Event()
    release_refresh = threading.Event()
    refresh_done = threading.Event()
    both_resolutions_arrived = threading.Event()
    resolution_lock = threading.Lock()
    resolution_count = 0
    static_calls: list[str] = []
    dispatch_calls: list[str] = []
    last_good = LLMProviderOverride(provider="openai", api_key="last-good-key")

    class FailingRepo:
        def __init__(self, _pool: object) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self) -> list[dict[str, Any]]:
            refresh_arrived.set()
            released = await asyncio.to_thread(release_refresh.wait, 10)
            if not released:
                raise TimeoutError("refresh test gate was not released")
            raise RuntimeError("override database unavailable at /private/secret.db")

    async def gated_resolver(provider: str, **kwargs: Any):
        nonlocal resolution_count
        with resolution_lock:
            resolution_count += 1
            if resolution_count == 2:
                both_resolutions_arrived.set()
        if not await asyncio.to_thread(refresh_done.wait, 10):
            raise ByokResolutionError("credential_store_unavailable", provider)
        fallback_resolver = kwargs.pop("fallback_resolver", None)
        if fallback_resolver is not None:
            kwargs["fallback_override"] = fallback_resolver(provider)
        return await byok_runtime.resolve_byok_credentials(provider, **kwargs)

    class GatedCredentialRuntime(ProviderCredentialRuntime):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(resolver=gated_resolver, **kwargs)

    _configure_no_dispatch_boundary(
        monkeypatch,
        static_calls=static_calls,
        dispatch_calls=dispatch_calls,
    )
    monkeypatch.setattr(overrides_module, "AuthnzLLMProviderOverridesRepo", FailingRepo)
    monkeypatch.setattr(chat_endpoint, "ProviderCredentialRuntime", GatedCredentialRuntime)
    overrides_module.set_llm_provider_overrides_cache_for_tests({"openai": last_good})

    def refresh() -> dict[str, LLMProviderOverride]:
        try:
            return asyncio.run(overrides_module.refresh_llm_provider_overrides(pool=object()))
        finally:
            refresh_done.set()

    def post():
        return authenticated_client.post("/api/v1/chat/completions", json=_chat_request())

    with ThreadPoolExecutor(max_workers=3) as executor:
        refresh_future = executor.submit(refresh)
        assert refresh_arrived.wait(10)
        first_response = executor.submit(post)
        second_response = executor.submit(post)
        try:
            assert both_resolutions_arrived.wait(10)
        finally:
            release_refresh.set()
        with pytest.raises(LLMProviderOverridesRefreshError):
            refresh_future.result(timeout=10)
        responses = (
            first_response.result(timeout=10),
            second_response.result(timeout=10),
        )

    with overrides_module._OVERRIDE_LOCK:
        assert dict(overrides_module._OVERRIDE_CACHE) == {"openai": last_good}
    assert [response.status_code for response in responses] == [503, 503]
    assert {
        response.json()["detail"]["error_code"] for response in responses
    } == {"credential_store_unavailable"}
    assert static_calls == []
    assert dispatch_calls == []


@pytest.mark.concurrent
def test_late_override_after_structured_absence_cannot_mix_chat_adapter_credentials(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late override cannot replace the static key after atomic absence."""
    from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime

    structured_absent = threading.Event()
    static_lookup_started = threading.Event()
    release_static_lookup = threading.Event()
    static_calls: list[str] = []
    adapter_calls: list[dict[str, Any]] = []

    def capture_snapshot(provider: str):
        snapshot = overrides_module.capture_provider_override_call_snapshot(provider)
        structured_absent.set()
        return snapshot

    real_static_lookup = (
        provider_credential_runtime.resolve_static_server_fallback_from_snapshot
    )

    def gated_static_lookup(provider: str, config_snapshot: dict[str, Any]):
        assert structured_absent.is_set()
        static_calls.append(provider)
        static_lookup_started.set()
        if not release_static_lookup.wait(10):
            raise TimeoutError("chat static-key race gate was not released")
        return real_static_lookup(provider, config_snapshot)

    def adapter_boundary(**kwargs: Any):
        adapter_calls.append(kwargs)
        return {
            "id": "chatcmpl-static-snapshot",
            "object": "chat.completion",
            "created": 1,
            "model": kwargs.get("model") or "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setenv("CHAT_FORCE_MOCK", "false")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        chat_endpoint,
        "capture_provider_override_call_snapshot",
        capture_snapshot,
    )
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "configured-static-key"}},
    )
    monkeypatch.setattr(
        provider_credential_runtime,
        "resolve_static_server_fallback_from_snapshot",
        gated_static_lookup,
    )
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", adapter_boundary)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: None)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False)
    monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            authenticated_client.post,
            "/api/v1/chat/completions",
            json=_chat_request(),
        )
        try:
            assert structured_absent.wait(10)
            assert static_lookup_started.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {
                    "openai": LLMProviderOverride(
                        provider="openai",
                        api_key="late-override-key",
                        credential_fields={
                            "base_url": "https://late-override.example/v1",
                        },
                    )
                }
            )
        finally:
            release_static_lookup.set()
        response = future.result(timeout=10)

    assert response.status_code == 200, response.text
    assert static_calls == ["openai"]
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "configured-static-key"
    assert "late-override.example" not in repr(adapter_calls[0].get("app_config"))


@pytest.mark.parametrize(
    ("initial_config", "expected_key", "expected_base_url"),
    (
        (
            {
                "openai_api": {
                    "api_key": "static-key-a",
                    "api_base_url": "https://static-a.example/v1",
                }
            },
            "static-key-a",
            "https://static-a.example/v1",
        ),
        ({}, None, None),
    ),
    ids=("a-to-b", "absent-to-b"),
)
def test_chat_static_fallback_freezes_one_config_generation_at_adapter_boundary(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
    initial_config: dict[str, Any],
    expected_key: str | None,
    expected_base_url: str | None,
) -> None:
    """Chat cannot combine an earlier static decision with a later config load."""
    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, provider_credential_runtime

    rotated_config = {
        "openai_api": {
            "api_key": "static-key-b",
            "api_base_url": "https://static-b.example/v1",
        }
    }
    legacy_lookup_finished = False
    dynamic_key_reads = 0
    adapter_calls: list[dict[str, Any]] = []

    def legacy_dynamic_keys() -> dict[str, str]:
        nonlocal dynamic_key_reads, legacy_lookup_finished
        dynamic_key_reads += 1
        legacy_lookup_finished = True
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        initial_section = initial_config.get("openai_api")
        initial_key = (
            initial_section.get("api_key")
            if isinstance(initial_section, dict)
            else None
        )
        return {"openai": initial_key} if isinstance(initial_key, str) else {}

    def load_static_snapshot() -> dict[str, Any]:
        selected = rotated_config if legacy_lookup_finished else initial_config
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        return copy.deepcopy(selected)

    def adapter_boundary(**kwargs: Any) -> dict[str, Any]:
        adapter_calls.append(kwargs)
        return {
            "id": "chatcmpl-static-config-snapshot",
            "object": "chat.completion",
            "created": 1,
            "model": kwargs.get("model") or "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setenv("CHAT_FORCE_MOCK", "false")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(byok_runtime, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_helpers, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        load_static_snapshot,
    )
    monkeypatch.setattr(byok_runtime, "loaded_config_data", initial_config)
    monkeypatch.setattr(chat_request_schemas, "get_api_keys", legacy_dynamic_keys)
    monkeypatch.setattr(chat_request_schemas, "API_KEYS", {})
    monkeypatch.setattr(chat_endpoint, "API_KEYS", {})
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", adapter_boundary)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: None)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False)
    monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=_chat_request(),
    )

    assert dynamic_key_reads == 0
    if expected_key is None:
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["error_code"] == "missing_provider_credentials"
        assert adapter_calls == []
        return

    assert response.status_code == 200, response.text
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == expected_key
    assert adapter_calls[0]["app_config"] == {
        "openai_api": {"api_base_url": expected_base_url}
    }


@pytest.mark.parametrize(
    ("late_policy", "expected_code"),
    (
        ({"is_enabled": False, "allowed_models": ["gpt-4o-mini"]}, "provider_disabled"),
        ({"is_enabled": True, "allowed_models": ["gpt-4.1"]}, "model_not_allowed"),
    ),
)
@pytest.mark.concurrent
def test_late_chat_override_policy_change_blocks_provider_dispatch(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
    late_policy: dict[str, Any],
    expected_code: str,
) -> None:
    """The credential-resolution snapshot rechecks policy before dispatch."""
    validation_complete = threading.Event()
    release_validation = threading.Event()
    dispatch_calls: list[str] = []
    original_validate = chat_endpoint.validate_provider_override

    def gated_validate(provider: str, model: str | None):
        decision = original_validate(provider, model)
        validation_complete.set()
        if not release_validation.wait(10):
            raise TimeoutError("chat policy race gate was not released")
        return decision

    def adapter_boundary(**kwargs: Any):
        dispatch_calls.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-policy-race",
            "object": "chat.completion",
            "created": 1,
            "model": kwargs.get("model") or "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "must not dispatch"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setenv("CHAT_FORCE_MOCK", "false")
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(chat_endpoint, "validate_provider_override", gated_validate)
    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", adapter_boundary)
    monkeypatch.setattr(chat_endpoint, "get_provider_manager", lambda: None)
    monkeypatch.setattr(chat_endpoint, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_service, "get_request_queue", lambda: None)
    monkeypatch.setattr(chat_endpoint, "ENABLE_PROVIDER_FALLBACK", False)
    monkeypatch.setattr(chat_endpoint, "QUEUED_EXECUTION", False)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=True,
                allowed_models=["gpt-4o-mini"],
                api_key="enabled-key",
            )
        }
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            authenticated_client.post,
            "/api/v1/chat/completions",
            json=_chat_request(),
        )
        try:
            assert validation_complete.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {
                    "openai": LLMProviderOverride(
                        provider="openai",
                        api_key="late-policy-key",
                        **late_policy,
                    )
                }
            )
        finally:
            release_validation.set()
        response = future.result(timeout=10)

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error_code"] == expected_code
    assert dispatch_calls == []


def test_chat_autoswitch_cannot_select_disabled_openai_override(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-to-OpenAI autoswitch applies OpenAI policy before dispatch."""
    static_calls: list[str] = []
    dispatch_calls: list[str] = []
    _configure_no_dispatch_boundary(
        monkeypatch,
        static_calls=static_calls,
        dispatch_calls=dispatch_calls,
    )
    monkeypatch.setattr(chat_endpoint, "_get_default_provider", lambda: "local-llm")
    monkeypatch.setattr(chat_endpoint, "ALLOW_AUTOSWITCH_TO_OPENAI", True)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=False,
                allowed_models=["gpt-4o-mini"],
                api_key="disabled-openai-key",
            )
        }
    )
    request_payload = _chat_request()
    request_payload.pop("api_provider")
    request_payload["model"] = "local-model"

    response = authenticated_client.post(
        "/api/v1/chat/completions",
        json=request_payload,
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error_code"] == "provider_disabled"
    assert dispatch_calls == []
