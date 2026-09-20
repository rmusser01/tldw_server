"""Credential-snapshot regressions for the Anthropic-compatible Messages API."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.core.AuthNZ import byok_runtime
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides_module
from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import LLMProviderOverride
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    is_runtime_issued_provider_call_credentials,
)

pytestmark = pytest.mark.integration


def _messages_request(model: str = "openai/gpt-4o-mini") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
    }


def _openai_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-messages-credential-snapshot",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok", "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


@pytest.fixture(autouse=True)
def healthy_override_cache_between_tests():
    # Capture raw test state: the public getter correctly fails closed when a
    # background refresh has marked the cache unhealthy, which must not make
    # this test module order-dependent.
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


def test_messages_dispatch_uses_one_atomic_override_snapshot(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter receives the key and endpoint from the same override read."""
    snapshot_loads: list[bool] = []
    legacy_field_calls: list[str] = []
    adapter_calls: list[dict[str, Any]] = []

    def load_static_snapshot() -> dict[str, Any]:
        snapshot_loads.append(True)
        return {
            "openai_api": {
                "api_key": "legacy-key",
                "api_base_url": "https://legacy-key.example/v1",
            }
        }

    def legacy_field_lookup(provider: str):
        legacy_field_calls.append(provider)
        return {
            "credential_fields": {"base_url": "https://legacy-fields.example/v1"}
        }

    async def adapter_boundary(**kwargs: Any):
        adapter_calls.append(kwargs)
        return _openai_response()

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(
        messages_endpoint,
        "get_override_credentials",
        legacy_field_lookup,
        raising=False,
    )
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="atomic-key",
                credential_fields={"base_url": "https://example.com/atomic/v1"},
            )
        }
    )

    response = client_user_only.post("/api/v1/messages", json=_messages_request())

    assert response.status_code == 200, response.text
    assert snapshot_loads == [True]
    assert legacy_field_calls == []
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "atomic-key"
    assert adapter_calls[0]["app_config"]["openai_api"]["api_base_url"] == (
        "https://example.com/atomic/v1"
    )


def test_messages_partial_override_inherits_frozen_static_adapter_config(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key-only override must not discard the configured adapter endpoint/model."""

    snapshot_loads: list[bool] = []
    adapter_calls: list[dict[str, Any]] = []

    def load_static_snapshot() -> dict[str, Any]:
        snapshot_loads.append(True)
        return {
            "openai_api": {
                "api_key": "static-key",
                "api_base_url": "https://static-options.example/v1",
                "model": "gpt-4o-mini",
            }
        }

    async def adapter_boundary(**kwargs: Any):
        adapter_calls.append(kwargs)
        return _openai_response()

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="override-key",
            )
        }
    )

    response = client_user_only.post("/api/v1/messages", json=_messages_request())

    assert response.status_code == 200, response.text
    assert snapshot_loads == [True]
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "override-key"
    adapter_config = adapter_calls[0]["app_config"]["openai_api"]
    assert adapter_config["api_base_url"] == "https://static-options.example/v1"
    assert adapter_config["model"] == "gpt-4o-mini"
    assert "api_key" not in adapter_config


@pytest.mark.concurrent
def test_late_override_after_structured_absence_cannot_mix_messages_credentials(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late override cannot pair its endpoint with an earlier static key."""
    structured_absent = threading.Event()
    static_lookup_started = threading.Event()
    release_static_lookup = threading.Event()
    adapter_calls: list[dict[str, Any]] = []

    def capture_snapshot(provider: str):
        captured = overrides_module.capture_provider_override_call_snapshot(provider)
        structured_absent.set()

        class _GatedAbsentSnapshot:
            provider = captured.provider

            def enforce(self, model: str | None) -> None:
                captured.enforce(model)

            def ensure_healthy(self) -> None:
                captured.ensure_healthy()

            def server_fallback(self, base_fallback=None):
                static_lookup_started.set()
                if not release_static_lookup.wait(10):
                    raise TimeoutError("messages static-key race gate was not released")
                return captured.server_fallback(base_fallback)

        return _GatedAbsentSnapshot()

    async def adapter_boundary(**kwargs: Any):
        adapter_calls.append(kwargs)
        return _openai_response()

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        messages_endpoint,
        "capture_provider_override_call_snapshot",
        capture_snapshot,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_server_config_snapshot",
        lambda: {
            "openai_api": {
                "api_key": "configured-static-key",
                "api_base_url": "https://configured-static.example/v1",
            }
        },
    )
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client_user_only.post,
            "/api/v1/messages",
            json=_messages_request(),
        )
        try:
            assert static_lookup_started.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {
                    "openai": LLMProviderOverride(
                        provider="openai",
                        api_key="late-override-key",
                        credential_fields={
                            "base_url": "https://late-override.example/v1"
                        },
                    )
                }
            )
        finally:
            release_static_lookup.set()
        response = future.result(timeout=10)

    assert response.status_code == 200, response.text
    assert structured_absent.is_set()
    assert len(adapter_calls) == 1
    assert adapter_calls[0]["api_key"] == "configured-static-key"
    assert "late-override.example" not in repr(adapter_calls[0].get("app_config"))


@pytest.mark.parametrize("operation", ["messages", "count_tokens"])
@pytest.mark.concurrent
def test_messages_policy_and_credentials_share_one_override_snapshot(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A late disabled override cannot replace an already-enforced snapshot."""
    policy_enforced = threading.Event()
    fallback_started = threading.Event()
    release_fallback = threading.Event()
    converted_calls: list[dict[str, Any]] = []
    native_calls: list[dict[str, Any]] = []
    provider = "openai" if operation == "messages" else "anthropic"
    model = (
        "gpt-4o-mini"
        if operation == "messages"
        else "claude-3-sonnet-20240229"
    )
    base_url = f"https://example.com/{provider}-snapshot-a/v1"

    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {
            provider: LLMProviderOverride(
                provider=provider,
                is_enabled=True,
                allowed_models=[model],
                api_key=f"{provider}-snapshot-key-a",
                credential_fields={"base_url": base_url},
            )
        }
    )
    captured = overrides_module.capture_provider_override_call_snapshot(provider)

    class GatedSnapshot:
        provider = captured.provider

        def enforce(self, selected_model: str | None) -> None:
            captured.enforce(selected_model)
            policy_enforced.set()

        def policy_error(self, selected_model: str | None):
            return captured.policy_error(selected_model)

        def server_fallback(self, base_fallback=None):
            fallback_started.set()
            if not release_fallback.wait(10):
                raise TimeoutError("messages override fallback gate was not released")
            return captured.server_fallback(base_fallback)

        def ensure_healthy(self) -> None:
            captured.ensure_healthy()

    async def converted_boundary(**kwargs: Any):
        converted_calls.append(kwargs)
        return _openai_response()

    async def native_boundary(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, int]:
        native_calls.append({"url": url, "headers": headers, "payload": payload})
        return {"input_tokens": 3}

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(
        messages_endpoint,
        "capture_provider_override_call_snapshot",
        lambda _provider: GatedSnapshot(),
    )
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", converted_boundary)
    monkeypatch.setattr(messages_endpoint, "_native_post_json", native_boundary)

    path = (
        "/api/v1/messages"
        if operation == "messages"
        else "/api/v1/messages/count_tokens"
    )
    request_model = f"{provider}/{model}"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client_user_only.post,
            path,
            json=_messages_request(request_model),
        )
        try:
            assert policy_enforced.wait(10)
            assert fallback_started.wait(10)
            overrides_module.set_llm_provider_overrides_cache_for_tests(
                {
                    provider: LLMProviderOverride(
                        provider=provider,
                        is_enabled=False,
                        api_key=f"{provider}-late-key-b",
                        credential_fields={
                            "base_url": f"https://{provider}-late-b.example/v1"
                        },
                    )
                }
            )
        finally:
            release_fallback.set()
        response = future.result(timeout=10)

    assert response.status_code == 200, response.text
    if operation == "messages":
        assert len(converted_calls) == 1
        call = converted_calls[0]
        credentials = call[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]
        assert isinstance(credentials, ProviderCallCredentials)
        assert is_runtime_issued_provider_call_credentials(
            credentials,
            provider="openai",
        )
        assert call["api_key"] == credentials.api_key == "openai-snapshot-key-a"
        assert call["app_config"] is credentials.app_config
        assert call["app_config"]["openai_api"]["api_base_url"] == base_url
        assert native_calls == []
    else:
        assert len(native_calls) == 1
        assert native_calls[0]["headers"]["x-api-key"] == "anthropic-snapshot-key-a"
        assert native_calls[0]["url"] == f"{base_url}/messages/count_tokens"
        assert PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in native_calls[0]["payload"]
        assert converted_calls == []


def test_unhealthy_override_store_fails_messages_closed_before_dispatch(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_loads: list[bool] = []
    adapter_calls: list[dict[str, Any]] = []

    def load_static_snapshot() -> dict[str, Any]:
        snapshot_loads.append(True)
        return {"openai_api": {"api_key": "must-not-be-used"}}

    async def adapter_boundary(**kwargs: Any):
        adapter_calls.append(kwargs)
        return _openai_response()

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", adapter_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )

    response = client_user_only.post("/api/v1/messages", json=_messages_request())

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == {
        "error_code": "credential_store_unavailable",
        "message": "Provider credential storage is temporarily unavailable.",
    }
    assert snapshot_loads == [True]
    assert adapter_calls == []


@pytest.mark.parametrize(
    ("route_kind", "initial_config", "expected_status", "expected_key", "expected_url"),
    (
        (
            "converted",
            {
                "openai_api": {
                    "api_key": "openai-key-a",
                    "api_base_url": "https://openai-a.example/v1",
                }
            },
            200,
            "openai-key-a",
            "https://openai-a.example/v1",
        ),
        ("converted", {}, 503, None, None),
        (
            "native",
            {
                "anthropic_api": {
                    "api_key": "anthropic-key-a",
                    "api_base_url": "https://anthropic-a.example/v1",
                }
            },
            200,
            "anthropic-key-a",
            "https://anthropic-a.example/v1/messages",
        ),
        ("native", {}, 503, None, None),
    ),
    ids=(
        "converted-a-to-b",
        "converted-absent-to-b",
        "native-a-to-b",
        "native-absent-to-b",
    ),
)
def test_messages_static_fallback_freezes_one_generation_at_adapter_boundary(
    client_user_only,
    monkeypatch: pytest.MonkeyPatch,
    route_kind: str,
    initial_config: dict[str, Any],
    expected_status: int,
    expected_key: str | None,
    expected_url: str | None,
) -> None:
    """Native and converted dispatch cannot adopt a later config generation."""
    rotated_config = {
        "openai_api": {
            "api_key": "openai-key-b",
            "api_base_url": "https://openai-b.example/v1",
        },
        "anthropic_api": {
            "api_key": "anthropic-key-b",
            "api_base_url": "https://anthropic-b.example/v1",
        },
    }
    converted_calls: list[dict[str, Any]] = []
    native_calls: list[dict[str, Any]] = []

    def load_static_snapshot() -> dict[str, Any]:
        monkeypatch.setattr(byok_runtime, "loaded_config_data", rotated_config)
        return copy.deepcopy(initial_config)

    async def converted_boundary(**kwargs: Any):
        converted_calls.append(kwargs)
        return _openai_response()

    async def native_boundary(
        url: str,
        headers: dict[str, str],
        _payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        native_calls.append({"url": url, "headers": headers})
        return {
            "id": "msg_static_snapshot",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-3-5-sonnet-latest",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: False)
    monkeypatch.setattr(runtime_module, "load_server_config_snapshot", load_static_snapshot)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", initial_config)
    monkeypatch.setattr(messages_endpoint, "perform_chat_api_call_async", converted_boundary)
    monkeypatch.setattr(messages_endpoint, "_native_post_json", native_boundary)
    overrides_module.set_llm_provider_overrides_cache_for_tests({})

    model = (
        "openai/gpt-4o-mini"
        if route_kind == "converted"
        else "anthropic/claude-3-5-sonnet-latest"
    )
    response = client_user_only.post(
        "/api/v1/messages",
        json=_messages_request(model),
    )

    assert response.status_code == expected_status, response.text
    if expected_status != 200:
        assert converted_calls == []
        assert native_calls == []
    elif route_kind == "converted":
        assert len(converted_calls) == 1
        assert converted_calls[0]["api_key"] == expected_key
        assert converted_calls[0]["app_config"]["openai_api"]["api_base_url"] == expected_url
        assert native_calls == []
    else:
        assert len(native_calls) == 1
        assert native_calls[0]["headers"]["x-api-key"] == expected_key
        assert native_calls[0]["url"] == expected_url
        assert converted_calls == []
