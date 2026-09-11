"""
Unit test to ensure completion pre-check uses efficient count instead of bulk-loading messages.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.LLM_Calls.routing.decision_store import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.LLM_Calls.routing.models import (
    RouterRequest,
    RoutingDecision,
    RoutingPolicy,
)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("captured_key", ["router-key-a", None], ids=["a-to-b", "absent-to-b"])
async def test_auto_router_dispatch_keeps_one_static_credential_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    captured_key: str | None,
) -> None:
    """Router dispatch must not splice a later server config into its key snapshot."""
    config_a = {"local_llm_api": {"base_url": "http://generation-a.invalid"}}
    current_key = {"value": captured_key}
    captured: dict[str, Any] = {}
    lifecycle: list[Any] = []
    handles: list[Any] = []

    class RecordingCredentialRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs["user_id"]))
            assert "fallback_resolver" not in kwargs
            self._api_key = captured_key
            self._app_config = config_a

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            current_key["value"] = "router-key-b"
            handle = SimpleNamespace(
                provider=provider,
                api_key=self._api_key,
                app_config=self._app_config,
                credentials_resolved=True,
            )
            handles.append(handle)
            return handle

        async def mark_used(self, _credentials: Any) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    async def provider_call(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": '{"provider":"local-llm","model":"routed"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    from tldw_Server_API.app.api.v1.schemas import chat_request_schemas

    monkeypatch.setattr(
        chat_request_schemas,
        "get_api_keys",
        lambda: {"local-llm": current_key["value"]},
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "ProviderCredentialRuntime",
        RecordingCredentialRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        character_chat_sessions,
        "perform_chat_api_call_async",
        provider_call,
    )

    await character_chat_sessions._select_auto_character_llm_router_choice(
        router_request=RouterRequest(
            model="auto",
            surface="character_chat",
            latest_user_turn="route me",
            scope="chat-1",
        ),
        policy=RoutingPolicy(
            request_model="auto",
            server_default_provider="local-llm",
            boundary_mode="server_default_provider",
        ),
        candidates=[
            {"provider": "local-llm", "model": "routed"},
            {"provider": "local-llm", "model": "alternate"},
        ],
        provider_listing={
            "default_provider": "local-llm",
            "providers": [
                {
                    "name": "local-llm",
                    "default_model": "router-model",
                    "is_configured": True,
                    "models_info": [{"name": "router-model"}],
                }
            ],
        },
        current_user=SimpleNamespace(id=1),
    )

    assert captured["api_key"] == captured_key
    assert captured["app_config"] == config_a
    assert captured["credentials_resolved"] is True
    assert captured[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handles[0]
    assert lifecycle == [
        ("init", 1),
        ("resolve", "local-llm", "router-model"),
        "mark_used",
        "close",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_router_logs_only_error_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Router/provider diagnostics must not copy exception text into logs."""

    usage_sentinel = "sk-router-usage-/private/usage.sqlite"
    provider_sentinel = "sk-router-provider-/private/provider.json"
    logged: list[str] = []

    class _Runtime:
        async def close(self) -> None:
            return None

    async def _usage_log(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(usage_sentinel)

    async def _select(**kwargs: Any):
        await kwargs["log_router_usage"](
            SimpleNamespace(provider="openai", model="router-model"),
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            1.0,
        )
        raise RuntimeError(provider_sentinel)

    def _debug(message: str, *args: Any, **_kwargs: Any) -> None:
        logged.append(message.format(*args))

    monkeypatch.setattr(character_chat_sessions, "log_model_router_usage", _usage_log)
    monkeypatch.setattr(character_chat_sessions, "select_llm_router_choice", _select)
    monkeypatch.setattr(character_chat_sessions.logger, "debug", _debug)

    choice, diagnostics = await character_chat_sessions._select_auto_character_llm_router_choice(
        router_request=RouterRequest(
            model="auto",
            surface="character_chat",
            latest_user_turn="route me",
            scope="chat-1",
        ),
        policy=RoutingPolicy(
            request_model="auto",
            server_default_provider="openai",
            boundary_mode="server_default_provider",
        ),
        candidates=[{"provider": "openai", "model": "router-model"}],
        provider_listing={"default_provider": "openai", "providers": []},
        current_user=SimpleNamespace(id=1),
        credential_runtime=_Runtime(),
    )

    assert choice is None
    assert diagnostics == {"error": "RuntimeError"}
    assert usage_sentinel not in "\n".join(logged)
    assert provider_sentinel not in "\n".join(logged)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_boundary", ["usage", "selection"])
async def test_auto_router_propagates_cancellation_and_closes_runtime_once(
    monkeypatch: pytest.MonkeyPatch,
    cancel_boundary: str,
) -> None:
    """Cancellation is control flow, not a recoverable auto-router failure."""
    entered = asyncio.Event()
    release = asyncio.Event()
    close_count = 0

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    async def blocking_usage_log(*_args: Any, **_kwargs: Any) -> None:
        entered.set()
        await release.wait()

    async def select(**kwargs: Any):
        if cancel_boundary == "usage":
            await kwargs["log_router_usage"](
                SimpleNamespace(provider="openai", model="router-model"),
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                1.0,
            )
            return None, {}
        entered.set()
        await release.wait()
        return None, {}

    monkeypatch.setattr(character_chat_sessions, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        character_chat_sessions,
        "log_model_router_usage",
        blocking_usage_log,
    )
    monkeypatch.setattr(character_chat_sessions, "select_llm_router_choice", select)

    task = asyncio.create_task(
        character_chat_sessions._select_auto_character_llm_router_choice(
            router_request=RouterRequest(
                model="auto",
                surface="character_chat",
                latest_user_turn="route me",
                scope="chat-1",
            ),
            policy=RoutingPolicy(
                request_model="auto",
                server_default_provider="openai",
                boundary_mode="server_default_provider",
            ),
            candidates=[{"provider": "openai", "model": "router-model"}],
            provider_listing={"default_provider": "openai", "providers": []},
            current_user=SimpleNamespace(id=1),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert close_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("late_outcome", "expected_marks"),
    [
        ("valid_route", 1),
        ("empty", 0),
        ("error", 0),
        ("mixed_error_route", 0),
        ("nested_error_route", 0),
        ("malformed_route", 0),
    ],
)
async def test_auto_router_cancellation_marks_and_closes_after_adapter_exit(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
    expected_marks: int,
) -> None:
    """Character routing marks only a semantic late route before teardown."""

    entered = asyncio.Event()
    release = asyncio.Event()
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="router-key",
        app_config={},
        credentials_resolved=True,
    )

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    async def provider_call(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["api_key"] == "router-key"
        entered.set()
        await release.wait()
        lifecycle.append("adapter-exit")
        if late_outcome == "valid_route":
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"provider":"openai","model":"routed"}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        if late_outcome == "empty":
            return {"choices": []}
        if late_outcome == "error":
            return {"error": {"code": "provider_unavailable"}}
        if late_outcome == "mixed_error_route":
            return {
                "error": {"code": "provider_unavailable"},
                "provider": "openai",
                "model": "routed",
            }
        if late_outcome == "nested_error_route":
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                'Error: {"provider":"openai","model":"routed"}'
                            )
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": "not a route"}}]}

    async def select(**kwargs: Any):
        router_model = SimpleNamespace(provider="openai", model="router-model")
        await kwargs["execute_router_call"](router_model, [])
        return {"provider": "openai", "model": "routed"}, {}

    monkeypatch.setattr(character_chat_sessions, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(character_chat_sessions, "select_llm_router_choice", select)

    task = asyncio.create_task(
        character_chat_sessions._select_auto_character_llm_router_choice(
            router_request=RouterRequest(
                model="auto",
                surface="character_chat",
                latest_user_turn="route me",
                scope="chat-1",
            ),
            policy=RoutingPolicy(
                request_model="auto",
                server_default_provider="openai",
                boundary_mode="server_default_provider",
            ),
            candidates=[{"provider": "openai", "model": "router-model"}],
            provider_listing={"default_provider": "openai", "providers": []},
            current_user=SimpleNamespace(id=1),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        assert lifecycle == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    expected_lifecycle = ["adapter-exit"]
    expected_lifecycle.extend(["mark"] * expected_marks)
    expected_lifecycle.append("close")
    assert lifecycle == expected_lifecycle


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router_result", "expected_choice", "expected_marks"),
    [
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"provider":"openai","model":"routed"}'
                        }
                    }
                ]
            },
            {"provider": "openai", "model": "routed"},
            1,
        ),
        ({"choices": []}, None, 0),
        ({"error": {"code": "provider_unavailable"}}, None, 0),
        (
            {
                "error": {"code": "provider_unavailable"},
                "provider": "openai",
                "model": "routed",
            },
            None,
            0,
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                'Error: {"provider":"openai","model":"routed"}'
                            )
                        }
                    }
                ]
            },
            None,
            0,
        ),
        ({"choices": [{"message": {"content": "not a route"}}]}, None, 0),
    ],
    ids=[
        "valid-route",
        "empty",
        "error",
        "mixed-error-route",
        "nested-error-route",
        "malformed-route",
    ],
)
async def test_auto_router_normal_result_marks_only_valid_route(
    monkeypatch: pytest.MonkeyPatch,
    router_result: dict[str, Any],
    expected_choice: dict[str, str] | None,
    expected_marks: int,
) -> None:
    """Normal routing must validate the route before usage accounting."""
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="router-key",
        app_config={},
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            lifecycle.append("mark")

        async def close(self) -> None:
            lifecycle.append("close")

    async def provider_call(**_kwargs: Any) -> dict[str, Any]:
        return router_result

    async def select(**kwargs: Any):
        router_model = SimpleNamespace(provider="openai", model="router-model")
        result = await kwargs["execute_router_call"](router_model, [])
        choice = character_chat_sessions.extract_router_choice(result)
        return choice, {"choice_received": choice is not None}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(character_chat_sessions, "select_llm_router_choice", select)

    choice, _diagnostics = (
        await character_chat_sessions._select_auto_character_llm_router_choice(
            router_request=RouterRequest(
                model="auto",
                surface="character_chat",
                latest_user_turn="route me",
                scope="chat-1",
            ),
            policy=RoutingPolicy(
                request_model="auto",
                server_default_provider="openai",
                boundary_mode="server_default_provider",
            ),
            candidates=[{"provider": "openai", "model": "routed"}],
            provider_listing={"default_provider": "openai", "providers": []},
            current_user=SimpleNamespace(id=1),
            credential_runtime=Runtime(),
        )
    )

    assert choice == expected_choice
    assert lifecycle.count("mark") == expected_marks
    assert "close" not in lifecycle


@pytest.mark.unit
@pytest.mark.asyncio
async def test_auto_router_valid_result_drains_mark_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation after a valid result cannot close before its usage mark exits."""
    mark_entered = asyncio.Event()
    mark_release = asyncio.Event()
    lifecycle: list[str] = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="router-key",
        app_config={},
        credentials_resolved=True,
    )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            del model
            return handle

        async def mark_used(self, selected_handle: Any) -> None:
            assert selected_handle is handle
            mark_entered.set()
            await mark_release.wait()
            lifecycle.append("mark-exit")

        async def close(self) -> None:
            lifecycle.append("close")

    async def provider_call(**_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openai","model":"routed"}'
                    }
                }
            ]
        }

    async def select(**kwargs: Any):
        router_model = SimpleNamespace(provider="openai", model="router-model")
        result = await kwargs["execute_router_call"](router_model, [])
        choice = character_chat_sessions.extract_router_choice(result)
        return choice, {"choice_received": choice is not None}

    monkeypatch.setattr(character_chat_sessions, "perform_chat_api_call_async", provider_call)
    monkeypatch.setattr(character_chat_sessions, "select_llm_router_choice", select)

    task = asyncio.create_task(
        character_chat_sessions._select_auto_character_llm_router_choice(
            router_request=RouterRequest(
                model="auto",
                surface="character_chat",
                latest_user_turn="route me",
                scope="chat-1",
            ),
            policy=RoutingPolicy(
                request_model="auto",
                server_default_provider="openai",
                boundary_mode="server_default_provider",
            ),
            candidates=[{"provider": "openai", "model": "routed"}],
            provider_listing={"default_provider": "openai", "providers": []},
            current_user=SimpleNamespace(id=1),
            credential_runtime=Runtime(),
        )
    )
    try:
        await asyncio.wait_for(mark_entered.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert lifecycle == []
        mark_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        mark_release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["mark-exit"]


@pytest.mark.unit
def test_completion_precheck_uses_count_not_bulk_get(test_client, auth_headers, character_db):
     # Create a character and a chat with a few messages
    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "CountCheck",
            "description": "",
            "personality": "",
            "first_message": "Hi"
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Count Test"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    # Add a few messages
    for i in range(3):
        test_client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"role": "user" if i % 2 == 0 else "assistant", "content": f"Msg {i}"},
            headers=auth_headers,
        )

    # Wrap DB methods to record usage
    original_count = character_db.count_messages_for_conversation
    original_get = character_db.get_messages_for_conversation

    calls: dict[str, Any] = {"count_calls": 0, "get_limits": []}

    def count_wrapper(conversation_id: str) -> int:
        calls["count_calls"] += 1
        return original_count(conversation_id)

    def get_wrapper(
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        calls["get_limits"].append(limit)
        return original_get(conversation_id, limit=limit, offset=offset, **kwargs)

    character_db.count_messages_for_conversation = count_wrapper
    character_db.get_messages_for_conversation = get_wrapper

    # Trigger completion pre-check (offline sim path)
    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "Check",
            "stream": False,
            "include_character_context": False,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200

    # Verify a count was used at least once and that no huge-limit fetch was used (10000)
    assert calls["count_calls"] >= 1
    assert 10000 not in calls["get_limits"], "Bulk get with 10000 limit should not be used for pre-check"


@pytest.mark.unit
def test_complete_v2_explicit_unavailable_model_returns_400(test_client, auth_headers, monkeypatch):
    monkeypatch.setenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "1")

    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "StrictModelCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Strict model check"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.is_model_known_for_provider",
        lambda provider, model: False,
    )

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "openai",
            "model": "gpt-not-installed",
            "append_user_message": "Hello",
            "stream": False,
            "include_character_context": False,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["error_code"] == "model_not_available"
    assert detail["provider"] == "openai"
    assert detail["model"] == "gpt-not-installed"


@pytest.mark.unit
def test_complete_v2_auto_model_routes_before_strict_availability_check(
    test_client,
    auth_headers,
    monkeypatch,
):
    monkeypatch.setenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION", "1")
    monkeypatch.setenv("ENABLE_LOCAL_LLM_PROVIDER", "1")

    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "AutoRouteCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Auto route strict check"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]
    injected_store = InMemoryRoutingDecisionStore()
    test_client.app.state.routing_decision_store = injected_store

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.get_configured_providers",
        lambda: {
            "default_provider": "local-llm",
            "providers": [
                {
                    "name": "local-llm",
                    "is_configured": True,
                    "default_model": "local-test-router",
                    "models_info": [
                        {"name": "local-test-router"},
                        {"name": "local-test-routed"},
                    ],
                }
            ],
        },
    )

    async def _stub_router_call(**kwargs):
        captured["router_call"] = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"local-llm","model":"local-test-routed"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            },
        }

    async def _stub_router_usage(**kwargs):
        captured.setdefault("router_usage", []).append(kwargs)

    def _stub_route_model(*, request, policy, candidates, provider_order, sticky_store=None, llm_router_choice=None):
        captured["routing_request"] = request
        captured["routing_policy"] = policy
        captured["routing_candidates"] = candidates
        captured["routing_provider_order"] = provider_order
        captured["routing_sticky_store"] = sticky_store
        captured["routing_llm_choice"] = llm_router_choice
        return RoutingDecision(
            provider="local-llm",
            model="local-test-routed",
            canonical=True,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.route_model",
        _stub_route_model,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.perform_chat_api_call_async",
        _stub_router_call,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.log_model_router_usage",
        _stub_router_usage,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.is_model_known_for_provider",
        lambda provider, model: False,
    )

    def _stub_chat_api_call(api_endpoint, messages_payload, **kwargs):
        captured["provider_call"] = {
            "api_endpoint": api_endpoint,
            "model": kwargs.get("model"),
            "streaming": kwargs.get("streaming"),
        }
        return {"choices": [{"message": {"content": "auto routed response"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.perform_chat_api_call",
        _stub_chat_api_call,
    )

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "model": "auto",
            "routing": {"mode": "sticky_session"},
            "append_user_message": "Route this automatically",
            "stream": False,
            "include_character_context": False,
            "save_to_db": False,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["provider"] == "local-llm"
    assert payload["model"] == "local-test-routed"
    assert payload["assistant_content"] == "auto routed response"
    assert captured["routing_request"].model == "auto"
    assert captured["routing_policy"].mode == "sticky_session"
    assert captured["routing_sticky_store"] is injected_store
    assert captured["routing_llm_choice"] == {
        "provider": "local-llm",
        "model": "local-test-routed",
    }
    assert captured["router_call"]["model"] == "local-test-router"
    assert captured["router_usage"][0]["provider"] == "local-llm"
    assert captured["provider_call"] == {
        "api_endpoint": "local-llm",
        "model": "local-test-routed",
        "streaming": False,
    }


@pytest.mark.unit
def test_complete_v2_rejects_routing_overrides_for_non_auto_models(
    test_client,
    auth_headers,
):
    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "RoutingValidatorCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Routing validator"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "model": "local-test",
            "routing": {"mode": "per_turn"},
            "append_user_message": "Hello",
            "stream": False,
            "include_character_context": False,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 422


@pytest.mark.unit
def test_complete_v2_surfaces_provider_model_resolution_failures(
    test_client,
    auth_headers,
    monkeypatch,
):
    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "ResolutionFailureCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Resolution failure"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    def _raise_resolution_failure(*args, **kwargs):
        raise RuntimeError("resolution exploded")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.resolve_provider_and_model",
        _raise_resolution_failure,
    )

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "Hello",
            "stream": False,
            "include_character_context": False,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error_code"] == "provider_model_resolution_failed"


@pytest.mark.unit
def test_complete_v2_maps_input_error_to_400(
    test_client,
    auth_headers,
    monkeypatch,
):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "InputErrorCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Input error completion"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    def _raise_input_error(*args, **kwargs):
        raise InputError("completion payload is invalid")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.post_message_to_conversation",
        _raise_input_error,
    )

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "Hello",
            "stream": False,
            "include_character_context": False,
            "save_to_db": True,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "completion payload is invalid"


@pytest.mark.unit
def test_complete_v2_maps_oversize_input_error_to_413(
    test_client,
    auth_headers,
    monkeypatch,
):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "OversizeInputCharacter",
            "description": "",
            "personality": "",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Oversize input completion"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    def _raise_input_error(*args, **kwargs):
        raise InputError("Completion attachment exceeds maximum size")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.post_message_to_conversation",
        _raise_input_error,
    )

    resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "Hello",
            "stream": False,
            "include_character_context": False,
            "save_to_db": True,
        },
        headers=auth_headers,
    )

    assert resp.status_code == 413
    assert resp.json()["detail"] == "Completion attachment exceeds maximum size"
