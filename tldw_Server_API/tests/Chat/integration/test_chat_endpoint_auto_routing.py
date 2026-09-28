"""Ordinary chat routing preserves admission before credentials and generation."""

from unittest.mock import patch

import pytest
from fastapi import status

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.LLM_Calls.routing.decision_store import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    _create_persona_conversation,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_client as persona_chat_client,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_db as persona_chat_db,
)

pytest_plugins = (
    "tldw_Server_API.tests.Chat.credential_runtime_fixtures",
)


@pytest.mark.integration
@pytest.mark.parametrize("model", ["gpt-4", "auto"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["inactive", "deleted", "missing", "wrong_owner", "disabled", "malformed"])
def test_persona_rejection_precedes_router_credentials_and_message_effects(
    persona_chat_client, persona_chat_db, monkeypatch, model, stream, failure,
):
    """Fixed/auto and streaming requests cannot route or write with an unusable Persona."""
    client, headers, provider_call = persona_chat_client
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: failure != "disabled")
    cid, _ = _create_persona_conversation(persona_chat_db, persona_id="admission-persona")
    expected = {"inactive": (409, "persona_unavailable"), "deleted": (404, "persona_not_found"),
                "missing": (404, "persona_not_found"), "wrong_owner": (404, "persona_not_found"),
                "disabled": (503, "persona_feature_disabled"), "malformed": (409, "persona_binding_invalid")}[failure]
    with persona_chat_db.transaction() as conn:
        if failure == "inactive":
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "admission-persona"))
        elif failure == "deleted":
            conn.execute("UPDATE persona_profiles SET deleted = ? WHERE id = ?", (True, "admission-persona"))
        elif failure == "wrong_owner":
            conn.execute("UPDATE persona_profiles SET user_id = ? WHERE id = ?", ("other", "admission-persona"))
        elif failure in {"missing", "malformed"}:
            conn.execute("UPDATE conversations SET assistant_id = ? WHERE id = ?",
                         (None if failure == "malformed" else "private-missing", cid))
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", wraps=chat_endpoint.ProviderCredentialRuntime) as credentials,
        patch.object(chat_endpoint, "_resolve_auto_chat_routing_decision", wraps=chat_endpoint._resolve_auto_chat_routing_decision) as router,
        patch.object(persona_chat_db, "add_message", wraps=persona_chat_db.add_message) as message,
    ):
        response = client.post("/api/v1/chat/completions", headers=headers, json={
            "model": model, "api_provider": "openai", "conversation_id": cid,
            "stream": stream, "save_to_db": True,
            "messages": [{"role": "user", "content": "Private turn"}],
        })
    assert response.status_code == expected[0], response.text
    assert response.json()["detail"]["code"] == expected[1]
    assert credentials.call_count == router.call_count == message.call_count == provider_call.call_count == 0
    assert persona_chat_db.get_messages_for_conversation(cid) == []


@pytest.mark.integration
def test_chat_endpoint_routes_auto_before_provider_normalization(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    execution_scoped_provider_credentials,
):
    request_data = ChatCompletionRequest(
        model="auto",
        api_provider="openrouter",
        messages=[ChatCompletionUserMessageParam(role="user", content="Summarize this")],
    )
    captured: dict[str, object] = {}

    async def fake_execute_non_stream_call(**kwargs):
        captured["selected_provider"] = kwargs.get("selected_provider")
        captured["model"] = kwargs.get("model")
        return {
            "id": "chatcmpl-auto-routing",
            "object": "chat.completion",
            "model": kwargs.get("model"),
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            return_value=RoutingDecision(
                provider="openrouter",
                model="anthropic/claude-4.5-sonnet",
                canonical=True,
                decision_source="rules_router",
            ),
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openrouter",
                        "models_info": [
                            {
                                "name": "anthropic/claude-4.5-sonnet",
                                "tool_support": True,
                                "vision_support": True,
                                "quality_rank": 20,
                            }
                        ],
                    }
                ],
                "default_provider": "openrouter",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert captured["selected_provider"] == "openrouter"
    assert captured["model"] == "anthropic/claude-4.5-sonnet"
    assert response.json()["model"] == "anthropic/claude-4.5-sonnet"


@pytest.mark.integration
def test_chat_endpoint_returns_503_when_auto_router_has_candidates_but_no_decision(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    request_data = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Route this")],
    )

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            return_value=None,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openai",
                        "models_info": [
                            {
                                "name": "gpt-test",
                                "tool_support": True,
                                "quality_rank": 1,
                            }
                        ],
                    }
                ],
                "default_provider": "openai",
            },
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "auto_routing_failed"
    assert response.json()["detail"]["routing"]["candidate_count"] == 1


@pytest.mark.integration
def test_chat_endpoint_returns_400_when_auto_router_has_no_candidates(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    request_data = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Route this")],
    )

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            return_value=None,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={"providers": [], "default_provider": "openai"},
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"]["error_code"] == "auto_routing_no_candidates"
    assert response.json()["detail"]["routing"]["candidate_count"] == 0


@pytest.mark.integration
def test_chat_endpoint_auto_routing_runs_llm_router_logs_usage_and_wires_sticky_mode(
    persona_chat_client,
    persona_chat_db,
    execution_scoped_provider_credentials,
):
    """Sticky routing uses a real accessible conversation before credential resolution."""
    authenticated_client, headers, _ = persona_chat_client
    persona_chat_db.add_conversation({"id": "conv-router", "client_id": "1", "title": "Sticky routing"})
    injected_store = InMemoryRoutingDecisionStore()
    authenticated_client.app.state.routing_decision_store = injected_store
    request_data = ChatCompletionRequest(
        model="auto",
        api_provider="openrouter",
        conversation_id="conv-router",
        routing={"mode": "sticky_session"},
        messages=[ChatCompletionUserMessageParam(role="user", content="Summarize this")],
    )
    captured: dict[str, object] = {}

    async def fake_router_call(**kwargs):
        captured["router_call"] = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"provider":"openrouter","model":"anthropic/claude-4.5-sonnet"}'
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        }

    async def fake_router_usage(**kwargs):
        captured.setdefault("router_usage", []).append(kwargs)

    def fake_route_model(**kwargs):
        captured["route_model_kwargs"] = kwargs
        decision = RoutingDecision(
            provider="openrouter",
            model="anthropic/claude-4.5-sonnet",
            canonical=True,
            decision_source="llm_router",
        )
        sticky_store = kwargs.get("sticky_store")
        request = kwargs.get("request")
        if sticky_store is not None and request is not None:
            sticky_store.save(
                scope=request.scope,
                fingerprint="persisted-for-test",
                provider=decision.provider,
                model=decision.model,
            )
        return decision

    async def fake_execute_non_stream_call(**kwargs):
        captured["selected_provider"] = kwargs.get("selected_provider")
        captured["model"] = kwargs.get("model")
        return {
            "id": "chatcmpl-auto-routing",
            "object": "chat.completion",
            "model": kwargs.get("model"),
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openrouter",
                        "default_model": "anthropic/claude-4.5-sonnet",
                        "models_info": [
                            {
                                "name": "anthropic/claude-4.5-sonnet",
                                "tool_support": True,
                                "vision_support": True,
                                "quality_rank": 20,
                            },
                            {
                                "name": "openai/gpt-4.1-mini",
                                "tool_support": True,
                                "vision_support": True,
                                "quality_rank": 10,
                            },
                        ],
                    }
                ],
                "default_provider": "openrouter",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call_async",
            side_effect=fake_router_call,
            create=True,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.log_model_router_usage",
            side_effect=fake_router_usage,
            create=True,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            side_effect=fake_route_model,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", headers=headers, json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert captured["selected_provider"] == "openrouter"
    assert captured["model"] == "anthropic/claude-4.5-sonnet"
    assert captured["route_model_kwargs"]["sticky_store"] is injected_store
    assert captured["route_model_kwargs"]["llm_router_choice"] == {
        "provider": "openrouter",
        "model": "anthropic/claude-4.5-sonnet",
    }
    assert captured["router_call"]["model"] == "anthropic/claude-4.5-sonnet"
    assert captured["router_usage"][0]["provider"] == "openrouter"


@pytest.mark.integration
@pytest.mark.parametrize("scope", ["correct", "omitted", "wrong_workspace", "deleted", "closing", "transferred", "archived"])
def test_workspace_persona_generation_requires_current_scope_access(
    persona_chat_client, persona_chat_db, monkeypatch, scope,
):
    """Ordinary generation supports Workspace Persona only with current parent access."""
    client, headers, provider_call = persona_chat_client
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    cid, _ = _create_persona_conversation(persona_chat_db, persona_id="workspace-persona")
    persona_chat_db.upsert_workspace("ws", name="Private Workspace")
    with persona_chat_db.transaction() as conn:
        conn.execute("UPDATE conversations SET scope_type = ?, workspace_id = ? WHERE id = ?", ("workspace", "ws", cid))
        if scope == "deleted":
            conn.execute("UPDATE workspaces SET deleted = ? WHERE id = ?", (True, "ws"))
        elif scope == "closing":
            conn.execute("UPDATE workspaces SET system_operation_state = ? WHERE id = ?", ("staged", "ws"))
        elif scope == "transferred":
            conn.execute("UPDATE workspaces SET client_id = ? WHERE id = ?", ("other", "ws"))
        elif scope == "archived":
            conn.execute("UPDATE workspaces SET archived = ? WHERE id = ?", (True, "ws"))
    query = {} if scope == "omitted" else {"scope_type": "workspace", "workspace_id": "other" if scope == "wrong_workspace" else "ws"}
    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", wraps=chat_endpoint.ProviderCredentialRuntime) as credentials,
        patch.object(chat_endpoint, "_resolve_auto_chat_routing_decision", wraps=chat_endpoint._resolve_auto_chat_routing_decision) as router,
        patch.object(persona_chat_db, "add_message", wraps=persona_chat_db.add_message) as message,
    ):
        response = client.post("/api/v1/chat/completions", headers=headers, params=query, json={
            "model": "gpt-4", "api_provider": "openai", "conversation_id": cid, "save_to_db": True,
            "messages": [{"role": "user", "content": "Private Workspace turn"}],
        })
    if scope in {"correct", "archived"}:
        assert response.status_code == 200, response.text
        assert provider_call.call_count == 1
        assert provider_call.call_args.kwargs["system_message"] == "You are Garden Helper."
    else:
        assert response.status_code == 404, response.text
        assert credentials.call_count == router.call_count == message.call_count == provider_call.call_count == 0
        assert persona_chat_db.get_messages_for_conversation(cid) == []


@pytest.mark.integration
def test_service_rechecks_persona_revoked_after_endpoint_admission(
    persona_chat_client, persona_chat_db, monkeypatch,
):
    """Endpoint approval is not cached across later ordinary context assembly."""
    client, headers, provider_call = persona_chat_client
    monkeypatch.setattr(feature_flags, "is_persona_enabled", lambda: True)
    cid, _ = _create_persona_conversation(persona_chat_db, persona_id="revoked-persona")
    original = chat_endpoint.require_current_persona

    def admit_then_revoke(*args, **kwargs):
        """Revoke the real profile after the point-in-time endpoint check."""
        profile = original(*args, **kwargs)
        with persona_chat_db.transaction() as conn:
            conn.execute("UPDATE persona_profiles SET is_active = ? WHERE id = ?", (False, "revoked-persona"))
        return profile

    monkeypatch.setattr(chat_endpoint, "require_current_persona", admit_then_revoke)
    with patch.object(persona_chat_db, "add_message", wraps=persona_chat_db.add_message) as message:
        response = client.post("/api/v1/chat/completions", headers=headers, json={
            "model": "gpt-4", "api_provider": "openai", "conversation_id": cid, "save_to_db": True,
            "messages": [{"role": "user", "content": "Private revoked turn"}],
        })
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "persona_unavailable"
    assert message.call_count == provider_call.call_count == 0
    assert persona_chat_db.get_messages_for_conversation(cid) == []


@pytest.mark.integration
def test_chat_endpoint_disables_provider_fallback_for_pinned_provider_auto_routing(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    execution_scoped_provider_credentials,
):
    request_data = ChatCompletionRequest(
        model="auto",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Route inside OpenAI")],
    )
    captured: dict[str, object] = {}

    async def fake_execute_non_stream_call(**kwargs):
        captured["enable_provider_fallback"] = kwargs.get("enable_provider_fallback")
        return {
            "id": "chatcmpl-auto-routing",
            "object": "chat.completion",
            "model": kwargs.get("model"),
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            return_value=RoutingDecision(
                provider="openai",
                model="gpt-4.1-mini",
                canonical=True,
                decision_source="rules_router",
            ),
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openai",
                        "default_model": "gpt-4.1-mini",
                        "models_info": [
                            {
                                "name": "gpt-4.1-mini",
                                "tool_support": True,
                                "quality_rank": 10,
                            }
                        ],
                    }
                ],
                "default_provider": "openai",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert captured["enable_provider_fallback"] is False


@pytest.mark.integration
def test_chat_endpoint_auto_routing_uses_post_validation_tool_capabilities(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    execution_scoped_provider_credentials,
):
    request_data = ChatCompletionRequest(
        model="auto",
        messages=[ChatCompletionUserMessageParam(role="user", content="Route after tool injection")],
    )
    captured: dict[str, object] = {}

    def fake_route_model(**kwargs):
        captured["requested_capabilities"] = kwargs["request"].requested_capabilities
        return RoutingDecision(
            provider="openai",
            model="gpt-4.1-mini",
            canonical=True,
            decision_source="rules_router",
        )

    async def fake_execute_non_stream_call(**kwargs):
        return {
            "id": "chatcmpl-auto-routing",
            "object": "chat.completion",
            "model": kwargs.get("model"),
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }

    async def fake_add_skill_tools(tools, **_kwargs):
        return [
            {
                "type": "function",
                "function": {"name": "skills.lookup", "parameters": {"type": "object"}},
            }
        ]

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.route_model",
            side_effect=fake_route_model,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "openai",
                        "default_model": "gpt-4.1-mini",
                        "models_info": [
                            {
                                "name": "gpt-4.1-mini",
                                "tool_support": True,
                                "quality_rank": 10,
                            }
                        ],
                    }
                ],
                "default_provider": "openai",
            },
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.add_skill_tool_to_tools_list_async",
            side_effect=fake_add_skill_tools,
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.execute_non_stream_call",
            side_effect=fake_execute_non_stream_call,
        ),
    ):
        response = authenticated_client.post("/api/v1/chat/completions", json=request_data.model_dump())

    assert response.status_code == status.HTTP_200_OK
    assert captured["requested_capabilities"]["tools"] is True
