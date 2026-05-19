"""
Unit test to ensure completion pre-check uses efficient count instead of bulk-loading messages.
"""

import pytest
from typing import List, Dict, Any

from tldw_Server_API.app.core.LLM_Calls.routing.decision_store import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


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

    calls: Dict[str, Any] = {"count_calls": 0, "get_limits": []}

    def count_wrapper(conversation_id: str) -> int:
        calls["count_calls"] += 1
        return original_count(conversation_id)

    def get_wrapper(conversation_id: str, limit: int = 100, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
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

    captured: Dict[str, Any] = {}

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
