"""Integration coverage for auto-model routing in character chat completions."""

from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.LLM_Calls.routing.models import RoutingDecision


@pytest.mark.integration
def test_complete_v2_routes_auto_model_before_provider_call(test_client, auth_headers):
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
        json={"character_id": char_id, "title": "Auto route integration"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    captured: dict[str, object] = {}

    def _stub_chat_api_call(
        api_endpoint: str, messages_payload: object, **kwargs: object
    ) -> dict[str, list[dict[str, dict[str, str]]]]:
        captured["provider"] = api_endpoint
        captured["model"] = kwargs.get("model")
        return {"choices": [{"message": {"content": "auto routed response"}}]}

    with (
        patch(
            "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.route_model",
            return_value=RoutingDecision(
                provider="local-llm",
                model="local-test-routed",
                canonical=True,
                decision_source="rules_router",
            ),
        ),
        patch(
            "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.get_configured_providers",
            return_value={
                "providers": [
                    {
                        "name": "local-llm",
                        "models_info": [
                            {
                                "name": "local-test-routed",
                                "tool_support": True,
                                "vision_support": False,
                                "quality_rank": 10,
                            }
                        ],
                        "is_configured": True,
                    }
                ],
                "default_provider": "local-llm",
            },
        ),
        patch(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call",
            side_effect=_stub_chat_api_call,
        ),
        patch(
            "tldw_Server_API.app.core.Utils.common.parse_boolean",
            return_value=True,
        ),
    ):
        response = test_client.post(
            f"/api/v1/chats/{chat_id}/complete-v2",
            json={
                "model": "auto",
                "append_user_message": "Route this automatically",
                "stream": False,
                "include_character_context": False,
                "save_to_db": False,
            },
            headers=auth_headers,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "local-llm"
    assert payload["model"] == "local-test-routed"
    assert payload["assistant_content"] == "auto routed response"
    assert captured == {
        "provider": "local-llm",
        "model": "local-test-routed",
    }
