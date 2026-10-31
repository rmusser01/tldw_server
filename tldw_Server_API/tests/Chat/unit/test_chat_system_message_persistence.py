from unittest.mock import patch

from fastapi import status

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)

pytest_plugins = (
    "tldw_Server_API.tests.Chat.credential_runtime_fixtures",
)


def test_chat_completion_persists_system_message(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
    execution_scoped_provider_credentials,
):
    """Ensure system prompt is stored with an ID and returned in message history."""
    _ = setup_dependencies  # Silence ARG001 by explicitly touching the fixture.
    mock_chacha_db.client_id = "1"
    request_data = ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[
            ChatCompletionUserMessageParam(role="user", content="Hello system prompt")
        ],
        save_to_db=True,
        # Endpoint auto-generates the system message via apply_prompt_templating().
    )

    with patch(
        "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call"
    ) as mock_llm:
        mock_llm.return_value = {
            "id": "chatcmpl-test",
            "choices": [{
                "message": {"role": "assistant", "content": "Acknowledged"},
                "finish_reason": "stop"
            }]
        }

        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump()
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        conv_id = data.get("tldw_conversation_id")
        system_id = data.get("tldw_system_message_id")
        assert conv_id
        assert system_id

    messages_response = authenticated_client.get(
        f"/api/v1/chats/{conv_id}/messages"
        "?limit=50&offset=0&format_for_completions=true&include_character_context=true&include_message_ids=true"
    )
    assert messages_response.status_code == status.HTTP_200_OK
    messages_payload = messages_response.json()
    messages = messages_payload.get("messages") or []
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 1
    assert system_messages[0].get("message_id") == system_id
