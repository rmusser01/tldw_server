"""
Simple chat endpoint test using the simplified fixtures.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import status

# Fixtures are imported automatically from conftest.py

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.app.core.security_utils import redact_secret
from tldw_Server_API.tests.helpers.provider_keys import api_keys_source


def test_chat_completion_works(client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server):
    """Test that chat completion works with proper auth."""

    # Set the app to debug mode for better error messages
    from tldw_Server_API.app.main import app as main_app

    original_debug = main_app.debug
    main_app.debug = True

    request_data = ChatCompletionRequest(
        model="local-llm",  # Use a local model that doesn't require API key
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
        api_provider="local-llm",  # Explicitly set the provider
    )

    # Mock the LLM call function and the provider key
    with (
        patch("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call") as mock_llm,
        patch.object(config_llm_providers, "get_api_keys", api_keys_source({"local-llm": "dummy-key"})),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_perform,
    ):
        # Set up the mock response
        mock_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "local-llm",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "I'm doing well!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        # Mock both functions to return the same response
        mock_llm.return_value = mock_response
        mock_perform.return_value = mock_response

        # Make request with authentication
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        settings = get_settings()
        print(f"AUTH_MODE: {settings.AUTH_MODE}")
        print(f"Expected API key is {'set' if settings.SINGLE_USER_API_KEY else 'not set'}")
        print(f"Auth token from fixture: {redact_secret(auth_token)}")

        # Debug: make the request manually to see what headers are sent
        headers = {"X-CSRF-Token": client.csrf_token}
        if settings.AUTH_MODE == "multi_user":
            headers["Authorization"] = auth_token
            print(f"Using Authorization header")
        else:
            # Use X-API-KEY header as expected by the endpoint in single-user mode
            headers["X-API-KEY"] = auth_token
            print(f"Using X-API-KEY header")

        redacted_headers = dict(headers)
        for key in ("Authorization", "X-API-KEY"):
            if key in redacted_headers:
                redacted_headers[key] = redact_secret(redacted_headers[key])
        print(f"Headers being sent: {redacted_headers}")

        try:
            response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)
        except Exception as e:
            print(f"Exception during request: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Check response
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
            from tldw_Server_API.app.main import app as main_app

            print(f"Main app overrides: {list(main_app.dependency_overrides.keys())}")

        assert (
            response.status_code == status.HTTP_200_OK
        ), f"Expected 200 but got {response.status_code}: {response.text}"
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "I'm doing well!"
