"""
Unit tests for chat functionality using isolated fixtures.
These tests don't require external services or global state modifications.
"""

import os
import pytest
from fastapi import status
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME

# Fixtures are automatically discovered from conftest.py


class TestChatUnit:
    """Unit tests for chat endpoint functionality."""

    def test_chat_completion_basic(self, unit_test_client, sample_chat_request):
        """Test basic chat completion with mocked LLM."""
        response = unit_test_client.post_with_auth(
            "/api/v1/chat/completions", json_data=sample_chat_request.model_dump()
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] == "This is a test response"

    def test_chat_with_character(self, unit_test_client, isolated_db):
        """Test chat with specific character."""
        # Add a test character
        char_id = isolated_db.add_character_card(
            {
                "name": "TestBot",
                "description": "A test bot",
                "personality": "Friendly",
                "scenario": "Testing",
                "system_prompt": "You are TestBot",
                "first_message": "Hello from TestBot!",
                "creator_notes": "Test",
            }
        )

        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "character_id": str(char_id),
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        if response.status_code != status.HTTP_200_OK:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.json()}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

    def test_conversation_persistence(self, unit_test_client, isolated_db):
        """Test that conversations are persisted correctly."""
        # Don't create a conversation upfront - let the endpoint create it
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "save_to_db": True,
            # No conversation_id - let endpoint create one
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_200_OK

        default_character = isolated_db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        assert default_character is not None, "Default character not found in test database"

        conversations = isolated_db.get_conversations_for_character(default_character["id"])
        assert len(conversations) > 0, "No conversations found after chat completion"

        # Get the latest conversation
        latest_conv = conversations[-1]
        conv_id = latest_conv.get("id")

        # Check that messages were saved
        messages = isolated_db.get_messages_for_conversation(conv_id)
        assert len(messages) > 0, f"No messages found in conversation {conv_id}"

    def test_invalid_api_provider(self, unit_test_client):
        """Test handling of missing API key for provider."""
        request_data = {
            "model": "test-model",
            "api_provider": "nonexistent_provider",  # Provider that doesn't exist
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Remove the provider from mock keys to simulate missing key
        with patch.dict(
            "tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "sk-mock-key-12345"}, clear=True
        ):
            response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code >= status.HTTP_400_BAD_REQUEST
        payload = response.json()
        assert isinstance(payload, dict)

    @pytest.mark.skip(reason="Streaming tests hang with TestClient")
    def test_streaming_request(self, unit_test_client):
        """Test that streaming requests are accepted (though not actually streamed in tests)."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        # Even with stream=True, TestClient returns regular response
        assert response.status_code == status.HTTP_200_OK

    def test_system_message_handling(self, unit_test_client):
        """Test that system messages are handled correctly."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "choices" in data

    def test_temperature_parameter(self, unit_test_client):
        """Test that temperature parameter is accepted."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_max_tokens_parameter(self, unit_test_client):
        """Test that max_tokens parameter is accepted."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_missing_messages(self, unit_test_client):
        """Test that missing messages field returns error."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            # messages field missing
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_empty_messages(self, unit_test_client):
        """Test that empty messages array returns error."""
        request_data = {"model": "test-model", "api_provider": "openai", "messages": []}

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code >= 400

    @pytest.mark.skip(reason="Authentication works differently in single-user mode")
    def test_authentication_required(self, isolated_db, isolated_chat_endpoint_mocks):
        """Test that authentication is required."""
        from fastapi.testclient import TestClient
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

        # Create a fresh client without auth overrides
        test_app = app
        original_overrides = test_app.dependency_overrides.copy()
        test_app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db

        client = TestClient(test_app)

        # Get CSRF token
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")

        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Make request without auth token
        response = client.post("/api/v1/chat/completions", json=request_data, headers={"X-CSRF-Token": csrf_token})

        # In single-user mode, should require Authorization/X-API-KEY header
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Restore overrides
        test_app.dependency_overrides = original_overrides

    @pytest.mark.skip(reason="Authentication works differently in single-user mode")
    def test_invalid_auth_token(self, isolated_db, isolated_chat_endpoint_mocks):
        """Test that invalid auth token is rejected."""
        from fastapi.testclient import TestClient
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

        # Create a fresh client without auth overrides
        test_app = app
        original_overrides = test_app.dependency_overrides.copy()
        test_app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db

        client = TestClient(test_app)

        # Get CSRF token
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")

        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Make request with invalid auth token
        response = client.post(
            "/api/v1/chat/completions",
            json=request_data,
            headers={"X-CSRF-Token": csrf_token, "Authorization": "Bearer invalid-token-xyz"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Restore overrides
        test_app.dependency_overrides = original_overrides


class TestChatErrorHandling:
    """Test error handling in chat endpoint."""

    def test_llm_api_error(self, isolated_db):
        """Test handling of LLM API errors."""
        from fastapi.testclient import TestClient
        from tldw_Server_API.app.main import app
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

        # Create test user
        test_user = User(id=1, username="test_user", email="test@example.com", is_active=True)

        async def mock_get_request_user(api_key=None, token=None):
            return test_user

        # Create client with auth but without LLM mocks
        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db
        app.dependency_overrides[get_request_user] = mock_get_request_user

        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Mock the LLM call to raise an exception and API keys
        try:
            with TestClient(app) as client:
                response = client.get("/api/v1/health")
                csrf_token = response.cookies.get("csrf_token", "")

                with (
                    patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_perform,
                    patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test-key"}),
                ):
                    mock_perform.side_effect = Exception("LLM API Error")

                    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

                    settings = get_settings()
                    api_key = settings.SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")

                    response = client.post(
                        "/api/v1/chat/completions",
                        json=request_data,
                        headers={"X-CSRF-Token": csrf_token, "X-API-KEY": api_key},
                    )

                    assert response.status_code >= 500
                    data = response.json()
                    assert "detail" in data
        finally:
            app.dependency_overrides = original_overrides

    def test_database_error(self, unit_test_client, isolated_db):
        """Test handling of invalid conversation ID."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_id": "invalid-conv-id",
        }

        # The endpoint should handle invalid conversation ID gracefully
        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        # Should handle the invalid ID gracefully - creates new conversation or returns success
        # When conversation ID is invalid, the endpoint creates a new conversation
        assert response.status_code == 200  # Should succeed with new conversation

    def test_invalid_message_format(self, unit_test_client):
        """Test handling of invalid message format."""
        request_data = {
            "model": "test-model",
            "api_provider": "openai",
            "messages": [{"role": "invalid-role", "content": "Hello"}],  # Invalid role
        }

        response = unit_test_client.post_with_auth("/api/v1/chat/completions", json_data=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
