"""
Isolated test configuration that avoids global state modifications.
This replaces the problematic autouse fixtures with explicit, isolated fixtures.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from typing import Dict, Any


def _get_app():
    from tldw_Server_API.app.main import app as _app

    return _app


from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core import config_llm_providers
from tldw_Server_API.tests.helpers.provider_keys import api_keys_source


# ============================================================================
# ISOLATED FIXTURES - No global state modifications
# ============================================================================


@pytest.fixture(scope="function")
def isolated_db():
    """Create an isolated database for each test."""
    import sqlite3

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    # Initialize database with WAL mode for better concurrency
    db = CharactersRAGDB(db_path, f"test_client_{id(tmp)}")
    conn = db.get_connection()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()

    # Add default character
    db.add_character_card(
        {
            "name": DEFAULT_CHARACTER_NAME,
            "description": "Test character",
            "personality": "Helpful",
            "scenario": "Testing",
            "system_prompt": "You are a helpful AI assistant.",
            "first_message": "Hello!",
            "creator_notes": "Test",
        }
    )

    yield db

    # Cleanup
    try:
        os.unlink(db_path)
        if os.path.exists(db_path + "-wal"):
            os.unlink(db_path + "-wal")
        if os.path.exists(db_path + "-shm"):
            os.unlink(db_path + "-shm")
    except:
        _ = None


@pytest.fixture(scope="function")
def isolated_client(isolated_db):
    """Create an isolated test client with its own database."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    # Create a new TestClient instance with isolated overrides
    test_app = _get_app()
    original_overrides = test_app.dependency_overrides.copy()

    # Override database dependency
    test_app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db

    client = TestClient(test_app)

    # Get CSRF token
    response = client.get("/api/v1/health")
    csrf_token = response.cookies.get("csrf_token", "")
    client.csrf_token = csrf_token
    client.cookies = {"csrf_token": csrf_token}

    yield client

    # Restore original overrides
    test_app.dependency_overrides = original_overrides


@pytest.fixture(scope="function")
def mock_api_keys():
    """Return mock API keys without modifying global state."""
    return {"openai": "sk-mock-key-12345", "local-llm": "dummy-key", "anthropic": "mock-anthropic-key"}


@pytest.fixture(scope="function")
def isolated_auth_token():
    """Generate an isolated auth token for testing."""
    settings = get_settings()
    if settings.AUTH_MODE == "multi_user":
        # In multi-user mode, create a JWT token
        from tldw_Server_API.app.core.Security.JWT import get_jwt_service

        jwt_service = get_jwt_service()
        access_token = jwt_service.create_access_token(user_id="test_user_id", username="test_user", role="user")
        return f"Bearer {access_token}"
    else:
        # In single-user mode, use the actual API key from settings
        # The endpoint expects Bearer prefix for consistency
        api_key = settings.SINGLE_USER_API_KEY or "test-api-key-12345"
        return f"Bearer {api_key}"


@pytest.fixture(scope="function")
def mock_llm_response():
    """Provide a standard mock LLM response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture(scope="function")
def isolated_chat_endpoint_mocks(mock_api_keys, mock_llm_response):
    """Create isolated mocks for chat endpoint without global modifications."""
    with (
        patch.object(config_llm_providers, "get_api_keys", api_keys_source(mock_api_keys)),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_perform,
        patch("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call") as mock_chat_call,
    ):

        mock_perform.return_value = mock_llm_response
        mock_chat_call.return_value = mock_llm_response

        yield {"perform_chat_api_call": mock_perform, "chat_api_call": mock_chat_call, "response": mock_llm_response}


# ============================================================================
# UNIT TEST FIXTURES - For tests that don't need external services
# ============================================================================


@pytest.fixture(scope="function")
def unit_test_client(isolated_db, isolated_chat_endpoint_mocks):
    """Client for unit tests with all external dependencies mocked."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    test_app = _get_app()
    original_overrides = test_app.dependency_overrides.copy()

    # Override database
    test_app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db

    client = TestClient(test_app)

    # Setup CSRF
    response = client.get("/api/v1/health")
    csrf_token = response.cookies.get("csrf_token", "")
    client.csrf_token = csrf_token

    # Add helper method for authenticated requests
    def post_with_auth(url, json_data, auth_token="Bearer sk-mock-key-12345"):
        # In single-user mode, the API expects X-API-KEY to match settings.
        # Use Authorization header for bearer/API-key compatibility in tests.
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

        _settings = _get_settings()
        headers = {"X-CSRF-Token": csrf_token}
        # Always include X-API-KEY for single-user auth
        try:
            api_key = _settings.SINGLE_USER_API_KEY or ""
            if api_key:
                headers["X-API-KEY"] = api_key
        except Exception:
            _ = None
        headers["Authorization"] = auth_token
        return client.post(url, json=json_data, headers=headers)

    client.post_with_auth = post_with_auth

    yield client

    test_app.dependency_overrides = original_overrides


# ============================================================================
# INTEGRATION TEST FIXTURES - For tests that need the mock server
# ============================================================================


@pytest.fixture(scope="session")
def mock_server_url():
    """Return the mock server URL if it's running."""
    import requests

    try:
        response = requests.get("http://localhost:8080/v1/models", timeout=1)
        if response.status_code == 200:
            return "http://localhost:8080"
    except:
        _ = None

    # If mock server is not running, skip integration tests
    pytest.skip("Mock OpenAI server not running - skipping integration tests")


@pytest.fixture(scope="function")
def integration_test_client(isolated_db, mock_server_url):
    """Client for integration tests that need the mock server."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    test_app = app
    original_overrides = test_app.dependency_overrides.copy()

    # Override database
    test_app.dependency_overrides[get_chacha_db_for_user] = lambda: isolated_db

    # Set environment for this test only using monkeypatch would be better
    # but for now we'll use a context manager approach
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-mock-key-12345",
            "OPENAI_API_BASE": f"{mock_server_url}/v1",
            "CUSTOM_OPENAI_API_IP": f"{mock_server_url}/v1/chat/completions",
        },
    ):
        # Temporarily give openai a key for this test
        with patch.object(config_llm_providers, "get_api_keys", api_keys_source({"openai": "sk-mock-key-12345"})):
            client = TestClient(test_app)

            # Setup CSRF
            response = client.get("/api/v1/health")
            csrf_token = response.cookies.get("csrf_token", "")
            client.csrf_token = csrf_token

            # Add helper for authenticated requests
            def post_with_auth(url, json_data, auth_token="Bearer sk-mock-key-12345"):
                from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

                _settings = _get_settings()
                headers = {"X-CSRF-Token": csrf_token}
                try:
                    api_key = _settings.SINGLE_USER_API_KEY or ""
                    if api_key:
                        headers["X-API-KEY"] = api_key
                except Exception:
                    _ = None
                headers["Authorization"] = auth_token
                return client.post(url, json=json_data, headers=headers)

            client.post_with_auth = post_with_auth

            yield client

    test_app.dependency_overrides = original_overrides


# ============================================================================
# HELPER FIXTURES
# ============================================================================


@pytest.fixture
def sample_chat_request():
    """Provide a sample chat request for testing."""
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
        ChatCompletionRequest,
        ChatCompletionUserMessageParam,
    )

    return ChatCompletionRequest(
        model="test-model",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello, how are you?")],
    )


@pytest.fixture
def mock_character():
    """Provide a mock character for testing."""
    return {
        "id": 1,
        "name": "TestCharacter",
        "description": "A test character",
        "personality": "Helpful and friendly",
        "scenario": "Testing",
        "system_prompt": "You are a test assistant",
        "first_message": "Hello! I'm TestCharacter.",
        "creator_notes": "Created for testing",
    }
