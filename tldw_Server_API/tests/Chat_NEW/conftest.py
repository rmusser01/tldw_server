"""
Chat Module Test Configuration and Fixtures

Provides fixtures for testing the chat functionality with proper separation
of unit, integration, and property tests. Focuses on the OpenAI-compatible
/chat/completions endpoint which is the primary production interface.
"""

import os
# Set test environment variables before any imports
os.environ["TEST_MODE"] = "true"
os.environ["DEFAULT_LLM_PROVIDER"] = "openai"
os.environ["API_BEARER"] = os.environ.get("API_BEARER", "test-api-key-12345")
os.environ["SINGLE_USER_API_KEY"] = os.environ.get("SINGLE_USER_API_KEY", "test-api-key-12345")
# Keep app imports lightweight in Chat_NEW tests; avoids torch/faster_whisper import paths.
os.environ.setdefault("MINIMAL_TEST_APP", "1")
_routes_disable = {
    part.strip()
    for part in str(os.environ.get("ROUTES_DISABLE", "")).split(",")
    if part and part.strip()
}
_routes_disable.add("media")
os.environ["ROUTES_DISABLE"] = ",".join(sorted(_routes_disable))
# Reduce background services during tests
os.environ.setdefault("DISABLE_AUTHNZ_SCHEDULER", "1")
os.environ.setdefault("WORKFLOWS_SCHEDULER_ENABLED", "false")
# Deterministic chat rate limits for integration tests
os.environ.setdefault("TEST_CHAT_PER_USER_RPM", "2")
os.environ.setdefault("TEST_CHAT_PER_CONVERSATION_RPM", "2")
os.environ.setdefault("TEST_CHAT_GLOBAL_RPM", "10")
os.environ.setdefault("TEST_CHAT_TOKENS_PER_MINUTE", "1000")

# Load config to get API keys
from tldw_Server_API.app.core.config import load_and_log_configs
_test_config = load_and_log_configs()
if _test_config and 'openai_api' in _test_config:
    _openai_key = _test_config['openai_api'].get('api_key')
    if _openai_key:
        os.environ["OPENAI_API_KEY"] = _openai_key

# Add dummy API keys for other providers used in tests
# Note: Do NOT set ANTHROPIC_API_KEY unless a real key is present, to allow skips
os.environ["GROQ_API_KEY"] = "test-groq-key-for-testing"
os.environ["MISTRAL_API_KEY"] = "test-mistral-key-for-testing"

from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock
from datetime import datetime
import uuid

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import actual components for integration tests
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

# Relax Character-Chat rate limits for this package to avoid flakiness when tests
# touch persona chat endpoints alongside Chat module coverage.
@pytest.fixture(autouse=True)
def _override_character_chat_rate_limits_for_chat_new(monkeypatch):
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_OPS", "1000000")
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("MAX_CHARACTERS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_CHATS_PER_USER", "1000000")
    monkeypatch.setenv("MAX_MESSAGES_PER_CHAT", "1000000")
    monkeypatch.setenv("MAX_CHAT_COMPLETIONS_PER_MINUTE", "1000000")
    monkeypatch.setenv("MAX_MESSAGE_SENDS_PER_MINUTE", "1000000")

    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl
        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    yield
    try:
        from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as _crl
        _crl._rate_limiter = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

# Note: FastAPI TestClient already triggers application lifespan shutdown
# which calls shutdown_all_audit_services() and performs DB cleanup.
# Adding an extra session-scope shutdown here risks duplicate teardown
# and can contend with SQLite locks. Intentionally omitted.

# =====================================================================
# Default LLM call mocking
# =====================================================================

@pytest.fixture(autouse=True)
def _mock_perform_chat_api_call(monkeypatch):
    """
    Provide a deterministic fake LLM response so integration tests do not rely on
    external OpenAI connectivity. Tests that need custom behaviour can override
    this by patching `perform_chat_api_call` in the test body.
    """
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint

    def _stream_generator():

        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
            'data: [DONE]\n\n',
        ]
        for chunk in chunks:
            yield chunk

    def _fake_chat_call(api_endpoint: str, messages_payload, *, streaming: bool | None = None, **kwargs):
        stream_flag = streaming or kwargs.get("stream")

        if stream_flag:
            return _stream_generator()

        return {
            "id": "chatcmpl-mocked",
            "object": "chat.completion",
            "created": 1,
            "model": kwargs.get("model") or "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mocked response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", _fake_chat_call)

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):

    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_llm: Tests requiring LLM API")
    config.addinivalue_line("markers", "streaming: Tests for streaming responses")

# =====================================================================
# Rate Limiter Reset Between Tests
# =====================================================================

@pytest.fixture(autouse=True)
def _reset_chat_rate_limiter_between_tests(monkeypatch):
    """Reset chat rate limiter state before each test to avoid cross-test 429s.

    Ensures deterministic behavior for tests that expect 200 responses by
    restoring token buckets to full capacity for the default test user.
    """
    try:
        # Force deterministic TEST_MODE rate limits for this test module.
        monkeypatch.setenv("TEST_MODE", "true")
        monkeypatch.setenv("RG_ENABLED", "0")
        monkeypatch.setenv("TEST_CHAT_PER_USER_RPM", "2")
        monkeypatch.setenv("TEST_CHAT_PER_CONVERSATION_RPM", "2")
        monkeypatch.setenv("TEST_CHAT_GLOBAL_RPM", "10")
        monkeypatch.setenv("TEST_CHAT_TOKENS_PER_MINUTE", "1000")
        monkeypatch.delenv("TEST_CHAT_BURST_MULTIPLIER", raising=False)
        from tldw_Server_API.app.core.Chat.rate_limiter import (
            initialize_rate_limiter,
        )
        # Reinitialize each test to ensure TEST_CHAT_* env overrides apply.
        rl = initialize_rate_limiter()
        # Reset per-user and global buckets (both common test ids)
        rl.reset_user_limits("test_user")
        rl.reset_user_limits("1")  # single_user mode default user id
        if hasattr(rl, "global_bucket") and hasattr(rl.global_bucket, "capacity"):
            rl.global_bucket.tokens = rl.global_bucket.capacity
    except Exception:
        # Best effort; tests that validate 429s will still function
        _ = None
    yield

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(scope="session")
def test_env_vars(tmp_path_factory):
    """Set deterministic env vars for Chat_NEW tests and isolate per-session user DBs."""
    original_user_db_base = os.environ.get("USER_DB_BASE_DIR")
    isolated_user_db_base = tmp_path_factory.mktemp("chat_new_user_databases")
    os.environ["USER_DB_BASE_DIR"] = str(isolated_user_db_base)
    try:
        yield
    finally:
        if original_user_db_base is None:
            os.environ.pop("USER_DB_BASE_DIR", None)
        else:
            os.environ["USER_DB_BASE_DIR"] = original_user_db_base

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path owned by pytest's temp cleanup."""
    return tmp_path / "test_chacha.db"

@pytest.fixture
def chacha_db(temp_db_path) -> Generator[CharactersRAGDB, None, None]:
    """Create a real CharactersRAGDB instance for testing."""
    db = CharactersRAGDB(
        db_path=str(temp_db_path),
        client_id="test_user"
    )
    # Database is initialized in __init__, no need to call initialize_db
    try:
        yield db
    finally:
        db.close_all_connections()

@pytest.fixture
def populated_chacha_db(chacha_db) -> CharactersRAGDB:
    """Create a CharactersRAGDB with test data."""
    # First, add a character card
    # Create the default character that the system expects
    character_data = {
        'name': DEFAULT_CHARACTER_NAME,
        'description': 'A helpful assistant',
        'personality': 'Helpful and friendly',
        'system_prompt': 'You are a helpful AI assistant.',
        'client_id': 'test_user'
    }
    character_id = chacha_db.add_character_card(character_data)

    # Add test conversations
    conversation_data = {
        'title': "Test Conversation",
        'character_id': character_id
    }
    conversation_id = chacha_db.add_conversation(conversation_data)

    # Add test messages
    chacha_db.add_message({
        'conversation_id': conversation_id,
        'sender': "user",
        'content': "Hello, how are you?"
    })

    chacha_db.add_message({
        'conversation_id': conversation_id,
        'sender': "assistant",
        'content': "I'm doing well, thank you! How can I help you today?"
    })

    return chacha_db

# =====================================================================
# Mock Fixtures for Unit Tests
# =====================================================================

@pytest.fixture
def mock_chacha_db():
    """Mock CharactersRAGDB for unit tests."""
    mock_db = MagicMock(spec=CharactersRAGDB)

    # Setup default return values
    mock_db.add_conversation.return_value = "test-conversation-id"
    mock_db.add_message.return_value = "test-message-id"
    mock_db.get_conversation.return_value = {
        "id": "test-conversation-id",
        "title": "Test Conversation",
        "character_id": 1
    }
    mock_db.get_messages.return_value = []
    mock_db.add_character_card.return_value = 1

    return mock_db

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for unit tests."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from the LLM."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

@pytest.fixture
def mock_streaming_response():
    """Mock streaming LLM response for unit tests."""
    async def stream_generator():
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        for chunk in chunks:
            yield chunk

    return stream_generator()

# =====================================================================
# Request Fixtures
# =====================================================================

@pytest.fixture
def basic_chat_request() -> Dict[str, Any]:
    """Basic chat completion request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }

@pytest.fixture
def multi_turn_chat_request() -> Dict[str, Any]:
    """Multi-turn conversation request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

@pytest.fixture
def streaming_chat_request() -> Dict[str, Any]:
    """Streaming chat completion request."""
    return {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Tell me a short story."}
        ],
        "stream": True
    }

@pytest.fixture
def provider_specific_request() -> Dict[str, Any]:
    """Request with specific provider."""
    return {
        "api_provider": "anthropic",
        "model": "claude-sonnet-4.5",
        "messages": [
            {"role": "user", "content": "Explain quantum computing."}
        ],
        "max_tokens": 200
    }

# =====================================================================
# Message Fixtures
# =====================================================================

@pytest.fixture
def valid_messages() -> List[Dict[str, str]]:
    """Collection of valid message formats."""
    return [
        [{"role": "user", "content": "Simple message"}],
        [{"role": "system", "content": "You are helpful."},
         {"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Question?"},
         {"role": "assistant", "content": "Answer."},
         {"role": "user", "content": "Follow-up?"}]
    ]

@pytest.fixture
def invalid_messages() -> List[Any]:
    """Collection of invalid message formats."""
    return [
        [],  # Empty messages
        [{"role": "invalid", "content": "test"}],  # Invalid role
        [{"content": "missing role"}],  # Missing role
        [{"role": "user"}],  # Missing content
        "not a list",  # Wrong type
        [{"role": "user", "content": ""}]  # Empty content
    ]

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars):
    """Create a test client for the FastAPI app with cleanup."""
    from tldw_Server_API.app.main import app
    with TestClient(app) as client:
        yield client

@pytest_asyncio.fixture
async def async_client(test_env_vars):
    """Create an async test client for streaming tests."""
    from tldw_Server_API.app.main import app
    # Use httpx.AsyncClient with transport instead of app parameter
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    api_key = os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    return {
        "Authorization": f"Bearer {api_key}",
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

# =====================================================================
# Provider Configuration Fixtures
# =====================================================================

@pytest.fixture
def provider_configs():
    """Configuration for different LLM providers."""
    return {
        "openai": {
            "api_key": "test-openai-key",
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-3.5-turbo", "gpt-4"]
        },
        "anthropic": {
            "api_key": "test-anthropic-key",
            "base_url": "https://api.anthropic.com",
            "models": ["claude-sonnet-4.5", "claude-opus-4.1"]
        },
        "local": {
            "base_url": "http://localhost:8080",
            "models": ["llama-2-7b"]
        }
    }

# =====================================================================
# Error Response Fixtures
# =====================================================================

@pytest.fixture
def rate_limit_error():
    """Rate limit error response."""
    return {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": 429
        }
    }

@pytest.fixture
def auth_error():
    """Authentication error response."""
    return {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": 401
        }
    }

@pytest.fixture
def validation_error():
    """Validation error response."""
    return {
        "error": {
            "message": "Invalid request format",
            "type": "validation_error",
            "code": 400
        }
    }

# =====================================================================
# Test Data Generators
# =====================================================================

@pytest.fixture
def message_generator():
    """Factory for generating test messages."""
    def _generate(count: int = 5) -> List[Dict[str, str]]:
        messages = []
        roles = ["user", "assistant"]
        for i in range(count):
            role = roles[i % 2]
            messages.append({
                "role": role,
                "content": f"Test message {i} from {role}"
            })
        return messages
    return _generate

@pytest.fixture
def conversation_generator():
    """Factory for generating test conversations."""
    def _generate(num_turns: int = 3) -> Dict[str, Any]:
        messages = []
        for i in range(num_turns):
            messages.append({
                "role": "user",
                "content": f"User message {i}"
            })
            if i < num_turns - 1:  # Don't add assistant message for last turn
                messages.append({
                    "role": "assistant",
                    "content": f"Assistant response {i}"
                })

        return {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.7
        }
    return _generate

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Any cleanup code here
    import gc
    gc.collect()
