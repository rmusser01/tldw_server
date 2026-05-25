# tests/unit/api/v1/endpoints/test_chat_endpoint.py
import pytest
from unittest.mock import patch, MagicMock, ANY
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
import json
import os
import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
# Ensure keys are loaded from the canonical Config_Files/.env
try:
    _this = Path(__file__).resolve()
    _project_root = _this.parents[3]  # repo_root/tldw_Server_API
    _env_path = _project_root / "Config_Files" / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=False)
except Exception:
    _ = None

# Import your FastAPI app instance
from tldw_Server_API.app.main import app

# Import helpers from conftest
# get_auth_headers is defined locally below for clarity
# Import schemas from your actual file path
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageURL,
    ResponseFormat,
    ToolDefinition,
    FunctionDefinition,
    ToolChoiceOption,
    ToolChoiceFunction,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatRateLimitError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatAPIError,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate, DEFAULT_RAW_PASSTHROUGH_TEMPLATE
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user, DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


# Helper function to make requests with CSRF token
def make_request_with_csrf(client, method, url, headers=None, **kwargs):
    """Helper to make requests with CSRF token included"""
    if headers is None:
        headers = {}

    # Get CSRF token from cookies if not already set
    if not hasattr(client, "csrf_token"):
        # Make a GET request to get CSRF token
        response = client.get("/api/v1/health")
        csrf_token = response.cookies.get("csrf_token", "")
        client.csrf_token = csrf_token

    # Add CSRF token to headers (preserving any existing headers)
    headers = dict(headers)  # Make a copy to avoid modifying the original
    # In single-user mode, ensure X-API-KEY is present so AuthNZ deps pass
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        settings = get_settings()
        if settings.AUTH_MODE == "single_user" and "X-API-KEY" not in headers:
            api_key = settings.SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
            if api_key:
                headers["X-API-KEY"] = api_key
    except Exception:
        _ = None

    headers["X-CSRF-Token"] = getattr(client, "csrf_token", "")

    print(f"DEBUG make_request_with_csrf: Final headers being sent: {headers}")

    # Make the request
    method_func = getattr(client, method.lower())
    return method_func(url, headers=headers, **kwargs)


def get_auth_headers(auth_token):
    """Get appropriate auth headers based on AUTH_MODE."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()

    if settings.AUTH_MODE == "multi_user":
        return {"Authorization": auth_token}
    else:
        # For single-user mode, include X-API-KEY so AuthNZ deps pass
        api_key = settings.SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_TEST_API_KEY", "test-api-key-12345")
        headers = {"Authorization": auth_token}
        if api_key:
            headers["X-API-KEY"] = api_key
        return headers


# Fixture for TestClient with proper CSRF token handling
@pytest.fixture(scope="function")
def client():
    print(f"DEBUG: Creating TestClient, app.dependency_overrides before: {app.dependency_overrides}")
    with TestClient(app) as c:
        print(f"DEBUG: Created TestClient, app.dependency_overrides after: {app.dependency_overrides}")
        # Get a CSRF token by making a GET request first
        response = c.get("/api/v1/health")  # Or any GET endpoint
        csrf_token = response.cookies.get("csrf_token", "")

        # Store the token in the client for use in tests
        c.csrf_token = csrf_token
        c.cookies = {"csrf_token": csrf_token}

        # Add helper method to client
        c.post_with_csrf = lambda url, **kwargs: make_request_with_csrf(c, "POST", url, **kwargs)
        c.put_with_csrf = lambda url, **kwargs: make_request_with_csrf(c, "PUT", url, **kwargs)
        c.patch_with_csrf = lambda url, **kwargs: make_request_with_csrf(c, "PATCH", url, **kwargs)
        c.delete_with_csrf = lambda url, **kwargs: make_request_with_csrf(c, "DELETE", url, **kwargs)

        yield c


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return User(id=1, username="test_user", email="test@example.com", is_active=True)


@pytest.fixture(autouse=True)
def setup_auth_override(mock_user):
    """Automatically override authentication for all tests.

    Note: The chat endpoint doesn't use dependency injection for auth,
    so we need to mock the auth utility functions instead.
    """
    from unittest.mock import patch

    # Mock the auth utility functions used by the chat endpoint
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.is_authentication_required", return_value=False):
        # When auth is not required, the endpoint won't check the token
        yield


@pytest.fixture
def valid_auth_token() -> str:
    """Generate appropriate auth token based on current AUTH_MODE."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()

    if settings.AUTH_MODE == "multi_user":
        # In multi-user mode, we need a proper JWT token
        # For testing, we'll create a mock JWT token using the actual JWT secret
        import jwt
        import datetime

        # Use the actual JWT secret from settings
        secret_key = settings.JWT_SECRET_KEY or os.getenv("JWT_SECRET_KEY", "test-secret-key")

        payload = {
            "sub": "1",  # User ID as string
            "username": "test_user",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
            "iat": datetime.datetime.utcnow(),
            "type": "access",
        }
        test_token = jwt.encode(payload, secret_key, algorithm=settings.JWT_ALGORITHM)
        return f"Bearer {test_token}"
    else:
        # In single-user mode, use SINGLE_USER_API_KEY with Bearer prefix
        api_key = settings.SINGLE_USER_API_KEY
        if not api_key:
            # Use the default key from environment or fallback
            api_key = os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
        # Ensure Bearer prefix is included
        if not api_key.startswith("Bearer "):
            return f"Bearer {api_key}"
        return api_key


# --- Test Data defined locally in this file ---
DEFAULT_MODEL_NAME = "test-model-unit"  # Changed name to avoid potential clash if importing
DEFAULT_USER_MESSAGES_FOR_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Hello from unit test")
]  # This was the missing declaration


# Fixture to provide default chat request data
@pytest.fixture
def default_chat_request_data():
    """Provides a default ChatCompletionRequest object for tests."""
    return ChatCompletionRequest(
        model=DEFAULT_MODEL_NAME, messages=DEFAULT_USER_MESSAGES_FOR_SCHEMA  # Use the locally defined constant
    )


@pytest.fixture
def default_chat_request_data_error_stream():
    """Provides a default ChatCompletionRequest object for error streaming tests."""
    return ChatCompletionRequest(
        model=DEFAULT_MODEL_NAME,  # Use the locally defined constant
        messages=DEFAULT_USER_MESSAGES_FOR_SCHEMA,  # Use the locally defined constant
        stream=True,  # Ensure stream is true for this test
    )


# Mocks for DB dependencies
@pytest.fixture
def mock_chat_db():
    db_mock = MagicMock(spec=CharactersRAGDB)
    db_mock.get_character_card_by_id.return_value = None
    # Add default character card for by_name lookup for default character
    db_mock.get_character_card_by_name.return_value = {
        "id": "mock_default_char_id_123",  # Or an int if your DB returns int
        "name": DEFAULT_CHARACTER_NAME,  # Assuming DEFAULT_CHARACTER_NAME is importable or defined
        "system_prompt": "Mock default system prompt",
        # Add any other fields your endpoint/templating might try to access from the default char
    }
    db_mock.add_conversation.return_value = "mock_conv_id_xyz"  # For new conversation creation
    db_mock.add_message.return_value = "mock_message_id_abc"  # Or whatever it should return

    # >>> ADD THIS LINE <<<
    db_mock.client_id = "test_client_unit"
    return db_mock


@pytest.fixture
def mock_media_db():
    return MagicMock()


# --- Unit Tests for the Endpoint ---


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch(
    "tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string"
)  # Keep patch in case logic changes, but expect no calls
def test_create_chat_completion_no_template(
    mock_apply_template,
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    # Simulate that when the endpoint tries to load the default template (or any specified one),
    # it's not found, making active_template=None in the endpoint.
    mock_load_template.return_value = None

    mock_response_data = {
        "id": "chatcmpl-no-template",
        "choices": [{"message": {"role": "assistant", "content": "Raw response"}}],
    }
    mock_chat_api_call.return_value = mock_response_data

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data_dict = default_chat_request_data.model_dump()
    # In default_chat_request_data, prompt_template_name is None.
    # The endpoint will try to load DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name.

    auth_headers = get_auth_headers(valid_auth_token)
    print(f"DEBUG: Using auth headers: {auth_headers}")
    print(f"DEBUG: app.dependency_overrides at request time: {app.dependency_overrides}")
    response = client.post_with_csrf("/api/v1/chat/completions", json=request_data_dict, headers=auth_headers)

    if response.status_code != status.HTTP_200_OK:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        print(f"Response headers: {response.headers}")
    assert response.status_code == status.HTTP_200_OK
    assert (
        response.json()["choices"][0]["message"]["content"] == mock_response_data["choices"][0]["message"]["content"]
    )  # Check relevant part

    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs

    # Even though active_template is None, if a default character is loaded,
    # its system_prompt may still be passed as system_message to the API call.
    # Update the assertion to check if system_message is present and has expected value
    if "system_message" in called_kwargs:
        # If present, it should be from the default character
        assert called_kwargs["system_message"] == "Mock default system prompt"

    # Messages should be passed through as-is when using the raw passthrough template
    expected_payload_messages_as_dicts = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_FOR_SCHEMA]
    actual_payload_messages = called_kwargs["messages_payload"]
    assert actual_payload_messages == expected_payload_messages_as_dicts

    # load_template IS called because request_data.prompt_template_name is None,
    # so it tries to load DEFAULT_RAW_PASSTHROUGH_TEMPLATE.
    mock_load_template.assert_called_once_with(DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name)

    # apply_template_to_string should NOT be called because active_template is None
    # (due to mock_load_template.return_value = None) and sys_msg_from_req is empty.
    mock_apply_template.assert_not_called()

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


# (The rest of the tests in test_chat_endpoint.py remain the same as the corrected version from the previous response)
# Ensure they use the `default_chat_request_data` fixture where appropriate.


@pytest.mark.unit
@pytest.mark.skip(reason="Streaming tests hang with TestClient - needs investigation")
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
def test_create_chat_completion_success_streaming(  # Added default_chat_request_data fixture
    mock_apply_template,
    mock_load_template,
    mock_chat_api_call,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    with patch(
        "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call"
    ) as mock_chat_api_call_inner:  # Renamed for clarity

        def mock_stream_generator():  # This mock is for the return value of perform_chat_api_call
            yield "Hello"  # Just the content delta
            yield " World"

        mock_chat_api_call_inner.return_value = mock_stream_generator()  # Corrected to use the inner mock

        app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
        app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

        streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
        response = client.post_with_csrf(
            "/api/v1/chat/completions", json=streaming_request_data.model_dump(), headers={"Authorization": valid_auth_token}
        )

        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers["content-type"].lower()

        stream_content = response.text
        events = [line for line in stream_content.split("\n\n") if line.strip()]
        assert events[0].startswith("event: stream_start")
        # Extract the data part and verify conversation_id
        data_line = events[0].split("\n")[1]  # Get the second line which contains the data
        assert json.loads(data_line.split("data: ", 1)[1])["conversation_id"] == "mock_conv_id_xyz"
        # events[1] would be 'data: {"choices": [{"delta": {"content": "Hello"}}]}' (after metadata)
        # events[2] would be 'data: {"choices": [{"delta": {"content": " World"}}]}'
        assert json.loads(events[1].split("data: ", 1)[1])["choices"][0]["delta"]["content"] == "Hello"
        assert json.loads(events[2].split("data: ", 1)[1])["choices"][0]["delta"]["content"] == " World"
        assert (
            "data: " in events[-1]
            and json.loads(events[-1].split("data: ", 1)[1]).get("choices")[0].get("finish_reason") == "stop"
        )

        mock_chat_api_call_inner.assert_called_once()
        call_args = mock_chat_api_call_inner.call_args[1]
        assert call_args["streaming"] is True

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
def test_system_message_extraction(
    mock_apply_template, mock_load_template, mock_chat_api_call, client, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    messages_with_system = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
        ChatCompletionUserMessageParam(role="user", content="Hello there."),
    ]
    # Use the specific Pydantic models from your schema
    request_data_obj = ChatCompletionRequest(model="test-model", messages=messages_with_system)

    client.post_with_csrf(
        "/api/v1/chat/completions", json=request_data_obj.model_dump(), headers={"Authorization": valid_auth_token}
    )

    mock_chat_api_call.assert_called_once()
    call_args = mock_chat_api_call.call_args.kwargs
    assert call_args["system_message"] == "You are a helpful assistant."
    # System messages are passed separately; payload should only include non-system roles.
    assert len(call_args["messages_payload"]) == 1
    assert call_args["messages_payload"][0]["role"] == "user"
    assert call_args["messages_payload"][0]["content"] == "Hello there."  # Assuming passthrough template
    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
def test_no_system_message_in_payload(
    mock_apply_template,
    mock_load_template,
    mock_chat_api_call,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    # Added default_chat_request_data
):
    mock_load_template.return_value = (
        None  # Default passthrough because prompt_template_name is None in default_chat_request_data
    )
    # Simulate passthrough for apply_template when default template is used
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    mock_chat_api_call.return_value = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    client.post_with_csrf(
        "/api/v1/chat/completions", json=default_chat_request_data.model_dump(), headers={"Authorization": valid_auth_token}
    )

    mock_chat_api_call.assert_called_once()
    # This line was missing:
    called_kwargs = mock_chat_api_call.call_args.kwargs
    # When no system message is provided, the endpoint uses the character's default system prompt
    # This is correct behavior
    system_msg = called_kwargs.get("system_message")
    assert system_msg is not None  # Should have some system message (from character or default)

    # Ensure the messages in the payload are dictionaries and match the input (since it's passthrough)
    expected_payload_messages_as_dicts = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_FOR_SCHEMA]
    # The endpoint logic (after templating with passthrough) should result in this payload
    actual_payload_messages = called_kwargs["messages_payload"]

    assert len(actual_payload_messages) == len(expected_payload_messages_as_dicts)
    for actual_msg, expected_msg in zip(actual_payload_messages, expected_payload_messages_as_dicts):
        assert actual_msg["role"] == expected_msg["role"]
        # The content from the user message (a string in this case) should pass through the template
        # The template application might wrap it in a list of content parts if it wasn't already
        # Your endpoint logic: `msg_dict["content"] = new_content_str` for string content
        # So it should be a direct string comparison if the passthrough template just returns the message_content
        assert actual_msg["content"] == expected_msg["content"]

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


VALID_ALTERNATIVE_PROVIDER_FOR_TEST = "groq"


@pytest.mark.unit
# Update the patch.dict to use this valid alternative provider name
@patch.dict(
    "tldw_Server_API.app.api.v1.schemas.chat_request_schemas.API_KEYS",
    {
        "openai": "key_from_config",
        VALID_ALTERNATIVE_PROVIDER_FOR_TEST: "alternative_key_for_test",  # Use the valid name
        "cohere": "cohere_test_key_if_needed_separately",  # If you still have a cohere specific part
    },
)
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
@pytest.mark.skip(reason="Flaky under composite run due to provider defaults; validated in isolation")
def test_api_key_used_from_config(
    mock_apply_template,
    mock_load_template,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
):
    mock_load_template.return_value = None
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_chat_api_call:
        mock_chat_api_call.return_value = {"id": "res_openai"}

        # Test 1: Default provider (should pick up "openai" key from patched dict)
        # default_chat_request_data has api_provider=None, so it will use DEFAULT_LLM_PROVIDER ("openai")
        client.post_with_csrf(
            "/api/v1/chat/completions", json=default_chat_request_data.model_dump(), headers={"Authorization": valid_auth_token}
        )
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "key_from_config"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == "openai"  # Check target endpoint

        mock_chat_api_call.reset_mock()
        mock_chat_api_call.return_value = {"id": "res_alternative"}

        # Test 2: Specific valid alternative provider
        request_data_alternative = default_chat_request_data.model_copy(
            update={
                "api_provider": VALID_ALTERNATIVE_PROVIDER_FOR_TEST,
                # Ensure a model is provided if the alternative provider requires it,
                # default_chat_request_data already includes a model='test-model-unit'
                # which might be fine if the mock_chat_api_call doesn't care about model validity for this unit test.
                # If it were an integration test, a valid model for the provider would be needed.
            }
        )
        response_alternative = client.post(
            "/api/v1/chat/completions", json=request_data_alternative.model_dump(), headers={"Authorization": valid_auth_token}
        )

        # This assertion should now pass if VALID_ALTERNATIVE_PROVIDER_FOR_TEST is correctly handled
        assert (
            response_alternative.status_code == status.HTTP_200_OK
        ), f"Alternative provider '{VALID_ALTERNATIVE_PROVIDER_FOR_TEST}' failed: {response_alternative.text}"

        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "alternative_key_for_test"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == VALID_ALTERNATIVE_PROVIDER_FOR_TEST

        # If you had a third part of the test for "cohere" specifically:
        mock_chat_api_call.reset_mock()
        mock_chat_api_call.return_value = {"id": "res_cohere"}
        request_data_cohere = default_chat_request_data.model_copy(
            update={"api_provider": "cohere", "model": "command-r"}  # Assuming 'cohere' is in SUPPORTED_API_ENDPOINTS
        )
        response_cohere = client.post(
            "/api/v1/chat/completions", json=request_data_cohere.model_dump(), headers={"Authorization": valid_auth_token}
        )
        assert response_cohere.status_code == status.HTTP_200_OK, f"Cohere provider failed: {response_cohere.text}"
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args.kwargs["api_key"] == "cohere_test_key_if_needed_separately"
        assert mock_chat_api_call.call_args.kwargs["api_endpoint"] == "cohere"

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": ""})  # Empty key for openai
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_missing_api_key_for_required_provider(
    mock_apply_template,
    mock_load_template,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    monkeypatch,
):
    # Simulate that the default template is found and is a passthrough
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # We don't need to mock chat_api_call here as the error should occur before it's called.
    # However, if any part of the setup before the key check calls it, a basic mock might be needed.
    # The key check happens quite early.

    # default_chat_request_data uses 'openai' as the default provider if api_provider is None.
    # We've patched API_KEYS so "openai" has an empty string key.
    # The endpoint's providers_requiring_keys list includes "openai".
    request_data_openai = default_chat_request_data.model_copy(update={"api_provider": "openai"})

    reg = get_metrics_registry()
    labels = {"provider": "openai", "operation": "chat"}
    before = reg.get_metric_stats("byok_missing_credentials_total", labels=labels).get("count", 0)

    # Ensure endpoint prefers module-level API_KEYS for this test to avoid env/interference
    # Do NOT enable TEST_MODE here, since TEST_MODE enables auto-mock for some providers and bypasses 503.
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.byok_runtime.resolve_server_default_key",
        lambda _provider: None,
    )
    from unittest.mock import patch as _patch

    with _patch("tldw_Server_API.app.api.v1.endpoints.chat.get_api_keys", return_value={}):
        response = client.post_with_csrf(
            "/api/v1/chat/completions", json=request_data_openai.model_dump(), headers={"Authorization": valid_auth_token}
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = response.json()["detail"]
    assert detail["error_code"] == "missing_provider_credentials"
    after = reg.get_metric_stats("byok_missing_credentials_total", labels=labels).get("count", 0)
    assert after >= before + 1

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")  # Mock template deps
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
def test_keyless_provider_proceeds_without_key(  # Added default_chat_request_data
    mock_apply_template,
    mock_load_template,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    monkeypatch,
    bypass_api_limits,
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY_99", raising=False)
    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY_1", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.byok_runtime.resolve_server_default_key",
        lambda _provider: None,
    )

    with (
        bypass_api_limits(app),
        patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {}, clear=True),
        patch.dict("tldw_Server_API.app.api.v1.schemas.chat_request_schemas.API_KEYS", {}, clear=True),
        patch("tldw_Server_API.app.api.v1.schemas.chat_request_schemas.get_api_keys", return_value={}),
        patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") as mock_chat_api_call,
    ):
        mock_chat_api_call.return_value = {"id": "res_custom_openai"}
        request_data_keyless = default_chat_request_data.model_copy(update={"api_provider": "custom-openai-api"})

        response = client.post_with_csrf(
            "/api/v1/chat/completions",
            json=request_data_keyless.model_dump(),
            headers={"Authorization": valid_auth_token},
        )
        assert response.status_code == status.HTTP_200_OK
        mock_chat_api_call.assert_called_once()
        api_key = mock_chat_api_call.call_args[1].get("api_key")
        assert api_key is None, f"Expected None for custom-openai-api, got: {api_key!r}"
    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")  # Corrected patch target
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
@pytest.mark.parametrize(
    "error_type, expected_status, expected_detail_substring",
    [
        (
            ChatAuthenticationError(provider="test", message="Auth failed detail from lib"),
            # Error from perform_chat_api_call
            status.HTTP_401_UNAUTHORIZED,
            "unauthorized",
        ),
        (
            ChatRateLimitError(provider="test", message="Rate limit detail from lib"),
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
        ),
        (
            ChatBadRequestError(provider="test", message="Bad request detail from lib"),
            status.HTTP_400_BAD_REQUEST,
            "invalid request",
        ),
        (
            ChatConfigurationError(provider="test", message="Config error from lib"),  # This is a 5xx type error
            status.HTTP_503_SERVICE_UNAVAILABLE,  # Endpoint maps ChatConfigurationError to 503
            "The chat service is temporarily unavailable.",
        ),  # Endpoint masks 5xx details
        (
            ChatProviderError(
                provider="test", message="Provider issue from lib", status_code=503
            ),  # This is a 5xx type error
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The chat service is temporarily unavailable.",
        ),
        (
            ChatProviderError(provider="test", message="Provider non-HTTP issue from lib", status_code=502),
            # This is a 5xx type error
            status.HTTP_502_BAD_GATEWAY,
            "The chat service provider is currently unavailable.",
        ),
        (
            ChatAPIError(
                provider="test", message="Generic API issue from lib", status_code=500
            ),  # This is a 5xx type error
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An internal server error occurred.",
        ),
        # Case: A non-library, non-HTTPException error from perform_chat_api_call (e.g., a raw ValueError)
        # The endpoint's final `except Exception` catches this.
        (
            ValueError("Value error from shim"),
            status.HTTP_500_INTERNAL_SERVER_ERROR,  # Endpoint's generic catch-all
            "An unexpected internal server error occurred.",
        ),
        # Case: An HTTPException raised directly by perform_chat_api_call
        # The endpoint's general exception handler will catch this and return 500
        (
            HTTPException(status_code=418, detail="I'm a teapot from shim"),
            status.HTTP_500_INTERNAL_SERVER_ERROR,  # General exception handler catches this
            "An unexpected internal server error occurred.",
        ),  # Generic error message
    ],
)
def test_chat_api_call_exception_handling_unit(
    mock_apply_template,
    mock_load_template,
    mock_perform_chat_api_call,  # Corrected mock name
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
    error_type,
    expected_status,
    expected_detail_substring,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    # This is the mock for perform_chat_api_call inside the endpoint
    mock_perform_chat_api_call.side_effect = error_type

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    response = client.post_with_csrf(
        "/api/v1/chat/completions",
        json=default_chat_request_data.model_dump(),
        headers={"Authorization": valid_auth_token},  # Ensure correct header name
    )

    assert response.status_code == expected_status

    response_json = response.json()
    assert "detail" in response_json, "Response JSON should contain a 'detail' field"
    response_detail_text = response_json["detail"]

    # For string details, check if the expected substring is present.
    # For dict details (e.g. Pydantic validation errors), this check might need adjustment,
    # but for these specific exception handlings, detail is expected to be a string.
    assert isinstance(
        response_detail_text, str
    ), f"Response detail should be a string, got {type(response_detail_text)}"
    detail_lower = response_detail_text.lower()
    assert (
        expected_detail_substring.lower() in detail_lower
    ), f"Expected detail '{expected_detail_substring}' not found in actual detail '{response_detail_text}'"
    if expected_status == status.HTTP_401_UNAUTHORIZED:
        assert (
            "key" in detail_lower or "token" in detail_lower or detail_lower.strip() == "unauthorized."
        ), f"Expected auth error with credential context, got: {response_detail_text}"
    if expected_status == status.HTTP_429_TOO_MANY_REQUESTS:
        assert (
            "retry" in detail_lower or "try again" in detail_lower
        ), f"Expected rate limit error with retry guidance, got: {response_detail_text}"
    if expected_status == status.HTTP_400_BAD_REQUEST:
        assert (
            "input" in detail_lower
            or "payload" in detail_lower
            or "parameter" in detail_lower
            or detail_lower.strip() == "invalid request."
        ), f"Expected invalid request error with input context, got: {response_detail_text}"

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
@patch("tldw_Server_API.app.core.Chat.chat_service.apply_template_to_string")
def test_non_iterable_stream_generator_from_shim(
    mock_apply_template,
    mock_load_template,
    mock_chat_api_call,
    client,
    default_chat_request_data,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
):
    # Simulate that the default template is found and is a passthrough
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    # Simulate chat_api_call (perform_chat_api_call) returning a non-iterable/non-async-iterable
    mock_chat_api_call.return_value = 123

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
    response = client.post_with_csrf(
        "/api/v1/chat/completions", json=streaming_request_data.model_dump(), headers={"Authorization": valid_auth_token}
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "Provider did not return a valid stream."

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.skip(reason="Async generator error handling causes timeout - needs refactoring")
@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")  # Patches the alias used in chat.py
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_error_within_stream_generator(
    mock_apply_template,
    mock_load_template,
    mock_chat_api_call,
    client,
    default_chat_request_data_error_stream,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
):
    # Simulate that the default template (or any) is found and is a passthrough
    # or correctly loaded if it's DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get(
        "message_content", data.get("original_system_message_from_request", "")
    )

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # This inner patch is fine if you want to ensure the outer mock_perform_chat_api_call
    # doesn't interfere, but the outer one should suffice.
    # For clarity, using the already patched mock_perform_chat_api_call directly.

    def faulty_stream_generator():  # This is what perform_chat_api_call (mocked) will return
        yield "Good start..."  # Raw content, not full SSE
        raise ValueError("Something broke mid-stream!")

    mock_chat_api_call.return_value = faulty_stream_generator()

    # Use the specific request data for this test
    request_data_dict = default_chat_request_data_error_stream.model_dump()

    response = client.post_with_csrf(
        "/api/v1/chat/completions", json=request_data_dict, headers={"Authorization": valid_auth_token}  # Corrected header name
    )
    assert response.status_code == status.HTTP_200_OK  # Stream should still start with 200

    # Process the SSE stream
    raw_stream_text = response.text
    print(f"DEBUG Raw Stream Output:\n{raw_stream_text}")  # For debugging

    events = []
    current_event_data = []
    for line in raw_stream_text.splitlines():
        if not line.strip():  # End of an event
            if current_event_data:
                events.append("\n".join(current_event_data))
                current_event_data = []
        else:
            current_event_data.append(line)
    if current_event_data:  # Append last event if any
        events.append("\n".join(current_event_data))

    print(f"DEBUG Parsed SSE Events: {events}")

    assert len(events) >= 3  # Expect metadata, data/error, DONE

    # 1. Check metadata event
    assert events[0].startswith("event: stream_start")
    # Extract the data part from the second line of the event
    data_line = events[0].split("\n")[1] if "\n" in events[0] else events[0]
    metadata_json = json.loads(data_line.split("data: ", 1)[1])
    expected_conv_id = metadata_json["conversation_id"]  # Get the actual conv_id from metadata
    assert expected_conv_id == "mock_conv_id_xyz"
    assert metadata_json["model"] == DEFAULT_MODEL_NAME

    # 2. Check for "Good start..." data chunk formatted by the endpoint
    # This part might be tricky if the error occurs before the first data yield is fully processed by sse_event_generator
    # The sse_event_generator will try to yield the "Good start..." data first.
    first_data_event_found = False
    error_event_found = False
    done_event_found = False

    for event_str in events[1:]:  # Skip metadata
        if event_str.startswith("data: "):
            try:
                payload_str = event_str.split("data: ", 1)[1]
                payload = json.loads(payload_str)

                if "choices" in payload and payload["choices"]:
                    delta = payload["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"] == "Good start...":
                        first_data_event_found = True
                    if payload["choices"][0].get("finish_reason") == "stop":
                        done_event_found = True
                        # Use the previously extracted expected_conv_id:
                        assert payload.get("tldw_conversation_id") == expected_conv_id

                if "error" in payload and "message" in payload["error"]:
                    if (
                        "Something broke mid-stream!" in payload["error"]["message"]
                        or "Stream failed due to provider error." in payload["error"]["message"]
                    ):  # Endpoint wraps the error
                        error_event_found = True
            except json.JSONDecodeError:
                print(f"WARN: Could not decode JSON from event: {payload_str}")
                continue

    # Depending on precise timing of exception in generator, "Good start..." might or might not appear
    # The error and DONE message are more crucial for this test.
    # If faulty_stream_generator raises immediately after yielding "Good start...",
    # the endpoint's sse_event_generator should process that yield, then catch the error.
    assert first_data_event_found, "The 'Good start...' data chunk was not found or not correctly formatted."
    assert error_event_found, "The SSE error message was not found or not correctly formatted."
    assert done_event_found, "The SSE DONE message was not found or not correctly formatted."

    # Ensure the background task (if any was triggered by the stream ending) is handled gracefully.
    # The test mock for _save_message_turn_to_db will be called by final_save_bg_task.
    # We can check its call if needed, but the main focus is the stream content.
    # mock_chat_db.add_message.assert_called() # Or more specific checks on what was saved.
    # The `_sse_state['full_reply']` would contain "Good start..." in this case.

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_create_chat_completion_with_optional_params(
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE  # Use the actual default
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-optional",
        "choices": [{"message": {"role": "assistant", "content": "Response with optionals"}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_with_optionals = default_chat_request_data.model_copy(
        update={
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 150,
            "n": 2,
            "response_format": ResponseFormat(type="json_object"),
            "seed": 12345,
            "stop": ["\n", "stopword"],
            "user": "test-user-id",
            "minp": 0.05,  # Custom extension
            "topk": 50,  # Custom extension
        }
    )
    request_data_dict = request_with_optionals.model_dump(exclude_none=True)

    response = client.post_with_csrf(
        "/api/v1/chat/completions", json=request_data_dict, headers={"Authorization": valid_auth_token}
    )

    assert response.status_code == status.HTTP_200_OK
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs

    assert called_kwargs["frequency_penalty"] == 0.5
    assert called_kwargs["presence_penalty"] == -0.5
    assert called_kwargs["logprobs"] is True
    assert called_kwargs["top_logprobs"] == 5
    # max_tokens is not directly mapped by chat_args, but by the schema to chat_api_call's provider logic
    # For a unit test of the endpoint, we check it's passed to chat_api_call if chat_api_call accepts it
    # The current chat_api_call doesn't explicitly list max_tokens in its signature,
    # so it depends on downstream adapter/handler forwarding behavior.
    # Let's focus on params explicitly in chat_api_call signature.
    # assert called_kwargs["n"] == 2 # 'n' is also not directly in chat_api_call signature
    # assert called_kwargs["response_format"] == {"type": "json_object"} # also not direct
    # assert called_kwargs["seed"] == 12345 # also not direct
    # assert called_kwargs["stop"] == ["\n", "stopword"] # also not direct
    assert called_kwargs["minp"] == 0.05
    assert called_kwargs["topk"] == 50
    # user is not directly passed to chat_api_call

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_non_stream_structured_json_schema_success_returns_metadata(
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-structured-success",
        "choices": [{"message": {"role": "assistant", "content": '{"answer":"ok"}'}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data_dict = default_chat_request_data.model_dump(exclude_none=True)
    request_data_dict["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    response = client.post_with_csrf(
        "/api/v1/chat/completions",
        json=request_data_dict,
        headers=get_auth_headers(valid_auth_token),
    )

    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["tldw_structured"]["validated"] is True
    assert payload["tldw_structured"]["validated_payload"] == {"answer": "ok"}
    assert payload["tldw_structured"]["mode_used"] in {"json_object", "json_schema"}
    assert isinstance(payload["tldw_structured"]["fallback_used"], bool)

    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_non_stream_structured_json_schema_failure_returns_400(
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-structured-failure",
        "choices": [{"message": {"role": "assistant", "content": '{"answer":123}'}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data_dict = default_chat_request_data.model_dump(exclude_none=True)
    request_data_dict["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    response = client.post_with_csrf(
        "/api/v1/chat/completions",
        json=request_data_dict,
        headers=get_auth_headers(valid_auth_token),
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    payload = response.json()
    assert payload["detail"]["code"] == "structured_output_schema_error"

    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_create_chat_completion_with_tools_unit(
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-tools",
        "choices": [{"message": {"role": "assistant", "tool_calls": []}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    tools_payload = [
        ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_current_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            ),
        )
    ]
    tool_choice_payload = ToolChoiceOption(type="function", function=ToolChoiceFunction(name="get_current_weather"))

    request_with_tools = default_chat_request_data.model_copy(
        update={
            "tools": tools_payload,  # Pass the actual ToolDefinition objects
            "tool_choice": tool_choice_payload,  # Pass the actual ToolChoiceOption object
        }
    )
    request_data_dict = request_with_tools.model_dump(exclude_none=True)

    response = client.post_with_csrf(
        "/api/v1/chat/completions", json=request_data_dict, headers={"Authorization": valid_auth_token}
    )
    if response.status_code != status.HTTP_200_OK:
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
    assert response.status_code == status.HTTP_200_OK
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    # tool_choice from request_data.model_dump() is a dict
    assert called_kwargs.get("tool_choice") == tool_choice_payload.model_dump(exclude_none=True)
    assert called_kwargs.get("tools") == [t.model_dump(exclude_none=True) for t in tools_payload]

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
def test_save_to_db_not_passed_to_chat_api_call(
    mock_load_template,
    mock_chat_api_call,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_chat_api_call.return_value = {
        "id": "chatcmpl-ok",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # Include save_to_db in request to ensure it's not forwarded to chat_api_call
    body = default_chat_request_data.model_copy(update={"save_to_db": True}).model_dump()
    response = client.post_with_csrf("/api/v1/chat/completions", json=body, headers={"Authorization": valid_auth_token})
    assert response.status_code == status.HTTP_200_OK

    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    assert "save_to_db" not in called_kwargs

    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")  # Mock this
def test_create_chat_completion_character_not_found_uses_defaults(
    mock_load_template,
    mock_chat_api_call_shim,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    # Mock DB to return None for character
    mock_chat_db.get_character_card_by_id.return_value = None
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE  # Or a specific test template

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    mock_chat_api_call_shim.return_value = {"id": "res", "choices": [{"message": {"content": "default response"}}]}

    request_with_char = default_chat_request_data.model_copy(
        update={
            "character_id": "non_existent_char_id",
            "prompt_template_name": "some_template_that_uses_char_vars",  # Assume this template exists for the test
        }
    )
    # If some_template_that_uses_char_vars is mocked by mock_load_template to use {char_name}
    # And char_name is not found, it will use the default "Character" from template_data initialization.

    client.post_with_csrf(
        "/api/v1/chat/completions", json=request_with_char.model_dump(), headers={"Authorization": valid_auth_token}
    )

    mock_chat_api_call_shim.assert_called_once()
    called_args_to_shim = mock_chat_api_call_shim.call_args.kwargs
    # Check that the system_message sent to chat_api_call uses default character values if the template was applied.
    # This depends on how DEFAULT_RAW_PASSTHROUGH_TEMPLATE is structured or the mock for "some_template_that_uses_char_vars"
    # If system_message_template in the active_template was "{char_name} says: {original_system_message_from_request}"
    # And no char was found, and no original system message, it might become "Character says: "
    # For the default passthrough, if original system message is empty, this would be empty.
    assert (
        called_args_to_shim.get("system_message") is not None
    )  # It will be at least "" if DEFAULT_RAW_PASSTHROUGH_TEMPLATE is used

    # Verify DB was called
    mock_chat_db.get_character_card_by_name.assert_called_once_with("non_existent_char_id")
    mock_chat_db.get_character_card_by_id.assert_not_called()

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "test_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call")
@patch("tldw_Server_API.app.core.Chat.chat_service.load_template")
def test_create_chat_completion_template_file_not_found(
    mock_load_template,
    mock_chat_api_call_shim,
    client,
    valid_auth_token,
    mock_media_db,
    mock_chat_db,
    default_chat_request_data,
):
    # Simulate load_template returning None (template not found)
    mock_load_template.return_value = None
    # chat_api_call should still be called with DEFAULT_RAW_PASSTHROUGH_TEMPLATE logic
    mock_chat_api_call_shim.return_value = {"id": "res", "choices": [{"message": {"content": "passthrough response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    request_data = default_chat_request_data.model_copy(update={"prompt_template_name": "definitely_missing_template"})

    response = client.post_with_csrf(
        "/api/v1/chat/completions", json=request_data.model_dump(), headers={"Authorization": valid_auth_token}
    )
    assert response.status_code == status.HTTP_200_OK  # Should fall back to default template
    mock_load_template.assert_called_once_with("definitely_missing_template")
    mock_chat_api_call_shim.assert_called_once()
    # Further assertions could check if the payload sent to chat_api_call reflects the passthrough template.

    # Clean up only the overrides we added (not the auth override from fixture)
    app.dependency_overrides.pop(get_media_db_for_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)
