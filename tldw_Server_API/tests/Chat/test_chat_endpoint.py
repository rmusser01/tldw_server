# tests/unit/api/v1/endpoints/test_chat_endpoint.py
import pytest
from unittest.mock import patch, MagicMock, ANY
from fastapi import status, HTTPException
from fastapi.testclient import TestClient
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your FastAPI app instance
from tldw_Server_API.app.main import app
# Import schemas from your actual file path
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartImageURL
)
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatConfigurationError, ChatProviderError, ChatAPIError
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate, DEFAULT_RAW_PASSTHROUGH_TEMPLATE
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user


# Fixture for TestClient
@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c


# Fixture for a valid auth token
@pytest.fixture
def valid_auth_token() -> str:
    token = os.getenv("TEST_AUTH_TOKEN", "test-user-token-if-auth-is-mocked")
    return token


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
        model=DEFAULT_MODEL_NAME,
        messages=DEFAULT_USER_MESSAGES_FOR_SCHEMA  # Use the locally defined constant
    )


# Mocks for DB dependencies
@pytest.fixture
def mock_chat_db():
    db_mock = MagicMock(spec=CharactersRAGDB)
    db_mock.get_character_card_by_id.return_value = None
    return db_mock


@pytest.fixture
def mock_media_db():
    return MagicMock()


# --- Unit Tests for the Endpoint ---

@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_no_template(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db, default_chat_request_data  # Use the fixture
):
    mock_load_template.return_value = None
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_response_data = {"id": "chatcmpl-no-template",
                          "choices": [{"message": {"role": "assistant", "content": "Raw response"}}]}
    mock_chat_api_call.return_value = mock_response_data

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    # Use the fixture for the request data
    request_data_dict = default_chat_request_data.model_dump()

    response = client.post("/api/v1/chat/completions", json=request_data_dict, headers={"token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == mock_response_data
    mock_chat_api_call.assert_called_once()
    called_kwargs = mock_chat_api_call.call_args.kwargs
    assert called_kwargs.get("system_message") == ""

    # Compare content of messages payload - Now uses the locally defined DEFAULT_USER_MESSAGES_FOR_SCHEMA
    expected_payload_messages_as_dicts = [msg.model_dump(exclude_none=True) for msg in DEFAULT_USER_MESSAGES_FOR_SCHEMA]
    actual_payload_messages = called_kwargs["messages_payload"]

    # Assuming default_chat_request_data has one user message
    assert len(actual_payload_messages) == len(expected_payload_messages_as_dicts)
    # The content of the user message (which is a string for this simple case)
    assert actual_payload_messages[0]["content"] == expected_payload_messages_as_dicts[0]["content"]
    assert actual_payload_messages[0]["role"] == expected_payload_messages_as_dicts[0]["role"]

    mock_load_template.assert_not_called()
    passthrough_user_template = DEFAULT_RAW_PASSTHROUGH_TEMPLATE.user_message_content_template
    user_content_template_call = any(
        cargs[0] == passthrough_user_template and cargs[1].get("message_content") == DEFAULT_USER_MESSAGES_FOR_SCHEMA[
            0].content
        for cargs, _ in mock_apply_template.call_args_list
    )
    assert user_content_template_call

    app.dependency_overrides = {}


# (The rest of the tests in test_chat_endpoint.py remain the same as the corrected version from the previous response)
# Ensure they use the `default_chat_request_data` fixture where appropriate.

@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_create_chat_completion_success_streaming(  # Added default_chat_request_data fixture
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    with patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call") as mock_chat_api_call_inner:  # Renamed for clarity
        def mock_stream_generator():
            yield "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n"
            yield "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \" World\"}}]}\n\n"

        mock_chat_api_call_inner.return_value = mock_stream_generator()  # Corrected to use the inner mock

        app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
        app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

        streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
        response = client.post(
            "/api/v1/chat/completions",
            json=streaming_request_data.model_dump(),
            headers={"token": valid_auth_token}
        )

        assert response.status_code == status.HTTP_200_OK
        assert "text/event-stream" in response.headers["content-type"].lower()

        stream_content = response.text
        chunks = [line for line in stream_content.split("\n\n") if line.strip()]

        assert "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}" in chunks[0]
        assert "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \" World\"}}]}" in chunks[1]
        assert "data: [DONE]" in chunks[-1]

        mock_chat_api_call_inner.assert_called_once()
        call_args = mock_chat_api_call_inner.call_args[1]
        assert call_args["streaming"] is True

    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_system_message_extraction(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_chat_api_call.return_value = {"id": "chatcmpl-123",
                                       "choices": [{"message": {"role": "assistant", "content": "Test response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    messages_with_system = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant."),
        ChatCompletionUserMessageParam(role="user", content="Hello there.")
    ]
    # Use the specific Pydantic models from your schema
    request_data_obj = ChatCompletionRequest(model="test-model", messages=messages_with_system)

    client.post("/api/v1/chat/completions", json=request_data_obj.model_dump(), headers={"token": valid_auth_token})

    mock_chat_api_call.assert_called_once()
    call_args = mock_chat_api_call.call_args.kwargs
    assert call_args["system_message"] == "You are a helpful assistant."
    assert len(call_args["messages_payload"]) == 1
    assert call_args["messages_payload"][0]["role"] == "user"
    assert call_args["messages_payload"][0]["content"] == "Hello there."  # Assuming passthrough template
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_no_system_message_in_payload(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
        # Added default_chat_request_data
):
    mock_load_template.return_value = None  # Default passthrough because prompt_template_name is None in default_chat_request_data
    # Simulate passthrough for apply_template when default template is used
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    mock_chat_api_call.return_value = {"id": "chatcmpl-123",
                                       "choices": [{"message": {"role": "assistant", "content": "Test response"}}]}

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    client.post("/api/v1/chat/completions", json=default_chat_request_data.model_dump(),
                headers={"token": valid_auth_token})

    mock_chat_api_call.assert_called_once()
    # This line was missing:
    called_kwargs = mock_chat_api_call.call_args.kwargs  # Get the keyword arguments passed to the mock

    # Assert that system_message passed to chat_api_call is None because:
    # 1. default_chat_request_data has no system message.
    # 2. No template is specified, so DEFAULT_RAW_PASSTHROUGH_TEMPLATE is used.
    # 3. DEFAULT_RAW_PASSTHROUGH_TEMPLATE.system_message_template is likely empty or just "{original_system_message_from_request}".
    # 4. original_system_message_from_request will be "" or None.
    # So, final_system_message_for_provider should be None or empty, leading to system_message=None in chat_args_cleaned.
    assert called_kwargs.get("system_message") == ""

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

    app.dependency_overrides = {}


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.schemas.chat_request_schemas.API_KEYS", {"openai": "key_from_config", "other_provider": "other_key"})
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")  # Mock template deps
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_api_key_used_from_config(  # Added default_chat_request_data fixture
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call") as mock_chat_api_call:
        mock_chat_api_call.return_value = {"id": "res"}

        client.post("/api/v1/chat/completions", json=default_chat_request_data.model_dump(),
                    headers={"token": valid_auth_token})
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args[1]["api_key"] == "key_from_config"

        mock_chat_api_call.reset_mock()
        request_data_other = default_chat_request_data.model_copy(update={"api_provider": "other_provider"})
        response_other = client.post("/api/v1/chat/completions", json=request_data_other.model_dump(),
                                     headers={"token": valid_auth_token})
        assert response_other.status_code == status.HTTP_200_OK
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args[1]["api_key"] == "other_key"
        mock_chat_api_call.reset_mock()
        # Test with 'cohere' (or another valid provider from SUPPORTED_API_ENDPOINTS)
        request_data_other = default_chat_request_data.model_copy(update={"api_provider": "cohere"})
        response_other = client.post("/api/v1/chat/completions", json=request_data_other.model_dump(),
                                     headers={"token": valid_auth_token})
        assert response_other.status_code == status.HTTP_200_OK  # Expect 200 now

        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args[1]["api_key"] == "cohere_test_key"
    app.dependency_overrides = {}


@pytest.mark.unit
@patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": ""})  # Empty key
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")  # Mock template deps
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_missing_api_key_for_required_provider(  # Added default_chat_request_data
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call"):
        request_data_openai = default_chat_request_data.model_copy(update={"api_provider": "openai"})
        response = client.post(
            "/api/v1/chat/completions",
            json=request_data_openai.model_dump(),
            headers={"token": valid_auth_token}
        )
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "API key is not configured" in response.json()["detail"]
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")  # Mock template deps
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_keyless_provider_proceeds_without_key(  # Added default_chat_request_data
        mock_apply_template, mock_load_template,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch.dict("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {}, clear=True), \
            patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call") as mock_chat_api_call:
        mock_chat_api_call.return_value = {"id": "res_ollama"}
        request_data_ollama = default_chat_request_data.model_copy(update={"api_provider": "ollama"})

        response = client.post(
            "/api/v1/chat/completions",
            json=request_data_ollama.model_dump(),
            headers={"token": valid_auth_token}
        )
        assert response.status_code == status.HTTP_200_OK
        mock_chat_api_call.assert_called_once()
        assert mock_chat_api_call.call_args[1].get(
            "api_key") is None  # Check that api_key was indeed None or not passed
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
@pytest.mark.parametrize("error_type, expected_status, error_message_detail", [
    (ChatAuthenticationError(provider="test", message="Auth failed"), status.HTTP_401_UNAUTHORIZED, "Auth failed"),
    (ChatRateLimitError(provider="test", message="Rate limit"), status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit"),
    (ChatBadRequestError(provider="test", message="Bad request"), status.HTTP_400_BAD_REQUEST, "Bad request"),
    (ChatConfigurationError(provider="test", message="Config error"), status.HTTP_500_INTERNAL_SERVER_ERROR,
     "Config error"),
    (ChatProviderError(provider="test", message="Provider issue", status_code=503), status.HTTP_503_SERVICE_UNAVAILABLE,
     "Provider issue"),
    (ChatProviderError(provider="test", message="Provider non-HTTP issue"), status.HTTP_502_BAD_GATEWAY,
     "Provider non-HTTP issue"),
    (ChatAPIError(provider="test", message="Generic API issue"), status.HTTP_500_INTERNAL_SERVER_ERROR,
     "Generic API issue"),
    (ValueError("Value error from shim"), status.HTTP_400_BAD_REQUEST,
     "Invalid parameter or internal configuration error: Value error from shim"),
    (HTTPException(status_code=418, detail="I'm a teapot from shim"), 418, "I'm a teapot from shim"),
])
def test_chat_api_call_exception_handling_unit(
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, valid_auth_token, mock_media_db, mock_chat_db,
        default_chat_request_data,  # Use the fixture
        error_type, expected_status, error_message_detail
):
    mock_load_template.return_value = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_chat_api_call.side_effect = error_type

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    response = client.post(
        "/api/v1/chat/completions",
        json=default_chat_request_data.model_dump(),
        headers={"token": valid_auth_token}
    )
    assert response.status_code == expected_status
    response_detail = response.json().get("detail", "")
    assert error_message_detail.lower() in response_detail.lower()
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_non_iterable_stream_generator_from_shim(  # Added default_chat_request_data
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))
    mock_chat_api_call.return_value = "not_a_generator"

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
    response = client.post(
        "/api/v1/chat/completions",
        json=streaming_request_data.model_dump(),
        headers={"token": valid_auth_token}
    )
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "did not return an iterator" in response.json()["detail"]
    app.dependency_overrides = {}


@pytest.mark.unit
@patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template")
@patch("tldw_Server_API.app.api.v1.endpoints.chat.apply_template_to_string")
def test_error_within_stream_generator(  # Added default_chat_request_data
        mock_apply_template, mock_load_template, mock_chat_api_call,
        client, default_chat_request_data, valid_auth_token, mock_media_db, mock_chat_db
):
    mock_load_template.return_value = None  # Default passthrough
    mock_apply_template.side_effect = lambda template_str, data: data.get("message_content", data.get(
        "original_system_message_from_request", ""))

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db

    with patch("tldw_Server_API.app.api.v1.endpoints.chat.chat_api_call") as mock_chat_api_call_inner:
        def faulty_stream_generator():
            yield "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Good start...\"}}]}\n\n"
            raise ValueError("Something broke mid-stream!")

        mock_chat_api_call_inner.return_value = faulty_stream_generator()

        streaming_request_data = default_chat_request_data.model_copy(update={"stream": True})
        response = client.post(
            "/api/v1/chat/completions",
            json=streaming_request_data.model_dump(),
            headers={"token": valid_auth_token}
        )
        assert response.status_code == status.HTTP_200_OK

        lines = response.text.splitlines()
        assert "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Good start...\"}}]}" in lines[0]

        error_payload_found = False
        done_found = False
        for line in lines:
            if "data:" in line and "error" in line and "Something broke mid-stream!" in line:
                error_payload_found = True
            if "data: [DONE]" in line:
                done_found = True

        assert error_payload_found
        assert done_found
    app.dependency_overrides = {}