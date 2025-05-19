# tests/integration/api/v1/test_chat_completions_integration.py
import pytest
import os
import json
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError

# Load environment variables from .env if you use one for test configurations
load_dotenv()

# Import your FastAPI app instance
from tldw_Server_API.app.main import app

# Import your actual schema definitions for constructing test requests
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    # If you test with multimodal content in integration tests:
    # ChatCompletionRequestMessageContentPartText,
    # ChatCompletionRequestMessageContentPartImage,
    # ChatCompletionRequestMessageContentPartImageURL
)
# For mocking DB dependencies if your endpoint uses them
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate  # For templating test


# --- Fixtures defined locally in this file ---
API_BEARER="default-secret-key-for-single-user"
@pytest.fixture(scope="function")
def client():
    """Yields a TestClient instance for making requests to the app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_auth_token() -> str:
    # This MUST match the value of os.getenv("API_BEARER") in the app's runtime environment.
    token = "default-secret-key-for-single-user"
    if os.getenv("API_BEARER") != token:
        pytest.fail(f"MISMATCH: API_BEARER is '{os.getenv('API_BEARER')}' but test token is '{token}'")
    return token


# --- Provider lists and helpers defined locally for this test file ---
try:
    # These are used to determine which providers are configured and have keys.
    # The actual APP_API_KEYS should be loaded by your application's schema/config logic.
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import API_KEYS as APP_API_KEYS_FROM_SCHEMA
    from tldw_Server_API.app.core.Chat.Chat_Functions import API_CALL_HANDLERS as APP_API_CALL_HANDLERS

    ALL_CONFIGURED_PROVIDERS_FROM_APP = list(APP_API_CALL_HANDLERS.keys())
except ImportError:
    APP_API_KEYS_FROM_SCHEMA = {}
    ALL_CONFIGURED_PROVIDERS_FROM_APP = []
    print(
        "Warning: Could not import APP_API_KEYS_FROM_SCHEMA or APP_API_CALL_HANDLERS for integration test parametrization.")


def get_commercial_providers_with_keys_integration():
    """
    Returns a list of commercial providers for which API keys are actually set
    and non-empty, as understood by the application's schema.
    """
    potentially_commercial = [
        "openai", "anthropic", "cohere", "groq", "openrouter",
        "deepseek", "mistral", "google", "huggingface"
        # Add others that are external and need keys from your config
    ]
    # Check against keys actually loaded by the app's config (via schemas)
    return [
        p for p in potentially_commercial
        if p in ALL_CONFIGURED_PROVIDERS_FROM_APP and APP_API_KEYS_FROM_SCHEMA.get(p)
    ]


def get_local_providers_integration():
    """
    Returns a list of providers considered "local" or that might use URLs/config
    instead of globally managed API keys.
    """
    local_provider_names = [
        "llama.cpp", "kobold", "ooba", "tabbyapi", "vllm",
        "local-llm", "ollama", "aphrodite",
        "custom-openai-api", "custom-openai-api-2"
        # Add other local providers from your config
    ]
    return [p for p in local_provider_names if p in ALL_CONFIGURED_PROVIDERS_FROM_APP]


# Test data using your actual Pydantic schema models
INTEGRATION_MESSAGES_NO_SYS_SCHEMA = [
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply in one sentence.")
]
INTEGRATION_MESSAGES_WITH_SYS_SCHEMA = [
    ChatCompletionSystemMessageParam(role="system",
                                     content="You are Albert Einstein. Explain things from your perspective, but keep it brief."),
    ChatCompletionUserMessageParam(role="user", content="Explain the theory of relativity simply.")
]
STREAM_INTEGRATION_MESSAGES_SCHEMA = [  # For streaming tests
    ChatCompletionUserMessageParam(role="user", content="Stream a very short poem about a star. Max 3 lines.")
]

COMMERCIAL_PROVIDERS_FOR_TEST = get_commercial_providers_with_keys_integration()


# Fixture to mock DB dependencies for integration tests if the endpoint uses them
# In tldw_Server_API/tests/Chat/test_chat_completions_integration.py
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME # Import

@pytest.fixture
def mock_db_dependencies_for_integration():
    mock_media_db_inst = MagicMock()
    mock_chat_db_inst = MagicMock(spec=CharactersRAGDB)

    # --- Configure mock_chat_db_inst ---
    # For default character loading by name:
    default_char_card_data = {
        'id': 'default_integration_char_id', # Or an int if appropriate
        'name': DEFAULT_CHARACTER_NAME,
        'system_prompt': 'This is a mock default system prompt for integration tests.',
        # Add other fields the endpoint might access from the default character
        # e.g., client_id if it's directly on the card (though less likely)
    }
    def mock_get_character_card_by_name(name_or_id):
        if name_or_id == DEFAULT_CHARACTER_NAME:
            return default_char_card_data
        return None # For any other name

    mock_chat_db_inst.get_character_card_by_name.side_effect = mock_get_character_card_by_name
    mock_chat_db_inst.get_character_card_by_id.return_value = None # Keep this for by_id lookups unless a test needs it

    # For new conversation creation
    mock_chat_db_inst.add_conversation.return_value = "integration_mock_conv_id"

    # For saving messages
    mock_chat_db_inst.add_message.return_value = "integration_mock_msg_id"

    # <<< Crucial: Add client_id for integration tests too! >>>
    mock_chat_db_inst.client_id = "test_client_integration"
    # <<< Crucial: Ensure 'get_conversation_by_id' returns something sensible or None based on test needs >>>
    # For cases where a conversation ID *is* provided in the request and is expected to exist or not.
    # For these "no template" tests, usually no conv_id is provided initially.
    mock_chat_db_inst.get_conversation_by_id.return_value = None # Default to not found

    # --- Dependency Override ---
    original_media_db_dep = app.dependency_overrides.get(get_media_db_for_user)
    original_chat_db_dep = app.dependency_overrides.get(get_chacha_db_for_user)

    app.dependency_overrides[get_media_db_for_user] = lambda: mock_media_db_inst
    app.dependency_overrides[get_chacha_db_for_user] = lambda: mock_chat_db_inst

    yield mock_media_db_inst, mock_chat_db_inst

    # --- Restore ---
    if original_media_db_dep:
        app.dependency_overrides[get_media_db_for_user] = original_media_db_dep
    elif get_media_db_for_user in app.dependency_overrides:
        del app.dependency_overrides[get_media_db_for_user]

    if original_chat_db_dep:
        app.dependency_overrides[get_chacha_db_for_user] = original_chat_db_dep
    elif get_chacha_db_for_user in app.dependency_overrides:
        del app.dependency_overrides[get_chacha_db_for_user]


# --- Commercial Provider Tests ---
@pytest.mark.external_api  # Custom marker (register in pytest.ini or pyproject.toml)
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", COMMERCIAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(not COMMERCIAL_PROVIDERS_FOR_TEST,
                    reason="No commercial providers with API keys configured for integration tests.")
def test_commercial_provider_non_streaming_no_template(
        client, provider_name, valid_auth_token, mock_db_dependencies_for_integration
):
    # This test uses the DEFAULT_RAW_PASSTHROUGH_TEMPLATE because prompt_template_name is None

    model_map = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "cohere": "command-r",
        "groq": "llama3-8b-8192",
        "openrouter": "mistralai/mistral-7b-instruct:free",  # A known free model
        "deepseek": "deepseek-chat",
        "mistral": "mistral-tiny",  # Or mistral-small for better results
        "google": "gemini-1.5-flash-latest",  # Cost-effective Gemini
        "huggingface": os.getenv("HF_TEST_MODEL", "Qwen/Qwen3-235B-A22B")  # Ensure HF_TOKEN is set
    }
    default_test_model = "test-model-default"  # Fallback, though ideally map should cover all in COMMERCIAL_PROVIDERS_FOR_TEST

    request_body = {
        "api_provider": provider_name,
        "model": model_map.get(provider_name, default_test_model),
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
        "temperature": 0.7,
        "stream": False,
        "prompt_template_name": None  # Explicitly no template, should use default passthrough
    }
    if provider_name == "anthropic":  # Anthropic (Claude) often requires max_tokens
        request_body["max_tokens"] = 200  # Adjusted for potentially longer explanations

    print(f"\nTesting NON-STREAMING (no template) with {provider_name} using model {request_body['model']}")
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK, f"Provider {provider_name} failed: {response.text}"
    data = response.json()

    assert data.get("id") is not None, f"Missing 'id' for {provider_name}"
    assert data.get("choices") and isinstance(data["choices"], list) and len(
        data["choices"]) > 0, f"Missing 'choices' for {provider_name}"
    message = data["choices"][0].get("message", {})
    assert message.get("role") == "assistant", f"Incorrect role for {provider_name}"
    content = message.get("content")
    assert isinstance(content, str) and len(content) > 5, f"Invalid content for {provider_name}: '{content}'"
    print(f"Response from {provider_name} (no template): {content[:80]}...")


@pytest.mark.external_api
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", COMMERCIAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(not COMMERCIAL_PROVIDERS_FOR_TEST,
                    reason="No commercial providers with API keys configured for streaming tests.")
def test_commercial_provider_streaming_no_template(
        client, provider_name, valid_auth_token, mock_db_dependencies_for_integration
):
    model_map = {  # Same as non-streaming
        "openai": "gpt-4o-mini", "anthropic": "claude-3-haiku-20240307", "cohere": "command-r",
        "groq": "llama3-8b-8192", "openrouter": "mistralai/mistral-7b-instruct:free",
        "deepseek": "deepseek-chat", "mistral": "mistral-tiny", "google": "gemini-1.5-flash-latest",
        "huggingface": os.getenv("HF_TEST_MODEL", "Qwen/Qwen3-235B-A22B")
    }
    default_test_model = "test-model-default-stream"

    request_body = {
        "api_provider": provider_name,
        "model": model_map.get(provider_name, default_test_model),
        "messages": [msg.model_dump(exclude_none=True) for msg in STREAM_INTEGRATION_MESSAGES_SCHEMA],
        "temperature": 0.7,
        "stream": True,
        "prompt_template_name": None
    }
    if provider_name == "anthropic": request_body["max_tokens"] = 300

    print(f"\nTesting STREAMING (no template) with {provider_name} using model {request_body['model']}")
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK, f"Provider {provider_name} streaming pre-check failed: {response.text}"
    assert 'text/event-stream' in response.headers['content-type'].lower()

    full_content = ""
    received_done = False
    raw_stream_text_for_debug = ""
    try:
        for line in response.iter_lines():
            raw_stream_text_for_debug += line + "\n"
            if line.startswith("data:") and "[DONE]" in line:
                received_done = True;
                break
            if line.startswith("data:"):
                try:
                    chunk_data_str = line[len("data:"):].strip()
                    if not chunk_data_str: continue
                    chunk = json.loads(chunk_data_str)

                    if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                        received_done = True
                        break  # Or continue if other final events might follow

                    delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta_content: full_content += delta_content
                except json.JSONDecodeError:
                    print(f"WARN: ({provider_name}) Test JSON decode error for line: '{line}' in stream.")
    except Exception as e:
        print(f"Raw stream for {provider_name} before error:\n{raw_stream_text_for_debug}")
        pytest.fail(f"Error consuming stream for {provider_name}: {e}")

    assert received_done, f"Stream for {provider_name} did not finish correctly. Last 500 chars:\n{raw_stream_text_for_debug[-500:]}"
    assert len(
        full_content) > 5, f"Streamed content for {provider_name} was too short or empty. Received: '{full_content}'"
    print(f"Streamed response from {provider_name} (no template): {full_content[:80]}...")


# --- Integration Test with Templating (Simplified - Mocks DB interaction) ---
@pytest.mark.external_api
@pytest.mark.integration
def test_commercial_provider_with_template_and_char_data_openai_integration(
        client, valid_auth_token, mock_db_dependencies_for_integration
):
    provider_name = "openai"
    if provider_name not in COMMERCIAL_PROVIDERS_FOR_TEST:
        pytest.skip(f"{provider_name} not configured with API key for this templating test.")

    # Unpack the chat_db mock to configure its return values
    _, mock_chat_db_inst = mock_db_dependencies_for_integration

    test_template_name = "pirate_speak_template"
    # This template would normally be loaded from a file by `load_template`
    # For this test, we mock `load_template` to return this specific PromptTemplate object.
    test_pirate_template_obj = PromptTemplate(
        name=test_template_name,
        system_message_template="Ye be talkin' to Cap'n {char_name}! {character_system_prompt}. The request mentioned: {original_system_message_from_request}",
        user_message_content_template="Arr, the landlubber {char_name} be askin': {message_content}",
        assistant_message_content_template=None  # Not used in request path
    )
    test_char_id_for_template = "pirate_blackheart"
    mock_character_data_for_template = {
        "id": test_char_id_for_template, "name": "Blackheart",
        "system_prompt": "I only speak in pirate tongues and seek treasure!",
        "personality": "Gruff but fair", "description": "A legendary pirate captain.",
        # Add other fields your template might use
    }
    # Configure the mock_chat_db_inst that the endpoint will use
    mock_chat_db_inst.get_character_card_by_id.return_value = None  # Or specific if ID is int

    # Ensure get_character_card_by_name returns the pirate when called with "pirate_blackheart"
    def specific_char_by_name_lookup(name_or_id):
        if name_or_id == test_char_id_for_template:  # "pirate_blackheart"
            return mock_character_data_for_template
        if name_or_id == DEFAULT_CHARACTER_NAME:  # Still handle default if needed elsewhere
            return {'id': 'default_id', 'name': DEFAULT_CHARACTER_NAME, 'system_prompt': 'Default'}
        return None

    mock_chat_db_inst.get_character_card_by_name.side_effect = specific_char_by_name_lookup

    # Patch `load_template` within the endpoint's module scope
    with patch("tldw_Server_API.app.api.v1.endpoints.chat.load_template", return_value=test_pirate_template_obj):
        request_body = {
            "api_provider": provider_name,
            "model": "gpt-4o-mini",  # A capable model for following instructions
            "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_WITH_SYS_SCHEMA],
            "prompt_template_name": test_template_name,
            "character_id": test_char_id_for_template,
            "temperature": 0.5,  # Give it some creativity
            "stream": False
        }

        print(f"\nTesting TEMPLATING with {provider_name} model {request_body['model']}")
        response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})

        assert response.status_code == status.HTTP_200_OK, f"{provider_name} with template failed: {response.text}"
        data = response.json()

        assert data.get("id") is not None
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str) and len(content) > 5
        # This is a loose check. A better check would be if you mocked chat_api_call
        # and verified the exact templated prompt, but this is an integration test for the LLM response.
        assert "arr" in content.lower() or "matey" in content.lower() or "treasure" in content.lower() or "cap'n" in content.lower(), \
            f"Response from {provider_name} with pirate template didn't sound pirate-y enough! Got: '{content}'"
        print(f"Templated response from {provider_name} (integration): {content[:100]}...")

        mock_chat_db_inst.get_character_card_by_name.assert_called_once_with(test_char_id_for_template)
        mock_chat_db_inst.get_character_card_by_id.assert_not_called()


# --- Local Provider Tests ---
LOCAL_PROVIDERS_FOR_TEST = get_local_providers_integration()


@pytest.mark.local_llm_service  # Custom marker
@pytest.mark.integration
@pytest.mark.parametrize("provider_name", LOCAL_PROVIDERS_FOR_TEST)
@pytest.mark.skipif(not LOCAL_PROVIDERS_FOR_TEST, reason="No local providers configured for integration tests.")
def test_local_provider_non_streaming_no_template(
        client, provider_name, valid_auth_token, mock_db_dependencies_for_integration, request
        # request is a pytest fixture
):
    # Configuration for local provider URLs (examples, adjust to your env var names)
    config_var_map = {
        "ollama": "OLLAMA_HOST", "llama.cpp": "LLAMA_CPP_URL", "ooba": "OOBA_URL",
        "vllm": "VLLM_URL", "tabbyapi": "TABBYAPI_URL",
        "local-llm": "LOCAL_LLM_API_IP", "aphrodite": "APHRODITE_API_IP",
        "kobold": "KOBOLD_API_IP",
        "custom-openai-api": "CUSTOM_OPENAI_API_IP_1",
        "custom-openai-api-2": "CUSTOM_OPENAI_API_IP_2"
    }
    required_env_var = config_var_map.get(provider_name)
    # Skip if the URL env var is not set AND the provider isn't one that might have its URL in APP_API_KEYS_FROM_SCHEMA (less common)
    if required_env_var and not os.getenv(required_env_var):
        # Some "local" providers might have their full URL defined in the API_KEYS/config if they are hosted
        # For truly local dev instances, checking the env var is more typical.
        # This skip logic might need refinement based on how you manage local endpoint URLs.
        is_url_in_app_config = APP_API_KEYS_FROM_SCHEMA.get(provider_name, "").startswith("http")
        if not is_url_in_app_config:
            pytest.skip(
                f"Environment variable like '{required_env_var}' for {provider_name} URL not set, and not found in app config. Skipping.")

    model_name = "test-local-model"  # Generic, may be ignored by server if model is pre-loaded
    if provider_name == "ollama": model_name = os.getenv("OLLAMA_TEST_MODEL", "phi3:mini")  # Example: use a small model
    if provider_name == "vllm": model_name = os.getenv("VLLM_TEST_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

    request_body = {
        "api_provider": provider_name,
        "model": model_name,
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA],
        "stream": False,
        "prompt_template_name": None  # Ensure default passthrough template
    }

    print(f"\nTesting LOCAL NON-STREAMING (no template) with {provider_name} using model {request_body['model']}")
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})

    assert response.status_code == status.HTTP_200_OK, f"Local provider {provider_name} failed: {response.text}"
    data = response.json()

    assert data.get("id") is not None, f"Missing 'id' for local {provider_name}"
    assert data.get("choices") and isinstance(data["choices"], list) and len(
        data["choices"]) > 0, f"Missing 'choices' for local {provider_name}"
    message = data["choices"][0].get("message", {})
    assert message.get("role") == "assistant", f"Incorrect role for local {provider_name}"
    content = message.get("content")
    # For local models, especially small ones, the content length can be very short.
    assert isinstance(content, str) and len(content) >= 1, f"Invalid content for local {provider_name}: '{content}'"
    print(f"Response from local {provider_name} (no template): {content[:80]}...")


# --- Invalid Key Test (for a commercial provider that needs a key) ---
@patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call") # Mock the shim
def test_chat_integration_invalid_key_for_commercial_provider_standalone(
    mock_chat_api_call_shim, client, valid_auth_token, mock_db_dependencies_for_integration
):
    provider_to_test_invalid_key = "openai"
    # Simulate that the call to the provider resulted in an auth error
    mock_chat_api_call_shim.side_effect = ChatAuthenticationError(
        provider=provider_to_test_invalid_key,
        message="Simulated auth error: Invalid API Key"
    )

    request_body = { # ... minimal request body ...
        "api_provider": provider_to_test_invalid_key,
        "model": "gpt-4o-mini",
        "messages": [msg.model_dump(exclude_none=True) for msg in INTEGRATION_MESSAGES_NO_SYS_SCHEMA]
    }
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})

    assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_400_BAD_REQUEST], \
        f"Expected 401/400 for invalid key with {provider_to_test_invalid_key}, got {response.status_code}. Response: {response.text}"

    detail = response.json().get("detail", "").lower()
    assert "authentication" in detail or "invalid" in detail or "key" in detail or "token" in detail, \
        f"Error detail for invalid key with {provider_to_test_invalid_key} did not match expected. Got: {detail}"
    print(f"\nInvalid Key Response Detail ({provider_to_test_invalid_key}): {response.json().get('detail')}")


# --- Bad Request Test (e.g., missing messages) ---
@pytest.mark.integration
def test_chat_integration_bad_request_missing_messages_standalone(
        client, valid_auth_token, mock_db_dependencies_for_integration  # Add DB mock for consistency
):
    request_body = {
        "api_provider": "openai",  # Could be any provider
        "model": "test-model",
        # "messages" field is intentionally missing
    }
    response = client.post("/api/v1/chat/completions", json=request_body, headers={"Token": valid_auth_token})
    # This is a Pydantic validation error from FastAPI itself before hitting your logic.
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    errors = response.json().get("detail")
    assert isinstance(errors, list)
    assert any("messages" in e.get("loc", []) and "field required" in e.get("msg", "").lower() for e in errors)