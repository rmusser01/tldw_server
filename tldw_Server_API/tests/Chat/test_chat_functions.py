# tests/unit/core/chat/test_chat_functions.py
import base64
import re

import pytest
from unittest.mock import patch, MagicMock, call
from typing import List, Dict, Any
import requests  # For mocking requests.exceptions

# Imports from your application
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    chat_api_call,
    chat,  # This is the multimodal chat coordinator
    save_chat_history_to_db_wrapper,  # Assuming you want to test this too
    API_CALL_HANDLERS,
    PROVIDER_PARAM_MAP, load_characters, save_character, ChatDictionary, parse_user_dict_markdown_file,
    # Import other functions you might want to unit test from Chat_Functions.py
    # e.g., process_user_input, parse_user_dict_markdown_file, etc.
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError,
    ChatConfigurationError, ChatProviderError, ChatAPIError
)
# Import Pydantic models from your schema file to construct valid `history` for the `chat` function tests
# Note: The `chat` function itself expects `history` as List[Dict[str, Any]],
# but for clarity in test setup, using Pydantic models first and then dumping is fine.
# However, the `chat` function's direct input type hint is List[Dict[str, Any]].
# For these unit tests, we'll construct the history as list of dicts directly.

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB  # For mocking


# Mock for load_and_log_configs as it's used in `chat` and `chat_api_call` (indirectly via LLM_API_Calls)
# It's better to mock it at the point of use within the functions being tested if it's too global.
# For now, let's assume it's used by the LLM call functions that are mocked out by mock_llm_api_call_handlers.
# If `chat` itself calls it directly, that patch needs to be in the `chat` function tests.

@pytest.fixture(autouse=True)  # Applied to all tests in this module
def mock_global_load_and_log_configs():
    """Mocks load_and_log_configs where it's used by Chat_Functions.py."""
    with patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs", return_value={
        "chat_dictionaries": {},
        # You might need to add more default keys here if Chat_Functions or its callees expect them
        # For example, if chat_api_call indirectly uses parts of the config via its LLM handlers:
        "openai_api": {"api_key": "mock_key"}, # Example
        "anthropic_api": {"api_key": "mock_key"}, # Example
        # Add other default structures your mocked LLM handlers might try to access
        # or ensure the LLM handlers are fully mocked not to need config.
    }) as mock_config:
        yield mock_config


@pytest.fixture
def mock_llm_api_call_handlers_for_chat_functions_unit():
    original_handlers = API_CALL_HANDLERS.copy()
    mocked_handlers_dict = {}
    for provider_name_key, original_func_ref in original_handlers.items():
        # Try to get the original function's name for the mock
        original_func_name = getattr(original_func_ref, '__name__', f"mock_{provider_name_key}_handler")

        # Create a MagicMock and explicitly set its __name__ attribute
        # and also pass it to the name argument of MagicMock for its repr.
        mock_handler = MagicMock(name=f"mock_for_{original_func_name}")
        mock_handler.__name__ = original_func_name  # Explicitly set it
        mocked_handlers_dict[provider_name_key] = mock_handler

    with patch("tldw_Server_API.app.core.Chat.Chat_Functions.API_CALL_HANDLERS", new=mocked_handlers_dict):
        yield mocked_handlers_dict


# --- Tests for chat_api_call ---
@pytest.mark.unit
def test_chat_api_call_routing_and_param_mapping_openai_unit(mock_llm_api_call_handlers_for_chat_functions_unit):
    provider = "openai"
    mock_openai_handler = mock_llm_api_call_handlers_for_chat_functions_unit[provider]
    mock_openai_handler.return_value = "OpenAI success"

    args = {
        "api_endpoint": provider,
        "messages_payload": [{"role": "user", "content": "Hi OpenAI"}],
        "api_key": "test_openai_key",
        "temp": 0.5,
        "system_message": "Be concise.",
        "streaming": False,
        "maxp": 0.9,
        "model": "gpt-4o-mini"
    }
    result = chat_api_call(**args)
    assert result == "OpenAI success"
    mock_openai_handler.assert_called_once()
    called_kwargs = mock_openai_handler.call_args.kwargs

    # Check that params were mapped correctly according to PROVIDER_PARAM_MAP['openai']
    param_map_for_provider = PROVIDER_PARAM_MAP[provider]
    expected_key = param_map_for_provider['messages_payload']
    print(
        f"Expected key from map for 'messages_payload': '{expected_key}' (type: {type(expected_key)})")  # Should be 'input_data'
    print(f"Actual keys in mock's called_kwargs: {list(called_kwargs.keys())}")

    assert expected_key in called_kwargs, f"Key '{expected_key}' not found in mock's called_kwargs"
    assert called_kwargs[expected_key] == args["messages_payload"]
    assert called_kwargs[param_map_for_provider['temp']] == args["temp"]
    assert called_kwargs[param_map_for_provider['system_message']] == args["system_message"]
    assert called_kwargs[param_map_for_provider['streaming']] == args["streaming"]
    assert called_kwargs[param_map_for_provider['maxp']] == args["maxp"]
    assert called_kwargs[param_map_for_provider['model']] == args["model"]


@pytest.mark.unit
def test_chat_api_call_routing_and_param_mapping_anthropic_unit(mock_llm_api_call_handlers_for_chat_functions_unit):
    provider = "anthropic"
    mock_anthropic_handler = mock_llm_api_call_handlers_for_chat_functions_unit[provider]
    mock_anthropic_handler.return_value = "Anthropic success"

    args = {
        "api_endpoint": provider,
        "messages_payload": [{"role": "user", "content": "Hi Anthropic"}],
        "api_key": "test_anthropic_key",
        "temp": 0.6,
        "system_message": "Be friendly.",  # This maps to 'system_prompt' for anthropic handler
        "streaming": True,
        "model": "claude-3",
        "topp": 0.92,
        "topk": 50
    }
    result = chat_api_call(**args)
    assert result == "Anthropic success"
    mock_anthropic_handler.assert_called_once()
    called_kwargs = mock_anthropic_handler.call_args.kwargs

    param_map_for_provider = PROVIDER_PARAM_MAP[provider]

    # Corrected assertions:
    # The key for param_map_for_provider is the GENERIC arg name from chat_api_call
    # The value is the PROVIDER-SPECIFIC arg name, which should be in called_kwargs

    # For 'api_key'
    provider_specific_api_key_name = param_map_for_provider['api_key']
    assert called_kwargs[provider_specific_api_key_name] == args["api_key"]

    # For 'messages_payload' -> 'input_data'
    provider_specific_messages_payload_name = param_map_for_provider['messages_payload']  # This will be 'input_data'
    assert called_kwargs[provider_specific_messages_payload_name] == args["messages_payload"]

    # For 'temp'
    provider_specific_temp_name = param_map_for_provider['temp']
    assert called_kwargs[provider_specific_temp_name] == args["temp"]

    # For 'system_message' -> 'system_prompt'
    provider_specific_system_message_name = param_map_for_provider['system_message']  # This will be 'system_prompt'
    assert called_kwargs[provider_specific_system_message_name] == args["system_message"]

    # For 'streaming'
    provider_specific_streaming_name = param_map_for_provider['streaming']
    assert called_kwargs[provider_specific_streaming_name] == args["streaming"]

    # For 'model'
    provider_specific_model_name = param_map_for_provider['model']
    assert called_kwargs[provider_specific_model_name] == args["model"]

    # For 'topp'
    provider_specific_topp_name = param_map_for_provider['topp']
    assert called_kwargs[provider_specific_topp_name] == args["topp"]

    # For 'topk'
    provider_specific_topk_name = param_map_for_provider['topk']
    assert called_kwargs[provider_specific_topk_name] == args["topk"]


@pytest.mark.unit
def test_chat_api_call_unsupported_provider_unit():
    with pytest.raises(ValueError, match="Unsupported API endpoint: non_existent_provider"):
        chat_api_call(api_endpoint="non_existent_provider", messages_payload=[])


@pytest.mark.unit
@pytest.mark.parametrize("raised_exception, expected_custom_error_type, expected_status_code_in_error", [
    (requests.exceptions.HTTPError(response=MagicMock(status_code=401, text="Auth error text")),
     ChatAuthenticationError, 401),
    (requests.exceptions.HTTPError(response=MagicMock(status_code=429, text="Rate limit text")), ChatRateLimitError,
     429),
    (requests.exceptions.HTTPError(response=MagicMock(status_code=400, text="Bad req text")), ChatBadRequestError, 400),
    # This should be the status_code on the error object
    (requests.exceptions.HTTPError(response=MagicMock(status_code=503, text="Provider down text")), ChatProviderError,
     503),
    (requests.exceptions.ConnectionError("Network fail"), ChatProviderError, 504),
    (ValueError("Internal value error"), ChatBadRequestError, 400),
    # Default status for bad request from value/type/key
    (TypeError("Internal type error"), ChatBadRequestError, 400),
    (KeyError("Internal key error"), ChatBadRequestError, 400),
    (Exception("Very generic error"), ChatAPIError, 500),
])
def test_chat_api_call_exception_propagation_and_mapping_unit(
        mock_llm_api_call_handlers_for_chat_functions_unit,
        raised_exception, expected_custom_error_type, expected_status_code_in_error
):
    provider = "openai"  # Use any mocked provider
    mock_handler = mock_llm_api_call_handlers_for_chat_functions_unit[provider]
    mock_handler.side_effect = raised_exception

    with pytest.raises(expected_custom_error_type) as exc_info:
        chat_api_call(api_endpoint=provider, messages_payload=[{"role": "user", "content": "test"}])

    assert exc_info.value.provider == provider
    # For ChatProviderError and ChatAPIError, status_code is an explicit attribute from the constructor
    if hasattr(exc_info.value, 'status_code') and exc_info.value.status_code is not None:
        assert exc_info.value.status_code == expected_status_code_in_error
    # For ChatBadRequestError from ValueError/TypeError/KeyError, it defaults to 400
    elif isinstance(exc_info.value, ChatBadRequestError) and \
            any(isinstance(raised_exception, E) for E in [ValueError, TypeError, KeyError]):
        # The endpoint might map these to 400, but the error itself might not store a status_code
        pass  # No specific status_code on these, but they are ChatBadRequestError

    # Check that original error message part is in the custom error
    if isinstance(raised_exception, requests.exceptions.HTTPError):
        assert "text" in exc_info.value.message.lower()  # Example check
    elif isinstance(raised_exception, requests.exceptions.RequestException):
        assert "network" in exc_info.value.message.lower()
    else:  # For ValueError, TypeError, KeyError, Exception
        assert str(raised_exception).lower() in exc_info.value.message.lower()


# --- Tests for the `chat` function (multimodal chat coordinator) ---

@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs")  # Patch it where `chat` uses it
def test_chat_function_basic_text_call_unit(
        mock_load_configs_chat, mock_process_input, mock_chat_api_call_shim
):
    mock_load_configs_chat.return_value = {"chat_dictionaries": {}}  # Config for chat dictionary part
    mock_process_input.side_effect = lambda text, *args, **kwargs: text  # Passthrough
    mock_chat_api_call_shim.return_value = "LLM Response from chat function"

    test_history_for_chat_func = []
    response = chat(
        message="Hello LLM",
        history=test_history_for_chat_func,
        media_content=None, selected_parts=[], api_endpoint="test_provider_for_chat",
        api_key="test_key_for_chat", custom_prompt="Be very brief.", temperature=0.1,
        system_message="You are a test bot for chat."
    )
    assert response == "LLM Response from chat function"
    mock_chat_api_call_shim.assert_called_once()
    call_args = mock_chat_api_call_shim.call_args.kwargs

    assert call_args["api_endpoint"] == "test_provider_for_chat"
    assert call_args["api_key"] == "test_key_for_chat"
    assert call_args["temp"] == 0.1
    assert call_args["system_message"] == "You are a test bot for chat."

    payload = call_args["messages_payload"]
    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    assert isinstance(payload[0]["content"], list)
    assert len(payload[0]["content"]) == 1
    assert payload[0]["content"][0]["type"] == "text"
    assert payload[0]["content"][0]["text"] == "Be very brief.\n\nHello LLM"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs", return_value={"chat_dictionaries": {}})
def test_chat_function_with_text_history_unit(mock_configs, mock_proc_input, mock_chat_shim):
    mock_chat_shim.return_value = "LLM Response with history"
    history_for_chat_func = [
        {"role": "user", "content": "Previous question?"},
        {"role": "assistant", "content": "Previous answer."}
    ]
    response = chat(
        message="New question", history=history_for_chat_func, media_content=None,
        selected_parts=[], api_endpoint="hist_provider", api_key="hist_key",
        custom_prompt=None, temperature=0.2, system_message="Sys History"
    )
    assert response == "LLM Response with history"
    mock_chat_shim.assert_called_once()
    payload = mock_chat_shim.call_args.kwargs["messages_payload"]
    assert len(payload) == 3
    # `chat` wraps string content in a list of text parts
    assert payload[0]["content"][0]["type"] == "text"
    assert payload[0]["content"][0]["text"] == "Previous question?"
    assert payload[1]["content"][0]["type"] == "text"
    assert payload[1]["content"][0]["text"] == "Previous answer."
    assert payload[2]["content"][0]["type"] == "text"
    assert payload[2]["content"][0]["text"] == "New question"
    assert mock_chat_shim.call_args.kwargs["system_message"] == "Sys History"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs", return_value={"chat_dictionaries": {}})
def test_chat_function_with_current_image_unit(mock_configs, mock_proc_input, mock_chat_shim):
    mock_chat_shim.return_value = "LLM image Response"
    current_image = {"base64_data": "fakeb64imagedata", "mime_type": "image/png"}

    response = chat(
        message="What is this image?", history=[], media_content=None, selected_parts=[],
        api_endpoint="img_provider", api_key="img_key", custom_prompt=None, temperature=0.3,
        current_image_input=current_image
    )
    assert response == "LLM image Response"
    mock_chat_shim.assert_called_once()
    payload = mock_chat_shim.call_args.kwargs["messages_payload"]
    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    user_content_parts = payload[0]["content"]
    assert isinstance(user_content_parts, list)
    assert {"type": "text", "text": "What is this image?"} in user_content_parts
    assert {"type": "image_url", "image_url": {"url": "data:image/png;base64,fakeb64imagedata"}} in user_content_parts


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs", return_value={"chat_dictionaries": {}})
def test_chat_function_image_history_tag_past_unit(mock_configs, mock_proc_input, mock_chat_shim):
    mock_chat_shim.return_value = "Tagged image history response"
    # History with multimodal content (list of parts)
    history_with_image = [
        {"role": "user", "content": [
            {"type": "text", "text": "Here is an image."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,previmgdata"}}
        ]},
        {"role": "assistant", "content": "I see the image."}  # Simple text content for assistant
    ]
    response = chat(
        message="What about that previous image?", history=history_with_image,
        media_content=None, selected_parts=[], api_endpoint="tag_provider",
        api_key="tag_key", custom_prompt=None, temperature=0.4,
        image_history_mode="tag_past"
    )
    mock_chat_shim.assert_called_once()
    payload = mock_chat_shim.call_args.kwargs["messages_payload"]

    assert len(payload) == 3
    user_hist_content = payload[0]["content"]  # This should be a list of dicts
    assert isinstance(user_hist_content, list)
    assert len(user_hist_content) == 2  # Original text part + generated tag part
    assert {"type": "text", "text": "Here is an image."} in user_hist_content
    assert {"type": "text", "text": "<image: prior_history.jpeg>"} in user_hist_content

    assistant_hist_content = payload[1]["content"]
    assert assistant_hist_content[0]["text"] == "I see the image."  # Will be wrapped in list by `chat`

    current_user_content = payload[2]["content"]
    assert current_user_content[0]["text"] == "What about that previous image?"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs")
def test_chat_function_streaming_passthrough(mock_load_configs, mock_process_input, mock_chat_api_call_shim):
    mock_load_configs.return_value = {"chat_dictionaries": {}}
    mock_process_input.side_effect = lambda text, *args, **kwargs: text

    # Simulate chat_api_call returning a generator for streaming
    def dummy_stream_gen():
        yield "stream chunk 1"
        yield "stream chunk 2"

    mock_chat_api_call_shim.return_value = dummy_stream_gen()

    response_gen = chat(
        message="Stream this", history=[], media_content=None, selected_parts=[],
        api_endpoint="stream_provider", api_key="stream_key", custom_prompt=None,
        temperature=0.1, system_message="Sys Stream", streaming=True  # Enable streaming
    )
    assert hasattr(response_gen, '__iter__'), "Response should be a generator for streaming"
    # Consume the generator
    result = list(response_gen)
    assert result == ["stream chunk 1", "stream chunk 2"]
    mock_chat_api_call_shim.assert_called_once()
    assert mock_chat_api_call_shim.call_args.kwargs["streaming"] is True


# --- Tests for save_chat_history_to_db_wrapper ---
@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.DEFAULT_CHARACTER_NAME", "TestDefaultChar")
def test_save_chat_history_new_conversation_default_char():
    mock_db = MagicMock(spec=CharactersRAGDB)
    mock_db.client_id = "unit_test_client"
    mock_db.get_character_card_by_name.return_value = {"id": 99, "name": "TestDefaultChar"}  # For default char lookup
    mock_db.add_conversation.return_value = "new_conv_id_123"  # Simulate new conversation ID
    mock_db.add_message.return_value = None  # add_message doesn't typically return a value

    history_to_save = [
        {"role": "user", "content": "Hello, default character!"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello, user!"}, {"type": "image_url", "image_url": {
            "url": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"}}]}
    ]

    # Use a context manager for the transaction if your DB class supports it that way
    mock_db.transaction.return_value.__enter__.return_value = None  # for 'with db.transaction():'

    conv_id, message = save_chat_history_to_db_wrapper(
        db=mock_db,
        chatbot_history=history_to_save,
        conversation_id=None,  # New conversation
        media_content_for_char_assoc=None,  # No specific media
        character_name_for_chat=None  # Should use default
    )

    assert conv_id == "new_conv_id_123"
    assert "success" in message.lower()
    mock_db.get_character_card_by_name.assert_called_once_with("TestDefaultChar")
    mock_db.add_conversation.assert_called_once()
    # add_conversation call details
    add_conv_args = mock_db.add_conversation.call_args.args[0]
    assert add_conv_args["character_id"] == 99
    assert "Chat with TestDefaultChar" in add_conv_args["title"]

    assert mock_db.add_message.call_count == 2  # User and Assistant messages
    # Check arguments of add_message calls
    first_message_call_args = mock_db.add_message.call_args_list[0].args[0]
    assert first_message_call_args["conversation_id"] == "new_conv_id_123"
    assert first_message_call_args["sender"] == "user"
    assert first_message_call_args["content"] == "Hello, default character!"
    assert first_message_call_args["image_data"] is None

    second_message_call_args = mock_db.add_message.call_args_list[1].args[0]
    assert second_message_call_args["sender"] == "assistant"
    assert second_message_call_args["content"] == "Hello, user!"  # Text part
    assert second_message_call_args["image_mime_type"] == "image/gif"
    assert isinstance(second_message_call_args["image_data"], bytes)  # Check it's decoded


@pytest.mark.unit
def test_save_chat_history_resave_conversation_specific_char():
    mock_db = MagicMock(spec=CharactersRAGDB)
    mock_db.client_id = "unit_test_client_resave"
    existing_conv_id = "existing_conv_456"
    char_id_for_resave = 77
    char_name_for_resave = "SpecificResaveChar"

    mock_db.get_character_card_by_name.return_value = {"id": char_id_for_resave, "name": char_name_for_resave,
                                                       "version": 1}
    mock_db.get_conversation_by_id.return_value = {"id": existing_conv_id, "character_id": char_id_for_resave,
                                                   "title": "Old Title", "version": 2}
    mock_db.get_messages_for_conversation.return_value = [  # Simulate some old messages
        {"id": "msg1", "version": 1}, {"id": "msg2", "version": 1}
    ]
    mock_db.soft_delete_message.return_value = None
    mock_db.add_message.return_value = None
    mock_db.update_conversation.return_value = None

    history_to_resave = [
        {"role": "user", "content": "Updated question for resave."}
    ]
    mock_db.transaction.return_value.__enter__.return_value = None

    conv_id, message = save_chat_history_to_db_wrapper(
        db=mock_db,
        chatbot_history=history_to_resave,
        conversation_id=existing_conv_id,
        media_content_for_char_assoc=None,
        character_name_for_chat=char_name_for_resave
    )

    assert conv_id == existing_conv_id
    assert "success" in message.lower()
    mock_db.get_character_card_by_name.assert_called_once_with(char_name_for_resave)
    mock_db.get_conversation_by_id.assert_any_call(existing_conv_id)  # Called multiple times
    mock_db.get_messages_for_conversation.assert_called_once_with(existing_conv_id, limit=10000,
                                                                  order_by_timestamp="ASC")
    assert mock_db.soft_delete_message.call_count == 2
    mock_db.add_message.assert_called_once()
    add_msg_args = mock_db.add_message.call_args.args[0]
    assert add_msg_args["content"] == "Updated question for resave."
    mock_db.update_conversation.assert_called_once()  # Called to bump version/timestamp


@pytest.mark.unit
def test_chat_api_call_provider_specific_params_unit(mock_llm_api_call_handlers_for_chat_functions_unit):
    provider_name = "openrouter"  # Example of a provider with minp, topk, topp
    mock_handler = mock_llm_api_call_handlers_for_chat_functions_unit[provider_name]
    mock_handler.return_value = "OpenRouter success"

    args = {
        "api_endpoint": provider_name,
        "messages_payload": [{"role": "user", "content": "Test OpenRouter"}],
        "api_key": "or_key",
        "temp": 0.7,
        "model": "some/model",
        "minp": 0.1,
        "topk": 40,
        "topp": 0.92  # Note: PROVIDER_PARAM_MAP maps 'topp' to 'top_p' for openrouter
    }
    chat_api_call(**args)
    mock_handler.assert_called_once()
    called_kwargs = mock_handler.call_args.kwargs

    param_map = PROVIDER_PARAM_MAP[provider_name]
    assert called_kwargs[param_map['minp']] == args['minp']
    assert called_kwargs[param_map['topk']] == args['topk']
    assert called_kwargs[param_map['topp']] == args['topp']  # This will be 'top_p'


@pytest.mark.unit
def test_chat_api_call_tools_and_tool_choice_unit(mock_llm_api_call_handlers_for_chat_functions_unit):
    provider = "openai"  # Assuming OpenAI handler is adapted for tools
    mock_handler = mock_llm_api_call_handlers_for_chat_functions_unit[provider]
    mock_handler.return_value = {"id": "tool_response"}

    tools_payload = [{"type": "function", "function": {"name": "get_weather"}}]
    args = {
        "api_endpoint": provider,
        "messages_payload": [{"role": "user", "content": "What's the weather?"}],
        "api_key": "key", "model": "gpt-4o-mini",
        "tools": tools_payload,
        "tool_choice": "auto"
    }
    chat_api_call(**args)
    mock_handler.assert_called_once()
    called_kwargs = mock_handler.call_args.kwargs
    # Assuming the openai handler function ('chat_with_openai') directly accepts 'tools' and 'tool_choice'
    # and PROVIDER_PARAM_MAP for openai would need to include these if names differ.
    # If they don't differ, these params would be passed through if they are in available_generic_params
    # and not None.
    # For this test, let's assume the handler receives them directly.
    # The current PROVIDER_PARAM_MAP for openai does not list 'tools' or 'tool_choice'.
    # This means the specific chat_with_openai function needs to accept **kwargs or these specific args.
    # For a more robust test, ensure the mock_handler is called with these or that your
    # PROVIDER_PARAM_MAP is updated if the internal handler uses different names.

    # Simplified assertion: check if tools and tool_choice are present in the call if the handler accepts them
    # This part of the test depends on how `chat_with_openai` is implemented to receive these.
    # If it uses **kwargs:
    assert "tools" in called_kwargs  # This will FAIL if not explicitly mapped or handled by **kwargs in chat_with_openai
    assert "tool_choice" in called_kwargs  # This will FAIL
    assert called_kwargs["tools"] == tools_payload
    assert called_kwargs["tool_choice"] == "auto"


# --- New Tests for chat function (multimodal coordinator) ---

@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
# mock_global_load_and_log_configs is already active via autouse=True
def test_chat_function_image_history_send_all_unit(mock_process_input, mock_chat_api_call_shim):
    mock_chat_api_call_shim.return_value = "Response"
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "First image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}}
        ]},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": [
            {"type": "text", "text": "Second image"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,img2"}}
        ]}
    ]
    chat(message="What about all images?", history=history, media_content=None, selected_parts=[],
         api_endpoint="test", api_key="key", custom_prompt=None, temperature=0.1,
         image_history_mode="send_all")

    mock_chat_api_call_shim.assert_called_once()
    payload = mock_chat_api_call_shim.call_args.kwargs["messages_payload"]
    assert len(payload) == 4  # 3 history messages + 1 current user message
    assert payload[0]["content"][1]["type"] == "image_url"
    assert payload[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,img1"
    assert payload[2]["content"][1]["type"] == "image_url"
    assert payload[2]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,img2"


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
def test_chat_function_image_history_send_last_user_image_unit(mock_process_input, mock_chat_api_call_shim):
    mock_chat_api_call_shim.return_value = "Response"
    history = [
        {"role": "user", "content": [  # This image should be ignored
            {"type": "text", "text": "Old image"},
            {"type": "image_url", "image_url": {"url": "data:image/gif;base64,old_img"}}
        ]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": [  # This is the last user image
            {"type": "text", "text": "Recent image"},
            {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,recent_img"}}
        ]},
        {"role": "assistant", "content": "Acknowledged."}
    ]
    chat(message="About the last one?", history=history, media_content=None, selected_parts=[],
         api_endpoint="test", api_key="key", custom_prompt=None, temperature=0.1,
         image_history_mode="send_last_user_image")

    mock_chat_api_call_shim.assert_called_once()
    payload = mock_chat_api_call_shim.call_args.kwargs["messages_payload"]
    # Expected: Old image text, assistant, recent image user message (with its image), assistant, current user message
    assert len(payload) == 5
    # The user message that contained "recent_img" should have it.
    # Check the message at index 2 (0-indexed)
    assert {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,recent_img"}} in payload[2]["content"]
    # Ensure the "old_img" is not present in the first user message if it wasn't originally multi-part
    # The current logic for "send_last_user_image" appends to the *last user message in the processed history*.
    # It doesn't remove other images if they were part of "send_all" style history input.
    # This test specifically ensures the "recent_img" is the one carried forward implicitly if mode is send_last_user_image.
    # And that "old_img" isn't *added again* to the first user message.
    first_user_msg_content = payload[0]["content"]
    assert not any(
        p.get("image_url", {}).get("url") == "data:image/gif;base64,old_img" for p in first_user_msg_content if
        p["type"] == "image_url")


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.process_user_input", side_effect=lambda x, *a, **kw: x)
def test_chat_function_with_rag_content_unit(mock_process_input, mock_chat_api_call_shim):
    mock_chat_api_call_shim.return_value = "RAG Response"
    media_content = {"summary": "This is a summary.", "content": "Full content here."}
    selected_parts = ["summary", "content"]

    chat(message="What about this?", history=[], media_content=media_content, selected_parts=selected_parts,
         api_endpoint="test_rag", api_key="key_rag", custom_prompt=None, temperature=0.1)

    mock_chat_api_call_shim.assert_called_once()
    payload = mock_chat_api_call_shim.call_args.kwargs["messages_payload"]
    user_message_text = payload[0]["content"][0]["text"]
    assert "Summary: This is a summary." in user_message_text
    assert "Content: Full content here." in user_message_text
    assert "\n\n---\n\nWhat about this?" in user_message_text


# --- New Tests for Chat Dictionary processing ---

@pytest.mark.unit
def test_parse_user_dict_markdown_file_various_formats(tmp_path):
    md_content = """
    key1: value1
    key2: |
        This is a
        multi-line value for key2.
        It has several lines.
    # ...
    """
    dict_file = tmp_path / "test_dict.md"
    dict_file.write_text(md_content)

    parsed = parse_user_dict_markdown_file(str(dict_file))
    expected_key2_value = "This is a\nmulti-line value for key2.\nIt has several lines.\n"  # Adjusted expectation
    assert parsed["key1"] == "value1"
    assert parsed["key2"] == expected_key2_value


@pytest.mark.unit
def test_chat_dictionary_class_methods():
    entry_plain = ChatDictionary(key="hello", content="hi there")
    entry_regex = ChatDictionary(key=r"/\bworld\b/", content="planet")  # Python re.IGNORECASE handles case

    assert entry_plain.matches("hello world")
    assert not entry_plain.matches("goodbye")
    assert isinstance(entry_plain.key, str)

    assert entry_regex.matches("Hello World!")
    assert entry_regex.matches("new world order")
    assert not entry_regex.matches("worldwide")
    assert isinstance(entry_regex.key, re.Pattern)


@pytest.mark.unit
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.load_and_log_configs")
@patch("tldw_Server_API.app.core.Chat.Chat_Functions.parse_user_dict_markdown_file")
def test_chat_function_with_chat_dictionary_post_replacement(
        mock_parse_dict, mock_load_configs_chat_func, tmp_path
):
    # Mock config for post-gen replacement
    mock_config_data = {
        "chat_dictionaries": {
            "post_gen_replacement": "True",  # String "True"
            "post_gen_replacement_dict": str(tmp_path / "post_gen.md")
        }
    }
    mock_load_configs_chat_func.return_value = mock_config_data

    # Create a dummy post_gen.md file
    post_gen_dict_file = tmp_path / "post_gen.md"
    post_gen_dict_file.write_text("AI: Artificial Intelligence\nLLM: Large Language Model")

    # Mock the return of parse_user_dict_markdown_file
    mock_parse_dict.return_value = {
        "AI": "Artificial Intelligence",
        "LLM": "Large Language Model"
    }

    # Mock chat_api_call (the one inside Chat_Functions, not the one in the endpoint)
    with patch("tldw_Server_API.app.core.Chat.Chat_Functions.chat_api_call") as mock_chat_api_call_inner:
        raw_llm_response = "The AI assistant uses an LLM."
        mock_chat_api_call_inner.return_value = raw_llm_response

        # Call the main chat function
        final_response = chat(
            message="Tell me about AI.",
            history=[],
            media_content=None,
            selected_parts=[],
            api_endpoint="openai",
            api_key="testkey",
            custom_prompt=None,
            temperature=0.7,
            system_message=None,
            streaming=False  # Important for this test case
        )

        expected_response = "The Artificial Intelligence assistant uses an Large Language Model."
        assert final_response == expected_response
        mock_parse_dict.assert_called_once_with(str(post_gen_dict_file))
        mock_chat_api_call_inner.assert_called_once()


# --- New Tests for save_character and load_characters ---
@pytest.mark.unit
def test_save_character_new_and_update_unit():
    mock_db = MagicMock(spec=CharactersRAGDB)
    mock_db.client_id = "char_test_client"

    char_data_v1 = {
        "name": "TestCharacter", "description": "A brave hero.", "system_prompt": "Be heroic.",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        # 1x1 red pixel
    }
    char_data_v2_update = {
        "name": "TestCharacter", "description": "An even braver hero.", "personality": "Bold"
    }

    # Scenario 1: Add new character
    mock_db.get_character_card_by_name.return_value = None  # Character does not exist
    mock_db.add_character_card.return_value = 1  # Simulate new character ID

    returned_id_v1 = save_character(db=mock_db, character_data=char_data_v1)
    assert returned_id_v1 == 1
    mock_db.add_character_card.assert_called_once()
    add_args = mock_db.add_character_card.call_args[0][0]
    assert add_args["name"] == "TestCharacter"
    assert add_args["description"] == "A brave hero."
    assert isinstance(add_args["image"], bytes)

    # Scenario 2: Update existing character
    mock_db.reset_mock()
    existing_char_from_db = {
        "id": 1, "name": "TestCharacter", "description": "A brave hero.", "system_prompt": "Be heroic.",
        "image": b"decoded_image_bytes", "version": 1, "personality": None
    }
    mock_db.get_character_card_by_name.return_value = existing_char_from_db
    mock_db.update_character_card.return_value = True  # Simulate successful update

    returned_id_v2 = save_character(db=mock_db, character_data=char_data_v2_update, expected_version=1)
    assert returned_id_v2 == 1
    mock_db.update_character_card.assert_called_once()
    update_args = mock_db.update_character_card.call_args[0]  # (char_id, data_to_update, version)
    assert update_args[0] == 1  # char_id
    assert update_args[1]["description"] == "An even braver hero."
    assert update_args[1]["personality"] == "Bold"
    assert "system_prompt" not in update_args[1]  # Should not be in update_payload as it wasn't in char_data_v2_update
    assert "image" not in update_args[1]  # Image wasn't in char_data_v2_update
    assert update_args[2] == 1  # expected_version


@pytest.mark.unit
def test_load_characters_empty_and_with_data_unit():
    mock_db = MagicMock(spec=CharactersRAGDB)

    # Scenario 1: No characters
    mock_db.list_character_cards.return_value = []
    chars = load_characters(db=mock_db)
    assert chars == {}

    # Scenario 2: Characters with data (including image for encoding)
    mock_db.reset_mock()
    db_cards_list = [
        {"id": 1, "name": "Hero", "description": "Good guy", "image": b"heroimagebytes"},
        {"id": 2, "name": "Villain", "description": "Bad guy", "image": None}
    ]
    mock_db.list_character_cards.return_value = db_cards_list

    loaded_chars_map = load_characters(db=mock_db)
    assert len(loaded_chars_map) == 2
    assert "Hero" in loaded_chars_map
    assert "Villain" in loaded_chars_map
    assert loaded_chars_map["Hero"]["description"] == "Good guy"
    assert loaded_chars_map["Hero"]["image_base64"] == base64.b64encode(b"heroimagebytes").decode('utf-8')
    assert loaded_chars_map["Villain"].get("image_base64") is None

# Add more tests for save_chat_history_to_db_wrapper:
# - Character not found (specific and default)
# - Conversation not found for resave
# - Character ID mismatch on resave
# - DB errors during operations (add_conversation, add_message, delete, update)
# - History with only system messages (should save no messages)
# - History with malformed image data URI
