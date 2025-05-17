# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
# FIXME
#
# Imports
import asyncio
import base64
import datetime
import logging
import json
from functools import partial
from typing import List, Optional, Dict, Any, AsyncIterator, Iterator
#
# 3rd-party imports
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
    UploadFile
)
from fastapi.encoders import jsonable_encoder
from requests import RequestException, HTTPError
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.responses import JSONResponse, StreamingResponse
#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest, API_KEYS, \
    DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.Chat.Chat_Functions import (
        chat_api_call,
        ChatAuthenticationError,
        ChatRateLimitError,
        ChatBadRequestError,
        ChatConfigurationError,
        ChatProviderError,
        ChatAPIError,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import PromptTemplate, load_template, \
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE, apply_template_to_string
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError, \
    InputError

#
# DB Mgmt
#from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
#
#
#######################################################################################################################
#
# Functions:

# --- FastAPI Router and Endpoint ---
router = APIRouter()

@router.post(
    "/completions",
    summary="Creates a model response via a proxy to the specified or default LLM provider.",
    tags=["Chat"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request or error response from the backend LLM API"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error or backend API key missing"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid authentication token"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Error from upstream provider"},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Upstream provider timed out"},
    }
)
async def create_chat_completion(
    request_data: ChatCompletionRequest = Body(...),
    media_db = Depends(get_media_db_for_user), # Keep if used elsewhere, or remove if not relevant to chat
    chat_db: Optional[CharactersRAGDB] = Depends(get_chacha_db_for_user), # Make optional if chat_db might not always be available/needed
    Token: Optional[str] = Header(None),
):
    """
    Acts as a proxy to various chat completion APIs via the `chat_api_call` shim
    which now takes a `messages_payload` (OpenAI format).

    Accepts a JSON body similar to the OpenAI Chat Completion API request structure,
    with an added **`api_provider`** field to select the backend (e.g., 'openai', 'anthropic', 'ollama').
    If `api_provider` is omitted, the server's default provider is used.

    - **api_provider**: (Optional) Specify the backend.
    - **model**: ID of the model to use.
    - **messages**: Conversation history (OpenAI message object format).
    - **prompt_template_name**: (Optional) Name of the prompt template to apply.
    - **character_id**: (Optional, if added to ChatCompletionRequest) ID of the character for template data.
    - **Optional parameters**: Forwarded to the selected backend API via `chat_api_call`
      (e.g., `temperature`, `stream`, `minp`, `topk`, `top_p`).

    *(Refer to the ChatCompletionRequest schema for all parameters)*
    """
    # --- Determine Target API Provider ---
    target_endpoint = request_data.api_provider or DEFAULT_LLM_PROVIDER
    logger.info(f"Routing chat completion request for model '{request_data.model}' to provider: '{target_endpoint}'")
    logger.debug(
        f"Incoming request data (brief): model={request_data.model}, provider={target_endpoint}, stream={request_data.stream}, template='{request_data.prompt_template_name}', messages_count={len(request_data.messages)}"
    )
    # For more verbose debugging:
    # logger.trace(f"Incoming request data (full): {request_data.model_dump_json(exclude_none=True)}")

    # --- API Key Check ---
    provider_api_key = API_KEYS.get(target_endpoint.lower())
    providers_requiring_keys = [  # This list should ideally be more centrally managed or derived
        "openai", "anthropic", "cohere", "groq", "openrouter",
        "deepseek", "mistral", "google", "huggingface"
        # Add other providers that always require a key
    ]
    if target_endpoint in providers_requiring_keys and not provider_api_key:  # Checks for None or empty string
        logger.error(f"API key for required provider '{target_endpoint}' is missing or empty.")
        raise ChatConfigurationError(  # This will be caught by the except ChatConfigurationError block below
            provider=target_endpoint,
            message=f"API key is not configured or is empty for provider: {target_endpoint}."
        )

    # --- Character and Conversation Context ---
    character_data_for_template: Optional[Dict[str, Any]] = None
    associated_character_db_id: Optional[int] = None # Actual DB primary key for the character
    final_conversation_id_for_turn: Optional[str] = request_data.conversation_id
    conversation_created_now = False
    historical_openai_messages: List[Dict[str, Any]] = []

    if request_data.character_id and chat_db:
        try:
            # Assuming character_id from request is the name or a string ID that get_character_card_by_id can handle
            # If it's a name, use get_character_card_by_name
            # For this example, let's assume request_data.character_id is the DB primary key (int) or a unique name.
            # Adjust the lookup method as per your actual character ID strategy.
            # If character_id is a name: char_obj = chat_db.get_character_card_by_name(request_data.character_id)
            # If character_id is an int ID: char_obj = chat_db.get_character_card_by_id(int(request_data.character_id))

            # For robust handling, try to determine if it's int or str:
            char_lookup_key = request_data.character_id
            try:
                char_id_int = int(char_lookup_key)
                character_data_for_template = chat_db.get_character_card_by_id(char_id_int)
            except ValueError: # Not an int, assume it's a name
                character_data_for_template = chat_db.get_character_card_by_name(char_lookup_key)

            if character_data_for_template:
                associated_character_db_id = character_data_for_template.get('id')
                logger.info(f"Context: Character '{character_data_for_template.get('name')}' (ID: {associated_character_db_id}) loaded for templating and conversation.")
            else:
                logger.warning(f"Character with identifier '{request_data.character_id}' not found in chat_db. Proceeding without character-specific conversation context.")
        except CharactersRAGDBError as e_char_db:
            logger.error(f"DB error fetching character '{request_data.character_id}': {e_char_db}")
        except Exception as e_char_fetch:
            logger.error(f"Error fetching character data for '{request_data.character_id}': {e_char_fetch}", exc_info=True)

    if not associated_character_db_id and request_data.character_id:
        # If a character_id was specified but no character was found or its ID couldn't be retrieved
        logger.warning(f"Specified character_id '{request_data.character_id}' did not resolve to a character for conversation context. Chat will not use DB history for this character.")

    if associated_character_db_id and chat_db: # Proceed with conversation handling only if a character context is established
        if final_conversation_id_for_turn:
            # Validate existing conversation_id
            try:
                conversation_details = chat_db.get_conversation_by_id(final_conversation_id_for_turn)
                if not conversation_details:
                    logger.warning(f"Provided conversation_id '{final_conversation_id_for_turn}' not found. A new conversation will be created.")
                    final_conversation_id_for_turn = None # Mark for creation
                elif conversation_details.get('character_id') != associated_character_db_id:
                    logger.warning(f"Conversation_id '{final_conversation_id_for_turn}' belongs to character {conversation_details.get('character_id')}, not current character {associated_character_db_id}. A new conversation will be created.")
                    final_conversation_id_for_turn = None # Mark for creation
                else:
                    logger.info(f"Using existing conversation_id '{final_conversation_id_for_turn}' for character {associated_character_db_id}.")
            except CharactersRAGDBError as e_conv_val:
                logger.error(f"DB error validating conversation_id '{final_conversation_id_for_turn}': {e_conv_val}. Assuming new conversation.")
                final_conversation_id_for_turn = None

        if not final_conversation_id_for_turn: # Need to create a new one
            try:
                char_name_for_title = character_data_for_template.get('name', 'Character') if character_data_for_template else "Unknown Character"
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_conv_title = f"Chat with {char_name_for_title} ({timestamp_str})"

                # Ensure client_id is available on chat_db or fetched separately
                client_id_for_conv = getattr(chat_db, 'client_id', None)
                if not client_id_for_conv:
                    # This part depends on how client_id is managed. If it's tied to user token:
                    # client_id_for_conv = await get_client_id_from_token(Token) # Example, if you have such a dependency
                    logger.error("Client ID not found on chat_db instance for new conversation. This is a configuration issue.")
                    raise ChatConfigurationError(provider=target_endpoint, message="Cannot create conversation due to missing client ID.")

                new_conv_data = {
                    'character_id': associated_character_db_id,
                    'title': new_conv_title,
                    'client_id': client_id_for_conv
                }
                newly_created_conv_id = chat_db.add_conversation(new_conv_data) # Returns the UUID string
                if newly_created_conv_id:
                    final_conversation_id_for_turn = newly_created_conv_id
                    conversation_created_now = True
                    logger.info(f"Created new conversation ID '{final_conversation_id_for_turn}' for character ID '{associated_character_db_id}'.")
                else:
                    logger.error("Failed to create new conversation in DB (add_conversation returned None/False). Proceeding without history.")
            except (InputError, ConflictError, CharactersRAGDBError, ChatConfigurationError) as e_conv_create:
                logger.error(f"Error creating new conversation for character {associated_character_db_id}: {e_conv_create}", exc_info=True)
                # Proceed without history if creation fails, final_conversation_id_for_turn remains None or previous invalid value
                final_conversation_id_for_turn = request_data.conversation_id # Revert to original if creation failed
            except Exception as e_unexpected_conv_create:
                logger.error(f"Unexpected error creating new conversation: {e_unexpected_conv_create}", exc_info=True)
                final_conversation_id_for_turn = request_data.conversation_id

        # Load history if conversation ID is valid and wasn't just created
        if final_conversation_id_for_turn and not conversation_created_now and chat_db:
            try:
                # TODO: Add configurable history length, e.g., from config or request
                raw_db_messages = chat_db.get_messages_for_conversation(
                    conversation_id=final_conversation_id_for_turn,
                    order_by_timestamp="ASC",
                    limit=20 # Example: Load last 10 turns (20 messages)
                )
                for db_msg in raw_db_messages:
                    role = db_msg.get("sender")
                    if role not in ["user", "assistant"]: # We only want user/assistant turns as direct history
                        continue

                    text_content = db_msg.get("content", "")
                    image_data_bytes = db_msg.get("image_data")
                    image_mime_type = db_msg.get("image_mime_type")

                    current_message_content_parts = []
                    if text_content:
                        current_message_content_parts.append({"type": "text", "text": text_content})

                    if image_data_bytes and image_mime_type:
                        try:
                            base64_image = base64.b64encode(image_data_bytes).decode('utf-8')
                            image_url = f"data:{image_mime_type};base64,{base64_image}"
                            current_message_content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                        except Exception as e_img_hist:
                            logger.warning(f"Could not encode image from history (msg_id: {db_msg.get('id')} for conv {final_conversation_id_for_turn}): {e_img_hist}")

                    if current_message_content_parts:
                         # Name is not stored per message in DB, so can't add it.
                        historical_openai_messages.append({"role": role, "content": current_message_content_parts})
                if historical_openai_messages:
                    logger.info(f"Loaded {len(historical_openai_messages)} messages from conversation '{final_conversation_id_for_turn}'.")
            except CharactersRAGDBError as e_hist_load:
                 logger.error(f"DB error loading history for conversation '{final_conversation_id_for_turn}': {e_hist_load}")
            except Exception as e_hist_processing:
                 logger.error(f"Error processing history for conversation '{final_conversation_id_for_turn}': {e_hist_processing}", exc_info=True)

    elif request_data.conversation_id: # conversation_id provided but no character context to use it with
        logger.warning(f"Conversation ID '{request_data.conversation_id}' was provided, but no character context could be established. Ignoring conversation ID for history loading.")
        final_conversation_id_for_turn = None # Cannot use or validate it

    # --- Prompt Templating ---
    active_template: Optional[PromptTemplate] = None
    if request_data.prompt_template_name:
        logger.info(f"Attempting to load prompt template: '{request_data.prompt_template_name}'")
        active_template = load_template(request_data.prompt_template_name)

    if not active_template:
        logger.info(f"No specific template loaded or specified for '{request_data.prompt_template_name}', using default raw passthrough.")
        active_template = DEFAULT_RAW_PASSTHROUGH_TEMPLATE

    # Prepare data for template placeholders
    template_data: Dict[str, Any] = {
        "original_system_message_from_request": "",
        # Defaults for character-specific placeholders
        "character_system_prompt": "",
        "char_name": "Character",
        "char_personality": "Default personality",
        "char_description": "Default description",
        "char_scenario": "Default scenario",
        "char_first_message": "",
        "char_message_example": ""
        # Add other common placeholders with default empty values
    }

    # If character_id is available in request_data and chat_db is available, fetch character data
    # Assuming ChatCompletionRequest is updated to include `character_id: Optional[str] = None`
    character_id_from_request = getattr(request_data, 'character_id', None)
    if character_id_from_request and chat_db:
        try:
            logger.info(f"Fetching character data for ID: {character_id_from_request} for templating.")
            # Note: ChaChaNotes_DB uses integer IDs for characters. Adjust if using UUIDs.
            # For this example, assuming character_id_from_request can be cast to int if needed by get_character_card_by_id.
            # If your character_id is a name, use get_character_card_by_name.
            # This is a placeholder for how you'd fetch the ID.
            # Let's assume character_id_from_request is the actual ID (int or str based on your DB).

            # If get_character_card_by_id expects an int:
            # char_id_for_db = int(character_id_from_request)
            # Else (if it's already correct type, e.g. string UUID):
            char_id_for_db = character_id_from_request

            character_data_for_template = chat_db.get_character_card_by_id(char_id_for_db) # Or by name if that's the identifier
            if character_data_for_template:
                logger.success(f"Character data found for '{character_data_for_template.get('name', 'Unknown')}'")
                template_data["character_system_prompt"] = character_data_for_template.get('system_prompt', "")
                template_data["char_name"] = character_data_for_template.get('name', "Character")
                template_data["char_personality"] = character_data_for_template.get('personality', "")
                template_data["char_description"] = character_data_for_template.get('description', "")
                template_data["char_scenario"] = character_data_for_template.get('scenario', "")
                template_data["char_first_message"] = character_data_for_template.get('first_message', "")
                template_data["char_message_example"] = character_data_for_template.get('message_example', "")
                # Add any other fields from character_data_for_template that your templates might use
            else:
                logger.warning(f"Character with ID '{character_id_from_request}' not found in chat_db.")
        except ValueError:
            logger.error(f"Invalid character_id format: '{character_id_from_request}'. Could not convert to expected type for DB lookup.")
        except Exception as e_char_fetch:
            logger.error(f"Error fetching character data for ID '{character_id_from_request}': {e_char_fetch}", exc_info=True)


    # --- Prepare messages_payload and Extract Original System Message for Template ---
    # This section is now primarily about extracting the original system message
    # and then applying the chosen template to all messages.

    messages_from_current_request: List[Dict[str, Any]] = []
    original_system_message_content: Optional[str] = None

    for msg_model in request_data.messages: # Process only messages from the current request here
        msg_dict = msg_model.model_dump(exclude_none=True)
        if msg_model.role == 'system' and original_system_message_content is None:
            # Capture the first system message from the request for the template placeholder
            if isinstance(msg_model.content, str):
                original_system_message_content = msg_model.content
            # ... (handle list content for system message as before) ...
            logger.debug(f"Original system message from current request: '{str(original_system_message_content)[:100]}...'")
            continue # Don't add system messages from request to the list that gets content-templated yet
        messages_from_current_request.append(msg_dict)

    template_data["original_system_message_from_request"] = original_system_message_content or ""

    # Apply template to generate the final system message to be passed to chat_api_call
    final_system_message_for_provider: Optional[str] = None
    if active_template.system_message_template:
        final_system_message_for_provider = apply_template_to_string(
            active_template.system_message_template,
            template_data
        )
        logger.info(f"Applied template to system message. Preview: '{str(final_system_message_for_provider)[:200]}...'")
    elif original_system_message_content:
        final_system_message_for_provider = original_system_message_content
        logger.info("Using original system message from request as no system_message_template was in active template.")
    else:
        logger.info("No system message by template and no original system message in request.")

    # Combine historical messages with current request's messages for content templating
    # History is already in OpenAI dict format. Current request messages are also dicts.
    all_messages_for_content_templating: List[Dict[str, Any]] = historical_openai_messages + messages_from_current_request

    templated_messages_payload_for_provider: List[Dict[str, Any]] = []
    for msg_dict in all_messages_for_content_templating: # Iterate over combined messages
        # ... (apply user/assistant content templates as before using active_template and template_data) ...
        # This logic (lines 170-203 approx in your original chat.py) should be applied here
        # to each msg_dict in all_messages_for_content_templating.
        current_content = msg_dict.get("content") # This should be List[Dict] if from history, or Str/List[Dict] if from request
        role = msg_dict.get("role")
        template_string_for_content = None

        # Create a copy of template_data for this specific message to avoid cross-contamination
        # if we were to add message-specific placeholders in the future.
        message_specific_template_data = template_data.copy()

        if role == "user":
            template_string_for_content = active_template.user_message_content_template
        elif role == "assistant":
            template_string_for_content = active_template.assistant_message_content_template

        # If it's not user/assistant, or no template string for that role, pass through.
        if not template_string_for_content or role not in ["user", "assistant"]:
            templated_messages_payload_for_provider.append(msg_dict)
            continue

        # Handle content: it could be a string (older/simpler system) or a list of parts (OpenAI format)
        new_content_to_set = None
        if isinstance(current_content, str): # Simple string content
            message_specific_template_data["message_content"] = current_content
            templated_text = apply_template_to_string(template_string_for_content, message_specific_template_data)
            if templated_text is not None:
                new_content_to_set = templated_text
        elif isinstance(current_content, list): # List of content parts (e.g. text and image)
            new_parts_list = []
            for part in current_content:
                if part.get("type") == "text" and part.get("text") is not None:
                    message_specific_template_data["message_content"] = part["text"]
                    templated_text_part = apply_template_to_string(template_string_for_content, message_specific_template_data)
                    if templated_text_part is not None:
                        new_parts_list.append({"type": "text", "text": templated_text_part})
                    else:
                        new_parts_list.append(part) # Keep original if template failed for text
                else:
                    new_parts_list.append(part) # Keep non-text parts (e.g. images) as is
            new_content_to_set = new_parts_list

        if new_content_to_set is not None:
            msg_dict_copy = msg_dict.copy() # Avoid modifying the original in all_messages_for_content_templating
            msg_dict_copy["content"] = new_content_to_set
            templated_messages_payload_for_provider.append(msg_dict_copy)
        else: # If content couldn't be templated for some reason, pass original
            templated_messages_payload_for_provider.append(msg_dict)

    if templated_messages_payload_for_provider:
        logger.debug(f"Combined & templated messages (first message content preview): {str(templated_messages_payload_for_provider[0].get('content'))[:200]}...")
    else:
        logger.debug("No user/assistant messages in payload after history and templating.")


    # --- Prepare Arguments for chat_api_call (as before, using templated_messages_payload_for_provider and final_system_message_for_provider) ---
    chat_args = {
        "api_endpoint": target_endpoint,
        "api_key": provider_api_key,
        "messages_payload": templated_messages_payload_for_provider,
        "temp": request_data.temperature,
        "system_message": final_system_message_for_provider, # Use templated system message
        "streaming": request_data.stream,
        "minp": request_data.minp,
        "maxp": request_data.top_p,  # chat_api_call maps 'maxp' to provider's top_p if applicable
        "model": request_data.model,
        "topk": request_data.topk,
        "topp": request_data.top_p,  # chat_api_call also has 'topp' for direct top_p mapping
        "logprobs": request_data.logprobs,
        "top_logprobs": request_data.top_logprobs,
        "logit_bias": request_data.logit_bias,
        "presence_penalty": request_data.presence_penalty,
        "frequency_penalty": request_data.frequency_penalty,
        "tools": request_data.tools,
        "tool_choice": request_data.tool_choice,
        "max_tokens": request_data.max_tokens,
        "seed": request_data.seed,
        "stop": request_data.stop,
        "response_format": request_data.response_format.model_dump() if request_data.response_format else None, # Pass as dict
        "n": request_data.n,      
        "user_identifier": request_data.user,
        # Ensure all new schema fields that map to chat_api_call are added here
    }

    # Clean chat_args: Remove None values as chat_api_call might not expect them,
    # or the underlying functions have their own defaults.
    chat_args_cleaned = {k: v for k, v in chat_args.items() if v is not None}
    # logger.trace(f"Arguments for chat_api_call (after templating, api_key excluded): { {k: v for k,v in chat_args_cleaned.items() if k != 'api_key'} }")

    if not request_data.tools and chat_args_cleaned.get("tool_choice") == "auto":
        logger.debug(
            f"No tools provided and tool_choice is 'auto' for {target_endpoint}. Removing 'tool_choice' from chat_args.")
        if "tool_choice" in chat_args_cleaned:
            del chat_args_cleaned["tool_choice"]

    loop = asyncio.get_running_loop()
    try:
        if request_data.stream:
            logger.info(f"Streaming requested for {target_endpoint} with model {request_data.model}.")
            func_call = partial(chat_api_call, **chat_args_cleaned)
            stream_generator_or_error = await loop.run_in_executor(None, func_call)

            if isinstance(stream_generator_or_error, Exception):  # Should be caught by specific exception handlers below now
                logger.error(
                    f"Error from chat_api_call during streaming setup for {target_endpoint}: {stream_generator_or_error}")
                raise stream_generator_or_error  # Re-raise to be caught by specific handlers

            # Check if it's a "proper" iterator/async iterator and NOT just a string/bytes
            is_valid_iterator_type = isinstance(stream_generator_or_error, (Iterator, AsyncIterator))
            is_simple_string_or_bytes = isinstance(stream_generator_or_error, (str, bytes))

            if not (is_valid_iterator_type and not is_simple_string_or_bytes):
                logger.error(
                    f"chat_api_call did not return a valid generator/iterator for streaming from {target_endpoint}. Type: {type(stream_generator_or_error)}")
                raise ChatProviderError(provider=target_endpoint,
                                        message="Streaming setup error: backend function did not return an iterator.",
                                        status_code=500)

            stream_generator = stream_generator_or_error

            async def async_stream_wrapper():
                # Yield metadata event first if conversation_id is available
                if final_conversation_id_for_turn:
                    metadata_event_payload = {"conversation_id": final_conversation_id_for_turn}
                    if conversation_created_now:
                         metadata_event_payload["conversation_created"] = True
                    yield f"event: tldw_metadata\ndata: {json.dumps(metadata_event_payload)}\n\n"

                # ... (the rest of the streaming logic as before, iterating stream_generator_or_error) ...
                try:
                    if hasattr(stream_generator_or_error, '__aiter__'): # Async iterator
                        async for chunk in stream_generator_or_error:
                            yield chunk
                    elif hasattr(stream_generator_or_error, '__iter__'): # Sync iterator
                        for chunk in stream_generator_or_error:
                            yield chunk
                            await asyncio.sleep(0.001)
                    else:
                        logger.error(f"Streaming error: Generator is not iterable from {target_endpoint}.")
                        yield f"data: {json.dumps({'error': {'message': 'Internal streaming error: Not iterable', 'type': 'stream_error'}})}\n\n"
                except Exception as stream_e:
                    logger.error(f"Error during streaming data generation from {target_endpoint}: {stream_e}", exc_info=True)
                    error_payload = json.dumps({"error": {"message": f"Stream data generation error: {str(stream_e)}", "type": "stream_error"}})
                    yield f"data: {error_payload}\n\n"
                finally:
                    logger.debug(f"Stream from {target_endpoint} finished or errored. Sending DONE if not already done.")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(async_stream_wrapper(), media_type="text/event-stream")

        else:  # Non-streaming
            logger.info(f"Non-streaming request to {target_endpoint} with model {request_data.model}.")
            func_call = partial(chat_api_call, **chat_args_cleaned)
            response_data_or_error = await loop.run_in_executor(None, func_call)

            response_data = response_data_or_error # Assuming it's the actual LLM response payload

            # Add conversation_id to the response
            final_response_payload = jsonable_encoder(response_data)
            if isinstance(final_response_payload, dict) and final_conversation_id_for_turn:
                final_response_payload["tldw_conversation_id"] = final_conversation_id_for_turn
                if conversation_created_now:
                    final_response_payload["tldw_conversation_created"] = True
            elif final_conversation_id_for_turn: # If original response wasn't a dict, wrap it
                 final_response_payload = {
                    "llm_response": final_response_payload,
                    "tldw_conversation_id": final_conversation_id_for_turn
                 }
                 if conversation_created_now:
                    final_response_payload["tldw_conversation_created"] = True

            logger.info(f"Successfully received non-streaming response from {target_endpoint} for model: {request_data.model}")
            return JSONResponse(content=final_response_payload)

    # --- Specific Chat*Error Handling (from chat_api_call or raised within this endpoint) ---
    except ChatAuthenticationError as e:
        logger.warning(f"Authentication error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except ChatRateLimitError as e:
        logger.warning(f"Rate limit error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except ChatBadRequestError as e:  # This will handle ChatBadRequestError raised by chat_api_call
        logger.warning(f"Bad request error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ChatConfigurationError as e:  # Server-side configuration issue (e.g., from API key check)
        logger.error(f"Configuration error for {e.provider or target_endpoint}: {e.message}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except ChatProviderError as e:  # MODIFIED
        logger.error(
            f"Provider error from {e.provider or target_endpoint} (Status {e.status_code}) in chat_api_call: {e.message}",
            exc_info=True)
        raise HTTPException(status_code=e.status_code if e.status_code else 502, detail=str(e))
    except ChatAPIError as e:  # MODIFIED - Catch other ChatAPIError (should be after ChatProviderError)
        logger.error(f"ChatAPIError in chat_api_call for {e.provider or target_endpoint}: {e.message}", exc_info=True)
        raise HTTPException(status_code=e.status_code if e.status_code else 500, detail=str(e))
    except ValueError as e:  # ADDED - To handle raw ValueErrors from chat_api_call (like in test_chat_api_call_exception_handling_unit[error_type7])
        logger.error(f"ValueError from API call for {target_endpoint}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid input or parameter: {str(e)}")
    except HTTPException as http_exc:  # ADDED - To re-raise HTTPExceptions if chat_api_call itself raises one (e.g. from a mock)
        logger.info(
            f"HTTPException propagated from API call for {target_endpoint}: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc  # Re-raise it as is

    # --- Catching errors from `requests` ---
    except HTTPError as e: # Explicitly from requests.exceptions
        status_code_from_exc = e.response.status_code if e.response is not None else 502 # Default to 502 for upstream HTTP errors
        detail_msg = f"Upstream API HTTP Error ({target_endpoint}): Status {status_code_from_exc}"
        try:
            error_text = e.response.text[:200] if e.response is not None else "No response body"
            detail_msg += f" - Details: {error_text}"
        except Exception:
            pass  # Ignore if cannot get response text
        logger.error(f"Unmapped HTTPError in endpoint: {detail_msg}", exc_info=True)
        final_status = status_code_from_exc
        # Ensure proper client status codes
        if final_status == 401: pass # Keep 401
        elif final_status == 429: pass # Keep 429
        elif not (400 <= final_status < 500): # If not a 4xx client error already
            final_status = 502 # Default to Bad Gateway for other upstream errors
        raise HTTPException(status_code=final_status, detail=detail_msg)
    except RequestException as e:  # Catch network errors like ConnectionError, Timeout
        logger.error(f"Unmapped RequestException (Network error) for {target_endpoint}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail=f"Network error while contacting {target_endpoint}: {str(e)}")

    # --- Catch potential config/value/key errors during argument preparation ---
    except (ValueError, TypeError, KeyError) as e:  # Existing block for local errors
        logger.error(f"Data or configuration validation error in endpoint setup: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid parameter or internal configuration error: {str(e)}")
    # --- Final Catch-All for truly unexpected errors ---
    except Exception as e:  # This will now catch genuinely unexpected errors not covered above.
        logger.critical(
            f"!!! UNEXPECTED ERROR in /completions endpoint for {target_endpoint} !!! Type: {type(e).__name__}, Error: {e}",
            exc_info=True)
        error_detail = f"An unexpected internal server error occurred. Error type: {type(e).__name__}."
        if final_conversation_id_for_turn:
            error_detail += f" (Context ConvID: {final_conversation_id_for_turn})"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)

#
# End of chat.py
#######################################################################################################################
