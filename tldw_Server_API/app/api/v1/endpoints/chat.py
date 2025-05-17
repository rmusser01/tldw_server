# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
# FIXME
#
# Imports
import asyncio
import logging
import json
from functools import partial
from typing import List, Optional, Dict, Any
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
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

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
    media_db = Depends(get_media_db_for_user),
    chat_db: CharactersRAGDB = Depends(get_chacha_db_for_user), # Added type hint
    Token: Optional[str] = Header(None), # Made Token optional as per Header(None)
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

    # --- Get API Key for the Target Provider ---
    current_api_key = API_KEYS.get(target_endpoint.lower())

    # Define which providers might not require a key from server-side config
    keyless_providers = ["llama.cpp", "local-llm", "ooba", "tabbyapi", "kobold", "ollama", "vllm"]
    is_key_required = target_endpoint.lower() not in keyless_providers

    if not current_api_key and is_key_required:
        logger.error(f"API Key for provider '{target_endpoint}' is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"API key is not configured for the '{target_endpoint}' chat service on the server."
        )

    if current_api_key and isinstance(current_api_key, str) and len(current_api_key) > 8:
        logger.info(f"Using API Key for {target_endpoint}: {current_api_key[:4]}...{current_api_key[-4:]}")
    elif current_api_key:
        logger.info(f"Using API Key for {target_endpoint}: Provided (length <= 8)")
    elif not is_key_required:
        logger.info(f"No API Key configured or needed for {target_endpoint}.")
    # No 'else' needed here as the error case is handled above.

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

            character_data = chat_db.get_character_card_by_id(char_id_for_db) # Or by name if that's the identifier
            if character_data:
                logger.success(f"Character data found for '{character_data.get('name', 'Unknown')}'")
                template_data["character_system_prompt"] = character_data.get('system_prompt', "")
                template_data["char_name"] = character_data.get('name', "Character")
                template_data["char_personality"] = character_data.get('personality', "")
                template_data["char_description"] = character_data.get('description', "")
                template_data["char_scenario"] = character_data.get('scenario', "")
                template_data["char_first_message"] = character_data.get('first_message', "")
                template_data["char_message_example"] = character_data.get('message_example', "")
                # Add any other fields from character_data that your templates might use
            else:
                logger.warning(f"Character with ID '{character_id_from_request}' not found in chat_db.")
        except ValueError:
            logger.error(f"Invalid character_id format: '{character_id_from_request}'. Could not convert to expected type for DB lookup.")
        except Exception as e_char_fetch:
            logger.error(f"Error fetching character data for ID '{character_id_from_request}': {e_char_fetch}", exc_info=True)


    # --- Prepare messages_payload and Extract Original System Message for Template ---
    # This section is now primarily about extracting the original system message
    # and then applying the chosen template to all messages.

    messages_from_request: List[Dict[str, Any]] = []
    original_system_message_content: Optional[str] = None

    for msg_model in request_data.messages:
        msg_dict = msg_model.model_dump(exclude_none=True)
        if msg_model.role == 'system' and original_system_message_content is None:
            # Capture the first system message from the request for the template placeholder
            if isinstance(msg_model.content, str):
                original_system_message_content = msg_model.content
            elif isinstance(msg_model.content, list) and msg_model.content and isinstance(msg_model.content[0].get("text"), str):
                original_system_message_content = msg_model.content[0]["text"]
            logger.debug(f"Original system message from request: '{str(original_system_message_content)[:100]}...'")
            # We will not add this system message to messages_from_request yet,
            # as the template will construct the final system message.
            continue # Move to next message model
        messages_from_request.append(msg_dict)

    template_data["original_system_message_from_request"] = original_system_message_content or ""

    # Apply template to generate the final system message to be passed to chat_api_call
    final_system_message_for_provider: Optional[str] = None
    if active_template.system_message_template:
        final_system_message_for_provider = apply_template_to_string(
            active_template.system_message_template,
            template_data
        )
        logger.info(f"Applied template to system message. Preview: '{str(final_system_message_for_provider)[:200]}...'")
    elif original_system_message_content: # No template string for system, but original system message exists
        final_system_message_for_provider = original_system_message_content # Use it directly
        logger.info("Using original system message from request as no system_message_template was defined in the active template.")
    else:
        logger.info("No system message generated by template and no original system message in request.")


    # Apply template to user/assistant message contents
    templated_messages_payload_for_provider: List[Dict[str, Any]] = []
    for msg_dict in messages_from_request: # Iterate over non-system messages from the request
        current_content = msg_dict.get("content")
        role = msg_dict.get("role")
        template_string_for_content = None

        # Create a copy of template_data for this specific message to avoid cross-contamination
        # if we were to add message-specific placeholders in the future.
        message_specific_template_data = template_data.copy()

        if role == "user":
            template_string_for_content = active_template.user_message_content_template
        elif role == "assistant":
            template_string_for_content = active_template.assistant_message_content_template
        else: # Other roles (tool, etc.) - pass through without content templating for now
            templated_messages_payload_for_provider.append(msg_dict)
            continue

        if isinstance(current_content, str):
            message_specific_template_data["message_content"] = current_content
            new_content_str = apply_template_to_string(template_string_for_content, message_specific_template_data)
            if new_content_str is not None: # Or handle error if template application fails critically
                msg_dict["content"] = new_content_str
        elif isinstance(current_content, list): # Multimodal content
            new_content_parts = []
            for part in current_content:
                if part.get("type") == "text" and part.get("text") is not None:
                    message_specific_template_data["message_content"] = part["text"]
                    templated_text = apply_template_to_string(template_string_for_content, message_specific_template_data)
                    if templated_text is not None:
                        new_content_parts.append({"type": "text", "text": templated_text})
                    else:
                        new_content_parts.append(part) # Keep original if template failed
                else:
                    new_content_parts.append(part) # Keep non-text parts (e.g., images) as is
            msg_dict["content"] = new_content_parts

        templated_messages_payload_for_provider.append(msg_dict)

    if templated_messages_payload_for_provider:
        logger.debug(f"Messages after templating (first message content preview): {str(templated_messages_payload_for_provider[0].get('content'))[:200]}...")
    else:
        logger.debug("No user/assistant messages in payload after processing for templating.")

    # --- Prepare Arguments for chat_api_call ---
    chat_args = {
        "api_endpoint": target_endpoint,
        "api_key": current_api_key,
        "messages_payload": templated_messages_payload_for_provider, # Use templated messages
        "temp": request_data.temperature,
        "system_message": final_system_message_for_provider, # Use templated system message
        "streaming": request_data.stream,
        "minp": request_data.minp,
        "maxp": request_data.top_p,  # chat_api_call maps 'maxp' to provider's top_p if applicable
        "model": request_data.model,
        "topk": request_data.topk,
        "topp": request_data.top_p  # chat_api_call also has 'topp' for direct top_p mapping
        # Add other parameters from ChatCompletionRequest if chat_api_call expects them
        # e.g. "max_tokens": request_data.max_tokens (if chat_api_call passes it on)
        # For now, assuming the PROVIDER_PARAM_MAP in Chat_Functions.py handles what's needed.
    }

    # Clean chat_args: Remove None values as chat_api_call might not expect them,
    # or the underlying functions have their own defaults.
    chat_args_cleaned = {k: v for k, v in chat_args.items() if v is not None}
    # logger.trace(f"Arguments for chat_api_call (after templating, api_key excluded): { {k: v for k,v in chat_args_cleaned.items() if k != 'api_key'} }")


    loop = asyncio.get_running_loop()
    try:
        if request_data.stream:
            logger.info(f"Streaming requested for {target_endpoint} with model {request_data.model}.")
            func_call = partial(chat_api_call, **chat_args_cleaned)
            stream_generator_or_error = await loop.run_in_executor(None, func_call)

            if isinstance(stream_generator_or_error,
                          Exception):  # Should be caught by specific exception handlers below now
                logger.error(
                    f"Error from chat_api_call during streaming setup for {target_endpoint}: {stream_generator_or_error}")
                raise stream_generator_or_error  # Re-raise to be caught by specific handlers

            if not hasattr(stream_generator_or_error, '__aiter__') and not hasattr(stream_generator_or_error,
                                                                                   '__iter__'):
                logger.error(
                    f"chat_api_call did not return a valid generator/iterator for streaming from {target_endpoint}. Type: {type(stream_generator_or_error)}")
                raise ChatProviderError(provider=target_endpoint,
                                        message="Streaming setup error: backend function did not return an iterator.",
                                        status_code=500)

            stream_generator = stream_generator_or_error

            async def async_stream_wrapper():
                try:
                    # Check if the executor returned an awaitable iterator (async generator)
                    if hasattr(stream_generator, '__aiter__'):
                        async for chunk in stream_generator:
                            yield chunk
                    # Check if the executor returned a synchronous iterator (regular generator)
                    elif hasattr(stream_generator, '__iter__'):
                        for chunk in stream_generator:
                            yield chunk
                            await asyncio.sleep(0.001)  # Yield control to event loop
                    else:  # Should have been caught above
                        logger.error(f"Streaming error: Generator from {target_endpoint} is not iterable.")
                        yield f"data: {json.dumps({'error': {'message': 'Internal streaming error: Not iterable', 'type': 'stream_error'}})}\n\n"

                except Exception as stream_e:  # Catch errors from within the generator iteration
                    logger.error(f"Error during streaming data generation from {target_endpoint}: {stream_e}",
                                 exc_info=True)
                    error_payload = json.dumps({"error": {"message": f"Stream data generation error: {str(stream_e)}",
                                                          "type": "stream_error"}})
                    yield f"data: {error_payload}\n\n"
                finally:
                    logger.debug(f"Stream from {target_endpoint} finished or errored. Sending DONE.")
                    yield "data: [DONE]\n\n"  # Ensure DONE is always sent

            return StreamingResponse(async_stream_wrapper(), media_type="text/event-stream")

        else:  # Non-streaming
            logger.info(f"Non-streaming request to {target_endpoint} with model {request_data.model}.")
            func_call = partial(chat_api_call, **chat_args_cleaned)
            response_data_or_error = await loop.run_in_executor(None, func_call)

            if isinstance(response_data_or_error, Exception):  # Should be caught by specific exception handlers below
                logger.error(
                    f"Error from chat_api_call for non-streaming request to {target_endpoint}: {response_data_or_error}")
                raise response_data_or_error  # Re-raise

            response_data = response_data_or_error
            logger.info(
                f"Successfully received non-streaming response from {target_endpoint} for model: {request_data.model}")
            # Before returning, ensure it's JSON serializable.
            # If response_data is already a dict/list from a JSON response, jsonable_encoder is fine.
            # If it's a custom object, ensure it can be encoded.
            try:
                encoded_content = jsonable_encoder(response_data)
                return JSONResponse(content=encoded_content)
            except TypeError as te:
                logger.error(
                    f"TypeError encoding response from {target_endpoint}: {te}. Response type: {type(response_data)}",
                    exc_info=True)
                raise ChatProviderError(provider=target_endpoint,
                                        message=f"Failed to serialize response from provider. Error: {te}",
                                        status_code=500)


    # --- Specific Chat*Error Handling (from chat_api_call or raised within this endpoint) ---
    except ChatAuthenticationError as e:
        logger.warning(f"Authentication error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except ChatRateLimitError as e:
        logger.warning(f"Rate limit error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except ChatBadRequestError as e:
        logger.warning(f"Bad request error for {e.provider or target_endpoint}: {e.message}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ChatConfigurationError as e:  # Server-side configuration issue
        logger.error(f"Configuration error for {e.provider or target_endpoint}: {e.message}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except ChatProviderError as e:  # Errors from the provider itself
        logger.error(f"Provider error from {e.provider or target_endpoint} (Status: {e.status_code}): {e.message}",
                     exc_info=True)
        http_status_code = e.status_code if e.status_code and 400 <= e.status_code < 600 else 502  # Default to 502 Bad Gateway
        raise HTTPException(status_code=http_status_code, detail=str(e))
    except ChatAPIError as e:  # More generic errors from the chat calling process
        logger.error(f"General API error for {e.provider or target_endpoint} (Status: {e.status_code}): {e.message}",
                     exc_info=True)
        http_status_code = e.status_code if e.status_code and 400 <= e.status_code < 600 else 500
        raise HTTPException(status_code=http_status_code, detail=str(e))

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
    except (ValueError, TypeError, KeyError) as e:  # Should be less common now with Pydantic
        logger.error(f"Data or configuration validation error in endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid parameter or internal configuration error: {str(e)}")
    except HTTPException as http_exc:  # Catch HTTPErrors specifically to re-raise them
        raise http_exc
    # --- Final Catch-All for truly unexpected errors ---
    except Exception as e:
        logger.critical(
            f"!!! UNEXPECTED ERROR in /completions endpoint for {target_endpoint} !!! Type: {type(e).__name__}, Error: {e}",
            exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected internal server error occurred. Please contact support. Error type: {type(e).__name__}")

#
# End of chat.py
#######################################################################################################################
