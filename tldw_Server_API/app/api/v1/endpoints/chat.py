# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
# FIXME
#
# Imports
import asyncio
import logging
import json
import os
from functools import partial
from typing import List, Optional, Union, Dict, Any, Literal
from dotenv import load_dotenv
import toml
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
from pydantic import BaseModel
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator, model_validator
import redis
import requests
from requests import RequestException, HTTPError
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.responses import JSONResponse, StreamingResponse
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest, API_KEYS, \
    DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_db_for_user
from tldw_Server_API.app.core.Chat.Chat_Functions import (
        chat_api_call,
        ChatAuthenticationError,
        ChatRateLimitError,
        ChatBadRequestError,
        ChatConfigurationError,
        ChatProviderError,
        ChatAPIError,
)
from tldw_Server_API.app.core.DB_Management.Media_DB import Database
#
# DB Mgmt
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
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
    }
)
async def create_chat_completion(
    request_data: ChatCompletionRequest = Body(...),
    db: Database = Depends(get_db_for_user), # Your existing auth dependency
):
    # FIXME - Add auth checks here
    """
    Acts as a proxy to various chat completion APIs via the `chat_api_call` shim.

    Accepts a JSON body similar to the OpenAI Chat Completion API request structure,
    with an added **`api_provider`** field to select the backend (e.g., 'openai', 'anthropic').
    If `api_provider` is omitted, the server's default provider is used.

    - **api_provider**: (Optional) Specify the backend (e.g., 'openai', 'anthropic').
    - **model**: ID of the model to use (specific compatibility depends on the provider).
    - **messages**: Conversation history.
    - **Optional parameters**: Forwarded to the selected backend API via `chat_api_call`
      (e.g., `temperature`, `max_tokens`, `stream`, `tools`, `minp`, `topk`).

    *(Refer to the ChatCompletionRequest schema for all parameters)*
    """
    # --- Determine Target API Provider ---
    target_endpoint = request_data.api_provider or DEFAULT_LLM_PROVIDER
    logger.info(f"Routing chat completion request for model '{request_data.model}' to provider: '{target_endpoint}'")
    logger.debug(f"Incoming request data: {request_data.model_dump_json(exclude_none=True)}")

    # --- Get API Key for the Target Provider ---
    current_api_key = API_KEYS.get(target_endpoint.lower())
    # Allow providers without keys (like some local LLMs)
    # if not current_api_key and target_endpoint.lower() not in ["llama.cpp", "local-llm", "ooba"]: # Add other keyless providers if needed
    is_key_required = target_endpoint.lower() not in ["llama.cpp", "local-llm", "ooba", "tabbyapi", "kobold"] # Example: List providers potentially not needing a key
    if not current_api_key and is_key_required:
         logger.error(f"API Key for provider '{target_endpoint}' is not configured on the server.")
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail=f"API key is not configured for the '{target_endpoint}' chat service."
         )
    # Log securely if key exists
    if current_api_key and isinstance(current_api_key, str) and len(current_api_key) > 8:
        logger.info(f"Using API Key for {target_endpoint}: {current_api_key[:4]}...{current_api_key[-4:]}")
    elif current_api_key:
        logger.info(f"Using API Key for {target_endpoint}: Provided (length <= 8)")
    elif not is_key_required:
        logger.info(f"No API Key needed or provided for {target_endpoint}.")
    else: # Should be caught by the error above, but for completeness
         logger.warning(f"API Key required but not found for {target_endpoint}.")


    # --- Extract System Message ---
    system_message_content: Optional[str] = None
    input_messages_for_provider = [] # Prepare the list of messages to send
    for msg_model in request_data.messages:
        # Convert Pydantic model to dict for the shim function
        msg_dict = msg_model.model_dump(exclude_none=True)
        input_messages_for_provider.append(msg_dict)
        # Extract first system message content if present
        if msg_model.role == 'system' and system_message_content is None:
            if isinstance(msg_model.content, str):
                system_message_content = msg_model.content
            # Note: OpenAI allows system message within the main list.
            # chat_api_call shim might handle this differently per provider.

    # --- Prepare Arguments for chat_api_call ---
    # Map ChatCompletionRequest fields to chat_api_call arguments
    chat_args = {
        "api_endpoint": target_endpoint, # Use the determined endpoint
        "api_key": current_api_key,     # Use the key for that endpoint
        "input_data": input_messages_for_provider, # Pass the list of message dicts
        "prompt": None,                 # Prompt arg isn't primary for chat models via messages
        "temp": request_data.temperature,
        "system_message": system_message_content,
        "streaming": request_data.stream,
        "minp": request_data.minp,
        "maxp": request_data.top_p,      # Map Pydantic top_p to chat_api_call maxp
        "model": request_data.model,
        "topk": request_data.topk,
        "topp": request_data.top_p       # Map Pydantic top_p to chat_api_call topp as well (shim handles provider specifics)
        # Pass other relevant OpenAI params if chat_api_call uses them internally
        # e.g., if chat_api_call passes max_tokens, tools, etc., add them here:
        # "max_tokens": request_data.max_tokens,
        # "tools": request_data.tools.model_dump() if request_data.tools else None,
        # ... etc.
    }

    # Clean chat_args: Remove None values as chat_api_call might not expect them
    chat_args_cleaned = {k: v for k, v in chat_args.items() if v is not None}

    # Get the current event loop
    loop = asyncio.get_running_loop()
    try:  # Single main try block for the core logic
        # --- Handle Streaming ---
        if request_data.stream:
            logger.info(f"Streaming requested for {target_endpoint}.")
            # Use partial to wrap the function call with arguments for the executor
            func_call = partial(chat_api_call, **chat_args_cleaned)
            # Exceptions from chat_api_call will be raised here by run_in_executor
            stream_generator = await loop.run_in_executor(None, func_call)

            # Check if the returned value is actually iterable *before* wrapping
            if not hasattr(stream_generator, '__aiter__') and not hasattr(stream_generator, '__iter__'):
                logger.error(
                    f"chat_api_call did not return a valid generator/iterator for streaming from {target_endpoint}.")
                # Raise error to be caught by handlers below
                raise ChatProviderError(provider=target_endpoint,
                                        message="Streaming setup error: backend function did not return iterator.",
                                        status_code=500)

            async def async_stream_wrapper():
                # Wrapper for handling sync/async iterators from executor
                try:
                    if hasattr(stream_generator, '__aiter__'):
                        async for chunk in stream_generator: yield chunk
                    elif hasattr(stream_generator, '__iter__'):
                        for chunk in stream_generator:
                            yield chunk
                            await asyncio.sleep(0)  # Yield control
                    # No need for 'else' here, the check above should prevent invalid types
                except Exception as stream_e:
                    logger.error(f"Error during streaming generation from {target_endpoint}: {stream_e}", exc_info=True)
                    error_payload = json.dumps(
                        {"error": {"message": f"Stream generation error: {str(stream_e)}", "type": "stream_error"}})
                    yield f"data: {error_payload}\n\n"
                finally:
                    # Always send DONE event
                    yield "data: [DONE]\n\n"

            return StreamingResponse(async_stream_wrapper(), media_type="text/event-stream")

        # --- Handle Non-Streaming ---
        else:
            logger.info(f"Non-streaming request to {target_endpoint}.")
            # Use partial for cleaner executor call
            func_call = partial(chat_api_call, **chat_args_cleaned)
            # Exceptions from chat_api_call will be raised here by run_in_executor
            response_data = await loop.run_in_executor(None, func_call)

            logger.info(f"Successfully received response from {target_endpoint} for model: {request_data.model}")
            return JSONResponse(content=jsonable_encoder(response_data))

    # --- CATCH SPECIFIC Chat*Errors FIRST ---
    except ChatAuthenticationError as e:
        logger.warning("Caught ChatAuthenticationError (%s): %s", e.provider, e.message)  # Safe logging
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=e.message)
    except ChatRateLimitError as e:
        logger.warning("Caught ChatRateLimitError (%s): %s", e.provider, e.message)  # Safe logging
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=e.message)
    except ChatBadRequestError as e:
        logger.warning("Caught ChatBadRequestError (%s): %s", e.provider, e.message)  # Safe logging
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except ChatConfigurationError as e:
        logger.error("Caught ChatConfigurationError (%s): %s", e.provider, e.message, exc_info=True)  # Safe logging
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message)
    except ChatProviderError as e:
        # This now also catches the stream setup error raised above
        logger.error("Caught ChatProviderError (%s, Status: %s): %s", e.provider, e.status_code, e.message,
                     exc_info=True)  # Safe logging
        http_status = e.status_code if 400 <= e.status_code < 600 else 502
        raise HTTPException(status_code=http_status, detail=e.message)
    except ChatAPIError as e:  # Catch base or other specific Chat errors
        logger.error("Caught ChatAPIError (%s): %s", e.provider, e.message, exc_info=True)  # Safe logging
        http_status = e.status_code if 400 <= e.status_code < 600 else 500
        raise HTTPException(status_code=http_status, detail=e.message)

    # --- CATCH GENERIC LIBRARY/NETWORK ERRORS (if they escape chat_api_call or happen in executor/stream setup) ---
    except HTTPError as e:  # From requests library
        status_code = e.response.status_code if e.response is not None else 500
        detail = f"Upstream API Error ({target_endpoint}): Status {status_code}"
        try:
            detail += f" - {e.response.text[:200]}" if e.response is not None else ""
        except:
            pass
        logger.error("Caught unmapped HTTPError in endpoint: %s", detail, exc_info=True)  # Safe logging
        http_status = status_code if 400 <= status_code < 600 else 500
        if status_code == 401: http_status = 401  # Ensure 401 is passed correctly if caught here
        raise HTTPException(status_code=http_status, detail=detail)
    except RequestException as e:  # From requests library
        logger.error("Caught unmapped RequestException in endpoint: %s", e, exc_info=True)  # Safe logging
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail=f"Network error contacting {target_endpoint}: {e}")

    # --- CATCH POTENTIAL CONFIG/VALUE ERRORS ---
    except (ValueError, TypeError, KeyError) as e:
        logger.error("Caught config/value/key error in endpoint: %s", e, exc_info=True)  # Safe logging
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid parameter or configuration: {e}")

    # --- FINAL CATCH-ALL ---
    except Exception as e:
        # Log details about the unknown exception 'e' using safe % formatting
        logger.error("!!! ENDPOINT CAUGHT UNEXPECTED EXCEPTION !!!")
        logger.error("!!! Type: %s", type(e).__name__)
        logger.error("!!! Args: %s", e.args)
        logger.error("!!! Str: %s", str(e))
        logger.exception(
            "FINAL CATCH-ALL - Unexpected error processing chat completion endpoint for %s:", target_endpoint)

        # Return a more informative generic error including the type
        detail_str = str(e)[:100]  # Get first 100 chars safely
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An unexpected internal server error occurred. Type: {type(e).__name__}. Details: {detail_str}")

#
# End of media.py
#######################################################################################################################
