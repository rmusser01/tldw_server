# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
# FIXME
#
# Imports
import asyncio
import logging
import json
from functools import partial
from typing import List, Optional, Union, Dict, Any, Literal
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
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.responses import JSONResponse, StreamingResponse

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest, \
    ChatCompletionSystemMessageParam, API_KEYS, DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_db_for_user
#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    get_character_names,
    get_conversation_name,
    alert_token_budget_exceeded, chat_api_call,
)
from tldw_Server_API.app.core.DB_Management.Media_DB import Database
#
# DB Mgmt
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
#from tldw_Server_API.app.core.DB_Management.DB_Manager import DBManager
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
    is_key_required = target_endpoint.lower() not in ["llama.cpp", "local-llm", "ooba", "tabbyapi"] # Example: List providers potentially not needing a key
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

    # --- Handle Streaming ---
    if request_data.stream:
        logger.info(f"Streaming requested for {target_endpoint}.")
        try:
            # Use partial to wrap the function call with arguments for the executor
            func_call = partial(chat_api_call, **chat_args_cleaned)
            stream_generator = await loop.run_in_executor(None, func_call)

            if not hasattr(stream_generator, '__aiter__') and not hasattr(stream_generator, '__iter__'):
                 logger.error(f"chat_api_call did not return a valid generator/iterator for streaming from {target_endpoint}.")
                 raise HTTPException(status_code=500, detail="Streaming setup error in backend.")

            async def async_stream_wrapper():
                # Wrapper for handling sync/async iterators from executor
                try:
                    if hasattr(stream_generator, '__aiter__'):
                         async for chunk in stream_generator:
                             yield chunk # Assume chunk is already formatted for SSE
                    elif hasattr(stream_generator, '__iter__'):
                         for chunk in stream_generator:
                             yield chunk # Assume chunk is already formatted for SSE
                             await asyncio.sleep(0) # Yield control
                    else:
                         logger.error(f"Backend function {target_endpoint} did not return iterable for streaming.")
                         yield f"data: {json.dumps({'error': 'Streaming error'})}\n\n"
                         yield "data: [DONE]\n\n"
                except Exception as e:
                     logger.error(f"Error during streaming from {target_endpoint}: {e}", exc_info=True)
                     error_payload = json.dumps({"error": {"message": f"Stream generation error: {str(e)}", "type": "stream_error"}})
                     yield f"data: {error_payload}\n\n"
                     yield "data: [DONE]\n\n" # Ensure stream terminates

            return StreamingResponse(async_stream_wrapper(), media_type="text/event-stream")

        except Exception as e:
             logger.error(f"Error setting up streaming response for {target_endpoint}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to initiate stream: {str(e)}")

    # --- Handle Non-Streaming ---
    else:
        logger.info(f"Non-streaming request to {target_endpoint}.")
        try:
            # Use partial for cleaner executor call
            func_call = partial(chat_api_call, **chat_args_cleaned)
            response_data = await loop.run_in_executor(None, func_call)

            if isinstance(response_data, str) and response_data.startswith("An error occurred"):
                 logger.error(f"Error from chat_api_call ({target_endpoint}): {response_data}")
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data)

            logger.info(f"Successfully received response from {target_endpoint} for model: {request_data.model}")
            # Return the raw response from the shim function for now
            # Ensure it's JSON serializable
            return JSONResponse(content=jsonable_encoder(response_data))

        except ValueError as ve:
             logger.error(f"Value error calling chat_api_call for {target_endpoint}: {ve}", exc_info=True)
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except Exception as e:
             logger.error(f"Unexpected error processing chat completion for {target_endpoint}: {e}", exc_info=True)
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error processing chat request for {target_endpoint}.")

#
# End of media.py
#######################################################################################################################
