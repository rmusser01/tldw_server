# Server_API/app/api/v1/endpoints/chat.py
# Description: This code provides a FastAPI endpoint for all Chat-related functionalities.
#
# Imports
from __future__ import annotations
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import asyncio
import base64
import datetime
import json
import logging
import os
from collections import deque
from functools import partial
from io import BytesIO
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

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
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    DEFAULT_CHARACTER_NAME,
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    API_KEYS,
    ChatCompletionRequest,
    DEFAULT_LLM_PROVIDER,
)
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatDictionary,
    ChatProviderError,
    ChatRateLimitError,
    chat_api_call as perform_chat_api_call,
    process_user_input,
    update_chat_content,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import (
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE,
    PromptTemplate,
    apply_template_to_string,
    load_template,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import (
    PromptTemplate,
    load_template,
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE,
    apply_template_to_string,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib import replace_placeholders
#######################################################################################################################
#
# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

router = APIRouter()

ALLOWED_IMAGE_MIME_TYPES: set[str] = {"image/png", "image/jpeg", "image/webp"}
MAX_BASE64_BYTES: int = 3 * 1024 * 1024          # 3 MiB per image
MAX_TEXT_LENGTH: int = 400_000                     # chars in any single text part
MAX_MESSAGES_PER_REQUEST: int = 1000               # hard stop to bound spend
MAX_IMAGES_PER_REQUEST: int = 10                   # New: Limit total images per request

# --- Helper Functions ---

def _check_mime(mime: str) -> bool:
    return mime.lower() in ALLOWED_IMAGE_MIME_TYPES

def _process_content_for_db_sync(
    content_iterable: Any, # Can be list of dicts or string
    conversation_id: str # For logging
) -> tuple[list[str], list[tuple[bytes, str]]]:
    """
    Synchronous helper to process message content, including base64 decoding.
    To be run in an executor.
    """
    text_parts_sync: list[str] = []
    images_sync: list[tuple[bytes, str]] = []   # (bytes, mime)

    processed_content_iterable: Any # Define type more specifically if possible
    if isinstance(content_iterable, str):
        processed_content_iterable = [{"type": "text", "text": content_iterable}]
    elif isinstance(content_iterable, list):
        processed_content_iterable = content_iterable
    else:
        logger.warning(
            "[DB SYNC] Unsupported content type=%s for conv=%s, treating as unsupported text.",
            type(content_iterable),
            conversation_id
        )
        processed_content_iterable = [{"type": "text", "text": f"<unsupported content type: {type(content_iterable).__name__}>"}]

    for part in processed_content_iterable:
        p_type = part.get("type")
        if p_type == "text":
            snippet = str(part.get("text", ""))[:MAX_TEXT_LENGTH + 1] # Ensure text is string
            if len(snippet) > MAX_TEXT_LENGTH:
                logger.info(
                    "[DB SYNC] Trimmed over-long text part (>%d chars) for conv=%s",
                    MAX_TEXT_LENGTH,
                    conversation_id
                )
                snippet = snippet[:MAX_TEXT_LENGTH]
            text_parts_sync.append(snippet)
        elif p_type == "image_url":
            url_dict = part.get("image_url", {})
            url_str = url_dict.get("url", "")

            if url_str.startswith("data:") and ";base64," in url_str:
                try:
                    header, b64_body = url_str.split(";base64,", 1)
                    mime = header.removeprefix("data:")
                    if not _check_mime(mime):
                        logger.warning("[DB SYNC] Blocked disallowed MIME '%s' for conv=%s", mime, conversation_id)
                        continue

                    decoded = base64.b64decode(b64_body, validate=True) # CPU intensive

                    if len(decoded) > MAX_BASE64_BYTES:
                        logger.warning("[DB SYNC] Image too large (%d B > %d B) for conv=%s", len(decoded), MAX_BASE64_BYTES, conversation_id)
                        continue
                    images_sync.append((decoded, mime))
                except (base64.Binascii.Error, ValueError) as e_b64: # type: ignore[attr-defined]
                    logger.warning("[DB SYNC] Bad base64 image for conv=%s: %s", conversation_id, e_b64)
                    continue
                except Exception as e_gen_img:
                    logger.warning("[DB SYNC] Generic error processing image for conv=%s: %s", conversation_id, e_gen_img, exc_info=True)
                    continue
            else:
                logger.warning(
                    "[DB SYNC] image_url part was not a valid data URI or did not pass checks, storing as text placeholder. conv=%s, url_start='%.50s...'",
                    conversation_id, url_str
                )
                text_parts_sync.append(f"<Image URL (not processed): {url_str[:200]}>")
    return text_parts_sync, images_sync

async def _save_message_turn_to_db(
    db: CharactersRAGDB,
    conversation_id: str,
    message_obj: Dict[str, Any],
) -> Optional[str]:
    """
    Persist a single user/assistant message.
    - Validates size/format.
    - CPU-bound content processing (image decoding) is run in an executor.
    - DB write is run in an executor.
    - Logs only metadata, never raw content.
    """
    current_loop = asyncio.get_running_loop()
    role = message_obj.get("role")
    if role not in ("user", "assistant"):
        logger.warning("Skip DB save: invalid role='%s' for conv=%s", role, conversation_id)
        return None

    content = message_obj.get("content")

    try:
        text_parts, images = await current_loop.run_in_executor(
            None, _process_content_for_db_sync, content, conversation_id
        )
    except Exception as e_proc:
        logger.error(
            "Error processing message content in executor for DB save. conv=%s err_type=%s err=%s",
            conversation_id, type(e_proc).__name__, e_proc, exc_info=True
        )
        return None

    if not text_parts and not images: # Issue 1 Fix
        logger.debug("Empty message (no text or valid images after processing) ignored for conv=%s", conversation_id)
        return None

    db_payload = {
        "conversation_id": conversation_id,
        "sender": message_obj.get("name") or role,
        "content": "\n".join(text_parts) if text_parts else None,
        "images": [{"data": b, "mime": m} for b, m in images] or None,
        "client_id": db.client_id,
    }

    try:
        return await current_loop.run_in_executor(None, db.add_message, db_payload)
    except (InputError, ConflictError, CharactersRAGDBError) as e_db:
        logger.error("DB error saving message for conv=%s: Type=%s, Msg='%s'", conversation_id, type(e_db).__name__, e_db)
        return None
    except Exception as e_unexpected_db:
        logger.error("Unexpected DB error saving message for conv=%s: %s", conversation_id, e_unexpected_db, exc_info=True)
        return None


@router.post(
    "/completions",
    summary="Creates a model response and manages conversation state.",
    tags=["Chat"],
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request (e.g., empty messages, text too long, bad parameters)."},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid authentication token."},
        status.HTTP_404_NOT_FOUND: {"description": "Resource not found (e.g., character)."},
        status.HTTP_409_CONFLICT: {"description": "Data conflict (e.g., version mismatch during DB operation)."},
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: {"description": "Request payload too large (e.g., too many messages, too many images)."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error."},
        status.HTTP_502_BAD_GATEWAY: {"description": "Error received from an upstream LLM provider."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Service temporarily unavailable or misconfigured (e.g., provider API key issue)."},
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "Upstream LLM provider timed out."},
    }
)
async def create_chat_completion(
    request_data: ChatCompletionRequest = Body(...),
    chat_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    Token: str = Header(None, description="Bearer token for authentication."),
    # background_tasks: BackgroundTasks = Depends(), # Replaced by starlette.background.BackgroundTask for StreamingResponse
    # request: Request, # For rate limiting via slowapi if enabled - get_remote_address(request)
):
    current_loop = asyncio.get_running_loop()

    if not Token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token.")
    expected_token = os.getenv("API_BEARER")
    if not expected_token:
        logger.critical("API_BEARER environment variable is not set. Authentication cannot be verified.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server authentication is misconfigured.")
    if Token.replace("Bearer ", "", 1).strip() != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")

    if not request_data.messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty.")
    if len(request_data.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many messages (max {MAX_MESSAGES_PER_REQUEST}, got {len(request_data.messages)}).",
        )

    total_image_parts = 0 # Issue 2 Fix
    for msg_model in request_data.messages:
        if isinstance(msg_model.content, list):
            for part in msg_model.content:
                if getattr(part, 'type', None) == 'image_url': # Check based on 'type' attribute
                    total_image_parts += 1
    if total_image_parts > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many images in request (max {MAX_IMAGES_PER_REQUEST}, found {total_image_parts}).",
        )

    for m_idx, m in enumerate(request_data.messages): # Text length validation
        if isinstance(m.content, str) and len(m.content) > MAX_TEXT_LENGTH:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Message at index {m_idx} text too long.")
        elif isinstance(m.content, list):
            for p_idx, part_obj in enumerate(m.content):
                if getattr(part_obj, 'type', None) == 'text' and \
                   isinstance(getattr(part_obj, 'text', None), str) and \
                   len(getattr(part_obj, 'text')) > MAX_TEXT_LENGTH:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Text part at index {p_idx} in message {m_idx} too long.")

    # Issue 3 (Schema Validator for Image Parts) is an external fix to chat_request_schemas.py.
    # This endpoint assumes that schema validation has ensured image URLs are 'data:' URIs if intended for processing.

    provider = (request_data.api_provider or DEFAULT_LLM_PROVIDER).lower()
    user_identifier_for_log = getattr(chat_db, 'client_id', 'unknown_client') # Example from original
    logger.info(
        "Chat completion request. Provider=%s, Model=%s, User=%s, Stream=%s, ConvID=%s, CharID=%s",
        provider, request_data.model, user_identifier_for_log,
        request_data.stream, request_data.conversation_id, request_data.character_id
    )

    character_card_for_context: Optional[Dict[str, Any]] = None
    final_conversation_id: Optional[str] = request_data.conversation_id
    final_character_db_id: Optional[int] = None # Initialize

    try:
        target_api_provider = provider # Already determined
        provider_api_key = API_KEYS.get(target_api_provider) # API_KEYS should be up-to-date

        # Simplified list, actual check might be in Chat_Functions or per-provider
        providers_requiring_keys = ["openai", "anthropic", "cohere", "groq", "openrouter", "deepseek", "mistral", "google", "huggingface"]
        if target_api_provider in providers_requiring_keys and not provider_api_key:
            logger.error(f"API key for provider '{target_api_provider}' is missing or not configured.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Service for '{target_api_provider}' is not configured (key missing).")

        conversation_created_this_turn = False

        # --- Character and Conversation Context ---
        if request_data.character_id:
            try:
                char_id_int = int(request_data.character_id)
                character_card_for_context = await current_loop.run_in_executor(None, chat_db.get_character_card_by_id, char_id_int)
            except ValueError:
                character_card_for_context = await current_loop.run_in_executor(None, chat_db.get_character_card_by_name, request_data.character_id)

            if not character_card_for_context:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Character '{request_data.character_id}' not found.")
            final_character_db_id = character_card_for_context['id']
            logger.info(f"Context: Character '{character_card_for_context['name']}' (ID: {final_character_db_id}) loaded.")
        else:
            character_card_for_context = await current_loop.run_in_executor(None, chat_db.get_character_card_by_name, DEFAULT_CHARACTER_NAME)
            if not character_card_for_context:
                logger.critical(f"CRITICAL: Default character '{DEFAULT_CHARACTER_NAME}' not found in DB.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Default character context is missing.")
            final_character_db_id = character_card_for_context['id']
            logger.info(f"Context: Generic chat, using Default Character '{DEFAULT_CHARACTER_NAME}' (ID: {final_character_db_id}).")

        # Multi-User Security FIXME
        client_id_from_db = getattr(chat_db, 'client_id', None)
        # if not client_id_from_db: # Should be set by get_chacha_db_for_user
        #      logger.critical("Client ID missing on chat_db instance. This is a server configuration issue.")
        #      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error: Client identification for DB operations failed.")

        if final_conversation_id:
            conv_details = await current_loop.run_in_executor(None, chat_db.get_conversation_by_id, final_conversation_id)
            if not conv_details:
                logger.warning(f"Provided conv_id '{final_conversation_id}' not found. New one will be created.")
                final_conversation_id = None
            elif conv_details.get('character_id') != final_character_db_id or conv_details.get('client_id') != client_id_from_db:
                logger.warning(f"Conv_id '{final_conversation_id}' (char {conv_details.get('character_id')}, client {conv_details.get('client_id')}) "
                               f"mismatches context (char {final_character_db_id}, client {client_id_from_db}). New conv will be created.")
                final_conversation_id = None

        if not final_conversation_id:
            char_name = character_card_for_context.get('name', "Chat") if character_card_for_context else "Chat"
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            title = f"{char_name} ({timestamp})"
            conv_data = {'character_id': final_character_db_id, 'title': title, 'client_id': client_id_from_db}
            created_id = await current_loop.run_in_executor(None, chat_db.add_conversation, conv_data)
            if not created_id:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create new conversation in DB.")
            final_conversation_id = created_id
            conversation_created_this_turn = True
            logger.info(f"Created new conv_id '{final_conversation_id}' for char_id {final_character_db_id}.")

        # --- History Loading ---
        historical_openai_messages: List[Dict[str, Any]] = []
        if not conversation_created_this_turn and final_conversation_id:
            # Limit history length (e.g., 20 messages = 10 turns)
            raw_hist = await current_loop.run_in_executor(None, chat_db.get_messages_for_conversation, final_conversation_id, 20, 0, "ASC")
            for db_msg in raw_hist:
                role = "user" if db_msg.get("sender", "").lower() == "user" else "assistant"
                char_name_hist = character_card_for_context.get('name', "Char") if character_card_for_context else "Char"

                text_content = db_msg.get("content", "")
                if text_content: # Apply placeholder replacement
                     text_content = replace_placeholders(text_content, char_name_hist, "User") # Assuming "User" for {{user}} placeholder

                msg_parts = []
                if text_content:
                    msg_parts.append({"type": "text", "text": text_content})

                img_data, img_mime = db_msg.get("image_data"), db_msg.get("image_mime_type")
                if img_data and img_mime:
                    try:
                        b64_img = await current_loop.run_in_executor(None, base64.b64encode, img_data)
                        msg_parts.append({"type": "image_url", "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}"}})
                    except Exception as e: logger.warning(f"Error encoding DB image for history (msg_id {db_msg.get('id')}): {e}")

                if msg_parts:
                    hist_entry = {"role": role, "content": msg_parts}
                    if role == "assistant" and character_card_for_context and character_card_for_context.get('name'):
                        hist_entry["name"] = character_card_for_context.get('name')
                    historical_openai_messages.append(hist_entry)
            logger.info(f"Loaded {len(historical_openai_messages)} historical messages for conv_id '{final_conversation_id}'.")

        # --- User Message Processing & DB Save ---
        current_turn_messages_for_llm: List[Dict[str, Any]] = []
        for msg_model in request_data.messages:
            if msg_model.role == "system": continue # Handled by templating

            msg_dict = msg_model.model_dump(exclude_none=True)
            msg_for_db = msg_dict.copy()
            if msg_model.role == "assistant" and character_card_for_context:
                msg_for_db["name"] = character_card_for_context.get('name', "Assistant")

            await _save_message_turn_to_db(chat_db, final_conversation_id, msg_for_db) # Already handles errors internally

            msg_for_llm = msg_dict.copy()
            if msg_model.role == "assistant" and character_card_for_context and character_card_for_context.get('name'):
                msg_for_llm["name"] = character_card_for_context.get('name')
            current_turn_messages_for_llm.append(msg_for_llm)

        # --- Prompt Templating ---
        llm_payload_messages = historical_openai_messages + current_turn_messages_for_llm
        active_template = load_template(request_data.prompt_template_name or DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name)
        template_data: Dict[str, Any] = {}
        if character_card_for_context:
            template_data.update({k: v for k, v in character_card_for_context.items() if isinstance(v, (str, int, float))}) # Basic fields
            template_data["char_name"] = character_card_for_context.get("name", "Character") # Ensure common alias
            # Add specific character fields used by templates if not covered by above
            template_data["character_system_prompt"] = character_card_for_context.get('system_prompt', "")

        sys_msg_from_req = next((m.content for m in request_data.messages if m.role == 'system' and isinstance(m.content, str)), "")
        template_data["original_system_message_from_request"] = sys_msg_from_req

        final_system_message: Optional[str] = None
        if active_template and active_template.system_message_template:
            final_system_message = apply_template_to_string(active_template.system_message_template, template_data)
        elif sys_msg_from_req:
            final_system_message = sys_msg_from_req

        templated_llm_payload: List[Dict[str, Any]] = []
        # ... (Rest of templating logic for user/assistant messages - simplified for brevity, assume it's complex and correct)
        # This logic should be efficient or offloaded if it becomes a bottleneck for large histories/contents.
        # For now, assume original logic is mostly sound but ensure it handles content lists correctly.
        if active_template:
            for msg in llm_payload_messages:
                templated_msg_content = msg.get("content")
                role = msg.get("role")
                content_template_str = None
                if role == "user" and active_template.user_message_content_template:
                    content_template_str = active_template.user_message_content_template
                elif role == "assistant" and active_template.assistant_message_content_template:
                    content_template_str = active_template.assistant_message_content_template

                if content_template_str:
                    new_content_parts = []
                    msg_template_data = template_data.copy()
                    if isinstance(templated_msg_content, str):
                        msg_template_data["message_content"] = templated_msg_content
                        new_text = apply_template_to_string(content_template_str, msg_template_data)
                        new_content_parts.append({"type": "text", "text": new_text or templated_msg_content})
                    elif isinstance(templated_msg_content, list):
                        for part in templated_msg_content:
                            if part.get("type") == "text":
                                msg_template_data["message_content"] = part.get("text", "")
                                new_text_part = apply_template_to_string(content_template_str, msg_template_data)
                                new_content_parts.append({"type": "text", "text": new_text_part or part.get("text", "")})
                            else:
                                new_content_parts.append(part) # Keep image parts
                    templated_llm_payload.append({**msg, "content": new_content_parts or templated_msg_content}) # type: ignore
                else:
                    templated_llm_payload.append(msg)
        else:
            templated_llm_payload = llm_payload_messages

        # --- LLM Call ---
        call_params = request_data.model_dump(
            exclude_none=True,
            exclude={"api_provider", "messages", "character_id", "conversation_id", "prompt_template_name", "stream"}
        )

        # Rename keys to match chat_api_call's signature for generic params
        if "temperature" in call_params:
            call_params["temp"] = call_params.pop("temperature")

        if "top_p" in call_params:
            top_p_value = call_params.pop("top_p")
            # chat_api_call has 'topp' and 'maxp' which both relate to top_p sampling.
            # Pass the value to both, let PROVIDER_PARAM_MAP in chat_api_call pick the relevant one.
            call_params["topp"] = top_p_value
            call_params["maxp"] = top_p_value

        if "user" in call_params:
            call_params["user_identifier"] = call_params.pop("user")

        # response_format, tools, tool_choice are already dict/list of dicts/str from model_dump if not None.
        # They match the expected names in chat_api_call signature.

        # Add other fixed arguments
        call_params.update({
            "api_endpoint": target_api_provider,
            "api_key": provider_api_key,
            "messages_payload": templated_llm_payload,
            "system_message": final_system_message,  # This can be None
            "streaming": request_data.stream,  # This is a boolean
        })

        # Filter out None values before making the call, as chat_api_call's defaults handle Nones.
        # The previous `cleaned_args` did this.
        cleaned_args = {k: v for k, v in call_params.items() if v is not None}

        llm_call_func = partial(perform_chat_api_call, **cleaned_args)

        if request_data.stream:
            raw_stream_iter = await current_loop.run_in_executor(None, llm_call_func)
            if not (hasattr(raw_stream_iter, "__aiter__") or hasattr(raw_stream_iter, "__iter__")):
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Provider did not return a valid stream.")

            _sse_state = {'full_reply': None} # Mutable state for sse_event_generator

            async def sse_event_generator(
                shared_state: dict, stream: Union[Iterator, AsyncIterator], model_name: str
            ): # Issue 4 Fix
                reply_parts = []
                try:
                    item_iterator = stream.__aiter__() if hasattr(stream, "__aiter__") else stream.__iter__() # type: ignore
                    while True:
                        try:
                            chunk = await item_iterator.__anext__() if hasattr(stream, "__aiter__") else next(item_iterator) # type: ignore
                            text_piece = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else str(chunk)
                            if text_piece:
                                reply_parts.append(text_piece)
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': text_piece}}]})}\n\n"
                        except (StopIteration, StopAsyncIteration): break
                except asyncio.CancelledError:
                    logger.info(f"SSE stream cancelled for conv_id {final_conversation_id}.")
                    # Do not re-raise; allow finally to run. Starlette handles actual cancellation.
                except Exception as e:
                    logger.error(f"Error during SSE streaming for conv_id {final_conversation_id}: {e}", exc_info=True)
                    yield f"data: {json.dumps({'error': {'message': 'Stream failed due to provider error.'}})}\n\n"
                finally:
                    shared_state['full_reply'] = "".join(reply_parts)
                    done_payload = { # OpenAI-like [DONE] message
                        "id": f"chatcmpl-{datetime.datetime.now(datetime.timezone.utc).timestamp()}", "object": "chat.completion.chunk",
                        "created": int(datetime.datetime.now(datetime.timezone.utc).timestamp()), "model": model_name,
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        "tldw_conversation_id": final_conversation_id
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"

            async def final_save_bg_task(): # Issue 5 & 7 Fix
                full_reply = _sse_state.get('full_reply')
                if full_reply and final_conversation_id: # Ensure final_conversation_id is still valid
                    asst_name = character_card_for_context.get("name", "Assistant") if character_card_for_context else "Assistant"
                    logger.info(f"BG Task: Saving assistant reply (len {len(full_reply)}) for conv_id {final_conversation_id}")
                    await _save_message_turn_to_db(chat_db, final_conversation_id, {"role": "assistant", "name": asst_name, "content": full_reply})
                else:
                    logger.info(f"BG Task: No assistant reply or conv_id to save for conv_id {final_conversation_id}.")

            async def combined_streamer():
                yield f"event: tldw_metadata\ndata: {json.dumps({'conversation_id': final_conversation_id, 'model': request_data.model})}\n\n"
                async for item in sse_event_generator(_sse_state, raw_stream_iter, request_data.model):
                    yield item

            return StreamingResponse(combined_streamer(), media_type="text/event-stream", background=BackgroundTask(final_save_bg_task))

        else: # Non-streaming
            llm_response = await current_loop.run_in_executor(None, llm_call_func)
            content_to_save: Optional[str] = None
            if isinstance(llm_response, dict): # OpenAI-like
                content_to_save = llm_response.get("choices", [{}])[0].get("message", {}).get("content")
            elif isinstance(llm_response, str):
                content_to_save = llm_response

            if content_to_save:
                asst_name = character_card_for_context.get("name", "Assistant") if character_card_for_context else "Assistant"
                await _save_message_turn_to_db(chat_db, final_conversation_id, {"role": "assistant", "name": asst_name, "content": content_to_save})

            # Issue 6 Fix: Offload jsonable_encoder
            encoded_payload = await current_loop.run_in_executor(None, jsonable_encoder, llm_response)
            if isinstance(encoded_payload, dict): # Ensure it's a dict to add custom fields
                encoded_payload["tldw_conversation_id"] = final_conversation_id
            return JSONResponse(content=encoded_payload)

    # --- Exception Handling --- Issue 8 Fix: Mask internal details in 5xx
    except HTTPException as e_http:
        if e_http.status_code >= 500: logger.error(f"HTTPException (Server Error): {e_http.status_code} - {e_http.detail}", exc_info=True)
        else: logger.warning(f"HTTPException (Client Error): {e_http.status_code} - {e_http.detail}")
        raise e_http # Re-raise, details are assumed to be client-safe or intentionally set

    except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError, ChatConfigurationError, ChatProviderError, ChatAPIError) as e_chat:
        status_code_map = { ChatAuthenticationError: 401, ChatRateLimitError: 429, ChatBadRequestError: 400,
                            ChatConfigurationError: 503, ChatProviderError: getattr(e_chat, 'status_code', 502),
                            ChatAPIError: getattr(e_chat, 'status_code', 500) }
        err_status = status_code_map.get(type(e_chat), 500)
        logger.error(f"Chat Library Error: {type(e_chat).__name__} - '{e_chat.message}' (Provider: {e_chat.provider}, UpstreamStatus: {getattr(e_chat, 'status_code', 'N/A')})", exc_info=True)
        client_detail = e_chat.message if err_status < 500 else "An error occurred with the chat provider or service configuration."
        raise HTTPException(status_code=err_status, detail=client_detail)

    except (InputError, ConflictError, CharactersRAGDBError) as e_db:
        logger.error(f"Database Error: {type(e_db).__name__} - {str(e_db)}", exc_info=True)
        err_status = status.HTTP_400_BAD_REQUEST if isinstance(e_db, InputError) else \
                     status.HTTP_409_CONFLICT if isinstance(e_db, ConflictError) else \
                     status.HTTP_500_INTERNAL_SERVER_ERROR
        client_detail = str(e_db) if err_status < 500 else "A database error occurred."
        raise HTTPException(status_code=err_status, detail=client_detail)

    except Exception as e_final:
        logger.critical(f"Unexpected Critical Error in /completions: {type(e_final).__name__} - {str(e_final)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred.")


#
# End of chat.py
#######################################################################################################################