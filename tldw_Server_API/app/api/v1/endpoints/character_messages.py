# character_messages.py
"""
API endpoints for message management within character chat sessions.
Provides CRUD operations for messages in conversations.
"""

from datetime import datetime, timezone
from typing import Any, Literal, Optional
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response, status
from loguru import logger
from pydantic import ValidationError

# Database and authentication dependencies
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

# Schemas
from tldw_Server_API.app.api.v1.schemas.chat_conversation_schemas import (
    ConversationScopeParams,
)
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    MessageUpdate,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User

# Character chat helpers
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import (
    edit_message_content,
    find_messages_in_conversation,
    map_sender_to_role,
    post_message_to_conversation,
    remove_message_from_conversation,
    replace_placeholders,
    retrieve_message_details,
)

# Rate limiting
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import get_character_rate_limiter
from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    build_character_system_prompt,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SyncServerOriginIdempotencyConflictError,
    SyncServerOriginMaterializationError,
    SyncServerOriginMutationNotSupportedError,
    capture_server_origin_mutation,
    get_active_server_origin_sync_service_for_user,
    server_origin_object_id,
    server_origin_stable_key,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service

_CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _detect_image_mime_type(data: bytes) -> Optional[str]:
    """
    Detect image MIME type from magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        MIME type string if recognized image format, None otherwise
    """
    if not data or len(data) < 12:
        return None

    # Check PNG
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'

    # Check JPEG (multiple signatures)
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'

    # Check GIF
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'

    # Check WebP (RIFF....WEBP)
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'

    # Check BMP
    if data[:2] == b'BM':
        return 'image/bmp'

    # Check ICO
    if data[:4] == b'\x00\x00\x01\x00':
        return 'image/x-icon'

    return None


router = APIRouter()

# ========================================================================
# Helper Functions
# ========================================================================

def _convert_db_message_to_response(msg_data: dict[str, Any]) -> MessageResponse:
    """Convert database message to response model."""
    return MessageResponse(
        id=msg_data.get('id', ''),
        conversation_id=msg_data.get('conversation_id', ''),
        parent_message_id=msg_data.get('parent_message_id'),
        sender=msg_data.get('sender', ''),
        content=msg_data.get('content') or '',
        timestamp=msg_data.get('timestamp', datetime.now(timezone.utc)),
        ranking=msg_data.get('ranking'),
        has_image=bool(msg_data.get('image_data')),
        version=msg_data.get('version', 1)
    )


def _message_sync_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SyncServerOriginIdempotencyConflictError):
        envelope = exc.envelope
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "sync_server_origin_idempotency_conflict",
                "message": "The idempotency key was already used for a different chat message change.",
                "server_cursor": envelope.server_cursor,
                "apply_status": envelope.apply_status,
            },
        )
    if isinstance(exc, SyncServerOriginMaterializationError):
        envelope = exc.envelope
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "sync_server_origin_materialization_failed",
                "message": "Sync accepted the server-origin message change but projection apply failed.",
                "server_cursor": envelope.server_cursor,
                "apply_status": envelope.apply_status,
                "apply_error_code": envelope.apply_error_code,
                "apply_error_message": envelope.apply_error_message,
            },
        )
    if isinstance(exc, SyncServerOriginMutationNotSupportedError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": exc.error_code,
                "message": str(exc),
                "dataset_id": exc.dataset.dataset_id,
                "domain": exc.domain,
            },
        )
    if isinstance(exc, SyncStoreError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "sync_server_origin_append_failed",
                "message": "Sync could not record the server-origin message change.",
            },
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error_code": "sync_server_origin_failed",
            "message": "Sync failed while recording the server-origin message change.",
        },
    )


def _active_message_sync_service(
    current_user: User,
    scope: ConversationScopeParams,
) -> SyncV2Service | None:
    if scope.scope_type == "workspace":
        return None
    return get_active_server_origin_sync_service_for_user(str(current_user.id))


def _verify_conversation_access(
    db: CharactersRAGDB,
    conversation_id: str,
    user_id: int,
    scope: ConversationScopeParams | None = None,
) -> dict[str, Any]:
    """
    Verify user has access to a conversation.

    Args:
        db: Database instance
        conversation_id: Conversation ID to check
        user_id: User ID to verify

    Returns:
        Conversation data if access allowed

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    conversation = db.get_conversation_by_id(conversation_id)

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {conversation_id} not found"
        )

    stored_client_id = str(conversation.get('client_id', '')).strip()
    request_user_id = str(user_id).strip()
    if stored_client_id != request_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this chat session"
        )
    expected_scope = scope or ConversationScopeParams()
    conversation_scope = conversation.get("scope_type") or "global"
    conversation_workspace_id = conversation.get("workspace_id")
    if conversation_scope != expected_scope.scope_type:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {conversation_id} not found",
        )
    if (
        expected_scope.scope_type == "workspace"
        and conversation_workspace_id != expected_scope.workspace_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {conversation_id} not found",
        )

    return conversation

def _verify_message_access(
    db: CharactersRAGDB,
    message_id: str,
    user_id: int,
    scope: ConversationScopeParams | None = None,
) -> dict[str, Any]:
    """
    Verify user has access to a message using DB abstractions.
    """
    message = db.get_message_by_id(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found"
        )
    conv = _verify_conversation_access(
        db,
        str(message.get('conversation_id')),
        user_id,
        scope,
    )
    message['client_id'] = conv.get('client_id')
    return message


def _resolve_message_scope(
    scope_type: Literal["global", "workspace"] | None,
    workspace_id: str | None,
) -> ConversationScopeParams:
    try:
        return ConversationScopeParams(
            scope_type=scope_type or "global",
            workspace_id=workspace_id,
        )
    except ValidationError as exc:
        detail = exc.errors()[0].get("msg") if exc.errors() else str(exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from exc

# ========================================================================
# Message Endpoints
# ========================================================================

@router.post("/chats/{chat_id}/messages", response_model=MessageResponse,
             status_code=status.HTTP_201_CREATED,
             summary="Send a message in a chat", tags=["Messages"])
async def send_message(
    message_data: MessageCreate,
    chat_id: str = Path(..., description="Chat session ID"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """
    Add a new message to a chat session.

    Args:
        chat_id: Chat session ID
        message_data: Message content and metadata
        db: Database instance
        current_user: Authenticated user

    Returns:
        Created message details

    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized, 429 if rate limited
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        # Check rate limits (per-minute + per-chat message count)
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_message_send_rate(current_user.id)

        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id, scope)
        # Enforce per-chat message cap (using efficient count instead of loading all messages)
        try:
            msg_count = db.count_messages_for_conversation(chat_id)
            await rate_limiter.check_message_limit(chat_id, msg_count + 1)
        except HTTPException:
            raise
        except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(
                "count_messages_for_conversation failed for chat_id={} include_deleted={} error={}",
                chat_id,
                False,
                e,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to validate chat message limit right now.",
            ) from e

        # Validate parent message if provided
        if message_data.parent_message_id:
            parent_msg = _verify_message_access(db, message_data.parent_message_id, current_user.id, scope)
            if parent_msg.get('conversation_id') != chat_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message must be from the same conversation"
                )

        # Map role to sender format used in database
        sender_map = {
            "user": "user",
            "assistant": "assistant",
            "system": "system"
        }
        sender_override = sender_map.get(message_data.role, "user")
        is_user_message = message_data.role == "user"

        # Resolve character/user names for placeholder handling and message sender defaults
        character_id = conversation.get('character_id')
        character = db.get_character_card_by_id(character_id) if character_id else None
        character_name = character.get('name', 'Assistant') if character else 'Assistant'
        user_name = conversation.get('user_name', 'User')

        # Handle image if provided
        image_data = None
        image_mime_type = None
        if message_data.image_base64:
            try:
                import base64
                raw_b64 = message_data.image_base64
                if isinstance(raw_b64, str) and raw_b64.startswith("data:") and "base64," in raw_b64:
                    raw_b64 = raw_b64.split("base64,", 1)[1]
                if isinstance(raw_b64, str):
                    raw_b64 = "".join(raw_b64.split())
                    if not raw_b64:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid image data. Empty base64 payload."
                        )
                    raw_b64 = raw_b64 + ("=" * (-len(raw_b64) % 4))
                # Preflight size check before decoding/DB layer
                try:
                    _max_img_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
                except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS:
                    _max_img_bytes = 5 * 1024 * 1024
                if isinstance(raw_b64, (str, bytes, bytearray)):
                    max_b64_len = ((_max_img_bytes + 2) // 3) * 4
                    if len(raw_b64) > max_b64_len:
                        raise HTTPException(
                            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                            detail=f"Image too large. Max {_max_img_bytes} bytes allowed."
                        )
                img_data = base64.b64decode(raw_b64, validate=True)
                if isinstance(img_data, (bytes, bytearray)) and len(img_data) > _max_img_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Image too large. Max {_max_img_bytes} bytes allowed."
                    )

                # Validate image by detecting MIME type from magic bytes
                detected_mime = _detect_image_mime_type(img_data)
                if detected_mime is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid image data. Supported formats: PNG, JPEG, GIF, WebP, BMP, ICO."
                    )

                image_data = img_data
                image_mime_type = detected_mime
            except HTTPException:
                raise
            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to decode image data: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to decode image data. Please provide valid base64-encoded image."
                ) from e

        sync_service = _active_message_sync_service(current_user, scope)
        if sync_service is not None and image_data is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "sync_v2_binary_message_unsupported",
                    "message": "Sync v2 M1 does not support binary chat message attachments.",
                },
            )
        if sync_service is not None:
            created_id = server_origin_object_id("chat.message", idempotency_key) or str(uuid.uuid4())
            stable_key = server_origin_stable_key(
                source="server_api",
                domain="chat.message",
                operation="append",
                idempotency_key=idempotency_key,
            )
            timestamp = sync_service.clock() or datetime.now(timezone.utc).isoformat()
            try:
                capture_server_origin_mutation(
                    sync_service,
                    user_id=str(current_user.id),
                    domain="chat.message",
                    operation="append",
                    object_id=created_id,
                    parent_id=chat_id,
                    payload={
                        "conversation_id": chat_id,
                        "parent_message_id": message_data.parent_message_id,
                        "sender": sender_override,
                        "content": message_data.content,
                        "timestamp": timestamp,
                        "client_id": str(current_user.id),
                    },
                    source="server_api",
                    stable_key=stable_key,
                )
            except Exception as sync_exc:
                raise _message_sync_http_error(sync_exc) from sync_exc
        else:
            # Add to database via Character_Chat guardrails
            created_id = post_message_to_conversation(
                db=db,
                conversation_id=chat_id,
                character_name=character_name,
                message_content=message_data.content,
                is_user_message=is_user_message,
                parent_message_id=message_data.parent_message_id,
                image_data=image_data,
                image_mime_type=image_mime_type,
                sender_override=sender_override,
            )

            if not created_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create message"
                )

        # Retrieve created message with placeholder parameters
        created_msg = retrieve_message_details(db, created_id, character_name, user_name)

        logger.info(f"Created message {created_id} in chat {chat_id} by user {current_user.id}")

        return _convert_db_message_to_response(created_msg)

    except HTTPException:
        raise
    except ConflictError as e:
        # Optimistic lock or state conflict during creation
        logger.warning(f"Conflict sending message to chat {chat_id}: {e}")
        raise map_db_error_to_http(e) from e
    except InputError as exc:
        logger.warning(f"Input error sending message to chat {chat_id}: {exc}")
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to send message",
            payload_too_large_substrings=("exceeds maximum size",),
        ) from exc
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to send message") from exc
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error sending message to chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending message"
        ) from e


@router.get("/chats/{chat_id}/messages",
            summary="Get messages in a chat", tags=["Messages"])
async def get_chat_messages(
    chat_id: str = Path(..., description="Chat session ID"),
    limit: int = Query(50, ge=1, le=200, description="Number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    include_deleted: bool = Query(False, description="Include deleted messages"),
    include_character_context: bool = Query(False, description="Include character context for chat completions"),
    format_for_completions: bool = Query(False, description="Format messages for use with chat/completions endpoint"),
    include_tool_calls: bool = Query(False, description="Include tool_calls metadata per message when available (standard format only)"),
    include_metadata: bool = Query(False, description="Include stored message metadata.extra JSON where available"),
    include_message_ids: bool = Query(
        False,
        description="Include message_id fields when formatting for completions (no effect on standard format)",
    ),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get messages from a chat session.

    Args:
        chat_id: Chat session ID
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        include_deleted: Whether to include soft-deleted messages
        include_character_context: Include character personality as system message
        format_for_completions: Return in format ready for /api/v1/chat/completions
        include_message_ids: Include message_id fields only in completions-formatted output
        db: Database instance
        current_user: Authenticated user

    Returns:
        List of messages with pagination info, or formatted for completions if requested

    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id, scope)

        # Get messages (honor include_deleted and DB pagination)
        messages = db.get_messages_for_conversation(chat_id, limit=limit, offset=offset, include_deleted=include_deleted)

        if not messages:
            messages = []
        paginated = messages

        # Compute total message count for pagination metadata
        try:
            total_count = db.count_messages_for_conversation(chat_id, include_deleted=include_deleted)
        except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(
                "count_messages_for_conversation failed for chat_id={} include_deleted={} error={}",
                chat_id,
                include_deleted,
                e,
                exc_info=True,
            )
            total_count = len(messages)

        # If character context or completions format requested
        if include_character_context or format_for_completions:
            # Get character info
            character_id = conversation.get('character_id')
            character = db.get_character_card_by_id(character_id) if character_id else None
            character_name = character.get('name', 'Assistant') if character else 'Assistant'
            user_name = conversation.get('user_name', 'User')

            def _replace_text(value: Any) -> str:
                if value is None:
                    return ""
                text = value if isinstance(value, str) else str(value)
                if not text:
                    return ""
                try:
                    return replace_placeholders(text, character_name, user_name)
                except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Placeholder replacement failed for chat_id={} value_type={}: {}",
                        chat_id,
                        type(value).__name__,
                        exc,
                    )
                    return text

            if format_for_completions:
                # Return format ready for chat completions endpoint
                formatted_messages = []
                metadata_extra_map: dict[str, Any] = {}

                # Add system prompt only on the first page and only if no system message exists in DB
                if character and include_character_context and offset == 0:
                    has_system_in_db = db.has_system_message_for_conversation(
                        chat_id,
                        include_deleted=include_deleted,
                    )
                    if not has_system_in_db:
                        system_prompt = build_character_system_prompt(
                            character,
                            character_name,
                            user_name,
                        )
                        formatted_messages.append({
                            "role": "system",
                            "content": system_prompt.strip()
                        })

                # Add conversation messages with optional tool role messages
                import re as _re
                _suffix_re = _re.compile(r"\[tool_calls\]\s*:\s*(\{.*|\[.*)$", _re.DOTALL)
                for msg in paginated:
                    role = map_sender_to_role(
                        msg.get('sender'),
                        character_name if character else None,
                        default_role="user",
                    )
                    content = _replace_text(msg.get('content'))
                    msg_id = msg.get('id')

                    base_message: dict[str, Any] = {"role": role, "content": content}
                    if include_message_ids and msg_id:
                        base_message["message_id"] = msg_id

                    # If assistant message has tool_calls in metadata, include tool role messages (OpenAI-compatible)
                    md = None
                    try:
                        md = db.get_message_metadata(msg_id)
                    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS:
                        md = None

                    if include_metadata and md and md.get('extra') is not None and msg_id:
                        metadata_extra_map[msg_id] = md.get('extra')

                    # Add the original message first
                    tool_calls_list = None
                    if role == 'assistant':
                        if md and isinstance(md.get('tool_calls'), list) and md.get('tool_calls'):
                            tool_calls_list = md.get('tool_calls')
                        else:
                            # Fallback: parse inline suffix if present
                            try:
                                match = _suffix_re.search(content or '')
                                if match:
                                    import json as _json
                                    parsed = _json.loads(match.group(1).strip())
                                    if isinstance(parsed, dict) and 'tool_calls' in parsed:
                                        tool_calls_list = parsed.get('tool_calls')
                                    else:
                                        tool_calls_list = parsed
                                    if not isinstance(tool_calls_list, list):
                                        tool_calls_list = None
                            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"character_messages: failed to parse tool_calls from suffix: {e}")
                                tool_calls_list = None

                    if role == 'assistant' and tool_calls_list:
                        # Optionally include tool_calls array on assistant message for completeness
                        base_message_with_tools = dict(base_message)
                        base_message_with_tools["tool_calls"] = tool_calls_list
                        formatted_messages.append(base_message_with_tools)

                        # Emit tool role messages after the assistant message
                        tool_results_by_id: dict[str, Any] = {}
                        try:
                            extra = md.get('extra') or {}
                            # Common pattern: extra.tool_results: { tool_call_id: { ... } }
                            tr = extra.get('tool_results') if isinstance(extra, dict) else None
                            if isinstance(tr, dict):
                                tool_results_by_id = tr
                        except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"character_messages: failed to extract tool_results: {e}")
                            tool_results_by_id = {}

                        for tc in tool_calls_list:
                            tc_id = None
                            tc_name = None
                            try:
                                tc_id = tc.get('id')
                                func = tc.get('function') or {}
                                tc_name = func.get('name')
                            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"character_messages: tool_call parse error: {e}")
                            tool_content = ""
                            # If we have stored results keyed by tool_call_id, include them
                            try:
                                if tc_id and tool_results_by_id.get(tc_id) is not None:
                                    # Convert result to string; JSON-encode if needed
                                    res = tool_results_by_id.get(tc_id)
                                    if isinstance(res, (dict, list)):
                                        import json as _json
                                        tool_content = _json.dumps(res)
                                    else:
                                        tool_content = str(res)
                            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"character_messages: failed to stringify tool result: {e}")
                            tool_msg: dict[str, Any] = {"role": "tool", "content": tool_content}
                            if tc_id:
                                tool_msg["tool_call_id"] = tc_id
                            if tc_name:
                                tool_msg["name"] = tc_name
                            formatted_messages.append(tool_msg)
                    else:
                        # No tools: append base message as-is
                        formatted_messages.append(base_message)

                pagination = build_offset_pagination_meta(
                    total=total_count,
                    limit=limit,
                    offset=offset,
                    count=len(paginated),
                )
                resp_obj: dict[str, Any] = {
                    "character_name": character.get('name') if character else None,
                    "character_id": character_id,
                    "chat_id": chat_id,
                    "messages": formatted_messages,
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "pagination": pagination.model_dump(mode="json"),
                    "has_more": pagination.has_more,
                    "next_offset": pagination.next_offset,
                    "usage_instructions": "Use these messages with POST /api/v1/chat/completions"
                }
                if include_metadata and metadata_extra_map:
                    # Provide sidecar of metadata.extra without polluting message objects
                    resp_obj["metadata_extra"] = metadata_extra_map
                return resp_obj

            # Otherwise return standard format with character info
            # Build standard response messages, optionally including tool_calls
            built_messages = []
            for m in paginated:
                msg_copy = dict(m)
                msg_copy["content"] = _replace_text(msg_copy.get("content"))
                resp = _convert_db_message_to_response(msg_copy)
                # Fetch metadata once if either flag is set (avoid duplicate queries)
                if include_tool_calls or include_metadata:
                    try:
                        md = db.get_message_metadata(resp.id)
                        if md:
                            updates = {}
                            if include_tool_calls and md.get('tool_calls') is not None:
                                updates["tool_calls"] = md.get('tool_calls')
                            if include_metadata and md.get('extra') is not None:
                                updates["metadata_extra"] = md.get('extra')
                            if updates:
                                resp = resp.model_copy(update=updates)
                    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"character_messages: failed to include metadata in response: {e}")
                built_messages.append(resp)
            response = MessageListResponse(
                messages=built_messages,
                total=total_count,
                limit=limit,
                offset=offset,
                pagination=build_offset_pagination_meta(
                    total=total_count,
                    limit=limit,
                    offset=offset,
                    count=len(built_messages),
                ),
            )

            # Add character context as additional field
            if character:
                response_dict = response.model_dump()
                response_dict['character_context'] = {
                    "name": character_name,
                    "description": _replace_text(character.get('description')),
                    "personality": _replace_text(character.get('personality')),
                    "system_prompt": _replace_text(character.get('system_prompt')),
                }
                return response_dict

            return response

        # Standard response (no character context)
        built_messages = []
        character_id = conversation.get('character_id')
        character = db.get_character_card_by_id(character_id) if character_id else None
        character_name = character.get('name', 'Assistant') if character else 'Assistant'
        user_name = conversation.get('user_name', 'User')

        def _replace_text_std(value: Any) -> str:
            if value is None:
                return ""
            text = value if isinstance(value, str) else str(value)
            if not text:
                return ""
            try:
                return replace_placeholders(text, character_name, user_name)
            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "Placeholder replacement failed for chat_id={} value_type={}: {}",
                    chat_id,
                    type(value).__name__,
                    exc,
                )
                return text

        for m in paginated:
            msg_copy = dict(m)
            msg_copy["content"] = _replace_text_std(msg_copy.get("content"))
            resp = _convert_db_message_to_response(msg_copy)
            # Fetch metadata once if either flag is set (avoid duplicate queries)
            if include_tool_calls or include_metadata:
                try:
                    md = db.get_message_metadata(resp.id)
                    if md:
                        updates = {}
                        if include_tool_calls and md.get('tool_calls') is not None:
                            updates["tool_calls"] = md.get('tool_calls')
                        if include_metadata and md.get('extra') is not None:
                            updates["metadata_extra"] = md.get('extra')
                        if updates:
                            resp = resp.model_copy(update=updates)
                except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"character_messages: failed to include metadata (std): {e}")
            built_messages.append(resp)
        return MessageListResponse(
            messages=built_messages,
            total=total_count,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total_count,
                limit=limit,
                offset=offset,
                count=len(built_messages),
            ),
        )

    except HTTPException:
        raise
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting messages for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving messages"
        ) from e


@router.get("/messages/{message_id}", response_model=MessageResponse,
            summary="Get a specific message", tags=["Messages"])
async def get_message(
    message_id: str = Path(..., description="Message ID"),
    include_tool_calls: bool = Query(False, description="Include tool_calls metadata when available"),
    include_metadata: bool = Query(False, description="Include stored message metadata.extra JSON where available"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get details of a specific message.

    Args:
        message_id: Message ID
        db: Database instance
        current_user: Authenticated user

    Returns:
        Message details

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        message = _verify_message_access(db, message_id, current_user.id, scope)
        resp = _convert_db_message_to_response(message)
        if include_tool_calls or include_metadata:
            try:
                md = db.get_message_metadata(resp.id)
                if include_tool_calls and md and md.get('tool_calls') is not None:
                    resp = resp.model_copy(update={"tool_calls": md.get('tool_calls')})
                if include_metadata and md and md.get('extra') is not None:
                    resp = resp.model_copy(update={"metadata_extra": md.get('extra')})
            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "Non-fatal: failed to load metadata for message {}: {}",
                    message_id,
                    exc,
                )
        return resp

    except HTTPException:
        raise
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving message"
        ) from e


@router.put("/messages/{message_id}", response_model=MessageResponse,
            summary="Edit a message", tags=["Messages"])
async def edit_message(
    update_data: MessageUpdate,
    message_id: str = Path(..., description="Message ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Edit message content and or metadata.

    Args:
        message_id: Message ID to edit
        update_data: New message content
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user

    Returns:
        Updated message details

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        # Check rate limits for message edits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "message_edit")

        # Verify message access
        message = _verify_message_access(db, message_id, current_user.id, scope)

        # Check version
        if message.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {message.get('version', 1)}"
            )
        if _active_message_sync_service(current_user, scope) is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "sync_v2_message_edit_not_supported",
                    "message": "Sync v2 M1 does not support editing chat messages.",
                },
            )

        content_updated = False
        if update_data.content is not None and str(update_data.content).strip():
            success = edit_message_content(
                db,
                message_id,
                update_data.content,
                expected_version,
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update message content",
                )
            content_updated = True

        metadata_updated = False
        if update_data.pinned is not None:
            metadata_updated = db.set_message_metadata_extra(
                message_id,
                {"pinned": bool(update_data.pinned)},
                merge=True,
            )
            if not metadata_updated:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update message metadata",
                )

        # Update conversation metadata (last_modified/version) via DB abstraction
        conv = db.get_conversation_by_id(message["conversation_id"])
        if conv and (content_updated or metadata_updated):
            try:
                db.update_conversation(
                    message["conversation_id"],
                    {},
                    conv.get("version", 1),
                )
            except (ConflictError, CharactersRAGDBError) as e:
                logger.warning(
                    "Non-fatal: failed to bump conversation metadata for {}: {}",
                    message["conversation_id"],
                    e,
                    exc_info=True,
                )

        # Get character details for placeholders
        conversation = db.get_conversation_by_id(message['conversation_id'])
        character_id = conversation.get('character_id') if conversation else None
        character = db.get_character_card_by_id(character_id) if character_id else None
        character_name = character.get('name', 'Assistant') if character else 'Assistant'
        user_name = conversation.get('user_name', 'User') if conversation else 'User'

        # Retrieve updated message with placeholder parameters
        updated_msg = retrieve_message_details(db, message_id, character_name, user_name)

        logger.info(f"Updated message {message_id} by user {current_user.id}")
        response_payload = _convert_db_message_to_response(updated_msg)
        if metadata_updated:
            try:
                metadata = db.get_message_metadata(message_id) or {}
                extra = metadata.get("extra")
                if isinstance(extra, dict):
                    response_payload = response_payload.model_copy(
                        update={"metadata_extra": extra}
                    )
            except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "Non-fatal: failed to include metadata in edit response for message {}: {}",
                    message_id,
                    exc,
                )
        return response_payload

    except HTTPException:
        raise
    except ConflictError as e:
        logger.warning(f"Conflict editing message {message_id}: {e}")
        raise map_db_error_to_http(e) from e
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to edit message") from exc
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error editing message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while editing message"
        ) from e


@router.delete(
    "/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a message",
    tags=["Messages"],
)
async def delete_message(
    message_id: str = Path(..., description="Message ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
) -> Response:
    """
    Soft delete a message from a conversation.

    Args:
        message_id: Message ID to delete
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        # Check rate limits for message deletions
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "message_delete")

        # Verify message access
        message = _verify_message_access(db, message_id, current_user.id, scope)

        # Check version
        if message.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {message.get('version', 1)}"
            )

        sync_service = _active_message_sync_service(current_user, scope)
        if sync_service is not None:
            try:
                capture_server_origin_mutation(
                    sync_service,
                    user_id=str(current_user.id),
                    domain="chat.message",
                    operation="tombstone",
                    object_id=message_id,
                    parent_id=str(message.get("conversation_id") or ""),
                    payload={
                        "id": message_id,
                        "deleted": True,
                        "conversation_id": str(message.get("conversation_id") or ""),
                        "client_id": str(current_user.id),
                        "owner_user_id": str(current_user.id),
                    },
                    source="server_api",
                )
            except Exception as sync_exc:
                raise _message_sync_http_error(sync_exc) from sync_exc
        else:
            # Soft delete the message
            success = remove_message_from_conversation(db, message_id, expected_version)

            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to delete message"
                )

        # Update conversation metadata (last_modified/version) via DB abstraction
        conv = db.get_conversation_by_id(message['conversation_id'])
        if conv and sync_service is None:
            try:
                db.update_conversation(message['conversation_id'], {}, conv.get('version', 1))
            except (ConflictError, CharactersRAGDBError) as e:
                logger.warning(
                    "Non-fatal: failed to bump conversation metadata for {}: {}",
                    message["conversation_id"],
                    e,
                    exc_info=True,
                )

        logger.info(f"Soft deleted message {message_id} by user {current_user.id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except HTTPException:
        raise
    except ConflictError as e:
        logger.warning(f"Conflict deleting message {message_id}: {e}")
        raise map_db_error_to_http(e) from e
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete message") from exc
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error deleting message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting message"
        ) from e


# Maximum search query length to prevent abuse
MAX_SEARCH_QUERY_LENGTH = 500


@router.get("/chats/{chat_id}/messages/search", response_model=MessageListResponse,
            summary="Search messages in a chat", tags=["Messages"])
async def search_messages(
    chat_id: str = Path(..., description="Chat session ID"),
    query: str = Query(..., description="Search query", min_length=1, max_length=MAX_SEARCH_QUERY_LENGTH),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for search pagination"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Search for messages in a chat session.

    Args:
        chat_id: Chat session ID
        query: Search query string
        limit: Maximum number of results
        offset: Offset for search pagination
        db: Database instance
        current_user: Authenticated user

    Returns:
        List of matching messages

    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        scope = _resolve_message_scope(scope_type, workspace_id)
        # Check rate limits for message search
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "message_search")

        # Verify conversation access
        conversation = _verify_conversation_access(db, chat_id, current_user.id, scope)

        # Resolve character/user names for placeholder-aware search
        character_id = conversation.get('character_id')
        character = db.get_character_card_by_id(character_id) if character_id else None
        character_name = character.get('name', 'Assistant') if character else 'Assistant'
        user_name = conversation.get('user_name', 'User')
        # Search messages with placeholder replacement
        results = find_messages_in_conversation(
            db,
            chat_id,
            query,
            character_name_for_placeholders=character_name,
            user_name_for_placeholders=user_name,
            limit=limit + 1,
            offset=offset,
        )

        if not results:
            results = []
        has_more = len(results) > limit
        page_results = results[:limit]

        return MessageListResponse(
            messages=[_convert_db_message_to_response(msg) for msg in page_results],
            total=len(page_results),
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=None,
                limit=limit,
                offset=offset,
                count=len(page_results),
                has_more=has_more,
            ),
        )

    except HTTPException:
        raise
    except _CHARACTER_MESSAGES_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error searching messages in chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while searching messages"
        ) from e


#
# End of character_messages.py
######################################################################################################################
