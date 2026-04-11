# character_chat_sessions.py
"""
API endpoints for character chat session management.
Provides CRUD operations for chat sessions and character-specific completions.
"""

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import secrets
import re
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Response,
    status,
)
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError

# Database and authentication dependencies
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.llm_routing_deps import (
    get_request_routing_decision_store,
)

# Schemas
from tldw_Server_API.app.api.v1.schemas.chat_conversation_schemas import (
    ConversationScopeParams,
)
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import (
    AuthorNoteInfoResponse,
    CharacterChatCompletionPrepRequest,
    CharacterChatCompletionPrepResponse,
    CharacterChatCompletionV2Request,
    CharacterChatCompletionV2Response,
    CharacterChatStreamPersistRequest,
    CharacterChatStreamPersistResponse,
    ChatLinkedResearchRunsListResponse,
    ChatSessionCreate,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionUpdate,
    ChatSettingsResponse,
    ChatSettingsUpdate,
    DiagnosticTurnEntry,
    GreetingItem,
    GreetingListResponse,
    GreetingSelectRequest,
    GreetingSelectResponse,
    LorebookDiagnosticExportResponse,
    MessageResponse,
    PresetCreate,
    PresetDetail,
    PresetListResponse,
    PresetTokenInfo,
    PresetUpdate,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    DEFAULT_LLM_PROVIDER,
)
from tldw_Server_API.app.api.v1.utils.deprecation import build_deprecation_headers
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    apply_llm_provider_overrides_to_listing,
    get_override_model_priority,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

# Character chat helpers
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import (
    map_sender_to_role,
    post_message_to_conversation,
    replace_placeholders,
)

# Rate limiting
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import (
    get_character_rate_limiter,
)

# Import shared constants
from tldw_Server_API.app.core.Character_Chat.constants import (
    MAX_STREAMING_BYTES,
    MAX_STREAMING_CHUNKS,
    MAX_TOOL_CALLS_COUNT,
    MAX_TOOL_CALLS_SIZE,
    THROTTLE_CACHE_MAX_KEYS,
    THROTTLE_STALE_SECONDS,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Character_Chat.modules.character_generation_presets import (
    resolve_character_generation_settings,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    DEFAULT_PROMPT_PRESET,
    build_character_system_prompt,
    build_custom_system_prompt,
    resolve_character_prompt_preset,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_utils import (
    sanitize_sender_name,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.Persona.exemplar_prompt_assembly import (
    assemble_persona_exemplar_prompt,
)

# Chat helpers and utilities
# For chat completions
from tldw_Server_API.app.core.Chat.chat_service import (
    is_model_known_for_provider,
    perform_chat_api_call,
    perform_chat_api_call_async,
    resolve_provider_and_model,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.LLM_Calls.routing import (
    InMemoryRoutingDecisionStore,
    RouterRequest,
    RoutingPolicy,
    RoutingUsageContext,
    build_provider_order_for_routing,
    flatten_provider_listing_for_routing,
    log_model_router_usage,
    resolve_routing_policy,
    route_model,
    select_llm_router_choice,
)
from tldw_Server_API.app.core.LLM_Calls.routing.candidate_pool import (
    build_candidate_pool,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.Research.service import ResearchService
from tldw_Server_API.app.core.LLM_Calls.sse import ensure_sse_line, normalize_provider_line, sse_done

# Completion schemas centralized in schemas/chat_session_schemas.py
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.Utils.common import parse_boolean
from tldw_Server_API.app.core.config import load_and_log_configs

from .llm_providers import get_configured_providers

_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS = (
    ChatAPIError,
    asyncio.CancelledError,
    asyncio.TimeoutError,
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
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    HTTPException,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

THROTTLE_WINDOW_SIZE = 100
MAX_CHAT_SETTINGS_BYTES = 200_000
MAX_AUTHOR_NOTE_CHARS = 20_000
DEFAULT_AUTO_SUMMARY_THRESHOLD_MESSAGES = 40
DEFAULT_AUTO_SUMMARY_WINDOW_MESSAGES = 12
MAX_AUTO_SUMMARY_LINES = 24
MAX_AUTO_SUMMARY_LINE_CHARS = 220
MAX_AUTO_SUMMARY_CONTENT_CHARS = 8_000

# Preserve the legacy patch point used by greeting tests without routing selection
# through the insecure stdlib PRNG.
_greeting_random = SimpleNamespace(choice=secrets.choice)
random = _greeting_random


@lru_cache(maxsize=1)
def _config_default_llm_provider() -> str | None:
    """Read default provider from config sections used by chat endpoints."""
    cfg = load_and_log_configs()
    if not isinstance(cfg, dict):
        return None

    def _extract(section: str) -> str | None:
        section_data = cfg.get(section)
        if not isinstance(section_data, dict):
            return None
        default_api = section_data.get("default_api")
        if not isinstance(default_api, str):
            return None
        value = default_api.strip()
        return value or None

    return _extract("llm_api_settings") or _extract("API")


def _is_char_chat_test_mode() -> bool:
    """Return True when running under test-mode semantics."""
    for env_key in ("TEST_MODE", "TLDW_TEST_MODE", "PYTEST_CURRENT_TEST"):
        raw = os.getenv(env_key)
        if isinstance(raw, str) and raw.strip():
            if env_key == "PYTEST_CURRENT_TEST":
                return True
            if is_truthy(raw):
                return True
    return False


def _get_default_provider() -> str:
    """Resolve default provider: config, then env, then test fallback, then schema default."""
    cfg_default = _config_default_llm_provider()
    if cfg_default:
        return cfg_default

    env_default = os.getenv("DEFAULT_LLM_PROVIDER")
    if isinstance(env_default, str) and env_default.strip():
        return env_default.strip()

    if _is_char_chat_test_mode():
        return "local-llm"

    return DEFAULT_LLM_PROVIDER


def _should_enforce_char_chat_strict_model_selection() -> bool:
    """Return whether explicit model/provider requests should be strictly enforced."""
    raw = os.getenv("CHAT_ENFORCE_STRICT_MODEL_SELECTION")
    if raw is not None:
        return is_truthy(raw)
    return not _is_char_chat_test_mode()

def _safe_replace_placeholders(value: Any, char_name: str, user_name: str) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if not text:
        return ""
    try:
        return replace_placeholders(text, char_name, user_name)
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Placeholder replacement failed: {}", exc)
        return text


def _extract_character_latest_user_turn_text(
    messages: list[dict[str, Any]],
    *,
    appended_user_message: str | None = None,
) -> str:
    """Return the latest user turn text for auto-routing decisions."""
    if isinstance(appended_user_message, str) and appended_user_message.strip():
        return appended_user_message.strip()

    for message in reversed(messages or []):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if str(part.get("type") or "").strip().lower() != "text":
                    continue
                text = str(part.get("text") or "").strip()
                if text:
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
    return ""


def _character_request_uses_vision_input(messages: list[dict[str, Any]]) -> bool:
    """Return True when the completion payload includes image parts."""
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if str(part.get("type") or "").strip().lower() == "image_url":
                return True
    return False


def _extract_character_routing_requested_capabilities(
    *,
    body: CharacterChatCompletionV2Request,
    formatted_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive hard routing capability filters from a character-chat request."""
    return {
        "tools": bool(body.tools),
        "vision": _character_request_uses_vision_input(formatted_messages),
        "json_mode": False,
        "reasoning": False,
    }


async def _select_auto_character_llm_router_choice(
    *,
    router_request: RouterRequest,
    policy: RoutingPolicy,
    candidates: list[dict[str, Any]],
    provider_listing: dict[str, Any],
    current_user: User | None,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Select a concrete router-model choice for character-chat auto routing."""

    def _fallback_resolver(name: str) -> Optional[str]:
        try:
            from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys

            return get_api_keys().get(name)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            return None

    user_id_int: Optional[int] = None
    if hasattr(current_user, "id_int"):
        user_id_int = current_user.id_int
    elif hasattr(current_user, "id"):
        with contextlib.suppress(TypeError, ValueError):
            user_id_int = int(current_user.id)

    async def _execute_router_call(router_model, router_messages):
        byok_resolution = await resolve_byok_credentials(
            router_model.provider,
            user_id=user_id_int,
            fallback_resolver=_fallback_resolver,
        )
        try:
            return await perform_chat_api_call_async(
                api_endpoint=router_model.provider,
                messages_payload=router_messages,
                api_key=byok_resolution.api_key,
                model=router_model.model,
                max_tokens=64,
                streaming=False,
                user_identifier=str(getattr(current_user, "id", "auto-router")),
                app_config=byok_resolution.app_config,
            )
        finally:
            await byok_resolution.touch_last_used()

    async def _log_router_usage(router_model, usage, latency_ms):
        try:
            await log_model_router_usage(
                context=RoutingUsageContext(
                    surface="character_chat",
                    endpoint="POST:/api/v1/chats/{chat_id}/complete-v2",
                    user_id=user_id_int,
                    conversation_id=router_request.scope,
                ),
                provider=router_model.provider,
                model=router_model.model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                latency_ms=latency_ms,
                estimated=usage["total_tokens"] == 0,
            )
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Auto character-chat router usage logging skipped: {}", exc)

    try:
        return await select_llm_router_choice(
            router_request=router_request,
            policy=policy,
            candidates=candidates,
            provider_listing=provider_listing,
            execute_router_call=_execute_router_call,
            log_router_usage=_log_router_usage,
        )
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Auto character-chat LLM router call failed: {}", exc)
        return None, {"error": type(exc).__name__}


async def _resolve_auto_character_chat_routing_decision(
    *,
    chat_id: str,
    body: CharacterChatCompletionV2Request,
    raw_provider: str | None,
    formatted_messages: list[dict[str, Any]],
    sticky_store: InMemoryRoutingDecisionStore,
    current_user: User | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Resolve `model='auto'` into a canonical provider/model pair for character chat."""
    provider_listing = apply_llm_provider_overrides_to_listing(get_configured_providers())
    default_provider = str(
        provider_listing.get("default_provider") or _get_default_provider()
    ).strip().lower() or _get_default_provider()
    policy = resolve_routing_policy(
        request_model=str(body.model or ""),
        explicit_provider=raw_provider,
        routing_override=body.routing,
        server_default_provider=default_provider,
    )
    requested_capabilities = _extract_character_routing_requested_capabilities(
        body=body,
        formatted_messages=formatted_messages,
    )
    candidates = build_candidate_pool(
        boundary_mode=policy.boundary_mode,
        pinned_provider=policy.pinned_provider,
        server_default_provider=policy.server_default_provider,
        requested_capabilities=requested_capabilities,
        catalog=flatten_provider_listing_for_routing(provider_listing),
    )
    router_request = RouterRequest(
        model="auto",
        surface="character_chat",
        latest_user_turn=_extract_character_latest_user_turn_text(
            formatted_messages,
            appended_user_message=body.append_user_message,
        ),
        scope=chat_id,
        requested_capabilities=requested_capabilities,
        routing_context={
            "stream": bool(body.stream),
            "include_character_context": bool(body.include_character_context),
            "directed_character_id": body.directed_character_id,
        },
    )
    llm_router_choice, llm_router_debug = await _select_auto_character_llm_router_choice(
        router_request=router_request,
        policy=policy,
        candidates=candidates,
        provider_listing=provider_listing,
        current_user=current_user,
    )
    decision = route_model(
        request=router_request,
        policy=policy,
        candidates=candidates,
        sticky_store=sticky_store,
        llm_router_choice=llm_router_choice,
        provider_order=build_provider_order_for_routing(
            provider_listing,
            objective=policy.objective,
            priority_resolver=get_override_model_priority,
        ),
    )
    return decision, {
        "policy": {
            "boundary_mode": policy.boundary_mode,
            "pinned_provider": policy.pinned_provider,
            "server_default_provider": policy.server_default_provider,
            "objective": policy.objective,
            "mode": policy.mode,
            "strategy": policy.strategy,
            "failure_mode": policy.failure_mode,
        },
        "candidate_count": len(candidates),
        "llm_router": llm_router_debug,
    }

def _validate_and_truncate_tool_calls(tool_calls: Any) -> Optional[list]:
    """
    Validate and truncate tool_calls to prevent unbounded storage.

    Args:
        tool_calls: Tool calls data from LLM response

    Returns:
        Validated and potentially truncated tool_calls, or None if invalid
    """
    if tool_calls is None:
        return None

    if not isinstance(tool_calls, list):
        logger.warning("tool_calls is not a list, discarding")
        return None

    # Limit number of tool calls
    if len(tool_calls) > MAX_TOOL_CALLS_COUNT:
        logger.warning(f"Truncating tool_calls from {len(tool_calls)} to {MAX_TOOL_CALLS_COUNT}")
        tool_calls = tool_calls[:MAX_TOOL_CALLS_COUNT]

    # Check total serialized size
    try:
        serialized = json.dumps(tool_calls)
        if len(serialized) > MAX_TOOL_CALLS_SIZE:
            logger.warning(f"tool_calls exceeds size limit ({len(serialized)} > {MAX_TOOL_CALLS_SIZE}), truncating")
            # Progressively remove tool calls until within size limit
            while tool_calls and len(serialized) > MAX_TOOL_CALLS_SIZE:
                tool_calls = tool_calls[:-1]
                serialized = json.dumps(tool_calls)
            if not tool_calls:
                return None
    except (TypeError, ValueError) as e:
        logger.warning(f"tool_calls not JSON serializable: {e}")
        return None

    return tool_calls


def _build_stream_persist_fingerprint(
    chat_id: str,
    assistant_content: str,
    user_message_id: str | None,
    speaker_character_id: int | None,
) -> str:
    """Build a deterministic fingerprint for streamed persist retries."""
    payload = {
        "chat_id": chat_id,
        "assistant_content": assistant_content.strip(),
        "user_message_id": user_message_id,
        "speaker_character_id": speaker_character_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_existing_stream_persist_message_by_id(
    db: CharactersRAGDB,
    chat_id: str,
    assistant_message_id: str,
    *,
    assistant_content: str,
    parent_message_id: str | None,
    speaker_name: str,
) -> tuple[str, dict[str, Any]] | None:
    """Return an existing persisted assistant message by stable message id."""
    message = db.get_message_by_id(assistant_message_id)
    if not message:
        return None
    if str(message.get("conversation_id") or "").strip() != str(chat_id).strip():
        raise ConflictError("assistant_message_id already exists in another conversation")
    if message.get("deleted"):
        raise ConflictError("assistant_message_id references a deleted message")
    existing_sender = str(message.get("sender") or "").strip()
    if existing_sender and existing_sender != speaker_name:
        raise ConflictError("assistant_message_id already exists for a different speaker")
    if str(message.get("content") or "") != assistant_content:
        raise ConflictError("assistant_message_id already exists for different content")
    existing_parent_message_id = str(message.get("parent_message_id") or "").strip() or None
    expected_parent_message_id = str(parent_message_id or "").strip() or None
    if existing_parent_message_id != expected_parent_message_id:
        raise ConflictError("assistant_message_id already exists for a different parent message")
    metadata = db.get_message_metadata(assistant_message_id) or {}
    extra = metadata.get("extra") or {}
    return assistant_message_id, extra


def _find_existing_stream_persist_message(
    db: CharactersRAGDB,
    chat_id: str,
    fingerprint: str,
) -> tuple[str, dict[str, Any]] | None:
    """Find a previously persisted assistant turn for the same streamed payload."""
    candidate_messages = [
        message
        for message in (
            db.get_messages_for_conversation(
                chat_id,
                limit=100,
                offset=0,
                order_by_timestamp="DESC",
            )
            or []
        )
        if not message.get("deleted") and message.get("sender") != "user"
    ]
    if not candidate_messages:
        return None

    message_ids = [str(message["id"]) for message in candidate_messages]
    load_metadata_map = getattr(db, "get_message_metadata_map", None)
    metadata_by_message_id = (
        load_metadata_map(message_ids)
        if callable(load_metadata_map)
        else {}
    ) or {}

    for message in candidate_messages:
        message_id = str(message["id"])
        metadata = metadata_by_message_id.get(message_id)
        if metadata is None:
            metadata = db.get_message_metadata(message_id) or {}
        extra = metadata.get("extra") or {}
        if extra.get("stream_persist_fingerprint") == fingerprint:
            return message_id, extra
    return None


def _build_persist_validation_degraded_detail(assistant_message_id: str) -> dict[str, Any]:
    """Build the structured 503 detail used when persistence succeeds but validation degrades."""
    return {
        "code": "persist_validation_degraded",
        "saved": True,
        "assistant_message_id": assistant_message_id,
    }


def _build_stream_persist_metadata_extra(
    *,
    speaker_character_id: int | None,
    speaker_character_name: str,
    turn_taking_mode: str,
    validation_degraded: bool,
    persist_fingerprint: str | None,
    mood_label: str | None,
    mood_confidence: float | None,
    mood_topic: str | None,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata_extra: dict[str, Any] = {
        "speaker_character_id": speaker_character_id,
        "speaker_character_name": speaker_character_name,
        "turn_taking_mode": turn_taking_mode,
        "persist_validation_degraded": validation_degraded,
    }
    if persist_fingerprint:
        metadata_extra["stream_persist_fingerprint"] = persist_fingerprint
    if isinstance(mood_label, str):
        trimmed_label = mood_label.strip()
        if trimmed_label:
            metadata_extra["mood_label"] = trimmed_label
    if mood_confidence is not None:
        metadata_extra["mood_confidence"] = float(mood_confidence)
    if isinstance(mood_topic, str):
        trimmed_topic = mood_topic.strip()
        if trimmed_topic:
            metadata_extra["mood_topic"] = trimmed_topic
    if usage is not None:
        metadata_extra["usage"] = usage
    return metadata_extra


def _try_store_stream_persist_metadata(
    db: CharactersRAGDB,
    *,
    assistant_message_id: str,
    tool_calls: Any,
    metadata_extra: dict[str, Any],
) -> None:
    try:
        validated_tool_calls = _validate_and_truncate_tool_calls(tool_calls)
        stored = db.add_message_metadata(
            assistant_message_id,
            tool_calls=validated_tool_calls,
            extra=metadata_extra,
        )
        if not stored:
            logger.debug(
                "Non-fatal: failed to persist metadata for message {}",
                assistant_message_id,
            )
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Non-fatal: failed to persist metadata for message {}: {}",
            assistant_message_id,
            exc,
        )


# Legacy local SSE helpers removed — unified streams handle normalization


def _verify_chat_ownership(
    conversation: Optional[dict[str, Any]],
    user_id: Any,
    chat_id: str,
    scope: ConversationScopeParams | None = None,
) -> dict[str, Any]:
    """Verify that the user owns the chat session.

    Args:
        conversation: The conversation dict from database (may be None)
        user_id: The current user's ID
        chat_id: The chat session ID (for error messages)

    Raises:
        HTTPException: 404 if conversation not found, 403 if not owner
    """
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {chat_id} not found"
        )

    # Normalize both IDs to strings and strip whitespace for consistent comparison
    # This handles cases where client_id might have different formatting
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
            detail=f"Chat session {chat_id} not found",
        )
    if (
        expected_scope.scope_type == "workspace"
        and conversation_workspace_id != expected_scope.workspace_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {chat_id} not found",
        )
    return conversation


def _resolve_chat_scope(
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


router = APIRouter()


def _get_research_service() -> ResearchService:
    """Return the default deep research service for chat-linked status reads."""
    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)

# Simple per-chat throttle used for legacy /complete endpoint in tests (TEST_MODE only)
# Bounded to prevent unbounded memory growth - uses constants from Character_Chat.constants
class _BoundedThrottleCache:
    """Bounded cache for throttle windows with automatic stale entry cleanup.

    Concurrency-safe implementation using asyncio.Lock for async context protection.
    """

    def __init__(self):
        self._data: dict[str, deque] = {}
        self._last_access: dict[str, float] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    async def get(self, key: str) -> deque:
        """Concurrency-safe access to throttle window for a given key."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        async with self._lock:
            now = time.time()
            # Cleanup if too many keys
            if len(self._data) > THROTTLE_CACHE_MAX_KEYS:
                self._cleanup(now)
            # Create or get entry
            if key not in self._data:
                self._data[key] = deque(maxlen=THROTTLE_WINDOW_SIZE)
            self._last_access[key] = now
            return self._data[key]

    async def check_and_record(self, key: str, max_hits: int, window_seconds: float) -> bool:
        """Atomically enforce windowed limit and record a hit if allowed."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        async with self._lock:
            now = time.time()
            if len(self._data) > THROTTLE_CACHE_MAX_KEYS:
                self._cleanup(now)
            window = self._data.get(key)
            if window is None:
                window = deque(maxlen=THROTTLE_WINDOW_SIZE)
                self._data[key] = window
            while window and (now - window[0]) > window_seconds:
                window.popleft()
            if len(window) >= max_hits:
                self._last_access[key] = now
                return False
            window.append(now)
            self._last_access[key] = now
            return True

    def _cleanup(self, now: float) -> None:
        """Remove entries not accessed recently."""
        stale_keys = [
            k for k, last in self._last_access.items()
            if (now - last) > THROTTLE_STALE_SECONDS
        ]
        for k in stale_keys:
            self._data.pop(k, None)
            self._last_access.pop(k, None)
        # If still over limit, evict oldest-accessed entries.
        if len(self._data) > THROTTLE_CACHE_MAX_KEYS:
            sorted_by_access = sorted(self._last_access.items(), key=lambda item: item[1])
            excess = len(self._data) - THROTTLE_CACHE_MAX_KEYS
            for k, _ in sorted_by_access[:excess]:
                self._data.pop(k, None)
                self._last_access.pop(k, None)

_complete_windows = _BoundedThrottleCache()

_MAX_DEEP_RESEARCH_ATTACHMENT_CLAIMS = 5
_MAX_DEEP_RESEARCH_ATTACHMENT_UNRESOLVED_QUESTIONS = 5
_MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY = 3
_DEEP_RESEARCH_ATTACHMENT_ALLOWED_KEYS = {
    "run_id",
    "query",
    "question",
    "outline",
    "key_claims",
    "unresolved_questions",
    "verification_summary",
    "source_trust_summary",
    "research_url",
    "attached_at",
    "updatedAt",
}


def reset_complete_windows() -> None:
    """Reset legacy /complete throttle cache (useful in tests)."""
    global _complete_windows
    _complete_windows = _BoundedThrottleCache()

# ========================================================================
# Helper Functions
# ========================================================================

def _convert_db_conversation_to_response(
    conv_data: dict[str, Any],
    *,
    settings: Optional[dict[str, Any]] = None,
) -> ChatSessionResponse:
    """Convert database conversation to response model."""
    character_id = conv_data.get('character_id')
    assistant_kind = conv_data.get('assistant_kind') or ("character" if character_id is not None else None)
    assistant_id = conv_data.get('assistant_id')
    if assistant_id is None and character_id is not None:
        assistant_id = str(character_id)
    return ChatSessionResponse(
        id=conv_data.get('id', ''),
        scope_type=conv_data.get("scope_type") or "global",
        workspace_id=conv_data.get("workspace_id"),
        character_id=character_id,
        assistant_kind=assistant_kind,
        assistant_id=assistant_id,
        persona_memory_mode=conv_data.get('persona_memory_mode'),
        title=conv_data.get('title'),
        rating=conv_data.get('rating'),
        state=conv_data.get('state', 'in-progress'),
        topic_label=conv_data.get('topic_label'),
        cluster_id=conv_data.get('cluster_id'),
        source=conv_data.get('source'),
        external_ref=conv_data.get('external_ref'),
        created_at=conv_data.get('created_at', datetime.now(timezone.utc)),
        last_modified=conv_data.get('last_modified', datetime.now(timezone.utc)),
        message_count=conv_data.get('message_count', 0),
        version=conv_data.get('version', 1),
        parent_conversation_id=conv_data.get('parent_conversation_id'),
        root_id=conv_data.get('root_id'),
        forked_from_message_id=conv_data.get('forked_from_message_id'),
        settings=settings,
    )

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

def _validate_chat_settings_payload(
    settings: dict[str, Any],
    *,
    owner_user_id: str | None = None,
    strip_invalid_deep_research: bool = False,
) -> dict[str, Any]:
    """Validate settings payload size, shape, and known enum fields."""
    settings = dict(settings)
    try:
        encoded = json.dumps(settings).encode("utf-8")
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid settings payload: {exc}"
        ) from exc

    if len(encoded) > MAX_CHAT_SETTINGS_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Settings payload exceeds {MAX_CHAT_SETTINGS_BYTES} bytes"
        )

    author_note = settings.get("authorNote")
    if isinstance(author_note, str) and len(author_note) > MAX_AUTHOR_NOTE_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"authorNote exceeds {MAX_AUTHOR_NOTE_CHARS} characters"
        )

    known_enum_fields = {
        "greetingScope": {"chat", "character"},
        "presetScope": {"chat", "character"},
        "memoryScope": {"shared", "character", "both"},
    }
    for key, allowed_values in known_enum_fields.items():
        value = settings.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or value.strip().lower() not in allowed_values:
            allowed_display = ", ".join(sorted(allowed_values))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {key}. Allowed values: {allowed_display}"
            )

    chat_preset_override = settings.get("chatPresetOverrideId")
    if chat_preset_override is not None and not isinstance(chat_preset_override, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid chatPresetOverrideId. Expected string or null",
        )

    schema_version = settings.get("schemaVersion")
    if schema_version is not None and (not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid schemaVersion. Expected integer >= 1"
        )

    updated_at = settings.get("updatedAt")
    if updated_at is not None and _parse_iso_timestamp(updated_at) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid updatedAt. Expected ISO timestamp string"
        )

    settings = _normalize_chat_settings_deep_research_fields(
        settings,
        owner_user_id=owner_user_id,
        strip_invalid_reference=strip_invalid_deep_research,
    )

    memory_by_id = settings.get("characterMemoryById")
    if memory_by_id is not None:
        if not isinstance(memory_by_id, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid characterMemoryById. Expected object map"
            )
        for character_id, entry in memory_by_id.items():
            if isinstance(entry, dict):
                note = entry.get("note")
                if note is not None and not isinstance(note, str):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid characterMemoryById.{character_id}.note. Expected string"
                    )
                if isinstance(note, str) and len(note) > MAX_AUTHOR_NOTE_CHARS:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"characterMemoryById.{character_id}.note exceeds {MAX_AUTHOR_NOTE_CHARS} characters"
                    )
                entry_updated_at = entry.get("updatedAt")
                if entry_updated_at is not None and _parse_iso_timestamp(entry_updated_at) is None:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid characterMemoryById.{character_id}.updatedAt. Expected ISO timestamp string"
                    )
                continue
            if isinstance(entry, str):
                if len(entry) > MAX_AUTHOR_NOTE_CHARS:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"characterMemoryById.{character_id} exceeds {MAX_AUTHOR_NOTE_CHARS} characters"
                    )
                continue
            if entry is None:
                continue
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid characterMemoryById.{character_id}. Expected object, string, or null"
            )

    def _validate_generation_override_object(value: Any, key_name: str) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {key_name}. Expected object",
            )

        enabled = value.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {key_name}.enabled. Expected boolean",
            )

        numeric_fields: tuple[tuple[str, float, float], ...] = (
            ("temperature", 0.0, 2.0),
            ("top_p", 0.0, 1.0),
            ("repetition_penalty", 0.0, 3.0),
        )
        for numeric_key, min_value, max_value in numeric_fields:
            numeric_value = value.get(numeric_key)
            if numeric_value is None:
                continue
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, (int, float)):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Invalid {key_name}.{numeric_key}. "
                        f"Expected number between {min_value} and {max_value}"
                    ),
                )
            if numeric_value < min_value or numeric_value > max_value:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Invalid {key_name}.{numeric_key}. "
                        f"Expected number between {min_value} and {max_value}"
                    ),
                )

        stop = value.get("stop")
        if stop is not None and (not isinstance(stop, list) or any(not isinstance(item, str) for item in stop)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {key_name}.stop. Expected array of strings",
            )

        override_updated_at = value.get("updatedAt")
        if override_updated_at is not None and _parse_iso_timestamp(override_updated_at) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {key_name}.updatedAt. Expected ISO timestamp string",
            )

    _validate_generation_override_object(settings.get("chatGenerationOverride"), "chatGenerationOverride")
    # Backward-compat: accept prior key spelling while clients migrate.
    _validate_generation_override_object(settings.get("generationOverrides"), "generationOverrides")

    summary_settings = settings.get("summary")
    if summary_settings is not None:
        if not isinstance(summary_settings, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid summary. Expected object"
            )
        source_range = summary_settings.get("sourceRange")
        if source_range is not None and not isinstance(source_range, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid summary.sourceRange. Expected object"
            )
        summary_updated_at = summary_settings.get("updatedAt")
        if summary_updated_at is not None and _parse_iso_timestamp(summary_updated_at) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid summary.updatedAt. Expected ISO timestamp string"
            )
    try:
        normalized_encoded = json.dumps(settings).encode("utf-8")
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid settings payload: {exc}"
        ) from exc
    if len(normalized_encoded) > MAX_CHAT_SETTINGS_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Settings payload exceeds {MAX_CHAT_SETTINGS_BYTES} bytes"
        )
    return settings


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    """Parse ISO timestamp and return epoch seconds, or None when invalid."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _build_canonical_research_attachment_url(run_id: str) -> str:
    return f"/research?run={run_id}"


def _load_owned_research_attachment_run(
    *,
    owner_user_id: str,
    run_id: str,
) -> Any | None:
    try:
        db = ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))
        session = db.get_session(run_id)
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        return None
    if session is None or str(session.owner_user_id) != str(owner_user_id):
        return None
    return session


def _validate_deep_research_attachment(
    value: Any,
    *,
    detail_prefix: str,
    owner_user_id: str | None = None,
    strip_invalid_reference: bool = False,
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {detail_prefix}. Expected object or null",
        )

    unknown_keys = sorted(set(value.keys()) - _DEEP_RESEARCH_ATTACHMENT_ALLOWED_KEYS)
    if unknown_keys:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid {detail_prefix}. Unknown keys: "
                + ", ".join(unknown_keys)
            ),
        )

    normalized: dict[str, Any] = {}
    required_string_fields = ("run_id", "query", "question", "research_url")
    for key in required_string_fields:
        raw = value.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {detail_prefix}.{key}. Expected non-empty string",
            )
        normalized[key] = raw.strip()

    for timestamp_key in ("attached_at", "updatedAt"):
        raw = value.get(timestamp_key)
        if not isinstance(raw, str) or _parse_iso_timestamp(raw) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.{timestamp_key}. "
                    "Expected ISO timestamp string"
                ),
            )
        normalized[timestamp_key] = raw

    outline = value.get("outline")
    if not isinstance(outline, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {detail_prefix}.outline. Expected array",
        )
    normalized_outline: list[dict[str, str]] = []
    for index, section in enumerate(outline):
        if not isinstance(section, dict) or set(section.keys()) != {"title"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.outline[{index}]. "
                    "Expected object with title"
                ),
            )
        title = section.get("title")
        if not isinstance(title, str) or not title.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.outline[{index}].title. "
                    "Expected non-empty string"
                ),
            )
        normalized_outline.append({"title": title.strip()})
    normalized["outline"] = normalized_outline

    key_claims = value.get("key_claims")
    if not isinstance(key_claims, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {detail_prefix}.key_claims. Expected array",
        )
    if len(key_claims) > _MAX_DEEP_RESEARCH_ATTACHMENT_CLAIMS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid {detail_prefix}.key_claims. "
                f"Maximum {_MAX_DEEP_RESEARCH_ATTACHMENT_CLAIMS} entries"
            ),
        )
    normalized_key_claims: list[dict[str, str]] = []
    for index, claim in enumerate(key_claims):
        if not isinstance(claim, dict) or set(claim.keys()) != {"text"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.key_claims[{index}]. "
                    "Expected object with text"
                ),
            )
        text = claim.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.key_claims[{index}].text. "
                    "Expected non-empty string"
                ),
            )
        normalized_key_claims.append({"text": text.strip()})
    normalized["key_claims"] = normalized_key_claims

    unresolved_questions = value.get("unresolved_questions")
    if not isinstance(unresolved_questions, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {detail_prefix}.unresolved_questions. Expected array",
        )
    if len(unresolved_questions) > _MAX_DEEP_RESEARCH_ATTACHMENT_UNRESOLVED_QUESTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid {detail_prefix}.unresolved_questions. "
                f"Maximum {_MAX_DEEP_RESEARCH_ATTACHMENT_UNRESOLVED_QUESTIONS} entries"
            ),
        )
    normalized_unresolved_questions: list[str] = []
    for index, question in enumerate(unresolved_questions):
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.unresolved_questions[{index}]. "
                    "Expected non-empty string"
                ),
            )
        normalized_unresolved_questions.append(question.strip())
    normalized["unresolved_questions"] = normalized_unresolved_questions

    verification_summary = value.get("verification_summary")
    if verification_summary is not None:
        if not isinstance(verification_summary, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {detail_prefix}.verification_summary. Expected object",
            )
        if set(verification_summary.keys()) - {"unsupported_claim_count"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.verification_summary. "
                    "Unsupported keys present"
                ),
            )
        unsupported_claim_count = verification_summary.get("unsupported_claim_count")
        if unsupported_claim_count is not None and (
            isinstance(unsupported_claim_count, bool)
            or not isinstance(unsupported_claim_count, int)
            or unsupported_claim_count < 0
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.verification_summary"
                    ".unsupported_claim_count. Expected non-negative integer"
                ),
            )
        normalized["verification_summary"] = (
            {"unsupported_claim_count": unsupported_claim_count}
            if unsupported_claim_count is not None
            else {}
        )
    else:
        normalized["verification_summary"] = None

    source_trust_summary = value.get("source_trust_summary")
    if source_trust_summary is not None:
        if not isinstance(source_trust_summary, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {detail_prefix}.source_trust_summary. Expected object",
            )
        if set(source_trust_summary.keys()) - {"high_trust_count"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.source_trust_summary. "
                    "Unsupported keys present"
                ),
            )
        high_trust_count = source_trust_summary.get("high_trust_count")
        if high_trust_count is not None and (
            isinstance(high_trust_count, bool)
            or not isinstance(high_trust_count, int)
            or high_trust_count < 0
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.source_trust_summary"
                    ".high_trust_count. Expected non-negative integer"
                ),
            )
        normalized["source_trust_summary"] = (
            {"high_trust_count": high_trust_count}
            if high_trust_count is not None
            else {}
        )
    else:
        normalized["source_trust_summary"] = None

    if owner_user_id is not None:
        session = _load_owned_research_attachment_run(
            owner_user_id=str(owner_user_id),
            run_id=normalized["run_id"],
        )
        if session is None:
            if strip_invalid_reference:
                return None
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.run_id. "
                    "Expected an owned completed deep research run"
                ),
            )
        if str(session.status) != "completed":
            if strip_invalid_reference:
                return None
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {detail_prefix}.run_id. "
                    "Attachment must reference a completed deep research run"
                ),
            )

    normalized["research_url"] = _build_canonical_research_attachment_url(normalized["run_id"])
    return normalized


def _validate_deep_research_attachment_history(
    value: Any,
    *,
    owner_user_id: str | None = None,
    strip_invalid_reference: bool = False,
) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid deepResearchAttachmentHistory. Expected array or null",
        )
    if len(value) > _MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid deepResearchAttachmentHistory. "
                f"Maximum {_MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY} entries"
            ),
        )
    normalized_history: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        normalized_entry = _validate_deep_research_attachment(
            entry,
            detail_prefix=f"deepResearchAttachmentHistory[{index}]",
            owner_user_id=owner_user_id,
            strip_invalid_reference=strip_invalid_reference,
        )
        if normalized_entry is not None:
            normalized_history.append(normalized_entry)
    return normalized_history


def _normalize_chat_settings_deep_research_fields(
    settings: dict[str, Any],
    *,
    owner_user_id: str | None = None,
    strip_invalid_reference: bool = False,
) -> dict[str, Any]:
    normalized = dict(settings)

    active_present = "deepResearchAttachment" in normalized
    active_raw = normalized.get("deepResearchAttachment")
    active_attachment = (
        _validate_deep_research_attachment(
            active_raw,
            detail_prefix="deepResearchAttachment",
            owner_user_id=owner_user_id,
            strip_invalid_reference=strip_invalid_reference,
        )
        if active_present
        else None
    )
    if active_present:
        if active_raw is None:
            normalized["deepResearchAttachment"] = None
        elif active_attachment is None:
            normalized.pop("deepResearchAttachment", None)
        else:
            normalized["deepResearchAttachment"] = active_attachment

    pinned_present = "deepResearchPinnedAttachment" in normalized
    pinned_raw = normalized.get("deepResearchPinnedAttachment")
    pinned_attachment = (
        _validate_deep_research_attachment(
            pinned_raw,
            detail_prefix="deepResearchPinnedAttachment",
            owner_user_id=owner_user_id,
            strip_invalid_reference=strip_invalid_reference,
        )
        if pinned_present
        else None
    )
    if pinned_present:
        if pinned_raw is None:
            normalized["deepResearchPinnedAttachment"] = None
        elif pinned_attachment is None:
            normalized.pop("deepResearchPinnedAttachment", None)
        else:
            normalized["deepResearchPinnedAttachment"] = pinned_attachment

    history_present = "deepResearchAttachmentHistory" in normalized
    history_raw = normalized.get("deepResearchAttachmentHistory")
    history_entries = (
        _validate_deep_research_attachment_history(
            history_raw,
            owner_user_id=owner_user_id,
            strip_invalid_reference=strip_invalid_reference,
        )
        if history_present
        else None
    )
    if history_present:
        if history_raw is None:
            normalized["deepResearchAttachmentHistory"] = None
            return normalized
        excluded_run_ids = {
            run_id
            for run_id in (
                active_attachment.get("run_id") if isinstance(active_attachment, dict) else None,
                pinned_attachment.get("run_id") if isinstance(pinned_attachment, dict) else None,
            )
            if isinstance(run_id, str) and run_id
        }
        deduped_history: list[dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        for entry in history_entries or []:
            run_id = str(entry.get("run_id") or "")
            if not run_id or run_id in excluded_run_ids or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            deduped_history.append(entry)
        if deduped_history:
            normalized["deepResearchAttachmentHistory"] = deduped_history
        else:
            normalized.pop("deepResearchAttachmentHistory", None)

    return normalized


def _merge_deep_research_attachment_history(
    server_history_raw: Any,
    incoming_history_raw: Any,
    *,
    excluded_run_ids: set[str] | None,
) -> Optional[list[dict[str, Any]]]:
    if not isinstance(server_history_raw, list) and not isinstance(incoming_history_raw, list):
        return None

    merged_by_run_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for history in (
        server_history_raw if isinstance(server_history_raw, list) else [],
        incoming_history_raw if isinstance(incoming_history_raw, list) else [],
    ):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            run_id = entry.get("run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                continue
            if excluded_run_ids and run_id in excluded_run_ids:
                continue
            entry_updated_at = _parse_iso_timestamp(entry.get("updatedAt")) or 0.0
            existing = merged_by_run_id.get(run_id)
            if existing is None or entry_updated_at > existing[0]:
                merged_by_run_id[run_id] = (entry_updated_at, entry)

    if not merged_by_run_id:
        return None

    ordered_entries = sorted(
        merged_by_run_id.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    return [
        entry
        for _timestamp, entry in ordered_entries[:_MAX_DEEP_RESEARCH_ATTACHMENT_HISTORY]
    ]


def _normalize_memory_entry(entry: Any) -> Optional[dict[str, Any]]:
    """Normalize a characterMemoryById entry while preserving extra fields."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        normalized = dict(entry)
        note = normalized.get("note")
        if note is None:
            normalized["note"] = ""
        elif not isinstance(note, str):
            normalized["note"] = str(note)
        return normalized
    if isinstance(entry, str):
        return {"note": entry}
    return None


def _merge_character_memory_by_id(
    server_map_raw: Any,
    incoming_map_raw: Any,
    *,
    server_updated_at_epoch: float,
    incoming_updated_at_epoch: float,
) -> Optional[dict[str, Any]]:
    """Merge memory map by entry updatedAt with server-wins tie-breaks."""
    if not isinstance(server_map_raw, dict) and not isinstance(incoming_map_raw, dict):
        return None

    server_map = server_map_raw if isinstance(server_map_raw, dict) else {}
    incoming_map = incoming_map_raw if isinstance(incoming_map_raw, dict) else {}

    merged: dict[str, Any] = {}
    keys = set(server_map.keys()) | set(incoming_map.keys())
    for key in keys:
        server_entry = _normalize_memory_entry(server_map.get(key))
        incoming_entry = _normalize_memory_entry(incoming_map.get(key))

        if server_entry is None and incoming_entry is None:
            continue
        if server_entry is None:
            merged[str(key)] = incoming_entry
            continue
        if incoming_entry is None:
            merged[str(key)] = server_entry
            continue

        server_entry_updated_at = _parse_iso_timestamp(server_entry.get("updatedAt")) or server_updated_at_epoch
        incoming_entry_updated_at = _parse_iso_timestamp(incoming_entry.get("updatedAt")) or incoming_updated_at_epoch
        if incoming_entry_updated_at > server_entry_updated_at:
            merged[str(key)] = incoming_entry
        else:
            merged[str(key)] = server_entry

    return merged


def _merge_conversation_settings(
    server_settings_raw: Any,
    incoming_settings_raw: Any,
) -> dict[str, Any]:
    """Merge settings with timestamp-aware conflict rules and per-entry memory reconciliation.

    Rules:
    - If incoming and server both provide valid ``updatedAt`` values, apply LWW with
      server-wins ties.
    - If incoming omits ``updatedAt``, treat it as an intentional patch update so
      incoming fields apply without client-managed timestamps.
    """
    server_settings = server_settings_raw if isinstance(server_settings_raw, dict) else {}
    incoming_settings = incoming_settings_raw if isinstance(incoming_settings_raw, dict) else {}

    server_updated_at_epoch = _parse_iso_timestamp(server_settings.get("updatedAt"))
    incoming_updated_at_epoch = _parse_iso_timestamp(incoming_settings.get("updatedAt"))
    server_has_updated_at = isinstance(server_settings.get("updatedAt"), str) and server_updated_at_epoch is not None
    incoming_has_updated_at = isinstance(incoming_settings.get("updatedAt"), str) and incoming_updated_at_epoch is not None

    if incoming_has_updated_at and server_has_updated_at:
        incoming_wins = incoming_updated_at_epoch > server_updated_at_epoch
    elif incoming_has_updated_at:
        incoming_wins = True
    else:
        # Untimestamped writes are treated as direct patch updates.
        incoming_wins = True

    if incoming_wins:
        merged = {**server_settings, **incoming_settings}
    else:
        # Server-backed ties resolve to server settings.
        merged = {**incoming_settings, **server_settings}

    server_memory_fallback_epoch = server_updated_at_epoch if server_updated_at_epoch is not None else 0.0
    if incoming_updated_at_epoch is not None:
        incoming_memory_fallback_epoch = incoming_updated_at_epoch
    else:
        incoming_memory_fallback_epoch = server_memory_fallback_epoch + 1.0

    merged_memory = _merge_character_memory_by_id(
        server_settings.get("characterMemoryById"),
        incoming_settings.get("characterMemoryById"),
        server_updated_at_epoch=server_memory_fallback_epoch,
        incoming_updated_at_epoch=incoming_memory_fallback_epoch,
    )
    if merged_memory is not None:
        merged["characterMemoryById"] = merged_memory

    if "deepResearchAttachment" in incoming_settings and incoming_settings.get("deepResearchAttachment") is None:
        merged.pop("deepResearchAttachment", None)
    else:
        server_attachment = server_settings.get("deepResearchAttachment")
        incoming_attachment = incoming_settings.get("deepResearchAttachment")
        if isinstance(server_attachment, dict) or isinstance(incoming_attachment, dict):
            if not isinstance(server_attachment, dict):
                merged["deepResearchAttachment"] = incoming_attachment
            elif not isinstance(incoming_attachment, dict):
                merged["deepResearchAttachment"] = server_attachment
            else:
                server_attachment_updated_at = _parse_iso_timestamp(server_attachment.get("updatedAt")) or 0.0
                incoming_attachment_updated_at = _parse_iso_timestamp(incoming_attachment.get("updatedAt")) or 0.0
                if incoming_attachment_updated_at > server_attachment_updated_at:
                    merged["deepResearchAttachment"] = incoming_attachment
                else:
                    merged["deepResearchAttachment"] = server_attachment

    if "deepResearchPinnedAttachment" in incoming_settings and incoming_settings.get("deepResearchPinnedAttachment") is None:
        merged.pop("deepResearchPinnedAttachment", None)
    else:
        server_pinned_attachment = server_settings.get("deepResearchPinnedAttachment")
        incoming_pinned_attachment = incoming_settings.get("deepResearchPinnedAttachment")
        if isinstance(server_pinned_attachment, dict) or isinstance(incoming_pinned_attachment, dict):
            if not isinstance(server_pinned_attachment, dict):
                merged["deepResearchPinnedAttachment"] = incoming_pinned_attachment
            elif not isinstance(incoming_pinned_attachment, dict):
                merged["deepResearchPinnedAttachment"] = server_pinned_attachment
            else:
                server_pinned_updated_at = _parse_iso_timestamp(server_pinned_attachment.get("updatedAt")) or 0.0
                incoming_pinned_updated_at = _parse_iso_timestamp(incoming_pinned_attachment.get("updatedAt")) or 0.0
                if incoming_pinned_updated_at > server_pinned_updated_at:
                    merged["deepResearchPinnedAttachment"] = incoming_pinned_attachment
                else:
                    merged["deepResearchPinnedAttachment"] = server_pinned_attachment

    active_attachment = merged.get("deepResearchAttachment")
    active_run_id = active_attachment.get("run_id") if isinstance(active_attachment, dict) else None
    pinned_attachment = merged.get("deepResearchPinnedAttachment")
    pinned_run_id = pinned_attachment.get("run_id") if isinstance(pinned_attachment, dict) else None
    excluded_run_ids = {
        run_id
        for run_id in (active_run_id, pinned_run_id)
        if isinstance(run_id, str) and run_id
    }

    if "deepResearchAttachmentHistory" in incoming_settings and incoming_settings.get("deepResearchAttachmentHistory") is None:
        merged.pop("deepResearchAttachmentHistory", None)
    else:
        merged_history = _merge_deep_research_attachment_history(
            server_settings.get("deepResearchAttachmentHistory"),
            incoming_settings.get("deepResearchAttachmentHistory"),
            excluded_run_ids=excluded_run_ids or None,
        )
        if merged_history:
            merged["deepResearchAttachmentHistory"] = merged_history
        else:
            merged.pop("deepResearchAttachmentHistory", None)

    schema_version = merged.get("schemaVersion")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        merged["schemaVersion"] = 2

    if incoming_wins and incoming_has_updated_at:
        merged["updatedAt"] = incoming_settings["updatedAt"]
    elif not incoming_has_updated_at:
        merged["updatedAt"] = datetime.now(timezone.utc).isoformat()
    elif server_has_updated_at:
        merged["updatedAt"] = server_settings["updatedAt"]
    else:
        merged["updatedAt"] = datetime.now(timezone.utc).isoformat()

    return merged


def _normalize_note_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_character_default_author_note(character: dict[str, Any]) -> str:
    if not isinstance(character, dict):
        return ""

    # Support direct fields and extension-based storage.
    for key in ("author_note", "authorNote", "memory_note", "memoryNote"):
        direct = _normalize_note_text(character.get(key))
        if direct:
            return direct

    extensions = character.get("extensions")
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            extensions = {}
    if not isinstance(extensions, dict):
        return ""

    for key in (
        "author_note",
        "authorNote",
        "default_author_note",
        "defaultAuthorNote",
        "memory_note",
        "memoryNote",
    ):
        ext_value = _normalize_note_text(extensions.get(key))
        if ext_value:
            return ext_value
    return ""


def _extract_character_memory_note(settings: dict[str, Any], character_id: Any) -> str:
    memory_by_id = settings.get("characterMemoryById")
    if not isinstance(memory_by_id, dict):
        return ""

    # Accept both raw and stringified keys for robustness across clients.
    lookup_keys = [character_id]
    if character_id is not None:
        lookup_keys.append(str(character_id))

    for key in lookup_keys:
        if key not in memory_by_id:
            continue
        entry = memory_by_id.get(key)
        if isinstance(entry, dict):
            note = _normalize_note_text(entry.get("note"))
            if note:
                return note
        else:
            note = _normalize_note_text(entry)
            if note:
                return note
    return ""


def _normalize_greeting_scope(settings: dict[str, Any]) -> str:
    scope_raw = settings.get("greetingScope")
    if isinstance(scope_raw, str) and scope_raw.strip().lower() == "character":
        return "character"
    return "chat"


def _normalize_preset_scope(settings: dict[str, Any]) -> str:
    scope_raw = settings.get("presetScope")
    if isinstance(scope_raw, str) and scope_raw.strip().lower() == "chat":
        return "chat"
    return "character"


def _normalize_memory_scope(settings: dict[str, Any]) -> str:
    scope_raw = settings.get("memoryScope")
    scope = str(scope_raw).strip().lower() if isinstance(scope_raw, str) else ""
    if scope in {"shared", "character", "both"}:
        return scope
    return "shared"


def _normalize_turn_taking_mode(settings: dict[str, Any]) -> str:
    mode_raw = settings.get("turnTakingMode")
    mode = str(mode_raw).strip().lower() if isinstance(mode_raw, str) else ""
    if mode in {"round_robin", "round-robin", "round robin"}:
        return "round_robin"
    return "single"


def _normalize_character_id(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = int(str(value).strip())
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        return None
    if normalized <= 0:
        return None
    return normalized


def _normalize_participant_character_ids(
    settings: dict[str, Any],
    primary_character_id: Any,
) -> list[int]:
    raw_ids = settings.get("participantCharacterIds")
    if raw_ids is None:
        raw_ids = settings.get("participant_character_ids")

    parsed_ids: list[Any]
    if isinstance(raw_ids, list):
        parsed_ids = raw_ids
    elif isinstance(raw_ids, str):
        text = raw_ids.strip()
        parsed_ids = []
        if text:
            try:
                loaded = json.loads(text)
                parsed_ids = loaded if isinstance(loaded, list) else []
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                parsed_ids = [part.strip() for part in text.split(",")]
    else:
        parsed_ids = []

    ordered_ids: list[int] = []
    seen_ids: set[int] = set()

    primary_id = _normalize_character_id(primary_character_id)
    if primary_id is not None:
        ordered_ids.append(primary_id)
        seen_ids.add(primary_id)

    for value in parsed_ids:
        normalized = _normalize_character_id(value)
        if normalized is None or normalized in seen_ids:
            continue
        ordered_ids.append(normalized)
        seen_ids.add(normalized)

    return ordered_ids


def _extract_settings(settings_row: Any) -> dict[str, Any]:
    if isinstance(settings_row, dict):
        settings = settings_row.get("settings")
        if isinstance(settings, dict):
            return settings
    return {}


def _resolve_effective_prompt_preset(
    settings: dict[str, Any],
    character: dict[str, Any],
    *,
    request_preset: Any = None,
    db: Optional[CharactersRAGDB] = None,
) -> str:
    """Resolve prompt formatting preset with scope-aware precedence.

    Precedence:
    1) request override (single-turn)
    2) chat override when presetScope=chat
    3) character preset when presetScope=character
    4) global default
    """

    def _coerce_preset_id(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        preset = value.strip()
        if not preset:
            return None
        return preset

    request_override = _coerce_preset_id(request_preset)
    if request_override:
        return request_override

    scope = _normalize_preset_scope(settings)
    chat_override = (
        _coerce_preset_id(settings.get("chatPresetOverrideId"))
        or _coerce_preset_id(settings.get("promptPreset"))
        or _coerce_preset_id(settings.get("prompt_preset"))
    )

    if scope == "chat":
        if chat_override in {"default", "st_default"}:
            return chat_override
        if chat_override and db is not None:
            try:
                if isinstance(db.get_prompt_preset(chat_override), dict):
                    return chat_override
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                pass
        return DEFAULT_PROMPT_PRESET

    character_preset = _coerce_preset_id(resolve_character_prompt_preset(character))
    return character_preset or DEFAULT_PROMPT_PRESET


def _build_system_prompt_for_preset(
    db: CharactersRAGDB,
    character: dict[str, Any],
    char_name: str,
    user_name: str,
    preset_id: Optional[str],
) -> str:
    preset = str(preset_id or "").strip()
    if not preset:
        preset = DEFAULT_PROMPT_PRESET

    if preset in {"default", "st_default"}:
        return build_character_system_prompt(character, char_name, user_name, preset=preset)

    try:
        custom_preset = db.get_prompt_preset(preset)
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Non-fatal: failed to fetch custom preset '{}': {}", preset, exc)
        custom_preset = None

    if isinstance(custom_preset, dict):
        raw_order = custom_preset.get("section_order")
        raw_templates = custom_preset.get("section_templates")
        section_order = [item for item in raw_order if isinstance(item, str)] if isinstance(raw_order, list) else []
        section_templates = (
            {key: value for key, value in raw_templates.items() if isinstance(key, str) and isinstance(value, str)}
            if isinstance(raw_templates, dict)
            else {}
        )
        return build_custom_system_prompt(
            character=character,
            char_name=char_name,
            user_name=user_name,
            section_order=section_order,
            section_templates=section_templates,
        )

    return build_character_system_prompt(character, char_name, user_name, preset=DEFAULT_PROMPT_PRESET)


def _resolve_effective_generation_settings(
    settings: dict[str, Any],
    character: dict[str, Any],
) -> dict[str, Any]:
    """Resolve generation parameters: chat override > character > None."""
    base = resolve_character_generation_settings(character)
    overrides = settings.get("chatGenerationOverride")
    if not isinstance(overrides, dict):
        # Backward-compat for earlier key spelling.
        overrides = settings.get("generationOverrides")
    if not isinstance(overrides, dict):
        return base
    if overrides.get("enabled") is False:
        return base
    for key in ("temperature", "top_p", "repetition_penalty", "stop"):
        if key in overrides and overrides[key] is not None:
            base[key] = overrides[key]
    return base


def _normalize_sender_aliases(names: list[str]) -> set[str]:
    aliases: set[str] = set()
    for name in names:
        if not isinstance(name, str):
            continue
        raw = name.strip().lower()
        if raw:
            aliases.add(raw)
        try:
            sanitized = sanitize_sender_name(name).strip().lower()
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            sanitized = ""
        if sanitized:
            aliases.add(sanitized)
    return aliases


def _map_sender_to_role_with_participants(
    sender: Any,
    primary_character_name: Optional[str],
    participant_aliases: set[str],
) -> str:
    role = map_sender_to_role(
        sender if isinstance(sender, str) else None,
        primary_character_name,
        default_role="user",
    )
    if role != "user":
        return role

    if not isinstance(sender, str):
        return "user"
    sender_raw = sender.strip().lower()
    if sender_raw in participant_aliases:
        return "assistant"
    try:
        sender_sanitized = sanitize_sender_name(sender).strip().lower()
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        sender_sanitized = ""
    if sender_sanitized and sender_sanitized in participant_aliases:
        return "assistant"
    return "user"


def _resolve_chat_turn_context(
    db: CharactersRAGDB,
    conversation: dict[str, Any],
    settings_row: Any,
    history_messages: list[dict[str, Any]],
    directed_character_id: Any = None,
) -> dict[str, Any]:
    settings = _extract_settings(settings_row)
    primary_character_id = conversation.get("character_id")
    participant_ids = _normalize_participant_character_ids(settings, primary_character_id)
    participants: list[dict[str, Any]] = []
    for participant_id in participant_ids:
        card = db.get_character_card_by_id(participant_id)
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or f"Character {participant_id}").strip()
        if not name:
            name = f"Character {participant_id}"
        participants.append(
            {"id": participant_id, "character": card, "name": name}
        )

    if not participants:
        fallback_card = db.get_character_card_by_id(primary_character_id) or {}
        fallback_id = _normalize_character_id(
            fallback_card.get("id") or primary_character_id
        ) or 0
        fallback_name = str(fallback_card.get("name") or "Assistant").strip() or "Assistant"
        participants = [
            {"id": fallback_id, "character": fallback_card, "name": fallback_name}
        ]

    turn_taking_mode = _normalize_turn_taking_mode(settings)
    if len(participants) <= 1:
        turn_taking_mode = "single"

    participant_aliases = _normalize_sender_aliases(
        [participant.get("name", "") for participant in participants]
    )
    participant_alias_to_index: dict[str, int] = {}
    for idx, participant in enumerate(participants):
        aliases = _normalize_sender_aliases([participant.get("name", "")])
        for alias in aliases:
            participant_alias_to_index.setdefault(alias, idx)
    primary_name = str(participants[0].get("name") or "Assistant")
    active_participant = participants[0]
    directed_id = _normalize_character_id(directed_character_id)
    directed_applied = False
    if directed_id is not None:
        directed_participant = next(
            (participant for participant in participants if participant.get("id") == directed_id),
            None,
        )
        if directed_participant is not None:
            active_participant = directed_participant
            directed_applied = True

    if turn_taking_mode == "round_robin" and not directed_applied:
        last_assistant_index: Optional[int] = None
        for message in reversed(history_messages):
            role = _map_sender_to_role_with_participants(
                message.get("sender"),
                primary_name,
                participant_aliases,
            )
            if role != "assistant":
                continue
            sender = message.get("sender")
            if isinstance(sender, str):
                sender_raw = sender.strip().lower()
                if sender_raw == "assistant":
                    last_assistant_index = 0
                else:
                    sender_sanitized = ""
                    try:
                        sender_sanitized = sanitize_sender_name(sender).strip().lower()
                    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                        sender_sanitized = ""
                    last_assistant_index = participant_alias_to_index.get(sender_raw)
                    if last_assistant_index is None and sender_sanitized:
                        last_assistant_index = participant_alias_to_index.get(sender_sanitized)
            if last_assistant_index is None:
                # Legacy messages may only store "assistant"; default to primary.
                last_assistant_index = 0
            break
        if last_assistant_index is not None:
            active_participant = participants[(last_assistant_index + 1) % len(participants)]

    return {
        "settings": settings,
        "participants": participants,
        "participant_aliases": participant_aliases,
        "primary_character_name": primary_name,
        "active_character_id": active_participant.get("id"),
        "active_character_name": str(active_participant.get("name") or "Assistant"),
        "active_character": active_participant.get("character") or {},
        "turn_taking_mode": turn_taking_mode,
        "directed_character_applied": directed_applied,
        "directed_character_id": directed_id if directed_applied else None,
    }


def _normalize_greeting_values(value: Any) -> list[str]:
    def _normalize_string_entries(entries: list[Any]) -> list[str]:
        normalized: list[str] = []
        for entry in entries:
            if not isinstance(entry, str):
                continue
            trimmed_entry = entry.strip()
            if trimmed_entry:
                normalized.append(trimmed_entry)
        return normalized

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return []
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            return [trimmed]
        if isinstance(parsed, list):
            return _normalize_string_entries(parsed)
        if isinstance(parsed, str):
            try:
                nested_parsed = json.loads(parsed)
            except json.JSONDecodeError:
                return [trimmed]
            if isinstance(nested_parsed, list):
                return _normalize_string_entries(nested_parsed)
        return [trimmed]
    if isinstance(value, list):
        return _normalize_string_entries(value)
    return []


def _collect_character_greeting_texts(character: dict[str, Any]) -> list[str]:
    greeting_fields = (
        "greeting",
        "first_message",
        "firstMessage",
        "greet",
        "alternate_greetings",
        "alternateGreetings",
    )
    greetings: list[str] = []
    seen: set[str] = set()
    for field_name in greeting_fields:
        values = _normalize_greeting_values(character.get(field_name))
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            greetings.append(value)
    return greetings


def _compute_greetings_checksum(character: dict[str, Any]) -> str:
    """Compute a stable checksum over all greeting texts for staleness detection."""
    greetings = _collect_character_greeting_texts(character)
    joined = "\n---\n".join(greetings)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _check_greeting_staleness(
    settings: dict[str, Any],
    character: dict[str, Any],
) -> Optional[str]:
    """Return a staleness warning if character greetings changed since chat creation.

    Returns ``None`` when no staleness is detected or when no checksum was
    persisted (pre-existing chats).
    """
    stored = settings.get("greetingsChecksum")
    if not isinstance(stored, str) or not stored:
        return None
    current = _compute_greetings_checksum(character)
    if current != stored:
        return (
            "Character greetings have changed since this chat was created. "
            "The stored greeting selection may be stale."
        )
    return None


def _parse_greeting_selection_index(selection_id: Any) -> Optional[int]:
    if not isinstance(selection_id, str):
        return None
    parts = selection_id.strip().split(":")
    if len(parts) < 3 or parts[0] != "greeting":
        return None
    try:
        index = int(parts[1])
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        return None
    if index < 0:
        return None
    return index


def _resolve_character_scoped_greeting(
    settings: dict[str, Any],
    character: dict[str, Any],
    char_name: str,
    user_name: str,
) -> str:
    greetings = _collect_character_greeting_texts(character)
    if not greetings:
        return ""

    selected_index = _parse_greeting_selection_index(settings.get("greetingSelectionId"))
    use_character_default = bool(settings.get("useCharacterDefault"))

    selected_greeting = ""
    if selected_index is not None and selected_index < len(greetings):
        selected_greeting = greetings[selected_index]
    elif use_character_default:
        selected_greeting = greetings[0]
    else:
        # Deterministic fallback to avoid per-request prompt drift.
        selected_greeting = greetings[0]

    return _safe_replace_placeholders(selected_greeting, char_name, user_name).strip()


def _message_sender_matches_character(
    sender: Any,
    active_character_name: str,
    primary_character_name: str,
) -> bool:
    if not isinstance(sender, str):
        return False

    active_aliases = _normalize_sender_aliases([active_character_name])
    sender_raw = sender.strip().lower()

    if sender_raw == "assistant":
        # Legacy assistant sender maps to the primary character.
        primary_aliases = _normalize_sender_aliases([primary_character_name])
        return bool(active_aliases.intersection(primary_aliases))

    if sender_raw in active_aliases:
        return True

    try:
        sender_sanitized = sanitize_sender_name(sender).strip().lower()
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        sender_sanitized = ""
    return bool(sender_sanitized and sender_sanitized in active_aliases)


def _has_prior_assistant_reply(
    history_messages: list[dict[str, Any]],
    primary_character_name: str,
    participant_aliases: set[str],
) -> bool:
    for message in history_messages:
        role = _map_sender_to_role_with_participants(
            message.get("sender"),
            primary_character_name,
            participant_aliases,
        )
        if role == "assistant":
            return True
    return False


def _has_prior_assistant_reply_for_character(
    history_messages: list[dict[str, Any]],
    active_character_name: str,
    primary_character_name: str,
    participant_aliases: set[str],
) -> bool:
    for message in history_messages:
        role = _map_sender_to_role_with_participants(
            message.get("sender"),
            primary_character_name,
            participant_aliases,
        )
        if role != "assistant":
            continue
        if _message_sender_matches_character(
            message.get("sender"),
            active_character_name,
            primary_character_name,
        ):
            return True
    return False


def _should_inject_character_scoped_greeting(
    settings: dict[str, Any],
    turn_context: dict[str, Any],
    history_messages: list[dict[str, Any]],
) -> bool:
    """Decide whether a greeting should be injected for the active turn.

    Scope behavior:
    - chat: inject once for the conversation (first assistant turn)
    - character: inject on each speaking character's first assistant turn
    """
    if settings.get("greetingEnabled") is False:
        return False

    greeting_scope = _normalize_greeting_scope(settings)
    primary_character_name = str(turn_context.get("primary_character_name") or "")
    active_character_name = str(turn_context.get("active_character_name") or "")
    participant_aliases = turn_context.get("participant_aliases") or set()
    participants = turn_context.get("participants") or []

    # Chat scope always behaves as chat-first regardless of participant count.
    if greeting_scope == "chat":
        return not _has_prior_assistant_reply(
            history_messages=history_messages,
            primary_character_name=primary_character_name,
            participant_aliases=participant_aliases,
        )

    if len(participants) <= 1:
        # Edge-case rule: one participant should behave like chat-first.
        return not _has_prior_assistant_reply(
            history_messages=history_messages,
            primary_character_name=primary_character_name,
            participant_aliases=participant_aliases,
        )

    return not _has_prior_assistant_reply_for_character(
        history_messages=history_messages,
        active_character_name=active_character_name,
        primary_character_name=primary_character_name,
        participant_aliases=participant_aliases,
    )


def _inject_character_scoped_greeting_from_settings(
    messages: list[dict[str, Any]],
    settings_row: Any,
    turn_context: dict[str, Any],
    history_messages: list[dict[str, Any]],
    user_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(settings_row, dict):
        return messages

    settings = _extract_settings(settings_row)
    if not settings:
        return messages

    if not _should_inject_character_scoped_greeting(
        settings=settings,
        turn_context=turn_context,
        history_messages=history_messages,
    ):
        return messages

    character = turn_context.get("active_character") or {}
    char_name = str(turn_context.get("active_character_name") or "Assistant")
    greeting_text = _resolve_character_scoped_greeting(
        settings=settings,
        character=character,
        char_name=char_name,
        user_name=user_name,
    )
    if not greeting_text:
        return messages

    insert_idx = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        role = str(messages[idx].get("role", "")).strip().lower()
        if role == "user":
            insert_idx = idx
            break

    if insert_idx == len(messages):
        insert_idx = 0
        while insert_idx < len(messages):
            role = str(messages[insert_idx].get("role", "")).strip().lower()
            if role != "system":
                break
            insert_idx += 1

    if insert_idx > 0:
        prev_message = messages[insert_idx - 1]
        prev_role = str(prev_message.get("role", "")).strip().lower()
        prev_content = str(prev_message.get("content", "")).strip()
        if prev_role == "assistant" and prev_content == greeting_text:
            return messages

    greeting_message = {
        "role": "assistant",
        "content": greeting_text,
    }
    return [*messages[:insert_idx], greeting_message, *messages[insert_idx:]]


def _resolve_author_note_text(
    settings: dict[str, Any],
    character: dict[str, Any],
    *,
    for_prompt: bool = True,
) -> str:
    # Optional toggles reserved for Stage D2 behavior; defaults keep note enabled.
    if settings.get("authorNoteEnabled") is False:
        return ""
    # GM-only and exclude-from-prompt flags suppress injection into LLM context
    # but the note text should still be resolvable for UI display (for_prompt=False).
    if for_prompt:
        if settings.get("authorNoteGmOnly") is True:
            return ""
        if settings.get("authorNoteExcludeFromPrompt") is True:
            return ""

    shared_note = _normalize_note_text(settings.get("authorNote"))
    character_note = _extract_character_memory_note(settings, character.get("id"))
    if not character_note:
        character_note = _extract_character_default_author_note(character)

    scope = _normalize_memory_scope(settings)

    if scope == "character":
        note = character_note
    elif scope == "both":
        note = "\n\n".join(part for part in (shared_note, character_note) if part)
    else:
        # Shared scope: keep chat note precedence, fallback to character default.
        note = shared_note or character_note

    return note.strip()


def _resolve_author_note_position(settings: dict[str, Any]) -> Any:
    for key in (
        "authorNotePosition",
        "authorNotePlacement",
        "authorNoteInjectionPosition",
    ):
        if key in settings:
            return settings.get(key)
    return "before_system"


def _insert_author_note_at_depth(
    messages: list[dict[str, Any]],
    author_note_message: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    if depth <= 0:
        return [author_note_message, *messages]

    non_system_count = 0
    for idx, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "system":
            continue
        if non_system_count >= depth:
            return [*messages[:idx], author_note_message, *messages[idx:]]
        non_system_count += 1
    return [*messages, author_note_message]


def _insert_author_note_message(
    messages: list[dict[str, Any]],
    author_note_message: dict[str, Any],
    position: Any,
) -> list[dict[str, Any]]:
    if isinstance(position, int):
        return _insert_author_note_at_depth(messages, author_note_message, position)

    if isinstance(position, dict):
        mode = str(position.get("mode", "")).strip().lower()
        if mode in {"depth", "at_depth"}:
            depth = position.get("depth")
            if isinstance(depth, int):
                return _insert_author_note_at_depth(messages, author_note_message, depth)
        if mode in {"before_system", "before-system", "before"}:
            return [author_note_message, *messages]

    if isinstance(position, str):
        normalized = position.strip().lower()
        if normalized in {"before_system", "before-system", "before system", "before"}:
            return [author_note_message, *messages]
        if normalized.startswith("depth"):
            suffix = normalized.replace("depth", "", 1).strip()
            if suffix.startswith(":"):
                suffix = suffix[1:].strip()
            if suffix.isdigit():
                return _insert_author_note_at_depth(messages, author_note_message, int(suffix))

    return [author_note_message, *messages]


def _inject_author_note_from_settings(
    messages: list[dict[str, Any]],
    settings_row: Any,
    character: dict[str, Any],
    char_name: str,
    user_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(settings_row, dict):
        return messages
    settings = settings_row.get("settings")
    if not isinstance(settings, dict):
        return messages

    note_text = _resolve_author_note_text(settings, character)
    if not note_text:
        return messages

    resolved_note = _safe_replace_placeholders(note_text, char_name, user_name).strip()
    if not resolved_note:
        return messages

    author_note_message = {
        "role": "system",
        "content": f"Author's note:\n{resolved_note}",
    }
    position = _resolve_author_note_position(settings)
    return _insert_author_note_message(messages, author_note_message, position)


# ---------------------------------------------------------------------------
# Cross-session character memory injection
# ---------------------------------------------------------------------------

_TOKEN_BUDGET_CHARACTER_MEMORY = 300  # approximate token cap for injected memory block


def _inject_character_memory_from_db(
    messages: list[dict[str, Any]],
    db: Any,
    user_id: str,
    character_id: str,
    char_name: str,
    user_name: str,
    token_budget: int = _TOKEN_BUDGET_CHARACTER_MEMORY,
) -> list[dict[str, Any]]:
    """Inject cross-session character memories as a system message.

    Queries ``persona_memory_entries`` for this user+character, groups by
    category, and appends a structured block after author note injection.
    Truncates to *token_budget* (approximate: 1 token ~ 4 chars).
    """
    from tldw_Server_API.app.api.v1.endpoints.character_memory import _persona_id_for_character

    persona_id = _persona_id_for_character(character_id)
    try:
        rows = db.list_persona_memory_entries(
            user_id=user_id,
            persona_id=persona_id,
            include_archived=False,
            include_deleted=False,
            limit=200,
        )
    except Exception as exc:
        logger.debug("Character memory injection skipped (DB error): {}", exc)
        return messages

    if not rows:
        return messages

    # Sort by salience DESC, then last_modified DESC
    rows.sort(key=lambda r: (float(r.get("salience", 0)), r.get("last_modified", "")), reverse=True)

    # Group by category
    categories: dict[str, list[str]] = {}
    for row in rows:
        cat = row.get("memory_type", "manual")
        content = row.get("content", "").strip()
        if content:
            categories.setdefault(cat, []).append(content)

    # Build text block
    category_labels = {
        "fact": "Facts",
        "relationship": "Relationship",
        "event": "Events",
        "preference": "Preferences",
        "manual": "Notes",
    }
    lines: list[str] = [f"Character memory about {user_name}:"]
    char_budget = token_budget * 4  # approximate chars
    total_chars = len(lines[0])

    for cat, label in category_labels.items():
        entries = categories.get(cat)
        if not entries:
            continue
        header = f"[{label}]"
        lines.append(header)
        total_chars += len(header) + 1
        for entry in entries:
            line = f"- {entry}"
            if total_chars + len(line) + 1 > char_budget:
                break
            lines.append(line)
            total_chars += len(line) + 1
        if total_chars >= char_budget:
            break

    if len(lines) <= 1:
        return messages

    memory_block = "\n".join(lines)
    memory_message = {"role": "system", "content": memory_block}
    return [*messages, memory_message]


# In-memory counters for extraction trigger (avoids extra DB write per message)
_extraction_message_counters: dict[str, int] = {}


def _maybe_trigger_character_memory_extraction(
    *,
    background_tasks: Any,
    db: Any,
    chat_id: str,
    settings_row: Any,
    conversation: dict[str, Any],
    character_id: str,
    char_name: str,
    user_id: str,
) -> None:
    """Check extraction settings and trigger background extraction if threshold met."""
    if background_tasks is None:
        return

    settings = {}
    if isinstance(settings_row, dict):
        settings = settings_row.get("settings", {}) or {}
    extraction_cfg = settings.get("characterMemoryExtraction")
    if not isinstance(extraction_cfg, dict) or not extraction_cfg.get("enabled"):
        return

    interval = int(extraction_cfg.get("intervalMessages", 10))
    if interval < 1:
        interval = 10

    counter_key = f"{user_id}:{chat_id}"
    count = _extraction_message_counters.get(counter_key, 0) + 1
    _extraction_message_counters[counter_key] = count

    if count < interval:
        return

    # Reset counter and trigger
    _extraction_message_counters[counter_key] = 0
    user_name = conversation.get("user_name", "User")
    provider = extraction_cfg.get("provider") or "openai"
    model = extraction_cfg.get("model")

    def _run_extraction() -> None:
        from tldw_Server_API.app.api.v1.endpoints.character_memory import (
            get_or_create_character_persona_profile,
            _persona_id_for_character,
        )
        from tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction import (
            extract_character_memories,
        )

        try:
            persona_id = get_or_create_character_persona_profile(db, character_id, char_name, user_id)
            existing = db.list_persona_memory_entries(
                user_id=user_id, persona_id=persona_id, include_archived=True, include_deleted=False, limit=1000,
            )
            messages = db.get_messages_for_conversation(chat_id, limit=50, offset=0) or []
            messages = [m for m in messages if not m.get("deleted")]
            if not messages:
                return

            extraction = extract_character_memories(
                messages=messages,
                char_name=char_name,
                user_name=user_name,
                existing_memories=existing,
                api_endpoint=provider,
                model=model,
            )
            for mem in extraction.unique:
                try:
                    db.add_persona_memory_entry({
                        "persona_id": persona_id,
                        "user_id": user_id,
                        "memory_type": mem["category"],
                        "content": mem["content"],
                        "salience": mem["salience"],
                        "source_conversation_id": chat_id,
                    })
                except Exception as exc:
                    logger.debug("Failed to persist auto-extracted memory: {}", exc)
            if extraction.unique:
                logger.info(
                    "Auto-extracted {} character memories for chat {} (char={})",
                    len(extraction.unique), chat_id, char_name,
                )
        except Exception as exc:
            logger.warning("Background character memory extraction failed: {}", exc)

    background_tasks.add_task(_run_extraction)


def _coerce_truthy_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if is_truthy(normalized):
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_positive_int(
    value: Any,
    default: int,
    *,
    min_value: int = 1,
    max_value: int = 10_000,
) -> int:
    try:
        parsed = int(value)
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _resolve_auto_summary_config(settings: dict[str, Any]) -> tuple[bool, int, int]:
    summary_block = settings.get("summary")
    summary_block = summary_block if isinstance(summary_block, dict) else {}

    enabled = _coerce_truthy_bool(
        settings.get(
            "autoSummaryEnabled",
            summary_block.get("enabled", False),
        ),
        default=False,
    )

    threshold_raw = settings.get("autoSummaryThresholdMessages")
    if threshold_raw is None:
        threshold_raw = settings.get("autoSummaryMessageThreshold")
    if threshold_raw is None:
        threshold_raw = summary_block.get("thresholdMessages")
    if threshold_raw is None:
        threshold_raw = summary_block.get("messageThreshold")
    threshold = _coerce_positive_int(
        threshold_raw if threshold_raw is not None else DEFAULT_AUTO_SUMMARY_THRESHOLD_MESSAGES,
        DEFAULT_AUTO_SUMMARY_THRESHOLD_MESSAGES,
        min_value=2,
        max_value=5_000,
    )

    window_raw = settings.get("autoSummaryWindowMessages")
    if window_raw is None:
        window_raw = settings.get("autoSummaryRecentWindow")
    if window_raw is None:
        window_raw = summary_block.get("windowMessages")
    if window_raw is None:
        window_raw = summary_block.get("recentWindowMessages")
    window = _coerce_positive_int(
        window_raw if window_raw is not None else DEFAULT_AUTO_SUMMARY_WINDOW_MESSAGES,
        DEFAULT_AUTO_SUMMARY_WINDOW_MESSAGES,
        min_value=1,
        max_value=2_000,
    )

    if window >= threshold:
        window = max(1, threshold - 1)

    return enabled, threshold, window


def _collect_pinned_message_ids(
    db: CharactersRAGDB,
    settings: dict[str, Any],
    prompt_messages: list[dict[str, Any]],
) -> set[str]:
    pinned_ids: set[str] = set()

    raw_pinned_ids = settings.get("pinnedMessageIds")
    if isinstance(raw_pinned_ids, list):
        for entry in raw_pinned_ids:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                pinned_ids.add(text)

    for message in prompt_messages:
        message_id_raw = message.get("id")
        if message_id_raw is None:
            continue
        message_id = str(message_id_raw).strip()
        if not message_id:
            continue
        try:
            metadata = db.get_message_metadata(message_id) or {}
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            continue
        extra = metadata.get("extra")
        if not isinstance(extra, dict):
            continue
        if _coerce_truthy_bool(extra.get("pinned"), default=False):
            pinned_ids.add(message_id)
    return pinned_ids


def _build_deterministic_conversation_summary(
    messages_to_summarize: list[dict[str, Any]],
    primary_character_name: str,
    participant_aliases: set[str],
    char_name: str,
    user_name: str,
) -> str:
    lines: list[str] = []
    for message in messages_to_summarize:
        role = _map_sender_to_role_with_participants(
            message.get("sender"),
            primary_character_name,
            participant_aliases,
        )
        content = _safe_replace_placeholders(
            message.get("content"),
            char_name,
            user_name,
        )
        content = " ".join(content.split()).strip()
        if not content:
            continue
        if len(content) > MAX_AUTO_SUMMARY_LINE_CHARS:
            content = f"{content[: MAX_AUTO_SUMMARY_LINE_CHARS - 3].rstrip()}..."
        lines.append(f"- {role}: {content}")
        if len(lines) >= MAX_AUTO_SUMMARY_LINES:
            break

    if not lines:
        return ""

    summary_content = "\n".join(lines)
    if len(summary_content) > MAX_AUTO_SUMMARY_CONTENT_CHARS:
        summary_content = summary_content[:MAX_AUTO_SUMMARY_CONTENT_CHARS].rstrip()
    return summary_content


def _summary_matches_existing(
    existing_summary: Any,
    content: str,
    source_from_id: str,
    source_to_id: str,
    threshold: int,
    window: int,
    compressed_count: int,
) -> bool:
    def _safe_int(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(str(value).strip())
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            return None

    if not isinstance(existing_summary, dict):
        return False
    if existing_summary.get("content") != content:
        return False
    source_range = existing_summary.get("sourceRange")
    if not isinstance(source_range, dict):
        return False
    if str(source_range.get("fromMessageId") or "") != source_from_id:
        return False
    if str(source_range.get("toMessageId") or "") != source_to_id:
        return False
    if _safe_int(existing_summary.get("thresholdMessages")) != threshold:
        return False
    if _safe_int(existing_summary.get("windowMessages")) != window:
        return False
    return _safe_int(existing_summary.get("compressedCount")) == compressed_count


def _persist_auto_summary_to_settings(
    db: CharactersRAGDB,
    chat_id: str,
    settings: dict[str, Any],
    content: str,
    source_from_id: str,
    source_to_id: str,
    threshold: int,
    window: int,
    compressed_count: int,
) -> None:
    existing_summary = settings.get("summary")
    if _summary_matches_existing(
        existing_summary,
        content=content,
        source_from_id=source_from_id,
        source_to_id=source_to_id,
        threshold=threshold,
        window=window,
        compressed_count=compressed_count,
    ):
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    merged_settings = dict(settings)
    merged_settings["summary"] = {
        "enabled": True,
        "content": content,
        "sourceRange": {
            "fromMessageId": source_from_id,
            "toMessageId": source_to_id,
        },
        "thresholdMessages": threshold,
        "windowMessages": window,
        "compressedCount": compressed_count,
        "updatedAt": now_iso,
    }
    merged_settings["schemaVersion"] = int(merged_settings.get("schemaVersion") or 2)
    merged_settings["updatedAt"] = now_iso

    try:
        merged_settings = _validate_chat_settings_payload(merged_settings)
        db.upsert_conversation_settings(chat_id, merged_settings)
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Non-fatal: failed to persist auto-summary settings for chat {}: {}",
            chat_id,
            exc,
        )


def _apply_auto_summary_to_prompt_messages(
    db: CharactersRAGDB,
    chat_id: str,
    settings_row: Any,
    prompt_messages: list[dict[str, Any]],
    *,
    offset: int,
    primary_character_name: str,
    participant_aliases: set[str],
    char_name: str,
    user_name: str,
) -> tuple[list[dict[str, Any]], str]:
    if offset != 0 or not prompt_messages:
        return prompt_messages, ""
    settings = _extract_settings(settings_row)
    if not settings:
        return prompt_messages, ""

    enabled, threshold, window = _resolve_auto_summary_config(settings)
    if not enabled:
        return prompt_messages, ""
    if len(prompt_messages) <= threshold:
        return prompt_messages, ""
    if len(prompt_messages) <= window:
        return prompt_messages, ""

    older_messages = prompt_messages[:-window]
    if not older_messages:
        return prompt_messages, ""

    pinned_ids = _collect_pinned_message_ids(
        db=db,
        settings=settings,
        prompt_messages=prompt_messages,
    )

    compressible: list[dict[str, Any]] = []
    compressible_ids: set[str] = set()
    compressible_object_ids: set[int] = set()
    for message in older_messages:
        message_id_raw = message.get("id")
        message_id = str(message_id_raw).strip() if message_id_raw is not None else ""
        if message_id and message_id in pinned_ids:
            continue
        compressible.append(message)
        if message_id:
            compressible_ids.add(message_id)
        else:
            compressible_object_ids.add(id(message))

    if not compressible:
        return prompt_messages, ""

    summary_content = _build_deterministic_conversation_summary(
        messages_to_summarize=compressible,
        primary_character_name=primary_character_name,
        participant_aliases=participant_aliases,
        char_name=char_name,
        user_name=user_name,
    )
    if not summary_content:
        return prompt_messages, ""

    summarized_messages: list[dict[str, Any]] = []
    for message in prompt_messages:
        message_id_raw = message.get("id")
        message_id = str(message_id_raw).strip() if message_id_raw is not None else ""
        if message_id and message_id in compressible_ids:
            continue
        if not message_id and id(message) in compressible_object_ids:
            continue
        summarized_messages.append(message)

    first_id_raw = compressible[0].get("id")
    last_id_raw = compressible[-1].get("id")
    source_from_id = str(first_id_raw).strip() if first_id_raw is not None else ""
    source_to_id = str(last_id_raw).strip() if last_id_raw is not None else ""
    if source_from_id and source_to_id:
        _persist_auto_summary_to_settings(
            db=db,
            chat_id=chat_id,
            settings=settings,
            content=summary_content,
            source_from_id=source_from_id,
            source_to_id=source_to_id,
            threshold=threshold,
            window=window,
            compressed_count=len(compressible),
        )

    return summarized_messages, summary_content


def _resolve_message_steering_flags(
    continue_as_user: Any,
    impersonate_user: Any,
    force_narrate: Any,
) -> tuple[bool, bool, bool, bool]:
    continue_flag = bool(continue_as_user)
    impersonate_flag = bool(impersonate_user)
    narrate_flag = bool(force_narrate)
    had_conflict = continue_flag and impersonate_flag
    if had_conflict:
        # PRD precedence: impersonate wins when both are selected.
        continue_flag = False
        impersonate_flag = True
    return continue_flag, impersonate_flag, narrate_flag, had_conflict


_DEFAULT_CONTINUE_AS_USER_PROMPT = (
    "Continue the user's current thought in the same voice and perspective."
)
_DEFAULT_IMPERSONATE_USER_PROMPT = (
    "Write this reply as if it is authored by the user, in first person, while preserving the user's intent."
)
_DEFAULT_FORCE_NARRATE_PROMPT = "Use narrative prose style for this reply."


def _normalize_message_steering_prompt_overrides(
    prompt_overrides: Any,
) -> dict[str, str]:
    raw_overrides: Mapping[str, Any] | dict[str, Any] = {}
    if hasattr(prompt_overrides, "model_dump"):
        try:
            raw_overrides = prompt_overrides.model_dump(exclude_none=True)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            raw_overrides = {}
    elif isinstance(prompt_overrides, Mapping):
        raw_overrides = dict(prompt_overrides)

    def _resolve_prompt(raw_value: Any, fallback: str) -> str:
        if not isinstance(raw_value, str):
            return fallback
        normalized = raw_value.strip()
        return normalized or fallback

    return {
        "continue_as_user": _resolve_prompt(
            raw_overrides.get("continue_as_user"),
            _DEFAULT_CONTINUE_AS_USER_PROMPT,
        ),
        "impersonate_user": _resolve_prompt(
            raw_overrides.get("impersonate_user"),
            _DEFAULT_IMPERSONATE_USER_PROMPT,
        ),
        "force_narrate": _resolve_prompt(
            raw_overrides.get("force_narrate"),
            _DEFAULT_FORCE_NARRATE_PROMPT,
        ),
    }


def _build_message_steering_instruction(
    continue_as_user: bool,
    impersonate_user: bool,
    force_narrate: bool,
    prompt_overrides: Any = None,
) -> str:
    prompt_templates = _normalize_message_steering_prompt_overrides(
        prompt_overrides
    )
    instructions: list[str] = []
    if impersonate_user:
        instructions.append(prompt_templates["impersonate_user"])
    elif continue_as_user:
        instructions.append(prompt_templates["continue_as_user"])
    if force_narrate:
        instructions.append(prompt_templates["force_narrate"])
    if not instructions:
        return ""
    return f"Steering instruction (single response): {' '.join(instructions)}"


def _inject_message_steering_instruction(
    messages: list[dict[str, Any]],
    continue_as_user: Any,
    impersonate_user: Any,
    force_narrate: Any,
    prompt_overrides: Any = None,
) -> tuple[list[dict[str, Any]], bool]:
    continue_flag, impersonate_flag, narrate_flag, had_conflict = (
        _resolve_message_steering_flags(
            continue_as_user=continue_as_user,
            impersonate_user=impersonate_user,
            force_narrate=force_narrate,
        )
    )
    instruction = _build_message_steering_instruction(
        continue_as_user=continue_flag,
        impersonate_user=impersonate_flag,
        force_narrate=narrate_flag,
        prompt_overrides=prompt_overrides,
    )
    if not instruction:
        return messages, had_conflict

    steering_message = {"role": "system", "content": instruction}

    # Keep steering scoped to the upcoming turn by inserting before the latest user message.
    for idx in range(len(messages) - 1, -1, -1):
        role = str(messages[idx].get("role", "")).strip().lower()
        if role == "user":
            return [*messages[:idx], steering_message, *messages[idx:]], had_conflict

    return [*messages, steering_message], had_conflict

"""Role mapping provided by Character_Chat utility: map_sender_to_role"""

# ========================================================================
# Chat Session Endpoints
# ========================================================================

@router.post("/", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED,
             summary="Create a new chat session", tags=["Chat Sessions"])
async def create_chat_session(
    session_data: ChatSessionCreate,
    response: Response,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    seed_first_message: bool = Query(False, description="If true, seed the chat with an initial assistant greeting"),
    greeting_strategy: Literal["default", "alternate_random", "alternate_index"] = Query("default", description="How to choose the initial assistant greeting when seeding"),
    alternate_index: Optional[int] = Query(None, ge=0, description="Index for alternate greeting when greeting_strategy=alternate_index"),
):
    """
    Create a new chat session with a character or persona assistant identity.

    Args:
        session_data: Chat session creation data
        db: Database instance
        current_user: Authenticated user

    Notes:
        This API does not automatically create a first assistant message. Clients
        should POST a first user/assistant message after chat creation. The library
        helper `start_new_chat_session` can seed a first message when used directly.

    Returns:
        Created chat session details

    Raises:
        HTTPException: 404 if character not found, 429 if rate limited
    """
    try:
        scope = _resolve_chat_scope(session_data.scope_type, session_data.workspace_id)
        # Check rate limits
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_rate_limit(current_user.id, "chat_create")
        # Enforce per-user chat count limit (approximate by scanning conversations per character)
        try:
            # Use DB-layer count for efficiency/accuracy. The helper expects
            # the current count (before this create) and rejects when
            # current_chat_count >= max_chats_per_user.
            user_chat_count = db.count_conversations_for_user(
                str(current_user.id),
                scope_type=scope.scope_type,
                workspace_id=scope.workspace_id,
            )
            await rate_limiter.check_chat_limit(current_user.id, user_chat_count)
        except HTTPException:
            # Propagate enforcement failures
            raise
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
            # Fail closed: quota enforcement must work to prevent resource exhaustion
            logger.error("Chat limit enforcement failed, denying request: {}", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quota enforcement unavailable. Please try again later."
            ) from e

        character: dict[str, Any] | None = None
        persona_profile: dict[str, Any] | None = None
        assistant_display_name = "Assistant"

        if session_data.assistant_kind == "persona":
            persona_profile = db.get_persona_profile(session_data.assistant_id or "", user_id=str(current_user.id))
            if not persona_profile:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Persona with ID {session_data.assistant_id} not found",
                )
            assistant_display_name = persona_profile.get("name") or assistant_display_name
        else:
            character = db.get_character_card_by_id(session_data.character_id)
            if not character:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Character with ID {session_data.character_id} not found"
                )
            assistant_display_name = character.get("name") or assistant_display_name

        # Validate parent conversation (if any) for ownership and root lineage
        parent_conversation = None
        validated_parent_id: Optional[str] = None
        parent_root_id: Optional[str] = None
        validated_forked_from_message_id: Optional[str] = None
        if session_data.parent_conversation_id:
            parent_conversation = db.get_conversation_by_id(session_data.parent_conversation_id)
            _verify_chat_ownership(
                parent_conversation,
                current_user.id,
                session_data.parent_conversation_id,
                scope,
            )
            if parent_conversation:
                validated_parent_id = parent_conversation.get("id") or session_data.parent_conversation_id
                parent_root_id = parent_conversation.get("root_id") or parent_conversation.get("id")
            # Cross-character forks are intentionally supported; do not enforce a character_id match.

        if session_data.forked_from_message_id:
            if not validated_parent_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="forked_from_message_id requires parent_conversation_id",
                )
            source_message = db.get_message_by_id(session_data.forked_from_message_id)
            if not source_message or source_message.get("conversation_id") != validated_parent_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="forked_from_message_id must belong to parent_conversation_id",
                )
            validated_forked_from_message_id = (
                source_message.get("id") or session_data.forked_from_message_id
            )

        # Generate chat ID and title
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = session_data.title or f"{assistant_display_name} Chat ({timestamp})"

        # Create conversation data
        conv_data = {
            'id': chat_id,
            'character_id': session_data.character_id,
            'assistant_kind': session_data.assistant_kind,
            'assistant_id': session_data.assistant_id,
            'persona_memory_mode': session_data.persona_memory_mode,
            'title': title,
            'root_id': parent_root_id or chat_id,  # Inherit root for forks
            'parent_conversation_id': validated_parent_id,
            'forked_from_message_id': validated_forked_from_message_id,
            'client_id': str(current_user.id),
            'version': 1,
            'state': session_data.state,
            'topic_label': session_data.topic_label,
            'cluster_id': session_data.cluster_id,
            'source': session_data.source,
            'external_ref': session_data.external_ref,
            'scope_type': session_data.scope_type or 'global',
            'workspace_id': session_data.workspace_id,
        }

        # Add to database
        created_id = db.add_conversation(conv_data)
        if not created_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )

        # Retrieve created conversation
        created_conv = db.get_conversation_by_id(created_id)
        if not created_conv:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created chat session"
            )

        # Optionally seed the chat with a greeting (first_message or alternate)
        seed_status: Optional[str] = None
        if seed_first_message and character is not None:
            try:
                raw_name = character.get('name') or 'Assistant'
                choice_text: Optional[str] = None
                if greeting_strategy in {"alternate_random", "alternate_index"}:
                    ag = character.get('alternate_greetings')
                    if isinstance(ag, list) and ag:
                        if greeting_strategy == "alternate_random":
                            choice_text = _greeting_random.choice(ag)
                        elif greeting_strategy == "alternate_index" and isinstance(alternate_index, int) and 0 <= alternate_index < len(ag):
                            choice_text = ag[alternate_index]
                if not choice_text:
                    fm = character.get('first_message')
                    if isinstance(fm, str) and fm.strip():
                        choice_text = fm
                if isinstance(choice_text, str) and choice_text.strip():
                    content = choice_text
                    post_message_to_conversation(
                        db=db,
                        conversation_id=created_id,
                        character_name=raw_name,
                        message_content=content,
                        is_user_message=False,
                    )
                    # Update in-memory message count (best-effort)
                    with contextlib.suppress(_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS):
                        created_conv['message_count'] = (created_conv.get('message_count') or 0) + 1
                    seed_status = "ok"
                else:
                    seed_status = "no_greeting"
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as _seed_err:
                seed_status = "failed"
                logger.warning(f"Failed to seed first message for chat {created_id}: {_seed_err}")
        elif seed_first_message:
            seed_status = "no_greeting"

        # Persist a greetings checksum so staleness can be detected later.
        if character is not None:
            try:
                checksum = _compute_greetings_checksum(character)
                updated_settings = db.upsert_conversation_settings(
                    created_id,
                    {"greetingsChecksum": checksum},
                )
                # Keep optimistic-locking version in response in sync with DB.
                if updated_settings:
                    latest_conv = db.get_conversation_by_id(created_id)
                    if isinstance(latest_conv, dict):
                        created_conv = latest_conv
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                pass  # best-effort; staleness detection degrades gracefully

        if response is not None and seed_status is not None:
            try:
                response.headers["X-Chat-Seed-Status"] = seed_status
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to set X-Chat-Seed-Status header: {}", exc)

        # Log creation
        logger.info(
            "Created chat session {} for {} {} by user {}",
            created_id,
            session_data.assistant_kind,
            session_data.assistant_id,
            current_user.id,
        )

        return _convert_db_conversation_to_response(created_conv)

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error creating chat session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating chat session"
        ) from e


# ========================================================================
# Preset Editor Endpoints (PRD 2 Stage B2)
# NOTE: These must be registered BEFORE /{chat_id} to avoid path capture.
# ========================================================================

# Template tokens available in custom presets
_PRESET_TEMPLATE_TOKENS = [
    PresetTokenInfo(token="{{char}}", description="Character name"),
    PresetTokenInfo(token="{{user}}", description="User/player name"),
    PresetTokenInfo(token="{{description}}", description="Character description field"),
    PresetTokenInfo(token="{{personality}}", description="Character personality field"),
    PresetTokenInfo(token="{{scenario}}", description="Character scenario field"),
    PresetTokenInfo(token="{{system_prompt}}", description="Character system prompt field"),
    PresetTokenInfo(token="{{message_example}}", description="Character example messages"),
    PresetTokenInfo(token="{{post_history}}", description="Post-history instructions"),
]

# Built-in preset metadata (non-deletable)
_BUILTIN_PRESETS = [
    PresetDetail(
        preset_id="default",
        name="Default",
        builtin=True,
        section_order=["identity", "description", "personality", "scenario", "system_prompt"],
        section_templates={
            "identity": "You are {{char}}.",
            "description": "{{description}}",
            "personality": "{{personality}}",
            "scenario": "{{scenario}}",
            "system_prompt": "{{system_prompt}}",
        },
    ),
    PresetDetail(
        preset_id="st_default",
        name="SillyTavern Default",
        builtin=True,
        section_order=["identity", "system_prompt", "description", "personality", "scenario", "message_example", "post_history"],
        section_templates={
            "identity": "You are {{char}}.",
            "system_prompt": "{{system_prompt}}",
            "description": "Description:\n{{description}}",
            "personality": "Personality:\n{{personality}}",
            "scenario": "Scenario:\n{{scenario}}",
            "message_example": "Example dialogue:\n{{message_example}}",
            "post_history": "Post-history instructions:\n{{post_history}}",
        },
    ),
]


@router.get(
    "/presets/tokens",
    summary="List available template tokens for custom presets",
    tags=["Presets"],
)
async def list_preset_tokens(
    current_user: User = Depends(get_request_user),
):
    """Return the list of template tokens that can be used in custom preset templates."""
    return {"tokens": _PRESET_TEMPLATE_TOKENS}


@router.get(
    "/presets",
    response_model=PresetListResponse,
    summary="List prompt presets",
    tags=["Presets"],
)
async def list_presets(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Return built-in and user-defined prompt presets."""
    all_presets: list[PresetDetail] = list(_BUILTIN_PRESETS)

    # Load user-defined presets from DB
    try:
        rows = db.list_prompt_presets()
        for row in rows:
            all_presets.append(PresetDetail(
                preset_id=row["preset_id"],
                name=row["name"],
                builtin=False,
                section_order=row.get("section_order", []),
                section_templates=row.get("section_templates", {}),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            ))
    except Exception as exc:
        logger.debug("Could not load user presets (table may not exist yet): {}", exc)

    return PresetListResponse(presets=all_presets)


@router.post(
    "/presets",
    response_model=PresetDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom prompt preset",
    tags=["Presets"],
)
async def create_preset(
    body: PresetCreate = Body(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Create a new user-defined prompt preset."""
    created = db.upsert_prompt_preset(
        preset_id=body.preset_id,
        name=body.name,
        section_order=body.section_order,
        section_templates=body.section_templates,
    )
    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create prompt preset",
        )
    return PresetDetail(
        preset_id=body.preset_id,
        name=body.name,
        builtin=False,
        section_order=body.section_order,
        section_templates=body.section_templates,
    )


@router.put(
    "/presets/{preset_id}",
    response_model=PresetDetail,
    summary="Update a custom prompt preset",
    tags=["Presets"],
)
async def update_preset(
    preset_id: str = Path(..., description="Preset identifier"),
    body: PresetUpdate = Body(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Update an existing user-defined prompt preset. Cannot modify built-in presets."""
    if preset_id in ("default", "st_default"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot modify built-in presets",
        )

    existing = db.get_prompt_preset(preset_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{preset_id}' not found",
        )

    name = body.name if body.name is not None else existing["name"]
    section_order = body.section_order if body.section_order is not None else existing.get("section_order", [])
    section_templates = body.section_templates if body.section_templates is not None else existing.get("section_templates", {})

    updated_ok = db.upsert_prompt_preset(
        preset_id=preset_id,
        name=name,
        section_order=section_order,
        section_templates=section_templates,
    )
    if not updated_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preset '{preset_id}'",  # nosec B608
        )
    updated = db.get_prompt_preset(preset_id)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load updated preset '{preset_id}'",
        )
    return PresetDetail(
        preset_id=preset_id,
        name=updated["name"],
        builtin=False,
        section_order=updated.get("section_order", []),
        section_templates=updated.get("section_templates", {}),
        created_at=updated.get("created_at"),
        updated_at=updated.get("updated_at"),
    )


@router.delete(
    "/presets/{preset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom prompt preset",
    tags=["Presets"],
)
async def delete_preset(
    preset_id: str = Path(..., description="Preset identifier"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Delete a user-defined preset. Cannot delete built-in presets."""
    if preset_id in ("default", "st_default"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot delete built-in presets",
        )

    existing = db.get_prompt_preset(preset_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset '{preset_id}' not found",
        )

    deleted = db.delete_prompt_preset(preset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete preset '{preset_id}'",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ========================================================================
# Chat Session Detail Endpoints
# ========================================================================

@router.get("/{chat_id}", response_model=ChatSessionResponse,
            summary="Get chat session details", tags=["Chat Sessions"])
async def get_chat_session(
    chat_id: str = Path(..., description="Chat session ID"),
    include_settings: bool = Query(
        False,
        description="Include per-chat settings payload in the response.",
    ),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Get details of a specific chat session.

    Args:
        chat_id: Chat session ID
        db: Database instance
        current_user: Authenticated user

    Returns:
        Chat session details

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized
    """
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        # Get message count efficiently
        try:
            conversation['message_count'] = db.count_messages_for_conversation(chat_id)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            messages = db.get_messages_for_conversation(chat_id, limit=1000)
            conversation['message_count'] = len(messages) if messages else 0

        settings_payload: Optional[dict[str, Any]] = None
        if include_settings:
            settings_row = db.get_conversation_settings(chat_id)
            settings_payload = (settings_row or {}).get("settings") or {}

        return _convert_db_conversation_to_response(
            conversation,
            settings=settings_payload,
        )

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving chat session"
        ) from e


@router.get(
    "/{chat_id}/research-runs",
    response_model=ChatLinkedResearchRunsListResponse,
    summary="List deep research runs linked to a chat session",
    tags=["Chat Sessions"],
)
async def list_chat_linked_research_runs(
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    research_service: ResearchService = Depends(_get_research_service),
):
    """Return compact deep research run status rows linked to the chat thread."""
    try:
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        runs = research_service.list_chat_linked_runs(
            owner_user_id=str(current_user.id),
            chat_id=chat_id,
            terminal_limit=10,
        )
        return ChatLinkedResearchRunsListResponse(runs=runs)

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error listing linked research runs for chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving linked research runs",
        ) from e


@router.get("/{chat_id}/context", summary="Get chat context for completions", tags=["Chat Sessions"])
async def get_chat_context(
    chat_id: str = Path(..., description="Chat session ID"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """Return chat context formatted for chat completions."""
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        conversation = db.get_conversation_by_id(chat_id)
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chat session {chat_id} not found")

        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        settings_row = db.get_conversation_settings(chat_id)
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get('deleted')]
        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
        )
        character = turn_context.get("active_character") or {}
        char_name = turn_context.get("active_character_name") or "Unknown"
        participant_aliases = turn_context.get("participant_aliases") or set()
        primary_character_name = turn_context.get("primary_character_name")
        user_name = conversation.get('user_name', 'User')

        messages = history_messages
        # Map DB messages to chat-completions messages with normalized roles
        formatted = []
        for m in messages:
            role = _map_sender_to_role_with_participants(
                m.get('sender'),
                primary_character_name,
                participant_aliases,
            )
            content = _safe_replace_placeholders(m.get('content'), char_name, user_name)
            formatted.append({"role": role, "content": content})

        # If no messages, include first_message as an initial assistant message (with placeholders resolved)
        if not formatted and character.get('first_message'):
            fm = _safe_replace_placeholders(character.get('first_message'), char_name, user_name)
            formatted.append({"role": "assistant", "content": fm})

        return {"character_name": char_name, "messages": formatted}

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error getting chat context for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while retrieving chat context") from e


@router.post(
    "/{chat_id}/complete",
    summary="Legacy completion endpoint with simple rate limit",
    tags=["Chat Sessions"],
    deprecated=True,
    description=(
        "DEPRECATED: The request body for this endpoint is ignored and will be rejected in a future release. "
        "Call this endpoint without a body. Prefer /{chat_id}/completions or /{chat_id}/complete-v2."
    ),
)
async def complete_chat_legacy(
    chat_id: str = Path(..., description="Chat session ID"),
    body: Optional[dict[str, Any]] = Body(
        default=None,
        description=(
            "DEPRECATED: Request body is ignored. This parameter will be removed and non-empty bodies will be rejected (422) in a future release."
        ),
    ),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    response: Response = None,
):
    """Legacy completion endpoint used by tests to validate rate limiting.

    Applies a very small per-conversation throttle when TEST_MODE=true so that
    burst requests trigger HTTP 429 as expected by tests.
    """
    try:
        # Validate chat ownership
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        # Per-minute completion limiter (global per-user)
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_chat_completion_rate(current_user.id)

        # Deprecation headers for clients; also used if we reject a non-empty body.
        dep_headers = build_deprecation_headers(
            f"/api/v1/chats/{chat_id}/complete-v2",
            default_sunset_days=90,
        )
        try:
            if response is not None:
                for k, v in dep_headers.items():
                    response.headers[k] = v
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Non-fatal: failed to set deprecation headers for chat {}: {}",
                chat_id,
                exc,
            )

        # If a non-empty body was provided, reject with 422 and include deprecation headers
        if isinstance(body, dict) and body:
            logger.warning("Legacy /complete rejected non-empty body; clients must omit the request body.")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Request body is deprecated and must be omitted. Use /{chat_id}/complete-v2 or /{chat_id}/completions.",
                headers=dep_headers,
            )

        # Test-mode throttle: 5 requests per second per (user, chat)
        key = f"{current_user.id}:{chat_id}"
        allowed = await _complete_windows.check_and_record(
            key,
            max_hits=5,
            window_seconds=1.0,
        )
        if not allowed:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded for chat completion")

        # Return a minimal success payload
        return {"status": "ok", "chat_id": chat_id}

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error in legacy complete for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during completion") from e


@router.post("/{chat_id}/completions", response_model=CharacterChatCompletionPrepResponse,
             summary="Prepare messages for chat completion (rate-limited)", tags=["Chat Sessions"])
async def prepare_chat_completion(
    chat_id: str = Path(..., description="Chat session ID"),
    body: CharacterChatCompletionPrepRequest = None,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """Prepare chat messages for use with the main Chat API, enforcing per-minute completion limits.

    This endpoint does not call an LLM. It returns messages formatted for
    POST /api/v1/chat/completions and applies the per-minute completion limiter.
    """
    try:
        body = body or CharacterChatCompletionPrepRequest()

        # Validate chat ownership
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        # Per-minute completion limiter (global per-user)
        rate_limiter = get_character_rate_limiter()
        await rate_limiter.check_chat_completion_rate(current_user.id)

        # Build messages
        user_name = conversation.get('user_name', 'User')
        include_ctx = bool(body.include_character_context)
        # Fields are validated by Pydantic; avoid redundant int() casting
        limit = body.limit
        offset = body.offset
        settings_row = db.get_conversation_settings(chat_id)

        messages = db.get_messages_for_conversation(chat_id, limit=limit, offset=offset) or []
        # Filter deleted
        messages = [m for m in messages if not m.get('deleted')]
        paginated = messages
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get('deleted')]

        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
            directed_character_id=body.directed_character_id,
        )
        if (
            body.directed_character_id is not None
            and not turn_context.get("directed_character_applied")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="directed_character_id must reference a selected participant in this chat",
            )
        character = turn_context.get("active_character") or {}
        character_name = turn_context.get("active_character_name")
        char_label = character_name or "Assistant"
        participant_aliases = turn_context.get("participant_aliases") or set()
        primary_character_name = turn_context.get("primary_character_name")
        summary_content = ""
        paginated, summary_content = _apply_auto_summary_to_prompt_messages(
            db=db,
            chat_id=chat_id,
            settings_row=settings_row,
            prompt_messages=paginated,
            offset=offset,
            primary_character_name=primary_character_name or "",
            participant_aliases=participant_aliases,
            char_name=char_label,
            user_name=user_name,
        )

        prep_settings = _extract_settings(settings_row)
        effective_preset = _resolve_effective_prompt_preset(
            prep_settings,
            character,
            request_preset=body.prompt_preset,
            db=db,
        )

        formatted: list[dict[str, Any]] = []
        if include_ctx and character and offset == 0:
            sys_text = _build_system_prompt_for_preset(
                db=db,
                character=character,
                char_name=char_label,
                user_name=user_name,
                preset_id=effective_preset,
            )
            if sys_text.strip():
                formatted.append({"role": "system", "content": sys_text.strip()})
        if summary_content:
            formatted.append(
                {"role": "system", "content": f"Conversation summary:\n{summary_content}"}
            )

        for msg in paginated:
            formatted.append({
                "role": _map_sender_to_role_with_participants(
                    msg.get('sender'),
                    primary_character_name,
                    participant_aliases,
                ),
                "content": _safe_replace_placeholders(msg.get('content'), char_label, user_name)
            })

        if body.append_user_message:
            formatted.append({"role": "user", "content": body.append_user_message})

        formatted = _inject_character_scoped_greeting_from_settings(
            formatted,
            settings_row=settings_row,
            turn_context=turn_context,
            history_messages=history_messages,
            user_name=user_name,
        )
        formatted = _inject_author_note_from_settings(
            formatted,
            settings_row=settings_row,
            character=character,
            char_name=char_label,
            user_name=user_name,
        )
        formatted = _inject_character_memory_from_db(
            formatted,
            db=db,
            user_id=str(current_user.id),
            character_id=str(turn_context.get("active_character_id") or conversation.get("character_id", "")),
            char_name=char_label,
            user_name=user_name,
        )
        formatted, steering_conflict = _inject_message_steering_instruction(
            formatted,
            continue_as_user=body.continue_as_user,
            impersonate_user=body.impersonate_user,
            force_narrate=body.force_narrate,
            prompt_overrides=body.message_steering_prompts,
        )
        if steering_conflict:
            logger.debug(
                "Steering conflict resolved for chat {} in /completions prep: impersonate_user overrides continue_as_user",
                chat_id,
            )

        return CharacterChatCompletionPrepResponse(
            chat_id=chat_id,
            character_id=turn_context.get("active_character_id") or conversation['character_id'],
            character_name=character_name,
            messages=formatted,
            total=len(messages)
        )

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error preparing completion for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while preparing completion") from e


# ── Token budget constants for supplemental prompt sections ──────────
_TOKEN_BUDGET_TOTAL = 1200
_TOKEN_BUDGET_GREETING = 120
_TOKEN_BUDGET_AUTHOR_NOTE = 240
_TOKEN_BUDGET_PRESET = 180
_TOKEN_BUDGET_STEERING = 120
_TOKEN_BUDGET_LOREBOOK = 420
_TOKEN_BUDGET_WORLD_BOOK = 240

# Priority order for truncation (highest priority first)
_TRUNCATION_PRIORITY = [
    "preset",
    "author_note",
    "message_steering",
    "lorebook",
    "world_book",
    "greeting",
]

_PREVIEW_CONFLICT_EXAMPLES = [
    "Preset sets temperature=0.7; later section sets temperature=0.2 -> effective temperature=0.2.",
    'Preset directive "Speak tersely"; later directive "Be verbose" -> later directive replaces earlier.',
    "Appendable source blocks concatenate only when both sides are appendable.",
]

_CONTRADICTORY_DIRECTIVE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("speak tersely", "be verbose", "Conflicting directives detected (terse vs verbose)."),
    ("concise", "detailed", "Conflicting directives detected (concise vs detailed)."),
    ("formal", "casual", "Conflicting directives detected (formal vs casual)."),
)


def _build_persona_preview_sections(
    *,
    conversation: dict[str, Any],
    exemplars: list[dict[str, Any]],
    requested_scenario_tags: list[str] | None = None,
    requested_tone: str | None = None,
    current_turn_text: str | None = None,
    conflicting_capability_tags: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Build persona exemplar preview sections using the shared assembly helper."""
    if conversation.get("assistant_kind") != "persona":
        return []

    persona_id = str(conversation.get("assistant_id") or "").strip()
    if not persona_id:
        return []

    assembly = assemble_persona_exemplar_prompt(
        persona_id=persona_id,
        exemplars=exemplars,
        requested_scenario_tags=requested_scenario_tags,
        requested_tone=requested_tone,
        current_turn_text=current_turn_text,
        conflicting_capability_tags=conflicting_capability_tags,
    )
    return assembly.sections


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to fit within a token budget (approximate)."""
    if not text:
        return text
    estimated = _estimate_tokens(text)
    if estimated <= token_budget:
        return text
    # Rough truncation: 4 chars per token
    max_chars = token_budget * 4
    return text[:max_chars] + "..."


def _extract_scalar_conflicts(texts: list[str]) -> list[dict[str, str]]:
    by_key: dict[str, set[str]] = {}
    pattern = re.compile(
        r"\b(temperature|top_p|top[\s-]?p|repetition_penalty|repetition[\s-]?penalty)\b\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for text in texts:
        for match in pattern.finditer(text or ""):
            raw_key = (match.group(1) or "").lower()
            key = "top_p" if "top" in raw_key else raw_key.replace(" ", "_").replace("-", "_")
            value = match.group(2) or ""
            by_key.setdefault(key, set()).add(value)

    conflicts: list[dict[str, str]] = []
    for key, values in by_key.items():
        if len(values) <= 1:
            continue
        joined_values = ", ".join(sorted(values))
        conflicts.append(
            {
                "type": "scalar_conflict",
                "message": f"Overlapping values detected for {key}: {joined_values}",
            }
        )
    return conflicts


def _extract_directive_conflicts(text: str) -> list[dict[str, str]]:
    lower = (text or "").lower()
    conflicts: list[dict[str, str]] = []
    for left, right, message in _CONTRADICTORY_DIRECTIVE_PAIRS:
        if left in lower and right in lower:
            conflicts.append({"type": "directive_conflict", "message": message})
    return conflicts


@router.post(
    "/{chat_id}/prompt-preview",
    summary="Preview assembled prompt with token budget breakdown",
    tags=["Chat Sessions"],
)
async def prompt_assembly_preview(
    chat_id: str = Path(..., description="Chat session ID"),
    body: CharacterChatCompletionPrepRequest = None,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Preview the full assembled prompt for a chat, including per-section token counts
    and budget enforcement.

    Returns the prompt sections (system prompt, greeting, author note, lorebook entries)
    with token estimates, budget caps, and truncation warnings.
    """
    try:
        body = body or CharacterChatCompletionPrepRequest()

        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        user_name = conversation.get("user_name", "User")
        include_ctx = bool(body.include_character_context)
        settings_row = db.get_conversation_settings(chat_id)
        settings = _extract_settings(settings_row) if isinstance(settings_row, dict) else {}

        messages = db.get_messages_for_conversation(chat_id, limit=body.limit, offset=body.offset) or []
        messages = [m for m in messages if not m.get("deleted")]
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get("deleted")]

        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
            directed_character_id=body.directed_character_id,
        )
        if (
            body.directed_character_id is not None
            and not turn_context.get("directed_character_applied")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="directed_character_id must reference a selected participant in this chat",
            )
        character = turn_context.get("active_character") or {}
        char_name = str(turn_context.get("active_character_name") or "Assistant")
        participant_aliases = turn_context.get("participant_aliases") or set()
        primary_character_name = str(turn_context.get("primary_character_name") or "")

        paginated = messages
        summary_content = ""
        paginated, summary_content = _apply_auto_summary_to_prompt_messages(
            db=db,
            chat_id=chat_id,
            settings_row=settings_row,
            prompt_messages=paginated,
            offset=body.offset,
            primary_character_name=primary_character_name,
            participant_aliases=participant_aliases,
            char_name=char_name,
            user_name=user_name,
        )

        preview_preset = _resolve_effective_prompt_preset(
            settings,
            character,
            request_preset=body.prompt_preset,
            db=db,
        )

        # Assemble prompt in the same order as completion-v2.
        formatted: list[dict[str, Any]] = []
        sys_text = ""
        if include_ctx and character and body.offset == 0:
            sys_text = _build_system_prompt_for_preset(
                db=db,
                character=character,
                char_name=char_name,
                user_name=user_name,
                preset_id=preview_preset,
            )
            if sys_text.strip():
                formatted.append({"role": "system", "content": sys_text.strip()})
        if summary_content:
            formatted.append({"role": "system", "content": f"Conversation summary:\n{summary_content}"})

        for msg in paginated:
            formatted.append({
                "role": _map_sender_to_role_with_participants(
                    msg.get("sender"),
                    primary_character_name,
                    participant_aliases,
                ),
                "content": _safe_replace_placeholders(msg.get("content"), char_name, user_name),
            })

        if body.append_user_message:
            formatted.append({"role": "user", "content": body.append_user_message})

        greeting_text = ""
        if _should_inject_character_scoped_greeting(
            settings=settings,
            turn_context=turn_context,
            history_messages=history_messages,
        ):
            greeting_text = _resolve_character_scoped_greeting(
                settings=settings,
                character=character,
                char_name=char_name,
                user_name=user_name,
            )
        formatted = _inject_character_scoped_greeting_from_settings(
            formatted,
            settings_row=settings_row,
            turn_context=turn_context,
            history_messages=history_messages,
            user_name=user_name,
        )

        note_text = _resolve_author_note_text(settings, character)
        if note_text:
            note_text = _safe_replace_placeholders(note_text, char_name, user_name).strip()
        author_note_text = f"Author's note:\n{note_text}" if note_text else ""
        formatted = _inject_author_note_from_settings(
            formatted,
            settings_row=settings_row,
            character=character,
            char_name=char_name,
            user_name=user_name,
        )
        formatted = _inject_character_memory_from_db(
            formatted,
            db=db,
            user_id=str(current_user.id),
            character_id=str(turn_context.get("active_character_id") or conversation.get("character_id", "")),
            char_name=char_name,
            user_name=user_name,
        )

        continue_flag, impersonate_flag, narrate_flag, steering_conflict = (
            _resolve_message_steering_flags(
                continue_as_user=body.continue_as_user,
                impersonate_user=body.impersonate_user,
                force_narrate=body.force_narrate,
            )
        )
        steering_text = _build_message_steering_instruction(
            continue_as_user=continue_flag,
            impersonate_user=impersonate_flag,
            force_narrate=narrate_flag,
            prompt_overrides=body.message_steering_prompts,
        )
        formatted, _ = _inject_message_steering_instruction(
            formatted,
            continue_as_user=body.continue_as_user,
            impersonate_user=body.impersonate_user,
            force_narrate=body.force_narrate,
            prompt_overrides=body.message_steering_prompts,
        )

        lorebook_text = ""
        lorebook_diagnostics: list[dict[str, Any]] = []
        try:
            from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

            wb_manager = WorldBookService(db)
            recent_text = " ".join(
                str(m.get("content", "")) for m in formatted if m.get("role") in ("user", "assistant")
            )[-2000:]
            if recent_text.strip():
                world_book_character_id = (
                    turn_context.get("active_character_id")
                    or conversation.get("character_id")
                )
                wb_result = wb_manager.process_context(
                    text=recent_text,
                    character_id=world_book_character_id,
                    include_diagnostics=True,
                )
                if isinstance(wb_result, dict):
                    wb_context = wb_result.get("processed_context", "")
                    lorebook_diagnostics = wb_result.get("diagnostics") or []
                    if wb_context and wb_context.strip():
                        lorebook_text = f"World info:\n{wb_context.strip()}"
                        insert_pos = 0
                        for idx, msg in enumerate(formatted):
                            role = str(msg.get("role", "")).strip().lower()
                            if role == "system":
                                insert_pos = idx + 1
                            else:
                                break
                        formatted.insert(
                            insert_pos,
                            {"role": "system", "content": lorebook_text},
                        )
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            pass

        persona_preview_sections: list[tuple[str, str, int]] = []
        if conversation.get("assistant_kind") == "persona" and conversation.get("assistant_id"):
            persona_preview_sections = _build_persona_preview_sections(
                conversation=conversation,
                exemplars=db.list_persona_exemplars(
                    user_id=str(current_user.id),
                    persona_id=str(conversation.get("assistant_id")),
                    include_disabled=False,
                    include_deleted=False,
                    limit=50,
                    offset=0,
                ),
                current_turn_text=body.append_user_message or next(
                    (
                        str(message.get("content") or "").strip()
                        for message in reversed(formatted)
                        if str(message.get("role") or "").strip().lower() == "user"
                        and str(message.get("content") or "").strip()
                    ),
                    "",
                ),
            )

        sections_raw: list[tuple[str, str, int]] = [
            ("preset", sys_text, _TOKEN_BUDGET_PRESET),
            ("author_note", author_note_text, _TOKEN_BUDGET_AUTHOR_NOTE),
            ("message_steering", steering_text, _TOKEN_BUDGET_STEERING),
            ("greeting", greeting_text, _TOKEN_BUDGET_GREETING),
            ("lorebook", lorebook_text, _TOKEN_BUDGET_LOREBOOK),
        ]
        sections_raw.extend(persona_preview_sections)
        sections: list[dict[str, Any]] = []
        total_supplemental_tokens = 0
        total_supplemental_effective_tokens = 0
        for name, content, budget in sections_raw:
            text = str(content or "").strip()
            estimated_tokens = _estimate_tokens(text)
            truncated_text = _truncate_to_budget(text, budget) if text else ""
            effective_tokens = _estimate_tokens(truncated_text)
            total_supplemental_tokens += estimated_tokens
            total_supplemental_effective_tokens += effective_tokens
            section_payload: dict[str, Any] = {
                "name": name,
                "content": truncated_text,
                "tokens_estimated": estimated_tokens,
                "tokens_effective": effective_tokens,
                "budget": budget,
                "truncated": bool(text) and truncated_text != text,
            }
            if name == "lorebook" and lorebook_diagnostics:
                section_payload["diagnostics"] = lorebook_diagnostics
            sections.append(section_payload)

        message_tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in messages)
        warnings: list[str] = []
        budget_status = "ok"
        if total_supplemental_effective_tokens > _TOKEN_BUDGET_TOTAL:
            budget_status = "error"
            warnings.append(
                f"Total supplemental tokens ({total_supplemental_effective_tokens}) exceed budget ({_TOKEN_BUDGET_TOTAL}). "
                f"Sections may be truncated in priority order: {', '.join(_TRUNCATION_PRIORITY)}."
            )
        elif total_supplemental_effective_tokens > int(_TOKEN_BUDGET_TOTAL * 0.9):
            budget_status = "caution"
            warnings.append(
                f"Total supplemental tokens ({total_supplemental_effective_tokens}) are at "
                f"{total_supplemental_effective_tokens * 100 // _TOKEN_BUDGET_TOTAL}% of budget ({_TOKEN_BUDGET_TOTAL})."
            )

        staleness = _check_greeting_staleness(settings, character)
        if staleness:
            warnings.append(staleness)
        if steering_conflict:
            warnings.append(
                "Steering conflict resolved: impersonate_user overrides continue_as_user."
            )

        conflict_source_sections = [str(content or "") for _, content, _ in sections_raw if str(content or "").strip()]
        conflict_text = "\n".join(conflict_source_sections)
        conflicts = _extract_scalar_conflicts(conflict_source_sections)
        conflicts.extend(_extract_directive_conflicts(conflict_text))

        return {
            "chat_id": chat_id,
            "character_id": turn_context.get("active_character_id") or conversation.get("character_id"),
            "character_name": char_name,
            "sections": sections,
            "total_supplemental_tokens": total_supplemental_tokens,
            "total_supplemental_effective_tokens": total_supplemental_effective_tokens,
            "supplemental_budget": _TOKEN_BUDGET_TOTAL,
            "budget_status": budget_status,
            "message_tokens_estimated": message_tokens,
            "message_count": len(messages),
            "warnings": warnings or None,
            "conflicts": conflicts or None,
            "examples": _PREVIEW_CONFLICT_EXAMPLES,
        }

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error generating prompt preview for {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating prompt preview",
        ) from e


@router.post(
    "/{chat_id}/complete-v2",
    response_model=CharacterChatCompletionV2Response,
    summary="Character Chat completion (operational, rate-limited)",
    description=(
        "Builds context from the chat and calls a provider.\n\n"
        "Streaming: when stream=true, response is sent as SSE and assistant content is NOT persisted,"
        " even if save_to_db=true. Use non-streaming to persist, or persist manually after streaming."
    ),
    tags=["Chat Sessions"],
)
async def character_chat_completion(
    chat_id: str = Path(..., description="Chat session ID"),
    body: CharacterChatCompletionV2Request = None,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    routing_decision_store: InMemoryRoutingDecisionStore = Depends(get_request_routing_decision_store),
    current_user: User = Depends(get_request_user),
    background_tasks: BackgroundTasks = None,
):
    """Perform a character chat completion using configured providers and persist results optionally.

    Behavior:
    - Enforces per-minute completion limiter (per-user)
    - Builds message context from the chat (and optional appended user message)
    - Calls provider via unified chat function
    - Persists appended user message and assistant reply when save_to_db is true

    Streaming behavior:
    - When `stream=True`, the response is returned as SSE and the assistant
      content is not persisted even if `save_to_db=True`. Use non-streaming
      mode to persist or persist separately after streaming completes.
    """
    try:
        import os
        body = body or CharacterChatCompletionV2Request()

        # Validate and ownership
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        # Prepare rate limiter
        rate_limiter = get_character_rate_limiter()

        # Gather character and context
        user_name = conversation.get('user_name', 'User')
        include_ctx = bool(body.include_character_context)
        limit = body.limit
        offset = body.offset
        stream_requested = bool(body.stream)
        save_to_db = body.save_to_db
        settings_row = db.get_conversation_settings(chat_id)
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get('deleted')]
        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
            directed_character_id=body.directed_character_id,
        )
        if (
            body.directed_character_id is not None
            and not turn_context.get("directed_character_applied")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="directed_character_id must reference a selected participant in this chat",
            )
        character = turn_context.get("active_character") or {}
        active_character_id = turn_context.get("active_character_id")
        active_character_name = turn_context.get("active_character_name")
        char_label = active_character_name or "Assistant"
        participant_aliases = turn_context.get("participant_aliases") or set()
        primary_character_name = turn_context.get("primary_character_name")
        completion_settings = _extract_settings(settings_row)
        character_generation_settings = _resolve_effective_generation_settings(completion_settings, character)
        resolved_temperature = (
            body.temperature
            if body.temperature is not None
            else character_generation_settings.get("temperature")
        )
        resolved_top_p = (
            body.top_p
            if body.top_p is not None
            else character_generation_settings.get("top_p")
        )
        resolved_repetition_penalty = (
            body.repetition_penalty
            if body.repetition_penalty is not None
            else character_generation_settings.get("repetition_penalty")
        )
        resolved_stop = (
            body.stop
            if body.stop is not None
            else character_generation_settings.get("stop")
        )
        if save_to_db is None:
            # Default from Chat API settings when not specified explicitly.
            try:
                from tldw_Server_API.app.api.v1.endpoints.chat import DEFAULT_SAVE_TO_DB as CHAT_DEFAULT_SAVE
                save_to_db = CHAT_DEFAULT_SAVE
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                save_to_db = False
        will_persist = bool(save_to_db) and not stream_requested

        messages = db.get_messages_for_conversation(chat_id, limit=limit, offset=offset) or []
        messages = [m for m in messages if not m.get('deleted')]
        paginated = messages
        summary_content = ""
        paginated, summary_content = _apply_auto_summary_to_prompt_messages(
            db=db,
            chat_id=chat_id,
            settings_row=settings_row,
            prompt_messages=paginated,
            offset=offset,
            primary_character_name=primary_character_name or "",
            participant_aliases=participant_aliases,
            char_name=char_label,
            user_name=user_name,
        )

        completion_preset = _resolve_effective_prompt_preset(
            completion_settings,
            character,
            request_preset=body.prompt_preset,
            db=db,
        )

        formatted: list[dict[str, Any]] = []
        if include_ctx and character and offset == 0:
            sys_text = _build_system_prompt_for_preset(
                db=db,
                character=character,
                char_name=char_label,
                user_name=user_name,
                preset_id=completion_preset,
            )
            if sys_text.strip():
                formatted.append({"role": "system", "content": sys_text.strip()})
        if summary_content:
            formatted.append(
                {"role": "system", "content": f"Conversation summary:\n{summary_content}"}
            )

        for msg in paginated:
            formatted.append({
                "role": _map_sender_to_role_with_participants(
                    msg.get('sender'),
                    primary_character_name,
                    participant_aliases,
                ),
                "content": _safe_replace_placeholders(msg.get('content'), char_label, user_name)
            })

        # Optional appended user message
        appended_user_id: Optional[str] = None
        if body.append_user_message:
            formatted.append({"role": "user", "content": body.append_user_message})

        formatted = _inject_character_scoped_greeting_from_settings(
            formatted,
            settings_row=settings_row,
            turn_context=turn_context,
            history_messages=history_messages,
            user_name=user_name,
        )
        formatted = _inject_author_note_from_settings(
            formatted,
            settings_row=settings_row,
            character=character,
            char_name=char_label,
            user_name=user_name,
        )
        formatted = _inject_character_memory_from_db(
            formatted,
            db=db,
            user_id=str(current_user.id),
            character_id=str(turn_context.get("active_character_id") or conversation.get("character_id", "")),
            char_name=char_label,
            user_name=user_name,
        )
        formatted, steering_conflict = _inject_message_steering_instruction(
            formatted,
            continue_as_user=body.continue_as_user,
            impersonate_user=body.impersonate_user,
            force_narrate=body.force_narrate,
            prompt_overrides=body.message_steering_prompts,
        )
        if steering_conflict:
            logger.debug(
                "Steering conflict resolved for chat {} in /complete-v2: impersonate_user overrides continue_as_user",
                chat_id,
            )

        # Inject world book context and capture diagnostics for this turn.
        turn_lorebook_diagnostics: Optional[list[dict[str, Any]]] = None
        try:
            from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
            wb_manager = WorldBookService(db)
            recent_text = " ".join(
                str(m.get("content", "")) for m in formatted if m.get("role") in ("user", "assistant")
            )[-2000:]
            if recent_text.strip():
                world_book_character_id = (
                    active_character_id
                    or conversation.get("character_id")
                )
                wb_result = wb_manager.process_context(
                    text=recent_text,
                    character_id=world_book_character_id,
                    include_diagnostics=True,
                )
                if isinstance(wb_result, dict):
                    wb_context = wb_result.get("processed_context", "")
                    turn_lorebook_diagnostics = wb_result.get("diagnostics") or None
                    if wb_context and wb_context.strip():
                        # Insert world book context as a system message after existing system messages.
                        insert_pos = 0
                        for idx, msg in enumerate(formatted):
                            if str(msg.get("role", "")).strip().lower() == "system":
                                insert_pos = idx + 1
                            else:
                                break
                        formatted.insert(insert_pos, {
                            "role": "system",
                            "content": f"World info:\n{wb_context.strip()}",
                        })
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            pass  # world book injection is best-effort

        # Determine provider/model with shared normalization and safe defaults.
        default_provider = _get_default_provider().strip()
        raw_model_input = str(body.model or "").strip()
        auto_model_requested = raw_model_input.lower() == "auto"
        explicit_model_requested = bool(raw_model_input) and not auto_model_requested
        strict_model_selection = _should_enforce_char_chat_strict_model_selection()
        raw_provider = (
            body.provider
            or os.getenv("CHAR_CHAT_PROVIDER")
            or None
        )
        if isinstance(raw_provider, str):
            raw_provider = raw_provider.strip() or None
        raw_model = (body.model or os.getenv("CHAR_CHAT_MODEL") or "local-test").strip()

        class _ProviderModelResolutionRequest:
            def __init__(self, provider_value: Optional[str], model_value: str):
                self.api_provider = provider_value
                self.model = model_value

        provider = (raw_provider or default_provider).strip()
        model = raw_model or "local-test"
        routing_decision = None
        if auto_model_requested:
            routing_decision, routing_debug = await _resolve_auto_character_chat_routing_decision(
            chat_id=chat_id,
            body=body,
            raw_provider=raw_provider,
            formatted_messages=formatted,
            sticky_store=routing_decision_store,
            current_user=current_user,
        )
            if routing_decision is None:
                candidate_count = int((routing_debug or {}).get("candidate_count") or 0)
                if candidate_count > 0:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={
                            "error_code": "auto_routing_failed",
                            "message": (
                                "Auto-routing failed and the current routing policy did not allow "
                                "deterministic fallback."
                            ),
                            "routing": routing_debug or {},
                        },
                    )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": "auto_routing_no_candidates",
                        "message": "No eligible models matched the current auto-routing constraints.",
                        "routing": routing_debug,
                    },
                )
        try:
            resolution_req = _ProviderModelResolutionRequest(raw_provider, raw_model)
            (
                _metrics_provider,
                _metrics_model,
                selected_provider,
                selected_model,
                provider_debug,
            ) = resolve_provider_and_model(
                request_data=resolution_req,
                metrics_default_provider=default_provider,
                normalize_default_provider=default_provider,
                routing_decision=routing_decision,
            )
            provider = (selected_provider or provider).strip()
            model = (selected_model or model).strip()
            logger.debug("Character provider/model resolution: {}", provider_debug)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
            logger.exception("Character provider/model resolution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_code": "provider_model_resolution_failed",
                    "message": "Failed to resolve provider and model for character chat completion.",
                },
            ) from exc

        if strict_model_selection and explicit_model_requested:
            availability = is_model_known_for_provider(provider, model)
            if availability is False:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error_code": "model_not_available",
                        "message": (
                            f"Model '{model}' is not available for provider '{provider}'. "
                            "Select one of the server-advertised models for this provider."
                        ),
                        "provider": provider,
                        "model": model,
                    },
                )

        # If we will persist, ensure message cap won't be exceeded.
        # Otherwise enforce a soft cap for non-persisted completions.
        try:
            current_count = db.count_messages_for_conversation(chat_id)
            if will_persist:
                will_add = 1 if body.append_user_message else 0
                will_add += 1  # assistant reply
                await rate_limiter.check_message_limit(chat_id, current_count + will_add)
            else:
                await rate_limiter.check_soft_message_limit(chat_id, current_count)
        except HTTPException:
            raise
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            logger.debug("Non-fatal: message cap pre-check skipped")

        # Resolve BYOK credentials (fall back to env/config)
        def _fallback_resolver(name: str) -> Optional[str]:
            try:
                from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
                return get_api_keys().get(name)
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                return None

        user_id_int: Optional[int] = None
        if hasattr(current_user, "id_int"):
            user_id_int = current_user.id_int
        elif hasattr(current_user, "id"):
            with contextlib.suppress(TypeError, ValueError):
                user_id_int = int(current_user.id)

        byok_resolution = await resolve_byok_credentials(
            provider,
            user_id=user_id_int,
            fallback_resolver=_fallback_resolver,
        )
        api_key = byok_resolution.api_key
        provider_key = (provider or "").strip().lower()
        if provider_requires_api_key(provider_key) and not api_key:
            record_byok_missing_credentials(provider_key, operation="character_chat")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "missing_provider_credentials",
                    "message": f"Provider '{provider}' requires an API key.",
                },
            )

        # Attempt provider call; allow offline simulation for local-llm in test/dev.
        # Offline simulation toggle (supports new flags for clarity, backward compatible with ALLOW_LOCAL_LLM_CALLS)
        enable_local_llm = parse_boolean(os.getenv("ENABLE_LOCAL_LLM_PROVIDER"))
        disable_offline_sim = parse_boolean(os.getenv("DISABLE_OFFLINE_SIM"))
        legacy_allow_local = parse_boolean(os.getenv("ALLOW_LOCAL_LLM_CALLS"))
        offline_sim = provider == "local-llm" and not (enable_local_llm or disable_offline_sim or legacy_allow_local)
        streams_unified = str(os.getenv("STREAMS_UNIFIED", "0")).strip().lower() in {"1", "true", "on", "yes"}
        llm_resp = None
        if not offline_sim:
            # Enforce per-minute completion rate only for real provider calls
            await rate_limiter.check_chat_completion_rate(current_user.id)
            try:
                llm_resp = perform_chat_api_call(
                    api_endpoint=provider,
                    messages_payload=formatted,
                    api_key=api_key,
                    temp=resolved_temperature,
                    top_p=resolved_top_p,
                    repetition_penalty=resolved_repetition_penalty,
                    stop=resolved_stop,
                    model=model,
                    max_tokens=body.max_tokens,
                    tools=body.tools,
                    tool_choice=body.tool_choice,
                    streaming=bool(body.stream),
                    user_identifier=str(current_user.id),
                    app_config=byok_resolution.app_config,
                )
                # Support async-returning provider hooks (test stubs or adapters)
                try:
                    if inspect.isawaitable(llm_resp):
                        llm_resp = await llm_resp  # type: ignore
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to await async LLM response: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="LLM provider error"
                    ) from e
            except ChatAPIError as e:
                logger.error("Chat provider call failed [{}]: {}", e.__class__.__name__, e)
                raise HTTPException(
                    status_code=int(getattr(e, "status_code", status.HTTP_502_BAD_GATEWAY)),
                    detail=str(e),
                ) from e
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Chat provider call failed: {e}")
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Chat provider error") from e
            await byok_resolution.touch_last_used()

        # Helper: Convert a provider chunk into a single SSE-formatted line
        def _coerce_sse_line(chunk: Any) -> Optional[str]:
            """Convert a provider chunk into a single SSE-formatted line.

            Prefer provider iterator output (already normalized). If chunk is not a
            string, attempt normalization; return None when nothing to forward.
            """
            try:
                if chunk is None:
                    return None
                if isinstance(chunk, (bytes, bytearray)):
                    text = chunk.decode("utf-8", errors="replace")
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    # As a fallback, stringify and normalize
                    text = str(chunk)
                if not text:
                    return None
                # If line looks like SSE control or data, keep as-is; otherwise normalize
                lower = text.strip().lower()
                if lower.startswith("data:") or lower.startswith("event:") or lower.startswith("id:") or lower.startswith("retry:") or lower.startswith(":"):
                    return ensure_sse_line(text.strip())
                normalized = normalize_provider_line(text)
                return normalized
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                return None

        # Extract assistant content from LLM response
        def _extract_text(resp: Any) -> str:
            if resp is None:
                return ""
            if isinstance(resp, str):
                return resp
            if isinstance(resp, (bytes, bytearray)):
                return resp.decode("utf-8", errors="replace")
            if isinstance(resp, dict):
                # OpenAI-style
                try:
                    return resp.get("choices", [{}])[0].get("message", {}).get("content", "") or resp.get("text", "")
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                    return resp.get("text", "")
            try:
                return str(resp)
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                return ""

        def _chunk_text(text: str, size: int = 2000) -> list[str]:
            if not text:
                return []
            return [text[i : i + size] for i in range(0, len(text), size)]

        def _stream_text_as_sse(text: str) -> StreamingResponse:
            async def _stream_text():
                try:
                    created_ts = int(time.time())
                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    model_id = model or "local-test"

                    head = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(head)}\n\n"

                    for chunk in _chunk_text(text):
                        data = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_id,
                            "choices": [
                                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    tail = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                    yield f"data: {json.dumps(tail)}\n\n"
                    yield "data: [DONE]\n\n"
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            headers = {}
            if streams_unified:
                headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            return StreamingResponse(_stream_text(), media_type="text/event-stream", headers=headers)

        # Initialize assistant_text to avoid potential UnboundLocalError in edge cases
        assistant_text = ""

        if offline_sim:
            # Simple deterministic response for tests/offline dev
            last_user = None
            for m in reversed(formatted):
                if m.get("role") == "user":
                    last_user = m.get("content")
                    break
            assistant_text = (last_user or "OK").strip()
            assistant_tool_calls = []
            # Streaming stub for offline-sim: emit SSE with plain text chunks and [DONE]
            if bool(body.stream):
                return _stream_text_as_sse(assistant_text or "OK")
        else:
            # For streaming, assistant text is not finalized here; skip extraction.
            assistant_tool_calls = []
            if not bool(body.stream):
                assistant_text = _extract_text(llm_resp).strip()
                # Try to extract tool calls if present (OpenAI-like shape)
                try:
                    if isinstance(llm_resp, dict):
                        tool_calls = llm_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls")
                        if isinstance(tool_calls, list):
                            assistant_tool_calls = tool_calls
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                    pass

        # If streaming requested and we have a generator, stream SSE (real providers)
        if not offline_sim and bool(body.stream):
            try:
                # Feature flag: use unified SSEStream when enabled
                if streams_unified:
                    # Unified path expects an iterator; fall back to text streaming for non-iterables.
                    if not hasattr(llm_resp, "__aiter__") and not (
                        hasattr(llm_resp, "__iter__") and not isinstance(llm_resp, (str, bytes, dict, list))
                    ):
                        assistant_text_fallback = _extract_text(llm_resp).strip()
                        return _stream_text_as_sse(assistant_text_fallback)

                    stream = SSEStream(
                        labels={"component": "chat", "endpoint": "character_chat_stream"}
                    )
                    with contextlib.suppress(_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS):
                        logger.debug(
                            f"Unified SSE enabled: interval={stream.heartbeat_interval_s} mode={stream.heartbeat_mode}"
                        )

                    chunk_count = 0
                    total_bytes = 0

                    async def _emit_stream_limit_error(message: str) -> None:
                        payload = {"error": message}
                        await stream.send_raw_sse_line(f"data: {json.dumps(payload)}")
                        await stream.done()

                    async def _handle_chunk(chunk: Any) -> bool:
                        nonlocal chunk_count, total_bytes
                        chunk_count += 1
                        if chunk_count > MAX_STREAMING_CHUNKS:
                            logger.warning(f"Streaming chunk limit exceeded ({MAX_STREAMING_CHUNKS})")
                            await _emit_stream_limit_error("Streaming limit exceeded.")
                            return False

                        line = _coerce_sse_line(chunk)
                        if not line:
                            return True

                        total_bytes += len(line.encode("utf-8"))
                        if total_bytes > MAX_STREAMING_BYTES:
                            logger.warning(f"Streaming byte limit exceeded ({MAX_STREAMING_BYTES})")
                            await _emit_stream_limit_error("Streaming size limit exceeded.")
                            return False

                        if line.strip().lower() == "data: [done]":
                            await stream.done()
                            return False

                        await stream.send_raw_sse_line(line)
                        return True

                    async def _produce_async():
                        try:
                            if hasattr(llm_resp, "__aiter__"):
                                async for chunk in llm_resp:  # type: ignore
                                    keep_going = await _handle_chunk(chunk)
                                    if not keep_going:
                                        return
                            elif hasattr(llm_resp, "__iter__") and not isinstance(llm_resp, (str, bytes, dict, list)):
                                for chunk in llm_resp:  # type: ignore
                                    keep_going = await _handle_chunk(chunk)
                                    if not keep_going:
                                        return
                            # Ensure DONE if provider didn't send one
                            await stream.done()
                        except ChatAPIError as e:
                            await stream.error("provider_error", str(e))
                        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                            await stream.error("internal_error", f"{e}")
                        except Exception:
                            logger.exception("Unhandled exception in character chat streaming producer")
                            await stream.error("internal_error", "An internal error has occurred.")

                    async def _generator():
                        producer = asyncio.create_task(_produce_async())
                        try:
                            async for line in stream.iter_sse():
                                yield line
                        except asyncio.CancelledError:
                            # Preserve cancellation semantics; cleanup happens in finally
                            raise
                        else:
                            # Ensure producer completes if stream ended without explicit DONE
                            if not producer.done():
                                with contextlib.suppress(_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS):
                                    await producer
                            # If DONE wasn’t enqueued for any reason, append one now
                            try:
                                if not getattr(stream, "_done_enqueued", False):
                                    yield sse_done()
                            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                                pass
                        finally:
                            # Always tear down the background producer to avoid leaks
                            if not producer.done():
                                with contextlib.suppress(_CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS):
                                    producer.cancel()
                                try:
                                    await producer
                                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                                    # Swallow any errors from producer teardown
                                    pass

                    headers = {
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    }
                    return StreamingResponse(_generator(), media_type="text/event-stream", headers=headers)
                # Legacy path (flag off): stream directly (provider iterator yields SSE lines)
                # Support async generators
                if hasattr(llm_resp, "__aiter__"):
                    async def _sse_async():
                        done_sent = False
                        chunk_count = 0
                        total_bytes = 0
                        try:
                            async for chunk in llm_resp:  # type: ignore
                                # Safety limits to prevent DoS
                                chunk_count += 1
                                if chunk_count > MAX_STREAMING_CHUNKS:
                                    logger.warning(f"Streaming chunk limit exceeded ({MAX_STREAMING_CHUNKS})")
                                    yield f"data: {json.dumps({'error': 'Streaming limit exceeded.'})}\n\n"
                                    break

                                line = _coerce_sse_line(chunk)
                                if not line:
                                    continue

                                total_bytes += len(line.encode('utf-8'))
                                if total_bytes > MAX_STREAMING_BYTES:
                                    logger.warning(f"Streaming byte limit exceeded ({MAX_STREAMING_BYTES})")
                                    yield f"data: {json.dumps({'error': 'Streaming size limit exceeded.'})}\n\n"
                                    break

                                normalized = line.strip().lower()
                                if normalized == "data: [done]":
                                    done_sent = True
                                yield ensure_sse_line(line)
                        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                            if isinstance(e, AttributeError) and "object has no attribute 'close'" in str(e):
                                logger.debug("Ignoring streaming session close error: {}", e)
                            else:
                                logger.exception("Exception occurred in streaming SSE async generator.")
                                yield f"data: {json.dumps({'error': 'An internal error has occurred.'})}\n\n"
                        finally:
                            if not done_sent:
                                yield "data: [DONE]\n\n"
                    # Note: streaming mode does not persist assistant content
                    return StreamingResponse(_sse_async(), media_type="text/event-stream")
                # Support sync generators/iterables that are not plain containers
                if hasattr(llm_resp, "__iter__") and not isinstance(llm_resp, (str, bytes, dict, list)):
                    async def _sse_gen():
                        done_sent = False
                        chunk_count = 0
                        total_bytes = 0
                        try:
                            for chunk in llm_resp:  # type: ignore
                                # Safety limits to prevent DoS
                                chunk_count += 1
                                if chunk_count > MAX_STREAMING_CHUNKS:
                                    logger.warning(f"Streaming chunk limit exceeded ({MAX_STREAMING_CHUNKS})")
                                    yield f"data: {json.dumps({'error': 'Streaming limit exceeded.'})}\n\n"
                                    break

                                line = _coerce_sse_line(chunk)
                                if not line:
                                    continue

                                total_bytes += len(line.encode('utf-8'))
                                if total_bytes > MAX_STREAMING_BYTES:
                                    logger.warning(f"Streaming byte limit exceeded ({MAX_STREAMING_BYTES})")
                                    yield f"data: {json.dumps({'error': 'Streaming size limit exceeded.'})}\n\n"
                                    break

                                normalized = line.strip().lower()
                                if normalized == "data: [done]":
                                    done_sent = True
                                yield ensure_sse_line(line)
                        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                            logger.exception("Exception occurred in streaming SSE generator.")
                            yield f"data: {json.dumps({'error': 'An internal error has occurred.'})}\n\n"
                        finally:
                            if not done_sent:
                                yield "data: [DONE]\n\n"
                    # Note: streaming mode does not persist assistant content
                    return StreamingResponse(_sse_gen(), media_type="text/event-stream")
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                # Fall through to non-streaming response
                pass
            if isinstance(llm_resp, (dict, str, bytes, bytearray)):
                assistant_text_fallback = _extract_text(llm_resp).strip()
                return _stream_text_as_sse(assistant_text_fallback)
        if not assistant_text:
            assistant_text = ""

        resolved_mood_label = (
            body.mood_label.strip()
            if isinstance(body.mood_label, str) and body.mood_label.strip()
            else None
        )
        resolved_mood_confidence = (
            float(body.mood_confidence) if body.mood_confidence is not None else None
        )
        resolved_mood_topic = (
            body.mood_topic.strip()
            if isinstance(body.mood_topic, str) and body.mood_topic.strip()
            else None
        )

        saved = False
        assistant_msg_id: Optional[str] = None
        if will_persist:
            # Persist appended user message first, if any
            if body.append_user_message:
                appended_user_id = post_message_to_conversation(
                    db=db,
                    conversation_id=chat_id,
                    character_name=char_label,
                    message_content=body.append_user_message,
                    is_user_message=True,
                    sender_override="user",
                )
                if not appended_user_id:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to persist appended user message"
                    )

            # Persist assistant response (tool_calls stored in metadata only)
            assistant_content_for_storage = assistant_text if assistant_text else " "
            assistant_msg_id = post_message_to_conversation(
                db=db,
                conversation_id=chat_id,
                character_name=char_label,
                message_content=assistant_content_for_storage,
                is_user_message=False,
                parent_message_id=appended_user_id,
            )
            if not assistant_msg_id:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to persist assistant message"
                )
            metadata_extra: dict[str, Any] = {
                "speaker_character_id": active_character_id,
                "speaker_character_name": char_label,
                "turn_taking_mode": turn_context.get("turn_taking_mode", "single"),
            }
            if resolved_mood_label:
                metadata_extra["mood_label"] = resolved_mood_label
            if resolved_mood_confidence is not None:
                metadata_extra["mood_confidence"] = resolved_mood_confidence
            if resolved_mood_topic:
                metadata_extra["mood_topic"] = resolved_mood_topic
            if turn_lorebook_diagnostics:
                metadata_extra["lorebook_diagnostics"] = turn_lorebook_diagnostics
            validated_tool_calls = (
                _validate_and_truncate_tool_calls(assistant_tool_calls)
                if assistant_tool_calls
                else None
            )
            try:
                db.add_message_metadata(
                    assistant_msg_id,
                    tool_calls=validated_tool_calls,
                    extra=metadata_extra,
                )
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Non-fatal: failed to persist assistant metadata: {exc}")
            saved = True

            # --- Character memory extraction trigger (opt-in) ---
            _maybe_trigger_character_memory_extraction(
                background_tasks=background_tasks,
                db=db,
                chat_id=chat_id,
                settings_row=settings_row,
                conversation=conversation,
                character_id=str(active_character_id or conversation.get("character_id", "")),
                char_name=char_label,
                user_id=str(current_user.id),
            )

        return CharacterChatCompletionV2Response(
            chat_id=chat_id,
            character_id=active_character_id or conversation['character_id'],
            provider=provider,
            model=model,
            saved=saved,
            user_message_id=appended_user_id,
            assistant_message_id=assistant_msg_id,
            assistant_content=assistant_text,
            speaker_character_id=active_character_id,
            speaker_character_name=char_label,
            mood_label=resolved_mood_label,
            mood_confidence=resolved_mood_confidence,
            mood_topic=resolved_mood_topic,
            lorebook_diagnostics=turn_lorebook_diagnostics,
        )

    except HTTPException:
        raise
    except InputError as e:
        msg = str(e)
        status_code = status.HTTP_400_BAD_REQUEST
        if "exceeds maximum" in msg.lower():
            status_code = status.HTTP_413_CONTENT_TOO_LARGE
        logger.warning(f"Input error in character chat completion for {chat_id}: {e}")
        raise HTTPException(status_code=status_code, detail=msg) from e
    except ConflictError as e:
        logger.warning(f"Conflict in character chat completion for {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error in character chat completion for {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error in character chat completion for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during character chat completion") from e


@router.get("/", response_model=ChatSessionListResponse,
            summary="List user's chat sessions", tags=["Chat Sessions"])
async def list_chat_sessions(
    character_id: Optional[int] = Query(None, description="Filter by character ID"),
    character_scope: Literal["all", "character", "non_character"] = Query(
        "all",
        description="Filter chats by whether they are character-backed or not",
    ),
    limit: int = Query(50, ge=1, le=200, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    include_deleted: bool = Query(False, description="Include soft-deleted chats"),
    deleted_only: bool = Query(False, description="Return only soft-deleted chats"),
    include_settings: bool = Query(
        False,
        description="Include per-chat settings payload for each returned chat.",
    ),
    scope_type: Optional[Literal["global", "workspace"]] = Query(None, description="Scope filter: 'global' or 'workspace'"),
    workspace_id: Optional[str] = Query(None, description="Workspace ID (required when scope_type='workspace')"),
    include_message_counts: bool = Query(
        True,
        description="Include message counts for each returned chat.",
    ),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    List all chat sessions for the current user.

    Args:
        character_id: Optional character ID filter
        limit: Maximum number of items to return
        offset: Number of items to skip
        db: Database instance
        current_user: Authenticated user

    Returns:
        List of chat sessions with pagination info
    """
    try:
        user_id_str = str(current_user.id)
        include_deleted_effective = include_deleted or deleted_only
        scope = _resolve_chat_scope(scope_type, workspace_id)
        if character_id is not None and character_scope == "non_character":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="character_scope=non_character cannot be combined with character_id",
            )

        if character_id is not None:
            # Get conversations for specific character scoped to current user
            conversations = db.get_conversations_for_user_and_character(
                user_id_str,
                character_id,
                limit=limit,
                offset=offset,
                include_deleted=include_deleted_effective,
                deleted_only=deleted_only,
                scope_type=scope.scope_type,
                workspace_id=scope.workspace_id,
            )
            try:
                total_count = db.count_conversations_for_user_by_character(
                    user_id_str,
                    character_id,
                    include_deleted=include_deleted_effective,
                    deleted_only=deleted_only,
                    scope_type=scope.scope_type,
                    workspace_id=scope.workspace_id,
                )
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                # Fallback: filter by client_id in-memory if efficient count isn't available
                total_count = len([c for c in conversations if c.get('client_id') == user_id_str])
        else:
            # Efficient path: list conversations for this user directly
            conversations = db.get_conversations_for_user(
                user_id_str,
                limit=limit,
                offset=offset,
                include_deleted=include_deleted_effective,
                deleted_only=deleted_only,
                scope_type=scope.scope_type,
                workspace_id=scope.workspace_id,
                character_scope=character_scope,
            )
            try:
                total_count = db.count_conversations_for_user(
                    user_id_str,
                    include_deleted=include_deleted_effective,
                    deleted_only=deleted_only,
                    character_scope=character_scope,
                    scope_type=scope.scope_type,
                    workspace_id=scope.workspace_id,
                )
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                total_count = len(conversations)

        # Filter by client_id for security (redundant in happy path, kept defensively)
        user_conversations = [conv for conv in conversations if conv.get('client_id') == user_id_str]

        # Sort by last_modified descending
        user_conversations.sort(key=lambda x: x.get('last_modified', ''), reverse=True)

        # Note: pagination was already applied at DB level, no need to slice again

        if include_message_counts:
            message_counts: dict[str, int] = {}
            countable_conversation_ids = [
                str(conv["id"])
                for conv in user_conversations
                if conv.get("id") and not conv.get("deleted")
            ]
            if countable_conversation_ids:
                try:
                    message_counts = db.count_messages_for_conversations(countable_conversation_ids)
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                    message_counts = {}

            for conv in user_conversations:
                if conv.get("deleted"):
                    conv["message_count"] = 0
                    continue

                conv_id = conv.get("id")
                if conv_id in message_counts:
                    conv["message_count"] = message_counts.get(conv_id, 0)
                    continue

                messages = db.get_messages_for_conversation(conv["id"], limit=1000)
                conv["message_count"] = len(messages) if messages else 0
        else:
            for conv in user_conversations:
                conv["message_count"] = None

        chats: list[ChatSessionResponse] = []
        for conv in user_conversations:
            settings_payload: Optional[dict[str, Any]] = None
            if include_settings:
                settings_row = db.get_conversation_settings(conv['id'])
                settings_payload = (settings_row or {}).get("settings") or {}
            chats.append(
                _convert_db_conversation_to_response(
                    conv,
                    settings=settings_payload,
                )
            )

        return ChatSessionListResponse(
            chats=chats,
            total=total_count,
            limit=limit,
            offset=offset
        )

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error listing chat sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing chat sessions"
        ) from e


@router.put("/{chat_id}", response_model=ChatSessionResponse,
            summary="Update chat session", tags=["Chat Sessions"])
async def update_chat_session(
    update_data: ChatSessionUpdate,
    chat_id: str = Path(..., description="Chat session ID"),
    expected_version: int = Query(..., description="Expected version for optimistic locking"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Update a chat session's metadata.

    Args:
        chat_id: Chat session ID
        update_data: Update data
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user

    Returns:
        Updated chat session details

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        # Get current conversation
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        # Check version
        if conversation.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {conversation.get('version', 1)}"
            )

        # Update fields via DB abstraction with optimistic locking
        update_fields = update_data.model_dump(exclude_unset=True)
        # Only allow supported fields
        allowed_update = {
            k: v
            for k, v in update_fields.items()
            if k in {"title", "rating", "state", "topic_label", "cluster_id", "source", "external_ref"}
        }
        # db.update_conversation updates metadata and bumps version even if payload is empty
        db.update_conversation(chat_id, allowed_update, expected_version)

        # Retrieve updated conversation
        updated_conv = db.get_conversation_by_id(chat_id)
        if updated_conv:
            messages = db.get_messages_for_conversation(chat_id, limit=1000)
            updated_conv['message_count'] = len(messages) if messages else 0

        return _convert_db_conversation_to_response(updated_conv)

    except ConflictError as e:
        # Optimistic locking or state conflicts
        logger.warning(f"Conflict updating chat session {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error updating chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error updating chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating chat session"
        ) from e


@router.get(
    "/{chat_id}/settings",
    response_model=ChatSettingsResponse,
    summary="Get chat settings",
    tags=["Chat Sessions"],
)
async def get_chat_settings(
    chat_id: str = Path(..., description="Chat session ID"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        settings_row = db.get_conversation_settings(chat_id)
        if not settings_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat settings not found")

        settings = settings_row.get("settings") or {}
        # Internal bootstrap metadata alone should not count as user-visible settings.
        if settings and set(settings.keys()) <= {"greetingsChecksum"}:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat settings not found")

        settings = _validate_chat_settings_payload(
            settings,
            owner_user_id=str(current_user.id),
            strip_invalid_deep_research=True,
        )

        # Normalize stored enum values so the client always gets valid scopes.
        _SCOPE_DEFAULTS = {
            "greetingScope": ("chat", {"chat", "character"}),
            "presetScope": ("character", {"chat", "character"}),
            "memoryScope": ("shared", {"shared", "character", "both"}),
        }
        for scope_key, (default_val, allowed) in _SCOPE_DEFAULTS.items():
            raw = settings.get(scope_key)
            if raw is not None:
                normalized = raw.strip().lower() if isinstance(raw, str) else None
                if normalized not in allowed:
                    settings[scope_key] = default_val

        # Normalize characterMemoryById string entries to dicts on read.
        mem_by_id = settings.get("characterMemoryById")
        if isinstance(mem_by_id, dict):
            for cid, entry in mem_by_id.items():
                if isinstance(entry, str):
                    mem_by_id[cid] = {"note": entry}

        # Check for greeting staleness if a character is associated.
        warnings: list[str] = []
        char_id = conversation.get("character_id") if conversation else None
        if char_id is not None:
            try:
                character = db.get_character_card_by_id(char_id)
                if character:
                    staleness_warning = _check_greeting_staleness(settings, character)
                    if staleness_warning:
                        warnings.append(staleness_warning)
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                pass  # best-effort

        return ChatSettingsResponse(
            conversation_id=chat_id,
            settings=settings,
            last_modified=settings_row.get("last_modified") or datetime.now(timezone.utc),
            warnings=warnings or None,
        )
    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Error fetching chat settings for {chat_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch chat settings") from exc


@router.put(
    "/{chat_id}/settings",
    response_model=ChatSettingsResponse,
    summary="Update chat settings",
    tags=["Chat Sessions"],
)
async def update_chat_settings(
    payload: ChatSettingsUpdate,
    chat_id: str = Path(..., description="Chat session ID"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        incoming_settings = _validate_chat_settings_payload(
            payload.settings or {},
            owner_user_id=str(current_user.id),
        )

        existing_row = db.get_conversation_settings(chat_id)
        existing_settings = (existing_row or {}).get("settings") or {}
        merged_settings = _merge_conversation_settings(existing_settings, incoming_settings)
        merged_settings = _validate_chat_settings_payload(
            merged_settings,
            owner_user_id=str(current_user.id),
        )

        if not db.upsert_conversation_settings(chat_id, merged_settings):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update chat settings")

        settings_row = db.get_conversation_settings(chat_id)

        return ChatSettingsResponse(
            conversation_id=chat_id,
            settings=(settings_row or {}).get("settings") or merged_settings,
            last_modified=(settings_row or {}).get("last_modified") or datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Error updating chat settings for {chat_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update chat settings") from exc


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete chat session",
    tags=["Chat Sessions"],
)
async def delete_chat_session(
    chat_id: str = Path(..., description="Chat session ID"),
    expected_version: Optional[int] = Query(None, description="Expected version for optimistic locking"),
    hard_delete: bool = Query(False, description="Permanently delete a chat already in trash"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
) -> Response:
    """
    Soft delete a chat session.

    Args:
        chat_id: Chat session ID
        expected_version: Expected version for optimistic locking
        db: Database instance
        current_user: Authenticated user

    Raises:
        HTTPException: 404 if not found, 403 if unauthorized, 409 if version conflict
    """
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        # Get current conversation. Hard-delete may target already deleted rows.
        conversation = db.get_conversation_by_id(chat_id, include_deleted=hard_delete)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        if hard_delete:
            if not conversation.get("deleted"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Chat must be in trash before permanent delete.",
                )

            if expected_version is not None and conversation.get('version', 1) != expected_version:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Version mismatch. Expected {expected_version}, found {conversation.get('version', 1)}"
                )

            deleted_ok = db.hard_delete_conversation(chat_id)
            if not deleted_ok:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {chat_id} not found",
                )
            logger.info(f"Hard deleted chat session {chat_id} by user {current_user.id}")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # Check version if provided
        if expected_version is not None and conversation.get('version', 1) != expected_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version mismatch. Expected {expected_version}, found {conversation.get('version', 1)}"
            )

        # Delete messages in batches using offset-based pagination to avoid unbounded memory
        # Track failed message IDs to prevent infinite loops when deletions fail
        batch_size = 100
        max_batches = 1000  # Safety limit: max 100,000 messages per conversation
        total_deleted = 0
        consecutive_empty_batches = 0
        max_empty_batches = 3  # Stop after consecutive empty batches (indicates completion)
        failed_message_ids: set = set()  # Track messages that failed to delete

        for _batch_num in range(max_batches):
            # Fetch non-deleted messages only (include_deleted=False is default)
            batch = db.get_messages_for_conversation(chat_id, limit=batch_size, offset=0)

            # Filter out messages that previously failed to delete to prevent infinite loops
            batch = [m for m in batch if m.get("id") not in failed_message_ids]

            if not batch:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_empty_batches:
                    break
                continue

            consecutive_empty_batches = 0  # Reset counter on successful batch

            # Delete messages in this batch
            batch_deleted = 0
            for msg in batch:
                msg_id = msg.get("id")
                if not msg_id:
                    continue
                try:
                    db.soft_delete_message(msg_id, msg.get("version", 1))
                    batch_deleted += 1
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                    logger.warning("Failed to soft-delete message {} during conversation delete.", msg_id)
                    failed_message_ids.add(msg_id)  # Track failed deletions

            total_deleted += batch_deleted

            # If we couldn't delete any messages in this batch, we might be stuck
            if batch_deleted == 0:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_empty_batches:
                    logger.warning(f"Stopping message deletion for {chat_id} after {total_deleted} messages - possible stuck state")
                    break

        if failed_message_ids:
            logger.warning(f"Failed to delete {len(failed_message_ids)} messages from conversation {chat_id}: {failed_message_ids}")

        logger.debug(f"Deleted {total_deleted} messages from conversation {chat_id}")

        # Soft delete conversation via DB abstraction (optimistic locking)
        exp_ver = expected_version if expected_version is not None else conversation.get('version', 1)
        # Finally soft delete the conversation
        db.soft_delete_conversation(chat_id, exp_ver)

        logger.info(f"Soft deleted chat session {chat_id} by user {current_user.id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except ConflictError as e:
        logger.warning(f"Conflict deleting chat session {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error deleting chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error deleting chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting chat session"
        ) from e


@router.post(
    "/{chat_id}/restore",
    response_model=ChatSessionResponse,
    summary="Restore chat session from trash",
    tags=["Chat Sessions"],
)
async def restore_chat_session(
    chat_id: str = Path(..., description="Chat session ID"),
    expected_version: Optional[int] = Query(None, description="Expected version for optimistic locking"),
    scope_type: Literal["global", "workspace"] | None = Query(None, description="Conversation scope type"),
    workspace_id: str | None = Query(None, description="Workspace ID when scope_type='workspace'"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    try:
        scope = _resolve_chat_scope(scope_type, workspace_id)
        conversation = db.get_conversation_by_id(chat_id, include_deleted=True)
        _verify_chat_ownership(conversation, current_user.id, chat_id, scope)

        # Already active: return current state as idempotent success.
        if not conversation.get("deleted"):
            try:
                conversation['message_count'] = db.count_messages_for_conversation(chat_id)
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                conversation['message_count'] = 0
            return _convert_db_conversation_to_response(conversation)

        exp_ver = expected_version if expected_version is not None else conversation.get("version", 1)
        db.restore_conversation(chat_id, exp_ver)

        restored = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(restored, current_user.id, chat_id, scope)
        try:
            restored['message_count'] = db.count_messages_for_conversation(chat_id)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            restored['message_count'] = 0
        return _convert_db_conversation_to_response(restored)

    except ConflictError as e:
        logger.warning(f"Conflict restoring chat session {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error restoring chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error restoring chat session {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while restoring chat session",
        ) from e


# ========================================================================
# Note: Character chat completions should use the main /api/v1/chat/completions endpoint
# To get messages formatted for completions, use:
# GET /api/v1/chats/{chat_id}/messages?format_for_completions=true&include_character_context=true
# ========================================================================


# ========================================================================
# Chat Export Endpoint
# ========================================================================

# Maximum messages per export page to prevent DoS
MAX_EXPORT_PAGE_SIZE = 5000
DEFAULT_EXPORT_PAGE_SIZE = 1000


@router.get("/{chat_id}/export",
            summary="Export chat history", tags=["Chat Export"])
async def export_chat_history(
    chat_id: str = Path(..., description="Chat session ID"),
    format: str = Query("json", description="Export format (json, markdown, text)"),
    include_metadata: bool = Query(True, description="Include chat metadata"),
    include_character: bool = Query(True, description="Include character info"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(DEFAULT_EXPORT_PAGE_SIZE, ge=1, le=MAX_EXPORT_PAGE_SIZE,
                          description=f"Messages per page (max {MAX_EXPORT_PAGE_SIZE})"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user)
):
    """
    Export chat history in various formats with pagination support.

    Args:
        chat_id: Chat session ID to export
        format: Export format (json, markdown, text)
        include_metadata: Whether to include metadata
        include_character: Whether to include character info
        page: Page number (1-indexed, default 1)
        page_size: Number of messages per page (default 1000, max 5000)
        db: Database instance
        current_user: Authenticated user

    Returns:
        Chat history in requested format with pagination info

    Raises:
        HTTPException: 404 if chat not found, 403 if unauthorized
    """
    try:
        # Get conversation
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        settings_row = db.get_conversation_settings(chat_id)
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get("deleted")]
        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
        )
        participant_aliases = turn_context.get("participant_aliases") or set()
        primary_character_name = turn_context.get("primary_character_name")

        # Get character info if requested
        character = None
        if include_character:
            character = db.get_character_card_by_id(conversation['character_id'])

        # Get total message count for pagination info
        try:
            total_messages = db.count_messages_for_conversation(chat_id)
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
            # Fallback if count method not available
            total_messages = None

        # Calculate offset for pagination
        offset = (page - 1) * page_size

        # Get messages with pagination
        messages = db.get_messages_for_conversation(chat_id, limit=page_size, offset=offset)

        # Calculate pagination metadata
        total_pages = None
        has_more = False
        if total_messages is not None:
            total_pages = (total_messages + page_size - 1) // page_size
            has_more = page < total_pages

        # Format based on requested type
        if format == "markdown":
            # Markdown format
            lines = []
            if include_metadata:
                lines.append(f"# Chat Export: {conversation.get('title', 'Untitled')}")
                if character:
                    lines.append(f"**Character**: {character.get('name', 'Unknown')}")
                lines.append(f"**Date**: {conversation.get('created_at', '')}")
                if total_messages is not None:
                    lines.append(f"**Messages**: {len(messages)} of {total_messages} (page {page})")
                else:
                    lines.append(f"**Messages**: {len(messages)}")
                lines.append("\n---\n")

            for msg in messages:
                if msg.get('deleted'):
                    continue
                sender = msg.get('sender', 'unknown')
                content = msg.get('content') or ''
                timestamp = msg.get('timestamp', '')
                lines.append(f"**{sender.title()}** ({timestamp}):")
                lines.append(f"{content}\n")

            return {"content": "\n".join(lines), "format": "markdown"}

        elif format == "text":
            # Plain text format
            lines = []
            if include_metadata:
                lines.append(f"Chat: {conversation.get('title', 'Untitled')}")
                if character:
                    lines.append(f"Character: {character.get('name', 'Unknown')}")
                lines.append(f"Date: {conversation.get('created_at', '')}")
                lines.append("-" * 40)

            for msg in messages:
                if msg.get('deleted'):
                    continue
                sender = msg.get('sender', 'unknown')
                content = msg.get('content') or ''
                lines.append(f"{sender}: {content}")

            return {"content": "\n".join(lines), "format": "text"}

        else:
            # JSON format (default)
            export_data = {
                "chat_id": chat_id,
                "character_name": character.get('name', 'Unknown') if character else 'Unknown',
                "character_id": conversation['character_id'],
                "title": conversation.get('title'),
                "created_at": str(conversation.get('created_at', '')),
                "messages": []
            }
            message_metadata_extra: dict[str, Any] = {}
            # Build messages with optional tool_calls per message
            for msg in messages:
                if msg.get('deleted'):
                    continue
                item = {
                    "id": msg.get('id'),
                    "role": msg.get('sender'),
                    "content": msg.get('content') or '',
                    "timestamp": str(msg.get('timestamp', '')),
                    "has_image": bool(msg.get('image_data'))
                }
                try:
                    md = db.get_message_metadata(msg.get('id'))
                except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                    md = None
                role_for_tool_calls = _map_sender_to_role_with_participants(
                    msg.get('sender'),
                    primary_character_name,
                    participant_aliases,
                )
                if md and md.get('tool_calls') is not None:
                    item["tool_calls"] = md.get('tool_calls')
                elif role_for_tool_calls == 'assistant':
                    # Fallback: parse inline suffix [tool_calls]: <json>
                    try:
                        import json as _json
                        import re as _re
                        m = _re.search(r"\[tool_calls\]\s*:\s*(\{.*|\[.*)$", (msg.get('content') or ''), _re.DOTALL)
                        if m:
                            parsed = _json.loads(m.group(1).strip())
                            if isinstance(parsed, dict) and 'tool_calls' in parsed:
                                tc_list = parsed.get('tool_calls')
                            else:
                                tc_list = parsed
                            if isinstance(tc_list, list):
                                item["tool_calls"] = tc_list
                    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS:
                        pass
                export_data["messages"].append(item)
                if include_metadata and md and md.get('extra') is not None and msg.get('id'):
                    message_metadata_extra[msg.get('id')] = md.get('extra')

            if include_metadata:
                export_data["metadata"] = {
                    "total_messages": total_messages if total_messages is not None else len(messages),
                    "rating": conversation.get('rating'),
                    "last_modified": str(conversation.get('last_modified', ''))
                }
                if message_metadata_extra:
                    export_data["message_metadata_extra"] = message_metadata_extra

            # Add pagination info to JSON export
            export_data["pagination"] = {
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "total_messages": total_messages,
                "has_more": has_more,
                "messages_in_page": len(messages)
            }

            return export_data

    except HTTPException:
        raise
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error exporting chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while exporting chat history"
        ) from e


# ========================================================================
# Persist streamed assistant content
# ========================================================================

@router.post(
    "/{chat_id}/completions/persist",
    response_model=CharacterChatStreamPersistResponse,
    summary="Persist streamed assistant content",
    description=(
        "Persist an assistant message after a streamed completion. "
        "Optionally links to a prior user_message_id and stores tool_calls metadata."
    ),
    tags=["Chat Sessions"],
)
async def persist_streamed_assistant_message(
    chat_id: str = Path(..., description="Chat session ID"),
    body: CharacterChatStreamPersistRequest = Body(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    try:
        conversation = db.get_conversation_by_id(chat_id)
        _verify_chat_ownership(conversation, current_user.id, chat_id)

        settings_row = db.get_conversation_settings(chat_id)
        history_messages = db.get_messages_for_conversation(chat_id, limit=1000, offset=0) or []
        history_messages = [m for m in history_messages if not m.get("deleted")]

        turn_context = _resolve_chat_turn_context(
            db=db,
            conversation=conversation,
            settings_row=settings_row,
            history_messages=history_messages,
            directed_character_id=body.speaker_character_id,
        )
        if (
            body.speaker_character_id is not None
            and not turn_context.get("directed_character_applied")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="speaker_character_id must reference a selected participant in this chat",
            )

        participants = turn_context.get("participants") or []
        participants_by_alias: dict[str, dict[str, Any]] = {}
        for participant in participants:
            aliases = _normalize_sender_aliases([participant.get("name", "")])
            for alias in aliases:
                participants_by_alias.setdefault(alias, participant)

        resolved_participant: Optional[dict[str, Any]] = None
        requested_speaker_name = (
            body.speaker_character_name.strip()
            if isinstance(body.speaker_character_name, str)
            else ""
        )

        if body.speaker_character_id is not None:
            requested_id = _normalize_character_id(body.speaker_character_id)
            resolved_participant = next(
                (
                    participant
                    for participant in participants
                    if _normalize_character_id(participant.get("id")) == requested_id
                ),
                None,
            )

        if requested_speaker_name:
            requested_aliases = _normalize_sender_aliases([requested_speaker_name])
            matched_by_name = next(
                (
                    participants_by_alias.get(alias)
                    for alias in requested_aliases
                    if alias in participants_by_alias
                ),
                None,
            )
            if matched_by_name is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="speaker_character_name must reference a selected participant in this chat",
                )
            if (
                resolved_participant is not None
                and _normalize_character_id(matched_by_name.get("id"))
                != _normalize_character_id(resolved_participant.get("id"))
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="speaker_character_id and speaker_character_name must reference the same participant",
                )
            resolved_participant = matched_by_name

        if resolved_participant is None:
            active_character_id = _normalize_character_id(turn_context.get("active_character_id"))
            resolved_participant = next(
                (
                    participant
                    for participant in participants
                    if _normalize_character_id(participant.get("id")) == active_character_id
                ),
                None,
            )
        if resolved_participant is None and participants:
            resolved_participant = participants[0]

        resolved_speaker_id = (
            _normalize_character_id(resolved_participant.get("id"))
            if isinstance(resolved_participant, dict)
            else _normalize_character_id(turn_context.get("active_character_id"))
        )
        resolved_speaker_name = (
            str((resolved_participant or {}).get("name") or turn_context.get("active_character_name") or "").strip()
        )
        if not resolved_speaker_name:
            char_card = db.get_character_card_by_id(conversation.get("character_id")) or {}
            resolved_speaker_name = str(char_card.get("name") or "Assistant").strip() or "Assistant"
        resolved_turn_mode = str(turn_context.get("turn_taking_mode") or "single").strip() or "single"
        requested_assistant_message_id = (
            body.assistant_message_id.strip()
            if isinstance(body.assistant_message_id, str)
            else ""
        ) or None
        persist_fingerprint = None
        if not requested_assistant_message_id and getattr(body, "user_message_id", None):
            persist_fingerprint = _build_stream_persist_fingerprint(
                chat_id,
                body.assistant_content,
                body.user_message_id,
                resolved_speaker_id,
            )

        # Validate optional parent message belongs to this conversation
        if getattr(body, "user_message_id", None):
            parent_msg = db.get_message_by_id(body.user_message_id)
            if not parent_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message not found"
                )
            if parent_msg.get("conversation_id") != chat_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parent message must belong to the same conversation"
                )

        existing_persist = None
        if requested_assistant_message_id:
            existing_persist = _load_existing_stream_persist_message_by_id(
                db,
                chat_id,
                requested_assistant_message_id,
                assistant_content=body.assistant_content,
                parent_message_id=body.user_message_id,
                speaker_name=resolved_speaker_name,
            )
        elif persist_fingerprint is not None:
            existing_persist = _find_existing_stream_persist_message(db, chat_id, persist_fingerprint)
        if existing_persist is not None:
            existing_id, existing_extra = existing_persist
            if existing_extra.get("persist_validation_degraded"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=_build_persist_validation_degraded_detail(existing_id),
                )
            return CharacterChatStreamPersistResponse(
                chat_id=chat_id,
                assistant_message_id=existing_id,
                saved=True,
            )

        # Enforce message cap (+1 assistant)
        validation_degraded: Exception | None = None
        try:
            current_count = db.count_messages_for_conversation(chat_id)
            limiter = get_character_rate_limiter()
            await limiter.check_message_limit(chat_id, current_count + 1)
        except HTTPException:
            raise
        except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
            validation_degraded = exc
            logger.debug("Non-fatal: message cap check degraded in persist endpoint: {}", exc)

        metadata_extra = _build_stream_persist_metadata_extra(
            speaker_character_id=resolved_speaker_id,
            speaker_character_name=resolved_speaker_name,
            turn_taking_mode=resolved_turn_mode,
            validation_degraded=validation_degraded is not None,
            persist_fingerprint=persist_fingerprint,
            mood_label=body.mood_label,
            mood_confidence=body.mood_confidence,
            mood_topic=body.mood_topic,
            usage=getattr(body, "usage", None),
        )

        # Persist assistant response via Character_Chat guardrails
        try:
            assistant_msg_id = post_message_to_conversation(
                db=db,
                conversation_id=chat_id,
                character_name=resolved_speaker_name,
                message_content=body.assistant_content,
                is_user_message=False,
                message_id=requested_assistant_message_id,
                parent_message_id=body.user_message_id,
                ranking=body.ranking if getattr(body, "ranking", None) is not None else None,
            )
        except ConflictError:
            if not requested_assistant_message_id:
                raise
            existing_persist = _load_existing_stream_persist_message_by_id(
                db,
                chat_id,
                requested_assistant_message_id,
                assistant_content=body.assistant_content,
                parent_message_id=body.user_message_id,
                speaker_name=resolved_speaker_name,
            )
            if existing_persist is None:
                raise
            existing_id, existing_extra = existing_persist
            _try_store_stream_persist_metadata(
                db,
                assistant_message_id=existing_id,
                tool_calls=getattr(body, "tool_calls", None),
                metadata_extra=metadata_extra,
            )
            if existing_extra.get("persist_validation_degraded"):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=_build_persist_validation_degraded_detail(existing_id),
                )
            return CharacterChatStreamPersistResponse(
                chat_id=chat_id,
                assistant_message_id=existing_id,
                saved=True,
            )
        if not assistant_msg_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist assistant message"
            )
        _try_store_stream_persist_metadata(
            db,
            assistant_message_id=assistant_msg_id,
            tool_calls=getattr(body, "tool_calls", None),
            metadata_extra=metadata_extra,
        )

        # Optionally update chat rating
        if getattr(body, 'chat_rating', None) is not None:
            try:
                conv_for_update = db.get_conversation_by_id(chat_id)
                if conv_for_update:
                    db.update_conversation(
                        chat_id,
                        {"rating": body.chat_rating},
                        conv_for_update.get('version', 1),
                    )
            except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(
                    "Failed to update chat rating for chat_id={} rating={} error={}",
                    chat_id,
                    getattr(body, "chat_rating", None),
                    e,
                    exc_info=True,
                )

        if validation_degraded is not None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_build_persist_validation_degraded_detail(assistant_msg_id),
            )

        return CharacterChatStreamPersistResponse(chat_id=chat_id, assistant_message_id=assistant_msg_id, saved=True)
    except HTTPException:
        raise
    except InputError as e:
        msg = str(e)
        status_code = status.HTTP_400_BAD_REQUEST
        if "exceeds maximum" in msg.lower():
            status_code = status.HTTP_413_CONTENT_TOO_LARGE
        logger.warning(f"Input error persisting streamed assistant message for {chat_id}: {e}")
        raise HTTPException(status_code=status_code, detail=msg) from e
    except ConflictError as e:
        logger.warning(f"Conflict persisting streamed assistant message for {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except CharactersRAGDBError as e:
        logger.error(f"DB error persisting streamed assistant message for {chat_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
    except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error persisting streamed assistant message for {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist assistant message") from e


# ========================================================================
# Greeting List Picker Endpoints (PRD 1 Stage A2)
# ========================================================================

@router.get(
    "/{chat_id}/greetings",
    response_model=GreetingListResponse,
    summary="List character greetings for this chat",
    tags=["Chat Sessions"],
)
async def list_greetings(
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Return all available greetings from the character card, current selection, and staleness info."""
    conversation = db.get_conversation_by_id(chat_id)
    _verify_chat_ownership(conversation, current_user.id, chat_id)

    character_id = conversation.get("character_id")
    character = db.get_character_card_by_id(character_id) if character_id else {}
    if not character:
        character = {}

    greetings_texts = _collect_character_greeting_texts(character)
    greetings = [
        GreetingItem(
            index=i,
            text=t,
            preview=t[:120],
        )
        for i, t in enumerate(greetings_texts)
    ]

    settings_row = db.get_conversation_settings(chat_id)
    settings = (settings_row or {}).get("settings") or {}
    current_selection = _parse_greeting_selection_index(settings.get("greetingSelectionId"))
    staleness = _check_greeting_staleness(settings, character)

    return GreetingListResponse(
        chat_id=chat_id,
        character_id=str(character_id) if character_id is not None else None,
        character_name=character.get("name"),
        greetings=greetings,
        current_selection=current_selection,
        staleness_warning=staleness,
    )


@router.put(
    "/{chat_id}/greetings/select",
    response_model=GreetingSelectResponse,
    summary="Select a greeting for this chat",
    tags=["Chat Sessions"],
)
async def select_greeting(
    chat_id: str = Path(..., description="Chat session ID"),
    body: GreetingSelectRequest = Body(...),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Select a specific greeting by index and update the chat settings."""
    conversation = db.get_conversation_by_id(chat_id)
    _verify_chat_ownership(conversation, current_user.id, chat_id)

    character_id = conversation.get("character_id")
    character = db.get_character_card_by_id(character_id) if character_id else {}
    if not character:
        character = {}

    greetings_texts = _collect_character_greeting_texts(character)
    if body.index < 0 or body.index >= len(greetings_texts):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Greeting index {body.index} out of range (0..{len(greetings_texts) - 1})",
        )

    settings_row = db.get_conversation_settings(chat_id)
    settings = (settings_row or {}).get("settings") or {}
    checksum = _compute_greetings_checksum(character)
    settings["greetingSelectionId"] = f"greeting:{body.index}:selected"
    settings["greetingsChecksum"] = checksum
    if not db.upsert_conversation_settings(chat_id, settings):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist greeting selection",
        )

    return GreetingSelectResponse(
        chat_id=chat_id,
        selected_index=body.index,
        greeting_preview=greetings_texts[body.index][:120],
        checksum_updated=True,
    )


# ========================================================================
# Author Note Info Endpoint (PRD 4 Stage D2)
# ========================================================================

@router.get(
    "/{chat_id}/author-note/info",
    response_model=AuthorNoteInfoResponse,
    summary="Get author note token info",
    tags=["Chat Sessions"],
)
async def get_author_note_info(
    chat_id: str = Path(..., description="Chat session ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Return author note text, token estimates, budget, and toggle states."""
    conversation = db.get_conversation_by_id(chat_id)
    _verify_chat_ownership(conversation, current_user.id, chat_id)

    character_id = conversation.get("character_id")
    character = db.get_character_card_by_id(character_id) if character_id else {}
    if not character:
        character = {}

    settings_row = db.get_conversation_settings(chat_id)
    settings = (settings_row or {}).get("settings") or {}

    text_display = _resolve_author_note_text(settings, character, for_prompt=False)
    text_prompt = _resolve_author_note_text(settings, character, for_prompt=True)

    tokens_display = _estimate_tokens(text_display)
    tokens_prompt = _estimate_tokens(text_prompt)

    enabled = settings.get("authorNoteEnabled") is not False
    gm_only = settings.get("authorNoteGmOnly") is True
    exclude_from_prompt = settings.get("authorNoteExcludeFromPrompt") is True
    scope = _normalize_memory_scope(settings)

    # Determine source
    shared_note = settings.get("authorNote")
    has_settings_note = isinstance(shared_note, str) and shared_note.strip()
    if has_settings_note:
        source = "settings"
    elif text_display:
        source = "character_default"
    else:
        source = "none"

    truncated = tokens_display > _TOKEN_BUDGET_AUTHOR_NOTE

    warnings: list[str] = []
    if truncated:
        warnings.append(
            f"Author note exceeds budget ({tokens_display} tokens > {_TOKEN_BUDGET_AUTHOR_NOTE} budget)"
        )
    if gm_only:
        warnings.append("Note is GM-only: visible in UI but excluded from LLM prompt")

    return AuthorNoteInfoResponse(
        chat_id=chat_id,
        text=text_display,
        text_for_prompt=text_prompt,
        tokens_estimated=tokens_display,
        tokens_for_prompt=tokens_prompt,
        budget=_TOKEN_BUDGET_AUTHOR_NOTE,
        truncated=truncated,
        enabled=enabled,
        gm_only=gm_only,
        exclude_from_prompt=exclude_from_prompt,
        scope=scope,
        source=source,
        warnings=warnings,
    )


# ========================================================================
# Lorebook Diagnostic Export Endpoint (PRD 6 Stage F2)
# ========================================================================

@router.get(
    "/{chat_id}/diagnostics/lorebook",
    response_model=LorebookDiagnosticExportResponse,
    summary="Export lorebook diagnostics across turns",
    tags=["Chat Sessions"],
)
async def export_lorebook_diagnostics(
    chat_id: str = Path(..., description="Chat session ID"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=200, description="Page size"),
    order: Literal["asc", "desc"] = Query(
        "asc",
        description="Sort order for turns by assistant turn number",
    ),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Iterate assistant messages and collect lorebook_diagnostics from message metadata."""
    conversation = db.get_conversation_by_id(chat_id)
    _verify_chat_ownership(conversation, current_user.id, chat_id)

    character_id = conversation.get("character_id")
    settings_row = db.get_conversation_settings(chat_id)

    # Fetch all messages ordered by timestamp
    messages = db.get_messages_for_conversation(chat_id, limit=10000, order_by_timestamp="ASC")
    history_messages = [msg for msg in messages if not msg.get("deleted")]
    turn_context = _resolve_chat_turn_context(
        db=db,
        conversation=conversation,
        settings_row=settings_row,
        history_messages=history_messages,
    )
    participant_aliases = turn_context.get("participant_aliases") or set()
    primary_character_name = str(turn_context.get("primary_character_name") or "")

    # Filter assistant messages that have lorebook diagnostics
    turns_with_diags: list[DiagnosticTurnEntry] = []
    turn_counter = 0
    for msg in messages:
        role = _map_sender_to_role_with_participants(
            msg.get("sender"),
            primary_character_name,
            participant_aliases,
        )
        if role != "assistant":
            continue
        turn_counter += 1

        meta = db.get_message_metadata(msg["id"])
        if not meta:
            continue
        extra = meta.get("extra")
        if not isinstance(extra, dict):
            continue
        diags = extra.get("lorebook_diagnostics")
        if not diags or not isinstance(diags, list):
            continue

        content = msg.get("content") or ""
        turns_with_diags.append(DiagnosticTurnEntry(
            message_id=msg["id"],
            timestamp=msg.get("timestamp"),
            turn_number=turn_counter,
            message_preview=content[:120],
            diagnostics=diags,
        ))

    # Pagination (allow newest-first retrieval without changing default behavior).
    ordered_turns = turns_with_diags if order == "asc" else list(reversed(turns_with_diags))
    total = len(turns_with_diags)
    start = (page - 1) * size
    page_items = ordered_turns[start:start + size]

    return LorebookDiagnosticExportResponse(
        chat_id=chat_id,
        character_id=str(character_id) if character_id is not None else None,
        total_turns_with_diagnostics=total,
        turns=page_items,
        page=page,
        size=size,
    )


#
# End of character_chat_sessions.py
######################################################################################################################
