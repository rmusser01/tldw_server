# tldw_Server_API/app/api/v1/endpoints/persona.py
# Placeholder endpoints for Persona Agent (catalog, session, WebSocket stream)

from __future__ import annotations

import asyncio
import base64
import binascii
from collections import defaultdict, deque
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import re
import time
import uuid
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect, status
from loguru import logger
from starlette.requests import Request as StarletteRequest

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.persona import (
    PersonaCommandDryRunRequest,
    PersonaCommandDryRunResponse,
    PersonaCommandPlannedActionResponse,
    PersonaCommandSafetyGateResponse,
    PersonaConnectionCreate,
    PersonaConnectionDeleteResponse,
    PersonaConnectionResponse,
    PersonaConnectionTestRequest,
    PersonaConnectionTestResponse,
    PersonaConnectionUpdate,
    PersonaDeleteResponse,
    PersonaExemplarCreate,
    PersonaExemplarDeleteResponse,
    PersonaExemplarImportRequest,
    PersonaExemplarReviewRequest,
    PersonaExemplarResponse,
    PersonaExemplarUpdate,
    PersonaInfo,
    PersonaPolicyRulesReplaceRequest,
    PersonaPolicyRulesResponse,
    PersonaBuddyResponse,
    PersonaProfileCreate,
    PersonaProfileResponse,
    PersonaProfileUpdate,
    PersonaSetupAnalyticsResponse,
    PersonaSetupAnalyticsRunSummary,
    PersonaSetupAnalyticsSummary,
    PersonaSetupEventCreate,
    PersonaSetupEventWriteResponse,
    PersonaSetupState,
    PersonaVoiceDefaults,
    PersonaStateHistoryResponse,
    PersonaStateRestoreRequest,
    PersonaStateResponse,
    PersonaStateUpdateRequest,
    PersonaSessionDetail,
    PersonaSessionRequest,
    PersonaSessionResponse,
    PersonaSessionSummary,
    PersonaLiveVoiceAnalyticsSummary,
    PersonaLiveVoiceSessionSummary,
    PersonaLiveVoiceSessionUpdateRequest,
    PersonaScopeRulesReplaceRequest,
    PersonaScopeRulesResponse,
    PersonaVoiceAnalyticsResponse,
    PersonaVoiceAnalyticsSummary,
    PersonaVoiceCommandAnalyticsItem,
    PersonaVoiceFallbackAnalytics,
)
from tldw_Server_API.app.api.v1.schemas.voice_assistant_schemas import (
    VoiceCommandDefinition,
    VoiceCommandInfo,
    VoiceCommandListResponse,
    VoiceCommandToggleRequest,
    VoiceActionType,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import verify_jwt_and_fetch_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.api_key_manager import (
    get_api_key_manager,
    has_scope,
    normalize_scope,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, InvalidTokenError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.feature_flags import (
    is_mcp_hub_policy_enforcement_enabled,
    is_persona_enabled,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.MCP_unified import MCPRequest, get_mcp_server
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.MCP_unified.persona_scope import normalize_persona_scope_payload
from tldw_Server_API.app.core.Metrics import increment_counter
from tldw_Server_API.app.core.Persona.connections import (
    PERSONA_CONNECTION_STATUS_FIELD,
    PersonaConnectionConfigError,
    PersonaConnectionSecretError,
    PersonaConnectionTargetError,
    build_connection_headers,
    build_connection_memory_content,
    build_updated_connection_memory_content,
    connection_content_from_row,
    get_connection_allowed_hosts,
    redact_connection_headers,
    render_nested_templates,
    render_template_value,
    safe_template_context,
    validate_connection_request_target,
)
from tldw_Server_API.app.core.Persona.memory_integration import (
    persist_persona_turn,
    persist_tool_outcome,
    retrieve_top_memories,
)
from tldw_Server_API.app.core.Personalization.companion_activity import (
    normalize_persona_activity_surface,
    record_persona_session_started,
    record_persona_session_summarized,
    record_persona_tool_executed,
)
from tldw_Server_API.app.core.Personalization.companion_context import load_companion_context
from tldw_Server_API.app.core.Persona.exemplar_runtime import (
    append_persona_exemplar_sections,
    resolve_persona_exemplar_runtime_context,
)
from tldw_Server_API.app.core.Persona.exemplar_turn_classifier import classify_persona_turn
from tldw_Server_API.app.core.Persona.exemplar_ingestion import (
    append_exemplar_review_note,
    build_transcript_exemplar_candidates,
)
from tldw_Server_API.app.core.Persona.policy_evaluator import (
    default_allow_rules,
    evaluate_canonical_policy,
    normalize_policy_rules,
)
from tldw_Server_API.app.core.Persona.buddy import (
    build_persona_buddy_summary,
    ensure_persona_buddy_for_profile,
)
from tldw_Server_API.app.core.Persona.session_manager import get_session_manager
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Skills.context_integration import handle_skill_tool_call
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.VoiceAssistant import (
    ActionType as VoiceActionTypeInternal,
    VoiceCommand,
    get_persona_live_voice_summary,
    delete_voice_command as delete_voice_command_db,
    get_voice_analytics_summary_stats,
    get_user_voice_commands,
    get_voice_command as get_voice_command_db,
    get_voice_command_registry,
    get_voice_command_router,
    get_voice_resolution_stats,
    get_voice_top_commands,
    record_persona_live_voice_event,
    save_voice_command,
)

router = APIRouter()

_PERSONA_KNOWN_TOOLS = {
    "ingest_url",
    "rag_search",
    "summarize",
}
_DEFAULT_PERSONA_ID = "research_assistant"
_DEFAULT_PERSONA_NAME = "Research Assistant"
_DEFAULT_PERSONA_DESCRIPTION = "Helps ingest, search, and summarize content"
_DEFAULT_PERSONA_DEFAULT_TOOLS = ["ingest_url", "rag_search", "summarize"]
_DEFAULT_PERSONA_POLICY_RULES: list[dict[str, Any]] = [
    {"rule_kind": "mcp_tool", "rule_name": "media.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "chats.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "notes.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "notes.create", "allowed": True, "require_confirmation": True},
]
_EXPLICIT_SCOPE_RULE_TYPES = {"conversation_id", "character_id", "media_id", "note_id"}
_PERSONA_RUNTIME_MODES = {"session_scoped", "persistent_scoped"}
_PERSONA_WS_REQUIRED_NOTICE_LEVELS = {"info", "warning", "error"}
_PERSONA_WS_ALLOWED_STEP_TYPES = {"mcp_tool", "skill", "rag_query", "final_answer"}
_PERSONA_LIVE_PROCESSING_NOTICE_DELAY_S = 2.0
_PERSONA_CONNECTION_MEMORY_TYPE = "persona_connection"
_PERSONA_CONNECTION_ALLOWED_AUTH_TYPES = {"none", "bearer", "api_key", "basic", "custom_header"}
_PERSONA_CONNECTION_TEST_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
_PERSONA_STATE_FIELD_TO_MEMORY_TYPE = {
    "soul_md": "persona_state_soul",
    "identity_md": "persona_state_identity",
    "heartbeat_md": "persona_state_heartbeat",
}
_PERSONA_STATE_MEMORY_TYPES = set(_PERSONA_STATE_FIELD_TO_MEMORY_TYPE.values())
_PERSONA_STATE_MEMORY_TYPE_TO_FIELD = {
    memory_type: field_name
    for field_name, memory_type in _PERSONA_STATE_FIELD_TO_MEMORY_TYPE.items()
}
_PERSONA_STATE_FIELD_LABELS = {
    "soul_md": "soul",
    "identity_md": "identity",
    "heartbeat_md": "heartbeat",
}


def _bounded_label(value: Any, *, allowed: set[str], fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in allowed:
        return candidate
    return fallback


def _redacted_id_for_logs(raw_id: Any) -> str:
    text = str(raw_id or "").strip()
    if not text:
        return "na"
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest[:12]


def _metric_reason_bucket(reason_code: Any) -> str:
    candidate = str(reason_code or "").strip().upper()
    allowed = {
        "POLICY_DENIED",
        "POLICY_SCOPE_MISSING",
        "POLICY_EXPORT_DISABLED",
        "POLICY_DELETE_DISABLED",
        "POLICY_PERSONA_NO_RULES",
        "POLICY_PERSONA_NO_MATCH",
        "POLICY_PERSONA_EXPLICIT_DENY",
        "POLICY_SESSION_NO_RULES",
        "POLICY_SESSION_NO_MATCH",
        "POLICY_SESSION_EXPLICIT_DENY",
        "POLICY_SKILL_NO_RULES",
        "POLICY_SKILL_NO_MATCH",
        "POLICY_SKILL_EXPLICIT_DENY",
        "POLICY_INVALID_ACTION",
    }
    if candidate in allowed:
        return candidate.lower()
    return "other"


def _increment_persona_metric(metric_name: str, labels: dict[str, str]) -> None:
    safe_labels = {str(k): str(v) for k, v in labels.items()}
    with contextlib.suppress(Exception):
        increment_counter(metric_name, 1, labels=safe_labels)


def _get_persona_max_tool_steps() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_MAX_TOOL_STEPS", 3))
    except Exception:
        value = 3
    return max(1, min(value, 20))


def _get_persona_memory_top_k() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_MEMORY_TOP_K", 3))
    except Exception:
        value = 3
    return max(1, min(value, 10))


def _get_persona_state_hint_max_chars() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_STATE_HINT_MAX_CHARS", 1024))
    except Exception:
        value = 1024
    return max(128, min(value, 8192))


def _get_persona_state_hint_per_doc_max_chars() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_STATE_HINT_PER_DOC_MAX_CHARS", 384))
    except Exception:
        value = 384
    return max(64, min(value, 2048))


def _get_persona_state_doc_max_chars() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_STATE_DOC_MAX_CHARS", 50_000))
    except Exception:
        value = 50_000
    return max(256, min(value, 1_000_000))


def _get_persona_state_history_max_entries() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_STATE_HISTORY_MAX_ENTRIES", 200))
    except Exception:
        value = 200
    return max(1, min(value, 2000))


def _get_persona_allowed_audio_formats() -> set[str]:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        raw = str(_app_settings.get("PERSONA_AUDIO_ALLOWED_FORMATS", "pcm16,wav,mp3,opus"))
    except Exception:
        raw = "pcm16,wav,mp3,opus"
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else {"pcm16"}


def _get_persona_audio_chunk_max_bytes() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_AUDIO_CHUNK_MAX_BYTES", 1_048_576))
    except Exception:
        value = 1_048_576
    return max(1024, min(value, 8_388_608))


def _get_persona_audio_chunks_per_minute() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_AUDIO_CHUNKS_PER_MINUTE", 120))
    except Exception:
        value = 120
    return max(1, min(value, 1200))


def _get_persona_tts_chunk_size_bytes() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_TTS_CHUNK_SIZE_BYTES", 8192))
    except Exception:
        value = 8192
    return max(256, min(value, 65536))


def _get_persona_tts_max_chunks() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_TTS_MAX_CHUNKS", 16))
    except Exception:
        value = 16
    return max(1, min(value, 256))


def _get_persona_tts_max_total_bytes() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_TTS_MAX_TOTAL_BYTES", 131072))
    except Exception:
        value = 131072
    return max(1024, min(value, 2_097_152))


def _get_persona_tts_max_in_flight_chunks() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = int(_app_settings.get("PERSONA_TTS_MAX_IN_FLIGHT_CHUNKS", 4))
    except Exception:
        value = 4
    return max(1, min(value, 32))


def _get_persona_ws_auth_revalidate_interval_s() -> float:
    """
    Periodic auth revalidation interval for long-lived persona WS sessions.

    A value <= 0 disables the background watchdog.
    """
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        value = float(_app_settings.get("PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S", 15.0))
    except Exception:
        value = 15.0
    if value <= 0:
        return 0.0
    return max(0.5, min(value, 300.0))


def _get_persona_rbac_flags() -> tuple[bool, bool]:
    """Return (allow_export, allow_delete) from runtime settings."""
    try:
        from tldw_Server_API.app.core.config import settings as _app_settings

        allow_export = bool(_app_settings.get("PERSONA_RBAC_ALLOW_EXPORT", False))
        allow_delete = bool(_app_settings.get("PERSONA_RBAC_ALLOW_DELETE", False))
    except Exception:
        allow_export = False
        allow_delete = False
    return allow_export, allow_delete


def _get_persona_session_scopes(*, allow_export: bool, allow_delete: bool) -> set[str]:
    scopes = {"read", "write:preview"}
    if allow_export:
        scopes.add("write:export")
    if allow_delete:
        scopes.add("write:delete")
    return scopes


def _normalize_persona_step_type(value: Any, *, tool_name: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"mcp_tool", "skill", "rag_query", "final_answer"}:
        return candidate
    normalized_tool = str(tool_name or "").strip().lower()
    if normalized_tool == "rag_search":
        return "rag_query"
    if normalized_tool == "summarize":
        return "final_answer"
    return "mcp_tool"


async def _run_persona_db_call(func, *args, **kwargs):
    """Offload synchronous persona DB calls from async HTTP handlers."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def _get_persona_profile_or_404(
    db: CharactersRAGDB,
    *,
    persona_id: str,
    user_id: str,
    include_deleted: bool,
) -> dict[str, Any]:
    profile = await _run_persona_db_call(
        db.get_persona_profile,
        persona_id,
        user_id=user_id,
        include_deleted=include_deleted,
    )
    if profile is None:
        raise HTTPException(status_code=404, detail="Persona profile not found")
    return profile


def _voice_command_to_response(command: VoiceCommand) -> VoiceCommandInfo:
    return VoiceCommandInfo(
        id=command.id,
        user_id=command.user_id,
        persona_id=command.persona_id,
        connection_id=command.connection_id,
        connection_status=None,
        connection_name=None,
        name=command.name,
        phrases=command.phrases,
        action_type=VoiceActionType(command.action_type.value),
        action_config=command.action_config,
        priority=command.priority,
        enabled=command.enabled,
        requires_confirmation=command.requires_confirmation,
        description=command.description,
        created_at=command.created_at,
    )


def _resolve_voice_command_connection_status(
    command: VoiceCommand,
    *,
    connections_by_id: dict[str, PersonaConnectionResponse] | None = None,
) -> tuple[str | None, str | None, str | None]:
    connection_id = str(command.connection_id or "").strip() or None
    if not connection_id:
        return None, None, None
    connection = (connections_by_id or {}).get(connection_id)
    if connection is None:
        return connection_id, "missing", None
    return connection_id, "ok", connection.name


def _voice_command_to_response_with_connections(
    command: VoiceCommand,
    *,
    connections_by_id: dict[str, PersonaConnectionResponse] | None = None,
) -> VoiceCommandInfo:
    connection_id, connection_status, connection_name = _resolve_voice_command_connection_status(
        command,
        connections_by_id=connections_by_id,
    )
    response = _voice_command_to_response(command)
    response.connection_id = connection_id
    response.connection_status = connection_status
    response.connection_name = connection_name
    return response


def _normalize_command_path_persona_id(path_persona_id: str, payload_persona_id: str | None) -> str:
    normalized_path_persona_id = str(path_persona_id or "").strip()
    normalized_payload_persona_id = str(payload_persona_id or "").strip()
    if normalized_payload_persona_id and normalized_payload_persona_id != normalized_path_persona_id:
        raise HTTPException(status_code=400, detail="persona_id in payload must match the route persona")
    return normalized_path_persona_id


def _connection_memory_content_from_payload(payload: PersonaConnectionCreate) -> dict[str, Any]:
    auth_type = str(payload.auth_type or "none").strip().lower()
    if auth_type not in _PERSONA_CONNECTION_ALLOWED_AUTH_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported auth_type")
    try:
        return build_connection_memory_content(
            name=payload.name,
            base_url=payload.base_url,
            auth_type=auth_type,
            headers_template=payload.headers_template,
            timeout_ms=int(payload.timeout_ms),
            secret=payload.secret,
        )
    except PersonaConnectionConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _connection_memory_content_from_update(
    existing_content: dict[str, Any],
    payload: PersonaConnectionUpdate,
) -> dict[str, Any]:
    auth_type = str(
        payload.auth_type
        if payload.auth_type is not None
        else existing_content.get("auth_type")
        or "none"
    ).strip().lower()
    if auth_type not in _PERSONA_CONNECTION_ALLOWED_AUTH_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported auth_type")

    try:
        return build_updated_connection_memory_content(
            existing_content,
            name=str(
                payload.name
                if payload.name is not None
                else existing_content.get("name")
                or ""
            ).strip(),
            base_url=str(
                payload.base_url
                if payload.base_url is not None
                else existing_content.get("base_url")
                or ""
            ).strip(),
            auth_type=auth_type,
            headers_template=(
                payload.headers_template
                if payload.headers_template is not None
                else dict(existing_content.get("headers_template") or {})
            ),
            timeout_ms=int(
                payload.timeout_ms
                if payload.timeout_ms is not None
                else existing_content.get("timeout_ms")
                or 15000
            ),
            secret=payload.secret,
            clear_secret=bool(payload.clear_secret),
        )
    except PersonaConnectionConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_persona_connection_test_response_body(response: Any) -> Any:
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    if "json" in content_type and callable(getattr(response, "json", None)):
        try:
            return response.json()
        except Exception:
            return getattr(response, "text", None)
    text = getattr(response, "text", None)
    return text if isinstance(text, str) else None


async def _close_persona_connection_test_response(response: Any) -> None:
    close = getattr(response, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _connection_response_from_row(persona_id: str, row: dict[str, Any]) -> PersonaConnectionResponse:
    content = connection_content_from_row(row)

    return PersonaConnectionResponse(
        id=str(row.get("id") or ""),
        persona_id=persona_id,
        name=str(content.get("name") or ""),
        base_url=str(content.get("base_url") or ""),
        auth_type=str(content.get("auth_type") or "none"),
        headers_template={
            str(k): str(v)
            for k, v in dict(content.get("headers_template") or {}).items()
        },
        timeout_ms=int(content.get("timeout_ms") or 15000),
        allowed_hosts=get_connection_allowed_hosts(content),
        secret_configured=bool(content.get(PERSONA_CONNECTION_STATUS_FIELD, False)),
        key_hint=(str(content.get("key_hint") or "").strip() or None),
        created_at=(str(row.get("created_at") or "").strip() or None),
        last_modified=(str(row.get("last_modified") or "").strip() or None),
    )


async def _list_persona_connection_rows(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
) -> list[dict[str, Any]]:
    rows = await _run_persona_db_call(
        db.list_persona_memory_entries,
        user_id=user_id,
        persona_id=persona_id,
        memory_type=_PERSONA_CONNECTION_MEMORY_TYPE,
        include_archived=False,
        include_deleted=False,
        limit=200,
        offset=0,
    )
    return [row for row in rows if row]


async def _list_persona_connections(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
) -> list[PersonaConnectionResponse]:
    rows = await _list_persona_connection_rows(db, user_id=user_id, persona_id=persona_id)
    return [_connection_response_from_row(persona_id, row) for row in rows]


async def _get_persona_connections_by_id(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
) -> dict[str, PersonaConnectionResponse]:
    responses = await _list_persona_connections(db, user_id=user_id, persona_id=persona_id)
    return {response.id: response for response in responses}


async def _get_persona_connection_row_or_404(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
    connection_id: str,
) -> dict[str, Any]:
    rows = await _list_persona_connection_rows(db, user_id=user_id, persona_id=persona_id)
    for row in rows:
        if str(row.get("id") or "").strip() == str(connection_id or "").strip():
            return row
    raise HTTPException(status_code=404, detail="Persona connection not found")


async def _test_persona_connection(
    connection_id: str,
    connection: dict[str, Any],
    payload: PersonaConnectionTestRequest,
) -> PersonaConnectionTestResponse:
    method = str(payload.method or "GET").strip().upper() or "GET"
    if method not in _PERSONA_CONNECTION_TEST_METHODS:
        raise HTTPException(status_code=422, detail="Unsupported connection test method")

    request_payload = render_nested_templates(
        dict(payload.payload or {}),
        safe_template_context(dict(payload.payload or {})),
    )
    if not isinstance(request_payload, dict):
        request_payload = {}

    base_url = str(connection.get("base_url") or "").strip()
    path = str(payload.path or "").strip()
    rendered_path = (
        render_template_value(path, safe_template_context(request_payload))
        if path
        else ""
    )
    url = urljoin(base_url.rstrip("/") + "/", rendered_path.lstrip("/")) if rendered_path else base_url

    try:
        validate_connection_request_target(url, connection)
    except (PersonaConnectionConfigError, PersonaConnectionTargetError) as exc:
        return PersonaConnectionTestResponse(
            ok=False,
            connection_id=connection_id,
            method=method,
            url=url,
            request_headers={},
            request_payload=request_payload,
            timeout_ms=int(connection.get("timeout_ms") or 15000),
            error=str(exc),
        )
    try:
        headers, secret = build_connection_headers(
            connection,
            payload=request_payload,
            extra_headers=dict(payload.headers or {}),
            auth_header_name=payload.auth_header_name,
        )
    except PersonaConnectionSecretError as exc:
        return PersonaConnectionTestResponse(
            ok=False,
            connection_id=connection_id,
            method=method,
            url=url,
            request_headers={},
            request_payload=request_payload,
            timeout_ms=int(connection.get("timeout_ms") or 15000),
            error=str(exc),
        )
    timeout_ms = int(connection.get("timeout_ms") or 15000)
    timeout_seconds = max(0.1, timeout_ms / 1000.0)
    retry_policy = RetryPolicy()
    retry_policy.attempts = 1
    retry_policy.retry_on_unsafe = False

    request_kwargs: dict[str, Any] = {
        "method": method,
        "url": url,
        "headers": headers or None,
        "timeout": timeout_seconds,
        "retry": retry_policy,
    }
    if method in {"GET", "DELETE", "HEAD"}:
        request_kwargs["params"] = request_payload or None
    else:
        request_kwargs["json"] = request_payload

    started_at = time.perf_counter()
    try:
        response = await afetch(**request_kwargs)
        try:
            body_preview = _parse_persona_connection_test_response_body(response)
        finally:
            await _close_persona_connection_test_response(response)
    except Exception as exc:
        return PersonaConnectionTestResponse(
            ok=False,
            connection_id=connection_id,
            method=method,
            url=url,
            request_headers=redact_connection_headers(headers, secret),
            request_payload=request_payload,
            timeout_ms=timeout_ms,
            latency_ms=max(1, int((time.perf_counter() - started_at) * 1000)),
            error=str(exc),
        )

    status_code = int(getattr(response, "status_code", 500))
    return PersonaConnectionTestResponse(
        ok=200 <= status_code < 400,
        connection_id=connection_id,
        method=method,
        url=url,
        request_headers=redact_connection_headers(headers, secret),
        request_payload=request_payload,
        timeout_ms=timeout_ms,
        status_code=status_code,
        body_preview=body_preview,
        latency_ms=max(1, int((time.perf_counter() - started_at) * 1000)),
        error=None if 200 <= status_code < 400 else str(body_preview),
    )


def _voice_target_name(command: VoiceCommand) -> str | None:
    if command.action_type == VoiceActionTypeInternal.MCP_TOOL:
        return str(command.action_config.get("tool_name") or "").strip() or None
    if command.action_type == VoiceActionTypeInternal.WORKFLOW:
        return (
            str(command.action_config.get("workflow_id") or "").strip()
            or str(command.action_config.get("workflow_name") or "").strip()
            or None
        )
    if command.action_type == VoiceActionTypeInternal.CUSTOM:
        return str(command.action_config.get("action") or "").strip() or None
    if command.action_type == VoiceActionTypeInternal.LLM_CHAT:
        return "persona_planner"
    return None


def _build_payload_preview(command: VoiceCommand, extracted_params: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    raw_slot_map = command.action_config.get("slot_to_param_map") or command.action_config.get("param_map") or {}
    if isinstance(raw_slot_map, dict):
        for param_name, source in raw_slot_map.items():
            if isinstance(source, str):
                slot_name = source.strip().strip("{}")
                if slot_name in extracted_params:
                    preview[str(param_name)] = extracted_params[slot_name]

    if not preview:
        if len(extracted_params) == 1:
            slot_name, slot_value = next(iter(extracted_params.items()))
            target_name = str(_voice_target_name(command) or "")
            if target_name.endswith("search"):
                preview["query"] = slot_value
            elif target_name.endswith("create"):
                preview["content"] = slot_value
            else:
                preview[slot_name] = slot_value
        else:
            preview.update(extracted_params)

    defaults = command.action_config.get("default_payload")
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            preview.setdefault(str(key), value)

    return preview


def _build_dry_run_safety_gate(
    *,
    command: VoiceCommand,
    persona_policy_rules: list[dict[str, Any]],
) -> PersonaCommandSafetyGateResponse:
    if command.connection_id:
        return PersonaCommandSafetyGateResponse(
            classification="calls_external_api",
            requires_confirmation=True,
            reason="persona_default",
        )

    target_name = _voice_target_name(command) or ""
    decision = _evaluate_step_policy(
        step_type="mcp_tool" if command.action_type == VoiceActionTypeInternal.MCP_TOOL else "skill",
        tool_name=target_name,
        args=command.action_config,
        persona_policy_rules=persona_policy_rules,
        session_policy_rules=_default_session_policy_rules(),
        session_scopes={"read", "write:preview", "write:export", "write:delete"},
        allow_export=True,
        allow_delete=True,
    )
    classification = "read_only" if str(decision.get("action") or "read") == "read" else "changes_data"
    reason = "persona_default" if decision.get("reason_code") in {None, ""} else str(decision.get("reason_code")).lower()
    return PersonaCommandSafetyGateResponse(
        classification=classification,
        requires_confirmation=bool(decision.get("requires_confirmation", False)),
        reason=reason,
    )


def _default_session_policy_rules() -> list[dict[str, Any]]:
    # Explicit session rules keep the intersection model deterministic.
    return [
        *default_allow_rules("mcp_tool"),
        *default_allow_rules("skill"),
    ]


def _session_policy_rules_from_preferences(preferences: dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(preferences, dict) and "session_policy_rules" in preferences:
        return normalize_policy_rules(preferences.get("session_policy_rules"))
    return _default_session_policy_rules()


def _normalize_persisted_persona_session_preferences(preferences: Any) -> dict[str, Any]:
    if not isinstance(preferences, dict):
        return {}
    normalized: dict[str, Any] = {}
    if "use_memory_context" in preferences:
        normalized["use_memory_context"] = _coerce_bool(preferences.get("use_memory_context"), default=True)
    if "use_companion_context" in preferences:
        normalized["use_companion_context"] = _coerce_bool(preferences.get("use_companion_context"), default=True)
    if "use_persona_state_context" in preferences:
        normalized["use_persona_state_context"] = _coerce_bool(
            preferences.get("use_persona_state_context"),
            default=True,
        )
    if "memory_top_k" in preferences:
        try:
            normalized_top_k = int(preferences.get("memory_top_k"))
        except (TypeError, ValueError):
            normalized_top_k = _get_persona_memory_top_k()
        normalized["memory_top_k"] = max(1, normalized_top_k)
    if "session_policy_rules" in preferences:
        normalized["session_policy_rules"] = normalize_policy_rules(preferences.get("session_policy_rules"))
    return normalized


def _merge_persisted_persona_session_preferences(*payloads: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged.update(_normalize_persisted_persona_session_preferences(payload))
    return merged


def _default_persisted_persona_session_preferences(profile: dict[str, Any] | None) -> dict[str, Any]:
    return _merge_persisted_persona_session_preferences(
        {
            "use_memory_context": True,
            "use_companion_context": True,
            "use_persona_state_context": _coerce_bool(
                (profile or {}).get("use_persona_state_context_default"),
                default=True,
            ),
            "memory_top_k": _get_persona_memory_top_k(),
        }
    )


def _resolve_step_action_name(
    *,
    step_type: str,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str:
    normalized_step_type = _normalize_persona_step_type(step_type, tool_name=tool_name)
    if normalized_step_type in {"mcp_tool", "rag_query"}:
        resolved_name, _ = _translate_persona_tool_request(tool_name, args or {})
        return str(resolved_name or tool_name or "").strip().lower()
    return str(tool_name or "").strip().lower()


def _load_persona_policy_rules_for_session(
    db: CharactersRAGDB | None,
    *,
    session_id: str,
    user_id: str,
) -> dict[str, Any]:
    default_payload = {
        "persona_id": _DEFAULT_PERSONA_ID,
        "runtime_mode": "session_scoped",
        "scope_snapshot_id": None,
        "policy_rules": normalize_policy_rules(_DEFAULT_PERSONA_POLICY_RULES),
        "persona_state_context_default": True,
        "preferences": {},
        "activity_surface": normalize_persona_activity_surface(None),
        "session_exists": False,
    }
    sid = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    if db is None or not sid or not uid:
        return dict(default_payload)
    try:
        session_row = db.get_persona_session(sid, user_id=uid, include_deleted=False)
        if not session_row:
            return dict(default_payload)
        persona_id = str(session_row.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
        runtime_mode = str(session_row.get("mode") or "session_scoped").strip().lower()
        if runtime_mode not in _PERSONA_RUNTIME_MODES:
            runtime_mode = "session_scoped"
        persona_profile = db.get_persona_profile(persona_id, user_id=uid, include_deleted=False)
        persona_state_context_default = True
        if isinstance(persona_profile, dict):
            persona_state_context_default = _coerce_bool(
                persona_profile.get("use_persona_state_context_default"),
                default=True,
            )
        scope_snapshot_id = _scope_snapshot_id_from_snapshot(session_row.get("scope_snapshot") or {})
        policy_rules = db.list_persona_policy_rules(persona_id=persona_id, user_id=uid, include_deleted=False)
        return {
            "persona_id": persona_id,
            "runtime_mode": runtime_mode,
            "scope_snapshot_id": scope_snapshot_id,
            "policy_rules": normalize_policy_rules(policy_rules),
            "persona_state_context_default": persona_state_context_default,
            "preferences": _normalize_persisted_persona_session_preferences(session_row.get("preferences")),
            "activity_surface": normalize_persona_activity_surface(session_row.get("activity_surface")),
            "session_exists": True,
        }
    except (OSError, RuntimeError, ValueError, CharactersRAGDBError) as exc:
        logger.debug(
            "persona ws policy lookup skipped for session_hash {}: {}",
            _redacted_id_for_logs(sid),
            exc,
        )
        return dict(default_payload)


def _get_session_preferences_with_activity_surface(
    *,
    session_manager: Any,
    session_id: str,
    user_id: str,
    persisted_preferences: Any = None,
    persisted_activity_surface: Any = None,
) -> tuple[dict[str, Any], str]:
    runtime_preferences = dict(
        session_manager.get_preferences(
            session_id=session_id,
            user_id=user_id,
        )
    )
    merged_preferences = dict(runtime_preferences)
    merged_preferences.update(
        _normalize_persisted_persona_session_preferences(persisted_preferences)
    )
    activity_surface = normalize_persona_activity_surface(
        runtime_preferences.get("companion_activity_surface", persisted_activity_surface)
    )
    merged_preferences["companion_activity_surface"] = activity_surface
    if merged_preferences == runtime_preferences:
        return merged_preferences, activity_surface

    updated_preferences = merged_preferences
    with contextlib.suppress(Exception):
        updated_preferences = session_manager.update_preferences(
            session_id=session_id,
            user_id=user_id,
            preferences=merged_preferences,
        )
    return dict(updated_preferences), activity_surface


def _persist_persona_session_preferences(
    db: CharactersRAGDB | None,
    *,
    session_id: str,
    user_id: str,
    base_preferences: Any = None,
    patch_preferences: Any = None,
) -> dict[str, Any]:
    merged_preferences = _merge_persisted_persona_session_preferences(
        base_preferences,
        patch_preferences,
    )
    sid = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    if db is None or not sid or not uid:
        return merged_preferences

    current_preferences = _normalize_persisted_persona_session_preferences(base_preferences)
    if current_preferences == merged_preferences:
        return merged_preferences

    try:
        _ = db.update_persona_session(
            session_id=session_id,
            user_id=user_id,
            update_data={"preferences_json": merged_preferences},
        )
    except (OSError, RuntimeError, ValueError, CharactersRAGDBError) as exc:
        logger.debug(
            "persona session preference persistence skipped for session_hash {}: {}",
            _redacted_id_for_logs(sid),
            exc,
        )
    return merged_preferences


def _skill_policy_rules_for_step(
    *,
    step_type: str,
) -> list[dict[str, Any]]:
    normalized_step_type = _normalize_persona_step_type(step_type, tool_name="")
    if normalized_step_type == "skill":
        return default_allow_rules("skill")
    return default_allow_rules("mcp_tool")


def _evaluate_step_policy(
    *,
    step_type: str,
    tool_name: str,
    args: dict[str, Any] | None,
    persona_policy_rules: list[dict[str, Any]],
    session_policy_rules: list[dict[str, Any]],
    session_scopes: set[str],
    allow_export: bool,
    allow_delete: bool,
) -> dict[str, Any]:
    normalized_step_type = _normalize_persona_step_type(step_type, tool_name=tool_name)
    action_name = _resolve_step_action_name(step_type=normalized_step_type, tool_name=tool_name, args=args)
    skill_policy_rules = _skill_policy_rules_for_step(step_type=normalized_step_type)
    return evaluate_canonical_policy(
        step_type=normalized_step_type,
        action_name=action_name,
        persona_policy_rules=persona_policy_rules,
        session_policy_rules=session_policy_rules,
        skill_policy_rules=skill_policy_rules,
        session_scopes=session_scopes,
        allow_export=allow_export,
        allow_delete=allow_delete,
    )


def _build_tool_result(
    *,
    ok: bool,
    output: Any = None,
    error: str | None = None,
    reason_code: str | None = None,
    policy: dict[str, Any] | None = None,
    approval: dict[str, Any] | None = None,
    external_access: dict[str, Any] | None = None,
    path_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "output": output,
        # Canonical wire key is `output`; keep `result` as a temporary compatibility alias.
        "result": output,
    }
    if error is not None:
        payload["error"] = str(error)
    if reason_code:
        payload["reason_code"] = str(reason_code)
    if isinstance(policy, dict):
        payload["policy"] = dict(policy)
    if isinstance(approval, dict):
        payload["approval"] = dict(approval)
    if isinstance(external_access, dict):
        payload["external_access"] = dict(external_access)
    if isinstance(path_scope, dict):
        payload["path_scope"] = dict(path_scope)
    return payload


def _summarize_retention_value(value: Any) -> tuple[str, int | None, int | None, str]:
    value_type = type(value).__name__
    if value is None:
        return value_type, 0, None, "na"
    if isinstance(value, str):
        digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, len(value), None, digest
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        digest = hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:16]
        return value_type, len(raw), None, digest
    if isinstance(value, dict):
        signature = f"dict:{len(value)}"
        digest = hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, None, len(value), digest
    if isinstance(value, (list, tuple, set)):
        signature = f"{value_type}:{len(value)}"
        digest = hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, None, len(value), digest
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return value_type, len(text), None, digest


def _summarize_tool_result_for_retention(result_payload: dict[str, Any] | None) -> str:
    payload = dict(result_payload or {})
    output_value = payload.get("output")
    if "output" not in payload:
        output_value = payload.get("result")
    output_type, output_char_count, output_item_count, output_digest = _summarize_retention_value(output_value)
    error_text = str(payload.get("error") or "").strip()
    error_digest = (
        hashlib.sha1(error_text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        if error_text
        else "na"
    )
    summary = {
        "ok": bool(payload.get("ok", False)),
        "reason_code": str(payload.get("reason_code") or ""),
        "output_type": output_type,
        "output_char_count": output_char_count,
        "output_item_count": output_item_count,
        "output_digest": output_digest,
        "error_present": bool(error_text),
        "error_char_count": len(error_text),
        "error_digest": error_digest,
    }
    return json.dumps(summary, ensure_ascii=True, sort_keys=True)


def _max_base64_length_for_decoded_bytes(max_decoded_bytes: int) -> int:
    safe_limit = max(0, int(max_decoded_bytes))
    return ((safe_limit + 2) // 3) * 4


def _project_base64_decoded_size(encoded_payload: str) -> int:
    payload = str(encoded_payload or "")
    payload_len = len(payload)
    if payload_len <= 0:
        return 0
    if payload_len % 4 != 0:
        return ((payload_len + 3) // 4) * 3

    padding = 0
    if payload.endswith("=="):
        padding = 2
    elif payload.endswith("="):
        padding = 1
    return (payload_len // 4) * 3 - padding


def _decode_audio_chunk(bytes_base64: str, *, max_decoded_bytes: int) -> bytes:
    encoded = str(bytes_base64 or "").strip()
    if not encoded:
        raise ValueError("bytes_base64 is required")
    max_encoded_bytes = _max_base64_length_for_decoded_bytes(max_decoded_bytes)
    if len(encoded) > max_encoded_bytes:
        raise ValueError(
            f"Audio chunk encoded payload exceeds max bytes ({len(encoded)} > {max_encoded_bytes})"
        )
    projected_decoded = _project_base64_decoded_size(encoded)
    if projected_decoded > max_decoded_bytes:
        raise ValueError(
            "Audio chunk projected decoded size exceeds max bytes "
            f"({projected_decoded} > {max_decoded_bytes})"
        )
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 payload for audio chunk") from exc
    if len(decoded) > max_decoded_bytes:
        raise ValueError(
            f"Audio chunk exceeds max bytes ({len(decoded)} > {max_decoded_bytes})"
        )
    return decoded


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "enabled"}:
            return True
        if normalized in {"false", "0", "no", "off", "disabled"}:
            return False
    return default


def _normalize_ws_identifier(raw_value: Any, *, fallback: str, max_len: int = 128) -> str:
    candidate = str(raw_value or "").strip()
    if not candidate:
        return fallback
    safe = "".join(ch for ch in candidate if ch.isalnum() or ch in {"-", "_", ":", "."})
    if not safe:
        return fallback
    return safe[:max_len]


def _memory_mode_allows_personalization_retrieval(runtime_mode: str, *, session_exists: bool) -> bool:
    normalized = str(runtime_mode or "").strip().lower()
    if normalized == "persistent_scoped":
        return True
    if normalized == "session_scoped" and session_exists:
        return False
    # Backward compatibility for pre-session-scaffold clients that don't create persisted sessions first.
    return True


def _require_current_user_id(current_user: User) -> str:
    user_id = str(getattr(current_user, "id", "") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_http_exception(exc: Exception, *, action: str) -> HTTPException:
    if isinstance(exc, InputError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, ConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, CharactersRAGDBError):
        logger.error("Persona DB error during {}: {}", action, exc)
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to {action}")
    logger.error("Unexpected persona error during {}: {}", action, exc)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to {action}")


def _scope_audit_from_snapshot(scope_snapshot: Any) -> dict[str, object]:
    if not isinstance(scope_snapshot, dict):
        return {}
    audit = scope_snapshot.get("audit")
    if isinstance(audit, dict):
        return {str(k): v for k, v in audit.items()}
    return {}


def _scope_snapshot_id_from_snapshot(scope_snapshot: Any) -> str | None:
    if not isinstance(scope_snapshot, dict):
        return None
    candidate = str(scope_snapshot.get("scope_snapshot_id") or "").strip()
    if candidate:
        return candidate
    audit = _scope_audit_from_snapshot(scope_snapshot)
    fallback = str(audit.get("scope_snapshot_id") or "").strip()
    return fallback or None


def _open_persona_ws_db(user_id: str) -> CharactersRAGDB | None:
    try:
        uid = int(str(user_id).strip())
    except (TypeError, ValueError):
        return None
    try:
        db_path = DatabasePaths.get_chacha_db_path(uid)
        return CharactersRAGDB(db_path=str(db_path), client_id=f"persona_ws_{uid}")
    except (OSError, RuntimeError, ValueError, CharactersRAGDBError) as exc:
        logger.debug("persona ws scope DB init skipped: {}", exc)
        return None


def _load_persona_scope_metadata_for_session(
    db: CharactersRAGDB | None,
    *,
    session_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    if db is None:
        return None
    sid = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid or not uid:
        return None
    try:
        row = db.get_persona_session(sid, user_id=uid, include_deleted=False)
    except (OSError, RuntimeError, ValueError, CharactersRAGDBError) as exc:
        logger.debug(
            "persona ws scope lookup skipped for session_hash {}: {}",
            _redacted_id_for_logs(sid),
            exc,
        )
        return None
    if not row:
        return None
    payload = normalize_persona_scope_payload(row.get("scope_snapshot") or {})
    if not payload:
        return None
    payload["persona_id"] = str(row.get("persona_id") or "")
    return payload


def _translate_persona_tool_request(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized = str(name or "").strip().lower()
    if normalized == "rag_search":
        query = str(arguments.get("query") or "").strip()
        try:
            limit = int(arguments.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = int(arguments.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        translated: dict[str, Any] = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "offset": max(0, offset),
            "sources": ["media", "chats", "notes"],
        }
        if isinstance(arguments.get("sources"), list):
            candidate_sources = [str(s).strip() for s in arguments["sources"] if str(s).strip()]
            if candidate_sources:
                translated["sources"] = candidate_sources
        if isinstance(arguments.get("filters"), dict):
            translated["filters"] = dict(arguments["filters"])
        return "knowledge.search", translated
    return name, arguments


def _build_scope_snapshot(rules: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, object]]:
    scope_snapshot_id = uuid.uuid4().hex
    materialized_at = _utc_now_iso()
    include_counts: dict[str, int] = {}
    explicit_ids: dict[str, list[str]] = {}
    selector_values: dict[str, list[str]] = {}
    include_rule_count = 0
    exclude_rule_count = 0
    explicit_id_rule_count = 0
    selector_rule_count = 0

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_type = str(rule.get("rule_type") or "").strip().lower()
        rule_value = str(rule.get("rule_value") or "").strip()
        include = bool(rule.get("include", True))
        if not rule_type:
            continue
        include_counts[rule_type] = include_counts.get(rule_type, 0) + 1
        if not include:
            exclude_rule_count += 1
            continue
        include_rule_count += 1
        if rule_type in _EXPLICIT_SCOPE_RULE_TYPES:
            if rule_value:
                explicit_ids.setdefault(rule_type, []).append(rule_value)
                explicit_id_rule_count += 1
        elif rule_value:
            selector_values.setdefault(rule_type, []).append(rule_value)
            selector_rule_count += 1

    for values in explicit_ids.values():
        values[:] = sorted(set(values))
    for values in selector_values.values():
        values[:] = sorted(set(values))

    audit: dict[str, object] = {
        "scope_snapshot_id": scope_snapshot_id,
        "materialized_at": materialized_at,
        "source_rule_count": len(rules),
        "include_rule_count": include_rule_count,
        "exclude_rule_count": exclude_rule_count,
        "source_rule_type_counts": include_counts,
        "explicit_id_rule_count": explicit_id_rule_count,
        "selector_rule_count": selector_rule_count,
        "selector_rule_types": sorted(selector_values.keys()),
    }
    snapshot = {
        "scope_snapshot_id": scope_snapshot_id,
        "materialized_at": materialized_at,
        "materialized_scope": {
            "explicit_ids": explicit_ids,
            "selectors": selector_values,
        },
        "audit": audit,
    }
    return snapshot, audit


def _persona_profile_to_response(
    profile: dict[str, Any],
    *,
    buddy_row: dict[str, Any] | None = None,
) -> PersonaProfileResponse:
    raw_voice_defaults = profile.get("voice_defaults")
    raw_setup = profile.get("setup")
    try:
        voice_defaults = (
            PersonaVoiceDefaults.model_validate(raw_voice_defaults)
            if isinstance(raw_voice_defaults, dict)
            else PersonaVoiceDefaults()
        )
    except Exception:
        voice_defaults = PersonaVoiceDefaults()
    try:
        setup = (
            PersonaSetupState.model_validate(raw_setup)
            if isinstance(raw_setup, dict)
            else PersonaSetupState()
        )
    except Exception:
        setup = PersonaSetupState()
    return PersonaProfileResponse(
        id=str(profile.get("id") or ""),
        name=str(profile.get("name") or ""),
        character_card_id=profile.get("character_card_id"),
        origin_character_id=profile.get("origin_character_id"),
        origin_character_name=profile.get("origin_character_name"),
        origin_character_snapshot_at=profile.get("origin_character_snapshot_at"),
        mode=str(profile.get("mode") or "session_scoped"),
        system_prompt=profile.get("system_prompt"),
        is_active=bool(profile.get("is_active", True)),
        use_persona_state_context_default=_coerce_bool(
            profile.get("use_persona_state_context_default"),
            default=True,
        ),
        voice_defaults=voice_defaults,
        setup=setup,
        created_at=str(profile.get("created_at") or _utc_now_iso()),
        last_modified=str(profile.get("last_modified") or _utc_now_iso()),
        version=int(profile.get("version") or 1),
        buddy_summary=_persona_buddy_summary_from_profile(profile, buddy_row=buddy_row),
    )


def _persona_buddy_to_response(buddy: dict[str, Any]) -> PersonaBuddyResponse:
    return PersonaBuddyResponse(
        persona_id=str(buddy.get("persona_id") or ""),
        resolved_profile=buddy.get("resolved_profile"),
        created_at=str(buddy.get("created_at") or _utc_now_iso()),
        last_modified=str(buddy.get("last_modified") or buddy.get("created_at") or _utc_now_iso()),
    )


class PersonaBuddyRollbackError(RuntimeError):
    """Raised when profile rollback fails after persona buddy validation errors."""


def _rollback_created_persona_profile_after_buddy_failure(
    db: CharactersRAGDB,
    *,
    persona_id: str,
    user_id: str,
    expected_version: int,
) -> None:
    """Hide a newly created profile when buddy sync fails after commit."""
    persona_hash = _redacted_id_for_logs(persona_id)
    try:
        rolled_back = db.soft_delete_persona_profile(
            persona_id=persona_id,
            user_id=user_id,
            expected_version=expected_version,
        )
        if rolled_back:
            logger.warning(
                "Rolled back newly created persona profile {} after buddy sync failure.",
                persona_hash,
            )
            return
        raise PersonaBuddyRollbackError(
            "Persona buddy validation failed and rollback did not complete."
        )
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        logger.error(
            "Error rolling back newly created persona profile {} after buddy sync failure: {}",
            persona_hash,
            exc,
        )
        raise PersonaBuddyRollbackError(
            "Persona buddy validation failed and rollback did not complete."
        ) from exc


def _rollback_updated_persona_profile_after_buddy_failure(
    db: CharactersRAGDB,
    *,
    persona_id: str,
    user_id: str,
    update_data: dict[str, Any],
    previous_profile: dict[str, Any],
    expected_version: int,
) -> None:
    """Restore the previous visible profile state when buddy sync fails after update."""
    persona_hash = _redacted_id_for_logs(persona_id)
    rollback_data = {field: previous_profile.get(field) for field in update_data.keys()}
    if not rollback_data:
        return
    try:
        rolled_back = db.update_persona_profile(
            persona_id=persona_id,
            user_id=user_id,
            update_data=rollback_data,
            expected_version=expected_version,
        )
        if rolled_back:
            logger.warning(
                "Rolled back persona profile {} after buddy sync failure.",
                persona_hash,
            )
            return
        raise PersonaBuddyRollbackError(
            "Persona buddy validation failed and rollback did not complete."
        )
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        logger.error(
            "Error rolling back persona profile {} after buddy sync failure: {}",
            persona_hash,
            exc,
        )
        raise PersonaBuddyRollbackError(
            "Persona buddy validation failed and rollback did not complete."
        ) from exc


def _ensure_persona_buddy_after_profile_mutation(
    *,
    db: CharactersRAGDB,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Keep buddy state aligned after a committed profile mutation."""
    return ensure_persona_buddy_for_profile(db, profile)


def _persona_exemplar_to_response(exemplar: dict[str, Any]) -> PersonaExemplarResponse:
    return PersonaExemplarResponse(
        id=str(exemplar.get("id") or ""),
        persona_id=str(exemplar.get("persona_id") or ""),
        user_id=str(exemplar.get("user_id") or ""),
        kind=str(exemplar.get("kind") or "style"),
        content=str(exemplar.get("content") or ""),
        tone=None if exemplar.get("tone") is None else str(exemplar.get("tone")),
        scenario_tags=[str(item) for item in list(exemplar.get("scenario_tags") or [])],
        capability_tags=[str(item) for item in list(exemplar.get("capability_tags") or [])],
        priority=int(exemplar.get("priority") or 0),
        enabled=bool(exemplar.get("enabled", True)),
        source_type=str(exemplar.get("source_type") or "manual"),
        source_ref=None if exemplar.get("source_ref") is None else str(exemplar.get("source_ref")),
        notes=None if exemplar.get("notes") is None else str(exemplar.get("notes")),
        created_at=str(exemplar.get("created_at") or _utc_now_iso()),
        last_modified=str(exemplar.get("last_modified") or exemplar.get("created_at") or _utc_now_iso()),
        deleted=bool(exemplar.get("deleted", False)),
        version=int(exemplar.get("version") or 1),
    )


def _persona_state_response_from_rows(
    *,
    persona_id: str,
    rows: list[dict[str, Any]] | None,
) -> PersonaStateResponse:
    payload: dict[str, Any] = {
        "persona_id": str(persona_id or ""),
        "soul_md": None,
        "identity_md": None,
        "heartbeat_md": None,
        "last_modified": None,
    }
    latest_last_modified: str | None = None
    for row in (rows or []):
        memory_type = str(row.get("memory_type") or "").strip()
        if memory_type not in _PERSONA_STATE_MEMORY_TYPES:
            continue
        field_name = _PERSONA_STATE_MEMORY_TYPE_TO_FIELD.get(memory_type)
        if not field_name or payload.get(field_name) is not None:
            continue
        payload[field_name] = str(row.get("content") or "")
        row_last_modified = str(row.get("last_modified") or "").strip() or None
        if row_last_modified and (latest_last_modified is None or row_last_modified > latest_last_modified):
            latest_last_modified = row_last_modified
    payload["last_modified"] = latest_last_modified
    return PersonaStateResponse(**payload)


def _persona_state_history_response_from_rows(
    *,
    persona_id: str,
    rows: list[dict[str, Any]] | None,
) -> PersonaStateHistoryResponse:
    entries: list[dict[str, Any]] = []
    for row in (rows or []):
        memory_type = str(row.get("memory_type") or "").strip()
        field_name = _PERSONA_STATE_MEMORY_TYPE_TO_FIELD.get(memory_type)
        if not field_name:
            continue
        entry_id = str(row.get("id") or "").strip()
        if not entry_id:
            continue
        is_archived = _coerce_bool(row.get("archived"), default=False)
        entries.append(
            {
                "entry_id": entry_id,
                "field": field_name,
                "content": str(row.get("content") or ""),
                "is_active": not is_archived,
                "created_at": str(row.get("created_at") or "").strip() or None,
                "last_modified": str(row.get("last_modified") or "").strip() or None,
                "version": int(row.get("version") or 1),
            }
        )
    return PersonaStateHistoryResponse(persona_id=str(persona_id or ""), entries=entries)


def _get_persona_state_rows(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
) -> list[dict[str, Any]]:
    return db.list_persona_memory_entries(
        user_id=user_id,
        persona_id=persona_id,
        include_archived=False,
        include_deleted=False,
        limit=500,
        offset=0,
    )


def _normalize_persona_state_hint_text(text: Any, *, max_chars: int) -> str:
    value = " ".join(str(text or "").strip().split())
    safe_limit = max(1, int(max_chars))
    if len(value) <= safe_limit:
        return value
    suffix = "... [truncated]"
    if safe_limit <= len(suffix):
        return value[:safe_limit]
    return f"{value[: safe_limit - len(suffix)]}{suffix}"


def _load_persona_state_hints_for_runtime(
    db: CharactersRAGDB | None,
    *,
    user_id: str,
    persona_id: str,
    runtime_mode: str,
) -> dict[str, str]:
    if db is None:
        return {}
    if str(runtime_mode or "").strip().lower() != "persistent_scoped":
        return {}

    per_doc_limit = _get_persona_state_hint_per_doc_max_chars()
    total_limit = _get_persona_state_hint_max_chars()
    try:
        rows = _get_persona_state_rows(
            db,
            user_id=user_id,
            persona_id=persona_id,
        )
    except Exception as exc:
        logger.debug(
            "persona state hint lookup skipped for persona_hash {}: {}",
            _redacted_id_for_logs(persona_id),
            exc,
        )
        return {}

    out: dict[str, str] = {}
    total_used = 0
    for row in rows:
        memory_type = str(row.get("memory_type") or "").strip()
        if memory_type not in _PERSONA_STATE_MEMORY_TYPES:
            continue
        field_name = _PERSONA_STATE_MEMORY_TYPE_TO_FIELD.get(memory_type)
        label = _PERSONA_STATE_FIELD_LABELS.get(str(field_name or ""), "")
        if not field_name or not label or label in out:
            continue
        normalized = _normalize_persona_state_hint_text(
            row.get("content"),
            max_chars=per_doc_limit,
        )
        if not normalized:
            continue
        projected = total_used + len(normalized)
        if projected > total_limit:
            remaining = total_limit - total_used
            if remaining <= 0:
                break
            normalized = _normalize_persona_state_hint_text(normalized, max_chars=remaining)
            if not normalized:
                break
            projected = total_used + len(normalized)
        out[label] = normalized
        total_used = projected
    return out


def _persona_identity_query_requested(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    signals = (
        "who are you",
        "your identity",
        "your personality",
        "about yourself",
        "what are your values",
        "how should you behave",
    )
    return any(signal in lowered for signal in signals)


def _build_persona_state_identity_answer(state_hints: dict[str, str]) -> str:
    if not state_hints:
        return "I don't have persistent persona state configured yet."
    fragments: list[str] = []
    if state_hints.get("identity"):
        fragments.append(f"Identity: {state_hints['identity']}")
    if state_hints.get("soul"):
        fragments.append(f"Soul: {state_hints['soul']}")
    if state_hints.get("heartbeat"):
        fragments.append(f"Heartbeat: {state_hints['heartbeat']}")
    answer = " ".join(fragment.strip() for fragment in fragments if fragment.strip())
    return _normalize_persona_state_hint_text(answer, max_chars=_get_persona_state_hint_max_chars())


def _build_persona_state_hint_lines(state_hints: dict[str, str]) -> list[str]:
    if not isinstance(state_hints, dict):
        return []
    lines: list[str] = []
    for field_name in ("identity", "soul", "heartbeat"):
        value = str(state_hints.get(field_name) or "").strip()
        if not value:
            continue
        label = field_name.capitalize()
        lines.append(f"- {label}: {value}")
    return lines


def _join_applied_context_labels(labels: list[str]) -> str:
    compact = [str(label or "").strip() for label in labels if str(label or "").strip()]
    if not compact:
        return ""
    if len(compact) == 1:
        return compact[0]
    if len(compact) == 2:
        return f"{compact[0]} and {compact[1]}"
    return f"{', '.join(compact[:-1])}, and {compact[-1]}"


def _replace_persona_state_docs(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
    updates: dict[str, str | None],
) -> None:
    now = _utc_now_iso()
    for field_name, value in updates.items():
        memory_type = _PERSONA_STATE_FIELD_TO_MEMORY_TYPE.get(field_name)
        if not memory_type:
            continue
        existing_rows = db.list_persona_memory_entries(
            user_id=user_id,
            persona_id=persona_id,
            memory_type=memory_type,
            include_archived=False,
            include_deleted=False,
            limit=200,
            offset=0,
        )
        for row in existing_rows:
            entry_id = str(row.get("id") or "").strip()
            if not entry_id:
                continue
            with contextlib.suppress(Exception):
                db.set_persona_memory_archived(
                    entry_id=entry_id,
                    user_id=user_id,
                    persona_id=persona_id,
                    archived=True,
                )
        if value is None:
            continue
        _ = db.add_persona_memory_entry(
            {
                "persona_id": persona_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "content": str(value),
                "salience": 0.0,
                "created_at": now,
                "last_modified": now,
            }
        )


def _increment_persona_state_metric(*, action: str, result: str) -> None:
    action_value = _bounded_label(action, allowed={"read", "write", "history", "restore"}, fallback="read")
    result_value = _bounded_label(
        result,
        allowed={
            "success",
            "error",
            "not_found",
            "entry_not_found",
            "rejected_empty",
            "rejected_too_large",
        },
        fallback="error",
    )
    _increment_persona_metric(
        "persona_state_docs_total",
        {"action": action_value, "result": result_value},
    )


def _persona_info_from_profile(
    profile: dict[str, Any],
    *,
    policy_rules: list[dict[str, Any]] | None = None,
    buddy_row: dict[str, Any] | None = None,
) -> PersonaInfo:
    mcp_tools = sorted(
        {
            str(rule.get("rule_name") or "").strip()
            for rule in (policy_rules or [])
            if str(rule.get("rule_kind") or "").strip() == "mcp_tool"
            and bool(rule.get("allowed", False))
            and str(rule.get("rule_name") or "").strip()
        }
    )
    skill_tools = sorted(
        {
            str(rule.get("rule_name") or "").strip()
            for rule in (policy_rules or [])
            if str(rule.get("rule_kind") or "").strip() == "skill"
            and bool(rule.get("allowed", False))
            and str(rule.get("rule_name") or "").strip()
        }
    )
    default_tools = mcp_tools or list(_DEFAULT_PERSONA_DEFAULT_TOOLS)
    capabilities = ["agentic", "mcp", "skills"]
    if mcp_tools:
        capabilities.append("mcp_tools_configured")
    if skill_tools:
        capabilities.append("skills_configured")
    description = str(profile.get("system_prompt") or "").strip()
    if not description:
        description = _DEFAULT_PERSONA_DESCRIPTION
    return PersonaInfo(
        id=str(profile.get("id") or _DEFAULT_PERSONA_ID),
        name=str(profile.get("name") or _DEFAULT_PERSONA_NAME),
        description=description[:300] if description else None,
        voice="default",
        avatar_url=None,
        capabilities=capabilities,
        default_tools=default_tools,
        buddy_summary=_persona_buddy_summary_from_profile(profile, buddy_row=buddy_row),
    )


def _persona_buddy_summary_from_profile(
    profile: dict[str, Any],
    *,
    buddy_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    description = str(profile.get("system_prompt") or "").strip()
    role_summary = description[:300] if description else _DEFAULT_PERSONA_DESCRIPTION
    return build_persona_buddy_summary(
        persona_name=str(profile.get("name") or _DEFAULT_PERSONA_NAME),
        role_summary=role_summary,
        buddy_row=buddy_row,
    )


def _load_persona_buddy_row_for_projection(
    db: CharactersRAGDB,
    *,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    persona_id = str(profile.get("id") or "").strip()
    user_id = str(profile.get("user_id") or "").strip()
    if not persona_id or not user_id:
        return None

    try:
        return db.get_persona_buddy(
            persona_id=persona_id,
            user_id=user_id,
            include_deleted_personas=bool(profile.get("deleted", False)),
        )
    except Exception as exc:
        logger.warning(
            "Failed to load persona buddy projection for persona_hash {}: {}",
            _redacted_id_for_logs(persona_id),
            exc,
        )
        return None


def _load_persona_buddy_rows_for_projection(
    db: CharactersRAGDB,
    *,
    profiles: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    normalized_profiles = [profile for profile in profiles if isinstance(profile, dict)]
    if not normalized_profiles:
        return {}

    sorted_profiles = sorted(
        normalized_profiles,
        key=lambda profile: (
            str(profile.get("name") or "").strip().lower(),
            str(profile.get("id") or "").strip(),
        ),
    )
    persona_ids = [
        str(profile.get("id") or "").strip()
        for profile in sorted_profiles
        if str(profile.get("id") or "").strip()
    ]
    user_ids = {
        str(profile.get("user_id") or "").strip()
        for profile in normalized_profiles
        if str(profile.get("user_id") or "").strip()
    }
    if not persona_ids or len(user_ids) != 1:
        return {}

    user_id = next(iter(user_ids))
    include_deleted_personas = any(bool(profile.get("deleted", False)) for profile in normalized_profiles)
    try:
        return db.list_persona_buddies(
            user_id=user_id,
            persona_ids=persona_ids,
            include_deleted_personas=include_deleted_personas,
        )
    except Exception as exc:
        logger.warning(
            "Failed bulk persona buddy projection load for user_hash {}: {}",
            _redacted_id_for_logs(user_id),
            exc,
        )
        return {}


def _ensure_default_persona_profile(db: CharactersRAGDB, *, user_id: str) -> dict[str, Any]:
    profile = db.get_persona_profile(_DEFAULT_PERSONA_ID, user_id=user_id, include_deleted=False)
    if profile is None:
        try:
            _ = db.create_persona_profile(
                {
                    "id": _DEFAULT_PERSONA_ID,
                    "user_id": user_id,
                    "name": _DEFAULT_PERSONA_NAME,
                    "mode": "session_scoped",
                    "system_prompt": _DEFAULT_PERSONA_DESCRIPTION,
                    "is_active": True,
                }
            )
        except ConflictError:
            # A raced creator likely inserted it first; re-fetch below.
            pass
        profile = db.get_persona_profile(_DEFAULT_PERSONA_ID, user_id=user_id, include_deleted=False)
    if profile is None:
        profiles = db.list_persona_profiles(user_id=user_id, active_only=True, limit=1)
        if not profiles:
            raise ConflictError(
                "Unable to resolve a default persona profile for user.",
                entity="persona_profiles",
                entity_id=_DEFAULT_PERSONA_ID,
            )
        profile = profiles[0]

    if str(profile.get("id") or "") == _DEFAULT_PERSONA_ID:
        try:
            existing = db.list_persona_policy_rules(persona_id=_DEFAULT_PERSONA_ID, user_id=user_id)
            if not existing:
                _ = db.replace_persona_policy_rules(
                    persona_id=_DEFAULT_PERSONA_ID,
                    user_id=user_id,
                    rules=_DEFAULT_PERSONA_POLICY_RULES,
                )
        except CharactersRAGDBError as exc:
            logger.warning("Failed to ensure default persona policy rules: {}", exc)
    return profile


def _persona_catalog_items() -> list[PersonaInfo]:
    return [
        PersonaInfo(
            id=_DEFAULT_PERSONA_ID,
            name=_DEFAULT_PERSONA_NAME,
            description=_DEFAULT_PERSONA_DESCRIPTION,
            voice="default",
            avatar_url=None,
            capabilities=["ingest", "rag_search", "summarize"],
            default_tools=list(_DEFAULT_PERSONA_DEFAULT_TOOLS),
        )
    ]


def _persona_session_summary_from_db(
    row: dict[str, Any],
    *,
    manager_row: dict[str, Any] | None = None,
) -> PersonaSessionSummary:
    scope_snapshot = row.get("scope_snapshot") or {}
    preferences = dict(row.get("preferences") or {})
    runtime_preferences = (manager_row or {}).get("preferences")
    if isinstance(runtime_preferences, dict):
        preferences.update(runtime_preferences)
    return PersonaSessionSummary(
        session_id=str(row.get("id") or ""),
        persona_id=str(row.get("persona_id") or ""),
        created_at=str(row.get("created_at") or _utc_now_iso()),
        updated_at=str(row.get("last_modified") or row.get("created_at") or _utc_now_iso()),
        turn_count=int((manager_row or {}).get("turn_count") or 0),
        pending_plan_count=int((manager_row or {}).get("pending_plan_count") or 0),
        preferences=preferences,
        runtime_mode=str(row.get("mode") or "session_scoped"),
        status=str(row.get("status") or "active"),
        reuse_allowed=bool(row.get("reuse_allowed", False)),
        scope_snapshot_id=_scope_snapshot_id_from_snapshot(scope_snapshot),
        scope_audit=_scope_audit_from_snapshot(scope_snapshot),
    )


def _persona_session_detail_from_db(
    row: dict[str, Any],
    *,
    manager_snapshot: dict[str, Any] | None = None,
) -> PersonaSessionDetail:
    scope_snapshot = row.get("scope_snapshot") or {}
    turns = list((manager_snapshot or {}).get("turns") or [])
    turn_count = int((manager_snapshot or {}).get("turn_count") or len(turns))
    preferences = dict(row.get("preferences") or {})
    runtime_preferences = (manager_snapshot or {}).get("preferences")
    if isinstance(runtime_preferences, dict):
        preferences.update(runtime_preferences)
    return PersonaSessionDetail(
        session_id=str(row.get("id") or ""),
        persona_id=str(row.get("persona_id") or ""),
        created_at=str(row.get("created_at") or _utc_now_iso()),
        updated_at=str(row.get("last_modified") or row.get("created_at") or _utc_now_iso()),
        turn_count=turn_count,
        pending_plan_count=int((manager_snapshot or {}).get("pending_plan_count") or 0),
        preferences=preferences,
        runtime_mode=str(row.get("mode") or "session_scoped"),
        status=str(row.get("status") or "active"),
        reuse_allowed=bool(row.get("reuse_allowed", False)),
        scope_snapshot_id=_scope_snapshot_id_from_snapshot(scope_snapshot),
        scope_audit=_scope_audit_from_snapshot(scope_snapshot),
        turns=turns,
    )


async def _transcribe_audio_chunk(audio_bytes: bytes, audio_format: str) -> str:
    """
    Lightweight scaffold transcription.

    This is intentionally simple to keep persona WS independent from heavy STT
    runtime requirements during early-stage rollout and tests.
    """
    if not audio_bytes:
        return ""
    try:
        text = audio_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        text = ""
    if text:
        return text
    return f"[audio:{audio_format or 'unknown'}:{len(audio_bytes)} bytes]"


def _normalize_persona_live_stt_model(raw_model: Any) -> tuple[str, str, str | None]:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.model_utils import (
        normalize_model_and_variant,
    )

    normalized_raw = str(raw_model or "").strip().lower() or None
    model_name, model_variant = normalize_model_and_variant(
        normalized_raw,
        "parakeet",
        "standard",
    )
    whisper_model_size: str | None = None
    if model_name == "whisper" and normalized_raw and normalized_raw not in {"whisper", "whisper-1"}:
        whisper_model_size = normalized_raw
    return model_name, model_variant, whisper_model_size


def _build_persona_live_stt_config(voice_runtime: dict[str, Any] | None) -> Any:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingConfig,
    )

    config = UnifiedStreamingConfig()
    model_name, model_variant, whisper_model_size = _normalize_persona_live_stt_model(
        (voice_runtime or {}).get("stt_model")
    )
    config.model = model_name
    config.model_variant = model_variant
    if whisper_model_size:
        config.whisper_model_size = whisper_model_size
    language = str((voice_runtime or {}).get("stt_language") or "").strip()
    config.language = language or None
    config.sample_rate = 16000
    config.enable_vad = False
    config.enable_partial = True
    config.partial_interval = 0.35
    config.min_partial_duration = 0.3
    return config


def _create_persona_live_stt_transcriber(*, voice_runtime: dict[str, Any] | None) -> Any:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingTranscriber,
    )

    config = _build_persona_live_stt_config(voice_runtime)
    return UnifiedStreamingTranscriber(config)


def _normalize_persona_live_stt_audio(audio_bytes: bytes, *, audio_format: str) -> bytes:
    import numpy as np

    fmt = str(audio_format or "").strip().lower()
    if fmt in {"pcm16", "pcm", "s16le"}:
        audio_np = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32, copy=False)
        if audio_np.size == 0:
            return b""
        audio_np = audio_np / 32768.0
        return audio_np.astype(np.float32, copy=False).tobytes()
    if fmt in {"float32", "f32le", "f32"}:
        if len(audio_bytes) % 4 != 0:
            raise ValueError("float32 audio size must be divisible by 4")
        return bytes(audio_bytes)
    raise ValueError(f"Unsupported live STT audio_format '{fmt}'")


def _persona_live_transcript_snapshot(
    *,
    transcriber: Any,
    result: dict[str, Any],
) -> str:
    finalized = str(getattr(transcriber, "get_full_transcript", lambda: "")() or "").strip()
    result_text = str(result.get("text") or "").strip()
    if str(result.get("type") or "").strip().lower() == "final":
        return finalized or result_text
    if finalized and result_text:
        if result_text.startswith(finalized):
            return result_text
        return f"{finalized} {result_text}".strip()
    return result_text or finalized


def _persona_live_forward_delta(previous_snapshot: str, next_snapshot: str) -> str:
    previous = str(previous_snapshot or "").strip()
    current = str(next_snapshot or "").strip()
    if not current:
        return ""
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous) :].strip()
    return ""


def _clamp_persona_live_float(
    value: Any,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    try:
        return max(min_value, min(max_value, float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_persona_live_int(
    value: Any,
    *,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        return int(max(min_value, min(max_value, int(value))))
    except (TypeError, ValueError):
        return default


def _create_persona_live_turn_detector(*, voice_runtime: dict[str, Any] | None) -> Any:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        SileroTurnDetector,
        UnifiedStreamingConfig,
    )

    config = UnifiedStreamingConfig()
    runtime = dict(voice_runtime or {})
    return SileroTurnDetector(
        sample_rate=int(config.sample_rate or 16000),
        enabled=_coerce_bool(runtime.get("enable_vad"), default=True),
        vad_threshold=_clamp_persona_live_float(
            runtime.get("vad_threshold"),
            default=float(config.vad_threshold),
            min_value=0.0,
            max_value=1.0,
        ),
        min_silence_ms=_clamp_persona_live_int(
            runtime.get("vad_min_silence_ms"),
            default=int(config.vad_min_silence_ms),
            min_value=50,
            max_value=10_000,
        ),
        turn_stop_secs=_clamp_persona_live_float(
            runtime.get("vad_turn_stop_secs"),
            default=float(config.vad_turn_stop_secs),
            min_value=0.05,
            max_value=10.0,
        ),
        min_utterance_secs=_clamp_persona_live_float(
            runtime.get("vad_min_utterance_secs"),
            default=float(config.vad_min_utterance_secs),
            min_value=0.0,
            max_value=10.0,
        ),
    )


def _apply_persona_live_trigger_phrases(
    transcript: str,
    *,
    trigger_phrases: list[str] | None,
) -> tuple[bool, str]:
    normalized_transcript = str(transcript or "").strip()
    normalized_phrases = [
        str(phrase or "").strip()
        for phrase in list(trigger_phrases or [])
        if str(phrase or "").strip()
    ]
    if not normalized_phrases:
        return True, normalized_transcript

    lowered_transcript = normalized_transcript.lower()
    matched = any(phrase.lower() in lowered_transcript for phrase in normalized_phrases)
    if not matched:
        return False, normalized_transcript

    cleaned = normalized_transcript
    for phrase in normalized_phrases:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return True, cleaned


async def _generate_tts_audio_chunks(
    text: str,
    audio_format: str,
    *,
    chunk_size_bytes: int,
    max_chunks: int,
    max_total_bytes: int,
) -> list[bytes]:
    """
    Lightweight scaffold TTS chunk generator.

    The event contract (`tts_audio` + binary frame) is implemented here; a full
    provider-backed synthesis path can be added without changing WS semantics.
    """
    spoken = str(text or "").strip()
    if not spoken:
        return []
    encoded = spoken.encode("utf-8")
    return _chunk_persona_audio_bytes(
        encoded,
        chunk_size_bytes=chunk_size_bytes,
        max_chunks=max_chunks,
        max_total_bytes=max_total_bytes,
    )


def _chunk_persona_audio_bytes(
    audio_bytes: bytes,
    *,
    chunk_size_bytes: int,
    max_chunks: int,
    max_total_bytes: int,
) -> list[bytes]:
    if not audio_bytes:
        return []
    data = bytes(audio_bytes)
    if max_total_bytes > 0:
        data = data[:max_total_bytes]
    if not data:
        return []
    size = max(1, int(chunk_size_bytes))
    chunks = [data[i : i + size] for i in range(0, len(data), size)]
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks


async def _generate_persona_live_tts_audio(
    text: str,
    *,
    provider: str | None,
    voice: str | None,
    response_format: str = "mp3",
) -> tuple[bytes, str]:
    provider_norm = str(provider or "").strip().lower()
    voice_norm = str(voice or "").strip()
    if not provider_norm or provider_norm == "browser":
        return b"", response_format

    model_name = "kokoro" if provider_norm == "tldw" else provider_norm

    from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

    tts_service = await get_tts_service_v2()
    request = OpenAISpeechRequest(
        model=model_name,
        input=text,
        voice=voice_norm or "af_heart",
        response_format=response_format,
        stream=False,
    )

    audio_chunks: list[bytes] = []
    async for chunk in tts_service.generate_speech(
        request=request,
        provider=provider_norm,
        fallback=True,
    ):
        if chunk:
            audio_chunks.append(chunk)

    return b"".join(audio_chunks), response_format


def _is_authnz_access_token(token: str) -> bool:
    """Return True when token verifies as an AuthNZ access token."""
    try:
        jwt_service = get_jwt_service()
        jwt_service.decode_access_token(token)
        return True
    except TokenExpiredError:
        return True
    except InvalidTokenError:
        return False
    except Exception:
        return False


def _looks_like_jwt(token: str | None) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    parts = raw.split(".")
    return len(parts) == 3 and all(bool(part.strip()) for part in parts)


def _should_treat_bearer_as_api_key(
    token: str | None,
    resolved_api_key: str | None,
) -> bool:
    """Mirror HTTP auth behavior for WS: single-user bearer or non-JWT bearer -> API key."""
    if not token or resolved_api_key:
        return False

    try:
        settings = get_settings()
        if getattr(settings, "AUTH_MODE", None) == "single_user":
            return True
    except Exception as settings_error:
        # Fall through to token-shape heuristics when settings resolution fails.
        logger.debug("Failed to resolve auth settings for WS bearer handling", exc_info=settings_error)

    return not _looks_like_jwt(token)


def _extract_auth_credentials(
    ws: WebSocket,
    token: str | None,
    api_key: str | None,
) -> tuple[str | None, str | None]:
    """Resolve auth credentials with headers taking precedence over query params."""
    auth_token = token
    resolved_api_key = api_key

    try:
        authz = ws.headers.get("authorization") or ws.headers.get("Authorization")
        if authz and authz.lower().startswith("bearer "):
            auth_token = authz.split(" ", 1)[1].strip()
    except Exception as authz_header_error:
        logger.debug("Failed to parse websocket authorization header", exc_info=authz_header_error)

    try:
        header_key = ws.headers.get("x-api-key") or ws.headers.get("X-API-KEY")
        if header_key:
            resolved_api_key = header_key.strip()
    except Exception as api_key_header_error:
        logger.debug("Failed to parse websocket x-api-key header", exc_info=api_key_header_error)

    try:
        proto = ws.headers.get("sec-websocket-protocol") or ws.headers.get("Sec-WebSocket-Protocol")
        if proto and not auth_token:
            parts = [p.strip() for p in proto.split(",")]
            if len(parts) >= 2 and parts[0].lower() == "bearer" and parts[1]:
                auth_token = parts[1]
    except Exception as protocol_header_error:
        logger.debug("Failed to parse websocket subprotocol auth header", exc_info=protocol_header_error)

    return auth_token, resolved_api_key


def _build_request_from_websocket(ws: WebSocket) -> StarletteRequest:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/persona/stream",
        "headers": [
            (k.encode("latin-1"), v.encode("latin-1"))
            for k, v in ws.headers.items()
        ],
    }
    try:
        client = ws.client
        if isinstance(client, (list, tuple)) and len(client) >= 2:
            scope["client"] = (client[0], client[1])
        elif client is not None and getattr(client, "host", None) is not None:
            scope["client"] = (client.host, getattr(client, "port", 0))
    except Exception as client_scope_error:
        logger.debug("Failed to propagate websocket client scope details", exc_info=client_scope_error)
    return StarletteRequest(scope)


async def _resolve_authenticated_user_id(
    ws: WebSocket,
    token: str | None,
    api_key: str | None,
    *,
    required_api_key_scope: str | None = "read",
) -> tuple[str | None, bool, bool]:
    """
    Resolve authenticated user id from WS credentials.

    Returns: (user_id, credentials_supplied, auth_ok)
    """
    auth_token, resolved_api_key = _extract_auth_credentials(ws, token, api_key)
    if _should_treat_bearer_as_api_key(auth_token, resolved_api_key):
        resolved_api_key = auth_token
        auth_token = None

    def _set_auth_context(*, method: str | None, api_key_scopes: set[str] | None = None) -> None:
        try:
            setattr(ws.state, "persona_auth_method", str(method or "").strip().lower())
            setattr(
                ws.state,
                "persona_api_key_scopes",
                sorted(str(scope).strip().lower() for scope in (api_key_scopes or set()) if str(scope).strip()),
            )
        except Exception:
            return

    def _clear_auth_context() -> None:
        _set_auth_context(method=None, api_key_scopes=set())

    credentials_supplied = bool(auth_token or resolved_api_key)
    user_id: str | None = None
    auth_method: str | None = None
    api_key_scopes: set[str] = set()

    if auth_token:
        auth_ok = False
        authnz_token_failed = False
        try:
            req = _build_request_from_websocket(ws)
            user = await verify_jwt_and_fetch_user(req, auth_token)
            uid = str(getattr(user, "id", None) or "")
            if uid:
                user_id = uid
                auth_ok = True
                auth_method = "jwt_authnz"
                logger.debug("persona stream: authenticated via AuthNZ JWT")
        except Exception as exc:
            logger.debug(f"persona stream: AuthNZ JWT auth failed: {exc}")
            if _is_authnz_access_token(auth_token):
                authnz_token_failed = True
                if not resolved_api_key:
                    _clear_auth_context()
                    return None, True, False
        if not auth_ok and not authnz_token_failed:
            try:
                token_data = get_jwt_manager().verify_token(auth_token)
                uid = str(getattr(token_data, "sub", "") or "")
                if uid:
                    user_id = uid
                    auth_ok = True
                    auth_method = "jwt_mcp"
                    logger.debug("persona stream: authenticated via MCP JWT")
            except Exception as exc:
                logger.debug(f"persona stream: MCP JWT auth failed: {exc}")
        if auth_token and not auth_ok and not resolved_api_key:
            _clear_auth_context()
            return None, True, False

    if resolved_api_key and not user_id:
        try:
            api_mgr = await get_api_key_manager()
            client_ip = resolve_client_ip(ws, None)
            required_scope = (
                str(required_api_key_scope or "").strip().lower()
                if required_api_key_scope is not None
                else None
            )
            info = await api_mgr.validate_api_key(
                resolved_api_key,
                required_scope=required_scope,
                ip_address=client_ip,
            )
            if info and info.get("user_id") is not None:
                user_id = str(info["user_id"])
                auth_method = "api_key"
                api_key_scopes = normalize_scope(info.get("scope"))
                logger.debug("persona stream: authenticated via API key")
            else:
                _clear_auth_context()
                return None, True, False
        except (DatabaseError, InvalidTokenError) as exc:
            logger.debug(f"persona stream: API key authentication failed: {exc}")
            _clear_auth_context()
            return None, True, False
        except Exception:
            logger.exception("persona stream: unexpected API key authentication error")
            _clear_auth_context()
            return None, True, False

    if not credentials_supplied:
        _clear_auth_context()
        return None, False, False
    if not user_id:
        _clear_auth_context()
        return None, True, False
    _set_auth_context(method=auth_method, api_key_scopes=api_key_scopes)
    return user_id, True, True


@router.get("/profiles", response_model=list[PersonaProfileResponse], tags=["persona"], status_code=status.HTTP_200_OK)
async def list_persona_profiles(
    active_only: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaProfileResponse]:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profiles = db.list_persona_profiles(
            user_id=user_id,
            include_deleted=include_deleted,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )
        if not profiles and not include_deleted:
            profiles = [_ensure_default_persona_profile(db, user_id=user_id)]
        buddy_rows = _load_persona_buddy_rows_for_projection(db, profiles=profiles)
        return [
            _persona_profile_to_response(
                profile,
                buddy_row=buddy_rows.get(str(profile.get("id") or "").strip()),
            )
            for profile in profiles
        ]
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona profiles") from exc


@router.post("/profiles", response_model=PersonaProfileResponse, tags=["persona"], status_code=status.HTTP_201_CREATED)
async def create_persona_profile(
    payload: PersonaProfileCreate = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaProfileResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        create_data = payload.model_dump(exclude_none=True)
        create_data["user_id"] = user_id
        persona_id = await _run_persona_db_call(db.create_persona_profile, create_data)
        if not await _run_persona_db_call(db.list_persona_policy_rules, persona_id=persona_id, user_id=user_id):
            _ = await _run_persona_db_call(
                db.replace_persona_policy_rules,
                persona_id=persona_id,
                user_id=user_id,
                rules=_DEFAULT_PERSONA_POLICY_RULES,
            )
        profile = await _run_persona_db_call(db.get_persona_profile, persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=500, detail="Failed to load created persona profile")
        try:
            buddy_row = await _run_persona_db_call(_ensure_persona_buddy_after_profile_mutation, db=db, profile=profile)
        except (ValueError, InputError, ConflictError, CharactersRAGDBError) as exc:
            try:
                await _run_persona_db_call(
                    _rollback_created_persona_profile_after_buddy_failure,
                    db,
                    persona_id=persona_id,
                    user_id=user_id,
                    expected_version=int(profile.get("version") or 1),
                )
            except PersonaBuddyRollbackError as rollback_exc:
                raise HTTPException(status_code=500, detail=str(rollback_exc)) from rollback_exc
            raise HTTPException(status_code=400, detail="Persona buddy validation failed") from exc
        return _persona_profile_to_response(profile, buddy_row=buddy_row)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="create persona profile") from exc


@router.get(
    "/profiles/{persona_id}",
    response_model=PersonaProfileResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def get_persona_profile(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaProfileResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        return _persona_profile_to_response(
            profile,
            buddy_row=_load_persona_buddy_row_for_projection(db, profile=profile),
        )
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="get persona profile") from exc


@router.patch(
    "/profiles/{persona_id}",
    response_model=PersonaProfileResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def update_persona_profile(
    persona_id: str,
    payload: PersonaProfileUpdate = Body(...),
    expected_version: int | None = Query(default=None, ge=1),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaProfileResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No profile fields provided for update")
    try:
        previous_profile: dict[str, Any] | None = None
        if expected_version is not None:
            previous_profile = await _run_persona_db_call(
                db.get_persona_profile, persona_id, user_id=user_id, include_deleted=False,
            )
            if previous_profile is None:
                raise HTTPException(status_code=404, detail="Persona profile not found")
        ok = await _run_persona_db_call(
            db.update_persona_profile,
            persona_id=persona_id,
            user_id=user_id,
            update_data=update_data,
            expected_version=expected_version,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        profile = await _run_persona_db_call(
            db.get_persona_profile, persona_id, user_id=user_id, include_deleted=False,
        )
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        try:
            buddy_row = await _run_persona_db_call(_ensure_persona_buddy_after_profile_mutation, db=db, profile=profile)
        except (ValueError, InputError, ConflictError, CharactersRAGDBError) as exc:
            try:
                await _run_persona_db_call(
                    _rollback_updated_persona_profile_after_buddy_failure,
                    db,
                    persona_id=persona_id,
                    user_id=user_id,
                    update_data=update_data,
                    previous_profile=previous_profile,
                    expected_version=int(profile.get("version") or 1),
                )
            except PersonaBuddyRollbackError as rollback_exc:
                raise HTTPException(status_code=500, detail=str(rollback_exc)) from rollback_exc
            raise HTTPException(status_code=400, detail="Persona buddy validation failed") from exc
        return _persona_profile_to_response(profile, buddy_row=buddy_row)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="update persona profile") from exc


@router.get(
    "/profiles/{persona_id}/buddy",
    response_model=PersonaBuddyResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def get_persona_buddy(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaBuddyResponse:
    """Return the resolved buddy profile for one active persona profile."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = await _run_persona_db_call(db.get_persona_profile, persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        buddy = await _run_persona_db_call(ensure_persona_buddy_for_profile, db, profile)
        if buddy is None:
            raise HTTPException(status_code=500, detail="Failed to load persona buddy")
        return _persona_buddy_to_response(buddy)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="get persona buddy") from exc


@router.delete(
    "/profiles/{persona_id}",
    response_model=PersonaDeleteResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def delete_persona_profile(
    persona_id: str,
    expected_version: int | None = Query(default=None, ge=1),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaDeleteResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        ok = await _run_persona_db_call(
            db.soft_delete_persona_profile,
            persona_id=persona_id,
            user_id=user_id,
            expected_version=expected_version,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        return PersonaDeleteResponse(status="deleted", persona_id=persona_id)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="delete persona profile") from exc


@router.post(
    "/profiles/{persona_id}/restore",
    response_model=PersonaProfileResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def restore_persona_profile(
    persona_id: str,
    expected_version: int = Query(..., ge=1),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaProfileResponse:
    """Restore a soft-deleted persona profile and resume its buddy projection."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        ok = await _run_persona_db_call(
            db.restore_persona_profile,
            persona_id=persona_id,
            user_id=user_id,
            expected_version=expected_version,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        profile = await _run_persona_db_call(
            db.get_persona_profile,
            persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        return _persona_profile_to_response(
            profile,
            buddy_row=_load_persona_buddy_row_for_projection(db, profile=profile),
        )
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="restore persona profile") from exc


@router.get(
    "/profiles/{persona_id}/exemplars",
    response_model=list[PersonaExemplarResponse],
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def list_persona_exemplars(
    persona_id: str,
    include_disabled: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    include_deleted_personas: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaExemplarResponse]:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=include_deleted_personas,
        )
        exemplars = await _run_persona_db_call(
            db.list_persona_exemplars,
            user_id=user_id,
            persona_id=persona_id,
            include_disabled=include_disabled,
            include_deleted=include_deleted,
            include_deleted_personas=include_deleted_personas,
            limit=limit,
            offset=offset,
        )
        return [_persona_exemplar_to_response(exemplar) for exemplar in exemplars]
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona exemplars") from exc


@router.post(
    "/profiles/{persona_id}/exemplars",
    response_model=PersonaExemplarResponse,
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def create_persona_exemplar(
    persona_id: str,
    payload: PersonaExemplarCreate = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaExemplarResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        create_data = payload.model_dump(exclude_none=True)
        create_data["persona_id"] = persona_id
        create_data["user_id"] = user_id
        exemplar_id = await _run_persona_db_call(db.create_persona_exemplar, create_data)
        exemplar = await _run_persona_db_call(
            db.get_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            include_disabled=True,
            include_deleted=False,
        )
        if exemplar is None:
            raise HTTPException(status_code=500, detail="Failed to load created persona exemplar")
        return _persona_exemplar_to_response(exemplar)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="create persona exemplar") from exc


@router.post(
    "/profiles/{persona_id}/exemplars/import",
    response_model=list[PersonaExemplarResponse],
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def import_persona_exemplars(
    persona_id: str,
    payload: PersonaExemplarImportRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaExemplarResponse]:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        candidate_rows = build_transcript_exemplar_candidates(
            transcript=payload.transcript,
            source_ref=payload.source_ref,
            notes=payload.notes,
            max_candidates=payload.max_candidates,
        )
        created: list[PersonaExemplarResponse] = []
        for candidate in candidate_rows:
            create_data = dict(candidate)
            create_data["persona_id"] = persona_id
            create_data["user_id"] = user_id
            exemplar_id = await _run_persona_db_call(db.create_persona_exemplar, create_data)
            exemplar = await _run_persona_db_call(
                db.get_persona_exemplar,
                exemplar_id=exemplar_id,
                persona_id=persona_id,
                user_id=user_id,
                include_disabled=True,
                include_deleted=False,
            )
            if exemplar is None:
                raise HTTPException(status_code=500, detail="Failed to load imported persona exemplar")
            created.append(_persona_exemplar_to_response(exemplar))
        return created
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError, ValueError) as exc:
        raise _to_http_exception(exc, action="import persona exemplars") from exc


@router.get(
    "/profiles/{persona_id}/exemplars/{exemplar_id}",
    response_model=PersonaExemplarResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def get_persona_exemplar(
    persona_id: str,
    exemplar_id: str,
    include_disabled: bool = Query(default=False),
    include_deleted: bool = Query(default=False),
    include_deleted_personas: bool = Query(default=False),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaExemplarResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=include_deleted_personas,
        )
        exemplar = await _run_persona_db_call(
            db.get_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            include_disabled=include_disabled,
            include_deleted=include_deleted,
            include_deleted_personas=include_deleted_personas,
        )
        if exemplar is None:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        return _persona_exemplar_to_response(exemplar)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="get persona exemplar") from exc


@router.patch(
    "/profiles/{persona_id}/exemplars/{exemplar_id}",
    response_model=PersonaExemplarResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def update_persona_exemplar(
    persona_id: str,
    exemplar_id: str,
    payload: PersonaExemplarUpdate = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaExemplarResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No exemplar fields provided for update")
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        ok = await _run_persona_db_call(
            db.update_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            update_data=update_data,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        exemplar = await _run_persona_db_call(
            db.get_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            include_disabled=True,
            include_deleted=False,
        )
        if exemplar is None:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        return _persona_exemplar_to_response(exemplar)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="update persona exemplar") from exc


@router.post(
    "/profiles/{persona_id}/exemplars/{exemplar_id}/review",
    response_model=PersonaExemplarResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def review_persona_exemplar(
    persona_id: str,
    exemplar_id: str,
    payload: PersonaExemplarReviewRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaExemplarResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        exemplar = await _run_persona_db_call(
            db.get_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            include_disabled=True,
            include_deleted=False,
        )
        if exemplar is None:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        if str(exemplar.get("source_type") or "") != "generated_candidate":
            raise HTTPException(status_code=400, detail="Only generated candidates can be reviewed")
        ok = await _run_persona_db_call(
            db.update_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            update_data={
                "enabled": payload.action == "approve",
                "notes": append_exemplar_review_note(
                    existing_notes=exemplar.get("notes"),
                    action=payload.action,
                    review_note=payload.notes,
                ),
            },
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        updated = await _run_persona_db_call(
            db.get_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
            include_disabled=True,
            include_deleted=False,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        return _persona_exemplar_to_response(updated)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError, ValueError) as exc:
        raise _to_http_exception(exc, action="review persona exemplar") from exc


@router.delete(
    "/profiles/{persona_id}/exemplars/{exemplar_id}",
    response_model=PersonaExemplarDeleteResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def delete_persona_exemplar(
    persona_id: str,
    exemplar_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaExemplarDeleteResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        ok = await _run_persona_db_call(
            db.soft_delete_persona_exemplar,
            exemplar_id=exemplar_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Persona exemplar not found")
        return PersonaExemplarDeleteResponse(status="deleted", persona_id=persona_id, exemplar_id=exemplar_id)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="delete persona exemplar") from exc


@router.get(
    "/profiles/{persona_id}/state",
    response_model=PersonaStateResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def get_persona_profile_state(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaStateResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            _increment_persona_state_metric(action="read", result="not_found")
            raise HTTPException(status_code=404, detail="Persona profile not found")
        rows = _get_persona_state_rows(db, user_id=user_id, persona_id=persona_id)
        _increment_persona_state_metric(action="read", result="success")
        return _persona_state_response_from_rows(persona_id=persona_id, rows=rows)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        _increment_persona_state_metric(action="read", result="error")
        raise _to_http_exception(exc, action="get persona profile state") from exc


@router.put(
    "/profiles/{persona_id}/state",
    response_model=PersonaStateResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def replace_persona_profile_state(
    persona_id: str,
    payload: PersonaStateUpdateRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaStateResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    updates_raw = payload.model_dump(exclude_unset=True)
    updates: dict[str, str | None] = {
        key: (None if value is None else str(value))
        for key, value in updates_raw.items()
        if key in _PERSONA_STATE_FIELD_TO_MEMORY_TYPE
    }
    if not updates:
        _increment_persona_state_metric(action="write", result="rejected_empty")
        raise HTTPException(status_code=400, detail="No state document fields provided for update")
    max_chars = _get_persona_state_doc_max_chars()
    for field_name, value in updates.items():
        if value is None:
            continue
        if len(value) > max_chars:
            _increment_persona_state_metric(action="write", result="rejected_too_large")
            raise HTTPException(
                status_code=413,
                detail=f"{field_name} exceeds max chars ({len(value)} > {max_chars})",
            )
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            _increment_persona_state_metric(action="write", result="not_found")
            raise HTTPException(status_code=404, detail="Persona profile not found")
        _replace_persona_state_docs(
            db,
            user_id=user_id,
            persona_id=persona_id,
            updates=updates,
        )
        rows = _get_persona_state_rows(db, user_id=user_id, persona_id=persona_id)
        _increment_persona_state_metric(action="write", result="success")
        return _persona_state_response_from_rows(persona_id=persona_id, rows=rows)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        _increment_persona_state_metric(action="write", result="error")
        raise _to_http_exception(exc, action="replace persona profile state") from exc


@router.get(
    "/profiles/{persona_id}/state/history",
    response_model=PersonaStateHistoryResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def list_persona_profile_state_history(
    persona_id: str,
    field: str | None = Query(default=None),
    include_archived: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=2000),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaStateHistoryResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)

    memory_type: str | None = None
    if field is not None:
        normalized_field = str(field).strip().lower()
        if normalized_field not in _PERSONA_STATE_FIELD_TO_MEMORY_TYPE:
            raise HTTPException(status_code=400, detail="Invalid state field filter")
        memory_type = _PERSONA_STATE_FIELD_TO_MEMORY_TYPE[normalized_field]

    bounded_limit = min(int(limit), _get_persona_state_history_max_entries())
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            _increment_persona_state_metric(action="history", result="not_found")
            raise HTTPException(status_code=404, detail="Persona profile not found")
        rows = db.list_persona_memory_entries(
            user_id=user_id,
            persona_id=persona_id,
            memory_type=memory_type,
            include_archived=bool(include_archived),
            include_deleted=False,
            limit=bounded_limit,
            offset=0,
        )
        _increment_persona_state_metric(action="history", result="success")
        return _persona_state_history_response_from_rows(persona_id=persona_id, rows=rows)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        _increment_persona_state_metric(action="history", result="error")
        raise _to_http_exception(exc, action="list persona profile state history") from exc


@router.post(
    "/profiles/{persona_id}/state/restore",
    response_model=PersonaStateResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def restore_persona_profile_state_entry(
    persona_id: str,
    payload: PersonaStateRestoreRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaStateResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    target_entry_id = str(payload.entry_id or "").strip()
    if not target_entry_id:
        raise HTTPException(status_code=400, detail="entry_id is required")

    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            _increment_persona_state_metric(action="restore", result="not_found")
            raise HTTPException(status_code=404, detail="Persona profile not found")

        rows = db.list_persona_memory_entries(
            user_id=user_id,
            persona_id=persona_id,
            include_archived=True,
            include_deleted=False,
            limit=_get_persona_state_history_max_entries(),
            offset=0,
        )
        target_row: dict[str, Any] | None = None
        for row in rows:
            row_id = str(row.get("id") or "").strip()
            if row_id != target_entry_id:
                continue
            if str(row.get("memory_type") or "").strip() not in _PERSONA_STATE_MEMORY_TYPES:
                continue
            target_row = row
            break
        if target_row is None:
            _increment_persona_state_metric(action="restore", result="entry_not_found")
            raise HTTPException(status_code=404, detail="State history entry not found")

        target_memory_type = str(target_row.get("memory_type") or "").strip()
        for row in rows:
            if str(row.get("memory_type") or "").strip() != target_memory_type:
                continue
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                continue
            if row_id == target_entry_id:
                continue
            if _coerce_bool(row.get("archived"), default=False):
                continue
            with contextlib.suppress(Exception):
                db.set_persona_memory_archived(
                    entry_id=row_id,
                    user_id=user_id,
                    persona_id=persona_id,
                    archived=True,
                )

        if _coerce_bool(target_row.get("archived"), default=False):
            restored = db.set_persona_memory_archived(
                entry_id=target_entry_id,
                user_id=user_id,
                persona_id=persona_id,
                archived=False,
            )
            if not restored:
                _increment_persona_state_metric(action="restore", result="entry_not_found")
                raise HTTPException(status_code=404, detail="State history entry not found")

        current_rows = _get_persona_state_rows(db, user_id=user_id, persona_id=persona_id)
        _increment_persona_state_metric(action="restore", result="success")
        return _persona_state_response_from_rows(persona_id=persona_id, rows=current_rows)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        _increment_persona_state_metric(action="restore", result="error")
        raise _to_http_exception(exc, action="restore persona profile state entry") from exc


@router.get(
    "/profiles/{persona_id}/scope-rules",
    response_model=PersonaScopeRulesResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def list_persona_scope_rules(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaScopeRulesResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        rules = db.list_persona_scope_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
        return PersonaScopeRulesResponse(persona_id=persona_id, rules=rules)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona scope rules") from exc


@router.put(
    "/profiles/{persona_id}/scope-rules",
    response_model=PersonaScopeRulesResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def replace_persona_scope_rules(
    persona_id: str,
    payload: PersonaScopeRulesReplaceRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaScopeRulesResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        replaced_count = db.replace_persona_scope_rules(
            persona_id=persona_id,
            user_id=user_id,
            rules=[rule.model_dump() for rule in payload.rules],
        )
        rules = db.list_persona_scope_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
        return PersonaScopeRulesResponse(persona_id=persona_id, replaced_count=replaced_count, rules=rules)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="replace persona scope rules") from exc


@router.get(
    "/profiles/{persona_id}/policy-rules",
    response_model=PersonaPolicyRulesResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def list_persona_policy_rules(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaPolicyRulesResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        rules = db.list_persona_policy_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
        return PersonaPolicyRulesResponse(persona_id=persona_id, rules=rules)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona policy rules") from exc


@router.put(
    "/profiles/{persona_id}/policy-rules",
    response_model=PersonaPolicyRulesResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def replace_persona_policy_rules(
    persona_id: str,
    payload: PersonaPolicyRulesReplaceRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaPolicyRulesResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise HTTPException(status_code=404, detail="Persona profile not found")
        replaced_count = db.replace_persona_policy_rules(
            persona_id=persona_id,
            user_id=user_id,
            rules=[rule.model_dump() for rule in payload.rules],
        )
        rules = db.list_persona_policy_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
        return PersonaPolicyRulesResponse(persona_id=persona_id, replaced_count=replaced_count, rules=rules)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="replace persona policy rules") from exc


@router.get(
    "/profiles/{persona_id}/setup-analytics",
    response_model=PersonaSetupAnalyticsResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def get_persona_setup_analytics(
    persona_id: str,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaSetupAnalyticsResponse:
    """Return aggregated setup analytics and recent setup runs for one persona."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        analytics = db.get_persona_setup_analytics_summary(
            user_id=int(user_id),
            persona_id=persona_id,
            days=days,
            recent_run_limit=limit,
        )
        return PersonaSetupAnalyticsResponse(
            persona_id=persona_id,
            summary=PersonaSetupAnalyticsSummary.model_validate(
                analytics.get("summary") or {}
            ),
            recent_runs=[
                PersonaSetupAnalyticsRunSummary.model_validate(item)
                for item in analytics.get("recent_runs") or []
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="get persona setup analytics") from exc


@router.post(
    "/profiles/{persona_id}/setup-events",
    response_model=PersonaSetupEventWriteResponse,
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def create_persona_setup_event(
    persona_id: str,
    payload: PersonaSetupEventCreate = Body(...),
    response: Response = None,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaSetupEventWriteResponse:
    """Append one persona setup analytics event and return its dedupe status."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        event_row = db.record_persona_setup_event(
            user_id=int(user_id),
            persona_id=persona_id,
            event_id=payload.event_id,
            run_id=payload.run_id,
            event_type=payload.event_type,
            event_key=payload.event_key,
            step=payload.step,
            completion_type=payload.completion_type,
            detour_source=payload.detour_source,
            action_target=payload.action_target,
            metadata=payload.metadata,
        )
        if bool(event_row.get("deduped")) and response is not None:
            response.status_code = status.HTTP_200_OK
        return PersonaSetupEventWriteResponse(
            event_id=str(event_row.get("event_id") or payload.event_id),
            run_id=str(event_row.get("run_id") or payload.run_id),
            event_type=str(event_row.get("event_type") or payload.event_type),
            deduped=bool(event_row.get("deduped")),
            created_at=str(event_row.get("created_at") or "").strip() or None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="record persona setup event") from exc


@router.get(
    "/profiles/{persona_id}/voice-analytics",
    response_model=PersonaVoiceAnalyticsResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def get_persona_voice_analytics(
    persona_id: str,
    days: int = Query(7, ge=1, le=365),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaVoiceAnalyticsResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        summary_stats = get_voice_analytics_summary_stats(
            db,
            user_id=int(user_id),
            days=days,
            persona_id=persona_id,
        )
        direct_command_stats = get_voice_resolution_stats(
            db,
            user_id=int(user_id),
            days=days,
            persona_id=persona_id,
            resolution_type="direct_command",
        )
        fallback_stats = get_voice_resolution_stats(
            db,
            user_id=int(user_id),
            days=days,
            persona_id=persona_id,
            resolution_type="planner_fallback",
        )
        command_rows = get_voice_top_commands(
            db,
            user_id=int(user_id),
            days=days,
            limit=50,
            persona_id=persona_id,
            resolution_type="direct_command",
        )
        live_voice_stats = get_persona_live_voice_summary(
            db,
            user_id=int(user_id),
            days=days,
            persona_id=persona_id,
        )
        recent_live_sessions = db.list_persona_live_voice_session_summaries(
            user_id=int(user_id),
            persona_id=persona_id,
            days=days,
            limit=10,
        )

        total_events = int(summary_stats.get("total_commands") or 0)
        planner_fallback_count = int(fallback_stats.get("total_invocations") or 0)

        def _session_summary_from_row(item: dict[str, Any]) -> PersonaLiveVoiceSessionSummary:
            return PersonaLiveVoiceSessionSummary(
                session_id=str(item.get("session_id") or ""),
                started_at=str(item.get("started_at") or "").strip() or None,
                ended_at=str(item.get("ended_at") or "").strip() or None,
                auto_commit_enabled=(
                    None
                    if item.get("auto_commit_enabled") is None
                    else bool(item.get("auto_commit_enabled"))
                ),
                vad_threshold=(
                    float(item.get("vad_threshold"))
                    if item.get("vad_threshold") is not None
                    else None
                ),
                min_silence_ms=(
                    int(item.get("min_silence_ms"))
                    if item.get("min_silence_ms") is not None
                    else None
                ),
                turn_stop_secs=(
                    float(item.get("turn_stop_secs"))
                    if item.get("turn_stop_secs") is not None
                    else None
                ),
                min_utterance_secs=(
                    float(item.get("min_utterance_secs"))
                    if item.get("min_utterance_secs") is not None
                    else None
                ),
                turn_detection_changed_during_session=bool(
                    item.get("turn_detection_changed_during_session")
                ),
                total_committed_turns=int(item.get("total_committed_turns") or 0),
                vad_auto_commit_count=int(item.get("vad_auto_commit_count") or 0),
                manual_commit_count=int(item.get("manual_commit_count") or 0),
                manual_mode_required_count=int(item.get("manual_mode_required_count") or 0),
                text_only_tts_count=int(item.get("text_only_tts_count") or 0),
                listening_recovery_count=int(item.get("listening_recovery_count") or 0),
                thinking_recovery_count=int(item.get("thinking_recovery_count") or 0),
            )

        return PersonaVoiceAnalyticsResponse(
            persona_id=persona_id,
            summary=PersonaVoiceAnalyticsSummary(
                total_events=total_events,
                direct_command_count=int(direct_command_stats.get("total_invocations") or 0),
                planner_fallback_count=planner_fallback_count,
                success_rate=float(summary_stats.get("success_rate") or 0.0),
                fallback_rate=(
                    float(planner_fallback_count) / float(total_events)
                    if total_events
                    else 0.0
                ),
                avg_response_time_ms=float(
                    summary_stats.get("avg_response_time_ms") or 0.0
                ),
            ),
            live_voice=PersonaLiveVoiceAnalyticsSummary(
                total_committed_turns=int(live_voice_stats.get("total_committed_turns") or 0),
                vad_auto_commit_count=int(live_voice_stats.get("vad_auto_commit_count") or 0),
                manual_commit_count=int(live_voice_stats.get("manual_commit_count") or 0),
                vad_auto_rate=float(live_voice_stats.get("vad_auto_rate") or 0.0),
                manual_commit_rate=float(live_voice_stats.get("manual_commit_rate") or 0.0),
                degraded_session_count=int(live_voice_stats.get("degraded_session_count") or 0),
            ),
            recent_live_sessions=[
                _session_summary_from_row(item)
                for item in recent_live_sessions
                if str(item.get("session_id") or "").strip()
            ],
            commands=[
                PersonaVoiceCommandAnalyticsItem(
                    command_id=str(item.get("command_id") or ""),
                    command_name=item.get("command_name"),
                    total_invocations=int(item.get("total_invocations") or 0),
                    success_count=int(item.get("success_count") or 0),
                    error_count=int(item.get("error_count") or 0),
                    avg_response_time_ms=float(
                        item.get("avg_response_time_ms") or 0.0
                    ),
                    last_used=(
                        str(item.get("last_used") or "").strip() or None
                    ),
                )
                for item in command_rows
                if str(item.get("command_id") or "").strip()
            ],
            fallbacks=PersonaVoiceFallbackAnalytics(
                total_invocations=int(fallback_stats.get("total_invocations") or 0),
                success_count=int(fallback_stats.get("success_count") or 0),
                error_count=int(fallback_stats.get("error_count") or 0),
                avg_response_time_ms=float(
                    fallback_stats.get("avg_response_time_ms") or 0.0
                ),
                last_used=str(fallback_stats.get("last_used") or "").strip() or None,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="get persona voice analytics") from exc


@router.put(
    "/profiles/{persona_id}/voice-analytics/live-sessions/{session_id}",
    response_model=PersonaLiveVoiceSessionSummary,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def update_persona_live_voice_session_analytics(
    persona_id: str,
    session_id: str,
    payload: PersonaLiveVoiceSessionUpdateRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaLiveVoiceSessionSummary:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        db.upsert_persona_live_voice_session_summary(
            user_id=int(user_id),
            persona_id=persona_id,
            session_id=session_id,
            listening_recovery_count=payload.listening_recovery_count,
            thinking_recovery_count=payload.thinking_recovery_count,
            finalize=payload.finalize,
            ended_at=payload.ended_at,
        )
        row = db.get_persona_live_voice_session_summary(
            user_id=int(user_id),
            persona_id=persona_id,
            session_id=session_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Live session summary not found")
        return PersonaLiveVoiceSessionSummary(
            session_id=str(row.get("session_id") or ""),
            started_at=str(row.get("started_at") or "").strip() or None,
            ended_at=str(row.get("ended_at") or "").strip() or None,
            auto_commit_enabled=(
                None
                if row.get("auto_commit_enabled") is None
                else bool(row.get("auto_commit_enabled"))
            ),
            vad_threshold=(
                float(row.get("vad_threshold"))
                if row.get("vad_threshold") is not None
                else None
            ),
            min_silence_ms=(
                int(row.get("min_silence_ms"))
                if row.get("min_silence_ms") is not None
                else None
            ),
            turn_stop_secs=(
                float(row.get("turn_stop_secs"))
                if row.get("turn_stop_secs") is not None
                else None
            ),
            min_utterance_secs=(
                float(row.get("min_utterance_secs"))
                if row.get("min_utterance_secs") is not None
                else None
            ),
            turn_detection_changed_during_session=bool(
                row.get("turn_detection_changed_during_session")
            ),
            total_committed_turns=int(row.get("total_committed_turns") or 0),
            vad_auto_commit_count=int(row.get("vad_auto_commit_count") or 0),
            manual_commit_count=int(row.get("manual_commit_count") or 0),
            manual_mode_required_count=int(row.get("manual_mode_required_count") or 0),
            text_only_tts_count=int(row.get("text_only_tts_count") or 0),
            listening_recovery_count=int(row.get("listening_recovery_count") or 0),
            thinking_recovery_count=int(row.get("thinking_recovery_count") or 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="update persona live voice session analytics") from exc


@router.get(
    "/profiles/{persona_id}/voice-commands",
    response_model=VoiceCommandListResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def list_persona_voice_commands(
    persona_id: str,
    include_system: bool = Query(default=False),
    include_disabled: bool = Query(default=False),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandListResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        registry = get_voice_command_registry()
        registry.load_defaults()
        registry.refresh_user_commands(
            db,
            user_id=int(user_id),
            include_disabled=include_disabled,
            persona_id=persona_id,
        )
        commands = registry.get_all_commands(
            int(user_id),
            include_system=include_system,
            include_disabled=include_disabled,
            persona_id=persona_id,
        )
        connections_by_id = await _get_persona_connections_by_id(
            db,
            user_id=user_id,
            persona_id=persona_id,
        )
        command_infos = [
            _voice_command_to_response_with_connections(
                command,
                connections_by_id=connections_by_id,
            )
            for command in commands
        ]
        return VoiceCommandListResponse(commands=command_infos, total=len(command_infos))
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="list persona voice commands") from exc


@router.post(
    "/profiles/{persona_id}/voice-commands",
    response_model=VoiceCommandInfo,
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def create_persona_voice_command(
    persona_id: str,
    payload: VoiceCommandDefinition = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    normalized_persona_id = _normalize_command_path_persona_id(persona_id, payload.persona_id)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=normalized_persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        connections_by_id: dict[str, PersonaConnectionResponse] = {}
        if payload.connection_id:
            connections_by_id = await _get_persona_connections_by_id(
                db,
                user_id=user_id,
                persona_id=normalized_persona_id,
            )
            if payload.connection_id not in connections_by_id:
                raise HTTPException(status_code=404, detail="Persona connection not found")

        command = VoiceCommand(
            id=str(uuid.uuid4()),
            user_id=int(user_id),
            persona_id=normalized_persona_id,
            connection_id=payload.connection_id,
            name=payload.name,
            phrases=payload.phrases,
            action_type=VoiceActionTypeInternal(payload.action_type.value),
            action_config=payload.action_config,
            priority=payload.priority,
            enabled=payload.enabled,
            requires_confirmation=payload.requires_confirmation,
            description=payload.description,
        )
        save_voice_command(db, command)
        registry = get_voice_command_registry()
        registry.load_defaults()
        registry.register_command(command)
        saved = get_voice_command_db(
            db,
            command.id,
            int(user_id),
            persona_id=normalized_persona_id,
        ) or command
        return _voice_command_to_response_with_connections(
            saved,
            connections_by_id=connections_by_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="create persona voice command") from exc


@router.put(
    "/profiles/{persona_id}/voice-commands/{command_id}",
    response_model=VoiceCommandInfo,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def update_persona_voice_command(
    persona_id: str,
    command_id: str,
    payload: VoiceCommandDefinition = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    normalized_persona_id = _normalize_command_path_persona_id(persona_id, payload.persona_id)
    try:
        await _get_persona_profile_or_404(
            db,
            persona_id=normalized_persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        existing = get_voice_command_db(
            db,
            command_id,
            int(user_id),
            persona_id=normalized_persona_id,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="Voice command not found")

        next_connection_id = (
            payload.connection_id
            if "connection_id" in payload.model_fields_set
            else existing.connection_id
        )
        connections_by_id: dict[str, PersonaConnectionResponse] = {}
        if next_connection_id:
            connections_by_id = await _get_persona_connections_by_id(
                db,
                user_id=user_id,
                persona_id=normalized_persona_id,
            )
            if next_connection_id not in connections_by_id:
                raise HTTPException(status_code=404, detail="Persona connection not found")

        updated = VoiceCommand(
            id=command_id,
            user_id=int(user_id),
            persona_id=normalized_persona_id,
            connection_id=next_connection_id,
            name=payload.name,
            phrases=payload.phrases,
            action_type=VoiceActionTypeInternal(payload.action_type.value),
            action_config=payload.action_config,
            priority=payload.priority,
            enabled=payload.enabled if "enabled" in payload.model_fields_set else existing.enabled,
            requires_confirmation=payload.requires_confirmation,
            description=payload.description,
            created_at=existing.created_at,
        )
        save_voice_command(db, updated)
        registry = get_voice_command_registry()
        registry.load_defaults()
        registry.register_command(updated)
        saved = get_voice_command_db(
            db,
            command_id,
            int(user_id),
            persona_id=normalized_persona_id,
        ) or updated
        return _voice_command_to_response_with_connections(
            saved,
            connections_by_id=connections_by_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="update persona voice command") from exc


@router.post(
    "/profiles/{persona_id}/voice-commands/{command_id}/toggle",
    response_model=VoiceCommandInfo,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def toggle_persona_voice_command(
    persona_id: str,
    command_id: str,
    payload: VoiceCommandToggleRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> VoiceCommandInfo:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        existing = get_voice_command_db(
            db,
            command_id,
            int(user_id),
            persona_id=persona_id,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="Voice command not found")

        updated = VoiceCommand(
            id=command_id,
            user_id=existing.user_id,
            persona_id=existing.persona_id,
            connection_id=existing.connection_id,
            name=existing.name,
            phrases=existing.phrases,
            action_type=existing.action_type,
            action_config=existing.action_config,
            priority=existing.priority,
            enabled=payload.enabled,
            requires_confirmation=existing.requires_confirmation,
            description=existing.description,
            created_at=existing.created_at,
        )
        save_voice_command(db, updated)
        registry = get_voice_command_registry()
        registry.load_defaults()
        registry.register_command(updated)
        saved = get_voice_command_db(
            db,
            command_id,
            int(user_id),
            persona_id=persona_id,
        ) or updated
        connections_by_id: dict[str, PersonaConnectionResponse] = {}
        if saved.connection_id:
            connections_by_id = await _get_persona_connections_by_id(
                db,
                user_id=user_id,
                persona_id=persona_id,
            )
        return _voice_command_to_response_with_connections(
            saved,
            connections_by_id=connections_by_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="toggle persona voice command") from exc


@router.delete(
    "/profiles/{persona_id}/voice-commands/{command_id}",
    tags=["persona"],
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    dependencies=[Depends(check_rate_limit)],
)
async def delete_persona_voice_command(
    persona_id: str,
    command_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> None:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        existing = get_voice_command_db(
            db,
            command_id,
            int(user_id),
            persona_id=persona_id,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="Voice command not found")
        deleted = delete_voice_command_db(db, command_id, int(user_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Voice command not found")
        registry = get_voice_command_registry()
        registry.unregister_command(command_id, int(user_id), persona_id=persona_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="delete persona voice command") from exc


@router.post(
    "/profiles/{persona_id}/voice-commands/test",
    response_model=PersonaCommandDryRunResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def dry_run_persona_voice_command(
    persona_id: str,
    payload: PersonaCommandDryRunRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaCommandDryRunResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        router_instance = get_voice_command_router()
        parsed = await router_instance.match_registered_command(
            payload.heard_text,
            user_id=int(user_id),
            persona_id=persona_id,
            db=db,
        )
        if parsed is None or not parsed.intent.command_id:
            return PersonaCommandDryRunResponse(
                heard_text=payload.heard_text,
                matched=False,
                fallback_to_persona_planner=True,
                failure_phase="no_match",
            )

        command = get_voice_command_db(
            db,
            parsed.intent.command_id,
            int(user_id),
            persona_id=persona_id,
        )
        if command is None:
            command = get_voice_command_registry().get_command(
                parsed.intent.command_id,
                int(user_id),
                persona_id=persona_id,
            )
        if command is None or not command.enabled:
            return PersonaCommandDryRunResponse(
                heard_text=payload.heard_text,
                matched=False,
                fallback_to_persona_planner=True,
                failure_phase="disabled_command" if command is not None and not command.enabled else "no_match",
            )

        connections_by_id: dict[str, PersonaConnectionResponse] = {}
        if command.connection_id:
            connections_by_id = await _get_persona_connections_by_id(
                db,
                user_id=user_id,
                persona_id=persona_id,
            )
        connection_id, connection_status, connection_name = _resolve_voice_command_connection_status(
            command,
            connections_by_id=connections_by_id,
        )
        persona_policy_rules = normalize_policy_rules(
            await _run_persona_db_call(
                db.list_persona_policy_rules,
                persona_id=persona_id,
                user_id=user_id,
                include_deleted=False,
            )
        )
        planned_action = PersonaCommandPlannedActionResponse(
            target_type=command.action_type.value,
            target_name=_voice_target_name(command),
            payload_preview=_build_payload_preview(command, parsed.intent.entities),
        )
        safety_gate = _build_dry_run_safety_gate(
            command=command,
            persona_policy_rules=persona_policy_rules,
        )
        return PersonaCommandDryRunResponse(
            heard_text=payload.heard_text,
            matched=True,
            match_reason=parsed.match_reason or parsed.match_method,
            command_id=command.id,
            command_name=command.name,
            connection_id=connection_id,
            connection_status=connection_status,
            connection_name=connection_name,
            extracted_params=parsed.intent.entities,
            planned_action=planned_action,
            safety_gate=safety_gate,
            fallback_to_persona_planner=False,
            failure_phase="missing_connection" if connection_status == "missing" else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="dry run persona voice command") from exc


@router.get(
    "/profiles/{persona_id}/connections",
    response_model=list[PersonaConnectionResponse],
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def list_persona_connections(
    persona_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaConnectionResponse]:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        return await _list_persona_connections(db, user_id=user_id, persona_id=persona_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="list persona connections") from exc


@router.post(
    "/profiles/{persona_id}/connections",
    response_model=PersonaConnectionResponse,
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def create_persona_connection(
    persona_id: str,
    payload: PersonaConnectionCreate = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaConnectionResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        connection_id = str(payload.id or uuid.uuid4())
        connection_content = _connection_memory_content_from_payload(payload)
        await _run_persona_db_call(
            db.add_persona_memory_entry,
            {
                "id": connection_id,
                "persona_id": persona_id,
                "user_id": user_id,
                "memory_type": _PERSONA_CONNECTION_MEMORY_TYPE,
                "content": json.dumps(connection_content),
                "salience": 0.0,
            },
        )
        responses = await _get_persona_connections_by_id(db, user_id=user_id, persona_id=persona_id)
        response = responses.get(connection_id)
        if response is None:
            raise HTTPException(status_code=500, detail="Failed to load created persona connection")
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="create persona connection") from exc


@router.put(
    "/profiles/{persona_id}/connections/{connection_id}",
    response_model=PersonaConnectionResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def update_persona_connection(
    persona_id: str,
    connection_id: str,
    payload: PersonaConnectionUpdate = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaConnectionResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        row = await _get_persona_connection_row_or_404(
            db,
            user_id=user_id,
            persona_id=persona_id,
            connection_id=connection_id,
        )
        updated_content = _connection_memory_content_from_update(
            connection_content_from_row(row),
            payload,
        )
        updated = await _run_persona_db_call(
            db.update_persona_memory_entry,
            entry_id=connection_id,
            persona_id=persona_id,
            user_id=user_id,
            update_data={"content": json.dumps(updated_content)},
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Persona connection not found")
        responses = await _get_persona_connections_by_id(db, user_id=user_id, persona_id=persona_id)
        response = responses.get(connection_id)
        if response is None:
            raise HTTPException(status_code=500, detail="Failed to load updated persona connection")
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="update persona connection") from exc


@router.delete(
    "/profiles/{persona_id}/connections/{connection_id}",
    response_model=PersonaConnectionDeleteResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def delete_persona_connection(
    persona_id: str,
    connection_id: str,
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaConnectionDeleteResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        deleted = await _run_persona_db_call(
            db.soft_delete_persona_memory_entry,
            entry_id=connection_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Persona connection not found")
        return PersonaConnectionDeleteResponse(
            status="deleted",
            persona_id=persona_id,
            connection_id=connection_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="delete persona connection") from exc


@router.post(
    "/profiles/{persona_id}/connections/{connection_id}/test",
    response_model=PersonaConnectionTestResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def test_persona_connection(
    persona_id: str,
    connection_id: str,
    payload: PersonaConnectionTestRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaConnectionTestResponse:
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        await _get_persona_profile_or_404(db, persona_id=persona_id, user_id=user_id, include_deleted=False)
        row = await _get_persona_connection_row_or_404(
            db,
            user_id=user_id,
            persona_id=persona_id,
            connection_id=connection_id,
        )
        connection = connection_content_from_row(row)
        connection["id"] = connection_id
        return await _test_persona_connection(connection_id, connection, payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_exception(exc, action="test persona connection") from exc


@router.get("/catalog", response_model=list[PersonaInfo], tags=["persona"], status_code=status.HTTP_200_OK)
async def persona_catalog(
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaInfo]:
    """Return persona catalog backed by ChaCha profile records."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        profiles = db.list_persona_profiles(user_id=user_id, active_only=True, limit=200)
        if not profiles:
            profiles = [_ensure_default_persona_profile(db, user_id=user_id)]
        buddy_rows = _load_persona_buddy_rows_for_projection(db, profiles=profiles)
        catalog: list[PersonaInfo] = []
        for profile in profiles:
            policy_rules = db.list_persona_policy_rules(
                persona_id=str(profile.get("id") or ""),
                user_id=user_id,
                include_deleted=False,
            )
            catalog.append(
                _persona_info_from_profile(
                    profile,
                    policy_rules=policy_rules,
                    buddy_row=buddy_rows.get(str(profile.get("id") or "").strip()),
                )
            )
        return catalog
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona catalog") from exc


@router.post("/session", response_model=PersonaSessionResponse, tags=["persona"], status_code=status.HTTP_200_OK)
async def persona_session(
    req: PersonaSessionRequest = Body(...),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaSessionResponse:
    """Create or resume a persona session with a materialized scope snapshot."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    requested_persona_id = str(req.persona_id or "").strip() or _DEFAULT_PERSONA_ID
    requested_activity_surface = normalize_persona_activity_surface(req.surface)
    session_manager = get_session_manager()

    try:
        profile = db.get_persona_profile(requested_persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            logger.info(
                "Unknown persona_id requested in API: {}; defaulting to {}",
                requested_persona_id,
                _DEFAULT_PERSONA_ID,
            )
            profile = _ensure_default_persona_profile(db, user_id=user_id)
        persona_id = str(profile.get("id") or _DEFAULT_PERSONA_ID)
        policy_rules = db.list_persona_policy_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
        persona = _persona_info_from_profile(
            profile,
            policy_rules=policy_rules,
            buddy_row=_load_persona_buddy_row_for_projection(db, profile=profile),
        )

        # Preserve scaffold ownership/persona binding semantics for resume IDs in process-local session manager
        # without creating new local entries before DB validation succeeds.
        if req.resume_session_id:
            local_session = session_manager.get(req.resume_session_id)
            if local_session is not None:
                if str(local_session.user_id) != user_id:
                    raise HTTPException(status_code=403, detail="session ownership mismatch")
                if str(local_session.persona_id) != persona_id:
                    raise HTTPException(status_code=409, detail="session persona mismatch")

        session_row: dict[str, Any] | None = None
        if req.resume_session_id:
            session_row = db.get_persona_session(req.resume_session_id, user_id=user_id, include_deleted=False)
            if session_row is not None:
                bound_persona_id = str(session_row.get("persona_id") or "").strip()
                if bound_persona_id and bound_persona_id != persona_id:
                    raise ConflictError(
                        "resume_session_id is bound to a different persona_id.",
                        entity="persona_sessions",
                        entity_id=str(req.resume_session_id),
                    )

        created_new_session = session_row is None
        if session_row is None:
            scope_rules = db.list_persona_scope_rules(persona_id=persona_id, user_id=user_id, include_deleted=False)
            scope_snapshot, scope_audit = _build_scope_snapshot(scope_rules)
            create_data: dict[str, Any] = {
                "persona_id": persona_id,
                "user_id": user_id,
                "conversation_id": req.project_id,
                "mode": str(profile.get("mode") or "session_scoped"),
                "scope_snapshot_json": scope_snapshot,
                "preferences_json": _default_persisted_persona_session_preferences(profile),
                "activity_surface": requested_activity_surface,
            }
            if req.resume_session_id:
                create_data["id"] = str(req.resume_session_id)
            session_id = db.create_persona_session(create_data)
            session_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False)
            if session_row is None:
                raise HTTPException(status_code=500, detail="Failed to load created persona session")
        else:
            scope_audit = _scope_audit_from_snapshot(session_row.get("scope_snapshot") or {})
            if req.surface is not None:
                current_surface = normalize_persona_activity_surface(session_row.get("activity_surface"))
                if current_surface != requested_activity_surface:
                    _ = db.update_persona_session(
                        session_id=str(session_row.get("id") or req.resume_session_id or ""),
                        user_id=user_id,
                        update_data={"activity_surface": requested_activity_surface},
                    )
                    refreshed_row = db.get_persona_session(
                        str(session_row.get("id") or req.resume_session_id or ""),
                        user_id=user_id,
                        include_deleted=False,
                    )
                    if refreshed_row is not None:
                        session_row = refreshed_row

        session_id = str(session_row.get("id") or req.resume_session_id or "").strip()
        if not session_id:
            raise HTTPException(status_code=500, detail="Persona session missing session_id")

        # Keep in-memory state synchronized for existing WS/tool-plan behavior.
        try:
            _ = session_manager.create(
                user_id=user_id,
                persona_id=persona_id,
                resume_session_id=session_id,
            )
            if created_new_session or req.surface is not None:
                session_manager.update_preferences(
                    session_id=session_id,
                    user_id=user_id,
                    preferences={"companion_activity_surface": requested_activity_surface},
                )
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        _session_preferences, activity_surface = _get_session_preferences_with_activity_surface(
            session_manager=session_manager,
            session_id=session_id,
            user_id=user_id,
            persisted_preferences=session_row.get("preferences"),
            persisted_activity_surface=session_row.get("activity_surface"),
        )

        allow_export, allow_delete = _get_persona_rbac_flags()
        scopes = sorted(_get_persona_session_scopes(allow_export=allow_export, allow_delete=allow_delete))
        scope_snapshot = session_row.get("scope_snapshot") or {}
        response = PersonaSessionResponse(
            session_id=session_id,
            persona=persona,
            scopes=scopes,
            runtime_mode=str(session_row.get("mode") or profile.get("mode") or "session_scoped"),
            scope_snapshot_id=_scope_snapshot_id_from_snapshot(scope_snapshot),
            scope_audit=scope_audit,
        )
        if created_new_session:
            record_persona_session_started(
                user_id=_current_user.id,
                session_id=response.session_id,
                persona_id=response.persona.id,
                runtime_mode=response.runtime_mode,
                scope_snapshot_id=response.scope_snapshot_id,
                surface=activity_surface,
            )
        return response
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="create or resume persona session") from exc


@router.get("/sessions", response_model=list[PersonaSessionSummary], tags=["persona"], status_code=status.HTTP_200_OK)
async def persona_sessions(
    persona_id: str | None = Query(default=None),
    surface: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> list[PersonaSessionSummary]:
    """List persisted persona sessions for the authenticated user."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    manager = get_session_manager()
    try:
        rows = db.list_persona_sessions(
            user_id=user_id,
            persona_id=persona_id,
            activity_surface=surface,
            include_deleted=False,
            limit=limit,
            offset=0,
        )
        manager_rows = {
            str(item.get("session_id") or ""): item
            for item in manager.list_sessions(user_id=user_id, persona_id=persona_id, limit=max(limit, 200))
        }
        return [
            _persona_session_summary_from_db(
                row,
                manager_row=manager_rows.get(str(row.get("id") or "")),
            )
            for row in rows
        ]
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="list persona sessions") from exc


@router.get(
    "/sessions/{session_id}",
    response_model=PersonaSessionDetail,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
)
async def persona_session_detail(
    session_id: str,
    limit_turns: int = Query(default=100, ge=0, le=1000),
    _current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> PersonaSessionDetail:
    """Get a single persisted persona session with local runtime turn snapshot (if present)."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    manager = get_session_manager()
    try:
        row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False)
        if row is None:
            raise HTTPException(status_code=404, detail="Persona session not found")
        snapshot = manager.get_session_snapshot(
            session_id=session_id,
            user_id=user_id,
            limit_turns=None if limit_turns <= 0 else limit_turns,
        )
        return _persona_session_detail_from_db(row, manager_snapshot=snapshot)
    except HTTPException:
        raise
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="get persona session detail") from exc


@router.websocket("/stream")
async def persona_stream(
    ws: WebSocket,
    token: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
):
    """
    Bi-directional placeholder stream.

    Standardized with WebSocketStream lifecycle/metrics; domain payloads unchanged.
    Accepts JSON text frames and echoes minimal notices.

    Security model:
    - Feature-gated via PERSONA_ENABLED.
    - Supports token/api-key auth resolution similar to MCP.
    - Tool execution requires an authenticated user_id.
    - Connections must authenticate before the stream is accepted.
    """
    if not is_persona_enabled():
        with contextlib.suppress(RuntimeError, OSError):
            await ws.accept()
            await ws.send_json(
                {
                    "event": "notice",
                    "session_id": "system",
                    "timestamp_ms": int(time.time() * 1000),
                    "event_seq": 0,
                    "level": "error",
                    "reason_code": "PERSONA_DISABLED",
                    "message": "Persona disabled",
                }
            )
            await ws.close(code=1000)
        return

    stream: WebSocketStream | None = None
    persona_scope_db: CharactersRAGDB | None = None
    auth_watchdog_task: asyncio.Task | None = None
    auth_watchdog_stop: asyncio.Event | None = None
    auth_revoked_event: asyncio.Event | None = None
    persona_live_stt_state_by_session: dict[str, dict[str, Any]] = {}
    persona_live_summary_sessions_seen: set[str] = set()
    try:
        user_id, credentials_supplied, auth_ok = await _resolve_authenticated_user_id(ws, token=token, api_key=api_key)
        if not auth_ok:
            auth_message = "Authentication failed" if credentials_supplied else "Authentication required"
            logger.info(f"persona stream rejected: {auth_message}")
            with contextlib.suppress(RuntimeError, OSError):
                await ws.close(code=1008)
            return

        # Wrap socket for lifecycle and metrics; keep domain payloads unchanged
        stream = WebSocketStream(
            ws,
            heartbeat_interval_s=0.0,  # disable WS pings for this scaffold
            idle_timeout_s=None,
            close_on_done=False,
            labels={"component": "persona", "endpoint": "persona_ws"},
        )
        await stream.start()
        authenticated_user_id = str(user_id or "").strip()
        if not authenticated_user_id:
            with contextlib.suppress(RuntimeError, OSError):
                await stream.ws.close(code=1008)
            return
        auth_watchdog_stop = asyncio.Event()
        auth_revoked_event = asyncio.Event()

        async def _is_stream_auth_valid() -> bool:
            try:
                revalidated_user_id, _credentials_supplied, revalidated_ok = await _resolve_authenticated_user_id(
                    ws,
                    token=token,
                    api_key=api_key,
                )
            except Exception as exc:
                logger.debug("persona stream auth revalidation failed with exception: {}", exc)
                return False
            if not revalidated_ok:
                return False
            return str(revalidated_user_id or "").strip() == authenticated_user_id

        async def _close_for_auth_revocation() -> None:
            if auth_revoked_event is not None and auth_revoked_event.is_set():
                return
            if auth_revoked_event is not None:
                auth_revoked_event.set()
            _increment_persona_metric(
                "persona_ws_auth_revalidation_total",
                {"result": "revoked"},
            )
            with contextlib.suppress(RuntimeError, OSError):
                await stream.ws.close(code=1008)

        auth_revalidate_interval_s = _get_persona_ws_auth_revalidate_interval_s()
        if auth_revalidate_interval_s > 0:

            async def _auth_revalidation_watchdog() -> None:
                if auth_watchdog_stop is None:
                    return
                while not auth_watchdog_stop.is_set():
                    try:
                        await asyncio.wait_for(
                            auth_watchdog_stop.wait(),
                            timeout=auth_revalidate_interval_s,
                        )
                        return
                    except asyncio.TimeoutError:
                        pass
                    if not await _is_stream_auth_valid():
                        await _close_for_auth_revocation()
                        return
                    _increment_persona_metric(
                        "persona_ws_auth_revalidation_total",
                        {"result": "ok"},
                    )

            auth_watchdog_task = asyncio.create_task(_auth_revalidation_watchdog())

        persona_scope_db = _open_persona_ws_db(authenticated_user_id)
        connection_user_id = authenticated_user_id
        session_manager = get_session_manager()
        default_session_id = uuid.uuid4().hex
        persona_id = "research_assistant"
        ws_event_seq_by_session: dict[str, int] = defaultdict(int)

        def _api_key_scope_allows(required_scope: Any) -> bool:
            try:
                auth_method = str(getattr(ws.state, "persona_auth_method", "") or "").strip().lower()
            except Exception:
                auth_method = ""
            if auth_method != "api_key":
                return True
            required = str(required_scope or "").strip().lower()
            if not required:
                return True
            try:
                key_scopes = normalize_scope(getattr(ws.state, "persona_api_key_scopes", None))
            except Exception:
                key_scopes = set()
            return has_scope(key_scopes, required)

        def _next_ws_event_meta(session_id: str) -> dict[str, Any]:
            sid = _normalize_ws_identifier(session_id, fallback=default_session_id)
            seq = ws_event_seq_by_session[sid]
            ws_event_seq_by_session[sid] += 1
            return {
                "session_id": sid,
                "timestamp_ms": int(time.time() * 1000),
                "event_seq": int(seq),
            }

        async def _emit_notice(
            *,
            session_id: str,
            level: str = "info",
            message: str,
            reason_code: str | None = None,
            **extra: Any,
        ) -> None:
            payload: dict[str, Any] = {
                "event": "notice",
                **_next_ws_event_meta(session_id),
                "level": _bounded_label(level, allowed=_PERSONA_WS_REQUIRED_NOTICE_LEVELS, fallback="info"),
                "message": str(message or ""),
                "reason_code": str(reason_code) if reason_code else None,
            }
            payload.update(extra)
            await stream.send_json(payload)

        async def _emit_assistant_delta(
            *,
            session_id: str,
            text_delta: str,
            **extra: Any,
        ) -> None:
            _mark_persona_live_processing_progress(session_id)
            payload: dict[str, Any] = {
                "event": "assistant_delta",
                **_next_ws_event_meta(session_id),
                "text_delta": str(text_delta or ""),
            }
            payload.update(extra)
            await stream.send_json(payload)

        async def _emit_tool_plan(
            *,
            session_id: str,
            plan_id: str,
            steps: list[dict[str, Any]],
            memory: dict[str, Any],
            companion: dict[str, Any],
            persona_id_value: str,
        ) -> None:
            _mark_persona_live_processing_progress(session_id)
            payload: dict[str, Any] = {
                "event": "tool_plan",
                **_next_ws_event_meta(session_id),
                "plan_id": str(plan_id or ""),
                "steps": list(steps),
                "memory": dict(memory or {}),
                "companion": dict(companion or {}),
                "persona_id": str(persona_id_value or ""),
            }
            await stream.send_json(payload)

        async def _emit_tool_call(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool: str,
            args: dict[str, Any],
            policy: dict[str, Any],
            why: str | None = None,
        ) -> None:
            _mark_persona_live_processing_progress(session_id)
            payload: dict[str, Any] = {
                "event": "tool_call",
                **_next_ws_event_meta(session_id),
                "plan_id": str(plan_id or ""),
                "step_idx": int(step_idx),
                "step_type": _bounded_label(step_type, allowed=_PERSONA_WS_ALLOWED_STEP_TYPES, fallback="mcp_tool"),
                "tool": str(tool or ""),
                "args": dict(args or {}),
                "policy": dict(policy or {}),
                "why": str(why or ""),
            }
            await stream.send_json(payload)
            _schedule_persona_live_processing_notice(
                session_id=session_id,
                reason_code="VOICE_TOOL_EXECUTION_PROCESSING",
                message="Tool execution is still in progress.",
                tool=str(tool or ""),
                step_idx=int(step_idx),
                why=str(why or ""),
            )

        async def _emit_tool_result(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool: str,
            result: dict[str, Any],
        ) -> None:
            _mark_persona_live_processing_progress(session_id)
            payload: dict[str, Any] = {
                "event": "tool_result",
                **_next_ws_event_meta(session_id),
                "plan_id": str(plan_id or ""),
                "step_idx": int(step_idx),
                "step_type": _bounded_label(step_type, allowed=_PERSONA_WS_ALLOWED_STEP_TYPES, fallback="mcp_tool"),
                "tool": str(tool or ""),
                **dict(result or {}),
            }
            payload.setdefault("ok", False)
            payload.setdefault("output", None)
            payload.setdefault("result", payload.get("output"))
            payload.setdefault("reason_code", None)
            await stream.send_json(payload)

        def _cancel_persona_live_processing_notice(session_id: str) -> None:
            task = persona_live_processing_notice_tasks_by_session.pop(session_id, None)
            if task is None:
                return
            with contextlib.suppress(Exception):
                task.cancel()

        def _mark_persona_live_processing_progress(session_id: str) -> None:
            _cancel_persona_live_processing_notice(session_id)

        def _schedule_persona_live_processing_notice(
            session_id: str,
            *,
            reason_code: str = "VOICE_TURN_PROCESSING",
            message: str = "Still processing this voice turn.",
            **extra: Any,
        ) -> None:
            _cancel_persona_live_processing_notice(session_id)
            notice_extra = dict(extra or {})

            async def _emit_after_delay() -> None:
                current_task = asyncio.current_task()
                try:
                    await asyncio.sleep(_PERSONA_LIVE_PROCESSING_NOTICE_DELAY_S)
                    if (
                        persona_live_processing_notice_tasks_by_session.get(session_id)
                        is not current_task
                    ):
                        return
                    await _emit_notice(
                        session_id=session_id,
                        level="info",
                        reason_code=reason_code,
                        message=message,
                        **notice_extra,
                    )
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.debug(
                        "persona live processing notice skipped for session {}: {}",
                        session_id,
                        exc,
                    )
                finally:
                    if (
                        persona_live_processing_notice_tasks_by_session.get(session_id)
                        is current_task
                    ):
                        persona_live_processing_notice_tasks_by_session.pop(session_id, None)

            persona_live_processing_notice_tasks_by_session[session_id] = (
                asyncio.create_task(_emit_after_delay())
            )

        await _emit_notice(
            session_id=default_session_id,
            level="info",
            message="persona stream connected (scaffold)",
            reason_code="WS_CONNECTED",
        )

        allow_export, allow_delete = _get_persona_rbac_flags()
        session_scopes = _get_persona_session_scopes(
            allow_export=allow_export,
            allow_delete=allow_delete,
        )
        allowed_audio_formats = _get_persona_allowed_audio_formats()
        audio_chunk_max_bytes = _get_persona_audio_chunk_max_bytes()
        audio_chunks_per_minute = _get_persona_audio_chunks_per_minute()
        tts_chunk_size_bytes = _get_persona_tts_chunk_size_bytes()
        tts_max_chunks = _get_persona_tts_max_chunks()
        tts_max_total_bytes = _get_persona_tts_max_total_bytes()
        tts_max_in_flight_chunks = _get_persona_tts_max_in_flight_chunks()
        audio_rate_windows: dict[str, deque[float]] = defaultdict(deque)
        transcript_seq_by_session: dict[str, int] = defaultdict(int)
        voice_transcript_buffer_by_session: dict[str, str] = defaultdict(str)
        tts_seq_by_session: dict[str, int] = defaultdict(int)
        tts_in_flight_by_session: dict[str, int] = defaultdict(int)
        persona_live_processing_notice_tasks_by_session: dict[str, asyncio.Task[Any]] = {}

        async def _record_turn(
            *,
            session_id: str,
            role: str,
            content: str,
            turn_type: str,
            metadata: dict[str, Any] | None = None,
            persist_as_memory: bool = False,
            persist_personalization: bool = True,
            persona_id_override: str | None = None,
            runtime_mode_override: str | None = None,
            scope_snapshot_id_override: str | None = None,
            memory_kind: str | None = None,
        ) -> None:
            effective_persona_id = str(persona_id_override or persona_id or "").strip() or persona_id
            effective_runtime_mode = str(runtime_mode_override or "session_scoped").strip().lower()
            effective_scope_snapshot_id = str(scope_snapshot_id_override or "").strip() or None
            should_store_memory = bool(
                persist_as_memory
                and effective_runtime_mode == "persistent_scoped"
                and str(memory_kind or "").strip().lower() == "summary"
            )
            try:
                session_manager.append_turn(
                    session_id=session_id,
                    user_id=connection_user_id,
                    persona_id=effective_persona_id,
                    role=role,
                    content=content,
                    turn_type=turn_type,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.debug(f"persona turn append skipped: {exc}")
            if persist_personalization:
                _ = await asyncio.to_thread(
                    persist_persona_turn,
                    user_id=authenticated_user_id,
                    session_id=session_id,
                    persona_id=effective_persona_id,
                    role=role,
                    content=content,
                    turn_type=turn_type,
                    metadata=metadata,
                    store_as_memory=should_store_memory,
                    runtime_mode=effective_runtime_mode,
                    scope_snapshot_id=effective_scope_snapshot_id,
                )

        async def _emit_persona_live_tts_for_assistant_text(
            *,
            session_id: str,
            assistant_text: str,
        ) -> None:
            preferences = session_manager.get_preferences(
                session_id=session_id,
                user_id=connection_user_id,
            )
            if str(preferences.get("last_turn_type") or "").strip().lower() != "voice_commit":
                return
            voice_runtime = preferences.get("voice_runtime")
            if not isinstance(voice_runtime, dict):
                return
            if bool(voice_runtime.get("text_only_due_to_tts_failure")):
                return

            provider = str(voice_runtime.get("tts_provider") or "").strip().lower()
            voice = str(voice_runtime.get("tts_voice") or "").strip() or None
            if not provider or provider == "browser":
                return

            try:
                audio_bytes, audio_format = await _generate_persona_live_tts_audio(
                    assistant_text,
                    provider=provider,
                    voice=voice,
                    response_format="mp3",
                )
                if not audio_bytes:
                    raise RuntimeError("TTS returned no audio")
            except Exception as exc:
                updated_voice_runtime = dict(voice_runtime)
                updated_voice_runtime["text_only_due_to_tts_failure"] = True
                with contextlib.suppress(Exception):
                    session_manager.update_preferences(
                        session_id=session_id,
                        user_id=connection_user_id,
                        preferences={"voice_runtime": updated_voice_runtime},
                    )
                logger.debug(f"persona live TTS unavailable for session {session_id}: {exc}")
                _mark_persona_live_processing_progress(session_id)
                _upsert_persona_live_session_summary_safe(
                    session_id=session_id,
                    voice_runtime=updated_voice_runtime,
                    text_only_tts_increment=1,
                )
                await _emit_notice(
                    session_id=session_id,
                    level="warning",
                    reason_code="TTS_UNAVAILABLE_TEXT_ONLY",
                    message="Live TTS unavailable for this session. Continuing in text-only mode.",
                )
                return

            tts_chunks = _chunk_persona_audio_bytes(
                audio_bytes,
                chunk_size_bytes=tts_chunk_size_bytes,
                max_chunks=tts_max_chunks,
                max_total_bytes=tts_max_total_bytes,
            )
            total_chunks = len(tts_chunks)
            for idx, chunk in enumerate(tts_chunks):
                if tts_in_flight_by_session[session_id] >= tts_max_in_flight_chunks:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        reason_code="TTS_BACKPRESSURE_DROP",
                        message=f"Dropping TTS chunk due to in-flight limit ({tts_max_in_flight_chunks})",
                    )
                    break

                chunk_seq = tts_seq_by_session[session_id]
                tts_seq_by_session[session_id] += 1
                chunk_id = uuid.uuid4().hex
                tts_in_flight_by_session[session_id] += 1
                await stream.send_json(
                    {
                        "event": "tts_audio",
                        "session_id": session_id,
                        "audio_format": str(audio_format or "mp3"),
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "chunk_count": total_chunks,
                        "seq": chunk_seq,
                        "timestamp_ms": int(time.time() * 1000),
                    }
                )
                try:
                    await stream.ws.send_bytes(chunk)
                except Exception as exc:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        message=f"Failed to send tts audio binary chunk: {exc}",
                        reason_code="TTS_SEND_FAILED",
                    )
                    tts_in_flight_by_session[session_id] = max(
                        0, tts_in_flight_by_session[session_id] - 1
                    )
                    break
                tts_in_flight_by_session[session_id] = max(
                    0, tts_in_flight_by_session[session_id] - 1
                )

        def _cleanup_persona_live_stt_state(session_id: str) -> None:
            _cancel_persona_live_processing_notice(session_id)
            state = persona_live_stt_state_by_session.pop(session_id, None)
            transcriber = state.get("transcriber") if isinstance(state, dict) else None
            turn_detector = state.get("turn_detector") if isinstance(state, dict) else None
            if transcriber is not None:
                with contextlib.suppress(Exception):
                    transcriber.cleanup()
            if turn_detector is not None:
                with contextlib.suppress(Exception):
                    turn_detector.reset()
            voice_transcript_buffer_by_session.pop(session_id, None)

        def _clear_persona_live_commit_state(state: dict[str, Any]) -> None:
            state["current_utterance_committed"] = False
            state["current_commit_source"] = None
            state["committed_transcript"] = ""

        def _reset_persona_live_active_turn(session_id: str) -> None:
            state = persona_live_stt_state_by_session.get(session_id) or {}
            transcriber = state.get("transcriber")
            turn_detector = state.get("turn_detector")
            if transcriber is not None:
                with contextlib.suppress(Exception):
                    transcriber.reset()
            if turn_detector is not None:
                with contextlib.suppress(Exception):
                    turn_detector.reset()
            voice_transcript_buffer_by_session.pop(session_id, None)

        def _get_or_create_persona_live_stt_state(
            session_id: str,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            preferences = session_manager.get_preferences(
                session_id=session_id,
                user_id=connection_user_id,
            )
            voice_runtime = preferences.get("voice_runtime")
            if not isinstance(voice_runtime, dict):
                return None, None

            existing_state = persona_live_stt_state_by_session.get(session_id)
            if isinstance(existing_state, dict):
                existing_state["voice_runtime"] = voice_runtime
                return existing_state, voice_runtime

            try:
                transcriber = _create_persona_live_stt_transcriber(
                    voice_runtime=voice_runtime,
                )
                transcriber.initialize()
            except Exception as exc:
                logger.debug(
                    "persona live STT initialization failed for session {}: {}",
                    session_id,
                    exc,
                )
                return None, voice_runtime

            turn_detector = None
            manual_mode_reason: str | None = None
            try:
                turn_detector = _create_persona_live_turn_detector(
                    voice_runtime=voice_runtime,
                )
                if turn_detector is not None and not bool(getattr(turn_detector, "available", False)):
                    manual_mode_reason = str(
                        getattr(turn_detector, "unavailable_reason", "") or "vad_unavailable"
                    )
            except Exception as exc:
                logger.debug(
                    "persona live VAD initialization failed for session {}: {}",
                    session_id,
                    exc,
                )
                turn_detector = None
                manual_mode_reason = str(exc)

            state = {
                "transcriber": transcriber,
                "turn_detector": turn_detector,
                "voice_runtime": voice_runtime,
                "current_utterance_committed": False,
                "current_commit_source": None,
                "committed_transcript": "",
                "manual_mode_notice_sent": False,
                "manual_mode_reason": manual_mode_reason,
            }
            persona_live_stt_state_by_session[session_id] = state
            return state, voice_runtime

        def _current_persona_live_transcript(session_id: str) -> str:
            buffered = str(voice_transcript_buffer_by_session.get(session_id) or "").strip()
            if buffered:
                return buffered
            state = persona_live_stt_state_by_session.get(session_id) or {}
            transcriber = state.get("transcriber")
            if transcriber is None:
                return ""
            with contextlib.suppress(Exception):
                return str(transcriber.get_full_transcript() or "").strip()
            return ""

        def _resolve_persona_live_event_persona_id(session_id: str) -> str | None:
            if persona_scope_db is None or not authenticated_user_id:
                return None
            try:
                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
            except Exception as exc:
                logger.debug(
                    "persona live analytics persona resolution failed for session {}: {}",
                    session_id,
                    exc,
                )
                return None
            normalized_persona_id = str(runtime_context.get("persona_id") or "").strip()
            return normalized_persona_id or None

        def _persona_live_session_summary_snapshot(
            voice_runtime: dict[str, Any] | None,
        ) -> dict[str, Any]:
            runtime = dict(voice_runtime or {})
            return {
                "auto_commit_enabled": _coerce_bool(runtime.get("enable_vad"), default=True),
                "vad_threshold": _clamp_persona_live_float(
                    runtime.get("vad_threshold"),
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                "min_silence_ms": _clamp_persona_live_int(
                    runtime.get("vad_min_silence_ms"),
                    default=250,
                    min_value=50,
                    max_value=10_000,
                ),
                "turn_stop_secs": _clamp_persona_live_float(
                    runtime.get("vad_turn_stop_secs"),
                    default=0.2,
                    min_value=0.05,
                    max_value=10.0,
                ),
                "min_utterance_secs": _clamp_persona_live_float(
                    runtime.get("vad_min_utterance_secs"),
                    default=0.4,
                    min_value=0.0,
                    max_value=10.0,
                ),
            }

        def _upsert_persona_live_session_summary_safe(
            *,
            session_id: str,
            voice_runtime: dict[str, Any] | None = None,
            commit_source: str | None = None,
            manual_mode_required_increment: int = 0,
            text_only_tts_increment: int = 0,
            finalize: bool = False,
        ) -> None:
            if persona_scope_db is None or not authenticated_user_id:
                return
            try:
                user_id_int = int(str(authenticated_user_id))
            except (TypeError, ValueError):
                return
            persona_id_value = _resolve_persona_live_event_persona_id(session_id)
            if not persona_id_value:
                return
            snapshot = (
                _persona_live_session_summary_snapshot(voice_runtime)
                if isinstance(voice_runtime, dict)
                else {}
            )
            persona_live_summary_sessions_seen.add(session_id)
            try:
                persona_scope_db.upsert_persona_live_voice_session_summary(
                    user_id=user_id_int,
                    persona_id=persona_id_value,
                    session_id=session_id,
                    auto_commit_enabled=snapshot.get("auto_commit_enabled"),
                    vad_threshold=snapshot.get("vad_threshold"),
                    min_silence_ms=snapshot.get("min_silence_ms"),
                    turn_stop_secs=snapshot.get("turn_stop_secs"),
                    min_utterance_secs=snapshot.get("min_utterance_secs"),
                    commit_source=commit_source,
                    manual_mode_required_increment=manual_mode_required_increment,
                    text_only_tts_increment=text_only_tts_increment,
                    finalize=finalize,
                )
            except Exception as exc:
                logger.debug(
                    "persona live session summary update skipped for session {}: {}",
                    session_id,
                    exc,
                )

        def _record_persona_live_voice_event_safe(
            *,
            session_id: str,
            event_type: str,
            commit_source: str | None = None,
        ) -> None:
            if persona_scope_db is None or not authenticated_user_id:
                return
            try:
                user_id_int = int(str(authenticated_user_id))
            except (TypeError, ValueError):
                return
            persona_id_value = _resolve_persona_live_event_persona_id(session_id)
            if not persona_id_value:
                return
            try:
                record_persona_live_voice_event(
                    persona_scope_db,
                    user_id=user_id_int,
                    persona_id=persona_id_value,
                    session_id=session_id,
                    event_type=event_type,
                    commit_source=commit_source,
                )
            except Exception as exc:
                logger.debug(
                    "persona live analytics event skipped for session {}: {}",
                    session_id,
                    exc,
                )

        async def _ensure_persona_live_manual_mode_notice(
            session_id: str,
            state: dict[str, Any],
        ) -> None:
            if bool(state.get("manual_mode_notice_sent")):
                return
            state["manual_mode_notice_sent"] = True
            await _emit_notice(
                session_id=session_id,
                level="warning",
                reason_code="VOICE_MANUAL_MODE_REQUIRED",
                message="Server VAD unavailable for this live session. Use Send now to commit heard speech manually.",
                details=str(state.get("manual_mode_reason") or "vad_unavailable"),
            )
            _record_persona_live_voice_event_safe(
                session_id=session_id,
                event_type="manual_mode_required",
            )
            _upsert_persona_live_session_summary_safe(
                session_id=session_id,
                voice_runtime=state.get("voice_runtime") if isinstance(state, dict) else None,
                manual_mode_required_increment=1,
            )

        async def _commit_persona_live_turn(
            *,
            session_id: str,
            transcript: str,
            commit_source: str,
            source: str,
        ) -> bool:
            state = persona_live_stt_state_by_session.get(session_id) or {}
            if bool(state.get("current_utterance_committed")):
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="VOICE_COMMIT_IGNORED_ALREADY_COMMITTED",
                    message="This utterance was already committed.",
                    commit_source=str(state.get("current_commit_source") or commit_source),
                    transcript=str(state.get("committed_transcript") or "").strip() or None,
                )
                return False

            voice_runtime = state.get("voice_runtime")
            trigger_phrases = (
                list(voice_runtime.get("trigger_phrases") or [])
                if isinstance(voice_runtime, dict)
                else []
            )
            trigger_matched, cleaned_transcript = _apply_persona_live_trigger_phrases(
                transcript,
                trigger_phrases=trigger_phrases,
            )
            if not trigger_matched:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="VOICE_TRIGGER_NOT_HEARD",
                    message="No trigger phrase was heard, so the transcript was ignored.",
                )
                _reset_persona_live_active_turn(session_id)
                return False
            if not cleaned_transcript:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="VOICE_EMPTY_COMMAND_AFTER_TRIGGER",
                    message="The trigger phrase was removed, but no spoken command remained.",
                )
                _reset_persona_live_active_turn(session_id)
                return False

            state["current_utterance_committed"] = True
            state["current_commit_source"] = commit_source
            state["committed_transcript"] = cleaned_transcript
            await _emit_notice(
                session_id=session_id,
                level="info",
                reason_code="VOICE_TURN_COMMITTED",
                message="Voice turn committed.",
                commit_source=commit_source,
                transcript=cleaned_transcript,
            )
            _record_persona_live_voice_event_safe(
                session_id=session_id,
                event_type="commit",
                commit_source=commit_source,
            )
            _upsert_persona_live_session_summary_safe(
                session_id=session_id,
                voice_runtime=voice_runtime if isinstance(voice_runtime, dict) else None,
                commit_source=commit_source,
            )
            _reset_persona_live_active_turn(session_id)
            _schedule_persona_live_processing_notice(session_id)
            await _handle_persona_live_turn(
                msg={
                    "session_id": session_id,
                    "transcript": cleaned_transcript,
                    "source": source,
                    "commit_source": commit_source,
                },
                text=cleaned_transcript,
                turn_type="voice_commit",
                source=source,
            )
            return True

        def _pending_retry_key(*, plan_id: str, step_idx: int, tool_name: str) -> str:
            """Return the stable storage key for a pending approval-backed retry."""
            return f"{str(plan_id or '').strip()}|{int(step_idx)}|{str(tool_name or '').strip()}"

        def _load_pending_retry_approvals(session_id: str) -> dict[str, dict[str, Any]]:
            """Load sanitized pending retry approvals from session preferences."""
            preferences = session_manager.get_preferences(
                session_id=session_id,
                user_id=connection_user_id,
            )
            raw = preferences.get("pending_retry_approvals")
            if not isinstance(raw, dict):
                return {}
            out: dict[str, dict[str, Any]] = {}
            for key, value in raw.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                out[key] = dict(value)
            return out

        def _store_pending_retry_approval(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool_name: str,
            args: dict[str, Any],
            why: str | None,
            description: str | None,
        ) -> None:
            """Persist approval retry state so the client cannot tamper with retried args."""
            retries = _load_pending_retry_approvals(session_id)
            key = _pending_retry_key(plan_id=plan_id, step_idx=step_idx, tool_name=tool_name)
            retries[key] = {
                "plan_id": str(plan_id or ""),
                "step_idx": int(step_idx),
                "step_type": _bounded_label(
                    step_type,
                    allowed=_PERSONA_WS_ALLOWED_STEP_TYPES,
                    fallback="mcp_tool",
                ),
                "tool": str(tool_name or ""),
                "args": dict(args or {}),
                "why": str(why or ""),
                "description": str(description or ""),
            }
            session_manager.update_preferences(
                session_id=session_id,
                user_id=connection_user_id,
                preferences={"pending_retry_approvals": retries},
            )

        def _consume_pending_retry_approval(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            tool_name: str,
        ) -> dict[str, Any] | None:
            """Remove and return a stored retry approval entry for one tool step."""
            retries = _load_pending_retry_approvals(session_id)
            key = _pending_retry_key(plan_id=plan_id, step_idx=step_idx, tool_name=tool_name)
            entry = retries.pop(key, None)
            session_manager.update_preferences(
                session_id=session_id,
                user_id=connection_user_id,
                preferences={"pending_retry_approvals": retries},
            )
            return dict(entry) if isinstance(entry, dict) else None

        async def _call_mcp_tool(
            name: str,
            arguments: dict,
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            policy: dict[str, Any],
            why: str | None = None,
            description: str | None = None,
            allowed_tools: list[str] | None = None,
        ) -> dict:
            if not authenticated_user_id:
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "mcp", "status": "auth_required"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error="Authentication required for tool execution",
                    reason_code="AUTH_REQUIRED",
                    policy=policy,
                )
            if not bool(policy.get("allow", False)):
                deny_reason = str(policy.get("reason") or f"Tool '{name}' not permitted by policy")
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "mcp", "status": "denied"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=deny_reason,
                    reason_code=str(policy.get("reason_code") or "POLICY_DENIED"),
                    policy=policy,
                )
            required_scope = str(policy.get("required_scope") or "").strip().lower()
            if not _api_key_scope_allows(required_scope):
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "mcp", "status": "api_key_scope_denied"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=f"API key missing required scope '{required_scope}'",
                    reason_code="API_KEY_SCOPE_MISSING",
                    policy=policy,
                )
            resolved_name, resolved_arguments = _translate_persona_tool_request(name, arguments or {})
            req = MCPRequest(method="tools/call", params={"name": resolved_name, "arguments": resolved_arguments})
            server = get_mcp_server()
            if not server.initialized:
                await server.initialize()
            audit_metadata = {
                "mcp_policy_context_enabled": is_mcp_hub_policy_enforcement_enabled(),
                "persona_id": runtime_persona_id,
                "session_id": session_id,
                "persona_audit": {
                    "source": "persona_ws",
                    "plan_id": plan_id,
                    "step_idx": step_idx,
                    "tool": name,
                    "mapped_tool": resolved_name,
                    "why": str(why or ""),
                    "description": str(description or ""),
                },
            }
            allowed_tools_effective = [str(t).strip() for t in (allowed_tools or []) if str(t).strip()]
            if resolved_name and resolved_name not in allowed_tools_effective:
                allowed_tools_effective.append(str(resolved_name))
            if allowed_tools_effective:
                audit_metadata["allowed_tools"] = allowed_tools_effective
            scope_metadata = _load_persona_scope_metadata_for_session(
                persona_scope_db,
                session_id=session_id,
                user_id=authenticated_user_id,
            )
            if scope_metadata:
                audit_metadata["persona_scope"] = scope_metadata
                _increment_persona_metric(
                    "persona_ws_scope_resolution_total",
                    {"source": "mcp_tool", "result": "hit"},
                )
            else:
                _increment_persona_metric(
                    "persona_ws_scope_resolution_total",
                    {"source": "mcp_tool", "result": "miss"},
                )
            resp = await server.handle_http_request(req, user_id=authenticated_user_id, metadata=audit_metadata)
            if resp.error:
                error_data = getattr(resp.error, "data", None)
                approval_payload = (
                    dict(error_data.get("approval") or {})
                    if isinstance(error_data, dict) and isinstance(error_data.get("approval"), dict)
                    else None
                )
                governance_payload = (
                    dict(error_data.get("governance") or {})
                    if isinstance(error_data, dict) and isinstance(error_data.get("governance"), dict)
                    else None
                )
                external_access_payload = (
                    dict(governance_payload.get("external_access") or {})
                    if isinstance(governance_payload, dict)
                    and isinstance(governance_payload.get("external_access"), dict)
                    else None
                )
                path_scope_payload = (
                    dict(governance_payload.get("path_scope") or {})
                    if isinstance(governance_payload, dict)
                    and isinstance(governance_payload.get("path_scope"), dict)
                    else None
                )
                reason_code = (
                    str(governance_payload.get("reason_code") or "").strip()
                    if isinstance(governance_payload, dict)
                    else ""
                )
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "mcp", "status": "error"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=resp.error.message,
                    reason_code=(
                        "APPROVAL_REQUIRED"
                        if approval_payload
                        else reason_code or "TOOL_EXECUTION_ERROR"
                    ),
                    policy=policy,
                    approval=approval_payload,
                    external_access=external_access_payload,
                    path_scope=path_scope_payload,
                )
            _increment_persona_metric(
                "persona_ws_tool_calls_total",
                {"kind": "mcp", "status": "success"},
            )
            return _build_tool_result(ok=True, output=resp.result, policy=policy)

        async def _call_skill(
            skill_name: str,
            skill_args: str,
            *,
            policy: dict[str, Any],
        ) -> dict[str, Any]:
            if not authenticated_user_id:
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "skill", "status": "auth_required"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error="Authentication required for skill execution",
                    reason_code="AUTH_REQUIRED",
                    policy=policy,
                )
            if not bool(policy.get("allow", False)):
                deny_reason = str(policy.get("reason") or f"Skill '{skill_name}' not permitted by policy")
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "skill", "status": "denied"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=deny_reason,
                    reason_code=str(policy.get("reason_code") or "POLICY_DENIED"),
                    policy=policy,
                )
            required_scope = str(policy.get("required_scope") or "").strip().lower()
            if not _api_key_scope_allows(required_scope):
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "skill", "status": "api_key_scope_denied"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=f"API key missing required scope '{required_scope}'",
                    reason_code="API_KEY_SCOPE_MISSING",
                    policy=policy,
                )
            try:
                uid_int = int(str(authenticated_user_id))
            except (TypeError, ValueError):
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error="Authenticated persona user must be numeric for skills execution",
                    reason_code="SKILL_USER_ID_INVALID",
                    policy=policy,
                )
            try:
                skill_result = await handle_skill_tool_call(
                    skill_name=skill_name,
                    args=skill_args,
                    user_id=uid_int,
                    base_path=DatabasePaths.get_user_base_directory(uid_int),
                    db=persona_scope_db,
                    request_context=None,
                )
            except Exception as exc:
                logger.debug("persona skill execution error for {}: {}", skill_name, exc)
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "skill", "status": "error"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=f"Skill execution failed: {exc}",
                    reason_code="SKILL_EXECUTION_ERROR",
                    policy=policy,
                )
            if not bool(skill_result.get("success")):
                _increment_persona_metric(
                    "persona_ws_tool_calls_total",
                    {"kind": "skill", "status": "error"},
                )
                return _build_tool_result(
                    ok=False,
                    output=None,
                    error=str(skill_result.get("error") or f"Skill '{skill_name}' failed"),
                    reason_code="SKILL_EXECUTION_ERROR",
                    policy=policy,
                )
            _increment_persona_metric(
                "persona_ws_tool_calls_total",
                {"kind": "skill", "status": "success"},
            )
            return _build_tool_result(ok=True, output=skill_result, policy=policy)

        async def _emit_and_persist_tool_step_result(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool_name: str,
            result: dict[str, Any],
            persona_id: str,
            runtime_mode_value: str,
            scope_snapshot_id: str | None,
        ) -> None:
            """Emit a tool result and persist its retention/audit side effects."""
            await _emit_tool_result(
                session_id=session_id,
                plan_id=plan_id,
                step_idx=step_idx,
                step_type=step_type,
                tool=tool_name,
                result=result,
            )
            await _record_turn(
                session_id=session_id,
                role="tool",
                content=_summarize_tool_result_for_retention(result),
                turn_type="tool_result",
                metadata={"tool": tool_name, "step_idx": step_idx, "step_type": step_type},
                persist_as_memory=False,
                persist_personalization=False,
                persona_id_override=persona_id,
                runtime_mode_override=runtime_mode_value,
                scope_snapshot_id_override=scope_snapshot_id,
            )
            _ = await asyncio.to_thread(
                persist_tool_outcome,
                user_id=authenticated_user_id,
                session_id=session_id,
                persona_id=persona_id,
                tool_name=tool_name,
                step_idx=step_idx,
                outcome=result,
                store_as_memory=False,
                runtime_mode=runtime_mode_value,
                scope_snapshot_id=scope_snapshot_id,
            )

        async def _emit_denied_tool_step(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool_name: str,
            step_policy: dict[str, Any],
            persona_id: str,
            runtime_mode_value: str,
            scope_snapshot_id: str | None,
        ) -> dict[str, Any]:
            """Emit, persist, and return a denied tool step result."""
            deny_reason = str(step_policy.get("reason") or f"Tool '{tool_name}' not permitted by policy")
            reason_code = str(step_policy.get("reason_code") or "POLICY_DENIED")
            _increment_persona_metric(
                "persona_ws_policy_denials_total",
                {
                    "step_type": _bounded_label(step_type, allowed=_PERSONA_WS_ALLOWED_STEP_TYPES, fallback="mcp_tool"),
                    "reason": _metric_reason_bucket(reason_code),
                },
            )
            await _emit_notice(
                session_id=session_id,
                step_idx=step_idx,
                tool=tool_name,
                step_type=step_type,
                level="warning",
                reason_code=reason_code,
                message=deny_reason,
            )
            result = _build_tool_result(
                ok=False,
                output=None,
                error=deny_reason,
                reason_code=reason_code,
                policy=step_policy,
            )
            await _emit_and_persist_tool_step_result(
                session_id=session_id,
                plan_id=plan_id,
                step_idx=step_idx,
                step_type=step_type,
                tool_name=tool_name,
                result=result,
                persona_id=persona_id,
                runtime_mode_value=runtime_mode_value,
                scope_snapshot_id=scope_snapshot_id,
            )
            return result

        async def _execute_persona_tool_step(
            *,
            session_id: str,
            plan_id: str,
            step_idx: int,
            step_type: str,
            tool_name: str,
            step_args: dict[str, Any],
            step_policy: dict[str, Any],
            why: str | None,
            description: str | None,
            persona_id: str,
            runtime_mode_value: str,
            scope_snapshot_id: str | None,
        ) -> dict[str, Any]:
            """Execute, emit, and persist one non-final persona tool step."""
            await _emit_tool_call(
                session_id=session_id,
                plan_id=plan_id,
                step_idx=step_idx,
                step_type=step_type,
                tool=tool_name,
                args=step_args,
                why=why,
                policy=step_policy,
            )
            if step_type == "skill":
                result = await _call_skill(
                    tool_name,
                    str(step_args.get("args") or ""),
                    policy=step_policy,
                )
            else:
                result = await _call_mcp_tool(
                    tool_name,
                    step_args,
                    session_id=session_id,
                    plan_id=plan_id,
                    step_idx=step_idx,
                    policy=step_policy,
                    why=why,
                    description=description,
                    allowed_tools=step_policy.get("effective_allowed_tools")
                    if isinstance(step_policy.get("effective_allowed_tools"), list)
                    else None,
                )
            if isinstance(result.get("approval"), dict):
                _store_pending_retry_approval(
                    session_id=session_id,
                    plan_id=plan_id,
                    step_idx=step_idx,
                    step_type=step_type,
                    tool_name=tool_name,
                    args=step_args,
                    why=why,
                    description=description,
                )
            await _emit_and_persist_tool_step_result(
                session_id=session_id,
                plan_id=plan_id,
                step_idx=step_idx,
                step_type=step_type,
                tool_name=tool_name,
                result=result,
                persona_id=persona_id,
                runtime_mode_value=runtime_mode_value,
                scope_snapshot_id=scope_snapshot_id,
            )
            return result

        async def _propose_plan(
            text: str,
            memory_context: list[str] | None = None,
            persona_state_hints: dict[str, str] | None = None,
            companion_context: dict[str, Any] | None = None,
            persona_exemplar_sections: list[tuple[str, str, int]] | None = None,
        ) -> dict:
            steps = []
            text_clean = str(text or "").strip()
            t = text_clean.lower()
            compact_state_hints = {
                str(k).strip().lower(): str(v).strip()
                for k, v in dict(persona_state_hints or {}).items()
                if str(k).strip() and str(v).strip()
            }
            if t.startswith("skill:"):
                payload = text_clean.split(":", 1)[1].strip()
                if payload:
                    skill_name, _, skill_args = payload.partition(" ")
                    steps.append(
                        {
                            "idx": 0,
                            "step_type": "skill",
                            "tool": skill_name.strip().lower(),
                            "args": {"args": skill_args.strip()},
                            "description": f"Execute skill '{skill_name.strip()}'",
                            "why": "Input explicitly requested skill execution.",
                        }
                    )
                    return {"steps": steps}
            if _persona_identity_query_requested(t) and compact_state_hints:
                state_answer = _build_persona_state_identity_answer(compact_state_hints)
                steps.append(
                    {
                        "idx": 0,
                        "step_type": "final_answer",
                        "tool": "summarize",
                        "args": {"text": state_answer},
                        "description": "Answer using persistent persona state docs",
                        "why": "Identity/personality question with persistent persona state available.",
                    }
                )
                return {"steps": steps}
            if "http" in t or "ingest" in t or "url" in t:
                steps.append(
                    {
                        "idx": 0,
                        "step_type": "mcp_tool",
                        "tool": "ingest_url",
                        "args": {"url": text},
                        "description": "Ingest the provided URL",
                        "why": "Input looks like a URL or ingestion request.",
                    }
                )
                steps.append(
                    {
                        "idx": 1,
                        "step_type": "final_answer",
                        "tool": "summarize",
                        "args": {},
                        "description": "Summarize the ingested content",
                        "why": "User likely wants a concise summary after ingestion.",
                    }
                )
            else:
                query_text = text
                compact_memories = [m.strip() for m in (memory_context or []) if str(m or "").strip()]
                compact_state_lines = _build_persona_state_hint_lines(compact_state_hints)
                companion_payload = dict(companion_context or {})
                companion_knowledge_lines = [
                    str(line).strip()
                    for line in companion_payload.get("knowledge_lines", [])
                    if str(line or "").strip()
                ]
                companion_goal_lines = [
                    str(line).strip()
                    for line in companion_payload.get("goal_lines", [])
                    if str(line or "").strip()
                ]
                companion_activity_lines = [
                    str(line).strip()
                    for line in companion_payload.get("activity_lines", [])
                    if str(line or "").strip()
                ]
                if compact_state_lines:
                    query_text = f"{query_text}\n\nPersistent persona state hints:\n" + "\n".join(compact_state_lines)
                if compact_memories:
                    memory_lines = "\n".join(f"- {m}" for m in compact_memories[: _get_persona_memory_top_k()])
                    query_text = f"{query_text}\n\nPersona memory hints:\n{memory_lines}"
                if companion_knowledge_lines:
                    query_text = (
                        f"{query_text}\n\nCompanion knowledge:\n"
                        + "\n".join(companion_knowledge_lines)
                    )
                if companion_goal_lines:
                    query_text = (
                        f"{query_text}\n\nActive companion goals:\n"
                        + "\n".join(companion_goal_lines)
                    )
                if companion_activity_lines:
                    query_text = (
                        f"{query_text}\n\nRecent explicit companion activity:\n"
                        + "\n".join(companion_activity_lines)
                    )
                applied_context_labels: list[str] = []
                if compact_state_lines:
                    applied_context_labels.append("persistent persona state")
                if compact_memories:
                    applied_context_labels.append("personalization memories")
                if companion_knowledge_lines or companion_goal_lines or companion_activity_lines:
                    applied_context_labels.append("companion context")
                query_text = append_persona_exemplar_sections(query_text, persona_exemplar_sections)
                has_exemplar_guidance = any(
                    str(content or "").strip()
                    for _, content, _ in list(persona_exemplar_sections or [])
                )
                if has_exemplar_guidance:
                    applied_context_labels.append("persona exemplar guidance")
                if applied_context_labels:
                    why_text = (
                        "Input appears to be a knowledge query with applied "
                        f"{_join_applied_context_labels(applied_context_labels)}."
                    )
                else:
                    why_text = "Input appears to be a knowledge query."
                steps.append(
                    {
                        "idx": 0,
                        "step_type": "rag_query",
                        "tool": "rag_search",
                        "args": {"query": query_text},
                        "description": "Search your knowledge base",
                        "why": why_text,
                    }
                )
            return {"steps": steps}

        async def _handle_persona_live_turn(
            *,
            msg: dict[str, Any],
            text: str,
            turn_type: str = "user_message",
            source: str = "ws",
        ) -> None:
            normalized_text = str(text or "").strip()
            original_session_id = msg.get("session_id")
            session_id = _normalize_ws_identifier(original_session_id, fallback=default_session_id)
            if str(original_session_id or "").strip() and str(original_session_id).strip() != session_id:
                await _emit_notice(
                    session_id=session_id,
                    level="warning",
                    message="Invalid session_id normalized to a safe identifier.",
                    reason_code="SESSION_ID_NORMALIZED",
                )
            forbidden_client_fields = [
                field_name
                for field_name in ("persona_scope", "allowed_tools", "persona_audit", "metadata")
                if field_name in msg
            ]
            if forbidden_client_fields:
                await _emit_notice(
                    session_id=session_id,
                    level="warning",
                    message="Ignored unsupported client-controlled security fields.",
                    reason_code="SECURITY_FIELDS_IGNORED",
                    ignored_fields=forbidden_client_fields,
                )

            runtime_context = _load_persona_policy_rules_for_session(
                persona_scope_db,
                session_id=session_id,
                user_id=authenticated_user_id,
            )
            runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
            runtime_mode = _bounded_label(
                runtime_context.get("runtime_mode"),
                allowed=_PERSONA_RUNTIME_MODES,
                fallback="session_scoped",
            )
            runtime_scope_snapshot_id = str(runtime_context.get("scope_snapshot_id") or "").strip() or None
            persona_policy_rules = normalize_policy_rules(runtime_context.get("policy_rules"))
            session_exists = bool(runtime_context.get("session_exists", False))
            _increment_persona_metric(
                "persona_ws_runtime_mode_total",
                {"mode": runtime_mode, "session_exists": "true" if session_exists else "false"},
            )
            _ = session_manager.create(
                user_id=connection_user_id,
                persona_id=runtime_persona_id,
                resume_session_id=session_id,
            )
            existing_preferences, activity_surface = _get_session_preferences_with_activity_surface(
                session_manager=session_manager,
                session_id=session_id,
                user_id=connection_user_id,
                persisted_preferences=runtime_context.get("preferences"),
                persisted_activity_surface=runtime_context.get("activity_surface"),
            )
            configured_top_k = _get_persona_memory_top_k()
            default_use_memory = _coerce_bool(
                existing_preferences.get("use_memory_context"),
                default=True,
            )
            requested_use_memory_context = _coerce_bool(
                msg.get("use_memory_context"),
                default=default_use_memory,
            )
            use_memory_context = requested_use_memory_context
            default_use_companion_context = _coerce_bool(
                existing_preferences.get("use_companion_context"),
                default=True,
            )
            requested_use_companion_context = _coerce_bool(
                msg.get("use_companion_context"),
                default=default_use_companion_context,
            )
            use_companion_context = requested_use_companion_context
            runtime_persona_state_context_default = _coerce_bool(
                runtime_context.get("persona_state_context_default"),
                default=True,
            )
            default_use_persona_state_context = _coerce_bool(
                existing_preferences.get("use_persona_state_context"),
                default=runtime_persona_state_context_default,
            )
            requested_use_persona_state_context = _coerce_bool(
                msg.get("use_persona_state_context"),
                default=default_use_persona_state_context,
            )
            state_context_override = "use_persona_state_context" in msg
            use_persona_state_context = requested_use_persona_state_context
            memory_allowed_by_mode = _memory_mode_allows_personalization_retrieval(
                runtime_mode,
                session_exists=session_exists,
            )
            if not memory_allowed_by_mode:
                use_memory_context = False
            state_context_allowed_by_mode = runtime_mode == "persistent_scoped"
            if not state_context_allowed_by_mode:
                use_persona_state_context = False
            pref_top_k_raw = existing_preferences.get("memory_top_k", configured_top_k)
            try:
                pref_top_k = int(pref_top_k_raw)
            except (TypeError, ValueError):
                pref_top_k = configured_top_k
            requested_top_k_raw = msg.get("memory_top_k", pref_top_k)
            try:
                memory_top_k = int(requested_top_k_raw)
            except (TypeError, ValueError):
                memory_top_k = pref_top_k
            memory_top_k = max(1, min(memory_top_k, configured_top_k))
            if "session_policy_rules" in msg:
                session_policy_rules = normalize_policy_rules(msg.get("session_policy_rules"))
            else:
                session_policy_rules = _session_policy_rules_from_preferences(existing_preferences)
            preferences_patch: dict[str, Any] = {
                "use_memory_context": use_memory_context,
                "use_companion_context": use_companion_context,
                "use_persona_state_context": use_persona_state_context,
                "memory_top_k": memory_top_k,
            }
            if "session_policy_rules" in msg:
                preferences_patch["session_policy_rules"] = session_policy_rules
            with contextlib.suppress(Exception):
                session_manager.update_preferences(
                    session_id=session_id,
                    user_id=connection_user_id,
                    preferences=preferences_patch,
                )
            with contextlib.suppress(Exception):
                session_manager.update_preferences(
                    session_id=session_id,
                    user_id=connection_user_id,
                    preferences={
                        "last_input_source": source,
                        "last_turn_type": turn_type,
                    },
                )
            _ = _persist_persona_session_preferences(
                persona_scope_db,
                session_id=session_id,
                user_id=authenticated_user_id,
                base_preferences=runtime_context.get("preferences"),
                patch_preferences=preferences_patch,
            )
            persona_turn_classifier = classify_persona_turn(normalized_text)
            persona_exemplar_context = await resolve_persona_exemplar_runtime_context(
                persona_scope_db=persona_scope_db,
                user_id=authenticated_user_id,
                persona_id=runtime_persona_id,
                classifier=persona_turn_classifier,
                current_turn_text=normalized_text,
                lookup_limit=50,
            )
            persona_exemplar_assembly = persona_exemplar_context.assembly
            persona_exemplar_selection = persona_exemplar_context.selection_metadata
            await _record_turn(
                session_id=session_id,
                role="user",
                content=normalized_text,
                turn_type=turn_type,
                metadata={
                    "source": source,
                    "use_memory_context": use_memory_context,
                    "use_companion_context": use_companion_context,
                    "use_persona_state_context": use_persona_state_context,
                    "memory_top_k": memory_top_k,
                    "session_policy_rule_count": len(session_policy_rules),
                    "runtime_mode": runtime_mode,
                    "session_exists": session_exists,
                    "persona_exemplar_selection": persona_exemplar_selection,
                },
                persist_as_memory=False,
                persona_id_override=runtime_persona_id,
                runtime_mode_override=runtime_mode,
                scope_snapshot_id_override=runtime_scope_snapshot_id,
            )
            memory_context: list[str] = []
            if use_memory_context and memory_allowed_by_mode:
                memories = retrieve_top_memories(
                    user_id=authenticated_user_id,
                    query_text=normalized_text,
                    top_k=memory_top_k,
                    persona_id=runtime_persona_id,
                    runtime_mode=runtime_mode,
                    scope_snapshot_id=runtime_scope_snapshot_id,
                    session_id=session_id,
                )
                memory_context = [m.content for m in memories]
            companion_context = {
                "knowledge_lines": [],
                "activity_lines": [],
                "card_count": 0,
                "activity_count": 0,
            }
            if use_companion_context:
                companion_context = await asyncio.to_thread(
                    load_companion_context,
                    user_id=authenticated_user_id,
                    query=normalized_text,
                )
            persona_state_hints: dict[str, str] = {}
            if use_persona_state_context and state_context_allowed_by_mode:
                persona_state_hints = _load_persona_state_hints_for_runtime(
                    persona_scope_db,
                    user_id=authenticated_user_id,
                    persona_id=runtime_persona_id,
                    runtime_mode=runtime_mode,
                )
            persona_state_fields = sorted(persona_state_hints.keys())
            memory_usage = {
                "enabled": use_memory_context,
                "requested_enabled": requested_use_memory_context,
                "requested_top_k": memory_top_k,
                "applied_count": len(memory_context),
                "runtime_mode": runtime_mode,
                "persona_state_enabled": use_persona_state_context,
                "persona_state_requested_enabled": requested_use_persona_state_context,
                "persona_state_profile_default": runtime_persona_state_context_default,
                "persona_state_mode_allowed": state_context_allowed_by_mode,
                "persona_state_applied_count": len(persona_state_fields),
                "persona_state_fields": persona_state_fields,
            }
            companion_usage = {
                "enabled": use_companion_context,
                "requested_enabled": requested_use_companion_context,
                "mode": str(companion_context.get("mode") or "recent_fallback"),
                "applied_card_count": int(companion_context.get("card_count", 0) or 0),
                "applied_goal_count": int(companion_context.get("goal_count", 0) or 0),
                "applied_activity_count": int(companion_context.get("activity_count", 0) or 0),
            }
            if use_memory_context and memory_context:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="MEMORY_CONTEXT_APPLIED",
                    message=f"Applied {len(memory_context)} personalization memories",
                )
            elif requested_use_memory_context and not memory_allowed_by_mode:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="MEMORY_CONTEXT_MODE_DISABLED",
                    message="Memory context is disabled for session_scoped personas.",
                )
            elif not use_memory_context:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="MEMORY_CONTEXT_DISABLED",
                    message="Memory context disabled for this message",
                )
            if use_companion_context and (
                companion_usage["applied_card_count"]
                or companion_usage["applied_goal_count"]
                or companion_usage["applied_activity_count"]
            ):
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="COMPANION_CONTEXT_APPLIED",
                    message=(
                        "Applied companion context from "
                        f"{companion_usage['applied_card_count']} knowledge cards and "
                        f"{companion_usage['applied_goal_count']} goals and "
                        f"{companion_usage['applied_activity_count']} recent activities"
                    ),
                )
            elif not use_companion_context:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="COMPANION_CONTEXT_DISABLED",
                    message="Companion context disabled for this message",
                )
            if persona_state_fields:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="PERSONA_STATE_HINTS_APPLIED",
                    message=f"Applied {len(persona_state_fields)} persona state docs",
                )
            elif state_context_override and requested_use_persona_state_context and not state_context_allowed_by_mode:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="PERSONA_STATE_MODE_DISABLED",
                    message="Persona state context is disabled for session_scoped personas.",
                )
            elif state_context_override and not use_persona_state_context:
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="PERSONA_STATE_DISABLED",
                    message="Persona state context disabled for this message",
                )
            plan = await _propose_plan(
                normalized_text,
                memory_context=memory_context,
                persona_state_hints=persona_state_hints,
                companion_context=companion_context,
                persona_exemplar_sections=persona_exemplar_assembly.sections,
            )
            plan_id = uuid.uuid4().hex
            max_tool_steps = _get_persona_max_tool_steps()
            proposed_steps = list(plan.get("steps", []))
            if len(proposed_steps) > max_tool_steps:
                proposed_steps = proposed_steps[:max_tool_steps]
                await _emit_notice(
                    session_id=session_id,
                    level="warning",
                    message=f"Plan truncated to max_tool_steps={max_tool_steps}",
                    reason_code="PLAN_TRUNCATED",
                )
            try:
                pending_plan = session_manager.put_plan(
                    session_id=session_id,
                    user_id=connection_user_id,
                    persona_id=runtime_persona_id,
                    plan_id=plan_id,
                    steps=proposed_steps,
                )
            except ValueError as exc:
                _cancel_persona_live_processing_notice(session_id)
                await _emit_notice(
                    session_id=session_id,
                    level="error",
                    message=str(exc),
                    reason_code="PLAN_INVALID",
                )
                return
            stored_steps: list[dict[str, Any]] = []
            for step in pending_plan.steps:
                step_type = _normalize_persona_step_type(step.step_type, tool_name=step.tool)
                policy = _evaluate_step_policy(
                    step_type=step_type,
                    tool_name=step.tool,
                    args=step.args,
                    persona_policy_rules=persona_policy_rules,
                    session_policy_rules=session_policy_rules,
                    session_scopes=session_scopes,
                    allow_export=allow_export,
                    allow_delete=allow_delete,
                )
                if not bool(policy.get("allow", False)):
                    _increment_persona_metric(
                        "persona_ws_policy_denials_total",
                        {
                            "step_type": _bounded_label(step_type, allowed=_PERSONA_WS_ALLOWED_STEP_TYPES, fallback="mcp_tool"),
                            "reason": _metric_reason_bucket(policy.get("reason_code")),
                        },
                    )
                stored_steps.append(
                    {
                        "idx": step.idx,
                        "step_type": step_type,
                        "tool": step.tool,
                        "args": step.args,
                        "description": step.description,
                        "why": step.why,
                        "policy": policy,
                    }
                )
            await _emit_tool_plan(
                session_id=session_id,
                plan_id=plan_id,
                steps=stored_steps,
                memory=memory_usage,
                companion=companion_usage,
                persona_id_value=runtime_persona_id,
            )

        def _normalize_voice_runtime_config(msg: dict[str, Any]) -> dict[str, Any]:
            voice_payload = msg.get("voice") if isinstance(msg.get("voice"), dict) else {}
            stt_payload = msg.get("stt") if isinstance(msg.get("stt"), dict) else {}
            tts_payload = msg.get("tts") if isinstance(msg.get("tts"), dict) else {}

            normalized_trigger_phrases: list[str] = []
            seen_phrases: set[str] = set()
            for raw_phrase in voice_payload.get("trigger_phrases") or []:
                phrase = str(raw_phrase or "").strip()
                if not phrase or phrase in seen_phrases:
                    continue
                seen_phrases.add(phrase)
                normalized_trigger_phrases.append(phrase)

            return {
                "trigger_phrases": normalized_trigger_phrases,
                "auto_resume": _coerce_bool(voice_payload.get("auto_resume"), default=False),
                "barge_in": _coerce_bool(voice_payload.get("barge_in"), default=False),
                "stt_language": str(stt_payload.get("language") or "").strip() or None,
                "stt_model": str(stt_payload.get("model") or "").strip() or None,
                "enable_vad": _coerce_bool(stt_payload.get("enable_vad"), default=True),
                "vad_threshold": _clamp_persona_live_float(
                    stt_payload.get("vad_threshold"),
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                "vad_min_silence_ms": _clamp_persona_live_int(
                    stt_payload.get("min_silence_ms"),
                    default=250,
                    min_value=50,
                    max_value=10_000,
                ),
                "vad_turn_stop_secs": _clamp_persona_live_float(
                    stt_payload.get("turn_stop_secs"),
                    default=0.2,
                    min_value=0.05,
                    max_value=10.0,
                ),
                "vad_min_utterance_secs": _clamp_persona_live_float(
                    stt_payload.get("min_utterance_secs"),
                    default=0.4,
                    min_value=0.0,
                    max_value=10.0,
                ),
                "tts_provider": str(tts_payload.get("provider") or "").strip() or None,
                "tts_voice": str(tts_payload.get("voice") or "").strip() or None,
                "text_only_due_to_tts_failure": False,
            }

        while True:
            raw = await stream.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                msg = {"type": "unknown", "raw": raw}

            mtype = msg.get("type") or msg.get("event") or "unknown"
            if mtype == "user_message":
                _cancel_persona_live_processing_notice(
                    _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                )
                await _handle_persona_live_turn(
                    msg=msg,
                    text=msg.get("text") or msg.get("message") or "",
                    turn_type="user_message",
                    source="ws",
                )
            elif mtype == "voice_config":
                original_session_id = msg.get("session_id")
                session_id = _normalize_ws_identifier(original_session_id, fallback="")
                if not session_id:
                    await _emit_notice(
                        session_id=default_session_id,
                        level="error",
                        message="session_id is required",
                        reason_code="SESSION_ID_REQUIRED",
                    )
                    continue
                if str(original_session_id or "").strip() != session_id:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        message="Invalid session_id normalized to a safe identifier.",
                        reason_code="SESSION_ID_NORMALIZED",
                    )
                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
                _ = session_manager.create(
                    user_id=connection_user_id,
                    persona_id=runtime_persona_id,
                    resume_session_id=session_id,
                )
                voice_runtime = _normalize_voice_runtime_config(msg)
                session_manager.update_preferences(
                    session_id=session_id,
                    user_id=connection_user_id,
                    preferences={"voice_runtime": voice_runtime},
                )
                _upsert_persona_live_session_summary_safe(
                    session_id=session_id,
                    voice_runtime=voice_runtime,
                )
                _cleanup_persona_live_stt_state(session_id)
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    message="Voice runtime updated for this live session.",
                    reason_code="VOICE_CONFIG_UPDATED",
                )
            elif mtype == "voice_commit":
                original_session_id = msg.get("session_id")
                session_id = _normalize_ws_identifier(original_session_id, fallback="")
                if not session_id:
                    await _emit_notice(
                        session_id=default_session_id,
                        level="error",
                        message="session_id is required",
                        reason_code="SESSION_ID_REQUIRED",
                    )
                    continue
                _cancel_persona_live_processing_notice(session_id)
                state = persona_live_stt_state_by_session.get(session_id) or {}
                if bool(state.get("current_utterance_committed")) and not _current_persona_live_transcript(
                    session_id
                ):
                    await _emit_notice(
                        session_id=session_id,
                        level="info",
                        reason_code="VOICE_COMMIT_IGNORED_ALREADY_COMMITTED",
                        message="This utterance was already committed.",
                        commit_source=str(state.get("current_commit_source") or "vad_auto"),
                        transcript=str(state.get("committed_transcript") or "").strip() or None,
                    )
                    continue
                transcript = str(msg.get("transcript") or msg.get("text") or "").strip()
                if not transcript:
                    transcript = _current_persona_live_transcript(session_id)
                if not transcript:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="transcript is required",
                        reason_code="TRANSCRIPT_REQUIRED",
                    )
                    continue
                await _commit_persona_live_turn(
                    session_id=session_id,
                    transcript=transcript,
                    commit_source="manual",
                    source=str(msg.get("source") or "persona_live_voice").strip()
                    or "persona_live_voice",
                )
            elif mtype == "audio_chunk":
                session_id = _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
                runtime_mode = _bounded_label(
                    runtime_context.get("runtime_mode"),
                    allowed=_PERSONA_RUNTIME_MODES,
                    fallback="session_scoped",
                )
                runtime_scope_snapshot_id = str(runtime_context.get("scope_snapshot_id") or "").strip() or None
                audio_format = str(msg.get("audio_format") or "pcm16").strip().lower()
                if audio_format not in allowed_audio_formats:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        reason_code="AUDIO_FORMAT_UNSUPPORTED",
                        message=f"Unsupported audio_format '{audio_format}'",
                    )
                    continue

                try:
                    audio_bytes = _decode_audio_chunk(
                        str(msg.get("bytes_base64") or ""),
                        max_decoded_bytes=audio_chunk_max_bytes,
                    )
                except ValueError as exc:
                    error_message = str(exc)
                    reason_code = "AUDIO_CHUNK_INVALID"
                    if (
                        "exceeds max bytes" in error_message
                        or "projected decoded size exceeds" in error_message
                    ):
                        reason_code = "AUDIO_CHUNK_TOO_LARGE"
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message=error_message,
                        reason_code=reason_code,
                    )
                    continue

                if len(audio_bytes) > audio_chunk_max_bytes:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        reason_code="AUDIO_CHUNK_TOO_LARGE",
                        message=f"Audio chunk exceeds max bytes ({len(audio_bytes)} > {audio_chunk_max_bytes})",
                    )
                    continue

                now_mono = time.monotonic()
                session_window = audio_rate_windows[session_id]
                while session_window and (now_mono - session_window[0]) >= 60.0:
                    session_window.popleft()
                if len(session_window) >= audio_chunks_per_minute:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        reason_code="AUDIO_RATE_LIMITED",
                        message=f"Audio chunk rate limit exceeded ({audio_chunks_per_minute}/minute)",
                    )
                    continue
                session_window.append(now_mono)

                timestamp_ms = int(time.time() * 1000)
                transcript_delta = ""
                auto_commit_triggered = False
                should_fallback_to_scaffold = True
                buffer_updated_from_snapshot = False
                stt_state, _voice_runtime = _get_or_create_persona_live_stt_state(session_id)
                if stt_state is not None:
                    should_fallback_to_scaffold = False
                    transcriber = stt_state.get("transcriber")
                    turn_detector = stt_state.get("turn_detector")
                    try:
                        normalized_audio = _normalize_persona_live_stt_audio(
                            audio_bytes,
                            audio_format=audio_format,
                        )
                        if turn_detector is None or not bool(getattr(turn_detector, "available", False)):
                            await _ensure_persona_live_manual_mode_notice(session_id, stt_state)
                        previous_snapshot = str(
                            voice_transcript_buffer_by_session.get(session_id) or ""
                        ).strip()
                        result = await transcriber.process_audio_chunk(normalized_audio)
                        if isinstance(result, dict):
                            next_snapshot = _persona_live_transcript_snapshot(
                                transcriber=transcriber,
                                result=result,
                            )
                        else:
                            next_snapshot = str(transcriber.get_full_transcript() or "").strip()
                        next_snapshot = str(next_snapshot or "").strip()
                        if next_snapshot and bool(stt_state.get("current_utterance_committed")):
                            _clear_persona_live_commit_state(stt_state)
                        if next_snapshot:
                            voice_transcript_buffer_by_session[session_id] = next_snapshot
                            buffer_updated_from_snapshot = True
                        transcript_delta = _persona_live_forward_delta(
                            previous_snapshot,
                            next_snapshot,
                        )
                        if turn_detector is not None and bool(getattr(turn_detector, "available", False)):
                            auto_commit_triggered = bool(turn_detector.observe(normalized_audio))
                            if not bool(getattr(turn_detector, "available", False)):
                                stt_state["manual_mode_reason"] = str(
                                    getattr(turn_detector, "unavailable_reason", "") or "vad_unavailable"
                                )
                                await _ensure_persona_live_manual_mode_notice(session_id, stt_state)
                                auto_commit_triggered = False
                    except Exception as exc:
                        logger.debug(
                            "persona live STT processing failed for session {}: {}",
                            session_id,
                            exc,
                        )
                        _cleanup_persona_live_stt_state(session_id)
                        should_fallback_to_scaffold = True

                if should_fallback_to_scaffold:
                    transcript_delta = await _transcribe_audio_chunk(
                        audio_bytes,
                        audio_format=audio_format,
                    )
                if transcript_delta:
                    transcript_seq = transcript_seq_by_session[session_id]
                    transcript_seq_by_session[session_id] += 1
                    if not buffer_updated_from_snapshot:
                        existing_buffer = str(
                            voice_transcript_buffer_by_session.get(session_id) or ""
                        ).strip()
                        voice_transcript_buffer_by_session[session_id] = (
                            f"{existing_buffer} {transcript_delta}".strip()
                            if existing_buffer
                            else transcript_delta
                        )
                    await stream.send_json(
                        {
                            "event": "partial_transcript",
                            "session_id": session_id,
                            "text_delta": transcript_delta,
                            "audio_format": audio_format,
                            "seq": transcript_seq,
                            "timestamp_ms": timestamp_ms,
                        }
                    )

                if auto_commit_triggered:
                    _ = await _commit_persona_live_turn(
                        session_id=session_id,
                        transcript=_current_persona_live_transcript(session_id),
                        commit_source="vad_auto",
                        source="persona_live_voice_auto",
                    )

                if "tts_text" not in msg:
                    continue

                tts_text = str(msg.get("tts_text") or f"You said: {transcript_delta or 'audio received.'}")
                tts_source_len = len(tts_text.encode("utf-8"))
                tts_chunks = await _generate_tts_audio_chunks(
                    tts_text,
                    audio_format=audio_format,
                    chunk_size_bytes=tts_chunk_size_bytes,
                    max_chunks=tts_max_chunks,
                    max_total_bytes=tts_max_total_bytes,
                )
                emitted_tts_bytes = sum(len(chunk) for chunk in tts_chunks)
                if tts_chunks and emitted_tts_bytes < tts_source_len:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        reason_code="TTS_OUTPUT_TRUNCATED",
                        message=f"TTS output truncated ({emitted_tts_bytes} of {tts_source_len} bytes)",
                    )

                total_chunks = len(tts_chunks)
                for idx, chunk in enumerate(tts_chunks):
                    if tts_in_flight_by_session[session_id] >= tts_max_in_flight_chunks:
                        await _emit_notice(
                            session_id=session_id,
                            level="warning",
                            reason_code="TTS_BACKPRESSURE_DROP",
                            message=f"Dropping TTS chunk due to in-flight limit ({tts_max_in_flight_chunks})",
                        )
                        break

                    chunk_seq = tts_seq_by_session[session_id]
                    tts_seq_by_session[session_id] += 1
                    chunk_id = uuid.uuid4().hex
                    tts_in_flight_by_session[session_id] += 1
                    await stream.send_json(
                        {
                            "event": "tts_audio",
                            "session_id": session_id,
                            "audio_format": audio_format,
                            "chunk_id": chunk_id,
                            "chunk_index": idx,
                            "chunk_count": total_chunks,
                            "seq": chunk_seq,
                            "timestamp_ms": int(time.time() * 1000),
                        }
                    )
                    try:
                        await stream.ws.send_bytes(chunk)
                    except Exception as exc:
                        await _emit_notice(
                            session_id=session_id,
                            level="warning",
                            message=f"Failed to send tts audio binary chunk: {exc}",
                            reason_code="TTS_SEND_FAILED",
                        )
                        tts_in_flight_by_session[session_id] = max(
                            0, tts_in_flight_by_session[session_id] - 1
                        )
                        break
                    tts_in_flight_by_session[session_id] = max(
                        0, tts_in_flight_by_session[session_id] - 1
                    )
                await _record_turn(
                    session_id=session_id,
                    role="assistant",
                    content=tts_text,
                    turn_type="tts_audio",
                    metadata={"audio_format": audio_format, "chunks": len(tts_chunks)},
                    persist_as_memory=False,
                    persona_id_override=runtime_persona_id,
                    runtime_mode_override=runtime_mode,
                    scope_snapshot_id_override=runtime_scope_snapshot_id,
                )
            elif mtype == "confirm_plan":
                if not await _is_stream_auth_valid():
                    await _close_for_auth_revocation()
                    break
                session_id = _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                plan_id = _normalize_ws_identifier(msg.get("plan_id"), fallback="")
                if not plan_id:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="plan_id is required",
                        reason_code="PLAN_ID_REQUIRED",
                    )
                    continue

                approved_steps_raw = msg.get("approved_steps", [])
                if not isinstance(approved_steps_raw, list):
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="approved_steps must be a list",
                        reason_code="APPROVED_STEPS_INVALID",
                    )
                    continue
                approved_step_indices: list[int] = []
                for raw_idx in approved_steps_raw:
                    try:
                        approved_step_indices.append(int(raw_idx))
                    except (TypeError, ValueError):
                        continue
                if not approved_step_indices:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        message="No valid approved steps",
                        reason_code="APPROVED_STEPS_EMPTY",
                    )
                    continue
                max_tool_steps = _get_persona_max_tool_steps()
                approved_step_indices = sorted(set(approved_step_indices))[:max_tool_steps]

                pending_plan = session_manager.get_plan(
                    session_id=session_id,
                    plan_id=plan_id,
                    user_id=connection_user_id,
                    consume=True,
                )
                if pending_plan is None:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="Invalid plan_id/session_id",
                        reason_code="PLAN_NOT_FOUND",
                    )
                    continue

                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
                runtime_mode = _bounded_label(
                    runtime_context.get("runtime_mode"),
                    allowed=_PERSONA_RUNTIME_MODES,
                    fallback="session_scoped",
                )
                runtime_scope_snapshot_id = str(runtime_context.get("scope_snapshot_id") or "").strip() or None
                persona_policy_rules = normalize_policy_rules(runtime_context.get("policy_rules"))
                current_preferences, activity_surface = _get_session_preferences_with_activity_surface(
                    session_manager=session_manager,
                    session_id=session_id,
                    user_id=connection_user_id,
                    persisted_preferences=runtime_context.get("preferences"),
                    persisted_activity_surface=runtime_context.get("activity_surface"),
                )
                if "session_policy_rules" in msg:
                    session_policy_rules = normalize_policy_rules(msg.get("session_policy_rules"))
                    with contextlib.suppress(Exception):
                        session_manager.update_preferences(
                            session_id=session_id,
                            user_id=connection_user_id,
                            preferences={"session_policy_rules": session_policy_rules},
                        )
                    _ = _persist_persona_session_preferences(
                        persona_scope_db,
                        session_id=session_id,
                        user_id=authenticated_user_id,
                        base_preferences=runtime_context.get("preferences"),
                        patch_preferences={"session_policy_rules": session_policy_rules},
                    )
                else:
                    session_policy_rules = _session_policy_rules_from_preferences(current_preferences)

                executed_steps = 0
                for step in pending_plan.steps:
                    if step.idx not in approved_step_indices:
                        continue
                    step_type = _normalize_persona_step_type(step.step_type, tool_name=step.tool)
                    step_policy = _evaluate_step_policy(
                        step_type=step_type,
                        tool_name=step.tool,
                        args=step.args,
                        persona_policy_rules=persona_policy_rules,
                        session_policy_rules=session_policy_rules,
                        session_scopes=session_scopes,
                        allow_export=allow_export,
                        allow_delete=allow_delete,
                    )
                    if not bool(step_policy.get("allow", False)):
                        await _emit_denied_tool_step(
                            session_id=session_id,
                            plan_id=plan_id,
                            step_idx=step.idx,
                            step_type=step_type,
                            tool_name=step.tool,
                            step_policy=step_policy,
                            persona_id=runtime_persona_id,
                            runtime_mode_value=runtime_mode,
                            scope_snapshot_id=runtime_scope_snapshot_id,
                        )
                        executed_steps += 1
                        continue

                    if step_type == "final_answer":
                        executed_steps += 1
                        assistant_text = str(
                            (step.args or {}).get("text")
                            or step.description
                            or "Final answer step executed."
                        )
                        _ = await asyncio.to_thread(
                            record_persona_session_summarized,
                            user_id=authenticated_user_id,
                            session_id=session_id,
                            persona_id=runtime_persona_id,
                            plan_id=plan_id,
                            step_idx=step.idx,
                            runtime_mode=runtime_mode,
                            scope_snapshot_id=runtime_scope_snapshot_id,
                            summary_text=assistant_text,
                            surface=activity_surface,
                        )
                        await _emit_assistant_delta(
                            session_id=session_id,
                            step_idx=step.idx,
                            text_delta=assistant_text,
                        )
                        await _record_turn(
                            session_id=session_id,
                            role="assistant",
                            content=assistant_text,
                            turn_type="assistant_delta",
                            metadata={"source": "plan", "step_idx": step.idx, "step_type": step_type},
                            persist_as_memory=True,
                            persona_id_override=runtime_persona_id,
                            runtime_mode_override=runtime_mode,
                            scope_snapshot_id_override=runtime_scope_snapshot_id,
                            memory_kind="summary",
                        )
                        await _emit_persona_live_tts_for_assistant_text(
                            session_id=session_id,
                            assistant_text=assistant_text,
                        )
                        continue

                    executed_steps += 1
                    result = await _execute_persona_tool_step(
                        session_id=session_id,
                        plan_id=plan_id,
                        step_idx=step.idx,
                        step_type=step_type,
                        tool_name=step.tool,
                        step_args=step.args or {},
                        step_policy=step_policy,
                        why=step.why,
                        description=step.description,
                        persona_id=runtime_persona_id,
                        runtime_mode_value=runtime_mode,
                        scope_snapshot_id=runtime_scope_snapshot_id,
                    )
                    _ = await asyncio.to_thread(
                        record_persona_tool_executed,
                        user_id=authenticated_user_id,
                        session_id=session_id,
                        persona_id=runtime_persona_id,
                        plan_id=plan_id,
                        step_idx=step.idx,
                        step_type=step_type,
                        tool_name=step.tool,
                        runtime_mode=runtime_mode,
                        scope_snapshot_id=runtime_scope_snapshot_id,
                        outcome=result,
                        surface=activity_surface,
                    )
                if executed_steps == 0:
                    await _emit_notice(
                        session_id=session_id,
                        level="warning",
                        message="No approved steps matched plan",
                        reason_code="APPROVED_STEPS_NO_MATCH",
                    )
            elif mtype == "retry_tool_call":
                if not await _is_stream_auth_valid():
                    await _close_for_auth_revocation()
                    break
                session_id = _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                plan_id = _normalize_ws_identifier(msg.get("plan_id"), fallback="")
                tool_name = str(msg.get("tool") or "").strip()
                if not tool_name:
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="tool is required",
                        reason_code="TOOL_REQUIRED",
                    )
                    continue
                try:
                    step_idx = int(msg.get("step_idx"))
                except (TypeError, ValueError):
                    await _emit_notice(
                        session_id=session_id,
                        level="error",
                        message="step_idx must be an integer",
                        reason_code="STEP_IDX_INVALID",
                    )
                    continue
                step_type = _normalize_persona_step_type(
                    str(msg.get("step_type") or "mcp_tool"),
                    tool_name=tool_name,
                )
                pending_retry = _consume_pending_retry_approval(
                    session_id=session_id,
                    plan_id=plan_id,
                    step_idx=step_idx,
                    tool_name=tool_name,
                )
                if pending_retry is None:
                    await _emit_notice(
                        session_id=session_id,
                        step_idx=step_idx,
                        tool=tool_name,
                        step_type=step_type,
                        level="warning",
                        message="No pending approval retry found for this tool step",
                        reason_code="APPROVAL_RETRY_NOT_FOUND",
                    )
                    continue
                plan_id = str(pending_retry.get("plan_id") or plan_id)
                step_idx = int(pending_retry.get("step_idx") or step_idx)
                tool_name = str(pending_retry.get("tool") or tool_name)
                step_type = _normalize_persona_step_type(
                    pending_retry.get("step_type"),
                    tool_name=tool_name,
                )
                step_args = pending_retry.get("args")
                if not isinstance(step_args, dict):
                    step_args = {}
                why = str(pending_retry.get("why") or "").strip() or None
                description = str(pending_retry.get("description") or "").strip() or None

                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
                runtime_mode = _bounded_label(
                    runtime_context.get("runtime_mode"),
                    allowed=_PERSONA_RUNTIME_MODES,
                    fallback="session_scoped",
                )
                runtime_scope_snapshot_id = str(runtime_context.get("scope_snapshot_id") or "").strip() or None
                persona_policy_rules = normalize_policy_rules(runtime_context.get("policy_rules"))
                current_preferences = session_manager.get_preferences(
                    session_id=session_id,
                    user_id=connection_user_id,
                )
                session_policy_rules = _session_policy_rules_from_preferences(current_preferences)
                step_policy = _evaluate_step_policy(
                    step_type=step_type,
                    tool_name=tool_name,
                    args=step_args,
                    persona_policy_rules=persona_policy_rules,
                    session_policy_rules=session_policy_rules,
                    session_scopes=session_scopes,
                    allow_export=allow_export,
                    allow_delete=allow_delete,
                )
                if not bool(step_policy.get("allow", False)):
                    await _emit_denied_tool_step(
                        session_id=session_id,
                        plan_id=plan_id,
                        step_idx=step_idx,
                        step_type=step_type,
                        tool_name=tool_name,
                        step_policy=step_policy,
                        persona_id=runtime_persona_id,
                        runtime_mode_value=runtime_mode,
                        scope_snapshot_id=runtime_scope_snapshot_id,
                    )
                    continue

                await _execute_persona_tool_step(
                    session_id=session_id,
                    plan_id=plan_id,
                    step_idx=step_idx,
                    step_type=step_type,
                    tool_name=tool_name,
                    step_args=step_args,
                    step_policy=step_policy,
                    why=why,
                    description=description,
                    persona_id=runtime_persona_id,
                    runtime_mode_value=runtime_mode,
                    scope_snapshot_id=runtime_scope_snapshot_id,
                )
            elif mtype == "cancel":
                session_id = _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                reason = str(msg.get("reason") or "user_cancelled")
                _cancel_persona_live_processing_notice(session_id)
                cleared = session_manager.clear_plans(session_id=session_id, user_id=connection_user_id)
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="PLAN_CANCELLED",
                    message=f"Cancelled pending work ({cleared} plan(s) cleared): {reason}",
                )
            else:
                session_id = _normalize_ws_identifier(msg.get("session_id"), fallback=default_session_id)
                runtime_context = _load_persona_policy_rules_for_session(
                    persona_scope_db,
                    session_id=session_id,
                    user_id=authenticated_user_id,
                )
                runtime_persona_id = str(runtime_context.get("persona_id") or _DEFAULT_PERSONA_ID).strip() or _DEFAULT_PERSONA_ID
                runtime_mode = _bounded_label(
                    runtime_context.get("runtime_mode"),
                    allowed=_PERSONA_RUNTIME_MODES,
                    fallback="session_scoped",
                )
                runtime_scope_snapshot_id = str(runtime_context.get("scope_snapshot_id") or "").strip() or None
                assistant_text = "(scaffold)"
                await _emit_assistant_delta(
                    session_id=session_id,
                    text_delta=assistant_text,
                )
                await _emit_notice(
                    session_id=session_id,
                    level="info",
                    reason_code="ECHO_EVENT",
                    message=f"echo: {mtype}",
                )
                await _record_turn(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_text,
                    turn_type="assistant_delta",
                    metadata={"echo_type": str(mtype)},
                    persist_as_memory=False,
                    persona_id_override=runtime_persona_id,
                    runtime_mode_override=runtime_mode,
                    scope_snapshot_id_override=runtime_scope_snapshot_id,
                )
    except WebSocketDisconnect:
        if auth_revoked_event is not None and auth_revoked_event.is_set():
            logger.info("Persona stream disconnected after auth revalidation failure")
        else:
            logger.info("Persona stream disconnected")
    except Exception as e:
        logger.warning(f"Persona stream error: {e}")
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.error("internal_error", "Internal error")
        else:
            with contextlib.suppress(Exception):
                await ws.close(code=1011)
    finally:
        if auth_watchdog_stop is not None:
            auth_watchdog_stop.set()
        if auth_watchdog_task is not None:
            auth_watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await auth_watchdog_task
        for session_id in list(persona_live_summary_sessions_seen):
            _upsert_persona_live_session_summary_safe(
                session_id=session_id,
                voice_runtime=None,
                finalize=True,
            )
        for session_id in list(persona_live_stt_state_by_session.keys()):
            _cleanup_persona_live_stt_state(session_id)
        if stream is not None:
            with contextlib.suppress(Exception):
                await stream.stop()
            with contextlib.suppress(Exception):
                await stream.ws.close()
        if persona_scope_db is not None:
            with contextlib.suppress(Exception):
                persona_scope_db.close_all_connections()
            with contextlib.suppress(Exception):
                persona_scope_db.close_connection()
