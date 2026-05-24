from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import threading
import tempfile
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import ValidationError
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.endpoints._in_memory_limits import SlidingWindowLimiter
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACP_COMPATIBILITY_DOCS_URL,
    ACPAgentCompatibilityStatus,
    ACPAgentInfo,
    ACPAgentListResponse,
    ACPAgentHealthEntry,
    ACPAgentHealthResponse,
    ACPAgentRegisterRequest,
    ACPAgentRegistrationResponse,
    ACPAgentUpdateRequest,
    ACPAsyncPromptRequest,
    ACPAsyncPromptResponse,
    ACPCheckpointListResponse,
    ACPHealthResponse,
    ACPSetupGuideResponse,
    ACPRollbackRequest,
    ACPRollbackResponse,
    ACPSessionCancelRequest,
    ACPSessionCloseRequest,
    ACPSessionDetailResponse,
    ACPSessionForkRequest,
    ACPSessionForkResponse,
    ACPSessionInfo,
    ACPSessionListResponse,
    ACPSessionNewRequest,
    ACPSessionNewResponse,
    ACPSessionPromptRequest,
    ACPSessionPromptResponse,
    ACPSessionUpdatesResponse,
    ACPSessionUsageResponse,
    ACPTaskStatusResponse,
    ACPTokenUsage,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPGovernanceDeniedError,
    get_runner_client,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import classify_agent_entrypoint
from tldw_Server_API.app.services.acp_runtime_policy_service import ACPRuntimePolicyService
from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPResponseError
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import is_single_user_ip_allowed, resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.checkpoint_consumer import CheckpointConsumer
from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.sse_consumer import SSEConsumer
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus

router = APIRouter(prefix="/acp", tags=["acp"])

# ---------------------------------------------------------------------------
# Per-session event bus registry (shared across SSE, WS, and ACP runtime)
# ---------------------------------------------------------------------------

_SESSION_EVENT_BUSES: dict[str, SessionEventBus] = {}
_SESSION_EVENT_BUSES_LOCK = threading.Lock()


def register_session_event_bus(session_id: str, bus: SessionEventBus) -> None:
    """Register a session's event bus so SSE/WS consumers can find it.

    Called by the ACP runtime when a session starts.
    """
    with _SESSION_EVENT_BUSES_LOCK:
        _SESSION_EVENT_BUSES[session_id] = bus


def get_session_event_bus(session_id: str) -> SessionEventBus | None:
    """Return the event bus registered for *session_id*, or ``None``."""
    with _SESSION_EVENT_BUSES_LOCK:
        return _SESSION_EVENT_BUSES.get(session_id)


def get_or_create_session_event_bus(session_id: str) -> SessionEventBus:
    """Return the SessionEventBus for *session_id*, creating one if needed.

    Prefers a bus registered by the ACP runtime via
    :func:`register_session_event_bus`.  Falls back to creating a
    standalone bus so that SSE consumers work even if the runtime has
    not yet registered one.
    """
    with _SESSION_EVENT_BUSES_LOCK:
        bus = _SESSION_EVENT_BUSES.get(session_id)
        if bus is None:
            bus = SessionEventBus(session_id=session_id)
            _SESSION_EVENT_BUSES[session_id] = bus
        return bus


# ---------------------------------------------------------------------------
# Per-session checkpoint consumer registry
# ---------------------------------------------------------------------------

_CHECKPOINT_CONSUMERS: dict[str, CheckpointConsumer] = {}
_CHECKPOINT_CONSUMERS_LOCK = threading.Lock()

_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS = (
    ACPResponseError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TokenExpiredError,
    TypeError,
    UnicodeDecodeError,
    InvalidTokenError,
    ValueError,
)

_ACP_WS_QUOTA_LOCK = threading.Lock()
_ACP_WS_ACTIVE_TOTAL = 0
_ACP_WS_ACTIVE_BY_USER: dict[str, int] = {}
_ACP_WS_ACTIVE_BY_PERSONA: dict[str, int] = {}
_ACP_WS_ACTIVE_BY_SESSION: dict[str, int] = {}


_ACP_CONTROL_RATE_LIMITER = SlidingWindowLimiter()
_ACP_AUDIT_LOCK = threading.Lock()
_ACP_AUDIT_EVENTS: deque[dict[str, Any]] = deque(maxlen=5000)
_ACP_RECONCILIATION_LOCK = threading.Lock()
_ACP_RECONCILIATION: OrderedDict[str, dict[str, Any]] = OrderedDict()
_ACP_DIAGNOSTIC_REASON_MAP: dict[str, str] = {
    "acp_governance_blocked": "blocked",
    "governance_blocked": "blocked",
    "blocked": "blocked",
    "acp_timeout": "timed_out",
    "timeout": "timed_out",
    "timed_out": "timed_out",
    "cancelled": "cancelled",
    "cancelled_by_user": "cancelled",
    "validation_error": "failed_validation",
    "failed_validation": "failed_validation",
    "authz_error": "authz_error",
    "authorization_error": "authz_error",
    "permission_denied": "authz_error",
    "session_not_found": "authz_error",
    "retry_exhausted": "retry_exhausted",
    "failed_runtime": "failed_runtime",
    "runtime_error": "failed_runtime",
    "invariant_violation": "invariant_violation",
}
_ACP_SUPPORT_STATES = frozenset({
    "supported",
    "supported_with_caveats",
    "experimental",
    "documented_unverified",
    "unsupported",
})
_ACP_VERIFICATION_LEVELS = frozenset({
    "documented_only",
    "stub_smoke_tested",
    "live_e2e_tested",
    "sandbox_tested",
    "production_supported",
})


def get_acp_runtime_policy_service() -> ACPRuntimePolicyService:
    """Provide the ACP runtime policy service for endpoint dependency injection."""
    return ACPRuntimePolicyService()


async def _build_initial_runtime_policy_snapshot(
    *,
    session_store: Any,
    session_record: Any,
    user_id: int,
    runtime_policy_service: ACPRuntimePolicyService,
) -> Any:
    """Build and persist the initial ACP runtime policy snapshot for a new session."""
    snapshot = await runtime_policy_service.build_snapshot(
        session_record=session_record,
        user_id=int(user_id),
    )
    return await runtime_policy_service.persist_snapshot(
        session_store=session_store,
        snapshot=snapshot,
    )


def _acp_env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return int(default)


def _acp_control_rate_limit_per_minute() -> int:
    return _acp_env_int("ACP_CONTROL_RATE_LIMIT_PER_MINUTE", 240)


def _acp_reconciliation_max_entries() -> int:
    return max(1, _acp_env_int("ACP_RECONCILIATION_MAX_ENTRIES", 5000))


def _acp_enforce_control_rate_limit(*, user_id: int, action: str) -> None:
    limit = max(1, _acp_control_rate_limit_per_minute())
    key = f"acp:control:user:{int(user_id)}:{str(action).strip().lower() or 'unknown'}"
    allowed, retry_after = _ACP_CONTROL_RATE_LIMITER.allow(key, limit)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "rate_limited",
                "message": "ACP control surface rate limit exceeded",
                "retry_after_seconds": retry_after,
                "action": action,
            },
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_ACP_AUDIT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "args",
    "arguments",
    "authorization",
    "bearer",
    "command",
    "content",
    "cwd",
    "env",
    "environment",
    "mcp_servers",
    "message_content",
    "messages",
    "prompt",
    "token",
}
_ACP_AUDIT_SENSITIVE_MARKERS = (
    "api_key",
    "access_token",
    "authorization",
    "bearer ",
    "xoxb-",
    "sk-",
)


def _sanitize_audit_value(key: str, value: Any) -> Any:
    """Return a redacted audit-safe representation for one metadata value."""
    key_l = str(key).strip().lower()
    if key_l in _ACP_AUDIT_SENSITIVE_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): _sanitize_audit_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_audit_value(key_l, item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _ACP_AUDIT_SENSITIVE_MARKERS):
            return "[redacted]"
        if len(value) > 300:
            return f"{value[:300]}..."
    return value


def _sanitize_audit_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return audit metadata with sensitive keys, payloads, and long strings removed."""
    return {
        str(key): _sanitize_audit_value(str(key), value)
        for key, value in dict(metadata or {}).items()
    }


def _acp_record_audit_event(
    *,
    action: str,
    user_id: int,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized_metadata = _sanitize_audit_metadata(metadata)
    event = {
        "timestamp": _now_iso(),
        "action": str(action),
        "user_id": int(user_id),
        "session_id": str(session_id),
        "metadata": sanitized_metadata,
    }
    with _ACP_AUDIT_LOCK:
        _ACP_AUDIT_EVENTS.append(event)
    # Persist to SQLite audit DB (best-effort)
    try:
        from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import get_acp_audit_db
        audit_db = get_acp_audit_db()
        audit_db.record_event(
            action=action,
            user_id=user_id,
            session_id=session_id,
            metadata=sanitized_metadata,
        )
        # Flush when buffer reaches threshold to balance durability vs performance
        audit_db.flush_if_needed(threshold=10)
    except Exception:
        logger.warning("ACP audit persistence failed")
    logger.info(
        "ACP audit event action={} user_id={} session_id={}",
        event["action"],
        event["user_id"],
        event["session_id"],
    )
    return event


def _acp_list_audit_events(*, session_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    retention_days = 30
    now_seconds = time.time()
    try:
        from tldw_Server_API.app.core.DB_Management.ACP_Audit_DB import get_acp_audit_db

        audit_db = get_acp_audit_db()
        retention_days = max(0, int(getattr(audit_db, "_retention_days", retention_days)))
        audit_db.flush()
        events.extend(audit_db.query_events(session_id=session_id, limit=5000))
        events.extend(
            item for item in audit_db.get_hot_cache(session_id=session_id)
            if _acp_audit_event_within_retention(item, retention_days=retention_days, now=now_seconds)
        )
    except Exception:
        logger.warning("ACP audit DB read failed")
    with _ACP_AUDIT_LOCK:
        events.extend(
            dict(item)
            for item in _ACP_AUDIT_EVENTS
            if str(item.get("session_id")) == str(session_id)
            and _acp_audit_event_within_retention(item, retention_days=retention_days, now=now_seconds)
        )

    deduped: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for event in events:
        metadata = event.get("metadata")
        try:
            metadata_key = json.dumps(metadata or {}, sort_keys=True, default=str)
        except TypeError:
            metadata_key = str(metadata)
        key = (
            str(event.get("timestamp") or ""),
            str(event.get("action") or ""),
            str(event.get("session_id") or ""),
            int(event.get("user_id") or 0),
            metadata_key,
        )
        deduped[key] = dict(event)
    return sorted(deduped.values(), key=lambda item: str(item.get("timestamp") or ""))


def _acp_audit_event_timestamp_seconds(event: dict[str, Any]) -> float | None:
    """Best-effort conversion of an audit event timestamp to epoch seconds."""
    value = event.get("timestamp")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _acp_audit_event_within_retention(
    event: dict[str, Any],
    *,
    retention_days: int,
    now: float,
) -> bool:
    """Return true when an in-memory audit event is inside the retention window."""
    timestamp_seconds = _acp_audit_event_timestamp_seconds(event)
    if timestamp_seconds is None:
        return True
    cutoff = float(now) - (max(0, int(retention_days)) * 86400)
    return timestamp_seconds >= cutoff


def _is_agent_audit_scope(session_id: str) -> bool:
    """Return true when an audit id names an admin-managed agent resource."""
    return str(session_id).startswith("agent:")


def _acp_mark_reconciliation(
    *,
    session_id: str,
    status_value: str,
    reason_code: str,
    error: str | None = None,
) -> dict[str, Any]:
    key = str(session_id)
    payload = {
        "session_id": key,
        "status": str(status_value),
        "reason_code": str(reason_code),
        "error": str(error) if error else None,
        "updated_at": _now_iso(),
    }
    with _ACP_RECONCILIATION_LOCK:
        _ACP_RECONCILIATION.pop(key, None)
        _ACP_RECONCILIATION[key] = payload
        max_entries = _acp_reconciliation_max_entries()
        while len(_ACP_RECONCILIATION) > max_entries:
            _ACP_RECONCILIATION.popitem(last=False)
    return dict(payload)


def _acp_get_reconciliation(session_id: str) -> dict[str, Any]:
    with _ACP_RECONCILIATION_LOCK:
        existing = _ACP_RECONCILIATION.get(str(session_id))
        if existing:
            return dict(existing)
    return {
        "session_id": str(session_id),
        "status": "not_recorded",
        "reason_code": "not_recorded",
        "error": None,
        "updated_at": None,
    }


def _sanitize_diagnostic_message(message: Any) -> str:
    text = str(message or "").strip()
    if not text:
        return "No diagnostic message available"
    lowered = text.lower()
    redaction_markers = ("api_key", "access_token", "authorization", "bearer ", "xoxb-", "sk-")
    if any(marker in lowered for marker in redaction_markers):
        return "Diagnostic message redacted (sensitive content)"
    if len(text) > 300:
        return f"{text[:300]}..."
    return text


_ACP_REDACTED_VALUE = "[redacted]"
_ACP_REDACTION_SENSITIVE_KEYS = _ACP_AUDIT_SENSITIVE_KEYS | {
    "access_token",
    "api-token",
    "artifact_path",
    "auth",
    "content",
    "file_path",
    "openai_api_key",
    "password",
    "path",
    "payload",
    "raw_data",
    "raw_prompt",
    "raw_result",
    "refresh_token",
    "secret",
    "secrets",
    "text",
    "tool_arguments",
}
_ACP_REDACTION_MARKERS = _ACP_AUDIT_SENSITIVE_MARKERS + (
    "api-key",
    "password=",
    "secret=",
    "token=",
)


def _redact_acp_string(value: str) -> str:
    """Return a support-safe representation of one ACP string value."""
    text = str(value)
    lowered = text.lower()
    if any(marker in lowered for marker in _ACP_REDACTION_MARKERS):
        return _ACP_REDACTED_VALUE
    if (
        text.startswith("/")
        or text.startswith(".")
        or text.startswith("~")
        or text.startswith("\\\\")
        or "/" in text
        or ":\\" in text
        or ":/" in text
    ):
        return _ACP_REDACTED_VALUE
    if len(text) > 300:
        return f"{text[:300]}..."
    return text


def _redact_acp_value(value: Any, *, key: str = "") -> Any:
    """Recursively redact sensitive ACP payload data while preserving shape."""
    key_l = str(key).strip().lower()
    if key_l in _ACP_REDACTION_SENSITIVE_KEYS:
        return _ACP_REDACTED_VALUE
    if isinstance(value, dict):
        return {
            str(child_key): _redact_acp_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_acp_value(item, key=key_l) for item in value]
    if isinstance(value, str):
        return _redact_acp_string(value)
    return value


def _redact_acp_message(message: dict[str, Any]) -> dict[str, Any]:
    """Redact transcript-level prompt/result payloads for support-safe views."""
    redacted: dict[str, Any] = {
        "role": message.get("role"),
        "content": _ACP_REDACTED_VALUE if message.get("content") is not None else None,
        "timestamp": message.get("timestamp"),
    }
    for raw_key in ("raw_prompt", "raw_result", "raw_data"):
        if raw_key in message:
            redacted[raw_key] = _ACP_REDACTED_VALUE
    return redacted


def _redact_acp_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return redacted copies of ACP transcript messages."""
    return [
        _redact_acp_message(msg)
        for msg in messages
        if isinstance(msg, dict)
    ]


def _normalize_reason_code(raw_reason: Any, raw_message: Any) -> str:
    candidate = str(raw_reason or "").strip().lower()
    if candidate in _ACP_DIAGNOSTIC_REASON_MAP:
        return _ACP_DIAGNOSTIC_REASON_MAP[candidate]
    message_text = str(raw_message or "").strip().lower()
    for marker, normalized in _ACP_DIAGNOSTIC_REASON_MAP.items():
        if marker in message_text:
            return normalized
    return "failed_runtime"


def _extract_session_diagnostics(session_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, dict):
            continue
        raw_reason = content.get("reason_code") or content.get("error_type") or content.get("status")
        raw_message = content.get("error") or content.get("message") or content.get("detail")
        diagnostic_uri = (
            content.get("diagnostic_uri")
            or content.get("diagnostic_url")
            or content.get("artifact_url")
            or content.get("artifact_uri")
        )
        if raw_reason is None and raw_message is None and diagnostic_uri is None:
            continue
        diagnostics.append(
            {
                "session_id": str(session_id),
                "index": idx,
                "timestamp": msg.get("timestamp"),
                "role": msg.get("role"),
                "reason_code": _normalize_reason_code(raw_reason, raw_message),
                "message": _sanitize_diagnostic_message(raw_message),
                "diagnostic_uri": str(diagnostic_uri) if diagnostic_uri else None,
            }
        )
    return diagnostics


def _acp_ws_limit(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, str(default)))
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return int(default)


def _acp_quota_inc(bucket: dict[str, int], key: str | None) -> None:
    if key is None:
        return
    bucket[key] = int(bucket.get(key, 0)) + 1


def _acp_quota_dec(bucket: dict[str, int], key: str | None) -> None:
    if key is None:
        return
    current = int(bucket.get(key, 0))
    if current <= 1:
        bucket.pop(key, None)
    else:
        bucket[key] = current - 1


def _acp_ws_try_acquire_quota(
    *,
    user_id: int,
    session_id: str,
    persona_id: str | None,
) -> tuple[dict[str, str | None] | None, str | None]:
    global _ACP_WS_ACTIVE_TOTAL
    total_limit = _acp_ws_limit("ACP_WS_MAX_CONNECTIONS_TOTAL", 1024)
    per_user_limit = _acp_ws_limit("ACP_WS_MAX_CONNECTIONS_PER_USER", 64)
    per_persona_limit = _acp_ws_limit("ACP_WS_MAX_CONNECTIONS_PER_PERSONA", 32)
    per_session_limit = _acp_ws_limit("ACP_WS_MAX_CONNECTIONS_PER_SESSION", 16)

    user_key = str(user_id)
    session_key = str(session_id).strip() if session_id else None
    persona_key = str(persona_id).strip() if persona_id else None

    with _ACP_WS_QUOTA_LOCK:
        if total_limit > 0 and _ACP_WS_ACTIVE_TOTAL >= total_limit:
            return None, "total_connections_quota_exceeded"
        if per_user_limit > 0 and int(_ACP_WS_ACTIVE_BY_USER.get(user_key, 0)) >= per_user_limit:
            return None, "user_connections_quota_exceeded"
        if persona_key and per_persona_limit > 0 and int(_ACP_WS_ACTIVE_BY_PERSONA.get(persona_key, 0)) >= per_persona_limit:
            return None, "persona_connections_quota_exceeded"
        if session_key and per_session_limit > 0 and int(_ACP_WS_ACTIVE_BY_SESSION.get(session_key, 0)) >= per_session_limit:
            return None, "session_connections_quota_exceeded"

        _ACP_WS_ACTIVE_TOTAL += 1
        _acp_quota_inc(_ACP_WS_ACTIVE_BY_USER, user_key)
        _acp_quota_inc(_ACP_WS_ACTIVE_BY_PERSONA, persona_key)
        _acp_quota_inc(_ACP_WS_ACTIVE_BY_SESSION, session_key)
        return {
            "user_key": user_key,
            "persona_key": persona_key,
            "session_key": session_key,
        }, None


def _acp_ws_release_quota(token: dict[str, str | None] | None) -> None:
    global _ACP_WS_ACTIVE_TOTAL
    if not token:
        return
    with _ACP_WS_QUOTA_LOCK:
        if _ACP_WS_ACTIVE_TOTAL > 0:
            _ACP_WS_ACTIVE_TOTAL -= 1
        _acp_quota_dec(_ACP_WS_ACTIVE_BY_USER, token.get("user_key"))
        _acp_quota_dec(_ACP_WS_ACTIVE_BY_PERSONA, token.get("persona_key"))
        _acp_quota_dec(_ACP_WS_ACTIVE_BY_SESSION, token.get("session_key"))


async def _resolve_acp_session_persona_id(client: Any, session_id: str, user_id: int) -> str | None:
    getter = getattr(client, "get_session_metadata", None)
    if not callable(getter):
        return None
    try:
        metadata = await getter(session_id, user_id=user_id)
    except TypeError:
        metadata = await getter(session_id)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return None
    if not isinstance(metadata, dict):
        return None
    persona_id = metadata.get("persona_id")
    if persona_id is None:
        return None
    return str(persona_id)


def _governance_action(decision: dict[str, Any] | None) -> str:
    if not isinstance(decision, dict):
        return ""
    return str(decision.get("action") or decision.get("status") or "").strip().lower()


def _governance_blocked_detail(
    decision: dict[str, Any] | None,
    *,
    message: str = "Prompt blocked by governance policy",
) -> dict[str, Any]:
    payload = dict(decision or {}) if isinstance(decision, dict) else {}
    return {
        "code": "governance_blocked",
        "message": message,
        "governance": payload,
    }


async def _check_prompt_governance(
    client: Any,
    *,
    session_id: str,
    prompt: list[dict[str, Any]],
    user_id: int,
) -> dict[str, Any] | None:
    checker = getattr(client, "check_prompt_governance", None)
    if not callable(checker):
        return None

    decision: Any | None = None
    try:
        decision = checker(
            session_id,
            prompt,
            user_id=int(user_id),
            metadata={"session_id": session_id, "user_id": int(user_id)},
        )
        if inspect.isawaitable(decision):
            decision = await decision
    except TypeError:
        try:
            decision = checker(
                session_id,
                prompt,
                user_id=int(user_id),
            )
            if inspect.isawaitable(decision):
                decision = await decision
        except TypeError:
            try:
                decision = checker(session_id, prompt)
                if inspect.isawaitable(decision):
                    decision = await decision
            except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                return None
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return None

    if isinstance(decision, dict):
        return decision
    return None


# -----------------------------------------------------------------------------
# WebSocket Authentication Helper
# -----------------------------------------------------------------------------

class _AuthNZJWTManagerCompat:
    """Compatibility shim exposing verify_token() with token_data.user_id."""

    async def verify_token(self, token: str) -> SimpleNamespace | None:
        try:
            payload = get_jwt_service().decode_access_token(token)
            session_manager = await get_session_manager()
            if await session_manager.is_token_blacklisted(token, payload.get("jti")):
                return None
            user_id = payload.get("user_id") or payload.get("sub")
            if user_id is None:
                return None
            return SimpleNamespace(user_id=int(user_id))
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            return None


def get_jwt_manager() -> _AuthNZJWTManagerCompat:
    """Return a compatibility JWT manager for ACP WebSocket auth and legacy tests."""
    return _AuthNZJWTManagerCompat()


async def _authenticate_ws(
    websocket: WebSocket,
    token: str | None = None,
    api_key: str | None = None,
    required_scope: str = "read",
) -> int | None:
    """Authenticate a WebSocket connection. Returns user_id or None."""
    # Try JWT token first
    if token:
        try:
            jwtm = get_jwt_manager()
            token_data = jwtm.verify_token(token)
            if inspect.isawaitable(token_data):
                token_data = await token_data
            if token_data and getattr(token_data, "user_id", None):
                return int(token_data.user_id)
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("JWT auth failed for WebSocket: {}", e)

    # Try API key (single-user mode)
    if api_key:
        try:
            settings = get_auth_settings()
            auth_mode = str(getattr(settings, "AUTH_MODE", "single_user")).strip().lower()
            client_ip = resolve_client_ip(websocket, settings)
            if auth_mode == "single_user":
                allowed_keys: set[str] = set()
                primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
                if isinstance(primary_key, str) and primary_key.strip():
                    allowed_keys.add(primary_key.strip())
                env_primary = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")
                if isinstance(env_primary, str) and env_primary.strip():
                    allowed_keys.add(env_primary.strip())
                if is_explicit_pytest_runtime():
                    test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                    if isinstance(test_key, str) and test_key.strip():
                        allowed_keys.add(test_key.strip())
                if api_key in allowed_keys and is_single_user_ip_allowed(client_ip, settings):
                    return int(getattr(settings, "SINGLE_USER_FIXED_ID", 1))
            else:
                api_mgr = await get_api_key_manager()
                info = await api_mgr.validate_api_key(
                    api_key=api_key,
                    required_scope=required_scope,
                    ip_address=client_ip,
                )
                user_id = info.get("user_id") if isinstance(info, dict) else None
                if user_id is not None:
                    return int(user_id)
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("API key auth failed for WebSocket: {}", e)

    # Try Authorization header
    auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            return await _authenticate_ws(websocket, token=auth_header[7:].strip(), required_scope=required_scope)
        elif auth_header.lower().startswith("x-api-key "):
            return await _authenticate_ws(websocket, api_key=auth_header[10:].strip(), required_scope=required_scope)

    # Try Sec-WebSocket-Protocol: bearer,<token> or x-api-key,<key>
    proto_header = websocket.headers.get("sec-websocket-protocol") or websocket.headers.get("Sec-WebSocket-Protocol")
    if proto_header:
        parts = [p.strip() for p in proto_header.split(",") if p.strip()]
        for idx in range(len(parts) - 1):
            scheme = parts[idx].lower()
            value = parts[idx + 1]
            if scheme == "bearer" and value:
                return await _authenticate_ws(websocket, token=value, required_scope=required_scope)
            if scheme in {"x-api-key", "api-key"} and value:
                return await _authenticate_ws(websocket, api_key=value, required_scope=required_scope)

    return None


async def _record_acp_prompt(
    *,
    session_id: str,
    prompt: list[dict[str, Any]],
    result: dict[str, Any],
) -> ACPTokenUsage | None:
    try:
        store = await get_acp_session_store()
        turn_usage_data = await store.record_prompt(session_id, prompt, result)
        if turn_usage_data:
            return ACPTokenUsage(
                prompt_tokens=turn_usage_data.prompt_tokens,
                completion_tokens=turn_usage_data.completion_tokens,
                total_tokens=turn_usage_data.total_tokens,
            )
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to record prompt for session {}", session_id)
    return None


async def _prepare_acp_runtime_prompt(
    *,
    session_id: str,
    prompt: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    used_bootstrap = False
    try:
        store = await get_acp_session_store()
        builder = getattr(store, "build_bootstrap_prompt", None)
        if callable(builder):
            prompt, used_bootstrap = await builder(session_id, prompt)
    except ValueError as exc:
        raise ACPResponseError(str(exc)) from exc
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to prepare ACP bootstrap prompt for session {}", session_id)

    # Preprocess @tool_name mentions and attach hints to message metadata.
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.prompt_utils import preprocess_mentions

        prompt, tool_hints = await preprocess_mentions(prompt)
        if tool_hints and prompt:
            last_msg = prompt[-1]
            meta = last_msg.setdefault("metadata", {})
            meta["tool_hints"] = tool_hints
    except Exception:
        logger.debug("@mention preprocessing failed for session {}", session_id)

    return prompt, used_bootstrap


async def _clear_acp_bootstrap_state(session_id: str) -> None:
    try:
        store = await get_acp_session_store()
        clearer = getattr(store, "clear_bootstrap", None)
        if callable(clearer):
            await clearer(session_id)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to clear ACP bootstrap state for session {}", session_id)


async def _execute_acp_prompt(
    *,
    client: Any,
    session_id: str,
    prompt: list[dict[str, Any]],
    user_id: int,
    require_access_check: bool = True,
) -> tuple[dict[str, Any], ACPTokenUsage | None]:
    if require_access_check:
        await _require_session_access(client, session_id=session_id, user_id=int(user_id))

    runtime_prompt, used_bootstrap = await _prepare_acp_runtime_prompt(
        session_id=session_id,
        prompt=prompt,
    )

    try:
        await _check_prompt_governance(
            client,
            session_id=session_id,
            prompt=prompt,
            user_id=int(user_id),
        )
        result = await client.prompt(session_id, runtime_prompt)
    except ACPGovernanceDeniedError:
        _acp_record_audit_event(
            action="prompt_blocked",
            user_id=int(user_id),
            session_id=session_id,
            metadata={"reason_code": "governance_blocked"},
        )
        raise
    except ACPResponseError:
        _acp_record_audit_event(
            action="prompt_failed",
            user_id=int(user_id),
            session_id=session_id,
            metadata={"reason_code": "failed_runtime"},
        )
        raise

    turn_usage = await _record_acp_prompt(
        session_id=session_id,
        prompt=prompt,
        result=result,
    )
    if used_bootstrap:
        await _clear_acp_bootstrap_state(session_id)

    # Check token budget after recording usage — auto-terminate if exceeded
    budget_terminated = False
    try:
        store = await get_acp_session_store()
        budget_terminated = await store.check_and_enforce_budget(session_id)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("Budget enforcement check failed for session {}", session_id)
    if budget_terminated:
        # Inject budget exhaustion notice into the result
        result["budget_exhausted"] = True
        result["budget_termination_notice"] = (
            "Session auto-terminated: token budget exhausted"
        )
        _acp_record_audit_event(
            action="budget_terminated",
            user_id=int(user_id),
            session_id=session_id,
            metadata={"reason": "token_budget_exhausted"},
        )

    _acp_record_audit_event(
        action="prompt",
        user_id=int(user_id),
        session_id=session_id,
        metadata={"prompt_items": len(prompt)},
    )
    return result, turn_usage


async def _require_session_access(
    client: Any,
    *,
    session_id: str,
    user_id: int,
) -> None:
    """Require that the authenticated user owns the requested ACP session."""
    verifier = getattr(client, "verify_session_access", None)
    if not callable(verifier):
        logger.warning(
            "ACP session access denied: client {} does not expose verify_session_access()",
            type(client).__name__,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    allowed = await verifier(session_id, user_id)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")


# -----------------------------------------------------------------------------
# WebSocket Endpoint
# -----------------------------------------------------------------------------


@router.websocket("/sessions/{session_id}/stream")
async def acp_session_stream(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(None),
    api_key: str | None = Query(None),
    last_sequence: int = Query(0),
) -> None:
    """
    WebSocket endpoint for real-time ACP session updates.

    Query Parameters:
    - last_sequence: If > 0, replay buffered events from this sequence on
      reconnect before switching to live delivery.

    Message types (Server → Client):
    - connected: Connection established
    - update: Session update from agent
    - permission_request: Permission required for tool execution
    - error: Error occurred
    - prompt_complete: Prompt execution completed

    Message types (Client → Server):
    - permission_response: Approve/deny permission request
    - cancel: Cancel current operation
    - prompt: Send a new prompt (alternative to REST endpoint)

    Authentication:
    - Pass token as query param: ?token=<jwt>
    - Pass api_key as query param: ?api_key=<key>
    - Or via Authorization header: Bearer <token>
    """
    # Authenticate
    user_id = await _authenticate_ws(websocket, token=token, api_key=api_key, required_scope="write")
    if user_id is None:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4401)
        return

    try:
        client = await get_runner_client()
        await _require_session_access(client, session_id=session_id, user_id=user_id)
    except HTTPException:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4404)
        return
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4404)
        return

    # Set up WebSocket stream wrapper for metrics
    persona_id = await _resolve_acp_session_persona_id(client, session_id=session_id, user_id=int(user_id))
    ws_quota_token, _ws_quota_reason = _acp_ws_try_acquire_quota(
        user_id=int(user_id),
        session_id=session_id,
        persona_id=persona_id,
    )
    if ws_quota_token is None:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4429)
        return

    stream = WebSocketStream(
        websocket,
        heartbeat_interval_s=30.0,
        idle_timeout_s=None,  # No idle timeout for ACP sessions
        close_on_done=False,
        labels={"component": "acp", "endpoint": "acp_session_stream"},
    )
    send_callback: Any | None = None

    try:
        await stream.start()

        # Define send callback for broadcasting
        async def _send_callback(message: dict[str, Any]) -> None:
            await stream.send_json(message)
        send_callback = _send_callback

        # Register this WebSocket with the session.
        # NOTE: client.register_websocket uses the runner's internal
        # broadcast mechanism which does not currently expose a
        # from_sequence parameter.  Replay of missed events on
        # reconnect requires wiring a WSBroadcaster with the
        # session's shared event bus.  For now we pass
        # last_sequence through when a bus-backed broadcaster is
        # available; otherwise fall back to the existing path.
        bus = get_session_event_bus(session_id)
        if bus is not None and last_sequence > 0:
            from tldw_Server_API.app.core.Agent_Client_Protocol.consumers.ws_broadcaster import WSBroadcaster

            broadcaster = WSBroadcaster()
            await broadcaster.start(bus)

            async def _ws_send_str(msg: str) -> None:
                await stream.send_json(json.loads(msg))

            await broadcaster.add_connection(
                conn_id=f"ws-{session_id}-{id(stream)}",
                send_callback=_ws_send_str,
                from_sequence=last_sequence,
            )
        await client.register_websocket(session_id, send_callback)

        # Send connected message
        await stream.send_json({
            "type": "connected",
            "session_id": session_id,
            "last_sequence": last_sequence,
            "agent_capabilities": client.agent_capabilities,
        })

        logger.info("WebSocket connected for ACP session {} (user={})", session_id, user_id)

        # Main message loop
        while True:
            try:
                data = await stream.receive_json()
                await _handle_client_message(
                    client,
                    session_id,
                    data,
                    stream,
                    user_id=int(user_id),
                )
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected for ACP session {}", session_id)
                break
            except json.JSONDecodeError as e:
                await stream.send_json({
                    "type": "error",
                    "code": "invalid_json",
                    "message": f"Invalid JSON: {e}",
                    "session_id": session_id,
                })
            except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as e:
                logger.exception("Error handling WebSocket message for session {}", session_id)
                await stream.send_json({
                    "type": "error",
                    "code": "internal_error",
                    "message": str(e),
                    "session_id": session_id,
                })

    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.exception("WebSocket error for ACP session {}", session_id)
    finally:
        # Unregister WebSocket
        if send_callback is not None:
            try:
                client = await get_runner_client()
                await client.unregister_websocket(session_id, send_callback)
            except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                pass
        await stream.stop()
        _acp_ws_release_quota(ws_quota_token)


@router.websocket("/sessions/{session_id}/ssh")
async def acp_session_ssh(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(None),
    api_key: str | None = Query(None),
) -> None:
    """WebSocket SSH proxy for an ACP sandbox session."""
    user_id = await _authenticate_ws(websocket, token=token, api_key=api_key, required_scope="write")
    if user_id is None:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4401)
        return

    try:
        client = await get_runner_client()
        await _require_session_access(client, session_id=session_id, user_id=int(user_id))
        if not hasattr(client, "get_ssh_info"):
            await websocket.close(code=4404)
            return
        try:
            ssh_info = await client.get_ssh_info(session_id, user_id=user_id)
        except TypeError:
            ssh_info = await client.get_ssh_info(session_id)
        if not ssh_info:
            await websocket.close(code=4404)
            return
        ssh_host, ssh_port, ssh_user, ssh_key = ssh_info
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4404)
        return

    persona_id = await _resolve_acp_session_persona_id(client, session_id=session_id, user_id=int(user_id))
    ws_quota_token, _ws_quota_reason = _acp_ws_try_acquire_quota(
        user_id=int(user_id),
        session_id=session_id,
        persona_id=persona_id,
    )
    if ws_quota_token is None:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=4429)
        return

    await websocket.accept()
    ssh_proc: asyncio.subprocess.Process | None = None
    temp_key_path: str | None = None
    try:
        try:
            import asyncssh  # type: ignore
        except ImportError:
            asyncssh = None  # type: ignore[assignment]

        if asyncssh is not None:
            key = asyncssh.import_private_key(ssh_key)
            async with asyncssh.connect(
                ssh_host,
                port=int(ssh_port),
                username=ssh_user,
                client_keys=[key],
                known_hosts=None,
            ) as conn:
                process = await conn.create_process(term_type="xterm", term_size=(80, 24))

                async def _read_output(reader: Any) -> None:
                    while True:
                        data = await reader.read(4096)
                        if not data:
                            return
                        await websocket.send_bytes(data.encode() if isinstance(data, str) else data)

                async def _write_input() -> None:
                    while True:
                        try:
                            msg = await websocket.receive()
                        except WebSocketDisconnect:
                            return
                        if msg.get("type") == "websocket.disconnect":
                            return
                        if msg.get("text"):
                            text = msg["text"]
                            try:
                                payload = json.loads(text)
                            except json.JSONDecodeError:
                                payload = None
                            if isinstance(payload, dict) and payload.get("type") == "resize":
                                cols = int(payload.get("cols") or 0)
                                rows = int(payload.get("rows") or 0)
                                if cols > 0 and rows > 0:
                                    with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                                        process.set_term_size(cols, rows)
                                continue
                            process.stdin.write(text)
                            await process.stdin.drain()
                        elif msg.get("bytes"):
                            process.stdin.write(msg["bytes"])
                            await process.stdin.drain()

                await asyncio.gather(
                    _read_output(process.stdout),
                    _read_output(process.stderr),
                    _write_input(),
                )
        else:
            with tempfile.NamedTemporaryFile("w", delete=False, prefix="acp_ssh_", suffix="_key") as tmp_key:
                tmp_key.write(ssh_key)
                temp_key_path = tmp_key.name
            os.chmod(temp_key_path, 0o600)

            ssh_proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-i",
                temp_key_path,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "IdentitiesOnly=yes",
                "-p",
                str(ssh_port),
                f"{ssh_user}@{ssh_host}",
                "-tt",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def _read_output(reader: asyncio.StreamReader | None) -> None:
                if reader is None:
                    return
                while True:
                    data = await reader.read(4096)
                    if not data:
                        return
                    await websocket.send_bytes(data)

            async def _write_input() -> None:
                if ssh_proc is None or ssh_proc.stdin is None:
                    return
                while True:
                    try:
                        msg = await websocket.receive()
                    except WebSocketDisconnect:
                        return
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("text"):
                        text = msg["text"]
                        # The ssh fallback cannot resize PTY directly; ignore resize control messages.
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            payload = None
                        if isinstance(payload, dict) and payload.get("type") == "resize":
                            continue
                        ssh_proc.stdin.write(text.encode("utf-8"))
                        await ssh_proc.stdin.drain()
                    elif msg.get("bytes"):
                        ssh_proc.stdin.write(msg["bytes"])
                        await ssh_proc.stdin.drain()

            tasks = {
                asyncio.create_task(_read_output(ssh_proc.stdout)),
                asyncio.create_task(_read_output(ssh_proc.stderr)),
                asyncio.create_task(_write_input()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                    await task
            for task in done:
                with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                    _ = task.result()
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=1011)
    finally:
        if ssh_proc is not None and ssh_proc.returncode is None:
            ssh_proc.terminate()
            with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                await asyncio.wait_for(ssh_proc.wait(), timeout=2)
        if temp_key_path:
            with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                os.unlink(temp_key_path)
        _acp_ws_release_quota(ws_quota_token)

async def _handle_client_message(
    client: Any,
    session_id: str,
    data: dict[str, Any],
    stream: WebSocketStream,
    user_id: int | None = None,
) -> None:
    """Handle a message from the WebSocket client."""
    msg_type = data.get("type")

    if msg_type == "permission_response":
        request_id = data.get("request_id")
        approved = data.get("approved", False)
        batch_approve_tier = data.get("batch_approve_tier")

        if not request_id:
            await stream.send_json({
                "type": "error",
                "code": "missing_request_id",
                "message": "permission_response requires request_id",
                "session_id": session_id,
            })
            return

        pending_metadata: dict[str, Any] = {}
        metadata_getter = getattr(client, "get_pending_permission_metadata", None)
        if callable(metadata_getter):
            try:
                maybe_metadata = metadata_getter(session_id, request_id)
                if inspect.isawaitable(maybe_metadata):
                    maybe_metadata = await maybe_metadata
                if isinstance(maybe_metadata, dict):
                    pending_metadata = {str(k): v for k, v in maybe_metadata.items()}
            except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Failed to load ACP pending permission metadata for session {} request {}: {}",
                    session_id,
                    request_id,
                    exc,
                )

        success = await client.respond_to_permission(
            session_id,
            request_id,
            approved,
            batch_approve_tier,
        )
        logger.debug(
            "ACP permission response processed: session_id={} request_id={} approved={} success={}",
            session_id,
            request_id,
            approved,
            success,
        )
        if not success:
            # Compatibility fallback for lightweight/mock runner clients that
            # track pending permissions in a simple dict.
            with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                pending = getattr(client, "_pending_permissions", None)
                if isinstance(pending, dict):
                    sess_pending = pending.get(session_id)
                    if isinstance(sess_pending, dict) and request_id in sess_pending:
                        sess_pending.pop(request_id, None)
                        success = True
        if success and user_id is not None:
            audit_metadata = {
                "approved": bool(approved),
                "request_id": str(request_id),
                "batch_approve_tier": batch_approve_tier,
            }
            audit_metadata.update({k: v for k, v in pending_metadata.items() if v is not None})
            _acp_record_audit_event(
                action="permission_response",
                user_id=int(user_id),
                session_id=session_id,
                metadata=audit_metadata,
            )
        if not success:
            await stream.send_json({
                "type": "error",
                "code": "permission_not_found",
                "message": f"Permission request {request_id} not found",
                "session_id": session_id,
            })

    elif msg_type == "cancel":
        try:
            await client.cancel(session_id)
            await stream.send_json({
                "type": "update",
                "session_id": session_id,
                "update_type": "cancelled",
                "data": {"message": "Operation cancelled"},
            })
        except ACPResponseError as e:
            await stream.send_json({
                "type": "error",
                "code": "cancel_failed",
                "message": str(e),
                "session_id": session_id,
            })

    elif msg_type == "prompt":
        prompt = data.get("prompt", [])
        if not prompt:
            await stream.send_json({
                "type": "error",
                "code": "missing_prompt",
                "message": "prompt message requires prompt array",
                "session_id": session_id,
            })
            return

        try:
            result, turn_usage = await _execute_acp_prompt(
                client=client,
                session_id=session_id,
                prompt=prompt,
                user_id=int(user_id),
                require_access_check=False,
            )
            response: dict[str, Any] = {
                "type": "prompt_complete",
                "session_id": session_id,
                "stop_reason": result.get("stopReason"),
                "raw_result": result,
            }
            if turn_usage is not None:
                response["usage"] = turn_usage.model_dump()
            await stream.send_json(response)
        except ACPGovernanceDeniedError as e:
            payload = dict(getattr(e, "governance", {}) or {})
            await stream.send_json({
                "type": "error",
                "code": "governance_blocked",
                "message": str(e),
                "session_id": session_id,
                "data": {"governance": payload},
            })
        except ACPResponseError as e:
            await stream.send_json({
                "type": "error",
                "code": "prompt_failed",
                "message": str(e),
                "session_id": session_id,
            })

    else:
        await stream.send_json({
            "type": "error",
            "code": "unknown_message_type",
            "message": f"Unknown message type: {msg_type}",
            "session_id": session_id,
        })


# ---------------------------------------------------------------------------
# Health check & setup-guide helpers
# ---------------------------------------------------------------------------



def _check_runner_binary() -> dict[str, Any]:
    """Check if the ACP runner binary is available."""
    import shutil

    from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_runner_config

    try:
        cfg = load_acp_runner_config()
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
        return {"status": "error", "detail": f"Config load failed: {exc}"}

    # Check binary_path shortcut first
    if cfg.binary_path:
        path = os.path.expanduser(cfg.binary_path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return {"status": "ok", "path": path, "source": "binary_path"}
        return {
            "status": "missing",
            "detail": f"Binary not found at configured path: {cfg.binary_path}",
            "configured_path": cfg.binary_path,
        }

    # Check command-based config
    if not cfg.command:
        return {
            "status": "missing",
            "detail": "No runner_command or runner_binary_path configured in [ACP] section",
        }

    # If command is an absolute/relative path, check it directly
    if os.sep in cfg.command or cfg.command.startswith("."):
        resolved = cfg.command
        if cfg.cwd:
            resolved = os.path.join(cfg.cwd, cfg.command)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return {"status": "ok", "path": resolved, "source": "runner_command"}
        return {
            "status": "missing",
            "detail": f"Runner command not found: {cfg.command}",
            "configured_command": cfg.command,
            "configured_cwd": cfg.cwd,
        }

    # Check if command is on PATH (e.g., "go", "node", or a binary name)
    which_result = shutil.which(cfg.command)
    if which_result:
        return {"status": "ok", "path": which_result, "source": "PATH"}

    return {
        "status": "missing",
        "detail": f"Runner command '{cfg.command}' not found on PATH",
        "configured_command": cfg.command,
    }


_ACP_ENTRYPOINT_STRATEGIES = frozenset({
    "native_acp",
    "adapter_acp",
    "documented_candidate",
    "custom_template",
})
_ACP_PROBE_STATES = frozenset({
    "ready_to_probe",
    "blocked",
    "custom_template",
    "documented_only",
})
_ACP_ENTRYPOINT_STEP_MAP = {
    "adapter_required": "Select and install a concrete ACP adapter command before live certification.",
    "adapter_missing": "Select and install a concrete ACP adapter command before live certification.",
    "binary_missing": "Install the ACP entrypoint command and ensure it is on PATH.",
    "credentials_missing": "Set the required provider credential before live certification.",
    "entrypoint_strategy_missing": "Identify and configure a concrete ACP stdio entrypoint before live certification.",
    "shell_builtin_collision": "Use an executable ACP command, not a shell builtin or alias.",
    "custom_template": "Create a named custom ACP profile with command, args, env, workspace policy, and evidence bundle.",
}


def _normalize_docs_url(value: Any) -> str | None:
    """Return a docs URL using the served docs-static path for repo docs."""
    if value is None:
        return ACP_COMPATIBILITY_DOCS_URL
    docs_url = str(value)
    if docs_url.startswith("Docs/"):
        return f"/docs-static/{docs_url.removeprefix('Docs/')}"
    return docs_url


def _string_list(value: Any) -> list[str]:
    """Normalize arbitrary list-like values into strings."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _entrypoint_status_from_entry(reg_entry: Any) -> dict[str, Any]:
    """Build ACP entrypoint status from a registry entry classifier."""
    status = classify_agent_entrypoint(reg_entry).as_dict()
    status["docs_url"] = _normalize_docs_url(status.get("docs_url"))
    return status


def _entrypoint_status_from_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize runner/static ACP entrypoint metadata with conservative defaults."""
    raw_entrypoint = item.get("entrypoint")
    source = raw_entrypoint if isinstance(raw_entrypoint, dict) else item
    profile_key = str(
        source.get("profile_key")
        or item.get("profile_key")
        or item.get("type")
        or item.get("agent_type")
        or ""
    )

    strategy = str(
        source.get("entrypoint_strategy")
        or item.get("entrypoint_strategy")
        or "documented_candidate"
    )
    if strategy not in _ACP_ENTRYPOINT_STRATEGIES:
        strategy = "documented_candidate"

    default_probe_state = "custom_template" if strategy == "custom_template" else "documented_only"
    if strategy in {"native_acp", "adapter_acp"}:
        default_probe_state = "blocked"
    probe_state = str(source.get("probe_state") or item.get("probe_state") or default_probe_state)
    if probe_state not in _ACP_PROBE_STATES:
        probe_state = default_probe_state

    primary_blocker_raw = (
        source.get("primary_blocker")
        if source.get("primary_blocker") is not None
        else item.get("primary_blocker")
    )
    if primary_blocker_raw is None:
        primary_blocker_raw = source.get("certification_blocker") or item.get("certification_blocker")
    if primary_blocker_raw is None and strategy == "custom_template":
        primary_blocker_raw = "custom_template"
    primary_blocker = str(primary_blocker_raw) if primary_blocker_raw else None

    blockers = _string_list(source.get("blockers") or item.get("blockers"))
    if not blockers and primary_blocker:
        blockers = [primary_blocker]

    status_message = str(source.get("status_message") or item.get("status_message") or "")
    if not status_message:
        if primary_blocker == "custom_template":
            status_message = _ACP_ENTRYPOINT_STEP_MAP["custom_template"]
        elif strategy == "documented_candidate":
            status_message = "Agent is documented as a candidate and is not eligible for live ACP probing yet."
        elif probe_state == "blocked":
            status_message = "ACP entrypoint readiness is blocked until setup is complete."
        elif probe_state == "ready_to_probe":
            status_message = "Configured ACP entrypoint is ready for a bounded initialize probe."

    return {
        "profile_key": profile_key,
        "entrypoint_strategy": strategy,
        "probe_state": probe_state,
        "acp_command": str(source.get("acp_command") or item.get("acp_command") or ""),
        "acp_args": _string_list(source.get("acp_args") or item.get("acp_args")),
        "primary_blocker": primary_blocker,
        "blockers": blockers,
        "status_message": status_message,
        "docs_url": _normalize_docs_url(
            source.get("docs_url")
            or item.get("docs_url")
            or item.get("compatibility_docs_url")
            or ACP_COMPATIBILITY_DOCS_URL
        ),
    }


def _entrypoint_setup_steps(entrypoint: dict[str, Any]) -> list[str]:
    """Return setup guide steps for ACP entrypoint readiness blockers."""
    keys = []
    primary_blocker = entrypoint.get("primary_blocker")
    if primary_blocker:
        keys.append(str(primary_blocker))
    keys.extend(_string_list(entrypoint.get("blockers")))
    if entrypoint.get("entrypoint_strategy") == "custom_template":
        keys.append("custom_template")

    steps: list[str] = []
    for key in keys:
        step = _ACP_ENTRYPOINT_STEP_MAP.get(key)
        if step and step not in steps:
            steps.append(step)
    return steps


def _check_agent_availability(agent_type: str) -> dict[str, Any]:
    """Check if a downstream agent binary and API keys are available."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    entry = registry.get_entry(agent_type)
    if entry is None:
        return {
            "agent_type": agent_type,
            "status": "unknown",
            "binary_found": False,
            "api_key_set": False,
            "support_state": "documented_unverified",
            "verification_level": "documented_only",
            "compatibility_notes": "Agent is not present in the registry; live-agent ACP compatibility has not been certified.",
            "compatibility_docs_url": ACP_COMPATIBILITY_DOCS_URL,
            "entrypoint": _entrypoint_status_from_dict({"agent_type": agent_type}),
        }
    result = entry.check_availability()
    result["agent_type"] = agent_type
    result["entrypoint"] = _entrypoint_status_from_entry(entry)
    return result


def _build_agent_compatibility_status(source: Any) -> ACPAgentCompatibilityStatus:
    """Return ACP compatibility metadata with conservative defaults."""
    get_value = source.get if isinstance(source, dict) else lambda key, default=None: getattr(source, key, default)

    support_state = str(get_value("support_state") or "documented_unverified")
    verification_level = str(get_value("verification_level") or "documented_only")
    if support_state not in _ACP_SUPPORT_STATES:
        support_state = "documented_unverified"
    if verification_level not in _ACP_VERIFICATION_LEVELS:
        verification_level = "documented_only"
    notes = (
        get_value("compatibility_notes")
        or "Configured locally; live-agent ACP compatibility has not been certified."
    )
    docs_url = _normalize_docs_url(get_value("compatibility_docs_url") or ACP_COMPATIBILITY_DOCS_URL)
    return ACPAgentCompatibilityStatus(
        support_state=support_state,
        verification_level=verification_level,
        notes=str(notes),
        docs_url=str(docs_url),
    )


def _compatibility_setup_steps(compatibility: ACPAgentCompatibilityStatus) -> list[str]:
    """Return setup-guide actions for ACP compatibility certification state."""
    support_state = compatibility.support_state
    if support_state == "documented_unverified":
        return [
            "Run the ACP certification checklist before claiming live-agent support.",
            "Record evidence in the ACP compatibility matrix after live verification.",
        ]
    if support_state == "unsupported":
        return [
            "Do not claim ACP support for this agent until the compatibility issue is resolved.",
            "Open or link a follow-up issue before release if this agent must ship.",
        ]
    if support_state == "experimental":
        return ["Treat this ACP integration as experimental and review the compatibility caveats before release."]
    if support_state == "supported_with_caveats":
        return ["Review the ACP compatibility matrix caveats before release claims."]
    return []


@router.get(
    "/health",
    response_model=ACPHealthResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.health"))],
)
async def acp_health(
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """
    ACP dependency chain health check.

    Validates the full stack: runner binary → downstream agent availability → API keys.
    Returns structured diagnostics for debugging ACP setup issues.
    """
    result: dict[str, Any] = {"timestamp": _now_iso()}

    # 1. Check runner binary
    runner_status = _check_runner_binary()
    result["runner"] = runner_status

    # 2. Check downstream agents
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    agents_status: list[dict[str, Any]] = []
    for entry in registry.entries:
        avail = entry.check_availability()
        agents_status.append({
            "agent_type": entry.type,
            "name": entry.name,
            "status": avail.get("status", "unknown"),
            "binary_found": avail.get("binary_found", False),
            "api_key_set": avail.get("api_key_set", False),
            "is_configured": avail.get("is_configured", False),
            "support_state": avail.get("support_state", "documented_unverified"),
            "verification_level": avail.get("verification_level", "documented_only"),
            "compatibility_notes": avail.get("compatibility_notes", ""),
            "compatibility_docs_url": avail.get("compatibility_docs_url"),
            "entrypoint": _entrypoint_status_from_entry(entry),
        })
    result["agents"] = agents_status

    # 3. Try to probe the runner (if binary is available)
    runner_probe: dict[str, Any] = {"status": "skipped"}
    if runner_status.get("status") == "ok":
        try:
            client = await get_runner_client()
            if hasattr(client, "is_running") and client.is_running:
                runner_probe = {"status": "ok", "detail": "Runner process is alive"}
                caps = getattr(client, "agent_capabilities", None)
                if caps:
                    runner_probe["agent_capabilities"] = caps
            else:
                runner_probe = {"status": "not_running", "detail": "Runner process is not started"}
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
            runner_probe = {"status": "error", "detail": str(exc)}
    result["runner_probe"] = runner_probe

    # 4. Route-gating status
    try:
        from tldw_Server_API.app.core.config import route_enabled as _route_enabled

        _stable_only = False
        try:
            from tldw_Server_API.app.core.config import _route_toggle_policy
            _policy = _route_toggle_policy()
            _stable_only = bool(_policy.get("stable_only", False))
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            pass

        _acp_enabled = _route_enabled("acp", default_stable=False)
        _route_note: str | None = None
        if _stable_only and not _acp_enabled:
            _route_note = (
                "ACP routes are hidden because stable_only is enabled. "
                "Add 'acp' to the enable list in [API-Routes] section of "
                "config.txt, or set ROUTES_STABLE_ONLY=false."
            )
        result["routes"] = {
            "stable_only": _stable_only,
            "acp_enabled": _acp_enabled,
            "note": _route_note,
        }
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        result["routes"] = None

    # 5. Overall status
    any_agent_available = any(a.get("status") == "available" for a in agents_status)
    runner_ok = runner_status.get("status") == "ok"

    if runner_ok and any_agent_available:
        result["overall"] = "ok"
    elif runner_ok and not any_agent_available:
        result["overall"] = "degraded"
        result["message"] = "Runner binary found but no agents are fully configured"
    elif not runner_ok:
        result["overall"] = "unavailable"
        result["message"] = "ACP runner binary not found or not configured"
    else:
        result["overall"] = "degraded"

    return result


@router.get(
    "/setup-guide",
    response_model=ACPSetupGuideResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.setup_guide"))],
)
async def acp_setup_guide(
    agent_type: str | None = Query(default=None, description="Filter to a specific agent type"),
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """
    Return agent-specific setup instructions.

    Checks current system state and returns actionable steps to get ACP working.
    """
    result: dict[str, Any] = {"timestamp": _now_iso(), "guides": []}

    # Runner setup
    runner_status = _check_runner_binary()
    runner_guide: dict[str, Any] = {
        "component": "runner",
        "status": runner_status.get("status", "unknown"),
        "steps": [],
    }
    if runner_status.get("status") != "ok":
        runner_guide["steps"] = [
            "Download the ACP runner binary: run Helper_Scripts/setup_acp.sh",
            "Or build from source: cd ../tldw-agent && go build -o bin/tldw-agent-acp ./cmd/tldw-agent-acp",
            "Set runner_binary_path in config.txt [ACP] section, or set ACP_RUNNER_BINARY_PATH env var",
        ]
    else:
        runner_guide["steps"] = ["Runner binary is available - no action needed"]
    result["runner"] = runner_guide

    # Agent guides from registry
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    guides: list[dict[str, Any]] = []

    # Filter to a specific agent if requested and it exists in the registry
    matched_entry = registry.get_entry(agent_type) if agent_type else None
    target_entries = [matched_entry] if matched_entry else registry.entries

    for reg_entry in target_entries:
        avail = reg_entry.check_availability()
        entrypoint = _entrypoint_status_from_entry(reg_entry)
        guide_item: dict[str, Any] = {
            "agent_type": reg_entry.type,
            "name": reg_entry.name,
            "status": avail.get("status", "unknown"),
            "compatibility": _build_agent_compatibility_status(reg_entry),
            "entrypoint": entrypoint,
            "steps": [],
        }

        if not avail.get("binary_found", True):
            steps = list(reg_entry.install_instructions) if reg_entry.install_instructions else []
            if not steps:
                steps = [f"Install {reg_entry.name} and ensure the '{reg_entry.command}' command is available"]
            guide_item["steps"].extend(steps)

        if not avail.get("api_key_set", True) and reg_entry.requires_api_key:
            guide_item["steps"].append(f"Set {reg_entry.requires_api_key} environment variable or add to .env file")

        guide_item["steps"].extend(_entrypoint_setup_steps(entrypoint))
        guide_item["steps"].extend(_compatibility_setup_steps(guide_item["compatibility"]))

        if not guide_item["steps"]:
            guide_item["steps"] = [f"{reg_entry.name} is fully configured"]

        if reg_entry.docs_url:
            guide_item["docs_url"] = reg_entry.docs_url

        guides.append(guide_item)

    result["guides"] = guides
    return result


def _get_static_agents() -> tuple[list[ACPAgentInfo], str]:
    """Fallback list of built-in agents when runner registry is unavailable."""
    import os

    agents: list[ACPAgentInfo] = []

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    agents.append(
        ACPAgentInfo(
            type="claude_code",
            name="Claude Code",
            description="Anthropic's Claude Code agent for software development tasks",
            is_configured=bool(anthropic_key),
            requires_api_key="ANTHROPIC_API_KEY" if not anthropic_key else None,
            support_state="documented_unverified",
            verification_level="documented_only",
            compatibility_notes="Static fallback only; live Claude Code ACP compatibility has not been certified.",
            entrypoint=_entrypoint_status_from_dict({
                "type": "claude_code",
                "entrypoint_strategy": "documented_candidate",
                "certification_blocker": "adapter_required",
            }),
        )
    )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    agents.append(
        ACPAgentInfo(
            type="codex",
            name="OpenAI Codex",
            description="OpenAI's Codex agent for code generation and analysis",
            is_configured=bool(openai_key),
            requires_api_key="OPENAI_API_KEY" if not openai_key else None,
            support_state="documented_unverified",
            verification_level="documented_only",
            compatibility_notes="Static fallback only; live Codex ACP compatibility has not been certified.",
            entrypoint=_entrypoint_status_from_dict({
                "type": "codex",
                "entrypoint_strategy": "documented_candidate",
                "certification_blocker": "adapter_required",
            }),
        )
    )

    agents.append(
        ACPAgentInfo(
            type="opencode",
            name="OpenCode",
            description="Open-source coding agent (github.com/sst/opencode)",
            is_configured=True,
            requires_api_key=None,
            support_state="documented_unverified",
            verification_level="documented_only",
            compatibility_notes="Static fallback only; live OpenCode ACP compatibility has not been certified.",
            entrypoint=_entrypoint_status_from_dict({
                "type": "opencode",
                "entrypoint_strategy": "native_acp",
                "acp_command": "opencode",
                "acp_args": ["acp"],
                "primary_blocker": "binary_missing",
            }),
        )
    )

    agents.append(
        ACPAgentInfo(
            type="custom",
            name="Custom Agent",
            description="Configure a custom agent with your own settings",
            is_configured=True,
            requires_api_key=None,
            support_state="documented_unverified",
            verification_level="documented_only",
            compatibility_notes="Custom agent profiles require certification evidence before release claims.",
            entrypoint=_entrypoint_status_from_dict({
                "type": "custom",
                "entrypoint_strategy": "custom_template",
            }),
        )
    )

    return agents, "claude_code"


def _get_registry_agents() -> tuple[list[ACPAgentInfo], str] | None:
    """Try to get agents from YAML registry. Returns None if registry is unavailable."""
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry
        registry = get_agent_registry()
        available = registry.get_available_agents()
        if not available:
            return None
        agents: list[ACPAgentInfo] = []
        for item in available:
            compatibility = _build_agent_compatibility_status(item)
            reg_entry = registry.get_entry(str(item["type"]))
            entrypoint = (
                _entrypoint_status_from_entry(reg_entry)
                if reg_entry is not None
                else _entrypoint_status_from_dict(item)
            )
            agents.append(
                ACPAgentInfo(
                    type=str(item["type"]),
                    name=str(item["name"]),
                    description=str(item.get("description", "")),
                    is_configured=bool(item.get("is_configured", False)),
                    requires_api_key=item.get("missing_api_key"),
                    support_state=compatibility.support_state,
                    verification_level=compatibility.verification_level,
                    compatibility_notes=compatibility.notes,
                    compatibility_docs_url=compatibility.docs_url,
                    entrypoint=entrypoint,
                )
            )
        return agents, registry.default_type
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return None


async def _get_available_agents() -> tuple[list[ACPAgentInfo], str]:
    """Get list of available agents: registry → runner → static fallback."""
    # 1. Try YAML registry first
    registry_result = _get_registry_agents()
    if registry_result:
        return registry_result

    # 2. Try runner's agent/list RPC
    try:
        client = await get_runner_client()
        raw = await client.list_agents()
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return _get_static_agents()

    agents_raw = raw.get("agents", []) if isinstance(raw, dict) else []
    default_agent = raw.get("defaultAgentType") if isinstance(raw, dict) else None

    agents: list[ACPAgentInfo] = []
    for item in agents_raw:
        if not isinstance(item, dict):
            continue
        agent_type = item.get("type")
        name = item.get("name")
        if not agent_type or not name:
            continue
        is_configured = item.get("isConfigured")
        if is_configured is None:
            is_configured = item.get("is_configured", False)
        requires_api_key = item.get("requiresApiKey")
        if requires_api_key is None:
            requires_api_key = item.get("requires_api_key")
        compatibility = _build_agent_compatibility_status(item)
        try:
            agent_info = ACPAgentInfo(
                type=str(agent_type),
                name=str(name),
                description=str(item.get("description") or ""),
                is_configured=bool(is_configured),
                requires_api_key=str(requires_api_key) if requires_api_key else None,
                support_state=compatibility.support_state,
                verification_level=compatibility.verification_level,
                compatibility_notes=compatibility.notes,
                compatibility_docs_url=compatibility.docs_url,
                entrypoint=_entrypoint_status_from_dict(item),
            )
        except ValidationError:
            logger.warning("Skipping invalid ACP runner agent entry: {}", agent_type)
            continue
        agents.append(agent_info)

    if not agents:
        return _get_static_agents()

    default_value = str(default_agent) if default_agent else agents[0].type
    return agents, default_value


@router.get(
    "/agents",
    response_model=ACPAgentListResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.agents.list"))],
)
async def acp_list_agents(
    user: User = Depends(get_request_user),
) -> ACPAgentListResponse:
    """
    List available ACP agents and their configuration status.

    Returns information about which agents are available and properly configured.
    """
    agents, default_agent = await _get_available_agents()
    return ACPAgentListResponse(
        agents=agents,
        default_agent=default_agent,
    )


@router.post(
    "/agents/register",
    response_model=ACPAgentRegistrationResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.agents.register"))],
)
async def acp_register_agent(
    request: ACPAgentRegisterRequest,
    user: User = Depends(get_request_user),
) -> ACPAgentRegistrationResponse:
    """Register a new agent type dynamically (admin only)."""
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin role required for agent registration")

    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    entry = registry.register_agent(
        type=request.agent_type,
        name=request.name,
        command=request.command,
        description=request.description,
        args=request.args,
        env=request.env,
        requires_api_key=request.requires_api_key,
        install_instructions=request.install_instructions,
        docs_url=request.docs_url,
        mcp_orchestration=request.mcp_orchestration,
        mcp_entry_tool=request.mcp_entry_tool,
        mcp_structured_response=request.mcp_structured_response,
        mcp_llm_provider=request.mcp_llm_provider,
        mcp_llm_model=request.mcp_llm_model,
        mcp_max_iterations=request.mcp_max_iterations,
        mcp_refresh_tools=request.mcp_refresh_tools,
        entrypoint_strategy=request.entrypoint_strategy,
        acp_command=request.acp_command,
        acp_args=request.acp_args,
        adapter_source=request.adapter_source,
        adapter_docs_url=request.adapter_docs_url,
        certification_blocker=request.certification_blocker,
    )
    _acp_record_audit_event(
        action="agent_registered",
        user_id=int(user.id),
        session_id=f"agent:{entry.type}",
        metadata={
            "agent_type": entry.type,
            "mcp_orchestration": request.mcp_orchestration,
            "requires_api_key": bool(request.requires_api_key),
        },
    )
    return ACPAgentRegistrationResponse(status="registered", agent_type=entry.type, name=entry.name)


@router.delete(
    "/agents/{agent_type}",
    response_model=ACPAgentRegistrationResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.agents.manage"))],
)
async def acp_deregister_agent(
    agent_type: str,
    user: User = Depends(get_request_user),
) -> ACPAgentRegistrationResponse:
    """Remove a dynamically registered agent (admin only)."""
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin role required for agent management")

    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    removed = registry.deregister_agent(agent_type)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_type}' not found or is a YAML-defined agent",
        )
    _acp_record_audit_event(
        action="agent_deregistered",
        user_id=int(user.id),
        session_id=f"agent:{agent_type}",
        metadata={"agent_type": agent_type},
    )
    return ACPAgentRegistrationResponse(status="deregistered", agent_type=agent_type)


@router.put(
    "/agents/{agent_type}",
    response_model=ACPAgentRegistrationResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.agents.manage"))],
)
async def acp_update_agent(
    agent_type: str,
    request: ACPAgentUpdateRequest,
    user: User = Depends(get_request_user),
) -> ACPAgentRegistrationResponse:
    """Update a dynamically registered agent (admin only)."""
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin role required for agent management")

    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry

    registry = get_agent_registry()
    updates = request.model_dump(exclude_unset=True)
    entry = registry.update_agent(agent_type, **updates)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_type}' not found in dynamic registry",
        )
    _acp_record_audit_event(
        action="agent_updated",
        user_id=int(user.id),
        session_id=f"agent:{entry.type}",
        metadata={
            "agent_type": entry.type,
            "updated_fields": sorted(updates.keys()),
            "requires_api_key": "requires_api_key" in updates,
        },
    )
    return ACPAgentRegistrationResponse(status="updated", agent_type=entry.type, name=entry.name)


@router.get(
    "/agents/health",
    response_model=ACPAgentHealthResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.agents.health"))],
)
async def acp_agents_health(
    user: User = Depends(get_request_user),
) -> ACPAgentHealthResponse:
    """Get health status for all monitored agents."""
    import asyncio as _asyncio
    from tldw_Server_API.app.core.Agent_Client_Protocol.health_monitor import get_health_monitor

    monitor = get_health_monitor()
    statuses = monitor.get_all_statuses()

    # If no cached statuses, trigger a check on-demand
    if not statuses and monitor._registry is not None:
        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, monitor.check_all)
        statuses = monitor.get_all_statuses()

    return ACPAgentHealthResponse(
        agents=[
            ACPAgentHealthEntry(
                agent_type=s.agent_type,
                health=s.health,
                consecutive_failures=s.consecutive_failures,
                last_check=s.last_check,
                last_healthy=s.last_healthy,
                details=s.details,
            )
            for s in statuses
        ]
    )


def _generate_session_name(cwd: str) -> str:
    """Generate a session name from the working directory."""
    from datetime import datetime

    # Extract project name from cwd
    parts = cwd.rstrip("/").split("/")
    project_name = parts[-1] if parts else "Session"

    # Add time stamp
    time_str = datetime.now().strftime("%H:%M")

    return f"{project_name} ({time_str})"


@router.post(
    "/sessions/new",
    response_model=ACPSessionNewResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_new(
    payload: ACPSessionNewRequest,
    user: User = Depends(get_request_user),
) -> ACPSessionNewResponse:
    """
    Create a new ACP session.

    Optionally specify a session name, agent type, tags, and MCP server configs.
    """
    # Quota check: max concurrent sessions per user
    try:
        store = await get_acp_session_store()
        quota_error = await store.check_session_quota(int(user.id))
        if quota_error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=quota_error,
            )
    except HTTPException:
        raise
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Session quota check failed (non-blocking): {}", exc)

    # Generate session name if not provided
    session_name = payload.name or _generate_session_name(payload.cwd)
    runtime_policy_service = get_acp_runtime_policy_service()

    # Convert MCP server configs to dicts for the runner client
    mcp_servers_dicts = None
    if payload.mcp_servers is not None:
        mcp_servers_dicts = [
            server.model_dump(exclude_none=True) for server in payload.mcp_servers
        ]

    try:
        client = await get_runner_client()
        create_session_params = set(inspect.signature(client.create_session).parameters.keys())
        create_session_kwargs: dict[str, Any] = {}
        if payload.agent_type is not None and "agent_type" in create_session_params:
            create_session_kwargs["agent_type"] = payload.agent_type
        if "user_id" in create_session_params:
            create_session_kwargs["user_id"] = user.id
        optional_tenancy_args = (
            ("persona_id", payload.persona_id),
            ("workspace_id", payload.workspace_id),
            ("workspace_group_id", payload.workspace_group_id),
            ("scope_snapshot_id", payload.scope_snapshot_id),
        )
        for field_name, field_value in optional_tenancy_args:
            if field_value is not None and field_name in create_session_params:
                create_session_kwargs[field_name] = field_value
        session_id = await client.create_session(
            payload.cwd,
            mcp_servers_dicts,
            **create_session_kwargs,
        )
    except ACPResponseError as exc:
        logger.error("ACP session/new failed for user {}: {}", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create ACP session",
        ) from exc

    sandbox_meta = None
    try:
        if hasattr(client, "get_session_metadata"):
            try:
                sandbox_meta = await client.get_session_metadata(session_id, user_id=user.id)
            except TypeError:
                sandbox_meta = await client.get_session_metadata(session_id)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        sandbox_meta = None

    resolved_agent_type = payload.agent_type
    if resolved_agent_type is None:
        try:
            _, default_agent = await _get_available_agents()
            resolved_agent_type = default_agent
        except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            resolved_agent_type = "custom"
    resolved_persona_id = payload.persona_id
    resolved_workspace_id = payload.workspace_id
    resolved_workspace_group_id = payload.workspace_group_id
    resolved_scope_snapshot_id = payload.scope_snapshot_id
    if sandbox_meta:
        resolved_persona_id = resolved_persona_id or sandbox_meta.get("persona_id")
        resolved_workspace_id = resolved_workspace_id or sandbox_meta.get("workspace_id")
        resolved_workspace_group_id = resolved_workspace_group_id or sandbox_meta.get("workspace_group_id")
        resolved_scope_snapshot_id = resolved_scope_snapshot_id or sandbox_meta.get("scope_snapshot_id")

    # Resolve model from agent registry for cost tracking
    resolved_model: str | None = None
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import get_agent_registry
        _reg = get_agent_registry()
        _agent_entry = _reg.get_entry(resolved_agent_type or "custom")
        if _agent_entry is not None:
            resolved_model = _agent_entry.model or _agent_entry.mcp_llm_model
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        pass

    # Persist session metadata and emit SSE event
    persisted_record = None
    try:
        store = await get_acp_session_store()
        persisted_record = await store.register_session(
            session_id=session_id,
            user_id=int(user.id),
            agent_type=resolved_agent_type or "custom",
            name=session_name,
            cwd=payload.cwd,
            tags=payload.tags,
            mcp_servers=mcp_servers_dicts,
            persona_id=resolved_persona_id,
            workspace_id=resolved_workspace_id,
            workspace_group_id=resolved_workspace_group_id,
            scope_snapshot_id=resolved_scope_snapshot_id,
            model=resolved_model,
        )
        if persisted_record is not None:
            try:
                persisted_record = await _build_initial_runtime_policy_snapshot(
                    session_store=store,
                    session_record=persisted_record,
                    user_id=int(user.id),
                    runtime_policy_service=runtime_policy_service,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to build initial ACP runtime policy snapshot for {}: {}",
                    session_id,
                    exc,
                )
                persisted_record = await store.update_policy_snapshot_state(
                    session_id,
                    policy_snapshot_version=None,
                    policy_snapshot_fingerprint=None,
                    policy_snapshot_refreshed_at=None,
                    policy_summary=None,
                    policy_provenance_summary=None,
                    policy_refresh_error=str(exc),
                )
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to persist ACP session metadata for {}", session_id)
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin.admin_events_stream import emit_admin_event
        await emit_admin_event("acp_session_created", {
            "session_id": session_id,
            "user_id": int(user.id),
            "agent_type": resolved_agent_type or "custom",
            "name": session_name,
        }, category="acp")
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        pass
    _acp_record_audit_event(
        action="session_created",
        user_id=int(user.id),
        session_id=session_id,
        metadata={
            "agent_type": resolved_agent_type or "custom",
            "mcp_server_count": len(mcp_servers_dicts or []),
            "persona_id": resolved_persona_id,
            "workspace_id": resolved_workspace_id,
            "workspace_group_id": resolved_workspace_group_id,
            "scope_snapshot_id": resolved_scope_snapshot_id,
            "policy_snapshot_version": getattr(persisted_record, "policy_snapshot_version", None),
            "policy_snapshot_fingerprint": getattr(persisted_record, "policy_snapshot_fingerprint", None),
            "policy_refresh_failed": bool(getattr(persisted_record, "policy_refresh_error", None)),
        },
    )

    return ACPSessionNewResponse(
        session_id=session_id,
        name=session_name,
        agent_type=resolved_agent_type,
        agent_capabilities=client.agent_capabilities,
        sandbox_session_id=(sandbox_meta or {}).get("sandbox_session_id") if sandbox_meta else None,
        sandbox_run_id=(sandbox_meta or {}).get("sandbox_run_id") if sandbox_meta else None,
        ssh_ws_url=(sandbox_meta or {}).get("ssh_ws_url") if sandbox_meta else None,
        ssh_user=(sandbox_meta or {}).get("ssh_user") if sandbox_meta else None,
        persona_id=resolved_persona_id,
        workspace_id=resolved_workspace_id,
        workspace_group_id=resolved_workspace_group_id,
        scope_snapshot_id=resolved_scope_snapshot_id,
        policy_snapshot_version=getattr(persisted_record, "policy_snapshot_version", None),
        policy_snapshot_fingerprint=getattr(persisted_record, "policy_snapshot_fingerprint", None),
        policy_snapshot_refreshed_at=getattr(persisted_record, "policy_snapshot_refreshed_at", None),
        policy_summary=getattr(persisted_record, "policy_summary", None),
        policy_provenance_summary=getattr(persisted_record, "policy_provenance_summary", None),
        policy_refresh_error=getattr(persisted_record, "policy_refresh_error", None),
    )


@router.post(
    "/sessions/prompt",
    response_model=ACPSessionPromptResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_prompt(
    payload: ACPSessionPromptRequest,
    user: User = Depends(get_request_user),
) -> ACPSessionPromptResponse:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="prompt")
    # Token quota check
    try:
        store = await get_acp_session_store()
        token_error = await store.check_token_quota(payload.session_id)
        if token_error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=token_error,
            )
    except HTTPException:
        raise
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Token quota check failed (non-blocking): {}", exc)
    # Budget exhaustion pre-check — reject prompts on budget-exhausted sessions
    try:
        store = await get_acp_session_store()
        rec = await store.get_session(payload.session_id)
        if rec and rec.budget_exhausted:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "budget_exhausted",
                    "message": "Session token budget has been exhausted",
                    "token_budget": rec.token_budget,
                    "total_tokens": rec.usage.total_tokens,
                },
            )
    except HTTPException:
        raise
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Budget exhaustion pre-check failed (non-blocking): {}", exc)
    try:
        client = await get_runner_client()
        result, turn_usage = await _execute_acp_prompt(
            client=client,
            session_id=payload.session_id,
            prompt=payload.prompt,
            user_id=int(user.id),
        )
    except ACPGovernanceDeniedError as exc:
        logger.warning("ACP session/prompt blocked by governance for user {}: {}", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_governance_blocked_detail(
                getattr(exc, "governance", None),
                message=str(exc) or "Prompt blocked by governance policy",
            ),
        ) from exc
    except HTTPException:
        raise
    except ACPResponseError as exc:
        logger.error("ACP session/prompt failed for user {}", user.id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ACP prompt failed",
        ) from exc
    return ACPSessionPromptResponse(
        stop_reason=result.get("stopReason"),
        raw_result=result,
        usage=turn_usage,
    )


@router.post(
    "/sessions/cancel",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_cancel(
    payload: ACPSessionCancelRequest,
    user: User = Depends(get_request_user),
) -> dict:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="cancel")
    try:
        client = await get_runner_client()
        await _require_session_access(client, session_id=payload.session_id, user_id=int(user.id))
        await client.cancel(payload.session_id)
    except ACPResponseError as exc:
        logger.error("ACP session/cancel failed for user {}", user.id)
        _acp_record_audit_event(
            action="cancel_failed",
            user_id=int(user.id),
            session_id=payload.session_id,
            metadata={"reason_code": "failed_runtime", "message": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ACP session cancel failed",
        ) from exc
    _acp_record_audit_event(
        action="cancel",
        user_id=int(user.id),
        session_id=payload.session_id,
    )
    return {"status": "ok"}


@router.post(
    "/sessions/close",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_close(
    payload: ACPSessionCloseRequest,
    user: User = Depends(get_request_user),
) -> dict:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="close")
    _acp_mark_reconciliation(
        session_id=payload.session_id,
        status_value="teardown_started",
        reason_code="teardown_requested",
    )
    try:
        client = await get_runner_client()
        await _require_session_access(client, session_id=payload.session_id, user_id=int(user.id))
        await client.close_session(payload.session_id)
    except ACPResponseError as exc:
        logger.error("ACP session/close failed for user {}", user.id)
        _acp_mark_reconciliation(
            session_id=payload.session_id,
            status_value="teardown_failed",
            reason_code="failed_runtime",
            error=str(exc),
        )
        _acp_record_audit_event(
            action="close_failed",
            user_id=int(user.id),
            session_id=payload.session_id,
            metadata={"reason_code": "failed_runtime", "message": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ACP session close failed",
        ) from exc
    # Mark session as closed in store and emit SSE event
    try:
        store = await get_acp_session_store()
        await store.close_session(payload.session_id)
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        from tldw_Server_API.app.api.v1.endpoints.admin.admin_events_stream import emit_admin_event
        await emit_admin_event("acp_session_closed", {
            "session_id": payload.session_id,
            "user_id": int(user.id),
        }, category="acp")
    except _ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        pass
    _acp_mark_reconciliation(
        session_id=payload.session_id,
        status_value="teardown_completed",
        reason_code="success",
    )
    _acp_record_audit_event(
        action="close",
        user_id=int(user.id),
        session_id=payload.session_id,
    )
    return {"status": "ok"}


@router.post(
    "/sessions/{session_id}/teardown",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_teardown(
    session_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="teardown")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    _acp_mark_reconciliation(
        session_id=session_id,
        status_value="teardown_started",
        reason_code="teardown_requested",
    )
    try:
        await client.close_session(session_id)
    except ACPResponseError as exc:
        record = _acp_mark_reconciliation(
            session_id=session_id,
            status_value="teardown_failed",
            reason_code="failed_runtime",
            error="ACP session teardown failed",
        )
        _acp_record_audit_event(
            action="teardown_failed",
            user_id=int(user.id),
            session_id=session_id,
            metadata={"reason_code": "failed_runtime", "message": str(exc)},
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=record) from exc

    with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
        store = await get_acp_session_store()
        await store.close_session(session_id)
    record = _acp_mark_reconciliation(
        session_id=session_id,
        status_value="teardown_completed",
        reason_code="success",
    )
    _acp_record_audit_event(
        action="teardown",
        user_id=int(user.id),
        session_id=session_id,
    )
    return {"status": "ok", "reconciliation": record}


@router.get(
    "/sessions/{session_id}/reconciliation",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_reconciliation(
    session_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="reconciliation")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    _acp_record_audit_event(
        action="reconciliation_query",
        user_id=int(user.id),
        session_id=session_id,
    )
    return {"session_id": session_id, "reconciliation": _acp_get_reconciliation(session_id)}


@router.post(
    "/sessions/{session_id}/reconcile",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_reconcile(
    session_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="reconcile")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    current = _acp_get_reconciliation(session_id)
    if str(current.get("status")) in {"teardown_completed", "reconciled"}:
        _acp_record_audit_event(
            action="reconcile_noop",
            user_id=int(user.id),
            session_id=session_id,
        )
        return {"status": "ok", "reconciliation": current}
    try:
        await client.close_session(session_id)
    except ACPResponseError as exc:
        updated = _acp_mark_reconciliation(
            session_id=session_id,
            status_value="reconcile_failed",
            reason_code="failed_runtime",
            error="ACP session reconcile failed",
        )
        _acp_record_audit_event(
            action="reconcile_failed",
            user_id=int(user.id),
            session_id=session_id,
            metadata={"reason_code": "failed_runtime", "message": str(exc)},
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=updated) from exc

    with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
        store = await get_acp_session_store()
        await store.close_session(session_id)
    updated = _acp_mark_reconciliation(
        session_id=session_id,
        status_value="reconciled",
        reason_code="success",
    )
    _acp_record_audit_event(
        action="reconcile",
        user_id=int(user.id),
        session_id=session_id,
    )
    return {"status": "ok", "reconciliation": updated}


@router.get(
    "/sessions/{session_id}/updates",
    response_model=ACPSessionUpdatesResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_updates(
    session_id: str,
    limit: int | None = Query(default=100, ge=1, le=1000),
    user: User = Depends(get_request_user),
) -> ACPSessionUpdatesResponse:
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="updates")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    updates = client.pop_updates(session_id, limit=limit or 100)
    _acp_record_audit_event(
        action="updates_query",
        user_id=int(user.id),
        session_id=session_id,
        metadata={"limit": int(limit or 100)},
    )
    return ACPSessionUpdatesResponse(updates=updates)


# -----------------------------------------------------------------------------
# Session Listing & Detail Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/sessions",
    response_model=ACPSessionListResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_list_sessions(
    status_filter: str | None = Query(default=None, alias="status"),
    agent_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_request_user),
) -> ACPSessionListResponse:
    """List ACP sessions for the authenticated user."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="list_sessions")
    store = await get_acp_session_store()
    client = await get_runner_client()
    records, total = await store.list_sessions(
        user_id=int(user.id),
        status=status_filter,
        agent_type=agent_type,
        limit=limit,
        offset=offset,
    )
    sessions = [
        ACPSessionInfo(**rec.to_info_dict(
            has_websocket=client.has_websocket_connections(rec.session_id),
        ))
        for rec in records
    ]
    return ACPSessionListResponse(
        sessions=sessions,
        total=total,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(sessions),
        ),
    )


@router.get(
    "/sessions/{session_id}/detail",
    response_model=ACPSessionDetailResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_detail(
    session_id: str,
    redacted: bool = Query(
        default=False,
        description="Return a support-safe redacted transcript view",
    ),
    user: User = Depends(get_request_user),
) -> ACPSessionDetailResponse:
    """Get detailed information about an ACP session."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="session_detail")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    fork_lineage = await store.get_fork_lineage(session_id)
    detail = rec.to_detail_dict(
        has_websocket=client.has_websocket_connections(session_id),
        fork_lineage=fork_lineage,
    )
    if redacted:
        detail["messages"] = _redact_acp_messages(detail.get("messages") or [])
        detail["cwd"] = _ACP_REDACTED_VALUE if detail.get("cwd") else detail.get("cwd")
    return ACPSessionDetailResponse(**detail)


@router.get(
    "/sessions/{session_id}/usage",
    response_model=ACPSessionUsageResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_usage(
    session_id: str,
    user: User = Depends(get_request_user),
) -> ACPSessionUsageResponse:
    """Get token usage for an ACP session."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="usage")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    _acp_record_audit_event(
        action="usage_query",
        user_id=int(user.id),
        session_id=session_id,
    )
    return ACPSessionUsageResponse(
        session_id=rec.session_id,
        user_id=rec.user_id,
        agent_type=rec.agent_type,
        usage=ACPTokenUsage(**rec.usage.to_dict()),
        message_count=rec.message_count,
        created_at=rec.created_at,
        last_activity_at=rec.last_activity_at,
    )


# -----------------------------------------------------------------------------
# Session Events & Artifacts Query
# -----------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/events",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_events(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    redacted: bool = Query(
        default=False,
        description="Return support-safe redacted event payloads",
    ),
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Query persisted ACP session events/messages."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="events")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    events: list[dict[str, Any]] = []
    for idx, msg in enumerate(getattr(rec, "messages", []) or []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        raw_reason = (
            content.get("reason_code") or content.get("error_type") or content.get("status")
            if isinstance(content, dict)
            else None
        )
        raw_error = (
            content.get("error") if isinstance(content, dict) else None
        ) or (
            content.get("message") if isinstance(content, dict) else None
        )
        data = _redact_acp_value(content, key="data") if redacted else content
        reason_code = _normalize_reason_code(raw_reason, raw_error) if isinstance(content, dict) else None
        if redacted and content is not None and not isinstance(content, dict):
            data = _ACP_REDACTED_VALUE
        event = {
            "index": idx,
            "event_type": "message",
            "role": msg.get("role"),
            "timestamp": msg.get("timestamp"),
            "data": data,
            "reason_code": reason_code,
        }
        events.append(event)

    total = len(events)
    sliced = events[offset:offset + limit]
    _acp_record_audit_event(
        action="events_query",
        user_id=int(user.id),
        session_id=session_id,
        metadata={"limit": int(limit), "offset": int(offset)},
    )
    pagination = build_offset_pagination_meta(
        total=total,
        limit=limit,
        offset=offset,
        count=len(sliced),
    )
    return {
        "session_id": session_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "pagination": pagination.model_dump(mode="json"),
        "has_more": pagination.has_more,
        "next_offset": pagination.next_offset,
        "events": sliced,
    }


# -----------------------------------------------------------------------------
# Session Events SSE Stream
# -----------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/events/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "ACP session event stream.",
            "content": {"text/event-stream": {}},
        },
    },
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_events_stream(
    request: Request,
    session_id: str,
    last_event_id: int = Query(0, description="Replay events from this sequence number"),
    user: User = Depends(get_request_user),
) -> StreamingResponse:
    """Stream ACP session events as Server-Sent Events.

    Events are formatted as typed SSE frames with ``event: {kind}`` and
    ``data: {json}`` fields.  A heartbeat comment is sent every 15 seconds
    to keep the connection alive through proxies and load balancers.

    Use the *last_event_id* query parameter to replay buffered events from
    a given sequence number (e.g. for reconnection catch-up).
    """
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="events_stream")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))

    # Verify session exists in the store
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    bus = get_or_create_session_event_bus(session_id)
    consumer = SSEConsumer(
        from_sequence=max(0, last_event_id),
        heartbeat_interval=15.0,
    )
    await consumer.start(bus)

    _acp_record_audit_event(
        action="events_stream_connect",
        user_id=int(user.id),
        session_id=session_id,
        metadata={"last_event_id": int(last_event_id), "consumer_id": consumer.consumer_id},
    )

    async def _event_generator():
        try:
            async for line in consumer.iter_sse_lines():
                if await request.is_disconnected():
                    break
                yield line
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            await consumer.stop()

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/sessions/{session_id}/artifacts",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_artifacts(
    session_id: str,
    redacted: bool = Query(
        default=False,
        description="Return support-safe redacted artifact payloads",
    ),
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Query artifacts emitted in ACP session messages."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="artifacts")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    artifacts: list[dict[str, Any]] = []
    for msg in getattr(rec, "messages", []) or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, dict):
            continue
        listed = content.get("artifacts")
        if isinstance(listed, list):
            for artifact in listed:
                if isinstance(artifact, dict):
                    artifact_payload = dict(artifact)
                    artifacts.append(
                        _redact_acp_value(artifact_payload, key="artifact")
                        if redacted
                        else artifact_payload
                    )
        single = content.get("artifact")
        if isinstance(single, dict):
            single_payload = dict(single)
            artifacts.append(
                _redact_acp_value(single_payload, key="artifact")
                if redacted
                else single_payload
            )

    _acp_record_audit_event(
        action="artifacts_query",
        user_id=int(user.id),
        session_id=session_id,
    )
    return {
        "session_id": session_id,
        "total": len(artifacts),
        "artifacts": artifacts,
    }


@router.get(
    "/sessions/{session_id}/diagnostics",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_diagnostics(
    session_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Return normalized, non-sensitive diagnostics for an ACP session."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="diagnostics")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    diagnostics = _extract_session_diagnostics(session_id, list(getattr(rec, "messages", []) or []))
    reconciliation = _acp_get_reconciliation(session_id)
    _acp_record_audit_event(
        action="diagnostics_query",
        user_id=int(user.id),
        session_id=session_id,
    )
    return {
        "session_id": session_id,
        "total": len(diagnostics),
        "diagnostics": diagnostics,
        "reconciliation": reconciliation,
    }


@router.get(
    "/sessions/{session_id}/audit",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.read"))],
)
async def acp_session_audit(
    session_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Return ACP audit trail for a session."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="audit")
    if _is_agent_audit_scope(session_id):
        if not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin role required for agent audit")
    else:
        client = await get_runner_client()
        await _require_session_access(client, session_id=session_id, user_id=int(user.id))
    events = _acp_list_audit_events(session_id=session_id)
    return {
        "session_id": session_id,
        "total": len(events),
        "events": events,
    }


# -----------------------------------------------------------------------------
# Session Forking
# -----------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/fork",
    response_model=ACPSessionForkResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_fork(
    session_id: str,
    payload: ACPSessionForkRequest,
    user: User = Depends(get_request_user),
) -> ACPSessionForkResponse:
    """Fork an ACP session from a specific message index."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="fork")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))

    store = await get_acp_session_store()
    source = await store.get_session(session_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    if payload.message_index >= len(source.messages):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"message_index {payload.message_index} exceeds message count {len(source.messages)}",
        )

    fork_messages = list(source.messages[:payload.message_index + 1])
    if not source.cwd or any(
        not isinstance(message, dict)
        or not str(message.get("role") or "").strip()
        or not isinstance(message.get("content"), str)
        or not str(message.get("content") or "").strip()
        for message in fork_messages
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="fork_not_resumable")

    create_session_params = set(inspect.signature(client.create_session).parameters.keys())
    create_session_kwargs: dict[str, Any] = {}
    if source.agent_type and "agent_type" in create_session_params:
        create_session_kwargs["agent_type"] = source.agent_type
    if "user_id" in create_session_params:
        create_session_kwargs["user_id"] = int(user.id)
    optional_tenancy_args = (
        ("persona_id", source.persona_id),
        ("workspace_id", source.workspace_id),
        ("workspace_group_id", source.workspace_group_id),
        ("scope_snapshot_id", source.scope_snapshot_id),
    )
    for field_name, field_value in optional_tenancy_args:
        if field_value is not None and field_name in create_session_params:
            create_session_kwargs[field_name] = field_value

    try:
        new_session_id = await client.create_session(
            source.cwd,
            list(source.mcp_servers),
            **create_session_kwargs,
        )
    except ACPResponseError as exc:
        logger.error("ACP session/fork create_session failed for user {}: {}", user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create forked ACP session",
        ) from exc

    forked = await store.fork_session(
        source_session_id=session_id,
        new_session_id=new_session_id,
        message_index=payload.message_index,
        user_id=int(user.id),
        name=payload.name,
    )
    if not forked:
        closer = getattr(client, "close_session", None)
        if callable(closer):
            with contextlib.suppress(_ACP_ENDPOINT_NONCRITICAL_EXCEPTIONS):
                await closer(new_session_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="fork_failed")

    return ACPSessionForkResponse(
        session_id=forked.session_id,
        name=forked.name,
        forked_from=session_id,
        message_count=forked.message_count,
    )


# ---------------------------------------------------------------------------
# Async fire-and-forget API (Scheduler-backed)
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/prompt-async",
    response_model=ACPAsyncPromptResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.prompt_async"))],
)
async def acp_prompt_async(
    body: ACPAsyncPromptRequest,
    user: User = Depends(get_request_user),
) -> ACPAsyncPromptResponse:
    """Submit an ACP prompt for asynchronous execution.

    Returns a ``task_id`` that can be polled via
    ``GET /api/v1/acp/tasks/{task_id}`` for status and results.

    Tasks are submitted to the global Scheduler and persisted in the
    database, so they survive process restarts and are visible from
    any worker.
    """
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="prompt_async")

    payload: dict[str, Any] = {
        "user_id": int(user.id),
        "prompt": body.prompt,
        "cwd": body.cwd,
    }
    if body.agent_type is not None:
        payload["agent_type"] = body.agent_type
    if body.persona_id is not None:
        payload["persona_id"] = body.persona_id
    if body.workspace_id is not None:
        payload["workspace_id"] = body.workspace_id

    from tldw_Server_API.app.core.Scheduler import get_global_scheduler

    scheduler = await get_global_scheduler()
    task_id = await scheduler.submit(
        handler="acp_run",
        payload=payload,
        queue_name="acp",
        metadata={"user_id": str(user.id)},
    )

    poll_url = f"/api/v1/acp/tasks/{task_id}"
    return ACPAsyncPromptResponse(task_id=task_id, poll_url=poll_url, status="queued")


@router.get(
    "/tasks/{task_id}",
    response_model=ACPTaskStatusResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.tasks.status"))],
)
async def acp_task_status(
    task_id: str,
    user: User = Depends(get_request_user),
) -> ACPTaskStatusResponse:
    """Poll for the status and result of an async ACP task."""
    from tldw_Server_API.app.core.Scheduler import get_global_scheduler

    scheduler = await get_global_scheduler()
    task = await scheduler.get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found",
        )

    # Verify ownership via task metadata
    task_user_id = (task.metadata or {}).get("user_id")
    if task_user_id != str(user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found",
        )

    # Map scheduler TaskStatus to the API status string
    status_map = {
        "pending": "queued",
        "queued": "queued",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "failed",
        "dead": "failed",
    }
    task_status_str = status_map.get(task.status.value, task.status.value)

    result_data = task.result if isinstance(task.result, dict) else None
    return ACPTaskStatusResponse(
        task_id=task_id,
        status=task_status_str,
        result=result_data.get("result") if result_data else None,
        usage=result_data.get("usage", {}) if result_data else {},
        error=task.error or (result_data.get("error") if result_data else None),
        duration_ms=result_data.get("duration_ms") if result_data else None,
    )


# ---------------------------------------------------------------------------
# Run history & cost tracking
# ---------------------------------------------------------------------------


@router.get(
    "/runs",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.runs.list"))],
)
async def list_acp_runs(
    user: User = Depends(get_request_user),
    status_filter: str | None = Query(None, alias="status", description="Filter by session status (active, closed, error)"),
    agent_type: str | None = Query(None, description="Filter by agent type"),
    from_date: str | None = Query(None, description="ISO date lower bound, e.g. 2026-01-01"),
    to_date: str | None = Query(None, description="ISO date upper bound, e.g. 2026-12-31"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Query ACP run history with optional filters."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="list_runs")
    store = await get_acp_session_store()
    result = await store.list_runs(
        user_id=int(user.id),
        status=status_filter,
        agent_type=agent_type,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )
    items = result.get("items", [])
    total = int(result.get("total", len(items)))
    page_limit = int(result.get("limit", limit))
    page_offset = int(result.get("offset", offset))
    pagination = build_offset_pagination_meta(
        total=total,
        limit=page_limit,
        offset=page_offset,
        count=len(items),
    )
    return {
        **result,
        "total": total,
        "limit": page_limit,
        "offset": page_offset,
        "pagination": pagination.model_dump(mode="json"),
        "has_more": pagination.has_more,
        "next_offset": pagination.next_offset,
    }


@router.get(
    "/runs/aggregate",
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.runs.aggregate"))],
)
async def aggregate_acp_runs(
    user: User = Depends(get_request_user),
    from_date: str | None = Query(None, description="ISO date lower bound"),
    to_date: str | None = Query(None, description="ISO date upper bound"),
) -> dict:
    """Aggregate token usage and costs across ACP runs."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="aggregate_runs")
    store = await get_acp_session_store()
    return await store.aggregate_runs(
        user_id=int(user.id),
        from_date=from_date,
        to_date=to_date,
    )


# -----------------------------------------------------------------------------
# Checkpoint-based Rollback
# -----------------------------------------------------------------------------


def _get_checkpoint_consumer(session_id: str) -> CheckpointConsumer | None:
    """Return the CheckpointConsumer for *session_id*, if one exists."""
    with _CHECKPOINT_CONSUMERS_LOCK:
        return _CHECKPOINT_CONSUMERS.get(session_id)


async def _ensure_checkpoint_consumer(session_id: str) -> CheckpointConsumer:
    """Return or create a CheckpointConsumer for *session_id*.

    Lazily creates the consumer and starts it on the session's event bus.
    The SandboxService is instantiated on demand so the endpoint module
    does not import it at module level.
    """
    with _CHECKPOINT_CONSUMERS_LOCK:
        consumer = _CHECKPOINT_CONSUMERS.get(session_id)
        if consumer is not None:
            return consumer

    # Build consumer outside the lock (SandboxService init may be expensive)
    try:
        from tldw_Server_API.app.core.Sandbox.service import SandboxService
        svc = SandboxService()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sandbox service unavailable",
        ) from exc

    consumer = CheckpointConsumer(sandbox_service=svc, session_id=session_id)
    bus = get_or_create_session_event_bus(session_id)
    await consumer.start(bus)

    with _CHECKPOINT_CONSUMERS_LOCK:
        # Double-check: another request may have raced us
        existing = _CHECKPOINT_CONSUMERS.get(session_id)
        if existing is not None:
            await consumer.stop()
            return existing
        _CHECKPOINT_CONSUMERS[session_id] = consumer

    return consumer


@router.post(
    "/sessions/{session_id}/rollback",
    response_model=ACPRollbackResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_rollback(
    session_id: str,
    body: ACPRollbackRequest,
    user: User = Depends(get_request_user),
) -> ACPRollbackResponse:
    """Rollback sandbox state to a checkpoint.

    Provide either ``to_sequence`` (finds nearest checkpoint at or before
    that sequence) or ``to_snapshot_id`` (restores directly).  Only works
    for sandbox-backed sessions with an active CheckpointConsumer.
    """
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="rollback")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))

    if body.to_sequence is None and body.to_snapshot_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either to_sequence or to_snapshot_id",
        )

    # Resolve snapshot_id
    snapshot_id: str | None = body.to_snapshot_id
    resolved_sequence: int | None = None

    consumer = _get_checkpoint_consumer(session_id)

    if snapshot_id is None:
        # Must resolve from sequence via the consumer
        if consumer is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No checkpoint consumer active for this session; provide to_snapshot_id directly or ensure checkpointing is enabled",
            )
        if body.to_sequence is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either to_sequence or to_snapshot_id",
            )
        result = consumer.get_nearest_checkpoint(body.to_sequence)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No checkpoint found at or before sequence {body.to_sequence}",
            )
        resolved_sequence, snapshot_id = result

    if snapshot_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Checkpoint snapshot not found",
        )

    # Perform the restore via SandboxService
    try:
        from tldw_Server_API.app.core.Sandbox.service import SandboxService
        svc = SandboxService()
        restored = svc.restore_snapshot(session_id, snapshot_id)
    except Exception as exc:
        logger.error(
            "Rollback failed for session {} snapshot {}: {}",
            session_id, snapshot_id, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Rollback failed",
        ) from exc

    if not restored:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="restore_snapshot returned False",
        )

    # Emit LIFECYCLE event on the bus
    from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind

    bus = get_or_create_session_event_bus(session_id)
    await bus.publish(AgentEvent(
        session_id=session_id,
        kind=AgentEventKind.LIFECYCLE,
        payload={
            "action": "rollback",
            "snapshot_id": snapshot_id,
            "to_sequence": resolved_sequence,
        },
    ))

    _acp_record_audit_event(
        action="rollback",
        user_id=int(user.id),
        session_id=session_id,
        metadata={"snapshot_id": snapshot_id, "to_sequence": resolved_sequence},
    )

    return ACPRollbackResponse(
        restored=True,
        snapshot_id=snapshot_id,
        sequence=resolved_sequence,
    )


@router.get(
    "/sessions/{session_id}/checkpoints",
    response_model=ACPCheckpointListResponse,
    dependencies=[Depends(TokenScopeGuard("any", require_if_present=True, endpoint_id="acp.sessions.manage"))],
)
async def acp_session_checkpoints(
    session_id: str,
    user: User = Depends(get_request_user),
) -> ACPCheckpointListResponse:
    """List available checkpoints for a sandbox-backed session."""
    _acp_enforce_control_rate_limit(user_id=int(user.id), action="list_checkpoints")
    client = await get_runner_client()
    await _require_session_access(client, session_id=session_id, user_id=int(user.id))

    consumer = _get_checkpoint_consumer(session_id)
    checkpoints: dict[int, str] = {}
    if consumer is not None:
        checkpoints = consumer.get_checkpoints()

    return ACPCheckpointListResponse(
        session_id=session_id,
        checkpoints=checkpoints,
    )
