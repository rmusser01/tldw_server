from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import mimetypes
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole, User, get_request_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.sandbox_service import sandbox_service
from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    ArtifactListResponse,
    CancelResponse,
    SandboxAdminIdempotencyItem,
    SandboxAdminIdempotencyListResponse,
    SandboxAdminMacOSDiagnosticsResponse,
    SandboxAdminMacOSImageStoreCleanupPlanResponse,
    SandboxAdminMacOSImageStoreCleanupRequest,
    SandboxAdminMacOSImageStoreCleanupResponse,
    SandboxAdminMacOSReconciliationRepairRequest,
    SandboxAdminMacOSReconciliationRepairResponse,
    SandboxAdminRunDetails,
    SandboxAdminRunListResponse,
    SandboxAdminRunSummary,
    SandboxAdminRuntimeDiagnosticsResponse,
    SandboxAdminUsageItem,
    SandboxAdminUsageResponse,
    SandboxFileUploadResponse,
    SandboxRunCreateRequest,
    SandboxRunStatus,
    SandboxRunStatusReasonDetails,
    SandboxRuntimesResponse,
    SandboxSession,
    SandboxSessionCreateRequest,
    SessionCloneRequest,
    SessionCloneResponse,
    SnapshotCreateResponse,
    SnapshotInfo,
    SnapshotListResponse,
    SnapshotRestoreRequest,
    SnapshotRestoreResponse,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
    AuditSeverity,
)
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import is_single_user_ip_allowed, resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
from tldw_Server_API.app.core.Sandbox.audit_metadata import build_run_completion_audit_metadata
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, SessionSpec
from tldw_Server_API.app.core.Sandbox.models import RuntimeType as CoreRuntimeType
from tldw_Server_API.app.core.Sandbox.models import TrustLevel as CoreTrustLevel
from tldw_Server_API.app.core.Sandbox.orchestrator import (
    IdempotencyConflict,
    QueueFull,
    SessionActiveRunsConflict,
)
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    normalize_run_status_reason,
    run_status_reason_details,
)
from tldw_Server_API.app.core.Sandbox.service import (
    SandboxImageStoreCleanupError,
    SandboxReconciliationRepairError,
    SandboxService,
)
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.testing import (
    is_explicit_pytest_runtime,
    is_test_mode,
    is_truthy,
)
from tldw_Server_API.app.core.Utils.path_utils import safe_join

_SANDBOX_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
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
    IdempotencyConflict,
    QueueFull,
    SandboxPolicy.RuntimeUnavailable,
    SandboxPolicy.PolicyUnsupported,
    SandboxService.InvalidSpecVersion,
    SandboxService.InvalidFirecrackerConfig,
)


class SandboxArtifactGuardRoute(APIRoute):
    """APIRoute that rejects unsafe artifact paths using raw ASGI path.

    This guard runs before dependency resolution/endpoint execution and inspects
    `scope['raw_path']` so traversal attempts aren't hidden by path normalization.
    Only affects sandbox artifact download URLs.
    """

    def get_route_handler(self):
        original = super().get_route_handler()

        async def custom_handler(request: Request):
            try:
                raw_path = request.scope.get("raw_path")
                path_raw = raw_path.decode("utf-8", "ignore") if isinstance(raw_path, (bytes, bytearray)) else (request.url.path or "")
                # Reject traversal early for any sandbox runs path if raw_path reveals '..'
                if "/api/v1/sandbox/runs/" in path_raw and "/../" in path_raw:
                    return JSONResponse({"detail": "invalid_path"}, status_code=400)
                if "/api/v1/sandbox/runs/" in path_raw and "/artifacts/" in path_raw:
                    import posixpath as _pp
                    from urllib.parse import unquote
                    idx = path_raw.find("/artifacts/")
                    tail = path_raw[idx + len("/artifacts/"):]
                    tail_unquoted = unquote(tail)
                    if (
                        ".." in tail_unquoted.split("/")
                        or tail_unquoted.startswith("/")
                        or "//" in tail
                        or _pp.normpath(tail_unquoted) != tail_unquoted
                    ):
                        return JSONResponse({"detail": "invalid_path"}, status_code=400)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                # Fail open on guard errors
                pass
            return await original(request)

        return custom_handler


router = APIRouter(prefix="/sandbox", tags=["sandbox"], route_class=SandboxArtifactGuardRoute)

_service = sandbox_service


def _status_reason_code(
    *,
    phase: RunPhase | str | None,
    message: str | None,
    exit_code: int | str | None,
    resource_usage: Mapping[str, Any] | None,
) -> str:
    return normalize_run_status_reason(
        phase=phase,
        message=message,
        exit_code=exit_code,
        resource_usage=resource_usage if isinstance(resource_usage, dict) else None,
    )


@lru_cache(maxsize=32)
def _cached_status_reason_details(
    code: str,
) -> SandboxRunStatusReasonDetails:
    """Cache schema metadata for the finite normalized status reason vocabulary."""

    return SandboxRunStatusReasonDetails.model_validate(run_status_reason_details(code))


def _status_reason_details(
    code: str | None,
) -> SandboxRunStatusReasonDetails:
    """Build the public schema metadata for a normalized status reason code."""

    return _cached_status_reason_details(str(code or "unknown").strip())


@router.on_event("startup")
async def _sandbox_startup() -> None:
    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
        _service.start_background_tasks()


@router.on_event("shutdown")
async def _sandbox_shutdown() -> None:
    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
        _service.shutdown()

_SANDBOX_WS_QUOTA_LOCK = threading.Lock()
_SANDBOX_WS_ACTIVE_TOTAL = 0
_SANDBOX_WS_ACTIVE_BY_USER: dict[str, int] = {}
_SANDBOX_WS_ACTIVE_BY_PERSONA: dict[str, int] = {}
_SANDBOX_WS_ACTIVE_BY_SESSION: dict[str, int] = {}
_SANDBOX_WS_ACTIVE_BY_RUN: dict[str, int] = {}


def _sandbox_ws_limit(env_key: str, settings_attr: str, default: int) -> int:
    try:
        raw = os.getenv(env_key)
        if raw is None:
            raw = getattr(app_settings, settings_attr, default)
        return int(raw)
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return int(default)


def _quota_inc(bucket: dict[str, int], key: str | None) -> None:
    if key is None:
        return
    bucket[key] = int(bucket.get(key, 0)) + 1


def _quota_dec(bucket: dict[str, int], key: str | None) -> None:
    if key is None:
        return
    current = int(bucket.get(key, 0))
    if current <= 1:
        bucket.pop(key, None)
    else:
        bucket[key] = current - 1


def _sandbox_ws_try_acquire_quota(
    *,
    user_id: int,
    run_id: str,
    persona_id: str | None,
    session_id: str | None,
) -> tuple[dict[str, str | None] | None, str | None]:
    global _SANDBOX_WS_ACTIVE_TOTAL
    total_limit = _sandbox_ws_limit("SANDBOX_WS_MAX_CONNECTIONS_TOTAL", "SANDBOX_WS_MAX_CONNECTIONS_TOTAL", 1024)
    per_user_limit = _sandbox_ws_limit("SANDBOX_WS_MAX_CONNECTIONS_PER_USER", "SANDBOX_WS_MAX_CONNECTIONS_PER_USER", 64)
    per_persona_limit = _sandbox_ws_limit("SANDBOX_WS_MAX_CONNECTIONS_PER_PERSONA", "SANDBOX_WS_MAX_CONNECTIONS_PER_PERSONA", 32)
    per_session_limit = _sandbox_ws_limit("SANDBOX_WS_MAX_CONNECTIONS_PER_SESSION", "SANDBOX_WS_MAX_CONNECTIONS_PER_SESSION", 16)
    per_run_limit = _sandbox_ws_limit("SANDBOX_WS_MAX_CONNECTIONS_PER_RUN", "SANDBOX_WS_MAX_CONNECTIONS_PER_RUN", 8)

    user_key = str(user_id)
    persona_key = str(persona_id).strip() if persona_id else None
    session_key = str(session_id).strip() if session_id else None
    run_key = str(run_id).strip() if run_id else None

    with _SANDBOX_WS_QUOTA_LOCK:
        if total_limit > 0 and total_limit <= _SANDBOX_WS_ACTIVE_TOTAL:
            return None, "total_connections_quota_exceeded"
        if per_user_limit > 0 and int(_SANDBOX_WS_ACTIVE_BY_USER.get(user_key, 0)) >= per_user_limit:
            return None, "user_connections_quota_exceeded"
        if persona_key and per_persona_limit > 0 and int(_SANDBOX_WS_ACTIVE_BY_PERSONA.get(persona_key, 0)) >= per_persona_limit:
            return None, "persona_connections_quota_exceeded"
        if session_key and per_session_limit > 0 and int(_SANDBOX_WS_ACTIVE_BY_SESSION.get(session_key, 0)) >= per_session_limit:
            return None, "session_connections_quota_exceeded"
        if run_key and per_run_limit > 0 and int(_SANDBOX_WS_ACTIVE_BY_RUN.get(run_key, 0)) >= per_run_limit:
            return None, "run_connections_quota_exceeded"

        _SANDBOX_WS_ACTIVE_TOTAL += 1
        _quota_inc(_SANDBOX_WS_ACTIVE_BY_USER, user_key)
        _quota_inc(_SANDBOX_WS_ACTIVE_BY_PERSONA, persona_key)
        _quota_inc(_SANDBOX_WS_ACTIVE_BY_SESSION, session_key)
        _quota_inc(_SANDBOX_WS_ACTIVE_BY_RUN, run_key)
        return {
            "user_key": user_key,
            "persona_key": persona_key,
            "session_key": session_key,
            "run_key": run_key,
        }, None


def _sandbox_ws_release_quota(token: dict[str, str | None] | None) -> None:
    global _SANDBOX_WS_ACTIVE_TOTAL
    if not token:
        return
    with _SANDBOX_WS_QUOTA_LOCK:
        if _SANDBOX_WS_ACTIVE_TOTAL > 0:
            _SANDBOX_WS_ACTIVE_TOTAL -= 1
        _quota_dec(_SANDBOX_WS_ACTIVE_BY_USER, token.get("user_key"))
        _quota_dec(_SANDBOX_WS_ACTIVE_BY_PERSONA, token.get("persona_key"))
        _quota_dec(_SANDBOX_WS_ACTIVE_BY_SESSION, token.get("session_key"))
        _quota_dec(_SANDBOX_WS_ACTIVE_BY_RUN, token.get("run_key"))


def _is_admin_user(user: User) -> bool:
    try:
        roles = {
            str(role).strip().lower()
            for role in (getattr(user, "roles", []) or [])
            if str(role).strip()
        }
        permissions = {
            str(perm).strip().lower()
            for perm in (getattr(user, "permissions", []) or [])
            if str(perm).strip()
        }
        if "admin" in roles:
            return True
        if "*" in permissions:
            return True
        if "system.configure" in permissions:
            return True
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass
    return False


def _require_run_owner(run_id: str, current_user: User) -> str:
    owner = _service._orch.get_run_owner(run_id)  # type: ignore[attr-defined]
    if owner is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    if not _is_admin_user(current_user) and str(owner) != str(current_user.id):
        raise HTTPException(status_code=404, detail="run_not_found")
    return str(owner)


def _require_session_owner(session_id: str, current_user: User) -> str:
    owner = _service._orch.get_session_owner(session_id)  # type: ignore[attr-defined]
    if owner is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    if not _is_admin_user(current_user) and str(owner) != str(current_user.id):
        raise HTTPException(status_code=404, detail="session_not_found")
    return str(owner)


def _model_field_names(model: object | None) -> set[str]:
    if model is None:
        return set()
    fields = getattr(model, "model_fields_set", None)
    if fields is None:
        fields = getattr(model, "__fields_set__", set())
    return {str(field) for field in (fields or set())}


def _workspace_usage_bytes_sync(root: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                if os.path.isfile(full_path):
                    total += int(os.path.getsize(full_path))
    return total


def _existing_file_size_sync(target: str) -> int:
    try:
        if os.path.isfile(target):
            return int(os.path.getsize(target))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return 0
    return 0


def _prepare_temp_target_sync(target: str) -> tuple[int, str, int]:
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    existing_size = _existing_file_size_sync(target)
    fd, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(target)}.",
        suffix=".upload",
    )
    return fd, temp_path, existing_size


def _write_fd_chunk_sync(fd: int, chunk: bytes) -> None:
    os.write(fd, chunk)


def _finalize_temp_target_sync(fd: int, temp_path: str, target: str) -> None:
    try:
        os.close(fd)
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass
    os.replace(temp_path, target)


def _cleanup_temp_target_sync(fd: int, temp_path: str) -> None:
    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
        os.close(fd)
    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
        os.unlink(temp_path)


def _make_directory_sync(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stream_reader_to_target_sync(
    target: str,
    read_chunk,
    *,
    chunk_size: int,
    workspace_used: int,
    cap_bytes: int,
) -> tuple[int, int]:
    fd, temp_path, existing_size = _prepare_temp_target_sync(target)
    file_bytes = 0
    try:
        while True:
            chunk = read_chunk(chunk_size)
            if not chunk:
                break
            next_total = workspace_used - existing_size + file_bytes + len(chunk)
            if next_total > cap_bytes:
                raise HTTPException(status_code=413, detail="workspace_cap_exceeded")
            _write_fd_chunk_sync(fd, chunk)
            file_bytes += len(chunk)
        _finalize_temp_target_sync(fd, temp_path, target)
        return file_bytes, workspace_used - existing_size + file_bytes
    except Exception:
        _cleanup_temp_target_sync(fd, temp_path)
        raise


def _looks_like_jwt(token: str | None) -> bool:
    return isinstance(token, str) and token.count(".") == 2


async def _resolve_sandbox_ws_user_id(
    websocket: WebSocket,
    *,
    token: str | None,
    api_key: str | None,
) -> int:
    if not token:
        auth_hdr = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
        if auth_hdr and auth_hdr.lower().startswith("bearer "):
            token = auth_hdr.split(" ", 1)[1].strip()
    if not api_key:
        api_key = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-KEY")

    if token and not api_key and not _looks_like_jwt(token):
        api_key = token
        token = None

    if token:
        try:
            from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
            from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager

            jwt_service = get_jwt_service()
            payload = await jwt_service.verify_token_async(token, token_type="access")  # nosec B106
            session_manager = await get_session_manager()
            if await session_manager.is_token_blacklisted(token, payload.get("jti")):
                raise HTTPException(status_code=401, detail="invalid_token")
        except HTTPException:
            raise
        except (InvalidTokenError, TokenExpiredError):
            raise HTTPException(status_code=401, detail="invalid_token") from None
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(status_code=401, detail="invalid_token") from None

        sub = payload.get("user_id") or payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=401, detail="invalid_token")
        try:
            return int(sub)
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(status_code=401, detail="invalid_token") from None

    if api_key:
        settings = get_settings()
        client_ip = resolve_client_ip(websocket, settings)
        if getattr(settings, "AUTH_MODE", None) == "single_user":
            allowed_keys: set[str] = set()
            primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
            if primary_key:
                allowed_keys.add(primary_key)
            if is_explicit_pytest_runtime():
                test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                if test_key:
                    allowed_keys.add(test_key)
            if api_key in allowed_keys and is_single_user_ip_allowed(client_ip, settings):
                return int(getattr(settings, "SINGLE_USER_FIXED_ID", 1))
            raise HTTPException(status_code=401, detail="invalid_api_key")

        api_mgr = await get_api_key_manager()
        info = await api_mgr.validate_api_key(api_key=api_key, required_scope="read", ip_address=client_ip)
        if not info:
            raise HTTPException(status_code=401, detail="invalid_api_key")
        user_id = info.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="invalid_api_key")
        return int(user_id)

    raise HTTPException(status_code=401, detail="auth_required")


def _normalize_reason(outcome: str, message: str | None) -> str:
    """Normalize a possibly long/unique error message into a small, bounded set for metrics labels.

    Prefers bucketing into a fixed taxonomy to prevent Prometheus label cardinality explosions.
    If no mapping applies, falls back to the outcome string.
    """
    try:
        o = (outcome or "").strip().lower() or "unknown"
        msg = (message or "").strip().lower()

        # Outcome-driven buckets
        if o in {"timeout", "timed_out"}:
            return "timeout"
        if o in {"killed"}:
            return "killed"
        if o in {"success", "completed"}:
            return "success"
        if o in {"failed", "error", "internal"} and not msg:
            return "internal"

        # Message-driven buckets (substring checks)
        if msg:
            if "out of memory" in msg or "oom" in msg or "memory limit" in msg:
                return "oom"
            if "validation" in msg or "invalid" in msg or "schema" in msg:
                return "validation_error"
            if "unauthorized" in msg or "forbidden" in msg or "permission denied" in msg:
                return "permission_denied"
            if "rate limit" in msg or "too many requests" in msg or "status code 429" in msg:
                return "rate_limited"
            if "queue full" in msg:
                return "queue_full"
            if "deadline exceeded" in msg:
                return "timeout"
            if "sigkill" in msg or "signal 9" in msg:
                return "killed"
            if "not found" in msg or "no such file" in msg:
                return "not_found"
            if "image pull" in msg or "manifest" in msg:
                return "image_error"
            if "network timeout" in msg or "connection timed out" in msg:
                return "timeout"
            if "cancelled" in msg or "canceled" in msg:
                return "killed"

        # Generic fallback buckets
        if o in {"failed", "error"}:
            return "internal"
        return o or "other"
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return "other"


def _runtime_name_from_policy_exception(exc: Exception) -> str:
    rt_attr = getattr(exc, "runtime", None)
    if rt_attr is None:
        return "unknown"
    try:
        return rt_attr.value if hasattr(rt_attr, "value") else str(rt_attr)
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return str(rt_attr) if rt_attr is not None else "unknown"


def _runtime_unavailable_suggestions(runtime_name: str) -> list[str]:
    suggestions: list[str] = []
    try:
        from tldw_Server_API.app.core.Sandbox.runners.docker_runner import docker_available as _dock_avail
        from tldw_Server_API.app.core.Sandbox.runners.firecracker_runner import (
            firecracker_available as _fc_avail,
        )

        rt = str(runtime_name or "").strip().lower()
        if rt == "lima":
            return []
        if rt == "firecracker" and _dock_avail():
            suggestions.append("docker")
        elif rt == "docker" and _fc_avail():
            suggestions.append("firecracker")
        return sorted(set(suggestions))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return []


@router.get("/runtimes", response_model=SandboxRuntimesResponse, summary="Discover available runtimes")
async def get_runtimes(current_user: User = Depends(get_request_user)) -> SandboxRuntimesResponse:
    info = _service.feature_discovery()
    return SandboxRuntimesResponse(runtimes=info)  # type: ignore[arg-type]


@router.get("/health", summary="Sandbox health and readiness probe")
async def sandbox_health(current_user: User = Depends(get_request_user)) -> dict:
    """Report sandbox store and Redis fan-out health.

    - Store: reports effective `store_mode` and a basic connectivity check in cluster mode.
    - Redis: reports whether WS fan-out is enabled and connected.
    """
    import time as _time

    from tldw_Server_API.app.core.Sandbox.store import get_store, get_store_mode
    store_info: dict = {"mode": None, "healthy": True}
    timings: dict = {}
    try:
        mode = str(get_store_mode())
        store_info["mode"] = mode
        if mode == "cluster":
            try:
                st = get_store()
                t0 = _time.perf_counter()
                # Minimal smoke call to exercise connectivity
                _ = int(st.count_runs())
                timings["store_ms"] = float((_time.perf_counter() - t0) * 1000.0)
                store_info["healthy"] = True
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                logger.exception("Sandbox health: store connectivity check failed")
                store_info["healthy"] = False
                store_info["code"] = "internal_error"
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.exception("Sandbox health: store mode detection failed")
        store_info["healthy"] = False
        store_info["code"] = "internal_error"
    # Redis status via hub
    try:
        hub = get_hub()  # type: ignore[attr-defined]
        redis_status = hub.get_redis_status()
        if redis_status.get("enabled") and redis_status.get("connected"):
            pong = hub.ping_redis()
            redis_status["ping_ms"] = pong.get("ms")
            timings["redis_ping_ms"] = pong.get("ms")
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        redis_status = {"enabled": False}
    ok = bool(store_info.get("healthy", True)) and (True if not redis_status.get("enabled") else bool(redis_status.get("connected")))
    return {"ok": ok, "store": store_info, "redis": redis_status, "timings": timings}


@router.get("/health/public", summary="Public sandbox health probe (no auth)")
async def sandbox_health_public() -> dict:
    """Public variant of the sandbox health endpoint; does not require auth.

    Reports the same payload as /sandbox/health, including store mode, connectivity,
    Redis fan-out status and ping timings when available.
    """
    import time as _time

    from tldw_Server_API.app.core.Sandbox.store import get_store, get_store_mode
    store_info: dict = {"mode": None, "healthy": True}
    timings: dict = {}
    try:
        mode = str(get_store_mode())
        store_info["mode"] = mode
        if mode == "cluster":
            try:
                st = get_store()
                t0 = _time.perf_counter()
                _ = int(st.count_runs())
                timings["store_ms"] = float((_time.perf_counter() - t0) * 1000.0)
                store_info["healthy"] = True
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                # Do not leak exception details publicly; log with traceback server-side
                logger.exception("Sandbox public health: store connectivity check failed")
                store_info["healthy"] = False
                store_info["code"] = "internal_error"
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        # Do not leak exception details publicly; log with traceback server-side
        logger.exception("Sandbox public health: store mode detection failed")
        store_info["healthy"] = False
        store_info["code"] = "internal_error"
    # Redis status via hub (no auth required)
    try:
        hub = get_hub()  # type: ignore[attr-defined]
        redis_status = hub.get_redis_status()
        if redis_status.get("enabled") and redis_status.get("connected"):
            pong = hub.ping_redis()
            redis_status["ping_ms"] = pong.get("ms")
            timings["redis_ping_ms"] = pong.get("ms")
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        redis_status = {"enabled": False}
    ok = bool(store_info.get("healthy", True)) and (True if not redis_status.get("enabled") else bool(redis_status.get("connected")))
    return {"ok": ok, "store": store_info, "redis": redis_status, "timings": timings}


@router.post("/sessions", response_model=SandboxSession, summary="Create a short-lived sandbox session")
async def create_session(
    request: Request,
    payload: SandboxSessionCreateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    audit_service=Depends(get_audit_service_for_user),
) -> SandboxSession:
    # Default execution timeout from settings (fallback handled in schema)
    try:
        default_exec_to = int(getattr(app_settings, "SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC", 300))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        default_exec_to = 300

    spec = SessionSpec(
        runtime=CoreRuntimeType(payload.runtime) if payload.runtime else None,
        base_image=payload.base_image,
        cpu_limit=payload.cpu_limit,
        memory_mb=payload.memory_mb,
        timeout_sec=payload.timeout_sec or default_exec_to,
        network_policy=payload.network_policy or "deny_all",
        env=payload.env or {},
        labels=payload.labels or {},
        trust_level=CoreTrustLevel(payload.trust_level) if payload.trust_level else None,
        persona_id=payload.persona_id,
        workspace_id=payload.workspace_id,
        workspace_group_id=payload.workspace_group_id,
        scope_snapshot_id=payload.scope_snapshot_id,
    )
    try:
        session = _service.create_session(
            user_id=current_user.id,
            spec=spec,
            spec_version=payload.spec_version,
            idem_key=idempotency_key,
            raw_body=payload.model_dump(exclude_none=True),
        )
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        if isinstance(e, SandboxPolicy.RuntimeUnavailable):
            # Map to 503 with details per PRD; read runtime from exception with safe fallback
            rt = _runtime_name_from_policy_exception(e)
            suggestions = _runtime_unavailable_suggestions(rt)
            reasons = list(getattr(e, "reasons", []) or [])
            logger.exception("RuntimeUnavailable error occurred on sandbox session creation: {}", str(e))
            return JSONResponse(status_code=503, content={
                "error": {
                    "code": "runtime_unavailable",
                    "message": "The requested runtime is currently unavailable.",
                    "details": {"runtime": rt, "available": False, "suggested": suggestions, "reasons": reasons}
                }
            })
        if isinstance(e, SandboxPolicy.PolicyUnsupported):
            rt = _runtime_name_from_policy_exception(e)
            requirement = str(getattr(e, "requirement", "unknown"))
            reasons = list(getattr(e, "reasons", []) or [])
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "policy_unsupported",
                        "message": "Sandbox policy is unsupported for the requested runtime.",
                        "details": {
                            "runtime": rt,
                            "requirement": requirement,
                            "reasons": reasons,
                        },
                    }
                },
            )
        if isinstance(e, IdempotencyConflict):
            raise HTTPException(status_code=409, detail={
                "error": {
                    "code": "idempotency_conflict",
                    "message": str(e),
                    "details": {
                        "prior_id": e.original_id,
                        "key": getattr(e, "key", None),
                        "prior_created_at": getattr(e, "prior_created_at", None),
                    }
                }
            }) from e
        if isinstance(e, QueueFull):
            # Backpressure: 429 with Retry-After
            retry_after = getattr(e, "retry_after", 30)
            details = {"retry_after": retry_after}
            quota_reason = getattr(e, "reason", None)
            quota_scope = getattr(e, "quota_scope", None)
            quota_limit = getattr(e, "limit", None)
            if quota_reason:
                details["reason"] = str(quota_reason)
            if quota_scope:
                details["quota_scope"] = str(quota_scope)
            if isinstance(quota_limit, int):
                details["limit"] = int(quota_limit)
            raise HTTPException(status_code=429, detail={
                "error": {
                    "code": "queue_full",
                    "message": "Sandbox run queue is full",
                    "details": details,
                }
            }, headers={"Retry-After": str(int(retry_after))}) from e
        if isinstance(e, SandboxService.InvalidSpecVersion):
            raise HTTPException(status_code=400, detail={
                "error": {
                    "code": "invalid_spec_version",
                    "message": str(e),
                    "details": {"supported": e.supported, "provided": e.provided}
                }
            }) from e
        if isinstance(e, SandboxService.InvalidFirecrackerConfig):
            return JSONResponse(status_code=400, content={
                "error": {
                    "code": "invalid_firecracker_config",
                    "message": "Firecracker kernel/rootfs configuration is invalid.",
                    "details": e.details,
                }
            })
        raise
    # Metrics
    try:
        increment_counter(
            "sandbox_sessions_created_total",
            labels={"runtime": session.runtime.value},
        )
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: sandbox_sessions_created_total failed")

    # Audit
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else "/api/v1/sandbox/sessions",
                method=(request.method if request else "POST"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="sandbox.session",
                resource_id=session.id,
                action="create",
                metadata={
                    "runtime": session.runtime.value,
                    "base_image": session.base_image,
                    "spec_version": payload.spec_version,
                    "labels": payload.labels or {},
                },
            )
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"sandbox audit(session.create) failed: {e}")

    # Compute canonical policy hash for reproducibility
    try:
        from tldw_Server_API.app.core.Sandbox.policy import compute_policy_hash
        cfg = _service.policy.cfg  # type: ignore[attr-defined]
        ph = compute_policy_hash(cfg)
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        ph = None
    return SandboxSession(
        id=session.id,
        runtime=session.runtime.value,
        base_image=session.base_image,
        cpu_limit=session.cpu_limit,
        memory_mb=session.memory_mb,
        timeout_sec=session.timeout_sec,
        network_policy=session.network_policy,
        env=dict(session.env or {}),
        labels=dict(session.labels or {}),
        trust_level=(session.trust_level.value if session.trust_level else None),
        expires_at=session.expires_at,
        policy_hash=ph,
        persona_id=session.persona_id,
        workspace_id=session.workspace_id,
        workspace_group_id=session.workspace_group_id,
        scope_snapshot_id=session.scope_snapshot_id,
    )


@router.delete("/sessions/{session_id}", summary="Destroy a sandbox session early")
async def delete_session(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> dict:
    _require_session_owner(session_id, current_user)
    try:
        ok = _service.destroy_session(session_id)
    except SessionActiveRunsConflict as e:
        err = str(e) or "session_has_active_runs"
        raise HTTPException(
            status_code=409,
            detail={
                "error": err,
                "active_runs": int(getattr(e, "active_runs", 0) or 0),
                "session_id": str(session_id),
            },
        ) from e
    if not ok:
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"ok": True}


# -----------------
# Snapshot/Clone API
# -----------------

@router.post(
    "/sessions/{session_id}/snapshot",
    response_model=SnapshotCreateResponse,
    summary="Create a snapshot of the session's current state",
)
async def create_snapshot(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> SnapshotCreateResponse:
    """Create a snapshot of the session's current workspace state.

    The snapshot can be used later to restore the session to this point in time.
    """
    _require_session_owner(session_id, current_user)
    try:
        result = _service.create_snapshot(session_id)
        return SnapshotCreateResponse(
            snapshot_id=result["snapshot_id"],
            created_at=result["created_at"],
            size_bytes=result["size_bytes"],
        )
    except SessionActiveRunsConflict as e:
        err = str(e) or "session_has_active_runs"
        raise HTTPException(
            status_code=409,
            detail={
                "error": err,
                "active_runs": int(getattr(e, "active_runs", 0) or 0),
                "session_id": str(session_id),
            },
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail="Failed to create snapshot") from e


@router.post(
    "/sessions/{session_id}/restore",
    response_model=SnapshotRestoreResponse,
    summary="Restore session to snapshot state",
)
async def restore_snapshot(
    session_id: str = Path(..., min_length=1),
    payload: SnapshotRestoreRequest = Body(...),
    current_user: User = Depends(get_request_user),
) -> SnapshotRestoreResponse:
    """Restore a session's workspace to a previous snapshot state.

    This will clear the current workspace and replace it with the snapshot contents.
    """
    _require_session_owner(session_id, current_user)
    try:
        restored = _service.restore_snapshot(session_id, payload.snapshot_id)
        return SnapshotRestoreResponse(
            restored=restored,
            snapshot_id=payload.snapshot_id,
        )
    except SessionActiveRunsConflict as e:
        err = str(e) or "session_has_active_runs"
        raise HTTPException(
            status_code=409,
            detail={
                "error": err,
                "active_runs": int(getattr(e, "active_runs", 0) or 0),
                "session_id": str(session_id),
            },
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail="Failed to restore snapshot") from e


@router.post(
    "/sessions/{session_id}/clone",
    response_model=SessionCloneResponse,
    summary="Clone a session including its workspace",
)
async def clone_session(
    session_id: str = Path(..., min_length=1),
    payload: SessionCloneRequest = Body(default=None),
    current_user: User = Depends(get_request_user),
) -> SessionCloneResponse:
    """Create a new session as a copy of the current one.

    The new session will have a copy of the original session's workspace.
    """
    _require_session_owner(session_id, current_user)
    try:
        new_session = _service.clone_session(
            session_id,
            new_name=(payload.new_session_name if payload else None),
        )
        return SessionCloneResponse(
            session_id=new_session.id,
            cloned_from=session_id,
        )
    except SessionActiveRunsConflict as e:
        err = str(e) or "session_has_active_runs"
        raise HTTPException(
            status_code=409,
            detail={
                "error": err,
                "active_runs": int(getattr(e, "active_runs", 0) or 0),
                "session_id": str(session_id),
            },
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(f"Clone session failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clone session") from e


@router.get(
    "/sessions/{session_id}/snapshots",
    response_model=SnapshotListResponse,
    summary="List available snapshots for a session",
)
async def list_snapshots(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> SnapshotListResponse:
    """List all available snapshots for a session, sorted by creation time (newest first)."""
    _require_session_owner(session_id, current_user)
    snapshots = _service.list_snapshots(session_id)
    items = [
        SnapshotInfo(
            snapshot_id=s.get("snapshot_id", ""),
            session_id=s.get("session_id", session_id),
            created_at=s.get("created_at", ""),
            size_bytes=s.get("size_bytes", 0),
        )
        for s in snapshots
    ]
    return SnapshotListResponse(items=items)


@router.delete(
    "/sessions/{session_id}/snapshots/{snapshot_id}",
    summary="Delete a specific snapshot",
)
async def delete_snapshot(
    session_id: str = Path(..., min_length=1),
    snapshot_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> dict:
    """Delete a specific snapshot from a session."""
    _require_session_owner(session_id, current_user)
    deleted = _service.delete_snapshot(session_id, snapshot_id)
    return {"ok": deleted, "snapshot_id": snapshot_id}


@router.post("/sessions/{session_id}/files", response_model=SandboxFileUploadResponse, summary="Upload files to a session workspace")
async def upload_files(
    request: Request,
    session_id: str = Path(..., min_length=1),
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_request_user),
    audit_service=Depends(get_audit_service_for_user),
) -> SandboxFileUploadResponse:
    _require_session_owner(session_id, current_user)
    written = 0
    count = 0

    try:
        async with _service.async_workspace_operation_lock(session_id) as ws_root:
            try:
                cap_mb = int(os.getenv("SANDBOX_WORKSPACE_CAP_MB") or 256)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                cap_mb = 256
            cap_bytes = cap_mb * 1024 * 1024
            chunk_size = 64 * 1024
            import tarfile
            import zipfile

            async def _stream_upload_to_target(
                upload: UploadFile,
                target: str,
                *,
                workspace_used: int,
            ) -> tuple[int, int]:
                fd, temp_path, existing_size = await asyncio.to_thread(_prepare_temp_target_sync, target)
                file_bytes = 0
                try:
                    while True:
                        chunk = await upload.read(chunk_size)
                        if not chunk:
                            break
                        next_total = workspace_used - existing_size + file_bytes + len(chunk)
                        if next_total > cap_bytes:
                            raise HTTPException(status_code=413, detail="workspace_cap_exceeded")
                        await asyncio.to_thread(_write_fd_chunk_sync, fd, chunk)
                        file_bytes += len(chunk)
                    await asyncio.to_thread(_finalize_temp_target_sync, fd, temp_path, target)
                    return file_bytes, workspace_used - existing_size + file_bytes
                except Exception:
                    await asyncio.to_thread(_cleanup_temp_target_sync, fd, temp_path)
                    raise

            workspace_used = await asyncio.to_thread(_workspace_usage_bytes_sync, ws_root)

            for up in files:
                lower = (up.filename or "").lower()
                if lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
                    try:
                        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                            await up.seek(0)
                        with tarfile.open(fileobj=up.file, mode="r:*") as tf:
                            for member in tf.getmembers():
                                if member.isdev() or member.issym() or member.islnk():
                                    continue
                                target = safe_join(ws_root, member.name)
                                if target is None:
                                    continue
                                if member.isdir():
                                    await asyncio.to_thread(_make_directory_sync, target)
                                    continue
                                fileobj = tf.extractfile(member)
                                if fileobj is None:
                                    continue
                                try:
                                    file_bytes, workspace_used = await asyncio.to_thread(
                                        _stream_reader_to_target_sync,
                                        target,
                                        fileobj.read,
                                        chunk_size=chunk_size,
                                        workspace_used=workspace_used,
                                        cap_bytes=cap_bytes,
                                    )
                                finally:
                                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                                        fileobj.close()
                                written += file_bytes
                                count += 1
                    except HTTPException:
                        raise
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Failed to extract tar: {e}")
                elif lower.endswith(".zip"):
                    try:
                        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                            await up.seek(0)
                        with zipfile.ZipFile(up.file) as zf:
                            for member in zf.infolist():
                                if member.is_dir():
                                    continue
                                if (member.external_attr >> 16) & 0xF000 == 0xA000:
                                    continue
                                target = safe_join(ws_root, member.filename)
                                if target is None:
                                    continue
                                with zf.open(member) as fileobj:
                                    file_bytes, workspace_used = await asyncio.to_thread(
                                        _stream_reader_to_target_sync,
                                        target,
                                        fileobj.read,
                                        chunk_size=chunk_size,
                                        workspace_used=workspace_used,
                                        cap_bytes=cap_bytes,
                                    )
                                written += file_bytes
                                count += 1
                    except HTTPException:
                        raise
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Failed to extract zip: {e}")
                else:
                    target = safe_join(ws_root, up.filename or f"file_{count}")
                    if target is None:
                        continue
                    try:
                        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                            await up.seek(0)
                        file_bytes, workspace_used = await _stream_upload_to_target(
                            up,
                            target,
                            workspace_used=workspace_used,
                        )
                    except HTTPException:
                        raise
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Failed reading upload file {up.filename}: {e}")
                        continue
                    written += file_bytes
                    count += 1
    except ValueError as e:
        if str(e) == "session_not_found":
            raise HTTPException(status_code=404, detail="session_not_found") from e
        raise

    # Metrics
    try:
        if written:
            increment_counter("sandbox_upload_bytes_total", value=float(written), labels={"kind": "session_upload"})
        if count:
            increment_counter("sandbox_upload_files_total", value=float(count), labels={"kind": "session_upload"})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: sandbox upload counters failed")

    # Audit
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else f"/api/v1/sandbox/sessions/{session_id}/files",
                method=(request.method if request else "POST"),
                session_id=session_id,
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="sandbox.session",
                resource_id=session_id,
                action="upload",
                metadata={"bytes_received": written, "file_count": count},
            )
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"sandbox audit(session.upload) failed: {e}")

    return SandboxFileUploadResponse(session_id=session_id, bytes_received=written, file_count=count)


@router.post("/runs", response_model=SandboxRunStatus, summary="Start a sandbox run (one-shot or for a session)")
async def start_run(
    request: Request,
    payload: SandboxRunCreateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    audit_service=Depends(get_audit_service_for_user),
) -> SandboxRunStatus:
    """
    Start a sandbox run (immediate scaffold or queued) using the provided request payload and return its status.

    Builds a RunSpec from the request payload, initiates the run via the sandbox service (respecting an optional idempotency key), records metrics and audit events, and may publish synthetic start/end frames when test-synthetic frames are enabled.

    Parameters:
        payload (SandboxRunCreateRequest): Run creation parameters (runtime, base_image, command, timeouts, resources, network policy, inline files, capture patterns, session association, and spec_version).
        idempotency_key (Optional[str]): Optional Idempotency-Key header used to deduplicate requests.
        request (Request): Incoming HTTP request (used for audit context and metadata).
        current_user (User): Authenticated user initiating the run.
        audit_service: Audit service dependency used to record API request/response events.

    Returns:
        SandboxRunStatus: Current status of the created run, including id, spec_version, runtime, base_image, image_digest, policy_hash, phase, exit_code, start/finish timestamps, message, resource_usage, and estimated_start_time.

    Raises:
        HTTPException: 409 with code "idempotency_conflict" when an idempotent request conflicts.
        HTTPException: 400 with code "invalid_spec_version" when the provided spec_version is unsupported.
    """
    if payload.session_id:
        _require_session_owner(payload.session_id, current_user)
    try:
        files_inline = _service.parse_inline_files([(f.model_dump() if hasattr(f, "model_dump") else f.dict()) for f in (payload.files or [])])
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "invalid_inline_file",
                    "message": str(e),
                }
            },
        ) from e
    try:
        default_exec_to = int(getattr(app_settings, "SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC", 300))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        default_exec_to = 300

    try:
        default_startup_to = int(getattr(app_settings, "SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC", 20))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        default_startup_to = 20

    payload_fields = _model_field_names(payload)
    resource_fields = _model_field_names(payload.resources)
    explicit_fields = set(payload_fields)
    explicit_fields.update({f"resources.{field}" for field in resource_fields})

    runtime = CoreRuntimeType(payload.runtime) if payload.runtime else None
    base_image = payload.base_image or None
    env = dict(payload.env or {})
    timeout_sec = int(payload.timeout_sec) if payload.timeout_sec is not None else default_exec_to
    cpu = payload.resources.cpu if payload.resources else None
    memory_mb = payload.resources.memory_mb if payload.resources else None
    network_policy = payload.network_policy
    trust_level = CoreTrustLevel(payload.trust_level) if payload.trust_level else None
    persona_id = payload.persona_id
    workspace_id = payload.workspace_id
    workspace_group_id = payload.workspace_group_id
    scope_snapshot_id = payload.scope_snapshot_id

    spec = RunSpec(
        session_id=payload.session_id,
        runtime=runtime,
        base_image=base_image,
        command=list(payload.command),
        env=env,
        startup_timeout_sec=payload.startup_timeout_sec or default_startup_to,
        timeout_sec=timeout_sec,
        cpu=cpu,
        memory_mb=memory_mb,
        network_policy=network_policy,
        files_inline=files_inline,
        capture_patterns=payload.capture_patterns or [],
        interactive=(bool(payload.interactive) if hasattr(payload, "interactive") and payload.interactive is not None else None),
        stdin_max_bytes=(int(payload.stdin_max_bytes) if hasattr(payload, "stdin_max_bytes") and payload.stdin_max_bytes is not None else None),
        stdin_max_frame_bytes=(int(payload.stdin_max_frame_bytes) if hasattr(payload, "stdin_max_frame_bytes") and payload.stdin_max_frame_bytes is not None else None),
        stdin_bps=(int(payload.stdin_bps) if hasattr(payload, "stdin_bps") and payload.stdin_bps is not None else None),
        stdin_idle_timeout_sec=(int(payload.stdin_idle_timeout_sec) if hasattr(payload, "stdin_idle_timeout_sec") and payload.stdin_idle_timeout_sec is not None else None),
        trust_level=trust_level,
        persona_id=persona_id,
        workspace_id=workspace_id,
        workspace_group_id=workspace_group_id,
        scope_snapshot_id=scope_snapshot_id,
    )
    # Scaffold: return immediate completed status without real execution
    try:
        status = _service.start_run_scaffold(
            user_id=current_user.id,
            spec=spec,
            spec_version=payload.spec_version,
            idem_key=idempotency_key,
            raw_body=payload.model_dump(exclude_none=True),
            explicit_fields=explicit_fields,
        )
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        if isinstance(e, SandboxPolicy.RuntimeUnavailable):
            # Use runtime from exception; fallback only if missing/None
            rt = _runtime_name_from_policy_exception(e)
            suggestions = _runtime_unavailable_suggestions(rt)
            reasons = list(getattr(e, "reasons", []) or [])
            return JSONResponse(status_code=503, content={
                    "error": {
                        "code": "runtime_unavailable",
                        "message": "The requested runtime is currently unavailable.",
                        "details": {"runtime": rt, "available": False, "suggested": suggestions, "reasons": reasons}
                    }
                })
        if isinstance(e, SandboxPolicy.PolicyUnsupported):
            rt = _runtime_name_from_policy_exception(e)
            requirement = str(getattr(e, "requirement", "unknown"))
            reasons = list(getattr(e, "reasons", []) or [])
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "policy_unsupported",
                        "message": "Sandbox policy is unsupported for the requested runtime.",
                        "details": {
                            "runtime": rt,
                            "requirement": requirement,
                            "reasons": reasons,
                        },
                    }
                },
            )
        if isinstance(e, IdempotencyConflict):
            return JSONResponse(status_code=409, content={
                "error": {
                    "code": "idempotency_conflict",
                    "message": str(e),
                    "details": {
                        "prior_id": e.original_id,
                        "key": getattr(e, "key", None),
                        "prior_created_at": getattr(e, "prior_created_at", None),
                    }
                }
            })
        if isinstance(e, QueueFull):
            # Backpressure: 429 with Retry-After + metric
            retry_after = getattr(e, "retry_after", 30)
            details = {"retry_after": retry_after}
            quota_reason = getattr(e, "reason", None)
            quota_scope = getattr(e, "quota_scope", None)
            quota_limit = getattr(e, "limit", None)
            if quota_reason:
                details["reason"] = str(quota_reason)
            if quota_scope:
                details["quota_scope"] = str(quota_scope)
            if isinstance(quota_limit, int):
                details["limit"] = int(quota_limit)
            try:
                # Include runtime label where possible for taxonomy consistency
                rt_label = str(payload.runtime or "unknown")
                metric_reason = str(quota_reason or "queue_full")
                increment_counter(
                    "sandbox_queue_full_total",
                    labels={"component": "sandbox", "runtime": rt_label, "reason": metric_reason},
                )
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            return JSONResponse(status_code=429, content={
                "error": {
                    "code": "queue_full",
                    "message": "Sandbox run queue is full",
                    "details": details,
                }
            }, headers={"Retry-After": str(int(retry_after))})
        if isinstance(e, SandboxService.InvalidSpecVersion):
            return JSONResponse(status_code=400, content={
                "error": {
                    "code": "invalid_spec_version",
                    "message": str(e),
                    "details": {"supported": e.supported, "provided": e.provided}
                }
            })
        if isinstance(e, SandboxService.InvalidFirecrackerConfig):
            return JSONResponse(status_code=400, content={
                "error": {
                    "code": "invalid_firecracker_config",
                    "message": "Firecracker kernel/rootfs configuration is invalid.",
                    "details": e.details,
                }
            })
        if isinstance(e, ValueError) and str(e) == "session_not_found":
            raise HTTPException(status_code=404, detail="session_not_found") from e
        if isinstance(e, ValueError) and str(e) == "session_base_image_required":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": {
                        "code": "session_base_image_required",
                        "message": "Session-backed runs require a base_image",
                    }
                },
            ) from e
        raise
    try:
        rt = (status.runtime.value if status.runtime else (payload.runtime or "unknown"))
        increment_counter("sandbox_runs_started_total", labels={"runtime": str(rt)})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: sandbox_runs_started_total failed")
    # Metrics and audit post-run (if completed)
    try:
        if status.started_at and status.finished_at:
            duration = max(0.0, (status.finished_at - status.started_at).total_seconds())
            outcome = (
                "success" if status.phase.value == "completed" and (status.exit_code or 0) == 0 else
                "timeout" if status.phase.value == "timed_out" else
                "killed" if status.phase.value == "killed" else
                "failed" if status.phase.value == "failed" else
                status.phase.value
            )
            # Metrics: completion + duration with normalized reason label
            try:
                reason_norm = _normalize_reason(outcome, getattr(status, "message", None))
                labels_completed = {
                    "runtime": str(status.runtime.value if status.runtime else (payload.runtime or "unknown")),
                    "outcome": outcome,
                    "reason": reason_norm,
                }
                increment_counter("sandbox_runs_completed_total", labels=labels_completed)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics: sandbox_runs_completed_total failed")
            try:
                reason_norm = _normalize_reason(outcome, getattr(status, "message", None))
                labels_duration = {
                    "runtime": str(status.runtime.value if status.runtime else (payload.runtime or "unknown")),
                    "outcome": outcome,
                    "reason": reason_norm,
                }
                observe_histogram("sandbox_run_duration_seconds", value=float(duration), labels=labels_duration)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics: sandbox_run_duration_seconds failed")

            # Audit: run completion
            try:
                if audit_service:
                    ctx = AuditContext(
                        request_id=(request.headers.get("X-Request-ID") if request else None),
                        user_id=str(current_user.id),
                        ip_address=(request.client.host if request and request.client else None),
                        user_agent=(request.headers.get("user-agent") if request else None),
                        endpoint=str(request.url.path) if request else "/api/v1/sandbox/runs",
                        method=(request.method if request else "POST"),
                        session_id=payload.session_id,
                    )
                    await audit_service.log_event(
                        event_type=AuditEventType.API_RESPONSE,
                        category=AuditEventCategory.API_CALL,
                        severity=AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING,
                        context=ctx,
                        resource_type="sandbox.run",
                        resource_id=status.id,
                        action="run",
                        result=("success" if outcome == "success" else outcome),
                        duration_ms=duration * 1000.0,
                        metadata=build_run_completion_audit_metadata(
                            status=status,
                            spec_version=payload.spec_version,
                            requested_runtime=runtime,
                            trust_level=spec.trust_level,
                            network_policy=spec.network_policy,
                            capture_patterns=spec.capture_patterns,
                        )
                    )
            except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"sandbox audit(run.complete) failed: {e}")
        else:
            # Audit: run started (background or queued)
            try:
                if audit_service:
                    ctx = AuditContext(
                        request_id=(request.headers.get("X-Request-ID") if request else None),
                        user_id=str(current_user.id),
                        ip_address=(request.client.host if request and request.client else None),
                        user_agent=(request.headers.get("user-agent") if request else None),
                        endpoint=str(request.url.path) if request else "/api/v1/sandbox/runs",
                        method=(request.method if request else "POST"),
                        session_id=payload.session_id,
                    )
                    await audit_service.log_event(
                        event_type=AuditEventType.API_REQUEST,
                        category=AuditEventCategory.API_CALL,
                        severity=AuditSeverity.INFO,
                        context=ctx,
                        resource_type="sandbox.run",
                        resource_id=status.id,
                        action="start",
                        metadata={
                            "runtime": status.runtime.value if status.runtime else None,
                            "base_image": status.base_image or payload.base_image,
                            "policy_hash": status.policy_hash,
                            "spec_version": payload.spec_version,
                        },
                    )
            except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"sandbox audit(run.start) failed: {e}")
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.debug("sandbox metrics/audit post-run block failed")

    # Optional synthetic frames for tests: when enabled via config/env,
    # publish minimal start/end so clients can drain frames immediately.
    try:
        if bool(getattr(app_settings, "SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS", False)):
            hub = get_hub()
            hub.publish_event(status.id, "start", {"source": "endpoint_synthetic"})
            hub.publish_event(status.id, "end", {"source": "endpoint_synthetic"})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass

    # Build optional log_stream_url (signed or unsigned)
    log_stream_url: str | None = None
    try:
        base_path = f"/api/v1/sandbox/runs/{status.id}/stream"
        # Prefer explicit env override to avoid stale cached settings in tests.
        signed_env = None
        try:
            import os as _os
            signed_env = _os.getenv("SANDBOX_WS_SIGNED_URLS")
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            signed_env = None
        if signed_env is not None:
            signed_flag = is_truthy(signed_env)
        else:
            signed_flag = bool(getattr(app_settings, "SANDBOX_WS_SIGNED_URLS", False))
        secret_env = None
        try:
            import os as _os
            secret_env = _os.getenv("SANDBOX_WS_SIGNING_SECRET")
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            secret_env = None
        if secret_env is not None:
            secret_val = secret_env or None
        else:
            secret_val = getattr(app_settings, "SANDBOX_WS_SIGNING_SECRET", None)
        if signed_flag and secret_val:
            ttl = int(getattr(app_settings, "SANDBOX_WS_SIGNED_URL_TTL_SEC", 60))
            exp = int(time.time()) + max(1, ttl)
            msg = f"{status.id}:{exp}".encode()
            secret = str(secret_val).encode("utf-8")
            token = hmac.new(secret, msg, hashlib.sha256).hexdigest()
            log_stream_url = f"{base_path}?token={token}&exp={exp}"
        else:
            log_stream_url = base_path
        # Append from_seq when requested via POST body (spec 1.1 convenience)
        try:
            if hasattr(payload, "resume_from_seq") and payload.resume_from_seq and int(payload.resume_from_seq) > 0:
                sep = "&" if ("?" in str(log_stream_url)) else "?"
                log_stream_url = f"{log_stream_url}{sep}from_seq={int(payload.resume_from_seq)}"
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        # Fail open: omit URL on error
        log_stream_url = None

    status_reason_code = _status_reason_code(
        phase=status.phase,
        message=status.message,
        exit_code=status.exit_code,
        resource_usage=status.resource_usage,
    )
    return SandboxRunStatus(
        id=status.id,
        spec_version=payload.spec_version,
        runtime=status.runtime.value if status.runtime else None,
        runtime_version=status.runtime_version,
        base_image=status.base_image,
        image_digest=status.image_digest,
        policy_hash=status.policy_hash,
        phase=status.phase.value,
        status_reason_code=status_reason_code,
        status_reason_details=_status_reason_details(status_reason_code),
        exit_code=status.exit_code,
        started_at=status.started_at,
        finished_at=status.finished_at,
        message=status.message,
        resource_usage=status.resource_usage,
        estimated_start_time=status.estimated_start_time,
        log_stream_url=log_stream_url,
        session_id=status.session_id,
        persona_id=status.persona_id,
        workspace_id=status.workspace_id,
        workspace_group_id=status.workspace_group_id,
        scope_snapshot_id=status.scope_snapshot_id,
    )


@router.get("/runs/{run_id}", response_model=SandboxRunStatus, summary="Get run status")
async def get_run_status(
    run_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> SandboxRunStatus:
    _require_run_owner(run_id, current_user)
    st = _service.get_run(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_not_found")
    status_reason_code = _status_reason_code(
        phase=st.phase,
        message=st.message,
        exit_code=st.exit_code,
        resource_usage=st.resource_usage,
    )
    return SandboxRunStatus(
        id=st.id,
        spec_version=st.spec_version,
        runtime=st.runtime.value if st.runtime else None,
        runtime_version=st.runtime_version,
        base_image=st.base_image,
        image_digest=st.image_digest,
        policy_hash=st.policy_hash,
        phase=st.phase.value,
        status_reason_code=status_reason_code,
        status_reason_details=_status_reason_details(status_reason_code),
        exit_code=st.exit_code,
        started_at=st.started_at,
        finished_at=st.finished_at,
        message=st.message,
        resource_usage=st.resource_usage,
        estimated_start_time=st.estimated_start_time,
        session_id=st.session_id,
        persona_id=st.persona_id,
        workspace_id=st.workspace_id,
        workspace_group_id=st.workspace_group_id,
        scope_snapshot_id=st.scope_snapshot_id,
    )


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactListResponse, summary="List captured artifacts")
async def list_artifacts(
    request: Request,
    run_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> ArtifactListResponse:
    _require_run_owner(run_id, current_user)
    # If a traversal attempt like `/artifacts/../x` was normalized to this route,
    # detect it via the raw ASGI path and reject with 400 to satisfy security tests.
    try:
        candidates: list[str] = []
        if request is not None:
            try:
                rp = request.scope.get("raw_path")
                if isinstance(rp, (bytes, bytearray)):
                    candidates.append(rp.decode("utf-8", "ignore"))
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                # Starlette URL may expose raw_path in some versions
                rp2 = getattr(request.url, "raw_path", None)
                if isinstance(rp2, str):
                    candidates.append(rp2)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                candidates.append(request.url.path)
            try:
                # HTTP/2 pseudo-header path may be present
                pseudo = None
                for (hk, hv) in request.scope.get("headers", []) or []:
                    try:
                        if hk.decode("latin-1").lower() == ":path":
                            pseudo = hv.decode("utf-8", "ignore")
                            break
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                        continue
                if pseudo:
                    candidates.append(pseudo)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
        if any("/../" in str(c) for c in candidates if c):
            raise HTTPException(status_code=400, detail="invalid_path")
    except HTTPException:
        raise
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        # If guard fails, continue with normal listing
        pass
    sizes = _service._orch.list_artifacts(run_id)  # type: ignore[attr-defined]
    items = []
    for p, sz in sizes.items():
        items.append({
            "path": p,
            "size": sz,
            "download_url": f"/api/v1/sandbox/runs/{run_id}/artifacts/{p}"
        })
    return ArtifactListResponse(items=items)


@router.get("/runs/{run_id}/artifacts/../{rest:path}", summary="Reject traversal in artifact path")
async def reject_artifact_traversal(
    run_id: str = Path(..., min_length=1),
    rest: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
):
    # Explicit guard for segment-level traversal attempts so that any path with
    # '/artifacts/../...' is rejected before hitting the generic download route.
    raise HTTPException(status_code=400, detail="invalid_path")


@router.get("/runs/{run_id}/artifacts/{path:path}", summary="Download an artifact")
async def download_artifact(
    request: Request,
    run_id: str = Path(..., min_length=1),
    path: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    range_header: str | None = Header(None, alias="Range"),
):
    _require_run_owner(run_id, current_user)
    # Basic path normalization checks (orchestrator also normalizes on FS)
    # Reject absolute or traversal attempts early (defense in depth). When the ASGI router
    # normalizes the URL (e.g., collapsing '/artifacts/../x' to '/artifacts/x'), the path
    # parameter may not include '..'. To catch this, also inspect the raw ASGI path.
    try:
        candidates: list[str] = []
        if request is not None:
            try:
                rp = request.scope.get("raw_path")
                if isinstance(rp, (bytes, bytearray)):
                    candidates.append(rp.decode("utf-8", "ignore"))
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                rp2 = getattr(request.url, "raw_path", None)
                if isinstance(rp2, str):
                    candidates.append(rp2)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                candidates.append(request.url.path)
            try:
                pseudo = None
                for (hk, hv) in request.scope.get("headers", []) or []:
                    try:
                        if hk.decode("latin-1").lower() == ":path":
                            pseudo = hv.decode("utf-8", "ignore")
                            break
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                        continue
                if pseudo:
                    candidates.append(pseudo)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
        if any("/../" in str(c) for c in candidates if c):
            raise HTTPException(status_code=400, detail="invalid_path")
    except HTTPException:
        raise
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass
    raw = str(path)
    if (
        raw.startswith("/")
        or raw.startswith("..")
        or "/.." in raw
        or "\\.." in raw
        or ".." in raw
    ):
        raise HTTPException(status_code=400, detail="invalid_path")

    artifact_path = _service._orch.get_artifact_path(run_id, path)  # type: ignore[attr-defined]
    data = None
    if artifact_path is not None:
        try:
            size = int(artifact_path.stat().st_size)
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(status_code=404, detail="artifact_not_found") from None
    else:
        data = _service._orch.get_artifact(run_id, path)  # type: ignore[attr-defined]
        if data is None:
            raise HTTPException(status_code=404, detail="artifact_not_found")
        size = len(data)
    ctype, _ = mimetypes.guess_type(path)
    ctype = ctype or "application/octet-stream"

    # Handle Range requests (e.g., Range: bytes=start-end)
    def _parse_range(h: str) -> tuple[int, int] | None:
        try:
            if not h.startswith("bytes="):
                return None
            rng = h[6:]
            if "," in rng:
                # Multiple ranges not supported
                return None
            start_s, end_s = (rng.split("-", 1) + [""])[:2]
            if start_s == "":
                # suffix bytes: bytes=-N
                n = int(end_s)
                if n <= 0:
                    return None
                start = max(0, size - n)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
            if start < 0 or end < start or start >= size:
                return None
            end = min(end, size - 1)
            return (start, end)
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            return None

    def _iter_artifact_file(file_path: str, start: int, end: int):
        if end < start:
            return
        remaining = end - start + 1
        with open(file_path, "rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Accept-Ranges": "bytes"}
    if range_header:
        rng = _parse_range(range_header)
        if rng is None:
            # Invalid or unsupported range
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        start, end = rng
        chunk_len = end - start + 1
        headers.update({
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(chunk_len),
        })
        if artifact_path is not None:
            return StreamingResponse(
                _iter_artifact_file(str(artifact_path), start, end),
                media_type=ctype,
                headers=headers,
                status_code=206,
            )
        return Response(content=data[start:end + 1], media_type=ctype, headers=headers, status_code=206)

    headers["Content-Length"] = str(size)
    if artifact_path is not None:
        return StreamingResponse(
            _iter_artifact_file(str(artifact_path), 0, size - 1),
            media_type=ctype,
            headers=headers,
        )
    return Response(content=data, media_type=ctype, headers=headers)


@router.post("/runs/{run_id}/cancel", response_model=CancelResponse, summary="Cancel a running sandbox job")
async def cancel_run(
    run_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
) -> CancelResponse:
    try:
        _require_run_owner(run_id, current_user)
        ok = _service.cancel_run(run_id)
        return CancelResponse(id=run_id, cancelled=bool(ok), message=("canceled_by_user" if ok else "not_running_or_not_found"))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail={
            "error": {
                "code": "cancel_failed",
                "message": "Failed to cancel sandbox run",
                "details": {"run_id": run_id}
            }
        }) from e


@router.websocket("/runs/{run_id}/stream")
async def stream_run_logs(websocket: WebSocket, run_id: str) -> None:
    """
    Stream live sandbox run events and frames to a connected WebSocket client.

    Subscribes to the run's event hub and forwards frames (events, logs, heartbeats) to the WebSocket. Periodically emits heartbeats and increments related metrics. When the configured synthetic-frames flag is enabled, publishes minimal synthetic start/end events to allow early-connected clients to observe non-heartbeat frames. Closes the connection after receiving an `end` event unless synthetic frames are enabled, and ensures heartbeat tasks are cancelled and the socket is closed on disconnect or error.

    Parameters:
        websocket (WebSocket): The active WebSocket connection to send frames to.
        run_id (str): Identifier of the sandbox run whose events should be streamed.
    """
    # Authenticate via JWT/session (Authorization header) or API key.
    signed_flag = False
    try:
        signed_env = os.getenv("SANDBOX_WS_SIGNED_URLS")
        if signed_env is not None:
            signed_flag = is_truthy(signed_env)
        else:
            signed_flag = bool(getattr(app_settings, "SANDBOX_WS_SIGNED_URLS", False))
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        signed_flag = False
    try:
        qp = websocket.query_params  # type: ignore[attr-defined]
        auth_token = None
        if qp is not None:
            # Prefer explicit auth_token; only reuse token param when signed URLs are disabled
            auth_token = qp.get("auth_token")
            if not auth_token and not signed_flag:
                auth_token = qp.get("token")
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        auth_token = None
    try:
        api_key = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-KEY")
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        api_key = None
    try:
        user_id = await _resolve_sandbox_ws_user_id(websocket, token=auth_token, api_key=api_key)
    except HTTPException:
        try:
            await websocket.close(code=4401)
        finally:
            return  # noqa: B012
    # Ownership check before streaming
    try:
        owner = _service._orch.get_run_owner(run_id)  # type: ignore[attr-defined]
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        owner = None
    if owner is None:
        # In test mode, allow streaming for untracked runs so hub-only WS tests
        # can publish frames without going through the run store.
        try:
            test_mode = bool(getattr(app_settings, "TEST_MODE", False))
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            test_mode = False
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            test_mode = test_mode or is_test_mode()
        try:
            allow_untracked = is_truthy(os.getenv("SANDBOX_WS_ALLOW_UNTRACKED_RUNS", ""))
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            allow_untracked = False
        if test_mode or allow_untracked:
            owner = str(user_id)
        else:
            try:
                await websocket.close(code=4404)
            finally:
                return  # noqa: B012
    if str(owner) != str(user_id):
        try:
            await websocket.close(code=4404)
        finally:
            return  # noqa: B012

    # Enforce signed WS URL validation when enabled (in addition to auth).
    try:
        if signed_flag:
            secret_env = os.getenv("SANDBOX_WS_SIGNING_SECRET")
            if secret_env is not None:
                secret = secret_env or None
            else:
                secret = getattr(app_settings, "SANDBOX_WS_SIGNING_SECRET", None)
            if not secret:
                # Signing is enabled but no secret configured: refuse connection
                try:
                    await websocket.close(code=1008)
                finally:
                    return  # noqa: B012
            qp = websocket.query_params  # type: ignore[attr-defined]
            token = qp.get("token") if qp else None  # type: ignore[assignment]
            exp = qp.get("exp") if qp else None  # type: ignore[assignment]
            try:
                exp_i = int(str(exp)) if exp is not None else 0
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                exp_i = 0
            now_i = int(time.time())
            if not token or exp_i <= now_i:
                try:
                    await websocket.close(code=1008)
                finally:
                    return  # noqa: B012
            try:
                msg = f"{run_id}:{exp_i}".encode()
                expected = hmac.new(str(secret).encode("utf-8"), msg, hashlib.sha256).hexdigest()
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                expected = ""
            if not hmac.compare_digest(str(token), expected):
                try:
                    await websocket.close(code=1008)
                finally:
                    return  # noqa: B012
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        # On any unexpected error during validation, fail closed
        try:
            await websocket.close(code=1008)
        finally:
            return  # noqa: B012

    run_status = None
    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
        run_status = _service.get_run(run_id)
    ws_quota_token, ws_quota_reason = _sandbox_ws_try_acquire_quota(
        user_id=int(user_id),
        run_id=run_id,
        persona_id=(getattr(run_status, "persona_id", None) if run_status is not None else None),
        session_id=(getattr(run_status, "session_id", None) if run_status is not None else None),
    )
    if ws_quota_token is None:
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            increment_counter(
                "sandbox_ws_quota_rejections_total",
                labels={"component": "sandbox", "reason": str(ws_quota_reason or "quota_exceeded")},
            )
        try:
            await websocket.close(code=4429)
        finally:
            return  # noqa: B012

    hub = get_hub()
    hub.set_loop(asyncio.get_running_loop())
    # Optional resume from specific sequence
    try:
        qp = websocket.query_params  # type: ignore[attr-defined]
        from_seq_raw = qp.get("from_seq") if qp else None  # type: ignore[assignment]
        from_seq = int(str(from_seq_raw)) if from_seq_raw is not None else 0
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        from_seq = 0
    # Subscribe with buffered frames prefilled to avoid races
    if from_seq and from_seq > 0:
        q = hub.subscribe_with_buffer_from_seq(run_id, int(from_seq))
    else:
        q = hub.subscribe_with_buffer(run_id)

    await websocket.accept()
    # Wrap for WS metrics; keep domain frames unchanged
    stream = WebSocketStream(
        websocket,
        heartbeat_interval_s=0.0,
        idle_timeout_s=None,
        close_on_done=False,
        labels={"component": "sandbox", "endpoint": "sandbox_run_ws"},
    )
    await stream.start()
    # Keep strong references to any background tasks spawned in this handler
    synth_task: asyncio.Task | None = None
    # No-op: retain default queue state; buffered frames are enqueued below.
    # In test environments (when explicitly enabled), proactively enqueue
    # minimal frames directly into this subscriber's queue if it's empty so
    # the client immediately receives non-heartbeat messages.
    try:
        _synth_env = os.getenv("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS")
        synth_enabled = is_truthy(_synth_env)
        if synth_enabled:
            st = _service.get_run(run_id)
            try:
                q_empty = q.empty()
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                q_empty = False
            # Optionally publish synthetic frames via hub for test determinism
            if st is not None and q_empty:
                # Publish synthetic frames via hub so they participate in ordering and seq stamping
                with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                    hub.publish_event(run_id, "start", {"source": "ws_synthetic"})
                async def _enqueue_end_later():
                    try:
                        await asyncio.sleep(0.05)
                        hub.publish_event(run_id, "end", {"source": "ws_synthetic"})
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                        return
                # Store task to avoid premature GC and enable cleanup
                synth_task = asyncio.create_task(_enqueue_end_later())
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass
    # Buffered frames are already enqueued for this subscriber by the hub,
    # ensuring seq-stamped history arrives before live frames.

    # If the run already ended, ensure an 'end' is present for late subscribers
    # (No second 'end' send here to avoid duplicates)

    # Metrics: connection opened
    try:
        increment_counter("sandbox_ws_connections_opened_total", labels={"component": "sandbox"})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: sandbox_ws_connections_opened_total failed")

    async def _heartbeats() -> None:
        try:
            while True:
                if hub.has_ended(run_id):
                    return
                await asyncio.sleep(10)
                if hub.has_ended(run_id):
                    return
                # Publish via hub to attach seq and flow through the same queue.
                # If this fails, skip the heartbeat to avoid injecting out-of-band frames
                # with potentially inconsistent sequencing.
                try:
                    if not hub.publish_heartbeat(run_id):
                        return
                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        increment_counter("sandbox_ws_heartbeats_sent_total", labels={"component": "sandbox"})
                except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                    continue
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            return

    spawn_hb = True
    try:
        # If run already ended, avoid spawning heartbeats that could interleave
        if hub.has_ended(run_id):
            spawn_hb = False
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        spawn_hb = True
    hb_task = asyncio.create_task(_heartbeats()) if spawn_hb else None

    # Background task to read inbound frames (stdin) when interactive is enabled
    async def _reader() -> None:
        try:
            while True:
                try:
                    msg = await websocket.receive_json()
                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        stream.mark_activity()
                except WebSocketDisconnect:
                    return
                except RuntimeError:
                    # Starlette raises RuntimeError when the websocket is no longer connected.
                    return
                except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                    # Non-JSON or decode error: ignore and continue
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") != "stdin":
                    continue
                # Determine encoding and data
                enc = str(msg.get("encoding") or "utf8").lower()
                data_field = msg.get("data")
                if not isinstance(data_field, str):
                    continue
                try:
                    if enc == "base64":
                        import base64 as _b64
                        raw = _b64.b64decode(data_field)
                    else:
                        raw = data_field.encode("utf-8")
                except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                    raw = b""
                if not raw:
                    continue
                # Enforce caps via hub
                allowed, reason = hub.consume_stdin(run_id, len(raw))
                if allowed <= 0:
                    # Rate limited or cap reached: notify client via truncated frame
                    if reason:
                        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                            hub.publish_truncated(run_id, str(reason))
                    continue
                if allowed < len(raw):
                    # Truncated by cap; notify once
                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        hub.publish_truncated(run_id, str(reason or "stdin_cap"))
                # Enqueue allowed bytes for runner-side stdin pump
                try:
                    if allowed > 0:
                        hub.push_stdin(run_id, raw[:allowed])
                except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                    pass
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            return

    reader_task = asyncio.create_task(_reader())

    # Idle-timeout watchdog for stdin
    async def _idle_watchdog() -> None:
        try:
            tout = hub.get_stdin_idle_timeout(run_id)
            if not tout or tout <= 0:
                return
            import time as _time
            while True:
                await asyncio.sleep(0.5)
                last = hub.get_last_stdin_input_time(run_id)
                if last is None:
                    continue
                if (_time.time() - float(last)) > float(tout):
                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        hub.publish_truncated(run_id, "stdin_idle")
                    with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        await stream.ws.close()
                    return
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            return

    idle_task = asyncio.create_task(_idle_watchdog())
    try:
        # Allow tests to reduce the poll timeout via settings/env (prefer env at runtime)
        try:
            _pt_env = os.getenv("SANDBOX_WS_POLL_TIMEOUT_SEC")
            poll_timeout = float(_pt_env) if _pt_env is not None else float(getattr(app_settings, "SANDBOX_WS_POLL_TIMEOUT_SEC", 30))
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            poll_timeout = 30.0
        while True:
            try:
                frame = await asyncio.wait_for(q.get(), timeout=poll_timeout)
            except asyncio.TimeoutError:
                continue
            await stream.send_json(frame)
            # Do not forcibly close on 'end'; allow clients/tests to disconnect.
            # This avoids race conditions with the Starlette TestClient where the
            # server closing first can lead to ClosedResourceError during reads.
            # We intentionally keep the socket open in both synthetic and normal modes.
    except WebSocketDisconnect:
        logger.info(f"WS disconnected for run {run_id}")
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            increment_counter("sandbox_ws_disconnects_total", labels={"component": "sandbox"})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"WS stream error: {e}")
    finally:
        try:
            if hb_task:
                hb_task.cancel()
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if reader_task:
                reader_task.cancel()
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if idle_task:
                idle_task.cancel()
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        # Ensure any synthetic end task is also cancelled if still pending
        try:
            if synth_task and not synth_task.done():
                synth_task.cancel()
        except _SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            hub.unsubscribe(run_id, q)
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            await stream.ws.close()
        with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
            _sandbox_ws_release_quota(ws_quota_token)




# -----------------------
# Admin API (list/details)
# -----------------------

def _sandbox_startup_warning_summary(request: Request) -> dict[str, object]:
    """Return compact sandbox startup warnings when the app registry exists."""

    empty_summary: dict[str, object] = {
        "present": False,
        "blocking": False,
        "codes": [],
    }
    registry = getattr(request.app.state, "startup_warning_registry", None)
    if registry is None:
        return empty_summary
    try:
        records = list(registry.list_warnings(component_prefix="sandbox.") or [])
        summary = dict(registry.summary(component_prefix="sandbox.") or {})
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        return empty_summary
    return {
        "present": bool(records),
        "blocking": bool(summary.get("has_blocking")),
        "codes": sorted(
            str(record.code) for record in records if getattr(record, "code", None)
        ),
    }


@router.get(
    "/admin/runtime-diagnostics",
    response_model=SandboxAdminRuntimeDiagnosticsResponse,
    summary="Admin: sandbox runtime diagnostics summary",
)
async def admin_runtime_diagnostics(
    request: Request,
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminRuntimeDiagnosticsResponse:
    """Return read-only operator diagnostics derived from runtime discovery."""
    payload = dict(await asyncio.to_thread(_service.runtime_diagnostics_summary))
    payload["startup_warning_summary"] = _sandbox_startup_warning_summary(request)
    return SandboxAdminRuntimeDiagnosticsResponse.model_validate(payload)


@router.get(
    "/admin/macos-diagnostics",
    response_model=SandboxAdminMacOSDiagnosticsResponse,
    summary="Admin: macOS sandbox diagnostics",
)
async def admin_macos_diagnostics(
    request: Request,
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminMacOSDiagnosticsResponse:
    """Return detailed macOS sandbox diagnostics for admin troubleshooting."""
    payload = dict(_service.macos_diagnostics())
    payload["startup_warning_summary"] = _sandbox_startup_warning_summary(request)

    return SandboxAdminMacOSDiagnosticsResponse.model_validate(payload)


@router.post(
    "/admin/macos-reconciliation/repair",
    response_model=SandboxAdminMacOSReconciliationRepairResponse,
    summary="Admin: repair macOS sandbox reconciliation state",
)
async def admin_repair_macos_reconciliation(
    request: SandboxAdminMacOSReconciliationRepairRequest = Body(default_factory=SandboxAdminMacOSReconciliationRepairRequest),
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminMacOSReconciliationRepairResponse:
    try:
        payload = await asyncio.to_thread(_service.repair_macos_reconciliation, **request.model_dump())
    except SandboxReconciliationRepairError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc
    return SandboxAdminMacOSReconciliationRepairResponse.model_validate(payload)


@router.get(
    "/admin/macos-image-store/cleanup-plan",
    response_model=SandboxAdminMacOSImageStoreCleanupPlanResponse,
    summary="Admin: plan macOS image store cleanup actions",
)
async def admin_macos_image_store_cleanup_plan(
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminMacOSImageStoreCleanupPlanResponse:
    payload = await asyncio.to_thread(_service.plan_macos_image_store_cleanup)
    return SandboxAdminMacOSImageStoreCleanupPlanResponse.model_validate(payload)


@router.post(
    "/admin/macos-image-store/cleanup",
    response_model=SandboxAdminMacOSImageStoreCleanupResponse,
    summary="Admin: cleanup macOS image store candidates",
)
async def admin_macos_image_store_cleanup(
    request: SandboxAdminMacOSImageStoreCleanupRequest = Body(default_factory=SandboxAdminMacOSImageStoreCleanupRequest),
    _principal: AuthPrincipal = Depends(RequireRole("admin")),
    _current_user: User = Depends(get_request_user),
) -> SandboxAdminMacOSImageStoreCleanupResponse:
    try:
        payload = await asyncio.to_thread(_service.cleanup_macos_image_store, **request.model_dump())
    except SandboxImageStoreCleanupError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc
    return SandboxAdminMacOSImageStoreCleanupResponse.model_validate(payload)


@router.get(
    "/admin/runs",
    response_model=SandboxAdminRunListResponse,
    summary="Admin: list sandbox runs with filters",
)
async def admin_list_runs(
    image_digest: str | None = Query(None, description="Filter by image digest"),
    user_id: str | None = Query(None, description="Filter by user id"),
    workspace_id: str | None = Query(None, description="Filter by workspace id"),
    workspace_group_id: str | None = Query(None, description="Filter by workspace-group id"),
    scope_snapshot_id: str | None = Query(None, description="Filter by scope snapshot id"),
    phase: str | None = Query(None, description="Filter by run phase"),
    started_at_from: str | None = Query(None, description="ISO timestamp inclusive lower bound"),
    started_at_to: str | None = Query(None, description="ISO timestamp inclusive upper bound"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str | None = Query("desc", pattern="^(asc|desc)$"),
    principal: AuthPrincipal = Depends(RequireRole("admin")),
    current_user: User = Depends(get_request_user),
) -> SandboxAdminRunListResponse:
    items_raw = _service._orch.list_runs(  # type: ignore[attr-defined]
        image_digest=image_digest,
        user_id=user_id,
        workspace_id=workspace_id,
        workspace_group_id=workspace_group_id,
        scope_snapshot_id=scope_snapshot_id,
        phase=phase,
        started_at_from=started_at_from,
        started_at_to=started_at_to,
        limit=limit,
        offset=offset,
        sort_desc=(str(sort).lower() != "asc"),
    )
    total = _service._orch.count_runs(  # type: ignore[attr-defined]
        image_digest=image_digest,
        user_id=user_id,
        workspace_id=workspace_id,
        workspace_group_id=workspace_group_id,
        scope_snapshot_id=scope_snapshot_id,
        phase=phase,
        started_at_from=started_at_from,
        started_at_to=started_at_to,
    )
    items: list[SandboxAdminRunSummary] = []
    for r in items_raw:
        status_reason_code = _status_reason_code(
            phase=r.get("phase"),
            message=r.get("message"),
            exit_code=r.get("exit_code"),
            resource_usage=r.get("resource_usage"),
        )
        items.append(
            SandboxAdminRunSummary(
                id=str(r.get("id")),
                user_id=(r.get("user_id") if r.get("user_id") is not None else None),
                spec_version=r.get("spec_version"),
                runtime=r.get("runtime"),
                runtime_version=r.get("runtime_version"),
                base_image=r.get("base_image"),
                image_digest=r.get("image_digest"),
                policy_hash=r.get("policy_hash"),
                phase=r.get("phase"),
                status_reason_code=status_reason_code,
                status_reason_details=_status_reason_details(status_reason_code),
                exit_code=r.get("exit_code"),
                started_at=(r.get("started_at") if isinstance(r.get("started_at"), str) else r.get("started_at")),
                finished_at=(r.get("finished_at") if isinstance(r.get("finished_at"), str) else r.get("finished_at")),
                message=r.get("message"),
                session_id=r.get("session_id"),
                persona_id=r.get("persona_id"),
                workspace_id=r.get("workspace_id"),
                workspace_group_id=r.get("workspace_group_id"),
                scope_snapshot_id=r.get("scope_snapshot_id"),
            )
        )
    has_more = (offset + len(items)) < int(total)
    total_count = int(total)
    return SandboxAdminRunListResponse(
        total=total_count,
        limit=int(limit),
        offset=int(offset),
        has_more=bool(has_more),
        pagination=build_offset_pagination_meta(
            total=total_count,
            limit=int(limit),
            offset=int(offset),
            count=len(items),
        ),
        items=items,
    )


@router.get(
    "/admin/runs/{run_id}",
    response_model=SandboxAdminRunDetails,
    summary="Admin: get sandbox run details",
)
async def admin_get_run_details(
    run_id: str = Path(..., min_length=1),
    principal: AuthPrincipal = Depends(RequireRole("admin")),
    current_user: User = Depends(get_request_user),
) -> SandboxAdminRunDetails:
    st = _service.get_run(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_not_found")
    try:
        owner = _service._orch.get_run_owner(run_id)  # type: ignore[attr-defined]
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        owner = None
    status_reason_code = _status_reason_code(
        phase=st.phase,
        message=st.message,
        exit_code=st.exit_code,
        resource_usage=st.resource_usage,
    )
    return SandboxAdminRunDetails(
        id=st.id,
        user_id=(owner if owner is not None else None),
        spec_version=st.spec_version,
        runtime=(st.runtime.value if st.runtime else None),
        runtime_version=st.runtime_version,
        base_image=st.base_image,
        image_digest=st.image_digest,
        policy_hash=st.policy_hash,
        phase=st.phase.value,
        status_reason_code=status_reason_code,
        status_reason_details=_status_reason_details(status_reason_code),
        exit_code=st.exit_code,
        started_at=st.started_at,
        finished_at=st.finished_at,
        message=st.message,
        resource_usage=st.resource_usage,
        session_id=st.session_id,
        persona_id=st.persona_id,
        workspace_id=st.workspace_id,
        workspace_group_id=st.workspace_group_id,
        scope_snapshot_id=st.scope_snapshot_id,
    )


# Fallback guard: catch normalized traversal paths that bypass artifacts route
@router.get("/runs/{run_id}/{rest:path}", include_in_schema=False)
async def sandbox_runs_fallback_guard(
    request: Request,
    run_id: str = Path(..., min_length=1),
    rest: str = Path(..., min_length=1),
):
    try:
        # Collect multiple candidates for the original request path across ASGI implementations
        candidates: list[str] = []
        if request is not None:
            try:
                rp = request.scope.get("raw_path")
                if isinstance(rp, (bytes, bytearray)):
                    candidates.append(rp.decode("utf-8", "ignore"))
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                rp2 = getattr(request.url, "raw_path", None)
                if isinstance(rp2, str):
                    candidates.append(rp2)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_SANDBOX_NONCRITICAL_EXCEPTIONS):
                candidates.append(request.url.path)
            try:
                # HTTP/2 pseudo-header path may be present
                pseudo = None
                for (hk, hv) in request.scope.get("headers", []) or []:
                    try:
                        if hk.decode("latin-1").lower() == ":path":
                            pseudo = hv.decode("utf-8", "ignore")
                            break
                    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                        continue
                if pseudo:
                    candidates.append(pseudo)
            except _SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
        if any(("/api/v1/sandbox/runs/" in c and "/artifacts/../" in c) for c in candidates if c):
            raise HTTPException(status_code=400, detail="invalid_path")
    except HTTPException:
        raise
    except _SANDBOX_NONCRITICAL_EXCEPTIONS:
        pass
    # Fallback: not found under /runs/{run_id}
    raise HTTPException(status_code=404, detail="Not Found")


# -----------------------------
# Admin API: Idempotency, Usage
# -----------------------------

@router.get(
    "/admin/idempotency",
    response_model=SandboxAdminIdempotencyListResponse,
    summary="Admin: list idempotency records",
)
async def admin_list_idempotency(
    endpoint: str | None = Query(None, description="Filter by endpoint, e.g., 'runs' or 'sessions'"),
    user_id: str | None = Query(None, description="Filter by user id"),
    key: str | None = Query(None, description="Filter by idempotency key"),
    created_at_from: str | None = Query(None, description="ISO timestamp inclusive lower bound"),
    created_at_to: str | None = Query(None, description="ISO timestamp inclusive upper bound"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str | None = Query("desc", pattern="^(asc|desc)$"),
    principal: AuthPrincipal = Depends(RequireRole("admin")),
    current_user: User = Depends(get_request_user),
) -> SandboxAdminIdempotencyListResponse:
    items_raw = _service._orch._store.list_idempotency(  # type: ignore[attr-defined]
        endpoint=endpoint,
        user_id=user_id,
        key=key,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
        limit=limit,
        offset=offset,
        sort_desc=(str(sort).lower() != "asc"),
    )
    total = _service._orch._store.count_idempotency(  # type: ignore[attr-defined]
        endpoint=endpoint,
        user_id=user_id,
        key=key,
        created_at_from=created_at_from,
        created_at_to=created_at_to,
    )
    items: list[SandboxAdminIdempotencyItem] = []
    for r in items_raw:
        items.append(
            SandboxAdminIdempotencyItem(
                endpoint=str(r.get("endpoint")),
                user_id=(r.get("user_id") if r.get("user_id") is not None else None),
                key=str(r.get("key")),
                fingerprint=(r.get("fingerprint") if r.get("fingerprint") is not None else None),
                object_id=str(r.get("object_id")),
                created_at=(r.get("created_at") if isinstance(r.get("created_at"), str) else None),
            )
        )
    has_more = (offset + len(items)) < int(total)
    total_count = int(total)
    return SandboxAdminIdempotencyListResponse(
        total=total_count,
        limit=int(limit),
        offset=int(offset),
        has_more=bool(has_more),
        pagination=build_offset_pagination_meta(
            total=total_count,
            limit=int(limit),
            offset=int(offset),
            count=len(items),
        ),
        items=items,
    )


@router.get(
    "/admin/usage",
    response_model=SandboxAdminUsageResponse,
    summary="Admin: usage aggregates per user",
)
async def admin_usage(
    user_id: str | None = Query(None, description="Filter by user id"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str | None = Query("desc", pattern="^(asc|desc)$"),
    principal: AuthPrincipal = Depends(RequireRole("admin")),
    current_user: User = Depends(get_request_user),
) -> SandboxAdminUsageResponse:
    items_raw = _service._orch._store.list_usage(  # type: ignore[attr-defined]
        user_id=user_id,
        limit=limit,
        offset=offset,
        sort_desc=(str(sort).lower() != "asc"),
    )
    total = _service._orch._store.count_usage(user_id=user_id)  # type: ignore[attr-defined]
    items: list[SandboxAdminUsageItem] = []
    for r in items_raw:
        items.append(
            SandboxAdminUsageItem(
                user_id=str(r.get("user_id")),
                runs_count=int(r.get("runs_count") or 0),
                log_bytes=int(r.get("log_bytes") or 0),
                artifact_bytes=int(r.get("artifact_bytes") or 0),
            )
        )
    has_more = (offset + len(items)) < int(total)
    total_count = int(total)
    return SandboxAdminUsageResponse(
        total=total_count,
        limit=int(limit),
        offset=int(offset),
        has_more=bool(has_more),
        pagination=build_offset_pagination_meta(
            total=total_count,
            limit=int(limit),
            offset=int(offset),
            count=len(items),
        ),
        items=items,
    )
