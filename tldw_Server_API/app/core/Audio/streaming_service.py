import asyncio
import contextlib
import inspect
import json
import os
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Any, Callable, Optional

from fastapi import WebSocket, WebSocketDisconnect, WebSocketException
from loguru import logger
from starlette.websockets import WebSocketState

from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import bearer_credential
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.websocket_principal import (
    WS_UNREAD_FRAME,
    auth_subprotocol,
    has_ws_credential,
    resolve_ws_principal,
    resolve_ws_principal_first_frame,
    resolve_ws_principal_or_deprecated_query,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Usage.llm_call_context import bind_llm_call_user

_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
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
    TypeError,
    UnicodeDecodeError,
    ValueError,
    WebSocketDisconnect,
)



def _ensure_audio_ws_state(websocket: WebSocket) -> Any:
    state = getattr(websocket, "state", None)
    if state is None:
        state = SimpleNamespace()
        with contextlib.suppress(Exception):
            websocket.state = state
    return state


async def receive_audio_websocket_text(websocket: WebSocket) -> str:
    """Receive text while replaying a first application frame inspected during auth."""
    state = _ensure_audio_ws_state(websocket)
    pending = getattr(state, WS_UNREAD_FRAME, None)
    if pending is not None:
        setattr(state, WS_UNREAD_FRAME, None)
        return str(pending)
    return await websocket.receive_text()


def _get_chat_history_max_messages() -> int:
    """
    Resolve the maximum number of chat history messages to retain
    for streaming audio chat sessions.

    Uses AUDIO_CHAT_HISTORY_MAX_MESSAGES env var when set, falling
    back to a sensible default of 40.
    """
    raw = os.getenv("AUDIO_CHAT_HISTORY_MAX_MESSAGES", "").strip()
    if not raw:
        return 40
    try:
        value = int(raw)
        return value if value > 0 else 40
    except (ValueError, TypeError) as e:
        logger.debug(f"AUDIO_CHAT_HISTORY_MAX_MESSAGES parse failed: {e}")
        return 40


CHAT_HISTORY_MAX_MESSAGES: int = _get_chat_history_max_messages()


def _get_tts_ws_queue_maxsize() -> int:
    """
    Resolve WS TTS producer/consumer queue depth.

    Environment variables (first non-empty wins):
      - AUDIO_TTS_WS_QUEUE_MAXSIZE
      - AUDIO_WS_TTS_QUEUE_MAXSIZE

    Values are clamped to [2, 256]. Invalid values fall back to 8.
    """
    raw = (os.getenv("AUDIO_TTS_WS_QUEUE_MAXSIZE") or os.getenv("AUDIO_WS_TTS_QUEUE_MAXSIZE") or "").strip()
    if not raw:
        return 8
    try:
        parsed = int(raw)
    except (ValueError, TypeError):
        logger.debug(f"Invalid WS TTS queue size value '{raw}'; using default=8")
        return 8
    return max(2, min(256, parsed))


def _allow_query_token_auth() -> bool:
    """Return True only when legacy WebSocket query-token auth is explicitly enabled."""
    return is_truthy(os.getenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", "0"))


# Register audio fail-open metrics (idempotent if already registered)
try:
    _reg = get_metrics_registry()
    _reg.register_metric(
        MetricDefinition(
            name="audio_failopen_minutes_total",
            type=MetricType.COUNTER,
            description="Minutes allowed during fail-open when quota store unavailable",
            unit="minutes",
            labels=["reason"],
        )
    )
    _reg.register_metric(
        MetricDefinition(
            name="audio_failopen_events_total",
            type=MetricType.COUNTER,
            description="Fail-open allowance events during streaming",
            labels=["reason"],
        )
    )
    _reg.register_metric(
        MetricDefinition(
            name="audio_failopen_cap_exhausted_total",
            type=MetricType.COUNTER,
            description="Fail-open cap exhausted; connection closed due to bounded fail-open",
            labels=["reason"],
        )
    )
except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
    # Metrics must never break imports; log for diagnostics
    logger.debug(f"audio: metrics registration skipped/failed: {e}")


async def _stream_tts_to_websocket(
    *,
    websocket: WebSocket,
    speech_req: Any,
    tts_service: Any,
    provider: Optional[str],
    provider_overrides: Optional[dict[str, Any]] = None,
    user_id: Optional[int] = None,
    on_first_output: Optional[Callable[[], Awaitable[None] | None]] = None,
    outer_stream: Optional[Any],
    reg: Any,
    route: str,
    component_label: str,
    voice_to_voice_start: Optional[float] = None,
    error_handler: Optional[Callable[[Exception], Awaitable[None]]] = None,
    asyncio_module: Optional[Any] = None,
) -> None:
    """
    Shared helper to stream TTS audio chunks over a WebSocket with backpressure and metrics.

    This consolidates the producer/consumer queue pattern used by both the
    audio.chat.stream and audio.stream.tts WebSocket handlers.
    """
    aio = asyncio_module or asyncio
    Queue = getattr(aio, "Queue", asyncio.Queue)
    QueueFull = getattr(aio, "QueueFull", asyncio.QueueFull)
    create_task = getattr(aio, "create_task", asyncio.create_task)
    wait = getattr(aio, "wait", asyncio.wait)
    FIRST_COMPLETED = getattr(aio, "FIRST_COMPLETED", asyncio.FIRST_COMPLETED)
    queue: asyncio.Queue[Optional[bytes]] = Queue(maxsize=_get_tts_ws_queue_maxsize())
    provider_label = (provider or getattr(speech_req, "model", None) or "default").lower()
    underrun_labels = {"provider": provider_label}
    error_labels = {"component": component_label, "provider": provider_label}
    sentinel_enqueued = False

    async def _producer() -> None:
        nonlocal sentinel_enqueued
        speech_stream: Any = None
        first_output_marked = False
        try:
            generate_kwargs: dict[str, Any] = {
                "provider": provider,
                "fallback": provider_overrides is None,
                "voice_to_voice_route": route,
                "user_id": user_id,
            }
            if provider_overrides is not None:
                generate_kwargs["provider_overrides"] = provider_overrides
            if voice_to_voice_start is not None:
                generate_kwargs["voice_to_voice_start"] = voice_to_voice_start

            speech_stream = tts_service.generate_speech(
                speech_req,
                **generate_kwargs,
            )
            async for chunk in speech_stream:
                if not chunk:
                    continue
                if not first_output_marked and on_first_output is not None:
                    mark_result = on_first_output()
                    if inspect.isawaitable(mark_result):
                        await mark_result
                    first_output_marked = True
                try:
                    queue.put_nowait(chunk)
                except QueueFull:
                    # Count underrun at overflow detection time even if recovery path changes.
                    try:
                        reg.increment("audio_stream_underruns_total", 1, labels=underrun_labels)
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"{route} underrun metrics update failed: error={m_err}")
                        with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                            reg.increment("audio_stream_errors_total", 1, labels=error_labels)
                    try:
                        _ = queue.get_nowait()
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as q_err:
                        logger.debug(f"{route} queue get_nowait failed: error={q_err}")
                    try:
                        queue.put_nowait(chunk)
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"{route} queue recovery after full failed: error={m_err}")
                        with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                            reg.increment("audio_stream_errors_total", 1, labels=error_labels)
        except asyncio.CancelledError:
            raise
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            try:
                reg.increment("audio_stream_errors_total", 1, labels=error_labels)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                logger.debug(f"{route} producer metrics update failed (outer): error={m_err}")
            if error_handler is not None:
                try:
                    await error_handler(exc)
                except asyncio.CancelledError:
                    raise
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                    logger.debug(f"{route} error handler failed: error={send_exc}")
        finally:
            close_stream = getattr(speech_stream, "aclose", None)
            if callable(close_stream):
                try:
                    await close_stream()
                except asyncio.CancelledError:
                    raise
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                    logger.debug(
                        f"{route} TTS iterator close failed: error={close_exc}"
                    )
            try:
                await queue.put(None)
                sentinel_enqueued = True
            except asyncio.CancelledError:
                raise
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as q_err:
                logger.debug(f"{route} queue sentinel enqueue failed: error={q_err}")

    async def _consumer() -> None:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                try:
                    await websocket.send_bytes(item)
                    if outer_stream:
                        outer_stream.mark_activity()
                except asyncio.CancelledError:
                    raise
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                    try:
                        reg.increment("audio_stream_errors_total", 1, labels=error_labels)
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"{route} consumer metrics update failed: error={m_err}")
                    try:
                        await websocket.close(code=1011)
                    except asyncio.CancelledError:
                        raise
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                        logger.debug(f"{route} websocket close in consumer failed: error={close_exc}")
                    if error_handler is not None:
                        try:
                            await error_handler(exc)
                        except asyncio.CancelledError:
                            raise
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                            logger.debug(f"{route} consumer error handler failed: error={send_exc}")
                    break
        except asyncio.CancelledError:
            raise
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            try:
                reg.increment("audio_stream_errors_total", 1, labels=error_labels)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                logger.debug(f"{route} consumer metrics update failed (outer): error={m_err}")

    producer_task = create_task(_producer())
    consumer_task = create_task(_consumer())

    async def _cancel_task(task: Any, task_name: str) -> None:
        """Cancel one child task while preserving cancellation of this stream."""
        if task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            logger.debug(f"{route} {task_name} task cancelled")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as wait_exc:
            logger.debug(f"{route} wait for {task_name} task failed after cancel: error={wait_exc}")

    def _observe_task_result(task: Any, task_name: str) -> bool:
        """Return whether a completed task exited without a handled stream error."""
        if not task.done():
            return False
        try:
            task.result()
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            logger.debug(f"{route} {task_name} task cancelled")
            return False
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as done_exc:
            logger.debug(f"{route} {task_name} task returned error: error={done_exc}")
            return False
        return True

    try:
        done, pending = await wait(
            {producer_task, consumer_task},
            return_when=FIRST_COMPLETED,
        )
        if producer_task in done:
            producer_ok = _observe_task_result(producer_task, "producer")
            if consumer_task in pending:
                if producer_ok and sentinel_enqueued:
                    try:
                        await consumer_task
                    except asyncio.CancelledError:
                        raise
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as wait_exc:
                        logger.debug(f"{route} wait for consumer task failed: error={wait_exc}")
                    _observe_task_result(consumer_task, "consumer")
                else:
                    await _cancel_task(consumer_task, "consumer")
            elif consumer_task in done:
                _observe_task_result(consumer_task, "consumer")
        elif consumer_task in done:
            _observe_task_result(consumer_task, "consumer")
            if producer_task in pending:
                await _cancel_task(producer_task, "producer")
    finally:
        # Re-raise the caught cancellation itself: a new bare CancelledError is not the host's
        # (anyio's cancel scope does not recognise it) and escapes it (TASK-13950.7).
        cleanup_cancelled: Optional[asyncio.CancelledError] = None
        for task, task_name in ((producer_task, "producer"), (consumer_task, "consumer")):
            try:
                await _cancel_task(task, task_name)
            except asyncio.CancelledError as cancelled:
                cleanup_cancelled = cleanup_cancelled or cancelled
                logger.debug(f"{route} parent cancellation while cleaning up {task_name} task")
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as cleanup_exc:
                logger.debug(f"{route} cleanup for {task_name} task failed: error={cleanup_exc}")
        if cleanup_cancelled is not None:
            raise cleanup_cancelled


_AUTH_FRAME_TIMEOUT_S = 5.0
# What the client is told when its WebSocket is refused, by the refusal's reason; never the credential.
_AUTH_REFUSAL_MESSAGES = {
    "authentication_timeout": 'Authentication timeout. Send {"type": "auth", "token": ...} first.',
    "expected_auth_frame": 'Authentication required. Send {"type": "auth", "token": ...} first.',
    "authentication_required": 'Authentication required. Send {"type": "auth", "token": ...} first.',
    "invalid_token": "Invalid or expired credentials",
    "user_principal_required": "Invalid or expired credentials",
    "insufficient_scope": "Credential not permitted",
    "cookie_auth_rejected": "Session not accepted from this origin",
    "auth_unavailable": "Authentication is unavailable",
}


def _policy_close_code() -> int:
    return 1008 if is_truthy(os.getenv("AUDIO_WS_QUOTA_CLOSE_1008", "0")) else 4003


def _json_value(raw: Any) -> Any:
    """A key's JSON column (endpoints, metadata) as a value; malformed text counts as unset, as before."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _check_allowed(endpoints: Any, paths: Any, *, endpoint_id: str, ws_path: str) -> None:
    if isinstance(endpoints, list) and endpoints and endpoint_id not in [str(item) for item in endpoints]:
        raise WebSocketException(code=4403, reason="Endpoint not permitted for this credential")
    if isinstance(paths, list) and paths and not any(ws_path.startswith(str(prefix)) for prefix in paths):
        raise WebSocketException(code=4403, reason="Path not permitted for this credential")


def _call_quota(limits: dict[str, Any]) -> tuple[Optional[int], Optional[str]]:
    """The credential's call limit (``max_runs``, else ``max_calls``) and its daily bucket, if any."""
    limit = limits.get("max_runs")
    if limit is None:
        limit = limits.get("max_calls")
    if not isinstance(limit, int) or limit < 0:
        return None, None
    period = limits.get("period")
    if isinstance(period, str) and period.lower() == "day":
        from datetime import datetime, timezone

        return limit, datetime.now(timezone.utc).date().isoformat()
    return limit, None


async def _enforce_scoped_credential_limits(
    websocket: WebSocket, principal: AuthPrincipal, *, endpoint_id: str, ws_path: str
) -> None:
    """A scoped JWT's or a multi-user API key's own limits (allowed endpoints and paths, a call quota),
    as the audio routes enforced them before the core resolver; admins (the JWT's role, the key's
    scope) are exempt. Raises ``WebSocketException``."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.quotas import (
        increment_and_check_api_key_quota,
        increment_and_check_jwt_quota,
    )

    verified = getattr(websocket.state, "ws_credential_headers", None) or {}
    if principal.token_type == "access":  # a JWT: a scoped one carries its limits as claims
        from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

        claims = get_jwt_service().decode_access_token(bearer_credential(verified.get("authorization")) or "")
        if str(claims.get("role", "")).lower() == "admin":
            return
        _check_allowed(claims.get("allowed_endpoints"), claims.get("allowed_paths"), endpoint_id=endpoint_id, ws_path=ws_path)
        limit, bucket = _call_quota(claims)
        if limit is None:
            return
        allowed, _count = await increment_and_check_jwt_quota(
            db_pool=await get_db_pool(), jti=str(claims.get("jti")), counter_type="call", limit=limit, bucket=bucket
        )
    elif principal.api_key_id is not None:  # a multi-user key: its limits live on the key
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
        from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        info = await (await get_api_key_manager()).validate_api_key(
            api_key=verified.get("x-api-key") or "",
            ip_address=resolve_client_ip(websocket, get_settings()),
            record_usage=False,
        )
        if not info:
            raise WebSocketException(code=4401, reason="invalid_token")
        if str(info.get("scope", "")).lower() == "admin":
            return
        metadata = _json_value(info.get("metadata"))
        metadata = metadata if isinstance(metadata, dict) else {}
        endpoints = _json_value(info.get("llm_allowed_endpoints"))
        _check_allowed(endpoints, metadata.get("allowed_paths"), endpoint_id=endpoint_id, ws_path=ws_path)
        limit, bucket = _call_quota(metadata)
        if limit is None:
            return
        allowed, _count = await increment_and_check_api_key_quota(
            db_pool=await get_db_pool(), api_key_id=int(info["id"]), counter_type="call", limit=limit, bucket=bucket
        )
    else:  # the single-user key or cookie: no per-credential limits
        return
    if not allowed:
        raise WebSocketException(code=_policy_close_code(), reason="Credential quota exceeded")


async def _refuse_audio_websocket(websocket: WebSocket, refused: WebSocketException) -> None:
    """Tell an open socket why it is refused (never the credential), then close with the refusal's code."""
    if websocket.application_state == WebSocketState.CONNECTED:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": _AUTH_REFUSAL_MESSAGES.get(refused.reason, refused.reason)})
    with contextlib.suppress(Exception):
        await websocket.close(code=refused.code)


async def _audio_ws_authenticate_impl(
    websocket: WebSocket,
    *,
    endpoint_id: str,
    ws_path: str,
    allow_initial_auth_message: bool = True,
) -> tuple[bool, Optional[int]]:
    """Resolve an audio WebSocket's caller with the core resolver (TASK-13950.7).

    Handshake credentials (``Authorization: Bearer``, ``X-API-KEY``, the bearer subprotocol) are
    verified before accept. Without them a streaming route (``allow_initial_auth_message``) reads an
    auth frame through ``resolve_ws_principal_first_frame``, where the single-user cookie's clients
    may send their data first; the realtime routes, whose protocol the server opens, take the cookie
    or refuse. ``?token=`` is read only with AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH set, through the
    deprecated query shim. On success a streaming route's socket is left accepted; a refusal is sent
    as an error frame when the socket is open, then closed with the refusal's code.
    """
    # Audio enforces a scoped token's own limits below, as an HTTP route with require_token_scope does; only
    # this module may set the mark (tests/lint/test_auth_token_parsing_ratchet.py). Not checked here yet: the
    # token's ``scope`` claim and ``allowed_methods``, which HTTP's TokenScopeGuard applies (TASK-13950.73).
    websocket.state.ws_token_scope_enforced = True
    try:
        if not has_ws_credential(websocket) and _allow_query_token_auth() and websocket.query_params.get("token"):
            # ``?token=`` only, as before; the shim's ``?api_key=`` was never an audio channel.
            principal = await resolve_ws_principal_or_deprecated_query(websocket, query_params=("token",))
        elif has_ws_credential(websocket) or not allow_initial_auth_message:
            principal = await resolve_ws_principal(websocket)
        else:
            principal = await resolve_ws_principal_first_frame(
                websocket, timeout=_AUTH_FRAME_TIMEOUT_S, cookie_may_send_data_first=True
            )
        if principal.user_id is None:  # a service token is no audio user
            raise WebSocketException(code=4401, reason="user_principal_required")
        await _enforce_scoped_credential_limits(websocket, principal, endpoint_id=endpoint_id, ws_path=ws_path)
        if allow_initial_auth_message and websocket.application_state != WebSocketState.CONNECTED:
            await websocket.accept(subprotocol=auth_subprotocol(websocket))
    except WebSocketException as refused:
        await _refuse_audio_websocket(websocket, refused)
        return False, None
    except WebSocketDisconnect:
        return False, None
    except Exception as exc:  # noqa: BLE001 - an audio socket fails closed, as it did before TASK-13950.7
        logger.warning("Audio WebSocket authentication failed error_type={}", type(exc).__name__)
        await _refuse_audio_websocket(websocket, WebSocketException(code=4401, reason="invalid_token"))
        return False, None
    state = websocket.state
    state.auth_principal, state.user_id = principal, principal.user_id
    state.org_ids, state.active_org_id = list(principal.org_ids or []), principal.active_org_id
    return True, principal.user_id


async def _audio_ws_authenticate(
    websocket: WebSocket,
    *,
    endpoint_id: str,
    ws_path: str,
    allow_initial_auth_message: bool = True,
) -> tuple[bool, Optional[int]]:
    """Authenticate an audio WebSocket, and bind its user for LLM usage accounting.

    The handler awaits this in its own task, so the bound user stays for the
    session and for the tasks the handler starts afterwards.
    """
    authenticated, user_id = await _audio_ws_authenticate_impl(
        websocket,
        endpoint_id=endpoint_id,
        ws_path=ws_path,
        allow_initial_auth_message=allow_initial_auth_message,
    )
    if authenticated:
        bind_llm_call_user(user_id)
    return authenticated, user_id
