"""
Embeddings audit adapter

Bridges legacy Embeddings audit calls to the unified audit service without
requiring FastAPI dependency injection at call sites. Exposes sync-friendly
wrappers that execute audit logging on a dedicated background loop and block
until completion to enforce mandatory auditing.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import sys
import threading
import warnings
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id_optional,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    UnifiedAuditService,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventType as UEvent,
)
from tldw_Server_API.app.core.testing import is_test_mode


def _parse_cache_size(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(f"Invalid {env_key}={raw!r}; using default {default}")
        return default
    if value < 1:
        logger.warning(f"{env_key} must be >= 1; using 1")
        return 1
    return value


_SYNC_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_LOOP_THREAD: threading.Thread | None = None
_SYNC_LOOP_LOCK = threading.Lock()
_SYNC_LOOP_READY = threading.Event()
_MANDATORY_TIMEOUT_S = 5.0


def _in_test_mode() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST")) or is_test_mode()


def _ensure_sync_loop() -> asyncio.AbstractEventLoop | None:
    """Ensure a background event loop exists for sync/threadpool calls."""
    global _SYNC_LOOP, _SYNC_LOOP_THREAD
    with _SYNC_LOOP_LOCK:
        if (
            _SYNC_LOOP_THREAD
            and _SYNC_LOOP_THREAD.is_alive()
            and _SYNC_LOOP
            and _SYNC_LOOP.is_running()
        ):
            return _SYNC_LOOP

        _SYNC_LOOP = None
        _SYNC_LOOP_READY.clear()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            global _SYNC_LOOP
            _SYNC_LOOP = loop
            _SYNC_LOOP_READY.set()
            loop.run_forever()
            loop.close()

        _SYNC_LOOP_THREAD = threading.Thread(
            target=_run,
            name="embeddings-audit-sync-loop",
            daemon=True,
        )
        _SYNC_LOOP_THREAD.start()

    if not _SYNC_LOOP_READY.wait(timeout=1.0):
        return None
    return _SYNC_LOOP


def _stop_sync_loop() -> None:
    """Stop the background sync loop if running."""
    global _SYNC_LOOP, _SYNC_LOOP_THREAD
    loop = _SYNC_LOOP
    thread = _SYNC_LOOP_THREAD

    if loop and loop.is_running():
        with contextlib.suppress(RuntimeError, TypeError, ValueError):
            loop.call_soon_threadsafe(loop.stop)

    if thread and thread.is_alive() and threading.current_thread() is not thread:
        thread.join(timeout=2.0)

    _SYNC_LOOP = None
    _SYNC_LOOP_THREAD = None
    _SYNC_LOOP_READY.clear()


async def _get_service_for_user(user_id: str | None) -> UnifiedAuditService:
    """Resolve the shared audit service via the central cache."""
    return await get_or_create_audit_service_for_user_id_optional(user_id)


async def _emit(
    user_id: str | None,
    event_type: UEvent,
    *,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    result: str = "success",
    metadata: dict[str, Any] | None = None,
    ip_address: str | None = None,
    endpoint: str | None = None,
    method: str | None = None,
) -> None:
    svc = await _get_service_for_user(user_id)
    ctx = AuditContext(
        user_id=str(user_id) if user_id is not None else None,
        ip_address=ip_address,
        endpoint=endpoint,
        method=method,
    )
    await svc.log_event(
        event_type=event_type,
        context=ctx,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        result=result,
        metadata=metadata,
    )
    # Mandatory audit: flush immediately and surface failures.
    await svc.flush(raise_on_failure=True)


def _schedule(coro) -> None:
    loop = _ensure_sync_loop()
    if loop is None:
        # No loop available; close coroutine and surface the failure.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
            try:
                coro.close()  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        raise RuntimeError("Embeddings audit adapter unavailable: no event loop")

    try:
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
            try:
                coro.close()  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        raise RuntimeError("Embeddings audit adapter failed to schedule") from exc

    try:
        fut.result(timeout=_MANDATORY_TIMEOUT_S)
    except Exception as exc:
        logger.error(f"Embeddings audit sync task failed: {exc}")
        raise


def log_security_violation(
    *, user_id: str | None, action: str, metadata: dict[str, Any] | None = None, ip_address: str | None = None
) -> None:
    """Log a security violation (sync wrapper)."""
    _schedule(_emit(user_id, UEvent.SECURITY_VIOLATION, action=action, result="failure", metadata=metadata, ip_address=ip_address))


def log_model_evicted(*, model_id: str, memory_usage_gb: float | None = None, reason: str | None = None) -> None:
    """Log a model eviction as a data deletion event."""
    meta = {"memory_usage_gb": memory_usage_gb, "reason": reason}
    _schedule(_emit(None, UEvent.DATA_DELETE, action="model_evicted", resource_type="embedding_model", resource_id=model_id, metadata=meta))


def log_memory_limit_exceeded(
    *, model_id: str, memory_usage_gb: float | None, current_usage_gb: float | None, limit_gb: float | None
) -> None:
    """Log memory limit exceeded as a system error event."""
    meta = {
        "memory_usage_gb": memory_usage_gb,
        "current_usage_gb": current_usage_gb,
        "limit_gb": limit_gb,
    }
    _schedule(_emit(None, UEvent.SYSTEM_ERROR, action="embeddings_memory_limit_exceeded", resource_type="embedding_model", resource_id=model_id, result="failure", metadata=meta))


# Async variants (use when already in an async context)
async def emit_security_violation_async(user_id: str | None, action: str, metadata: dict[str, Any] | None = None) -> None:
    await _emit(user_id, UEvent.SECURITY_VIOLATION, action=action, result="failure", metadata=metadata)


async def emit_model_evicted_async(model_id: str, memory_usage_gb: float | None = None, reason: str | None = None) -> None:
    await _emit(None, UEvent.DATA_DELETE, action="model_evicted", resource_type="embedding_model", resource_id=model_id, metadata={"memory_usage_gb": memory_usage_gb, "reason": reason})


async def emit_memory_limit_exceeded_async(
    model_id: str, memory_usage_gb: float | None, current_usage_gb: float | None, limit_gb: float | None
) -> None:
    await _emit(None, UEvent.SYSTEM_ERROR, action="embeddings_memory_limit_exceeded", resource_type="embedding_model", resource_id=model_id, result="failure", metadata={"memory_usage_gb": memory_usage_gb, "current_usage_gb": current_usage_gb, "limit_gb": limit_gb})


def shutdown_local_audit_adapter_loop() -> None:
    """Shutdown the adapter-local sync loop."""
    _stop_sync_loop()


async def shutdown_audit_adapter_services() -> None:
    """Backward-compatible async wrapper for adapter-local cleanup."""
    shutdown_local_audit_adapter_loop()


# Ensure services are shutdown at interpreter exit to avoid hanging tests
def _shutdown_on_exit() -> None:
    try:
        try:
            if getattr(sys, "stderr", None) is None or sys.stderr.closed:
                logger.remove()
            else:
                logger.disable("tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps")
                logger.disable("tldw_Server_API.app.core.Embeddings.audit_adapter")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        shutdown_local_audit_adapter_loop()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass


atexit.register(_shutdown_on_exit)
