"""
Evaluations audit adapter

Bridges legacy Evaluations audit events to the unified audit service.
Provides sync and async wrappers that enforce mandatory audit logging
without requiring DI at call sites.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import threading
import warnings
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id_optional,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    MandatoryAuditWriteError,
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
    except Exception:
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
            name="evaluations-audit-sync-loop",
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
        with contextlib.suppress(Exception):
            loop.call_soon_threadsafe(loop.stop)

    if thread and thread.is_alive() and threading.current_thread() is not thread:
        thread.join(timeout=2.0)

    _SYNC_LOOP = None
    _SYNC_LOOP_THREAD = None
    _SYNC_LOOP_READY.clear()


async def _get_svc(user_id: str | None) -> UnifiedAuditService:
    """Resolve the shared audit service via the central cache."""
    return await get_or_create_audit_service_for_user_id_optional(user_id)


def _schedule(coro) -> None:
    loop = _ensure_sync_loop()
    if loop is None:
        # No loop available; close coroutine and surface the failure.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
            try:
                coro.close()  # type: ignore[attr-defined]
            except Exception as close_error:
                logger.debug("Failed to close coroutine while handling missing event loop", exc_info=close_error)
        raise RuntimeError("Evaluations audit adapter unavailable: no event loop")

    try:
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
            try:
                coro.close()  # type: ignore[attr-defined]
            except Exception as close_error:
                logger.debug("Failed to close coroutine while handling scheduling failure", exc_info=close_error)
        raise RuntimeError("Evaluations audit adapter failed to schedule") from exc

    try:
        fut.result(timeout=_MANDATORY_TIMEOUT_S)
    except Exception as exc:
        logger.error(f"Evaluations audit sync task failed: {exc}")
        raise


async def _emit(
    *,
    user_id: str | None,
    event_type: UEvent,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    result: str = "success",
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        svc = await _get_svc(user_id)
        ctx = AuditContext(user_id=user_id)
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
    except Exception as exc:
        logger.error(
            "Mandatory evaluations audit write failed for {}:{} ({})",
            action,
            resource_id,
            type(exc).__name__,
            exc_info=True,
        )
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc


# Convenience wrappers

async def log_evaluation_created_async(*, user_id: str, eval_id: str, name: str, eval_type: str) -> None:
    await _emit(user_id=user_id, event_type=UEvent.DATA_WRITE, action="evaluation_create", resource_type="evaluation", resource_id=eval_id, metadata={"name": name, "type": eval_type})


async def log_evaluation_updated_async(*, user_id: str, eval_id: str, updates: dict[str, Any]) -> None:
    await _emit(user_id=user_id, event_type=UEvent.DATA_UPDATE, action="evaluation_update", resource_type="evaluation", resource_id=eval_id, metadata={"updates": list(updates.keys())})


async def log_evaluation_deleted_async(*, user_id: str, eval_id: str) -> None:
    await _emit(user_id=user_id, event_type=UEvent.DATA_DELETE, action="evaluation_delete", resource_type="evaluation", resource_id=eval_id)


async def log_evaluation_exported_async(*, user_id: str, eval_id: str, eval_type: str, export_format: str, total: int | None = None) -> None:
    metadata: dict[str, Any] = {"type": eval_type, "format": export_format}
    if total is not None:
        metadata["total"] = int(total)
    await _emit(user_id=user_id, event_type=UEvent.DATA_EXPORT, action="evaluation_export", resource_type="evaluation", resource_id=eval_id, metadata=metadata)


async def log_run_started_async(*, user_id: str, run_id: str, eval_id: str, target_model: str) -> None:
    await _emit(user_id=user_id, event_type=UEvent.EVAL_STARTED, action="run_create", resource_type="evaluation_run", resource_id=run_id, metadata={"eval_id": eval_id, "target_model": target_model})


async def log_run_cancelled_async(*, user_id: str, run_id: str) -> None:
    await _emit(user_id=user_id, event_type=UEvent.EVAL_FAILED, action="run_cancelled", resource_type="evaluation_run", resource_id=run_id, result="failure")


async def log_dataset_created_async(*, user_id: str, dataset_id: str, name: str, samples: int) -> None:
    await _emit(user_id=user_id, event_type=UEvent.DATA_WRITE, action="dataset_create", resource_type="dataset", resource_id=dataset_id, metadata={"name": name, "samples": samples})


async def log_dataset_deleted_async(*, user_id: str, dataset_id: str) -> None:
    await _emit(user_id=user_id, event_type=UEvent.DATA_DELETE, action="dataset_delete", resource_type="dataset", resource_id=dataset_id)


async def log_webhook_registration_async(*, user_id: str, webhook_id: str | None, url: str, events: list[str], success: bool, error: str | None = None) -> None:
    event = UEvent.SECURITY_SCAN if success else UEvent.SECURITY_VIOLATION
    res = "success" if success else "failure"
    await _emit(user_id=user_id, event_type=event, action="webhook_register", resource_type="webhook", resource_id=str(webhook_id) if webhook_id else None, result=res, metadata={"url": url, "events": events, "error": error})


async def log_webhook_unregistration_async(*, user_id: str, webhook_id: str | None, url: str, events: list[str], success: bool, error: str | None = None) -> None:
    event = UEvent.SECURITY_SCAN if success else UEvent.SECURITY_VIOLATION
    res = "success" if success else "failure"
    await _emit(user_id=user_id, event_type=event, action="webhook_unregister", resource_type="webhook", resource_id=str(webhook_id) if webhook_id else None, result=res, metadata={"url": url, "events": events, "error": error})


def log_evaluation_created(*, user_id: str, eval_id: str, name: str, eval_type: str) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_WRITE, action="evaluation_create", resource_type="evaluation", resource_id=eval_id, metadata={"name": name, "type": eval_type}))


def log_evaluation_updated(*, user_id: str, eval_id: str, updates: dict[str, Any]) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_UPDATE, action="evaluation_update", resource_type="evaluation", resource_id=eval_id, metadata={"updates": list(updates.keys())}))


def log_evaluation_deleted(*, user_id: str, eval_id: str) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_DELETE, action="evaluation_delete", resource_type="evaluation", resource_id=eval_id))


def log_evaluation_exported(*, user_id: str, eval_id: str, eval_type: str, export_format: str, total: int | None = None) -> None:
    metadata: dict[str, Any] = {"type": eval_type, "format": export_format}
    if total is not None:
        metadata["total"] = int(total)
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_EXPORT, action="evaluation_export", resource_type="evaluation", resource_id=eval_id, metadata=metadata))


def log_run_started(*, user_id: str, run_id: str, eval_id: str, target_model: str) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.EVAL_STARTED, action="run_create", resource_type="evaluation_run", resource_id=run_id, metadata={"eval_id": eval_id, "target_model": target_model}))


def log_run_cancelled(*, user_id: str, run_id: str) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.EVAL_FAILED, action="run_cancelled", resource_type="evaluation_run", resource_id=run_id, result="failure"))


def log_dataset_created(*, user_id: str, dataset_id: str, name: str, samples: int) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_WRITE, action="dataset_create", resource_type="dataset", resource_id=dataset_id, metadata={"name": name, "samples": samples}))


def log_dataset_deleted(*, user_id: str, dataset_id: str) -> None:
    _schedule(_emit(user_id=user_id, event_type=UEvent.DATA_DELETE, action="dataset_delete", resource_type="dataset", resource_id=dataset_id))


def log_webhook_registration(*, user_id: str, webhook_id: str | None, url: str, events: list[str], success: bool, error: str | None = None) -> None:
    event = UEvent.SECURITY_SCAN if success else UEvent.SECURITY_VIOLATION
    res = "success" if success else "failure"
    _schedule(_emit(user_id=user_id, event_type=event, action="webhook_register", resource_type="webhook", resource_id=str(webhook_id) if webhook_id else None, result=res, metadata={"url": url, "events": events, "error": error}))


def log_webhook_unregistration(*, user_id: str, webhook_id: str | None, url: str, events: list[str], success: bool, error: str | None = None) -> None:
    event = UEvent.SECURITY_SCAN if success else UEvent.SECURITY_VIOLATION
    res = "success" if success else "failure"
    _schedule(_emit(user_id=user_id, event_type=event, action="webhook_unregister", resource_type="webhook", resource_id=str(webhook_id) if webhook_id else None, result=res, metadata={"url": url, "events": events, "error": error}))


def shutdown_local_evaluations_audit_loop() -> None:
    """Shutdown the adapter-local sync loop."""
    _stop_sync_loop()


async def shutdown_evaluations_audit_services() -> None:
    """Backward-compatible async wrapper for adapter-local cleanup."""
    shutdown_local_evaluations_audit_loop()


def _shutdown_on_exit() -> None:
    """Atexit handler for best-effort local cleanup without late logging."""
    with contextlib.suppress(Exception):
        shutdown_local_evaluations_audit_loop()


atexit.register(_shutdown_on_exit)
