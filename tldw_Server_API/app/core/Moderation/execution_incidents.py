"""Metadata-only operational incidents, independent of moderation admission.

Deployment: quiesce old phase-restricted response workers before activation.
Rollback must retain neutral records and compatible readers, never relabel/drop
incidents. Schema preparation runs before traffic, not inside incident writes.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter
from tldw_Server_API.app.core.Moderation.execution_models import ExecutionScope, FailureMetadata, Operation

ENGINE_FAILURE_EXCERPT = "[Moderation execution failed; content withheld]"
_lock = threading.Lock()
_report_thread: threading.Thread | None = None
_closed = False
_last_error = float("-inf")


def _after_fork() -> None:
    global _lock, _report_thread, _closed, _last_error
    _lock, _report_thread, _closed, _last_error = threading.Lock(), None, False, float("-inf")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def prepare_moderation_incident_store() -> None:
    """Have the DB owner transactionally prepare the schema before traffic."""
    ModerationReviewStore()


def _emit_reporter_error() -> None:
    """Rate limit the sanitized ERROR independently of metric availability."""
    global _last_error
    try:
        with _lock:
            now = time.monotonic()
            emit = now - _last_error >= 60
            if emit:
                _last_error = now
        if emit:
            logger.error("Moderation execution incident reporting unavailable; enforcement remains blocked")
    except Exception:  # noqa: BLE001 - diagnostics must preserve enforcement.
        # Observability must not mask the original enforcement error either.
        return


def _reporter_failure() -> None:
    """Signal degraded reporting without exception text, traceback or metadata."""
    try:
        log_counter("moderation_execution_reporter_failures_total", labels={})
    except Exception:  # noqa: BLE001 - attempt the independent ERROR signal too.
        _emit_reporter_error()
        return
    _emit_reporter_error()


def _claim(error: ModerationExecutionError, metadata: FailureMetadata, scope: ExecutionScope) -> bool:
    ModerationExecutionError(error.reason)
    if not scope.claim_report():
        return False
    try:
        log_counter(
            "moderation_execution_failures_total",
            labels={
                "reason": error.reason,
                "operation": metadata.operation,
                "phase": metadata.phase or "unspecified",
            },
        )
    except Exception:  # noqa: BLE001 - metrics cannot replace a typed block.
        _reporter_failure()
    return True


def _persist_failure(error: ModerationExecutionError, *, metadata: FailureMetadata, scope: ExecutionScope) -> None:
    """Persist only an explicit allowlist, atomically deduplicated by the store."""
    phase = metadata.phase or "unspecified"
    bucket = int(time.time() // 300)
    key = f"moderation-engine-v1:{metadata.operation}:{phase}:{error.reason}:{bucket}"
    context: dict[str, str | int] = {
        "reason": error.reason,
        "operation": metadata.operation,
        "input_chars": metadata.input_chars,
    }
    try:
        context["correlation_id"] = uuid.UUID(scope.correlation_id).hex
    except (ValueError, TypeError, AttributeError):
        pass
    store = ModerationReviewStore(busy_timeout_ms=250)
    store.upsert_item(
        {
            "idempotency_key": hashlib.sha256(key.encode("ascii")).hexdigest(),
            "phase": phase,
            "source_type": "moderation_engine",
            "category": "moderation_engine_failure",
            "severity": "high",
            "status": "needs_review",
            "recommended_action": "escalate",
            "excerpt": ENGINE_FAILURE_EXCERPT,
            "context": context,
            "safe_fields": {"excerpt": True, "context": True, "effective_policy": True, "matches": True},
            "effective_policy": {},
            "matches": [],
        }
    )


def report_execution_failure(
    error: ModerationExecutionError, *, operation: Operation, phase: str | None, scope: ExecutionScope, input_chars: int
) -> None:
    """Report once for direct sync callers; never replace the original block."""
    try:
        metadata = FailureMetadata(operation, phase, input_chars)
        if _claim(error, metadata, scope):
            _persist_failure(error, metadata=metadata, scope=scope)
    except Exception:  # noqa: BLE001 - store failure must preserve enforcement.
        _reporter_failure()


def report_execution_failure_background(
    error: ModerationExecutionError, *, metadata: FailureMetadata, scope: ExecutionScope
) -> threading.Thread | None:
    """Submit once to the single independent I/O slot, without waiting or queueing.

    Cleanup owners can report even after their caller's event loop disappears.
    The actual thread retains its lease through exit, including filesystem stalls.
    """
    global _report_thread
    try:
        metadata = FailureMetadata(metadata.operation, metadata.phase, metadata.input_chars)
        if not _claim(error, metadata, scope):
            return

        def run() -> None:
            try:
                _persist_failure(error, metadata=metadata, scope=scope)
            except Exception:  # noqa: BLE001 - sanitize all persistence failures.
                _reporter_failure()

        with _lock:
            if _report_thread is not None and not _report_thread.is_alive():
                _report_thread = None
            busy = _closed or _report_thread is not None
            if not busy:
                thread = threading.Thread(target=run, name="moderation-incident", daemon=True)
                _report_thread = thread
                try:
                    thread.start()
                except (RuntimeError, OSError):
                    _report_thread = None
                    raise
        if busy:
            _reporter_failure()
            return
        return thread
    except Exception:  # noqa: BLE001 - preserve the original execution error.
        _reporter_failure()
        return None


async def report_execution_failure_async(
    error: ModerationExecutionError, *, metadata: FailureMetadata, scope: ExecutionScope
) -> None:
    """Use the independent I/O slot and await it for at most 0.5 seconds."""
    global _report_thread
    try:
        thread = report_execution_failure_background(error, metadata=metadata, scope=scope)
        if thread is None:
            return
        deadline = time.monotonic() + 0.5
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _reporter_failure()
                return
            await asyncio.sleep(min(0.005, remaining))
        with _lock:
            if _report_thread is thread:
                _report_thread = None
    except Exception:  # noqa: BLE001 - preserve the original execution error.
        _reporter_failure()


async def close_moderation_incident_reporting() -> None:
    """Stop admission and wait at most 0.5 seconds without relinquishing leases."""
    global _closed, _report_thread
    with _lock:
        _closed = True
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        with _lock:
            if _report_thread is not None and not _report_thread.is_alive():
                _report_thread = None
            if _report_thread is None:
                return
        await asyncio.sleep(0.005)
