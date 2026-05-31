from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
from tldw_Server_API.app.core.testing import is_truthy

_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return is_truthy(raw)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _compute_next_backoff(attempts: int) -> int:
    base = int(os.getenv("MEETINGS_WEBHOOK_DLQ_BASE_SEC", "30") or "30")
    cap = int(os.getenv("MEETINGS_WEBHOOK_DLQ_MAX_BACKOFF_SEC", "3600") or "3600")
    return max(1, min(cap, int(base * (2 ** max(0, attempts - 1)))))


def _next_attempt_iso(delay_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, int(delay_seconds)))).replace(microsecond=0).isoformat()


def _dispatch_exception_message(exc: BaseException) -> str:
    return "Meeting webhook delivery timed out" if isinstance(exc, TimeoutError) else "Meeting webhook delivery failed"


def discover_meetings_db_targets() -> list[tuple[Path, str]]:
    """Discover candidate per-user media DB paths for meeting dispatch processing."""
    targets: list[tuple[Path, str]] = []
    try:
        single_user_id = str(DatabasePaths.get_single_user_id())
        base_dir = Path(DatabasePaths.get_user_base_directory(single_user_id)).parent
    except _MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS:
        return targets

    candidate_user_ids: set[str] = {single_user_id}
    with contextlib.suppress(_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS):
        if base_dir.exists():
            for entry in base_dir.iterdir():
                if entry.is_dir() and entry.name:
                    candidate_user_ids.add(str(entry.name))

    for user_id in sorted(candidate_user_ids):
        db_path = base_dir / user_id / DatabasePaths.MEDIA_DB_NAME
        if db_path.exists():
            targets.append((db_path, user_id))
    return targets


async def _close_response(resp: object) -> None:
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()


async def _attempt_dispatch(
    *,
    dispatch_row: dict[str, Any],
    timeout_sec: float,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    payload = dispatch_row.get("payload_json") or {}
    if not isinstance(payload, dict):
        return False, "invalid_payload", None

    destination = payload.get("destination") or {}
    request_body = payload.get("request_body")
    webhook_url = str((destination or {}).get("url") or "").strip()
    if not webhook_url:
        return False, "missing_webhook_url", None

    policy = evaluate_url_policy(webhook_url)
    if not getattr(policy, "allowed", False):
        reason = str(getattr(policy, "reason", "") or "denied")
        return False, f"denied_by_policy:{reason}", {"reason": reason}

    try:
        response = await afetch(
            method="POST",
            url=webhook_url,
            json=request_body,
            timeout=timeout_sec,
            retry=RetryPolicy(attempts=1),
        )
        response_payload = {"status_code": int(getattr(response, "status_code", 0))}
        with contextlib.suppress(_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS):
            response_payload["body"] = str(getattr(response, "text", "") or "")[:500]
        try:
            status_code = int(getattr(response, "status_code", 0))
            if status_code < 400:
                return True, None, response_payload
            return False, f"status={status_code}", response_payload
        finally:
            await _close_response(response)
    except _MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS as exc:
        return False, _dispatch_exception_message(exc), None


async def run_meetings_webhook_dlq_worker(stop_event: asyncio.Event | None = None) -> None:
    """Retry and deliver queued meetings webhook/slack dispatches."""
    if not _env_bool("MEETINGS_WEBHOOK_DLQ_ENABLED", False):
        logger.info("Meetings webhook DLQ worker disabled")
        return

    stop = stop_event or asyncio.Event()
    interval = int(os.getenv("MEETINGS_WEBHOOK_DLQ_INTERVAL_SEC", "15") or "15")
    batch = int(os.getenv("MEETINGS_WEBHOOK_DLQ_BATCH", "25") or "25")
    timeout_sec = float(os.getenv("MEETINGS_WEBHOOK_DLQ_TIMEOUT_SEC", "10") or "10")
    max_attempts = int(os.getenv("MEETINGS_WEBHOOK_DLQ_MAX_ATTEMPTS", "8") or "8")

    logger.info(
        "Starting Meetings webhook DLQ worker "
        f"(interval={interval}s, batch={batch}, timeout={timeout_sec}s, max_attempts={max_attempts})"
    )

    db_instances: dict[tuple[str, str], MeetingsDatabase] = {}
    try:
        while not stop.is_set():
            targets = discover_meetings_db_targets()
            processed_any = False

            for db_path, user_id in targets:
                if stop.is_set():
                    break

                cache_key = (str(db_path), str(user_id))
                db = db_instances.get(cache_key)
                if db is None:
                    db = MeetingsDatabase(
                        db_path=db_path,
                        client_id=f"meetings-dlq-{user_id}",
                        user_id=user_id,
                    )
                    db_instances[cache_key] = db

                rows = db.claim_due_integration_dispatches(limit=batch, max_attempts=max_attempts)
                if not rows:
                    continue

                for row in rows:
                    if stop.is_set():
                        break
                    processed_any = True

                    dispatch_id = int(row.get("id") or 0)
                    session_id = str(row.get("session_id") or "")
                    integration_type = str(row.get("integration_type") or "")
                    attempts = int(row.get("attempts") or 0) + 1

                    ok, error, response_json = await _attempt_dispatch(dispatch_row=row, timeout_sec=timeout_sec)
                    if ok:
                        db.update_integration_dispatch(
                            dispatch_id=dispatch_id,
                            status="delivered",
                            attempts=attempts,
                            next_attempt_at=None,
                            last_error=None,
                            response_json=response_json,
                        )
                        with contextlib.suppress(_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS):
                            db.append_event(
                                session_id=session_id,
                                event_type="integration.delivered",
                                payload_json={
                                    "dispatch_id": dispatch_id,
                                    "integration_type": integration_type,
                                },
                            )
                        continue

                    denied_by_policy = bool(error and str(error).startswith("denied_by_policy:"))
                    if denied_by_policy or attempts >= max_attempts:
                        status = "failed"
                        next_attempt_at = None
                        event_type = "integration.failed"
                    else:
                        status = "retrying"
                        backoff_seconds = _compute_next_backoff(attempts)
                        next_attempt_at = _next_attempt_iso(backoff_seconds)
                        event_type = "integration.retrying"

                    db.update_integration_dispatch(
                        dispatch_id=dispatch_id,
                        status=status,
                        attempts=attempts,
                        next_attempt_at=next_attempt_at,
                        last_error=error or "unknown_error",
                        response_json=response_json,
                    )
                    with contextlib.suppress(_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS):
                        db.append_event(
                            session_id=session_id,
                            event_type=event_type,
                            payload_json={
                                "dispatch_id": dispatch_id,
                                "integration_type": integration_type,
                                "attempts": attempts,
                                "next_attempt_at": next_attempt_at,
                                "error": error,
                            },
                        )

            if processed_any:
                continue

            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=max(1, interval))
    finally:
        for db in db_instances.values():
            with contextlib.suppress(_MEETINGS_DLQ_NONCRITICAL_EXCEPTIONS):
                db.close_connection()

    logger.info("Meetings webhook DLQ worker stopped")
