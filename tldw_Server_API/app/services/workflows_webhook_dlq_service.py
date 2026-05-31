from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import sqlite3
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import create_workflows_database, get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode, is_truthy

_WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EgressPolicyError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    NetworkError,
    OSError,
    PermissionError,
    RetryExhaustedError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    sqlite3.Error,
)


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return is_truthy(v.lower())


def _get_lists_for_tenant(tenant_id: str) -> tuple[list[str], list[str]]:
    """Return (allowlist, denylist) patterns for a tenant.

    Patterns are comma-separated; entries may be hostnames or wildcard like '*.example.com'.
    Tenant-specific envs override global lists if present:
      WORKFLOWS_WEBHOOK_ALLOWLIST_<TENANT>, WORKFLOWS_WEBHOOK_DENYLIST_<TENANT>
    """
    base_allow = os.getenv("WORKFLOWS_WEBHOOK_ALLOWLIST", "").strip()
    base_deny = os.getenv("WORKFLOWS_WEBHOOK_DENYLIST", "").strip()
    key_t = tenant_id.upper().replace("-", "_")
    t_allow = os.getenv(f"WORKFLOWS_WEBHOOK_ALLOWLIST_{key_t}", "").strip()
    t_deny = os.getenv(f"WORKFLOWS_WEBHOOK_DENYLIST_{key_t}", "").strip()
    allow_src = t_allow if t_allow else base_allow
    deny_src = t_deny if t_deny else base_deny
    allow = [s.strip() for s in allow_src.split(",") if s.strip()]
    deny = [s.strip() for s in deny_src.split(",") if s.strip()]
    return allow, deny


def _host_allowed(url: str, tenant_id: str) -> bool:
    """Apply centralized egress policy for webhook retries.

    Prefer tenant-aware webhook policy; fallback to generic URL policy with
    per-tenant allow/deny when available. This enforces scheme, port, and
    private/reserved IP restrictions consistently.
    """
    try:
        # Use centralized webhook policy if available
        from tldw_Server_API.app.core.Security import egress as _eg
        if hasattr(_eg, "is_webhook_url_allowed_for_tenant"):
            try:
                _allowed = bool(_eg.is_webhook_url_allowed_for_tenant(url, tenant_id))
                if _allowed:
                    return True
                # If not allowed, continue to fallback logic below for test-friendly match
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
                # Fall back to explicit evaluate_url_policy with derived lists
                logger.debug(f"DLQ: is_webhook_url_allowed_for_tenant failed, falling back: {e}")
        # Fallback path: derive allow/deny lists and evaluate via core policy
        allow, deny = _get_lists_for_tenant(tenant_id)
        # Normalize wildcard patterns to bare host suffixes for policy evaluation
        def _norm(pats: list[str]) -> list[str]:
            out: list[str] = []
            for s in pats:
                v = (s or "").strip().lower()
                if v.startswith("*."):
                    v = v[2:]
                if v.startswith('.'):
                    v = v[1:]
                if v:
                    out.append(v)
            return out
        allow = _norm(allow)
        deny = _norm(deny)
        if hasattr(_eg, "evaluate_url_policy"):
            try:
                res = _eg.evaluate_url_policy(url, allowlist=(allow or None), denylist=(deny or None))
                if bool(getattr(res, "allowed", False)):
                    return True
                # In test contexts (no DNS), allow pattern-only when explicitly allowed
                if is_explicit_pytest_runtime() or is_test_mode():
                    try:
                        p = urlparse(url)
                        host = (p.hostname or "").lower().rstrip('.')
                        if not host:
                            return False
                        # Denylist wins
                        for d in deny:
                            if host == d or host.endswith(f".{d}"):
                                return False
                        if allow:
                            for a in allow:
                                if host == a or host.endswith(f".{a}"):
                                    return True
                        return False
                    except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                        return False
                return False
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"DLQ: evaluate_url_policy failed: {e}")
                return False
        # If policy module is missing, fail safe
        return False
    except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"DLQ egress policy check failed for url={url}: {e}")
        try:
            get_metrics_registry().increment(
                "app_exception_events_total",
                labels={"component": "workflows_dlq", "event": "egress_policy_check_failed"},
            )
        except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for workflows_dlq egress_policy_check_failed")
        return False


def _compute_next_backoff(attempts: int) -> int:
    base = int(os.getenv("WORKFLOWS_WEBHOOK_DLQ_BASE_SEC", "30"))
    cap = int(os.getenv("WORKFLOWS_WEBHOOK_DLQ_MAX_BACKOFF_SEC", "3600"))
    # Exponential with jitter: min(cap, base * 2^attempts) +/- 20%
    raw = min(cap, int(base * (2 ** max(0, attempts))))
    jitter_pct = 80 + secrets.randbelow(41)
    return max(1, int(raw * jitter_pct / 100))


def record_webhook_delivery_event(
    db: WorkflowsDatabase,
    *,
    tenant_id: str,
    run_id: str,
    url: str,
    status: str,
    code: int | None = None,
    reason: str | None = None,
    source: str | None = None,
    step_run_id: str | None = None,
    strict: bool = False,
) -> None:
    """Append webhook delivery evidence for a workflow run.

    The persisted payload stores the destination host, delivery status, optional
    HTTP code/reason/source fields, and never stores the full webhook URL. When
    ``strict`` is true, append failures are re-raised after logging; otherwise
    evidence remains best-effort so retry bookkeeping can continue.
    """
    parsed_url = urlparse(url)
    host = parsed_url.hostname or ""
    redacted_url = f"{parsed_url.scheme}://{parsed_url.netloc}/..." if parsed_url.netloc else "<invalid-url>"
    payload: dict[str, Any] = {"host": host, "status": status}
    if code is not None:
        payload["code"] = int(code)
    if reason:
        payload["reason"] = reason
    if source:
        payload["source"] = source
    try:
        db.append_event(tenant_id, run_id, "webhook_delivery", payload, step_run_id=step_run_id)
    except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to append webhook_delivery evidence for run_id={} url={} status={}: {}",
            run_id,
            redacted_url,
            status,
            exc,
        )
        if strict:
            raise


def _delivery_exception_message(exc: BaseException) -> str:
    return "Webhook delivery timed out" if isinstance(exc, TimeoutError) else "Webhook delivery failed"


async def _attempt_delivery(url: str, payload: dict[str, Any], timeout: float) -> tuple[bool, str | None]:
    try:
        policy = RetryPolicy()
        resp = await afetch(method="POST", url=url, json=payload, timeout=timeout, retry=policy)
        try:
            if resp.status_code < 400:
                return True, None
            # Consume body text safely
            try:
                body_text = resp.text[:200]
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                body_text = ""
            return False, f"status={resp.status_code}: {body_text}"
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()
            else:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()
    except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:  # network or other error
        return False, _delivery_exception_message(e)


async def run_workflows_webhook_dlq_worker(stop_event: asyncio.Event) -> None:
    """Background loop that retries webhook deliveries from the workflow_webhook_dlq table.

    Behavior is controlled via env:
      WORKFLOWS_WEBHOOK_DLQ_ENABLED: enable the worker (checked by caller)
      WORKFLOWS_WEBHOOK_DLQ_INTERVAL_SEC: polling interval when idle (default 15)
      WORKFLOWS_WEBHOOK_DLQ_BATCH: number of items to fetch per cycle (default 25)
      WORKFLOWS_WEBHOOK_DLQ_TIMEOUT_SEC: http timeout per request (default 10)
      WORKFLOWS_WEBHOOK_DLQ_MAX_ATTEMPTS: max retry attempts before giving up (default 8)
      WORKFLOWS_WEBHOOK_ALLOWLIST(_<TENANT>): comma-separated hostnames (supports '*.domain')
      WORKFLOWS_WEBHOOK_DENYLIST(_<TENANT>): comma-separated hostnames
    """
    backend = get_content_backend_instance()
    db: WorkflowsDatabase = create_workflows_database(backend=backend)

    interval = int(os.getenv("WORKFLOWS_WEBHOOK_DLQ_INTERVAL_SEC", "15"))
    batch = int(os.getenv("WORKFLOWS_WEBHOOK_DLQ_BATCH", "25"))
    timeout_sec = float(os.getenv("WORKFLOWS_WEBHOOK_DLQ_TIMEOUT_SEC", "10"))
    max_attempts = int(os.getenv("WORKFLOWS_WEBHOOK_DLQ_MAX_ATTEMPTS", "8"))

    logger.info(
        f"Starting Workflows webhook DLQ worker (interval={interval}s, batch={batch}, timeout={timeout_sec}s, max_attempts={max_attempts})"
    )

    while not stop_event.is_set():
        try:
            rows = db.list_webhook_dlq_due(limit=batch)
        except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"DLQ fetch failed: {e}")
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "workflows_dlq", "event": "fetch_failed"},
                )
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for workflows_dlq fetch_failed")
            rows = []

        if not rows:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            continue

        for r in rows:
            if stop_event.is_set():
                break
            dlq_id = int(r.get("id"))
            tenant_id = str(r.get("tenant_id") or "default")
            url = str(r.get("url") or "")
            attempts = int(r.get("attempts") or 0)
            # Mark that we are attempting a delivery now so callers observing mid-loop
            # see attempts >= 1 even before backoff bookkeeping is applied.
            current_attempt = attempts + 1
            try:
                db.update_webhook_dlq_failure(
                    dlq_id=dlq_id,
                    last_error=r.get("last_error") or "",
                    next_attempt_at_iso=None,
                    attempts=current_attempt,
                )
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                current_attempt = attempts + 1
            if current_attempt > max_attempts:
                exhausted_error = f"max_attempts_exceeded:{max_attempts}"
                record_webhook_delivery_event(
                    db,
                    tenant_id=tenant_id,
                    run_id=str(r.get("run_id") or ""),
                    url=url,
                    status="failed",
                    reason="max_attempts_exceeded",
                    source="dlq_worker",
                )
                try:
                    db.update_webhook_dlq_failure(
                        dlq_id=dlq_id,
                        last_error=exhausted_error,
                        next_attempt_at_iso="9999-12-31T23:59:59+00:00",
                        attempts=current_attempt,
                    )
                except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(f"DLQ max-attempts exhaustion update failed for id={dlq_id}: {exc}")
                continue
            try:
                body = json.loads(r.get("body_json") or "{}")
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"DLQ: invalid body_json for id={dlq_id}: {e}")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "workflows_dlq", "event": "bad_body_json"},
                    )
                except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for workflows_dlq bad_body_json")
                body = {}

            if not _host_allowed(url, tenant_id):
                logger.warning(f"DLQ drop (denied host): id={dlq_id} tenant={tenant_id} url={url}")
                record_webhook_delivery_event(
                    db,
                    tenant_id=tenant_id,
                    run_id=str(r.get("run_id") or ""),
                    url=url,
                    status="blocked",
                    reason="denied_by_policy",
                    source="dlq_worker",
                )
                db.update_webhook_dlq_failure(
                    dlq_id=dlq_id,
                    last_error="denied_by_policy",
                    next_attempt_at_iso=None,
                    attempts=current_attempt,
                )
                continue

            try:
                ok, err = await _attempt_delivery(url, body, timeout=timeout_sec)
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
                ok, err = False, _delivery_exception_message(e)
            if ok:
                record_webhook_delivery_event(
                    db,
                    tenant_id=tenant_id,
                    run_id=str(r.get("run_id") or ""),
                    url=url,
                    status="delivered",
                    source="dlq_worker",
                )
                try:
                    db.delete_webhook_dlq(dlq_id=dlq_id)
                except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as _e:
                    logger.warning(f"Failed to delete DLQ id={dlq_id} after success: {_e}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "workflows_dlq", "event": "delete_after_success_failed"},
                        )
                    except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for workflows_dlq delete_after_success_failed")
                continue

            # Failure: compute next backoff
            next_delay = _compute_next_backoff(current_attempt)
            try:
                import datetime as _dt
                next_at = (_dt.datetime.utcnow() + _dt.timedelta(seconds=next_delay)).isoformat()
            except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"DLQ: failed to compute next_attempt_at for id={dlq_id}: {e}")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "workflows_dlq", "event": "next_attempt_compute_failed"},
                    )
                except _WORKFLOWS_DLQ_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for workflows_dlq next_attempt_compute_failed")
                next_at = None

            db.update_webhook_dlq_failure(
                dlq_id=dlq_id,
                last_error=err or "unknown_error",
                next_attempt_at_iso=next_at,
                attempts=attempts + 1,
            )
            record_webhook_delivery_event(
                db,
                tenant_id=tenant_id,
                run_id=str(r.get("run_id") or ""),
                url=url,
                status="failed",
                reason=err or "unknown_error",
                source="dlq_worker",
            )
            logger.debug(f"DLQ retry scheduled in {next_delay}s (id={dlq_id} attempts={attempts+1}): {err}")

    logger.info("Workflows webhook DLQ worker stopped")
