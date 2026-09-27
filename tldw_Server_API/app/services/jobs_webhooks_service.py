from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os

from loguru import logger

from tldw_Server_API.app.core.http_client import afetch
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

_JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS = (
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
    UnicodeDecodeError,
    ValueError,
)

def _sign(secret: bytes, payload: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()

async def _close_response(resp: object) -> None:
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()


async def run_jobs_webhooks_worker(stop_event: asyncio.Event | None = None) -> None:
    """Emit signed webhooks on job.completed/job.failed from job_events outbox.

    Env:
      - JOBS_EVENTS_OUTBOX=true
      - JOBS_WEBHOOKS_ENABLED=true
      - JOBS_WEBHOOKS_URL=https://...
      - JOBS_WEBHOOKS_SECRET_KEYS=key1,key2 (rotating; first used for signing)
      - JOBS_WEBHOOKS_INTERVAL_SEC=1.0
      - JOBS_WEBHOOKS_TIMEOUT_SEC=5
    Headers:
      - X-Jobs-Event: job.completed|job.failed
      - X-Jobs-Event-Id: outbox id
      - X-Jobs-Timestamp: epoch seconds
      - X-Jobs-Signature: v1=<hex>
    Body: JSON of {event, attrs, job}
    """
    url = os.getenv("JOBS_WEBHOOKS_URL")
    if not (env_flag_enabled("JOBS_WEBHOOKS_ENABLED") and url):
        logger.info("Jobs webhooks worker disabled")
        return
    if not env_flag_enabled("JOBS_EVENTS_OUTBOX"):
        logger.warning("Jobs webhooks require JOBS_EVENTS_OUTBOX=true; refusing to start")
        return
    secrets = [(s.strip()).encode("utf-8") for s in (os.getenv("JOBS_WEBHOOKS_SECRET_KEYS", "").split(",")) if s.strip()]
    if not secrets:
        logger.warning("Jobs webhooks enabled but no secrets configured; refusing to start")
        return
    interval = float(os.getenv("JOBS_WEBHOOKS_INTERVAL_SEC", "1.0") or "1.0")
    timeout_s = float(os.getenv("JOBS_WEBHOOKS_TIMEOUT_SEC", "5") or "5")
    # Admin context for reading outbox across domains
    try:
        JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None)
    except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Jobs webhooks: failed to set RLS admin context: {e}")
        try:
            get_metrics_registry().increment(
                "app_warning_events_total",
                labels={"component": "jobs_webhooks", "event": "set_rls_context_failed"},
            )
        except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for set_rls_context_failed")
    try:
        jm = JobManager()
        after_id = 0
        # Persistent cursor path (opt-in via env, defaults under project Databases/)
        cp = os.getenv("JOBS_WEBHOOKS_CURSOR_PATH")
        if cp:
            cursor_path = cp
        else:
            try:
                from pathlib import Path as _Path

                from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr
                cursor_path = str(_Path(_gpr()) / "Databases" / "jobs_webhooks_cursor.txt")
            except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                # Last resort: relative to this module's package root
                from pathlib import Path as _Path
                cursor_path = str(_Path(__file__).resolve().parents[3] / "Databases" / "jobs_webhooks_cursor.txt")
        # Resume from persisted cursor if present, unless explicitly overridden by env.
        # In TEST_MODE: allow resume only when an explicit JOBS_WEBHOOKS_CURSOR_PATH is provided,
        # to avoid cross-test interference with a shared global file.
        persisted_after = None
        try:
            _is_test = is_test_mode()
            allow_resume = True
            if _is_test and not os.getenv("JOBS_WEBHOOKS_CURSOR_PATH"):
                allow_resume = False
            if cursor_path and os.path.exists(cursor_path) and allow_resume:
                with open(cursor_path, encoding="utf-8") as f:
                    persisted_after = int((f.read() or "0").strip() or 0)
        except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Jobs webhooks: failed to read cursor file: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "jobs_webhooks", "event": "read_cursor_failed"},
                )
            except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for read_cursor_failed")
            persisted_after = None
        try:
            env_after = int(os.getenv("JOBS_WEBHOOKS_AFTER_ID", "0") or 0)
        except (TypeError, ValueError) as e:
            logger.debug(f"Jobs webhooks: invalid JOBS_WEBHOOKS_AFTER_ID; using 0: {e}")
            env_after = 0
        if env_after:
            after_id = env_after
        elif persisted_after:
            after_id = persisted_after
        logger.info("Starting Jobs webhooks worker")
        # Enforce egress policy per URL
        _is_test = is_test_mode()
        if not _is_test:
            try:
                from tldw_Server_API.app.core.Security.egress import evaluate_url_policy as _eval_policy
                pol = _eval_policy(url)
                if not getattr(pol, "allowed", False):
                    logger.warning(f"Jobs webhooks disabled: URL not allowed by egress policy ({getattr(pol, 'reason', 'denied')})")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "jobs_webhooks", "event": "egress_policy_denied"},
                        )
                    except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for egress_policy_denied")
                    return
            except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Jobs webhooks: egress policy check failed; refusing to start for safety: {e}")
                try:
                    get_metrics_registry().increment(
                        "app_exception_events_total",
                        labels={"component": "jobs_webhooks", "event": "egress_policy_check_failed"},
                    )
                except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for egress_policy_check_failed")
                return

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stopping Jobs webhooks worker on shutdown signal")
                return
            conn = jm._connect()
            try:
                rows = []
                if jm.backend == "postgres":
                    with jm._pg_cursor(conn) as cur:
                        cur.execute(
                            "SELECT id, event_type, attrs_json, job_id, domain, queue, job_type, created_at FROM job_events WHERE id > %s AND event_type IN ('job.completed','job.failed') ORDER BY id ASC LIMIT 200",
                            (int(after_id),),
                        )
                        rows = cur.fetchall() or []
                else:
                    rows = conn.execute(
                        "SELECT id, event_type, attrs_json, job_id, domain, queue, job_type, created_at FROM job_events WHERE id > ? AND event_type IN ('job.completed','job.failed') ORDER BY id ASC LIMIT 200",
                        (int(after_id),),
                    ).fetchall() or []
                if not rows:
                    await asyncio.sleep(interval)
                    continue
                for r in rows:
                    try:
                        if isinstance(r, dict):
                            eid = int(r.get("id"))
                            et = str(r.get("event_type"))
                            attrs = r.get("attrs_json")
                            job_id = r.get("job_id")
                            dom=r.get("domain")
                            que=r.get("queue")
                            jt=r.get("job_type")
                            ts=str(r.get("created_at"))
                        else:
                            eid = int(r[0])
                            et = str(r[1])
                            attrs = r[2]
                            job_id = r[3]
                            dom=r[4]
                            que=r[5]
                            jt=r[6]
                            ts=str(r[7])
                        # Construct payload
                        job_stub = {"id": job_id, "domain": dom, "queue": que, "job_type": jt}
                        try:
                            attrs_obj = json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
                        except (json.JSONDecodeError, TypeError):
                            attrs_obj = {}
                        body = json.dumps({"event": et, "attrs": attrs_obj, "job": job_stub, "created_at": ts}).encode("utf-8")
                        # Sign
                        import time
                        ts_epoch = int(time.time())
                        sig = _sign(secrets[0], str(ts_epoch).encode("utf-8") + b"." + body)
                        headers = {
                            "X-Jobs-Event": et,
                            "X-Jobs-Event-Id": str(eid),
                            "X-Jobs-Timestamp": str(ts_epoch),
                            "X-Jobs-Signature": f"v1={sig}",
                            "Content-Type": "application/json",
                        }
                        resp = await afetch(
                            method="POST",
                            url=url,
                            headers=headers,
                            data=body,
                            timeout=timeout_s,
                        )
                        try:
                            if resp.status_code >= 400:
                                logger.debug(f"Jobs webhook delivery failed {resp.status_code}: {resp.text}")
                                try:
                                    get_metrics_registry().increment(
                                        "app_warning_events_total",
                                        labels={"component": "jobs_webhooks", "event": "delivery_failed"},
                                    )
                                except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                                    logger.debug("metrics increment failed for delivery_failed")
                        finally:
                            await _close_response(resp)
                        after_id = eid
                    except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Jobs webhook send error: {e}")
                        try:
                            get_metrics_registry().increment(
                                "app_exception_events_total",
                                labels={"component": "jobs_webhooks", "event": "send_error"},
                            )
                        except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                            logger.debug("metrics increment failed for send_error")
                # Persist latest cursor for resume across restarts
                try:
                    if cursor_path:
                        os.makedirs(os.path.dirname(cursor_path), exist_ok=True)
                        with open(cursor_path, "w", encoding="utf-8") as f:
                            f.write(str(after_id))
                except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Jobs webhooks: failed to persist cursor: {e}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "jobs_webhooks", "event": "persist_cursor_failed"},
                        )
                    except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for persist_cursor_failed")
            finally:
                try:
                    conn.close()
                except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Jobs webhooks: failed to close connection: {e}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "jobs_webhooks", "event": "conn_close_failed"},
                        )
                    except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for conn_close_failed")
    finally:
        try:
            JobManager.clear_rls_context()
        except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Jobs webhooks: failed to clear RLS context: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "jobs_webhooks", "event": "clear_rls_context_failed"},
                )
            except _JOBS_WEBHOOKS_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for clear_rls_context_failed")
