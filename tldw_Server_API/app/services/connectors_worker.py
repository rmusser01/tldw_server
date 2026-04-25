from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Any

from loguru import logger

try:
    from tldw_Server_API.app.core.Jobs.manager import JobManager
except ImportError:  # pragma: no cover - optional
    JobManager = None  # type: ignore
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    get_media_repository,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

_CONNECTOR_NONCRITICAL_EXCEPTIONS = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


DOMAIN = "connectors"
_EMAIL_SYNC_METRICS_REGISTERED = False


def _email_sync_metric_labels(
    *,
    provider: str,
    status: str | None = None,
    reason: str | None = None,
    outcome: str | None = None,
) -> dict[str, str]:
    labels: dict[str, str] = {"provider": str(provider or "unknown")}
    if status:
        labels["status"] = str(status)
    if reason:
        labels["reason"] = str(reason)
    if outcome:
        labels["outcome"] = str(outcome)
    return labels


def _get_email_sync_metrics_registry():
    global _EMAIL_SYNC_METRICS_REGISTERED
    try:
        from tldw_Server_API.app.core.Metrics import (
            MetricDefinition,
            MetricType,
            get_metrics_registry,
        )
    except Exception:
        return None

    try:
        registry = get_metrics_registry()
    except Exception:
        return None

    if _EMAIL_SYNC_METRICS_REGISTERED:
        return registry

    with contextlib.suppress(Exception):
        registry.register_metric(
            MetricDefinition(
                name="email_sync_runs_total",
                type=MetricType.COUNTER,
                description="Total email sync runs by provider and status",
                labels=["provider", "status"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="email_sync_failures_total",
                type=MetricType.COUNTER,
                description="Total email sync failure events by provider and reason",
                labels=["provider", "reason"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="email_sync_recovery_events_total",
                type=MetricType.COUNTER,
                description="Email sync recovery events for cursor invalidation and replay",
                labels=["provider", "outcome"],
            )
        )
        registry.register_metric(
            MetricDefinition(
                name="email_sync_lag_seconds",
                type=MetricType.HISTOGRAM,
                description="Observed lag in seconds for successful email sync cycles",
                unit="s",
                labels=["provider"],
                buckets=[30, 60, 120, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400],
            )
        )
    _EMAIL_SYNC_METRICS_REGISTERED = True
    return registry


def _email_sync_metrics_increment(
    metric_name: str,
    *,
    labels: dict[str, str],
    value: float = 1,
) -> None:
    registry = _get_email_sync_metrics_registry()
    if registry is None:
        return
    with contextlib.suppress(Exception):
        registry.increment(metric_name, value, labels=labels)


def _email_sync_metrics_observe(
    metric_name: str,
    value: float,
    *,
    labels: dict[str, str],
) -> None:
    registry = _get_email_sync_metrics_registry()
    if registry is None:
        return
    with contextlib.suppress(Exception):
        registry.observe(metric_name, float(value), labels=labels)


def _decode_gmail_base64url(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    pad = "=" * ((4 - (len(raw) % 4)) % 4)
    try:
        decoded = base64.urlsafe_b64decode((raw + pad).encode("utf-8"))
        return decoded.decode("utf-8", errors="replace")
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return ""


def _gmail_headers_map(payload: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(payload, dict):
        return out
    headers = payload.get("headers") or []
    if not isinstance(headers, list):
        return out
    for header in headers:
        if not isinstance(header, dict):
            continue
        key = str(header.get("name") or "").strip().lower()
        value = str(header.get("value") or "").strip()
        if not key or not value:
            continue
        if key not in out:
            out[key] = value
    return out


def _gmail_part_headers_map(part: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(part, dict):
        return {}
    headers = part.get("headers")
    if not isinstance(headers, list):
        return {}
    out: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, dict):
            continue
        key = str(header.get("name") or "").strip().lower()
        value = str(header.get("value") or "").strip()
        if key and value and key not in out:
            out[key] = value
    return out


def _gmail_attachment_disposition(part: dict[str, Any] | None) -> str:
    headers_map = _gmail_part_headers_map(part)
    return str(headers_map.get("content-disposition") or "").strip().lower()


def _html_to_text(value: str) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    no_script = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
    no_style = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", no_script)
    no_tags = re.sub(r"(?is)<[^>]+>", " ", no_style)
    collapsed = re.sub(r"\s+", " ", html.unescape(no_tags)).strip()
    return collapsed


def _safe_int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return 0


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _ingest_connector_media(
    *,
    media_db: Any,
    url: str,
    title: str,
    media_type: str,
    content: str,
    keywords: list[str],
    overwrite: bool = False,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Route connector ingest through the repository API for real DB sessions.

    Lightweight test doubles that only expose ``add_media_with_keywords`` keep
    using that method directly so worker tests can stay narrow.
    """
    media_writer = get_media_repository(media_db)
    return media_writer.add_media_with_keywords(
        url=url,
        title=title,
        media_type=media_type,
        content=content,
        keywords=keywords,
        overwrite=overwrite,
        **kwargs,
    )


def _create_connector_media_db(user_id: int):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    return create_media_database(
        client_id=str(user_id),
        db_path=str(DatabasePaths.get_media_db_path(user_id)),
    )


def _close_connector_media_db(media_db: Any) -> None:
    with contextlib.suppress(_CONNECTOR_NONCRITICAL_EXCEPTIONS):
        media_db.close_connection()


def _parse_iso_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _gmail_retry_policy() -> tuple[int, int, int]:
    max_attempts = _safe_positive_int(os.getenv("EMAIL_SYNC_RETRY_MAX_ATTEMPTS"), 5)
    base_seconds = _safe_positive_int(os.getenv("EMAIL_SYNC_RETRY_BASE_SECONDS"), 60)
    max_backoff_seconds = _safe_positive_int(
        os.getenv("EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS"), 3600
    )
    if max_backoff_seconds < base_seconds:
        max_backoff_seconds = base_seconds
    return max_attempts, base_seconds, max_backoff_seconds


def _compute_exponential_backoff_seconds(
    retry_count: int,
    *,
    base_seconds: int,
    max_backoff_seconds: int,
) -> int:
    retry_n = max(1, int(retry_count))
    try:
        delay = int(base_seconds) * (2 ** (retry_n - 1))
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        delay = int(max_backoff_seconds)
    return min(int(max_backoff_seconds), max(int(base_seconds), int(delay)))


def _extract_error_status_code(err: Exception) -> int | None:
    try:
        response = getattr(err, "response", None)
        if response is not None:
            for attr in ("status_code", "status"):
                value = getattr(response, attr, None)
                if value is not None:
                    return int(value)
        for attr in ("status_code", "status"):
            value = getattr(err, attr, None)
            if value is not None:
                return int(value)
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return None
    return None


def _is_gmail_history_cursor_invalid(err: Exception) -> bool:
    status_code = _extract_error_status_code(err)
    text = str(err or "").strip().lower()
    if "starthistoryid" in text and ("too old" in text or "invalid" in text):
        return True
    if "history id" in text and ("too old" in text or "invalid" in text):
        return True
    if "historyid" in text and ("too old" in text or "invalid" in text):
        return True
    if status_code in {400, 404} and ("history" in text or "gmail/v1/users/me/history" in text):
        return True
    return False


def _append_query_term(query: str | None, term: str | None) -> str | None:
    base = str(query or "").strip()
    addon = str(term or "").strip()
    if not addon:
        return base or None
    if not base:
        return addon
    return f"{base} {addon}"


def _collect_gmail_body_text(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""

    plain_parts: list[str] = []
    html_parts: list[str] = []

    def _walk(part: dict[str, Any]) -> None:
        mime = str(part.get("mimeType") or "").strip().lower()
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        disposition = _gmail_attachment_disposition(part)
        filename = str(part.get("filename") or "").strip()
        is_attachment = bool(filename) or ("attachment" in disposition)
        if not is_attachment and mime.startswith("text/"):
            decoded = _decode_gmail_base64url(body.get("data"))
            if decoded.strip():
                if mime == "text/plain":
                    plain_parts.append(decoded.strip())
                elif mime == "text/html":
                    html_text = _html_to_text(decoded)
                    if html_text:
                        html_parts.append(html_text)
                else:
                    plain_parts.append(decoded.strip())
        for child in part.get("parts") or []:
            if isinstance(child, dict):
                _walk(child)

    _walk(payload)
    if plain_parts:
        return "\n\n".join(part for part in plain_parts if part).strip()
    if html_parts:
        return "\n\n".join(part for part in html_parts if part).strip()
    # Some messages place plain data at the root body with no parts.
    root_body = payload.get("body") if isinstance(payload.get("body"), dict) else {}
    return _decode_gmail_base64url(root_body.get("data")).strip()


def _collect_gmail_attachments(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    attachments: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    def _walk(part: dict[str, Any]) -> None:
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        filename = str(part.get("filename") or "").strip()
        attachment_id = str(body.get("attachmentId") or "").strip()
        disposition = _gmail_attachment_disposition(part)
        has_attachment_like_disposition = ("attachment" in disposition) or (
            "inline" in disposition and bool(filename or attachment_id)
        )
        if filename or attachment_id or has_attachment_like_disposition:
            headers_map = _gmail_part_headers_map(part)
            content_type = str(part.get("mimeType") or "").strip().lower() or None
            size_bytes = _safe_int_or_zero(body.get("size"))
            dedupe_key = (
                filename.lower(),
                content_type,
                size_bytes,
                attachment_id,
                headers_map.get("content-id"),
            )
            if dedupe_key in seen_keys:
                for child in part.get("parts") or []:
                    if isinstance(child, dict):
                        _walk(child)
                return
            seen_keys.add(dedupe_key)
            attachments.append(
                {
                    "filename": filename or None,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "content_id": headers_map.get("content-id"),
                    "disposition": headers_map.get("content-disposition"),
                }
            )
        for child in part.get("parts") or []:
            if isinstance(child, dict):
                _walk(child)

    _walk(payload)
    return attachments


def _gmail_internal_date_iso(message: dict[str, Any]) -> str | None:
    raw = str(message.get("internalDate") or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
        return dt.isoformat()
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return None


def _normalize_gmail_history_cursor(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _merge_gmail_history_cursor(current: str | None, candidate: Any) -> str | None:
    current_text = _normalize_gmail_history_cursor(current)
    candidate_text = _normalize_gmail_history_cursor(candidate)
    if not candidate_text:
        return current_text
    if not current_text:
        return candidate_text
    try:
        return candidate_text if int(candidate_text) >= int(current_text) else current_text
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        return candidate_text if candidate_text >= current_text else current_text


def _utc_now_db_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _file_sync_cursor_kind(provider: str) -> str | None:
    normalized = str(provider or "").strip().lower()
    if normalized == "drive":
        return "drive_start_page_token"
    if normalized == "onedrive":
        return "graph_delta_link"
    return None


def _determine_drive_export_mime(
    *,
    mime: str | None,
    allowed_export_formats: list[str],
    allowed_export_set: set[str],
    export_overrides: dict[str, str],
) -> str | None:
    export_mime = None
    if (mime or "").startswith("application/vnd.google-apps."):
        override_key = mime or ""
        ov = export_overrides.get(override_key) or export_overrides.get(override_key.split(".")[-1])
        if ov in {"pdf", "txt", "md"} and ov in allowed_export_set:
            export_mime = "application/pdf" if ov == "pdf" else "text/plain"
        else:
            if mime == "application/vnd.google-apps.presentation":
                export_mime = "application/pdf" if "pdf" in allowed_export_formats else "text/plain"
            elif mime == "application/vnd.google-apps.document":
                export_mime = (
                    "text/plain"
                    if ("txt" in allowed_export_formats or "md" in allowed_export_formats)
                    else ("application/pdf" if "pdf" in allowed_export_formats else "text/plain")
                )
            elif mime == "application/vnd.google-apps.spreadsheet":
                export_mime = "text/csv" if "txt" in allowed_export_formats else ("application/pdf" if "pdf" in allowed_export_formats else "text/csv")
    return export_mime


def _file_sync_skip_reason(
    *,
    name: str,
    mime: str | None,
    size: Any,
    is_folder: bool,
    include_types: list[str],
    exclude_patterns: list[str],
    allowed_file_types: list[str],
    max_bytes: int | None,
) -> str | None:
    from tldw_Server_API.app.core.External_Sources.policy import is_file_type_allowed

    if is_folder:
        return "folder"
    if exclude_patterns and any(fnmatch(name, pattern) for pattern in exclude_patterns):
        return "excluded_by_pattern"
    if include_types:
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext not in include_types:
            return "extension_not_included"
    if not is_file_type_allowed(name=name, mime=mime, allowed=allowed_file_types):
        return "disallowed_file_type"
    if max_bytes is not None and size is not None:
        try:
            if int(size) > max_bytes:
                return "file_too_large"
        except (TypeError, ValueError):
            return None
    return None


async def _convert_document_bytes_to_text(*, raw: bytes, name: str, effective_mime: str) -> str:
    content_text = ""
    try:
        if effective_mime == "application/pdf" or (not effective_mime and name.lower().endswith(".pdf")):
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf

                res = await asyncio.to_thread(process_pdf, file_input=raw, filename=name, parser="docling")
            except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import process_pdf

                res = await asyncio.to_thread(process_pdf, file_input=raw, filename=name, parser="pymupdf4llm")
            if isinstance(res, dict):
                content_text = (res.get("content") or "").strip()
        else:
            if raw:
                content_text = raw.decode("utf-8", errors="replace")
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("content conversion failed for {}: {}", name, exc)
    return content_text


async def run_connectors_worker(stop_event: asyncio.Event | None = None) -> None:
    """Minimal worker that acknowledges and completes connector jobs.

    Scaffold behavior: picks up jobs with domain 'connectors' and completes them
    immediately. Real ingestion/sync logic will be implemented later.
    """
    if JobManager is None:
        logger.warning("Jobs manager unavailable; connectors worker disabled")
        return
    jm = JobManager()
    worker_id = "connectors-worker"
    poll_sleep = float(os.getenv("CONNECTORS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    logger.info("Starting connectors worker (scaffold import processor)")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping connectors worker on shutdown signal")
            return
        try:
            job = jm.acquire_next_job(domain=DOMAIN, queue="default", lease_seconds=120, worker_id=worker_id)
            if not job:
                await asyncio.sleep(poll_sleep)
                continue
            jid = int(job["id"]) if job.get("id") is not None else None
            lease_id = str(job.get("lease_id")) if job.get("lease_id") else None
            try:
                # Process import job
                payload: dict[str, Any] = job.get("payload") or {}
                source_id = int(payload.get("source_id")) if payload.get("source_id") is not None else None
                user_id = int(payload.get("user_id")) if payload.get("user_id") is not None else None
                if not source_id or not user_id:
                    raise ValueError("invalid job payload")
                from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
                from tldw_Server_API.app.core.External_Sources.connectors_service import (
                    finish_source_sync_job,
                    start_source_sync_job,
                )

                pool = await get_db_pool()
                async with pool.transaction() as db:
                    sync_state = await start_source_sync_job(
                        db,
                        source_id=source_id,
                        job_id=str(jid),
                    )
                if str((sync_state or {}).get("active_job_id") or "") != str(jid):
                    jm.fail_job(
                        jid,
                        error=f"Active sync job already exists for source {source_id}",
                        retryable=True,
                        backoff_seconds=30,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        completion_token=lease_id,
                    )
                    continue
                job_type = str(job.get("job_type") or "import")
                await _process_import_job(
                    jm,
                    jid,
                    lease_id,
                    worker_id,
                    source_id,
                    user_id,
                    job_type=job_type,
                )
                async with pool.transaction() as db:
                    await finish_source_sync_job(
                        db,
                        source_id=source_id,
                        job_id=str(jid),
                        outcome="success",
                    )
            except _CONNECTOR_NONCRITICAL_EXCEPTIONS as _e:
                with contextlib.suppress(_CONNECTOR_NONCRITICAL_EXCEPTIONS):
                    if source_id:
                        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
                        from tldw_Server_API.app.core.External_Sources.connectors_service import finish_source_sync_job

                        pool = await get_db_pool()
                        async with pool.transaction() as db:
                            await finish_source_sync_job(
                                db,
                                source_id=source_id,
                                job_id=str(jid),
                                outcome="failure",
                                error=str(_e),
                            )
                jm.fail_job(jid, error=str(_e), retryable=False, worker_id=worker_id, lease_id=lease_id, completion_token=lease_id)
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
            await asyncio.sleep(poll_sleep)


async def start_connectors_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    enabled = env_flag_enabled("CONNECTORS_WORKER_ENABLED")
    if not enabled:
        return None
    managed_stop_event = stop_event or asyncio.Event()
    task = asyncio.create_task(run_connectors_worker(managed_stop_event), name="connectors-worker")
    return task


async def _process_import_job(
    jm,
    jid: int,
    lease_id: str | None,
    worker_id: str,
    source_id: int,
    user_id: int,
    *,
    job_type: str = "import",
) -> None:
    """Fetch source/account, enumerate items, and ingest into Media DB."""
    # DB access
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.External_Sources import get_connector_by_name
    from tldw_Server_API.app.core.External_Sources.connectors_service import (
        FILE_SYNC_PROVIDERS,
        REFERENCE_MANAGER_PROVIDERS,
        get_external_item_binding,
        get_account_for_user,
        get_account_tokens,
        record_item_event,
        get_source_by_id,
        get_source_sync_state,
        record_ingested_item,
        should_ingest_item,
        upsert_external_item_binding,
        upsert_source_sync_state,
        update_account_tokens,
    )

    pool = await get_db_pool()
    async with pool.transaction() as db:
        src = await get_source_by_id(db, user_id, source_id)
        if not src:
            raise ValueError("source not found or not owned by user")
        provider = str(src.get("provider"))
        account_id = int(src.get("account_id"))
        options = src.get("options") or {}
        remote_id = str(src.get("remote_id"))
        account_row = {}
        if provider in REFERENCE_MANAGER_PROVIDERS:
            account_row = await get_account_for_user(db, user_id, account_id) or {}
        tokens = await get_account_tokens(db, user_id, account_id)
        acct = dict(account_row)
        acct["tokens"] = dict(tokens or {})
        acct.setdefault("email", src.get("email"))
        for key, value in dict(tokens or {}).items():
            if key in {"access_token", "refresh_token"} or value in (None, ""):
                continue
            acct.setdefault(str(key), value)
        # Load org policy for this user (best-effort)
        try:
            from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
            from tldw_Server_API.app.core.External_Sources.connectors_service import get_policy as get_org_policy
            from tldw_Server_API.app.core.External_Sources.policy import get_default_policy_from_env
            memberships = await list_memberships_for_user(user_id)
            org_id = int((memberships[0] or {}).get("org_id") if memberships else 1)
            policy = await get_org_policy(db, org_id)
            if not policy:
                policy = get_default_policy_from_env(org_id)
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
            from tldw_Server_API.app.core.External_Sources.policy import get_default_policy_from_env
            policy = get_default_policy_from_env(1)

    # Prepare connector
    conn = get_connector_by_name(provider)
    # Helper to detect 401-like unauthorized errors from provider responses
    def _is_unauthorized(err: Exception) -> bool:
        status = None
        try:
            resp = getattr(err, "response", None)
            if resp is not None:
                status = getattr(resp, "status_code", None)
                if status is None:
                    status = getattr(resp, "status", None)
            if status is None:
                status = getattr(err, "status_code", None)
            if status is None:
                status = getattr(err, "status", None)
            if status is not None and int(status) == 401:
                return True
        except (AttributeError, TypeError, ValueError):
            pass
        # Fallback: inspect message
        msg = str(err).lower()
        return '401' in msg or 'unauthorized' in msg

    async def _attempt_with_refresh(call_coro, *args, **kwargs):
        nonlocal acct
        try:
            return await call_coro(*args, **kwargs)
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
            if not _is_unauthorized(e):
                raise
            # Try refresh if possible
            rtok = (acct.get('tokens') or {}).get('refresh_token')
            if not rtok:
                raise
            try:
                new_toks = None
                if provider in {"drive", "notion", "gmail"} and hasattr(conn, "refresh_access_token"):
                    new_toks = await conn.refresh_access_token(rtok)
                if not new_toks or not new_toks.get('access_token'):
                    raise
                # Persist and update local token cache
                async with pool.transaction() as db:
                    await update_account_tokens(db, user_id, account_id, new_toks)
                acct['tokens'].update(new_toks)
                # Retry once
                return await call_coro(*args, **kwargs)
            except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
                raise

    async def _renew_reference_manager_lease_until_stopped(stop_event: asyncio.Event) -> None:
        if not lease_id:
            return

        while not stop_event.is_set():
            try:
                renewed = jm.renew_job_lease(
                    jid,
                    seconds=120,
                    worker_id=worker_id,
                    lease_id=lease_id,
                )
                if not renewed:
                    logger.warning("Lease renewal failed for job {}, stopping heartbeat", jid)
                    return
            except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Lease renewal error for job {}: {}", jid, exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=30)
            except TimeoutError:
                continue

    # Prepare DB instance
    mdb = _create_connector_media_db(user_id)

    gmail_sync_state_supported = bool(
        provider == "gmail"
        and hasattr(mdb, "get_email_sync_state")
        and hasattr(mdb, "mark_email_sync_run_started")
        and hasattr(mdb, "mark_email_sync_run_succeeded")
        and hasattr(mdb, "mark_email_sync_run_failed")
    )
    gmail_sync_state_started = False
    gmail_previous_cursor: str | None = None
    gmail_cursor_candidate: str | None = None
    gmail_sync_errors: list[str] = []
    gmail_sync_source_key = str(source_id)
    gmail_sync_tenant_id = str(user_id)
    gmail_previous_success_at: datetime | None = None
    gmail_latest_message_internal_at: datetime | None = None
    gmail_run_status: str | None = None
    gmail_failure_reason: str | None = None
    gmail_cursor_recovery_active = False
    gmail_cursor_recovery_window_days = _safe_positive_int(
        os.getenv("EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS"),
        7,
    )
    gmail_cursor_recovery_max_messages = _safe_positive_int(
        os.getenv("EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES"),
        2000,
    )

    if gmail_sync_state_supported:
        try:
            state = mdb.get_email_sync_state(
                provider="gmail",
                source_key=gmail_sync_source_key,
                tenant_id=gmail_sync_tenant_id,
            )
            retry_count = int((state or {}).get("retry_backoff_count") or 0)
            last_error = str((state or {}).get("error_state") or "").strip()
            if retry_count > 0 and last_error:
                max_attempts, base_seconds, max_backoff_seconds = _gmail_retry_policy()
                backoff_seconds = _compute_exponential_backoff_seconds(
                    retry_count,
                    base_seconds=base_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                )
                if retry_count >= max_attempts:
                    logger.warning(
                        "gmail sync retry budget exhausted for source_id={} user_id={} "
                        "(retry_count={}, max_attempts={})",
                        source_id,
                        user_id,
                        retry_count,
                        max_attempts,
                    )
                    jm.complete_job(
                        jid,
                        result={
                            "processed": 0,
                            "total": 0,
                            "skipped": "retry_budget_exhausted",
                            "retry_backoff_count": retry_count,
                            "retry_backoff_seconds": backoff_seconds,
                            "last_error": last_error,
                        },
                        worker_id=worker_id,
                        lease_id=lease_id,
                        completion_token=lease_id,
                    )
                    _email_sync_metrics_increment(
                        "email_sync_runs_total",
                        labels=_email_sync_metric_labels(
                            provider="gmail",
                            status="skipped",
                        ),
                    )
                    _close_connector_media_db(mdb)
                    return

                last_run_at = _parse_iso_utc((state or {}).get("last_run_at"))
                if last_run_at is not None:
                    retry_after_ts = last_run_at.timestamp() + float(backoff_seconds)
                    now_ts = datetime.now(timezone.utc).timestamp()
                    if now_ts < retry_after_ts:
                        retry_after_iso = datetime.fromtimestamp(
                            retry_after_ts,
                            tz=timezone.utc,
                        ).isoformat()
                        logger.info(
                            "gmail sync backoff active for source_id={} user_id={} "
                            "(retry_count={}, retry_after={})",
                            source_id,
                            user_id,
                            retry_count,
                            retry_after_iso,
                        )
                        jm.complete_job(
                            jid,
                            result={
                                "processed": 0,
                                "total": 0,
                                "skipped": "backoff_active",
                                "retry_backoff_count": retry_count,
                                "retry_backoff_seconds": backoff_seconds,
                                "retry_after": retry_after_iso,
                                "last_error": last_error,
                            },
                            worker_id=worker_id,
                            lease_id=lease_id,
                            completion_token=lease_id,
                        )
                        _email_sync_metrics_increment(
                            "email_sync_runs_total",
                            labels=_email_sync_metric_labels(
                                provider="gmail",
                                status="skipped",
                            ),
                        )
                        _close_connector_media_db(mdb)
                        return

            gmail_previous_cursor = _normalize_gmail_history_cursor((state or {}).get("cursor"))
            gmail_previous_success_at = _parse_iso_utc((state or {}).get("last_success_at"))
            mdb.mark_email_sync_run_started(
                provider="gmail",
                source_key=gmail_sync_source_key,
                tenant_id=gmail_sync_tenant_id,
                cursor=gmail_previous_cursor,
            )
            gmail_sync_state_started = True
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"gmail sync-state start failed for source {source_id}: {e}")
    # Determine listing function
    async def _enumerate_items() -> list[dict[str, Any]]:
        nonlocal gmail_cursor_candidate
        nonlocal gmail_cursor_recovery_active
        items: list[dict[str, Any]] = []
        page_size = 100
        recursive = bool(options.get("recursive", True))
        if provider == "drive":
            # BFS traversal when recursive; otherwise single level
            queue: list[str] = [remote_id or "root"]
            visited: set[str] = set()
            while queue:
                parent = queue.pop(0)
                if parent in visited:
                    continue
                visited.add(parent)
                cursor = None
                while True:
                    batch, cursor = await _attempt_with_refresh(conn.list_files, acct, parent, page_size=page_size, cursor=cursor)
                    for f in (batch or []):
                        items.append(f)
                        if recursive and f.get("is_folder"):
                            fid2 = str(f.get("id")) if f.get("id") else None
                            if fid2:
                                queue.append(fid2)
                    if not cursor:
                        break
        elif provider == "notion":
            typ = str(src.get("type"))
            if typ == "page":
                items = [{"id": remote_id, "name": src.get("path") or remote_id, "mimeType": "text/markdown", "last_edited_time": None, "size": None}]
            elif typ == "database":
                cursor = None
                while True:
                    batch, cursor = await _attempt_with_refresh(conn.list_sources, acct, parent_remote_id=remote_id, page_size=page_size, cursor=cursor)
                    items.extend(batch or [])
                    if not cursor:
                        break
            else:
                cursor = None
                while True:
                    batch, cursor = await _attempt_with_refresh(conn.list_sources, acct, parent_remote_id=None, page_size=page_size, cursor=cursor)
                    items.extend(batch or [])
                    if not cursor:
                        break
        elif provider == "gmail":
            cursor = None
            query = str(options.get("query") or "").strip() or None
            max_messages = _safe_int_or_zero(options.get("max_messages"))
            label_id = None
            remote_hint = (remote_id or "").strip()
            if remote_hint and remote_hint.lower() not in {"root", "all"}:
                label_id = remote_hint
            can_use_history_api = bool(
                gmail_previous_cursor and hasattr(conn, "list_history")
            )
            if can_use_history_api:
                seen_ids: set[str] = set()
                while True:
                    try:
                        batch, cursor, history_cursor = await _attempt_with_refresh(
                            conn.list_history,
                            acct,
                            start_history_id=gmail_previous_cursor,
                            label_id=label_id,
                            page_size=page_size,
                            cursor=cursor,
                        )
                    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
                        if not _is_gmail_history_cursor_invalid(exc):
                            raise
                        gmail_cursor_recovery_active = True
                        logger.warning(
                            "gmail history cursor invalid for source_id={} user_id={} "
                            "(cursor={}): {}",
                            source_id,
                            user_id,
                            gmail_previous_cursor,
                            exc,
                        )
                        break
                    gmail_cursor_candidate = _merge_gmail_history_cursor(
                        gmail_cursor_candidate,
                        history_cursor,
                    )
                    for row in (batch or []):
                        if not isinstance(row, dict):
                            continue
                        msg_id = str(row.get("id") or "").strip()
                        if not msg_id or msg_id in seen_ids:
                            continue
                        seen_ids.add(msg_id)
                        items.append(row)
                    if not cursor:
                        break
            if gmail_cursor_recovery_active or not can_use_history_api:
                replay_query = query
                replay_limit = max_messages
                if gmail_cursor_recovery_active:
                    replay_query = _append_query_term(
                        query,
                        f"newer_than:{gmail_cursor_recovery_window_days}d",
                    )
                    replay_limit = gmail_cursor_recovery_max_messages
                    if max_messages > 0:
                        replay_limit = min(replay_limit, max_messages)
                seen_ids: set[str] = set()
                for row in items:
                    if not isinstance(row, dict):
                        continue
                    msg_id = str(row.get("id") or "").strip()
                    if msg_id:
                        seen_ids.add(msg_id)
                cursor = None
                while True:
                    batch, cursor = await _attempt_with_refresh(
                        conn.list_messages,
                        acct,
                        label_id=label_id,
                        page_size=page_size,
                        cursor=cursor,
                        query=replay_query,
                    )
                    for row in (batch or []):
                        if not isinstance(row, dict):
                            continue
                        msg_id = str(row.get("id") or "").strip()
                        if not msg_id or msg_id in seen_ids:
                            continue
                        seen_ids.add(msg_id)
                        items.append(row)
                    if replay_limit > 0 and len(items) >= replay_limit:
                        items = items[:replay_limit]
                        break
                    if not cursor:
                        break
        return items

    processed = 0
    total = 1
    failed = 0
    degraded = 0
    bootstrap_sync_storage_enabled = False
    # Policy helpers
    allowed_export_formats = [str(f).lower() for f in (policy.get("allowed_export_formats") or [])]
    allowed_export_set = set(allowed_export_formats)
    allowed_file_types = [str(t).lower() for t in (policy.get("allowed_file_types") or [])]
    max_file_size_mb = int(policy.get("max_file_size_mb") or 0)
    max_bytes = max_file_size_mb * 1024 * 1024 if max_file_size_mb > 0 else None
    include_types = [str(x).lower() for x in (options.get("include_types") or [])]
    exclude_patterns = [str(x) for x in (options.get("exclude_patterns") or [])]
    export_overrides = {str(k): str(v).lower() for k, v in (options.get("export_format_overrides") or {}).items()}
    try:
        from tldw_Server_API.app.core.config import settings as app_settings

        email_native_persist_enabled = bool(
            app_settings.get("EMAIL_NATIVE_PERSIST_ENABLED", True)
        )
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
        email_native_persist_enabled = True

    async def _process_subscription_renewal() -> bool:
        if job_type != "subscription_renewal":
            return False
        if provider not in FILE_SYNC_PROVIDERS or not hasattr(conn, "renew_webhook"):
            jm.complete_job(
                jid,
                result={"processed": 0, "total": 0, "skipped": "unsupported_provider"},
                worker_id=worker_id,
                lease_id=lease_id,
                completion_token=lease_id,
            )
            return True

        from tldw_Server_API.app.core.External_Sources.sync_adapter import FileSyncWebhookSubscription

        async with pool.transaction() as db:
            sync_state = await get_source_sync_state(db, source_id=source_id) or {}
        subscription_id = str(sync_state.get("webhook_subscription_id") or "").strip() or None
        expires_at = str(sync_state.get("webhook_expires_at") or "").strip() or None
        webhook_metadata = dict(sync_state.get("webhook_metadata") or {})
        if not subscription_id:
            jm.complete_job(
                jid,
                result={"processed": 0, "total": 0, "skipped": "missing_subscription"},
                worker_id=worker_id,
                lease_id=lease_id,
                completion_token=lease_id,
            )
            return True

        renewed = await _attempt_with_refresh(
            conn.renew_webhook,
            acct,
            subscription=FileSyncWebhookSubscription(
                subscription_id=subscription_id,
                expires_at=expires_at,
                metadata=webhook_metadata,
            ),
        )
        if not renewed or not renewed.subscription_id:
            raise ValueError(f"Webhook renewal failed for provider={provider} source_id={source_id}")  # noqa: TRY003

        async with pool.transaction() as db:
            await upsert_source_sync_state(
                db,
                source_id=source_id,
                webhook_status="active",
                webhook_subscription_id=renewed.subscription_id,
                webhook_expires_at=renewed.expires_at,
                webhook_metadata=renewed.metadata or webhook_metadata,
                last_error=None,
            )
        jm.complete_job(
            jid,
            result={
                "processed": 1,
                "total": 1,
                "subscription_id": renewed.subscription_id,
                "webhook_expires_at": renewed.expires_at,
            },
            worker_id=worker_id,
            lease_id=lease_id,
            completion_token=lease_id,
        )
        return True

    async def _process_file_sync_changes() -> bool:
        nonlocal processed
        nonlocal total
        nonlocal failed
        nonlocal degraded

        if provider not in FILE_SYNC_PROVIDERS:
            return False

        try:
            async with pool.transaction() as db:
                sync_state = await get_source_sync_state(db, source_id=source_id)
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
            return False
        file_cursor = str((sync_state or {}).get("cursor") or "").strip() or None
        if not file_cursor:
            return False
        if bool((sync_state or {}).get("needs_full_rescan")):
            return False

        from tldw_Server_API.app.core.External_Sources import sync_coordinator

        cursor_kind = str((sync_state or {}).get("cursor_kind") or "").strip() or _file_sync_cursor_kind(provider)
        async with pool.transaction() as db:
            await upsert_source_sync_state(
                db,
                source_id=source_id,
                cursor=file_cursor,
                cursor_kind=cursor_kind,
                last_sync_started_at=_utc_now_db_text(),
                last_error=None,
            )

        try:
            changes, next_cursor, cursor_hint = await _attempt_with_refresh(
                conn.list_changes,
                acct,
                cursor=file_cursor,
                page_size=100,
            )
            total = max(1, len(changes))
            for idx, change in enumerate(changes):
                try:
                    pct = int((idx / total) * 100)
                    jm.renew_job_lease(
                        jid,
                        seconds=120,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        progress_percent=pct,
                    )
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
                    pass

                async with pool.transaction() as db:
                    binding = await get_external_item_binding(
                        db,
                        source_id=source_id,
                        provider=provider,
                        external_id=change.remote_id,
                    )

                if not binding and change.event_type != "created":
                    logger.warning(
                        "Skipping file sync change without binding for provider={} source_id={} remote_id={}",
                        provider,
                        source_id,
                        change.remote_id,
                    )
                    continue

                try:
                    content_payload = None
                    if change.event_type in {"created", "content_updated"}:
                        metadata = dict(change.metadata or {})
                        export_mime = None
                        mime = str(metadata.get("mime_type") or metadata.get("mimeType") or "").strip() or None
                        name = str(change.remote_name or change.remote_id)
                        skip_reason = _file_sync_skip_reason(
                            name=name,
                            mime=mime,
                            size=metadata.get("size"),
                            is_folder=bool(metadata.get("is_folder")),
                            include_types=include_types,
                            exclude_patterns=exclude_patterns,
                            allowed_file_types=allowed_file_types,
                            max_bytes=max_bytes,
                        )
                        if skip_reason:
                            if binding:
                                raise ValueError(f"File sync change blocked by policy: {skip_reason}")  # noqa: TRY003
                            logger.info(
                                "Skipping file sync change for provider={} source_id={} remote_id={} reason={}",
                                provider,
                                source_id,
                                change.remote_id,
                                skip_reason,
                            )
                            continue
                        if provider == "drive":
                            export_mime = _determine_drive_export_mime(
                                mime=mime,
                                allowed_export_formats=allowed_export_formats,
                                allowed_export_set=allowed_export_set,
                                export_overrides=export_overrides,
                            )
                            if export_mime:
                                metadata["export_mime"] = export_mime
                        raw = await _attempt_with_refresh(
                            conn.download_or_export,
                            acct,
                            change.remote_id,
                            metadata=metadata,
                        )
                        content_text = await _convert_document_bytes_to_text(
                            raw=raw,
                            name=name,
                            effective_mime=(export_mime or mime or "").lower(),
                        )
                        content_payload = sync_coordinator.FileSyncContentPayload(
                            text=content_text,
                            safe_metadata={"export_mime": export_mime} if export_mime else None,
                        )

                    async with pool.transaction() as db:
                        reconcile_result = await sync_coordinator.reconcile_file_change(
                            db,
                            mdb,
                            source_id=source_id,
                            provider=provider,
                            change=change,
                            content=content_payload,
                            job_id=str(jid),
                        )
                    if reconcile_result.action:
                        processed += 1
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
                    failed += 1
                    logger.warning(
                        "File sync item reconcile failed for provider={} source_id={} remote_id={}: {}",
                        provider,
                        source_id,
                        change.remote_id,
                        exc,
                    )
                    if await _mark_file_binding_degraded(
                        binding=binding,
                        change=change,
                        error=exc,
                    ):
                        degraded += 1
                    continue

            final_cursor = str(cursor_hint or next_cursor or file_cursor).strip() or file_cursor
            async with pool.transaction() as db:
                await upsert_source_sync_state(
                    db,
                    source_id=source_id,
                    cursor=final_cursor,
                    cursor_kind=cursor_kind,
                    last_sync_succeeded_at=_utc_now_db_text(),
                    last_error=None,
                )
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
            async with pool.transaction() as db:
                await upsert_source_sync_state(
                    db,
                    source_id=source_id,
                    cursor=file_cursor,
                    cursor_kind=cursor_kind,
                    last_sync_failed_at=_utc_now_db_text(),
                    last_error=str(exc),
                )
            raise
        return True

    async def _resolve_post_bootstrap_cursor() -> tuple[str | None, str | None]:
        cursor_kind = _file_sync_cursor_kind(provider)
        try:
            async with pool.transaction() as db:
                sync_state = await get_source_sync_state(db, source_id=source_id) or {}
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
            sync_state = {}

        existing_cursor = str(sync_state.get("cursor") or "").strip() or None
        existing_kind = str(sync_state.get("cursor_kind") or "").strip() or cursor_kind
        if existing_cursor:
            return existing_cursor, existing_kind

        try:
            if provider == "drive" and hasattr(conn, "get_start_page_token"):
                drive_cursor = await _attempt_with_refresh(conn.get_start_page_token, acct)
                resolved = str(drive_cursor or "").strip() or None
                return resolved, existing_kind
            if hasattr(conn, "list_changes"):
                _changes, next_cursor, cursor_hint = await _attempt_with_refresh(
                    conn.list_changes,
                    acct,
                    cursor=None,
                    page_size=1,
                )
                resolved = str(cursor_hint or next_cursor or "").strip() or None
                return resolved, existing_kind
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Failed to resolve post-bootstrap cursor for provider={} source_id={}: {}",
                provider,
                source_id,
                exc,
            )
        return None, existing_kind

    def _supports_connector_sync_storage(db: Any) -> bool:
        return callable(getattr(db, "execute", None))

    async def _mark_file_binding_degraded(*, binding: dict[str, Any] | None, change, error: Exception) -> bool:
        if not binding or binding.get("id") is None:
            return False

        change_metadata = dict(change.metadata or {})
        provider_metadata = dict(binding.get("provider_metadata") or {})
        provider_metadata.update(
            {key: value for key, value in change_metadata.items() if value is not None}
        )
        provider_metadata.update(
            {
                "failed_remote_revision": change.remote_revision,
                "failed_remote_hash": change.remote_hash,
                "failed_event_type": change.event_type,
            }
        )

        async with pool.transaction() as db:
            updated = await upsert_external_item_binding(
                db,
                source_id=source_id,
                provider=provider,
                external_id=change.remote_id,
                name=change.remote_name or binding.get("name"),
                mime=change_metadata.get("mime_type") or binding.get("mime"),
                size=change_metadata.get("size") or binding.get("size"),
                version=binding.get("version"),
                modified_at=change_metadata.get("modified_at") or binding.get("modified_at"),
                content_hash=binding.get("hash"),
                media_id=binding.get("media_id"),
                sync_status="degraded",
                current_version_number=binding.get("current_version_number"),
                remote_parent_id=change.remote_parent_id or binding.get("remote_parent_id"),
                remote_path=change.remote_path or binding.get("remote_path"),
                last_seen_at=_utc_now_db_text(),
                last_metadata_sync_at=_utc_now_db_text(),
                provider_metadata=provider_metadata,
            )
            await record_item_event(
                db,
                external_item_id=int(updated["id"]),
                event_type="ingest_failed",
                job_id=str(jid),
                payload={
                    "error": str(error),
                    "change_event_type": change.event_type,
                    "remote_revision": change.remote_revision,
                    "remote_hash": change.remote_hash,
                },
            )
        return True

    try:
        if await _process_subscription_renewal():
            _close_connector_media_db(mdb)
            return
        if not await _process_file_sync_changes():
            if provider in REFERENCE_MANAGER_PROVIDERS:
                from tldw_Server_API.app.core.External_Sources.reference_manager_import import (
                    sync_reference_manager_source,
                )

                async with pool.transaction() as db:
                    reference_sync_state = await get_source_sync_state(
                        db,
                        source_id=source_id,
                    ) or {}
                    await upsert_source_sync_state(
                        db,
                        source_id=source_id,
                        last_sync_started_at=_utc_now_db_text(),
                        last_error=None,
                    )
                heartbeat_stop = asyncio.Event()
                heartbeat_task = asyncio.create_task(
                    _renew_reference_manager_lease_until_stopped(heartbeat_stop)
                )
                try:
                    reference_result = await sync_reference_manager_source(
                        connectors_pool=pool,
                        connector=conn,
                        account=acct,
                        source=src,
                        sync_state=reference_sync_state,
                        media_db=mdb,
                        job_id=str(jid),
                        convert_bytes_to_text=_convert_document_bytes_to_text,
                    )
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS as exc:
                    async with pool.transaction() as db:
                        await upsert_source_sync_state(
                            db,
                            source_id=source_id,
                            last_sync_failed_at=_utc_now_db_text(),
                            last_error=str(exc),
                        )
                    raise
                finally:
                    heartbeat_stop.set()
                    await heartbeat_task

                processed = int(reference_result.get("processed") or 0)
                total = int(reference_result.get("total") or 0)
                failed = int(reference_result.get("failed") or 0)
                if "cursor" in reference_result:
                    cursor_value = reference_result.get("cursor")
                else:
                    cursor_value = reference_sync_state.get("cursor")
                final_cursor = str(cursor_value).strip() or None if cursor_value is not None else None
                result_payload = {
                    "processed": processed,
                    "total": total,
                    "failed": failed,
                    "imported": int(reference_result.get("imported") or 0),
                    "duplicates": int(reference_result.get("duplicates") or 0),
                    "metadata_only": int(reference_result.get("metadata_only") or 0),
                }
                async with pool.transaction() as db:
                    await upsert_source_sync_state(
                        db,
                        source_id=source_id,
                        cursor=final_cursor,
                        last_sync_succeeded_at=_utc_now_db_text(),
                        last_error=None,
                    )
                jm.complete_job(
                    jid,
                    result=result_payload,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
                _close_connector_media_db(mdb)
                return
            bootstrap_file_sync_scan = provider in FILE_SYNC_PROVIDERS
            if bootstrap_file_sync_scan:
                async with pool.transaction() as db:
                    bootstrap_sync_storage_enabled = _supports_connector_sync_storage(db)
                    if bootstrap_sync_storage_enabled:
                        await upsert_source_sync_state(
                            db,
                            source_id=source_id,
                            cursor_kind=_file_sync_cursor_kind(provider),
                            last_sync_started_at=_utc_now_db_text(),
                            last_error=None,
                        )
            if bootstrap_file_sync_scan and not bootstrap_sync_storage_enabled:
                logger.debug(
                    "Connector sync storage unavailable during bootstrap for provider={} source_id={}; falling back to legacy import path",
                    provider,
                    source_id,
                )
            items = await _enumerate_items()
            total = max(1, len(items))

            for idx, it in enumerate(items):
                try:
                    # Renew lease with progress
                    pct = int((idx / total) * 100)
                    jm.renew_job_lease(
                        jid,
                        seconds=120,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        progress_percent=pct,
                    )
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS:
                    pass
                if provider == "gmail":
                    fid = str(it.get("id") or "").strip()
                    if not fid:
                        continue
                    history_id_hint = _normalize_gmail_history_cursor(it.get("historyId"))
                    gmail_cursor_candidate = _merge_gmail_history_cursor(
                        gmail_cursor_candidate,
                        history_id_hint,
                    )
                    history_message_added = bool(it.get("message_added"))
                    history_message_deleted = bool(it.get("message_deleted"))

                    labels_added_raw = it.get("labels_added")
                    labels_removed_raw = it.get("labels_removed")
                    labels_added: list[str] = []
                    labels_removed: list[str] = []
                    if isinstance(labels_added_raw, list):
                        seen_added: set[str] = set()
                        for label in labels_added_raw:
                            label_text = str(label or "").strip()
                            key = label_text.lower()
                            if not label_text or key in seen_added:
                                continue
                            seen_added.add(key)
                            labels_added.append(label_text)
                    if isinstance(labels_removed_raw, list):
                        seen_removed: set[str] = set()
                        for label in labels_removed_raw:
                            label_text = str(label or "").strip()
                            key = label_text.lower()
                            if not label_text or key in seen_removed:
                                continue
                            seen_removed.add(key)
                            labels_removed.append(label_text)

                    if (
                        history_message_deleted
                        and not history_message_added
                        and hasattr(mdb, "reconcile_email_message_state")
                    ):
                        try:
                            state_res = mdb.reconcile_email_message_state(
                                provider="gmail",
                                source_key=str(source_id),
                                source_message_id=fid,
                                tenant_id=str(user_id),
                                deleted=True,
                            )
                            if bool((state_res or {}).get("applied")):
                                processed += 1
                            continue
                        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                            logger.warning(f"gmail state reconcile failed for {fid}: {e}")
                            gmail_sync_errors.append(f"state_reconcile:{fid}")
                            continue

                    if (
                        not history_message_added
                        and not history_message_deleted
                        and (labels_added or labels_removed)
                        and hasattr(mdb, "apply_email_label_delta")
                    ):
                        try:
                            delta_res = mdb.apply_email_label_delta(
                                provider="gmail",
                                source_key=str(source_id),
                                source_message_id=fid,
                                labels_added=labels_added,
                                labels_removed=labels_removed,
                                tenant_id=str(user_id),
                            )
                            if bool((delta_res or {}).get("applied")):
                                processed += 1
                                continue
                            # If we cannot resolve the local message yet, fall back to full fetch.
                            delta_reason = str((delta_res or {}).get("reason") or "").strip().lower()
                            if delta_reason not in {"message_not_found", "source_not_found"}:
                                continue
                        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                            logger.warning(f"gmail label delta apply failed for {fid}: {e}")
                            gmail_sync_errors.append(f"label_delta:{fid}")
                            continue

                    try:
                        message = await _attempt_with_refresh(conn.get_message, acct, message_id=fid, format="full")
                    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"gmail get_message failed for {fid}: {e}")
                        gmail_sync_errors.append(f"get_message:{fid}")
                        continue

                    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
                    headers_map = _gmail_headers_map(payload)
                    subject = str(headers_map.get("subject") or "").strip() or f"Gmail message {fid}"
                    from_text = str(headers_map.get("from") or "").strip() or None
                    to_text = str(headers_map.get("to") or "").strip() or None
                    cc_text = str(headers_map.get("cc") or "").strip() or None
                    bcc_text = str(headers_map.get("bcc") or "").strip() or None
                    message_id_header = str(headers_map.get("message-id") or "").strip() or None
                    date_header = str(headers_map.get("date") or "").strip() or None

                    body_text = _collect_gmail_body_text(payload)
                    if not body_text:
                        body_text = str(message.get("snippet") or "").strip()

                    attachments = _collect_gmail_attachments(payload)
                    label_ids: list[str] = []
                    seen_labels: set[str] = set()
                    for label in (message.get("labelIds") or []):
                        label_text = str(label).strip()
                        if not label_text:
                            continue
                        key = label_text.lower()
                        if key in seen_labels:
                            continue
                        seen_labels.add(key)
                        label_ids.append(label_text)
                    internal_date = _gmail_internal_date_iso(message)
                    internal_date_dt = _parse_iso_utc(internal_date)
                    if internal_date_dt is not None:
                        if (
                            gmail_latest_message_internal_at is None
                            or internal_date_dt > gmail_latest_message_internal_at
                        ):
                            gmail_latest_message_internal_at = internal_date_dt
                    history_id = str(message.get("historyId") or "").strip() or None
                    gmail_cursor_candidate = _merge_gmail_history_cursor(
                        gmail_cursor_candidate,
                        history_id,
                    )
                    content_hash = hashlib.sha256(body_text.encode()).hexdigest() if body_text else None

                    async with pool.transaction() as db:
                        should = await should_ingest_item(
                            db,
                            source_id=source_id,
                            provider=provider,
                            external_id=fid,
                            version=history_id,
                            modified_at=internal_date,
                            content_hash=content_hash,
                        )
                    if not should:
                        continue

                    metadata_map: dict[str, Any] = {
                        "source": "gmail_connector",
                        "labels": label_ids,
                        "email": {
                            "source_message_id": fid,
                            "message_id": message_id_header,
                            "subject": subject,
                            "date": date_header,
                            "internal_date": internal_date,
                            "from": from_text,
                            "to": to_text,
                            "cc": cc_text,
                            "bcc": bcc_text,
                            "labels": label_ids,
                            "headers_map": headers_map,
                            "attachments": attachments,
                        },
                    }
                    safe_metadata_json = json.dumps(metadata_map, ensure_ascii=False)

                    url = f"gmail://{source_id}/{fid}"
                    try:
                        media_id, _, _ = _ingest_connector_media(
                            media_db=mdb,
                            url=url,
                            title=subject,
                            media_type="email",
                            content=body_text or f"[empty content for gmail:{fid}]",
                            keywords=[],
                            safe_metadata=safe_metadata_json,
                            author=from_text,
                            ingestion_date=internal_date,
                            overwrite=False,
                        )
                    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"add_media_with_keywords failed for gmail:{fid}: {e}")
                        gmail_sync_errors.append(f"ingest:{fid}")
                        continue

                    if media_id and email_native_persist_enabled:
                        try:
                            mdb.upsert_email_message_graph(
                                media_id=int(media_id),
                                metadata=metadata_map,
                                body_text=body_text,
                                tenant_id=str(user_id),
                                provider="gmail",
                                source_key=str(source_id),
                                source_message_id=fid,
                                labels=label_ids,
                            )
                        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                            logger.warning(f"email native upsert failed for gmail:{fid}: {e}")

                    async with pool.transaction() as db:
                        await record_ingested_item(
                            db,
                            source_id=source_id,
                            provider=provider,
                            external_id=fid,
                            name=subject,
                            mime="message/rfc822",
                            size=None,
                            version=history_id,
                            modified_at=internal_date,
                            content_hash=content_hash,
                        )
                    processed += 1
                    continue
                fid = str(it.get("id"))
                name = str(it.get("name") or fid)
                modified_at = it.get("modifiedTime") or it.get("last_edited_time")
                size = it.get("size")
                mime = it.get("mimeType") or ("text/markdown" if provider == "notion" else None)
                version = it.get("md5Checksum") or None

                skip_reason = _file_sync_skip_reason(
                    name=name,
                    mime=mime,
                    size=size,
                    is_folder=bool(it.get("is_folder"))
                    or str(it.get("mimeType") or "").startswith("application/vnd.google-apps.folder"),
                    include_types=include_types,
                    exclude_patterns=exclude_patterns,
                    allowed_file_types=allowed_file_types,
                    max_bytes=max_bytes,
                )
                if skip_reason:
                    continue

                # Determine desired export for Drive Google types according to policy/overrides
                export_mime = None
                if provider == "drive":
                    export_mime = _determine_drive_export_mime(
                        mime=mime,
                        allowed_export_formats=allowed_export_formats,
                        allowed_export_set=allowed_export_set,
                        export_overrides=export_overrides,
                    )

                # Download/export
                try:
                    if provider == "drive":
                        raw = await _attempt_with_refresh(conn.download_file, acct, fid, mime_type=mime, export_mime=export_mime)
                    else:
                        raw = await _attempt_with_refresh(conn.download_file, acct, fid) if provider == "notion" else await _attempt_with_refresh(conn.download_file, acct, fid, mime_type=mime)
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"download failed for {provider}:{fid}: {e}")
                    if provider in FILE_SYNC_PROVIDERS and bootstrap_sync_storage_enabled:
                        failed += 1
                        continue
                    raw = b""

                effective_mime = (export_mime or mime or "").lower()
                content_text = await _convert_document_bytes_to_text(
                    raw=raw,
                    name=name,
                    effective_mime=effective_mime,
                )
                # Hash for dedup
                content_hash = hashlib.sha256(content_text.encode()).hexdigest() if content_text else None
                # Dedup check
                async with pool.transaction() as db:
                    should = await should_ingest_item(
                        db,
                        source_id=source_id,
                        provider=provider,
                        external_id=fid,
                        version=version,
                        modified_at=modified_at,
                        content_hash=content_hash,
                    )
                if not should:
                    continue
                if provider in FILE_SYNC_PROVIDERS and bootstrap_sync_storage_enabled:
                    from tldw_Server_API.app.core.External_Sources import sync_coordinator

                    change = sync_coordinator.FileSyncChange(
                        event_type="created",
                        remote_id=fid,
                        remote_name=name,
                        remote_revision=version,
                        remote_hash=content_hash,
                        metadata={
                            "mime_type": mime,
                            "size": size,
                            "modified_at": modified_at,
                        },
                    )
                    existing_binding = None
                    try:
                        async with pool.transaction() as db:
                            existing_binding = await get_external_item_binding(
                                db,
                                source_id=source_id,
                                provider=provider,
                                external_id=fid,
                            )
                        if existing_binding:
                            change = sync_coordinator.FileSyncChange(
                                event_type="content_updated",
                                remote_id=fid,
                                remote_name=name,
                                remote_revision=version,
                                remote_hash=content_hash,
                                metadata={
                                    "mime_type": mime,
                                    "size": size,
                                    "modified_at": modified_at,
                                },
                            )

                        reconcile_content = sync_coordinator.FileSyncContentPayload(
                            text=content_text,
                            safe_metadata={"export_mime": export_mime} if export_mime else None,
                        )
                        async with pool.transaction() as db:
                            reconcile_result = await sync_coordinator.reconcile_file_change(
                                db,
                                mdb,
                                source_id=source_id,
                                provider=provider,
                                change=change,
                                content=reconcile_content,
                                job_id=str(jid),
                            )
                        if reconcile_result.action:
                            processed += 1
                    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                        failed += 1
                        logger.warning(
                            "Bootstrap file sync reconcile failed for provider={} source_id={} remote_id={}: {}",
                            provider,
                            source_id,
                            fid,
                            e,
                        )
                        if await _mark_file_binding_degraded(
                            binding=existing_binding,
                            change=change,
                            error=e,
                        ):
                            degraded += 1
                    continue

                # Ingest minimal record
                title = name
                url = f"{provider}://{fid}"
                ingested = False
                try:
                    _mid, _m_uuid, _msg = _ingest_connector_media(
                        media_db=mdb,
                        url=url,
                        title=title,
                        media_type="document",
                        content=content_text or f"[empty content for {provider}:{fid}]",
                        keywords=[],
                        overwrite=False,
                    )
                    processed += 1
                    ingested = True
                except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"add_media_with_keywords failed: {e}")
                if ingested:
                    # Record ingestion cache
                    async with pool.transaction() as db:
                        await record_ingested_item(
                            db,
                            source_id=source_id,
                            provider=provider,
                            external_id=fid,
                            name=name,
                            mime=mime,
                            size=size,
                            version=version,
                            modified_at=modified_at,
                            content_hash=content_hash,
                        )

            if bootstrap_file_sync_scan and bootstrap_sync_storage_enabled:
                resolved_cursor, resolved_cursor_kind = await _resolve_post_bootstrap_cursor()
                async with pool.transaction() as db:
                    await upsert_source_sync_state(
                        db,
                        source_id=source_id,
                        cursor=resolved_cursor,
                        cursor_kind=resolved_cursor_kind,
                        last_bootstrap_at=_utc_now_db_text(),
                        last_sync_succeeded_at=_utc_now_db_text(),
                        last_error=None,
                    )
    except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
        if gmail_sync_state_supported and gmail_sync_state_started:
            with contextlib.suppress(_CONNECTOR_NONCRITICAL_EXCEPTIONS):
                mdb.mark_email_sync_run_failed(
                    provider="gmail",
                    source_key=gmail_sync_source_key,
                    tenant_id=gmail_sync_tenant_id,
                    error_state=f"run_exception:{type(e).__name__}",
                )
        if provider == "gmail":
            _email_sync_metrics_increment(
                "email_sync_failures_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    reason="run_exception",
                ),
            )
            _email_sync_metrics_increment(
                "email_sync_runs_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    status="failed",
                ),
            )
        _close_connector_media_db(mdb)
        raise

    cursor_recovery_full_backfill_required = bool(
        provider == "gmail"
        and gmail_cursor_recovery_active
        and not bool(_normalize_gmail_history_cursor(gmail_cursor_candidate))
    )

    if gmail_sync_state_supported and gmail_sync_state_started:
        try:
            if cursor_recovery_full_backfill_required:
                mdb.mark_email_sync_run_failed(
                    provider="gmail",
                    source_key=gmail_sync_source_key,
                    tenant_id=gmail_sync_tenant_id,
                    error_state=(
                        "cursor_invalid_full_backfill_required:"
                        f"window={gmail_cursor_recovery_window_days}d"
                    ),
                )
                gmail_run_status = "failed"
                gmail_failure_reason = "cursor_invalid_full_backfill_required"
            elif gmail_sync_errors:
                summary = ", ".join(gmail_sync_errors[:5])
                if len(gmail_sync_errors) > 5:
                    summary = f"{summary}, +{len(gmail_sync_errors) - 5} more"
                mdb.mark_email_sync_run_failed(
                    provider="gmail",
                    source_key=gmail_sync_source_key,
                    tenant_id=gmail_sync_tenant_id,
                    error_state=f"partial_failure:{summary}",
                )
                gmail_run_status = "failed"
                gmail_failure_reason = "partial_failure"
            else:
                final_cursor = _merge_gmail_history_cursor(
                    gmail_previous_cursor,
                    gmail_cursor_candidate,
                )
                mdb.mark_email_sync_run_succeeded(
                    provider="gmail",
                    source_key=gmail_sync_source_key,
                    tenant_id=gmail_sync_tenant_id,
                    cursor=final_cursor,
                )
                gmail_run_status = "success"
                gmail_failure_reason = None
        except _CONNECTOR_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"gmail sync-state finalize failed for source {source_id}: {e}")

    if provider == "gmail" and gmail_run_status is None:
        if cursor_recovery_full_backfill_required:
            gmail_run_status = "failed"
            gmail_failure_reason = "cursor_invalid_full_backfill_required"
        elif gmail_sync_errors:
            gmail_run_status = "failed"
            gmail_failure_reason = "partial_failure"
        else:
            gmail_run_status = "success"
            gmail_failure_reason = None

    if provider == "gmail":
        if gmail_run_status == "failed":
            _email_sync_metrics_increment(
                "email_sync_failures_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    reason=gmail_failure_reason or "sync_failed",
                ),
            )
            _email_sync_metrics_increment(
                "email_sync_runs_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    status="failed",
                ),
            )
        elif gmail_run_status == "success":
            _email_sync_metrics_increment(
                "email_sync_runs_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    status="success",
                ),
            )
            lag_reference = gmail_latest_message_internal_at or gmail_previous_success_at
            lag_seconds = 0.0
            if lag_reference is not None:
                lag_seconds = max(
                    0.0,
                    (datetime.now(timezone.utc) - lag_reference).total_seconds(),
                )
            _email_sync_metrics_observe(
                "email_sync_lag_seconds",
                lag_seconds,
                labels=_email_sync_metric_labels(provider="gmail"),
            )
        if gmail_cursor_recovery_active:
            _email_sync_metrics_increment(
                "email_sync_recovery_events_total",
                labels=_email_sync_metric_labels(
                    provider="gmail",
                    outcome=(
                        "full_backfill_required"
                        if cursor_recovery_full_backfill_required
                        else "bounded_replay"
                    ),
                ),
            )

    result_payload: dict[str, Any] = {
        "processed": processed,
        "total": total,
        "failed": failed,
        "degraded": degraded,
    }
    if provider == "gmail" and gmail_cursor_recovery_active:
        result_payload["cursor_recovery"] = (
            "full_backfill_required"
            if cursor_recovery_full_backfill_required
            else "bounded_replay"
        )
        result_payload["cursor_recovery_window_days"] = gmail_cursor_recovery_window_days

    # Complete job
    jm.complete_job(
        jid,
        result=result_payload,
        worker_id=worker_id,
        lease_id=lease_id,
        completion_token=lease_id,
    )
    _close_connector_media_db(mdb)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_connectors_worker())
