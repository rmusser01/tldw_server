from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Callable
from urllib.parse import urlparse

from fastapi import BackgroundTasks, HTTPException, Request, UploadFile, status
from loguru import logger
from starlette.responses import JSONResponse

from tldw_Server_API.app.core.Claims_Extraction.claims_utils import (
    extract_claims_if_requested,
    persist_claims_if_applicable,
)
from tldw_Server_API.app.core.config import loaded_config_data, settings
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
    normalize_media_dedupe_url,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    get_media_repository,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    upsert_transcript,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    async_resolve_chunking_options_and_plan,
    prepare_chunking_options_dict,
    prepare_common_options,
    resolve_chunking_options_and_plan,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
    open_safe_local_path_async,
    resolve_safe_local_path,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    DEFAULT_MEDIA_TYPE_CONFIG,
)
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Metrics.stt_metrics import emit_stt_run_write_total
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
)

try:
    from tldw_Server_API.app.core.Resource_Governance import RGRequest
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional in minimal profiles
    RGRequest = None  # type: ignore[assignment]

try:  # Align HTTP 413 compatibility across FastAPI/Starlette versions
    HTTP_413_TOO_LARGE = status.HTTP_413_CONTENT_TOO_LARGE
except AttributeError:  # Starlette < 0.27
    HTTP_413_TOO_LARGE = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

_PERSISTENCE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
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
    sqlite3.Error,
    HTTPException,
    ConflictError,
    DatabaseError,
    InputError,
)

_MEDIA_INGESTION_POLICY_ID = "media.default"
_MEDIA_INGESTION_BYTES_CATEGORY = "ingestion_bytes"

_media_ingestion_daily_ledger = None
_media_ingestion_daily_ledger_lock = asyncio.Lock()

_SHARED_TRANSCRIPT_REUSABLE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        return int(default)


def _resolve_media_writer(db_instance: Any) -> Any:
    return get_media_repository(db_instance)


def _with_media_db_session(
    *,
    db_path: str,
    client_id: str,
    operation: Callable[[MediaDatabase], Any],
) -> Any:
    worker_db = create_media_database(
        client_id,
        db_path=db_path,
    )
    try:
        return operation(worker_db)
    finally:
        worker_db.close_connection()


def _build_ingestion_budget_headers(
    *,
    category_details: dict[str, Any] | None,
    retry_after: int | None,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if retry_after is not None and int(retry_after) > 0:
        headers["Retry-After"] = str(int(retry_after))
    cat = category_details or {}
    limit_raw = cat.get("daily_cap")
    if limit_raw is None:
        limit_raw = cat.get("limit")
    remaining_raw = cat.get("daily_remaining")
    if remaining_raw is None:
        remaining_raw = cat.get("remaining")
    if limit_raw is not None:
        headers["X-RateLimit-Limit"] = str(max(0, _safe_int(limit_raw, 0)))
    if remaining_raw is not None:
        headers["X-RateLimit-Remaining"] = str(max(0, _safe_int(remaining_raw, 0)))
    return headers


def _resolve_media_budget_context(
    *,
    request: Request | None,
    current_user: Any,
) -> tuple[Any | None, str, dict[str, Any], str]:
    """Return (governor, policy_id, policy, entity) for media budget checks."""
    if request is None:
        return None, _MEDIA_INGESTION_POLICY_ID, {}, ""
    try:
        app_state = getattr(request.app, "state", None)
        gov = getattr(app_state, "rg_governor", None)
        loader = getattr(app_state, "rg_policy_loader", None)
        policy_id = str(getattr(request.state, "rg_policy_id", None) or _MEDIA_INGESTION_POLICY_ID)
        policy: dict[str, Any] = {}
        if loader is not None:
            try:
                policy = dict(loader.get_policy(policy_id) or {})
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                policy = {}
        if hasattr(current_user, "id") and getattr(current_user, "id") is not None:
            entity = f"user:{int(current_user.id)}"
        else:
            entity = ""
        return gov, policy_id, policy, entity
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        return None, _MEDIA_INGESTION_POLICY_ID, {}, ""


async def _get_media_ingestion_daily_ledger():
    global _media_ingestion_daily_ledger
    if _media_ingestion_daily_ledger is not None:
        return _media_ingestion_daily_ledger
    async with _media_ingestion_daily_ledger_lock:
        if _media_ingestion_daily_ledger is not None:
            return _media_ingestion_daily_ledger
        try:
            from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
                ResourceDailyLedger,
            )
        except (ImportError, ModuleNotFoundError):
            return None
        try:
            ledger = ResourceDailyLedger()
            await ledger.initialize()
            _media_ingestion_daily_ledger = ledger
            return ledger
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Media ingestion budget: failed to initialize ResourceDailyLedger: {}",
                exc,
            )
            return None


async def _record_media_ingestion_bytes_ledger_entry(
    *,
    entity_scope: str,
    entity_value: str,
    units: int,
    op_id: str,
) -> bool:
    if units <= 0:
        return False
    ledger = await _get_media_ingestion_daily_ledger()
    if ledger is None:
        return False
    try:
        from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
            LedgerEntry,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        inserted = await ledger.add(
            LedgerEntry(
                entity_scope=str(entity_scope),
                entity_value=str(entity_value),
                category=_MEDIA_INGESTION_BYTES_CATEGORY,
                units=max(0, int(units)),
                op_id=str(op_id),
                occurred_at=datetime.now(timezone.utc),
            )
        )
        return bool(inserted)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Media ingestion budget: ledger write failed for {}:{} units={}: {}",
            entity_scope,
            entity_value,
            units,
            exc,
        )
        return False


def _ensure_warnings_list(result: dict[str, Any]) -> list[str]:
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        result["warnings"] = warnings
    return warnings


def _callable_accepts_keyword(
    callable_obj: Callable[..., Any],
    keyword: str,
) -> bool:
    """Return True if callable_obj accepts keyword directly or via **kwargs."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    parameter = signature.parameters.get(keyword)
    if parameter and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True

    return any(candidate.kind == inspect.Parameter.VAR_KEYWORD for candidate in signature.parameters.values())


def _normalize_analysis_text_chunk(
    value: Any,
    *,
    max_chars: int = 4000,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False).strip()
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            text = str(value).strip()
    else:
        text = str(value).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _extract_analysis_extra_chunks_for_indexing(
    process_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build retrieval/embedding chunks from structured analysis outputs.

    Stage-2 parity path:
    - OCR table extraction -> `chunk_type=table`
    - VLM/image detections/captions -> `chunk_type=media` (or `table` for table labels)
    """
    if not isinstance(process_result, dict):
        return []
    details = process_result.get("analysis_details")
    if not isinstance(details, dict):
        return []

    out: list[dict[str, Any]] = []

    # OCR structured tables
    ocr_details = details.get("ocr")
    if isinstance(ocr_details, dict):
        structured = ocr_details.get("structured")
        if isinstance(structured, dict):
            tables_any = structured.get("tables")
            if isinstance(tables_any, list):
                for idx, table_entry in enumerate(tables_any):
                    table_format = "unknown"
                    content_any = table_entry
                    if isinstance(table_entry, dict):
                        table_format = str(table_entry.get("format") or "unknown")
                        content_any = table_entry.get("content")
                    table_text = _normalize_analysis_text_chunk(content_any)
                    if not table_text:
                        continue
                    out.append(
                        {
                            "text": f"OCR table ({table_format}): {table_text}",
                            "start_char": None,
                            "end_char": None,
                            "chunk_type": "table",
                            "metadata": {
                                "source": "ocr_structured_table",
                                "table_format": table_format,
                                "table_index": idx,
                            },
                        }
                    )

            pages_any = structured.get("pages")
            if isinstance(pages_any, list):
                for page_entry in pages_any:
                    if not isinstance(page_entry, dict):
                        continue
                    page_no = page_entry.get("page")
                    page_tables = page_entry.get("tables")
                    if not isinstance(page_tables, list):
                        continue
                    for idx, table_entry in enumerate(page_tables):
                        table_format = "unknown"
                        content_any = table_entry
                        if isinstance(table_entry, dict):
                            table_format = str(table_entry.get("format") or "unknown")
                            content_any = table_entry.get("content")
                        table_text = _normalize_analysis_text_chunk(content_any)
                        if not table_text:
                            continue
                        out.append(
                            {
                                "text": f"OCR table (page {page_no}, {table_format}): {table_text}",
                                "start_char": None,
                                "end_char": None,
                                "chunk_type": "table",
                                "metadata": {
                                    "source": "ocr_structured_page_table",
                                    "page": page_no,
                                    "table_format": table_format,
                                    "table_index": idx,
                                },
                            }
                        )

    # VLM/image detections and captions
    vlm_details = details.get("vlm")
    if isinstance(vlm_details, dict):
        by_page = vlm_details.get("by_page")
        if isinstance(by_page, list):
            for page_entry in by_page:
                if not isinstance(page_entry, dict):
                    continue
                page_no = page_entry.get("page")
                detections = page_entry.get("detections")
                if not isinstance(detections, list):
                    continue
                for det_idx, det in enumerate(detections):
                    if not isinstance(det, dict):
                        continue
                    label = str(det.get("label") or "").strip()
                    score = det.get("score")
                    bbox = det.get("bbox")
                    caption_text = _normalize_analysis_text_chunk(
                        det.get("caption") or det.get("description") or det.get("summary"),
                        max_chars=1200,
                    )
                    if caption_text:
                        text = f"Image caption: {caption_text}"
                    elif label:
                        if page_no is not None:
                            text = f"Detected {label} visual element on page {page_no}"
                        else:
                            text = f"Detected {label} visual element"
                    else:
                        continue
                    chunk_type = "table" if label.lower() == "table" else "media"
                    out.append(
                        {
                            "text": text,
                            "start_char": None,
                            "end_char": None,
                            "chunk_type": chunk_type,
                            "metadata": {
                                "source": "vlm_detection",
                                "page": page_no,
                                "label": label or None,
                                "score": score,
                                "bbox": bbox,
                                "detection_index": det_idx,
                            },
                        }
                    )

    # Stable dedupe by content + core metadata keys.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for chunk in out:
        md = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        key = (
            str(chunk.get("chunk_type") or ""),
            str(chunk.get("text") or ""),
            md.get("source"),
            md.get("page"),
            md.get("label"),
            md.get("table_format"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    return deduped


def _is_httpx_request_error(exc: Exception) -> bool:
    module = getattr(exc.__class__, "__module__", "")
    if not module.startswith("httpx"):
        return False
    return exc.__class__.__name__ in {"HTTPStatusError", "RequestError"}


def _document_like_concurrency_limit(default: int = 10) -> int:
    env_raw = os.getenv("DOCUMENT_LIKE_CONCURRENCY")
    if env_raw:
        try:
            env_val = int(str(env_raw).strip())
            if env_val > 0:
                return env_val
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        cfg = loaded_config_data.get("media_processing", {}) if loaded_config_data else {}
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        cfg = {}
    cfg_val = None
    if isinstance(cfg, dict):
        cfg_val = cfg.get("document_like_concurrency") or cfg.get("doc_like_concurrency")
    if cfg_val is not None:
        try:
            cfg_int = int(str(cfg_val).strip())
            if cfg_int > 0:
                return cfg_int
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass
    return max(1, int(default))


def _classify_upload_error(file_save_errors: list[dict[str, Any]]) -> tuple[int, str] | None:
    if not file_save_errors:
        return None
    priority = {HTTP_413_TOO_LARGE: 3, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: 2, status.HTTP_400_BAD_REQUEST: 1}
    chosen_status = status.HTTP_400_BAD_REQUEST
    chosen_detail = "File upload failed."
    for err_info in file_save_errors:
        error_msg = str(err_info.get("error", "") or "").strip()
        error_lower = error_msg.lower()
        if "exceeds maximum allowed size" in error_lower:
            candidate = (HTTP_413_TOO_LARGE, error_msg or "Uploaded file too large.")
        elif "not allowed for security reasons" in error_lower:
            candidate = (status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, error_msg or "Unsupported media type.")
        elif "empty" in error_lower:
            candidate = (status.HTTP_400_BAD_REQUEST, error_msg or "Uploaded file content is empty.")
        else:
            candidate = (status.HTTP_400_BAD_REQUEST, error_msg or "File upload failed.")
        candidate_priority = priority.get(candidate[0], 0)
        chosen_priority = priority.get(chosen_status, 0)
        if candidate_priority > chosen_priority or (
            candidate_priority == chosen_priority
            and chosen_detail in {"", "File upload failed."}
            and bool(candidate[1])
        ):
            chosen_status, chosen_detail = candidate
    return chosen_status, chosen_detail


def _is_nonfatal_upload_validation_error(error_msg: str) -> bool:
    """
    Return True when an upload error is a format/validation mismatch that should
    surface as item-level errors in a multi-status response, not a request-level
    hard failure.
    """
    lower = str(error_msg or "").strip().lower()
    if not lower:
        return False
    if (
        "not allowed for security reasons" in lower
        or "exceeds maximum allowed size" in lower
        or "too large" in lower
        or "quota exceeded" in lower
    ):
        return False
    return (
        "validation failed" in lower
        or "validation error" in lower
        or "invalid file type" in lower
        or "unsupported media type" in lower
        or "detected mime type" in lower
        or "unable to determine mime" in lower
        or "claimed extension" in lower
        or "could not identify file" in lower
    )


def _coerce_ingestion_label(value: Any, *, default: str = "unknown") -> str:
    raw = str(value or "").strip().lower()
    return raw or default


def _classify_ingestion_validation_failure_reason(message: str | None) -> str:
    lower = str(message or "").strip().lower()
    if not lower:
        return "unknown"
    if "empty" in lower:
        return "empty_file"
    if "security policy" in lower or "url blocked" in lower or "ssrf" in lower:
        return "security_policy"
    if (
        "maximum allowed size" in lower
        or "exceeds maximum" in lower
        or "too large" in lower
        or "quota exceeded" in lower
    ):
        return "size_limit"
    if "not allowed" in lower or "unsupported media type" in lower or "mime" in lower or "extension" in lower:
        return "file_type"
    if "validation" in lower or "validator" in lower:
        return "validator_rejected"
    return "other"


def _emit_ingestion_metric_increment(
    metric_name: str,
    value: float = 1,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    try:
        get_metrics_registry().increment(metric_name, value, labels=labels)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        # Metrics must never break ingestion execution paths.
        pass


def _emit_ingestion_metric_observe(
    metric_name: str,
    value: float,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    try:
        get_metrics_registry().observe(metric_name, value, labels=labels)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        # Metrics must never break ingestion execution paths.
        pass


def _emit_ingestion_request_metric(*, media_type: Any, outcome: str) -> None:
    _emit_ingestion_metric_increment(
        "ingestion_requests_total",
        1,
        labels={
            "media_type": _coerce_ingestion_label(media_type),
            "outcome": _coerce_ingestion_label(outcome),
        },
    )


def _emit_ingestion_processing_duration_metric(
    *,
    media_type: Any,
    processor: str,
    duration_seconds: float,
) -> None:
    safe_duration = float(duration_seconds) if duration_seconds > 0 else 0.0
    _emit_ingestion_metric_observe(
        "ingestion_processing_seconds",
        safe_duration,
        labels={
            "media_type": _coerce_ingestion_label(media_type),
            "processor": _coerce_ingestion_label(processor),
        },
    )


def _emit_ingestion_validation_failure_metric(*, reason: str, path_kind: str) -> None:
    _emit_ingestion_metric_increment(
        "ingestion_validation_failures_total",
        1,
        labels={
            "reason": _coerce_ingestion_label(reason),
            "path_kind": _coerce_ingestion_label(path_kind),
        },
    )


def _emit_ingestion_chunks_metric(
    *,
    media_type: Any,
    chunk_method: Any,
    chunk_count: int,
) -> None:
    if chunk_count <= 0:
        return
    _emit_ingestion_metric_increment(
        "ingestion_chunks_total",
        float(chunk_count),
        labels={
            "media_type": _coerce_ingestion_label(media_type),
            "chunk_method": _coerce_ingestion_label(chunk_method, default="none"),
        },
    )


def _emit_ingestion_embeddings_enqueue_metric(*, path_kind: str, outcome: str) -> None:
    _emit_ingestion_metric_increment(
        "ingestion_embeddings_enqueue_total",
        1,
        labels={
            "path_kind": _coerce_ingestion_label(path_kind),
            "outcome": _coerce_ingestion_label(outcome),
        },
    )


def _is_email_native_persist_enabled() -> bool:
    try:
        return bool(settings.get("EMAIL_NATIVE_PERSIST_ENABLED", True))
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        return True


def _emit_email_native_persist_metric(*, path_kind: str, outcome: str) -> None:
    _emit_ingestion_metric_increment(
        "email_native_persist_total",
        1,
        labels={
            "path_kind": _coerce_ingestion_label(path_kind),
            "outcome": _coerce_ingestion_label(outcome),
        },
    )


def _ingestion_request_outcome_from_status(status_code: int) -> str:
    if status_code == status.HTTP_200_OK:
        return "success"
    if status_code == status.HTTP_207_MULTI_STATUS:
        return "partial"
    return "error"


_METADATA_CONTRACT_POLICIES = {"off", "warn", "error"}
_CHUNK_CONSISTENCY_POLICIES = {"off", "warn", "error"}
_CHUNK_CONSISTENCY_SKIP_DB_MESSAGE_MARKERS = (
    "already exists",
    "url canonicalized",
)

_METADATA_COMMON_TYPED_KEYS: dict[str, tuple[type, ...]] = {
    "title": (str,),
    "author": (str,),
    "source_url": (str,),
    "url": (str,),
    "duration": (int, float),
    "page_count": (int,),
    "word_count": (int,),
    "parser_used": (str,),
    "model": (str,),
    "provider": (str,),
    "language": (str,),
    "source_format": (str,),
    "source_hash": (str,),
    "keywords": (list,),
    "tags": (list,),
    "email": (dict,),
    "chunking_plan": (dict,),
}

_METADATA_CONTRACTS: dict[str, dict[str, Any]] = {
    "default": {
        "required_any": (("title", "source_url", "url"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "audio": {
        "required_any": (("title", "source_url", "url", "model", "provider"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "duration",
            "language",
            "model",
            "provider",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "video": {
        "required_any": (("title", "source_url", "url", "model", "provider"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "duration",
            "language",
            "model",
            "provider",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "document": {
        "required_any": (("title", "source_url", "url", "parser_used"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "parser_used",
            "word_count",
            "language",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "json": {
        "required_any": (("title", "source_url", "url", "parser_used"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "parser_used",
            "word_count",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "pdf": {
        "required_any": (("title", "source_url", "url", "parser_used"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "page_count",
            "word_count",
            "parser_used",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "ebook": {
        "required_any": (("title", "source_url", "url", "parser_used"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "page_count",
            "word_count",
            "parser_used",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
    "email": {
        "required_any": (("title", "source_url", "url", "email"),),
        "optional": (
            "title",
            "author",
            "source_url",
            "url",
            "email",
            "parser_used",
            "source_format",
            "keywords",
            "tags",
            "source_hash",
        ),
        "typed": _METADATA_COMMON_TYPED_KEYS,
    },
}


def _resolve_metadata_contract_policy(form_data: Any | None = None) -> str:
    candidates: list[Any] = []
    if form_data is not None:
        with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
            candidates.append(getattr(form_data, "metadata_contract_policy", None))
    candidates.append(os.getenv("MEDIA_METADATA_CONTRACT_POLICY"))
    try:
        media_cfg = loaded_config_data.get("media_processing", {}) if loaded_config_data else {}
        if isinstance(media_cfg, dict):
            candidates.append(media_cfg.get("metadata_contract_policy"))
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass

    for raw in candidates:
        normalized = str(raw or "").strip().lower()
        if normalized in _METADATA_CONTRACT_POLICIES:
            return normalized
    return "warn"


def _resolve_metadata_contract_for_media_type(media_type: Any) -> dict[str, Any]:
    media_key = _coerce_ingestion_label(media_type)
    return _METADATA_CONTRACTS.get(media_key, _METADATA_CONTRACTS["default"])


def _metadata_value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _describe_expected_types(expected: tuple[type, ...]) -> str:
    names = [tp.__name__ for tp in expected]
    return "|".join(names)


def _evaluate_metadata_contract_issues(
    *,
    media_type: Any,
    metadata: Any,
) -> list[str]:
    if not isinstance(metadata, dict):
        return [f"metadata must be an object/dict (got {type(metadata).__name__})"]

    contract = _resolve_metadata_contract_for_media_type(media_type)
    issues: list[str] = []

    required_any = contract.get("required_any", ()) or ()
    for group in required_any:
        candidate_keys = [str(key).strip() for key in group if str(key).strip()]
        if not candidate_keys:
            continue
        if not any(_metadata_value_is_present(metadata.get(key)) for key in candidate_keys):
            issues.append(
                "missing one of required metadata keys: " + ", ".join(candidate_keys),
            )

    typed_keys = contract.get("typed", {}) or {}
    if isinstance(typed_keys, dict):
        for key, expected_types in typed_keys.items():
            if key not in metadata:
                continue
            value = metadata.get(key)
            if value is None:
                continue
            if not isinstance(expected_types, tuple):
                continue
            if not isinstance(value, expected_types):
                issues.append(
                    f"metadata.{key} expected {_describe_expected_types(expected_types)} "
                    f"(got {type(value).__name__})",
                )

    return issues


def _enforce_metadata_contract_on_result(
    *,
    result: dict[str, Any],
    media_type: Any,
    form_data: Any,
    path_kind: str,
    processor: str,
) -> None:
    policy = _resolve_metadata_contract_policy(form_data)
    if policy == "off":
        return

    status_value = _coerce_ingestion_label(result.get("status"), default="")
    if status_value not in {"success", "warning"}:
        return

    issues = _evaluate_metadata_contract_issues(
        media_type=media_type,
        metadata=result.get("metadata"),
    )
    if not issues:
        return

    issue_text = "; ".join(issues)
    input_ref = result.get("input_ref") or result.get("processing_source") or "unknown"
    logger.warning(
        "Metadata contract violations (policy={}) for {} item {} via {}: {}",
        policy,
        _coerce_ingestion_label(media_type),
        input_ref,
        processor,
        issue_text,
    )
    _emit_ingestion_validation_failure_metric(
        reason="metadata_contract",
        path_kind=path_kind,
    )

    warning_message = f"Metadata contract warning: {issue_text}"
    if policy == "error":
        error_message = f"Metadata contract validation failed: {issue_text}"
        existing_error = str(result.get("error", "") or "").strip()
        result["status"] = "Error"
        result["error"] = f"{existing_error} | {error_message}" if existing_error else error_message
        result["db_message"] = "DB operation skipped (metadata contract failure)."
        result["db_id"] = None
        result["media_uuid"] = None
        _ensure_warnings_list(result).append(error_message)
        return

    _ensure_warnings_list(result).append(warning_message)


def _resolve_chunk_consistency_policy(form_data: Any | None = None) -> str:
    candidates: list[Any] = []
    if form_data is not None:
        with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
            candidates.append(getattr(form_data, "chunk_consistency_policy", None))
    candidates.append(os.getenv("MEDIA_CHUNK_CONSISTENCY_POLICY"))
    try:
        media_cfg = loaded_config_data.get("media_processing", {}) if loaded_config_data else {}
        if isinstance(media_cfg, dict):
            candidates.append(media_cfg.get("chunk_consistency_policy"))
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass

    for raw in candidates:
        normalized = str(raw or "").strip().lower()
        if normalized in _CHUNK_CONSISTENCY_POLICIES:
            return normalized
    return "warn"


def _skip_chunk_consistency_for_db_message(db_message: Any) -> bool:
    normalized = str(db_message or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _CHUNK_CONSISTENCY_SKIP_DB_MESSAGE_MARKERS)


async def _fetch_unvectorized_chunk_count(
    *,
    db_path: str,
    client_id: str,
    media_id: int,
    loop: Any,
) -> int | None:
    media_id_int = _coerce_positive_int(media_id, 0)
    if media_id_int <= 0:
        return None

    def _count_worker() -> int | None:
        try:
            return _with_media_db_session(
                db_path=db_path,
                client_id=client_id,
                operation=lambda worker_db: worker_db.get_unvectorized_chunk_count(media_id_int),
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            return None

    try:
        return await loop.run_in_executor(None, _count_worker)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        return None


async def _enforce_chunk_consistency_after_persist(
    *,
    result: dict[str, Any],
    form_data: Any,
    media_type: Any,
    path_kind: str,
    processor: str,
    expected_chunk_count: int | None,
    db_message: Any,
    media_id: Any,
    db_path: str,
    client_id: str,
    loop: Any,
) -> None:
    policy = _resolve_chunk_consistency_policy(form_data)
    if policy == "off":
        return

    status_value = _coerce_ingestion_label(result.get("status"), default="")
    if status_value not in {"success", "warning"}:
        return

    if expected_chunk_count is None:
        return
    expected = _coerce_positive_int(expected_chunk_count, 0)

    media_id_int = _coerce_positive_int(media_id, 0)
    if media_id_int <= 0:
        return

    if _skip_chunk_consistency_for_db_message(db_message):
        return

    persisted_chunk_count = await _fetch_unvectorized_chunk_count(
        db_path=db_path,
        client_id=client_id,
        media_id=media_id_int,
        loop=loop,
    )
    if persisted_chunk_count is None:
        logger.warning(
            "Chunk consistency count unavailable for media_id={} via {}",
            media_id_int,
            processor,
        )
        return
    if persisted_chunk_count == expected:
        return

    issue_text = f"expected {expected} chunk rows but found {persisted_chunk_count} " f"(media_id={media_id_int})"
    logger.warning(
        "Chunk consistency mismatch (policy={}) for {} via {} [{}]: {}",
        policy,
        _coerce_ingestion_label(media_type),
        processor,
        path_kind,
        issue_text,
    )
    _emit_ingestion_validation_failure_metric(
        reason="chunk_consistency",
        path_kind=path_kind,
    )

    if policy == "error":
        error_message = f"Chunk consistency validation failed: {issue_text}"
        existing_error = str(result.get("error", "") or "").strip()
        result["status"] = "Error"
        result["error"] = f"{existing_error} | {error_message}" if existing_error else error_message
        _ensure_warnings_list(result).append(error_message)
        existing_db_message = str(result.get("db_message", "") or "").strip()
        result["db_message"] = f"{existing_db_message} | {error_message}" if existing_db_message else error_message
        return

    warning_message = f"Chunk consistency warning: {issue_text}"
    _ensure_warnings_list(result).append(warning_message)


def _compute_source_hash_safe(
    file_path: FilePath,
    base_dir: FilePath,
    *,
    chunk_size: int = 1024 * 1024,
) -> tuple[str | None, str | None]:
    """
    Compute a hash only for a validated local path within ``base_dir``.

    Returns (hash, warning_message). The warning is set only when the path is rejected.
    """
    safe_path = resolve_safe_local_path(file_path, base_dir)
    if safe_path is None:
        return None, "Source hash skipped: local path rejected outside allowed base directory."
    if not safe_path.is_file():
        logger.debug(
            "Skipping source hash computation for non-file path: {}",
            safe_path,
        )
        return None, None
    try:
        hasher = hashlib.sha256()
        handle = open_safe_local_path(safe_path, base_dir, mode="rb")
        if handle is None:
            logger.warning("Source hash compute skipped for rejected path: {}", safe_path)
            return None, None
        with handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest(), None
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        try:
            if safe_path.exists():
                logger.warning(
                    "Source hash compute failed for existing file {}: {}",
                    safe_path,
                    exc,
                )
            else:
                logger.debug("Source hash compute failed for {}: {}", safe_path, exc)
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            logger.debug("Source hash compute failed for {}: {}", safe_path, exc)
        return None, None


def _media_has_source_hash_column(db: MediaDatabase) -> bool:
    for table_name in ("Media", "media"):
        try:
            columns = {col.get("name", "").lower() for col in db.backend.get_table_info(table_name)}
            if columns:
                return "source_hash" in columns
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            continue
    return False


def _allowed_url_extensions(media_type: str, form_data: Any) -> set[str] | None:
    media_key = str(media_type).lower()
    if media_key == "json":
        return {".json"}
    if media_key == "email":
        allowed = {".eml"}
        if getattr(form_data, "accept_archives", False):
            allowed.add(".zip")
        if getattr(form_data, "accept_mbox", False):
            allowed.add(".mbox")
        if getattr(form_data, "accept_pst", False):
            allowed.update({".pst", ".ost"})
        return allowed
    cfg = DEFAULT_MEDIA_TYPE_CONFIG.get(media_key) if isinstance(DEFAULT_MEDIA_TYPE_CONFIG, dict) else None
    if cfg:
        extensions = cfg.get("allowed_extensions")
        if isinstance(extensions, (set, list, tuple)):
            normalized = {str(ext).lower() for ext in extensions if ext}
            if media_key == "document":
                # URL-based "document" ingestion is expected to accept web content
                # and XML-like payloads handled by the document processor.
                normalized.update({".html", ".htm", ".xml"})
            return normalized
    return None


def _normalize_dedupe_url_for_db(value: str) -> str:
    """Normalize URL-like identifiers before persistence while preserving non-URL refs."""
    raw = str(value).strip()
    if not raw:
        return raw
    normalized = normalize_media_dedupe_url(raw)
    return str(normalized).strip() if normalized else raw


def _merge_video_lite_summary_safe_metadata(
    safe_meta: dict[str, Any],
    *,
    process_result: dict[str, Any],
) -> dict[str, Any]:
    """Reserve a stable lightweight-summary slot separate from generic analysis content."""
    summary_value = process_result.get("summary")
    if summary_value is None:
        summary_value = process_result.get("analysis")

    merged = dict(safe_meta)
    existing_block = merged.get("video_lite")
    if isinstance(existing_block, dict):
        video_lite_meta = dict(existing_block)
    else:
        video_lite_meta = {}

    summary_status = str(process_result.get("video_lite_summary_status") or "").strip().lower()
    if summary_value is not None:
        summary_text = str(summary_value).strip()
        if summary_text:
            video_lite_meta["summary"] = summary_text
            video_lite_meta["summary_status"] = "ready"
            merged["video_lite"] = video_lite_meta
            return merged

    if summary_status == "failed":
        video_lite_meta["summary_status"] = "failed"
        merged["video_lite"] = video_lite_meta
        return merged

    if not video_lite_meta:
        return safe_meta

    merged["video_lite"] = video_lite_meta
    return merged


_SAFE_METADATA_ALLOWED_KEYS = frozenset(
    {
        "title",
        "author",
        "doi",
        "pmid",
        "pmcid",
        "arxiv_id",
        "s2_paper_id",
        "url",
        "pdf_url",
        "pmc_url",
        "date",
        "year",
        "venue",
        "journal",
        "license",
        "license_url",
        "publisher",
        "source",
        "creators",
        "rights",
        "source_hash",
        "chunking_plan",
    }
)


def _json_safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            cleaned = _json_safe_metadata_value(item)
            if cleaned is not None or item is None:
                cleaned_list.append(cleaned)
        return cleaned_list
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            if key is None:
                continue
            cleaned = _json_safe_metadata_value(item)
            if cleaned is not None or item is None:
                cleaned_dict[str(key)] = cleaned
        return cleaned_dict
    return None


def build_safe_metadata_subset(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Copy safe metadata fields while preserving nested JSON-safe plan data."""
    safe_meta: dict[str, Any] = {}
    if not isinstance(metadata, dict):
        return safe_meta

    for key, value in metadata.items():
        if key not in _SAFE_METADATA_ALLOWED_KEYS:
            continue
        cleaned = _json_safe_metadata_value(value)
        if cleaned is not None:
            safe_meta[key] = cleaned

    ext_ids = metadata.get("externalIds")
    if isinstance(ext_ids, dict):
        for ext_key in ("DOI", "ArXiv", "PMID", "PMCID"):
            ext_value = ext_ids.get(ext_key)
            if ext_value:
                target_key = "arxiv_id" if ext_key == "ArXiv" else ext_key.lower()
                safe_meta[target_key] = ext_value

    return safe_meta


def _shared_transcript_dedupe_candidates(
    *,
    source_path_or_url: str,
    source_hash: str | None,
    form_data: Any,
) -> list[tuple[str, str]]:
    raw_source = str(source_path_or_url or "").strip()
    if not raw_source:
        return []

    signature_payload = {
        "transcription_model": getattr(form_data, "transcription_model", None),
        "transcription_language": getattr(form_data, "transcription_language", None),
        "start_time": getattr(form_data, "start_time", None),
        "end_time": getattr(form_data, "end_time", None),
        "timestamp_option": getattr(form_data, "timestamp_option", None),
        "diarize": bool(getattr(form_data, "diarize", False)),
        "vad_use": bool(getattr(form_data, "vad_use", False)),
    }
    normalized_signature = {key: value for key, value in signature_payload.items() if value not in (None, "")}
    reuse_signature = json.dumps(
        normalized_signature,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    candidates: list[tuple[str, str]] = []
    if raw_source.lower().startswith(("http://", "https://")):
        normalized = normalize_media_dedupe_url(raw_source) or raw_source
        parsed = urlparse(str(normalized))
        host = (parsed.netloc or "").strip().lower()
        if ":" in host:
            host = host.split(":", 1)[0]
        if host in _SHARED_TRANSCRIPT_REUSABLE_HOSTS:
            candidates.append(("url", f"{str(normalized).strip()}|{reuse_signature}"))
        return candidates

    source_hash_value = str(source_hash or "").strip()
    if source_hash_value:
        candidates.append(("source_hash", f"{source_hash_value}|{reuse_signature}"))
    return candidates


async def _get_media_ingest_dedupe_repo():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.media_ingest_dedupe_repo import (
        MediaIngestDedupeRepo,
    )

    pool = await get_db_pool()
    repo = MediaIngestDedupeRepo(pool)
    await repo.ensure_tables()
    return repo


def _parse_normalized_stt_artifact(raw_value: Any) -> dict[str, Any] | None:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        return None
    stripped = raw_value.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def _extract_reusable_transcript_payload(
    *,
    source_media: dict[str, Any],
    transcript_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    normalized_stt = None
    for row in transcript_rows:
        normalized_stt = _parse_normalized_stt_artifact(row.get("transcription"))
        if normalized_stt:
            break

    content_text = ""
    if isinstance(normalized_stt, dict):
        content_text = str(normalized_stt.get("text") or "").strip()
    if not content_text:
        content_text = str(source_media.get("content") or "").strip()

    if normalized_stt is None and content_text:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            to_normalized_stt_artifact,
        )

        normalized_stt = to_normalized_stt_artifact(
            text=content_text,
            segments=None,
            language=None,
            provider="dedupe",
            model=str(source_media.get("transcription_model") or "dedupe"),
        )

    return content_text, normalized_stt


def _build_reused_process_result(
    *,
    original_input_ref: str,
    media_type: str,
    content_text: str,
    source_media: dict[str, Any],
    normalized_stt: dict[str, Any] | None,
) -> dict[str, Any]:
    reusable_content_text = str(content_text or "").strip()
    if not reusable_content_text and isinstance(normalized_stt, dict):
        reusable_content_text = str(normalized_stt.get("text") or "").strip()
    artifact_meta = dict(normalized_stt.get("metadata") or {}) if isinstance(normalized_stt, dict) else {}
    metadata = {
        "title": source_media.get("title"),
        "author": source_media.get("author"),
        "url": source_media.get("url"),
        "source_url": source_media.get("url"),
        "model": artifact_meta.get("model") or source_media.get("transcription_model"),
        "provider": artifact_meta.get("provider"),
        "source_hash": source_media.get("source_hash"),
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "")}

    analysis_details: dict[str, Any] = {}
    language_value = artifact_meta.get("language")
    if language_value:
        analysis_details["transcription_language"] = language_value

    return {
        "status": "Success",
        "input_ref": original_input_ref,
        "processing_source": original_input_ref,
        "media_type": media_type,
        "metadata": metadata,
        "content": reusable_content_text,
        "segments": (normalized_stt.get("segments") if isinstance(normalized_stt, dict) else None),
        "chunks": None,
        "analysis": None,
        "summary": None,
        "analysis_details": analysis_details,
        "error": None,
        "warnings": None,
        "normalized_stt": normalized_stt,
        "video_lite_summary_status": None,
    }


def _synthesize_reused_transcript_analysis_sync(
    *,
    process_result: dict[str, Any],
    form_data: Any,
    chunk_options: dict[str, Any] | None,
) -> tuple[str | None, list[str], list[dict[str, Any]] | None]:
    from tldw_Server_API.app.core.Chunking import improved_chunking_process
    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze

    warnings: list[str] = []
    text_to_analyze = str(process_result.get("content") or "").strip()
    if not text_to_analyze:
        return None, warnings, None

    api_name = getattr(form_data, "api_name", None)
    if not api_name or str(api_name).strip().lower() == "none":
        return None, warnings, None

    custom_prompt = getattr(form_data, "custom_prompt", None)
    system_prompt = getattr(form_data, "system_prompt", None)
    perform_chunking = bool(getattr(form_data, "perform_chunking", True))
    summarize_recursively = bool(getattr(form_data, "summarize_recursively", False))

    if not perform_chunking or not chunk_options:
        return (
            analyze(
                api_name,
                text_to_analyze,
                custom_prompt,
                None,
                system_message=system_prompt,
            ),
            warnings,
            None,
        )

    try:
        chunked_texts_list = improved_chunking_process(text_to_analyze, chunk_options)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        warnings.append(f"Chunking process failed during dedupe reuse: {exc}")
        return (
            analyze(
                api_name,
                text_to_analyze,
                custom_prompt,
                None,
                system_message=system_prompt,
            ),
            warnings,
            None,
        )

    if not chunked_texts_list:
        warnings.append("Chunking yielded no chunks during dedupe reuse.")
        return (
            analyze(
                api_name,
                text_to_analyze,
                custom_prompt,
                None,
                system_message=system_prompt,
            ),
            warnings,
            None,
        )

    chunk_summaries: list[str] = []
    for index, chunk_block in enumerate(chunked_texts_list):
        chunk_text = str((chunk_block or {}).get("text") or "").strip()
        if not chunk_text:
            continue
        try:
            chunk_summary = analyze(
                api_name,
                chunk_text,
                custom_prompt,
                None,
                system_message=system_prompt,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
            warnings.append(f"Chunk summarization failed during dedupe reuse for chunk {index}: {exc}")
            continue
        if chunk_summary:
            chunk_summaries.append(str(chunk_summary))
            if isinstance(chunk_block.get("metadata"), dict):
                chunk_block["metadata"]["summary"] = str(chunk_summary)
            else:
                chunk_block["metadata"] = {"summary": str(chunk_summary)}

    if not chunk_summaries:
        warnings.append("No chunk summaries were produced during dedupe reuse.")
        return None, warnings, chunked_texts_list

    if summarize_recursively and len(chunk_summaries) > 1:
        combined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries)
        return (
            analyze(
                api_name,
                combined_chunk_summaries,
                custom_prompt or "Summarize the key points from the preceding text sections.",
                None,
                system_message=system_prompt,
            ),
            warnings,
            chunked_texts_list,
        )

    return "\n\n---\n\n".join(chunk_summaries), warnings, chunked_texts_list


async def _populate_reused_transcript_summary(
    *,
    process_result: dict[str, Any],
    form_data: Any,
    chunk_options: dict[str, Any] | None,
    loop: asyncio.AbstractEventLoop,
) -> None:
    if process_result.get("analysis") or process_result.get("summary"):
        return
    if not getattr(form_data, "perform_analysis", False):
        return

    analysis_details = process_result.get("analysis_details")
    if not isinstance(analysis_details, dict):
        analysis_details = {}
        process_result["analysis_details"] = analysis_details
    analysis_details["llm_api"] = getattr(form_data, "api_name", None)
    analysis_details["custom_prompt"] = getattr(form_data, "custom_prompt", None)
    analysis_details["system_prompt"] = getattr(form_data, "system_prompt", None)
    if chunk_options:
        analysis_details["chunking_options"] = dict(chunk_options)

    try:
        analysis_text, warnings, chunks = await loop.run_in_executor(
            None,
            functools.partial(
                _synthesize_reused_transcript_analysis_sync,
                process_result=process_result,
                form_data=form_data,
                chunk_options=chunk_options,
            ),
        )
        if chunks is not None:
            process_result["chunks"] = chunks
        if warnings:
            _ensure_warnings_list(process_result).extend(warnings)
        process_result["analysis"] = analysis_text
        process_result["summary"] = analysis_text
        if analysis_text:
            process_result["video_lite_summary_status"] = "ready"
        else:
            process_result["video_lite_summary_status"] = "failed"
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        _ensure_warnings_list(process_result).append(f"Summary generation failed during dedupe reuse: {exc}")
        process_result["video_lite_summary_status"] = "failed"


async def _maybe_build_reused_transcript_process_result(
    *,
    media_type: str,
    original_input_ref: str,
    source_path_or_url: str,
    source_hash: str | None,
    form_data: Any,
    chunk_options: dict[str, Any] | None,
    loop: asyncio.AbstractEventLoop,
) -> dict[str, Any] | None:
    if str(media_type) != "video":
        return None

    dedupe_candidates = _shared_transcript_dedupe_candidates(
        source_path_or_url=source_path_or_url,
        source_hash=source_hash,
        form_data=form_data,
    )
    if not dedupe_candidates:
        return None

    repo = await _get_media_ingest_dedupe_repo()
    hit = await repo.lookup_source(
        media_type=str(media_type),
        dedupe_keys=[f"{kind}:{value}" for kind, value in dedupe_candidates],
    )
    if not hit:
        return None

    source_user_id = hit.get("source_user_id")
    source_media_id = hit.get("source_media_id")
    try:
        source_user_id_int = int(source_user_id)
        source_media_id_int = int(source_media_id)
    except (TypeError, ValueError):
        return None

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        get_media_transcripts,
    )

    source_db = create_media_database(
        client_id=f"media_ingest_dedupe_source:{source_user_id_int}",
        db_path=str(DatabasePaths.get_media_db_path(str(source_user_id_int))),
    )
    try:
        source_media = source_db.get_media_by_id(
            source_media_id_int,
            include_deleted=False,
            include_trash=False,
        )
        if not source_media:
            return None
        transcript_rows = get_media_transcripts(source_db, source_media_id_int)
        content_text, normalized_stt = _extract_reusable_transcript_payload(
            source_media=source_media,
            transcript_rows=transcript_rows,
        )
        if not content_text:
            return None
        reused_result = _build_reused_process_result(
            original_input_ref=original_input_ref,
            media_type=media_type,
            content_text=content_text,
            source_media=source_media,
            normalized_stt=normalized_stt,
        )
    finally:
        source_db.close_connection()

    await _populate_reused_transcript_summary(
        process_result=reused_result,
        form_data=form_data,
        chunk_options=chunk_options,
        loop=loop,
    )
    return reused_result


async def _register_reusable_transcript_source(
    *,
    media_type: str,
    original_input_ref: str,
    source_hash: str | None,
    process_result: dict[str, Any],
    user_id: int | None,
    form_data: Any,
) -> None:
    if str(media_type) != "video" or user_id is None:
        return

    media_id = process_result.get("db_id")
    if media_id is None:
        return
    try:
        media_id_int = int(media_id)
    except (TypeError, ValueError):
        return

    dedupe_candidates = _shared_transcript_dedupe_candidates(
        source_path_or_url=original_input_ref,
        source_hash=source_hash,
        form_data=form_data,
    )
    if not dedupe_candidates:
        return

    repo = await _get_media_ingest_dedupe_repo()
    for key_type, key_value in dedupe_candidates:
        await repo.upsert_source(
            dedupe_key=f"{key_type}:{key_value}",
            key_type=key_type,
            media_type=str(media_type),
            source_user_id=int(user_id),
            source_media_id=media_id_int,
            source_url=original_input_ref if key_type == "url" else None,
            source_hash=source_hash if key_type == "source_hash" else None,
        )


def _build_url_match_clause(
    url_candidates: tuple[str, ...],
    *,
    column: str = "url",
) -> tuple[str, tuple[str, ...]]:
    """Build a SQL match clause for one or more candidate URLs."""
    if not url_candidates:
        return f"{column} = ?", ("",)
    if len(url_candidates) == 1:
        return f"{column} = ?", (url_candidates[0],)
    placeholders = ", ".join(["?"] * len(url_candidates))
    return f"{column} IN ({placeholders})", url_candidates


def _resolve_ingestion_file_validator(media_mod: Any | None) -> Any:
    """
    Resolve the shared file validator instance used by media ingestion flows.

    Prefer the endpoint-exported validator patchpoint (for tests that
    monkeypatch `endpoints.media.file_validator_instance`) and fall back to
    the core dependency singleton.
    """
    try:
        from tldw_Server_API.app.api.v1.API_Deps.validations_deps import (  # type: ignore  # noqa: E501
            file_validator_instance as core_file_validator_instance,
        )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (  # type: ignore  # noqa: E501
            FileValidator,
        )

        core_file_validator_instance = FileValidator()

    if media_mod is not None:
        try:
            return getattr(
                media_mod,
                "file_validator_instance",
                core_file_validator_instance,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            return core_file_validator_instance
    return core_file_validator_instance


def _validate_downloaded_url_file(
    *,
    downloaded_path: FilePath,
    processing_filename: str | None,
    media_type: Any,
    form_data: Any,
    media_mod: Any | None,
    allowed_extensions: set[str] | None,
) -> None:
    """
    Apply upload-equivalent validation rules to files fetched from URLs.

    This mirrors upload-path behavior for archive/pst special-cases while
    reusing `process_and_validate_file` for normal document-like inputs.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (  # type: ignore  # noqa: E501
        FileValidationError,
        _resolve_media_type_key,
        process_and_validate_file,
    )

    validator = _resolve_ingestion_file_validator(media_mod)
    normalized_allowed_extensions = (
        {str(ext).lower() for ext in allowed_extensions if ext} if allowed_extensions is not None else None
    )

    file_extension = downloaded_path.suffix.lower()
    archive_exts = {
        ".zip",
        ".tar",
        ".tgz",
        ".tar.gz",
        ".tbz2",
        ".tar.bz2",
        ".txz",
        ".tar.xz",
    }
    validate_email_archive_contents = True
    try:
        media_cfg = loaded_config_data.get("media_processing", {}) if loaded_config_data else {}
        if isinstance(media_cfg, dict):
            validate_email_archive_contents = bool(
                media_cfg.get("validate_email_archive_contents", True),
            )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        validate_email_archive_contents = True

    skip_archive_scanning = (
        str(media_type).lower() == "email"
        and bool(getattr(form_data, "accept_archives", False))
        and file_extension in archive_exts
        and not validate_email_archive_contents
    )
    is_pst_ost = file_extension in {".pst", ".ost"}
    pst_accepted = normalized_allowed_extensions is not None and (
        ".pst" in normalized_allowed_extensions or ".ost" in normalized_allowed_extensions
    )

    try:
        if skip_archive_scanning:
            validation_result = validator.validate_file(
                downloaded_path,
                original_filename=processing_filename,
                media_type_key="archive",
            )
        elif is_pst_ost and pst_accepted:
            validation_result = validator.validate_file(
                downloaded_path,
                original_filename=processing_filename,
                media_type_key="email",
                allowed_mimetypes_override=set(),
            )
        else:
            try:
                inferred_media_key = _resolve_media_type_key(
                    processing_filename or str(downloaded_path),
                )
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                inferred_media_key = None

            media_key_override = inferred_media_key or str(media_type)
            validation_result = process_and_validate_file(
                downloaded_path,
                validator,
                original_filename=processing_filename,
                media_type_key_override=media_key_override,
            )
    except FileValidationError as validation_err:
        issues = getattr(validation_err, "issues", None) or [str(validation_err)]
        issue_msg = "; ".join(str(issue) for issue in issues)
        _emit_ingestion_validation_failure_metric(
            reason=_classify_ingestion_validation_failure_reason(issue_msg),
            path_kind="url",
        )
        raise ValueError(f"Downloaded file failed validation: {issue_msg}") from validation_err

    if not validation_result:
        issues = getattr(validation_result, "issues", None) or ["Unknown validation failure"]
        issue_msg = "; ".join(str(issue) for issue in issues)
        _emit_ingestion_validation_failure_metric(
            reason=_classify_ingestion_validation_failure_reason(issue_msg),
            path_kind="url",
        )
        raise ValueError(f"Downloaded file failed validation: {issue_msg}")


def sync_media_add_results_to_collections(
    *,
    results: list[dict[str, Any]],
    form_data: Any,
    current_user: Any,
    db: Any,
) -> None:
    """
    Dual-write successful `/media/add` items into Collections `content_items`.

    This is intentionally non-fatal: failures are recorded as warnings on each
    result item and do not fail the ingestion request.
    """
    user_id = getattr(current_user, "id", None)
    if user_id is None or not isinstance(results, list):
        return

    from tldw_Server_API.app.core.DB_Management.Collections_DB import (  # type: ignore
        CollectionsDatabase,
    )

    collections_origin = "media_add"
    form_origin = getattr(form_data, "collections_origin", None)
    if isinstance(form_origin, str) and form_origin.strip():
        collections_origin = form_origin.strip()

    collections_db = None
    try:
        backend = getattr(db, "backend", None)
        if backend is not None:
            collections_db = CollectionsDatabase.from_backend(
                user_id=user_id,
                backend=backend,
            )
        else:
            collections_db = CollectionsDatabase.for_user(user_id=user_id)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Collections dual-write initialization failed: {}", exc)
        return

    try:
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("status") not in {"Success", "Warning"}:
                continue
            db_id = result.get("db_id")
            if db_id is None:
                continue
            try:
                media_id = int(db_id)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                continue

            input_ref = str(result.get("input_ref") or "")
            processing_source = str(result.get("processing_source") or "")
            media_uuid = result.get("media_uuid")
            media_type = str(result.get("media_type") or getattr(form_data, "media_type", ""))

            metadata = result.get("metadata")
            metadata_map = metadata if isinstance(metadata, dict) else {}

            title = (
                metadata_map.get("title")
                or getattr(form_data, "title", None)
                or (FilePath(input_ref).stem if input_ref else None)
                or f"Media {media_id}"
            )

            content_val = result.get("content")
            if content_val is None:
                content_val = result.get("transcript")
            if content_val is None:
                content_text = ""
            elif isinstance(content_val, str):
                content_text = content_val
            else:
                try:
                    content_text = json.dumps(content_val, ensure_ascii=False)
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    content_text = str(content_val or "")

            analysis_val = result.get("analysis")
            if analysis_val is None:
                analysis_val = result.get("summary")
            if analysis_val is None:
                analysis_val = metadata_map.get("summary")
            summary_text = str(analysis_val or "").strip()
            if not summary_text and content_text:
                summary_text = content_text[:600]
            if len(summary_text) > 600:
                summary_text = summary_text[:600]

            source_url = (
                input_ref
                if isinstance(input_ref, str) and input_ref.lower().startswith(("http://", "https://"))
                else None
            )
            url_value = source_url or input_ref or f"media://{media_id}"
            canonical_url = f"media://{media_id}"
            domain = None
            if source_url:
                with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                    domain = (urlparse(source_url).hostname or "").lower() or None

            content_hash = (
                hashlib.sha256(content_text.encode("utf-8", errors="ignore")).hexdigest() if content_text else None
            )
            word_count = len(content_text.split()) if content_text else None
            published_at = metadata_map.get("published_at") or metadata_map.get("publish_date")

            tags: list[str] = []
            seen: set[str] = set()
            for keyword in getattr(form_data, "keywords", []) or []:
                if isinstance(keyword, str):
                    normalized = keyword.strip().lower()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        tags.append(normalized)
            for keyword in metadata_map.get("keywords", []) if isinstance(metadata_map.get("keywords"), list) else []:
                if isinstance(keyword, str):
                    normalized = keyword.strip().lower()
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        tags.append(normalized)

            provenance_payload = {
                "entrypoint": "/api/v1/media/add",
                "origin": collections_origin,
                "media_id": media_id,
                "media_uuid": media_uuid,
                "media_type": media_type,
                "input_ref": input_ref,
                "processing_source": processing_source,
                "source_url": source_url,
            }
            metadata_payload: dict[str, Any] = {
                "origin": collections_origin,
                "provenance": provenance_payload,
                "media_type": media_type,
                "media_uuid": media_uuid,
                "input_ref": input_ref,
                "processing_source": processing_source,
                "title": title,
                "summary": summary_text,
                "source_url": source_url,
                "source_domain": domain,
                "tags": tags,
            }

            item_row = collections_db.upsert_content_item(
                origin=collections_origin,
                origin_type=media_type or None,
                origin_id=media_id,
                url=url_value,
                canonical_url=canonical_url,
                domain=domain,
                title=title,
                summary=summary_text or None,
                notes=None,
                content_hash=content_hash,
                word_count=word_count,
                published_at=str(published_at) if published_at else None,
                status="saved",
                favorite=False,
                metadata=metadata_payload,
                media_id=media_id,
                job_id=None,
                run_id=None,
                source_id=media_id,
                read_at=None,
                tags=tags,
                merge_tags=True,
            )
            result["collections_item_id"] = item_row.id
            result["collections_origin"] = collections_origin

            planned_item_id_raw = (
                getattr(form_data, "media_collection_item_id", None)
                or getattr(form_data, "planned_item_id", None)
            )
            try:
                planned_item_id = int(planned_item_id_raw) if planned_item_id_raw is not None else None
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                planned_item_id = None
            if planned_item_id is not None and planned_item_id > 0:
                latest_job_id = getattr(form_data, "media_ingest_job_id", None)
                resolved_status = "completed"
                status_text = str(result.get("status") or "").strip().lower()
                message_text = str(result.get("db_message") or result.get("message") or "").lower()
                if status_text in {"skipped", "duplicate", "skipped_existing"} or "already exists" in message_text:
                    resolved_status = "skipped_existing"
                collections_db.resolve_media_collection_item(
                    planned_item_id,
                    media_id=media_id,
                    status=resolved_status,
                    latest_job_id=str(latest_job_id).strip() if latest_job_id is not None else None,
                )
                result["media_collection_item_id"] = planned_item_id

    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Collections dual-write failed: {}", exc)
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("status") not in {"Success", "Warning"}:
                continue
            if result.get("db_id") is None:
                continue
            _ensure_warnings_list(result).append(
                f"Collections dual-write failed: {exc}",
            )
    finally:
        with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
            collections_db.close()


def _normalize_embeddings_dispatch_mode(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "job": "jobs",
        "queue": "jobs",
        "queued": "jobs",
        "direct": "background",
        "legacy": "background",
        "bg": "background",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"auto", "jobs", "background"}:
        return normalized
    return None


def _resolve_media_add_embeddings_mode(form_data: Any) -> str:
    form_mode = _normalize_embeddings_dispatch_mode(
        getattr(form_data, "embedding_dispatch_mode", None),
    )
    if form_mode:
        return form_mode

    env_mode = _normalize_embeddings_dispatch_mode(
        os.getenv("MEDIA_ADD_EMBEDDINGS_MODE"),
    )
    if env_mode:
        return env_mode

    cfg_mode: Any = None
    try:
        cfg = loaded_config_data.get("media_processing", {}) if loaded_config_data else {}
        if isinstance(cfg, dict):
            cfg_mode = cfg.get("media_add_embeddings_mode")
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        cfg_mode = None

    normalized_cfg_mode = _normalize_embeddings_dispatch_mode(cfg_mode)
    if normalized_cfg_mode:
        return normalized_cfg_mode
    return "auto"


def _mark_media_embeddings_complete(db: Any, media_id: int) -> bool:
    """Mark embeddings generation as completed for a media row."""
    try:
        mark_media_as_processed(db_instance=db, media_id=media_id)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to mark embeddings complete for media {}: {}",
            media_id,
            exc,
        )
        return False
    return True


def _mark_media_embeddings_error(db: Any, media_id: int, error_detail: Any) -> bool:
    """Record an embeddings failure against a media row when possible."""
    detail = str(error_detail or "Embedding generation failed").strip() or "Embedding generation failed"
    try:
        mark_error = getattr(db, "mark_embeddings_error", None)
        if not callable(mark_error):
            raise AttributeError("Database instance does not expose mark_embeddings_error")
        mark_error(media_id, detail)
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Failed to mark embeddings error for media {}: {}",
            media_id,
            exc,
        )
        return False
    except AttributeError as exc:
        logger.warning(
            "Failed to mark embeddings error for media {}: {}",
            media_id,
            exc,
        )
        return False
    return True


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass
    return int(default)


def _resolve_media_add_embedding_config(form_data: Any) -> tuple[str, str, int, int]:
    embedding_settings = settings.get("EMBEDDING_CONFIG", {}) or {}
    embedding_model = (
        getattr(form_data, "embedding_model", None)
        or embedding_settings.get("embedding_model")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_provider = (
        getattr(form_data, "embedding_provider", None) or embedding_settings.get("embedding_provider") or "huggingface"
    )
    chunk_size = _coerce_positive_int(getattr(form_data, "chunk_size", None), 1000)
    chunk_overlap = _coerce_positive_int(
        getattr(form_data, "chunk_overlap", None) or getattr(form_data, "overlap", None),
        200,
    )
    return (
        str(embedding_model),
        str(embedding_provider),
        chunk_size,
        chunk_overlap,
    )


def _build_media_add_embeddings_provenance(
    *,
    result: dict[str, Any],
    form_data: Any,
    current_user: Any,
    media_id: int,
) -> dict[str, Any]:
    origin = str(result.get("collections_origin") or getattr(form_data, "collections_origin", None) or "media_add")
    media_type = str(result.get("media_type") or getattr(form_data, "media_type", "")).strip()
    input_ref = str(result.get("input_ref") or "")
    processing_source = str(result.get("processing_source") or "")
    source_url = (
        input_ref if isinstance(input_ref, str) and input_ref.lower().startswith(("http://", "https://")) else None
    )

    collections_item_id = result.get("collections_item_id")
    try:
        collections_item_id = int(collections_item_id) if collections_item_id is not None else None
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        collections_item_id = None

    return {
        "origin": origin,
        "origin_type": media_type or None,
        "origin_id": media_id,
        "source_id": media_id,
        "run_id": None,
        "job_id": None,
        "media_id": media_id,
        "media_uuid": result.get("media_uuid"),
        "collections_item_id": collections_item_id,
        "request_source": "media_add",
        "entrypoint": "/api/v1/media/add",
        "user_id": getattr(current_user, "id", None),
        "input_ref": input_ref,
        "processing_source": processing_source,
        "source_url": source_url,
    }


async def schedule_media_add_embeddings(
    *,
    results: list[dict[str, Any]],
    form_data: Any,
    background_tasks: BackgroundTasks,
    db: MediaDatabase,
    current_user: Any,
) -> None:
    """
    Schedule embeddings for successful `/media/add` items.

    Dispatch strategy is selected from:
    1) `form_data.embedding_dispatch_mode`
    2) `MEDIA_ADD_EMBEDDINGS_MODE` env var
    3) `media_processing.media_add_embeddings_mode` config
    4) default `auto` (jobs first, fallback to background)
    """
    generate_embeddings = bool(getattr(form_data, "generate_embeddings", False))
    logger.info("generate_embeddings flag: {}", generate_embeddings)
    if not generate_embeddings:
        return

    dispatch_mode = _resolve_media_add_embeddings_mode(form_data)
    logger.info(
        "Scheduling embeddings for successfully processed media items (dispatch_mode={})",
        dispatch_mode,
    )

    embedding_model, embedding_provider, chunk_size, chunk_overlap = _resolve_media_add_embedding_config(form_data)
    user_id = str(getattr(current_user, "id", "1"))

    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("status") not in {"Success", "Warning"}:
            continue
        db_id = result.get("db_id")
        if db_id is None:
            continue
        try:
            media_id = int(db_id)
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            continue

        provenance = _build_media_add_embeddings_provenance(
            result=result,
            form_data=form_data,
            current_user=current_user,
            media_id=media_id,
        )
        result["embeddings_provenance"] = provenance

        dispatched = False
        if dispatch_mode in {"auto", "jobs"}:
            try:
                from tldw_Server_API.app.core.Embeddings.jobs_adapter import (  # type: ignore
                    EmbeddingsJobsAdapter,
                )

                adapter = EmbeddingsJobsAdapter()
                job_row = adapter.create_job(
                    user_id=user_id,
                    media_id=media_id,
                    embedding_model=embedding_model,
                    embedding_provider=embedding_provider,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    request_source="media_add",
                    force_regenerate=False,
                    stage="chunking",
                    embedding_priority=None,
                    provenance=provenance,
                )
                job_id = str((job_row or {}).get("uuid") or (job_row or {}).get("id") or "").strip()
                result["embeddings_scheduled"] = True
                result["embeddings_dispatch"] = "jobs"
                if job_id:
                    result["embeddings_job_id"] = job_id
                _emit_ingestion_embeddings_enqueue_metric(
                    path_kind="jobs",
                    outcome="success",
                )
                dispatched = True
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as jobs_err:
                logger.warning(
                    "Failed to enqueue embeddings job for media {}: {}",
                    media_id,
                    jobs_err,
                )
                _emit_ingestion_embeddings_enqueue_metric(
                    path_kind="jobs",
                    outcome="failure",
                )
                _ensure_warnings_list(result).append(
                    f"Embeddings jobs enqueue failed: {jobs_err}",
                )
                if dispatch_mode == "jobs":
                    continue

        if not dispatched and dispatch_mode in {"auto", "background"}:
            logger.info(
                "Scheduling background embedding generation for media ID {}",
                media_id,
            )

            async def generate_embeddings_task(
                media_id: int,
                provenance_payload: dict[str, Any],
            ) -> None:
                try:
                    from tldw_Server_API.app.api.v1.endpoints.media_embeddings import (  # type: ignore
                        generate_embeddings_for_media,
                        get_media_content,
                    )

                    media_content = await get_media_content(media_id, db)
                    media_item = media_content.get("media_item")
                    if isinstance(media_item, dict):
                        meta = media_item.get("metadata")
                        metadata_payload = meta if isinstance(meta, dict) else {}
                        metadata_payload["embedding_provenance"] = provenance_payload
                        media_item["metadata"] = metadata_payload

                    result_emb = await generate_embeddings_for_media(
                        media_id=media_id,
                        media_content=media_content,
                        embedding_model=embedding_model,
                        embedding_provider=embedding_provider,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        user_id=user_id,
                    )
                    if result_emb.get("status") == "success":
                        if not _mark_media_embeddings_complete(db, media_id):
                            _mark_media_embeddings_error(
                                db,
                                media_id,
                                "Failed to persist embeddings completion status",
                            )
                    else:
                        _mark_media_embeddings_error(
                            db,
                            media_id,
                            result_emb.get("error") or result_emb.get("message") or "Embedding generation failed",
                        )
                    logger.info(
                        "Embedding generation result for media {}: {}",
                        media_id,
                        result_emb,
                    )
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as embed_err:
                    _mark_media_embeddings_error(db, media_id, embed_err)
                    logger.error(
                        "Failed to generate embeddings for media {}: {}",
                        media_id,
                        embed_err,
                    )

            background_tasks.add_task(
                generate_embeddings_task,
                media_id,
                provenance,
            )
            result["embeddings_scheduled"] = True
            result["embeddings_dispatch"] = "background"
            _emit_ingestion_embeddings_enqueue_metric(
                path_kind="background",
                outcome="success",
            )


def validate_add_media_inputs(
    media_type: Any,
    urls: list[str] | None,
    files: list[UploadFile] | None,
) -> None:
    """
    Validate basic inputs for the `/media/add` endpoint.

    This is the core implementation of the historical endpoint
    `_validate_inputs` helper.
    """
    if not urls and not files:
        logger.warning("No URLs or files provided in add_media request")
        _emit_ingestion_request_metric(media_type=media_type, outcome="error")
        _emit_ingestion_processing_duration_metric(
            media_type=media_type,
            processor="media_add_orchestrate",
            duration_seconds=0.0,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No valid media sources supplied. At least one 'url' in the "
                "'urls' list or one 'file' in the 'files' list must be provided."
            ),
        )


def determine_add_media_final_status(results: list[dict[str, Any]]) -> int:
    """
    Determine the overall HTTP status code for `/media/add` responses.

    Mirrors the previous endpoint-local `_determine_final_status` behavior
    while living in the core ingestion module.
    """
    if not results:
        # This case should ideally be handled earlier if no inputs were valid.
        return status.HTTP_400_BAD_REQUEST

    processing_results = results
    if not processing_results:
        return status.HTTP_200_OK

    if all(str(r.get("status", "")).lower() == "success" for r in processing_results):
        return status.HTTP_200_OK
    return status.HTTP_207_MULTI_STATUS


async def add_media_orchestrate(
    background_tasks: BackgroundTasks,
    form_data: Any,
    files: list[UploadFile] | None,
    db: MediaDatabase,
    current_user: Any,
    usage_log: Any,
    response: Any = None,
    request: Request | None = None,
) -> Any:
    """
    Orchestration helper for the `/media/add` endpoint.

    This function now owns the full ingestion and processing pipeline
    that previously lived in endpoint-local helpers, while resolving
    selected helpers from modular `media` exports so tests can
    monkeypatch `endpoints.media.*` patch points.
    """
    # Resolve helpers via the modular `media` exports when available so
    # tests that patch `endpoints.media.*` continue to work. Fall back
    # to core implementations when those exports are unavailable.
    try:
        from tldw_Server_API.app.api.v1.endpoints import (  # type: ignore
            media as media_mod,
        )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - ultra-minimal profiles
        media_mod = None  # type: ignore[assignment]

    _validate_inputs = validate_add_media_inputs
    _prepare_common_options = prepare_common_options
    _determine_final_status = determine_add_media_final_status

    from tldw_Server_API.app.api.v1.API_Deps.validations_deps import (  # type: ignore  # noqa: E501
        file_validator_instance as core_file_validator_instance,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing import (  # type: ignore  # noqa: E501
        input_sourcing as input_sourcing_mod,
    )

    CoreTempDirManager = input_sourcing_mod.TempDirManager

    if media_mod is not None:
        file_validator_instance = getattr(  # type: ignore[assignment]
            media_mod,
            "file_validator_instance",
            core_file_validator_instance,
        )
        TemplateClassifier = getattr(  # type: ignore[assignment]
            media_mod,
            "TemplateClassifier",
            None,
        )
        TempDirManagerCls = getattr(  # type: ignore[assignment]
            media_mod,
            "TempDirManager",
            CoreTempDirManager,
        )
    else:  # pragma: no cover - fallback for minimal profiles
        file_validator_instance = core_file_validator_instance  # type: ignore[assignment]
        TemplateClassifier = None  # type: ignore[assignment]
        TempDirManagerCls = CoreTempDirManager  # type: ignore[assignment]

    request_started_at = time.monotonic()
    request_outcome = "error"
    total_uploaded_bytes = 0
    rg_media_handle_id: str | None = None
    rg_governor, rg_policy_id, rg_policy, rg_entity = _resolve_media_budget_context(
        request=request,
        current_user=current_user,
    )
    rg_jobs_limit = _safe_int((rg_policy.get("jobs") or {}).get("max_concurrent"), 0)
    rg_daily_bytes_cap = _safe_int(
        (rg_policy.get(_MEDIA_INGESTION_BYTES_CATEGORY) or {}).get("daily_cap"),
        0,
    )

    # --- 1. Validation (form parsing handled by get_add_media_form) ---
    _validate_inputs(form_data.media_type, form_data.urls, files)

    # TEST_MODE diagnostics for auth and DB context
    try:
        if is_test_mode():
            _dbp = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
            logger.info(
                "TEST_MODE: add_media db_path={} user_id={}",
                _dbp,
                getattr(current_user, "id", "?"),
            )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass

    logger.info("Received request to add {} media.", form_data.media_type)
    try:
        usage_log.log_event(
            "media.add",
            tags=[str(form_data.media_type or "")],
            metadata={
                "has_urls": bool(form_data.urls),
                "files_count": len(files) if files else 0,
                "perform_analysis": bool(form_data.perform_analysis),
            },
        )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        # Usage logging must never break the endpoint path.
        pass

    # --- 1b. Resource Governor per-user concurrency budget ---
    if rg_governor is not None and RGRequest is not None and rg_entity and rg_jobs_limit > 0:
        try:
            rg_decision, rg_handle = await rg_governor.reserve(
                RGRequest(
                    entity=rg_entity,
                    categories={"jobs": {"units": 1}},
                    tags={
                        "policy_id": rg_policy_id,
                        "endpoint": request.url.path if request is not None else "/api/v1/media/add",
                    },
                ),
                op_id=f"media-add-jobs:{rg_entity}:{time.time_ns()}",
            )
            if not bool(getattr(rg_decision, "allowed", False)) or not rg_handle:
                cat_details = ((getattr(rg_decision, "details", {}) or {}).get("categories", {}) or {}).get(
                    "jobs"
                ) or {}
                retry_after = _safe_int(
                    getattr(rg_decision, "retry_after", None) or cat_details.get("retry_after"),
                    1,
                )
                headers = _build_ingestion_budget_headers(
                    category_details=cat_details,
                    retry_after=retry_after,
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Media ingestion concurrency limit reached.",
                    headers=headers or None,
                )
            rg_media_handle_id = str(rg_handle)
        except HTTPException:
            raise
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as rg_exc:
            logger.debug(
                "Media ingestion RG concurrency reserve skipped for entity={} policy_id={}: {}",
                rg_entity,
                rg_policy_id,
                rg_exc,
            )

    # --- 2. Database dependency / client_id guard ---
    if not hasattr(db, "client_id") or not db.client_id:
        logger.error("CRITICAL: Database instance dependency missing client_id.")
        db.client_id = settings.get("SERVER_CLIENT_ID", "SERVER_API_V1_FALLBACK")
        logger.warning("Manually set missing client_id on DB instance to: {}", db.client_id)

    results: list[dict[str, Any]] = []
    temp_dir_manager = TempDirManagerCls(  # type: ignore[call-arg]
        cleanup=not form_data.keep_original_file,
    )
    temp_dir_path: FilePath | None = None
    loop = asyncio.get_running_loop()

    try:
        # --- 3. Setup Temporary Directory ---
        with temp_dir_manager as temp_dir:
            temp_dir_path = FilePath(str(temp_dir))
            logger.info("Using temporary directory: {}", temp_dir_path)

            # --- 4. Save Uploaded Files ---
            # Restrict allowed extensions based on declared media_type to avoid mismatches
            allowed_ext_map = {
                "video": [
                    ".mp4",
                    ".mkv",
                    ".avi",
                    ".mov",
                    ".flv",
                    ".webm",
                    ".wmv",
                    ".mpg",
                    ".mpeg",
                ],
                "audio": [
                    ".mp3",
                    ".aac",
                    ".flac",
                    ".wav",
                    ".ogg",
                    ".m4a",
                    ".wma",
                ],
                "pdf": [".pdf"],
                "ebook": [".epub", ".mobi", ".azw"],
                "email": [".eml"]
                + ([".zip"] if getattr(form_data, "accept_archives", False) else [])
                + ([".mbox"] if getattr(form_data, "accept_mbox", False) else [])
                + ([".pst", ".ost"] if getattr(form_data, "accept_pst", False) else []),
                "json": [".json"],
                # For "document", allow a broad set; leave None to let validator handle.
            }
            allowed_exts = allowed_ext_map.get(str(form_data.media_type).lower())

            saved_files_info, file_save_errors = await input_sourcing_mod.save_uploaded_files(
                files or [],
                temp_dir_path,
                validator=file_validator_instance,
                allowed_extensions=allowed_exts,
                skip_archive_scanning=(
                    str(form_data.media_type).lower() == "email" and bool(getattr(form_data, "accept_archives", False))
                ),
            )

            upload_error_status = _classify_upload_error(file_save_errors)

            # Adapt file saving errors to the standard result format
            for err_info in file_save_errors:
                _emit_ingestion_validation_failure_metric(
                    reason=_classify_ingestion_validation_failure_reason(
                        str(err_info.get("error", "")),
                    ),
                    path_kind="upload",
                )
                results.append(
                    {
                        "status": "Error",
                        "input_ref": err_info.get("input_ref", "Unknown Upload"),
                        "processing_source": None,
                        "media_type": form_data.media_type,
                        "metadata": {},
                        "content": None,
                        "transcript": None,
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "summary": None,
                        "analysis_details": None,
                        "error": err_info.get("error", "File save failed."),
                        "warnings": None,
                        "db_id": None,
                        "db_message": "File saving failed.",
                        "message": "File saving failed.",
                    }
                )

            # --- Quota check for uploaded files and upload metrics ---
            try:
                if saved_files_info:
                    total_uploaded_bytes = 0
                    for pf in saved_files_info:
                        try:
                            # Use filesystem Path (not FastAPI's Path) to compute size.
                            total_uploaded_bytes += FilePath(str(pf["path"]).strip()).stat().st_size
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            pass
                    if total_uploaded_bytes > 0:
                        from tldw_Server_API.app.services.storage_quota_service import (  # type: ignore
                            get_storage_quota_service,
                        )

                        quota_service = get_storage_quota_service()
                        has_quota, info = await quota_service.check_quota(
                            current_user.id,
                            total_uploaded_bytes,
                            raise_on_exceed=False,
                        )
                        if not has_quota:
                            detail = (
                                "Storage quota exceeded. "
                                f"Current: {info['current_usage_mb']}MB, "
                                f"New: {info['new_size_mb']}MB, "
                                f"Quota: {info['quota_mb']}MB, "
                                f"Available: {info['available_mb']}MB"
                            )
                            raise HTTPException(
                                status_code=HTTP_413_TOO_LARGE,
                                detail=detail,
                            )
                        # Record upload metrics
                        try:
                            reg = get_metrics_registry()
                            reg.increment(
                                "uploads_total",
                                len(saved_files_info),
                                labels={
                                    "user_id": str(current_user.id),
                                    "media_type": form_data.media_type,
                                },
                            )
                            reg.increment(
                                "upload_bytes_total",
                                float(total_uploaded_bytes),
                                labels={
                                    "user_id": str(current_user.id),
                                    "media_type": form_data.media_type,
                                },
                            )
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            pass
            except HTTPException:
                raise
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as quota_err:
                logger.warning("Quota check failed (non-fatal): {}", quota_err)

            # --- Resource Governor per-user upload-bytes budget ---
            if (
                rg_governor is not None
                and RGRequest is not None
                and rg_entity
                and rg_daily_bytes_cap > 0
                and total_uploaded_bytes > 0
            ):
                try:
                    rg_decision = await rg_governor.check(
                        RGRequest(
                            entity=rg_entity,
                            categories={
                                _MEDIA_INGESTION_BYTES_CATEGORY: {
                                    "units": int(total_uploaded_bytes),
                                }
                            },
                            tags={
                                "policy_id": rg_policy_id,
                                "endpoint": request.url.path if request is not None else "/api/v1/media/add",
                            },
                        )
                    )
                    if not bool(getattr(rg_decision, "allowed", False)):
                        cat_details = (
                            (getattr(rg_decision, "details", {}) or {}).get(
                                "categories",
                                {},
                            )
                            or {}
                        ).get(_MEDIA_INGESTION_BYTES_CATEGORY) or {}
                        retry_after = _safe_int(
                            getattr(rg_decision, "retry_after", None) or cat_details.get("retry_after"),
                            1,
                        )
                        headers = _build_ingestion_budget_headers(
                            category_details=cat_details,
                            retry_after=retry_after,
                        )
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Daily ingestion size budget exceeded.",
                            headers=headers or None,
                        )

                    if ":" in rg_entity:
                        entity_scope, entity_value = rg_entity.split(":", 1)
                    else:
                        entity_scope, entity_value = "entity", rg_entity
                    request_id_part = request.headers.get("X-Request-ID", "") if request is not None else ""
                    if not request_id_part:
                        request_id_part = str(time.time_ns())
                    await _record_media_ingestion_bytes_ledger_entry(
                        entity_scope=entity_scope,
                        entity_value=entity_value,
                        units=int(total_uploaded_bytes),
                        op_id=(
                            f"media-ingestion-bytes:{entity_scope}:{entity_value}:"
                            f"{request_id_part}:{int(total_uploaded_bytes)}"
                        ),
                    )
                except HTTPException:
                    raise
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as rg_exc:
                    logger.debug(
                        "Media ingestion RG bytes check skipped for entity={} policy_id={} bytes={}: {}",
                        rg_entity,
                        rg_policy_id,
                        total_uploaded_bytes,
                        rg_exc,
                    )

            # --- 5. Prepare Inputs and Options ---
            uploaded_file_paths = [str(pf["path"]) for pf in saved_files_info]
            url_list = form_data.urls or []
            if url_list and form_data.media_type in ["video", "audio"]:
                try:
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import (
                        parse_and_expand_urls,
                    )

                    expanded_urls = parse_and_expand_urls(url_list)
                    if expanded_urls != url_list:
                        logger.info(
                            "Expanded playlist and shortcut URLs into {} concrete entries.",
                            len(expanded_urls),
                        )
                    url_list = expanded_urls
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as expand_err:
                    logger.warning(
                        "Failed to expand playlist URLs; continuing with originals: {}",
                        expand_err,
                    )
            all_valid_input_sources = url_list + uploaded_file_paths

            # Check if any valid sources remain after potential save errors
            if not all_valid_input_sources:
                if file_save_errors:
                    logger.warning("No valid inputs remaining after file handling errors.")
                    if upload_error_status and upload_error_status[0] == HTTP_413_TOO_LARGE:
                        raise HTTPException(
                            status_code=upload_error_status[0],
                            detail=upload_error_status[1],
                        )
                    if all(
                        _is_nonfatal_upload_validation_error(str(err_info.get("error", "") or ""))
                        for err_info in file_save_errors
                    ):
                        logger.info("Returning multi-status response for upload validation-only failures.")
                    elif upload_error_status:
                        raise HTTPException(
                            status_code=upload_error_status[0],
                            detail=upload_error_status[1],
                        )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="File upload failed; no valid media sources found to process.",
                        )
                else:
                    logger.error("No input URLs or successfully saved files found for /media/add.")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No valid media sources found to process.",
                    )

            first_url = (form_data.urls or [None])[0]
            first_filename = None
            try:
                if saved_files_info:
                    first_filename = saved_files_info[0]["original_filename"]
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                first_filename = None
            first_source = all_valid_input_sources[0] if all_valid_input_sources else first_url or first_filename

            # Prepare chunking options and auto-apply templates.
            chunking_options_dict, chunking_plan = resolve_chunking_options_and_plan(
                form_data,
                media_type=str(form_data.media_type) if form_data.media_type else None,
                source_name=str(first_source) if first_source else None,
            )

            # Apply explicit or auto-selected chunking templates for legacy/manual
            # requests. Auto requests already own the resolved planner options.
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (  # type: ignore  # noqa: E501
                    apply_chunking_template_if_any as _apply_tpl,
                )

                if chunking_plan is None:
                    chunking_options_dict = _apply_tpl(
                        form_data=form_data,
                        db=db,
                        chunking_options_dict=chunking_options_dict,
                        TemplateClassifier=TemplateClassifier,
                        first_url=first_url,
                        first_filename=first_filename,
                    )
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as auto_err:
                logger.warning("Auto-apply chunking template failed: {}", auto_err)

            # Even if not used directly here, preserve the existing call
            # to common options preparation to keep side effects/logging.
            _prepare_common_options(form_data, chunking_options_dict)

            # Map input sources back to original refs (URL or original filename)
            source_to_ref_map: dict[str, str] = {src: src for src in url_list}
            source_to_ref_map.update(
                {str(path): pf["original_filename"] for pf in saved_files_info if (path := pf.get("path"))}
            )
            for pf in saved_files_info:
                path = pf.get("path")
                if not path:
                    continue
                try:
                    resolved_path = resolve_safe_local_path(
                        FilePath(path),
                        temp_dir_path,
                    )
                    if resolved_path is None:
                        logger.warning(
                            "Skipping source path {} outside temp dir {}",
                            path,
                            temp_dir_path,
                        )
                        continue
                except OSError as exc:
                    logger.warning(
                        "Skipping source path {} due to resolve error: {}",
                        path,
                        exc,
                    )
                    continue
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(
                        "Skipping source path {} due to unexpected resolve error: {}",
                        path,
                        exc,
                    )
                    continue
                source_to_ref_map[str(resolved_path)] = pf["original_filename"]

            # --- 6. Process Media based on Type ---
            db_path_for_workers = db.db_path_str
            client_id_for_workers = db.client_id

            logger.info(
                "Processing {} items of type '{}'",
                len(all_valid_input_sources),
                form_data.media_type,
            )

            if form_data.media_type in ["video", "audio"]:
                batch_results = await process_batch_media(
                    media_type=str(form_data.media_type),
                    urls=url_list,
                    uploaded_file_paths=uploaded_file_paths,
                    source_to_ref_map=source_to_ref_map,
                    form_data=form_data,
                    chunk_options=chunking_options_dict,
                    loop=loop,
                    db_path=db_path_for_workers,
                    client_id=client_id_for_workers,
                    temp_dir=temp_dir_path,
                    user_id=int(current_user.id) if str(getattr(current_user, "id", "")).isdigit() else None,
                )
                results.extend(batch_results)
            else:
                # PDF / Document / Ebook / Email
                doc_like_concurrency = _document_like_concurrency_limit()
                semaphore = asyncio.Semaphore(doc_like_concurrency)

                async def _run_doc_item(source: str) -> dict[str, Any]:
                    async with semaphore:
                        return await process_document_like_item(
                            item_input_ref=source_to_ref_map.get(source, source),
                            processing_source=source,
                            media_type=form_data.media_type,
                            is_url=(source in url_list),
                            form_data=form_data,
                            chunk_options=chunking_options_dict,
                            temp_dir=temp_dir_path,
                            loop=loop,
                            db_path=db_path_for_workers,
                            client_id=client_id_for_workers,
                            user_id=(current_user.id if hasattr(current_user, "id") else None),
                        )

                tasks = [_run_doc_item(source) for source in all_valid_input_sources]
                individual_results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )
                for source, result in zip(all_valid_input_sources, individual_results):
                    if isinstance(result, Exception):
                        ref_info = source_to_ref_map.get(source, source)
                        input_ref = ref_info[0] if isinstance(ref_info, tuple) else ref_info
                        detail = getattr(result, "detail", None)
                        status_code = getattr(result, "status_code", None)
                        detail_text = str(detail) if detail else str(result)
                        if status_code:
                            error_msg = f"Processing failed (HTTP {status_code}): {detail_text}"
                        else:
                            error_msg = f"Processing failed: {type(result).__name__}: {detail_text}"
                        logger.error(
                            "Document-like processing failed for {}: {}",
                            source,
                            error_msg,
                            exc_info=True,
                        )
                        results.append(
                            {
                                "status": "Error",
                                "input_ref": input_ref,
                                "processing_source": source,
                                "media_type": form_data.media_type,
                                "metadata": {},
                                "content": None,
                                "transcript": None,
                                "segments": None,
                                "chunks": None,
                                "analysis": None,
                                "summary": None,
                                "analysis_details": None,
                                "error": error_msg,
                                "warnings": None,
                                "db_id": None,
                                "db_message": "DB operation skipped (processing failed).",
                                "message": None,
                                "media_uuid": None,
                            }
                        )
                    else:
                        results.append(result)

            # --- 6a. Store Original Files if Requested ---
            # For PDFs, documents, and ebooks, save originals to permanent storage when keep_original_file=True
            if form_data.keep_original_file and form_data.media_type in ["pdf", "document", "ebook"]:
                try:
                    from tldw_Server_API.app.core.Storage import get_storage_backend

                    storage = get_storage_backend()
                    user_id_str = str(current_user.id) if hasattr(current_user, "id") else "anonymous"

                    for result in results:
                        if result.get("status") != "Success" or not result.get("db_id"):
                            continue

                        media_id = result["db_id"]
                        input_ref = result.get("input_ref")
                        source_path = result.get("original_processing_source") or result.get("processing_source")

                        # Only store uploaded files, not URLs
                        if not source_path or input_ref in url_list:
                            continue

                        # Check if the source file exists and is within temp_dir
                        safe_source = resolve_safe_local_path(
                            FilePath(source_path),
                            temp_dir_path,
                        )
                        if safe_source is None:
                            logger.warning(
                                "Original file path rejected outside temp dir: {}",
                                source_path,
                            )
                            _ensure_warnings_list(result).append("Original file not stored: unsafe local path")
                            continue
                        source_file = safe_source
                        if not source_file.exists():
                            logger.warning(
                                "Original file not found for storage: {}",
                                source_path,
                            )
                            continue

                        try:
                            # Get file info
                            file_size = source_file.stat().st_size
                            original_filename = input_ref or source_file.name
                            if form_data.media_type == "pdf":
                                mime_type = "application/pdf"
                            elif form_data.media_type == "ebook":
                                mime_type = "application/epub+zip"
                            else:
                                mime_type = "application/octet-stream"

                            # Check storage quota before storing
                            try:
                                from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

                                quota_service = StorageQuotaService()
                                await quota_service.initialize()

                                has_quota, _info = await quota_service.check_quota(
                                    user_id=int(current_user.id) if hasattr(current_user, "id") else 0,
                                    new_bytes=file_size,
                                    raise_on_exceed=False,
                                )
                                if not has_quota:
                                    logger.warning(
                                        f"Storage quota exceeded for user {user_id_str}, "
                                        f"skipping original file storage for media_id={media_id}"
                                    )
                                    _ensure_warnings_list(result).append(
                                        "Original file not stored: storage quota exceeded"
                                    )
                                    continue
                            except ImportError:
                                # Quota service not available, proceed without check
                                pass
                            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as quota_err:
                                # Non-fatal quota check failure - log and proceed
                                logger.debug(f"Quota check failed, proceeding: {quota_err}")

                            # Read file bytes
                            handle = open_safe_local_path(source_file, temp_dir_path, mode="rb")
                            if handle is None:
                                raise InputError("Original file path rejected outside temp directory.")
                            try:
                                # Compute checksum without loading the entire file into memory
                                hasher = hashlib.sha256()
                                while True:
                                    chunk = handle.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    hasher.update(chunk)
                                checksum = hasher.hexdigest()
                                try:
                                    handle.seek(0)
                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                    raise InputError("Original file stream is not seekable for storage.") from None

                                # Store in permanent storage
                                storage_path = await storage.store(
                                    user_id=user_id_str,
                                    media_id=media_id,
                                    filename="original" + source_file.suffix,
                                    data=handle,
                                    mime_type=mime_type,
                                )
                            finally:
                                try:
                                    handle.close()
                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                    logger.debug("Failed to close original file handle for {}", source_file)

                            # Insert database record
                            db.insert_media_file(
                                media_id=media_id,
                                file_type="original",
                                storage_path=storage_path,
                                original_filename=original_filename,
                                file_size=file_size,
                                mime_type=mime_type,
                                checksum=checksum,
                            )

                            logger.info(f"Stored original file for media_id={media_id}: {storage_path}")
                            result["original_file_stored"] = True

                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as store_err:
                            logger.error(f"Failed to store original file for media_id={media_id}: {store_err}")
                            # Non-fatal - don't fail the entire ingestion
                            result["original_file_stored"] = False
                            _ensure_warnings_list(result).append(f"Failed to store original file: {store_err}")

                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as storage_init_err:
                    logger.error(f"Failed to initialize storage backend: {storage_init_err}")

        # --- 6b. Dual-write to Collections content_items ---
        try:
            sync_media_add_results_to_collections(
                results=results,
                form_data=form_data,
                current_user=current_user,
                db=db,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as collections_err:
            logger.warning("Collections dual-write step failed: {}", collections_err)

        # --- 7. Generate Embeddings if Requested ---
        try:
            await schedule_media_add_embeddings(
                results=results,
                form_data=form_data,
                background_tasks=background_tasks,
                db=db,
                current_user=current_user,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as embeddings_err:
            logger.warning(
                "Embeddings scheduling step failed: {}",
                embeddings_err,
            )

        # --- 8. Determine Final Status Code and Return Response ---
        final_status_code = _determine_final_status(results)
        request_outcome = _ingestion_request_outcome_from_status(final_status_code)

        # Special-case: Email container parent with children should return 200
        # even when some children include guardrail errors.
        try:
            if (
                isinstance(results, list)
                and len(results) == 1
                and isinstance(results[0], dict)
                and results[0].get("media_type") == "email"
                and results[0].get("status") == "Success"
                and isinstance(results[0].get("children"), list)
            ):
                final_status_code = status.HTTP_200_OK
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass

        log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
        logger.log(
            log_level,
            "Request finished with status {}. Results count: {}",
            final_status_code,
            len(results),
        )

        # TEST_MODE: emit diagnostic headers to assist tests
        try:
            if is_test_mode() and response is not None:
                try:
                    _dbp = getattr(db, "db_path_str", getattr(db, "db_path", "?"))
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    _dbp = "?"
                response.headers["X-TLDW-DB-Path"] = str(_dbp)
                response.headers["X-TLDW-Add-Results-Len"] = str(len(results))
                try:
                    ok_with_id = sum(
                        1 for r in results if isinstance(r, dict) and r.get("status") == "Success" and r.get("db_id")
                    )
                    response.headers["X-TLDW-Add-OK-With-Id"] = str(ok_with_id)
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    pass
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass

        return JSONResponse(
            status_code=final_status_code,
            content={"results": results},
        )

    except HTTPException as exc:
        request_outcome = "error"
        logger.warning(
            "HTTP Exception encountered in /media/add: Status={}, Detail={}",
            exc.status_code,
            exc.detail,
        )
        raise
    except OSError as os_err:
        request_outcome = "error"
        logger.error("OSError during /media/add setup: {}", os_err, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OS error during setup: {os_err}",
        ) from os_err
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as unexpected:
        request_outcome = "error"
        logger.error(
            "Unhandled exception in /media/add endpoint: {} - {}",
            type(unexpected).__name__,
            unexpected,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal error: {type(unexpected).__name__}",
        ) from unexpected
    finally:
        if rg_media_handle_id and rg_governor is not None:
            try:
                await rg_governor.release(rg_media_handle_id)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as rg_release_err:
                logger.debug(
                    "Media ingestion RG release failed for handle_id={}: {}",
                    rg_media_handle_id,
                    rg_release_err,
                )
        _emit_ingestion_request_metric(
            media_type=getattr(form_data, "media_type", None),
            outcome=request_outcome,
        )
        _emit_ingestion_processing_duration_metric(
            media_type=getattr(form_data, "media_type", None),
            processor="media_add_orchestrate",
            duration_seconds=time.monotonic() - request_started_at,
        )


async def add_media_persist(
    background_tasks: BackgroundTasks,
    form_data: Any,
    files: list[UploadFile] | None,
    db: MediaDatabase,
    current_user: Any,
    usage_log: Any,
    response: Any = None,
    request: Request | None = None,
) -> Any:
    """
    Persistence entry point used by the modular `media/add` endpoint.

    This delegates to `add_media_orchestrate` so future refactors can
    move more orchestration logic into this module while keeping the
    FastAPI route stable.
    """
    return await add_media_orchestrate(
        background_tasks=background_tasks,
        form_data=form_data,
        files=files,
        db=db,
        current_user=current_user,
        usage_log=usage_log,
        response=response,
        request=request,
    )


async def _resolve_auto_chunking_options_for_persistence(
    form_data: Any,
    *,
    media_type: Any,
    source_name: str,
    extracted_text: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return await async_resolve_chunking_options_and_plan(
            form_data,
            media_type=media_type,
            source_name=source_name,
            extracted_text=extracted_text,
        )
    except asyncio.CancelledError:
        raise
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        return None, None


async def persist_primary_av_item(
    *,
    process_result: dict[str, Any],
    form_data: Any,
    media_type: Any,
    original_input_ref: str,
    chunk_options: dict[str, Any] | None,
    path_kind: str,
    db_path: str,
    client_id: str,
    loop: Any,
    claims_context: dict[str, Any] | None,
) -> None:
    """
    Persist a single audio/video item processed by the /add orchestration.

    This helper lifts the DB write + claims persistence logic used by
    `_process_batch_media` so it can be reused from the core ingestion module.
    """
    # Match historical guard: only attempt DB writes when we have a DB path,
    # client id, and a successful or warning status.
    if not (db_path and client_id and process_result.get("status") in ["Success", "Warning"]):
        return

    # Use transcript as content for audio/video.
    content_for_db = process_result.get("transcript", process_result.get("content"))
    analysis_for_db = process_result.get("summary", process_result.get("analysis"))
    metadata_for_db = process_result.get("metadata", {}) or {}
    if not isinstance(metadata_for_db, dict):
        metadata_for_db = {}
    resolved_chunk_options = chunk_options
    if getattr(form_data, "chunking_mode", None) == "auto":
        auto_chunking_options, auto_chunking_plan = await _resolve_auto_chunking_options_for_persistence(
            form_data,
            media_type=media_type,
            source_name=str(original_input_ref or ""),
            extracted_text=content_for_db if isinstance(content_for_db, str) else None,
        )
        if auto_chunking_options is not None:
            resolved_chunk_options = auto_chunking_options
        if auto_chunking_plan:
            metadata_for_db = dict(metadata_for_db)
            metadata_for_db["chunking_plan"] = auto_chunking_plan
            process_result["metadata"] = metadata_for_db

    # Use the model reported by the processor if available, else fall back.
    transcription_model_used = metadata_for_db.get(
        "model",
        getattr(form_data, "transcription_model", None),
    )
    extracted_keywords = metadata_for_db.get("keywords", [])

    combined_keywords = set(getattr(form_data, "keywords", []) or [])
    if isinstance(extracted_keywords, list):
        combined_keywords.update(
            k.strip().lower() for k in extracted_keywords if k and isinstance(k, str) and k.strip()
        )
    final_keywords_list = sorted(combined_keywords)

    # Use original input ref for default title to match previous endpoint behavior.
    default_title = FilePath(str(original_input_ref)).stem if original_input_ref else "Untitled"

    title_for_db = metadata_for_db.get(
        "title",
        getattr(form_data, "title", None) or default_title,
    )
    author_for_db = metadata_for_db.get("author", getattr(form_data, "author", None))

    # When there is no content, mirror prior behavior: skip DB writes but
    # still persist claims (with media_id=None) and update db_message/db_id.
    if not content_for_db:
        process_result["db_message"] = "DB persistence skipped (no content)."
        process_result["db_id"] = None
        process_result["media_uuid"] = None
        await persist_claims_if_applicable(
            claims_context=claims_context,
            media_id=None,
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            process_result=process_result,
        )
        logger.warning(
            "Skipping DB persistence for {} due to missing content.",
            original_input_ref,
        )
        return

    try:
        logger.info("Attempting DB persistence for item: {}", process_result.get("input_ref"))

        # Build a safe metadata subset for persistence.
        safe_meta: dict[str, Any] = {}
        try:
            safe_meta = build_safe_metadata_subset(metadata_for_db)
            safe_meta = _merge_video_lite_summary_safe_metadata(
                safe_meta,
                process_result=process_result,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            safe_meta = {}

        safe_metadata_json: str | None = None
        try:
            if safe_meta:
                from tldw_Server_API.app.core.Utils.metadata_utils import (  # type: ignore
                    normalize_safe_metadata as _norm_sm,
                )

                try:
                    safe_meta = _norm_sm(safe_meta)
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    # Best-effort normalization; ignore failures here.
                    pass
                safe_metadata_json = json.dumps(safe_meta, ensure_ascii=False)
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            safe_metadata_json = None

        source_hash_for_db = None
        raw_source_hash = metadata_for_db.get("source_hash")
        if raw_source_hash is None:
            raw_source_hash = safe_meta.get("source_hash")
        if raw_source_hash is not None:
            raw_source_hash_str = str(raw_source_hash).strip()
            source_hash_for_db = raw_source_hash_str if raw_source_hash_str else None

        effective_chunk_options = resolved_chunk_options
        if effective_chunk_options is None and getattr(form_data, "perform_chunking", False):
            try:
                effective_chunk_options = prepare_chunking_options_dict(form_data)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as chunk_opts_err:
                logger.warning(
                    "Failed to derive chunk options during AV persistence for {}: {}",
                    original_input_ref,
                    chunk_opts_err,
                )

        # Build plaintext chunks for chunk-level FTS if chunking is requested.
        chunks_for_sql: list[dict[str, Any]] | None = None
        try:
            _opts = effective_chunk_options or {}
            if _opts:
                from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                    Chunker as _Chunker,
                )

                _ck = _Chunker()
                _flat = _ck.chunk_text_hierarchical_flat(
                    content_for_db,
                    method=_opts.get("method") or "sentences",
                    max_size=_opts.get("max_size") or 500,
                    overlap=_opts.get("overlap") or 50,
                )
                chunks_for_sql = []
                for _it in _flat:
                    _md = _it.get("metadata") or {}
                    _ctype = _ck.normalize_chunk_type(_md.get("chunk_type") or _md.get("paragraph_kind")) or "text"
                    _small: dict[str, Any] = {}
                    if _md.get("ancestry_titles"):
                        _small["ancestry_titles"] = _md.get("ancestry_titles")
                    if _md.get("section_path"):
                        _small["section_path"] = _md.get("section_path")
                    chunks_for_sql.append(
                        {
                            "text": _it.get("text", ""),
                            "start_char": _md.get("start_offset"),
                            "end_char": _md.get("end_offset"),
                            "chunk_type": _ctype,
                            "metadata": _small,
                        }
                    )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as chunk_err:
            logger.warning(
                "Failed to build AV chunks during persistence for {}: {}",
                original_input_ref,
                chunk_err,
            )
            chunks_for_sql = None

        # Merge processor-provided and analysis-derived extra chunks (VLM/OCR)
        # even if chunking was disabled or failed.
        try:
            extra_chunks_any = (process_result or {}).get("extra_chunks")
            derived_extra_chunks = _extract_analysis_extra_chunks_for_indexing(process_result)
            combined_extra_chunks: list[dict[str, Any]] = []
            if isinstance(extra_chunks_any, list) and extra_chunks_any:
                combined_extra_chunks.extend(extra_chunks_any)
            if derived_extra_chunks:
                combined_extra_chunks.extend(derived_extra_chunks)

            if combined_extra_chunks:
                if chunks_for_sql is None:
                    chunks_for_sql = []
                seen_extra_keys: set[tuple[Any, ...]] = set()
                for ec in combined_extra_chunks:
                    if not isinstance(ec, dict) or "text" not in ec:
                        continue
                    ec_md = ec.get("metadata") if isinstance(ec.get("metadata"), dict) else {}
                    dedupe_key = (
                        str(ec.get("chunk_type") or ""),
                        str(ec.get("text") or ""),
                        ec_md.get("source"),
                        ec_md.get("page"),
                        ec_md.get("label"),
                        ec_md.get("table_format"),
                    )
                    if dedupe_key in seen_extra_keys:
                        continue
                    seen_extra_keys.add(dedupe_key)
                    raw_chunk_type = ec.get("chunk_type") or "vlm"
                    try:
                        normalized_chunk_type = _ck.normalize_chunk_type(raw_chunk_type)  # type: ignore[name-defined]
                    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                        try:
                            from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                                Chunker as _Chunker,
                            )

                            normalized_chunk_type = _Chunker.normalize_chunk_type(raw_chunk_type)
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            normalized_chunk_type = raw_chunk_type
                    chunks_for_sql.append(
                        {
                            "text": ec.get("text", ""),
                            "start_char": ec.get("start_char"),
                            "end_char": ec.get("end_char"),
                            "chunk_type": normalized_chunk_type or raw_chunk_type,
                            "metadata": ec_md,
                        }
                    )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass

        db_add_kwargs = {
            "url": _normalize_dedupe_url_for_db(str(original_input_ref)),
            "title": title_for_db,
            "media_type": media_type,
            "content": content_for_db,
            "keywords": final_keywords_list,
            "prompt": getattr(form_data, "custom_prompt", None),
            "analysis_content": analysis_for_db,
            "safe_metadata": safe_metadata_json,
            "source_hash": source_hash_for_db,
            "transcription_model": transcription_model_used,
            "author": author_for_db,
            "overwrite": getattr(form_data, "overwrite_existing", False),
            "chunk_options": effective_chunk_options,
            "chunks": chunks_for_sql,
        }

        def _db_worker() -> Any:
            return _with_media_db_session(
                db_path=db_path,
                client_id=client_id,
                operation=lambda worker_db: _resolve_media_writer(worker_db).add_media_with_keywords(**db_add_kwargs),
            )

        media_id_result, media_uuid_result, db_message_result = await loop.run_in_executor(
            None,
            _db_worker,
        )

        process_result["db_id"] = media_id_result
        process_result["db_message"] = db_message_result
        process_result["media_uuid"] = media_uuid_result
        await _enforce_chunk_consistency_after_persist(
            result=process_result,
            form_data=form_data,
            media_type=media_type,
            path_kind=path_kind,
            processor=f"{_coerce_ingestion_label(media_type)}_primary_persist",
            expected_chunk_count=(len(chunks_for_sql) if isinstance(chunks_for_sql, list) else None),
            db_message=db_message_result,
            media_id=media_id_result,
            db_path=db_path,
            client_id=client_id,
            loop=loop,
        )
        _emit_ingestion_chunks_metric(
            media_type=media_type,
            chunk_method=(effective_chunk_options or {}).get("method"),
            chunk_count=len(chunks_for_sql) if isinstance(chunks_for_sql, list) else 0,
        )

        # Optionally persist a normalized STT transcript into the Transcripts table
        # for audio/video items when a transcription model is known.
        try:
            if media_type in ["audio", "video"] and media_id_result and transcription_model_used and content_for_db:
                artifact = process_result.get("normalized_stt")
                provider_name = None
                if isinstance(artifact, dict):
                    artifact = dict(artifact)
                    artifact_meta = artifact.get("metadata") or {}
                    if isinstance(artifact_meta, dict):
                        provider_name = artifact_meta.get("provider")
                else:
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
                        to_normalized_stt_artifact,
                    )
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
                        get_stt_provider_registry,
                    )

                    registry = get_stt_provider_registry()
                    provider_name, provider_model, _ = registry.resolve_provider_for_model(
                        str(transcription_model_used)
                    )
                    analysis_details = process_result.get("analysis_details") or {}
                    lang_for_stt = analysis_details.get("transcription_language")

                    artifact = to_normalized_stt_artifact(
                        text=str(content_for_db),
                        segments=process_result.get("segments"),
                        language=lang_for_stt,
                        provider=provider_name,
                        model=provider_model or str(transcription_model_used),
                    )
                serialized_artifact = json.dumps(artifact, default=str)
                artifact_meta = artifact.get("metadata") or {}
                whisper_model_value = str(artifact_meta.get("model") or transcription_model_used)

                def _upsert_worker() -> dict[str, Any]:
                    return _with_media_db_session(
                        db_path=db_path,
                        client_id=client_id,
                        operation=lambda db: upsert_transcript(
                            db_instance=db,
                            media_id=int(media_id_result),
                            transcription=serialized_artifact,
                            whisper_model=whisper_model_value,
                        ),
                    )

                write_payload = await loop.run_in_executor(None, _upsert_worker)
                with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                    emit_stt_run_write_total(
                        provider=provider_name,
                        write_result=(write_payload or {}).get("write_result", "created"),
                    )
                # Attach normalized artifact to the process_result for callers
                process_result["normalized_stt"] = artifact
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as stt_err:
            with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                emit_stt_run_write_total(
                    provider=locals().get("provider_name"),
                    write_result="failed",
                )
            logger.debug(
                "STT transcript upsert skipped/failed for {} (media_id={}): {}",
                original_input_ref,
                media_id_result,
                stt_err,
            )

        # Optionally persist VisualDocuments for eligible media types (currently PDFs via VLM summary).
        try:
            if media_type in ["pdf"] and media_id_result:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.visual_ingestion import (  # type: ignore
                    persist_visual_documents_from_analysis,
                )

                def _visual_docs_worker() -> int:
                    return persist_visual_documents_from_analysis(
                        db_path=db_path,
                        client_id=client_id,
                        media_id=int(media_id_result),
                        analysis_details=process_result.get("analysis_details") or {},
                    )

                created_visual_docs = await loop.run_in_executor(None, _visual_docs_worker)
                if created_visual_docs:
                    logger.info(
                        "Persisted {} VisualDocuments for media_id={} (input_ref={})",
                        created_visual_docs,
                        media_id_result,
                        original_input_ref,
                    )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as visual_err:
            logger.debug(
                "Visual RAG ingestion skipped/failed for {} (media_id={}): {}",
                original_input_ref,
                media_id_result,
                visual_err,
            )

        await persist_claims_if_applicable(
            claims_context=claims_context,
            media_id=process_result.get("db_id"),
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            process_result=process_result,
        )

        logger.info(
            "DB persistence result for {}: ID={}, UUID={}, Msg='{}'",
            original_input_ref,
            media_id_result,
            media_uuid_result,
            db_message_result,
        )

    except (DatabaseError, InputError, ConflictError) as db_err:
        logger.error(
            "Database operation failed for {}: {}",
            original_input_ref,
            db_err,
            exc_info=True,
        )
        process_result["status"] = "Warning"
        process_result["error"] = (process_result.get("error") or "") + f" | DB Error: {db_err}"
        _ensure_warnings_list(process_result).append(
            f"Database operation failed: {db_err}",
        )
        process_result["db_message"] = f"DB Error: {db_err}"
        process_result["db_id"] = None
        process_result["media_uuid"] = None
        await persist_claims_if_applicable(
            claims_context=claims_context,
            media_id=None,
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            process_result=process_result,
        )

    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            "Unexpected error during DB persistence for {}: {}",
            original_input_ref,
            exc,
            exc_info=True,
        )
        process_result["status"] = "Warning"
        process_result["error"] = process_result.get("error") or ""
        _ensure_warnings_list(process_result).append(
            f"Unexpected persistence error: {exc}",
        )
        process_result["db_message"] = f"Persistence Error: {type(exc).__name__}"
        process_result["db_id"] = None
        process_result["media_uuid"] = None
        await persist_claims_if_applicable(
            claims_context=claims_context,
            media_id=None,
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            process_result=process_result,
        )


async def process_batch_media(
    media_type: Any,
    urls: list[str],
    uploaded_file_paths: list[str],
    source_to_ref_map: dict[str, Any],
    form_data: Any,
    chunk_options: dict[str, Any] | None,
    loop: asyncio.AbstractEventLoop,
    db_path: str,
    client_id: str,
    temp_dir: FilePath,
    user_id: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """
    Core implementation of the audio/video batch processing helper used by `/media/add`.

    This function mirrors the previous endpoint-local `_process_batch_media`
    behavior while living in the core ingestion module for shared reuse.
    """
    combined_results: list[dict[str, Any]] = []
    all_processing_sources = urls + uploaded_file_paths
    items_to_process: list[str] = []
    reused_processing_results: list[dict[str, Any]] = []
    source_hash_by_ref: dict[str, list[str]] = {}
    source_hash_by_source: dict[str, str] = {}
    source_hash_column_available: bool | None = None

    def _is_cancelled() -> bool:
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            return False

    logger.debug(
        "Starting pre-check for {} {} items...",
        len(all_processing_sources),
        media_type,
    )

    # --- 1. Pre-check ---
    for source_path_or_url in all_processing_sources:
        input_ref_info = source_to_ref_map.get(source_path_or_url)
        input_ref = input_ref_info[0] if isinstance(input_ref_info, tuple) else input_ref_info
        if not input_ref:
            logger.error(
                "CRITICAL: Could not find original input reference for {}.",
                source_path_or_url,
            )
            input_ref = source_path_or_url

        identifier_for_check = str(input_ref)
        should_process = True
        existing_id: int | None = None
        reason = "Ready for processing."
        pre_check_warning: str | None = None
        source_hash: str | None = None
        is_url = isinstance(source_path_or_url, str) and source_path_or_url.lower().startswith(("http://", "https://"))
        if is_url:
            url_dedupe_candidates = media_dedupe_url_candidates(identifier_for_check)
            if not url_dedupe_candidates:
                url_dedupe_candidates = (identifier_for_check,)
            identifier_for_check = url_dedupe_candidates[0]
        else:
            url_dedupe_candidates = (identifier_for_check,)

        if is_url:
            try:
                from tldw_Server_API.app.core.Security.egress import (  # type: ignore
                    evaluate_url_policy,
                )

                block_override: bool | None = None
                if is_explicit_pytest_runtime() or env_flag_enabled("TESTING") or is_test_mode():
                    block_override = False

                policy_result = evaluate_url_policy(
                    str(source_path_or_url),
                    block_private_override=block_override,
                )
                if not getattr(policy_result, "allowed", False):
                    reason = policy_result.reason or "URL blocked by security policy"
                    with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                        get_metrics_registry().increment(
                            "security_ssrf_block_total",
                            1,
                        )
                    _emit_ingestion_validation_failure_metric(
                        reason="security_policy",
                        path_kind="url",
                    )
                    combined_results.append(
                        {
                            "status": "Error",
                            "input_ref": input_ref,
                            "processing_source": source_path_or_url,
                            "media_type": media_type,
                            "error": f"URL blocked by security policy: {reason}",
                            "metadata": None,
                            "content": None,
                            "transcript": None,
                            "segments": None,
                            "chunks": None,
                            "analysis": None,
                            "summary": None,
                            "analysis_details": None,
                            "warnings": None,
                            "db_id": None,
                            "db_message": "URL blocked by security policy.",
                        }
                    )
                    continue
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as policy_err:
                logger.warning(
                    "URL policy check failed for {}: {}",
                    source_path_or_url,
                    policy_err,
                )

        if not is_url:
            try:
                source_hash, hash_warning = _compute_source_hash_safe(
                    FilePath(source_path_or_url),
                    temp_dir,
                )
                if hash_warning:
                    pre_check_warning = (
                        hash_warning if not pre_check_warning else f"{pre_check_warning}; {hash_warning}"
                    )
                if source_hash and input_ref:
                    source_hash_by_ref.setdefault(str(input_ref), []).append(source_hash)
                    source_hash_by_source[str(source_path_or_url)] = source_hash
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as hash_err:
                logger.debug(
                    "Source hash computation failed for {}: {}",
                    source_path_or_url,
                    hash_err,
                )

        if not getattr(form_data, "overwrite_existing", False) and str(media_type) in ["video", "audio"]:
            try:
                model_for_check = getattr(form_data, "transcription_model", None)
                url_clause, url_params = _build_url_match_clause(
                    url_dedupe_candidates,
                    column="url",
                )
                url_clause_alias, url_params_alias = _build_url_match_clause(
                    url_dedupe_candidates,
                    column="m.url",
                )
                if source_hash and not is_url:

                    def _source_hash_precheck(db: MediaDatabase) -> Any:
                        nonlocal source_hash_column_available
                        if source_hash_column_available is None:
                            source_hash_column_available = _media_has_source_hash_column(db)
                        if source_hash_column_available:
                            pre_check_query_template = """
                                SELECT id
                                FROM Media
                                WHERE {url_clause}
                                  AND source_hash = ?
                                  AND is_trash = 0
                                  AND deleted = 0
                                LIMIT 1
                            """
                            pre_check_query = pre_check_query_template.format(url_clause=url_clause)  # nosec B608
                            cursor = db.execute_query(
                                pre_check_query,
                                (*url_params, source_hash),
                            )
                            existing_record = cursor.fetchone()
                            if not existing_record:
                                pre_check_query_template = """
                                    SELECT m.id
                                    FROM Media m
                                    JOIN DocumentVersions dv ON dv.media_id = m.id
                                    WHERE {url_clause_alias}
                                      AND m.is_trash = 0
                                      AND m.deleted = 0
                                      AND dv.deleted = 0
                                      AND dv.safe_metadata LIKE ?
                                    LIMIT 1
                                """
                                pre_check_query = pre_check_query_template.format(
                                    url_clause_alias=url_clause_alias
                                )  # nosec B608
                                hash_fragment = f'%"source_hash":"{source_hash}"%'
                                cursor = db.execute_query(
                                    pre_check_query,
                                    (*url_params_alias, hash_fragment),
                                )
                                existing_record = cursor.fetchone()
                        else:
                            pre_check_query_template = """
                                SELECT m.id
                                FROM Media m
                                JOIN DocumentVersions dv ON dv.media_id = m.id
                                WHERE {url_clause_alias}
                                  AND m.is_trash = 0
                                  AND m.deleted = 0
                                  AND dv.deleted = 0
                                  AND dv.safe_metadata LIKE ?
                                LIMIT 1
                            """
                            pre_check_query = pre_check_query_template.format(
                                url_clause_alias=url_clause_alias
                            )  # nosec B608
                            hash_fragment = f'%"source_hash":"{source_hash}"%'
                            cursor = db.execute_query(
                                pre_check_query,
                                (*url_params_alias, hash_fragment),
                            )
                            existing_record = cursor.fetchone()
                        return existing_record

                    existing_record = _with_media_db_session(
                        db_path=db_path,
                        client_id=client_id,
                        operation=_source_hash_precheck,
                    )

                    if existing_record:
                        existing_id = existing_record["id"]
                        should_process = False
                        reason = (
                            f"Media exists (ID: {existing_id}) with the same filename "
                            "and source hash. "
                            "Overwrite is False."
                        )
                    else:
                        should_process = True
                        reason = "Media not found with this filename and source hash."
                elif not is_url and not source_hash:
                    should_process = True
                    reason = "Local file pre-check skipped (no source hash available)."
                else:

                    def _url_precheck(db: MediaDatabase) -> Any:
                        pre_check_query_template = """
                                          SELECT id
                                          FROM Media
                                          WHERE {url_clause}
                                            AND is_trash = 0
                                            AND deleted = 0
                                          LIMIT 1
                                          """
                        pre_check_query = pre_check_query_template.format(url_clause=url_clause)  # nosec B608
                        cursor = db.execute_query(
                            pre_check_query,
                            url_params,
                        )
                        return cursor.fetchone()

                    existing_record = _with_media_db_session(
                        db_path=db_path,
                        client_id=client_id,
                        operation=_url_precheck,
                    )

                    if existing_record:
                        existing_id = existing_record["id"]
                        should_process = False
                        reason = (
                            f"Media exists (ID: {existing_id}) with the same URL/identifier "
                            "for this user. Overwrite is False."
                        )
                    else:
                        should_process = True
                        reason = "Media not found with this URL/identifier."
            except (DatabaseError, sqlite3.Error) as check_err:
                logger.error(
                    "DB pre-check (custom query) failed for {}: {}",
                    identifier_for_check,
                    check_err,
                    exc_info=True,
                )
                should_process, existing_id, reason = (
                    True,
                    None,
                    f"DB pre-check failed: {check_err}",
                )
                pre_check_warning = f"Database pre-check failed: {check_err}"
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as check_err:
                logger.error(
                    "Unexpected error during DB pre-check (custom query) for {}: {}",
                    identifier_for_check,
                    check_err,
                    exc_info=True,
                )
                should_process, existing_id, reason = (
                    True,
                    None,
                    f"Unexpected pre-check error: {check_err}",
                )
                pre_check_warning = f"Unexpected database pre-check error: {check_err}"
        else:
            should_process = True
            reason = "Overwrite requested or not applicable, proceeding regardless " "of existence."

        if should_process:
            reused_process_result = await _maybe_build_reused_transcript_process_result(
                media_type=str(media_type),
                original_input_ref=str(input_ref),
                source_path_or_url=str(source_path_or_url),
                source_hash=source_hash,
                form_data=form_data,
                chunk_options=chunk_options,
                loop=loop,
            )
            if reused_process_result is not None:
                if pre_check_warning:
                    _ensure_warnings_list(reused_process_result).append(pre_check_warning)
                reused_processing_results.append(reused_process_result)
                logger.info(
                    "Reused internal transcript source for {} without rerunning processor work.",
                    input_ref,
                )
                continue

        if not should_process:
            logger.info("Skipping processing for {}: {}", input_ref, reason)
            skipped_warnings = [pre_check_warning] if pre_check_warning else None
            skipped_result = {
                "status": "Skipped",
                "input_ref": input_ref,
                "processing_source": source_path_or_url,
                "media_type": media_type,
                "message": reason,
                "db_id": existing_id,
                "metadata": {},
                "content": None,
                "transcript": None,
                "segments": None,
                "chunks": None,
                "analysis": None,
                "summary": None,
                "analysis_details": None,
                "error": None,
                "warnings": skipped_warnings,
                "db_message": "Skipped processing, no DB action.",
            }
            combined_results.append(skipped_result)
        else:
            items_to_process.append(source_path_or_url)
            log_msg = f"Proceeding with processing for {input_ref}: {reason}"
            if pre_check_warning:
                log_msg += f" (Pre-check Warning: {pre_check_warning})"
                source_to_ref_map[source_path_or_url] = (input_ref, pre_check_warning)
            logger.info(log_msg)

    if not items_to_process and not reused_processing_results:
        logging.info("No items require processing after pre-checks.")
        return combined_results

    if _is_cancelled():
        for item in items_to_process:
            ref_info = source_to_ref_map.get(item)
            if isinstance(ref_info, tuple):
                err_input_ref = ref_info[0]
            elif isinstance(ref_info, str):
                err_input_ref = ref_info
            else:
                err_input_ref = item
            combined_results.append(
                {
                    "status": "Cancelled",
                    "input_ref": err_input_ref,
                    "processing_source": item,
                    "media_type": media_type,
                    "error": "Cancelled by user",
                    "metadata": {},
                    "content": None,
                    "transcript": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "summary": None,
                    "analysis_details": None,
                    "warnings": None,
                    "db_id": None,
                    "db_message": "DB operation skipped (cancelled).",
                    "media_uuid": None,
                }
            )
        return combined_results

    processing_output: dict[str, Any] | None = None
    try:
        if not items_to_process:
            processing_output = {
                "results": [],
                "errors_count": 0,
                "errors": [],
            }
        elif str(media_type) == "video":
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import (  # type: ignore  # noqa: E501
                process_videos,
            )

            target_callable: Callable[..., Any] = process_videos
            attach_chunk_options = _callable_accepts_keyword(
                target_callable,
                "chunk_options",
            )

            video_args = {
                "inputs": items_to_process,
                "temp_dir": str(temp_dir),
                "start_time": getattr(form_data, "start_time", None),
                "end_time": getattr(form_data, "end_time", None),
                "diarize": getattr(form_data, "diarize", False),
                "vad_use": getattr(form_data, "vad_use", False),
                "transcription_model": getattr(form_data, "transcription_model", None),
                "transcription_language": getattr(
                    form_data,
                    "transcription_language",
                    None,
                ),
                "custom_prompt": getattr(form_data, "custom_prompt", None),
                "system_prompt": getattr(form_data, "system_prompt", None),
                "perform_analysis": getattr(form_data, "perform_analysis", False),
                "perform_chunking": getattr(form_data, "perform_chunking", True),
                "chunk_method": chunk_options.get("method") if chunk_options else None,
                "max_chunk_size": (chunk_options.get("max_size") if chunk_options else 500),
                "chunk_overlap": (chunk_options.get("overlap") if chunk_options else 200),
                "use_adaptive_chunking": (chunk_options.get("adaptive", False) if chunk_options else False),
                "use_multi_level_chunking": (chunk_options.get("multi_level", False) if chunk_options else False),
                "chunk_language": (chunk_options.get("language") if chunk_options else None),
                "summarize_recursively": getattr(
                    form_data,
                    "summarize_recursively",
                    False,
                ),
                "api_name": (
                    getattr(form_data, "api_name", None) if getattr(form_data, "perform_analysis", False) else None
                ),
                "use_cookies": getattr(form_data, "use_cookies", False),
                "cookies": getattr(form_data, "cookies", None),
                "timestamp_option": getattr(form_data, "timestamp_option", None),
                "perform_confabulation_check": getattr(
                    form_data,
                    "perform_confabulation_check_of_analysis",
                    False,
                ),
                "keep_original": getattr(form_data, "keep_original_file", False),
                "user_id": user_id,
                "cancel_check": cancel_check,
            }
            if attach_chunk_options:
                video_args["chunk_options"] = chunk_options
            logger.debug(
                "Calling external process_videos with args including temp_dir: {}",
                list(video_args.keys()),
            )
            if asyncio.iscoroutinefunction(target_callable):
                processing_output = await target_callable(**video_args)
            else:
                target_func = functools.partial(target_callable, **video_args)
                processing_output = await loop.run_in_executor(None, target_func)

        elif str(media_type) == "audio":
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import (
                process_audio_files,  # type: ignore  # noqa: E501
            )

            target_callable: Callable[..., Any] = process_audio_files
            attach_chunk_options = _callable_accepts_keyword(
                target_callable,
                "chunk_options",
            )

            audio_args = {
                "inputs": items_to_process,
                "temp_dir": str(temp_dir),
                "transcription_model": getattr(
                    form_data,
                    "transcription_model",
                    None,
                ),
                "transcription_language": getattr(
                    form_data,
                    "transcription_language",
                    None,
                ),
                "perform_chunking": getattr(form_data, "perform_chunking", True),
                "chunk_method": chunk_options.get("method") if chunk_options else None,
                "max_chunk_size": (chunk_options.get("max_size") if chunk_options else 500),
                "chunk_overlap": (chunk_options.get("overlap") if chunk_options else 200),
                "use_adaptive_chunking": (chunk_options.get("adaptive", False) if chunk_options else False),
                "use_multi_level_chunking": (chunk_options.get("multi_level", False) if chunk_options else False),
                "chunk_language": (chunk_options.get("language") if chunk_options else None),
                "diarize": getattr(form_data, "diarize", False),
                "vad_use": getattr(form_data, "vad_use", False),
                "timestamp_option": getattr(form_data, "timestamp_option", None),
                "perform_analysis": getattr(form_data, "perform_analysis", False),
                "api_name": (
                    getattr(form_data, "api_name", None) if getattr(form_data, "perform_analysis", False) else None
                ),
                "custom_prompt_input": getattr(form_data, "custom_prompt", None),
                "system_prompt_input": getattr(form_data, "system_prompt", None),
                "summarize_recursively": getattr(
                    form_data,
                    "summarize_recursively",
                    False,
                ),
                "use_cookies": getattr(form_data, "use_cookies", False),
                "cookies": getattr(form_data, "cookies", None),
                "keep_original": getattr(form_data, "keep_original_file", False),
                "custom_title": getattr(form_data, "title", None),
                "author": getattr(form_data, "author", None),
                "user_id": user_id,
                "cancel_check": cancel_check,
            }
            if attach_chunk_options:
                audio_args["chunk_options"] = chunk_options
            logger.debug(
                "Calling external process_audio_files with args including temp_dir: {}",
                list(audio_args.keys()),
            )
            if asyncio.iscoroutinefunction(target_callable):
                processing_output = await target_callable(**audio_args)
            else:
                target_func = functools.partial(target_callable, **audio_args)
                processing_output = await loop.run_in_executor(None, target_func)
        else:
            raise ValueError(f"Invalid media type '{media_type}' for batch processing.")

    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as call_e:
        logger.error(
            "Error calling external batch processor for {}: {}",
            media_type,
            call_e,
            exc_info=True,
        )
        failed_items_results = []
        for item in items_to_process:
            ref_info = source_to_ref_map.get(item)
            if isinstance(ref_info, tuple):
                err_input_ref = ref_info[0]
            elif isinstance(ref_info, str):
                err_input_ref = ref_info
            else:
                err_input_ref = item
            failed_items_results.append(
                {
                    "status": "Error",
                    "input_ref": err_input_ref,
                    "processing_source": item,
                    "media_type": media_type,
                    "error": f"Failed to call processor: {type(call_e).__name__}",
                    "metadata": None,
                    "content": None,
                    "transcript": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "summary": None,
                    "analysis_details": None,
                    "warnings": None,
                    "db_id": None,
                    "db_message": None,
                }
            )
        combined_results.extend(failed_items_results)
        return combined_results

    final_batch_results: list[dict[str, Any]] = []
    processing_results_list: list[dict[str, Any]] = list(reused_processing_results)

    if processing_output and isinstance(processing_output.get("results"), list):
        processing_results_list.extend(processing_output["results"])
        if processing_output.get("errors_count", 0) > 0:
            logger.warning(
                "Batch {} processor reported errors: {}",
                media_type,
                processing_output.get("errors"),
            )
    else:
        logger.error(
            "Batch {} processor returned unexpected output: {}",
            media_type,
            processing_output,
        )
        return combined_results

    for process_result in processing_results_list:
        if not isinstance(process_result, dict):
            logger.error("Processor returned non-dict item: {}", process_result)
            malformed_result = {
                "status": "Error",
                "input_ref": "Unknown Input",
                "processing_source": "Unknown",
                "media_type": media_type,
                "error": "Processor returned invalid result format.",
                "metadata": None,
                "content": None,
                "transcript": None,
                "segments": None,
                "chunks": None,
                "analysis": None,
                "summary": None,
                "analysis_details": None,
                "warnings": None,
                "db_id": None,
                "db_message": None,
            }
            final_batch_results.append(malformed_result)
            continue

        input_ref = process_result.get("input_ref")
        processing_source = process_result.get("processing_source")
        if processing_source:
            ref_info = source_to_ref_map.get(str(processing_source))
            if isinstance(ref_info, tuple):
                original_input_ref = ref_info[0]
            elif isinstance(ref_info, str):
                original_input_ref = ref_info
            else:
                logger.warning(
                    "Could not find original input reference in source_to_ref_map "
                    "for processing_source: {}. Falling back.",
                    processing_source,
                )
                original_input_ref = process_result.get("input_ref") or processing_source or "Unknown Input"
        else:
            original_input_ref = process_result.get("input_ref") or "Unknown Input (Missing Source)"
            logger.warning(
                "Processing result missing 'processing_source'. Using fallback input_ref: {}",
                original_input_ref,
            )
            process_result["processing_source"] = str(original_input_ref) if original_input_ref else "Unknown"

        process_result["input_ref"] = str(original_input_ref) if original_input_ref else "Unknown"

        pre_check_info = source_to_ref_map.get(processing_source) if processing_source else None
        pre_check_warning_msg = None
        if isinstance(pre_check_info, tuple):
            pre_check_warning_msg = pre_check_info[1]
        if pre_check_warning_msg:
            _ensure_warnings_list(process_result).append(pre_check_warning_msg)

        source_hash = None
        if processing_source:
            source_hash = source_hash_by_source.get(str(processing_source))
        if not source_hash and original_input_ref:
            hash_list = source_hash_by_ref.get(str(original_input_ref))
            if hash_list:
                source_hash = hash_list.pop(0)
        if source_hash:
            metadata = process_result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("source_hash", source_hash)
            process_result["metadata"] = metadata

        path_kind = (
            "url"
            if isinstance(original_input_ref, str) and original_input_ref.lower().startswith(("http://", "https://"))
            else "upload"
        )
        _enforce_metadata_contract_on_result(
            result=process_result,
            media_type=media_type,
            form_data=form_data,
            path_kind=path_kind,
            processor=f"{_coerce_ingestion_label(media_type)}_batch_processor",
        )

        if _is_cancelled():
            process_result["status"] = "Cancelled"
            process_result["error"] = "Cancelled by user"
            process_result["db_message"] = "DB operation skipped (cancelled)."
            process_result["db_id"] = None
            process_result["media_uuid"] = None
            final_batch_results.append(process_result)
            continue

        claims_context: dict[str, Any] | None = None
        if process_result.get("status") in ("Success", "Warning"):
            try:
                claims_context = await extract_claims_if_requested(
                    process_result,
                    form_data,
                    loop,
                )
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as claims_err:
                logger.debug(
                    "Claim extraction skipped for {}: {}",
                    original_input_ref,
                    claims_err,
                )

        await persist_primary_av_item(
            process_result=process_result,
            form_data=form_data,
            media_type=media_type,
            original_input_ref=str(original_input_ref) if original_input_ref else "",
            chunk_options=chunk_options,
            path_kind=path_kind,
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            claims_context=claims_context,
        )

        await _register_reusable_transcript_source(
            media_type=str(media_type),
            original_input_ref=str(original_input_ref) if original_input_ref else "",
            source_hash=source_hash,
            process_result=process_result,
            user_id=user_id,
            form_data=form_data,
        )

        final_batch_results.append(process_result)

    combined_results.extend(final_batch_results)

    final_standardized_results: list[dict[str, Any]] = []
    processed_keys: set[tuple[str, str]] = set()

    for res in combined_results:
        input_ref = res.get("input_ref", "Unknown")
        processing_source = str(res.get("processing_source") or "")
        if input_ref != "Unknown" and processing_source:
            key = (input_ref, processing_source)
            if key in processed_keys:
                continue
            processed_keys.add(key)

        standardized = {
            "status": res.get("status", "Error"),
            "input_ref": input_ref,
            "processing_source": res.get("processing_source", "Unknown"),
            "media_type": res.get("media_type", media_type),
            "metadata": res.get("metadata", {}),
            "content": res.get("content", res.get("transcript")),
            "transcript": res.get("transcript"),
            "segments": res.get("segments"),
            "chunks": res.get("chunks"),
            "analysis": res.get("analysis", res.get("summary")),
            "summary": res.get("summary"),
            "analysis_details": res.get("analysis_details"),
            "claims": res.get("claims"),
            "claims_details": res.get("claims_details"),
            "error": res.get("error"),
            "warnings": res.get("warnings"),
            "db_id": res.get("db_id"),
            "db_message": res.get("db_message"),
            "message": res.get("message"),
            "media_uuid": res.get("media_uuid"),
        }
        if isinstance(standardized.get("warnings"), list) and not standardized["warnings"]:
            standardized["warnings"] = None

        final_standardized_results.append(standardized)

    return final_standardized_results


async def process_document_like_item(
    item_input_ref: str,
    processing_source: str,
    media_type: Any,
    is_url: bool,
    form_data: Any,
    chunk_options: dict[str, Any] | None,
    temp_dir: FilePath,
    loop: asyncio.AbstractEventLoop,
    db_path: str,
    client_id: str,
    user_id: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Core helper that handles download/prep, processing, and DB persistence for
    document-like items (PDF, generic documents/JSON, ebooks, and emails)
    used by the `/media/add` endpoint.

    This mirrors the behavior of the previous endpoint-local
    `_process_document_like_item` implementation while living in core.
    """
    # Resolve media-module exports when available so validator monkeypatching
    # (`endpoints.media.file_validator_instance`) continues to apply.
    try:  # type: ignore[assignment]
        from tldw_Server_API.app.api.v1.endpoints import (  # type: ignore
            media as _media_mod,
        )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - ultra-minimal profiles
        _media_mod = None  # type: ignore[assignment]

    final_result: dict[str, Any] = {
        "status": "Pending",
        "input_ref": item_input_ref,
        "processing_source": processing_source,
        "media_type": media_type,
        "metadata": {},
        "content": None,
        "segments": None,
        "chunks": None,
        "analysis": None,
        "summary": None,
        "analysis_details": None,
        "error": None,
        "warnings": [],
        "db_id": None,
        "db_message": None,
        "message": None,
    }
    claims_context: dict[str, Any] | None = None

    # --- 2. Download/Prepare File ---
    file_bytes: bytes | None = None
    processing_filepath: FilePath | None = None
    processing_filename: str | None = None
    downloaded_path: FilePath | None = None

    try:
        if is_url:
            logger.info("Downloading URL: {}", processing_source)
            # SSRF guard for individual item
            try:
                from tldw_Server_API.app.core.Security.url_validation import (  # type: ignore
                    assert_url_safe,
                )

                assert_url_safe(processing_source)
            except HTTPException as exc:
                # In TEST_MODE, treat host resolution failures as an
                # environment quirk so tests that stub downloads can
                # still execute the ingestion path.
                detail = getattr(exc, "detail", "")
                if is_test_mode() and isinstance(detail, str) and "Host could not be resolved" in detail:
                    logger.warning(
                        "TEST_MODE: ignoring host resolution error for {}: {}",
                        processing_source,
                        detail,
                    )
                else:
                    get_metrics_registry().increment(
                        "security_ssrf_block_total",
                        1,
                    )
                    _emit_ingestion_validation_failure_metric(
                        reason="security_policy",
                        path_kind="url",
                    )
                    raise

            from tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils import (  # type: ignore  # noqa: E501
                download_url_async as _core_download_url_async,
            )

            download_url_async = _core_download_url_async

            allowed_extensions = _allowed_url_extensions(media_type, form_data)
            check_extension = bool(allowed_extensions)
            downloaded_path = await download_url_async(
                client=None,
                url=processing_source,
                target_dir=temp_dir,
                allowed_extensions=allowed_extensions,
                check_extension=check_extension,
                media_type_key=None,
            )
            if downloaded_path and isinstance(downloaded_path, FilePath) and downloaded_path.exists():
                safe_downloaded_path = resolve_safe_local_path(
                    downloaded_path,
                    temp_dir,
                )
                if safe_downloaded_path is None:
                    raise FileNotFoundError("Downloaded file path rejected outside temp directory.")
                processing_filepath = safe_downloaded_path
                processing_filename = safe_downloaded_path.name
                _validate_downloaded_url_file(
                    downloaded_path=safe_downloaded_path,
                    processing_filename=processing_filename,
                    media_type=media_type,
                    form_data=form_data,
                    media_mod=_media_mod,
                    allowed_extensions=allowed_extensions,
                )

                if user_id is not None:
                    try:
                        from tldw_Server_API.app.services.storage_quota_service import (  # type: ignore  # noqa: E501
                            get_storage_quota_service,
                        )

                        quota_service = get_storage_quota_service()
                        size_bytes = downloaded_path.stat().st_size
                        has_quota, info = await quota_service.check_quota(
                            user_id,
                            size_bytes,
                            raise_on_exceed=False,
                        )
                        if not has_quota:
                            raise HTTPException(
                                status_code=HTTP_413_TOO_LARGE,
                                detail=(
                                    "Storage quota exceeded. Current: "
                                    f"{info['current_usage_mb']}MB, "
                                    f"New: {info['new_size_mb']}MB, "
                                    f"Quota: {info['quota_mb']}MB, "
                                    f"Available: {info['available_mb']}MB"
                                ),
                            )
                        try:
                            reg = get_metrics_registry()
                            reg.increment(
                                "uploads_total",
                                1,
                                labels={
                                    "user_id": str(user_id),
                                    "media_type": str(media_type),
                                },
                            )
                            reg.increment(
                                "upload_bytes_total",
                                float(size_bytes),
                                labels={
                                    "user_id": str(user_id),
                                    "media_type": str(media_type),
                                },
                            )
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            # Metrics must never break ingestion.
                            pass
                    except HTTPException:
                        raise
                    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as quota_err:
                        logger.warning(
                            "Per-item quota check failed (non-fatal): {}",
                            quota_err,
                        )

                # Read bytes for types that operate on raw content.
                if str(media_type) in {"pdf", "email"}:
                    async with open_safe_local_path_async(
                        processing_filepath,
                        temp_dir,
                        mode="rb",
                    ) as file_obj:
                        if file_obj is None:
                            raise FileNotFoundError(
                                "Downloaded file path rejected outside temp directory.",
                            )
                        file_bytes = await file_obj.read()

                final_result["processing_source"] = str(processing_filepath)
            else:
                raise OSError(
                    f"Download failed or did not return a valid path for {processing_source}",
                )
        else:
            path_obj = FilePath(processing_source)
            safe_path = resolve_safe_local_path(path_obj, temp_dir)
            if safe_path is None:
                raise FileNotFoundError(
                    f"Uploaded file path rejected outside temp directory: {processing_source}",
                )
            if not safe_path.is_file():
                raise FileNotFoundError(
                    f"Uploaded file path not found or is not a file: {processing_source}",
                )
            processing_filepath = safe_path
            processing_filename = safe_path.name

            if str(media_type) in {"pdf", "email"}:
                async with open_safe_local_path_async(
                    processing_filepath,
                    temp_dir,
                    mode="rb",
                ) as file_obj:
                    if file_obj is None:
                        raise FileNotFoundError(
                            "Uploaded file path rejected outside temp directory.",
                        )
                    file_bytes = await file_obj.read()

            final_result["processing_source"] = processing_source

    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as prep_err:
        error_detail = str(getattr(prep_err, "detail", prep_err))
        prep_error_type = type(prep_err).__name__
        temp_dir_exists: bool | None = None
        processing_source_exists: bool | None = None
        processing_filepath_exists: bool | None = None
        downloaded_path_exists: bool | None = None

        with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
            temp_dir_exists = FilePath(temp_dir).exists()
        if not is_url:
            with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                processing_source_exists = FilePath(processing_source).exists()
        if processing_filepath is not None:
            with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                processing_filepath_exists = processing_filepath.exists()
        if downloaded_path is not None:
            with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                downloaded_path_exists = downloaded_path.exists()

        logger.exception(
            "File preparation/download error for {}: {} ({}) | context: "
            "is_url={} temp_dir={} temp_dir_exists={} processing_source={} "
            "processing_source_exists={} downloaded_path={} downloaded_path_exists={} "
            "processing_filepath={} processing_filepath_exists={} processing_filename={}",
            item_input_ref,
            error_detail,
            prep_error_type,
            is_url,
            temp_dir,
            temp_dir_exists,
            processing_source,
            processing_source_exists,
            downloaded_path,
            downloaded_path_exists,
            processing_filepath,
            processing_filepath_exists,
            processing_filename,
        )
        validation_reason = _classify_ingestion_validation_failure_reason(error_detail)
        if validation_reason != "other":
            _emit_ingestion_validation_failure_metric(
                reason=validation_reason,
                path_kind="url" if is_url else "upload",
            )
        final_result.update(
            {
                "status": "Error",
                "error": f"File preparation/download failed: {error_detail}",
            },
        )
        if not final_result.get("warnings"):
            final_result["warnings"] = None
        return final_result

    # --- 3. Select and Call Processing Function ---
    process_result_dict: dict[str, Any] | None = None

    try:
        processing_func: Callable[..., Any] | None = None
        common_args: dict[str, Any] = {
            "title_override": getattr(form_data, "title", None),
            "author_override": getattr(form_data, "author", None),
            "keywords": getattr(form_data, "keywords", None),
            "perform_chunking": getattr(form_data, "perform_chunking", True),
            "chunk_options": chunk_options,
            "perform_analysis": getattr(form_data, "perform_analysis", True),
            "api_name": getattr(form_data, "api_name", None),
            "api_key": None,
            "custom_prompt": getattr(form_data, "custom_prompt", None),
            "system_prompt": getattr(form_data, "system_prompt", None),
            "summarize_recursively": getattr(
                form_data,
                "summarize_recursively",
                False,
            ),
        }
        specific_args: dict[str, Any] = {}
        run_in_executor = True

        media_type_str = str(media_type)

        if media_type_str == "pdf":
            if file_bytes is None:
                raise ValueError(
                    "PDF processing requires file bytes, but they were not read.",
                )
            from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib import (  # type: ignore  # noqa: E501
                process_pdf_task,
            )

            processing_func = process_pdf_task
            run_in_executor = False
            specific_args = {
                "file_bytes": file_bytes,
                "filename": processing_filename or item_input_ref,
                "parser": str(
                    getattr(form_data, "pdf_parsing_engine", "pymupdf4llm"),
                ),
                "enable_ocr": bool(getattr(form_data, "enable_ocr", False)),
                "ocr_backend": getattr(form_data, "ocr_backend", None),
                "ocr_lang": getattr(form_data, "ocr_lang", "eng"),
                "ocr_dpi": getattr(form_data, "ocr_dpi", 300),
                "ocr_mode": getattr(form_data, "ocr_mode", "fallback"),
                "ocr_min_page_text_chars": getattr(form_data, "ocr_min_page_text_chars", 40),
                "ocr_output_format": getattr(form_data, "ocr_output_format", None),
                "ocr_prompt_preset": getattr(form_data, "ocr_prompt_preset", None),
                "chunk_method": (chunk_options or {}).get("method"),
                "max_chunk_size": (chunk_options or {}).get("max_size"),
                "chunk_overlap": (chunk_options or {}).get("overlap"),
            }
            common_args.pop("chunk_options", None)

        elif media_type_str == "document":
            if processing_filepath is None:
                raise ValueError("Document processing requires a file path.")
            import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs  # type: ignore  # noqa: E501

            # Prefer endpoint-exported `media.process_document_content` so
            # tests can patch it; fall back to the core implementation.
            if _media_mod is not None:
                try:
                    processing_func = getattr(  # type: ignore[assignment]
                        _media_mod,
                        "process_document_content",
                        docs.process_document_content,
                    )
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
                    processing_func = docs.process_document_content
            else:  # pragma: no cover - minimal profiles
                processing_func = docs.process_document_content

            specific_args = {
                "doc_path": processing_filepath,
                "base_dir": temp_dir,
            }

        elif media_type_str == "json":
            if processing_filepath is None:
                raise ValueError("JSON processing requires a file path.")
            import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs  # type: ignore  # noqa: E501

            if _media_mod is not None:
                try:
                    processing_func = getattr(  # type: ignore[assignment]
                        _media_mod,
                        "process_document_content",
                        docs.process_document_content,
                    )
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
                    processing_func = docs.process_document_content
            else:  # pragma: no cover
                processing_func = docs.process_document_content

            specific_args = {
                "doc_path": processing_filepath,
                "base_dir": temp_dir,
            }

        elif media_type_str == "ebook":
            if processing_filepath is None:
                raise ValueError("Ebook processing requires a file path.")
            import tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib as books  # type: ignore  # noqa: E501

            def _sync_process_ebook_wrapper(**kwargs: Any) -> Any:
                return books.process_epub(**kwargs)

            processing_func = _sync_process_ebook_wrapper
            specific_args = {
                "file_path": str(processing_filepath),
                "extraction_method": "filtered",
                "base_dir": temp_dir,
            }
            custom_pattern = getattr(form_data, "custom_chapter_pattern", None)
            if custom_pattern:
                specific_args["custom_chapter_pattern"] = custom_pattern

        elif media_type_str == "email":
            if file_bytes is None and processing_filepath is not None:
                try:
                    async with open_safe_local_path_async(
                        processing_filepath,
                        temp_dir,
                        mode="rb",
                    ) as file_obj:
                        if file_obj is None:
                            raise FileNotFoundError(
                                "Email file path rejected outside temp directory.",
                            )
                        file_bytes = await file_obj.read()
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as read_err:
                    raise ValueError(
                        f"Email processing requires file bytes: {read_err}",
                    ) from read_err
            if file_bytes is None:
                raise ValueError(
                    "Email processing requires file bytes, but they were not available.",
                )

            import tldw_Server_API.app.core.Ingestion_Media_Processing.Email.Email_Processing_Lib as email_lib  # type: ignore  # noqa: E501

            name_lower = (processing_filename or item_input_ref).lower()
            if name_lower.endswith(".zip") and getattr(
                form_data,
                "accept_archives",
                False,
            ):
                processing_func = email_lib.process_eml_archive_bytes
                specific_args = {
                    "file_bytes": file_bytes,
                    "archive_name": processing_filename or item_input_ref,
                    "ingest_attachments": getattr(
                        form_data,
                        "ingest_attachments",
                        False,
                    ),
                    "max_depth": getattr(form_data, "max_depth", 2),
                }
            elif name_lower.endswith(".mbox") and getattr(
                form_data,
                "accept_mbox",
                False,
            ):
                processing_func = email_lib.process_mbox_bytes
                specific_args = {
                    "file_bytes": file_bytes,
                    "mbox_name": processing_filename or item_input_ref,
                    "ingest_attachments": getattr(
                        form_data,
                        "ingest_attachments",
                        False,
                    ),
                    "max_depth": getattr(form_data, "max_depth", 2),
                }
            elif (name_lower.endswith(".pst") or name_lower.endswith(".ost")) and getattr(  # noqa: E501
                form_data,
                "accept_pst",
                False,
            ):
                processing_func = email_lib.process_pst_bytes
                specific_args = {
                    "file_bytes": file_bytes,
                    "pst_name": processing_filename or item_input_ref,
                    "ingest_attachments": getattr(
                        form_data,
                        "ingest_attachments",
                        False,
                    ),
                    "max_depth": getattr(form_data, "max_depth", 2),
                }
            else:
                processing_func = email_lib.process_email_task
                specific_args = {
                    "file_bytes": file_bytes,
                    "filename": processing_filename or item_input_ref,
                    "ingest_attachments": getattr(
                        form_data,
                        "ingest_attachments",
                        False,
                    ),
                    "max_depth": getattr(form_data, "max_depth", 2),
                }

        else:
            raise NotImplementedError(
                f"Processor not implemented for media type: '{media_type}'",
            )

        all_args = {**common_args, **specific_args}
        final_args = all_args

        if processing_func is not None:
            func_name = getattr(
                processing_func,
                "__name__",
                str(processing_func),
            )
            logger.info(
                "Calling document-like processor '{}' for '{}' {}",
                func_name,
                item_input_ref,
                "in executor" if run_in_executor else "directly",
            )
            if run_in_executor:
                target_func = functools.partial(processing_func, **final_args)
                process_result_dict = await loop.run_in_executor(
                    None,
                    target_func,
                )
            else:
                process_result_dict = await processing_func(**final_args)

            # Email containers may return a list of children.
            if (
                media_type_str == "email"
                and isinstance(
                    process_result_dict,
                    list,
                )
                and (
                    getattr(form_data, "accept_archives", False)
                    or getattr(form_data, "accept_mbox", False)
                    or getattr(form_data, "accept_pst", False)
                )
            ):
                final_result.update(
                    {
                        "status": "Success",
                        "media_type": "email",
                        "content": None,
                        "metadata": {
                            "title": (getattr(form_data, "title", None) or (processing_filename or item_input_ref)),
                            "parser_used": "builtin-email",
                        },
                        "children": process_result_dict,
                    },
                )
                try:
                    archive_name = processing_filename or item_input_ref
                    archive_keyword: str | None = None
                    if archive_name:
                        lower_name = str(archive_name).lower()
                        if lower_name.endswith(".zip"):
                            archive_keyword = f"email_archive:{FilePath(archive_name).stem}"
                        elif lower_name.endswith(".mbox"):
                            archive_keyword = f"email_mbox:{FilePath(archive_name).stem}"
                        elif lower_name.endswith(".pst") or lower_name.endswith(
                            ".ost",
                        ):
                            archive_keyword = f"email_pst:{FilePath(archive_name).stem}"
                    if archive_keyword:
                        base_keywords: list[str] = []
                        try:
                            keywords_from_form = getattr(
                                form_data,
                                "keywords",
                                None,
                            )
                            if isinstance(keywords_from_form, list):
                                base_keywords = [
                                    str(keyword).strip().lower() for keyword in keywords_from_form if keyword
                                ]
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            base_keywords = []
                        merged = sorted(
                            set(
                                (final_result.get("keywords") or []) + base_keywords + [archive_keyword],
                            ),
                        )
                        final_result["keywords"] = merged
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    # Keyword enrichment is best-effort only.
                    pass
            else:
                if not isinstance(process_result_dict, dict):
                    raise TypeError(
                        f"Processor '{func_name}' returned non-dict: " f"{type(process_result_dict)}",
                    )
                if (
                    isinstance(process_result_dict, dict)
                    and process_result_dict.get("processing_source")
                    and final_result.get("processing_source")
                    and final_result.get("processing_source") != process_result_dict.get("processing_source")
                ):
                    final_result.setdefault(
                        "original_processing_source",
                        final_result.get("processing_source"),
                    )
                final_result.update(process_result_dict)
                final_result["status"] = process_result_dict.get(
                    "status",
                    "Error" if process_result_dict.get("error") else "Success",
                )

            proc_warnings: Any | None = None
            if isinstance(process_result_dict, dict):
                proc_warnings = process_result_dict.get("warnings")
            elif isinstance(process_result_dict, list):
                try:
                    aggregated: list[str] = []
                    for child in process_result_dict:
                        if isinstance(child, dict):
                            warnings_value = child.get("warnings")
                            if isinstance(warnings_value, list):
                                aggregated.extend(warnings_value)
                            elif warnings_value:
                                aggregated.append(str(warnings_value))
                    proc_warnings = aggregated or None
                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                    proc_warnings = None

            if isinstance(proc_warnings, list):
                if not isinstance(final_result.get("warnings"), list):
                    final_result["warnings"] = []
                final_result["warnings"].extend(proc_warnings)
            elif proc_warnings:
                if not isinstance(final_result.get("warnings"), list):
                    final_result["warnings"] = []
                final_result["warnings"].append(str(proc_warnings))
        else:
            final_result.update(
                {
                    "status": "Error",
                    "error": "No processing function selected.",
                },
            )

    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as proc_err:
        logger.error(
            "Error during processing call for {}: {}",
            item_input_ref,
            proc_err,
            exc_info=True,
        )
        final_result.update(
            {
                "status": "Error",
                "error": ("Processing error: " f"{type(proc_err).__name__}: {proc_err}"),
            },
        )

    # --- 4. Post-Processing DB Logic ---
    final_result.setdefault("status", "Error")
    final_result["input_ref"] = item_input_ref
    final_result["media_type"] = media_type

    if cancel_check is not None:
        try:
            if cancel_check():
                final_result.update(
                    {
                        "status": "Cancelled",
                        "error": "Cancelled by user",
                        "db_message": "DB operation skipped (cancelled).",
                        "db_id": None,
                        "media_uuid": None,
                    },
                )
                if not final_result.get("warnings"):
                    final_result["warnings"] = None
                return final_result
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
            pass

    _enforce_metadata_contract_on_result(
        result=final_result,
        media_type=media_type,
        form_data=form_data,
        path_kind="url" if is_url else "upload",
        processor=f"{_coerce_ingestion_label(media_type)}_document_like_processor",
    )

    if final_result.get("status") in ["Success", "Warning"]:
        try:
            claims_context = await extract_claims_if_requested(
                final_result,
                form_data,
                loop,
            )
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as claims_err:
            logger.debug(
                "Claim extraction failed for {}: {}",
                item_input_ref,
                claims_err,
            )
            _ensure_warnings_list(final_result).append(
                f"Claim extraction failed: {claims_err}",
            )
            claims_context = None
        await persist_doc_item_and_children(
            final_result=final_result,
            form_data=form_data,
            media_type=str(media_type),
            item_input_ref=item_input_ref,
            processing_filename=processing_filename,
            chunk_options=chunk_options,
            path_kind="url" if is_url else "upload",
            db_path=db_path,
            client_id=client_id,
            loop=loop,
            claims_context=claims_context,
        )
    else:
        final_result["db_message"] = "DB operation skipped (processing failed)."
        final_result["db_id"] = None
        final_result["media_uuid"] = None

    if not final_result.get("warnings"):
        final_result["warnings"] = None

    final_result["content"] = final_result.get("content")
    final_result["transcript"] = final_result.get("content")
    final_result["analysis"] = final_result.get("analysis")
    if "claims" not in final_result:
        final_result["claims"] = None
    if "claims_details" not in final_result:
        final_result["claims_details"] = None

    return final_result


async def persist_doc_item_and_children(
    *,
    final_result: dict[str, Any],
    form_data: Any,
    media_type: str,
    item_input_ref: str,
    processing_filename: str | None,
    chunk_options: dict[str, Any] | None,
    path_kind: str,
    db_path: str,
    client_id: str,
    loop: Any,
    claims_context: dict[str, Any] | None,
) -> None:
    """
    Persist a single document/email item (and any children) produced by the /add
    orchestration, mirroring the previous post-processing DB logic.
    """
    content_for_db = final_result.get("content", "")
    analysis_for_db = final_result.get("summary") or final_result.get("analysis")
    metadata_for_db = final_result.get("metadata", {}) or {}
    if not isinstance(metadata_for_db, dict):
        metadata_for_db = {}
    resolved_chunk_options = chunk_options
    if getattr(form_data, "chunking_mode", None) == "auto":
        auto_chunking_options, auto_chunking_plan = await _resolve_auto_chunking_options_for_persistence(
            form_data,
            media_type=media_type,
            source_name=str(item_input_ref or processing_filename or ""),
            extracted_text=content_for_db if isinstance(content_for_db, str) else None,
        )
        if auto_chunking_options is not None:
            resolved_chunk_options = auto_chunking_options
        if auto_chunking_plan:
            metadata_for_db = dict(metadata_for_db)
            metadata_for_db["chunking_plan"] = auto_chunking_plan
            final_result["metadata"] = metadata_for_db

    extracted_keywords = final_result.get("keywords", [])
    combined_keywords = set(getattr(form_data, "keywords", None) or [])
    if isinstance(extracted_keywords, list):
        combined_keywords.update(k.strip().lower() for k in extracted_keywords if isinstance(k, str) and k.strip())

    try:
        if media_type == "email":
            children = final_result.get("children")
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        child_keywords = child.get("keywords") or []
                        for kw in child_keywords:
                            if isinstance(kw, str) and kw.strip():
                                combined_keywords.add(kw.strip())
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        if media_type == "email" and getattr(form_data, "ingest_attachments", False):
            parent_msg_id = None
            try:
                parent_msg_id = ((metadata_for_db or {}).get("email") or {}).get("message_id")
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                parent_msg_id = None
            if parent_msg_id:
                combined_keywords.add(f"email_group:{str(parent_msg_id)}")
        if media_type == "email" and (
            getattr(form_data, "accept_archives", False)
            or getattr(form_data, "accept_mbox", False)
            or getattr(form_data, "accept_pst", False)
        ):
            try:
                arch_name = processing_filename or item_input_ref
                if arch_name:
                    lower = str(arch_name).lower()
                    if lower.endswith(".zip"):
                        arch_tag = f"email_archive:{FilePath(arch_name).stem}"
                        combined_keywords.add(arch_tag)
                    elif lower.endswith(".mbox"):
                        arch_tag = f"email_mbox:{FilePath(arch_name).stem}"
                        combined_keywords.add(arch_tag)
                    elif lower.endswith(".pst") or lower.endswith(".ost"):
                        pst_tag = f"email_pst:{FilePath(arch_name).stem}"
                        combined_keywords.add(pst_tag)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                pass
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
        pass

    final_keywords_list = sorted(combined_keywords)
    try:
        final_result["keywords"] = final_keywords_list
        logger.info(
            "Archive parent keywords set for {}: {}",
            item_input_ref,
            final_keywords_list,
        )
    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as kw_err:
        logger.warning("Failed to set parent keywords for {}: {}", item_input_ref, kw_err)

    model_used = metadata_for_db.get("parser_used", "Imported")
    if not model_used and media_type == "pdf":
        model_used = (final_result.get("analysis_details") or {}).get("parser", "Imported")

    default_title = FilePath(item_input_ref).stem if item_input_ref else "Untitled"

    title_for_db = getattr(form_data, "title", None) or metadata_for_db.get("title") or default_title
    author_for_db = metadata_for_db.get(
        "author",
        getattr(form_data, "author", None) or "Unknown",
    )

    if content_for_db:
        try:
            logger.info(
                "Attempting DB persistence for item: {} using user DB",
                item_input_ref,
            )
            safe_meta: dict[str, Any] = {}
            try:
                safe_meta = build_safe_metadata_subset(metadata_for_db)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                safe_meta = {}

            safe_metadata_json: str | None = None
            try:
                if safe_meta:
                    from tldw_Server_API.app.core.Utils.metadata_utils import (  # type: ignore
                        normalize_safe_metadata,
                    )

                    with contextlib.suppress(_PERSISTENCE_NONCRITICAL_EXCEPTIONS):
                        safe_meta = normalize_safe_metadata(safe_meta)
                    safe_metadata_json = json.dumps(safe_meta, ensure_ascii=False)
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                safe_metadata_json = None

            source_hash_for_db = None
            try:
                raw_source_hash = metadata_for_db.get("source_hash")
                if raw_source_hash is None:
                    raw_source_hash = safe_meta.get("source_hash")
                if raw_source_hash is not None:
                    raw_source_hash_str = str(raw_source_hash).strip()
                    source_hash_for_db = raw_source_hash_str if raw_source_hash_str else None
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                source_hash_for_db = None

            chunks_for_sql: list[dict[str, Any]] | None = None
            try:
                opts = resolved_chunk_options or {}
                if opts:
                    from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                        Chunker as _Chunker,
                    )

                    chunker = _Chunker()
                    flat_chunks = chunker.chunk_text_hierarchical_flat(
                        content_for_db,
                        method=opts.get("method") or "sentences",
                        max_size=opts.get("max_size") or 500,
                        overlap=opts.get("overlap") or 50,
                    )
                    chunks_for_sql = []
                    for item in flat_chunks:
                        meta = item.get("metadata") or {}
                        chunk_type = (
                            chunker.normalize_chunk_type(meta.get("chunk_type") or meta.get("paragraph_kind")) or "text"
                        )
                        small_meta: dict[str, Any] = {}
                        if meta.get("ancestry_titles"):
                            small_meta["ancestry_titles"] = meta.get("ancestry_titles")
                        if meta.get("section_path"):
                            small_meta["section_path"] = meta.get("section_path")
                        chunks_for_sql.append(
                            {
                                "text": item.get("text", ""),
                                "start_char": meta.get("start_offset"),
                                "end_char": meta.get("end_offset"),
                                "chunk_type": chunk_type,
                                "metadata": small_meta,
                            }
                        )
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                chunks_for_sql = None

            db_add_kwargs = {
                "url": _normalize_dedupe_url_for_db(item_input_ref),
                "title": title_for_db,
                "media_type": media_type,
                "content": content_for_db,
                "keywords": final_keywords_list,
                "prompt": getattr(form_data, "custom_prompt", None),
                "analysis_content": analysis_for_db,
                "safe_metadata": safe_metadata_json,
                "source_hash": source_hash_for_db,
                "transcription_model": model_used,
                "author": author_for_db,
                "overwrite": getattr(form_data, "overwrite_existing", False),
                "chunk_options": resolved_chunk_options,
                "chunks": chunks_for_sql,
            }

            def _db_worker() -> Any:
                def _persist_document(worker_db: MediaDatabase) -> Any:
                    media_writer = _resolve_media_writer(worker_db)
                    media_id_local, media_uuid_local, db_message_local = media_writer.add_media_with_keywords(
                        **db_add_kwargs
                    )
                    email_graph_local: dict[str, Any] | None = None
                    if media_type == "email" and media_id_local:
                        if _is_email_native_persist_enabled():
                            try:
                                email_graph_local = worker_db.upsert_email_message_graph(
                                    media_id=int(media_id_local),
                                    metadata=metadata_for_db if isinstance(metadata_for_db, dict) else {},
                                    body_text=str(content_for_db or ""),
                                    tenant_id=str(client_id),
                                    provider="upload",
                                    source_key=str(processing_filename or item_input_ref or "upload"),
                                    labels=(
                                        (metadata_for_db or {}).get("labels")
                                        if isinstance(metadata_for_db, dict)
                                        else None
                                    ),
                                )
                                _emit_email_native_persist_metric(
                                    path_kind="primary",
                                    outcome=(
                                        "success"
                                        if isinstance(email_graph_local, dict)
                                        and email_graph_local.get("email_message_id")
                                        else "noop"
                                    ),
                                )
                            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
                                logger.debug(
                                    "Email native upsert skipped due to non-fatal error (primary): {}",
                                    exc,
                                )
                                _emit_email_native_persist_metric(
                                    path_kind="primary",
                                    outcome="error",
                                )
                        else:
                            _emit_email_native_persist_metric(
                                path_kind="primary",
                                outcome="skipped_flag",
                            )
                    return (
                        media_id_local,
                        media_uuid_local,
                        db_message_local,
                        email_graph_local,
                    )

                return _with_media_db_session(
                    db_path=db_path,
                    client_id=client_id,
                    operation=_persist_document,
                )

            db_worker_result = await loop.run_in_executor(  # type: ignore[arg-type]
                None,
                _db_worker,
            )
            if isinstance(db_worker_result, tuple) and len(db_worker_result) == 4:
                (
                    media_id_result,
                    media_uuid_result,
                    db_message_result,
                    email_graph_result,
                ) = db_worker_result
            else:
                media_id_result, media_uuid_result, db_message_result = db_worker_result
                email_graph_result = None

            final_result["db_id"] = media_id_result
            final_result["db_message"] = db_message_result
            final_result["media_uuid"] = media_uuid_result
            if isinstance(email_graph_result, dict) and email_graph_result.get("email_message_id"):
                final_result["email_message_id"] = email_graph_result.get("email_message_id")
            await _enforce_chunk_consistency_after_persist(
                result=final_result,
                form_data=form_data,
                media_type=media_type,
                path_kind=path_kind,
                processor=f"{_coerce_ingestion_label(media_type)}_document_persist",
                expected_chunk_count=(len(chunks_for_sql) if isinstance(chunks_for_sql, list) else None),
                db_message=db_message_result,
                media_id=media_id_result,
                db_path=db_path,
                client_id=client_id,
                loop=loop,
            )
            _emit_ingestion_chunks_metric(
                media_type=media_type,
                chunk_method=(resolved_chunk_options or {}).get("method"),
                chunk_count=len(chunks_for_sql) if isinstance(chunks_for_sql, list) else 0,
            )
            logger.info(
                "DB persistence result for {}: ID={}, UUID={}, Msg='{}'",
                item_input_ref,
                media_id_result,
                media_uuid_result,
                db_message_result,
            )

            try:
                if media_type == "email" and getattr(form_data, "ingest_attachments", False):
                    children = final_result.get("children") or []
                    if isinstance(children, list) and children:
                        if any(isinstance(child, dict) and child.get("status") != "Success" for child in children):
                            final_result["child_db_results"] = None
                        else:
                            child_db_results: list[dict[str, Any]] = []
                            for child in children:
                                try:
                                    child_content = child.get("content")
                                    child_meta = child.get("metadata") or {}
                                    if not child_content:
                                        continue
                                    allowed_keys_child = {
                                        "title",
                                        "author",
                                        "doi",
                                        "pmid",
                                        "pmcid",
                                        "arxiv_id",
                                        "s2_paper_id",
                                        "url",
                                        "pdf_url",
                                        "pmc_url",
                                        "date",
                                        "year",
                                        "venue",
                                        "journal",
                                        "license",
                                        "license_url",
                                        "publisher",
                                        "source",
                                        "creators",
                                        "rights",
                                        "parent_media_uuid",
                                    }
                                    safe_child_meta = {
                                        key: value
                                        for key, value in child_meta.items()
                                        if key in allowed_keys_child
                                        and isinstance(value, (str, int, float, bool, list))
                                    }
                                    safe_child_meta["parent_media_uuid"] = media_uuid_result
                                    try:
                                        from tldw_Server_API.app.core.Utils.metadata_utils import (  # type: ignore
                                            normalize_safe_metadata,
                                        )

                                        safe_child_meta = normalize_safe_metadata(safe_child_meta)
                                        safe_child_meta_json = json.dumps(safe_child_meta, ensure_ascii=False)
                                    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                        safe_child_meta_json = None

                                    child_chunks_for_sql: list[dict[str, Any]] | None = None
                                    try:
                                        opts_child = resolved_chunk_options or {}
                                        if opts_child:
                                            from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                                                Chunker as _Chunker,
                                            )

                                            chunker_child = _Chunker()
                                            flat_child = chunker_child.chunk_text_hierarchical_flat(
                                                child_content,
                                                method=opts_child.get("method") or "sentences",
                                                max_size=opts_child.get("max_size") or 500,
                                                overlap=opts_child.get("overlap") or 50,
                                            )
                                            child_chunks_for_sql = []
                                            for item in flat_child:
                                                meta = item.get("metadata") or {}
                                                chunk_type = (
                                                    chunker_child.normalize_chunk_type(
                                                        meta.get("chunk_type") or meta.get("paragraph_kind")
                                                    )
                                                    or "text"
                                                )
                                                small_meta: dict[str, Any] = {}
                                                if meta.get("ancestry_titles"):
                                                    small_meta["ancestry_titles"] = meta.get("ancestry_titles")
                                                if meta.get("section_path"):
                                                    small_meta["section_path"] = meta.get("section_path")
                                                child_chunks_for_sql.append(
                                                    {
                                                        "text": item.get("text", ""),
                                                        "start_char": meta.get("start_offset"),
                                                        "end_char": meta.get("end_offset"),
                                                        "chunk_type": chunk_type,
                                                        "metadata": small_meta,
                                                    }
                                                )
                                    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                        child_chunks_for_sql = None

                                    child_title = (
                                        getattr(form_data, "title", None)
                                        or child_meta.get("title")
                                        or f"{FilePath(item_input_ref).stem} (child)"
                                    )
                                    child_author = child_meta.get(
                                        "author",
                                        getattr(form_data, "author", None) or "Unknown",
                                    )
                                    child_url = (
                                        f"{item_input_ref}::child::" f"{child_meta.get('filename') or child_title}"
                                    )

                                    def _db_child_worker(
                                        child_url: str = child_url,
                                        child_title: str = child_title,
                                        child_content: str = child_content,
                                        child_metadata_local: dict[str, Any] = (
                                            child_meta if isinstance(child_meta, dict) else {}
                                        ),
                                        final_keywords: list[str] = final_keywords_list,
                                        safe_child_meta_json_local: str | None = safe_child_meta_json,
                                        model_used_local: str | None = model_used,
                                        child_author_local: str = child_author,
                                        child_chunks_for_sql_local: list[dict[str, Any]] | None = child_chunks_for_sql,
                                        chunk_options_local: dict[str, Any] | None = resolved_chunk_options,
                                        form_data_local: Any = form_data,
                                        media_type_local: str = media_type,
                                        client_id_local: str = client_id,
                                        db_path_local: str = db_path,
                                    ) -> Any:
                                        def _persist_child(worker_db: MediaDatabase) -> Any:
                                            media_writer = _resolve_media_writer(worker_db)
                                            child_id_local, child_uuid_local, child_msg_local = (
                                                media_writer.add_media_with_keywords(
                                                    url=_normalize_dedupe_url_for_db(child_url),
                                                    title=child_title,
                                                    media_type=media_type_local,
                                                    content=child_content,
                                                    keywords=final_keywords,
                                                    prompt=getattr(form_data_local, "custom_prompt", None),
                                                    analysis_content=None,
                                                    safe_metadata=safe_child_meta_json_local,
                                                    transcription_model=model_used_local,
                                                    author=child_author_local,
                                                    overwrite=getattr(form_data_local, "overwrite_existing", False),
                                                    chunk_options=chunk_options_local,
                                                    chunks=child_chunks_for_sql_local,
                                                )
                                            )
                                            if media_type_local == "email" and child_id_local:
                                                if _is_email_native_persist_enabled():
                                                    try:
                                                        child_email_graph_local = worker_db.upsert_email_message_graph(
                                                            media_id=int(child_id_local),
                                                            metadata=child_metadata_local,
                                                            body_text=str(child_content or ""),
                                                            tenant_id=str(client_id_local),
                                                            provider="upload",
                                                            source_key=str(child_url),
                                                        )
                                                        _emit_email_native_persist_metric(
                                                            path_kind="attachment_child",
                                                            outcome=(
                                                                "success"
                                                                if isinstance(child_email_graph_local, dict)
                                                                and child_email_graph_local.get("email_message_id")
                                                                else "noop"
                                                            ),
                                                        )
                                                    except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
                                                        logger.debug(
                                                            "Email native upsert skipped due to non-fatal error (attachment_child): {}",
                                                            exc,
                                                        )
                                                        _emit_email_native_persist_metric(
                                                            path_kind="attachment_child",
                                                            outcome="error",
                                                        )
                                                else:
                                                    _emit_email_native_persist_metric(
                                                        path_kind="attachment_child",
                                                        outcome="skipped_flag",
                                                    )
                                            return child_id_local, child_uuid_local, child_msg_local

                                        return _with_media_db_session(
                                            db_path=db_path_local,
                                            client_id=client_id_local,
                                            operation=_persist_child,
                                        )

                                    (
                                        child_id,
                                        child_uuid,
                                        child_msg,
                                    ) = await loop.run_in_executor(  # type: ignore[arg-type]
                                        None,
                                        _db_child_worker,
                                    )
                                    await _enforce_chunk_consistency_after_persist(
                                        result=final_result,
                                        form_data=form_data,
                                        media_type=media_type,
                                        path_kind=path_kind,
                                        processor="email_child_attachment_persist",
                                        expected_chunk_count=(
                                            len(child_chunks_for_sql)
                                            if isinstance(child_chunks_for_sql, list)
                                            else None
                                        ),
                                        db_message=child_msg,
                                        media_id=child_id,
                                        db_path=db_path,
                                        client_id=client_id,
                                        loop=loop,
                                    )
                                    _emit_ingestion_chunks_metric(
                                        media_type=media_type,
                                        chunk_method=(resolved_chunk_options or {}).get("method"),
                                        chunk_count=(
                                            len(child_chunks_for_sql) if isinstance(child_chunks_for_sql, list) else 0
                                        ),
                                    )
                                    child_db_results.append(
                                        {
                                            "db_id": child_id,
                                            "media_uuid": child_uuid,
                                            "message": child_msg,
                                            "title": child_title,
                                        }
                                    )
                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as child_db_err:
                                    logger.warning(
                                        "Child email persistence failed: {}",
                                        child_db_err,
                                    )
                            if child_db_results:
                                final_result["child_db_results"] = child_db_results
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                pass

        except (DatabaseError, InputError, ConflictError) as db_err:
            logger.error(
                "Database operation failed for {}: {}",
                item_input_ref,
                db_err,
                exc_info=True,
            )
            final_result["status"] = "Warning"
            final_result["error"] = (final_result.get("error") or "") + f" | DB Error: {db_err}"
            if not isinstance(final_result.get("warnings"), list):
                final_result["warnings"] = []
            final_result["warnings"].append(f"Database operation failed: {db_err}")
            final_result["db_message"] = f"DB Error: {db_err}"
            final_result["db_id"] = None
            final_result["media_uuid"] = None
        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(
                "Unexpected error during DB persistence for {}: {}",
                item_input_ref,
                exc,
                exc_info=True,
            )
            final_result["status"] = "Warning"
            final_result["error"] = final_result.get("error") or ""
            if not isinstance(final_result.get("warnings"), list):
                final_result["warnings"] = []
            final_result["warnings"].append(f"Unexpected persistence error: {exc}")
            final_result["db_message"] = f"Persistence Error: {type(exc).__name__}"
            final_result["db_id"] = None
            final_result["media_uuid"] = None
    else:
        persisted_any_children = False
        if media_type == "email" and (
            getattr(form_data, "accept_archives", False)
            or getattr(form_data, "accept_mbox", False)
            or getattr(form_data, "accept_pst", False)
        ):
            try:
                children = final_result.get("children") or []
                if isinstance(children, list) and children:
                    if any(isinstance(child, dict) and child.get("status") != "Success" for child in children):
                        final_result["child_db_results"] = None
                        persisted_any_children = False
                    else:
                        child_db_results = []
                        for child in children:
                            try:
                                child_content = child.get("content")
                                child_meta = child.get("metadata") or {}
                                if not child_content:
                                    continue
                                allowed_keys_child = {
                                    "title",
                                    "author",
                                    "doi",
                                    "pmid",
                                    "pmcid",
                                    "arxiv_id",
                                    "s2_paper_id",
                                    "url",
                                    "pdf_url",
                                    "pmc_url",
                                    "date",
                                    "year",
                                    "venue",
                                    "journal",
                                    "license",
                                    "license_url",
                                    "publisher",
                                    "source",
                                    "creators",
                                    "rights",
                                }
                                safe_child_meta = {
                                    key: value
                                    for key, value in child_meta.items()
                                    if key in allowed_keys_child and isinstance(value, (str, int, float, bool, list))
                                }
                                safe_child_meta_json: str | None = None
                                try:
                                    from tldw_Server_API.app.core.Utils.metadata_utils import (  # type: ignore
                                        normalize_safe_metadata,
                                    )

                                    safe_child_meta = normalize_safe_metadata(safe_child_meta)
                                    safe_child_meta_json = json.dumps(safe_child_meta, ensure_ascii=False)
                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                    pass

                                child_chunks_for_sql: list[dict[str, Any]] | None = None
                                try:
                                    opts_child = resolved_chunk_options or {}
                                    if opts_child:
                                        from tldw_Server_API.app.core.Chunking.chunker import (  # type: ignore
                                            Chunker as _Chunker,
                                        )

                                        chunker_child = _Chunker()
                                        flat_child = chunker_child.chunk_text_hierarchical_flat(
                                            child_content,
                                            method=opts_child.get("method") or "sentences",
                                            max_size=opts_child.get("max_size") or 500,
                                            overlap=opts_child.get("overlap") or 50,
                                        )
                                        child_chunks_for_sql = []
                                        for item in flat_child:
                                            meta = item.get("metadata") or {}
                                            chunk_type = (
                                                chunker_child.normalize_chunk_type(
                                                    meta.get("chunk_type") or meta.get("paragraph_kind")
                                                )
                                                or "text"
                                            )
                                            small_meta: dict[str, Any] = {}
                                            if meta.get("ancestry_titles"):
                                                small_meta["ancestry_titles"] = meta.get("ancestry_titles")
                                            if meta.get("section_path"):
                                                small_meta["section_path"] = meta.get("section_path")
                                            child_chunks_for_sql.append(
                                                {
                                                    "text": item.get("text", ""),
                                                    "start_char": meta.get("start_offset"),
                                                    "end_char": meta.get("end_offset"),
                                                    "chunk_type": chunk_type,
                                                    "metadata": small_meta,
                                                }
                                            )
                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                                    child_chunks_for_sql = None

                                child_title = (
                                    getattr(form_data, "title", None)
                                    or child_meta.get("title")
                                    or f"{FilePath(item_input_ref).stem} (archive child)"
                                )
                                child_author = child_meta.get(
                                    "author",
                                    getattr(form_data, "author", None) or "Unknown",
                                )
                                child_url = (
                                    f"{item_input_ref}::archive::" f"{child_meta.get('filename') or child_title}"
                                )

                                def _db_child_arch_worker(
                                    child_url_local: str = child_url,
                                    child_title_local: str = child_title,
                                    child_content_local: str = child_content,
                                    child_metadata_local: dict[str, Any] = (
                                        child_meta if isinstance(child_meta, dict) else {}
                                    ),
                                    final_keywords_local: list[str] = final_keywords_list,
                                    safe_child_meta_json_local: str | None = safe_child_meta_json,
                                    model_used_local: str | None = model_used,
                                    child_author_local: str = child_author,
                                    child_chunks_for_sql_local: list[dict[str, Any]] | None = child_chunks_for_sql,
                                    media_type_local: str = media_type,
                                    form_data_local: Any = form_data,
                                    chunk_options_local: dict[str, Any] | None = resolved_chunk_options,
                                    db_path_local: str = db_path,
                                    client_id_local: str = client_id,
                                ) -> Any:
                                    def _persist_archive_child(worker_db: MediaDatabase) -> Any:
                                        media_writer = _resolve_media_writer(worker_db)
                                        child_id_local, child_uuid_local, child_msg_local = (
                                            media_writer.add_media_with_keywords(
                                                url=_normalize_dedupe_url_for_db(child_url_local),
                                                title=child_title_local,
                                                media_type=media_type_local,
                                                content=child_content_local,
                                                keywords=final_keywords_local,
                                                prompt=getattr(form_data_local, "custom_prompt", None),
                                                analysis_content=None,
                                                safe_metadata=safe_child_meta_json_local,
                                                transcription_model=model_used_local,
                                                author=child_author_local,
                                                overwrite=getattr(form_data_local, "overwrite_existing", False),
                                                chunk_options=chunk_options_local,
                                                chunks=child_chunks_for_sql_local,
                                            )
                                        )
                                        if media_type_local == "email" and child_id_local:
                                            if _is_email_native_persist_enabled():
                                                try:
                                                    child_email_graph_local = worker_db.upsert_email_message_graph(
                                                        media_id=int(child_id_local),
                                                        metadata=child_metadata_local,
                                                        body_text=str(child_content_local or ""),
                                                        tenant_id=str(client_id_local),
                                                        provider="upload",
                                                        source_key=str(child_url_local),
                                                    )
                                                    _emit_email_native_persist_metric(
                                                        path_kind="archive_child",
                                                        outcome=(
                                                            "success"
                                                            if isinstance(child_email_graph_local, dict)
                                                            and child_email_graph_local.get("email_message_id")
                                                            else "noop"
                                                        ),
                                                    )
                                                except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as exc:
                                                    logger.debug(
                                                        "Email native upsert skipped due to non-fatal error (archive_child): {}",
                                                        exc,
                                                    )
                                                    _emit_email_native_persist_metric(
                                                        path_kind="archive_child",
                                                        outcome="error",
                                                    )
                                            else:
                                                _emit_email_native_persist_metric(
                                                    path_kind="archive_child",
                                                    outcome="skipped_flag",
                                                )
                                        return child_id_local, child_uuid_local, child_msg_local

                                    return _with_media_db_session(
                                        db_path=db_path_local,
                                        client_id=client_id_local,
                                        operation=_persist_archive_child,
                                    )

                                (
                                    child_id,
                                    child_uuid,
                                    child_msg,
                                ) = await loop.run_in_executor(  # type: ignore[arg-type]
                                    None,
                                    _db_child_arch_worker,
                                )
                                await _enforce_chunk_consistency_after_persist(
                                    result=final_result,
                                    form_data=form_data,
                                    media_type=media_type,
                                    path_kind=path_kind,
                                    processor="email_child_archive_persist",
                                    expected_chunk_count=(
                                        len(child_chunks_for_sql) if isinstance(child_chunks_for_sql, list) else None
                                    ),
                                    db_message=child_msg,
                                    media_id=child_id,
                                    db_path=db_path,
                                    client_id=client_id,
                                    loop=loop,
                                )
                                _emit_ingestion_chunks_metric(
                                    media_type=media_type,
                                    chunk_method=(resolved_chunk_options or {}).get("method"),
                                    chunk_count=(
                                        len(child_chunks_for_sql) if isinstance(child_chunks_for_sql, list) else 0
                                    ),
                                )
                                child_db_results.append(
                                    {
                                        "db_id": child_id,
                                        "media_uuid": child_uuid,
                                        "message": child_msg,
                                        "title": child_title,
                                    }
                                )
                                persisted_any_children = True
                            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS as child_db_err:
                                logger.warning(
                                    "Archive child email persistence failed: {}",
                                    child_db_err,
                                )
                        try:
                            if child_db_results:
                                final_result["child_db_results"] = child_db_results
                        except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                            pass
            except _PERSISTENCE_NONCRITICAL_EXCEPTIONS:
                pass

        if not persisted_any_children:
            logger.warning(
                "Skipping DB persistence for {} due to missing content.",
                item_input_ref,
            )
            final_result["db_message"] = "DB persistence skipped (no content)."
            final_result["db_id"] = None
            final_result["media_uuid"] = None
        else:
            final_result["db_message"] = "Persisted archive children."

    await persist_claims_if_applicable(
        claims_context=claims_context,
        media_id=final_result.get("db_id"),
        db_path=db_path,
        client_id=client_id,
        loop=loop,
        process_result=final_result,
    )


__all__ = [
    "add_media_orchestrate",
    "add_media_persist",
    "persist_primary_av_item",
    "persist_doc_item_and_children",
    "schedule_media_add_embeddings",
    "sync_media_add_results_to_collections",
]
