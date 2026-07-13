from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from cachetools import LRUCache
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    check_rate_limit,
    get_auth_principal,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import try_get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_add_deps import get_add_media_form
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.exceptions import BadRequestError, JobSubmissionLimitError
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
    _RUN_BOUND_JOB_SENTINEL,
    PlaylistIngestConflictError,
    PlaylistIngestNotFoundError,
    PlaylistIngestStore,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
    classify_playlist_url,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work
from tldw_Server_API.app.services.worker_startup_policy import worker_path_enabled

router = APIRouter()

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
MAX_MEDIA_INGEST_JOBS_OFFSET = 10_000
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})
_TERMINAL_MEDIA_INGEST_JOB_STATUSES = frozenset({"completed", "failed", "cancelled", "quarantined"})
_MAX_RUN_BOUND_SUBMISSIONS = 500
_MAX_RUN_BOUND_ID_LENGTH = 255
_MAX_RUN_BOUND_ARRAY_BYTES = 256 * 1024
_MAX_STAGING_JOB_REFERENCE_SCAN = 100
_SAFE_STAGING_CLEANUP_RESULTS = frozenset({"absent", "deleted"})
_RUN_FILE_STAGING_MANIFEST = ".tldw-upload.json"
_MAX_RUN_FILE_STAGING_MANIFEST_BYTES = 8 * 1024


def get_job_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = f"url:{db_url}" if db_url else f"path:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if not db_url:
            job_manager = JobManager(db_path=Path(db_path)) if db_path else JobManager()
        else:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)

        _job_manager_cache[cache_key] = job_manager
        return job_manager


class MediaIngestJobItem(BaseModel):
    id: int
    uuid: str | None
    source: str
    source_kind: str
    status: str
    collection_id: str | None = None
    planned_item_id: str | None = None
    idempotency_key: str | None = None


class MediaIngestOccurrenceSubmission(BaseModel):
    occurrence_id: str
    status: str
    accepted: bool
    job_id: int | None = None
    batch_id: str
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False
    attempt: int


class SubmitMediaIngestJobsResponse(BaseModel):
    batch_id: str
    jobs: list[MediaIngestJobItem]
    errors: list[str] = Field(default_factory=list)
    submissions: list[MediaIngestOccurrenceSubmission] = Field(default_factory=list)


class MediaIngestJobStatus(BaseModel):
    id: int
    uuid: str | None
    status: str
    job_type: str
    owner_user_id: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    cancelled_at: str | None
    cancellation_reason: str | None
    progress_percent: float | None
    progress_message: str | None
    result: dict[str, Any] | None
    error_message: str | None
    media_type: str | None = None
    source: str | None = None
    source_kind: str | None = None
    batch_id: str | None = None
    collection_id: str | None = None
    planned_item_id: str | None = None
    idempotency_key: str | None = None


class CancelMediaIngestJobResponse(BaseModel):
    success: bool
    job_id: int
    status: str
    message: str | None = None


class CancelMediaIngestBatchResponse(BaseModel):
    success: bool
    batch_id: str
    requested: int
    cancelled: int
    already_terminal: int
    failed: int = 0
    message: str | None = None


class MediaIngestJobListResponse(BaseModel):
    batch_id: str
    jobs: list[MediaIngestJobStatus]
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None
    pagination: OffsetPaginationMeta


def _cleanup_dir(path_str: str) -> None:
    try:
        shutil.rmtree(path_str, ignore_errors=True)
    except Exception:
        logger.debug("Failed to cleanup media ingest temp dir")


def _validate_submit_inputs(
    media_type: Any,
    urls: list[str] | None,
    files: list[UploadFile] | None,
) -> None:
    if urls or files:
        return

    logger.warning("No URLs or files provided in media ingest job submit request")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "No valid media sources supplied. At least one 'url' in the "
            "'urls' list or one 'file' in the 'files' list must be provided."
        ),
    )


def _coerce_form_string_list(value: Any) -> list[str]:
    """Return trimmed form values from repeated fields, JSON strings, or scalar input."""
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            out.extend(_coerce_form_string_list(list(raw)))
            continue
        text = str(raw).strip()
        if not text:
            continue
        if text.startswith("[") or text.startswith('"'):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, list):
                out.extend(str(item).strip() for item in parsed if str(item).strip())
                continue
            if isinstance(parsed, str) and parsed.strip():
                out.append(parsed.strip())
                continue
        out.append(text)
    return out


def _coerce_form_string(value: Any) -> str | None:
    """Return the first normalized form string value, if one is present."""
    values = _coerce_form_string_list(value)
    return values[0] if values else None


def _coerce_positive_int(value: Any) -> int | None:
    """Parse a positive integer identifier from form data, ignoring invalid values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _strict_positive_int_list(value: Any, *, name: str) -> list[int]:
    """Parse bounded canonical positive integers without bool/float coercion."""
    values = _strict_run_bound_array(value, name=name)
    parsed: list[int] = []
    for raw in values:
        if type(raw) is int:
            number = raw
        elif type(raw) is str and raw == raw.strip() and len(raw) <= 10 and raw.isascii() and raw.isdigit():
            if raw.startswith("0"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"{name} must contain positive integers.",
                )
            number = int(raw)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must contain positive integers.",
            )
        if number < 1 or number > 2_147_483_647:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must contain positive integers.",
            )
        parsed.append(number)
    return parsed


def _strict_run_bound_array(value: Any, *, name: str) -> list[Any]:
    """Decode one repeated-form array or one JSON array without element coercion."""
    if value is None:
        return []
    values = list(value) if type(value) in {list, tuple} else [value]
    if len(values) > _MAX_RUN_BOUND_SUBMISSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{name} must contain no more than {_MAX_RUN_BOUND_SUBMISSIONS} items.",
        )
    if len(values) == 1 and type(values[0]) is str:
        encoded = values[0]
        stripped = encoded.lstrip()
        looks_encoded = stripped.startswith(("[", "{", '"')) or stripped in {"null", "true", "false"}
        if not looks_encoded:
            return values
        try:
            encoded_bytes = len(encoded.encode("utf-8"))
        except MemoryError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must be an array.",
            ) from exc
        if encoded_bytes > _MAX_RUN_BOUND_ARRAY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} encoded array is too large.",
            )
        try:
            decoded = json.loads(encoded)
        except (TypeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must be an array.",
            ) from exc
        if type(decoded) is not list:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must be an array.",
            )
        if len(decoded) > _MAX_RUN_BOUND_SUBMISSIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must contain no more than {_MAX_RUN_BOUND_SUBMISSIONS} items.",
            )
        return decoded
    return values


def _bounded_identity_list(value: Any, *, name: str) -> list[str]:
    values = _strict_run_bound_array(value, name=name)
    if any(
        type(item) is not str or not item or item != item.strip() or len(item) > _MAX_RUN_BOUND_ID_LENGTH
        for item in values
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{name} contains an invalid identifier.",
        )
    return list(values)


def _validate_aligned(values: list[Any], *, count: int, name: str, required: bool) -> None:
    if (required or values) and len(values) != count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{name} must match the number of submitted items.",
        )


def _derive_run_occurrence_idempotency(
    *,
    key: bytes,
    owner_user_id: str,
    run_id: str,
    occurrence_id: str,
    attempt: int,
) -> str:
    message = json.dumps(
        ["media_ingest_occurrence", owner_user_id, run_id, occurrence_id, attempt],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"playlist-ingest-v1:{hmac.new(key, message, hashlib.sha256).hexdigest()}"


def _run_file_staging_prefix(*, batch_id: str, idempotency_identity: str) -> str:
    """Return an opaque temp-dir prefix shared by retries of one reservation."""
    marker = hashlib.sha256(f"{batch_id}\0{idempotency_identity}".encode()).hexdigest()[:24]
    return f"media_ingest_job_{marker}_"


def _validated_run_file_staging_dir(
    *,
    temp_dir: Any,
    batch_id: str,
    idempotency_identity: str,
) -> Path | None:
    """Resolve one reservation staging directory without accepting aliases."""
    if not isinstance(temp_dir, str) or not temp_dir:
        return None
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        candidate = Path(temp_dir).resolve()
        prefix = _run_file_staging_prefix(
            batch_id=batch_id,
            idempotency_identity=idempotency_identity,
        )
    except (OSError, RuntimeError):
        return None
    if candidate.parent != temp_root or not candidate.name.startswith(prefix):
        return None
    return candidate


def _write_run_file_staging_manifest(
    *,
    temp_dir: str,
    batch_id: str,
    idempotency_identity: str,
    saved_file: dict[str, Any],
) -> None:
    """Atomically mark one validated staged upload as reusable by exact retries."""
    candidate = _validated_run_file_staging_dir(
        temp_dir=temp_dir,
        batch_id=batch_id,
        idempotency_identity=idempotency_identity,
    )
    source_value = saved_file.get("path")
    if candidate is None or not isinstance(source_value, (str, os.PathLike)):
        raise ValueError("invalid staged upload")
    source = Path(source_value)
    try:
        resolved_source = source.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid staged upload") from exc
    if (
        source.is_symlink()
        or not resolved_source.is_file()
        or resolved_source.parent != candidate
        or resolved_source.name.casefold() == _RUN_FILE_STAGING_MANIFEST.casefold()
    ):
        raise ValueError("invalid staged upload")

    original_filename = saved_file.get("original_filename")
    input_ref = saved_file.get("input_ref")
    if not isinstance(original_filename, str) or not isinstance(input_ref, str):
        raise ValueError("invalid staged upload metadata")
    if not original_filename or len(original_filename) > 1024 or not input_ref or len(input_ref) > 1024:
        raise ValueError("invalid staged upload metadata")
    manifest = {
        "version": 1,
        "source_name": resolved_source.name,
        "original_filename": original_filename,
        "input_ref": input_ref,
    }
    encoded = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > _MAX_RUN_FILE_STAGING_MANIFEST_BYTES:
        raise ValueError("invalid staged upload metadata")

    manifest_path = candidate / _RUN_FILE_STAGING_MANIFEST
    pending_path = candidate / f"{_RUN_FILE_STAGING_MANIFEST}.{uuid4().hex}.tmp"
    try:
        pending_path.write_bytes(encoded)
        os.replace(pending_path, manifest_path)
    finally:
        with contextlib.suppress(OSError):
            pending_path.unlink()


def _read_run_file_staging_manifest(
    *,
    temp_dir: str,
    batch_id: str,
    idempotency_identity: str,
) -> dict[str, str] | None:
    """Return a bounded completed-upload manifest for an exact reservation path."""
    candidate = _validated_run_file_staging_dir(
        temp_dir=temp_dir,
        batch_id=batch_id,
        idempotency_identity=idempotency_identity,
    )
    if candidate is None:
        return None
    manifest_path = candidate / _RUN_FILE_STAGING_MANIFEST
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return None
        size = manifest_path.stat().st_size
        if size < 2 or size > _MAX_RUN_FILE_STAGING_MANIFEST_BYTES:
            return None
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, MemoryError):
        return None
    if not isinstance(manifest, dict) or set(manifest) != {
        "version",
        "source_name",
        "original_filename",
        "input_ref",
    }:
        return None
    source_name = manifest.get("source_name")
    original_filename = manifest.get("original_filename")
    input_ref = manifest.get("input_ref")
    if manifest.get("version") != 1:
        return None
    if not all(
        isinstance(value, str) and 0 < len(value) <= 1024 for value in (source_name, original_filename, input_ref)
    ):
        return None
    if Path(source_name).name != source_name or source_name.casefold() == _RUN_FILE_STAGING_MANIFEST.casefold():
        return None
    source = candidate / source_name
    try:
        resolved_source = source.resolve()
    except (OSError, RuntimeError):
        return None
    if source.is_symlink() or not resolved_source.is_file() or resolved_source.parent != candidate:
        return None
    return {
        "source": str(Path(temp_dir) / source_name),
        "original_filename": original_filename,
        "input_ref": input_ref,
        "temp_dir": temp_dir,
    }


def _cleanup_exact_run_file_staging(
    *,
    temp_dir: Any,
    batch_id: str,
    idempotency_identity: str,
    authoritative_temp_dir: Any = None,
) -> str:
    """Classify cleanup of one exact persisted path without scanning the temp root."""
    candidate = _validated_run_file_staging_dir(
        temp_dir=temp_dir,
        batch_id=batch_id,
        idempotency_identity=idempotency_identity,
    )
    if candidate is None:
        return "invalid"
    try:
        if isinstance(authoritative_temp_dir, str) and authoritative_temp_dir:
            if candidate == Path(authoritative_temp_dir).resolve():
                return "protected"
        if not candidate.exists():
            return "absent"
        shutil.rmtree(candidate)
        return "deleted" if not candidate.exists() else "failed"
    except (OSError, RuntimeError):
        logger.debug("Failed to reconcile media ingest staging directory")
        return "failed"


def _retire_persisted_run_file_staging(
    *,
    store: PlaylistIngestStore,
    owner_user_id: str,
    run_id: str,
    occurrence_id: str,
    attempt: int,
    batch_id: str,
    idempotency_identity: str,
    temp_dir: str | None,
    authoritative_temp_dir: Any = None,
) -> bool:
    """Delete an obsolete exact path before clearing its durable reservation pointer."""
    if not temp_dir:
        return True
    result = _cleanup_exact_run_file_staging(
        temp_dir=temp_dir,
        batch_id=batch_id,
        idempotency_identity=idempotency_identity,
        authoritative_temp_dir=authoritative_temp_dir,
    )
    if result == "protected":
        return True
    if result not in _SAFE_STAGING_CLEANUP_RESULTS:
        return False
    try:
        return bool(
            store.clear_run_item_staging(
                owner_user_id,
                run_id,
                occurrence_id,
                attempt=attempt,
                batch_id=batch_id,
                idempotency_identity=idempotency_identity,
                temp_dir=temp_dir,
            )
        )
    except Exception:
        logger.warning("Failed to clear retired media-ingest staging metadata")
        return False


def _cleanup_abandoned_run_file_staging(
    *,
    store: PlaylistIngestStore,
    jm: JobManager,
    owner_user_id: str,
    retention_seconds: int,
    limit: int,
) -> int:
    """Delete capped expired reservation paths only when no exact Jobs row exists."""
    cutoff = datetime.now().astimezone() - timedelta(seconds=max(0, retention_seconds))
    candidates = store.list_abandoned_run_item_staging(
        owner_user_id,
        older_than=cutoff,
        limit=limit,
    )
    deleted = 0
    for candidate in candidates:
        batch = str(candidate.batch_id or "")
        identity = str(candidate.idempotency_identity or "")
        queue = str(candidate.submission_queue or "")
        temp_dir = str(candidate.staging_temp_dir or "")
        if not batch or not identity or not queue or not temp_dir:
            continue
        try:
            job = _find_exact_occurrence_job(
                jm,
                queue=queue,
                owner_user_id=owner_user_id,
                batch_id=batch,
                idempotency_key=identity,
            )
        except Exception:
            logger.warning("Skipping abandoned media-ingest staging cleanup because exact job lookup failed")
            continue
        if job is not None:
            continue
        try:
            if store.has_live_run_item_staging_reference(
                owner_user_id,
                temp_dir,
                excluding_run_id=str(candidate.run_id),
                excluding_occurrence_id=str(candidate.occurrence_id),
            ):
                continue
            owner_jobs = jm.list_jobs(
                domain="media_ingest",
                owner_user_id=owner_user_id,
                job_type="media_ingest_item",
                limit=_MAX_STAGING_JOB_REFERENCE_SCAN + 1,
            )
            if len(owner_jobs) > _MAX_STAGING_JOB_REFERENCE_SCAN:
                continue
            job_reference_found = False
            for owner_job in owner_jobs:
                binding_view = jm.normalize_job_binding_view(
                    owner_job,
                    owner_user_id=owner_user_id,
                )
                if binding_view is None:
                    job_reference_found = True
                    break
                if binding_view["payload"].get("temp_dir") == temp_dir:
                    job_reference_found = True
                    break
            if job_reference_found:
                continue
        except Exception:
            logger.warning("Skipping abandoned media-ingest staging cleanup because live references are ambiguous")
            continue
        cleanup_result = _cleanup_exact_run_file_staging(
            temp_dir=temp_dir,
            batch_id=batch,
            idempotency_identity=identity,
        )
        if cleanup_result not in _SAFE_STAGING_CLEANUP_RESULTS:
            continue
        try:
            cleared = store.clear_run_item_staging(
                owner_user_id,
                candidate.run_id,
                candidate.occurrence_id,
                attempt=int(candidate.attempt),
                batch_id=batch,
                idempotency_identity=identity,
                temp_dir=temp_dir,
            )
        except Exception:
            cleared = False
        if cleanup_result == "deleted" and cleared:
            deleted += 1
    return deleted


def _submission_rejected(
    *,
    occurrence_id: str,
    batch_id: str,
    attempt: int,
    code: str,
    message: str,
    retryable: bool = False,
) -> MediaIngestOccurrenceSubmission:
    return MediaIngestOccurrenceSubmission(
        occurrence_id=occurrence_id,
        status="rejected",
        accepted=False,
        batch_id=batch_id,
        error_code=code,
        message=message,
        retryable=retryable,
        attempt=attempt,
    )


def _submission_accepted(
    *,
    occurrence_id: str,
    batch_id: str,
    attempt: int,
    job_id: int,
) -> MediaIngestOccurrenceSubmission:
    return MediaIngestOccurrenceSubmission(
        occurrence_id=occurrence_id,
        status="accepted",
        accepted=True,
        job_id=job_id,
        batch_id=batch_id,
        retryable=False,
        attempt=attempt,
    )


def _job_item_from_row(
    row: dict[str, Any],
    *,
    fallback_kind: str,
    idempotency_key: str,
) -> MediaIngestJobItem:
    payload = _normalize_payload(row.get("payload"))
    return MediaIngestJobItem(
        id=int(row["id"]),
        uuid=row.get("uuid"),
        source=str(payload.get("source") or ""),
        source_kind=str(payload.get("source_kind") or fallback_kind),
        status=str(row.get("status") or "queued"),
        collection_id=(str(payload["collection_id"]) if payload.get("collection_id") is not None else None),
        planned_item_id=(str(payload["planned_item_id"]) if payload.get("planned_item_id") is not None else None),
        idempotency_key=idempotency_key,
    )


def _find_exact_occurrence_job(
    jm: JobManager,
    *,
    queue: str,
    owner_user_id: str,
    batch_id: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    return jm.get_job_by_idempotency(
        domain="media_ingest",
        queue=queue,
        job_type="media_ingest_item",
        idempotency_key=idempotency_key,
        owner_user_id=owner_user_id,
        batch_group=batch_id,
    )


def _validate_per_url_binding_list(
    *,
    name: str,
    values: list[str],
    url_count: int,
) -> None:
    """Ensure optional per-URL binding arrays line up with the submitted URL count."""
    if not values:
        return
    if len(values) == url_count:
        return
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"{name} must match the number of URL items.",
    )


def _resolve_submit_bindings(
    *,
    url_count: int,
    media_collection_id: Any,
    collection_id: Any,
    media_collection_item_id: Any,
    planned_item_ids: Any,
    idempotency_key: Any,
    idempotency_keys: Any,
) -> tuple[str | None, list[str], list[str]]:
    """Normalize collection, planned-item, and idempotency bindings for URL jobs."""
    collection_id_value = _coerce_form_string(media_collection_id) or _coerce_form_string(collection_id)
    planned_values = _coerce_form_string_list(planned_item_ids)
    single_planned = _coerce_form_string(media_collection_item_id)
    if single_planned and not planned_values:
        planned_values = [single_planned]

    key_values = _coerce_form_string_list(idempotency_keys)
    single_key = _coerce_form_string(idempotency_key)
    if single_key and not key_values:
        key_values = [single_key]

    _validate_per_url_binding_list(
        name="planned_item_ids",
        values=planned_values,
        url_count=url_count,
    )
    _validate_per_url_binding_list(
        name="idempotency_keys",
        values=key_values,
        url_count=url_count,
    )
    return collection_id_value, planned_values, key_values


def _apply_collection_binding_to_payload(
    payload: dict[str, Any],
    *,
    collection_id: str | None,
    planned_item_id: str | None,
    idempotency_key: str | None,
) -> None:
    """Attach durable collection binding fields to a job payload when supplied."""
    if collection_id:
        payload["collection_id"] = collection_id
    if planned_item_id:
        payload["planned_item_id"] = planned_item_id
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key


def _submit_failure_message(exc: Exception) -> str:
    """Return a bounded, user-safe message for collection submit-failure tracking."""
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            return detail.strip() or "Media ingest job submission failed"
        return str(detail).strip() or "Media ingest job submission failed"
    return str(exc).strip() or "Media ingest job submission failed"


def _mark_collection_item_submit_failed(
    *,
    collections_db: CollectionsDatabase | None,
    planned_item_id: Any,
    error_summary: str,
) -> None:
    """Mark a planned collection item as submit_failed using the injected DB handle."""
    item_id = _coerce_positive_int(planned_item_id)
    if item_id is None or collections_db is None:
        return

    try:
        collections_db.update_media_collection_item_status(
            item_id,
            status="submit_failed",
            latest_job_id=None,
            error_summary=error_summary[:1000],
        )
    except Exception as exc:
        logger.warning(
            "Media collection item submit-failure sync failed for item {}: {}",
            item_id,
            exc,
        )


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.debug(
                "Failed to parse payload as JSON (length={}, error={})",
                len(payload),
                exc,
            )
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_job_created_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_heavy_media_ingest_request(form_data: AddMediaForm) -> bool:
    media_type = str(getattr(form_data, "media_type", "") or "").strip().lower()
    if media_type in {"audio", "video"}:
        return True
    return bool(getattr(form_data, "enable_ocr", False))


def _heavy_media_ingest_worker_available() -> bool:
    return worker_path_enabled(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        default_stable=False,
        # Queue routing should still honor explicit route policy in tests so
        # integration coverage can model a deployed heavy-worker path without
        # auto-starting local workers.
        test_mode=False,
    )


def _resolve_media_ingest_queue(form_data: AddMediaForm) -> str:
    default_queue = (os.getenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE") or "default").strip() or "default"
    route_heavy = _is_truthy(os.getenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true"))
    if not route_heavy:
        return default_queue
    if not _is_heavy_media_ingest_request(form_data):
        return default_queue
    if not _heavy_media_ingest_worker_available():
        return default_queue
    # Keep fallback within JobManager standard queue names unless explicitly overridden.
    heavy_queue = (os.getenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE") or "low").strip() or "low"
    return heavy_queue


def _create_media_ingest_job(
    *,
    jm: JobManager,
    selected_queue: str,
    payload: dict[str, Any],
    current_user: User,
    batch_id: str,
    request_id: str | None,
    trace_id: str | None,
    available_at: datetime | None = None,
) -> dict[str, Any]:
    try:
        return jm.create_job(
            domain="media_ingest",
            queue=selected_queue,
            job_type="media_ingest_item",
            payload=payload,
            batch_group=batch_id,
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=3,
            idempotency_key=payload.get("idempotency_key"),
            request_id=request_id,
            trace_id=trace_id,
            available_at=available_at,
        )
    except JobSubmissionLimitError as exc:
        message = str(exc).strip() or "Media ingest job submission limit exceeded"
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": exc.code, "message": message},
            headers=headers,
        ) from exc
    except BadRequestError as exc:
        message = str(exc).strip() or "Invalid media ingest job request"
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message) from exc


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {str(role).strip().lower() for role in (principal.roles or []) if str(role).strip()}
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower() for permission in (principal.permissions or []) if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _job_to_status(job: dict[str, Any]) -> MediaIngestJobStatus:
    payload = _normalize_payload(job.get("payload"))
    id_value = job.get("id")
    if id_value is None:
        raise ValueError(f"Missing job id in job: {job!r}")
    try:
        job_id = int(id_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid job id {id_value!r} in job: {job!r}") from exc
    return MediaIngestJobStatus(
        id=job_id,
        uuid=job.get("uuid"),
        status=job.get("status"),
        job_type=job.get("job_type"),
        owner_user_id=job.get("owner_user_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        cancelled_at=job.get("cancelled_at"),
        cancellation_reason=job.get("cancellation_reason"),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        result=job.get("result"),
        error_message=job.get("error_message"),
        media_type=payload.get("media_type"),
        source=payload.get("source"),
        source_kind=payload.get("source_kind"),
        batch_id=payload.get("batch_id"),
        collection_id=payload.get("collection_id"),
        planned_item_id=payload.get("planned_item_id"),
        idempotency_key=payload.get("idempotency_key"),
    )


def _collect_jobs_for_batch(
    *,
    jm: JobManager,
    batch_id: str,
    owner_filter: str | None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if not batch_id:
        return []

    matched = jm.list_jobs(
        domain="media_ingest",
        owner_user_id=owner_filter,
        batch_group=batch_id,
        limit=limit,
        sort_by="created_at",
        sort_order="desc",
    )
    if len(matched) >= limit:
        return matched[:limit]

    page_limit = min(500, max(limit, 100))
    seen_ids = {int(job.get("id")) for job in matched if job.get("id") is not None}
    cursor_created_at: datetime | None = None
    cursor_id: int | None = None
    last_cursor: tuple[datetime, int] | None = None

    while len(matched) < limit:
        jobs = jm.list_jobs(
            domain="media_ingest",
            owner_user_id=owner_filter,
            limit=page_limit,
            created_before=cursor_created_at,
            before_id=cursor_id,
            sort_by="created_at",
            sort_order="desc",
        )
        if not jobs:
            break
        for job in jobs:
            raw_job_id = job.get("id")
            if raw_job_id is not None and int(raw_job_id) in seen_ids:
                continue
            payload = _normalize_payload(job.get("payload"))
            if str(payload.get("batch_id") or "") == batch_id:
                matched.append(job)
                if raw_job_id is not None:
                    seen_ids.add(int(raw_job_id))
                if len(matched) >= limit:
                    return matched[:limit]
        if len(jobs) < page_limit:
            break
        last_job = jobs[-1]
        next_created_at = _parse_job_created_at(last_job.get("created_at"))
        next_id_raw = last_job.get("id")
        if next_created_at is None or next_id_raw is None:
            break
        next_cursor = (next_created_at, int(next_id_raw))
        if last_cursor == next_cursor:
            break
        last_cursor = next_cursor
        cursor_created_at, cursor_id = next_cursor

    return matched[:limit]


def _batch_exists_for_any_owner(
    *,
    jm: JobManager,
    batch_id: str,
) -> bool:
    if not batch_id:
        return False
    jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=batch_id,
        owner_filter=None,
        limit=1,
    )
    return bool(jobs)


def _resolve_batch_or_session_id(
    *,
    batch_id: str | None,
    session_id: str | None,
) -> str:
    resolved = str(batch_id or session_id or "").strip()
    if resolved:
        return resolved
    raise HTTPException(status_code=400, detail="Either batch_id or session_id is required")


async def _submit_run_bound_media_ingest_jobs(
    *,
    request: Request,
    form_data: AddMediaForm,
    files: list[UploadFile] | None,
    run_id: str,
    occurrence_ids: Any,
    attempts: Any,
    planned_item_ids: Any,
    file_occurrence_ids: Any,
    file_attempts: Any,
    file_planned_item_ids: Any,
    current_user: User,
    jm: JobManager,
) -> SubmitMediaIngestJobsResponse | JSONResponse:
    """Submit occurrence-bound jobs while preserving legacy response fields."""
    owner = str(current_user.id)
    run_identity = str(run_id).strip()
    if not run_identity or len(run_identity) > _MAX_RUN_BOUND_ID_LENGTH:
        raise HTTPException(status_code=422, detail="run_id is invalid.")

    url_list = [str(url).strip() for url in (form_data.urls or []) if str(url).strip()]
    uploads = list(files or [])
    url_occurrences = _bounded_identity_list(occurrence_ids, name="occurrence_ids")
    url_attempts = _strict_positive_int_list(attempts, name="attempts")
    url_planned = _strict_positive_int_list(planned_item_ids, name="planned_item_ids")
    upload_occurrences = _bounded_identity_list(file_occurrence_ids, name="file_occurrence_ids")
    upload_attempts = _strict_positive_int_list(file_attempts, name="file_attempts")
    upload_planned = _strict_positive_int_list(file_planned_item_ids, name="file_planned_item_ids")
    _validate_aligned(url_occurrences, count=len(url_list), name="occurrence_ids", required=bool(url_list))
    _validate_aligned(url_attempts, count=len(url_list), name="attempts", required=bool(url_list))
    _validate_aligned(url_planned, count=len(url_list), name="planned_item_ids", required=False)
    _validate_aligned(
        upload_occurrences,
        count=len(uploads),
        name="file_occurrence_ids",
        required=bool(uploads),
    )
    _validate_aligned(upload_attempts, count=len(uploads), name="file_attempts", required=bool(uploads))
    _validate_aligned(
        upload_planned,
        count=len(uploads),
        name="file_planned_item_ids",
        required=False,
    )
    total = len(url_list) + len(uploads)
    if not 1 <= total <= _MAX_RUN_BOUND_SUBMISSIONS:
        raise HTTPException(status_code=422, detail="Run-bound submissions must contain between 1 and 500 items.")
    all_occurrences = [*url_occurrences, *upload_occurrences]
    if len(set(all_occurrences)) != len(all_occurrences):
        raise HTTPException(status_code=422, detail="occurrence_ids must be unique within a submission.")

    store = PlaylistIngestStore(jm)
    try:
        retention_seconds = max(
            300,
            min(int(os.getenv("MEDIA_INGEST_STAGING_RETENTION_SECONDS", "86400")), 2_592_000),
        )
        cleanup_limit = max(0, min(int(os.getenv("MEDIA_INGEST_STAGING_CLEANUP_LIMIT", "10")), 100))
    except ValueError:
        retention_seconds, cleanup_limit = 86400, 10
    if cleanup_limit:
        with contextlib.suppress(Exception):
            _cleanup_abandoned_run_file_staging(
                store=store,
                jm=jm,
                owner_user_id=owner,
                retention_seconds=retention_seconds,
                limit=cleanup_limit,
            )
    try:
        run = store.get_run(owner, run_identity)
        items = {item.occurrence_id: item for item in store.list_run_items(owner, run_identity, limit=500)}
    except PlaylistIngestNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Playlist ingest run not found.") from exc
    if run.status not in {"staged", "running"}:
        raise HTTPException(status_code=409, detail="Playlist ingest run is not accepting jobs.")

    batch_id = str(uuid4())
    hmac_key = derive_hmac_key()
    base_options = form_data.model_dump(mode="json")
    base_options.pop("urls", None)
    base_options.pop("keywords", None)
    if run.processing_options:
        base_options.update(run.processing_options)
    candidate_queue = _resolve_media_ingest_queue(form_data)
    rid = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""

    prepared: list[dict[str, Any]] = []
    submissions: list[MediaIngestOccurrenceSubmission] = []
    for index, (source, occurrence, attempt) in enumerate(zip(url_list, url_occurrences, url_attempts, strict=True)):
        item = items.get(occurrence)
        client_planned = url_planned[index] if url_planned else None
        rejection: MediaIngestOccurrenceSubmission | None = None
        if item is None:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_not_found",
                message="Run occurrence was not found.",
            )
        elif item.attempt != attempt:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_attempt_mismatch",
                message="Submitted attempt does not match the run occurrence.",
            )
        elif client_planned is not None and client_planned != item.planned_collection_item_id:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="planned_item_mismatch",
                message="Submitted planned item does not match the run occurrence.",
            )
        elif (
            item.input_kind == "file_stub"
            or item.action not in {"ingest", "overwrite"}
            or item.state
            not in {
                "staged",
                "submit_pending",
                "queued",
            }
        ):
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_not_processable",
                message="Run occurrence does not require URL processing.",
            )
        elif source != item.source_url:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_source_mismatch",
                message="Submitted source does not match the run occurrence.",
            )
        else:
            try:
                classified = classify_playlist_url(str(item.source_url))
            except ValueError:
                rejection = _submission_rejected(
                    occurrence_id=occurrence,
                    batch_id=batch_id,
                    attempt=attempt,
                    code="occurrence_source_invalid",
                    message="Stored run source is not a valid media URL.",
                )
            else:
                if classified.is_playlist:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "playlist_preflight_required",
                            "message": "Playlist URLs must be inspected before job submission.",
                        },
                    )
        if rejection is not None:
            submissions.append(rejection)
            continue
        prepared.append(
            {
                "kind": "url",
                "index": index,
                "item": item,
                "occurrence_id": occurrence,
                "attempt": attempt,
                "planned_item_id": item.planned_collection_item_id,
            }
        )

    for index, (occurrence, attempt) in enumerate(zip(upload_occurrences, upload_attempts, strict=True)):
        item = items.get(occurrence)
        client_planned = upload_planned[index] if upload_planned else None
        rejection = None
        if item is None:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_not_found",
                message="Run occurrence was not found.",
            )
        elif item.attempt != attempt:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_attempt_mismatch",
                message="Submitted attempt does not match the run occurrence.",
            )
        elif client_planned is not None and client_planned != item.planned_collection_item_id:
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="planned_item_mismatch",
                message="Submitted planned item does not match the run occurrence.",
            )
        elif (
            item.input_kind != "file_stub"
            or item.action not in {"ingest", "overwrite"}
            or item.state
            not in {
                "awaiting_upload",
                "submit_pending",
                "queued",
            }
        ):
            rejection = _submission_rejected(
                occurrence_id=occurrence,
                batch_id=batch_id,
                attempt=attempt,
                code="occurrence_not_processable",
                message="Run occurrence is not awaiting this file upload.",
            )
        if rejection is not None:
            submissions.append(rejection)
            continue
        prepared.append(
            {
                "kind": "file",
                "index": index,
                "item": item,
                "occurrence_id": occurrence,
                "attempt": attempt,
                "planned_item_id": item.planned_collection_item_id,
            }
        )

    jobs: list[MediaIngestJobItem] = []
    errors: list[str] = []
    for entry in prepared:
        occurrence = str(entry["occurrence_id"])
        attempt = int(entry["attempt"])
        identity = _derive_run_occurrence_idempotency(
            key=hmac_key,
            owner_user_id=owner,
            run_id=run_identity,
            occurrence_id=occurrence,
            attempt=attempt,
        )
        try:
            reserved = store.prepare_run_item_job_submission(
                owner,
                run_identity,
                occurrence,
                attempt=attempt,
                batch_id=batch_id,
                idempotency_identity=identity,
                submission_queue=candidate_queue,
                source_kind=str(entry["kind"]),
                planned_item_id=entry["planned_item_id"],
            )
        except (PlaylistIngestConflictError, PlaylistIngestNotFoundError):
            submissions.append(
                _submission_rejected(
                    occurrence_id=occurrence,
                    batch_id=batch_id,
                    attempt=attempt,
                    code="occurrence_not_processable",
                    message="Run occurrence state changed before submission.",
                    retryable=True,
                )
            )
            continue

        identity = str(reserved.idempotency_identity or "")
        selected_queue = str(reserved.submission_queue or "")
        if not identity or not selected_queue:
            submissions.append(
                _submission_rejected(
                    occurrence_id=occurrence,
                    batch_id=str(reserved.batch_id or batch_id),
                    attempt=attempt,
                    code="occurrence_binding_pending",
                    message="Accepted job binding is temporarily unavailable.",
                    retryable=True,
                )
            )
            continue
        entry_options = dict(base_options)
        entry_options["overwrite_existing"] = reserved.action == "overwrite"
        reserved_batch = str(reserved.batch_id or batch_id)
        owns_reservation = reserved_batch == batch_id
        if reserved.state == "queued" and reserved.job_id is not None:
            existing = jm.normalize_job_binding_view(
                jm.get_job(reserved.job_id),
                owner_user_id=owner,
            )
            try:
                bound = (
                    store.bind_run_item_job(
                        owner,
                        run_identity,
                        occurrence,
                        attempt=attempt,
                        job_id=int(reserved.job_id),
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                    )
                    if existing is not None
                    else None
                )
            except Exception:
                bound = None
            if existing is None or bound is None:
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                continue
            jobs.append(_job_item_from_row(existing, fallback_kind=str(entry["kind"]), idempotency_key=identity))
            submissions.append(
                _submission_accepted(
                    occurrence_id=occurrence,
                    batch_id=str(bound.batch_id or reserved_batch),
                    attempt=attempt,
                    job_id=int(existing["id"]),
                )
            )
            continue

        recovering_reservation = reserved.state == "submit_pending" and not owns_reservation
        if recovering_reservation:
            try:
                existing = _find_exact_occurrence_job(
                    jm,
                    queue=selected_queue,
                    owner_user_id=owner,
                    batch_id=reserved_batch,
                    idempotency_key=identity,
                )
            except Exception:
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                continue
            existing = jm.normalize_job_binding_view(existing, owner_user_id=owner)
            if existing is not None and entry["kind"] == "file" and reserved.staging_temp_dir:
                if not _retire_persisted_run_file_staging(
                    store=store,
                    owner_user_id=owner,
                    run_id=run_identity,
                    occurrence_id=occurrence,
                    attempt=attempt,
                    batch_id=reserved_batch,
                    idempotency_identity=identity,
                    temp_dir=reserved.staging_temp_dir,
                    authoritative_temp_dir=existing["payload"].get("temp_dir"),
                ):
                    existing = None
            if existing is not None:
                try:
                    bound = store.bind_run_item_job(
                        owner,
                        run_identity,
                        occurrence,
                        attempt=attempt,
                        job_id=int(existing["id"]),
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                    )
                except Exception:
                    existing = None
            if existing is not None:
                jobs.append(_job_item_from_row(existing, fallback_kind=str(entry["kind"]), idempotency_key=identity))
                submissions.append(
                    _submission_accepted(
                        occurrence_id=occurrence,
                        batch_id=str(bound.batch_id or reserved_batch),
                        attempt=attempt,
                        job_id=int(existing["id"]),
                    )
                )
                continue

        prior_staging_dir = reserved.staging_temp_dir if entry["kind"] == "file" else None
        recovered_staging: dict[str, str] | None = None
        if recovering_reservation and entry["kind"] == "file":
            if prior_staging_dir:
                recovered_staging = _read_run_file_staging_manifest(
                    temp_dir=prior_staging_dir,
                    batch_id=reserved_batch,
                    idempotency_identity=identity,
                )
            if recovered_staging is None:
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                continue

        temp_dir_path: str | None = None
        file_staging_prefix: str | None = None
        completed_staging_is_shared = recovered_staging is not None
        if entry["kind"] == "url":
            source = str(reserved.source_url)
            payload = {
                "batch_id": reserved_batch,
                "media_type": str(form_data.media_type),
                "source": source,
                "source_kind": "url",
                "input_ref": source,
                "options": entry_options,
            }
        elif recovered_staging is not None:
            source = recovered_staging["source"]
            payload = {
                "batch_id": reserved_batch,
                "media_type": str(form_data.media_type),
                "source": source,
                "source_kind": "file",
                "input_ref": recovered_staging["input_ref"],
                "original_filename": recovered_staging["original_filename"],
                "temp_dir": recovered_staging["temp_dir"],
                "cleanup_temp_dir": True,
                "options": entry_options,
            }
        else:
            upload = uploads[int(entry["index"])]
            file_staging_prefix = _run_file_staging_prefix(
                batch_id=reserved_batch,
                idempotency_identity=identity,
            )
            try:
                with TempDirManager(prefix=file_staging_prefix, cleanup=False) as temp_dir:
                    temp_dir_path = str(temp_dir)
                    store.record_run_item_staging(
                        owner,
                        run_identity,
                        occurrence,
                        attempt=attempt,
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                        temp_dir=temp_dir_path,
                    )
                    saved_files, file_errors = await save_uploaded_files(
                        [upload],
                        temp_dir=temp_dir,
                        validator=file_validator_instance,
                        expected_media_type_key=str(form_data.media_type),
                    )
                if file_errors or len(saved_files) != 1:
                    raise ValueError("upload_staging_failed")
                saved = saved_files[0]
                source = str(saved.get("path"))
                original_filename = saved.get("original_filename")
                _write_run_file_staging_manifest(
                    temp_dir=temp_dir_path,
                    batch_id=reserved_batch,
                    idempotency_identity=identity,
                    saved_file=saved,
                )
                completed_staging_is_shared = True
                payload = {
                    "batch_id": reserved_batch,
                    "media_type": str(form_data.media_type),
                    "source": source,
                    "source_kind": "file",
                    "input_ref": saved.get("input_ref") or original_filename or source,
                    "original_filename": original_filename,
                    "temp_dir": temp_dir_path,
                    "cleanup_temp_dir": True,
                    "options": entry_options,
                }
            except Exception:
                staging_retired = temp_dir_path is None
                if temp_dir_path:
                    cleanup_result = _cleanup_exact_run_file_staging(
                        temp_dir=temp_dir_path,
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                    )
                    staging_retired = cleanup_result in _SAFE_STAGING_CLEANUP_RESULTS
                    if staging_retired:
                        try:
                            store.clear_run_item_staging(
                                owner,
                                run_identity,
                                occurrence,
                                attempt=attempt,
                                batch_id=reserved_batch,
                                idempotency_identity=identity,
                                temp_dir=temp_dir_path,
                            )
                        except Exception:
                            logger.warning("Failed to clear retired media-ingest staging metadata")
                if staging_retired:
                    try:
                        store.reset_run_item_job_submission(
                            owner,
                            run_identity,
                            occurrence,
                            attempt=attempt,
                            batch_id=reserved_batch,
                            idempotency_identity=identity,
                        )
                    except Exception:
                        logger.warning("Failed to release media-ingest upload reservation")
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="upload_staging_failed",
                        message="Upload staging failed.",
                        retryable=True,
                    )
                )
                errors.append("Upload staging failed")
                continue

        payload.update(
            {
                "run_id": run_identity,
                "occurrence_id": occurrence,
                "attempt": attempt,
                "idempotency_key": identity,
            }
        )
        if run.collection_id is not None:
            payload["collection_id"] = run.collection_id
        if entry["planned_item_id"] is not None:
            payload["planned_item_id"] = entry["planned_item_id"]
        try:
            row = _create_media_ingest_job(
                jm=jm,
                selected_queue=selected_queue,
                payload=payload,
                current_user=current_user,
                batch_id=reserved_batch,
                request_id=rid,
                trace_id=tp or None,
                available_at=_RUN_BOUND_JOB_SENTINEL,
            )
            row = jm.normalize_job_binding_view(row, owner_user_id=owner)
            row_id = row.get("id") if row is not None else None
            if row_id is None:
                raise ValueError("job_create_failed")
        except Exception as exc:
            if isinstance(exc, HTTPException) and isinstance(exc.__cause__, JobSubmissionLimitError):
                if temp_dir_path and not completed_staging_is_shared:
                    _retire_persisted_run_file_staging(
                        store=store,
                        owner_user_id=owner,
                        run_id=run_identity,
                        occurrence_id=occurrence,
                        attempt=attempt,
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                        temp_dir=temp_dir_path,
                    )
                raise
            try:
                existing = _find_exact_occurrence_job(
                    jm,
                    queue=selected_queue,
                    owner_user_id=owner,
                    batch_id=reserved_batch,
                    idempotency_key=identity,
                )
            except Exception:
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                errors.append("Accepted job binding is temporarily unavailable")
                continue
            existing = jm.normalize_job_binding_view(existing, owner_user_id=owner)
            if existing is not None:
                stored_payload = existing["payload"]
                if temp_dir_path and stored_payload.get("source") != payload.get("source"):
                    if not _retire_persisted_run_file_staging(
                        store=store,
                        owner_user_id=owner,
                        run_id=run_identity,
                        occurrence_id=occurrence,
                        attempt=attempt,
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                        temp_dir=temp_dir_path,
                    ):
                        submissions.append(
                            _submission_rejected(
                                occurrence_id=occurrence,
                                batch_id=reserved_batch,
                                attempt=attempt,
                                code="occurrence_binding_pending",
                                message="Accepted job binding is temporarily unavailable.",
                                retryable=True,
                            )
                        )
                        errors.append("Accepted job binding is temporarily unavailable")
                        continue
                    temp_dir_path = None
                try:
                    bound = store.bind_run_item_job(
                        owner,
                        run_identity,
                        occurrence,
                        attempt=attempt,
                        job_id=int(existing["id"]),
                        batch_id=reserved_batch,
                        idempotency_identity=identity,
                    )
                except Exception:
                    submissions.append(
                        _submission_rejected(
                            occurrence_id=occurrence,
                            batch_id=reserved_batch,
                            attempt=attempt,
                            code="occurrence_binding_pending",
                            message="Accepted job binding is temporarily unavailable.",
                            retryable=True,
                        )
                    )
                    errors.append("Accepted job binding is temporarily unavailable")
                    continue
                jobs.append(_job_item_from_row(existing, fallback_kind=str(entry["kind"]), idempotency_key=identity))
                submissions.append(
                    _submission_accepted(
                        occurrence_id=occurrence,
                        batch_id=str(bound.batch_id or reserved_batch),
                        attempt=attempt,
                        job_id=int(existing["id"]),
                    )
                )
                continue

            if temp_dir_path and not completed_staging_is_shared:
                _retire_persisted_run_file_staging(
                    store=store,
                    owner_user_id=owner,
                    run_id=run_identity,
                    occurrence_id=occurrence,
                    attempt=attempt,
                    batch_id=reserved_batch,
                    idempotency_identity=identity,
                    temp_dir=temp_dir_path,
                )
            if isinstance(exc, HTTPException) and exc.status_code in {429, 503}:
                raise
            if not owns_reservation:
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                errors.append("Accepted job binding is temporarily unavailable")
                continue
            submissions.append(
                _submission_rejected(
                    occurrence_id=occurrence,
                    batch_id=reserved_batch,
                    attempt=attempt,
                    code="job_submission_failed",
                    message="Media ingest job submission failed.",
                    retryable=True,
                )
            )
            errors.append("Media ingest job submission failed")
            continue

        stored_payload = row["payload"]
        if temp_dir_path and stored_payload and stored_payload.get("source") != payload.get("source"):
            if not _retire_persisted_run_file_staging(
                store=store,
                owner_user_id=owner,
                run_id=run_identity,
                occurrence_id=occurrence,
                attempt=attempt,
                batch_id=reserved_batch,
                idempotency_identity=identity,
                temp_dir=temp_dir_path,
            ):
                submissions.append(
                    _submission_rejected(
                        occurrence_id=occurrence,
                        batch_id=reserved_batch,
                        attempt=attempt,
                        code="occurrence_binding_pending",
                        message="Accepted job binding is temporarily unavailable.",
                        retryable=True,
                    )
                )
                errors.append("Accepted job binding is temporarily unavailable")
                continue
            temp_dir_path = None
        try:
            bound = store.bind_run_item_job(
                owner,
                run_identity,
                occurrence,
                attempt=attempt,
                job_id=int(row_id),
                batch_id=reserved_batch,
                idempotency_identity=identity,
            )
        except Exception:
            submissions.append(
                _submission_rejected(
                    occurrence_id=occurrence,
                    batch_id=reserved_batch,
                    attempt=attempt,
                    code="occurrence_binding_pending",
                    message="Accepted job binding is temporarily unavailable.",
                    retryable=True,
                )
            )
            errors.append("Accepted job binding is temporarily unavailable")
            continue
        effective_payload = stored_payload or payload
        jobs.append(
            MediaIngestJobItem(
                id=int(row_id),
                uuid=row.get("uuid"),
                source=str(effective_payload.get("source") or ""),
                source_kind=str(effective_payload.get("source_kind") or entry["kind"]),
                status=str(row.get("status") or "queued"),
                collection_id=(
                    str(effective_payload["collection_id"])
                    if effective_payload.get("collection_id") is not None
                    else None
                ),
                planned_item_id=(
                    str(effective_payload["planned_item_id"])
                    if effective_payload.get("planned_item_id") is not None
                    else None
                ),
                idempotency_key=identity,
            )
        )
        submissions.append(
            _submission_accepted(
                occurrence_id=occurrence,
                batch_id=str(bound.batch_id or reserved_batch),
                attempt=attempt,
                job_id=int(row_id),
            )
        )

    order = {occurrence: index for index, occurrence in enumerate(all_occurrences)}
    submissions.sort(key=lambda record: order[record.occurrence_id])
    response = SubmitMediaIngestJobsResponse(
        batch_id=batch_id,
        jobs=jobs,
        errors=errors,
        submissions=submissions,
    )
    if any(not record.accepted for record in submissions):
        return JSONResponse(status_code=status.HTTP_207_MULTI_STATUS, content=response.model_dump())
    return response


@router.post(
    "/ingest/jobs",
    response_model=SubmitMediaIngestJobsResponse,
    summary="Submit async media ingestion jobs (one job per item)",
    tags=["Media Ingestion Jobs"],
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
        Depends(guard_storage_quota),
        # Pessimistic pre-check: verifies at least 1 MB of storage quota
        # remains.  Actual size is unknown until after ingestion completes,
        # so the real usage is recorded post-ingestion by the job worker.
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def submit_media_ingest_jobs(
    request: Request,
    form_data: AddMediaForm = Depends(get_add_media_form),
    files: list[UploadFile] | None = File(None, description="Optional media uploads"),
    run_id: str | None = Form(None, description="Owner-scoped playlist ingest run identifier"),
    occurrence_ids: list[str] | None = Form(
        None,
        description="Aligned run occurrence identifiers for URL items",
    ),
    attempts: list[str] | None = Form(None, description="Aligned positive attempt numbers for URL items"),
    file_occurrence_ids: list[str] | None = Form(
        None,
        description="Aligned run occurrence identifiers for uploaded files",
    ),
    file_attempts: list[str] | None = Form(
        None,
        description="Aligned positive attempt numbers for uploaded files",
    ),
    file_planned_item_ids: list[str] | None = Form(
        None,
        description="Optional aligned planned collection item identifiers for files",
    ),
    media_collection_id: str | None = Form(
        None,
        description="Optional durable media collection id for this job batch",
    ),
    collection_id: str | None = Form(
        None,
        description="Alias for media_collection_id",
    ),
    media_collection_item_id: str | None = Form(
        None,
        description="Optional planned collection item id for single-item submits",
    ),
    planned_item_ids: list[str] | None = Form(
        None,
        description="Optional JSON/list of planned collection item ids, one per URL",
    ),
    idempotency_key: str | None = Form(
        None,
        description="Optional idempotency key for single-item submits",
    ),
    idempotency_keys: list[str] | None = Form(
        None,
        description="Optional JSON/list of idempotency keys, one per URL",
    ),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
    collections_db: CollectionsDatabase | None = Depends(try_get_collections_db_for_user),
) -> SubmitMediaIngestJobsResponse:
    assert_may_start_work(request.app, "media.ingest.jobs.submit")
    rid = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""

    # Normalize sentinel urls=[''] from some clients.
    if form_data.urls and form_data.urls == [""]:
        form_data.urls = None

    _validate_submit_inputs(form_data.media_type, form_data.urls, files)

    run_id_value = _coerce_form_string(run_id)
    if run_id_value is not None:
        return await _submit_run_bound_media_ingest_jobs(
            request=request,
            form_data=form_data,
            files=files,
            run_id=run_id_value,
            occurrence_ids=occurrence_ids,
            attempts=attempts,
            planned_item_ids=planned_item_ids,
            file_occurrence_ids=file_occurrence_ids,
            file_attempts=file_attempts,
            file_planned_item_ids=file_planned_item_ids,
            current_user=current_user,
            jm=jm,
        )

    options = form_data.model_dump(mode="json")
    options.pop("urls", None)
    options.pop("keywords", None)
    selected_queue = _resolve_media_ingest_queue(form_data)

    batch_id = str(uuid4())
    jobs: list[MediaIngestJobItem] = []
    errors: list[str] = []
    first_url_submit_exception: Exception | None = None
    url_failure_count = 0

    url_list = form_data.urls or []
    valid_url_count = len([url for url in url_list if url and str(url).strip()])
    for url in url_list:
        if not url or not str(url).strip():
            continue
        try:
            is_playlist = classify_playlist_url(str(url)).is_playlist
        except ValueError:
            is_playlist = False
        if is_playlist:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "playlist_preflight_required",
                    "message": "Playlist URLs must be inspected before job submission.",
                },
            )
    collection_id_value, planned_item_values, idempotency_key_values = _resolve_submit_bindings(
        url_count=valid_url_count,
        media_collection_id=media_collection_id,
        collection_id=collection_id,
        media_collection_item_id=media_collection_item_id,
        planned_item_ids=planned_item_ids,
        idempotency_key=idempotency_key,
        idempotency_keys=idempotency_keys,
    )

    url_index = 0
    for url in url_list:
        if not url or not str(url).strip():
            continue
        planned_item_id = planned_item_values[url_index] if planned_item_values else None
        item_idempotency_key = idempotency_key_values[url_index] if idempotency_key_values else None
        payload = {
            "batch_id": batch_id,
            "media_type": str(form_data.media_type),
            "source": str(url).strip(),
            "source_kind": "url",
            "input_ref": str(url).strip(),
            "options": options,
        }
        _apply_collection_binding_to_payload(
            payload,
            collection_id=collection_id_value,
            planned_item_id=planned_item_id,
            idempotency_key=item_idempotency_key,
        )
        try:
            row = _create_media_ingest_job(
                jm=jm,
                selected_queue=selected_queue,
                payload=payload,
                current_user=current_user,
                batch_id=batch_id,
                request_id=rid,
                trace_id=tp or None,
            )
            row_id = row.get("id")
            if row_id is None:
                raise ValueError(f"Job creation returned no id: {row!r}")
        except Exception as exc:
            url_failure_count += 1
            if first_url_submit_exception is None:
                first_url_submit_exception = exc
            error_message = _submit_failure_message(exc)
            _mark_collection_item_submit_failed(
                collections_db=collections_db,
                planned_item_id=planned_item_id,
                error_summary=error_message,
            )
            errors.append(f"{payload['source']}: {error_message}")
            url_index += 1
            continue
        jobs.append(
            MediaIngestJobItem(
                id=int(row_id),
                uuid=row.get("uuid"),
                source=payload["source"],
                source_kind="url",
                status=row.get("status"),
                collection_id=payload.get("collection_id"),
                planned_item_id=payload.get("planned_item_id"),
                idempotency_key=payload.get("idempotency_key"),
            )
        )
        url_index += 1

    if files:
        for upload in files:
            temp_dir_path = None
            try:
                with TempDirManager(prefix="media_ingest_job_", cleanup=False) as temp_dir:
                    temp_dir_path = str(temp_dir)
                    saved_files, file_errors = await save_uploaded_files(
                        [upload],
                        temp_dir=temp_dir,
                        validator=file_validator_instance,
                        expected_media_type_key=str(form_data.media_type),
                    )
                if file_errors:
                    for err in file_errors:
                        msg = err.get("error") or "Failed to stage upload"
                        errors.append(msg)
                    if temp_dir_path:
                        _cleanup_dir(temp_dir_path)
                    continue
                if not saved_files:
                    errors.append("Failed to stage upload")
                    if temp_dir_path:
                        _cleanup_dir(temp_dir_path)
                    continue

                saved = saved_files[0]
                source_path = str(saved.get("path"))
                original_filename = saved.get("original_filename")
                input_ref = saved.get("input_ref") or original_filename or source_path

                payload = {
                    "batch_id": batch_id,
                    "media_type": str(form_data.media_type),
                    "source": source_path,
                    "source_kind": "file",
                    "input_ref": input_ref,
                    "original_filename": original_filename,
                    "temp_dir": temp_dir_path,
                    "cleanup_temp_dir": True,
                    "options": options,
                }
                row = _create_media_ingest_job(
                    jm=jm,
                    selected_queue=selected_queue,
                    payload=payload,
                    current_user=current_user,
                    batch_id=batch_id,
                    request_id=rid,
                    trace_id=tp or None,
                )
                row_id = row.get("id")
                if row_id is None:
                    raise ValueError(f"Job creation returned no id: {row!r}")
                jobs.append(
                    MediaIngestJobItem(
                        id=int(row_id),
                        uuid=row.get("uuid"),
                        source=source_path,
                        source_kind="file",
                        status=row.get("status"),
                    )
                )
            except HTTPException:
                raise
            except Exception:
                logger.warning("Failed to stage upload for ingest jobs")
                errors.append("Upload staging failed")
                if temp_dir_path:
                    _cleanup_dir(temp_dir_path)

    if not jobs:
        if first_url_submit_exception is not None and valid_url_count == 1 and url_failure_count == 1 and not files:
            raise first_url_submit_exception
        if errors:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content=SubmitMediaIngestJobsResponse(
                    batch_id=batch_id,
                    jobs=[],
                    errors=errors,
                ).model_dump(),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid media sources supplied.",
        )

    return SubmitMediaIngestJobsResponse(batch_id=batch_id, jobs=jobs, errors=errors)


@router.get(
    "/ingest/jobs/{job_id}",
    response_model=MediaIngestJobStatus,
    summary="Get media ingest job status",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def get_media_ingest_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> MediaIngestJobStatus:
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "media_ingest":
        raise HTTPException(status_code=404, detail="Job not found")

    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="Not authorized for this job")

    return _job_to_status(job)


@router.get(
    "/ingest/jobs",
    response_model=MediaIngestJobListResponse,
    summary="List media ingest jobs for a batch",
    tags=["Media Ingestion Jobs"],
)
async def list_media_ingest_jobs(
    batch_id: str = Query(..., min_length=1, description="Batch identifier from submit response"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=MAX_MEDIA_INGEST_JOBS_OFFSET),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    _: None = Depends(check_rate_limit),
    jm: JobManager = Depends(get_job_manager),
) -> MediaIngestJobListResponse:
    owner_filter = None if _principal_has_admin_claims(principal) else str(current_user.id)
    # Backward compatibility: legacy rows may only have payload.batch_id, so the
    # shared batch collector remains the source of truth for both storage forms.
    window_end = offset + limit
    collected_jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=batch_id,
        owner_filter=owner_filter,
        limit=window_end + 1,
    )
    page_jobs = collected_jobs[offset:window_end]
    statuses = [_job_to_status(job) for job in page_jobs]
    has_more = len(collected_jobs) > window_end
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        count=len(statuses),
        has_more=has_more,
    )

    return MediaIngestJobListResponse(
        batch_id=batch_id,
        jobs=statuses,
        limit=limit,
        offset=offset,
        has_more=pagination.has_more,
        next_offset=pagination.next_offset,
        pagination=pagination,
    )


@router.get(
    "/ingest/jobs/events/stream",
    summary="Stream media ingest job events (SSE)",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Server-sent events stream of media ingest job updates",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_media_ingest_job_events(
    request: Request,
    batch_id: str | None = Query(
        None,
        min_length=1,
        description="Optional batch identifier to scope events to a single submit response",
    ),
    after_id: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> StreamingResponse:
    assert_may_start_work(request.app, "media.ingest.jobs.events.stream")
    is_admin = _principal_has_admin_claims(principal)
    owner_filter = None if is_admin else str(current_user.id)

    tracked_jobs: list[dict[str, Any]]
    if batch_id:
        tracked_jobs = _collect_jobs_for_batch(
            jm=jm,
            batch_id=batch_id,
            owner_filter=owner_filter,
        )
        if not tracked_jobs and not is_admin and _batch_exists_for_any_owner(jm=jm, batch_id=batch_id):
            raise HTTPException(status_code=403, detail="Not authorized for this batch")
    else:
        tracked_jobs = jm.list_jobs(
            domain="media_ingest",
            owner_user_id=owner_filter,
            limit=200,
            sort_by="created_at",
            sort_order="desc",
        )

    tracked_job_ids = {int(job.get("id")) for job in tracked_jobs if job.get("id") is not None}

    poll_interval = float(os.getenv("JOBS_EVENTS_POLL_INTERVAL", "1.0") or "1.0")
    max_duration_s: float | None = None
    try:
        if is_test_mode():
            max_duration_s = float(os.getenv("JOBS_SSE_TEST_MAX_SECONDS", "1.0") or "1.0")
    except (OSError, ValueError, TypeError):
        max_duration_s = 1.0

    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        max_duration_s=max_duration_s,
        labels={"component": "jobs", "endpoint": "media_ingest_events_sse"},
    )

    async def _producer() -> None:
        nonlocal after_id
        snapshot_jobs = [_job_to_status(job).model_dump(mode="json") for job in tracked_jobs]
        await stream.send_event(
            "snapshot",
            {
                "domain": "media_ingest",
                "batch_id": batch_id,
                "jobs": snapshot_jobs,
            },
        )

        while True:
            try:
                if getattr(stream, "_closed", False):
                    break
            except (AttributeError, RuntimeError):
                pass

            try:
                rows = jm.list_job_events_after(
                    after_id=int(after_id),
                    limit=500,
                    domain="media_ingest",
                    owner_user_id=owner_filter,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                rows = []

            if rows:
                for row in rows:
                    event_id = int(row.get("id"))
                    job_id = int(row.get("job_id"))
                    event_type = str(row.get("event_type"))
                    attrs_raw = row.get("attrs_json")
                    if tracked_job_ids and job_id not in tracked_job_ids:
                        after_id = event_id
                        continue
                    try:
                        attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else (attrs_raw or {})
                    except (TypeError, ValueError):
                        attrs = {}
                    await stream.send_event(
                        "job",
                        {
                            "event_id": event_id,
                            "job_id": job_id,
                            "event_type": event_type,
                            "attrs": attrs,
                        },
                        event_id=str(event_id),
                    )
                    after_id = event_id

            if tracked_job_ids:
                refreshed = [jm.get_job(job_id) for job_id in tracked_job_ids]
                if (
                    all(
                        (job or {}).get("status") in {"completed", "failed", "cancelled", "quarantined"}
                        for job in refreshed
                    )
                    and not rows
                ):
                    break

            await asyncio.sleep(poll_interval)

    async def _gen():
        producer = asyncio.create_task(_producer())
        try:
            async for line in stream.iter_sse():
                yield line
        finally:
            if not producer.done():
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    await producer

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete(
    "/ingest/jobs/{job_id}",
    response_model=CancelMediaIngestJobResponse,
    summary="Cancel a media ingest job",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def cancel_media_ingest_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
    reason: str | None = Query(None, description="Reason for cancellation"),
) -> CancelMediaIngestJobResponse:
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "media_ingest":
        raise HTTPException(status_code=404, detail="Job not found")

    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="Not authorized for this job")

    status_val = str(job.get("status") or "").lower()
    if status_val in {"completed", "failed", "cancelled", "quarantined"}:
        raise HTTPException(status_code=400, detail="Cannot cancel terminal job")

    ok = jm.cancel_job(int(job_id), reason=reason)
    if not ok:
        raise HTTPException(status_code=400, detail="Cancellation failed")

    return CancelMediaIngestJobResponse(
        success=True,
        job_id=int(job_id),
        status="cancelled",
        message="Job cancellation requested",
    )


@router.post(
    "/ingest/jobs/cancel",
    response_model=CancelMediaIngestBatchResponse,
    summary="Cancel media ingest jobs by batch/session id",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def cancel_media_ingest_jobs_batch(
    batch_id: str | None = Query(None, min_length=1, description="Batch identifier to cancel"),
    session_id: str | None = Query(
        None,
        min_length=1,
        description="Session identifier alias for batch-level cancellation",
    ),
    reason: str | None = Query(None, description="Reason for cancellation"),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> CancelMediaIngestBatchResponse:
    resolved_batch_id = _resolve_batch_or_session_id(batch_id=batch_id, session_id=session_id)
    is_admin = _principal_has_admin_claims(principal)
    owner_filter = None if is_admin else str(current_user.id)

    matched_jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=resolved_batch_id,
        owner_filter=owner_filter,
        limit=5000,
    )
    if not matched_jobs:
        if not is_admin and _batch_exists_for_any_owner(jm=jm, batch_id=resolved_batch_id):
            raise HTTPException(status_code=403, detail="Not authorized for this batch")
        raise HTTPException(status_code=404, detail="Batch not found")

    requested = len(matched_jobs)
    cancelled = 0
    already_terminal = 0
    failed = 0

    for job in matched_jobs:
        raw_job_id = job.get("id")
        if raw_job_id is None:
            failed += 1
            continue
        job_id = int(raw_job_id)
        status_value = str(job.get("status") or "").lower()
        if status_value in _TERMINAL_MEDIA_INGEST_JOB_STATUSES:
            already_terminal += 1
            continue
        if jm.cancel_job(job_id, reason=reason):
            cancelled += 1
            continue

        refreshed = jm.get_job(job_id) or {}
        refreshed_status = str(refreshed.get("status") or "").lower()
        if refreshed_status in _TERMINAL_MEDIA_INGEST_JOB_STATUSES:
            already_terminal += 1
        else:
            failed += 1

    if cancelled > 0:
        message = f"Cancellation requested for {cancelled} job(s)"
    elif already_terminal == requested:
        message = "All jobs already terminal"
    else:
        message = "No jobs were cancelled"

    return CancelMediaIngestBatchResponse(
        success=failed == 0,
        batch_id=resolved_batch_id,
        requested=requested,
        cancelled=cancelled,
        already_terminal=already_terminal,
        failed=failed,
        message=message,
    )


__all__ = ["router"]
