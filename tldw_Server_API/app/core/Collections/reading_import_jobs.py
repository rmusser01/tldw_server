from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Collections.reading_importers import (
    detect_import_source,
    parse_reading_import,
)
from tldw_Server_API.app.core.Collections.reading_service import ReadingService
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    normalize_output_storage_filename,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.testing import is_truthy

READING_IMPORT_DOMAIN = "reading"
READING_IMPORT_JOB_TYPE = "reading_import"


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid integer for {}: {!r}; using default={}", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("Out-of-range integer for {}: {}; minimum={}; using default={}", name, value, minimum, default)
        return default
    return value


MAX_READING_IMPORT_BYTES = _env_int("READING_IMPORT_MAX_BYTES", 10 * 1024 * 1024, minimum=1)


class ReadingImportJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.backoff_seconds = backoff_seconds


def reading_import_queue() -> str:
    queue = (os.getenv("READING_IMPORT_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def stage_reading_import_file(
    *,
    user_id: int | str,
    filename: str | None,
    raw_bytes: bytes,
) -> Path:
    imports_dir = DatabasePaths.get_user_reading_imports_dir(user_id)
    imports_dir_resolved = imports_dir.resolve()
    try:
        safe_name = normalize_output_storage_filename(
            filename or "reading_import",
            allow_absolute=False,
            reject_relative_with_separators=True,
            expand_user=False,
            base_resolved=imports_dir_resolved,
            check_relative_containment=True,
        )
    except InvalidStoragePathError:
        safe_name = "reading_import"
    if not safe_name:
        safe_name = "reading_import"
    token = uuid4().hex
    target = (imports_dir / f"reading_import_{token}_{safe_name}").resolve()
    try:
        target.relative_to(imports_dir_resolved)
    except ValueError:
        safe_name = "reading_import"
        target = (imports_dir / f"reading_import_{token}_{safe_name}").resolve()
    target.write_bytes(raw_bytes)
    return target


def resolve_reading_import_file(user_id: int | str, file_token: str) -> Path:
    token = str(file_token or "").strip()
    if not token:
        raise ReadingImportJobError("reading_import_missing_file", retryable=False)
    try:
        safe_token = normalize_output_storage_filename(
            token,
            allow_absolute=False,
            reject_relative_with_separators=True,
            expand_user=False,
        )
    except InvalidStoragePathError as exc:
        raise ReadingImportJobError("reading_import_invalid_file_token", retryable=False) from exc
    if safe_token != token:
        raise ReadingImportJobError("reading_import_invalid_file_token", retryable=False)

    imports_dir = DatabasePaths.get_user_reading_imports_dir(user_id).resolve()
    candidate = (imports_dir / safe_token).resolve()
    try:
        candidate.relative_to(imports_dir)
    except ValueError as exc:
        raise ReadingImportJobError("reading_import_path_escape", retryable=False) from exc
    if not candidate.exists():
        raise ReadingImportJobError("reading_import_file_not_found", retryable=False)
    return candidate


def _parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return is_truthy(str(value).strip().lower())


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        return int(DatabasePaths.get_single_user_id())
    return int(owner)


async def handle_reading_import_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_payload(job.get("payload"))
    user_id = _resolve_user_id(job, payload)
    file_token = payload.get("file_token") or payload.get("file_path")
    if not file_token:
        raise ReadingImportJobError("reading_import_missing_file", retryable=False)
    source = (payload.get("source") or "auto").strip().lower()
    merge_tags = _coerce_bool(payload.get("merge_tags"), True)
    filename = payload.get("filename")

    import_path = resolve_reading_import_file(user_id, str(file_token))
    try:
        raw_bytes = await asyncio.to_thread(import_path.read_bytes)
        if not raw_bytes:
            raise ReadingImportJobError("reading_import_empty", retryable=False)
        if len(raw_bytes) > MAX_READING_IMPORT_BYTES:
            raise ReadingImportJobError("reading_import_too_large", retryable=False)

        if source == "auto":
            source = detect_import_source(filename, raw_bytes)
        items = parse_reading_import(raw_bytes, source=source, filename=filename)
        service = ReadingService(user_id)
        result = await asyncio.to_thread(
            service.import_items,
            items=items,
            merge_tags=merge_tags,
            origin_type=source,
        )
        return {
            "source": source,
            "imported": result.imported,
            "updated": result.updated,
            "skipped": result.skipped,
            "errors": result.errors,
        }
    except ValueError as exc:
        raise ReadingImportJobError(f"reading_import_invalid:{exc}", retryable=False) from exc
    except ReadingImportJobError:
        raise
    except Exception as exc:
        logger.error(f"reading_import_job_failed: {exc}")
        raise ReadingImportJobError(f"reading_import_failed:{exc}", retryable=False) from exc
    finally:
        try:
            await asyncio.to_thread(import_path.unlink, missing_ok=True)
        except Exception as cleanup_exc:
            logger.debug(f"reading_import_job: cleanup failed for {import_path}: {cleanup_exc}")
