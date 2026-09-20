"""Source-free fenced reconciliation for standalone HTML generation receipts."""

from __future__ import annotations

import heapq
import hmac
import json
import os
import re
import sqlite3
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from tldw_Server_API.app.core.Slides.slides_db import (
    SchemaError,
    SlidesDatabase,
    SlidesDatabaseError,
    SlidesGenerationReconciliationRow,
)

_CURSOR_VERSION = 1
_MAX_CURSOR_BYTES = 1024
_MAX_DISCOVERY_BATCH = 1000
_MAX_SWEEP_LAG_SECONDS = 15 * 60
_UNSAFE_DATABASE_CODE = "standalone_html_slides_database_unsafe"
_MISSING_JOB_MARKER = "generation_receipt_unresolved_pending"
_MISSING_JOB_GRACE = timedelta(minutes=15)
_INPUT_TTL = timedelta(hours=24)
_TERMINAL_RETENTION = timedelta(days=30)
_JOBS_KEY_RE = re.compile(r"slides:v1:[0-9a-f]{64}\Z")
_SAFE_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_TERMINAL_RECEIPT_STATES = frozenset({"completed", "failed", "cancelled"})
_CURSOR_PHASES = frozenset({"active", "dormant"})
_DEFAULT_LEASE_SECONDS = 90
_DEFAULT_RECEIPT_PAGE = 100


class UnsafeSlidesDatabaseError(RuntimeError):
    """A canonical Slides database cannot be safely reconciled."""

    def __init__(self) -> None:
        super().__init__(_UNSAFE_DATABASE_CODE)


@dataclass(frozen=True, slots=True)
class DiscoveredSlidesDatabase:
    """One canonical owner/database pair containing no presentation content."""

    owner_user_id: str
    path: Path
    root_identity: tuple[int, int, int]
    user_directory_identity: tuple[int, int, int]
    file_identity: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class OwnerReconciliationResult:
    """Bounded source-free result for one owner receipt page."""

    processed_count: int
    last_receipt_id: str | None
    has_more: bool
    jobs_available: bool


@dataclass(frozen=True, slots=True)
class ReconciliationBatchResult:
    """One bounded fenced step suitable for an interruptible lifecycle loop."""

    leader: bool
    startup_ready: bool
    completed_pass: bool = False
    jobs_available: bool = True
    lost_leadership: bool = False
    processed_owner_user_id: str | None = None
    diagnostic_code: str | None = None
    local_sweep_state: Literal["not_run", "progressed", "completed", "blocked"] = "not_run"


def _canonical_owner_user_id(value: object, *, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError("reconciliation cursor is invalid")
    if value == "0" or value.startswith("0") or str(int(value)) != value:
        raise ValueError("reconciliation cursor is invalid")
    return value


def _canonical_receipt_id(value: object, *, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError("reconciliation cursor is invalid")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("reconciliation cursor is invalid") from None
    if str(parsed) != value:
        raise ValueError("reconciliation cursor is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ReconciliationCursor:
    """Bounded source-free progress through the canonical owner registry."""

    phase: Literal["active", "dormant"] = "active"
    after_owner_user_id: str | None = None
    owner_user_id: str | None = None
    after_receipt_id: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in _CURSOR_PHASES:
            raise ValueError("reconciliation cursor is invalid")
        after_owner = _canonical_owner_user_id(
            self.after_owner_user_id,
            allow_none=True,
        )
        owner = _canonical_owner_user_id(self.owner_user_id, allow_none=True)
        receipt = _canonical_receipt_id(self.after_receipt_id, allow_none=True)
        if receipt is not None and owner is None:
            raise ValueError("reconciliation cursor is invalid")
        if after_owner is not None and owner is not None:
            ordered_after_owner = owner <= after_owner if self.phase == "active" else int(owner) <= int(after_owner)
            if ordered_after_owner:
                raise ValueError("reconciliation cursor is invalid")


def encode_reconciliation_cursor(cursor: ReconciliationCursor) -> str:
    """Encode a validated progress cursor for the shared Jobs row."""
    if not isinstance(cursor, ReconciliationCursor):
        raise TypeError("cursor must be a ReconciliationCursor")
    encoded = json.dumps(
        {
            "after_owner_user_id": cursor.after_owner_user_id,
            "after_receipt_id": cursor.after_receipt_id,
            "owner_user_id": cursor.owner_user_id,
            "phase": cursor.phase,
            "v": _CURSOR_VERSION,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("ascii")) > _MAX_CURSOR_BYTES:
        raise ValueError("reconciliation cursor is invalid")
    return encoded


def decode_reconciliation_cursor(value: str) -> ReconciliationCursor:
    """Decode a closed, bounded progress cursor or fail closed."""
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_CURSOR_BYTES:
        raise ValueError("reconciliation cursor is invalid")
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, RecursionError, UnicodeError):
        raise ValueError("reconciliation cursor is invalid") from None
    expected_keys = {
        "after_owner_user_id",
        "after_receipt_id",
        "owner_user_id",
        "phase",
        "v",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload.get("v") != _CURSOR_VERSION:
        raise ValueError("reconciliation cursor is invalid")
    return ReconciliationCursor(
        phase=payload.get("phase"),
        after_owner_user_id=payload.get("after_owner_user_id"),
        owner_user_id=payload.get("owner_user_id"),
        after_receipt_id=payload.get("after_receipt_id"),
    )


def _read_only_schema_is_complete(path: Path) -> bool:
    uri = path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        return SlidesDatabase._schema_is_complete(connection)
    finally:
        connection.close()


def _resolve_registry_root(base_dir: str | Path) -> Path | None:
    """Resolve one existing registry root without accepting symlink traversal."""
    root = Path(os.path.abspath(os.fspath(Path(base_dir))))
    try:
        root_stat = os.stat(root, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise UnsafeSlidesDatabaseError() from None
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise UnsafeSlidesDatabaseError()
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise UnsafeSlidesDatabaseError() from None
    if resolved_root != root:
        raise UnsafeSlidesDatabaseError()
    return root


def _validate_candidate(
    *,
    base_dir: Path,
    owner_user_id: str,
    user_directory: Path,
) -> DiscoveredSlidesDatabase:
    """Validate one existing canonical owner database without creating it."""
    expected_user_directory = base_dir / owner_user_id
    db_path = user_directory / "Slides.db"
    try:
        relative = db_path.relative_to(base_dir)
        root_stat = os.stat(base_dir, follow_symlinks=False)
        user_stat = os.stat(user_directory, follow_symlinks=False)
        file_stat = os.stat(db_path, follow_symlinks=False)
        resolved_user_directory = user_directory.resolve(strict=True)
        resolved_db_path = db_path.resolve(strict=True)
        if (
            relative.parts != (owner_user_id, "Slides.db")
            or user_directory != expected_user_directory
            or resolved_user_directory != expected_user_directory
            or resolved_db_path != db_path
        ):
            raise UnsafeSlidesDatabaseError()
    except FileNotFoundError:
        raise UnsafeSlidesDatabaseError() from None
    except (OSError, RuntimeError, ValueError):
        raise UnsafeSlidesDatabaseError() from None
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(user_stat.st_mode)
        or not stat.S_ISDIR(user_stat.st_mode)
        or stat.S_ISLNK(file_stat.st_mode)
        or not stat.S_ISREG(file_stat.st_mode)
    ):
        raise UnsafeSlidesDatabaseError()
    try:
        if not _read_only_schema_is_complete(db_path):
            raise UnsafeSlidesDatabaseError()
    except (OSError, sqlite3.Error, ValueError):
        raise UnsafeSlidesDatabaseError() from None
    try:
        refreshed_root_stat = os.stat(base_dir, follow_symlinks=False)
        refreshed_user_stat = os.stat(user_directory, follow_symlinks=False)
        refreshed_file_stat = os.stat(db_path, follow_symlinks=False)
        if (
            user_directory.resolve(strict=True) != expected_user_directory
            or db_path.resolve(strict=True) != db_path
            or (refreshed_root_stat.st_dev, refreshed_root_stat.st_ino, stat.S_IFMT(refreshed_root_stat.st_mode))
            != (root_stat.st_dev, root_stat.st_ino, stat.S_IFMT(root_stat.st_mode))
            or (
                refreshed_user_stat.st_dev,
                refreshed_user_stat.st_ino,
                stat.S_IFMT(refreshed_user_stat.st_mode),
            )
            != (user_stat.st_dev, user_stat.st_ino, stat.S_IFMT(user_stat.st_mode))
            or (
                refreshed_file_stat.st_dev,
                refreshed_file_stat.st_ino,
                stat.S_IFMT(refreshed_file_stat.st_mode),
            )
            != (file_stat.st_dev, file_stat.st_ino, stat.S_IFMT(file_stat.st_mode))
        ):
            raise UnsafeSlidesDatabaseError()
    except FileNotFoundError:
        raise UnsafeSlidesDatabaseError() from None
    except (OSError, RuntimeError):
        raise UnsafeSlidesDatabaseError() from None
    return DiscoveredSlidesDatabase(
        owner_user_id=owner_user_id,
        path=db_path,
        root_identity=(root_stat.st_dev, root_stat.st_ino, stat.S_IFMT(root_stat.st_mode)),
        user_directory_identity=(
            user_stat.st_dev,
            user_stat.st_ino,
            stat.S_IFMT(user_stat.st_mode),
        ),
        file_identity=(file_stat.st_dev, file_stat.st_ino, stat.S_IFMT(file_stat.st_mode)),
    )


def discover_canonical_slides_databases(
    base_dir: str | Path,
    *,
    after_owner_user_id: str | None,
    limit: int,
) -> tuple[DiscoveredSlidesDatabase, ...]:
    """Return one bounded numeric-owner page without following symlinks."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_DISCOVERY_BATCH:
        raise ValueError("discovery limit is invalid")
    after_owner = _canonical_owner_user_id(after_owner_user_id, allow_none=True)
    root = _resolve_registry_root(base_dir)
    if root is None:
        return ()

    def canonical_directories():
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    name = entry.name
                    try:
                        owner = _canonical_owner_user_id(name, allow_none=False)
                    except ValueError:
                        continue
                    if after_owner is not None and int(owner) <= int(after_owner):
                        continue
                    try:
                        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                            raise UnsafeSlidesDatabaseError()
                    except OSError:
                        raise UnsafeSlidesDatabaseError() from None
                    db_path = Path(entry.path) / "Slides.db"
                    try:
                        db_stat = os.stat(db_path, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    except OSError:
                        raise UnsafeSlidesDatabaseError() from None
                    if stat.S_ISLNK(db_stat.st_mode) or not stat.S_ISREG(db_stat.st_mode):
                        raise UnsafeSlidesDatabaseError()
                    yield int(owner), owner, Path(entry.path)
        except OSError:
            raise UnsafeSlidesDatabaseError() from None

    selected = heapq.nsmallest(limit, canonical_directories(), key=lambda item: item[0])
    discovered: list[DiscoveredSlidesDatabase] = []
    for _numeric_owner, owner, user_directory in selected:
        discovered.append(
            _validate_candidate(
                base_dir=root,
                owner_user_id=owner,
                user_directory=user_directory,
            )
        )
    return tuple(discovered)


def _aware_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(timezone.utc)


def reconciliation_admission_ready(
    state: Mapping[str, Any],
    *,
    config_epoch: str,
    now: datetime,
    max_lag_seconds: int = _MAX_SWEEP_LAG_SECONDS,
) -> bool:
    """Return whether shared fenced cleanup is current enough for acquisition."""
    if (
        not isinstance(state, Mapping)
        or not isinstance(config_epoch, str)
        or not config_epoch
        or now.tzinfo is None
        or now.utcoffset() != timedelta(0)
        or isinstance(max_lag_seconds, bool)
        or not isinstance(max_lag_seconds, int)
        or max_lag_seconds <= 0
    ):
        return False
    holder = state.get("holder_uuid")
    lease_expires_at = _aware_utc(state.get("lease_expires_at"))
    if not isinstance(holder, str) or not holder or lease_expires_at is None or lease_expires_at <= now:
        return False
    if state.get("config_revision") != config_epoch or state.get("startup_complete_epoch") != config_epoch:
        return False
    last_complete_epoch = state.get("last_complete_epoch")
    lag = state.get("lag")
    if (
        isinstance(last_complete_epoch, bool)
        or not isinstance(last_complete_epoch, (int, float))
        or isinstance(lag, bool)
        or not isinstance(lag, int)
        or lag < 0
        or lag > max_lag_seconds
    ):
        return False
    age = now.timestamp() - float(last_complete_epoch)
    return 0 <= age <= max_lag_seconds


def _canonical_utc(value: object) -> datetime | None:
    parsed = _aware_utc(value)
    if parsed is None or parsed.replace(microsecond=0).isoformat() != value:
        return None
    return parsed


def _utc_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be an aware UTC datetime")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _valid_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if str(parsed) == value.lower() else None


def _matching_completed_presentation(row: SlidesGenerationReconciliationRow) -> bool:
    return bool(
        row.receipt_status == "completed"
        and row.presentation_id
        and row.presentation_exists
        and row.presentation_content_kind == "standalone_html"
        and row.job_uuid
        and row.presentation_generation_job_uuid == row.job_uuid
    )


def _logical_input_deadline(row: SlidesGenerationReconciliationRow) -> datetime | None:
    created_at = _canonical_utc(row.created_at)
    return None if created_at is None else created_at + _INPUT_TTL


def _job_identity(
    job: object,
    *,
    row: SlidesGenerationReconciliationRow,
) -> tuple[int | None, str, str] | None:
    if not isinstance(job, Mapping):
        return None
    job_uuid = _valid_uuid(job.get("uuid"))
    archived = job.get("archived") is True
    job_id = job.get("id")
    candidate_key = job.get("idempotency_key")
    stored_key = row.jobs_idempotency_key
    key_matches = bool(
        isinstance(candidate_key, str)
        and isinstance(stored_key, str)
        and _JOBS_KEY_RE.fullmatch(candidate_key) is not None
        and _JOBS_KEY_RE.fullmatch(stored_key) is not None
        and hmac.compare_digest(candidate_key, stored_key)
    )
    if (
        job_uuid is None
        or isinstance(job_id, bool)
        or (job_id is not None and (not isinstance(job_id, int) or job_id <= 0))
        or (job_id is None and not archived)
        or job.get("domain") != "slides"
        or job.get("queue") != "default"
        or job.get("job_type") != "presentation.generate"
        or job.get("owner_user_id") != row.owner_user_id
        or not key_matches
        or job.get("payload") != {"receipt_id": row.id}
        or (row.job_uuid is not None and row.job_uuid != job_uuid)
        or (not archived and row.job_id is not None and row.job_id != job_id)
    ):
        return None
    status = job.get("status")
    if not isinstance(status, str):
        return None
    return job_id, job_uuid, status


def _terminalize_receipt(
    slides_db: SlidesDatabase,
    row: SlidesGenerationReconciliationRow,
    *,
    job_uuid: str | None,
    status: str,
    error_code: str,
    error_message: str,
    now: datetime,
) -> bool:
    terminal_at = _utc_text(now)
    return slides_db.terminalize_generation_receipt(
        receipt_id=row.id,
        owner_user_id=row.owner_user_id,
        job_uuid=job_uuid,
        status=status,
        error_code=error_code,
        error_message=error_message,
        terminal_at=terminal_at,
        expires_at=_utc_text(now + _TERMINAL_RETENTION),
    )


def _terminalize_active_job(
    job_manager: Any,
    *,
    job: Mapping[str, Any],
    row: SlidesGenerationReconciliationRow,
    error_code: str,
    error_message: str,
) -> bool:
    expected_status = job.get("status")
    job_uuid = _valid_uuid(job.get("uuid"))
    if expected_status not in {"queued", "processing"} or job_uuid is None:
        return False
    job_id = job.get("id")
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        return False
    outcome = job_manager.terminalize_slides_generation_job_from_reconciler(
        job_uuid=job_uuid,
        owner_user_id=row.owner_user_id,
        expected_status=expected_status,
        status="failed",
        error_code=error_code,
        error_message=error_message,
        completion_token=f"slides-reconciler:{row.id}:{error_code}",
        job_id=job_id,
        require_processing_lease_expired=False,
    )
    return outcome in {"APPLIED", "IDEMPOTENT"}


def _reconcile_expired_jobs_state(
    job_manager: Any,
    row: SlidesGenerationReconciliationRow,
) -> bool:
    try:
        lookup_kwargs: dict[str, Any] = {
            "owner_user_id": row.owner_user_id,
            "idempotency_key": row.jobs_idempotency_key,
        }
        if row.job_uuid is not None:
            lookup_kwargs["expected_job_uuid"] = row.job_uuid
        job = job_manager.lookup_slides_generation_job(
            **lookup_kwargs,
        )
        if job is None and row.job_uuid is not None:
            job = job_manager.lookup_slides_generation_job(
                owner_user_id=row.owner_user_id,
                idempotency_key=row.jobs_idempotency_key,
            )
        if job is None:
            return True
        identity = _job_identity(job, row=row)
        if identity is None or not isinstance(job, Mapping):
            return False
        status = identity[2]
        if status in {"queued", "processing"}:
            return _terminalize_active_job(
                job_manager,
                job=job,
                row=row,
                error_code="generation_expired",
                error_message="Generation input expired.",
            )
        return status in {"failed", "cancelled", "quarantined"}
    except Exception:  # noqa: BLE001 - Jobs outages must not expose implementation detail
        return False


def _reconcile_expired_receipt(
    slides_db: SlidesDatabase,
    job_manager: Any,
    row: SlidesGenerationReconciliationRow,
    *,
    now_text: str,
) -> bool:
    expired = slides_db.terminalize_expired_generation_receipt(
        receipt_id=row.id,
        owner_user_id=row.owner_user_id,
        expected_job_uuid=row.job_uuid,
        as_of=now_text,
    )
    if not expired:
        return False
    return _reconcile_expired_jobs_state(job_manager, row)


def _map_terminal_job(
    slides_db: SlidesDatabase,
    row: SlidesGenerationReconciliationRow,
    *,
    job: Mapping[str, Any],
    job_uuid: str,
    now: datetime,
) -> None:
    job_status = job.get("status")
    if job_status == "cancelled":
        _terminalize_receipt(
            slides_db,
            row,
            job_uuid=job_uuid,
            status="cancelled",
            error_code="generation_cancelled",
            error_message="Generation was cancelled.",
            now=now,
        )
        return
    if job_status == "quarantined":
        _terminalize_receipt(
            slides_db,
            row,
            job_uuid=job_uuid,
            status="failed",
            error_code="generation_quarantined",
            error_message="Generation was quarantined.",
            now=now,
        )
        return
    if job_status == "failed":
        candidate_code = job.get("error_code")
        error_code = (
            candidate_code
            if isinstance(candidate_code, str) and _SAFE_ERROR_CODE_RE.fullmatch(candidate_code) is not None
            else "generation_failed"
        )
        _terminalize_receipt(
            slides_db,
            row,
            job_uuid=job_uuid,
            status="failed",
            error_code=error_code,
            error_message="Generation failed.",
            now=now,
        )
        return
    if job_status == "completed":
        _terminalize_receipt(
            slides_db,
            row,
            job_uuid=job_uuid,
            status="failed",
            error_code="generation_correlation_mismatch",
            error_message="Generation correlation failed.",
            now=now,
        )


def _reconcile_authoritative_job(
    slides_db: SlidesDatabase,
    row: SlidesGenerationReconciliationRow,
    *,
    job: Mapping[str, Any],
    identity: tuple[int | None, str, str],
    now: datetime,
) -> None:
    job_id, job_uuid, job_status = identity
    receipt_status = "running" if job_status == "processing" else "queued"
    if job_status not in {
        "queued",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "quarantined",
    }:
        _terminalize_receipt(
            slides_db,
            row,
            job_uuid=row.job_uuid,
            status="failed",
            error_code="generation_correlation_mismatch",
            error_message="Generation correlation failed.",
            now=now,
        )
        return
    repaired = slides_db.repair_generation_receipt_job(
        receipt_id=row.id,
        owner_user_id=row.owner_user_id,
        expected_job_uuid=row.job_uuid,
        job_id=job_id,
        job_uuid=job_uuid,
        receipt_status=receipt_status,
        updated_at=_utc_text(now),
    )
    if not repaired or job_status in {"queued", "processing"}:
        return
    _map_terminal_job(
        slides_db,
        row,
        job=job,
        job_uuid=job_uuid,
        now=now,
    )


def _reconcile_missing_job(
    slides_db: SlidesDatabase,
    row: SlidesGenerationReconciliationRow,
    *,
    now: datetime,
) -> None:
    first_missing = slides_db.mark_generation_receipt_job_missing(
        receipt_id=row.id,
        owner_user_id=row.owner_user_id,
        expected_job_uuid=row.job_uuid,
        observed_at=_utc_text(now),
    )
    parsed_first_missing = _canonical_utc(first_missing)
    if parsed_first_missing is None or now - parsed_first_missing < _MISSING_JOB_GRACE:
        return
    _terminalize_receipt(
        slides_db,
        row,
        job_uuid=row.job_uuid,
        status="failed",
        error_code="generation_receipt_unresolved",
        error_message="Generation job could not be resolved.",
        now=now,
    )


def reconcile_owner_generation_receipts(
    slides_db: SlidesDatabase,
    job_manager: Any,
    *,
    owner_user_id: str,
    now: datetime,
    after_receipt_id: str | None,
    limit: int,
) -> OwnerReconciliationResult:
    """Reconcile one bounded owner page without reading execution or HTML bodies."""
    if not isinstance(slides_db, SlidesDatabase):
        raise TypeError("slides_db must be a SlidesDatabase")
    now_text = _utc_text(now)
    rows = slides_db.list_generation_receipts_for_reconciliation(
        owner_user_id=owner_user_id,
        after_receipt_id=after_receipt_id,
        limit=limit,
    )
    jobs_available = True
    for row in rows:
        if row.owner_user_id != owner_user_id:
            raise SlidesDatabaseError("generation_owner_mismatch")
        if row.receipt_status == "completed":
            slides_db.delete_terminal_generation_input(
                receipt_id=row.id,
                owner_user_id=owner_user_id,
            )
            if not _matching_completed_presentation(row):
                jobs_available = False
            continue
        if row.receipt_status in {"failed", "cancelled"}:
            slides_db.delete_terminal_generation_input(
                receipt_id=row.id,
                owner_user_id=owner_user_id,
            )
            if row.error_code == "generation_expired":
                jobs_available = _reconcile_expired_jobs_state(job_manager, row) and jobs_available
            continue
        input_deadline = _logical_input_deadline(row)
        if input_deadline is not None and now >= input_deadline:
            jobs_available = (
                _reconcile_expired_receipt(
                    slides_db,
                    job_manager,
                    row,
                    now_text=now_text,
                )
                and jobs_available
            )
            continue
        if not row.input_exists or input_deadline is None or _canonical_utc(row.input_expires_at) != input_deadline:
            terminalized = _terminalize_receipt(
                slides_db,
                row,
                job_uuid=row.job_uuid,
                status="failed",
                error_code="generation_correlation_mismatch",
                error_message="Generation correlation failed.",
                now=now,
            )
            if terminalized:
                slides_db.delete_terminal_generation_input(
                    receipt_id=row.id,
                    owner_user_id=owner_user_id,
                )
            continue
        try:
            lookup_kwargs = {
                "owner_user_id": owner_user_id,
                "idempotency_key": row.jobs_idempotency_key,
            }
            if row.job_uuid is not None:
                lookup_kwargs["expected_job_uuid"] = row.job_uuid
            job = job_manager.lookup_slides_generation_job(**lookup_kwargs)
        except Exception:  # noqa: BLE001 - Jobs outages are a closed source-free state
            jobs_available = False
            continue
        if job is None and row.job_uuid is not None:
            try:
                job = job_manager.lookup_slides_generation_job(
                    owner_user_id=owner_user_id,
                    idempotency_key=row.jobs_idempotency_key,
                )
            except Exception:  # noqa: BLE001 - Jobs outages are a closed source-free state
                jobs_available = False
                continue
        if job is None:
            _reconcile_missing_job(slides_db, row, now=now)
            continue
        identity = _job_identity(job, row=row)
        if identity is None or not isinstance(job, Mapping):
            _terminalize_receipt(
                slides_db,
                row,
                job_uuid=row.job_uuid,
                status="failed",
                error_code="generation_correlation_mismatch",
                error_message="Generation correlation failed.",
                now=now,
            )
            continue
        _reconcile_authoritative_job(
            slides_db,
            row,
            job=job,
            identity=identity,
            now=now,
        )

    slides_db.delete_expired_generation_receipts(
        owner_user_id=owner_user_id,
        expires_before=now_text,
        limit=limit,
    )
    return OwnerReconciliationResult(
        processed_count=len(rows),
        last_receipt_id=rows[-1].id if rows else after_receipt_id,
        has_more=len(rows) == limit,
        jobs_available=jobs_available,
    )


def reconcile_owner_local_expiry(
    slides_db: SlidesDatabase,
    *,
    owner_user_id: str,
    now: datetime,
    after_receipt_id: str | None,
    limit: int,
) -> OwnerReconciliationResult:
    """Apply only deterministic local expiry/retention without consulting Jobs."""
    if not isinstance(slides_db, SlidesDatabase):
        raise TypeError("slides_db must be a SlidesDatabase")
    now_text = _utc_text(now)
    rows = slides_db.list_generation_receipts_for_reconciliation(
        owner_user_id=owner_user_id,
        after_receipt_id=after_receipt_id,
        limit=limit,
    )
    for row in rows:
        if row.owner_user_id != owner_user_id:
            raise SlidesDatabaseError("generation_owner_mismatch")
        if row.receipt_status in _TERMINAL_RECEIPT_STATES:
            slides_db.delete_terminal_generation_input(
                receipt_id=row.id,
                owner_user_id=owner_user_id,
            )
            continue
        input_deadline = _logical_input_deadline(row)
        if input_deadline is not None and now >= input_deadline:
            slides_db.terminalize_expired_generation_receipt(
                receipt_id=row.id,
                owner_user_id=owner_user_id,
                expected_job_uuid=row.job_uuid,
                as_of=now_text,
            )
            continue
        if not row.input_exists or input_deadline is None or _canonical_utc(row.input_expires_at) != input_deadline:
            terminalized = _terminalize_receipt(
                slides_db,
                row,
                job_uuid=row.job_uuid,
                status="failed",
                error_code="generation_correlation_mismatch",
                error_message="Generation correlation failed.",
                now=now,
            )
            if terminalized:
                slides_db.delete_terminal_generation_input(
                    receipt_id=row.id,
                    owner_user_id=owner_user_id,
                )
    slides_db.delete_expired_generation_receipts(
        owner_user_id=owner_user_id,
        expires_before=now_text,
        limit=limit,
    )
    return OwnerReconciliationResult(
        processed_count=len(rows),
        last_receipt_id=rows[-1].id if rows else after_receipt_id,
        has_more=len(rows) == limit,
        jobs_available=False,
    )


def _resolve_canonical_owner_database(
    base_dir: Path,
    *,
    owner_user_id: str,
) -> DiscoveredSlidesDatabase | None:
    owner = _canonical_owner_user_id(owner_user_id, allow_none=False)
    root = _resolve_registry_root(base_dir)
    if root is None:
        return None
    user_directory = root / owner
    try:
        user_stat = os.stat(user_directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise UnsafeSlidesDatabaseError() from None
    if stat.S_ISLNK(user_stat.st_mode) or not stat.S_ISDIR(user_stat.st_mode):
        raise UnsafeSlidesDatabaseError()
    db_path = user_directory / "Slides.db"
    try:
        db_stat = os.stat(db_path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise UnsafeSlidesDatabaseError() from None
    if stat.S_ISLNK(db_stat.st_mode) or not stat.S_ISREG(db_stat.st_mode):
        raise UnsafeSlidesDatabaseError()
    return _validate_candidate(
        base_dir=root,
        owner_user_id=owner,
        user_directory=user_directory,
    )


def _revalidate_discovered_database(
    discovered: DiscoveredSlidesDatabase,
) -> DiscoveredSlidesDatabase:
    """Repeat canonical validation immediately before a database factory open."""
    base_dir = _resolve_registry_root(discovered.path.parent.parent)
    if base_dir is None:
        raise UnsafeSlidesDatabaseError()
    revalidated = _validate_candidate(
        base_dir=base_dir,
        owner_user_id=discovered.owner_user_id,
        user_directory=discovered.path.parent,
    )
    if revalidated != discovered:
        raise UnsafeSlidesDatabaseError()
    return revalidated


class FencedStandaloneHtmlReconciler:
    """Run bounded active-first reconciliation under the shared Jobs fence."""

    def __init__(
        self,
        *,
        job_manager: Any,
        user_db_base_dir: str | Path,
        config_epoch: str,
        holder_uuid: str,
        now: Callable[[], datetime] | None = None,
        slides_db_factory: Callable[..., SlidesDatabase] = SlidesDatabase.open_existing_complete,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
        receipt_page_size: int = _DEFAULT_RECEIPT_PAGE,
    ) -> None:
        if not isinstance(config_epoch, str) or not config_epoch or len(config_epoch) > 512:
            raise ValueError("config_epoch must be a bounded nonblank string")
        if not isinstance(holder_uuid, str) or not holder_uuid.strip() or len(holder_uuid) > 128:
            raise ValueError("holder_uuid must be a bounded nonblank string")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if (
            isinstance(receipt_page_size, bool)
            or not isinstance(receipt_page_size, int)
            or not 1 <= receipt_page_size <= 500
        ):
            raise ValueError("receipt_page_size must be between 1 and 500")
        if not callable(slides_db_factory):
            raise TypeError("slides_db_factory must be callable")
        self._job_manager = job_manager
        self._base_dir = Path(user_db_base_dir)
        self._config_epoch = config_epoch
        self._holder_uuid = holder_uuid
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._slides_db_factory = slides_db_factory
        self._lease_seconds = lease_seconds
        self._receipt_page_size = receipt_page_size
        self._fencing_token: int | None = None
        self._cursor: ReconciliationCursor | None = None
        self._startup_complete_epoch: str | None = None
        self._last_complete_epoch: float | None = None
        self._pass_started_epoch: float | None = None
        self._proof_required = False
        self._proof_key_id: str | None = None
        self._proof_started_at: datetime | None = None
        self._proof_reference_count = 0
        self._last_jobs_available = False
        self._local_cursor = ReconciliationCursor(phase="dormant")

    def _utc_now(self) -> datetime:
        current = self._now()
        if not isinstance(current, datetime) or current.tzinfo is None or current.utcoffset() != timedelta(0):
            raise ValueError("reconciliation clock must return an aware UTC datetime")
        return current.astimezone(timezone.utc)

    @staticmethod
    def _retiring_key_id(raw_registry: object, *, config_epoch: str) -> str | None:
        if not isinstance(raw_registry, Mapping) or raw_registry.get("config_revision") != config_epoch:
            raise RuntimeError("standalone digest registry is unavailable")
        records = raw_registry.get("records")
        if not isinstance(records, (list, tuple)):
            raise RuntimeError("standalone digest registry is unavailable")
        retiring: list[tuple[datetime, str]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise RuntimeError("standalone digest registry is unavailable")
            key_id = record.get("key_id")
            state = record.get("state")
            if (
                not isinstance(key_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", key_id) is None
                or state not in {"current", "retiring"}
            ):
                raise RuntimeError("standalone digest registry is unavailable")
            if state == "retiring":
                retired_at = _aware_utc(record.get("retired_at"))
                if retired_at is None:
                    raise RuntimeError("standalone digest registry is unavailable")
                retiring.append((retired_at, key_id))
        return min(retiring)[1] if retiring else None

    def _start_pass(self, *, now: datetime, fresh: bool) -> None:
        registry = self._job_manager.load_slides_digest_key_registry()
        retiring_key_id = self._retiring_key_id(
            registry,
            config_epoch=self._config_epoch,
        )
        self._proof_required = retiring_key_id is not None
        self._proof_key_id = retiring_key_id if fresh else None
        self._proof_started_at = now if self._proof_key_id is not None else None
        self._proof_reference_count = 0
        if fresh:
            self._pass_started_epoch = now.timestamp()

    def _clear_leadership(self) -> None:
        self._fencing_token = None
        self._cursor = None
        self._proof_key_id = None
        self._proof_started_at = None
        self._proof_reference_count = 0
        self._pass_started_epoch = None

    def _shared_admission_ready(self, *, now: datetime) -> bool:
        if not self._last_jobs_available:
            return False
        try:
            readiness = self._job_manager.get_slides_generation_readiness()
            state = self._job_manager.get_slides_reconciliation_state()
            observed_at = max(now, self._utc_now())
        except Exception:  # noqa: BLE001 - admission is deliberately fail closed
            self._last_jobs_available = False
            return False
        return bool(
            isinstance(readiness, Mapping)
            and readiness.get("ready") is True
            and isinstance(state, Mapping)
            and state.get("diagnostic_code") is None
            and reconciliation_admission_ready(
                state,
                config_epoch=self._config_epoch,
                now=observed_at,
            )
        )

    def admission_ready(self) -> bool:
        """Return local-plus-shared handler admission for this process."""
        try:
            now = self._utc_now()
        except ValueError:
            return False
        return self._shared_admission_ready(now=now)

    def _result(
        self,
        *,
        now: datetime,
        leader: bool,
        completed_pass: bool = False,
        jobs_available: bool = True,
        lost_leadership: bool = False,
        processed_owner_user_id: str | None = None,
        diagnostic_code: str | None = None,
        local_sweep_state: Literal["not_run", "progressed", "completed", "blocked"] = "not_run",
    ) -> ReconciliationBatchResult:
        try:
            readiness_now = max(now, self._utc_now())
        except ValueError:
            readiness_now = None
        startup_ready = bool(
            readiness_now is not None
            and jobs_available
            and not lost_leadership
            and self._shared_admission_ready(now=readiness_now)
        )
        return ReconciliationBatchResult(
            leader=leader,
            startup_ready=startup_ready,
            completed_pass=completed_pass,
            jobs_available=jobs_available,
            lost_leadership=lost_leadership,
            processed_owner_user_id=processed_owner_user_id,
            diagnostic_code=diagnostic_code,
            local_sweep_state=local_sweep_state,
        )

    def _release_acquired_fence(self, *, fencing_token: int, now: datetime) -> None:
        """Best-effort release a valid fence whose acquired state is unusable."""
        try:
            self._job_manager.release_slides_reconciliation_lease(
                holder_uuid=self._holder_uuid,
                fencing_token=fencing_token,
                config_revision=self._config_epoch,
                now=now,
            )
        except Exception:  # noqa: BLE001 - an unusable acquisition still fails closed
            return

    def _acquire_if_needed(self, *, now: datetime) -> ReconciliationBatchResult | None:
        if self._fencing_token is not None:
            if self._cursor is None:
                try:
                    self._start_pass(now=now, fresh=True)
                except Exception:  # noqa: BLE001 - source-free Jobs/config failure
                    self._release_acquired_fence(
                        fencing_token=self._fencing_token,
                        now=now,
                    )
                    self._clear_leadership()
                    self._last_jobs_available = False
                    return self._result(
                        now=now,
                        leader=False,
                        jobs_available=False,
                        diagnostic_code="generation_reconciler_unavailable",
                    )
                self._cursor = ReconciliationCursor(phase="active")
            return None
        try:
            readiness = self._job_manager.get_slides_generation_readiness()
            if not isinstance(readiness, Mapping) or readiness.get("ready") is not True:
                self._last_jobs_available = False
                return self._result(
                    now=now,
                    leader=False,
                    jobs_available=False,
                    diagnostic_code="generation_reconciler_unavailable",
                )
            acquired = self._job_manager.acquire_slides_reconciliation_lease(
                holder_uuid=self._holder_uuid,
                lease_seconds=self._lease_seconds,
                config_revision=self._config_epoch,
                now=now,
            )
        except Exception:  # noqa: BLE001 - shared Jobs failures close admission
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code="generation_reconciler_unavailable",
            )
        self._last_jobs_available = True
        if acquired is None:
            return self._result(now=now, leader=False)
        if not isinstance(acquired, Mapping):
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code="generation_reconciler_unavailable",
            )
        fencing_token = acquired.get("fencing_token")
        raw_cursor = acquired.get("cursor")
        if (
            isinstance(fencing_token, bool)
            or not isinstance(fencing_token, int)
            or fencing_token <= 0
            or acquired.get("holder_uuid") != self._holder_uuid
            or acquired.get("config_revision") != self._config_epoch
        ):
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code="generation_reconciler_unavailable",
            )
        if raw_cursor is not None and not isinstance(raw_cursor, str):
            self._release_acquired_fence(fencing_token=fencing_token, now=now)
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code="generation_reconciler_unavailable",
            )
        try:
            cursor = (
                decode_reconciliation_cursor(raw_cursor)
                if isinstance(raw_cursor, str)
                else ReconciliationCursor(phase="active")
            )
            self._start_pass(now=now, fresh=raw_cursor is None)
        except Exception:  # noqa: BLE001 - malformed shared progress fails closed
            self._release_acquired_fence(fencing_token=fencing_token, now=now)
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code="generation_reconciler_unavailable",
            )
        self._fencing_token = fencing_token
        self._cursor = cursor
        startup_epoch = acquired.get("startup_complete_epoch")
        self._startup_complete_epoch = startup_epoch if startup_epoch == self._config_epoch else None
        last_complete_epoch = acquired.get("last_complete_epoch")
        self._last_complete_epoch = (
            float(last_complete_epoch)
            if not isinstance(last_complete_epoch, bool)
            and isinstance(last_complete_epoch, (int, float))
            and float(last_complete_epoch) >= 0
            else None
        )
        if raw_cursor is not None and self._last_complete_epoch is None:
            raw_lag = acquired.get("lag")
            self._pass_started_epoch = (
                now.timestamp() - float(raw_lag)
                if not isinstance(raw_lag, bool) and isinstance(raw_lag, (int, float)) and float(raw_lag) >= 0
                else now.timestamp()
            )
        return None

    def _renew(self, *, now: datetime) -> bool:
        if self._fencing_token is None:
            return False
        try:
            renewed = self._job_manager.renew_slides_reconciliation_lease(
                holder_uuid=self._holder_uuid,
                fencing_token=self._fencing_token,
                config_revision=self._config_epoch,
                lease_seconds=self._lease_seconds,
                now=now,
            )
        except Exception:  # noqa: BLE001 - lease store failure is leadership loss
            renewed = False
        if renewed:
            self._last_jobs_available = True
            return True
        self._last_jobs_available = False
        self._clear_leadership()
        return False

    def _lag(self, *, now: datetime) -> int:
        anchor = self._last_complete_epoch
        if anchor is None:
            anchor = self._pass_started_epoch
        if anchor is None:
            return 0
        return max(0, int(now.timestamp() - anchor))

    def _checkpoint(
        self,
        *,
        now: datetime,
        cursor: ReconciliationCursor | None,
        completed: bool,
        startup_complete_epoch: str | None,
    ) -> bool:
        if self._fencing_token is None:
            return False
        try:
            now = max(now, self._utc_now())
        except ValueError:
            self._last_jobs_available = False
            self._clear_leadership()
            return False
        kwargs: dict[str, Any] = {
            "holder_uuid": self._holder_uuid,
            "fencing_token": self._fencing_token,
            "config_revision": self._config_epoch,
            "cursor": None if completed else encode_reconciliation_cursor(cursor),
            "startup_complete_epoch": startup_complete_epoch,
            "last_complete_epoch": now.timestamp() if completed else self._last_complete_epoch,
            "lag": 0 if completed else self._lag(now=now),
            "now": now,
            "completed": completed,
        }
        if self._proof_key_id is not None:
            kwargs.update(
                sweep_key_id=self._proof_key_id,
                sweep_started_at=self._proof_started_at,
                unexpired_reference_count=self._proof_reference_count,
            )
        try:
            published = self._job_manager.checkpoint_slides_reconciliation(**kwargs)
        except Exception:  # noqa: BLE001 - publication failure loses the fence locally
            published = False
        if not published:
            self._last_jobs_available = False
            self._clear_leadership()
            return False
        self._last_jobs_available = True
        self._cursor = cursor
        self._startup_complete_epoch = startup_complete_epoch
        if completed:
            self._last_complete_epoch = now.timestamp()
            self._pass_started_epoch = None
        return True

    def _active_owner(self, cursor: ReconciliationCursor) -> str | None:
        if cursor.owner_user_id is not None:
            return cursor.owner_user_id
        owners = self._job_manager.list_active_slides_generation_owner_ids(
            after_owner_user_id=cursor.after_owner_user_id,
            limit=1,
        )
        if not isinstance(owners, (list, tuple)) or len(owners) > 1:
            raise RuntimeError("active Slides owner registry is unavailable")
        if not owners:
            return None
        owner = _canonical_owner_user_id(owners[0], allow_none=False)
        if cursor.after_owner_user_id is not None and owner <= cursor.after_owner_user_id:
            raise RuntimeError("active Slides owner registry is unavailable")
        return owner

    def _open_discovered_database(
        self,
        discovered: DiscoveredSlidesDatabase,
    ) -> SlidesDatabase:
        """Open a revalidated database without allowing creation or migration."""
        try:
            slides_db = self._slides_db_factory(
                db_path=discovered.path,
                client_id=discovered.owner_user_id,
                expected_file_identity=discovered.file_identity,
                expected_directory_identities=(
                    (discovered.path.parent.parent, discovered.root_identity),
                    (discovered.path.parent, discovered.user_directory_identity),
                ),
            )
        except (OSError, sqlite3.Error, SchemaError, ValueError):
            raise UnsafeSlidesDatabaseError() from None
        if not isinstance(slides_db, SlidesDatabase):
            raise TypeError("slides_db_factory must return SlidesDatabase")
        return slides_db

    def _process_owner(
        self,
        *,
        discovered: DiscoveredSlidesDatabase | None,
        owner_user_id: str,
        cursor: ReconciliationCursor,
        now: datetime,
    ) -> OwnerReconciliationResult:
        if discovered is None:
            return OwnerReconciliationResult(
                processed_count=0,
                last_receipt_id=cursor.after_receipt_id,
                has_more=False,
                jobs_available=True,
            )
        if discovered.owner_user_id != owner_user_id:
            raise UnsafeSlidesDatabaseError()
        discovered = _revalidate_discovered_database(discovered)
        slides_db = self._open_discovered_database(discovered)
        try:
            result = reconcile_owner_generation_receipts(
                slides_db,
                self._job_manager,
                owner_user_id=owner_user_id,
                now=now,
                after_receipt_id=cursor.after_receipt_id,
                limit=self._receipt_page_size,
            )
            if cursor.phase == "dormant" and not result.has_more and self._proof_key_id is not None:
                self._proof_reference_count += slides_db.count_unexpired_generation_receipts_for_digest_key(
                    owner_user_id=owner_user_id,
                    digest_key_id=self._proof_key_id,
                    as_of=_utc_text(now),
                )
            return result
        finally:
            slides_db.close_connection()

    def _run_active_batch(
        self,
        *,
        cursor: ReconciliationCursor,
        now: datetime,
    ) -> ReconciliationBatchResult:
        owner = self._active_owner(cursor)
        if owner is None:
            next_cursor = ReconciliationCursor(phase="dormant")
            if not self._checkpoint(
                now=now,
                cursor=next_cursor,
                completed=False,
                startup_complete_epoch=self._startup_complete_epoch,
            ):
                return self._result(
                    now=now,
                    leader=False,
                    lost_leadership=True,
                    jobs_available=False,
                )
            return self._result(now=now, leader=True)
        discovered = _resolve_canonical_owner_database(
            self._base_dir,
            owner_user_id=owner,
        )
        result = self._process_owner(
            discovered=discovered,
            owner_user_id=owner,
            cursor=cursor,
            now=now,
        )
        if not result.jobs_available:
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=True,
                jobs_available=False,
                processed_owner_user_id=owner,
                diagnostic_code="generation_reconciler_unavailable",
            )
        next_cursor = (
            ReconciliationCursor(
                phase="active",
                after_owner_user_id=cursor.after_owner_user_id,
                owner_user_id=owner,
                after_receipt_id=result.last_receipt_id,
            )
            if result.has_more
            else ReconciliationCursor(
                phase="active",
                after_owner_user_id=owner,
            )
        )
        if not self._checkpoint(
            now=now,
            cursor=next_cursor,
            completed=False,
            startup_complete_epoch=self._startup_complete_epoch,
        ):
            return self._result(
                now=now,
                leader=False,
                lost_leadership=True,
                jobs_available=False,
                processed_owner_user_id=owner,
            )
        return self._result(
            now=now,
            leader=True,
            processed_owner_user_id=owner,
        )

    def _run_dormant_batch(
        self,
        *,
        cursor: ReconciliationCursor,
        now: datetime,
    ) -> ReconciliationBatchResult:
        if cursor.owner_user_id is not None:
            owner = cursor.owner_user_id
            discovered = _resolve_canonical_owner_database(
                self._base_dir,
                owner_user_id=owner,
            )
            if discovered is None:
                raise UnsafeSlidesDatabaseError()
        else:
            page = discover_canonical_slides_databases(
                self._base_dir,
                after_owner_user_id=cursor.after_owner_user_id,
                limit=1,
            )
            if not page:
                startup_epoch = (
                    self._config_epoch
                    if not self._proof_required or self._proof_key_id is not None
                    else self._startup_complete_epoch
                )
                if not self._checkpoint(
                    now=now,
                    cursor=None,
                    completed=True,
                    startup_complete_epoch=startup_epoch,
                ):
                    return self._result(
                        now=now,
                        leader=False,
                        lost_leadership=True,
                        jobs_available=False,
                    )
                self._cursor = None
                return self._result(
                    now=now,
                    leader=True,
                    completed_pass=True,
                )
            discovered = page[0]
            owner = discovered.owner_user_id
        result = self._process_owner(
            discovered=discovered,
            owner_user_id=owner,
            cursor=cursor,
            now=now,
        )
        if not result.jobs_available:
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=True,
                jobs_available=False,
                processed_owner_user_id=owner,
                diagnostic_code="generation_reconciler_unavailable",
            )
        next_cursor = (
            ReconciliationCursor(
                phase="dormant",
                after_owner_user_id=cursor.after_owner_user_id,
                owner_user_id=owner,
                after_receipt_id=result.last_receipt_id,
            )
            if result.has_more
            else ReconciliationCursor(
                phase="dormant",
                after_owner_user_id=owner,
            )
        )
        if not self._checkpoint(
            now=now,
            cursor=next_cursor,
            completed=False,
            startup_complete_epoch=self._startup_complete_epoch,
        ):
            return self._result(
                now=now,
                leader=False,
                lost_leadership=True,
                jobs_available=False,
                processed_owner_user_id=owner,
            )
        return self._result(
            now=now,
            leader=True,
            processed_owner_user_id=owner,
        )

    def _run_local_expiry_batch(
        self,
        *,
        now: datetime,
        diagnostic_code: str,
    ) -> ReconciliationBatchResult:
        cursor = self._local_cursor
        try:
            if cursor.owner_user_id is not None:
                owner = cursor.owner_user_id
                discovered = _resolve_canonical_owner_database(
                    self._base_dir,
                    owner_user_id=owner,
                )
                if discovered is None:
                    raise UnsafeSlidesDatabaseError()
            else:
                page = discover_canonical_slides_databases(
                    self._base_dir,
                    after_owner_user_id=cursor.after_owner_user_id,
                    limit=1,
                )
                if not page:
                    self._local_cursor = ReconciliationCursor(phase="dormant")
                    return self._result(
                        now=now,
                        leader=False,
                        jobs_available=False,
                        diagnostic_code=diagnostic_code,
                        local_sweep_state="completed",
                    )
                discovered = page[0]
                owner = discovered.owner_user_id
            discovered = _revalidate_discovered_database(discovered)
            slides_db = self._open_discovered_database(discovered)
            try:
                result = reconcile_owner_local_expiry(
                    slides_db,
                    owner_user_id=owner,
                    now=now,
                    after_receipt_id=cursor.after_receipt_id,
                    limit=self._receipt_page_size,
                )
            finally:
                slides_db.close_connection()
            self._local_cursor = (
                ReconciliationCursor(
                    phase="dormant",
                    after_owner_user_id=cursor.after_owner_user_id,
                    owner_user_id=owner,
                    after_receipt_id=result.last_receipt_id,
                )
                if result.has_more
                else ReconciliationCursor(
                    phase="dormant",
                    after_owner_user_id=owner,
                )
            )
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                processed_owner_user_id=owner,
                diagnostic_code=diagnostic_code,
                local_sweep_state="progressed",
            )
        except UnsafeSlidesDatabaseError:
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code=_UNSAFE_DATABASE_CODE,
                local_sweep_state="blocked",
            )
        except Exception:  # noqa: BLE001 - local cleanup remains fail closed/redacted
            return self._result(
                now=now,
                leader=False,
                jobs_available=False,
                diagnostic_code=diagnostic_code,
                local_sweep_state="blocked",
            )

    def _continue_local_expiry_after_outage(
        self,
        *,
        now: datetime,
        fenced_result: ReconciliationBatchResult,
    ) -> ReconciliationBatchResult:
        """Advance an independent local cursor without publishing fenced progress."""
        if fenced_result.diagnostic_code == _UNSAFE_DATABASE_CODE:
            return self._result(
                now=now,
                leader=fenced_result.leader,
                jobs_available=False,
                lost_leadership=fenced_result.lost_leadership,
                diagnostic_code=_UNSAFE_DATABASE_CODE,
                local_sweep_state="blocked",
            )
        local_result = self._run_local_expiry_batch(
            now=now,
            diagnostic_code=(fenced_result.diagnostic_code or "generation_reconciler_unavailable"),
        )
        return self._result(
            now=now,
            leader=fenced_result.leader,
            jobs_available=False,
            lost_leadership=fenced_result.lost_leadership,
            processed_owner_user_id=local_result.processed_owner_user_id,
            diagnostic_code=local_result.diagnostic_code,
            local_sweep_state=local_result.local_sweep_state,
        )

    def run_batch(self) -> ReconciliationBatchResult:
        """Run at most one owner receipt page and one fenced checkpoint."""
        now = self._utc_now()
        acquisition_result = self._acquire_if_needed(now=now)
        if acquisition_result is not None:
            if not acquisition_result.jobs_available:
                return self._run_local_expiry_batch(
                    now=now,
                    diagnostic_code=(acquisition_result.diagnostic_code or "generation_reconciler_unavailable"),
                )
            return acquisition_result
        if not self._renew(now=now):
            return self._continue_local_expiry_after_outage(
                now=now,
                fenced_result=self._result(
                    now=now,
                    leader=False,
                    jobs_available=False,
                    lost_leadership=True,
                ),
            )
        cursor = self._cursor
        if cursor is None:
            raise RuntimeError("reconciliation cursor is unavailable")
        try:
            result = (
                self._run_active_batch(cursor=cursor, now=now)
                if cursor.phase == "active"
                else self._run_dormant_batch(cursor=cursor, now=now)
            )
            if not result.jobs_available:
                return self._continue_local_expiry_after_outage(
                    now=now,
                    fenced_result=result,
                )
            return result
        except UnsafeSlidesDatabaseError:
            self._last_jobs_available = False
            return self._result(
                now=now,
                leader=True,
                jobs_available=False,
                diagnostic_code=_UNSAFE_DATABASE_CODE,
                local_sweep_state="blocked",
            )
        except Exception:  # noqa: BLE001 - lifecycle observes one redacted closed state
            self._last_jobs_available = False
            return self._continue_local_expiry_after_outage(
                now=now,
                fenced_result=self._result(
                    now=now,
                    leader=True,
                    jobs_available=False,
                    diagnostic_code="generation_reconciler_unavailable",
                ),
            )

    def run_local_expiry_batch(self) -> ReconciliationBatchResult:
        """Run one unfenced local-only cleanup step without Jobs publication."""
        now = self._utc_now()
        self._last_jobs_available = False
        return self._run_local_expiry_batch(
            now=now,
            diagnostic_code="generation_reconciler_unavailable",
        )

    def release(self) -> bool:
        """Best-effort fenced lease release for deterministic shutdown."""
        fencing_token = self._fencing_token
        if fencing_token is None:
            return False
        try:
            released = self._job_manager.release_slides_reconciliation_lease(
                holder_uuid=self._holder_uuid,
                fencing_token=fencing_token,
                config_revision=self._config_epoch,
                now=self._utc_now(),
            )
        except Exception:  # noqa: BLE001 - shutdown remains best effort
            released = False
        self._clear_leadership()
        self._last_jobs_available = False
        return bool(released)


__all__ = [
    "DiscoveredSlidesDatabase",
    "FencedStandaloneHtmlReconciler",
    "OwnerReconciliationResult",
    "ReconciliationBatchResult",
    "ReconciliationCursor",
    "UnsafeSlidesDatabaseError",
    "decode_reconciliation_cursor",
    "discover_canonical_slides_databases",
    "encode_reconciliation_cursor",
    "reconcile_owner_generation_receipts",
    "reconcile_owner_local_expiry",
    "reconciliation_admission_ready",
]
