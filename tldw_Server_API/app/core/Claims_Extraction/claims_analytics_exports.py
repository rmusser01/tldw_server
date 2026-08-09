"""Deterministic request normalization and rendering for Claims analytics exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.config import settings

DEFAULT_EXPORT_MAX_BYTES = 10_485_760
DEFAULT_EXPORT_ORPHAN_GRACE_SEC = 300
DEFAULT_EXPORT_RETENTION_HOURS = 24
CLEANUP_ROTATION_SECONDS = 300
EXPORT_SCAN_PAGE_SIZE = 1000
EXPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CSV_COLUMNS = ("id", "event_type", "severity", "created_at", "payload_json")

_OWNER_ID_RE = re.compile(r"^[1-9][0-9]*$")
_SCALAR_FILTERS = ("event_type", "severity", "provider", "model")
_EVENT_COLUMNS = ("id", "user_id", "event_type", "severity", "created_at", "delivered_at")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_NORMALIZED_REQUEST_KEYS = {
    "owner_user_id",
    "format",
    "filters",
    "pagination",
    "snapshot_at",
}
_EXPORT_TRANSITIONS = {
    ("queued", "processing"),
    ("queued", "failed"),
    ("processing", "ready"),
    ("processing", "failed"),
    ("failed", "processing"),
    ("ready", "ready"),
}
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "quarantined"}
_EXPORT_JOB_TYPE = "claims_generate_analytics_export"
_EXPORT_BATCH_GROUP_PREFIX = "claims-analytics-export:"


class ClaimsAnalyticsExportError(RuntimeError):
    """Safe domain failure surfaced by Claims analytics export operations."""

    def __init__(
        self,
        public_message: str,
        *,
        code: str,
        retryable: bool = False,
        http_status: int = 400,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.code = code
        self.retryable = retryable
        self.http_status = http_status


def _invalid_payload_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Invalid Claims analytics export request.",
        code="claims_export_invalid_payload",
    )


def _owner_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Invalid Claims analytics export owner.",
        code="claims_owner_scope_violation",
    )


def _unsupported_format_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Unsupported Claims analytics export format.",
        code="claims_export_unsupported_format",
    )


def _serialization_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Claims analytics export could not be serialized.",
        code="claims_export_serialization_failed",
    )


def _missing_artifact_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Claims analytics export was not found.",
        code="claims_export_missing",
        http_status=404,
    )


def _invalid_artifact_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Claims analytics export artifact is invalid.",
        code="claims_export_invalid_artifact",
    )


def _storage_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Claims analytics export storage is temporarily unavailable.",
        code="claims_export_storage_unavailable",
        retryable=True,
        http_status=503,
    )


def _enqueue_failed_error() -> ClaimsAnalyticsExportError:
    return ClaimsAnalyticsExportError(
        "Claims analytics export could not be queued.",
        code="claims_export_enqueue_failed",
        http_status=503,
    )


def _canonical_owner_id(value: Any) -> str:
    if not isinstance(value, str) or _OWNER_ID_RE.fullmatch(value) is None:
        raise _owner_error()
    return value


def _parse_iso8601(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise _invalid_payload_error()
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_payload_error() from exc


def _normalize_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        raise _invalid_payload_error()
    try:
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid_payload_error() from exc


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce_pagination_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _setting_value(settings_obj: Any, key: str) -> Any:
    if settings_obj is None:
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        source = settings
    else:
        source = settings_obj
    try:
        if isinstance(source, Mapping):
            return source.get(key)
        return getattr(source, key, None)
    except Exception:  # noqa: BLE001 - hostile settings adapters must fall back safely.
        return None


def _integer_setting(settings_obj: Any, key: str) -> int | None:
    value = _setting_value(settings_obj, key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if re.fullmatch(r"[+-]?[0-9]+", normalized) is None:
        return None
    try:
        return int(normalized)
    except (ValueError, OverflowError):
        return None


def _positive_int_setting(settings_obj: Any, key: str, default: int) -> int:
    parsed = _integer_setting(settings_obj, key)
    if parsed is None or parsed <= 0:
        return default
    return parsed


def _non_negative_int_setting(settings_obj: Any, key: str, default: int) -> int:
    parsed = _integer_setting(settings_obj, key)
    if parsed is None or parsed < 0:
        return default
    return parsed


def normalize_export_request(
    payload: Any,
    *,
    owner_user_id: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize an analytics export request for persistence."""
    owner = _canonical_owner_id(owner_user_id)
    if not isinstance(payload, dict):
        raise _invalid_payload_error()

    raw_format = payload.get("format", "json")
    if not isinstance(raw_format, str):
        raise _unsupported_format_error()
    normalized_format = raw_format.lower()
    if normalized_format not in {"json", "csv"}:
        raise _unsupported_format_error()

    raw_filters = payload.get("filters", {})
    raw_pagination = payload.get("pagination", {})
    if not isinstance(raw_filters, dict) or not isinstance(raw_pagination, dict):
        raise _invalid_payload_error()

    snapshot = _normalize_now(now)
    normalized_filters: dict[str, Any] = {}
    for key in _SCALAR_FILTERS:
        value = raw_filters.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise _invalid_payload_error()
        normalized_filters[key] = value

    start = None
    if raw_filters.get("start_time") is not None:
        start = _parse_iso8601(raw_filters["start_time"])
        normalized_filters["start_time"] = _format_utc(start)

    caller_end = None
    if raw_filters.get("end_time") is not None:
        caller_end = _parse_iso8601(raw_filters["end_time"])
    effective_end = min(caller_end, snapshot) if caller_end is not None else snapshot
    if start is not None and start > effective_end:
        raise _invalid_payload_error()
    normalized_filters["end_time"] = _format_utc(effective_end)

    limit = _coerce_pagination_int(raw_pagination.get("limit", 1000), 1000)
    offset = _coerce_pagination_int(raw_pagination.get("offset", 0), 0)
    normalized_pagination = {
        "limit": max(1, min(10_000, limit)),
        "offset": max(0, offset),
    }

    return {
        "owner_user_id": owner,
        "format": normalized_format,
        "filters": normalized_filters,
        "pagination": normalized_pagination,
        "snapshot_at": _format_utc(snapshot),
    }


def validate_export_id(value: Any) -> str:
    """Return a validated server-generated export ID."""
    if not isinstance(value, str) or EXPORT_ID_RE.fullmatch(value) is None:
        raise _invalid_payload_error()
    return value


def export_max_bytes(settings_obj: Any = None) -> int:
    """Resolve the positive UTF-8 export byte limit from mapping or object settings."""
    return _positive_int_setting(
        settings_obj,
        "CLAIMS_ANALYTICS_EXPORT_MAX_BYTES",
        DEFAULT_EXPORT_MAX_BYTES,
    )


def orphan_grace_seconds(settings_obj: Any = None) -> int:
    """Resolve the non-negative orphan reconciliation grace period in seconds."""
    return _non_negative_int_setting(
        settings_obj,
        "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC",
        DEFAULT_EXPORT_ORPHAN_GRACE_SEC,
    )


def export_retention_hours(settings_obj: Any = None) -> float:
    """Resolve the positive completed-export retention period in hours."""
    value = _setting_value(settings_obj, "CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS")
    if isinstance(value, bool):
        return DEFAULT_EXPORT_RETENTION_HOURS
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_EXPORT_RETENTION_HOURS
    return parsed if math.isfinite(parsed) and parsed > 0 else DEFAULT_EXPORT_RETENTION_HOURS


def spreadsheet_safe(value: Any) -> Any:
    """Prefix formula-like spreadsheet strings with a literal single quote."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def _database_timestamp(value: Any) -> tuple[datetime, str]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _serialization_error() from exc
    else:
        raise _serialization_error()
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        normalized = parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _serialization_error() from exc
    return normalized, _format_utc(normalized)


def _decode_event(
    row: Any,
    *,
    owner_user_id: str,
    previous_key: tuple[datetime, int] | None,
) -> tuple[dict[str, Any], tuple[datetime, int], Any]:
    if not isinstance(row, Mapping):
        raise _serialization_error()
    event_id = row.get("id")
    if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
        raise _serialization_error()
    event_type = row.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        raise _serialization_error()
    if "user_id" not in row or str(row.get("user_id")) != owner_user_id:
        raise _serialization_error()
    if "created_at" not in row:
        raise _serialization_error()
    raw_created_at = row["created_at"]
    created_at, canonical_created_at = _database_timestamp(raw_created_at)
    order_key = (created_at, event_id)
    if previous_key is not None and order_key <= previous_key:
        raise _serialization_error()

    event = {column: row.get(column) for column in _EVENT_COLUMNS if column in row}
    event["created_at"] = canonical_created_at
    if event.get("delivered_at") is not None:
        _, event["delivered_at"] = _database_timestamp(event["delivered_at"])
    raw_payload = row.get("payload_json")
    try:
        payload = json.loads(raw_payload) if raw_payload else {}
    except (TypeError, ValueError, UnicodeError):
        payload = {}
    event["payload"] = payload
    return event, order_key, raw_created_at


def _payload_matches(event: Mapping[str, Any], *, provider: str | None, model: str | None) -> bool:
    if provider is None and model is None:
        return True
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        return False
    if provider is not None and str(payload.get("provider")) != provider:
        return False
    return model is None or str(payload.get("model")) == model


def _scan_events(
    db: Any,
    *,
    owner_user_id: str,
    filters: dict[str, Any],
    pagination: dict[str, int],
    snapshot_event_id: int | None,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    total = 0
    offset = pagination["offset"]
    upper_bound = offset + pagination["limit"]
    after_created_at: Any = None
    after_id: int | None = None
    previous_key: tuple[datetime, int] | None = None

    while True:
        page = db.list_claims_monitoring_events_page(
            user_id=owner_user_id,
            event_type=filters.get("event_type"),
            severity=filters.get("severity"),
            start_time=filters.get("start_time"),
            end_time=filters["end_time"],
            after_created_at=after_created_at,
            after_id=after_id,
            max_event_id=snapshot_event_id,
            limit=EXPORT_SCAN_PAGE_SIZE,
        )
        if not isinstance(page, list) or len(page) > EXPORT_SCAN_PAGE_SIZE:
            raise _serialization_error()
        if not page:
            break

        page_last_created_at: Any = None
        page_last_id: int | None = None
        for raw_row in page:
            event, previous_key, page_last_created_at = _decode_event(
                raw_row,
                owner_user_id=owner_user_id,
                previous_key=previous_key,
            )
            page_last_id = previous_key[1]
            if not _payload_matches(
                event,
                provider=filters.get("provider"),
                model=filters.get("model"),
            ):
                continue
            if offset <= total < upper_bound:
                selected.append(event)
            total += 1

        if len(page) < EXPORT_SCAN_PAGE_SIZE:
            break
        after_created_at = page_last_created_at
        after_id = page_last_id

    return selected, total


def _render_json(
    events: list[dict[str, Any]],
    *,
    filters: dict[str, Any],
    pagination: dict[str, int],
) -> str:
    return json.dumps(
        {"events": events, "filters": filters, "pagination": pagination},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _render_csv(events: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([spreadsheet_safe(value) for value in CSV_COLUMNS])
    for event in events:
        payload_json = json.dumps(
            event.get("payload", {}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        cells = (
            event.get("id"),
            event.get("event_type"),
            event.get("severity"),
            event.get("created_at"),
            payload_json,
        )
        writer.writerow([spreadsheet_safe(value) for value in cells])
    return output.getvalue()


def render_export(
    db: Any,
    *,
    owner_user_id: str,
    format: str,
    filters: dict[str, Any],
    pagination: dict[str, int],
    snapshot_at: str,
    max_bytes: int,
    snapshot_event_id: int | None = None,
) -> dict[str, Any]:
    """Render one owner-scoped export through bounded keyset database pages."""
    owner = _canonical_owner_id(owner_user_id)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise _invalid_payload_error()
    snapshot = _parse_iso8601(snapshot_at)
    if (
        snapshot_event_id is not None
        and (
            isinstance(snapshot_event_id, bool)
            or not isinstance(snapshot_event_id, int)
            or snapshot_event_id < 0
        )
    ):
        raise _invalid_payload_error()
    normalized = normalize_export_request(
        {
            "format": format,
            "filters": filters,
            "pagination": pagination,
        },
        owner_user_id=owner,
        now=snapshot,
    )

    events, total = _scan_events(
        db,
        owner_user_id=owner,
        filters=normalized["filters"],
        pagination=normalized["pagination"],
        snapshot_event_id=snapshot_event_id,
    )
    pagination_meta = {
        "limit": normalized["pagination"]["limit"],
        "offset": normalized["pagination"]["offset"],
        "total": total,
    }
    try:
        if normalized["format"] == "csv":
            payload_text = _render_csv(events)
        else:
            payload_text = _render_json(
                events,
                filters=normalized["filters"],
                pagination=pagination_meta,
            )
        size_bytes = len(payload_text.encode("utf-8"))
    except ClaimsAnalyticsExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - serialization details must never become public text.
        raise _serialization_error() from exc

    if size_bytes > max_bytes:
        raise ClaimsAnalyticsExportError(
            "Claims analytics export exceeds the configured size limit.",
            code="claims_export_too_large",
            http_status=413,
        )

    return {
        "payload_json": payload_text if normalized["format"] == "json" else None,
        "payload_csv": payload_text if normalized["format"] == "csv" else None,
        "format": normalized["format"],
        "event_count": len(events),
        "size_bytes": size_bytes,
    }


def apply_export_transition(current: str, requested: str) -> str:
    """Apply the Claims artifact transition table without side effects."""
    return requested if (current, requested) in _EXPORT_TRANSITIONS else current


def _compact_json(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _invalid_payload_error() from exc


def _validate_normalized_request(
    normalized: Any,
    *,
    owner_user_id: str,
    persisted: bool,
) -> dict[str, Any]:
    if not isinstance(normalized, dict):
        raise _invalid_artifact_error() if persisted else _invalid_payload_error()
    if normalized.get("owner_user_id") != owner_user_id:
        raise _owner_error()
    try:
        _canonical_owner_id(normalized["owner_user_id"])
    except (KeyError, ClaimsAnalyticsExportError) as exc:
        raise _owner_error() from exc

    invalid = _invalid_artifact_error if persisted else _invalid_payload_error
    if set(normalized) != _NORMALIZED_REQUEST_KEYS:
        raise invalid()
    try:
        snapshot = _parse_iso8601(normalized["snapshot_at"])
        rebuilt = normalize_export_request(
            {
                "format": normalized["format"],
                "filters": normalized["filters"],
                "pagination": normalized["pagination"],
            },
            owner_user_id=owner_user_id,
            now=snapshot,
        )
    except (KeyError, ClaimsAnalyticsExportError) as exc:
        raise invalid() from exc
    if rebuilt != normalized:
        raise invalid()
    return rebuilt


def _create_artifact(
    db: Any,
    *,
    owner_user_id: str,
    normalized: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    owner = _canonical_owner_id(owner_user_id)
    request = _validate_normalized_request(
        normalized,
        owner_user_id=owner,
        persisted=False,
    )
    snapshot_event_id = db.get_claims_monitoring_event_high_water(user_id=owner)
    return db.create_claims_analytics_export(
        export_id=uuid4().hex,
        user_id=owner,
        format=request["format"],
        status=status,
        filters_json=_compact_json(request["filters"]),
        pagination_json=_compact_json(request["pagination"]),
        snapshot_at=request["snapshot_at"],
        snapshot_event_id=snapshot_event_id,
    )


def create_queued_artifact(
    db: Any,
    *,
    owner_user_id: str,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    """Create one queued artifact containing only its normalized request."""
    return _create_artifact(
        db,
        owner_user_id=owner_user_id,
        normalized=normalized,
        status="queued",
    )


def _already_ready(export_id: str) -> dict[str, Any]:
    return {
        "outcome": "skipped",
        "reason": "already_ready",
        "export_id": export_id,
    }


def _success_result(export_id: str, rendered: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": "ok",
        "export_id": export_id,
        "format": rendered["format"],
        "event_count": rendered["event_count"],
        "size_bytes": rendered["size_bytes"],
    }


def _record_processing_failure(
    db: Any,
    *,
    owner_user_id: str,
    export_id: str,
    error: ClaimsAnalyticsExportError,
) -> bool:
    return db.transition_claims_analytics_export_status(
        export_id=export_id,
        user_id=owner_user_id,
        from_statuses=("processing",),
        to_status="failed",
        error_code=error.code,
        error_message=error.public_message,
    )


def _ready_after_lost_failure_transition(
    db: Any,
    *,
    owner_user_id: str,
    export_id: str,
    transitioned: bool,
) -> dict[str, Any] | None:
    if transitioned:
        return None
    current = db.get_claims_analytics_export(export_id, user_id=owner_user_id)
    if current and current.get("user_id") == owner_user_id and current.get("status") == "ready":
        return _already_ready(export_id)
    return None


def _mark_rendered_ready(
    db: Any,
    *,
    owner_user_id: str,
    export_id: str,
    rendered: dict[str, Any],
) -> bool:
    return db.mark_claims_analytics_export_ready(
        export_id=export_id,
        user_id=owner_user_id,
        payload_json=rendered["payload_json"],
        payload_csv=rendered["payload_csv"],
    )


def _render_normalized(
    db: Any,
    *,
    owner_user_id: str,
    normalized: dict[str, Any],
    snapshot_event_id: int | None = None,
) -> dict[str, Any]:
    return render_export(
        db,
        owner_user_id=owner_user_id,
        format=normalized["format"],
        filters=normalized["filters"],
        pagination=normalized["pagination"],
        snapshot_at=normalized["snapshot_at"],
        max_bytes=export_max_bytes(),
        snapshot_event_id=snapshot_event_id,
    )


def create_ready_artifact(
    db: Any,
    *,
    owner_user_id: str,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    """Synchronously render and atomically complete one artifact."""
    owner = _canonical_owner_id(owner_user_id)
    request = _validate_normalized_request(
        normalized,
        owner_user_id=owner,
        persisted=False,
    )
    row = _create_artifact(
        db,
        owner_user_id=owner,
        normalized=request,
        status="processing",
    )
    export_id = row["export_id"]
    try:
        rendered = _render_normalized(
            db,
            owner_user_id=owner,
            normalized=request,
            snapshot_event_id=row.get("snapshot_event_id"),
        )
        if not _mark_rendered_ready(
            db,
            owner_user_id=owner,
            export_id=export_id,
            rendered=rendered,
        ):
            current = db.get_claims_analytics_export(export_id, user_id=owner)
            if current.get("status") != "ready":
                raise _storage_error()
    except ClaimsAnalyticsExportError as exc:
        _record_processing_failure(
            db,
            owner_user_id=owner,
            export_id=export_id,
            error=exc,
        )
        raise
    except Exception as exc:
        storage_error = _storage_error()
        try:
            _record_processing_failure(
                db,
                owner_user_id=owner,
                export_id=export_id,
                error=storage_error,
            )
        except Exception as persistence_exc:  # noqa: BLE001
            logger.warning(
                "Claims export failure state could not be persisted: operation={} "
                "export_id={} error_code={} original_error_type={} persistence_error_type={}",
                "record_processing_failure",
                export_id,
                storage_error.code,
                type(exc).__name__,
                type(persistence_exc).__name__,
            )
        raise
    return db.get_claims_analytics_export(export_id, user_id=owner)


def _decode_persisted_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise _invalid_artifact_error()
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _invalid_artifact_error() from exc
    if not isinstance(decoded, dict):
        raise _invalid_artifact_error()
    return decoded


def _persisted_request(row: Any, *, owner_user_id: str) -> dict[str, Any]:
    if not isinstance(row, Mapping) or row.get("user_id") != owner_user_id:
        raise _missing_artifact_error()
    normalized = {
        "owner_user_id": owner_user_id,
        "format": row.get("format"),
        "filters": _decode_persisted_object(row.get("filters_json")),
        "pagination": _decode_persisted_object(row.get("pagination_json")),
        "snapshot_at": row.get("snapshot_at"),
    }
    return _validate_normalized_request(
        normalized,
        owner_user_id=owner_user_id,
        persisted=True,
    )


def _validate_job_id(job_id: Any) -> int:
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        raise _invalid_artifact_error()
    return job_id


def process_export_artifact(
    db: Any,
    *,
    owner_user_id: str,
    export_id: str,
    job_id: int,
) -> dict[str, Any]:
    """Render one persisted artifact safely across retries and late races."""
    owner = _canonical_owner_id(owner_user_id)
    validated_export_id = validate_export_id(export_id)
    validated_job_id = _validate_job_id(job_id)
    row = db.get_claims_analytics_export(validated_export_id, user_id=owner)
    if not row or row.get("user_id") != owner:
        raise _missing_artifact_error()
    if row.get("status") == "ready":
        return _already_ready(validated_export_id)

    persisted_job_id = row.get("job_id")
    if persisted_job_id is None:
        attached = db.attach_claims_analytics_export_job(
            export_id=validated_export_id,
            user_id=owner,
            job_id=validated_job_id,
        )
        row = db.get_claims_analytics_export(validated_export_id, user_id=owner)
        if row.get("status") == "ready":
            return _already_ready(validated_export_id)
        if not attached and row.get("job_id") is None:
            raise _storage_error()
        persisted_job_id = row.get("job_id")
    if persisted_job_id != validated_job_id:
        raise _invalid_artifact_error()

    status = row.get("status")
    if status in {"queued", "failed"}:
        transitioned = db.transition_claims_analytics_export_status(
            export_id=validated_export_id,
            user_id=owner,
            from_statuses=(status,),
            to_status="processing",
        )
        row = db.get_claims_analytics_export(validated_export_id, user_id=owner)
        if row.get("status") == "ready":
            return _already_ready(validated_export_id)
        if not transitioned and row.get("status") != "processing":
            raise _storage_error()
    elif status != "processing":
        raise _invalid_artifact_error()

    try:
        row = db.get_claims_analytics_export(validated_export_id, user_id=owner)
        if not row or row.get("user_id") != owner:
            raise _missing_artifact_error()
        if row.get("status") == "ready":
            return _already_ready(validated_export_id)
        if row.get("status") != "processing":
            raise _storage_error()
        if row.get("job_id") != validated_job_id:
            raise _invalid_artifact_error()
        normalized = _persisted_request(row, owner_user_id=owner)
        rendered = _render_normalized(
            db,
            owner_user_id=owner,
            normalized=normalized,
            snapshot_event_id=row.get("snapshot_event_id"),
        )
        if _mark_rendered_ready(
            db,
            owner_user_id=owner,
            export_id=validated_export_id,
            rendered=rendered,
        ):
            return _success_result(validated_export_id, rendered)
        current = db.get_claims_analytics_export(validated_export_id, user_id=owner)
        if current.get("status") == "ready":
            return _already_ready(validated_export_id)
        raise _storage_error()
    except ClaimsAnalyticsExportError as exc:
        transitioned = _record_processing_failure(
            db,
            owner_user_id=owner,
            export_id=validated_export_id,
            error=exc,
        )
        ready_result = _ready_after_lost_failure_transition(
            db,
            owner_user_id=owner,
            export_id=validated_export_id,
            transitioned=transitioned,
        )
        if ready_result is not None:
            return ready_result
        raise
    except Exception as exc:
        storage_error = _storage_error()
        try:
            transitioned = _record_processing_failure(
                db,
                owner_user_id=owner,
                export_id=validated_export_id,
                error=storage_error,
            )
            ready_result = _ready_after_lost_failure_transition(
                db,
                owner_user_id=owner,
                export_id=validated_export_id,
                transitioned=transitioned,
            )
            if ready_result is not None:
                return ready_result
        except Exception as persistence_exc:  # noqa: BLE001
            logger.warning(
                "Claims export failure state could not be persisted: operation={} "
                "export_id={} error_code={} original_error_type={} persistence_error_type={}",
                "record_processing_failure",
                validated_export_id,
                storage_error.code,
                type(exc).__name__,
                type(persistence_exc).__name__,
            )
        raise


def _valid_artifact_job_ids(rows: list[dict[str, Any]]) -> list[int]:
    job_ids: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        job_id = row.get("job_id")
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0 or job_id in seen:
            continue
        seen.add(job_id)
        job_ids.append(job_id)
    return job_ids


def hydrate_job_statuses(
    rows: list[dict[str, Any]],
    *,
    owner_user_id: str,
    job_manager: Any,
) -> dict[int, str | None]:
    """Project scoped Jobs statuses without mutating artifact rows."""
    owner = _canonical_owner_id(owner_user_id)
    job_ids = _valid_artifact_job_ids(rows)
    if not job_ids:
        return {}
    statuses = dict.fromkeys(job_ids)
    try:
        jobs = job_manager.get_jobs_by_ids(
            job_ids,
            domain="claims",
            owner_user_id=owner,
            include_archived=True,
        )
    except Exception:  # noqa: BLE001 - Jobs outages degrade to null projections.
        return statuses
    if not isinstance(jobs, Mapping):
        return statuses
    for job_id in job_ids:
        job = jobs.get(job_id)
        status = job.get("status") if isinstance(job, Mapping) else None
        statuses[job_id] = status if isinstance(status, str) else None
    return statuses


def _maintenance_limit(value: Any) -> int:
    if isinstance(value, bool):
        return 100
    try:
        return max(1, min(100, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 100


def _artifact_age_seconds(row: Mapping[str, Any], *, field: str, now: datetime) -> float | None:
    try:
        timestamp, _ = _database_timestamp(row.get(field))
    except ClaimsAnalyticsExportError:
        return None
    return (now - timestamp).total_seconds()


def _job_matches_reconciliation(
    job: Any,
    *,
    owner_user_id: str,
    batch_group: str,
) -> int | None:
    if not isinstance(job, Mapping):
        return None
    job_id = job.get("id")
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        return None
    if (
        job.get("batch_group") != batch_group
        or job.get("domain") != "claims"
        or job.get("owner_user_id") != owner_user_id
        or job.get("job_type") != _EXPORT_JOB_TYPE
    ):
        return None
    return job_id


def reconcile_export_artifacts(
    db: Any,
    *,
    owner_user_id: str,
    job_manager: Any,
    now: datetime | None = None,
    limit: int = 100,
) -> dict[str, int]:
    """Repair missing Job links and fail only Jobs-proven queued orphans."""
    owner = _canonical_owner_id(owner_user_id)
    current_time = _normalize_now(now)
    rows = db.list_claims_analytics_exports_for_maintenance(
        user_id=owner,
        limit=_maintenance_limit(limit),
        statuses=("queued",),
        job_id_missing=True,
    )
    candidates = [
        row for row in rows if isinstance(row, Mapping) and row.get("status") == "queued" and row.get("job_id") is None
    ]
    counters = {
        "examined": len(candidates),
        "repaired": 0,
        "failed": 0,
        "unchanged": 0,
    }
    grace = orphan_grace_seconds()
    enqueue_error = _enqueue_failed_error()
    for row in candidates:
        export_id = row.get("export_id")
        try:
            validated_export_id = validate_export_id(export_id)
        except ClaimsAnalyticsExportError:
            counters["unchanged"] += 1
            continue
        batch_group = f"{_EXPORT_BATCH_GROUP_PREFIX}{validated_export_id}"
        try:
            job = job_manager.find_job_by_batch_group(
                batch_group=batch_group,
                domain="claims",
                owner_user_id=owner,
                job_type=_EXPORT_JOB_TYPE,
                include_archived=True,
            )
        except Exception:  # noqa: BLE001 - uncertainty must not fail an artifact.
            counters["unchanged"] += 1
            continue
        if job is not None:
            job_id = _job_matches_reconciliation(
                job,
                owner_user_id=owner,
                batch_group=batch_group,
            )
            if job_id is not None and db.attach_claims_analytics_export_job(
                export_id=validated_export_id,
                user_id=owner,
                job_id=job_id,
            ):
                counters["repaired"] += 1
            else:
                counters["unchanged"] += 1
            continue
        age = _artifact_age_seconds(row, field="created_at", now=current_time)
        if age is None or age < grace:
            counters["unchanged"] += 1
            continue
        if db.transition_claims_analytics_export_status(
            export_id=validated_export_id,
            user_id=owner,
            from_statuses=("queued",),
            to_status="failed",
            error_code=enqueue_error.code,
            error_message=enqueue_error.public_message,
        ):
            counters["failed"] += 1
        else:
            counters["unchanged"] += 1
    return counters


def _cleanup_retention_seconds(retention_hours: Any) -> float | None:
    if isinstance(retention_hours, bool):
        return None
    try:
        seconds = float(retention_hours) * 3600
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _cleanup_job_status(
    job: Any,
    *,
    job_id: int,
    owner_user_id: str,
) -> str | None:
    if not isinstance(job, Mapping):
        return None
    if (
        job.get("id") != job_id
        or job.get("domain") != "claims"
        or job.get("owner_user_id") != owner_user_id
        or job.get("job_type") != _EXPORT_JOB_TYPE
    ):
        return None
    status = job.get("status")
    return status if isinstance(status, str) else None


def _cleanup_rotation_anchor(*, owner_user_id: str, now: datetime) -> str:
    """Return a stable rotating anchor for bounded failed-artifact scans."""
    bucket = int(now.timestamp()) // CLEANUP_ROTATION_SECONDS
    seed = f"{owner_user_id}:{bucket}".encode("ascii")
    return hashlib.sha256(seed).hexdigest()[:32]


def _cleanup_page_limits(limit: Any) -> tuple[int, int]:
    bounded = _maintenance_limit(limit)
    ready_limit = max(1, bounded // 2)
    failed_limit = max(1, bounded - ready_limit)
    return ready_limit, failed_limit


def cleanup_export_artifacts(
    db: Any,
    *,
    owner_user_id: str,
    job_manager: Any,
    now: datetime | None = None,
    retention_hours: float = 24,
    limit: int = 100,
) -> int:
    """Delete only bounded artifacts with proven terminal lifecycle state."""
    owner = _canonical_owner_id(owner_user_id)
    current_time = _normalize_now(now)
    retention_seconds = _cleanup_retention_seconds(retention_hours)
    if retention_seconds is None:
        return 0
    cutoff = current_time - timedelta(seconds=retention_seconds)
    cutoff_text = _format_utc(cutoff)
    ready_limit, failed_limit = _cleanup_page_limits(limit)
    ready_rows = db.list_claims_analytics_exports_for_maintenance(
        user_id=owner,
        limit=ready_limit,
        statuses=("ready",),
        updated_before=cutoff_text,
    )
    anchor = _cleanup_rotation_anchor(owner_user_id=owner, now=current_time)
    failed_rows = db.list_claims_analytics_exports_for_maintenance(
        user_id=owner,
        limit=failed_limit,
        statuses=("failed",),
        updated_before=cutoff_text,
        export_id_after=anchor,
    )
    remaining = failed_limit - len(failed_rows)
    if remaining > 0:
        failed_rows.extend(
            db.list_claims_analytics_exports_for_maintenance(
                user_id=owner,
                limit=remaining,
                statuses=("failed",),
                updated_before=cutoff_text,
                export_id_at_or_before=anchor,
            )
        )
    rows = [*ready_rows, *failed_rows]

    old_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        age = _artifact_age_seconds(row, field="updated_at", now=current_time)
        if age is not None and age > retention_seconds:
            old_rows.append(dict(row))

    failed_with_jobs = [
        row
        for row in old_rows
        if row.get("status") == "failed"
        and isinstance(row.get("job_id"), int)
        and not isinstance(row.get("job_id"), bool)
        and row["job_id"] > 0
    ]
    job_ids = _valid_artifact_job_ids(failed_with_jobs)
    jobs_lookup_succeeded = True
    jobs: Mapping[int, Any] = {}
    if job_ids:
        try:
            result = job_manager.get_jobs_by_ids(
                job_ids,
                domain="claims",
                owner_user_id=owner,
                include_archived=True,
            )
            if isinstance(result, Mapping):
                jobs = result
            else:
                jobs_lookup_succeeded = False
        except Exception:  # noqa: BLE001 - uncertain Jobs state blocks deletion.
            jobs_lookup_succeeded = False

    grace = orphan_grace_seconds()
    selected: list[str] = []
    for row in old_rows:
        export_id = row.get("export_id")
        if not isinstance(export_id, str):
            continue
        status = row.get("status")
        if status == "ready":
            selected.append(export_id)
            continue
        if status != "failed":
            continue
        age = _artifact_age_seconds(row, field="updated_at", now=current_time)
        if age is None:
            continue
        job_id = row.get("job_id")
        if job_id is None:
            if row.get("error_code") != "claims_export_enqueue_failed" or age <= retention_seconds + grace:
                continue
            try:
                validated_export_id = validate_export_id(export_id)
            except ClaimsAnalyticsExportError:
                continue
            batch_group = f"{_EXPORT_BATCH_GROUP_PREFIX}{validated_export_id}"
            try:
                job = job_manager.find_job_by_batch_group(
                    batch_group=batch_group,
                    domain="claims",
                    owner_user_id=owner,
                    job_type=_EXPORT_JOB_TYPE,
                    include_archived=True,
                )
            except Exception as exc:  # noqa: BLE001 - uncertainty must preserve the artifact.
                logger.warning(
                    "Claims export cleanup Jobs lookup unavailable: operation={} export_id={} error_type={}",
                    "find_job_by_batch_group",
                    validated_export_id,
                    type(exc).__name__,
                )
                continue
            if job is None:
                selected.append(validated_export_id)
            elif (
                _job_matches_reconciliation(
                    job,
                    owner_user_id=owner,
                    batch_group=batch_group,
                )
                is None
            ):
                continue
            continue
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
            continue
        if not jobs_lookup_succeeded:
            continue
        job = jobs.get(job_id)
        if isinstance(job, Mapping):
            job_status = _cleanup_job_status(
                job,
                job_id=job_id,
                owner_user_id=owner,
            )
            if job_status in _TERMINAL_JOB_STATUSES:
                selected.append(export_id)
        elif age > grace:
            selected.append(export_id)

    if not selected:
        return 0
    return db.delete_claims_analytics_exports(
        user_id=owner,
        export_ids=selected,
        updated_before=cutoff_text,
    )
