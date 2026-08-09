"""Deterministic request normalization and rendering for Claims analytics exports."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.config import settings

DEFAULT_EXPORT_MAX_BYTES = 10_485_760
DEFAULT_EXPORT_ORPHAN_GRACE_SEC = 300
EXPORT_SCAN_PAGE_SIZE = 1000
EXPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CSV_COLUMNS = ("id", "event_type", "severity", "created_at", "payload_json")

_OWNER_ID_RE = re.compile(r"^[1-9][0-9]*$")
_SCALAR_FILTERS = ("event_type", "severity", "provider", "model")
_EVENT_COLUMNS = ("id", "user_id", "event_type", "severity", "created_at", "delivered_at")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


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
    source = settings if settings_obj is None else settings_obj
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
) -> dict[str, Any]:
    """Render one owner-scoped export through bounded keyset database pages."""
    owner = _canonical_owner_id(owner_user_id)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise _invalid_payload_error()
    snapshot = _parse_iso8601(snapshot_at)
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
