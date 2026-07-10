"""Validation and serialization helpers for source-grounded review plans."""

from __future__ import annotations

import json
from calendar import monthrange
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackSourceSelection

SourceReviewActivity = Literal["reread", "quiz", "flashcards", "cloze"]
SourceReviewOffsetUnit = Literal["day", "month"]

_OFFSET_CAPS = {"day": 3650, "month": 120}
_LAUNCH_METADATA_MAX_BYTES = 16 * 1024
_ACTIVITY_LAUNCH_FIELDS = {
    "reread": (
        "/flashcards",
        "source_review_due_panel",
        "show_reread_snapshot",
        "source_bundle",
    ),
    "quiz": (
        "/quiz",
        "quiz_generation",
        "prefill_generation_sources",
        "source_items",
    ),
    "flashcards": (
        "/flashcards",
        "flashcard_generation",
        "prefill_generation_sources",
        "source_items",
    ),
    "cloze": (
        "/flashcards",
        "cloze_flashcard_generation",
        "prefill_generation_sources",
        "source_items",
    ),
}


def compute_source_review_due_at(
    *,
    starts_on: date,
    timezone_name: str,
    offset_value: int,
    offset_unit: SourceReviewOffsetUnit,
) -> datetime:
    """Compute a review's local-midnight due time and return it in UTC."""

    if not isinstance(offset_unit, str) or offset_unit not in _OFFSET_CAPS:
        raise ValueError("offset_unit must be 'day' or 'month'")
    if isinstance(offset_value, bool) or not isinstance(offset_value, int):
        raise ValueError("offset_value must be an integer")
    if offset_value <= 0:
        raise ValueError("offset_value must be positive")
    if offset_value > _OFFSET_CAPS[offset_unit]:
        raise ValueError(f"offset_value exceeds the {_OFFSET_CAPS[offset_unit]} {offset_unit} cap")

    try:
        if offset_unit == "day":
            due_date = starts_on + timedelta(days=offset_value)
        else:
            month_index = starts_on.month - 1 + offset_value
            due_year = starts_on.year + month_index // 12
            due_month = month_index % 12 + 1
            due_day = min(starts_on.day, monthrange(due_year, due_month)[1])
            due_date = date(due_year, due_month, due_day)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Source review due date is outside the supported date range") from exc

    try:
        plan_timezone = ZoneInfo(timezone_name)
    except (OSError, TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"Invalid timezone: {timezone_name!r}") from exc

    local_midnight = datetime.combine(due_date, time.min, tzinfo=plan_timezone)
    due_at = local_midnight.astimezone(timezone.utc)
    round_trip = due_at.astimezone(plan_timezone)
    if round_trip.date() != due_date or round_trip.time().replace(tzinfo=None) != time.min:
        raise ValueError(
            f"Source review due date {due_date.isoformat()} does not exist in timezone {timezone_name!r}"
        )
    return due_at


def compute_source_review_schedule(
    *,
    starts_on: date,
    timezone_name: str,
    schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute schedule due times and reject duplicate due/activity pairs."""

    computed_rows: list[dict[str, Any]] = []
    seen: set[tuple[datetime, str]] = set()

    for row in schedule:
        activity_type = row.get("activity_type")
        if not isinstance(activity_type, str) or activity_type not in _ACTIVITY_LAUNCH_FIELDS:
            raise ValueError("Unsupported activity_type")

        due_at = compute_source_review_due_at(
            starts_on=starts_on,
            timezone_name=timezone_name,
            offset_value=row.get("offset_value"),
            offset_unit=row.get("offset_unit"),
        )
        unique_key = (due_at, activity_type)
        if unique_key in seen:
            raise ValueError("Duplicate computed due_at and activity_type")
        seen.add(unique_key)

        computed_row = dict(row)
        computed_row["due_at"] = due_at
        computed_rows.append(computed_row)

    return computed_rows


def normalize_source_review_bundle(
    source_items: Sequence[StudyPackSourceSelection | Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate source selections and serialize them with canonical field names."""

    return {
        "items": [
            (
                source_item
                if isinstance(source_item, StudyPackSourceSelection)
                else StudyPackSourceSelection.model_validate(source_item)
            ).model_dump(mode="json", exclude_none=True)
            for source_item in source_items
        ]
    }


def build_source_review_launch_metadata(
    *,
    activity_type: SourceReviewActivity,
    plan_id: int,
    occurrence_id: int,
    created_at: str,
) -> dict[str, Any]:
    """Build bounded launch metadata for an existing activity surface."""

    if not isinstance(activity_type, str) or activity_type not in _ACTIVITY_LAUNCH_FIELDS:
        raise ValueError("Unsupported activity_type")

    target_route, target_surface, action, source_payload_field = _ACTIVITY_LAUNCH_FIELDS[activity_type]
    metadata = {
        "activity_type": activity_type,
        "plan_id": plan_id,
        "occurrence_id": occurrence_id,
        "target_route": target_route,
        "target_surface": target_surface,
        "action": action,
        "source_payload_field": source_payload_field,
        "completion_required": True,
        "created_at": created_at,
    }
    serialized = json.dumps(metadata).encode("utf-8")
    if len(serialized) > _LAUNCH_METADATA_MAX_BYTES:
        raise ValueError("Launch metadata exceeds the 16 KiB limit")
    return metadata
