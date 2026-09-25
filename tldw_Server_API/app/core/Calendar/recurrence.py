"""Bounded recurrence expansion for local Calendar items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as date_parser
from dateutil import rrule

from tldw_Server_API.app.core.Calendar.constants import (
    MAX_EXPANDED_OCCURRENCES,
    MAX_QUERY_WINDOW_DAYS,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError

RecurrenceFrequency = Literal["daily", "weekly", "monthly"]
TemporalValue = date | datetime | str

_FREQUENCY_TO_DATEUTIL = {
    "daily": rrule.DAILY,
    "weekly": rrule.WEEKLY,
    "monthly": rrule.MONTHLY,
}
_DATEUTIL_TO_FREQUENCY = {
    "DAILY": "daily",
    "WEEKLY": "weekly",
    "MONTHLY": "monthly",
}
_WEEKDAYS = {
    "MO": rrule.MO,
    "TU": rrule.TU,
    "WE": rrule.WE,
    "TH": rrule.TH,
    "FR": rrule.FR,
    "SA": rrule.SA,
    "SU": rrule.SU,
}
_SUPPORTED_RRULE_KEYS = {"FREQ", "INTERVAL", "BYDAY", "COUNT", "UNTIL"}


@dataclass(frozen=True)
class LocalRecurrenceRule:
    """Normalized local recurrence rule for the supported Calendar v1 subset."""

    frequency: RecurrenceFrequency
    interval: int = 1
    weekdays: tuple[str, ...] = field(default_factory=tuple)
    count: int | None = None
    until: TemporalValue | None = None

    def __post_init__(self) -> None:
        if self.frequency not in _FREQUENCY_TO_DATEUTIL:
            raise CalendarValidationError(f"Unsupported recurrence frequency: {self.frequency}")
        if self.interval < 1:
            raise CalendarValidationError("Recurrence interval must be at least 1")
        if self.count is not None and self.count < 1:
            raise CalendarValidationError("Recurrence count must be at least 1")
        normalized_weekdays = tuple(day.upper() for day in self.weekdays)
        invalid_weekdays = [day for day in normalized_weekdays if day not in _WEEKDAYS]
        if invalid_weekdays:
            raise CalendarValidationError(f"Unsupported recurrence weekdays: {', '.join(invalid_weekdays)}")
        if normalized_weekdays and self.frequency != "weekly":
            raise CalendarValidationError("Weekday recurrence is only supported for weekly rules")
        object.__setattr__(self, "weekdays", normalized_weekdays)

    def to_rrule(self) -> str:
        """Serialize the local recurrence subset to an RFC 5545-style RRULE line."""

        parts = [
            f"FREQ={self.frequency.upper()}",
            f"INTERVAL={self.interval}",
        ]
        if self.weekdays:
            parts.append(f"BYDAY={','.join(self.weekdays)}")
        if self.count is not None:
            parts.append(f"COUNT={self.count}")
        if self.until is not None:
            parts.append(f"UNTIL={_format_until(self.until)}")
        return ";".join(parts)

    @classmethod
    def from_rrule(cls, value: str) -> "LocalRecurrenceRule":
        """Parse the local recurrence subset from an RRULE value."""

        text = value.strip()
        if text.upper().startswith("RRULE:"):
            text = text.split(":", 1)[1]
        fields: dict[str, str] = {}
        for part in text.split(";"):
            if not part:
                continue
            if "=" not in part:
                raise CalendarValidationError(f"Invalid recurrence rule part: {part}")
            key, raw_value = part.split("=", 1)
            normalized_key = key.upper()
            if normalized_key not in _SUPPORTED_RRULE_KEYS:
                raise CalendarValidationError(f"Unsupported recurrence rule key: {normalized_key}")
            fields[normalized_key] = raw_value

        raw_frequency = fields.get("FREQ", "").upper()
        frequency = _DATEUTIL_TO_FREQUENCY.get(raw_frequency)
        if frequency is None:
            raise CalendarValidationError(f"Unsupported recurrence frequency: {raw_frequency}")

        try:
            interval = int(fields.get("INTERVAL", "1"))
            count = int(fields["COUNT"]) if "COUNT" in fields else None
        except ValueError as exc:
            raise CalendarValidationError("Recurrence interval and count must be integers") from exc
        weekdays = tuple(day.strip().upper() for day in fields.get("BYDAY", "").split(",") if day.strip())
        try:
            until = _parse_until(fields["UNTIL"]) if "UNTIL" in fields else None
        except (TypeError, ValueError) as exc:
            raise CalendarValidationError("Recurrence UNTIL must be a valid date or timestamp") from exc
        return cls(
            frequency=frequency,  # type: ignore[arg-type]
            interval=interval,
            weekdays=weekdays,
            count=count,
            until=until,
        )


@dataclass(frozen=True)
class RecurrenceOccurrence:
    """One expanded local recurrence occurrence."""

    start_at: date | datetime
    end_at: date | datetime | None
    occurrence_index: int


def validate_query_window(window_start: TemporalValue, window_end: TemporalValue) -> None:
    """Reject missing, reversed, or over-broad Calendar query windows."""

    start = _coerce_datetime(window_start)
    end = _coerce_datetime(window_end)
    if end <= start:
        raise CalendarValidationError("Calendar query window end must be after start")
    if end - start > timedelta(days=MAX_QUERY_WINDOW_DAYS):
        raise CalendarValidationError(
            f"Calendar query window cannot exceed {MAX_QUERY_WINDOW_DAYS} days"
        )


def expand_recurrence(
    *,
    master_start: TemporalValue,
    master_end: TemporalValue | None,
    recurrence: LocalRecurrenceRule,
    window_start: TemporalValue,
    window_end: TemporalValue,
    timezone_name: str | None = None,
    all_day: bool = False,
) -> list[RecurrenceOccurrence]:
    """Expand a local recurrence rule inside a bounded query window."""

    validate_query_window(window_start, window_end)
    if all_day:
        return _expand_all_day_recurrence(
            master_start=master_start,
            master_end=master_end,
            recurrence=recurrence,
            window_start=window_start,
            window_end=window_end,
            timezone_name=timezone_name,
        )
    return _expand_timed_recurrence(
        master_start=master_start,
        master_end=master_end,
        recurrence=recurrence,
        window_start=window_start,
        window_end=window_end,
        timezone_name=timezone_name,
    )


def expand_rrule(
    *,
    master_start: TemporalValue,
    master_end: TemporalValue | None,
    rrule_text: str,
    window_start: TemporalValue,
    window_end: TemporalValue,
    timezone_name: str | None = None,
    all_day: bool = False,
) -> list[RecurrenceOccurrence]:
    """Parse and expand a local recurrence RRULE value."""

    return expand_recurrence(
        master_start=master_start,
        master_end=master_end,
        recurrence=LocalRecurrenceRule.from_rrule(rrule_text),
        window_start=window_start,
        window_end=window_end,
        timezone_name=timezone_name,
        all_day=all_day,
    )


def _expand_timed_recurrence(
    *,
    master_start: TemporalValue,
    master_end: TemporalValue | None,
    recurrence: LocalRecurrenceRule,
    window_start: TemporalValue,
    window_end: TemporalValue,
    timezone_name: str | None,
) -> list[RecurrenceOccurrence]:
    tz = _zoneinfo(timezone_name)
    start = _coerce_datetime(master_start, tz)
    end = _coerce_datetime(master_end, tz) if master_end is not None else None
    duration = (end - start) if end is not None else timedelta(0)
    query_start = _coerce_datetime(window_start, tz)
    query_end = _coerce_datetime(window_end, tz)
    search_start = query_start - duration
    rule = _dateutil_rule(recurrence, dtstart=start)
    candidates = rule.between(search_start, query_end, inc=True)
    occurrences: list[RecurrenceOccurrence] = []
    for candidate in candidates:
        occurrence_end = candidate + duration if end is not None else None
        if not _overlaps(candidate, occurrence_end, query_start, query_end):
            continue
        occurrences.append(
            RecurrenceOccurrence(
                start_at=candidate,
                end_at=occurrence_end,
                occurrence_index=len(occurrences),
            )
        )
        if len(occurrences) >= MAX_EXPANDED_OCCURRENCES:
            break
    return occurrences


def _expand_all_day_recurrence(
    *,
    master_start: TemporalValue,
    master_end: TemporalValue | None,
    recurrence: LocalRecurrenceRule,
    window_start: TemporalValue,
    window_end: TemporalValue,
    timezone_name: str | None,
) -> list[RecurrenceOccurrence]:
    tz = _zoneinfo(timezone_name)
    start_date = _coerce_date(master_start, tz)
    end_date = _coerce_date(master_end, tz) if master_end is not None else None
    duration = (end_date - start_date) if end_date is not None else timedelta(0)
    query_start = _coerce_date(window_start, tz)
    query_end = _coerce_date(window_end, tz)
    dtstart = datetime.combine(start_date, time.min, tzinfo=tz)
    search_start = datetime.combine(query_start - duration, time.min, tzinfo=tz)
    search_end = datetime.combine(query_end, time.max, tzinfo=tz)
    rule = _dateutil_rule(recurrence, dtstart=dtstart)

    occurrences: list[RecurrenceOccurrence] = []
    for candidate in rule.between(search_start, search_end, inc=True):
        occurrence_start = candidate.date()
        occurrence_end = occurrence_start + duration if end_date is not None else None
        if not _date_overlaps(occurrence_start, occurrence_end, query_start, query_end):
            continue
        occurrences.append(
            RecurrenceOccurrence(
                start_at=occurrence_start,
                end_at=occurrence_end,
                occurrence_index=len(occurrences),
            )
        )
        if len(occurrences) >= MAX_EXPANDED_OCCURRENCES:
            break
    return occurrences


def _dateutil_rule(recurrence: LocalRecurrenceRule, *, dtstart: datetime) -> rrule.rrule:
    kwargs = {
        "freq": _FREQUENCY_TO_DATEUTIL[recurrence.frequency],
        "dtstart": dtstart,
        "interval": recurrence.interval,
    }
    if recurrence.weekdays:
        kwargs["byweekday"] = [_WEEKDAYS[day] for day in recurrence.weekdays]
    if recurrence.count is not None:
        kwargs["count"] = recurrence.count
    if recurrence.until is not None:
        kwargs["until"] = _coerce_until(recurrence.until, dtstart)
    return rrule.rrule(**kwargs)


def _overlaps(
    occurrence_start: datetime,
    occurrence_end: datetime | None,
    window_start: datetime,
    window_end: datetime,
) -> bool:
    effective_end = occurrence_end or occurrence_start
    return effective_end >= window_start and occurrence_start <= window_end


def _date_overlaps(
    occurrence_start: date,
    occurrence_end: date | None,
    window_start: date,
    window_end: date,
) -> bool:
    effective_end = occurrence_end or occurrence_start
    return effective_end >= window_start and occurrence_start <= window_end


def _coerce_until(value: TemporalValue, dtstart: datetime) -> datetime:
    if isinstance(value, date) and not isinstance(value, datetime):
        coerced = datetime.combine(value, time.max, tzinfo=dtstart.tzinfo)
    else:
        coerced = _coerce_datetime(value, dtstart.tzinfo)
    if dtstart.tzinfo is None:
        return coerced.replace(tzinfo=None)
    if coerced.tzinfo is None:
        return coerced.replace(tzinfo=dtstart.tzinfo)
    return coerced.astimezone(dtstart.tzinfo)


def _coerce_datetime(value: TemporalValue | None, tz: ZoneInfo | timezone | None = None) -> datetime:
    if value is None:
        raise CalendarValidationError("Calendar query window boundaries are required")
    default_tz = tz or timezone.utc
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str):
        parsed = date_parser.isoparse(value)
    else:
        raise CalendarValidationError(f"Unsupported temporal value: {value!r}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_tz)
    if tz is not None:
        return parsed.astimezone(tz)
    return parsed


def _coerce_date(value: TemporalValue | None, tz: ZoneInfo | timezone) -> date:
    if value is None:
        raise CalendarValidationError("Calendar query window boundaries are required")
    if isinstance(value, datetime):
        return value.astimezone(tz).date() if value.tzinfo is not None else value.date()
    if isinstance(value, date):
        return value
    parsed = date_parser.isoparse(value)
    return parsed.astimezone(tz).date() if parsed.tzinfo is not None else parsed.date()


def _zoneinfo(timezone_name: str | None) -> ZoneInfo | timezone:
    if not timezone_name:
        return timezone.utc
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CalendarValidationError(f"Unsupported calendar timezone: {timezone_name}") from exc


def _format_until(value: TemporalValue) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    parsed = _coerce_datetime(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def _parse_until(value: str) -> date | datetime:
    if len(value) == 8 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d").date()
    return date_parser.parse(value)


__all__ = [
    "LocalRecurrenceRule",
    "RecurrenceOccurrence",
    "expand_recurrence",
    "expand_rrule",
    "validate_query_window",
]
