"""iCalendar adapter for file artifact normalization, validation, and export."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from tldw_Server_API.app.core.exceptions import FileArtifactsError, FileArtifactsValidationError
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult, ValidationIssue


class IcalAdapter:
    """Normalize, validate, and export iCalendar payloads as ICS."""
    file_type: ClassVar[str] = "ical"
    export_formats: ClassVar[set[str]] = {"ics"}
    _UTC_EQUIVALENTS: ClassVar[set[str]] = {"UTC", "ETC/UTC", "ETC/GMT", "GMT", "Z"}

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize calendar payload shape and apply default values."""
        calendar = payload.get("calendar")
        if not isinstance(calendar, dict):
            raise FileArtifactsValidationError("calendar_required")
        prodid = calendar.get("prodid") or "-//tldw//files//EN"
        version = calendar.get("version") or "2.0"
        calendar_timezone = calendar.get("timezone")
        if calendar_timezone is not None and not isinstance(calendar_timezone, str):
            raise FileArtifactsValidationError("calendar_timezone_invalid")
        events = calendar.get("events")
        if events is None:
            events = []
        if not isinstance(events, list):
            raise FileArtifactsValidationError("events_must_be_list")
        return {
            "calendar": {
                "prodid": str(prodid),
                "version": str(version),
                "timezone": calendar_timezone,
                "events": events,
            }
        }

    def validate(self, structured: dict[str, Any]) -> list[ValidationIssue]:
        """Validate a structured calendar payload and return issues."""
        issues: list[ValidationIssue] = []
        calendar = structured.get("calendar")
        if not isinstance(calendar, dict):
            return [ValidationIssue(code="calendar_required", message="calendar must be an object", path="calendar")]
        events = calendar.get("events")
        if not isinstance(events, list):
            issues.append(ValidationIssue(code="events_required", message="events must be a list", path="calendar.events"))
            return issues
        calendar_tz = calendar.get("timezone")
        if calendar_tz is not None and not isinstance(calendar_tz, str):
            issues.append(
                ValidationIssue(code="calendar_timezone_invalid", message="calendar timezone must be a string", path="calendar.timezone")
            )
            return issues
        if calendar_tz and not self._timezone_valid(calendar_tz):
            issues.append(
                ValidationIssue(code="calendar_timezone_invalid", message="calendar timezone must be a valid IANA timezone", path="calendar.timezone")
            )
            return issues
        for idx, event in enumerate(events):
            if not isinstance(event, dict):
                issues.append(ValidationIssue(code="event_invalid", message="event must be an object", path=f"calendar.events[{idx}]"))
                continue
            for field in ("uid", "summary", "start"):
                if not event.get(field):
                    issues.append(
                        ValidationIssue(
                            code="event_missing_field",
                            message=f"event missing {field}",
                            path=f"calendar.events[{idx}].{field}",
                        )
                    )
            event_tz = event.get("timezone") or calendar_tz
            start_dt, start_issues = self._parse_event_datetime(
                event.get("start"),
                event_tz,
                path=f"calendar.events[{idx}].start",
            )
            issues.extend(start_issues)
            end_dt, end_issues = self._parse_event_datetime(
                event.get("end"),
                event_tz,
                path=f"calendar.events[{idx}].end",
            )
            issues.extend(end_issues)
            if start_dt and end_dt:
                if (type(start_dt) is date) != (type(end_dt) is date):
                    issues.append(
                        ValidationIssue(
                            code="event_date_type_mismatch",
                            message="start and end must both be date or datetime",
                            path=f"calendar.events[{idx}]",
                        )
                    )
                elif end_dt < start_dt:
                    issues.append(
                        ValidationIssue(
                            code="event_end_before_start",
                            message="event end must be after start",
                            path=f"calendar.events[{idx}].end",
                        )
                    )
        if issues:
            return issues
        try:
            _ = self._build_calendar(structured)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    code="icalendar_validation_failed",
                    message=str(exc),
                    path="calendar",
                )
            )
        return issues

    def export(self, structured: dict[str, Any], *, format: str) -> ExportResult:
        """Export the structured payload as an ICS file."""
        if format != "ics":
            raise FileArtifactsValidationError("unsupported_format")
        cal = self._build_calendar(structured)
        data = cal.to_ical()
        return ExportResult(status="ready", content_type="text/calendar", bytes_len=len(data), content=data)

    @staticmethod
    def _timezone_valid(tzid: str) -> bool:
        """Return True when tzid is a valid IANA timezone identifier."""
        try:
            ZoneInfo(tzid)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_date_only(raw: str) -> bool:
        """Return True if the string is in YYYY-MM-DD format."""
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", raw))

    def _parse_event_datetime(
        self,
        raw: Any,
        tzid: str | None,
        *,
        path: str,
    ) -> tuple[datetime | date | None, list[ValidationIssue]]:
        """Parse event date/datetime values and collect validation issues."""
        issues: list[ValidationIssue] = []
        if raw is None:
            return None, issues
        if not isinstance(raw, str):
            issues.append(ValidationIssue(code="event_datetime_invalid", message="value must be a string", path=path))
            return None, issues
        candidate = raw.strip()
        if self._is_date_only(candidate):
            try:
                return datetime.strptime(candidate, "%Y-%m-%d").date(), issues
            except ValueError:
                issues.append(ValidationIssue(code="event_date_invalid", message="date must be YYYY-MM-DD", path=path))
                return None, issues

        try:
            from dateutil.parser import isoparse
        except Exception as exc:
            issues.append(ValidationIssue(code="datetime_parser_unavailable", message=str(exc), path=path))
            return None, issues

        try:
            dt = isoparse(candidate)
        except Exception:
            issues.append(ValidationIssue(code="event_datetime_invalid", message="datetime must be ISO8601", path=path))
            return None, issues

        if dt.tzinfo is None:
            if not tzid:
                issues.append(
                    ValidationIssue(
                        code="event_timezone_required",
                        message="timezone required for datetime values",
                        path=path,
                    )
                )
                return None, issues
            try:
                tzinfo = ZoneInfo(tzid)
            except Exception:
                issues.append(
                    ValidationIssue(
                        code="event_timezone_invalid",
                        message="timezone must be a valid IANA timezone",
                        path=path,
                    )
                )
                return None, issues
            dt = dt.replace(tzinfo=tzinfo)
        else:
            if tzid and not self._tz_matches(dt, tzid):
                issues.append(
                    ValidationIssue(
                        code="event_timezone_mismatch",
                        message="datetime timezone does not match event timezone",
                        path=path,
                    )
                )
                return None, issues
        return dt, issues

    def _tz_matches(self, dt: datetime, tzid: str) -> bool:
        """Return True when the datetime timezone matches the requested tzid."""
        tzinfo = dt.tzinfo
        if tzinfo is None:
            return False
        tz_key = getattr(tzinfo, "key", None)
        if tz_key == tzid:
            return True
        dt_offset = dt.utcoffset()
        if dt_offset is None:
            return False
        offset = self._parse_offset_tzid(tzid)
        if offset is not None:
            return dt_offset == offset
        try:
            expected_tz = ZoneInfo(tzid)
        except Exception:
            return False
        expected_offset = expected_tz.utcoffset(dt)
        if expected_offset is None:
            return False
        return dt_offset == expected_offset

    @classmethod
    def _parse_offset_tzid(cls, tzid: str) -> timedelta | None:
        """Parse a timezone id into a fixed offset when possible."""
        tzid = tzid.strip()
        tz_upper = tzid.upper()
        if tz_upper in cls._UTC_EQUIVALENTS:
            return timedelta(0)
        if tz_upper.startswith(("UTC", "GMT")) and len(tz_upper) > 3:
            offset = cls._parse_utc_offset(tz_upper[3:])
            if offset is not None:
                return offset
        return cls._parse_utc_offset(tz_upper)

    @staticmethod
    def _parse_utc_offset(raw: str) -> timedelta | None:
        """Parse a UTC offset string like +HHMM or +HH:MM."""
        match = re.match(r"^(?P<sign>[+-])(?P<hours>\d{2})(?::?(?P<minutes>\d{2}))?$", raw)
        if not match:
            return None
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes") or "0")
        if hours > 23 or minutes > 59:
            return None
        sign = -1 if match.group("sign") == "-" else 1
        return timedelta(hours=sign * hours, minutes=sign * minutes)

    def _build_calendar(self, structured: dict[str, Any]):
        """Build an icalendar.Calendar from a structured payload."""
        try:
            from icalendar import Calendar, Event
        except Exception as exc:
            raise FileArtifactsError("icalendar_library_unavailable", detail="icalendar export backend unavailable") from exc

        calendar = structured.get("calendar") or {}
        cal = Calendar()
        cal.add("prodid", calendar.get("prodid") or "-//tldw//files//EN")
        cal.add("version", calendar.get("version") or "2.0")
        calendar_tz = calendar.get("timezone")
        for idx, event in enumerate(calendar.get("events") or []):
            if not isinstance(event, dict):
                continue
            event_uid = event.get("uid")
            event_ref = f"index={idx}"
            if event_uid:
                event_ref = f"{event_ref} uid={event_uid}"
            event_tz = event.get("timezone") or calendar_tz
            start_dt, start_issues = self._parse_event_datetime(
                event.get("start"),
                event_tz,
                path=f"calendar.events[{idx}].start",
            )
            if start_issues or not start_dt:
                raise FileArtifactsValidationError(f"event_start_invalid ({event_ref})")
            end_dt, end_issues = self._parse_event_datetime(
                event.get("end"),
                event_tz,
                path=f"calendar.events[{idx}].end",
            )
            if end_issues:
                raise FileArtifactsValidationError(f"event_end_invalid ({event_ref})")

            ical_event = Event()
            ical_event.add("uid", event.get("uid"))
            ical_event.add("summary", event.get("summary"))
            ical_event.add("dtstart", start_dt)
            if end_dt is not None:
                ical_event.add("dtend", end_dt)
            if event.get("description"):
                ical_event.add("description", event.get("description"))
            if event.get("location"):
                ical_event.add("location", event.get("location"))
            cal.add_component(ical_event)
        return cal
