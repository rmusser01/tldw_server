"""Read-only HTTP-level CalDAV provider adapter."""

from __future__ import annotations

import base64
import ipaddress
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html import escape as html_escape
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from dateutil import parser as date_parser
from dateutil import tz
from defusedxml import ElementTree
from icalendar import Calendar as ICalendar
from icalendar.parser import Contentlines

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.provider_operations import log_calendar_failure
from tldw_Server_API.app.core.Calendar.temporal import add_ical_duration
from tldw_Server_API.app.core.http_client import _prepare_pinned_transport_target, create_client
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy

_DAV_NS = "DAV:"
_CALDAV_NS = "urn:ietf:params:xml:ns:caldav"
_CALSERVER_NS = "http://calendarserver.org/ns/"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_ICS_BYTES = 1024 * 1024
_SECRET_METADATA_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "password",
    "token",
    "access_token",
    "refresh_token",
    "client_secret",
    "secret_ref",
}


@dataclass(frozen=True)
class CalDavVerificationResult:
    verified: bool
    status: str
    error: str | None = None


@dataclass(frozen=True)
class DiscoveredCalendar:
    remote_calendar_id: str
    remote_display_name: str | None
    provider_capabilities: dict[str, Any]


@dataclass(frozen=True)
class CalDavEvent:
    uid: str
    title: str
    start_at: str | None
    end_at: str | None
    location: str | None
    description: str | None
    source_updated_at: str | None = None
    provider_payload: dict[str, Any] | None = None
    all_day: bool = False
    timezone: str | None = None
    status: str = "confirmed"
    rrule: str | None = None
    rdate: list[str] = field(default_factory=list)
    exdate: list[str] = field(default_factory=list)
    recurrence_id: str | None = None


class CalDavProvider:
    """Minimal CalDAV client for account verification and read-only discovery."""

    def __init__(
        self,
        *,
        http_client: Any | None = None,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_ics_bytes: int = MAX_ICS_BYTES,
    ) -> None:
        self.http_client = http_client
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.max_ics_bytes = max_ics_bytes

    def verify_account(self, *, server_url: str, username: str, password: str) -> CalDavVerificationResult:
        safe_url = self._validate_http_url(server_url)
        try:
            response = self._request(
                "OPTIONS",
                safe_url,
                username=username,
                password=password,
            )
            self._raise_for_status(response)
        except (httpx.HTTPError, OSError, CalendarValidationError) as exc:
            log_calendar_failure("verify_account", exc)
            return CalDavVerificationResult(
                verified=False, status="error", error=f"CalDAV verification failed ({type(exc).__name__})"
            )
        return CalDavVerificationResult(verified=True, status="ok", error=None)

    def discover_calendars(
        self,
        *,
        server_url: str,
        username: str,
        password: str,
    ) -> list[DiscoveredCalendar]:
        safe_url = self._validate_http_url(server_url)
        options_response = self._request("OPTIONS", safe_url, username=username, password=password)
        self._raise_for_status(options_response)
        dav_header = str(getattr(options_response, "headers", {}) or {}).lower()
        server_supports_sync = "sync-collection" in dav_header

        principal_response = self._request(
            "PROPFIND",
            safe_url,
            username=username,
            password=password,
            depth="0",
            body="""<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:"><d:prop><d:current-user-principal /></d:prop></d:propfind>""",
        )
        principal_href = self._first_href(
            self._parse_xml(principal_response),
            parent_tag=f"{{{_DAV_NS}}}current-user-principal",
        )
        principal_url = self.same_origin_url(safe_url, principal_href or safe_url)

        home_response = self._request(
            "PROPFIND",
            principal_url,
            username=username,
            password=password,
            depth="0",
            body="""<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav">
  <d:prop><cal:calendar-home-set /></d:prop>
</d:propfind>""",
        )
        home_href = self._first_href(
            self._parse_xml(home_response),
            parent_tag=f"{{{_CALDAV_NS}}}calendar-home-set",
        )
        home_url = self.same_origin_url(safe_url, home_href or safe_url)

        calendar_response = self._request(
            "PROPFIND",
            home_url,
            username=username,
            password=password,
            depth="1",
            body="""<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav" xmlns:cs="http://calendarserver.org/ns/">
  <d:prop>
    <d:displayname />
    <d:resourcetype />
    <cs:getctag />
    <d:sync-token />
    <cal:supported-calendar-component-set />
  </d:prop>
</d:propfind>""",
        )
        return self._parse_calendar_home(
            self._parse_xml(calendar_response),
            base_url=safe_url,
            server_supports_sync=server_supports_sync,
        )

    def parse_vevents(self, ics_payload: str) -> list[CalDavEvent]:
        if len(ics_payload.encode("utf-8")) > self.max_ics_bytes:
            raise CalendarValidationError("iCalendar payload exceeds byte limit")
        try:
            calendar = ICalendar.from_ical(ics_payload)
            duration_texts = _event_duration_texts(ics_payload)
        except ValueError as exc:
            raise CalendarValidationError("Invalid iCalendar payload") from exc

        events: list[CalDavEvent] = []
        for event_index, component in enumerate(calendar.walk("VEVENT")):
            uid = str(component.get("UID") or "").strip()
            if not uid:
                continue
            if "\x00" in uid:
                raise CalendarValidationError("CalDAV UID cannot contain a NUL character")
            start_value = (
                component.decoded("DTSTART")
                if component.get("DTSTART")
                else (component.decoded("RECURRENCE-ID") if component.get("RECURRENCE-ID") else None)
            )
            recurrence_id = self._component_datetime_iso(component, "RECURRENCE-ID")
            rule = component.get("RRULE")
            rule_text = rule.to_ical().decode("utf-8") if rule else None
            rdates = self._component_dates(component, "RDATE")
            exdates = self._component_dates(component, "EXDATE")
            tzid = component.get("DTSTART").params.get("TZID") if component.get("DTSTART") else None
            end_at = self._component_datetime_iso(component, "DTEND")
            if component.get("DURATION") is not None:
                if end_at is not None:
                    raise CalendarValidationError("VEVENT cannot specify both DTEND and DURATION")
                duration_text = duration_texts[event_index]
                if start_value is None or duration_text is None:
                    raise CalendarValidationError("Invalid VEVENT duration")
                end_at = add_ical_duration(start_value, duration_text).isoformat()
            events.append(
                CalDavEvent(
                    uid=uid,
                    title=str(component.get("SUMMARY") or "Untitled event"),
                    start_at=self._component_datetime_iso(component, "DTSTART") or recurrence_id,
                    end_at=end_at,
                    location=str(component.get("LOCATION") or "") or None,
                    description=str(component.get("DESCRIPTION") or "") or None,
                    source_updated_at=self._component_datetime_iso(component, "LAST-MODIFIED"),
                    all_day=isinstance(start_value, date) and not isinstance(start_value, datetime),
                    timezone=str(tzid) if tzid else None,
                    status=str(component.get("STATUS") or "confirmed").lower(),
                    rrule=rule_text,
                    rdate=rdates,
                    exdate=exdates,
                    recurrence_id=recurrence_id,
                    provider_payload=sanitize_provider_metadata(
                        {
                            "uid": uid,
                            "rrule": rule_text,
                            "rdate": rdates,
                            "exdate": exdates,
                            "recurrence_id": recurrence_id,
                            "duration": duration_texts[event_index],
                        }
                    ),
                )
            )
        return events

    def fetch_vevents(
        self,
        *,
        remote_calendar_url: str,
        username: str,
        password: str,
        window_start: str | None = None,
        window_end: str | None = None,
        limit: int = 500,
    ) -> list[CalDavEvent]:
        safe_url = self._validate_http_url(remote_calendar_url)
        time_range = self._calendar_query_time_range(window_start=window_start, window_end=window_end)
        response = self._request(
            "REPORT",
            safe_url,
            username=username,
            password=password,
            depth="1",
            body=f"""<?xml version="1.0" encoding="utf-8" ?>
<cal:calendar-query xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav">
  <d:prop><d:getetag /><cal:calendar-data /></d:prop>
  <cal:filter><cal:comp-filter name="VCALENDAR"><cal:comp-filter name="VEVENT">{time_range}</cal:comp-filter></cal:comp-filter></cal:filter>
</cal:calendar-query>""",
        )
        root = self._parse_xml(response)
        events: list[CalDavEvent] = []
        for calendar_data in root.findall(f".//{{{_CALDAV_NS}}}calendar-data"):
            if calendar_data.text:
                events.extend(self.parse_vevents(calendar_data.text))
            if len(events) >= limit:
                raise CalendarValidationError("CalDAV response reached the event limit; widen or split the sync window")
        return events

    @staticmethod
    def _calendar_query_time_range(
        *,
        window_start: str | None,
        window_end: str | None,
    ) -> str:
        if not window_start or not window_end:
            return ""
        start = CalDavProvider._caldav_timestamp(window_start)
        end = CalDavProvider._caldav_timestamp(window_end)
        return f'<cal:time-range start="{html_escape(start, quote=True)}" end="{html_escape(end, quote=True)}" />'

    @staticmethod
    def _caldav_timestamp(value: str) -> str:
        parsed = date_parser.parse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _request(
        self,
        method: str,
        url: str,
        *,
        username: str,
        password: str,
        depth: str | None = None,
        body: str | None = None,
    ) -> Any:
        headers = {"Accept": "application/xml, text/calendar;q=0.9, */*;q=0.1"}
        if depth is not None:
            headers["Depth"] = depth
        if body is not None:
            headers["Content-Type"] = "application/xml; charset=utf-8"
        auth_value = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {auth_value}"
        safe_url = self._validate_http_url(url)
        content = body.encode("utf-8") if body is not None else None
        if self.http_client is not None:
            response = self.http_client.request(
                method,
                safe_url,
                headers=headers,
                content=content,
                timeout=self.timeout_seconds,
            )
            raw = getattr(response, "content", None)
            if raw is None:
                raw = str(getattr(response, "text", "")).encode("utf-8")
            if len(raw) > self.max_response_bytes:
                raise CalendarValidationError("CalDAV response exceeds byte limit")
            return response
        policy = evaluate_url_policy(safe_url, block_private_override=True)
        if not policy.allowed or not policy.resolved_ips:
            raise CalendarValidationError("CalDAV server URL is blocked by outbound network policy")
        transport_url, transport_headers, sni_hostname = _prepare_pinned_transport_target(
            safe_url, headers, tuple(policy.resolved_ips)
        )
        with create_client(timeout=self.timeout_seconds, trust_env=False) as client:
            with client.stream(
                method,
                transport_url,
                headers=transport_headers,
                content=content,
                extensions={"sni_hostname": sni_hostname} if sni_hostname else None,
                follow_redirects=False,
            ) as response:
                self._raise_for_status(response)
                declared_size = response.headers.get("Content-Length")
                if declared_size and declared_size.isdigit() and int(declared_size) > self.max_response_bytes:
                    raise CalendarValidationError("CalDAV response exceeds byte limit")
                content_buffer = bytearray()
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if len(content_buffer) + len(chunk) > self.max_response_bytes:
                        raise CalendarValidationError("CalDAV response exceeds byte limit")
                    content_buffer.extend(chunk)
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=bytes(content_buffer),
                    request=httpx.Request(method, safe_url),
                )

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code >= 300:
            raise CalendarValidationError(f"CalDAV provider returned HTTP {status_code}")

    @staticmethod
    def _parse_xml(response: Any) -> ElementTree.Element:
        CalDavProvider._raise_for_status(response)
        raw_text = getattr(response, "text", None)
        if raw_text is None:
            content = getattr(response, "content", b"")
            raw_text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        if len(raw_text.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise CalendarValidationError("CalDAV XML exceeds byte limit")
        try:
            return ElementTree.fromstring(raw_text)
        except ElementTree.ParseError as exc:
            raise CalendarValidationError("CalDAV provider returned invalid XML") from exc

    @staticmethod
    def _first_href(root: ElementTree.Element, *, parent_tag: str) -> str | None:
        parent = root.find(f".//{parent_tag}")
        if parent is None:
            return None
        href = parent.find(f"{{{_DAV_NS}}}href")
        return href.text.strip() if href is not None and href.text else None

    @staticmethod
    def _parse_calendar_home(
        root: ElementTree.Element,
        *,
        base_url: str,
        server_supports_sync: bool,
    ) -> list[DiscoveredCalendar]:
        calendars: list[DiscoveredCalendar] = []
        for response in root.findall(f"{{{_DAV_NS}}}response"):
            prop = response.find(f".//{{{_DAV_NS}}}prop")
            href = response.find(f"{{{_DAV_NS}}}href")
            if prop is None or href is None or not href.text:
                continue
            resource_type = prop.find(f"{{{_DAV_NS}}}resourcetype")
            if resource_type is None or resource_type.find(f"{{{_CALDAV_NS}}}calendar") is None:
                continue
            display_name = _element_text(prop.find(f"{{{_DAV_NS}}}displayname"))
            ctag = _element_text(prop.find(f"{{{_CALSERVER_NS}}}getctag"))
            sync_token = _element_text(prop.find(f"{{{_DAV_NS}}}sync-token"))
            component_names = {
                str(component.attrib.get("name", "")).upper() for component in prop.findall(f".//{{{_CALDAV_NS}}}comp")
            }
            supports_sync_token = bool(server_supports_sync and sync_token)
            capabilities = sanitize_provider_metadata(
                {
                    "supports_vevent": "VEVENT" in component_names or not component_names,
                    "supports_vtodo": "VTODO" in component_names,
                    "supports_sync_token": supports_sync_token,
                    "sync_strategy": "sync_token" if supports_sync_token else "bounded_polling",
                    "ctag": ctag,
                    "sync_token": sync_token if supports_sync_token else None,
                }
            )
            calendars.append(
                DiscoveredCalendar(
                    remote_calendar_id=CalDavProvider.same_origin_url(base_url, href.text.strip()),
                    remote_display_name=display_name,
                    provider_capabilities={key: value for key, value in capabilities.items() if value is not None},
                )
            )
        return calendars

    @staticmethod
    def _component_datetime_iso(component: Any, name: str) -> str | None:
        if component.get(name) is None:
            return None
        raw_value = component.decoded(name)
        tzid = None
        try:
            tzid = component.get(name).params.get("TZID")
        except AttributeError:
            tzid = None
        if isinstance(raw_value, datetime):
            value = raw_value
        elif isinstance(raw_value, date):
            return raw_value.isoformat()
        else:
            value = date_parser.parse(str(raw_value))
        if value.tzinfo is None:
            value = value.replace(tzinfo=tz.gettz(str(tzid)) or timezone.utc)
        if name == "RECURRENCE-ID":
            value = value.astimezone(timezone.utc)
        return value.isoformat()

    @staticmethod
    def _component_dates(component: Any, name: str) -> list[str]:
        properties = component.get(name)
        if properties is None:
            return []
        dates = []
        for prop in properties if isinstance(properties, list) else [properties]:
            for entry in prop.dts:
                value = entry.dt
                if isinstance(value, datetime) and value.tzinfo is None:
                    value = value.replace(tzinfo=tz.gettz(str(prop.params.get("TZID"))) or timezone.utc)
                if not isinstance(value, (date, datetime)):
                    raise CalendarValidationError("CalDAV recurrence periods are not supported")
                dates.append(value.isoformat())
        return dates

    @staticmethod
    def _validate_http_url(url: str) -> str:
        """Validate public HTTPS URL syntax, including ports, without network I/O.

        Return its normalized spelling or raise CalendarValidationError; never
        propagate urlparse/port ValueError for account or provider-supplied input.
        """
        try:
            parsed = urlparse(str(url).strip())
            port = parsed.port
        except ValueError as exc:
            raise CalendarValidationError("CalDAV server URL must have a valid host and port") from exc
        if port is not None and not 1 <= port <= 65535:
            raise CalendarValidationError("CalDAV server URL port must be between 1 and 65535")
        if parsed.scheme != "https":
            raise CalendarValidationError("CalDAV server URL must use https")
        if not parsed.hostname:
            raise CalendarValidationError("CalDAV server URL must include a host")
        if parsed.username or parsed.password or parsed.fragment:
            raise CalendarValidationError("CalDAV server URL cannot include credentials or a fragment")
        hostname = parsed.hostname.lower()
        if hostname == "localhost":
            raise CalendarValidationError("CalDAV server URL cannot target localhost")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return parsed.geturl()
        if (
            address.is_loopback
            or address.is_link_local
            or address.is_private
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise CalendarValidationError("CalDAV server URL cannot target private or local addresses")
        return parsed.geturl()

    @staticmethod
    def same_origin_url(base_url: str, href: str) -> str:
        """Resolve a collection URL and reject invalid or cross-origin destinations."""
        base = urlparse(CalDavProvider._validate_http_url(base_url))
        try:
            joined = urljoin(base_url, href)
        except ValueError as exc:
            raise CalendarValidationError("CalDAV calendar URL must have a valid host and port") from exc
        resolved = CalDavProvider._validate_http_url(joined)
        target = urlparse(resolved)
        base_origin = (base.scheme, base.hostname, base.port or (443 if base.scheme == "https" else 80))
        target_origin = (target.scheme, target.hostname, target.port or (443 if target.scheme == "https" else 80))
        if target_origin != base_origin:
            raise CalendarValidationError("CalDAV calendar URL must use the same origin as the account server")
        return resolved


def _event_duration_texts(payload: str) -> list[str | None]:
    """Retain lexical DURATION: the library's timedelta loses P1D versus PT24H."""
    stack: list[str] = []
    durations: list[str | None] = []
    for line in Contentlines.from_ical(payload):
        if not line:
            continue
        name, _params, value = line.parts()
        name = name.upper()
        if name == "BEGIN":
            stack.append(value.upper())
            if stack[-1] == "VEVENT":
                durations.append(None)
        elif name == "END":
            stack.pop()
        elif name == "DURATION" and stack and stack[-1] == "VEVENT":
            if durations[-1] is not None:
                raise CalendarValidationError("VEVENT cannot specify multiple durations")
            durations[-1] = value
    return durations


def sanitize_provider_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_METADATA_KEYS:
                continue
            cleaned = sanitize_provider_metadata(item)
            if cleaned is not None:
                sanitized[key] = cleaned
        return sanitized
    if isinstance(value, list):
        return [item for item in (sanitize_provider_metadata(item) for item in value) if item is not None]
    return value


def _element_text(element: ElementTree.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text or None
