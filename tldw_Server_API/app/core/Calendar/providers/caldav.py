"""Read-only HTTP-level CalDAV provider adapter."""

from __future__ import annotations

import base64
import ipaddress
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from defusedxml import ElementTree
import httpx
from dateutil import parser as date_parser
from dateutil import tz
from icalendar import Calendar as ICalendar

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError

_DAV_NS = "DAV:"
_CALDAV_NS = "urn:ietf:params:xml:ns:caldav"
_CALSERVER_NS = "http://calendarserver.org/ns/"
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


class CalDavProvider:
    """Minimal CalDAV client for account verification and read-only discovery."""

    def __init__(self, *, http_client: Any | None = None, timeout_seconds: float = 10.0) -> None:
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds, follow_redirects=False)
        self.timeout_seconds = timeout_seconds

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
        except Exception as exc:
            return CalDavVerificationResult(verified=False, status="error", error=str(exc))
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
        principal_url = urljoin(safe_url, principal_href or safe_url)

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
        home_url = urljoin(safe_url, home_href or safe_url)

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
        try:
            calendar = ICalendar.from_ical(ics_payload)
        except ValueError as exc:
            raise CalendarValidationError("Invalid iCalendar payload") from exc

        events: list[CalDavEvent] = []
        for component in calendar.walk("VEVENT"):
            uid = str(component.get("UID") or "").strip()
            if not uid:
                continue
            events.append(
                CalDavEvent(
                    uid=uid,
                    title=str(component.get("SUMMARY") or "Untitled event"),
                    start_at=self._component_datetime_iso(component, "DTSTART"),
                    end_at=self._component_datetime_iso(component, "DTEND"),
                    location=str(component.get("LOCATION") or "") or None,
                    description=str(component.get("DESCRIPTION") or "") or None,
                    source_updated_at=self._component_datetime_iso(component, "LAST-MODIFIED"),
                    provider_payload=sanitize_provider_metadata({"uid": uid}),
                )
            )
        return events

    def fetch_vevents(
        self,
        *,
        remote_calendar_url: str,
        username: str,
        password: str,
        limit: int = 500,
    ) -> list[CalDavEvent]:
        safe_url = self._validate_http_url(remote_calendar_url)
        response = self._request(
            "REPORT",
            safe_url,
            username=username,
            password=password,
            depth="1",
            body="""<?xml version="1.0" encoding="utf-8" ?>
<cal:calendar-query xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav">
  <d:prop><d:getetag /><cal:calendar-data /></d:prop>
  <cal:filter><cal:comp-filter name="VCALENDAR"><cal:comp-filter name="VEVENT" /></cal:comp-filter></cal:filter>
</cal:calendar-query>""",
        )
        root = self._parse_xml(response)
        events: list[CalDavEvent] = []
        for calendar_data in root.findall(f".//{{{_CALDAV_NS}}}calendar-data"):
            if calendar_data.text:
                events.extend(self.parse_vevents(calendar_data.text))
            if len(events) >= limit:
                return events[:limit]
        return events

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
        auth_value = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {auth_value}"
        return self.http_client.request(
            method,
            url,
            headers=headers,
            content=body.encode("utf-8") if body is not None else None,
            timeout=self.timeout_seconds,
        )

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
            return
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code >= 400:
            raise CalendarValidationError(f"CalDAV provider returned HTTP {status_code}")

    @staticmethod
    def _parse_xml(response: Any) -> ElementTree.Element:
        CalDavProvider._raise_for_status(response)
        raw_text = getattr(response, "text", None)
        if raw_text is None:
            content = getattr(response, "content", b"")
            raw_text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
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
                str(component.attrib.get("name", "")).upper()
                for component in prop.findall(f".//{{{_CALDAV_NS}}}comp")
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
                    remote_calendar_id=urljoin(base_url, href.text.strip()),
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
            value = datetime.combine(raw_value, time.min)
        else:
            value = date_parser.parse(str(raw_value))
        if value.tzinfo is None:
            value = value.replace(tzinfo=tz.gettz(str(tzid)) or timezone.utc)
        return value.isoformat()

    @staticmethod
    def _validate_http_url(url: str) -> str:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"}:
            raise CalendarValidationError("CalDAV server URL must use http or https")
        if not parsed.hostname:
            raise CalendarValidationError("CalDAV server URL must include a host")
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
