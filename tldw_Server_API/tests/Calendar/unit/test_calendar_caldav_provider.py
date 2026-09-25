from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider, sanitize_provider_metadata
from tldw_Server_API.app.core.Calendar.providers import caldav as caldav_module

pytestmark = pytest.mark.unit


@dataclass
class _FakeResponse:
    status_code: int = 207
    text: str = ""
    headers: dict[str, str] | None = None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected {method} request to {url}")
        return self.responses.pop(0)


def _principal_xml() -> str:
    return """
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/dav/</d:href>
        <d:propstat>
          <d:prop>
            <d:current-user-principal><d:href>/principals/user/</d:href></d:current-user-principal>
          </d:prop>
        </d:propstat>
      </d:response>
    </d:multistatus>
    """


def _home_set_xml() -> str:
    return """
    <d:multistatus xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav">
      <d:response>
        <d:href>/principals/user/</d:href>
        <d:propstat>
          <d:prop>
            <cal:calendar-home-set><d:href>/calendars/user/</d:href></cal:calendar-home-set>
          </d:prop>
        </d:propstat>
      </d:response>
    </d:multistatus>
    """


def _calendar_home_xml(*, sync_token: str | None = "sync-1") -> str:
    sync_token_xml = f"<d:sync-token>{sync_token}</d:sync-token>" if sync_token else ""
    return f"""
    <d:multistatus
        xmlns:d="DAV:"
        xmlns:cal="urn:ietf:params:xml:ns:caldav"
        xmlns:cs="http://calendarserver.org/ns/">
      <d:response>
        <d:href>/calendars/user/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection /></d:resourcetype></d:prop></d:propstat>
      </d:response>
      <d:response>
        <d:href>/calendars/user/work/</d:href>
        <d:propstat>
          <d:prop>
            <d:displayname>Work</d:displayname>
            <d:resourcetype><d:collection /><cal:calendar /></d:resourcetype>
            <cs:getctag>ctag-1</cs:getctag>
            {sync_token_xml}
            <cal:supported-calendar-component-set>
              <cal:comp name="VEVENT" />
            </cal:supported-calendar-component-set>
          </d:prop>
        </d:propstat>
      </d:response>
    </d:multistatus>
    """


def test_verify_account_rejects_non_http_urls() -> None:
    provider = CalDavProvider()

    with pytest.raises(CalendarValidationError, match="http"):
        provider.verify_account(server_url="file:///etc/passwd", username="reader", password="secret")


def test_verify_account_rejects_plain_http_credentials() -> None:
    provider = CalDavProvider()

    with pytest.raises(CalendarValidationError, match="https"):
        provider.verify_account(server_url="http://calendar.example.test/dav/", username="reader", password="secret")


def test_default_transport_rejects_dns_targets_denied_by_egress_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []
    monkeypatch.setattr(
        caldav_module,
        "evaluate_url_policy",
        lambda url, **kwargs: SimpleNamespace(allowed=False),
    )
    monkeypatch.setattr(
        caldav_module,
        "create_client",
        lambda **kwargs: requested.append("created"),
    )
    provider = CalDavProvider()

    with pytest.raises(CalendarValidationError, match="outbound network policy"):
        provider._request("OPTIONS", "https://calendar.example.test/dav/", username="user", password="secret")

    assert requested == []


def test_default_transport_connects_to_vetted_ip_with_original_host_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"client_options": kwargs})

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append({"method": method, "url": url, **kwargs})
            return _FakeResponse(status_code=200)

    monkeypatch.setattr(caldav_module, "create_client", _Client)
    monkeypatch.setattr(
        caldav_module,
        "evaluate_url_policy",
        lambda url, **kwargs: SimpleNamespace(allowed=True, resolved_ips=("93.184.216.34",)),
    )

    CalDavProvider()._request("OPTIONS", "https://calendar.example.test/dav/", username="user", password="secret")

    assert calls[0]["client_options"]["trust_env"] is False
    assert calls[1]["url"] == "https://93.184.216.34/dav/"
    assert calls[1]["headers"]["Host"] == "calendar.example.test"
    assert calls[1]["extensions"] == {"sni_hostname": "calendar.example.test"}


def test_discovery_rejects_private_principal_url() -> None:
    http_client = _FakeHttpClient(
        [
            _FakeResponse(status_code=200),
            _FakeResponse(text=_principal_xml().replace("/principals/user/", "https://127.0.0.1/private/")),
        ]
    )

    with pytest.raises(CalendarValidationError, match="private or local"):
        CalDavProvider(http_client=http_client).discover_calendars(
            server_url="https://caldav.example.test/dav/", username="user", password="secret"
        )

    assert len(http_client.requests) == 2


def test_discovery_rejects_cross_origin_principal_before_sending_credentials() -> None:
    http_client = _FakeHttpClient(
        [
            _FakeResponse(status_code=200),
            _FakeResponse(text=_principal_xml().replace("/principals/user/", "https://other.example.test/user/")),
        ]
    )

    with pytest.raises(CalendarValidationError, match="same origin"):
        CalDavProvider(http_client=http_client).discover_calendars(
            server_url="https://caldav.example.test/dav/", username="user", password="secret"
        )

    assert len(http_client.requests) == 2


def test_fetch_rejects_capped_result_instead_of_returning_partial_snapshot() -> None:
    response = _FakeResponse(
        text="""<d:multistatus xmlns:d="DAV:" xmlns:cal="urn:ietf:params:xml:ns:caldav">
<d:response><d:propstat><d:prop><cal:calendar-data>BEGIN:VCALENDAR
BEGIN:VEVENT
UID:event-1
SUMMARY:Planning
DTSTART:20260605T160000Z
END:VEVENT
END:VCALENDAR</cal:calendar-data></d:prop></d:propstat></d:response>
</d:multistatus>"""
    )
    provider = CalDavProvider(http_client=_FakeHttpClient([response]))

    with pytest.raises(CalendarValidationError, match="event limit"):
        provider.fetch_vevents(
            remote_calendar_url="https://caldav.example.test/calendar/",
            username="user",
            password="secret",
            limit=1,
        )


def test_discovery_records_sync_token_capabilities() -> None:
    http_client = _FakeHttpClient(
        [
            _FakeResponse(status_code=200, headers={"DAV": "1, 3, calendar-access, sync-collection"}),
            _FakeResponse(text=_principal_xml()),
            _FakeResponse(text=_home_set_xml()),
            _FakeResponse(text=_calendar_home_xml(sync_token="sync-1")),
        ]
    )
    provider = CalDavProvider(http_client=http_client)

    calendars = provider.discover_calendars(
        server_url="https://caldav.example.test/dav/",
        username="reader@example.test",
        password="app-secret",
    )

    assert len(calendars) == 1
    discovered = calendars[0]
    assert discovered.remote_calendar_id == "https://caldav.example.test/calendars/user/work/"
    assert discovered.remote_display_name == "Work"
    assert discovered.provider_capabilities["supports_vevent"] is True
    assert discovered.provider_capabilities["supports_sync_token"] is True
    assert discovered.provider_capabilities["sync_strategy"] == "sync_token"
    assert discovered.provider_capabilities["ctag"] == "ctag-1"
    assert discovered.provider_capabilities["sync_token"] == "sync-1"
    assert "app-secret" not in repr(discovered.provider_capabilities)
    assert http_client.requests[0]["method"] == "OPTIONS"


def test_discovery_without_sync_token_falls_back_to_bounded_polling() -> None:
    http_client = _FakeHttpClient(
        [
            _FakeResponse(status_code=200, headers={"DAV": "1, calendar-access"}),
            _FakeResponse(text=_principal_xml()),
            _FakeResponse(text=_home_set_xml()),
            _FakeResponse(text=_calendar_home_xml(sync_token=None)),
        ]
    )
    provider = CalDavProvider(http_client=http_client)

    calendars = provider.discover_calendars(
        server_url="https://caldav.example.test/dav/",
        username="reader@example.test",
        password="app-secret",
    )

    assert calendars[0].provider_capabilities["supports_sync_token"] is False
    assert calendars[0].provider_capabilities["sync_strategy"] == "bounded_polling"


def test_parse_vevents_ignores_vtodo_and_returns_timezone_aware_dates() -> None:
    provider = CalDavProvider()
    ics = """
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:event-1
SUMMARY:Planning
DTSTART;TZID=America/Los_Angeles:20260605T090000
DTEND;TZID=America/Los_Angeles:20260605T100000
LOCATION:Room 3
END:VEVENT
BEGIN:VTODO
UID:todo-1
SUMMARY:Do not import yet
DUE:20260605T190000Z
END:VTODO
END:VCALENDAR
"""

    events = provider.parse_vevents(ics)

    assert len(events) == 1
    assert events[0].uid == "event-1"
    assert events[0].title == "Planning"
    assert events[0].start_at == "2026-06-05T09:00:00-07:00"
    assert events[0].end_at == "2026-06-05T10:00:00-07:00"
    assert events[0].location == "Room 3"


def test_provider_metadata_scrubs_auth_values() -> None:
    metadata = sanitize_provider_metadata(
        {
            "headers": {"Authorization": "Basic secret", "Depth": "1"},
            "token": "secret-token",
            "nested": {"password": "secret-password", "safe": True},
        }
    )

    assert metadata == {"headers": {"Depth": "1"}, "nested": {"safe": True}}
