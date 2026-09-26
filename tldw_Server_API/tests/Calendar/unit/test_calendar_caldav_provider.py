from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.providers import caldav as caldav_module
from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider, sanitize_provider_metadata

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("url", ["https://example.test:bad/dav/", "https://example.test:65536/dav/", "https://example.test:0/dav/", "https://[invalid/dav/"])
def test_url_validation_rejects_malformed_ports_and_hosts(url: str) -> None:
    """Malformed URL parsing remains a domain validation error before any I/O."""
    with pytest.raises(CalendarValidationError):
        CalDavProvider._validate_http_url(url)


@pytest.mark.parametrize("base, href", [
    ("https://example.test:bad/dav/", "/calendar/"),
    ("https://example.test/dav/", "https://example.test:bad/calendar/"),
    ("https://example.test/dav/", "https://[invalid/calendar/"),
    ("https://example.test/dav/", "//[invalid/calendar/"),
])
def test_same_origin_rejects_malformed_ports_with_domain_error(base: str, href: str) -> None:
    """Untrusted account or collection authorities never escape as ValueError."""
    with pytest.raises(CalendarValidationError):
        CalDavProvider.same_origin_url(base, href)


@pytest.mark.parametrize(
    ("start", "duration", "expected_end"),
    [
        ("DTSTART:20260605T090000Z", "PT1H", "2026-06-05T10:00:00+00:00"),
        ("DTSTART;VALUE=DATE:20260605", "P2D", "2026-06-07"),
    ],
)
def test_parse_derives_exclusive_end_from_duration(start: str, duration: str, expected_end: str) -> None:
    """DURATION supplies the same exclusive end as an equivalent DTEND."""
    event = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:duration\n{start}\nDURATION:{duration}\nEND:VEVENT\nEND:VCALENDAR"
    )[0]
    assert event.end_at == expected_end


@pytest.mark.parametrize("duration", ["-PT1H", "PT0S", "PT1H"])
def test_parse_rejects_invalid_all_day_duration(duration: str) -> None:
    """All-day events require a strictly positive whole-day duration."""
    with pytest.raises(CalendarValidationError):
        CalDavProvider().parse_vevents(
            f"BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:duration\nDTSTART;VALUE=DATE:20260605\n"
            f"DURATION:{duration}\nEND:VEVENT\nEND:VCALENDAR"
        )


@pytest.mark.parametrize(
    ("start", "duration", "expected_end"),
    [
        ("20260308T013000", "PT2H", "2026-03-08T04:30:00-07:00"),
        ("20261101T003000", "PT2H", "2026-11-01T01:30:00-08:00"),
        ("20260307T120000", "PT24H", "2026-03-08T13:00:00-07:00"),
        ("20260307T120000", "P1D", "2026-03-08T12:00:00-07:00"),
    ],
)
def test_duration_preserves_elapsed_hours_and_nominal_days_at_dst(
    start: str, duration: str, expected_end: str,
) -> None:
    """Accurate hour durations and nominal civil days retain distinct DST semantics."""
    event = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:dst\nDTSTART;TZID=America/Los_Angeles:{start}\n"
        f"DURATION:{duration}\nEND:VEVENT\nEND:VCALENDAR"
    )[0]
    assert event.end_at == expected_end


def test_parse_preserves_date_only_events_and_exclusive_end() -> None:
    event = CalDavProvider().parse_vevents("""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:holiday
DTSTART;VALUE=DATE:20260605
DTEND;VALUE=DATE:20260607
END:VEVENT
END:VCALENDAR""")[0]
    assert event.start_at == "2026-06-05"
    assert event.end_at == "2026-06-07"
    assert event.all_day is True


def test_parse_preserves_master_recurrence_and_detached_identity() -> None:
    events = CalDavProvider().parse_vevents("""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:series
DTSTART:20260605T090000Z
RRULE:FREQ=DAILY;COUNT=3
RDATE:20260610T090000Z
EXDATE:20260606T090000Z
END:VEVENT
BEGIN:VEVENT
UID:series
RECURRENCE-ID:20260607T090000Z
DTSTART:20260607T110000Z
SUMMARY:Moved
END:VEVENT
END:VCALENDAR""")
    assert events[0].rrule == "FREQ=DAILY;COUNT=3"
    assert events[0].rdate == ["2026-06-10T09:00:00+00:00"]
    assert events[0].exdate == ["2026-06-06T09:00:00+00:00"]
    assert events[1].recurrence_id == "2026-06-07T09:00:00+00:00"


def test_parse_preserves_floating_and_explicit_recurrence_date_semantics() -> None:
    """Floating dates remain wall times; UTC and explicit TZID values retain their offsets."""
    event = CalDavProvider().parse_vevents(
        "BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:floating-dates\n"
        "DTSTART;TZID=America/Los_Angeles:20260306T090000\nRRULE:FREQ=DAILY;COUNT=3\n"
        "RDATE:20260309T090000\nRDATE:20260310T160000Z\n"
        "RDATE;TZID=Europe/Paris:20260311T090000\nEXDATE:20260308T090000\n"
        "EXDATE:20260312T160000Z\nEXDATE;TZID=Europe/Paris:20260313T090000\n"
        "END:VEVENT\nEND:VCALENDAR"
    )[0]
    assert event.rdate == [
        "2026-03-09T09:00:00", "2026-03-10T16:00:00+00:00", "2026-03-11T09:00:00+01:00",
    ]
    assert event.exdate == [
        "2026-03-08T09:00:00", "2026-03-12T16:00:00+00:00", "2026-03-13T09:00:00+01:00",
    ]


@pytest.mark.parametrize("rule", [
    "FREQ=YEARLY;BYMONTHDAY=31", "FREQ=YEARLY;BYMONTHDAY=-31",
    "FREQ=YEARLY;BYMONTH=2,3;BYMONTHDAY=31", "FREQ=YEARLY;BYMONTH=2,3;BYMONTHDAY=-31",
    "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=28,30", "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=-28,-30",
    "FREQ=YEARLY;BYMONTH=2,4;BYMONTHDAY=29", "FREQ=YEARLY;BYMONTH=2,4;BYMONTHDAY=-29",
])
def test_provider_accepts_productive_timezone_month_day_sets(rule: str) -> None:
    """Invalid dates in some months do not suppress valid annual transitions in others."""
    from datetime import datetime

    from dateutil import rrule

    from tldw_Server_API.app.core.Calendar.recurrence import _validate_timezone_rule

    finite_rule = f"{rule};UNTIL=20271231T235959Z"
    estimate = _validate_timezone_rule({"DTSTART": "20260101T000000", "RRULE": finite_rule})
    actual = list(rrule.rrulestr(finite_rule, dtstart=datetime(2026, 1, 1), ignoretz=True))
    assert estimate >= len(actual) > 0
    event = CalDavProvider().parse_vevents(
        "BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/MonthDays\nBEGIN:STANDARD\n"
        "DTSTART:20260101T000000\nTZOFFSETFROM:+0100\nTZOFFSETTO:+0100\n"
        f"RRULE:{finite_rule}\nEND:STANDARD\nEND:VTIMEZONE\nBEGIN:VEVENT\nUID:month-days\n"
        "DTSTART;TZID=Custom/MonthDays:20260605T090000\nEND:VEVENT\nEND:VCALENDAR"
    )[0]
    assert event.start_at == "2026-06-05T09:00:00+01:00"


@pytest.mark.parametrize("start_property", ["DTSTART", "RECURRENCE-ID"])
def test_embedded_timezone_rules_are_scoped_to_each_payload(start_property: str) -> None:
    """A library TZID cache must not substitute another account's custom definition."""
    events = []
    for offset in ("+0100", "+0200"):
        events.extend(CalDavProvider().parse_vevents(
            "BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/PayloadScoped\nBEGIN:STANDARD\n"
            f"DTSTART:19700101T000000\nTZOFFSETFROM:{offset}\nTZOFFSETTO:{offset}\n"
            "END:STANDARD\nEND:VTIMEZONE\nBEGIN:VEVENT\nUID:scoped\n"
            f"{start_property};TZID=Custom/PayloadScoped:20260605T090000\nDURATION:PT1H\n"
            "RDATE;TZID=Custom/PayloadScoped:20260606T090000\nEND:VEVENT\nEND:VCALENDAR"
        ))
    expected_starts = (
        ["2026-06-05T09:00:00+01:00", "2026-06-05T09:00:00+02:00"] if start_property == "DTSTART"
        else ["2026-06-05T08:00:00+00:00", "2026-06-05T07:00:00+00:00"]
    )
    assert [event.start_at for event in events] == expected_starts
    assert [event.end_at for event in events] == ["2026-06-05T10:00:00+01:00", "2026-06-05T10:00:00+02:00"]
    assert [event.rdate for event in events] == [["2026-06-06T09:00:00+01:00"], ["2026-06-06T09:00:00+02:00"]]


def test_provider_rejects_oversized_ics_before_parsing() -> None:
    with pytest.raises(CalendarValidationError, match="byte limit"):
        CalDavProvider(max_ics_bytes=32).parse_vevents("x" * 33)


@pytest.mark.parametrize("payload", [
    "END:VEVENT",
    "BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/MissingOffset\nBEGIN:STANDARD\n"
    "DTSTART:19700101T000000\nTZOFFSETFROM:+0100\nEND:STANDARD\nEND:VTIMEZONE\nEND:VCALENDAR",
])
def test_invalid_timezone_structure_is_a_domain_validation_error(payload: str) -> None:
    """Malformed structures remain controlled import failures rather than parser exceptions."""
    with pytest.raises(CalendarValidationError):
        CalDavProvider().parse_vevents(payload)


@pytest.mark.parametrize("rule", [
    "FREQ=SECONDLY", "FREQ=YEARLY;BYSECOND=0,1,2", "FREQ=YEARLY;INTERVAL=0", "FREQ=YEARLY;COUNT=0",
    "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=30", "FREQ=YEARLY;INTERVAL=4;BYMONTH=2;BYMONTHDAY=29",
    "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=-30", "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=30,31",
    "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=0,28", "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=28,32",
    "FREQ=YEARLY\nEXRULE:FREQ=SECONDLY", "FREQ=YEARLY\nEXDATE:19700101T000000",
])
def test_provider_rejects_unsafe_timezone_rules_before_parsing(
    monkeypatch: pytest.MonkeyPatch, rule: str,
) -> None:
    """Custom timezone offset lookups must never enter unbounded or invalid rule scans."""
    def must_not_parse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Unsafe VTIMEZONE reached the calendar parser")

    monkeypatch.setattr(caldav_module.ICalendar, "from_ical", must_not_parse)
    with pytest.raises(CalendarValidationError, match="VTIMEZONE"):
        CalDavProvider().parse_vevents(
            "BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/Unsafe\nBEGIN:STANDARD\n"
            "DTSTART:19700101T000000\nTZOFFSETFROM:+0100\nTZOFFSETTO:+0200\n"
            f"RRULE:{rule}\nEND:STANDARD\nEND:VTIMEZONE\nEND:VCALENDAR"
        )


@pytest.mark.parametrize("start", ["19700101T000000", "20260101T020000"])
def test_timezone_productivity_guard_does_not_scan_to_year_9999(
    monkeypatch: pytest.MonkeyPatch, start: str,
) -> None:
    """Instrument actual dateutil years: UNTIL must not masquerade as a non-yielding work bound."""
    from dateutil import rrule

    years: list[int] = []
    original = rrule._iterinfo.rebuild

    def counted_rebuild(self: Any, year: int, month: int) -> None:
        years.append(year)
        original(self, year, month)

    monkeypatch.setattr(rrule._iterinfo, "rebuild", counted_rebuild)
    with pytest.raises(CalendarValidationError):
        caldav_module.validate_provider_timezones(
            f"BEGIN:VTIMEZONE\nTZID:Custom/Never\nBEGIN:STANDARD\nDTSTART:{start}\n"
            "TZOFFSETFROM:+0100\nTZOFFSETTO:+0100\nRRULE:FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=30\n"
            "END:STANDARD\nEND:VTIMEZONE"
        )
    assert years == [], f"{len(years)} year scans ending at {years[-1]}"


@pytest.mark.parametrize(("rule", "copies"), [
    ("FREQ=YEARLY;BYDAY=MO,TU,WE,TH,FR,SA,SU", 1),
    ("FREQ=YEARLY;BYMONTH=3;BYDAY=2SU", 32),
    ("FREQ=YEARLY;BYMONTHDAY=1", 1),
    ("FREQ=YEARLY;BYMONTHDAY=31", 1),
])
def test_provider_timezone_transition_history_budget_rejects_before_parsing(
    monkeypatch: pytest.MonkeyPatch, rule: str, copies: int,
) -> None:
    """Small definitions cannot trigger excessive dense or cumulative historical transitions."""
    def must_not_parse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Excessive timezone history reached the calendar parser")

    monkeypatch.setattr(caldav_module.ICalendar, "from_ical", must_not_parse)
    observance = (
        "BEGIN:STANDARD\nDTSTART:16010101T000000\nTZOFFSETFROM:+0100\nTZOFFSETTO:+0100\n"
        f"RRULE:{rule}\nEND:STANDARD\n"
    )
    with pytest.raises(CalendarValidationError, match="transition budget"):
        CalDavProvider().parse_vevents(
            f"BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/Dense\n{observance * copies}"
            "END:VTIMEZONE\nEND:VCALENDAR"
        )


@pytest.mark.parametrize("lifespan", ["COUNT=2", "UNTIL=16020101T000000Z"])
def test_provider_timezone_transition_budget_respects_bounded_rule_lifespan(lifespan: str) -> None:
    """Explicit finite lifespan bounds allow old definitions without changing their semantics."""
    caldav_module.validate_provider_timezones(
        "BEGIN:VTIMEZONE\nTZID:Custom/Finite\nBEGIN:STANDARD\nDTSTART:16010101T000000\n"
        "TZOFFSETFROM:+0100\nTZOFFSETTO:+0100\n"
        f"RRULE:FREQ=YEARLY;BYDAY=MO,TU,WE,TH,FR,SA,SU;{lifespan}\nEND:STANDARD\nEND:VTIMEZONE"
    )


def test_timezone_rejects_mixed_ordinal_weekdays_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intersecting ordinal/plain weekdays may never yield, regardless of a small COUNT."""
    def must_not_parse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Nonproductive mixed weekdays reached the calendar parser")

    monkeypatch.setattr(caldav_module.ICalendar, "from_ical", must_not_parse)
    with pytest.raises(CalendarValidationError, match="VTIMEZONE"):
        CalDavProvider().parse_vevents(
            "BEGIN:VCALENDAR\nBEGIN:VTIMEZONE\nTZID:Custom/Mixed\nBEGIN:STANDARD\n"
            "DTSTART:19700101T000000\nTZOFFSETFROM:+0100\nTZOFFSETTO:+0100\n"
            "RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=1MO,TU;COUNT=1\n"
            "END:STANDARD\nEND:VTIMEZONE\nEND:VCALENDAR"
        )


def test_timezone_implicit_month_density_bounds_actual_finite_candidates() -> None:
    """A month-day filter without BYMONTH visits every month, not just DTSTART's month."""
    from datetime import datetime

    from dateutil import rrule

    from tldw_Server_API.app.core.Calendar.recurrence import _validate_timezone_rule

    rule = "FREQ=YEARLY;BYMONTHDAY=1;UNTIL=20261231T235959Z"
    estimate = _validate_timezone_rule({"DTSTART": "20260101T000000", "RRULE": rule})
    actual = list(rrule.rrulestr(rule, dtstart=datetime(2026, 1, 1), ignoretz=True))
    assert estimate >= len(actual) == 12


@pytest.mark.parametrize(("copies", "rejected"), [(10, False), (11, True)])
def test_timezone_explicit_dates_consume_cumulative_transition_budget(copies: int, rejected: bool) -> None:
    """RDATE parsing and initial starts share the same budget as generated transitions."""
    dates = ",".join(["20000101T000000"] * 1999)
    payload = "BEGIN:VCALENDAR\n" + "".join(
        f"BEGIN:VTIMEZONE\nTZID:Custom/Explicit{index}\nBEGIN:STANDARD\nDTSTART:19700101T000000\n"
        f"TZOFFSETFROM:+0100\nTZOFFSETTO:+0100\nRDATE:{dates}\nEND:STANDARD\nEND:VTIMEZONE\n"
        for index in range(copies)
    ) + "END:VCALENDAR"
    if rejected:
        with pytest.raises(CalendarValidationError, match="transition budget"):
            caldav_module.validate_provider_timezones(payload)
    else:
        caldav_module.validate_provider_timezones(payload)


@pytest.mark.parametrize(("zone_count", "observance_count", "rejected"), [(2, 16, False), (3, 11, True)])
def test_provider_timezone_observance_limit_is_cumulative(
    monkeypatch: pytest.MonkeyPatch, zone_count: int, observance_count: int, rejected: bool,
) -> None:
    """Several individually small timezone definitions cannot bypass the total observance cap."""
    def must_not_parse(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Oversized timezone set reached the calendar parser")

    if rejected:
        monkeypatch.setattr(caldav_module.ICalendar, "from_ical", must_not_parse)
    observance = (
        "BEGIN:STANDARD\nDTSTART:19700101T000000\nTZOFFSETFROM:+0100\nTZOFFSETTO:+0100\nEND:STANDARD\n"
    )
    payload = "BEGIN:VCALENDAR\n" + "".join(
        f"BEGIN:VTIMEZONE\nTZID:Custom/Many{index}\n{observance * observance_count}END:VTIMEZONE\n"
        for index in range(zone_count)
    ) + "END:VCALENDAR"
    if rejected:
        with pytest.raises(CalendarValidationError, match="VTIMEZONE"):
            CalDavProvider().parse_vevents(payload)
    else:
        assert CalDavProvider().parse_vevents(payload) == []


def test_provider_rejects_oversized_buffered_response() -> None:
    client = _FakeHttpClient([_FakeResponse(text="x" * 33)])
    with pytest.raises(CalendarValidationError, match="byte limit"):
        CalDavProvider(http_client=client, max_response_bytes=32)._request(
            "REPORT", "https://calendar.example.test/", username="user", password="secret"
        )


@pytest.mark.parametrize("content_length", [None, "1000"])
def test_default_transport_bounds_stream_before_buffering(
    monkeypatch: pytest.MonkeyPatch, content_length: str | None
) -> None:
    consumed: list[int] = []
    closed: list[bool] = []

    class Response:
        headers = {"Content-Length": content_length} if content_length else {}
        status_code = 207

        def iter_bytes(self, chunk_size: int) -> Any:
            for index in range(10):
                consumed.append(index)
                yield b"x" * 16

    class Client:
        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        @contextlib.contextmanager
        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            try:
                yield Response()
            finally:
                closed.append(True)

    monkeypatch.setattr(caldav_module, "create_client", lambda **kwargs: Client())
    monkeypatch.setattr(
        caldav_module,
        "evaluate_url_policy",
        lambda *args, **kwargs: SimpleNamespace(allowed=True, resolved_ips=("93.184.216.34",)),
    )
    with pytest.raises(CalendarValidationError, match="byte limit"):
        CalDavProvider(max_response_bytes=32)._request(
            "REPORT", "https://calendar.example.test/", username="user", password="secret"
        )
    assert len(consumed) == (0 if content_length else 3)
    assert closed == [True]


def test_verification_logs_safe_diagnostics_and_does_not_expose_credentials() -> None:
    class Client:
        def request(self, *_args: Any, **_kwargs: Any) -> Any:
            raise httpx.ConnectError("password=private-secret https://host/?token=private-token")

    messages: list[str] = []
    sink = logger.add(messages.append, format="{message} {extra}")
    try:
        result = CalDavProvider(http_client=Client()).verify_account(
            server_url="https://calendar.example.test/", username="user", password="private-secret"
        )
    finally:
        logger.remove(sink)
    assert result.error == "CalDAV verification failed (ConnectError)"
    assert "verify_account" in "".join(messages)
    assert "private-secret" not in repr(result) + "".join(messages)
    assert "private-token" not in repr(result) + "".join(messages)


def test_verification_does_not_swallow_programming_errors() -> None:
    class Client:
        def request(self, *_args: Any, **_kwargs: Any) -> Any:
            raise TypeError("programming error")

    with pytest.raises(TypeError, match="programming error"):
        CalDavProvider(http_client=Client()).verify_account(
            server_url="https://calendar.example.test/", username="user", password="secret"
        )


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

        @contextlib.contextmanager
        def stream(self, method: str, url: str, **kwargs: Any) -> Any:
            calls.append({"method": method, "url": url, **kwargs})
            yield httpx.Response(200, content=b"")

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
