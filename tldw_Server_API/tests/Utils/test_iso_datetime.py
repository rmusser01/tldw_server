from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Utils.iso_datetime import parse_iso_utc, utc_now_iso

_EXPECTED = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "2024-01-02 03:04:05",  # SQLite CURRENT_TIMESTAMP
        "2024-01-02T03:04:05",
        "2024-01-02T03:04:05Z",
        "2024-01-02T03:04:05+00:00",
        "2024-01-02T05:04:05+02:00",
        " 2024-01-02T03:04:05Z ",
        datetime(2024, 1, 2, 3, 4, 5),
        datetime(2024, 1, 2, 5, 4, 5, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_awareness_does_not_depend_on_input_format(value):
    parsed = parse_iso_utc(value)
    assert parsed == _EXPECTED
    assert parsed.utcoffset() == timedelta(0)


@pytest.mark.unit
def test_fractional_seconds_are_kept():
    assert parse_iso_utc("2024-01-02 03:04:05.250000").microsecond == 250000


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "", "   ", "garbage", "2024-13-40", 1700000000, object()])
def test_unparseable_input_is_none_never_now(value):
    assert parse_iso_utc(value) is None


@pytest.mark.unit
def test_utc_now_iso_is_aware_and_round_trips():
    parsed = parse_iso_utc(utc_now_iso())
    assert utc_now_iso().endswith("+00:00")
    assert abs(datetime.now(timezone.utc) - parsed) < timedelta(seconds=5)


@pytest.mark.unit
def test_chat_dictionary_entry_timestamps_are_aware_utc_for_sqlite_format():
    """The old canonical returned a NAIVE datetime for SQLite's "YYYY-MM-DD HH:MM:SS"
    but an aware one for ISO-with-offset input, so one response mixed the two."""
    from tldw_Server_API.app.api.v1.endpoints.chat_dictionaries import _entry_dict_to_response

    response = _entry_dict_to_response(
        {
            "id": 1,
            "dictionary_id": 1,
            "pattern": "a",
            "replacement": "b",
            "created_at": "2024-01-02 03:04:05",
            "updated_at": "2024-01-02T05:04:05+02:00",
            "last_used_at": "2024-01-02 03:04:05",
        }
    )
    assert response.created_at == _EXPECTED
    assert response.created_at.utcoffset() == timedelta(0)
    assert response.updated_at == _EXPECTED
    assert response.last_used_at == _EXPECTED
