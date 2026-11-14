"""Event-page contracts retained by the concurrent SQLite replay optimization."""

import json
import math
import sqlite3
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase


def _create_run(db, run_id="run-1"):
    db.create_run(
        run_id=run_id,
        tenant_id="tenant",
        user_id="owner",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "event-pages", "steps": []},
    )


@pytest.fixture
def event_db(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "events.db"))
    _create_run(db)
    try:
        yield db
    finally:
        db.close()


@pytest.mark.parametrize(
    "stored, expected",
    [
        (
            '{"text":"hello ☃","zero":"\\u0000","big":1267650600228229401496703205376}',
            {"text": "hello ☃", "zero": "\x00", "big": 1267650600228229401496703205376},
        ),
        ('["1",null,false,1.5]', ["1", None, False, 1.5]),
        ('"1"', "1"),
        ("0", 0),
        ("false", False),
        ("null", None),
        ("", {}),
        (None, {}),
        ("  ", "  "),
        ('{"broken"', '{"broken"'),
        (b'{"binary":true}', {"binary": True}),
    ],
)
def test_event_page_preserves_stored_payload_types(event_db, stored, expected):
    event_db.append_event("tenant", "run-1", "created", {"seed": True})
    event_db._conn.execute("UPDATE workflow_events SET payload_json = ?", (stored,))
    event_db._conn.commit()

    event = event_db.get_events("run-1")[0]

    assert event["payload_json"] == expected
    assert set(event) == {
        "event_id",
        "tenant_id",
        "run_id",
        "step_run_id",
        "event_seq",
        "event_type",
        "payload_json",
        "created_at",
    }
    assert (event["tenant_id"], event["run_id"], event["step_run_id"], event["event_seq"]) == (
        "tenant",
        "run-1",
        None,
        1,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_event_page_preserves_python_json_nonfinite_values(event_db, value):
    event_db.append_event("tenant", "run-1", "nonfinite", {"value": value})
    actual = event_db.get_events("run-1")[0]["payload_json"]["value"]
    assert math.isnan(actual) if math.isnan(value) else actual == value


def test_event_page_legacy_fallback_does_not_decode_strings_twice(event_db):
    event_db.append_event("tenant", "run-1", "created", {"seed": True})
    event_db.append_event("tenant", "run-1", "created", {"seed": True})
    event_db._conn.execute("UPDATE workflow_events SET payload_json = ? WHERE event_seq = 1", ('"1"',))
    event_db._conn.execute("UPDATE workflow_events SET payload_json = ? WHERE event_seq = 2", ("broken",))
    event_db._conn.commit()
    assert [event["payload_json"] for event in event_db.get_events("run-1")] == ["1", "broken"]


@pytest.mark.parametrize("decoder_limit", ["integer", "nesting"])
def test_event_page_preserves_per_payload_decoding_at_python_limits(event_db, decoder_limit):
    event_db.append_event("tenant", "run-1", "history", {"seed": True})
    old_digits = sys.get_int_max_str_digits()
    old_recursion = sys.getrecursionlimit()
    try:
        if decoder_limit == "integer":
            sys.set_int_max_str_digits(640)
            stored = '{"value":' + "9" * 641 + "}"
        else:
            sys.setrecursionlimit(400)
            stored = "[" * 600 + "0" + "]" * 600
        try:
            expected = json.loads(stored)
        except (ValueError, RecursionError):
            expected = stored
        event_db._conn.execute("UPDATE workflow_events SET payload_json = ?", (stored,))
        event_db._conn.commit()
        actual = event_db.get_events("run-1")[0]["payload_json"]
    finally:
        sys.set_int_max_str_digits(old_digits)
        sys.setrecursionlimit(old_recursion)
    assert actual == expected


def test_event_page_filters_before_limiting_and_keeps_run_isolation(event_db):
    run_id = "run' OR 1=1 --"
    _create_run(event_db, run_id)
    for kind in ["skip", "match", "skip", "match", "match"]:
        event_db.append_event("tenant", run_id, kind, {"kind": kind})
    event_db.append_event("tenant", "run-1", "match", {"foreign": True})

    events = event_db.get_events(run_id, since=2, types=["match"], limit=1)

    assert [(event["run_id"], event["event_seq"], event["event_type"]) for event in events] == [(run_id, 4, "match")]
    assert event_db.get_events("missing") == []
    assert event_db.get_events(run_id, types=["match') OR 1=1 --"]) == []
    assert event_db.get_events(run_id, limit=0) == []


def test_event_pages_preserve_order_across_default_page_boundary(event_db):
    for index in range(503):
        event_db.append_event("tenant", "run-1", "history", {"index": index})
    first = event_db.get_events("run-1")
    second = event_db.get_events("run-1", since=first[-1]["event_seq"])
    assert len(first) == 500
    assert [event["event_seq"] for event in first + second] == list(range(1, 504))
    assert [event["payload_json"]["index"] for event in first + second] == list(range(503))


def test_event_page_keeps_event_id_tie_breaker_for_legacy_rows(event_db):
    for index in range(3):
        event_db.append_event("tenant", "run-1", "history", {"index": index})
    event_db._conn.execute("DROP INDEX ux_events_run_seq")
    event_db._conn.execute("UPDATE workflow_events SET event_seq = 1")
    event_db._conn.commit()
    events = event_db.get_events("run-1")
    assert [event["payload_json"]["index"] for event in events] == [0, 1, 2]
    assert [event["event_id"] for event in events] == sorted(event["event_id"] for event in events)


def test_event_page_handles_results_larger_than_sqlite_single_value_limit(event_db):
    payload = {"body": "x" * 256}
    for _ in range(20):
        event_db.append_event("tenant", "run-1", "history", payload)
    previous = event_db._conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2000)
    try:
        events = event_db.get_events("run-1")
    finally:
        event_db._conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, previous)
    assert [event["event_seq"] for event in events] == list(range(1, 21))
    assert all(event["payload_json"] == payload for event in events)


def test_event_page_keeps_working_without_optional_sqlite_json_functions(monkeypatch, tmp_path):
    original_connect = sqlite3.connect

    class LegacyCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if "json_group_array" in sql.lower():
                raise sqlite3.OperationalError("no such function: json_group_array")
            return super().execute(sql, parameters)

    class LegacyConnection(sqlite3.Connection):
        def cursor(self, factory=LegacyCursor):
            return super().cursor(factory)

    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=LegacyConnection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    db = WorkflowsDatabase(str(tmp_path / "legacy.db"))
    try:
        _create_run(db)
        db.append_event("tenant", "run-1", "history", {"value": 42})
        assert db.get_events("run-1")[0]["payload_json"] == {"value": 42}
    finally:
        db.close()


def test_event_page_does_not_swallow_sqlite_read_authorization_errors(event_db):
    event_db.append_event("tenant", "run-1", "history", {"value": 42})

    def authorize(action, table, column, database, trigger):
        return (
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == "workflow_events" else sqlite3.SQLITE_OK
        )

    event_db._conn.set_authorizer(authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            event_db.get_events("run-1")
    finally:
        event_db._conn.set_authorizer(None)
    assert event_db.get_events("run-1")[0]["payload_json"] == {"value": 42}


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**130), max_value=2**130)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda values: st.lists(values, max_size=4) | st.dictionaries(st.text(), values, max_size=4),
    max_leaves=12,
)


@given(json_values)
@settings(max_examples=25, deadline=None)
def test_event_page_round_trips_nested_json_values(value):
    db = WorkflowsDatabase(":memory:")
    try:
        _create_run(db)
        db.append_event("tenant", "run-1", "generated", {"value": value})
        assert db.get_events("run-1")[0]["payload_json"] == {"value": value}
    finally:
        db.close()
