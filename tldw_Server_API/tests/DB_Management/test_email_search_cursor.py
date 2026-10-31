"""Real SQLite cursor traversal and scope regression tests."""

import base64
import json
from datetime import datetime, timezone
from email.utils import format_datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit


@pytest.fixture()
def cursor_db(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "cursor.db"), client_id="cursor-test")
    try:
        yield db
    finally:
        db.close_connection()


def add_message(db, key, date=None, *, tenant=None, subject="Budget"):
    """Persist a unique synthetic email via the production graph writer."""
    if date is not None:
        date = format_datetime(datetime.fromisoformat(date))
    media_id, _, _ = db.add_media_with_keywords(
        url=f"email://cursor/{key}",
        title=subject,
        media_type="email",
        content=f"Synthetic body {key}",
        keywords=["email"],
    )
    result = db.upsert_email_message_graph(
        media_id=media_id,
        tenant_id=tenant,
        source_key="cursor-fixture",
        source_message_id=key,
        body_text=f"Synthetic body {key}",
        metadata={"email": {"subject": subject, "date": date, "message_id": f"<{key}@example.test>"}},
    )
    return int(result["email_message_id"]), media_id


def test_cursor_traverses_ties_nulls_and_newer_inserts(cursor_db):
    db = cursor_db
    for key, date in [
        ("null-a", None),
        ("null-b", None),
        ("old", "2025-01-01T00:00:00+00:00"),
        ("tie-a", "2025-02-01T00:00:00+00:00"),
        ("tie-b", "2025-02-01T00:00:00+00:00"),
    ]:
        add_message(db, key, date)
    rows, total, token = db.search_email_messages(cursor="", limit=1)
    assert ([row["source_message_id"] for row in rows], total) == (["tie-b"], 5)
    add_message(db, "new", "2026-01-01T00:00:00+00:00")
    add_message(db, "new-tie", "2025-02-01T00:00:00+00:00")
    visited = []
    while token:
        rows, total, token = db.search_email_messages(cursor=token, limit=1)
        visited.extend(row["source_message_id"] for row in rows)
    assert visited == ["tie-a", "old", "null-b", "null-a"]
    assert total == 7


@pytest.mark.parametrize(
    "change", [{"query": "subject:Other"}, {"tenant_id": "other"}, {"include_deleted": True}, {"offset": 1}]
)
def test_cursor_rejects_changed_scope_and_offset(cursor_db, change):
    add_message(cursor_db, "a")
    add_message(cursor_db, "b")
    _, _, token = cursor_db.search_email_messages(cursor="", limit=1)
    with pytest.raises(InputError):
        cursor_db.search_email_messages(cursor=token, **change)


@pytest.mark.parametrize("token", ["garbage", "%%%", "e30", "bnVsbA", "A" * 4097])
def test_cursor_rejects_malformed_tokens(cursor_db, token):
    with pytest.raises(InputError):
        cursor_db.search_email_messages(cursor=token)


def test_cursor_filters_tenant_deleted_trash_and_query(cursor_db):
    db = cursor_db
    add_message(db, "other-tenant", tenant="other")
    add_message(db, "wrong-subject", subject="Other")
    _, deleted_id = add_message(db, "deleted")
    db.soft_delete_media(deleted_id, cascade=True)
    _, trash_id = add_message(db, "trash")
    with db.transaction() as conn:
        db._execute_with_connection(
            conn,
            "UPDATE Media SET is_trash = 1, version = version + 1 WHERE id = ?",
            (trash_id,),
        )
    add_message(db, "visible-a")
    add_message(db, "visible-b")
    rows, total, token = db.search_email_messages(query="subject:Budget", cursor="", limit=1)
    assert ([r["source_message_id"] for r in rows], total) == (["visible-b"], 2)
    rows, total, token = db.search_email_messages(query="subject:Budget", cursor=token, limit=1)
    assert ([r["source_message_id"] for r in rows], total, token) == (["visible-a"], 2, None)


def test_cursor_empty_results(cursor_db):
    assert cursor_db.search_email_messages(cursor="") == ([], 0, None)


def test_cursor_rejects_relative_date_underflow(cursor_db):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_search_cursor import (
        email_cursor_scope,
        encode_email_cursor,
    )

    query = "newer_than:1d"
    token = encode_email_cursor(
        email_cursor_scope(str(cursor_db.client_id), query, False),
        datetime(1, 1, 1, tzinfo=timezone.utc),
        {"internal_date": None, "email_message_id": 1},
    )
    with pytest.raises(InputError):
        cursor_db.search_email_messages(query=query, cursor=token)


def test_search_rejects_overflowing_relative_window(cursor_db):
    with pytest.raises(InputError):
        cursor_db.search_email_messages(query="newer_than:999999999999999999999d")


def test_offset_contract_stays_two_values(cursor_db):
    add_message(cursor_db, "a")
    add_message(cursor_db, "b")
    rows, total = cursor_db.search_email_messages(limit=1, offset=1)
    assert ([r["source_message_id"] for r in rows], total) == (["a"], 2)


@pytest.mark.parametrize(
    "index,value",
    [(0, 2), (0, True), (2, "bad-date"), (2, "2025-01-01"), (3, []), (3, "bad-date"), (4, True), (4, -1), (4, 2**63)],
)
def test_cursor_rejects_invalid_payload_fields(cursor_db, index, value):
    add_message(cursor_db, "a")
    add_message(cursor_db, "b")
    _, _, token = cursor_db.search_email_messages(cursor="", limit=1)
    payload = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    payload[index] = value
    invalid = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    with pytest.raises(InputError):
        cursor_db.search_email_messages(cursor=invalid)


def test_cursor_freezes_relative_query_window(cursor_db, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import email_query_ops

    class Clock(datetime):
        current = datetime(2025, 2, 2, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(email_query_ops, "datetime", Clock)
    add_message(cursor_db, "older", "2025-02-01T01:00:00+00:00")
    add_message(cursor_db, "newer", "2025-02-01T02:00:00+00:00")
    _, _, token = cursor_db.search_email_messages(query="newer_than:1d", cursor="", limit=1)
    Clock.current = datetime(2025, 2, 3, tzinfo=timezone.utc)
    rows, total, token = cursor_db.search_email_messages(query="newer_than:1d", cursor=token, limit=1)
    assert ([r["source_message_id"] for r in rows], total, token) == (["older"], 2, None)


@given(order=st.permutations([0, 1, 2, 3]), page_size=st.integers(min_value=1, max_value=4))
@settings(max_examples=12, deadline=None)
def test_cursor_all_rows_once_for_insertion_orders_and_page_sizes(order, page_size):
    db = MediaDatabase(db_path=":memory:", client_id="cursor-property")
    try:
        dates = [None, "2025-01-01T00:00:00+00:00", "2025-02-01T00:00:00+00:00", None]
        expected = []
        for index in order:
            message_id, _ = add_message(db, str(index), dates[index])
            expected.append((dates[index] or "", message_id))
        expected_ids = [message_id for _, message_id in sorted(expected, reverse=True)]
        cursor = ""
        actual = []
        while cursor is not None:
            rows, _, cursor = db.search_email_messages(cursor=cursor, limit=page_size)
            actual.extend(r["email_message_id"] for r in rows)
        assert actual == expected_ids
    finally:
        db.close_connection()
