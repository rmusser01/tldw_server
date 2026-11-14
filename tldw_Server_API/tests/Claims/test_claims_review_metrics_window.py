"""Behavioral coverage of atomic, explicit Claims review windows."""

import json
from datetime import date, datetime, timedelta

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit
START = date(2026, 9, 6)
END = date(2026, 9, 7)


@pytest.fixture
def review_db(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "owner-42.db"), client_id="42")
    try:
        yield db
    finally:
        db.close_connection()


def _seed(db, *, extractor="heuristic", events=None):
    """Seed real Claims reviews, then give their logs fixed UTC timestamps."""
    events = events or [("2026-09-06 00:00:00", "approved", "spam", "Edited")]
    media_id, _, _ = db.add_media_with_keywords(
        title=extractor,
        media_type="text",
        content=f"Content {extractor}",
        keywords=None,
    )
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "claim_text": "Original",
                "extractor": extractor,
                "extractor_version": "v1",
                "chunk_hash": extractor,
            }
        ]
    )
    claim = db.get_claims_by_media(media_id)[0]
    for created_at, status, reason, text in events:
        db.update_claim_review(
            int(claim["id"]),
            review_status=status.lower(),
            reviewer_id=42,
            review_reason_code=reason,
            corrected_text=text,
        )
        log = db.execute_query(
            "SELECT MAX(id) AS id FROM claims_review_log WHERE claim_id = ?",
            (claim["id"],),
        ).fetchone()
        db.execute_query(
            "UPDATE claims_review_log SET created_at = ?, new_status = ?, "
            "old_text = 'Original', new_text = ? WHERE id = ?",
            (created_at, status, text, log["id"]),
            commit=True,
        )


def _aggregate(db, *, owner="42", start=START, end=END):
    from tldw_Server_API.app.core.Claims_Extraction.claims_review_metrics import (
        aggregate_claims_review_metrics_window,
    )

    return aggregate_claims_review_metrics_window(
        db=db,
        owner_user_id=owner,
        start_date=start,
        end_date=end,
    )


def test_explicit_window_counts_statuses_edits_and_stable_reasons(review_db):
    _seed(
        review_db,
        events=[
            ("2026-09-05 23:59:59", "approved", "outside", "Edited"),
            ("2026-09-06 00:00:00", "APPROVED", " zeta ", "Edited"),
            ("2026-09-06 01:00:00", "rejected", "alpha", "Original"),
            ("2026-09-06 02:00:00", "flagged", "zeta", None),
            ("2026-09-06 03:00:00", "reassigned", "  ", "Edited"),
            ("2026-09-07 23:59:59", "approved", None, "Original"),
            ("2026-09-08 00:00:00", "approved", "outside", "Edited"),
        ],
    )
    assert _aggregate(review_db) == 2
    rows = review_db.list_claims_review_extractor_metrics_daily(user_id="42")
    first = next(row for row in rows if str(row["report_date"]) == START.isoformat())
    assert {
        key: first[key]
        for key in (
            "total_reviewed",
            "approved_count",
            "rejected_count",
            "flagged_count",
            "reassigned_count",
            "edited_count",
        )
    } == {
        "total_reviewed": 4,
        "approved_count": 1,
        "rejected_count": 1,
        "flagged_count": 1,
        "reassigned_count": 1,
        "edited_count": 2,
    }
    assert first["reason_code_counts_json"] == '{"alpha":1,"zeta":2}'
    assert rows[0]["reason_code_counts_json"] is None


def test_owner_sqlite_database_is_the_source_boundary(review_db, tmp_path):
    _seed(review_db)
    other = MediaDatabase(db_path=str(tmp_path / "owner-43.db"), client_id="43")
    try:
        _seed(other, extractor="other")
        assert _aggregate(review_db) == 1
        assert review_db.list_claims_review_extractor_metrics_daily(user_id="43") == []
        assert other.list_claims_review_extractor_metrics_daily(user_id="42") == []
    finally:
        other.close_connection()


@pytest.mark.parametrize("use_legacy_adapter", [False, True])
def test_second_group_failure_rolls_back_the_entire_window(review_db, monkeypatch, use_legacy_adapter):
    _seed(review_db, extractor="alpha")
    _seed(review_db, extractor="beta")
    upsert = review_db.upsert_claims_review_extractor_metrics_daily
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second group failure")
        return upsert(**kwargs)

    monkeypatch.setattr(review_db, "upsert_claims_review_extractor_metrics_daily", fail_second)
    with pytest.raises(RuntimeError, match="injected second group failure"):
        if use_legacy_adapter:
            claims_service.aggregate_claims_review_extractor_metrics_daily(
                db=review_db,
                target_user_id="42",
                report_date=START.isoformat(),
            )
        else:
            _aggregate(review_db)
    assert review_db.list_claims_review_extractor_metrics_daily(user_id="42") == []


def test_rerun_updates_observed_groups_without_deleting_historical_groups(review_db):
    _seed(review_db)
    review_db.upsert_claims_review_extractor_metrics_daily(
        user_id="42",
        report_date=START.isoformat(),
        extractor="historic",
        total_reviewed=9,
    )
    assert _aggregate(review_db) == 1
    assert _aggregate(review_db) == 1
    rows = review_db.list_claims_review_extractor_metrics_daily(user_id="42")
    assert {row["extractor"]: row["total_reviewed"] for row in rows} == {
        "historic": 9,
        "heuristic": 1,
    }


def test_empty_window_writes_nothing(review_db):
    assert _aggregate(review_db) == 0
    assert review_db.list_claims_review_extractor_metrics_daily(user_id="42") == []


@pytest.mark.parametrize("owner", [None, True, 42, "0", "01", " 42", "-1", "9223372036854775808"])
def test_rejects_noncanonical_owner_before_database_access(review_db, owner):
    with pytest.raises(ValueError, match="owner"):
        _aggregate(review_db, owner=owner)


@pytest.mark.parametrize(
    "start,end",
    [
        (END, START),
        (START, START + timedelta(days=366)),
        (START.isoformat(), END),
        (START, datetime(2026, 9, 7)),
        (date.max, date.max),
    ],
)
def test_rejects_invalid_or_unbounded_window(review_db, start, end):
    with pytest.raises(ValueError, match="date|window"):
        _aggregate(review_db, start=start, end=end)


def test_accepts_inclusive_366_day_window(review_db):
    assert _aggregate(review_db, end=START + timedelta(days=365)) == 0


def test_legacy_adapter_resolves_report_date_and_delegates(review_db, monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import claims_review_metrics

    captured = {}

    def aggregate(**kwargs):
        captured.update(kwargs)
        return 7

    monkeypatch.setattr(claims_review_metrics, "aggregate_claims_review_metrics_window", aggregate)
    assert (
        claims_service.aggregate_claims_review_extractor_metrics_daily(
            db=review_db,
            target_user_id="42",
            report_date=START.isoformat(),
            lookback_days=30,
        )
        == 7
    )
    assert captured == {"db": review_db, "owner_user_id": "42", "start_date": START, "end_date": START}


def test_legacy_adapter_bounds_lookback(review_db, monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction import claims_review_metrics

    captured = {}

    def aggregate(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(claims_review_metrics, "aggregate_claims_review_metrics_window", aggregate)
    claims_service.aggregate_claims_review_extractor_metrics_daily(
        db=review_db,
        target_user_id="42",
        lookback_days=999999,
    )
    assert (captured["end_date"] - captured["start_date"]).days == 365


def test_window_fetches_counts_and_reasons_in_one_source_statement(review_db, monkeypatch):
    _seed(review_db)
    execute = review_db.execute_query
    source_queries = []

    def record(query, params=None, **kwargs):
        if "FROM claims_review_log" in query:
            source_queries.append(query)
        return execute(query, params, **kwargs)

    monkeypatch.setattr(review_db, "execute_query", record)
    assert _aggregate(review_db) == 1
    assert len(source_queries) == 1
    row = review_db.list_claims_review_extractor_metrics_daily(user_id="42")[0]
    assert json.loads(row["reason_code_counts_json"]) == {"spam": 1}
