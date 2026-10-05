"""Review metrics against the official restricted-role PostgreSQL fixture."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers, claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_review_metrics import (
    aggregate_claims_review_metrics_window,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
START = date(2026, 9, 6)
END = date(2026, 9, 7)


@pytest.fixture
def pg_review_db(pg_restricted_backend):
    """No custom provisioning: the official fixture owns DB, role, and pool."""
    with scoped_context(user_id=None, is_admin=True):
        db = MediaDatabase(db_path=":memory:", client_id="42", backend=pg_restricted_backend)
        try:
            yield db
        finally:
            db.close_connection()


def _add_event(db, claim_id, *, timestamp="2026-09-06T00:00:00Z", reason="spam"):
    db.execute_query(
        "INSERT INTO claims_review_log "
        "(claim_id, old_status, new_status, old_text, new_text, reason_code, created_at) "
        "VALUES (?, 'pending', 'approved', 'Original', 'Edited', ?, ?)",
        (claim_id, reason, timestamp),
        commit=True,
    )


def _seed(
    db, *, owner="42", extractor="heuristic", visibility="personal", deleted=False, timestamp="2026-09-06T00:00:00Z"
):
    with scoped_context(user_id=int(owner), is_admin=True, org_ids=[8], team_ids=[9]):
        media_id, _, _ = db.add_media_with_keywords(
            title=f"{owner}-{extractor}",
            media_type="text",
            content=f"Content {owner} {extractor}",
            keywords=None,
            owner_user_id=int(owner),
            visibility=visibility,
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
        claim = db.execute_query(
            "SELECT id FROM claims WHERE media_id = ?",
            (media_id,),
        ).fetchone()
        _add_event(db, claim["id"], timestamp=timestamp)
        if deleted:
            db.execute_query("UPDATE media SET deleted = 1 WHERE id = ?", (media_id,), commit=True)
    return int(claim["id"])


def _aggregate(db, owner="42"):
    with scoped_context(user_id=int(owner), org_ids=[], team_ids=[], is_admin=True):
        return aggregate_claims_review_metrics_window(
            db=db,
            owner_user_id=owner,
            start_date=START,
            end_date=END,
        )


@pytest.mark.asyncio
async def test_postgres_review_metrics_admission_to_worker_and_atomic_owner_metrics(
    pg_review_db, tmp_path, monkeypatch
):
    db = pg_review_db
    _seed(db, visibility="team", deleted=True)
    _seed(db, owner="43", extractor="other-owner", visibility="org")
    manager = JobManager(tmp_path / "pg-metrics-jobs.sqlite")
    original_scope = get_scope()
    monkeypatch.setattr(claims_job_handlers.media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(
        claims_job_handlers,
        "managed_media_database",
        lambda **kwargs: managed_media_database(backend=db.backend, **kwargs),
    )
    admission = claims_jobs.enqueue_claims_review_metrics(
        owner_user_id="42",
        scheduled_for="2026-09-07T00:00:00Z",
        start_date=START.isoformat(),
        end_date=END.isoformat(),
        interval_seconds=86400,
        job_manager=manager,
        settings_obj={"CLAIMS_JOBS_QUEUE": "default"},
    )
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="claims",
            queue="default",
            worker_id="postgres-metrics-e2e",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    results = []

    async def completed(job, result):
        results.append(result)
        sdk.stop()

    await asyncio.wait_for(sdk.run(handler=claims_job_handlers.process_claims_job, on_completed=completed), 30)
    assert manager.get_job(int(admission.row["id"]))["status"] == "completed"
    assert results[0]["groups_written"] == 1
    assert get_scope() is original_scope
    rows = db.list_claims_review_extractor_metrics_daily(user_id="42")
    assert len(rows) == 1 and rows[0]["total_reviewed"] == 1
    assert db.list_claims_review_extractor_metrics_daily(user_id="43") == []


def test_postgres_utc_grouping_matches_sqlite_even_in_non_utc_session(pg_review_db, tmp_path):
    db = pg_review_db
    claim = _seed(db, timestamp="2026-09-06T00:15:00Z")
    _add_event(db, claim, timestamp="2026-09-07T23:59:59Z", reason="typo")
    _add_event(db, claim, timestamp="2026-09-08T00:00:00Z", reason="outside")
    sqlite_db = MediaDatabase(db_path=str(tmp_path / "parity.db"), client_id="42")
    try:
        sqlite_claim = _seed(sqlite_db, timestamp="2026-09-06 00:15:00")
        _add_event(sqlite_db, sqlite_claim, timestamp="2026-09-07 23:59:59", reason="typo")
        _add_event(sqlite_db, sqlite_claim, timestamp="2026-09-08 00:00:00", reason="outside")
        with db.transaction():
            db.execute_query("SET LOCAL TIME ZONE 'America/Los_Angeles'")
            assert _aggregate(db) == 2
        assert _aggregate(sqlite_db) == 2
        keys = (
            "report_date",
            "extractor",
            "extractor_version",
            "total_reviewed",
            "approved_count",
            "edited_count",
            "reason_code_counts_json",
        )
        pg_rows = db.list_claims_review_extractor_metrics_daily(user_id="42")
        sqlite_rows = sqlite_db.list_claims_review_extractor_metrics_daily(user_id="42")
        assert [{key: str(row[key]) for key in keys} for row in pg_rows] == [
            {key: str(row[key]) for key in keys} for row in sqlite_rows
        ]
    finally:
        sqlite_db.close_connection()


def test_privileged_owner_pages_and_aggregate_include_team_org_deleted_media(pg_review_db):
    db = pg_review_db
    for visibility, deleted in [("personal", False), ("team", False), ("org", True)]:
        _seed(db, extractor=visibility, visibility=visibility, deleted=deleted)
    _seed(db, owner="43", extractor="other")
    _seed(db, owner="10", extractor="old", timestamp="2026-08-01T00:00:00Z")
    with scoped_context(user_id=999, is_admin=False):
        assert db.list_claims_review_user_ids_page(start_date=START, end_date=END) == []
        with scoped_context(user_id=None, org_ids=[], team_ids=[], is_admin=True):
            assert db.list_claims_review_user_ids_page(start_date=START, end_date=END, limit=1) == ["42"]
            assert db.list_claims_review_user_ids_page(
                start_date=START,
                end_date=END,
                after_user_id="42",
                limit=1,
            ) == ["43"]
        assert _aggregate(db) == 3
        assert get_scope().is_admin is False
        assert db.list_claims_review_user_ids_page(start_date=START, end_date=END) == []
    rows = db.list_claims_review_extractor_metrics_daily(user_id="42")
    assert {row["extractor"] for row in rows} == {"personal", "team", "org"}
    assert db.list_claims_review_extractor_metrics_daily(user_id="43") == []
    flags = db.execute_query(
        "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user",
    ).fetchone()
    assert not flags["rolsuper"] and not flags["rolbypassrls"]
    forced = db.execute_query("SELECT relforcerowsecurity FROM pg_class WHERE relname = 'media'").fetchone()
    assert forced["relforcerowsecurity"]


def test_owner_pages_use_text_order_and_client_id_fallback(pg_review_db):
    db = pg_review_db
    for owner in ("9", "10", "42"):
        _seed(db, owner=owner, extractor=owner)
    db.execute_query("UPDATE media SET owner_user_id = NULL, client_id = '42' WHERE owner_user_id = 42", commit=True)
    assert db.list_claims_review_user_ids_page(start_date=START, end_date=END, limit=2) == ["10", "42"]
    assert db.list_claims_review_user_ids_page(start_date=START, end_date=END, after_user_id="42") == ["9"]


def test_source_totals_and_reasons_keep_one_snapshot_when_new_event_commits(pg_review_db, monkeypatch):
    db = pg_review_db
    claim = _seed(db)
    execute = db.execute_query
    source_statements = []

    def concurrent_write(query, params=None, **kwargs):
        result = execute(query, params, **kwargs)
        if "FROM claims_review_log" in query:
            source_statements.append(query)
            # Commit on a separate connection before the domain consumes results.
            db.backend.execute(
                "INSERT INTO claims_review_log (claim_id, new_status, reason_code, created_at) "
                "VALUES (%s, 'approved', 'spam', %s)",
                (claim, datetime(2026, 9, 6, 1, tzinfo=timezone.utc)),
            )
        return result

    monkeypatch.setattr(db, "execute_query", concurrent_write)
    assert _aggregate(db) == 1
    row = db.list_claims_review_extractor_metrics_daily(user_id="42")[0]
    assert row["total_reviewed"] == json.loads(row["reason_code_counts_json"])["spam"] == 1
    assert len(source_statements) == 1
    monkeypatch.setattr(db, "execute_query", execute)
    assert _aggregate(db) == 1
    assert db.list_claims_review_extractor_metrics_daily(user_id="42")[0]["total_reviewed"] == 2


def test_owner_lock_serializes_overlapping_snapshots_but_not_other_owners(pg_review_db, monkeypatch):
    db = pg_review_db
    claim = _seed(db)
    _seed(db, owner="43")
    waiting_db = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend)
    independent_db = MediaDatabase(db_path=":memory:", client_id="43", backend=db.backend)
    attempted_lock = threading.Event()
    execute = waiting_db.execute_query

    def record_lock(query, params=None, **kwargs):
        if "pg_advisory_xact_lock(" in query:
            attempted_lock.set()
        return execute(query, params, **kwargs)

    monkeypatch.setattr(waiting_db, "execute_query", record_lock)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            with db.transaction():
                db.lock_claims_review_metrics_owner(owner_user_id="42")
                old = db.get_claims_review_metrics_window_rows(owner_user_id="42", start_date=START, end_date=END)
                assert next(row for row in old if row["kind"] == "metrics")["total_reviewed"] == 1
                db.backend.execute(
                    "INSERT INTO claims_review_log (claim_id, new_status, reason_code, created_at) "
                    "VALUES (%s, 'approved', 'spam', %s)",
                    (claim, datetime(2026, 9, 6, 1, tzinfo=timezone.utc)),
                )
                waiting = pool.submit(_aggregate, waiting_db)
                assert attempted_lock.wait(5)
                independent = pool.submit(_aggregate, independent_db, "43")
                assert independent.result(timeout=5) == 1
                assert not waiting.done()
                db.upsert_claims_review_extractor_metrics_daily(
                    user_id="42",
                    report_date=START.isoformat(),
                    extractor="heuristic",
                    extractor_version="v1",
                    total_reviewed=1,
                    approved_count=1,
                )
            assert waiting.result(timeout=5) == 1
        assert db.list_claims_review_extractor_metrics_daily(user_id="42")[0]["total_reviewed"] == 2
    finally:
        waiting_db.close_connection()
        independent_db.close_connection()


def test_concurrent_initial_upserts_converge_on_one_postgres_row(pg_review_db):
    db = pg_review_db
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend)
    barrier = threading.Barrier(2)

    def write(session):
        with scoped_context(user_id=42, is_admin=True):
            barrier.wait(timeout=5)
            return session.upsert_claims_review_extractor_metrics_daily(
                user_id="42",
                report_date=START.isoformat(),
                extractor="heuristic",
                total_reviewed=3,
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, session) for session in (db, other)]
            rows = [future.result(timeout=10) for future in futures]
        assert rows[0]["id"] == rows[1]["id"]
        assert db.count_claims_review_extractor_metrics_daily(user_id="42") == 1
    finally:
        other.close_connection()


def test_postgres_partial_failure_rolls_back_and_restores_scope_on_pool_reuse(pg_review_db, monkeypatch):
    db = pg_review_db
    _seed(db, extractor="alpha")
    _seed(db, extractor="beta")
    upsert = db.upsert_claims_review_extractor_metrics_daily
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected window failure")
        return upsert(**kwargs)

    monkeypatch.setattr(db, "upsert_claims_review_extractor_metrics_daily", fail_second)
    with scoped_context(user_id=999, org_ids=[88], team_ids=[99], is_admin=False) as original:
        with pytest.raises(RuntimeError, match="injected window failure"):
            _aggregate(db)
        assert get_scope() is original
        for _ in range(5):
            assert db.list_claims_review_user_ids_page(start_date=START, end_date=END) == []
        settings = db.execute_query(
            "SELECT current_setting('app.current_user_id', true) AS owner, "
            "current_setting('app.is_admin', true) AS admin",
        ).fetchone()
        assert settings["owner"] == "999"
        assert settings["admin"] in {"false", "0"}
    assert db.list_claims_review_extractor_metrics_daily(user_id="42") == []
