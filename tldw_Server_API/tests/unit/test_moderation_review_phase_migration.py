"""Neutral phase migration preserves review identity, history and undo leases."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from tldw_Server_API.app.core.DB_Management import Moderation_Review_DB as db
from tldw_Server_API.tests.unit.test_moderation_review_store import _item_payload

pytestmark = pytest.mark.unit


@pytest.fixture
def legacy(tmp_path):
    path = tmp_path / "legacy.db"
    store = db.ModerationReviewStore(path)
    item = store.upsert_item(_item_payload())
    decision = store.record_decision(item["id"], action="block", decided_by="ops", reason="retained")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE UNIQUE INDEX retained_custom_key ON moderation_review_items(source_id,user_id)")
        conn.execute(
            "CREATE TRIGGER retained_review_trigger AFTER UPDATE OF excerpt ON moderation_review_items BEGIN SELECT 1; END"
        )
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=replace(sql, ?, ?) WHERE name='moderation_review_items'",
            (
                db._PHASE_CONSTRAINT,
                "CHECK (phase IN ('input', 'output'))",
            ),
        )
        conn.commit()
    return path, item, decision


def snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        return {
            "moderation_review_items": conn.execute("SELECT * FROM moderation_review_items ORDER BY id").fetchall(),
            "moderation_review_decisions": conn.execute(
                "SELECT * FROM moderation_review_decisions ORDER BY id"
            ).fetchall(),
            "moderation_review_audit_events": conn.execute(
                "SELECT * FROM moderation_review_audit_events ORDER BY id"
            ).fetchall(),
        }


def related_schema(path):
    with closing(sqlite3.connect(path)) as conn:
        return {
            "indexes_and_triggers": conn.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE type IN ('index','trigger') ORDER BY name"
            ).fetchall(),
            "decision_fks": conn.execute("PRAGMA foreign_key_list(moderation_review_decisions)").fetchall(),
            "audit_fks": conn.execute("PRAGMA foreign_key_list(moderation_review_audit_events)").fetchall(),
        }


def test_migration_preserves_rows_ids_audit_undo_foreign_keys_and_indexes(legacy):
    path, item, decision = legacy
    before = snapshot(path)
    schema_before = related_schema(path)
    store = db.ModerationReviewStore(path)
    assert snapshot(path) == before
    assert related_schema(path) == schema_before
    db.ModerationReviewStore(path)
    assert snapshot(path) == before
    assert related_schema(path) == schema_before
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='idx_moderation_review_items_status_created'"
        ).fetchone()
    store.undo_decision(item["id"], undo_token=decision["undo_token"], actor_id="ops")
    assert store.get_item(item["id"])["status"] == "needs_review"


def test_migration_rolls_back_on_injected_failure(legacy, monkeypatch):
    path, _, _ = legacy
    before = snapshot(path)
    original = db.ModerationReviewStore._migrate_phase_constraint

    def fail(self, conn):
        original(self, conn)
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(db.ModerationReviewStore, "_migrate_phase_constraint", fail)
    with pytest.raises(RuntimeError):
        db.ModerationReviewStore(path)
    assert snapshot(path) == before
    with closing(sqlite3.connect(path)) as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='moderation_review_items'").fetchone()[0]
    assert "unspecified" not in sql


def test_concurrent_initialization_is_idempotent(legacy):
    path, _, _ = legacy
    before = snapshot(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: db.ModerationReviewStore(path), range(16)))
    assert snapshot(path) == before


@pytest.mark.parametrize(
    "source,category",
    [
        (None, None),
        (None, "moderation_engine_failure"),
        ("moderation_engine", None),
        ("chat", "moderation_engine_failure"),
        ("moderation_engine", "pii"),
    ],
)
def test_neutral_phase_is_reserved_in_sql_including_null(tmp_path, source, category):
    store = db.ModerationReviewStore(tmp_path / "review.db")
    with closing(sqlite3.connect(store.db_path)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO moderation_review_items(id,idempotency_key,phase,source_type,category,created_at,excerpt) VALUES ('x','x','unspecified',?,?,'now','safe')",
            (source, category),
        )


def test_incident_first_use_never_migrates_legacy_schema(legacy):
    path, _, _ = legacy
    with pytest.raises(RuntimeError, match="not prepared"):
        db.ModerationReviewStore(path, busy_timeout_ms=250)


@pytest.mark.parametrize("failure", [None, "setup", "write"])
def test_incident_connections_close_after_success_setup_and_write_failures(tmp_path, monkeypatch, failure):
    path = tmp_path / "review.db"
    db.ModerationReviewStore(path)
    store = db.ModerationReviewStore(path, busy_timeout_ms=250)
    connections = []
    connect = sqlite3.connect

    def track(*args, **kwargs):
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(db.sqlite3, "connect", track)
    if failure == "setup":
        monkeypatch.setattr(
            db, "configure_sqlite_connection", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("setup"))
        )
    if failure == "write":
        monkeypatch.setattr(store, "_append_audit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("write")))
    if failure:
        with pytest.raises(RuntimeError):
            store.upsert_item(_item_payload())
    else:
        store.upsert_item(_item_payload())
    assert connections
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    with closing(connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM moderation_review_items").fetchone()[0] == (
            1 if failure is None else 0
        )


def test_incident_initial_connect_and_wal_setup_obey_budget_and_close(tmp_path, monkeypatch):
    path = tmp_path / "locked.db"
    db.ModerationReviewStore(path)
    connect = sqlite3.connect
    connections, timeouts = [], []

    def track(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn

    with closing(connect(path)) as blocker:
        blocker.execute("PRAGMA journal_mode=DELETE")
        blocker.execute("BEGIN EXCLUSIVE")
        monkeypatch.setattr(db.sqlite3, "connect", track)
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db.ModerationReviewStore(path, busy_timeout_ms=250)
        assert time.monotonic() - started < 0.8
        blocker.rollback()
    assert timeouts == [0.25]
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_incident_connection_policy_has_bounded_wal_and_foreign_keys(tmp_path, monkeypatch):
    path = tmp_path / "review.db"
    db.ModerationReviewStore(path)
    configure = db.configure_sqlite_connection
    calls = []

    def track(conn, **kwargs):
        calls.append(kwargs)
        configure(conn, **kwargs)
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 250
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    monkeypatch.setattr(db, "configure_sqlite_connection", track)
    store = db.ModerationReviewStore(path, busy_timeout_ms=250)
    store.upsert_item(_item_payload())
    assert all(call == {"busy_timeout_ms": 250, "foreign_keys": True} for call in calls)


def test_review_response_only_accepts_reserved_neutral_phase():
    from pydantic import ValidationError

    from tldw_Server_API.app.api.v1.schemas.moderation_schemas import (
        ModerationOverrideRule,
        ModerationReviewItem,
        ModerationTestRequest,
    )

    payload = {
        **_item_payload(phase="unspecified", source_type="moderation_engine", category="moderation_engine_failure"),
        "id": "incident",
        "status": "needs_review",
        "created_at": "now",
    }
    assert ModerationReviewItem(**payload).phase == "unspecified"
    with pytest.raises(ValidationError):
        ModerationReviewItem(**{**payload, "source_type": None})
    with pytest.raises(ValidationError):
        ModerationTestRequest(text="safe", phase="unspecified")
    with pytest.raises(ValidationError):
        ModerationOverrideRule(id="r", pattern="safe", action="block", phase="unspecified")
