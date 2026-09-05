"""Persisted revision-clock contracts for guarded Reading mutations."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

pytestmark = pytest.mark.unit
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    if request.param == "postgres":
        # The shared fixture otherwise replaces a named container on connection failure.
        monkeypatch.setenv("TLDW_TEST_NO_DOCKER", "1")
        _, db_name = request.getfixturevalue("isolated_test_environment")
        backend = DatabaseBackendFactory.create_backend(
            DatabaseConfig(
                backend_type=BackendType.POSTGRESQL,
                pg_host=os.getenv("TEST_DB_HOST", "localhost"),
                pg_port=int(os.getenv("TEST_DB_PORT", "5432")),
                pg_database=db_name,
                pg_user=os.getenv("TEST_DB_USER", "tldw_user"),
                pg_password=os.getenv("TEST_DB_PASSWORD", "TestPassword123!"),
                pg_sslmode=os.getenv("TEST_DB_SSLMODE", "prefer"),
            )
        )
        return CollectionsDatabase.from_backend(user_id="780", backend=backend)
    return CollectionsDatabase.for_user(780)


def make_reading(db, *, origin="reading"):
    return db.upsert_content_item(
        origin=origin,
        url="https://example.org/a",
        canonical_url="https://example.org/a",
        domain="example.org",
        title="Original",
        summary="Body",
        content_hash="a",
        word_count=1,
        published_at=None,
        tags=["news"],
    )


def test_item_edits_advance_once_and_equivalent_values_are_noops(db):
    original = make_reading(db)
    changed = db.update_content_item(
        original.id,
        title="Changed",
        tags=[" news ", "new", "new"],
        metadata={"one": 1, "two": 2},
        favorite=True,
    )
    assert changed.revision == original.revision + 1
    same = db.update_content_item(
        original.id,
        title="Changed",
        tags=["new", "news"],
        metadata={"two": 2, "one": 1},
        favorite=True,
    )
    assert same.revision == changed.revision
    assert same.updated_at == changed.updated_at
    assert same.tags == ["new", "news"]
    assert json.loads(same.metadata_json) == {"one": 1, "two": 2}


def test_upsert_identical_record_is_noop_and_changes_advance(db):
    original = make_reading(db)
    same = make_reading(db)
    assert same.revision == original.revision
    assert same.updated_at == original.updated_at
    db.update_content_item(original.id, title="Changed")
    before = db.get_content_item(original.id)
    restored = make_reading(db)
    assert restored.title == "Original"
    assert restored.revision == before.revision + 1


def test_nonreading_upsert_keeps_existing_refresh_timestamp_semantics(db, monkeypatch):
    original = make_reading(db, origin="watchlist")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    refreshed = make_reading(db, origin="watchlist")
    assert refreshed.updated_at == "2099-01-01T00:00:00+00:00"
    assert refreshed.revision == original.revision


def test_nonreading_explicit_update_keeps_timestamp_semantics(db, monkeypatch):
    original = make_reading(db, origin="watchlist")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB._utcnow_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )
    refreshed = db.update_content_item(original.id, title=original.title)
    assert refreshed.updated_at == "2099-01-01T00:00:00+00:00"
    assert refreshed.revision == original.revision


def test_recreated_reading_item_never_reuses_deleted_revision(db):
    original = make_reading(db)
    db.delete_content_item(original.id)
    recreated = make_reading(db)
    assert recreated.revision > original.revision


def test_failed_tag_write_rolls_back_item_and_revision(db, monkeypatch):
    original = make_reading(db)
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    original_replace = db._replace_item_tags

    def replace_then_fail(*args, **kwargs):
        original_replace(*args, **kwargs)
        raise RuntimeError("abort tags")

    monkeypatch.setattr(db, "_replace_item_tags", replace_then_fail)
    with pytest.raises(RuntimeError, match="abort tags"):
        db.update_content_item(original.id, title="Do not save", tags=["replacement"])
    remaining = db.get_content_item(original.id)
    assert remaining.title == "Original"
    assert remaining.tags == ["news"]
    assert remaining.revision == original.revision
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


def test_failed_insert_rolls_back_item_tags_and_clock(db, monkeypatch):
    clock = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar
    replace = db._replace_item_tags

    def replace_then_fail(*args, **kwargs):
        replace(*args, **kwargs)
        raise RuntimeError("abort insert")

    monkeypatch.setattr(db, "_replace_item_tags", replace_then_fail)
    with pytest.raises(RuntimeError, match="abort insert"):
        make_reading(db)
    assert db.list_content_items(origin="reading")[1] == 0
    assert db.backend.execute("SELECT COUNT(*) FROM content_item_tags", ()).scalar == 0
    assert db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).scalar == clock


def test_concurrent_identical_upserts_create_one_item_and_revision(db):
    ready = Barrier(2)

    def save():
        ready.wait(timeout=10)
        return make_reading(db)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first, second = [workers.submit(save) for _ in range(2)]
        rows = [first.result(timeout=15), second.result(timeout=15)]
    assert rows[0].id == rows[1].id
    assert rows[0].revision == rows[1].revision
    assert sorted(row.is_new for row in rows) == [False, True]


def test_read_paths_return_persisted_revision(db):
    original = make_reading(db)
    changed = db.update_content_item(original.id, notes="New notes")
    assert db.get_content_item(original.id).revision == changed.revision
    assert db.get_content_item_by_url(original.url).revision == changed.revision
    rows, total = db.list_content_items(origin="reading")
    assert total == 1
    assert rows[0].revision == changed.revision
    db.backend.execute("UPDATE content_items SET media_id = 42 WHERE id = ?", (original.id,))
    assert db.get_content_item_by_media_id(42).revision == changed.revision


def test_fts_failure_rolls_back_item_tags_and_revision(db, monkeypatch):
    original = make_reading(db)
    before = db.get_content_item(original.id)

    def fail_index(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(db, "_update_content_fts_entry", fail_index)
    with pytest.raises(RuntimeError, match="index unavailable"):
        db.update_content_item(original.id, title="Not committed", tags=["not-committed"])
    assert db.get_content_item(original.id) == before


def test_concurrent_disjoint_item_updates_preserve_both_changes(db):
    original = make_reading(db)
    ready = Barrier(2)

    def update(**kwargs):
        ready.wait(timeout=10)
        return db.update_content_item(original.id, **kwargs).revision

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(update, title="Edited")
        second = workers.submit(update, notes="Annotated")
        revisions = [first.result(timeout=15), second.result(timeout=15)]
    final = db.get_content_item(original.id)
    assert len(set(revisions)) == 2
    assert final.revision == max(revisions)
    assert final.title == "Edited"
    assert final.notes == "Annotated"


def test_url_read_keeps_tags_and_revision_in_one_snapshot(db, monkeypatch):
    original = make_reading(db)
    fetch = db._fetch_tags_for_item_ids
    armed = True

    def fetch_after_other_writer(item_ids, *, connection=None):
        nonlocal armed
        if armed:
            armed = False
            with ThreadPoolExecutor(max_workers=1) as workers:
                workers.submit(db.update_content_item, original.id, tags=["changed"]).result(timeout=15)
        return fetch(item_ids, connection=connection)

    monkeypatch.setattr(db, "_fetch_tags_for_item_ids", fetch_after_other_writer)
    snapshot = db.get_content_item_by_url(original.url)
    assert snapshot.tags == ["news"]
    assert snapshot.revision == original.revision
    assert db.get_content_item(original.id).tags == ["changed"]


def test_later_adapters_update_existing_item_and_search_index(db):
    original = make_reading(db)
    # First construction may create the file; only subsequent constructions hit the schema memo.
    CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    second = CollectionsDatabase.from_backend(user_id=db.user_id, backend=db.backend)
    changed = second.update_content_item(original.id, title="Replacement")
    assert changed.revision > original.revision
    matches, total = second.list_content_items(origin="reading", q="Replacement")
    assert total == 1
    assert matches[0].id == original.id
    assert second.list_content_items(origin="reading", q="Original")[1] == 0


def test_revision_clock_survives_schema_reinitialization(db):
    with db.transaction() as conn:
        first = db._next_reading_revision(conn)
    db.ensure_schema()
    with db.transaction() as conn:
        second = db._next_reading_revision(conn)
    assert first > 0
    assert second > first


def test_revision_allocation_rolls_back_with_transaction(db):
    with db.transaction() as conn:
        first = db._next_reading_revision(conn)
    with pytest.raises(RuntimeError, match="abort"):
        with db.transaction() as conn:
            db._next_reading_revision(conn)
            raise RuntimeError("abort")
    with db.transaction() as conn:
        next_committed = db._next_reading_revision(conn)
    assert next_committed == first + 1


def test_revision_clock_exhaustion_fails_without_wraparound(db):
    maximum = 2**63 - 1
    db.backend.execute("UPDATE reading_revision_clock SET value = ? WHERE id = 1", (maximum,))
    with pytest.raises(OverflowError, match="revision"):
        with db.transaction() as conn:
            db._next_reading_revision(conn)
    row = db.backend.execute("SELECT value FROM reading_revision_clock WHERE id = 1", ()).first
    assert row["value"] == maximum


def test_existing_items_receive_positive_persisted_revision(db):
    item = make_reading(db)
    row = db.backend.execute("SELECT revision FROM content_items WHERE id = ?", (item.id,)).first
    assert row["revision"] > 0


def test_clock_migration_preserves_existing_item_revision(db):
    item = make_reading(db)
    db.backend.execute("UPDATE content_items SET revision = 57 WHERE id = ?", (item.id,))
    db.ensure_schema()
    row = db.backend.execute("SELECT revision FROM content_items WHERE id = ?", (item.id,)).first
    with db.transaction() as conn:
        allocated = db._next_reading_revision(conn)
    assert row["revision"] == 57
    assert allocated > 57


def test_legacy_schema_upgrade_preserves_item_data(db):
    item = make_reading(db)
    # Reconstruct the pre-feature shape, retaining a real existing capture.
    db.backend.execute("ALTER TABLE content_items DROP COLUMN revision", ())
    db.backend.execute("DROP TABLE reading_revision_clock", ())
    db.ensure_schema()
    db.ensure_schema()
    row = db.backend.execute(
        "SELECT title, revision FROM content_items WHERE id = ?",
        (item.id,),
    ).first
    assert row["title"] == "Original"
    assert row["revision"] > 0


def test_deleted_item_does_not_reset_clock(db):
    item = make_reading(db)
    with db.transaction() as conn:
        issued = db._next_reading_revision(conn)
        db.backend.execute(
            "UPDATE content_items SET revision = ? WHERE id = ?",
            (issued, item.id),
            connection=conn,
        )
    db.delete_content_item(item.id)
    db.ensure_schema()
    with db.transaction() as conn:
        assert db._next_reading_revision(conn) > issued


def test_concurrent_transactions_never_issue_same_revision(db):
    ready = Barrier(2)

    def allocate():
        ready.wait(timeout=10)
        with db.transaction() as conn:
            return db._next_reading_revision(conn)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(allocate) for _ in range(2)]
        revisions = [future.result(timeout=15) for future in futures]
    assert len(set(revisions)) == 2
    assert min(revisions) > 0


def test_postgres_search_path_cannot_reset_public_revision_clock(db, monkeypatch):
    if db.backend.backend_type != BackendType.POSTGRESQL:
        pytest.skip("PostgreSQL search-path regression")
    with db.transaction() as conn:
        issued = db._next_reading_revision(conn)
        db.backend.execute("CREATE SCHEMA reading_revision_probe", (), connection=conn)
    original_transaction = db.transaction

    @contextmanager
    def prefixed_transaction():
        with original_transaction() as conn:
            db.backend.execute(
                "SET LOCAL search_path TO reading_revision_probe, public",
                (),
                connection=conn,
            )
            yield conn

    monkeypatch.setattr(db, "transaction", prefixed_transaction)
    db._ensure_reading_revision_schema()
    db._ensure_reading_revision_schema()
    with db.transaction() as conn:
        assert db._next_reading_revision(conn) > issued
