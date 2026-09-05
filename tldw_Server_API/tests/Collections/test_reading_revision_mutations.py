"""Persisted revision-clock contracts for guarded Reading mutations."""

from __future__ import annotations

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


def make_reading(db):
    return db.upsert_content_item(
        origin="reading",
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
