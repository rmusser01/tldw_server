"""Non-provisioning access to Watchlists on caller-owned Media storage."""

import importlib

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def existing_media(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "existing.db"), client_id="42")
    yield db
    db.close_connection()


def _inventory(db, root):
    tables = [dict(row) for row in db.execute_query("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()]
    return tables, sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _forbid_bootstrap(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("existing-only access must not provision or resolve global storage")

    for name in (
        "__init__",
        "for_user",
        "_resolve_backend",
        "_ensure_schema_for_key",
        "_ensure_schema_and_tag_identity",
        "_ensure_schema_deduplicated",
        "ensure_schema",
        "ensure_schema_once",
    ):
        monkeypatch.setattr(WatchlistsDatabase, name, forbidden)
    module = importlib.import_module(WatchlistsDatabase.__module__)
    monkeypatch.setattr(module, "ensure_tag_identity", forbidden)


def test_missing_watchlists_schema_is_not_initialized(existing_media, tmp_path, monkeypatch):
    before = _inventory(existing_media, tmp_path)
    _forbid_bootstrap(monkeypatch)
    assert WatchlistsDatabase.open_existing(user_id="42", backend=existing_media.backend) is None
    assert _inventory(existing_media, tmp_path) == before


@pytest.mark.parametrize(
    "table,column",
    [
        ("watchlist_clusters", "job_id"),
        ("watchlist_clusters", "cluster_id"),
        ("watchlist_clusters", "created_at"),
        ("scrape_jobs", "id"),
        ("scrape_jobs", "user_id"),
        ("claim_clusters", "id"),
        ("claim_clusters", "user_id"),
        ("claim_clusters", "canonical_claim_text"),
        ("claim_clusters", "updated_at"),
        ("claim_clusters", "watchlist_count"),
        ("claim_cluster_membership", "cluster_id"),
        ("claims_notifications", "id"),
        ("claims_notifications", "user_id"),
        ("claims_notifications", "kind"),
        ("claims_notifications", "target_user_id"),
        ("claims_notifications", "target_review_group"),
        ("claims_notifications", "resource_type"),
        ("claims_notifications", "resource_id"),
        ("claims_notifications", "payload_json"),
        ("claims_notifications", "created_at"),
        ("claims_notifications", "delivered_at"),
    ],
)
def test_partial_evaluator_schema_is_absent_without_repair(
    existing_media,
    tmp_path,
    monkeypatch,
    table,
    column,
):
    WatchlistsDatabase(user_id="42", backend=existing_media.backend)
    info = existing_media.backend.get_table_info

    def missing_column(name, connection=None):
        return [row for row in info(name, connection=connection) if not (name == table and row["name"] == column)]

    monkeypatch.setattr(existing_media.backend, "get_table_info", missing_column)
    before = _inventory(existing_media, tmp_path)
    _forbid_bootstrap(monkeypatch)
    assert WatchlistsDatabase.open_existing(user_id="42", backend=existing_media.backend) is None
    assert _inventory(existing_media, tmp_path) == before


def test_missing_notifications_table_is_not_repaired(existing_media, tmp_path, monkeypatch):
    WatchlistsDatabase(user_id="42", backend=existing_media.backend)
    existing_media.execute_query("DROP TABLE claims_notifications", commit=True)
    before = _inventory(existing_media, tmp_path)
    _forbid_bootstrap(monkeypatch)
    assert WatchlistsDatabase.open_existing(user_id="42", backend=existing_media.backend) is None
    assert _inventory(existing_media, tmp_path) == before


def test_existing_handle_pins_owner_backend_across_operations_and_transaction(
    existing_media,
    monkeypatch,
):
    ordinary = WatchlistsDatabase(user_id="42", backend=existing_media.backend)
    other = WatchlistsDatabase(user_id="43", backend=existing_media.backend)
    job = _job(ordinary)
    foreign_job = _job(other)
    ordinary.add_watchlist_cluster(job.id, 100)
    other.add_watchlist_cluster(foreign_job.id, 200)
    _forbid_bootstrap(monkeypatch)
    db = WatchlistsDatabase.open_existing(user_id="42", backend=existing_media.backend)
    assert db is not None
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    for _ in range(2):
        assert db.backend is existing_media.backend
        assert db.list_watchlist_cluster_counts() == {100: 1}
        assert [row["cluster_id"] for row in db.list_watchlist_cluster_subscriptions()] == [100]
        with db.transaction():
            assert db.backend is existing_media.backend
            assert db.list_watchlist_cluster_counts() == {100: 1}
        assert db.list_watchlist_clusters(foreign_job.id) == []
    assert existing_media.execute_query("SELECT 1").fetchone()[0] == 1


def _job(db):
    return db.create_job(
        name="Subscribed",
        description=None,
        scope_json="{}",
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
        job_filters_json=None,
    )


@pytest.mark.parametrize("error", [PermissionError, OSError, DatabaseError])
def test_access_errors_are_not_schema_absence(existing_media, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error("private SQL provider path credential")

    monkeypatch.setattr(existing_media.backend, "get_table_info", fail)
    with pytest.raises(error):
        WatchlistsDatabase.open_existing(user_id="42", backend=existing_media.backend)


def test_legacy_for_user_still_initializes_storage(tmp_path, monkeypatch):
    module = importlib.import_module(WatchlistsDatabase.__module__)
    path = tmp_path / "legacy" / "media.db"
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.setattr(module, "load_comprehensive_config", lambda: None)
    monkeypatch.setattr(module.DatabasePaths, "get_media_db_path", lambda owner: path)
    db = WatchlistsDatabase.for_user("42")
    assert path.exists()
    assert db.backend.table_exists("watchlist_clusters")
