"""Post-publication best effort does not grant access via subscriptions."""

import importlib
import json

import pytest
from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.claims_notifications import (
    record_watchlist_cluster_notifications,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.tests.Watchlists.test_watchlists_existing_only import (
    _forbid_bootstrap,
    _inventory,
    _job,
)

MODULE = "tldw_Server_API.app.core.Claims_Extraction.claims_cluster_watchlists"


def _evaluate(db):
    return importlib.import_module(MODULE).evaluate_published_cluster_watchlists(db, owner_user_id="42")


@pytest.fixture(params=["sqlite", "postgres"])
def shared_media(request, tmp_path):
    if request.param == "postgres":
        backend = request.getfixturevalue("pg_restricted_backend")
        db = MediaDatabase(db_path=":memory:", client_id="42", backend=backend)
    else:
        db = MediaDatabase(db_path=str(tmp_path / "shared.db"), client_id="42")
    yield db
    db.close_connection()


def _seed_cluster(db, owner, text):
    if db.backend_type == BackendType.POSTGRESQL and owner != 42:
        owner_db = MediaDatabase(
            db_path=":memory:",
            client_id=str(owner),
            backend=db.backend,
            existing_only=True,
        )
        try:
            return _seed_cluster_for_owner(owner_db, owner, text)
        finally:
            owner_db.close_connection()
    return _seed_cluster_for_owner(db, owner, text)


def _seed_cluster_for_owner(db, owner, text):
    with scoped_context(user_id=owner, is_admin=False):
        media, _, _ = db.add_media_with_keywords(
            title=text,
            media_type="text",
            content=text,
            keywords=None,
            owner_user_id=owner,
        )
        db.upsert_claims(
            [
                {
                    "media_id": media,
                    "chunk_index": 0,
                    "claim_text": text,
                    "extractor": "test",
                    "extractor_version": "1",
                    "chunk_hash": text,
                }
            ]
        )
        db.rebuild_claim_clusters_exact(user_id=str(owner), min_size=1)
        return int(db.list_claim_clusters(str(owner), limit=10, offset=0)[0]["id"])


def _mixed_subscriptions(db):
    own = _seed_cluster(db, 42, "Owner A claim")
    foreign = _seed_cluster(db, 43, "Private owner B claim")
    watch = WatchlistsDatabase(user_id="42", backend=db.backend)
    job = _job(watch)
    # The existing API legitimately accepts a foreign cluster for an owned job.
    watch.add_watchlist_cluster(job.id, foreign)
    watch.add_watchlist_cluster(job.id, own)
    db.update_claim_clusters_watchlist_counts({foreign: 17})
    return own, foreign, job.id


def test_foreign_cluster_subscription_neither_discloses_nor_mutates(shared_media, monkeypatch):
    db = shared_media
    own, foreign, job = _mixed_subscriptions(db)
    cluster_read = db.get_claim_clusters_by_ids
    member_read = db.get_claim_cluster_member_counts
    update = db.update_claim_clusters_watchlist_counts

    def clusters(ids, *, owner_user_id=None):
        assert owner_user_id == "42"
        rows = cluster_read(ids, owner_user_id=owner_user_id)
        assert [row["id"] for row in rows] == [own]
        return rows

    def members(ids, *, owner_user_id=None):
        assert ids == [own]
        assert owner_user_id == "42"
        return member_read(ids, owner_user_id=owner_user_id)

    def counts(values, *, owner_user_id=None):
        assert values == {own: 1}
        assert owner_user_id == "42"
        return update(values, owner_user_id=owner_user_id)

    monkeypatch.setattr(db, "get_claim_clusters_by_ids", clusters)
    monkeypatch.setattr(db, "get_claim_cluster_member_counts", members)
    monkeypatch.setattr(db, "update_claim_clusters_watchlist_counts", counts)
    _forbid_bootstrap(monkeypatch)
    with scoped_context(user_id=42, is_admin=False):
        assert _evaluate(db) is None
        assert _evaluate(db) is None  # existing member-count deduplication
        notifications = db.list_claim_notifications(user_id="42", kind="watchlist_cluster_update")
    assert len(notifications) == 1
    assert json.loads(notifications[0]["payload_json"]) == {
        "cluster_id": own,
        "canonical_claim_text": "Owner A claim",
        "member_count": 1,
        "watchlist_job_ids": [job],
    }
    assert db.execute_query("SELECT watchlist_count FROM claim_clusters WHERE id = ?", (foreign,)).fetchone()[0] == 17
    assert db.execute_query("SELECT watchlist_count FROM claim_clusters WHERE id = ?", (own,)).fetchone()[0] == 1
    assert db.list_claim_notifications(user_id="43", kind="watchlist_cluster_update") == []


def test_owner_scoped_aggregate_helpers_reject_foreign_and_stale_ids(shared_media):
    db = shared_media
    own, foreign, _ = _mixed_subscriptions(db)
    stale = max(own, foreign) + 10000
    if db.backend_type == BackendType.POSTGRESQL:
        # The official role is restricted; cluster tables currently have no RLS.
        flags = db.execute_query("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone()
        assert not flags[0] and not flags[1]
    with scoped_context(user_id=42, is_admin=False):
        # Prove the explicit owner predicate, not an ambient RLS filter, protects B.
        assert len(db.get_claim_clusters_by_ids([own, foreign])) == 2
        assert [
            row["id"]
            for row in db.get_claim_clusters_by_ids(
                [own, foreign, stale],
                owner_user_id="42",
            )
        ] == [own]
        assert db.get_claim_cluster_member_counts([own, foreign, stale], owner_user_id="42") == {own: 1}
        assert db.get_claim_cluster_member_counts([foreign, stale], owner_user_id="42") == {}
        assert (
            db.update_claim_clusters_watchlist_counts(
                {own: 3, foreign: 99, stale: 7},
                owner_user_id="42",
            )
            == 1
        )
        assert db.update_claim_clusters_watchlist_counts({foreign: 99, stale: 7}, owner_user_id="42") == 0
        assert db.get_claim_clusters_by_ids([], owner_user_id="42") == []
        assert db.get_claim_cluster_member_counts([], owner_user_id="42") == {}
        assert db.update_claim_clusters_watchlist_counts({}, owner_user_id="42") == 0
    assert db.execute_query("SELECT watchlist_count FROM claim_clusters WHERE id = ?", (foreign,)).fetchone()[0] == 17
    assert db.execute_query("SELECT watchlist_count FROM claim_clusters WHERE id = ?", (own,)).fetchone()[0] == 3


@pytest.mark.parametrize(
    "helper", ["get_claim_clusters_by_ids", "get_claim_cluster_member_counts", "update_claim_clusters_watchlist_counts"]
)
def test_each_storage_helper_independently_enforces_owner(shared_media, helper):
    db = shared_media
    own, foreign, _ = _mixed_subscriptions(db)
    stale = max(own, foreign) + 10000
    with scoped_context(user_id=42, is_admin=False):
        if helper == "update_claim_clusters_watchlist_counts":
            assert getattr(db, helper)({own: 2, foreign: 99, stale: 7}, owner_user_id="42") == 1
            assert (
                db.execute_query("SELECT watchlist_count FROM claim_clusters WHERE id = ?", (foreign,)).fetchone()[0]
                == 17
            )
        elif helper == "get_claim_cluster_member_counts":
            assert getattr(db, helper)([own, foreign, stale], owner_user_id="42") == {own: 1}
        else:
            assert [row["id"] for row in getattr(db, helper)([own, foreign, stale], owner_user_id="42")] == [own]


def test_postgres_missing_watchlists_schema_is_not_initialized(request):
    backend = request.getfixturevalue("pg_restricted_backend")
    db = MediaDatabase(db_path=":memory:", client_id="42", backend=backend)
    query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    try:
        before = [dict(row) for row in db.execute_query(query).fetchall()]
        assert WatchlistsDatabase.open_existing(user_id="42", backend=backend) is None
        assert _evaluate(db) is None
        assert [dict(row) for row in db.execute_query(query).fetchall()] == before
    finally:
        db.close_connection()


def test_postgres_physical_missing_evaluator_column_preserves_schema_and_data(pg_restricted_backend, monkeypatch):
    """A real dropped column must not trigger schema repair or evaluation writes."""
    backend = pg_restricted_backend
    db = MediaDatabase(db_path=":memory:", client_id="42", backend=backend)

    def snapshot():
        tables = db.execute_query(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
        schema_queries = (
            "SELECT * FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename",
            "SELECT * FROM pg_catalog.pg_indexes WHERE schemaname = 'public' ORDER BY tablename, indexname",
            "SELECT c.relname, a.attnum, a.attname, a.attisdropped, a.attnotnull, "
            "pg_catalog.format_type(a.atttypid, a.atttypmod) AS type, "
            "pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS default_expr "
            "FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
            "WHERE n.nspname = 'public' AND a.attnum > 0 ORDER BY c.relname, a.attnum",
            "SELECT c.relname, k.conname, pg_catalog.pg_get_constraintdef(k.oid) AS definition "
            "FROM pg_catalog.pg_constraint k "
            "JOIN pg_catalog.pg_class c ON c.oid = k.conrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' ORDER BY c.relname, k.conname",
        )
        schema = [[dict(row) for row in db.execute_query(query).fetchall()] for query in schema_queries]
        data = {}
        for table in tables:
            name = table["tablename"]
            identifier = backend.escape_identifier(name)
            rows = db.execute_query(
                f"SELECT row_to_json(t)::text AS row_json FROM public.{identifier} AS t ORDER BY row_json"
            ).fetchall()
            data[name] = [row["row_json"] for row in rows]
        return schema, data

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete evaluator schema must not reach count or notification operations")

    try:
        with scoped_context(user_id=42, is_admin=False):
            cluster = _seed_cluster(db, 42, "Subscribed physical-column claim")
            watch = WatchlistsDatabase(user_id="42", backend=backend)
            job = _job(watch)
            watch.add_watchlist_cluster(job.id, cluster)
            db.update_claim_clusters_watchlist_counts({cluster: 17})
            notification = db.insert_claim_notification(user_id="42", kind="sentinel", payload_json='{"keep":true}')
            assert notification["payload_json"] == '{"keep":true}'
            role = db.execute_query(
                "SELECT current_user AS role, session_user AS session_role, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
            assert role["role"] == role["session_role"]
            assert not role["rolsuper"] and not role["rolbypassrls"]
            assert db.execute_query(
                "SELECT relowner = current_user::regrole AS owned "
                "FROM pg_catalog.pg_class WHERE oid = 'public.watchlist_clusters'::regclass"
            ).fetchone()["owned"]
            assert {row["name"] for row in backend.get_table_info("watchlist_clusters")} == {
                "job_id",
                "cluster_id",
                "created_at",
            }
            # Deliberate fixture damage precedes the baseline, never the operation under test.
            backend.execute("ALTER TABLE public.watchlist_clusters DROP COLUMN created_at")
            assert {row["name"] for row in backend.get_table_info("watchlist_clusters")} == {"job_id", "cluster_id"}
            before = snapshot()
            assert [json.loads(row) for row in before[1]["watchlist_clusters"]] == [
                {"job_id": job.id, "cluster_id": cluster}
            ]
            assert json.loads(before[1]["claim_clusters"][0])["watchlist_count"] == 17
            assert len(before[1]["claims_notifications"]) == 1
            _forbid_bootstrap(monkeypatch)
            for name in (
                "get_claim_cluster_member_counts",
                "update_claim_clusters_watchlist_counts",
                "get_latest_claim_notification",
                "insert_claim_notification",
            ):
                monkeypatch.setattr(db, name, forbidden)
            monkeypatch.setattr(WatchlistsDatabase, "list_watchlist_cluster_counts", forbidden)
            monkeypatch.setattr(importlib.import_module(MODULE), "record_watchlist_cluster_notifications", forbidden)
            assert WatchlistsDatabase.open_existing(user_id="42", backend=backend) is None
            assert snapshot() == before
            assert _evaluate(db) is None
            assert snapshot() == before
    finally:
        db.close_connection()


def test_postgres_permission_denial_is_not_schema_absence(request):
    backend = request.getfixturevalue("pg_restricted_backend")
    db = MediaDatabase(db_path=":memory:", client_id="42", backend=backend)
    WatchlistsDatabase(user_id="42", backend=backend)
    try:
        backend.execute("REVOKE SELECT ON watchlist_clusters FROM CURRENT_USER")
        with pytest.raises(DatabaseError):
            WatchlistsDatabase.open_existing(user_id="42", backend=backend)
    finally:
        backend.execute("GRANT SELECT ON watchlist_clusters TO CURRENT_USER")
        db.close_connection()


def test_missing_watchlists_schema_evaluation_preserves_inventory(tmp_path, monkeypatch):
    db = MediaDatabase(db_path=str(tmp_path / "existing.db"), client_id="42")
    try:
        before = _inventory(db, tmp_path)
        _forbid_bootstrap(monkeypatch)
        assert _evaluate(db) is None
        assert _inventory(db, tmp_path) == before
    finally:
        db.close_connection()


@pytest.mark.parametrize("foreign_only", [False, True])
def test_no_eligible_subscription_does_not_count_update_or_notify(shared_media, monkeypatch, foreign_only):
    db = shared_media
    own, foreign, job = _mixed_subscriptions(db)
    watch = WatchlistsDatabase.open_existing(user_id="42", backend=db.backend)
    watch.remove_watchlist_cluster(job, own)
    if not foreign_only:
        watch.remove_watchlist_cluster(job, foreign)

    def forbidden(*args, **kwargs):
        pytest.fail("no eligible cluster may reach count/update/notification operations")

    for name in (
        "get_claim_cluster_member_counts",
        "update_claim_clusters_watchlist_counts",
        "get_latest_claim_notification",
        "insert_claim_notification",
    ):
        monkeypatch.setattr(db, name, forbidden)
    _forbid_bootstrap(monkeypatch)
    assert _evaluate(db) is None


@pytest.mark.parametrize(
    "stage", ["open", "subscriptions", "clusters", "members", "counts", "update", "latest", "insert"]
)
def test_best_effort_access_errors_are_sanitized_once(shared_media, monkeypatch, stage):
    db = shared_media
    _mixed_subscriptions(db)
    module = importlib.import_module(MODULE)
    secret = "SELECT private_claim FROM provider /secret/path credential=hunter2"

    def fail(*args, **kwargs):
        raise DatabaseError(secret)

    if stage == "open":
        monkeypatch.setattr(WatchlistsDatabase, "open_existing", fail)
    elif stage in {"subscriptions", "counts"}:
        name = "list_watchlist_cluster_subscriptions" if stage == "subscriptions" else "list_watchlist_cluster_counts"
        monkeypatch.setattr(WatchlistsDatabase, name, fail)
    else:
        name = {
            "clusters": "get_claim_clusters_by_ids",
            "members": "get_claim_cluster_member_counts",
            "update": "update_claim_clusters_watchlist_counts",
            "latest": "get_latest_claim_notification",
            "insert": "insert_claim_notification",
        }[stage]
        monkeypatch.setattr(db, name, fail)
    captured = []
    sink = logger.add(lambda message: captured.append(message.record), level="DEBUG")
    try:
        assert module.evaluate_published_cluster_watchlists(db, owner_user_id="42") is None
    finally:
        logger.remove(sink)
    assert len(captured) == 1
    assert secret not in str(captured)
    assert captured[0]["exception"] is None


def test_swallowed_notification_insert_error_does_not_log_raw_exception(shared_media, monkeypatch):
    secret = "SQL SELECT provider /private/path token=secret claim text"

    def fail(**kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(shared_media, "insert_claim_notification", fail)
    captured = []
    sink = logger.add(lambda message: captured.append(message.record), level="DEBUG")
    try:
        assert (
            record_watchlist_cluster_notifications(
                db=shared_media,
                owner_user_id="42",
                clusters={1: {"canonical_claim_text": "Private claim"}},
                member_counts={1: 1},
                subscriptions={1: [2]},
            )
            == 0
        )
    finally:
        logger.remove(sink)
    assert len(captured) == 1
    assert secret not in str(captured)
    assert captured[0]["exception"] is None
