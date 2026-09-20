"""Real historical keyword schemas preserve data while adding merge metadata."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


class HistoricalKeywords(CharactersRAGDB):
    _CURRENT_SCHEMA_VERSION = 67
    _POSTGRES_SCHEMA_VERSION = 69


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def historical_keywords(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    path = tmp_path / "historical.db"
    old = HistoricalKeywords(path, client_id="2", backend=backend)
    instances = [old]
    table = old._map_table_for_backend("keywords")

    def rows():
        return [dict(row) for row in old.execute_query(f"SELECT * FROM {table} ORDER BY id", read_only=True).fetchall()]  # nosec B608

    def schema_metadata():
        if backend is None:
            return [
                dict(row)
                for row in old.execute_query(
                    "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = 'keywords' AND type <> 'table' ORDER BY type, name",
                    read_only=True,
                ).fetchall()
            ]
        indexes = backend.execute(
            "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname='public' AND tablename='chacha_keywords' ORDER BY indexname"
        ).rows
        triggers = backend.execute(
            "SELECT tgname,pg_get_triggerdef(oid) AS definition FROM pg_trigger WHERE tgrelid='chacha_keywords'::regclass AND NOT tgisinternal ORDER BY tgname"
        ).rows
        policies = backend.execute(
            "SELECT * FROM pg_policies WHERE schemaname='public' AND tablename='chacha_keywords' ORDER BY policyname"
        ).rows
        flags = backend.execute(
            "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='chacha_keywords'::regclass"
        ).rows
        return (indexes, triggers, policies, flags)

    def reopen():
        old.close_connection()
        db = CharactersRAGDB(path, client_id="2", backend=backend)
        instances.append(db)
        return db

    try:
        # Seed historical columns directly: current writers may require the new column.
        with old.transaction() as conn:
            for label, deleted in (("Historical active", False), ("Historical deleted", True)):
                conn.execute(
                    f"INSERT INTO {table}(sync_id, keyword, created_at, last_modified, deleted, client_id, version) "  # nosec B608
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (old._generate_uuid(), label, "2026-08-01T12:34:56Z", "2026-08-02T12:34:56Z", deleted, "2", 7),
                )
        before = rows()
        assert all("merged_into_sync_id" not in row for row in before)
        version = old.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (old._SCHEMA_NAME,), read_only=True
        ).fetchone()["version"]
        assert version == (69 if backend is not None else 67)
        yield SimpleNamespace(
            old=old,
            backend=backend,
            path=path,
            table=table,
            before=before,
            rows=rows,
            reopen=reopen,
            metadata_before=schema_metadata(),
            schema_metadata=schema_metadata,
        )
    finally:
        for instance in instances:
            instance.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def test_genuine_prior_schema_upgrade_preserves_rows_and_reopens(historical_keywords):
    f = historical_keywords
    for _ in range(2):
        db = f.reopen()
        after = f.rows()
        assert all("merged_into_sync_id" in row and row["merged_into_sync_id"] is None for row in after)
        assert [{key: value for key, value in row.items() if key != "merged_into_sync_id"} for row in after] == f.before
        version = db.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,), read_only=True
        ).fetchone()["version"]
        assert version == (CharactersRAGDB._POSTGRES_SCHEMA_VERSION if f.backend is not None else 68)
        assert f.schema_metadata() == f.metadata_before


@pytest.mark.parametrize("invalid", ["active-redirect", "self-redirect"])
def test_new_constraint_rejects_impossible_survivor(historical_keywords, invalid):
    f = historical_keywords
    db = f.reopen()
    live, deleted = f.rows()
    row, target = (live, deleted["sync_id"]) if invalid == "active-redirect" else (deleted, deleted["sync_id"])
    # A real database constraint error must roll back the attempted mutation.
    with pytest.raises(Exception) as error:
        with db.transaction() as conn:
            conn.execute(f"UPDATE {f.table} SET merged_into_sync_id = ? WHERE id = ?", (target, row["id"]))  # nosec B608
    if f.backend is not None:
        # The backend intentionally removes driver messages and SQLSTATE. Verify
        # the validated constraint catalog alongside the real rejected write.
        assert isinstance(error.value, DatabaseError)
        constraint = db.execute_query(
            "SELECT convalidated, pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid='chacha_keywords'::regclass AND conname='keyword_merge_tombstone'",
            read_only=True,
        ).fetchone()
        assert constraint["convalidated"]
        assert (
            constraint["definition"]
            == "CHECK (((merged_into_sync_id IS NULL) OR ((deleted = true) AND (merged_into_sync_id <> sync_id))))"
        )
    else:
        assert "keyword_merge_tombstone" in str(error.value)
    assert [{key: value for key, value in row.items() if key != "merged_into_sync_id"} for row in f.rows()] == f.before


def test_failed_upgrade_rolls_back_column_and_version(historical_keywords, monkeypatch):
    f = historical_keywords
    name = "_migrate_from_v69_to_v70_postgres" if f.backend is not None else "_migrate_from_v67_to_v68"
    original = getattr(CharactersRAGDB, name)
    reached = []

    def fail_after_migration(db, conn):
        original(db, conn)
        reached.append(True)
        raise RuntimeError("Controlled keyword migration failure")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, name, fail_after_migration)
        with pytest.raises(Exception, match="Controlled keyword migration failure"):
            f.reopen()
    assert reached == [True]
    assert f.rows() == f.before
    version = f.old.execute_query(
        "SELECT version FROM db_schema_version WHERE schema_name = ?", (f.old._SCHEMA_NAME,), read_only=True
    ).fetchone()["version"]
    assert version == (69 if f.backend is not None else 67)
    f.reopen()
    assert all("merged_into_sync_id" in row for row in f.rows())
