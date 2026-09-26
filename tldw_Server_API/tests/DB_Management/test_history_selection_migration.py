"""Schema upgrades preserve legacy history and introduce protected provenance."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_upgrade_and_reopen_leave_legacy_transcript_unchanged(backend_name, request, tmp_path, monkeypatch):
    kwargs = {"db_path": str(tmp_path / "legacy.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 67)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 67)
        old = CharactersRAGDB(**kwargs)
        cid = old.add_conversation({"character_id": 1, "title": "Legacy"})
        mid = old.add_message({"conversation_id": cid, "sender": "user", "content": "keep this"})
        old.upsert_conversation_settings(cid, {"temperature": 0.25})
        before = dict(old.execute_query("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone())
        settings_before = dict(
            old.execute_query("SELECT * FROM conversation_settings WHERE conversation_id = ?", (cid,)).fetchone()
        )
        old.close_connection()
    for _ in range(2):
        upgraded = CharactersRAGDB(**kwargs)
        row = dict(upgraded.execute_query("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone())
        assert "history_admission_json" in row
        assert row.pop("history_admission_json") is None
        assert row == before
        assert (
            dict(
                upgraded.execute_query(
                    "SELECT * FROM conversation_settings WHERE conversation_id = ?", (cid,)
                ).fetchone()
            )
            == settings_before
        )
        assert upgraded.execute_query("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == 0
        upgraded.close_connection()
    if backend_name == "postgres":
        kwargs["backend"].get_pool().close_all()


def test_postgres_projection_rls_filters_actual_owner(pg_database_config):
    """A non-superuser sees only projections tied to its actual live conversation."""
    import uuid

    from tldw_Server_API.tests.DB_Management.test_history_selection_transactions import (
        add,
        confirm,
        confirm_body,
        snapshot,
    )

    db = CharactersRAGDB(
        db_path=":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    cid = db.add_conversation({"character_id": 1, "title": "RLS"})
    add(db, cid)
    confirm(db, cid, confirm_body(snapshot(db, cid)))
    role = "h1_probe_" + uuid.uuid4().hex

    class CleanupRole(Exception):
        pass

    try:
        with pytest.raises(CleanupRole):
            with db.transaction() as conn:
                conn.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER")  # nosec B608 - generated hex identifier
                conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")  # nosec B608
                conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}")  # nosec B608
                conn.execute(f"SET LOCAL ROLE {role}")  # nosec B608
                conn.execute("SELECT set_config('app.current_user_id', 'alice', true)")
                assert conn.execute("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == 1
                conn.execute("SELECT set_config('app.current_user_id', 'bob', true)")
                assert conn.execute("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == 0
                raise CleanupRole()
    finally:
        db.close_connection()
        db.backend.get_pool().close_all()


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_migration_failure_rolls_back_schema_and_version(backend_name, request, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    kwargs = {"db_path": str(tmp_path / "rollback.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 67)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 67)
        old = CharactersRAGDB(**kwargs)
        old.close_connection()
    method_name = "_migrate_from_v72_to_v73_postgres" if backend_name == "postgres" else "_migrate_from_v68_to_v69"
    migrate = getattr(CharactersRAGDB, method_name)

    def failing_migration(self, conn):
        migrate(self, conn)
        raise RuntimeError("migration interrupted after DDL")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, method_name, failing_migration)
        with pytest.raises((RuntimeError, CharactersRAGDBError)):
            CharactersRAGDB(**kwargs)
    assert (
        old.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
        ).fetchone()["version"]
        == 67
    )
    catalog = (
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'conversation_history_projections'"
        if backend_name == "postgres"
        else "SELECT name FROM sqlite_master WHERE name = 'conversation_history_projections'"
    )
    assert old.execute_query(catalog).fetchone() is None
    old.close_connection()
    restored = CharactersRAGDB(**kwargs)
    assert restored.execute_query("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == 0
    restored.close_connection()
    if backend_name == "postgres":
        kwargs["backend"].get_pool().close_all()


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
@pytest.mark.parametrize("lineage", ["h1-v68", "dev"])
def test_colliding_v68_lineages_upgrade_without_losing_history(
    backend_name, lineage, request, tmp_path, monkeypatch,
):
    """Exercise real historical DDL, including H1's conflicting version marker."""
    kwargs = {"db_path": str(tmp_path / "lineage.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 67 if lineage == "h1-v68" else 68)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 67 if lineage == "h1-v68" else 72)
        old = CharactersRAGDB(**kwargs)
        cid = old.add_conversation({"character_id": 1, "title": "Preserved"})
        mid = old.add_message({"conversation_id": cid, "sender": "user", "content": "original"})
        if lineage == "h1-v68":
            # Historical H1 DDL intentionally lacks dev's keyword/name migrations.
            with old.transaction() as conn:
                conn.execute(CharactersRAGDB._HISTORY_PROJECTIONS_SCHEMA_SQL)
                conn.execute("ALTER TABLE messages ADD COLUMN history_admission_json TEXT")
                conn.execute(
                    "UPDATE messages SET history_admission_json = ? WHERE id = ?",
                    ('{"historical":"admission"}', mid),
                )
                conn.execute(
                    "INSERT INTO conversation_history_projections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("projection", cid, "alice", "owner", 1, "digest", "fence", "[]", "[]", "{}", "digest", "2026-01-01"),
                )
                conn.execute(
                    "UPDATE db_schema_version SET version = 68 WHERE schema_name = ?",
                    (old._SCHEMA_NAME,),
                )
        before = dict(old.execute_query("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone())
        old.close_connection()
    try:
        for _ in range(2):
            db = CharactersRAGDB(**kwargs)
            try:
                after = dict(db.execute_query("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone())
                assert after == {**before, "history_admission_json": before.get("history_admission_json")}
                assert db.execute_query("SELECT COUNT(*) AS n FROM conversation_history_projections").fetchone()["n"] == (1 if lineage == "h1-v68" else 0)
                table = db._map_table_for_backend("keywords")
                db.execute_query(f"SELECT merged_into_sync_id FROM {table}").fetchall()  # nosec B608 - backend table mapping
                version = db.execute_query(
                    "SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,),
                ).fetchone()["version"]
                assert version == (73 if backend_name == "postgres" else 69)
                if backend_name == "postgres":
                    constraints = db.execute_query(
                        "SELECT conname FROM pg_constraint WHERE conname IN "
                        "('character_cards_client_id_name_key', 'decks_client_id_name_key', 'keyword_merge_tombstone')",
                    ).fetchall()
                    assert len(constraints) == 3
                if lineage == "h1-v68":
                    with pytest.raises(Exception) as immutable_error:
                        with db.transaction() as conn:
                            conn.execute("UPDATE conversation_history_projections SET source_digest = 'changed'")
                    if backend_name == "sqlite":
                        assert "immutable" in str(immutable_error.value)
                    assert db.execute_query(
                        "SELECT source_digest FROM conversation_history_projections",
                    ).fetchone()["source_digest"] == "digest"
            finally:
                db.close_connection()
    finally:
        if backend_name == "postgres":
            kwargs["backend"].get_pool().close_all()
