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
    method_name = "_migrate_from_v67_to_v68_postgres" if backend_name == "postgres" else "_migrate_from_v67_to_v68"
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
