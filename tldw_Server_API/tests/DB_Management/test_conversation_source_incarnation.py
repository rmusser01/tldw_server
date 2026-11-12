"""Original conversation rows differ from same-ID recreation."""

from __future__ import annotations

import pytest

from tldw_Server_API.tests.DB_Management.test_native_fork_transactions import native_fork_db as native_fork_db

pytestmark = pytest.mark.integration


def incarnation(db, cid):
    return db.execute_query("SELECT source_incarnation_id FROM conversations WHERE id = ?", (cid,)).fetchone()[
        "source_incarnation_id"
    ]


def test_server_issues_identity_preserves_restore_and_retires_same_id(native_fork_db, monkeypatch):
    db = native_fork_db
    monkeypatch.setattr(db, "_get_current_utc_timestamp_iso", lambda: "2026-10-12T12:00:00.000Z")
    cid = db.add_conversation({"id": "original", "character_id": None, "source_incarnation_id": "copied-authority"})
    original = incarnation(db, cid)
    assert original and original != "copied-authority"
    version = db.get_conversation_by_id(cid)["version"]
    assert db.soft_delete_conversation(cid, version)
    assert db.restore_conversation(cid, version + 1)
    assert incarnation(db, cid) == original
    assert db.hard_delete_conversation(cid)
    assert db.add_conversation({"id": cid, "character_id": None, "source_incarnation_id": original}) == cid
    assert incarnation(db, cid) != original


def test_sync_insert_issues_identity_but_existing_upsert_preserves_it(native_fork_db):
    db = native_fork_db
    kwargs = {
        "conversation_id": "sync-source",
        "sync_client_id": "alice",
        "object_revision": 1,
        "object_hash": "hash",
        "title": "Source",
    }
    assert db.upsert_conversation_from_sync(**kwargs)
    original = incarnation(db, "sync-source")
    assert original
    assert db.upsert_conversation_from_sync(**(kwargs | {"object_revision": 2, "title": "Updated"}))
    assert incarnation(db, "sync-source") == original
    assert db.hard_delete_conversation("sync-source")
    assert db.upsert_conversation_from_sync(**kwargs)
    assert incarnation(db, "sync-source") != original


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_registered_incarnation_upgrade_preserves_rows_and_reopen(backend_name, request, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.tests.DB_Management.test_native_reference_role_migration import close_pool, database_kwargs

    kwargs = database_kwargs(backend_name, request, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 80)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 84)
        old = CharactersRAGDB(**kwargs)
        first = old.add_conversation({"character_id": None, "title": "Original row"})
        old.add_conversation({"character_id": None, "title": "Other row"})
        old.add_message({"conversation_id": first, "sender": "user", "content": "Preserved résumé 😀"})
        rows = [dict(row) for row in old.execute_query("SELECT * FROM conversations ORDER BY id").fetchall()]
        assert all("source_incarnation_id" not in row for row in rows)
        messages = [dict(row) for row in old.execute_query("SELECT * FROM messages").fetchall()]
        old.close_connection()
    try:
        original = None
        for _ in range(2):
            db = CharactersRAGDB(**kwargs)
            try:
                upgraded = [dict(row) for row in db.execute_query("SELECT * FROM conversations ORDER BY id").fetchall()]
                identities = [row.pop("source_incarnation_id") for row in upgraded]
                assert upgraded == rows
                assert len(set(identities)) == 2 and all(identities)
                if original is not None:
                    assert identities == original
                original = identities
                assert [dict(row) for row in db.execute_query("SELECT * FROM messages").fetchall()] == messages
                from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError

                before_forgery = [
                    dict(row) for row in db.execute_query("SELECT * FROM conversations ORDER BY id").fetchall()
                ]
                error_type = BackendDatabaseError if backend_name == "postgres" else Exception
                error_text = (
                    "^PostgreSQL query execution failed$"
                    if backend_name == "postgres"
                    else "conversation_incarnation_immutable"
                )
                with pytest.raises(error_type, match=error_text):
                    with db.transaction() as conn:
                        conn.execute("UPDATE conversations SET source_incarnation_id = 'forged' WHERE id = ?", (first,))
                assert [
                    dict(row) for row in db.execute_query("SELECT * FROM conversations ORDER BY id").fetchall()
                ] == before_forgery
                assert [dict(row) for row in db.execute_query("SELECT * FROM messages").fetchall()] == messages
            finally:
                db.close_connection()
    finally:
        close_pool(kwargs)


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_interrupted_incarnation_migration_rolls_back_rows_schema_and_triggers(
    backend_name, request, tmp_path, monkeypatch
):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.tests.DB_Management.test_native_reference_role_migration import close_pool, database_kwargs

    kwargs = database_kwargs(backend_name, request, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 80)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 84)
        old = CharactersRAGDB(**kwargs)
        cid = old.add_conversation({"character_id": None, "title": "Preserve interrupted upgrade"})
        before = dict(old.execute_query("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone())
        old.close_connection()
    name = "_migrate_from_v84_to_v85_postgres" if backend_name == "postgres" else "_migrate_from_v80_to_v81"
    migrate = getattr(CharactersRAGDB, name)

    def interrupted(self, conn):
        migrate(self, conn)
        raise RuntimeError("incarnation migration interrupted")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(CharactersRAGDB, name, interrupted)
            with pytest.raises(Exception, match="incarnation migration interrupted"):
                CharactersRAGDB(**kwargs)
        assert dict(old.execute_query("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()) == before
        assert old.execute_query("SELECT version FROM db_schema_version").fetchone()["version"] == (
            84 if backend_name == "postgres" else 80
        )
        if backend_name == "sqlite":
            assert (
                old.execute_query(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'conversation_incarnation_%'"
                ).fetchall()
                == []
            )
        else:
            assert (
                old.execute_query(
                    "SELECT tgname FROM pg_trigger WHERE tgname = 'conversation_incarnation_guard'"
                ).fetchall()
                == []
            )
    finally:
        old.close_connection()
        close_pool(kwargs)


def test_direct_insert_cannot_copy_an_original_row_incarnation(native_fork_db):
    db = native_fork_db
    cid = db.add_conversation({"character_id": None, "title": "Original"})
    copied = incarnation(db, cid)

    def insert():
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id,root_id,client_id,source_incarnation_id) VALUES ('direct-copy','direct-copy','alice',?)",
                (copied,),
            )

    if db.backend_type.value == "sqlite":
        with pytest.raises(Exception, match="conversation_incarnation_reserved"):
            insert()
    else:
        insert()
        assert incarnation(db, "direct-copy") != copied
