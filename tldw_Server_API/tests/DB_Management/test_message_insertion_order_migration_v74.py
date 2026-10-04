"""Real SQLite migration and transactional message insertion-order contracts."""

import sqlite3
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError

pytestmark = pytest.mark.integration


@pytest.fixture
def legacy_v73(tmp_path, monkeypatch):
    path = tmp_path / "legacy-v73.db"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        db = CharactersRAGDB(path, client_id="order-owner")
        try:
            cid = db.add_conversation({"title": "Preserved"})
            ids = [
                db.add_message({"conversation_id": cid, "sender": sender, "content": content, "timestamp": timestamp})
                for sender, content, timestamp in (
                    ("user", "Legacy user", "2099-01-01T00:00:00Z"),
                    ("assistant", "Legacy answer", "1999-01-01T00:00:00Z"),
                )
            ]
            rows = [db.get_message_by_id(mid) for mid in ids]
        finally:
            db.close_all_connections()
    return path, cid, ids, rows


def order_schema(db):
    return [tuple(row) for row in db.get_connection().execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name LIKE '%insertion_order%' ORDER BY name"
    )]


def test_v74_upgrade_preserves_legacy_and_rejects_unknown_anchor(legacy_v73):
    path, cid, ids, rows = legacy_v73
    with sqlite3.connect(path) as conn:
        upstream_schema = set(conn.execute("SELECT type, name, sql FROM sqlite_master"))
    db = CharactersRAGDB(path, client_id="order-owner")
    try:
        assert upstream_schema <= {tuple(row) for row in db.get_connection().execute(
            "SELECT type, name, sql FROM sqlite_master"
        )}
        assert [db.get_message_by_id(mid) for mid in ids] == rows
        assert db.get_connection().execute("SELECT COUNT(*) FROM message_insertion_order").fetchone()[0] == 0
        context = db.get_conversation_by_id(cid)
        with pytest.raises(ConflictError, match="insertion order"):
            db.insert_or_validate_user_turn(
                cid, ids[0], "Legacy user", owner_client_id=db.client_id, conversation_context=context
            )
        uid = str(uuid4())
        for limit, expected in ((20, [ids[1], ids[0], uid]), (1, [ids[0], uid]), (0, [uid])):
            history = db.insert_or_validate_user_turn(
                cid, uid, "New user", owner_client_id=db.client_id, conversation_context=context, history_limit=limit
            )
            assert [row["id"] for row in history] == expected
        assert [db.get_message_by_id(mid) for mid in ids] == rows
        assert db._get_db_version(db.get_connection()) == 75
    finally:
        db.close_all_connections()


def test_v74_fresh_and_upgraded_schema_equivalence(legacy_v73, tmp_path):
    upgraded = CharactersRAGDB(legacy_v73[0], client_id="order-owner")
    fresh = CharactersRAGDB(tmp_path / "fresh.db", client_id="order-owner")
    try:
        assert order_schema(fresh)
        assert order_schema(fresh) == order_schema(upgraded)
    finally:
        upgraded.close_all_connections()
        fresh.close_all_connections()


def test_v74_failed_migration_rolls_back_schema_version_and_data(legacy_v73, monkeypatch):
    path, _, ids, rows = legacy_v73

    class FailedMigration(CharactersRAGDB):
        def _migrate_from_v74_to_v75(self, conn):
            super()._migrate_from_v74_to_v75(conn)
            raise sqlite3.OperationalError("injected migration failure")

    with pytest.raises(CharactersRAGDBError, match="injected migration failure"):
        FailedMigration(path, client_id="order-owner")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM db_schema_version").fetchone()[0] == 73
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE '%insertion_order%'").fetchall() == []
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        db = CharactersRAGDB(path, client_id="order-owner")
        try:
            assert [db.get_message_by_id(mid) for mid in ids] == rows
        finally:
            db.close_all_connections()
    recovered = CharactersRAGDB(path, client_id="order-owner")
    try:
        assert recovered._get_db_version(recovered.get_connection()) == 75
    finally:
        recovered.close_all_connections()


def test_v74_order_is_atomic_immutable_and_cascades(tmp_path):
    db = CharactersRAGDB(tmp_path / "atomic.db", client_id="order-owner")
    try:
        cid = db.add_conversation({"title": "Atomic"})
        rolled_back = str(uuid4())
        with pytest.raises(RuntimeError, match="rollback"):
            with db.transaction() as conn:
                db.add_message({"id": rolled_back, "conversation_id": cid, "sender": "user", "content": "Rollback"}, conn=conn)
                assert conn.execute("SELECT sequence FROM message_insertion_order WHERE message_id = ?", (rolled_back,)).fetchone()
                raise RuntimeError("rollback")
        assert db.get_message_by_id(rolled_back) is None
        assert db.get_connection().execute("SELECT COUNT(*) FROM message_insertion_order").fetchone()[0] == 0
        mid = db.add_message({"conversation_id": cid, "sender": "user", "content": "Kept"})
        original = tuple(db.get_connection().execute("SELECT * FROM message_insertion_order WHERE message_id = ?", (mid,)).fetchone())
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with db.transaction() as conn:
                conn.execute("UPDATE message_insertion_order SET sequence = sequence + 100 WHERE message_id = ?", (mid,))
        with db.transaction() as conn:
            conn.execute("UPDATE messages SET timestamp = ?, content = ? WHERE id = ?", ("1999-01-01T00:00:00Z", "Edited", mid))
        assert tuple(db.get_connection().execute("SELECT * FROM message_insertion_order WHERE message_id = ?", (mid,)).fetchone()) == original
        with db.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
        assert db.get_connection().execute("SELECT COUNT(*) FROM message_insertion_order").fetchone()[0] == 0
    finally:
        db.close_all_connections()
