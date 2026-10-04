"""Durable turns use the existing AuthNZ isolated PostgreSQL fixture."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError


@pytest.fixture
def order_db(isolated_test_environment):
    backend = DatabaseBackendFactory.create_backend(DatabaseConfig.from_env())
    db = CharactersRAGDB(":memory:", client_id="durable-pg", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()


@pytest.mark.integration
def test_postgres_durable_retry_rejects_independently_appended_image(order_db):
    db = order_db
    db.upsert_workspace("durable-image-workspace", "Durable image workspace")
    cid = db.add_conversation({
        "title": "Imaged durable anchor", "scope_type": "workspace", "workspace_id": "durable-image-workspace",
    })
    db.upsert_conversation_settings(cid, {"provider": "openai", "model": "gpt-4o-mini"})
    uid = str(uuid4())
    context = db.get_conversation_by_id(cid)
    db.insert_or_validate_user_turn(
        cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context,
    )
    db.append_message_image(uid, b"independently appended image", "image/png")
    assert db.get_message_by_id(uid)["image_data"] is None
    before_rows = db.get_messages_for_conversation(cid)
    before_images = db.get_message_images(uid, strict=True)
    before_settings = db.get_conversation_settings(cid)
    before_conversation = db.get_conversation_by_id(cid)
    with pytest.raises(ConflictError, match="identity conflicts"):
        db.insert_or_validate_user_turn(
            cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context,
        )
    assert db.get_messages_for_conversation(cid) == before_rows
    assert db.get_message_images(uid, strict=True) == before_images
    assert db.get_conversation_settings(cid) == before_settings
    assert db.get_conversation_by_id(cid) == before_conversation


def insert_writer(
    db: CharactersRAGDB,
    writer: str,
    cid: str,
    mid: str,
    sender: str,
    timestamp: str,
) -> None:
    """Use each real shared writer, including SQL outside add_message."""
    if writer == "ordinary":
        db.add_message({"id": mid, "conversation_id": cid, "sender": sender, "content": "Later", "timestamp": timestamp})
    elif writer == "sync":
        db.append_message_from_sync(
            stable_message_id=mid, conversation_id=cid, sender=sender, content="Later", timestamp=timestamp,
            sync_client_id=db.client_id, object_revision=1, payload_hash=mid,
        )
    else:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO messages(id, conversation_id, sender, content, timestamp, client_id) VALUES (?, ?, ?, ?, ?, ?)",
                (mid, cid, sender, "Later", timestamp, db.client_id),
            )


@pytest.mark.integration
def test_postgres_insertion_order_is_owner_isolated(order_db):
    db, backend = order_db, order_db.backend
    cid = db.add_conversation({"title": "Private order"})
    mid = db.add_message({"conversation_id": cid, "sender": "user", "content": "Private"})
    role = backend.escape_identifier(f"durable_order_{uuid4().hex[:12]}")
    created_role = False
    db.close_connection()
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT SELECT ON messages, conversations, message_insertion_order TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created_role = True
        with backend.transaction() as conn:
            backend.execute(f"SET LOCAL ROLE {role}", connection=conn)
            backend.execute("SELECT set_config('app.current_user_id', %s, true)", (db.client_id,), connection=conn)
            assert backend.execute("SELECT message_id FROM message_insertion_order", connection=conn).rows == [{"message_id": mid}]
            backend.execute("SELECT set_config('app.current_user_id', %s, true)", ("foreign-owner",), connection=conn)
            assert backend.execute("SELECT message_id FROM message_insertion_order", connection=conn).rows == []
    finally:
        if created_role:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {role}", connection=conn)
                backend.execute(f"DROP ROLE {role}", connection=conn)


@pytest.mark.integration
def test_postgres_boundary_covers_all_writers_and_timestamps(order_db):
    db = order_db
    for writer in ("ordinary", "sync", "sql"):
        for timestamp_kind in ("natural", "equal", "backdated"):
            cid = db.add_conversation({"title": f"{writer} {timestamp_kind}"})
            prior = db.add_message({"conversation_id": cid, "sender": "assistant", "content": "Future", "timestamp": "2099-01-01T00:00:00Z"})
            uid = str(uuid4())
            context = db.get_conversation_by_id(cid)
            db.insert_or_validate_user_turn(cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context)
            timestamp = {
                "natural": db._get_current_utc_timestamp_iso(),
                "equal": db.get_message_by_id(uid)["timestamp"],
                "backdated": "1999-01-01T00:00:00Z",
            }[timestamp_kind]
            insert_writer(db, writer, cid, str(uuid4()), "assistant", timestamp)
            history = db.insert_or_validate_user_turn(cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context)
            assert [row["id"] for row in history] == [prior, uid]
            insert_writer(db, writer, cid, str(uuid4()), "user", timestamp)
            with pytest.raises(ConflictError, match="later user"):
                db.insert_or_validate_user_turn(cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context)


@pytest.mark.integration
def test_postgres_mixed_concurrent_writers_allocate_after_conversation_lock(order_db):
    db = order_db
    cid = db.add_conversation({"title": "Mixed writers"})
    uid = str(uuid4())
    context = db.get_conversation_by_id(cid)
    db.close_connection()
    peers = [CharactersRAGDB(":memory:", client_id=db.client_id, backend=db.backend) for _ in range(4)]
    barrier = Barrier(4)

    def insert(index):
        peer = peers[index]
        try:
            barrier.wait(timeout=10)
            if index == 0:
                return peer.insert_or_validate_user_turn(
                    cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
                )
            insert_writer(peer, ("ordinary", "sync", "sql")[index - 1], cid, str(uuid4()), "assistant", "1999-01-01T00:00:00Z")
            return None
        finally:
            peer.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(insert, range(4)))
        with db.transaction() as conn:
            ordered = conn.execute(
                "SELECT m.id, o.sequence FROM messages m JOIN message_insertion_order o ON o.message_id = m.id "
                "WHERE m.conversation_id = ? ORDER BY o.sequence", (cid,),
            ).fetchall()
        ids = [row["id"] for row in ordered]
        expected = ids[:ids.index(uid) + 1]
        assert [row["id"] for row in results[0]] == expected
        history = db.insert_or_validate_user_turn(cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context)
        assert [row["id"] for row in history] == expected
        assert len(ids) == 4
        assert len({row["sequence"] for row in ordered}) == 4
    finally:
        for peer in peers:
            peer.close_all_connections()


@pytest.mark.integration
def test_postgres_v78_upgrade_legacy_and_atomic_migration(isolated_test_environment, monkeypatch):
    backend = DatabaseBackendFactory.create_backend(DatabaseConfig.from_env())
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 77)
        legacy = CharactersRAGDB(":memory:", client_id="durable-pg", backend=backend)
        try:
            cid = legacy.add_conversation({"title": "Legacy"})
            mid = legacy.add_message({"conversation_id": cid, "sender": "user", "content": "Legacy", "timestamp": "2099-01-01T00:00:00Z"})
            before = legacy.get_message_by_id(mid)
        finally:
            legacy.close_all_connections()

    backend = DatabaseBackendFactory.create_backend(DatabaseConfig.from_env())

    class FailedMigration(CharactersRAGDB):
        def _migrate_from_v77_to_v78_postgres(self, conn):
            super()._migrate_from_v77_to_v78_postgres(conn)
            raise RuntimeError("injected migration failure")

    with pytest.raises(CharactersRAGDBError, match="injected migration failure"):
        FailedMigration(":memory:", client_id="durable-pg", backend=backend)
    with backend.transaction() as conn:
        assert backend.execute("SELECT version FROM db_schema_version WHERE schema_name = %s", (CharactersRAGDB._SCHEMA_NAME,), connection=conn).rows[0]["version"] == 77
        assert not backend.table_exists("message_insertion_order", connection=conn)
    db = CharactersRAGDB(":memory:", client_id="durable-pg", backend=backend)
    try:
        assert db.get_message_by_id(mid) == before
        with db.transaction() as conn:
            assert conn.execute("SELECT COUNT(*) AS count FROM message_insertion_order").fetchone()["count"] == 0
        context = db.get_conversation_by_id(cid)
        with pytest.raises(ConflictError, match="insertion order"):
            db.insert_or_validate_user_turn(cid, mid, "Legacy", owner_client_id=db.client_id, conversation_context=context)
        uid = str(uuid4())
        history = db.insert_or_validate_user_turn(cid, uid, "New", owner_client_id=db.client_id, conversation_context=context)
        assert [row["id"] for row in history] == [mid, uid]
        with db.transaction() as conn:
            sequence = conn.execute("SELECT sequence FROM message_insertion_order WHERE message_id = ?", (uid,)).fetchone()["sequence"]
        with pytest.raises(DatabaseError):
            with db.transaction() as conn:
                conn.execute("UPDATE message_insertion_order SET sequence = sequence + 100 WHERE message_id = ?", (uid,))
        with db.transaction() as conn:
            assert conn.execute("SELECT sequence FROM message_insertion_order WHERE message_id = ?", (uid,)).fetchone()["sequence"] == sequence
            conn.execute("DELETE FROM messages WHERE id = ?", (uid,))
            assert conn.execute("SELECT COUNT(*) AS count FROM message_insertion_order").fetchone()["count"] == 0
    finally:
        db.close_all_connections()


@pytest.mark.integration
def test_postgres_durable_identity_concurrency_conflicts_and_history(isolated_test_environment):
    """Separate connections serialize matching IDs and leave conflicting rows intact."""
    config = DatabaseConfig.from_env()
    backend = DatabaseBackendFactory.create_backend(config)
    db = CharactersRAGDB(":memory:", client_id="durable-pg", backend=backend)
    peers = []
    try:
        cid = db.add_conversation({"title": "Durable PostgreSQL"})
        legacy_id = db.add_message({"conversation_id": cid, "sender": "user", "content": "Legacy"})
        uid = str(uuid4())
        context = db.get_conversation_by_id(cid)
        db.close_connection()
        peers = [CharactersRAGDB(":memory:", client_id=db.client_id, backend=backend) for _ in range(4)]
        barrier = Barrier(len(peers))

        def insert(peer):
            barrier.wait(timeout=10)
            try:
                return peer.insert_or_validate_user_turn(
                    cid,
                    uid,
                    "Question",
                    owner_client_id=db.client_id,
                    conversation_context=context,
                    history_limit=20,
                )
            finally:
                peer.close_connection()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(insert, peers))
        assert all([row["id"] for row in result] == [legacy_id, uid] for result in results)
        assert db.count_messages_for_conversation(cid) == 2

        partial_id = db.add_message(
            {"conversation_id": cid, "sender": "assistant", "content": "Partial", "parent_message_id": uid}
        )
        before = db.get_messages_for_conversation(cid)
        history = db.insert_or_validate_user_turn(
            cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
        )
        assert [row["id"] for row in history] == [legacy_id, uid]
        with pytest.raises(ConflictError):
            db.insert_or_validate_user_turn(
                cid, uid, "Mismatch", owner_client_id=db.client_id, conversation_context=context
            )
        with pytest.raises(ConflictError):
            db.insert_or_validate_user_turn(
                cid, partial_id, "Partial", owner_client_id=db.client_id, conversation_context=context
            )
        assert db.get_messages_for_conversation(cid) == before
        with pytest.raises(ConflictError):
            db.add_message({"id": uid, "conversation_id": cid, "sender": "user", "content": "Question"})

        next_id = str(uuid4())
        db.insert_or_validate_user_turn(
            cid, next_id, "Question", owner_client_id=db.client_id, conversation_context=context
        )
        with pytest.raises(ConflictError):
            db.insert_or_validate_user_turn(
                cid, uid, "Question", owner_client_id=db.client_id, conversation_context=context
            )
        db.soft_delete_message(next_id, 1)
        with pytest.raises(ConflictError):
            db.insert_or_validate_user_turn(
                cid, next_id, "Question", owner_client_id=db.client_id, conversation_context=context
            )
        foreign_cid = db.add_conversation({"title": "Other"})
        with pytest.raises(ConflictError):
            db.insert_or_validate_user_turn(
                foreign_cid,
                uid,
                "Question",
                owner_client_id=db.client_id,
                conversation_context=db.get_conversation_by_id(foreign_cid),
            )
    finally:
        for peer in peers:
            peer.close_all_connections()
        db.close_all_connections()
