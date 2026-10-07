"""Owner-bound, independently versioned Knowledge provenance persistence."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError

PAYLOAD = {"origin": "knowledge_qa", "trust_state": "cited_answer"}


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "provenance.db", client_id="alice")
    database.add_note("Note", "Body", note_id="own")
    yield database
    database.close_all_connections()


def test_independent_version_and_reopen(db):
    store = db.note_provenance_store
    assert store.get("own") is None
    record = store.put("own", PAYLOAD, expected_version=0)
    assert record["version"] == 1
    with pytest.raises(ConflictError):
        store.put("own", PAYLOAD, expected_version=0)
    assert store.put("own", PAYLOAD, expected_version=1)["version"] == 2
    assert db.get_note_by_id("own")["version"] == 1
    db.close_all_connections()
    reopened = CharactersRAGDB(db.db_path, client_id="alice")
    try:
        assert reopened.note_provenance_store.get("own")["payload"] == PAYLOAD
    finally:
        reopened.close_all_connections()


def test_owner_and_parent_lifecycle(db):
    foreign = CharactersRAGDB(db.db_path, client_id="bob")
    try:
        foreign.add_note("Foreign", "Body", note_id="foreign")
        with pytest.raises(InputError):
            db.note_provenance_store.put("foreign", PAYLOAD, expected_version=0)
        assert db.note_provenance_store.get("foreign", include_deleted=True) is None
        assert foreign.note_provenance_store.get("own", include_deleted=True) is None
    finally:
        foreign.close_all_connections()
    db.note_provenance_store.put("own", PAYLOAD, expected_version=0)
    with db.transaction() as conn:
        conn.execute("UPDATE notes SET deleted = 1 WHERE id = ?", ("own",))
    with pytest.raises(InputError):
        db.note_provenance_store.put("own", PAYLOAD, expected_version=1)
    assert db.note_provenance_store.get("own") is None
    tombstone = db.note_provenance_store.tombstone("own", expected_version=1)
    assert tombstone["deleted"] and tombstone["payload"] == PAYLOAD
    with db.transaction() as conn:
        conn.execute("UPDATE notes SET deleted = 0 WHERE id = ?", ("own",))
    with pytest.raises(ConflictError):
        db.note_provenance_store.put("own", PAYLOAD, expected_version=2)
    assert db.note_provenance_store.restore("own", expected_version=2)["version"] == 3


def test_atomic_rollback(db):
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            db.note_provenance_store.put("own", PAYLOAD, expected_version=0, conn=conn)
            raise RuntimeError("rollback")
    assert db.note_provenance_store.get("own") is None


def test_sync_revision_hash_replay(db):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import notes_provenance_object_hash

    digest = notes_provenance_object_hash(PAYLOAD)
    record = db.note_provenance_store.apply_sync("own", PAYLOAD, 4, digest)
    assert record["version"] == 4
    assert db.note_provenance_store.apply_sync("own", PAYLOAD, 4, digest) == record
    for revision, hash_value in [(3, digest), (4, "sha256:" + "a" * 64), (5, "sha256:" + "b" * 64)]:
        with pytest.raises((InputError, ConflictError)):
            db.note_provenance_store.apply_sync("own", PAYLOAD, revision, hash_value)


def test_upgrade_v74_repeatable(db):
    with db.transaction() as conn:
        conn.execute("DROP TABLE notes_knowledge_provenance")
        conn.execute("UPDATE db_schema_version SET version = 74 WHERE schema_name = ?", (db._SCHEMA_NAME,))
    db.close_all_connections()
    for _ in range(2):
        migrated = CharactersRAGDB(db.db_path, client_id="alice")
        try:
            assert migrated.note_provenance_store.get("own") is None
            assert migrated._get_db_version(migrated.get_connection()) == 76
        finally:
            migrated.close_all_connections()


@pytest.mark.postgres
def test_postgres_migration_and_forced_owner_rls(pg_database_config):
    database = CharactersRAGDB(
        ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    try:
        database.add_note("Note", "Body", note_id="own")
        assert database.note_provenance_store.put("own", PAYLOAD, expected_version=0)["version"] == 1
        with database.transaction() as conn:
            row = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = 'notes_knowledge_provenance'"
            ).fetchone()
            assert row["relrowsecurity"] and row["relforcerowsecurity"]
            policies = conn.execute(
                "SELECT qual, with_check FROM pg_policies WHERE tablename = 'notes_knowledge_provenance'"
            ).fetchall()
            assert len(policies) == 1 and "app.current_user_id" in policies[0]["qual"]
        assert database._POSTGRES_SCHEMA_VERSION == 80
    finally:
        database.close_all_connections()


@pytest.mark.parametrize("method", ["soft_delete_note", "delete_note", "tombstone_note_from_sync"])
def test_note_deletion_tombstones_sidecar_without_restore(db, method):
    db.note_provenance_store.put("own", PAYLOAD, expected_version=0)
    if method == "tombstone_note_from_sync":
        db.tombstone_note_from_sync(
            note_id="own", object_revision=2, sync_client_id="alice", object_hash="sha256:" + "a" * 64
        )
    else:
        getattr(db, method)("own", expected_version=1)
    retained = db.note_provenance_store.get("own", include_deleted=True)
    assert retained["deleted"] and retained["version"] == 2
    db.restore_note("own", expected_version=2)
    assert db.note_provenance_store.get("own") is None
    assert db.note_provenance_store.get("own", include_deleted=True) == retained
    assert db.delete_note("own", hard_delete=True)
    assert db.note_provenance_store.get("own", include_deleted=True) is None


@pytest.mark.postgres
def test_postgres_safe_integer_revision(pg_database_config):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import notes_provenance_object_hash

    database = CharactersRAGDB(
        ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    try:
        database.add_note("Note", "Body", note_id="own")
        record = database.note_provenance_store.apply_sync("own", PAYLOAD, 2**32, notes_provenance_object_hash(PAYLOAD))
        assert record["version"] == 2**32
    finally:
        database.close_all_connections()


@pytest.mark.postgres
def test_restricted_postgres_rls_denies_foreign_and_unbound_access(pg_restricted_backend):
    database = CharactersRAGDB(":memory:", client_id="alice", backend=pg_restricted_backend)
    try:
        database.add_note("Note", "Body", note_id="own")
        record = database.note_provenance_store.put("own", PAYLOAD, expected_version=0)
        with pg_restricted_backend.transaction() as conn:
            for owner in ("bob", ""):
                conn.execute("SELECT set_config('app.current_user_id', %s, true)", (owner,))
                assert conn.execute("SELECT * FROM notes_knowledge_provenance").fetchall() == []
            conn.execute("SELECT set_config('app.current_user_id', %s, true)", ("alice",))
            assert len(conn.execute("SELECT * FROM notes_knowledge_provenance").fetchall()) == 1
        with pytest.raises(Exception, match="row-level security"):
            with pg_restricted_backend.transaction() as conn:
                conn.execute("SELECT set_config('app.current_user_id', %s, true)", ("bob",))
                conn.execute(
                    "INSERT INTO notes_knowledge_provenance(owner_user_id,note_id,payload_json,version,object_hash) VALUES (%s,%s,%s,%s,%s)",
                    ("alice", "own", '{"origin":"knowledge_qa"}', 2, record["object_hash"]),
                )
        database.delete_note("own", expected_version=1)
        assert database.note_provenance_store.get("own", include_deleted=True)["deleted"]
    finally:
        database.close_all_connections()


@pytest.mark.postgres
def test_postgres_v78_upgrade_and_reopen(pg_database_config):
    database = CharactersRAGDB(
        ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    try:
        database.add_note("Note", "Body", note_id="own")
        with database.transaction() as conn:
            conn.execute("DROP TABLE notes_knowledge_provenance")
            conn.execute("UPDATE db_schema_version SET version = 78 WHERE schema_name = ?", (database._SCHEMA_NAME,))
    finally:
        database.close_all_connections()
    for _ in range(2):
        migrated = CharactersRAGDB(
            ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
        )
        try:
            assert migrated._runtime_schema_version == 80
            assert migrated.note_provenance_store.get("own") is None
        finally:
            migrated.close_all_connections()


def test_explicit_restore_requires_exact_retained_payload(db):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import notes_provenance_object_hash

    db.note_provenance_store.put("own", PAYLOAD, expected_version=0)
    db.note_provenance_store.tombstone("own", expected_version=1)
    changed = {"origin": "reviewed_sources"}
    with pytest.raises(ConflictError):
        db.note_provenance_store.put("own", changed, expected_version=2, restore=True)
    with pytest.raises(ConflictError):
        db.note_provenance_store.apply_sync("own", changed, 3, notes_provenance_object_hash(changed), restore=True)
    assert db.note_provenance_store.restore("own", expected_version=2)["payload"] == PAYLOAD


def test_receipt_rollback_reopen_changed_input_and_owner_isolation(db):
    store = db.note_provenance_store
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as conn:
            assert store.claim_receipt("lost", "fingerprint", conn) is None
            store.put("own", PAYLOAD, 0, conn=conn)
            store.complete_receipt("lost", "fingerprint", {"id": "own", "version": 1}, conn)
            raise RuntimeError("rollback")
    assert store.read_receipt("lost", "fingerprint") is None
    assert store.get("own") is None
    with db.transaction() as conn:
        assert store.claim_receipt("lost", "fingerprint", conn) is None
        store.put("own", PAYLOAD, 0, conn=conn)
        store.complete_receipt("lost", "fingerprint", {"id": "own", "version": 1}, conn)
    db.update_note("own", {"content": "Later"}, 1)
    db.close_all_connections()
    reopened = CharactersRAGDB(db.db_path, client_id="alice")
    foreign = CharactersRAGDB(db.db_path, client_id="bob")
    try:
        assert reopened.note_provenance_store.read_receipt("lost", "fingerprint") == {"id": "own", "version": 1}
        with reopened.transaction() as conn:
            assert reopened.note_provenance_store.claim_receipt("lost", "fingerprint", conn) == {
                "id": "own",
                "version": 1,
            }
        with pytest.raises(ConflictError):
            reopened.note_provenance_store.read_receipt("lost", "changed")
        assert foreign.note_provenance_store.read_receipt("lost", "fingerprint") is None
        with foreign.transaction() as conn:
            assert foreign.note_provenance_store.claim_receipt("lost", "different", conn) is None
            foreign.note_provenance_store.complete_receipt("lost", "different", {"id": "foreign"}, conn)
        assert reopened.note_provenance_store.read_receipt("lost", "fingerprint")["id"] == "own"
    finally:
        reopened.close_all_connections()
        foreign.close_all_connections()


def test_upgrade_v75_receipts_repeatable(db):
    with db.transaction() as conn:
        conn.execute("DROP TABLE notes_provenance_receipts")
        conn.execute("UPDATE db_schema_version SET version = 75 WHERE schema_name = ?", (db._SCHEMA_NAME,))
    db.close_all_connections()
    for _ in range(2):
        migrated = CharactersRAGDB(db.db_path, client_id="alice")
        try:
            assert migrated._get_db_version(migrated.get_connection()) == 76
            assert migrated.note_provenance_store.read_receipt("missing", "fp") is None
        finally:
            migrated.close_all_connections()


@pytest.mark.postgres
def test_postgres_receipt_upgrade_reopen_and_rls(pg_restricted_backend):
    database = CharactersRAGDB(":memory:", client_id="alice", backend=pg_restricted_backend)
    with database.transaction() as conn:
        conn.execute("DROP TABLE notes_provenance_receipts")
        conn.execute("UPDATE db_schema_version SET version = 79 WHERE schema_name = ?", (database._SCHEMA_NAME,))
    database = CharactersRAGDB(":memory:", client_id="alice", backend=pg_restricted_backend)
    assert database._runtime_schema_version == 80
    store = database.note_provenance_store
    with database.transaction() as conn:
        store.claim_receipt("same", "fingerprint", conn)
        store.complete_receipt("same", "fingerprint", {"id": "own"}, conn)
        flags = conn.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = 'notes_provenance_receipts'"
        ).fetchone()
        assert flags["relrowsecurity"] and flags["relforcerowsecurity"]
    reopened = CharactersRAGDB(":memory:", client_id="alice", backend=pg_restricted_backend)
    assert reopened.note_provenance_store.read_receipt("same", "fingerprint") == {"id": "own"}
    with pg_restricted_backend.transaction() as conn:
        for owner in ("bob", ""):
            conn.execute("SELECT set_config('app.current_user_id', %s, true)", (owner,))
            assert conn.execute("SELECT * FROM notes_provenance_receipts").fetchall() == []
        conn.execute("SELECT set_config('app.current_user_id', %s, true)", ("alice",))
        assert len(conn.execute("SELECT * FROM notes_provenance_receipts").fetchall()) == 1
    with pytest.raises(Exception, match="row-level security"):
        with pg_restricted_backend.transaction() as conn:
            conn.execute("SELECT set_config('app.current_user_id', %s, true)", ("bob",))
            conn.execute(
                "INSERT INTO notes_provenance_receipts(owner_user_id,request_key,request_fingerprint) VALUES (%s,%s,%s)",
                ("alice", "foreign", "fp"),
            )


@pytest.mark.postgres
@pytest.mark.parametrize("read_method", ["get", "read_receipt", "list_parent_notes"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_postgres_provenance_reads_preserve_caller_pending_writes(pg_database_config, read_method, commit):
    """Pure history reads retain the caller's pending transaction and completion choice."""
    database = CharactersRAGDB(
        ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    try:
        database.add_note("Note", "Body", note_id="own")
        store = database.note_provenance_store
        store.put("own", PAYLOAD, expected_version=0)
        with database.transaction() as conn:
            store.claim_receipt("request", "fingerprint", conn)
            store.complete_receipt("request", "fingerprint", {"id": "own"}, conn)
        raw = database._get_thread_connection()
        database.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", "own"))

        if read_method == "get":
            assert store.get("own")["payload"] == PAYLOAD
        elif read_method == "read_receipt":
            assert store.read_receipt("request", "fingerprint") == {"id": "own"}
        else:
            assert store.list_parent_notes()[0]["title"] == "Pending title"

        assert raw.info.transaction_status.name == "INTRANS"
        assert database.backend.execute("SELECT title FROM notes WHERE id=%s", ("own",)).scalar == "Note"
        if commit:
            raw.commit()
        else:
            raw.rollback()
        expected = "Pending title" if commit else "Note"
        assert database.backend.execute("SELECT title FROM notes WHERE id=%s", ("own",)).scalar == expected
    finally:
        database.close_all_connections()
