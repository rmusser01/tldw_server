"""Reconcile the relevant released catalog shapes without reconstructing history."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration


def _incoming_fingerprint_migration(db, connection):
    """Freeze the released incoming step; the union now owns a different catalog."""
    postgres = db.backend_type.value == "postgresql"
    sql = db._MIGRATION_SQL_V78_TO_V79_POSTGRES if postgres else db._MIGRATION_SQL_V74_TO_V75
    for statement in split_sql_statements(sql):
        if postgres:
            db.backend.execute(statement, connection=connection)
        else:
            connection.execute(statement)
    version = db._get_schema_version_postgres(connection) if postgres else db._get_db_version(connection)
    if version != (79 if postgres else 75):
        raise CharactersRAGDBError("Frozen incoming fingerprint migration failed version verification.")


def _parent_order_migration(db, connection):
    """Install the released parent catalog, not incoming provenance at 77/81."""
    postgres = db.backend_type.value == "postgresql"
    if postgres:
        version = db._get_schema_version_postgres(connection)
        db._ensure_message_order_and_create_fingerprint_postgres(connection)
        db._set_schema_version_postgres(connection, version + 1)
    else:
        version = db._get_db_version(connection)
        db._ensure_message_order_and_create_fingerprint_sqlite(connection)
        connection.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ? AND version = ?",
            (version + 1, db._SCHEMA_NAME, version),
        )


@pytest.mark.parametrize("catalog", ["pr3071", "incoming"])
def test_published_catalog_shapes_reconcile_without_changing_existing_messages(db_factory, monkeypatch, catalog):
    # The fixture supplies the real isolated backend; remove only the features
    # that the other released migration catalog had not installed.
    with monkeypatch.context() as previous:
        previous.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 75 if catalog == "pr3071" else 76)
        previous.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 79 if catalog == "pr3071" else 80)
        if catalog == "pr3071":
            previous.setattr(CharactersRAGDB, "_migrate_from_v74_to_v75", _parent_order_migration)
            previous.setattr(CharactersRAGDB, "_migrate_from_v78_to_v79_postgres", _parent_order_migration)
        old = db_factory()
    postgres = old.backend_type.value == "postgresql"
    conversation = old.add_conversation({"title": "Preserved catalog"})
    message = old.add_message({"conversation_id": conversation, "sender": "user", "content": "Preserved question"})
    before = old.get_message_by_id(message)
    with old.transaction() as connection:
        if catalog == "pr3071":
            old.backend.execute(
                "ALTER TABLE conversations DROP COLUMN create_request_fingerprint", connection=connection
            )
        else:
            statements = (
                (
                    "DROP TRIGGER IF EXISTS messages_record_insertion_order ON messages",
                    "DROP TRIGGER IF EXISTS messages_lock_insertion_order ON messages",
                )
                if postgres
                else (
                    "DROP TRIGGER IF EXISTS messages_record_insertion_order",
                    "DROP TRIGGER IF EXISTS messages_lock_insertion_order",
                )
            )
            for sql in statements:
                old.backend.execute(sql, connection=connection)
            old.backend.execute("DROP TABLE IF EXISTS message_insertion_order", connection=connection)
            if postgres:
                for sql in (
                    "DROP FUNCTION IF EXISTS messages_record_insertion_order()",
                    "DROP FUNCTION IF EXISTS messages_lock_insertion_order()",
                    "DROP FUNCTION IF EXISTS message_insertion_order_immutable()",
                ):
                    old.backend.execute(sql, connection=connection)
    old.close_all_connections()

    upgraded = db_factory()
    with upgraded.transaction() as connection:
        columns = {row["name"] for row in upgraded.backend.get_table_info("conversations", connection=connection)}
        assert "create_request_fingerprint" in columns
        assert upgraded.backend.table_exists("workspace_chat_startup_receipts", connection=connection)
        assert upgraded.backend.table_exists("message_insertion_order", connection=connection)
        count = upgraded.backend.execute(
            "SELECT COUNT(*) AS total FROM message_insertion_order", connection=connection
        ).rows[0]["total"]
        assert count == (1 if catalog == "pr3071" else 0)
    assert upgraded.get_message_by_id(message) == before
    added = upgraded.add_message({"conversation_id": conversation, "sender": "assistant", "content": "New answer"})
    with upgraded.transaction() as connection:
        rows = upgraded.backend.execute(
            "SELECT message_id FROM message_insertion_order ORDER BY sequence", connection=connection
        ).rows
        assert [row["message_id"] for row in rows] == ([message, added] if catalog == "pr3071" else [added])
    upgraded.close_all_connections()
    reopened = db_factory()
    assert reopened.get_message_by_id(message) == before
    assert reopened.get_message_by_id(added)["content"] == "New answer"


@pytest.fixture(params=["parent", "incoming", "current"])
def released_catalog(db_factory, monkeypatch, request):
    """Use the owning isolated factory for each published lineage and union head."""
    catalog = request.param
    with monkeypatch.context() as historical:
        historical.setattr(
            CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", {"parent": 77, "incoming": 78, "current": 79}[catalog]
        )
        historical.setattr(
            CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", {"parent": 81, "incoming": 82, "current": 83}[catalog]
        )
        if catalog == "parent":
            for method in (
                "_migrate_from_v74_to_v75",
                "_migrate_from_v76_to_v77",
                "_migrate_from_v78_to_v79_postgres",
                "_migrate_from_v80_to_v81_postgres",
            ):
                historical.setattr(CharactersRAGDB, method, _parent_order_migration)
        elif catalog == "incoming":
            historical.setattr(CharactersRAGDB, "_migrate_from_v74_to_v75", _incoming_fingerprint_migration)
            historical.setattr(CharactersRAGDB, "_migrate_from_v78_to_v79_postgres", _incoming_fingerprint_migration)
        database = db_factory()
    if catalog == "incoming":
        assert not database.backend.table_exists("message_insertion_order")
        assert database.backend.table_exists("workspace_chat_startup_receipts")
    conversation = database.add_conversation({"title": "Retained union"})
    message = database.add_message({"conversation_id": conversation, "sender": "user", "content": "Retained question"})
    database.add_note("Retained note", "Original body", note_id="catalog-note")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE conversations SET create_request_fingerprint = ? WHERE id = ?", ("a" * 64, conversation)
        )
        connection.execute(
            "INSERT INTO workspace_chat_startup_receipts "
            "(owner_user_id,key_digest,request_fingerprint,binding_digest,workspace_id,conversation_id) "
            "VALUES (?,?,?,?,?,?)",
            (database.owner_user_id, "b" * 64, "c" * 64, "d" * 64, "retained-workspace", conversation),
        )
    if catalog != "parent":
        store = database.note_provenance_store
        store.put("catalog-note", {"origin": "knowledge_qa", "trust_state": "cited_answer"}, expected_version=0)
        with database.transaction() as connection:
            store.claim_receipt("retained-request", "e" * 64, connection)
            store.complete_receipt("retained-request", "e" * 64, {"accepted_version": 1}, connection)
    return catalog, database, conversation, message


def _catalog_rows(database):
    """Capture durable values, including sequence numbers and immutable receipts."""
    queries = (
        ("conversations", "SELECT * FROM conversations ORDER BY 1"),
        ("messages", "SELECT * FROM messages ORDER BY 1"),
        ("notes", "SELECT * FROM notes ORDER BY 1"),
        ("message_insertion_order", "SELECT * FROM message_insertion_order ORDER BY 1"),
        ("workspace_chat_startup_receipts", "SELECT * FROM workspace_chat_startup_receipts ORDER BY 1"),
        ("notes_knowledge_provenance", "SELECT * FROM notes_knowledge_provenance ORDER BY 1"),
        ("notes_provenance_receipts", "SELECT * FROM notes_provenance_receipts ORDER BY 1"),
    )
    with database.transaction() as connection:
        rows = {}
        for table, query in queries:
            if database.backend.table_exists(table, connection=connection):
                rows[table] = [dict(row) for row in connection.execute(query).fetchall()]
            else:
                rows[table] = None
        return rows


def _assert_union_catalog(database):
    """The union owns all three catalogs and keeps PostgreSQL owner RLS forced."""
    with database.transaction() as connection:
        postgres = database.backend_type.value == "postgresql"
        version = (
            database._get_schema_version_postgres(connection) if postgres else database._get_db_version(connection)
        )
        assert version == (83 if postgres else 79)
        for table in ("message_insertion_order", "notes_knowledge_provenance", "notes_provenance_receipts"):
            assert database.backend.table_exists(table, connection=connection)
        if postgres:
            rows = connection.execute(
                "SELECT relname,relrowsecurity,relforcerowsecurity FROM pg_class "
                "WHERE relname IN ('message_insertion_order','notes_knowledge_provenance','notes_provenance_receipts')"
            ).fetchall()
            assert len(rows) == 3
            assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in rows)
            policies = connection.execute(
                "SELECT tablename,qual,with_check FROM pg_policies "
                "WHERE tablename IN ('notes_knowledge_provenance','notes_provenance_receipts')"
            ).fetchall()
            assert {row["tablename"] for row in policies} == {"notes_knowledge_provenance", "notes_provenance_receipts"}
            assert all(
                "app.current_user_id" in row["qual"] and "app.current_user_id" in row["with_check"] for row in policies
            )


def test_released_lineage_upgrade_and_reopen_preserve_durable_catalogs(db_factory, released_catalog):
    """Missing order/provenance must be installed without reconstructing old data."""
    catalog, old, conversation, message = released_catalog
    before = _catalog_rows(old)
    old.close_all_connections()
    upgraded = db_factory()
    _assert_union_catalog(upgraded)
    expected = {table: rows if rows is not None else [] for table, rows in before.items()}
    assert _catalog_rows(upgraded) == expected
    added = upgraded.add_message({"conversation_id": conversation, "sender": "assistant", "content": "Union answer"})
    with upgraded.transaction() as connection:
        rows = connection.execute("SELECT message_id FROM message_insertion_order ORDER BY sequence").fetchall()
        assert [row["message_id"] for row in rows] == ([added] if catalog == "incoming" else [message, added])
    accepted = _catalog_rows(upgraded)
    with pytest.raises(RuntimeError, match="abort union write"):
        with upgraded.transaction() as connection:
            connection.execute(
                "UPDATE conversations SET create_request_fingerprint = ? WHERE id = ?", ("f" * 64, conversation)
            )
            upgraded.add_message(
                {"conversation_id": conversation, "sender": "assistant", "content": "Rolled back"}, conn=connection
            )
            store = upgraded.note_provenance_store
            store.put(
                "catalog-note",
                {"origin": "knowledge_qa", "trust_state": "cited_answer"},
                expected_version=0 if catalog == "parent" else 1,
                conn=connection,
            )
            store.claim_receipt("aborted-request", "f" * 64, connection)
            store.complete_receipt("aborted-request", "f" * 64, {"accepted_version": 2}, connection)
            connection.execute(
                "UPDATE workspace_chat_startup_receipts SET request_fingerprint = ? WHERE owner_user_id = ?",
                ("f" * 64, upgraded.owner_user_id),
            )
            raise RuntimeError("abort union write")
    assert _catalog_rows(upgraded) == accepted
    upgraded.close_all_connections()
    reopened = db_factory()
    _assert_union_catalog(reopened)
    assert _catalog_rows(reopened) == accepted


@pytest.mark.parametrize("released_catalog", ["parent", "incoming"], indirect=True)
def test_union_migration_failure_rolls_back_catalog_and_reopens_old_lineage(db_factory, released_catalog, monkeypatch):
    """Failure after union DDL/version update must retain the released source catalog."""
    catalog, old, _, _ = released_catalog
    before = _catalog_rows(old)
    postgres = old.backend_type.value == "postgresql"
    method = "_migrate_from_v82_to_v83_postgres" if postgres else "_migrate_from_v78_to_v79"
    migrate = getattr(CharactersRAGDB, method)
    old.close_all_connections()

    def fail_after_union(db, connection):
        migrate(db, connection)
        raise CharactersRAGDBError("injected union migration failure")

    with monkeypatch.context() as failure:
        failure.setattr(CharactersRAGDB, method, fail_after_union)
        with pytest.raises(CharactersRAGDBError, match="injected union migration failure"):
            db_factory()
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 77 if catalog == "parent" else 78)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 81 if catalog == "parent" else 82)
        rolled_back = db_factory()
    assert _catalog_rows(rolled_back) == before
    rolled_back.close_all_connections()
    repaired = db_factory()
    _assert_union_catalog(repaired)
    assert _catalog_rows(repaired) == {table: rows if rows is not None else [] for table, rows in before.items()}


@pytest.mark.parametrize("db_factory", ["sqlite"], indirect=True)
@pytest.mark.parametrize("initial_catalog", ["fresh", "incoming78"])
def test_both_sqlite_legacy_dispatchers_install_union(db_factory, monkeypatch, initial_catalog):
    """Both historical initializer branches must dispatch the new union migration."""
    if initial_catalog == "incoming78":
        with monkeypatch.context() as incoming:
            incoming.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 78)
            incoming.setattr(CharactersRAGDB, "_migrate_from_v74_to_v75", _incoming_fingerprint_migration)
            old = db_factory()
        assert not old.backend.table_exists("message_insertion_order")
        old.close_all_connections()

    def legacy_initializer(database):
        database._initialize_schema_sqlite_legacy(target_version=79)

    with monkeypatch.context() as legacy:
        legacy.setattr(CharactersRAGDB, "_initialize_schema_sqlite", legacy_initializer)
        database = db_factory()
    _assert_union_catalog(database)
