"""Reconcile the relevant released catalog shapes without reconstructing history."""

from types import SimpleNamespace
from unittest.mock import Mock

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


def _published_union_migration(db, connection):
    """Reproduce the published 79/83 catalog, which had no quota owner column."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_notes_provenance_rls_sql
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import notes_provenance_schema_sql

    postgres = db.backend_type.value == "postgresql"
    _parent_order_migration(db, connection)
    connection.execute(notes_provenance_schema_sql(postgres=postgres))
    if postgres:
        for statement in build_notes_provenance_rls_sql():
            connection.execute(statement)


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


@pytest.fixture(params=["parent", "incoming", "published_union", "dev"])
def released_catalog(db_factory, monkeypatch, request):
    """Use the owning isolated factory for each published lineage and union head."""
    catalog = request.param
    with monkeypatch.context() as historical:
        historical.setattr(
            CharactersRAGDB, "_CURRENT_SCHEMA_VERSION",
            {"parent": 77, "incoming": 78, "published_union": 79, "dev": 85}[catalog],
        )
        historical.setattr(
            CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION",
            {"parent": 81, "incoming": 82, "published_union": 83, "dev": 91}[catalog],
        )
        if catalog == "parent":
            for method in (
                "_migrate_from_v74_to_v75",
                "_migrate_from_v76_to_v77",
                "_migrate_from_v78_to_v79_postgres",
                "_migrate_from_v80_to_v81_postgres",
            ):
                historical.setattr(CharactersRAGDB, method, _parent_order_migration)
        elif catalog in ("incoming", "dev"):
            historical.setattr(CharactersRAGDB, "_migrate_from_v74_to_v75", _incoming_fingerprint_migration)
            historical.setattr(CharactersRAGDB, "_migrate_from_v78_to_v79_postgres", _incoming_fingerprint_migration)
        else:
            historical.setattr(CharactersRAGDB, "_migrate_from_v78_to_v79", _published_union_migration)
            historical.setattr(CharactersRAGDB, "_migrate_from_v82_to_v83_postgres", _published_union_migration)
        database = db_factory()
    if catalog in ("incoming", "dev"):
        assert not database.backend.table_exists("message_insertion_order")
        assert database.backend.table_exists("workspace_chat_startup_receipts")
    conversation = database.add_conversation({"title": "Retained union"})
    message = database.add_message({"conversation_id": conversation, "sender": "user", "content": "Retained question"})
    database.add_note("Retained note", "Original body", note_id="catalog-note")
    with database.transaction() as connection:
        quota_columns = database.backend.get_table_info("native_chat_quota_intents", connection=connection)
        assert ("quota_owner_json" in {row["name"] for row in quota_columns}) == (catalog == "dev")
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
        assert version == (92 if postgres else 86)
        for table, column in (
            ("native_chat_quota_intents", "quota_owner_json"),
            ("native_chat_asset_references", "role"),
            ("conversations", "source_incarnation_id"),
        ):
            assert column in {row["name"] for row in database.backend.get_table_info(table, connection=connection)}
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


def _assert_preserved_rows(database, before):
    after = _catalog_rows(database)
    for table, rows in before.items():
        assert len(after[table]) == len(rows or [])
        for original, upgraded in zip(rows or [], after[table], strict=True):
            assert {column: upgraded[column] for column in original} == original


def test_released_lineage_upgrade_and_reopen_preserve_durable_catalogs(db_factory, released_catalog):
    """Missing order/provenance must be installed without reconstructing old data."""
    catalog, old, conversation, message = released_catalog
    before = _catalog_rows(old)
    old.close_all_connections()
    upgraded = db_factory()
    _assert_union_catalog(upgraded)
    _assert_preserved_rows(upgraded, before)
    added = upgraded.add_message({"conversation_id": conversation, "sender": "assistant", "content": "Union answer"})
    with upgraded.transaction() as connection:
        rows = connection.execute("SELECT message_id FROM message_insertion_order ORDER BY sequence").fetchall()
        assert [row["message_id"] for row in rows] == ([added] if catalog in ("incoming", "dev") else [message, added])
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


@pytest.mark.parametrize("released_catalog", ["incoming", "dev"], indirect=True)
def test_union_migration_failure_rolls_back_catalog_and_reopens_old_lineage(db_factory, released_catalog, monkeypatch):
    """Failure after union DDL/version update must retain the released source catalog."""
    catalog, old, _, _ = released_catalog
    before = _catalog_rows(old)
    postgres = old.backend_type.value == "postgresql"
    method = "_migrate_from_v91_to_v92_postgres" if postgres else "_migrate_from_v85_to_v86"
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
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 85 if catalog == "dev" else 78)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 91 if catalog == "dev" else 82)
        rolled_back = db_factory()
    assert _catalog_rows(rolled_back) == before
    rolled_back.close_all_connections()
    repaired = db_factory()
    _assert_union_catalog(repaired)
    _assert_preserved_rows(repaired, before)


@pytest.mark.parametrize("db_factory", ["sqlite"], indirect=True)
@pytest.mark.parametrize("initial_catalog", ["fresh", "incoming78"])
def test_sqlite_bootstrap_and_upgrade_dispatch_final_union(db_factory, monkeypatch, initial_catalog):
    """Fresh and existing databases must reach the final union migration."""
    if initial_catalog == "incoming78":
        with monkeypatch.context() as incoming:
            incoming.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 78)
            incoming.setattr(CharactersRAGDB, "_migrate_from_v74_to_v75", _incoming_fingerprint_migration)
            old = db_factory()
        assert not old.backend.table_exists("message_insertion_order")
        old.close_all_connections()

    _assert_union_catalog(db_factory())


@pytest.mark.parametrize("dev_catalog", [False, True], ids=["old-union", "dev-catalog"])
def test_pg_forward_migration_installs_only_the_missing_authority(dev_catalog, monkeypatch):
    backend = Mock()
    backend.table_exists.side_effect = lambda table, **_: table != "message_insertion_order" or not dev_catalog
    backend.get_table_info.side_effect = lambda table, **_: (
        [{"name": "create_request_fingerprint"}] if table == "conversations"
        else [{"name": "quota_owner_json"}] if dev_catalog
        else []
    )
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = SimpleNamespace()
    db._backend = backend
    db._uses_shared_content_backend = False
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _: 92)
    connection = object()

    db._migrate_from_v91_to_v92_postgres(connection)

    calls = backend.execute.call_args_list
    assert all(call.kwargs["connection"] is connection for call in calls)
    ddl = "\n".join(call.args[0].strip() for call in calls[:-1])
    assert ("ADD COLUMN quota_owner_json" in ddl) is not dev_catalog
    assert ("CREATE TABLE message_insertion_order" in ddl) is dev_catalog
    if dev_catalog:
        assert "FORCE ROW LEVEL SECURITY" in ddl
        assert "CREATE POLICY message_insertion_order_tenant_isolation" in ddl
        assert "WITH CHECK" in ddl
        assert "CREATE TRIGGER messages_lock_insertion_order" in ddl
        assert "CREATE TRIGGER messages_record_insertion_order" in ddl
    assert "GRANT " not in ddl
    assert calls[-1].args == (
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 91",
        (92, db._SCHEMA_NAME),
    )


@pytest.mark.parametrize("allowed", [False, True])
def test_pg_runtime_requires_order_ledger_privileges(allowed):
    backend = Mock()
    backend.execute.return_value.scalar = allowed
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = SimpleNamespace()
    db._backend = backend
    db._uses_shared_content_backend = False
    connection = object()

    if allowed:
        db._verify_message_order_privileges_postgres(connection)
    else:
        with pytest.raises(CharactersRAGDBError, match="schema owner.*message insertion order"):
            db._verify_message_order_privileges_postgres(connection)

    backend.execute.assert_called_once()
    sql, = backend.execute.call_args.args
    assert "has_any_column_privilege(current_user, 'messages', 'INSERT')" in sql
    assert "has_column_privilege(current_user, 'messages', 'id', 'SELECT')" in sql
    assert "has_column_privilege(current_user, 'messages', 'client_id', 'SELECT')" in sql
    assert "has_column_privilege(current_user, 'message_insertion_order', 'message_id', 'INSERT')" in sql
    assert "pg_get_serial_sequence('message_insertion_order', 'sequence')" in sql
    assert backend.execute.call_args.kwargs["connection"] is connection
