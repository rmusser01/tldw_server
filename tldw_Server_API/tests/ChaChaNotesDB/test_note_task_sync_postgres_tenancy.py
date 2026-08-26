"""Live PostgreSQL tenancy proof for the schema-v60 Notes task graph."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    SchemaError,
)

pytestmark = pytest.mark.integration


def _postgres_session_dataset_scope(db: CharactersRAGDB) -> str | None:
    """Read and end the current connection's session-level dataset setting."""
    conn = db.get_connection()
    row = conn.execute(
        "SELECT current_setting('app.current_dataset_id', true) AS dataset_id"
    ).fetchone()
    conn.rollback()
    return row["dataset_id"] if row else None


def test_postgres_task_operations_do_not_leak_dataset_scope_to_the_session(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950000"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        note_id = db.add_note("Transaction-local task scope", "Body")
        task = db.create_task(
            owner_user_id=owner,
            dataset_id="local-unbound",
            note_id=note_id,
            text="Do not leak RLS scope",
        )
        assert _postgres_session_dataset_scope(db) in (None, "")

        assert db.get_task(
            owner_user_id=owner,
            dataset_id="local-unbound",
            task_id=str(task["id"]),
        )["id"] == task["id"]
        assert _postgres_session_dataset_scope(db) in (None, "")
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _restore_reviewed_postgres_v59_task_source(db: CharactersRAGDB) -> None:
    """Replace the fresh v61 graph with the exact reviewed empty v59 source."""
    with db.transaction() as conn:
        for table, constraint in (
            ("moodboard_notes", "moodboard_notes_v61_note_fk"),
            ("note_studio_documents", "note_studio_documents_v61_note_fk"),
        ):
            db.backend.execute(
                f"ALTER TABLE {table} DROP CONSTRAINT {constraint}",  # nosec B608
                connection=conn,
            )
        for table in (
            "note_task_scope_authority",
            "task_projection_drifts",
            "task_event_read_state",
            "task_note_projections",
            "note_task_reconciliation_state",
            "task_events",
            "note_tasks",
            "chacha_schema_migration_progress",
        ):
            db.backend.execute(f"DROP TABLE {table}", connection=conn)  # nosec B608
        db.backend.execute("DROP INDEX uq_notes_owner_id", connection=conn)
        statements = db._convert_sqlite_schema_to_postgres_statements(
            db._MIGRATION_SQL_V47_TO_V48_POSTGRES
        )
        for statement in statements:
            if not statement.lstrip().upper().startswith("UPDATE DB_SCHEMA_VERSION"):
                db.backend.execute(statement, connection=conn)
        db._set_schema_version_postgres(conn, 59)


@pytest.mark.parametrize(
    "drift_sql",
    (
        "ALTER TABLE notes NO FORCE ROW LEVEL SECURITY",
        "ALTER TABLE note_tasks ENABLE ROW LEVEL SECURITY",
        "DROP POLICY notes_tenant_isolation ON notes",
        "CREATE POLICY note_tasks_open ON note_tasks USING (true) WITH CHECK (true)",
        "CREATE POLICY notes_open ON notes USING (true) WITH CHECK (true)",
    ),
)
def test_postgres_v59_source_authority_drift_fails_before_migration_ddl(
    pg_database_config: DatabaseConfig,
    drift_sql: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="950000", backend=backend)

    try:
        _restore_reviewed_postgres_v59_task_source(db)
        with db.transaction() as conn:
            backend.execute(drift_sql, connection=conn)

        with pytest.raises(SchemaError, match="v59 PostgreSQL source authority"):
            with db.transaction() as conn:
                db._migrate_from_v59_to_v60_postgres(conn)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _create_complete_postgres_local_task_graph(
    db: CharactersRAGDB,
) -> tuple[str, str, dict[str, int]]:
    owner = str(db.client_id)
    note_id = str(uuid4())
    task_id = str(uuid4())
    db.add_note("Bound task", "- [ ] Bind this task\n", note_id=note_id)
    task = db.create_task(
        owner_user_id=owner,
        dataset_id="local-unbound",
        task_id=task_id,
        note_id=note_id,
        text="Bind this task",
        actor_type="user",
    )
    db.set_task_projection(
        owner_user_id=owner,
        dataset_id="local-unbound",
        task_id=task_id,
        note_id=note_id,
        note_version=1,
        line_number=1,
        start_offset=0,
        end_offset=20,
        normalized_text_hash="sha256:bind",
        occurrence_index=0,
        block_fingerprint="bind-block",
        raw_line="- [ ] Bind this task",
        has_child_content=False,
    )
    event = db.record_task_event(
        owner_user_id=owner,
        dataset_id="local-unbound",
        task_id=task_id,
        note_id=note_id,
        event_type="updated",
        actor_type="user",
    )
    db.mark_task_activity_read(
        owner_user_id=owner,
        dataset_id="local-unbound",
        event_id=event["id"],
        user_id=owner,
    )
    db.set_reconciliation_state(
        owner_user_id=owner,
        dataset_id="local-unbound",
        note_id=note_id,
        note_version=1,
        status="clean",
        item_count=1,
        warning_count=0,
    )
    with db.transaction() as conn:
        conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", ("local-unbound",))
        conn.execute(
            """
            INSERT INTO task_projection_drifts(
                owner_user_id,dataset_id,id,note_id,task_id,marker_base_revision,
                marker_base_hash,reason_code,status,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                owner, "local-unbound", str(uuid4()), note_id, task_id, 1,
                task["canonical_hash"], "both_changed", "open",
                task["created_at"], task["updated_at"],
            ),
        )
    return note_id, task_id, {
        "note_tasks": 1,
        "task_note_projections": 1,
        "task_events": 2,
        "task_event_read_state": 1,
        "note_task_reconciliation_state": 1,
        "task_projection_drifts": 1,
    }


def _seed_occupied_target_without_authority(
    db: CharactersRAGDB,
    *,
    owner: str,
    target: str,
    note_id: str,
) -> None:
    """Create a deliberate split-scope fixture outside the guarded product path."""
    db.execute_query(
        "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
        (owner, target),
    )
    try:
        db.create_task(
            owner_user_id=owner,
            dataset_id=target,
            task_id=str(uuid4()),
            note_id=note_id,
            text="Occupied target",
            actor_type=None,
        )
    finally:
        db.execute_query(
            "DELETE FROM note_task_scope_authority WHERE owner_user_id=?",
            (owner,),
        )


def _postgres_task_force_flags(conn: Any) -> dict[str, bool]:
    rows = conn.execute(
        "SELECT relname,relforcerowsecurity FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=current_schema() AND c.relname=ANY(?)",
        (list(CharactersRAGDB._NOTE_TASK_V60_RELATIONS),),
    ).fetchall()
    return {str(row["relname"]): bool(row["relforcerowsecurity"]) for row in rows}


def _postgres_rls_authority_snapshot(
    backend: Any,
    *,
    table: str,
) -> tuple[tuple[object, ...], ...]:
    with backend.transaction() as conn:
        relation = backend.execute(
            "SELECT relrowsecurity,relforcerowsecurity,relowner=current_user::regrole "
            "AS is_table_owner,pg_has_role(current_user,n.nspowner,'USAGE') "
            "AS is_schema_owner FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=?",
            (table,),
            connection=conn,
        ).rows
        policies = backend.execute(
            "SELECT policyname,permissive,roles::text AS roles,cmd,qual,with_check "
            "FROM pg_policies WHERE schemaname=current_schema() AND tablename=? "
            "ORDER BY policyname",
            (table,),
            connection=conn,
        ).rows
    return (
        tuple(relation[0].values()),
        *(tuple(row.values()) for row in policies),
    )


def test_postgres_bind_local_task_graph_rekeys_complete_six_table_scope(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950020"
    target = f"dataset-{uuid4()}"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        _note_id, task_id, expected_counts = _create_complete_postgres_local_task_graph(db)

        assert db.bind_local_task_graph_to_dataset(
            owner_user_id=owner, target_dataset_id=target
        ) == expected_counts
        assert db.get_task(
            owner_user_id=owner, dataset_id="local-unbound", task_id=task_id
        ) is None
        assert db.get_task(
            owner_user_id=owner, dataset_id=target, task_id=task_id
        )["id"] == task_id
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_bind_local_task_graph_rejects_cross_owner_and_occupied_target(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950021"
    target = f"dataset-{uuid4()}"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        note_id, task_id, _counts = _create_complete_postgres_local_task_graph(db)
        with pytest.raises(ConflictError, match="authenticated PostgreSQL client"):
            db.bind_local_task_graph_to_dataset(
                owner_user_id="950022", target_dataset_id=target
            )
        _seed_occupied_target_without_authority(
            db, owner=owner, target=target, note_id=note_id
        )
        with pytest.raises(ConflictError, match="target collision"):
            db.bind_local_task_graph_to_dataset(
                owner_user_id=owner, target_dataset_id=target
            )
        assert db.get_task(
            owner_user_id=owner, dataset_id="local-unbound", task_id=task_id
        )["id"] == task_id
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_bind_local_task_graph_rejects_inconsistent_parent_scope(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950023"
    target = f"dataset-{uuid4()}"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        _note_id, task_id, _counts = _create_complete_postgres_local_task_graph(db)
        other_note_id = str(uuid4())
        db.add_note("Wrong parent", "Body", note_id=other_note_id)
        with db.transaction() as conn:
            conn.execute("ALTER TABLE task_note_projections NO FORCE ROW LEVEL SECURITY")
            conn.execute(
                "UPDATE task_note_projections SET note_id = ? "
                "WHERE owner_user_id = ? AND dataset_id = ? AND task_id = ?",
                (other_note_id, owner, "local-unbound", task_id),
            )
            conn.execute("ALTER TABLE task_note_projections FORCE ROW LEVEL SECURITY")

        with pytest.raises(ConflictError, match="parent proof"):
            db.bind_local_task_graph_to_dataset(
                owner_user_id=owner, target_dataset_id=target
            )
        assert db.get_task(
            owner_user_id=owner, dataset_id="local-unbound", task_id=task_id
        )["id"] == task_id
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_bind_caught_collision_restores_force_in_caller_transaction(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950024"
    target = f"dataset-{uuid4()}"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        note_id, task_id, _counts = _create_complete_postgres_local_task_graph(db)
        _seed_occupied_target_without_authority(
            db, owner=owner, target=target, note_id=note_id
        )
        with db.transaction() as conn:
            with pytest.raises(ConflictError, match="target collision"):
                db.bind_local_task_graph_to_dataset(
                    owner_user_id=owner,
                    target_dataset_id=target,
                    conn=conn,
                )
            assert _postgres_task_force_flags(conn) == dict.fromkeys(
                CharactersRAGDB._NOTE_TASK_V60_RELATIONS, True
            )

        assert db.get_task(
            owner_user_id=owner, dataset_id="local-unbound", task_id=task_id
        )["id"] == task_id
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_bind_caught_hash_failure_rolls_back_rekey_and_restores_force(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "950025"
    target = f"dataset-{uuid4()}"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        _note_id, task_id, _counts = _create_complete_postgres_local_task_graph(db)
        original_hash = db._note_task_v60_hash
        hash_calls = 0

        def drift_rebound_hash(value: object) -> str:
            nonlocal hash_calls
            hash_calls += 1
            digest = original_hash(value)
            rebound_start = 3 * len(CharactersRAGDB._NOTE_TASK_V60_TABLES)
            return f"{digest}-drift" if hash_calls > rebound_start else digest

        monkeypatch.setattr(db, "_note_task_v60_hash", drift_rebound_hash)
        with db.transaction() as conn:
            with pytest.raises(ConflictError, match="complete-set verification"):
                db.bind_local_task_graph_to_dataset(
                    owner_user_id=owner,
                    target_dataset_id=target,
                    conn=conn,
                )
            assert _postgres_task_force_flags(conn) == dict.fromkeys(
                CharactersRAGDB._NOTE_TASK_V60_RELATIONS, True
            )

        assert db.get_task(
            owner_user_id=owner, dataset_id="local-unbound", task_id=task_id
        )["id"] == task_id
        assert db.get_task(
            owner_user_id=owner, dataset_id=target, task_id=task_id
        ) is None
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_note_task_schema_remains_authoritative_at_v61(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950001"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        with db.transaction() as conn:
            version = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()["version"]
            policy_rows = conn.execute(
                "SELECT tablename, policyname FROM pg_policies "
                "WHERE schemaname = current_schema() AND tablename = ANY(?) "
                "ORDER BY tablename, policyname",
                (list(CharactersRAGDB._NOTE_TASK_V60_RELATIONS),),
            ).fetchall()
        assert version == 61
        assert [(row["tablename"], row["policyname"]) for row in policy_rows] == [
            (table, f"{table}_tenant_isolation")
            for table in sorted(CharactersRAGDB._NOTE_TASK_V60_RELATIONS)
        ]
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize(
    "drift_statements",
    (
        ("ALTER TABLE notes NO FORCE ROW LEVEL SECURITY",),
        ("DROP POLICY notes_tenant_isolation ON notes",),
        ("CREATE POLICY notes_open ON notes USING (true) WITH CHECK (true)",),
        (
            "DROP POLICY notes_tenant_isolation ON notes",
            "CREATE POLICY notes_tenant_isolation ON notes USING (true) WITH CHECK (true)",
        ),
    ),
    ids=("no-force", "missing-policy", "extra-policy", "weakened-policy"),
)
def test_postgres_current_v60_rejects_notes_authority_drift_without_repair(
    pg_database_config: DatabaseConfig,
    drift_statements: tuple[str, ...],
) -> None:
    owner = "950026"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
    drift_backend = None
    observer_backend = None
    startup_db = None

    try:
        with db.transaction() as conn:
            for statement in drift_statements:
                conn.execute(statement)
        drifted = _postgres_rls_authority_snapshot(backend, table="notes")
        db.close_all_connections()
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="ownership or RLS|policy catalog"):
            startup_db = CharactersRAGDB(":memory:", client_id=owner, backend=drift_backend)
        observer_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        assert _postgres_rls_authority_snapshot(observer_backend, table="notes") == drifted
    finally:
        if startup_db is not None:
            startup_db.close_all_connections()
        db.close_all_connections()
        backend.get_pool().close_all()
        if drift_backend is not None:
            drift_backend.get_pool().close_all()
        if observer_backend is not None:
            observer_backend.get_pool().close_all()


@pytest.mark.parametrize(
    "drift_statements",
    (
        ("ALTER TABLE note_task_scope_authority NO FORCE ROW LEVEL SECURITY",),
        ("DROP POLICY note_task_scope_authority_tenant_isolation ON note_task_scope_authority",),
        (
            "CREATE POLICY note_task_scope_authority_open ON note_task_scope_authority "
            "USING (true) WITH CHECK (true)",
        ),
        (
            "DROP POLICY note_task_scope_authority_tenant_isolation "
            "ON note_task_scope_authority",
            "CREATE POLICY note_task_scope_authority_tenant_isolation "
            "ON note_task_scope_authority USING (true) WITH CHECK (true)",
        ),
    ),
    ids=("no-force", "missing-policy", "extra-policy", "weakened-policy"),
)
def test_postgres_current_v60_rejects_scope_authority_drift_without_repair(
    pg_database_config: DatabaseConfig,
    drift_statements: tuple[str, ...],
) -> None:
    owner = "950028"
    table = "note_task_scope_authority"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
    drift_backend = None
    observer_backend = None
    startup_db = None

    try:
        with db.transaction() as conn:
            for statement in drift_statements:
                conn.execute(statement)
        drifted = _postgres_rls_authority_snapshot(backend, table=table)
        db.close_all_connections()
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        with pytest.raises(CharactersRAGDBError, match="ownership or RLS|policy catalog"):
            startup_db = CharactersRAGDB(":memory:", client_id=owner, backend=drift_backend)
        observer_backend = DatabaseBackendFactory.create_backend(pg_database_config)
        assert _postgres_rls_authority_snapshot(observer_backend, table=table) == drifted
    finally:
        if startup_db is not None:
            startup_db.close_all_connections()
        db.close_all_connections()
        backend.get_pool().close_all()
        if drift_backend is not None:
            drift_backend.get_pool().close_all()
        if observer_backend is not None:
            observer_backend.get_pool().close_all()


def test_postgres_dataset_cursor_page_uses_exact_index(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950027"
    dataset_id = "cursor-dataset"
    other_dataset_id = "cursor-other"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
    note_id = db.add_note("Cursor parent", "Body")

    try:
        db.bind_local_task_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=dataset_id,
        )
        with db.transaction() as conn:
            for cursor in range(1, 65):
                target_event = db.record_task_event(
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    note_id=note_id,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )
                conn.execute(
                    "UPDATE task_events SET sync_server_cursor=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (cursor, owner, dataset_id, target_event["id"]),
                )
            conn.execute(
                "UPDATE note_task_scope_authority SET dataset_id=? WHERE owner_user_id=?",
                (other_dataset_id, owner),
            )
            conn.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (other_dataset_id,),
            )
            for cursor in range(1, 65):
                other_event = db.record_task_event(
                    owner_user_id=owner,
                    dataset_id=other_dataset_id,
                    note_id=note_id,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )
                conn.execute(
                    "UPDATE task_events SET sync_server_cursor=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (cursor, owner, other_dataset_id, other_event["id"]),
                )
            conn.execute(
                "UPDATE note_task_scope_authority SET dataset_id=? WHERE owner_user_id=?",
                (dataset_id, owner),
            )
            conn.execute("ANALYZE task_events")
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,))
            conn.execute("SET LOCAL enable_seqscan=off")
            index_definition = conn.execute(
                "SELECT pg_get_indexdef('idx_task_events_scope_cursor'::regclass,0,true)"
            ).fetchone()["pg_get_indexdef"]
            assert index_definition == (
                "CREATE INDEX idx_task_events_scope_cursor ON task_events USING btree "
                "(owner_user_id, dataset_id, sync_server_cursor, id)"
            )
            plan_rows = conn.execute(
                "EXPLAIN SELECT id FROM task_events "
                "WHERE owner_user_id=? AND dataset_id=? AND sync_server_cursor>? "
                "ORDER BY sync_server_cursor,id LIMIT ?",
                (owner, dataset_id, 0, 10),
            ).fetchall()
        plan = [str(next(iter(dict(row).values()))) for row in plan_rows]
        relevant_plan = [
            line.strip()
            for line in plan
            if "Sort" in line or "Index" in line or "Filter:" in line
        ]
        assert any("idx_task_events_scope_cursor" in line for line in plan), relevant_plan
        assert not any("Sort" in line for line in plan), relevant_plan
        assert not any("Filter: (sync_server_cursor" in line for line in plan), relevant_plan
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_note_tasks_allow_same_id_for_two_owners(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "950011"
    owner_b = "950012"
    dataset_id = f"dataset-{uuid4()}"
    other_dataset_id = f"dataset-{uuid4()}"
    note_a = str(uuid4())
    note_b = str(uuid4())
    task_id = str(uuid4())
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(":memory:", client_id=owner_b, backend=backend_b)
    ident = backend_a.escape_identifier  # type: ignore[attr-defined]
    role_name = f"note_task_rls_{uuid4().hex[:8]}"
    role_created = False

    try:
        db_a.add_note("Owner A", "Body", note_id=note_a)
        db_b.add_note("Owner B", "Body", note_id=note_b)
        db_a.bind_local_task_graph_to_dataset(
            owner_user_id=owner_a,
            target_dataset_id=dataset_id,
        )
        db_b.bind_local_task_graph_to_dataset(
            owner_user_id=owner_b,
            target_dataset_id=dataset_id,
        )
        first = db_a.create_task(
            owner_user_id=owner_a,
            dataset_id=dataset_id,
            task_id=task_id,
            note_id=note_a,
            text="Same ID",
            actor_type=None,
        )
        second = db_b.create_task(
            owner_user_id=owner_b,
            dataset_id=dataset_id,
            task_id=task_id,
            note_id=note_b,
            text="Same ID",
            actor_type=None,
        )
        with pytest.raises(ConflictError, match="bound authority"):
            db_a.create_task(
                owner_user_id=owner_a,
                dataset_id=other_dataset_id,
                task_id=task_id,
                note_id=note_a,
                text="Same ID, other dataset",
                actor_type=None,
            )
        event_a = db_a.record_task_event(
            owner_user_id=owner_a,
            dataset_id=dataset_id,
            task_id=task_id,
            note_id=note_a,
            event_type="updated",
            actor_type="user",
        )
        event_b = db_b.record_task_event(
            owner_user_id=owner_b,
            dataset_id=dataset_id,
            task_id=task_id,
            note_id=note_b,
            event_type="updated",
            actor_type="user",
        )
        db_a.mark_task_activity_read(
            owner_user_id=owner_a,
            dataset_id=dataset_id,
            event_id=event_a["id"],
            user_id=owner_a,
        )
        db_b.mark_task_activity_read(
            owner_user_id=owner_b,
            dataset_id=dataset_id,
            event_id=event_b["id"],
            user_id=owner_b,
        )

        assert first["owner_user_id"] == owner_a
        assert second["owner_user_id"] == owner_b
        assert db_a.get_task(
            owner_user_id=owner_a,
            dataset_id=dataset_id,
            task_id=task_id,
        )["note_id"] == note_a
        assert db_b.get_task(
            owner_user_id=owner_b,
            dataset_id=dataset_id,
            task_id=task_id,
        )["note_id"] == note_b

        with backend_a.transaction() as conn:
            backend_a.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend_a.execute(f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}", connection=conn)
            backend_a.execute(f"GRANT SELECT ON notes TO {ident(role_name)}", connection=conn)
            for table in CharactersRAGDB._NOTE_TASK_V60_TABLES:
                backend_a.execute(
                    f"GRANT SELECT, INSERT, UPDATE ON {ident(table)} TO {ident(role_name)}",
                    connection=conn,
                )
            backend_a.execute(
                f"GRANT SELECT ON note_task_scope_authority TO {ident(role_name)}",
                connection=conn,
            )
            backend_a.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        with backend_a.transaction() as conn:
            backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend_a.execute(
                "SELECT set_config('app.current_user_id', ?, true)", (owner_a,), connection=conn
            )
            backend_a.execute(
                "SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,), connection=conn
            )
            principal = backend_a.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
                connection=conn,
            ).rows[0]
            assert principal == {"rolsuper": False, "rolbypassrls": False}
            visible_authority = backend_a.execute(
                "SELECT owner_user_id,dataset_id FROM note_task_scope_authority",
                connection=conn,
            ).rows
            assert visible_authority == [
                {"owner_user_id": owner_a, "dataset_id": dataset_id}
            ]
            visible_tasks = backend_a.execute(
                "SELECT owner_user_id,note_id FROM note_tasks WHERE id=?", (task_id,), connection=conn
            ).rows
            assert visible_tasks == [{"owner_user_id": owner_a, "note_id": note_a}]
            visible_events = backend_a.execute(
                "SELECT owner_user_id,id FROM task_events WHERE task_id=? ORDER BY id",
                (task_id,),
                connection=conn,
            ).rows
            assert {row["id"] for row in visible_events} == {event_a["id"]}
            visible_read_state = backend_a.execute(
                "SELECT owner_user_id,event_id FROM task_event_read_state ORDER BY event_id",
                connection=conn,
            ).rows
            assert visible_read_state == [{"owner_user_id": owner_a, "event_id": event_a["id"]}]
            hidden_update = backend_a.execute(
                "UPDATE note_tasks SET text=? WHERE owner_user_id=? AND dataset_id=? AND id=?",
                ("cross-owner overwrite", owner_b, dataset_id, task_id),
                connection=conn,
            )
            assert hidden_update.rowcount == 0
            hidden_dataset_update = backend_a.execute(
                "UPDATE note_tasks SET text=? WHERE owner_user_id=? AND dataset_id=? AND id=?",
                ("cross-dataset overwrite", owner_a, other_dataset_id, task_id),
                connection=conn,
            )
            assert hidden_dataset_update.rowcount == 0

        with pytest.raises(DatabaseError):
            with backend_a.transaction() as conn:
                backend_a.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
                backend_a.execute(
                    "SELECT set_config('app.current_user_id', ?, true)", (owner_a,), connection=conn
                )
                backend_a.execute(
                    "SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,), connection=conn
                )
                backend_a.execute(
                    """
                    INSERT INTO note_tasks(
                      owner_user_id,dataset_id,id,note_id,text,status,metadata_json,
                      projection_status,deleted,created_at,updated_at,client_id,version,
                      canonical_revision,canonical_hash
                    ) VALUES (?,?,?,?,?,'open','{}','live',FALSE,CURRENT_TIMESTAMP,
                              CURRENT_TIMESTAMP,?,1,1,?)
                    """,
                    (owner_a, dataset_id, str(uuid4()), note_b, "cross-owner parent", owner_a,
                     "sha256:" + "a" * 64),
                    connection=conn,
                )

        # Exercise a real task page amid same-scope activity so the ordered task
        # index is both selective and useful for satisfying the LIMIT.
        with db_a.transaction() as conn:
            for _ in range(64):
                db_a.record_task_event(
                    owner_user_id=owner_a,
                    dataset_id=dataset_id,
                    task_id=task_id,
                    note_id=note_a,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )
                db_a.record_task_event(
                    owner_user_id=owner_a,
                    dataset_id=dataset_id,
                    note_id=note_a,
                    event_type="updated",
                    actor_type="user",
                    conn=conn,
                )

        db_a.get_task(owner_user_id=owner_a, dataset_id=dataset_id, task_id=task_id)
        with db_a.transaction() as conn:
            conn.execute("ANALYZE task_events")
            conn.execute("SET LOCAL enable_seqscan=off")
            task_plan_rows = conn.execute(
                "EXPLAIN SELECT id FROM note_tasks WHERE owner_user_id=? AND dataset_id=? "
                "AND note_id=? AND deleted=FALSE ORDER BY created_at,id LIMIT ?",
                (owner_a, dataset_id, note_a, 50),
            ).fetchall()
            event_plan_rows = conn.execute(
                "EXPLAIN SELECT id FROM task_events WHERE owner_user_id=? AND dataset_id=? "
                "AND task_id=? ORDER BY created_at,id LIMIT ?",
                (owner_a, dataset_id, task_id, 50),
            ).fetchall()
        task_plan = " ".join(str(next(iter(dict(row).values()))) for row in task_plan_rows)
        event_plan = " ".join(str(next(iter(dict(row).values()))) for row in event_plan_rows)
        assert "idx_note_tasks_scope_note_page" in task_plan
        assert "idx_task_events_scope_task_created" in event_plan
    finally:
        if role_created:
            with backend_a.transaction() as conn:
                backend_a.execute(f"REVOKE {ident(role_name)} FROM CURRENT_USER", connection=conn)
                backend_a.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                backend_a.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        db_a.close_all_connections()
        db_b.close_all_connections()
        backend_a.get_pool().close_all()
        backend_b.get_pool().close_all()


@pytest.mark.parametrize(
    "replacement_sql",
    [
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks("
        "owner_user_id,dataset_id,projection_status,deleted,id) INCLUDE(text)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks("
        "owner_user_id,dataset_id,projection_status,deleted,id DESC NULLS FIRST)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks("
        "owner_user_id text_pattern_ops,dataset_id,projection_status,deleted,id)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks("
        "owner_user_id COLLATE \"C\",dataset_id,projection_status,deleted,id)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks("
        "(lower(owner_user_id)),dataset_id,projection_status,deleted,id)",
    ],
    ids=("include", "descending-nulls", "opclass", "collation", "expression"),
)
def test_postgres_note_task_current_v60_rejects_full_index_shape_drift(
    pg_database_config: DatabaseConfig,
    replacement_sql: str,
) -> None:
    owner = "950021"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        with db.transaction() as conn:
            canonical = conn.execute(
                "SELECT pg_get_indexdef(indexrelid,0,true) AS definition "
                "FROM pg_index WHERE indexrelid='idx_note_tasks_scope_projection'::regclass"
            ).fetchone()["definition"]
            assert canonical == (
                "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks USING btree "
                "(owner_user_id, dataset_id, projection_status, deleted, id)"
            )
            conn.execute("DROP INDEX idx_note_tasks_scope_projection")
            conn.execute(replacement_sql)
        db.close_all_connections()
        drift_backend = DatabaseBackendFactory.create_backend(pg_database_config)

        try:
            with pytest.raises(CharactersRAGDBError, match="index catalog drifted"):
                CharactersRAGDB(":memory:", client_id=owner, backend=drift_backend)
        finally:
            drift_backend.get_pool().close_all()
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _populate_reviewed_postgres_v59_source(
    db: CharactersRAGDB,
) -> tuple[str, str, str]:
    owner = str(db.client_id)
    note_id = str(uuid4())
    task_id = str(uuid4())
    event_id = str(uuid4())
    timestamp = "2026-08-14T00:00:00+00:00"
    db.add_note("Legacy task", "- [ ] Migrate this task\n", note_id=note_id)
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO note_tasks(
                id,note_id,text,status,metadata_json,projection_status,deleted,
                created_at,updated_at,completed_at,client_id,version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task_id, note_id, "Migrate this task", "open", "{}", "live", False,
                timestamp, timestamp, None, owner, 1,
            ),
        )
        conn.execute(
            """
            INSERT INTO task_note_projections(
                task_id,note_id,note_version,line_number,start_offset,end_offset,
                normalized_text_hash,occurrence_index,block_fingerprint,raw_line,
                has_child_content,projection_status,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task_id, note_id, 1, 1, 0, 23, "sha256:legacy", 0,
                "legacy-block", "- [ ] Migrate this task", False, "live", timestamp,
            ),
        )
        conn.execute(
            """
            INSERT INTO task_events(
                id,task_id,note_id,event_type,actor_type,actor_id,tool_name,policy_mode,
                approval_id,old_value_json,new_value_json,created_at,client_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id, task_id, note_id, "created", "user", owner, None, None,
                None, None, "{}", timestamp, owner,
            ),
        )
        conn.execute(
            "INSERT INTO task_event_read_state(event_id,user_id,read_at,dismissed_at) "
            "VALUES (?,?,?,?)",
            (event_id, owner, timestamp, None),
        )
        conn.execute(
            """
            INSERT INTO note_task_reconciliation_state(
                note_id,note_version,status,reconciled_at,item_count,warning_count,cursor
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (note_id, 1, "clean", timestamp, 1, 0, "legacy-cursor"),
        )
    return note_id, task_id, event_id


def _postgres_v60_task_catalog_snapshot(db: CharactersRAGDB) -> dict[str, object]:
    tables = list(CharactersRAGDB._NOTE_TASK_V60_RELATIONS)
    with db.transaction() as conn:
        relation_rows = conn.execute(
            "SELECT relname,relrowsecurity,relforcerowsecurity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) ORDER BY relname",
            (tables,),
        ).fetchall()
        column_rows = conn.execute(
            "SELECT c.relname,a.attnum,a.attname,format_type(a.atttypid,a.atttypmod) AS type,"
            "a.attnotnull,pg_get_expr(d.adbin,d.adrelid,false) AS default_expression "
            "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) "
            "AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attnum",
            (tables,),
        ).fetchall()
        constraint_rows = conn.execute(
            "SELECT c.relname,k.conname,k.contype,k.convalidated,"
            "pg_get_constraintdef(k.oid,false) AS definition "
            "FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) AND k.contype<>'n' "
            "ORDER BY c.relname,k.conname",
            (tables,),
        ).fetchall()
        index_rows = conn.execute(
            "SELECT table_row.relname,index_row.relname AS index_name,"
            "pg_get_indexdef(i.indexrelid,0,true) AS definition "
            "FROM pg_index i JOIN pg_class table_row ON table_row.oid=i.indrelid "
            "JOIN pg_class index_row ON index_row.oid=i.indexrelid "
            "JOIN pg_namespace n ON n.oid=table_row.relnamespace "
            "WHERE n.nspname=current_schema() AND (table_row.relname=ANY(?) OR "
            "index_row.relname='uq_notes_owner_id') ORDER BY table_row.relname,index_row.relname",
            (tables,),
        ).fetchall()
        policy_rows = conn.execute(
            "SELECT tablename,policyname,permissive,roles::text,cmd,qual,with_check "
            "FROM pg_policies WHERE schemaname=current_schema() AND tablename=ANY(?) "
            "ORDER BY tablename,policyname",
            (tables,),
        ).fetchall()
    return {
        "relations": [dict(row) for row in relation_rows],
        "columns": [dict(row) for row in column_rows],
        "constraints": [dict(row) for row in constraint_rows],
        "indexes": [dict(row) for row in index_rows],
        "policies": [dict(row) for row in policy_rows],
    }


def _postgres_v59_source_snapshot(db: CharactersRAGDB) -> dict[str, object]:
    source_tables = ("notes", *CharactersRAGDB._NOTE_TASK_V60_TABLES[:-1])
    legacy_tables = CharactersRAGDB._NOTE_TASK_V60_TABLES[:-1]
    data_order = {
        "note_tasks": "id",
        "task_note_projections": "task_id",
        "task_events": "id",
        "task_event_read_state": "event_id,user_id",
        "note_task_reconciliation_state": "note_id",
    }
    collision_names = (
        "task_projection_drifts",
        "note_task_scope_authority",
        "uq_notes_owner_id",
        *(f"{table}_v60" for table in CharactersRAGDB._NOTE_TASK_V60_RELATIONS),
    )
    with db.transaction() as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()["version"]
        relations = conn.execute(
            "SELECT c.relname,c.relrowsecurity,c.relforcerowsecurity "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) ORDER BY c.relname",
            (list(source_tables),),
        ).fetchall()
        policies = conn.execute(
            "SELECT tablename,policyname,permissive,roles::text,cmd,qual,with_check "
            "FROM pg_policies WHERE schemaname=current_schema() AND tablename=ANY(?) "
            "ORDER BY tablename,policyname",
            (list(source_tables),),
        ).fetchall()
        constraints = conn.execute(
            "SELECT c.relname,k.conname,pg_get_constraintdef(k.oid,false) AS definition "
            "FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) AND k.contype<>'n' "
            "ORDER BY c.relname,k.conname",
            (list(legacy_tables),),
        ).fetchall()
        data = {
            table: [dict(row) for row in conn.execute(
                f"SELECT * FROM {table} ORDER BY {ordering}"  # nosec B608
            ).fetchall()]
            for table, ordering in data_order.items()
        }
        notes = conn.execute("SELECT id,client_id FROM notes ORDER BY id").fetchall()
        remnants = conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname=ANY(?) ORDER BY c.relname",
            (list(collision_names),),
        ).fetchall()
    return db._note_task_v60_json_safe(
        {
            "version": version,
            "relations": [dict(row) for row in relations],
            "policies": [dict(row) for row in policies],
            "constraints": [dict(row) for row in constraints],
            "data": data,
            "notes": [dict(row) for row in notes],
            "target_remnants": [dict(row) for row in remnants],
        }
    )


def test_postgres_populated_v59_upgrade_matches_fresh_v60_catalog(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "950030"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)

    try:
        fresh_catalog = _postgres_v60_task_catalog_snapshot(db)
        fresh_catalog["columns"] = [
            row
            for row in fresh_catalog["columns"]
            if row["attname"]
            not in {"task_graph_bound", "moodboard_graph_bound", "studio_graph_bound"}
        ]
        _restore_reviewed_postgres_v59_task_source(db)
        note_id, task_id, event_id = _populate_reviewed_postgres_v59_source(db)
        with db.transaction() as conn:
            db._migrate_from_v59_to_v60_postgres(conn)

        assert _postgres_v60_task_catalog_snapshot(db) == fresh_catalog
        with db.transaction() as conn:
            db._verify_note_task_schema_postgres(conn)
            task = conn.execute(
                "SELECT owner_user_id,dataset_id,id,note_id FROM note_tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            event = conn.execute(
                "SELECT owner_user_id,dataset_id,id,note_id FROM task_events WHERE id=?",
                (event_id,),
            ).fetchone()
        assert dict(task) == {
            "owner_user_id": owner,
            "dataset_id": "local-unbound",
            "id": task_id,
            "note_id": note_id,
        }
        assert dict(event) == {
            "owner_user_id": owner,
            "dataset_id": "local-unbound",
            "id": event_id,
            "note_id": note_id,
        }
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("stage", ("validate", "create", "copy", "index", "verify"))
def test_postgres_v60_checkpoint_failure_restores_exact_populated_v59_state(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="950031", backend=backend)

    try:
        _restore_reviewed_postgres_v59_task_source(db)
        _populate_reviewed_postgres_v59_source(db)
        before = _postgres_v59_source_snapshot(db)
        assert before["version"] == 59
        assert before["target_remnants"] == []

        def fail_at_stage(_db: CharactersRAGDB, current_stage: str) -> None:
            if current_stage == stage:
                raise SchemaError(f"injected PostgreSQL v60 {stage} failure")

        monkeypatch.setattr(
            CharactersRAGDB,
            "_note_task_v60_migration_checkpoint",
            fail_at_stage,
        )
        with pytest.raises(SchemaError, match=f"injected PostgreSQL v60 {stage} failure"):
            with db.transaction() as conn:
                db._migrate_from_v59_to_v60_postgres(conn)

        assert _postgres_v59_source_snapshot(db) == before
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_concurrent_v59_initializers_serialize_one_migration(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "950032"
    setup_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    setup_db = CharactersRAGDB(":memory:", client_id=owner, backend=setup_backend)
    _restore_reviewed_postgres_v59_task_source(setup_db)
    _populate_reviewed_postgres_v59_source(setup_db)
    setup_db.close_all_connections()
    setup_backend.get_pool().close_all()

    barrier = threading.Barrier(2)
    record_lock = threading.Lock()
    checkpoints: list[str] = []
    verify_threads: list[int] = []
    original_checkpoint = CharactersRAGDB._note_task_v60_migration_checkpoint
    original_verify = CharactersRAGDB._verify_note_task_schema_postgres

    def track_checkpoint(_db: CharactersRAGDB, stage: str) -> None:
        with record_lock:
            checkpoints.append(stage)
        original_checkpoint(stage)

    def track_verify(db: CharactersRAGDB, conn: object) -> None:
        with record_lock:
            verify_threads.append(threading.get_ident())
        original_verify(db, conn)

    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        track_checkpoint,
    )
    monkeypatch.setattr(
        CharactersRAGDB,
        "_verify_note_task_schema_postgres",
        track_verify,
    )

    def initialize() -> int:
        backend = DatabaseBackendFactory.create_backend(pg_database_config)
        barrier.wait(timeout=30)
        db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
        try:
            with db.transaction() as conn:
                return db._get_schema_version_postgres(conn)
        finally:
            db.close_all_connections()
            backend.get_pool().close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize) for _index in range(2)]
        versions = [future.result(timeout=120) for future in futures]

    assert versions == [61, 61]
    assert checkpoints == ["validate", "create", "copy", "index", "verify"]
    assert len(verify_threads) == 2
    assert len(set(verify_threads)) == 2

    check_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    try:
        with check_backend.transaction() as conn:
            version = check_backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name=%s",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            ).scalar
            remnants = check_backend.execute(
                "SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname=current_schema() AND relname LIKE %s",
                ("%_v60",),
                connection=conn,
            ).rows
        assert version == 61
        assert remnants == []
    finally:
        check_backend.get_pool().close_all()
