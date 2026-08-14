"""Live PostgreSQL tenancy proof for the schema-v60 Notes task graph."""

from __future__ import annotations

from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_note_task_schema_v60_is_authoritative(
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
                (list(CharactersRAGDB._NOTE_TASK_V60_TABLES),),
            ).fetchall()
        assert version == 60
        assert [(row["tablename"], row["policyname"]) for row in policy_rows] == [
            (table, f"{table}_tenant_isolation")
            for table in sorted(CharactersRAGDB._NOTE_TASK_V60_TABLES)
        ]
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
        third = db_a.create_task(
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
        assert third["dataset_id"] == other_dataset_id
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
