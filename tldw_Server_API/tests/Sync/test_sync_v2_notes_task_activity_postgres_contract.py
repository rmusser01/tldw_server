"""PostgreSQL contracts for dormant ``notes.task_activity`` storage."""

from __future__ import annotations

import inspect
from typing import Any
from uuid import UUID, uuid4

import pytest
from psycopg import sql as psycopg_sql

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityV1,
    notes_task_activity_object_hash,
    parse_notes_task_activity_v1,
)

pytestmark = pytest.mark.integration

DATASET_ID = "local-unbound"
NOW = "2026-08-23T10:00:00+00:00"


def _execute_role_statement(connection: Any, statement: str, role_name: str) -> None:
    raw_connection = getattr(connection, "_connection", connection)
    with raw_connection.cursor() as cursor:
        cursor.execute(psycopg_sql.SQL(statement).format(psycopg_sql.Identifier(role_name)))


def _activity(
    *,
    owner: str,
    activity_id: str,
    note_id: str,
    task_id: str | None,
) -> NotesTaskActivityV1:
    return parse_notes_task_activity_v1(
        {
            "activity_id": activity_id,
            "note_id": note_id,
            "task_id": task_id,
            "event_type": "created",
            "actor_type": "user",
            "actor_id": owner,
            "source_device_id": None,
            "client_occurred_at": NOW,
            "source_kind": "rest",
            "corrects_activity_id": None,
            "old_value": None,
            "new_value": {
                "title": "PostgreSQL activity",
                "status": "open",
                "completed_at": None,
                "metadata": {
                    "description": None,
                    "priority": None,
                    "due_date": None,
                    "estimate": None,
                    "recurrence": None,
                    "assignee_id": None,
                    "tags": [],
                    "custom": {},
                },
            },
            "metadata": {},
        },
        owner_user_id=owner,
        bound_actor_type="user",
        bound_actor_id=owner,
        authenticated_device_id=None,
        trusted_server_origin=True,
    )


def _seed_task(db: CharactersRAGDB, *, owner: str) -> tuple[str, str]:
    note_id = str(uuid4())
    task_id = str(uuid4())
    db.add_note("PostgreSQL activity", "Body\n", note_id=note_id)
    db.task_store.create_task(
        owner_user_id=owner,
        dataset_id=DATASET_ID,
        note_id=note_id,
        text="PostgreSQL activity",
        task_id=task_id,
        projection_status="unlinked",
    )
    return note_id, task_id


def _insert_activity(
    db: CharactersRAGDB,
    *,
    owner: str,
    role_name: str,
    payload: NotesTaskActivityV1,
    cursor: int,
) -> dict[str, Any]:
    object_hash = notes_task_activity_object_hash(payload, revision=1, deleted=False)
    with db.transaction() as conn:
        _execute_role_statement(conn, "SET LOCAL ROLE {}", role_name)
        conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner,))
        return db.task_store.create_sync_task_activity(
            owner_user_id=owner,
            dataset_id=DATASET_ID,
            payload=payload,
            sync_object_hash=object_hash,
            sync_server_cursor=cursor,
            conn=conn,
        )


def test_activity_sql_binds_scope_parent_and_cursor_id_ordering() -> None:
    get_source = inspect.getsource(TaskStore.get_sync_task_activity)
    page_source = inspect.getsource(TaskStore.page_sync_task_activity)
    parent_source = inspect.getsource(TaskStore._require_sync_activity_parents)
    indexes = " ".join(CharactersRAGDB._note_task_v60_postgres_indexes())
    policy = CharactersRAGDB._note_task_v60_policy_predicates()["task_events"]

    assert "e.owner_user_id = ? AND e.dataset_id = ? AND e.id = ?" in get_source
    assert "t.id = e.task_id AND t.note_id = e.note_id" in get_source
    assert "e.sync_server_cursor > ?" in page_source
    assert "e.sync_server_cursor = ? AND e.id > ?" in page_source
    assert "ORDER BY e.sync_server_cursor ASC, e.id ASC" in page_source
    assert 'task["note_id"] != payload.note_id' in parent_source
    assert "idx_task_events_scope_cursor" in indexes
    assert "task.id=task_events.task_id" in policy
    assert "task.note_id=task_events.note_id" in policy


def test_postgres_activity_rls_parent_paging_replay_and_rollback(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_a = "980001"
    owner_b = "980002"
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(":memory:", client_id=owner_b, backend=backend_b)
    role_name = f"notes_task_activity_{uuid4().hex[:8]}"
    role_created = False

    try:
        with backend_a.transaction() as conn:
            _execute_role_statement(conn, "CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS", role_name)
            _execute_role_statement(conn, "GRANT USAGE ON SCHEMA public TO {}", role_name)
            for table in ("notes", "note_tasks"):
                _execute_role_statement(conn, f"GRANT SELECT ON {table} TO {{}}", role_name)
            _execute_role_statement(
                conn,
                "GRANT SELECT, UPDATE ON note_task_scope_authority TO {}",
                role_name,
            )
            _execute_role_statement(conn, "GRANT SELECT, INSERT, UPDATE ON task_events TO {}", role_name)
            _execute_role_statement(conn, "GRANT {} TO CURRENT_USER", role_name)
        role_created = True

        note_a, task_a = _seed_task(db_a, owner=owner_a)
        note_b, task_b = _seed_task(db_b, owner=owner_b)
        wrong_note_a = str(uuid4())
        db_a.add_note("Wrong activity parent", "Body\n", note_id=wrong_note_a)
        activity_ids = [str(UUID(int=index, version=4)) for index in (1, 2, 3)]
        payloads = [
            _activity(
                owner=owner_a,
                activity_id=activity_id,
                note_id=note_a,
                task_id=task_a,
            )
            for activity_id in activity_ids
        ]
        _insert_activity(db_a, owner=owner_a, role_name=role_name, payload=payloads[1], cursor=50)
        _insert_activity(db_a, owner=owner_a, role_name=role_name, payload=payloads[0], cursor=50)
        inserted = _insert_activity(
            db_a,
            owner=owner_a,
            role_name=role_name,
            payload=payloads[2],
            cursor=51,
        )
        foreign = _activity(
            owner=owner_b,
            activity_id=str(uuid4()),
            note_id=note_b,
            task_id=task_b,
        )
        _insert_activity(db_b, owner=owner_b, role_name=role_name, payload=foreign, cursor=60)

        page = db_a.task_store.page_sync_task_activity(
            owner_user_id=owner_a,
            dataset_id=DATASET_ID,
            limit=2,
        )
        next_page = db_a.task_store.page_sync_task_activity(
            owner_user_id=owner_a,
            dataset_id=DATASET_ID,
            after_server_cursor=int(page[-1]["sync_server_cursor"]),
            after_activity_id=str(page[-1]["id"]),
            limit=2,
        )
        assert [(row["sync_server_cursor"], row["id"]) for row in [*page, *next_page]] == [
            (50, activity_ids[0]),
            (50, activity_ids[1]),
            (51, activity_ids[2]),
        ]
        assert db_a.task_store.verify_sync_task_activity_postcondition(
            owner_user_id=owner_a,
            dataset_id=DATASET_ID,
            payload=payloads[2],
            sync_revision=1,
            sync_object_hash=inserted["sync_object_hash"],
            sync_server_cursor=51,
        )

        with db_a.transaction() as conn:
            _execute_role_statement(conn, "SET LOCAL ROLE {}", role_name)
            conn.execute("SELECT set_config('app.current_user_id', ?, true)", (owner_a,))
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (DATASET_ID,))
            visible = conn.execute("SELECT id FROM task_events ORDER BY id").fetchall()
        assert {str(row["id"]) for row in visible} == set(activity_ids)

        wrong_parent = _activity(
            owner=owner_a,
            activity_id=str(uuid4()),
            note_id=wrong_note_a,
            task_id=task_a,
        )
        with pytest.raises(ConflictError, match="scope does not match its note"):
            _insert_activity(
                db_a,
                owner=owner_a,
                role_name=role_name,
                payload=wrong_parent,
                cursor=70,
            )

        rolled_back = _activity(
            owner=owner_a,
            activity_id=str(uuid4()),
            note_id=note_a,
            task_id=task_a,
        )
        monkeypatch.setattr(
            db_a.task_store,
            "_sync_task_activity_materialization_checkpoint",
            lambda _phase: (_ for _ in ()).throw(RuntimeError("forced activity rollback")),
        )
        with pytest.raises(RuntimeError, match="forced activity rollback"):
            _insert_activity(
                db_a,
                owner=owner_a,
                role_name=role_name,
                payload=rolled_back,
                cursor=71,
            )
        assert db_a.task_store.get_sync_task_activity(
            owner_user_id=owner_a,
            dataset_id=DATASET_ID,
            activity_id=rolled_back.activity_id,
        ) is None
    finally:
        if role_created:
            with backend_a.transaction() as conn:
                _execute_role_statement(conn, "REVOKE {} FROM CURRENT_USER", role_name)
                _execute_role_statement(conn, "DROP OWNED BY {}", role_name)
                _execute_role_statement(conn, "DROP ROLE {}", role_name)
        db_b.close_all_connections()
        db_a.close_all_connections()
        backend_b.get_pool().close_all()
        backend_a.get_pool().close_all()
