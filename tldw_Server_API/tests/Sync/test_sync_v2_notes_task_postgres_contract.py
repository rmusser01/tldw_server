"""PostgreSQL contracts for dormant ``notes.task`` product materialization."""

from __future__ import annotations

import inspect
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sync.v2.materializers.notes_task import (
    NotesTaskMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskV1Payload,
    notes_task_object_hash,
    parse_notes_task_v1,
)

pytestmark = pytest.mark.integration

_DATASET_ID = "local-unbound"


def _payload(
    *,
    owner: str,
    task_id: str,
    note_id: str,
    title: str,
    status: str = "open",
    completed_at: str | None = None,
) -> NotesTaskV1Payload:
    return parse_notes_task_v1(
        {
            "task_id": task_id,
            "note_id": note_id,
            "title": title,
            "description": None,
            "status": status,
            "completed_at": completed_at,
            "priority": "high",
            "due_date": None,
            "estimate": "30m",
            "recurrence": None,
            "assignee_id": None,
            "tags": [],
            "custom": {},
        },
        owner_user_id=owner,
    )


def _apply_lifecycle(
    db: CharactersRAGDB,
    *,
    owner: str,
    role_identifier: str,
) -> dict[str, Any]:
    note_id = str(uuid4())
    task_id = str(uuid4())
    db.add_note("PostgreSQL task lifecycle", "Body\n", note_id=note_id)

    created_payload = _payload(
        owner=owner,
        task_id=task_id,
        note_id=note_id,
        title="Create on PostgreSQL",
    )
    created_hash = notes_task_object_hash(
        created_payload,
        revision=1,
        deleted=False,
    )
    with db.transaction() as conn:
        conn.execute(f"SET LOCAL ROLE {role_identifier}")  # nosec B608
        conn.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (owner,),
        )
        created = db.task_store.apply_sync_task_create(
            owner_user_id=owner,
            dataset_id=_DATASET_ID,
            payload=created_payload,
            canonical_revision=1,
            canonical_hash=created_hash,
            conn=conn,
        )

    updated_payload = _payload(
        owner=owner,
        task_id=task_id,
        note_id=note_id,
        title="Updated on PostgreSQL",
        status="done",
        completed_at="2026-08-21T12:00:00+00:00",
    )
    updated_hash = notes_task_object_hash(
        updated_payload,
        revision=2,
        deleted=False,
    )
    with db.transaction() as conn:
        conn.execute(f"SET LOCAL ROLE {role_identifier}")  # nosec B608
        conn.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (owner,),
        )
        updated = db.task_store.apply_sync_task_upsert(
            owner_user_id=owner,
            dataset_id=_DATASET_ID,
            payload=updated_payload,
            base_revision=1,
            base_hash=created_hash,
            canonical_revision=2,
            canonical_hash=updated_hash,
            conn=conn,
        )

    tombstone_hash = notes_task_object_hash(
        updated_payload,
        revision=3,
        deleted=True,
    )
    with db.transaction() as conn:
        conn.execute(f"SET LOCAL ROLE {role_identifier}")  # nosec B608
        conn.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (owner,),
        )
        tombstone = db.task_store.apply_sync_task_tombstone(
            owner_user_id=owner,
            dataset_id=_DATASET_ID,
            payload=updated_payload,
            base_revision=2,
            base_hash=updated_hash,
            canonical_revision=3,
            canonical_hash=tombstone_hash,
            conn=conn,
        )

    restored_hash = notes_task_object_hash(
        updated_payload,
        revision=4,
        deleted=False,
    )
    with db.transaction() as conn:
        conn.execute(f"SET LOCAL ROLE {role_identifier}")  # nosec B608
        conn.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (owner,),
        )
        restored = db.task_store.apply_sync_task_restore(
            owner_user_id=owner,
            dataset_id=_DATASET_ID,
            payload=updated_payload,
            base_revision=3,
            base_hash=tombstone_hash,
            canonical_revision=4,
            canonical_hash=restored_hash,
            conn=conn,
        )

    assert created["canonical_revision"] == 1
    assert updated["canonical_revision"] == 2
    assert bool(tombstone["deleted"]) is True
    assert bool(restored["deleted"]) is False
    assert restored["projection_status"] == "unlinked"
    assert restored["canonical_hash"] == restored_hash
    return restored


def test_postgres_task_sources_bind_scope_lock_cas_and_page_by_primary_key() -> None:
    fetch_source = inspect.getsource(TaskStore._fetch_task)
    transition_source = inspect.getsource(TaskStore._apply_sync_task_transition)
    page_source = inspect.getsource(TaskStore.page_tasks_for_sync_bootstrap)
    materializer_source = inspect.getsource(NotesTaskMaterializer.apply)
    ddl = " ".join(CharactersRAGDB._note_task_v60_postgres_ddl())

    assert "WHERE owner_user_id = ? AND dataset_id = ? AND id = ?" in fetch_source
    assert "for_update=True" in transition_source
    assert "FOR UPDATE" in fetch_source
    assert "WHERE owner_user_id = ? AND dataset_id = ?" in page_source
    assert "AND id > ? ORDER BY id ASC LIMIT ?" in page_source
    assert "PRIMARY KEY(owner_user_id,dataset_id,id)" in ddl
    assert '"owner_user_id": str(self.note_db.client_id)' in materializer_source
    assert '"dataset_id": envelope.dataset_id' in materializer_source


def test_postgres_task_lifecycle_is_rls_isolated_and_index_backed_for_two_owners(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_a = "970001"
    owner_b = "970002"
    backend_a = DatabaseBackendFactory.create_backend(pg_database_config)
    backend_b = DatabaseBackendFactory.create_backend(pg_database_config)
    db_a = CharactersRAGDB(":memory:", client_id=owner_a, backend=backend_a)
    db_b = CharactersRAGDB(":memory:", client_id=owner_b, backend=backend_b)
    ident = backend_a.escape_identifier  # type: ignore[attr-defined]
    role_name = f"notes_task_lifecycle_{uuid4().hex[:8]}"
    role_identifier = ident(role_name)
    role_created = False

    try:
        with backend_a.transaction() as conn:
            backend_a.execute(
                f"CREATE ROLE {role_identifier} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT USAGE ON SCHEMA public TO {role_identifier}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT ON notes TO {role_identifier}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT, INSERT, UPDATE ON note_tasks TO {role_identifier}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT SELECT, UPDATE ON note_task_scope_authority TO {role_identifier}",
                connection=conn,
            )
            backend_a.execute(
                f"GRANT {role_identifier} TO CURRENT_USER",
                connection=conn,
            )
        role_created = True

        with db_a.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role_identifier}")  # nosec B608
            role = conn.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
            ).fetchone()
            assert role is not None
            assert bool(role["rolsuper"]) is False
            assert bool(role["rolbypassrls"]) is False

        task_a = _apply_lifecycle(
            db_a,
            owner=owner_a,
            role_identifier=role_identifier,
        )
        task_b = _apply_lifecycle(
            db_b,
            owner=owner_b,
            role_identifier=role_identifier,
        )

        assert db_a.get_task(
            owner_user_id=owner_a,
            dataset_id=_DATASET_ID,
            task_id=str(task_b["id"]),
            include_deleted=True,
        ) is None
        assert db_b.get_task(
            owner_user_id=owner_b,
            dataset_id=_DATASET_ID,
            task_id=str(task_a["id"]),
            include_deleted=True,
        ) is None

        with db_a.transaction() as conn:
            conn.execute("ANALYZE note_tasks")
            conn.execute("SET LOCAL enable_seqscan=off")
            plan_rows = conn.execute(
                "EXPLAIN SELECT id FROM note_tasks "
                "WHERE owner_user_id=? AND dataset_id=? AND id>? "
                "ORDER BY id ASC LIMIT ?",
                (owner_a, _DATASET_ID, "", 50),
            ).fetchall()
        plan = " ".join(str(next(iter(dict(row).values()))) for row in plan_rows)
        assert "note_tasks_pkey" in plan
    finally:
        if role_created:
            with backend_a.transaction() as conn:
                backend_a.execute(
                    f"REVOKE {role_identifier} FROM CURRENT_USER",
                    connection=conn,
                )
                backend_a.execute(
                    f"DROP OWNED BY {role_identifier}",
                    connection=conn,
                )
                backend_a.execute(
                    f"DROP ROLE {role_identifier}",
                    connection=conn,
                )
        db_b.close_all_connections()
        db_a.close_all_connections()
        backend_b.get_pool().close_all()
        backend_a.get_pool().close_all()
