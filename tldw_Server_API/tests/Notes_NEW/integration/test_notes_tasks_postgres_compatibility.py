"""Live PostgreSQL continuity for product-owned Notes task compatibility scope."""

from __future__ import annotations

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule
from tldw_Server_API.app.core.Notes_Tasks import service as notes_task_service_module
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import (
    NotesTaskService,
    resolve_task_compatibility_scope,
)

pytestmark = pytest.mark.integration

LOCAL_UNBOUND = "local-unbound"


class _NoopRateLimiter:
    async def check_user_rate_limit(self, *_args: Any, **_kwargs: Any) -> tuple[bool, dict[str, Any]]:
        return True, {}


def _bound_task_graph(
    pg_database_config: DatabaseConfig,
    *,
    owner: str,
) -> tuple[Any, CharactersRAGDB, NotesTaskService, str, dict[str, Any], str]:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
    service = NotesTaskService()
    note_id = str(uuid4())
    db.add_note("Bound task", "- [ ] Preserve continuity\n", note_id=note_id)
    service.reconcile_note_current(
        db=db,
        note_id=note_id,
        owner_user_id=owner,
        actor=TaskActor(actor_type="user", actor_id=owner),
    )
    tasks = db.list_tasks(
        owner_user_id=owner,
        dataset_id=LOCAL_UNBOUND,
        note_id=note_id,
        limit=10,
    )
    assert len(tasks) == 1
    target = f"dataset-{uuid4()}"
    db.bind_local_task_graph_to_dataset(
        owner_user_id=owner,
        target_dataset_id=target,
    )
    return backend, db, service, note_id, tasks[0], target


def _assert_only_target_task(
    db: CharactersRAGDB,
    *,
    owner: str,
    target: str,
    task_id: str,
) -> dict[str, Any]:
    assert (
        db.list_tasks(
            owner_user_id=owner,
            dataset_id=LOCAL_UNBOUND,
            include_deleted=True,
            limit=10,
        )
        == []
    )
    for table, _ordering in db.task_store._BIND_TABLE_ORDER:
        # Table names come only from the fixed class-level task graph list.
        row = db.execute_query(
            f"SELECT COUNT(*) AS row_count FROM {table} "  # nosec B608
            "WHERE owner_user_id = ? AND dataset_id = ?",
            (owner, LOCAL_UNBOUND),
        ).fetchone()
        assert int(row["row_count"]) == 0, table
    rows = db.list_tasks(
        owner_user_id=owner,
        dataset_id=target,
        include_deleted=True,
        limit=10,
    )
    assert [row["id"] for row in rows] == [task_id]
    return rows[0]


def test_postgres_empty_bind_persists_authority_before_first_task(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960000"
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id=owner, backend=backend)
    service = NotesTaskService()
    try:
        assert (
            resolve_task_compatibility_scope(
                db,
                authenticated_owner_user_id=owner,
            ).dataset_id
            == LOCAL_UNBOUND
        )

        target = f"dataset-{uuid4()}"
        zero_counts = {
            table: 0 for table, _ordering in db.task_store._BIND_TABLE_ORDER
        }
        assert db.bind_local_task_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        ) == zero_counts
        assert db.bind_local_task_graph_to_dataset(
            owner_user_id=owner,
            target_dataset_id=target,
        ) == zero_counts
        with pytest.raises(ConflictError, match="immutable"):
            db.bind_local_task_graph_to_dataset(
                owner_user_id=owner,
                target_dataset_id=f"dataset-{uuid4()}",
            )

        note_id = str(uuid4())
        db.add_note("Bound-first task", "- [ ] Stay bound\n", note_id=note_id)
        service.reconcile_note_current(
            db=db,
            note_id=note_id,
            owner_user_id=owner,
            actor=TaskActor(actor_type="user", actor_id=owner),
        )
        assert (
            resolve_task_compatibility_scope(
                db,
                authenticated_owner_user_id=owner,
            ).dataset_id
            == target
        )
        assert len(db.list_tasks(
            owner_user_id=owner,
            dataset_id=target,
            note_id=note_id,
            limit=10,
        )) == 1
        assert db.list_tasks(
            owner_user_id=owner,
            dataset_id=LOCAL_UNBOUND,
            note_id=note_id,
            limit=10,
        ) == []
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_service_updates_bound_task_without_recreating_sentinel_rows(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960001"
    backend, db, service, note_id, task, target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )
    try:
        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner,
        )
        assert scope.dataset_id == target

        updated = service.update_task(
            db=db,
            owner_user_id=owner,
            task_id=str(task["id"]),
            expected_task_version=int(task["version"]),
            expected_note_version=1,
            text="Continuity preserved",
            actor=TaskActor(actor_type="user", actor_id=owner),
        )

        assert updated["text"] == "Continuity preserved"
        assert db.get_note_by_id(note_id)["content"] == "- [ ] Continuity preserved\n"
        assert (
            _assert_only_target_task(
                db,
                owner=owner,
                target=target,
                task_id=str(task["id"]),
            )["text"]
            == "Continuity preserved"
        )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_projected_update_serializes_before_same_target_bind(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960007"
    backend, db, service, _note_id, task, target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )
    bind_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    bind_db = CharactersRAGDB(":memory:", client_id=owner, backend=bind_backend)
    write_entered = threading.Event()
    allow_write = threading.Event()
    original_write = notes_task_service_module._write_note_content

    def paused_write(*args: Any, **kwargs: Any) -> None:
        write_entered.set()
        assert allow_write.wait(30), "projected update was not released"
        original_write(*args, **kwargs)

    monkeypatch.setattr(notes_task_service_module, "_write_note_content", paused_write)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            update_future = executor.submit(
                service.update_task,
                db=db,
                owner_user_id=owner,
                task_id=str(task["id"]),
                expected_task_version=int(task["version"]),
                expected_note_version=1,
                text="Serialized update",
                actor=TaskActor(actor_type="user", actor_id=owner),
            )
            assert write_entered.wait(30), "projected update did not reach the note write"
            bind_future = executor.submit(
                bind_db.bind_local_task_graph_to_dataset,
                owner_user_id=owner,
                target_dataset_id=target,
            )
            try:
                with pytest.raises(FutureTimeoutError):
                    bind_future.result(timeout=0.25)
            finally:
                allow_write.set()

            assert update_future.result(timeout=30)["text"] == "Serialized update"
            counts = bind_future.result(timeout=30)
            assert counts["note_tasks"] == 1
    finally:
        allow_write.set()
        bind_db.close_all_connections()
        bind_backend.get_pool().close_all()
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_rest_reads_and_updates_bound_task_without_dataset_selector(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960002"
    backend, db, _service, note_id, task, target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )

    async def override_user() -> User:
        return User(
            id=int(owner),
            username="pg-task-owner",
            email="pg-task-owner@example.com",
            is_active=True,
            is_admin=True,
        )

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_user,
    )

    app = FastAPI()
    endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.notes_tasks")
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()

    try:
        with TestClient(app) as client:
            fetched = client.get(f"/api/v1/notes/tasks/{task['id']}")
            assert fetched.status_code == 200, fetched.text
            assert fetched.json()["text"] == "Preserve continuity"
            assert fetched.json()["created_at"] == task["created_at"].isoformat()

            updated = client.patch(
                f"/api/v1/notes/tasks/{task['id']}",
                json={
                    "text": "REST continuity",
                    "expected_task_version": task["version"],
                    "expected_note_version": 1,
                },
            )
            assert updated.status_code == 200, updated.text
            assert updated.json()["text"] == "REST continuity"

        assert db.get_note_by_id(note_id)["content"] == "- [ ] REST continuity\n"
        assert (
            _assert_only_target_task(
                db,
                owner=owner,
                target=target,
                task_id=str(task["id"]),
            )["text"]
            == "REST continuity"
        )
    finally:
        app.dependency_overrides.clear()
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_rest_activity_after_bind_serializes_read_and_dismiss_timestamps(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960006"
    backend, db, _service, note_id, task, target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )
    event = db.record_task_event(
        owner_user_id=owner,
        dataset_id=target,
        task_id=str(task["id"]),
        note_id=note_id,
        event_type="updated",
        actor_type="agent",
        actor_id="assistant",
    )

    async def override_user() -> User:
        return User(
            id=int(owner),
            username="pg-task-activity-owner",
            email="pg-task-activity-owner@example.com",
            is_active=True,
            is_admin=True,
        )

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
        get_chacha_db_for_user,
    )

    app = FastAPI()
    endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.notes_tasks")
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()

    try:
        with TestClient(app) as client:
            activity = client.get("/api/v1/notes/tasks/activity")
            assert activity.status_code == 200, activity.text
            assert activity.json()["events"][0]["id"] == event["id"]
            assert activity.json()["events"][0]["created_at"] == event["created_at"].isoformat()

            read = client.patch(
                f"/api/v1/notes/tasks/activity/{event['id']}",
                json={"read": True},
            )
            assert read.status_code == 200, read.text
            read_state = db.get_task_activity_read_state(
                owner_user_id=owner,
                dataset_id=target,
                event_id=str(event["id"]),
                user_id=owner,
            )
            assert read_state is not None
            assert read.json()["read_at"] == read_state["read_at"].isoformat()
            assert read.json()["dismissed_at"] is None

            dismissed = client.patch(
                f"/api/v1/notes/tasks/activity/{event['id']}",
                json={"dismissed": True},
            )
            assert dismissed.status_code == 200, dismissed.text
            dismissed_state = db.get_task_activity_read_state(
                owner_user_id=owner,
                dataset_id=target,
                event_id=str(event["id"]),
                user_id=owner,
            )
            assert dismissed_state is not None
            assert dismissed.json()["read_at"] == dismissed_state["read_at"].isoformat()
            assert dismissed.json()["dismissed_at"] == dismissed_state["dismissed_at"].isoformat()
    finally:
        app.dependency_overrides.clear()
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.asyncio
async def test_postgres_mcp_reads_and_updates_bound_task_without_dataset_selector(
    pg_database_config: DatabaseConfig,
) -> None:
    owner = "960003"
    backend, db, _service, note_id, task, target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )
    module = NotesModule(ModuleConfig(name="notes"))
    module._open_db = lambda _context: db  # type: ignore[method-assign]
    module._close_task_db = lambda _db, _operation: None  # type: ignore[method-assign]
    context = SimpleNamespace(
        request_id="req-pg-task",
        user_id=owner,
        client_id="pg-task-client",
        session_id="pg-task-session",
        metadata={},
    )

    try:
        fetched = await module.execute_tool(
            "notes.tasks.get",
            {"task_id": task["id"]},
            context=context,
        )
        assert fetched["text"] == "Preserve continuity"

        updated = await module.execute_tool(
            "notes.tasks.update",
            {
                "task_id": task["id"],
                "text": "MCP continuity",
                "expected_task_version": task["version"],
                "expected_note_version": 1,
            },
            context=context,
        )
        assert updated["text"] == "MCP continuity"
        assert db.get_note_by_id(note_id)["content"] == "- [ ] MCP continuity\n"
        assert (
            _assert_only_target_task(
                db,
                owner=owner,
                target=target,
                task_id=str(task["id"]),
            )["text"]
            == "MCP continuity"
        )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_compatibility_resolver_performs_no_graph_lock_or_ddl(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "960005"
    backend, db, _service, _note_id, _task, _target = _bound_task_graph(
        pg_database_config,
        owner=owner,
    )
    queries: list[str] = []
    original_read = db.task_store._read

    def record_read(query: str, *args: Any, **kwargs: Any) -> Any:
        queries.append(query)
        return original_read(query, *args, **kwargs)

    monkeypatch.setattr(db.task_store, "_read", record_read)
    monkeypatch.setattr(
        db.task_store,
        "_execute",
        lambda *_args, **_kwargs: pytest.fail("resolver must not execute DDL or locks"),
    )
    try:
        assert resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=owner,
        ).dataset_id == _target
        rendered = " ".join(queries).upper()
        assert "NOTE_TASK_SCOPE_AUTHORITY" in rendered
        assert "LOCK TABLE" not in rendered
        assert "ALTER TABLE" not in rendered
        assert "SELECT DISTINCT" not in rendered
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
