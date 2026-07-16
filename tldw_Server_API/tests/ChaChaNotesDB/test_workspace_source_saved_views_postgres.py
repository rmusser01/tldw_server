"""PostgreSQL persistence and isolation tests for workspace source saved views."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import ensure_chacha_rls
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)

pytestmark = pytest.mark.integration

OWNER_A = "owner-a"
OWNER_B = "owner-b"


def _database(
    config: DatabaseConfig,
    *,
    owner: str = OWNER_A,
) -> tuple[Any, CharactersRAGDB]:
    backend = DatabaseBackendFactory.create_backend(config)
    return backend, CharactersRAGDB(db_path=":memory:", client_id=owner, backend=backend)


def _create(
    db: CharactersRAGDB,
    workspace_id: str,
    *,
    owner: str = OWNER_A,
    name: str = "Saved view",
    schema_version: int = 1,
    state_json: str = '{"schema_version":1}',
) -> dict[str, Any]:
    return db.create_workspace_source_saved_view(
        owner,
        workspace_id,
        name=name,
        schema_version=schema_version,
        state_json=state_json,
    )


def _policy_state(backend: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    relation = list(
        backend.execute(
            """
            SELECT relrowsecurity, relforcerowsecurity
              FROM pg_class
             WHERE oid = to_regclass('workspace_source_saved_views')
            """
        )
    )[0]
    policy = list(
        backend.execute(
            """
            SELECT qual, with_check
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND tablename = 'workspace_source_saved_views'
               AND policyname = 'workspace_source_saved_views_tenant_isolation'
            """
        )
    )[0]
    return relation, policy


def _assert_forced_active_workspace_policy(backend: Any) -> None:
    relation, policy = _policy_state(backend)
    assert relation["relrowsecurity"] is True
    assert relation["relforcerowsecurity"] is True
    for expression in (policy["qual"], policy["with_check"]):
        assert expression
        assert "owner_user_id" in expression
        assert "app.current_user_id" in expression
        assert "workspace_source_saved_views.workspace_id" in expression
        assert "w.client_id" in expression
        assert "NOT w.deleted" in expression or "w.deleted = false" in expression


def test_postgres_missing_workspace_route_leaves_connection_idle(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _database(pg_database_config, owner="1")
    try:
        connection = db.get_connection()
        connection.rollback()
        assert connection.info.transaction_status.name == "IDLE"

        with pytest.raises(HTTPException) as exc_info:
            workspaces_endpoint.list_source_saved_views(
                "missing-workspace",
                db=db,
                current_user=SimpleNamespace(id=1),
            )

        assert exc_info.value.status_code == 404
        assert connection.info.transaction_status.name == "IDLE"
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_fresh_schema_named_unique_crud_order_and_owner_predicates(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _database(pg_database_config)
    owner_b = CharactersRAGDB(db_path=":memory:", client_id=OWNER_B, backend=backend)
    workspace_a = "ws-a"
    workspace_b = "ws-b"
    try:
        db.upsert_workspace(workspace_a, "Workspace A")
        owner_b.upsert_workspace(workspace_b, "Workspace B")
        assert backend.table_exists("workspace_source_saved_views")
        constraint = list(
            backend.execute(
                """
                SELECT conname, pg_get_constraintdef(oid) AS definition
                  FROM pg_constraint
                 WHERE conrelid = 'workspace_source_saved_views'::regclass
                   AND conname = 'uq_workspace_source_saved_views_owner_name'
                """
            )
        )
        assert len(constraint) == 1
        assert "UNIQUE (owner_user_id, workspace_id, name_key)" in constraint[0]["definition"]
        _assert_forced_active_workspace_policy(backend)

        zulu = _create(db, workspace_a, name="Zulu")
        alpha = _create(db, workspace_a, name="Alpha")
        db.execute_query(
            "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00.000Z", zulu["id"]),
            commit=True,
        )
        db.execute_query(
            "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id = ?",
            ("2026-01-02T00:00:00.000Z", alpha["id"]),
            commit=True,
        )
        assert [row["id"] for row in db.list_workspace_source_saved_views(OWNER_A, workspace_a)] == [
            alpha["id"],
            zulu["id"],
        ]
        updated = db.update_workspace_source_saved_view(
            OWNER_A,
            workspace_a,
            zulu["id"],
            expected_version=1,
            name="Bravo",
            schema_version=9,
            state_json="raw pg state",
        )
        assert updated["version"] == 2
        assert updated["state_json"] == "raw pg state"
        assert db.get_workspace_source_saved_view(OWNER_A, workspace_a, zulu["id"]) == updated

        with pytest.raises(CharactersRAGDBError) as exc_info:
            owner_b.get_workspace_source_saved_view(OWNER_B, workspace_a, zulu["id"])
        assert exc_info.value.code == "source_view_not_found"
        assert exc_info.value.metadata == {}
        db.delete_workspace_source_saved_view(OWNER_A, workspace_a, zulu["id"])
        assert [row["id"] for row in db.list_workspace_source_saved_views(OWNER_A, workspace_a)] == [alpha["id"]]
    finally:
        db.close_all_connections()
        owner_b.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_v53_migration_creates_table_and_forced_policy_immediately(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, seed = _database(pg_database_config)
    try:
        with backend.transaction() as conn:
            backend.execute("DROP TABLE workspace_source_saved_views", connection=conn)
            backend.execute(
                "UPDATE db_schema_version SET version = 53 WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
                connection=conn,
            )
        seed.close_connection()

        migrated = CharactersRAGDB(db_path=":memory:", client_id=OWNER_A, backend=backend)
        version = list(
            backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            )
        )[0]["version"]

        assert version == 54
        assert backend.table_exists("workspace_source_saved_views")
        _assert_forced_active_workspace_policy(backend)
    finally:
        seed.close_all_connections()
        if "migrated" in locals():
            migrated.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_concurrent_count_duplicate_and_rename_conflicts_have_safe_metadata(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = "ws-concurrency"
    db.upsert_workspace(workspace_id, "Concurrency")

    def run_two(workers: tuple[Callable[[], object], Callable[[], object]]) -> list[object]:
        barrier = threading.Barrier(2)
        results: list[object] = []
        lock = threading.Lock()

        def run(worker: Callable[[], object]) -> None:
            barrier.wait(timeout=5)
            try:
                result: object = worker()
            except Exception as exc:  # noqa: BLE001 - caller asserts the captured conflict
                result = exc
            finally:
                db.close_connection()
            with lock:
                results.append(result)

        threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()
        return results

    try:
        duplicate_results = run_two(
            (
                lambda: _create(db, workspace_id, name="Stra\u00dfe"),
                lambda: _create(db, workspace_id, name="STRASSE"),
            )
        )
        winner = next(result for result in duplicate_results if isinstance(result, dict))
        duplicate = next(result for result in duplicate_results if isinstance(result, CharactersRAGDBError))
        assert duplicate.code == "source_view_name_exists"
        assert duplicate.metadata == {"view_id": winner["id"], "version": 1}

        left = _create(db, workspace_id, name="Left")
        right = _create(db, workspace_id, name="Right")
        rename_results = run_two(
            (
                lambda: db.update_workspace_source_saved_view(
                    OWNER_A,
                    workspace_id,
                    left["id"],
                    expected_version=1,
                    name="Target",
                ),
                lambda: db.update_workspace_source_saved_view(
                    OWNER_A,
                    workspace_id,
                    right["id"],
                    expected_version=1,
                    name="target",
                ),
            )
        )
        rename_winner = next(result for result in rename_results if isinstance(result, dict))
        rename_conflict = next(result for result in rename_results if isinstance(result, CharactersRAGDBError))
        assert rename_conflict.code == "source_view_name_exists"
        assert rename_conflict.metadata == {"view_id": rename_winner["id"], "version": 2}

        now = "2026-01-01T00:00:00.000Z"
        with db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO workspace_source_saved_views (
                    id, workspace_id, owner_user_id, name, name_key, schema_version,
                    state_json, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, '{}', 1, ?, ?)
                """,
                [
                    (str(uuid4()), workspace_id, OWNER_A, f"Seed {index}", f"seed {index}", now, now)
                    for index in range(96)
                ],
            )
        limit_results = run_two(
            (
                lambda: _create(db, workspace_id, name="Hundred A"),
                lambda: _create(db, workspace_id, name="Hundred B"),
            )
        )
        assert sum(isinstance(result, dict) for result in limit_results) == 1
        limit_conflict = next(result for result in limit_results if isinstance(result, CharactersRAGDBError))
        assert limit_conflict.code == "source_view_limit_reached"
        assert limit_conflict.metadata == {"limit": 100}
        assert len(db.list_workspace_source_saved_views(OWNER_A, workspace_id)) == 100

        with pytest.raises(CharactersRAGDBError) as duplicate_at_capacity:
            _create(db, workspace_id, name="STRASSE")
        assert duplicate_at_capacity.value.code == "source_view_name_exists"
        assert duplicate_at_capacity.value.metadata == {"view_id": winner["id"], "version": 1}
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["create", "update"])
def test_postgres_named_unique_recovery_uses_independent_connection_when_nested(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = f"ws-named-unique-{operation}"
    db.upsert_workspace(workspace_id, "Named unique recovery")
    conflict = _create(db, workspace_id, name="Conflict")
    conflict = db.update_workspace_source_saved_view(
        OWNER_A,
        workspace_id,
        conflict["id"],
        expected_version=1,
        state_json="conflicting row version two",
        schema_version=2,
    )
    candidate = _create(db, workspace_id, name="Candidate")
    original_find = db._find_workspace_source_saved_view_name_with_conn
    original_detect = db._is_workspace_source_saved_view_postgres_unique_error
    find_connection_ids: list[int] = []
    detected_errors: list[tuple[str | None, str | None]] = []

    def stale_once(
        conn: Any,
        owner_user_id: str,
        scoped_workspace_id: str,
        name_key: str,
        *,
        exclude_view_id: str | None = None,
    ) -> dict[str, Any] | None:
        raw_connection = getattr(conn, "_connection", conn)
        find_connection_ids.append(id(raw_connection))
        if len(find_connection_ids) == 1:
            return None
        return original_find(
            conn,
            owner_user_id,
            scoped_workspace_id,
            name_key,
            exclude_view_id=exclude_view_id,
        )

    def record_named_unique(exc: Exception) -> bool:
        current: BaseException | None = exc
        while current is not None:
            sqlstate = getattr(current, "sqlstate", None) or getattr(current, "pgcode", None)
            diagnostics = getattr(current, "diag", None)
            constraint_name = getattr(diagnostics, "constraint_name", None)
            if sqlstate is not None or constraint_name is not None:
                detected_errors.append((sqlstate, constraint_name))
            current = current.__cause__
        return original_detect(exc)

    monkeypatch.setattr(db, "_find_workspace_source_saved_view_name_with_conn", stale_once)
    monkeypatch.setattr(db, "_is_workspace_source_saved_view_postgres_unique_error", record_named_unique)

    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            with db.transaction() as outer_conn:
                outer_conn.execute(
                    "UPDATE workspaces SET description = ? WHERE id = ?",
                    ("must roll back", workspace_id),
                )
                if operation == "create":
                    _create(db, workspace_id, name="CONFLICT")
                else:
                    db.update_workspace_source_saved_view(
                        OWNER_A,
                        workspace_id,
                        candidate["id"],
                        expected_version=1,
                        name="CONFLICT",
                    )

        assert exc_info.value.code == "source_view_name_exists"
        assert exc_info.value.metadata == {"view_id": conflict["id"], "version": 2}
        assert (
            "23505",
            "uq_workspace_source_saved_views_owner_name",
        ) in detected_errors
        assert len(find_connection_ids) == 2
        assert find_connection_ids[0] != find_connection_ids[1]
        unchanged = db.get_workspace_source_saved_view(OWNER_A, workspace_id, candidate["id"])
        assert unchanged["name"] == "Candidate"
        assert unchanged["version"] == 1
        workspace = db.get_workspace(workspace_id)
        assert workspace is not None
        assert workspace["description"] is None
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("outcome", ["conflict", "not_found"])
def test_postgres_recovery_closes_idle_direct_connection_before_domain_error(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = f"ws-recovery-cleanup-{outcome}"
    db.upsert_workspace(workspace_id, "Recovery cleanup")
    conflict = _create(db, workspace_id, name="Conflict")
    original_connect = backend.connect
    original_disconnect = backend.disconnect
    recovery_connections: list[Any] = []
    disconnect_states: list[tuple[str | None, bool]] = []

    def tracked_connect() -> Any:
        conn = original_connect()
        recovery_connections.append(conn)
        return conn

    def tracked_disconnect(conn: Any) -> None:
        status_name = getattr(conn.info.transaction_status, "name", None)
        original_disconnect(conn)
        disconnect_states.append((status_name, bool(conn.closed)))

    monkeypatch.setattr(backend, "connect", tracked_connect)
    monkeypatch.setattr(backend, "disconnect", tracked_disconnect)
    name_key = conflict["name_key"] if outcome == "conflict" else "missing"

    try:
        for _ in range(3):
            with pytest.raises(CharactersRAGDBError) as exc_info:
                db._raise_workspace_source_saved_view_duplicate_from_fresh_transaction(
                    OWNER_A,
                    workspace_id,
                    name_key,
                )
            if outcome == "conflict":
                assert exc_info.value.code == "source_view_name_exists"
                assert exc_info.value.metadata == {"view_id": conflict["id"], "version": 1}
            else:
                assert exc_info.value.code == "source_view_not_found"
                assert exc_info.value.metadata == {}

        assert len(recovery_connections) == 3
        assert disconnect_states == [("IDLE", True)] * 3
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_nested_duplicate_recovery_does_not_borrow_pool_capacity(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one_connection_config = replace(
        pg_database_config,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    backend, db = _database(one_connection_config)
    workspace_id = "ws-pool-one-recovery"
    db.upsert_workspace(workspace_id, "Pool one recovery")
    conflict = _create(db, workspace_id, name="Conflict")
    original_find = db._find_workspace_source_saved_view_name_with_conn
    original_pool_get = backend.get_pool().get_connection
    original_disconnect = backend.disconnect
    find_calls = 0
    pool_borrows = 0
    disconnect_states: list[tuple[str | None, bool]] = []

    def stale_preflight(
        conn: Any,
        owner_user_id: str,
        scoped_workspace_id: str,
        name_key: str,
        *,
        exclude_view_id: str | None = None,
    ) -> dict[str, Any] | None:
        nonlocal find_calls
        find_calls += 1
        if find_calls % 2 == 1:
            return None
        return original_find(
            conn,
            owner_user_id,
            scoped_workspace_id,
            name_key,
            exclude_view_id=exclude_view_id,
        )

    def tracked_pool_get() -> Any:
        nonlocal pool_borrows
        pool_borrows += 1
        return original_pool_get()

    def tracked_disconnect(conn: Any) -> None:
        status_name = getattr(conn.info.transaction_status, "name", None)
        original_disconnect(conn)
        disconnect_states.append((status_name, bool(conn.closed)))

    monkeypatch.setattr(db, "_find_workspace_source_saved_view_name_with_conn", stale_preflight)
    monkeypatch.setattr(backend.get_pool(), "get_connection", tracked_pool_get)
    monkeypatch.setattr(backend, "disconnect", tracked_disconnect)

    try:
        for attempt in range(2):
            with pytest.raises(CharactersRAGDBError) as exc_info:
                with db.transaction() as outer_conn:
                    outer_conn.execute(
                        "UPDATE workspaces SET description = ? WHERE id = ?",
                        (f"must roll back {attempt}", workspace_id),
                    )
                    _create(db, workspace_id, name="CONFLICT")
            assert exc_info.value.code == "source_view_name_exists"
            assert exc_info.value.metadata == {"view_id": conflict["id"], "version": 1}

        assert pool_borrows == 0
        assert disconnect_states == [("IDLE", True)] * 2
        workspace = db.get_workspace(workspace_id)
        assert workspace is not None
        assert workspace["description"] is None
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "nul\x00name"),
        ("name", "surrogate\ud800name"),
        ("state_json", "nul\x00state"),
        ("state_json", "surrogate\ud800state"),
    ],
)
def test_postgres_saved_view_text_validation_is_driver_independent(
    pg_database_config: DatabaseConfig,
    field: str,
    value: str,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = f"ws-text-validation-{field}-{uuid4()}"
    try:
        db.upsert_workspace(workspace_id, "Text validation")

        with pytest.raises(InputError, match=field):
            _create(db, workspace_id, **{field: value})
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_saved_view_integer_boundaries_are_driver_independent(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = "ws-integer-validation"
    try:
        db.upsert_workspace(workspace_id, "Integer validation")
        created = db.create_workspace_source_saved_view(
            OWNER_A,
            workspace_id,
            name="Maximum schema",
            schema_version=2_147_483_647,
            state_json=r'{"value":"\u0000"}',
        )
        assert created["schema_version"] == 2_147_483_647
        assert created["state_json"] == r'{"value":"\u0000"}'

        for value in (True, 0, -1, 2_147_483_648, 1.5):
            with pytest.raises(InputError, match="schema_version"):
                db.create_workspace_source_saved_view(
                    OWNER_A,
                    workspace_id,
                    name=f"Invalid schema {value!r}",
                    schema_version=value,  # type: ignore[arg-type]
                    state_json="{}",
                )
            with pytest.raises(InputError, match="expected_version"):
                db.update_workspace_source_saved_view(
                    OWNER_A,
                    workspace_id,
                    created["id"],
                    expected_version=value,  # type: ignore[arg-type]
                    name="Updated",
                )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_update_rejects_maximum_stored_version_without_mutating_row(
    pg_database_config: DatabaseConfig,
) -> None:
    maximum = 2_147_483_647
    backend, db = _database(pg_database_config)
    workspace_id = "ws-maximum-saved-view-version"
    try:
        db.upsert_workspace(workspace_id, "Maximum saved view version")
        created = _create(db, workspace_id, name="Maximum version")
        db.execute_query(
            "UPDATE workspace_source_saved_views SET version = ? WHERE id = ?",
            (maximum, created["id"]),
            commit=True,
        )

        with pytest.raises(CharactersRAGDBError) as stale_exc:
            db.update_workspace_source_saved_view(
                OWNER_A,
                workspace_id,
                created["id"],
                expected_version=maximum - 1,
                name="Stale change",
            )
        assert stale_exc.value.code == "source_view_version_conflict"
        assert stale_exc.value.metadata == {
            "view_id": created["id"],
            "current_version": maximum,
        }

        with pytest.raises(InputError, match="maximum"):
            db.update_workspace_source_saved_view(
                OWNER_A,
                workspace_id,
                created["id"],
                expected_version=maximum,
                name="Overflowing change",
            )

        current = db.get_workspace_source_saved_view(OWNER_A, workspace_id, created["id"])
        assert current["version"] == maximum
        assert current["name"] == "Maximum version"
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
@pytest.mark.parametrize("delete_first", [False, True], ids=["mutation-first", "delete-first"])
def test_postgres_saved_view_mutations_serialize_with_workspace_soft_delete(
    pg_database_config: DatabaseConfig,
    operation: str,
    delete_first: bool,
) -> None:
    backend, db = _database(pg_database_config)
    workspace_id = f"ws-{operation}-{delete_first}"
    db.upsert_workspace(workspace_id, "Workspace")
    existing = _create(db, workspace_id)
    first_done = threading.Event()
    allow_commit = threading.Event()
    results: dict[str, object] = {}

    def mutate() -> None:
        try:
            if delete_first:
                assert first_done.wait(timeout=5)
                if operation == "create":
                    results["mutation"] = _create(db, workspace_id, name="Concurrent")
                elif operation == "update":
                    results["mutation"] = db.update_workspace_source_saved_view(
                        OWNER_A,
                        workspace_id,
                        existing["id"],
                        expected_version=1,
                        name="Concurrent",
                    )
                else:
                    results["mutation"] = db.delete_workspace_source_saved_view(
                        OWNER_A,
                        workspace_id,
                        existing["id"],
                    )
            else:
                with db.transaction():
                    if operation == "create":
                        results["mutation"] = _create(db, workspace_id, name="Concurrent")
                    elif operation == "update":
                        results["mutation"] = db.update_workspace_source_saved_view(
                            OWNER_A,
                            workspace_id,
                            existing["id"],
                            expected_version=1,
                            name="Concurrent",
                        )
                    else:
                        results["mutation"] = db.delete_workspace_source_saved_view(
                            OWNER_A,
                            workspace_id,
                            existing["id"],
                        )
                    first_done.set()
                    assert allow_commit.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001 - thread boundary surfaces the exact error
            results["mutation_error"] = exc
        finally:
            db.close_connection()

    def soft_delete() -> None:
        try:
            if not delete_first:
                assert first_done.wait(timeout=5)
                workspace = db.get_workspace(workspace_id)
                assert workspace is not None
                results["deleted"] = db.delete_workspace(
                    workspace_id,
                    expected_version=workspace["version"],
                )
            else:
                with db.transaction():
                    workspace = db.get_workspace(workspace_id)
                    assert workspace is not None
                    results["deleted"] = db.delete_workspace(
                        workspace_id,
                        expected_version=workspace["version"],
                    )
                    first_done.set()
                    assert allow_commit.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001 - thread boundary surfaces the exact error
            results["delete_error"] = exc
        finally:
            db.close_connection()

    mutation_thread = threading.Thread(target=mutate)
    deletion_thread = threading.Thread(target=soft_delete)
    try:
        mutation_thread.start()
        deletion_thread.start()
        assert first_done.wait(timeout=5)
        allow_commit.set()
        mutation_thread.join(timeout=15)
        deletion_thread.join(timeout=15)
        assert not mutation_thread.is_alive()
        assert not deletion_thread.is_alive()
        assert "delete_error" not in results
        assert results["deleted"] is True
        if delete_first:
            error = results.get("mutation_error")
            assert isinstance(error, CharactersRAGDBError)
            assert error.code == "source_view_not_found"
            assert error.metadata == {}
        else:
            assert "mutation_error" not in results
    finally:
        allow_commit.set()
        mutation_thread.join(timeout=1)
        deletion_thread.join(timeout=1)
        db.close_all_connections()
        backend.get_pool().close_all()


def test_postgres_rls_isolates_two_principals_and_active_workspaces(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, owner_db = _database(pg_database_config)
    other_db = CharactersRAGDB(db_path=":memory:", client_id=OWNER_B, backend=backend)
    owner_db.upsert_workspace("ws-a", "Workspace A")
    other_db.upsert_workspace("ws-b", "Workspace B")
    owner_view = _create(owner_db, "ws-a")
    other_view = _create(other_db, "ws-b", owner=OWNER_B)
    ident = backend.escape_identifier
    role_name = f"saved_view_rls_{uuid4().hex[:8]}"
    role_created = False
    try:
        with pytest.raises(CharactersRAGDBError):
            _create(other_db, "ws-a", owner=OWNER_B, name="Probe")
        with pytest.raises(CharactersRAGDBError):
            other_db.update_workspace_source_saved_view(
                OWNER_B,
                "ws-a",
                owner_view["id"],
                expected_version=1,
                name="Probe",
            )
        with pytest.raises(CharactersRAGDBError):
            other_db.delete_workspace_source_saved_view(OWNER_B, "ws-a", owner_view["id"])

        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {ident(role_name)} NOLOGIN", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}", connection=conn)
            backend.execute(
                f"GRANT SELECT ON workspaces TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON workspace_source_saved_views TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        def set_constrained_principal(conn: Any, owner: str) -> None:
            backend.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (owner,),
                connection=conn,
            )

        def direct_insert(conn: Any, *, owner: str, workspace_id: str, name: str) -> str:
            view_id = str(uuid4())
            backend.execute(
                """
                INSERT INTO workspace_source_saved_views (
                    id, workspace_id, owner_user_id, name, name_key, schema_version,
                    state_json, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, '{}', 1, ?, ?)
                """,
                (
                    view_id,
                    workspace_id,
                    owner,
                    name,
                    name.casefold(),
                    "2026-01-01T00:00:00.000Z",
                    "2026-01-01T00:00:00.000Z",
                ),
                connection=conn,
            )
            return view_id

        with backend.transaction() as conn:
            set_constrained_principal(conn, OWNER_B)
            visible = list(
                backend.execute(
                    "SELECT id FROM workspace_source_saved_views ORDER BY id",
                    connection=conn,
                )
            )
            updated = backend.execute(
                "UPDATE workspace_source_saved_views SET name = ? WHERE id = ?",
                ("Cross tenant", owner_view["id"]),
                connection=conn,
            )
            deleted = backend.execute(
                "DELETE FROM workspace_source_saved_views WHERE id = ?",
                (owner_view["id"],),
                connection=conn,
            )

        assert [row["id"] for row in visible] == [other_view["id"]]
        assert updated.rowcount == 0
        assert deleted.rowcount == 0

        with pytest.raises(DatabaseError, match="row-level security"):
            with backend.transaction() as conn:
                set_constrained_principal(conn, OWNER_B)
                direct_insert(conn, owner=OWNER_A, workspace_id="ws-a", name="Wrong owner")

        with pytest.raises(DatabaseError, match="row-level security"):
            with backend.transaction() as conn:
                set_constrained_principal(conn, OWNER_B)
                direct_insert(conn, owner=OWNER_B, workspace_id="ws-a", name="Wrong workspace")

        with backend.transaction() as conn:
            set_constrained_principal(conn, OWNER_A)
            visible_a = list(
                backend.execute(
                    "SELECT id FROM workspace_source_saved_views ORDER BY id",
                    connection=conn,
                )
            )
            allowed_id = direct_insert(
                conn,
                owner=OWNER_A,
                workspace_id="ws-a",
                name="Allowed insert",
            )
            selected = list(
                backend.execute(
                    "SELECT id FROM workspace_source_saved_views WHERE id = ?",
                    (allowed_id,),
                    connection=conn,
                )
            )
            allowed_update = backend.execute(
                "UPDATE workspace_source_saved_views SET name = ?, name_key = ? WHERE id = ?",
                ("Allowed update", "allowed update", allowed_id),
                connection=conn,
            )
            allowed_delete = backend.execute(
                "DELETE FROM workspace_source_saved_views WHERE id = ?",
                (allowed_id,),
                connection=conn,
            )

        assert [row["id"] for row in visible_a] == [owner_view["id"]]
        assert selected == [{"id": allowed_id}]
        assert allowed_update.rowcount == 1
        assert allowed_delete.rowcount == 1
        assert owner_db.get_workspace_source_saved_view(OWNER_A, "ws-a", owner_view["id"])["id"] == owner_view["id"]
    finally:
        owner_db.close_connection()
        other_db.close_connection()
        if role_created:
            with backend.transaction() as conn:
                backend.execute(f"REVOKE {ident(role_name)} FROM CURRENT_USER", connection=conn)
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                backend.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        backend.get_pool().close_all()


def test_ensure_chacha_rls_succeeds_when_saved_view_table_is_absent(
    pg_database_config: DatabaseConfig,
) -> None:
    backend, db = _database(pg_database_config)
    try:
        backend.execute("DROP TABLE workspace_source_saved_views")

        assert ensure_chacha_rls(backend) is True
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
