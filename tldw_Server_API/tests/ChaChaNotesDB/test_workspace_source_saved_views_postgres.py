from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import ensure_chacha_rls
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
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
) -> dict[str, Any]:
    return db.create_workspace_source_saved_view(
        owner,
        workspace_id,
        name=name,
        schema_version=1,
        state_json='{"schema_version":1}',
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

        with backend.transaction() as conn:
            backend.execute(f"SET LOCAL ROLE {ident(role_name)}", connection=conn)
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                (OWNER_B,),
                connection=conn,
            )
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
            with pytest.raises(Exception, match="row-level security"):
                backend.execute(
                    """
                    INSERT INTO workspace_source_saved_views (
                        id, workspace_id, owner_user_id, name, name_key, schema_version,
                        state_json, version, created_at, updated_at
                    ) VALUES (?, 'ws-a', ?, 'Injected', 'injected', 1, '{}', 1, ?, ?)
                    """,
                    (str(uuid4()), OWNER_A, "2026-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z"),
                    connection=conn,
                )

        assert [row["id"] for row in visible] == [other_view["id"]]
        assert updated.rowcount == 0
        assert deleted.rowcount == 0
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
