"""PostgreSQL integration coverage for staged Workspace clone targets."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, QueryResult
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

pytestmark = [pytest.mark.integration, pytest.mark.timeout(60)]


@pytest.fixture
def postgres_db(pg_database_config: DatabaseConfig) -> Iterator[CharactersRAGDB]:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="user-1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _reserve(
    db: CharactersRAGDB,
    *,
    workspace_id: str = "workspace-target",
    operation_id: str = "operation-1",
    request_fingerprint: str = "fingerprint-1",
    name: str = "Target Workspace",
    description: str | None = "Cloned workspace",
    workspace_profile: str = "research",
) -> dict[str, object]:
    return db.reserve_clone_target(
        workspace_id=workspace_id,
        operation_id=operation_id,
        request_fingerprint=request_fingerprint,
        name=name,
        description=description,
        workspace_profile=workspace_profile,
    )


def _seed_workspace_snapshot(db: CharactersRAGDB) -> None:
    db.upsert_workspace("workspace-source", "Workspace v1")
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO workspace_sources "
            "(id, workspace_id, media_id, title, source_type, position, selected, added_at, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)",
            ("source-1", "workspace-source", 101, "Source v1", "document", 0, True),
        )
    db.add_workspace_resource_membership(
        "workspace-source",
        {
            "resource_type": "media",
            "resource_id": "101",
            "role": "source",
            "label": "Membership v1",
        },
        user_id="user-1",
    )
    db.add_workspace_note(
        "workspace-source",
        {"title": "Note v1", "content": "Note content v1"},
    )
    db.add_workspace_artifact(
        "workspace-source",
        {
            "id": "artifact-1",
            "artifact_type": "report",
            "title": "Artifact v1",
            "content": "Artifact content v1",
        },
    )


def test_postgres_workspace_clone_snapshot_is_repeatable_and_returns_connection(
    postgres_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace_snapshot(postgres_db)
    caller_connection = postgres_db.get_connection()
    backend = postgres_db.backend
    pool = backend.get_pool()
    original_execute = backend.execute
    original_get_connection = pool.get_connection
    original_return_connection = pool.return_connection
    acquired_connections = []
    returned_connections: list[tuple[object, str]] = []
    observed_modes: dict[str, str] = {}
    observed_scope: dict[str, str] = {}

    def tracked_get_connection():
        connection = original_get_connection()
        acquired_connections.append(connection)
        return connection

    def tracked_return_connection(connection) -> None:
        status = getattr(connection.info.transaction_status, "name", "")
        returned_connections.append((connection, str(status)))
        original_return_connection(connection)

    def interleaved_execute(query, params=None, connection=None, **kwargs):
        result = original_execute(query, params, connection=connection, **kwargs)
        if not observed_modes and "FROM workspaces" in query:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT current_setting('app.current_user_id', true) AS current_user_id, "
                    "current_setting('app.user_id', true) AS user_id, "
                    "current_setting('app.org_ids', true) AS org_ids, "
                    "current_setting('app.team_ids', true) AS team_ids, "
                    "current_setting('app.is_admin', true) AS is_admin, "
                    "current_role::text AS current_role, session_user::text AS session_user"
                )
                observed_scope.update(cursor.fetchone())
                cursor.execute("SHOW transaction_isolation")
                observed_modes["isolation"] = str(next(iter(cursor.fetchone().values())))
                cursor.execute("SHOW transaction_read_only")
                observed_modes["read_only"] = str(next(iter(cursor.fetchone().values())))

            writer = pool.get_connection()
            try:
                writer.commit()
                with writer.cursor() as cursor:
                    cursor.execute(
                        "UPDATE workspaces SET name = %s, version = version + 1 WHERE id = %s",
                        ("Workspace v2", "workspace-source"),
                    )
                    cursor.execute(
                        "UPDATE workspace_sources SET title = %s, version = version + 1 "
                        "WHERE workspace_id = %s",
                        ("Source v2", "workspace-source"),
                    )
                    cursor.execute(
                        "UPDATE workspace_resource_memberships "
                        "SET label = %s, version = version + 1 WHERE workspace_id = %s",
                        ("Membership v2", "workspace-source"),
                    )
                    cursor.execute(
                        "UPDATE workspace_notes SET title = %s, version = version + 1 "
                        "WHERE workspace_id = %s",
                        ("Note v2", "workspace-source"),
                    )
                    cursor.execute(
                        "UPDATE workspace_artifacts SET title = %s, version = version + 1 "
                        "WHERE workspace_id = %s",
                        ("Artifact v2", "workspace-source"),
                    )
                writer.commit()
            finally:
                pool.return_connection(writer)
        return result

    monkeypatch.setattr(pool, "get_connection", tracked_get_connection)
    monkeypatch.setattr(pool, "return_connection", tracked_return_connection)
    monkeypatch.setattr(backend, "execute", interleaved_execute)

    with scoped_context(user_id="user-1", org_ids=[12], team_ids=[77], is_admin=True):
        snapshot = postgres_db.read_workspace_clone_snapshot("workspace-source")

    snapshot_connection = acquired_connections[0]
    assert snapshot_connection is not caller_connection
    assert returned_connections.count((snapshot_connection, "IDLE")) == 1
    assert observed_modes == {"isolation": "repeatable read", "read_only": "on"}
    assert observed_scope["current_user_id"] == "user-1"
    assert observed_scope["user_id"] == "user-1"
    assert observed_scope["org_ids"] == "12"
    assert observed_scope["team_ids"] == "77"
    assert observed_scope["is_admin"] == "1"
    assert observed_scope["current_role"] == observed_scope["session_user"]
    assert snapshot.workspace["name"] == "Workspace v1"
    assert snapshot.sources[0]["title"] == "Source v1"
    assert snapshot.memberships[0]["label"] == "Membership v1"
    assert snapshot.notes[0]["title"] == "Note v1"
    assert snapshot.artifacts[0]["title"] == "Artifact v1"


def test_postgres_workspace_clone_snapshot_rolls_back_and_returns_on_failure(
    postgres_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = postgres_db.backend.get_pool()
    original_get_connection = pool.get_connection
    original_return_connection = pool.return_connection
    acquired_connections = []
    returned_connections: list[tuple[object, str]] = []

    def tracked_get_connection():
        connection = original_get_connection()
        acquired_connections.append(connection)
        return connection

    def tracked_return_connection(connection) -> None:
        status = getattr(connection.info.transaction_status, "name", "")
        returned_connections.append((connection, str(status)))
        original_return_connection(connection)

    monkeypatch.setattr(pool, "get_connection", tracked_get_connection)
    monkeypatch.setattr(pool, "return_connection", tracked_return_connection)

    with pytest.raises(CloneSnapshotUnavailable):
        postgres_db.read_workspace_clone_snapshot("missing-workspace")

    assert len(acquired_connections) == 1
    assert returned_connections == [(acquired_connections[0], "IDLE")]


def test_postgres_workspace_clone_graph_is_tenant_isolated(
    postgres_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace_snapshot(postgres_db)
    other_user_db = CharactersRAGDB(
        ":memory:",
        client_id="user-2",
        backend=postgres_db.backend,
    )
    backend = postgres_db.backend
    role_name = f"workspace_clone_reader_{uuid4().hex[:12]}"
    ident = backend.escape_identifier
    role_created = False
    try:
        other_user_db.upsert_workspace("workspace-user-2", "User 2 Workspace")
        with backend.transaction() as connection:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=connection,
            )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                "GRANT SELECT ON workspaces, workspace_sources, workspace_notes, "
                "workspace_artifacts, workspace_artifact_versions, "
                f"workspace_resource_memberships TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                f"GRANT {ident(role_name)} TO CURRENT_USER",
                connection=connection,
            )
        role_created = True
        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)

        for database, user_id in (
            (postgres_db, "user-1"),
            (other_user_db, "user-2"),
        ):
            database.close_connection()
            connection = database.get_connection()
            cursor = connection.cursor()
            cursor.execute(f"SET ROLE {ident(role_name)}")
            cursor.execute("SET row_security = on")
            cursor.execute(
                "SELECT set_config('app.current_user_id', %s, false)",
                (user_id,),
            )
            connection.commit()

        assert other_user_db.get_workspace("workspace-source") is None
        assert {row["id"] for row in other_user_db.list_workspaces()} == {
            "workspace-user-2"
        }
        assert other_user_db.list_workspace_sources("workspace-source") == []
        assert other_user_db.list_workspace_notes("workspace-source") == []
        assert other_user_db.list_workspace_artifacts("workspace-source") == []
        assert other_user_db.list_workspace_artifact_versions(
            "workspace-source",
            "artifact-1",
        ) == []
        assert other_user_db.list_workspace_resource_memberships(
            "workspace-source"
        ) == []
        with scoped_context(
            user_id="user-2",
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            with pytest.raises(CloneSnapshotUnavailable):
                other_user_db.read_workspace_clone_snapshot("workspace-source")

        assert postgres_db.get_workspace("workspace-user-2") is None
        assert {row["id"] for row in postgres_db.list_workspaces()} == {
            "workspace-source"
        }
    finally:
        try:
            for database in (postgres_db, other_user_db):
                connection = database.get_connection()
                cursor = connection.cursor()
                cursor.execute("RESET ROLE")
                cursor.execute("RESET row_security")
                cursor.execute("RESET app.current_user_id")
                connection.commit()
        finally:
            try:
                if role_created:
                    with backend.transaction() as connection:
                        backend.execute(
                            f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                            connection=connection,
                        )
                        backend.execute(
                            f"DROP OWNED BY {ident(role_name)}",
                            connection=connection,
                        )
                        backend.execute(
                            f"DROP ROLE {ident(role_name)}",
                            connection=connection,
                        )
            finally:
                other_user_db.close_connection()


@pytest.mark.parametrize(
    "failure_mode",
    ["scope_apply", "scope_verification", "transaction_mode"],
)
def test_postgres_workspace_snapshot_setup_failure_stops_before_table_queries(
    postgres_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    _seed_workspace_snapshot(postgres_db)
    caller_connection = postgres_db.get_connection()
    backend = postgres_db.backend
    pool = backend.get_pool()
    original_get_connection = pool.get_connection
    original_return_connection = pool.return_connection
    original_execute = backend.execute
    acquired_connections = []
    returned_connections: list[tuple[object, str]] = []
    table_queries: list[str] = []

    def tracked_get_connection():
        connection = original_get_connection()
        acquired_connections.append(connection)
        return connection

    def tracked_return_connection(connection) -> None:
        status = getattr(connection.info.transaction_status, "name", "")
        returned_connections.append((connection, str(status)))
        original_return_connection(connection)

    def mismatched_execute(query, params=None, connection=None, **kwargs):
        if any(
            table_name in query
            for table_name in (
                "FROM workspaces",
                "FROM workspace_resource_memberships",
                "FROM workspace_sources",
                "FROM workspace_notes",
                "FROM workspace_artifacts",
            )
        ):
            table_queries.append(query)
        if (
            failure_mode == "scope_verification"
            and "current_setting('app.current_user_id'" in query
        ):
            return QueryResult(
                rows=[
                    {
                        "current_user_id": "wrong-user",
                        "user_id": "user-1",
                        "org_ids": "",
                        "team_ids": "",
                        "is_admin": "0",
                        "row_security": "on",
                        "current_role": "postgres",
                        "session_user": "postgres",
                    }
                ],
                rowcount=1,
            )
        if failure_mode == "transaction_mode" and query.strip().upper() == (
            "SHOW TRANSACTION_READ_ONLY"
        ):
            return QueryResult(
                rows=[{"transaction_read_only": "off"}],
                rowcount=1,
            )
        return original_execute(query, params, connection=connection, **kwargs)

    monkeypatch.setattr(pool, "get_connection", tracked_get_connection)
    monkeypatch.setattr(pool, "return_connection", tracked_return_connection)
    monkeypatch.setattr(backend, "execute", mismatched_execute)

    scope_kwargs: dict[str, object] = {
        "user_id": "user-1",
        "org_ids": [],
        "team_ids": [],
        "is_admin": False,
    }
    if failure_mode == "scope_apply":
        monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
        monkeypatch.setenv(
            "TLDW_CONTENT_PG_ROLE_WHITELIST",
            "task3_missing_snapshot_role",
        )
        scope_kwargs["session_role"] = "task3_missing_snapshot_role"

    with scoped_context(**scope_kwargs):
        with pytest.raises(CloneSnapshotUnavailable):
            postgres_db.read_workspace_clone_snapshot("workspace-source")

    assert table_queries == []
    assert len(acquired_connections) == 1
    snapshot_connection = acquired_connections[0]
    assert snapshot_connection is not caller_connection
    assert returned_connections == [(snapshot_connection, "IDLE")]


def test_postgres_reserve_publish_replay_and_confirm_are_hidden_and_fenced(
    postgres_db: CharactersRAGDB,
) -> None:
    postgres_db.upsert_workspace("workspace-ordinary", "Ordinary Workspace")

    with pytest.raises(ConflictError):
        _reserve(postgres_db, workspace_id="workspace-ordinary")

    reserved = _reserve(postgres_db, name="  Target   Workspace  ")
    replayed = _reserve(postgres_db, name="Target Workspace")

    assert replayed == reserved
    assert bool(reserved["archived"]) is True
    assert reserved["system_operation_state"] == "staged"
    assert postgres_db.get_workspace("workspace-target") is None
    assert {row["id"] for row in postgres_db.list_workspaces()} == {"workspace-ordinary"}

    with pytest.raises(ConflictError):
        _reserve(postgres_db, operation_id="operation-2")

    published = postgres_db.publish_clone_target(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )
    pending_replay = _reserve(postgres_db)

    assert pending_replay == published
    assert bool(pending_replay["archived"]) is False
    assert pending_replay["system_operation_state"] == "publication_pending"
    assert postgres_db.get_workspace("workspace-target") is None
    assert {row["id"] for row in postgres_db.list_workspaces()} == {"workspace-ordinary"}

    confirmed = postgres_db.confirm_clone_target_publication(
        workspace_id="workspace-target",
        operation_id="operation-1",
    )

    assert confirmed["system_operation_id"] is None
    assert confirmed["system_operation_kind"] is None
    assert confirmed["system_operation_state"] is None
    assert confirmed["system_request_fingerprint"] is None
    assert postgres_db.get_workspace("workspace-target") == confirmed
    assert {row["id"] for row in postgres_db.list_workspaces()} == {
        "workspace-ordinary",
        "workspace-target",
    }


def test_postgres_reconciliation_and_discard_use_bounded_boolean_rowcounts(
    postgres_db: CharactersRAGDB,
) -> None:
    _reserve(postgres_db, workspace_id="workspace-a", operation_id="operation-a")
    _reserve(postgres_db, workspace_id="workspace-b", operation_id="operation-b")
    postgres_db.publish_clone_target(workspace_id="workspace-b", operation_id="operation-b")
    _reserve(postgres_db, workspace_id="workspace-c", operation_id="operation-c")

    limited = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-a"],
        limit=1,
    )
    correlated = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-a"],
        limit=2,
    )

    assert [(row["id"], row["system_operation_state"]) for row in limited] == [
        ("workspace-a", "staged")
    ]
    assert [(row["id"], row["system_operation_state"]) for row in correlated] == [
        ("workspace-a", "staged"),
        ("workspace-b", "publication_pending"),
    ]

    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-other",
    ) is False
    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-a",
    ) is True
    assert postgres_db.discard_clone_target(
        workspace_id="workspace-a",
        operation_id="operation-a",
    ) is False
    assert postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-a"],
    ) == []

    assert postgres_db.discard_clone_target(
        workspace_id="workspace-b",
        operation_id="operation-b",
    ) is True
    remaining = postgres_db.list_clone_targets_for_reconciliation(
        operation_ids=["operation-b", "operation-c"],
    )

    assert [(row["id"], row["system_operation_state"]) for row in remaining] == [
        ("workspace-c", "staged")
    ]
