"""Unit tests for PrototypeWorkspacesRepo and AuthNZ prototype migration."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _workspace_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "pws_sql_review",
        "owner_user_id": 1,
        "title": "SQL review",
        "description": None,
        "creation_source": "prompt",
        "canonical_snapshot_id": "snap_existing",
        "last_known_good_snapshot_id": "snap_existing",
        "canonical_preview_status": "pending",
        "publish_validation_status": "unknown",
        "preview_policy_json": "{}",
        "share_policy_json": "{}",
        "runtime_policy_json": "{}",
        "designated_promoter_ids_json": "[]",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "archived_at": None,
    }
    row.update(overrides)
    return row


def _session_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "pss_sql_review",
        "prototype_workspace_id": "pws_sql_review",
        "base_snapshot_id": "snap_existing",
        "actor_user_id": None,
        "actor_shared_actor_id": "psa_sql_review",
        "actor_type": "external_collaborator",
        "share_link_id": 101,
        "acp_session_id": None,
        "sandbox_session_id": None,
        "sandbox_run_id": None,
        "runtime_status": "running",
        "preview_handle": None,
        "preview_status": "ready",
        "last_saved_snapshot_id": None,
        "last_activity_at": "2026-01-01T00:00:00+00:00",
        "expires_at": "2999-01-01T00:00:00+00:00",
        "revoked_at": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_migration_086_creates_prototype_workspace_tables() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_086_create_prototype_workspace_tables,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_086_create_prototype_workspace_tables(conn)
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        table_names = {str(row[0]) for row in rows}
        assert {
            "prototype_workspaces",
            "prototype_snapshots",
            "prototype_sessions",
            "prototype_shared_actors",
            "prototype_promotion_requests",
        }.issubset(table_names)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_create_workspace_snapshot_and_session_enforce_single_actor_identity(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_1",
        created_by_user_id=1,
    )
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=11,
        display_name="Stakeholder A",
        runtime_policy_profile="locked_collab",
    )

    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=snapshot["snapshot_id"],
        actor_shared_actor_id=actor["id"],
        actor_type="external_collaborator",
        share_link_id=11,
    )

    assert session["actor_shared_actor_id"] == actor["id"]
    assert session["actor_user_id"] is None
    assert session["share_link_id"] == 11

    with pytest.raises(ValueError, match="external_collaborator requires share_link_id"):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_shared_actor_id=actor["id"],
            actor_type="external_collaborator",
        )

    with pytest.raises(ValueError, match="share_link_id must match actor_shared_actor_id"):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_shared_actor_id=actor["id"],
            actor_type="external_collaborator",
            share_link_id=12,
        )

    with pytest.raises(
        ValueError, match="owner/internal_collaborator requires actor_user_id and forbids actor_shared_actor_id"
    ):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_type="internal_collaborator",
            actor_user_id=2,
            actor_shared_actor_id=actor["id"],
        )

    with pytest.raises(
        ValueError, match="owner/internal_collaborator requires actor_user_id and forbids actor_shared_actor_id"
    ):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_type="internal_collaborator",
        )


@pytest.mark.asyncio
async def test_create_session_rejects_external_collaborator_with_actor_user_id(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_user_invalid",
        created_by_user_id=1,
    )
    with pytest.raises(
        ValueError, match="external_collaborator requires actor_shared_actor_id and forbids actor_user_id"
    ):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_type="external_collaborator",
            actor_user_id=2,
        )


@pytest.mark.asyncio
async def test_create_session_rejects_internal_collaborator_with_actor_shared_actor_id(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_internal_invalid",
        created_by_user_id=1,
    )
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=22,
        display_name="Stakeholder B",
        runtime_policy_profile="locked_collab",
    )
    with pytest.raises(
        ValueError, match="owner/internal_collaborator requires actor_user_id and forbids actor_shared_actor_id"
    ):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_type="internal_collaborator",
            actor_shared_actor_id=actor["id"],
        )


@pytest.mark.asyncio
async def test_create_session_rejects_internal_collaborator_with_nonexistent_user(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_missing_user",
        created_by_user_id=1,
    )
    with pytest.raises(ValueError, match="actor_user_id must reference an existing user"):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id=snapshot["snapshot_id"],
            actor_type="internal_collaborator",
            actor_user_id=999,
        )


@pytest.mark.asyncio
async def test_create_session_rejects_missing_base_snapshot_id(repo):
    workspace = await repo.create_workspace(owner_user_id=1, title="demo", creation_source="prompt")
    with pytest.raises(ValueError, match="base_snapshot_id is required"):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id="",
            actor_type="internal_collaborator",
            actor_user_id=2,
        )

    with pytest.raises(ValueError, match="base_snapshot_id must reference a snapshot in the same workspace"):
        await repo.create_session(
            prototype_workspace_id=workspace["id"],
            base_snapshot_id="missing_snapshot",
            actor_type="internal_collaborator",
            actor_user_id=2,
        )


@pytest.mark.asyncio
async def test_create_session_rejects_cross_workspace_snapshot(repo):
    workspace_one = await repo.create_workspace(owner_user_id=1, title="w1", creation_source="prompt")
    workspace_two = await repo.create_workspace(owner_user_id=1, title="w2", creation_source="prompt")
    await repo.create_snapshot(
        prototype_workspace_id=workspace_two["id"],
        snapshot_id="snap_other_workspace",
        created_by_user_id=1,
    )

    with pytest.raises(ValueError, match="base_snapshot_id must reference a snapshot in the same workspace"):
        await repo.create_session(
            prototype_workspace_id=workspace_one["id"],
            base_snapshot_id="snap_other_workspace",
            actor_type="internal_collaborator",
            actor_user_id=2,
        )


@pytest.mark.asyncio
async def test_create_session_rejects_cross_workspace_shared_actor(repo):
    workspace_one = await repo.create_workspace(owner_user_id=1, title="w1", creation_source="prompt")
    workspace_two = await repo.create_workspace(owner_user_id=1, title="w2", creation_source="prompt")

    snapshot_one = await repo.create_snapshot(
        prototype_workspace_id=workspace_one["id"],
        snapshot_id="snap_workspace_one",
        created_by_user_id=1,
    )
    actor_two = await repo.create_shared_actor(
        prototype_workspace_id=workspace_two["id"],
        share_link_id=31,
        display_name="Stakeholder Other Workspace",
        runtime_policy_profile="locked_collab",
    )

    with pytest.raises(
        ValueError, match="actor_shared_actor_id must reference an active shared actor in the same workspace"
    ):
        await repo.create_session(
            prototype_workspace_id=workspace_one["id"],
            base_snapshot_id=snapshot_one["snapshot_id"],
            actor_type="external_collaborator",
            actor_shared_actor_id=actor_two["id"],
        )


@pytest.mark.asyncio
async def test_ensure_tables_uses_information_schema_for_postgres() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class RecordingPool:
        pool = object()

        def __init__(self) -> None:
            self.fetchall_calls: list[tuple[str, tuple[Any, ...]]] = []

        async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, str]]:
            self.fetchall_calls.append((sql, params))
            return [
                {"name": "prototype_workspaces"},
                {"name": "prototype_snapshots"},
                {"name": "prototype_sessions"},
                {"name": "prototype_shared_actors"},
                {"name": "prototype_promotion_requests"},
                {"name": "prototype_preview_handles"},
            ]

    pool = RecordingPool()
    repo = PrototypeWorkspacesRepo(db_pool=pool)  # type: ignore[arg-type]

    await repo.ensure_tables()

    table_query = pool.fetchall_calls[0][0]
    assert "information_schema.tables" in table_query
    assert "sqlite_master" not in table_query


def test_row_to_dict_logs_conversion_failures(monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos import prototype_workspaces_repo as repo_module

    class BrokenRow:
        def keys(self) -> list[str]:
            return ["id"]

        def __getitem__(self, _key: str) -> str:
            raise TypeError("broken row access")

    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        repo_module,
        "logger",
        SimpleNamespace(debug=lambda *args: warnings.append(args)),
        raising=False,
    )

    result = repo_module.PrototypeWorkspacesRepo._row_to_dict(BrokenRow())

    assert result == {}
    assert warnings
    assert "Failed to convert prototype workspace row" in warnings[0][0]


@pytest.mark.asyncio
async def test_update_workspace_state_preserves_columns_in_sql_without_pre_read() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class RecordingPool:
        pool = None

        def __init__(self) -> None:
            self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
            self.fetchone_calls = 0

        async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
            self.execute_calls.append((sql, params))

        async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
            self.fetchone_calls += 1
            if not self.execute_calls:
                pytest.fail("update_workspace_state should not fetch existing state before updating")
            return _workspace_row(id=params[0])

    pool = RecordingPool()
    repo = PrototypeWorkspacesRepo(db_pool=pool)  # type: ignore[arg-type]

    updated = await repo.update_workspace_state(
        "pws_sql_review",
        canonical_snapshot_id="snap_new",
    )

    assert updated is not None
    update_sql, update_params = pool.execute_calls[0]
    assert "COALESCE(?, canonical_snapshot_id)" in update_sql
    assert "COALESCE(?, last_known_good_snapshot_id)" in update_sql
    assert "COALESCE(?, canonical_preview_status)" in update_sql
    assert "COALESCE(?, publish_validation_status)" in update_sql
    assert update_params[0] == "snap_new"


@pytest.mark.asyncio
async def test_find_active_session_filters_candidate_in_sql() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class RecordingPool:
        pool = None

        def __init__(self) -> None:
            self.fetchone_calls: list[tuple[str, tuple[Any, ...]]] = []
            self.fetchall_calls: list[tuple[str, tuple[Any, ...]]] = []

        async def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
            self.fetchone_calls.append((sql, params))
            return _session_row()

        async def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
            self.fetchall_calls.append((sql, params))
            return [_session_row()]

    pool = RecordingPool()
    repo = PrototypeWorkspacesRepo(db_pool=pool)  # type: ignore[arg-type]

    session = await repo.find_active_session(
        prototype_workspace_id="pws_sql_review",
        base_snapshot_id="snap_existing",
        actor_type="external_collaborator",
        actor_shared_actor_id="psa_sql_review",
        share_link_id=101,
    )

    assert session is not None
    query, params = (pool.fetchone_calls or pool.fetchall_calls)[0]
    assert "base_snapshot_id = ?" in query
    assert "actor_type = ?" in query
    assert "actor_shared_actor_id = ?" in query
    assert "share_link_id = ?" in query
    assert "runtime_status" in query
    assert "expires_at" in query
    assert "LIMIT 1" in query
    assert params[:3] == ("pws_sql_review", "snap_existing", "external_collaborator")


def test_apply_authnz_migrations_from_v85_includes_prototype_tables(tmp_path) -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations

    db_path = tmp_path / "authnz_upgrade_path.db"
    apply_authnz_migrations(db_path, target_version=85)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        table_names = {str(row[0]) for row in rows}
        assert "prototype_workspaces" not in table_names
    finally:
        conn.close()

    apply_authnz_migrations(db_path)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        table_names = {str(row[0]) for row in rows}
        assert {
            "prototype_workspaces",
            "prototype_snapshots",
            "prototype_sessions",
            "prototype_shared_actors",
            "prototype_promotion_requests",
        }.issubset(table_names)
    finally:
        conn.close()
