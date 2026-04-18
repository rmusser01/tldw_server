"""Unit tests for PrototypeWorkspacesRepo and AuthNZ prototype migration."""
from __future__ import annotations

import sqlite3

import pytest

pytestmark = pytest.mark.unit


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
    )

    assert session["actor_shared_actor_id"] == actor["id"]
    assert session["actor_user_id"] is None

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
