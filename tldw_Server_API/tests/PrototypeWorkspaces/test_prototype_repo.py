"""Unit tests for PrototypeWorkspacesRepo and AuthNZ prototype migration."""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
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


def test_migration_086_creates_risk_gate_2_query_indexes() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_086_create_prototype_workspace_tables,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_086_create_prototype_workspace_tables(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_prototype_%'"
        ).fetchall()
        index_names = {str(row[0]) for row in rows}
        assert {
            "idx_prototype_workspaces_owner_updated",
            "idx_prototype_workspaces_archived_at_cleanup",
            "idx_prototype_sessions_workspace_active_updated",
            "idx_prototype_sessions_active_lookup",
            "idx_prototype_sessions_revoked_at_cleanup",
            "idx_prototype_sessions_expires_at_cleanup",
            "idx_prototype_shared_actors_active_lookup",
            "idx_prototype_shared_actors_expires_revoked_cleanup",
            "idx_prototype_promotion_requests_workspace_status_updated",
            "idx_prototype_preview_handles_workspace",
            "idx_prototype_preview_handles_session",
            "idx_prototype_preview_handles_scope_active",
            "idx_prototype_preview_handles_active_scope",
            "idx_prototype_preview_handles_inactive_revoked_cleanup",
        }.issubset(index_names)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_transaction_yields_repo_bound_to_transaction_connection() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class TxConn:
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

    class TxPool:
        pool = None

        def __init__(self) -> None:
            self.conn = TxConn()
            self.transaction_opened = False

        @asynccontextmanager
        async def transaction(self):
            self.transaction_opened = True
            yield self.conn

        async def fetchall(self, _sql: str, _params: tuple[Any, ...] = ()) -> list[dict[str, str]]:
            pytest.fail("transaction-bound repo should not use the top-level pool fetchall")

    pool = TxPool()
    repo = PrototypeWorkspacesRepo(db_pool=pool)  # type: ignore[arg-type]

    async with repo.transaction() as tx_repo:
        await tx_repo.ensure_tables()

    assert pool.transaction_opened is True
    assert pool.conn.fetchall_calls


@pytest.mark.asyncio
async def test_transaction_bound_postgres_repo_converts_question_mark_placeholders() -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class PgConn:
        def __init__(self) -> None:
            self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

        async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any]:
            self.fetchrow_calls.append((sql, tuple(params)))
            return _workspace_row(id=params[0])

    class PgPool:
        pool = object()

        def __init__(self) -> None:
            self.conn = PgConn()

        @asynccontextmanager
        async def transaction(self):
            yield self.conn

        async def fetchone(self, _sql: str, _params: tuple[Any, ...] = ()) -> dict[str, Any]:
            pytest.fail("transaction-bound repo should not use the top-level pool fetchone")

    pool = PgPool()
    repo = PrototypeWorkspacesRepo(db_pool=pool)  # type: ignore[arg-type]

    async with repo.transaction() as tx_repo:
        workspace = await tx_repo.get_workspace("pws_sql_review")

    assert workspace is not None
    query, params = pool.conn.fetchrow_calls[0]
    assert "$1" in query
    assert "?" not in query
    assert params == ("pws_sql_review",)


@pytest.mark.asyncio
async def test_transaction_preserves_wrapped_domain_exception() -> None:
    from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    class TxPool:
        pool = None

        @asynccontextmanager
        async def transaction(self):
            try:
                yield object()
            except RuntimeError as exc:
                raise TransactionError("SQLite transaction", str(exc)) from exc

    repo = PrototypeWorkspacesRepo(db_pool=TxPool())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="domain failure"):
        async with repo.transaction():
            raise RuntimeError("domain failure")


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


@pytest.mark.asyncio
async def test_list_promotion_requests_for_workspace_filters_and_orders_requests(
    repo,
    prototype_db,
) -> None:
    workspace = await repo.create_workspace(owner_user_id=1, title="review queue", creation_source="prompt")
    other_workspace = await repo.create_workspace(owner_user_id=1, title="other queue", creation_source="prompt")
    base_snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_review_queue_base",
        created_by_user_id=1,
    )
    other_snapshot = await repo.create_snapshot(
        prototype_workspace_id=other_workspace["id"],
        snapshot_id="snap_review_queue_other_base",
        created_by_user_id=1,
    )
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=91,
        display_name="Review queue stakeholder",
        runtime_policy_profile="locked_collab",
    )
    other_actor = await repo.create_shared_actor(
        prototype_workspace_id=other_workspace["id"],
        share_link_id=92,
        display_name="Other stakeholder",
        runtime_policy_profile="locked_collab",
    )
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=base_snapshot["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
        share_link_id=91,
    )
    other_session = await repo.create_session(
        prototype_workspace_id=other_workspace["id"],
        base_snapshot_id=other_snapshot["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=other_actor["id"],
        share_link_id=92,
    )
    older_candidate = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_review_queue_candidate_old",
        created_by_shared_actor_id=actor["id"],
        parent_snapshot_id=base_snapshot["snapshot_id"],
        created_from_session_id=session["id"],
    )
    newer_candidate = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_review_queue_candidate_new",
        created_by_shared_actor_id=actor["id"],
        parent_snapshot_id=base_snapshot["snapshot_id"],
        created_from_session_id=session["id"],
    )
    other_candidate = await repo.create_snapshot(
        prototype_workspace_id=other_workspace["id"],
        snapshot_id="snap_review_queue_other_candidate",
        created_by_shared_actor_id=other_actor["id"],
        parent_snapshot_id=other_snapshot["snapshot_id"],
        created_from_session_id=other_session["id"],
    )

    older_request = await repo.create_promotion_request(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        candidate_snapshot_id=older_candidate["snapshot_id"],
        requested_by_shared_actor_id=actor["id"],
    )
    newer_request = await repo.create_promotion_request(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        candidate_snapshot_id=newer_candidate["snapshot_id"],
        requested_by_shared_actor_id=actor["id"],
    )
    await repo.create_promotion_request(
        prototype_workspace_id=other_workspace["id"],
        prototype_session_id=other_session["id"],
        candidate_snapshot_id=other_candidate["snapshot_id"],
        requested_by_shared_actor_id=other_actor["id"],
    )
    prototype_db.execute(
        "UPDATE prototype_promotion_requests SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", older_request["id"]),
    )
    prototype_db.execute(
        "UPDATE prototype_promotion_requests SET updated_at = ? WHERE id = ?",
        ("2026-01-02T00:00:00+00:00", newer_request["id"]),
    )
    prototype_db.commit()

    requests = await repo.list_promotion_requests_for_workspace(workspace["id"])

    assert [request["id"] for request in requests] == [newer_request["id"], older_request["id"]]
    assert {request["prototype_workspace_id"] for request in requests} == {workspace["id"]}
    assert requests[0]["candidate_snapshot_id"] == newer_candidate["snapshot_id"]


@pytest.mark.asyncio
async def test_cleanup_retained_state_revokes_expired_records_and_stales_pending_promotions(
    repo,
    prototype_db,
) -> None:
    old = "2026-01-01T00:00:00+00:00"
    cutoff = "2026-01-15T00:00:00+00:00"
    now = "2026-02-01T00:00:00+00:00"

    workspace = await repo.create_workspace(owner_user_id=1, title="cleanup", creation_source="prompt")
    base_snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_cleanup_base",
        created_by_user_id=1,
    )
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=71,
        display_name="Expired collaborator",
        runtime_policy_profile="locked_collab",
        expires_at=old,
    )
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=base_snapshot["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
        share_link_id=71,
        expires_at=old,
    )
    preview = await repo.replace_active_preview_handle_record(
        preview_handle="pph_cleanup_expired",
        preview_scope="session",
        scope_id=f"session:{session['id']}",
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        actor_key=f"shared_actor:{actor['id']}",
        target_ref="runtime://expired",
        runtime_policy_profile="locked_collab",
        created_at=old,
    )
    candidate = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_cleanup_candidate",
        created_by_shared_actor_id=actor["id"],
        parent_snapshot_id=base_snapshot["snapshot_id"],
        created_from_session_id=session["id"],
    )
    promotion = await repo.create_promotion_request(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        requested_by_shared_actor_id=actor["id"],
    )
    prototype_db.execute(
        "UPDATE prototype_promotion_requests SET updated_at = ? WHERE id = ?",
        (old, promotion["id"]),
    )
    prototype_db.commit()

    result = await repo.cleanup_retained_state(
        now=now,
        expired_before=cutoff,
        stale_promotion_before=cutoff,
        inactive_preview_before=old,
    )

    assert result["expired_shared_actors_revoked"] == 1
    assert result["expired_sessions_revoked"] == 1
    assert result["preview_handles_revoked"] == 1
    assert result["stale_promotion_requests_marked"] == 1
    assert result["inactive_preview_handles_deleted"] == 0

    updated_actor = await repo.get_shared_actor(actor["id"])
    updated_session = await repo.get_session(session["id"])
    updated_preview = await repo.get_preview_handle_record(preview["preview_handle"])
    updated_promotion = await repo.get_promotion_request(promotion["id"])

    assert updated_actor["is_revoked"] is True
    assert updated_session["is_revoked"] is True
    assert updated_session["runtime_status"] == "revoked"
    assert updated_session["preview_status"] == "revoked"
    assert updated_preview["is_active"] is False
    assert updated_preview["revoked_at"] == now
    assert updated_promotion["status"] == "stale"


@pytest.mark.asyncio
async def test_cleanup_retained_state_runs_inside_existing_transaction(repo) -> None:
    async with repo.transaction() as tx_repo:
        result = await tx_repo.cleanup_retained_state(
            now="2026-02-01T00:00:00+00:00",
            expired_before="2026-01-15T00:00:00+00:00",
        )

    assert result["expired_shared_actors_revoked"] == 0
    assert result["expired_sessions_revoked"] == 0
    assert result["preview_handles_revoked"] == 0


@pytest.mark.asyncio
async def test_cleanup_retained_state_deletes_archived_workspaces_after_cutoff(repo) -> None:
    workspace = await repo.create_workspace(owner_user_id=1, title="archived", creation_source="prompt")
    snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_archived_cleanup",
        created_by_user_id=1,
    )
    await repo.update_workspace_state(
        workspace["id"],
        canonical_snapshot_id=snapshot["snapshot_id"],
    )
    await repo.archive_workspace(
        workspace["id"],
        archived_at="2026-01-01T00:00:00+00:00",
    )

    result = await repo.cleanup_retained_state(
        now="2026-02-01T00:00:00+00:00",
        archived_workspace_before="2026-01-15T00:00:00+00:00",
    )

    assert result["archived_workspaces_deleted"] == 1
    assert await repo.get_workspace(workspace["id"]) is None
    assert await repo.get_snapshot(snapshot["snapshot_id"]) is None


@pytest.mark.asyncio
async def test_cleanup_retained_state_preserves_active_recent_records(repo) -> None:
    workspace = await repo.create_workspace(owner_user_id=1, title="active cleanup", creation_source="prompt")
    base_snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_active_cleanup_base",
        created_by_user_id=1,
    )
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=72,
        display_name="Active collaborator",
        runtime_policy_profile="locked_collab",
        expires_at="2026-03-01T00:00:00+00:00",
    )
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=base_snapshot["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
        share_link_id=72,
        expires_at="2026-03-01T00:00:00+00:00",
    )
    preview = await repo.replace_active_preview_handle_record(
        preview_handle="pph_cleanup_active",
        preview_scope="session",
        scope_id=f"session:{session['id']}",
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        actor_key=f"shared_actor:{actor['id']}",
        target_ref="runtime://active",
        runtime_policy_profile="locked_collab",
        created_at="2026-02-01T00:00:00+00:00",
    )
    candidate = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_active_cleanup_candidate",
        created_by_shared_actor_id=actor["id"],
        parent_snapshot_id=base_snapshot["snapshot_id"],
        created_from_session_id=session["id"],
    )
    promotion = await repo.create_promotion_request(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        requested_by_shared_actor_id=actor["id"],
    )

    result = await repo.cleanup_retained_state(
        now="2026-02-01T00:00:00+00:00",
        expired_before="2026-01-15T00:00:00+00:00",
        stale_promotion_before="2026-01-15T00:00:00+00:00",
        inactive_preview_before="2026-01-15T00:00:00+00:00",
        archived_workspace_before="2026-01-15T00:00:00+00:00",
    )

    assert all(count == 0 for count in result.values())
    assert (await repo.get_shared_actor(actor["id"]))["is_revoked"] is False
    assert (await repo.get_session(session["id"]))["is_revoked"] is False
    assert (await repo.get_preview_handle_record(preview["preview_handle"]))["is_active"] is True
    assert (await repo.get_promotion_request(promotion["id"]))["status"] == "pending"


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
