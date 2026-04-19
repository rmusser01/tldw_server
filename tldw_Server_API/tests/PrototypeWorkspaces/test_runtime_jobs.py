"""Task 4 coverage for prototype runtime job orchestration."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Module {module.__name__} does not define any of: {', '.join(names)}")


@pytest.fixture
def prototype_jobs(repo):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    jobs_cls = _load_attr(module, "PrototypeWorkspaceJobs", "PrototypeRuntimeJobs")
    return jobs_cls(repo=repo)


async def _seed_branch_workspace(repo, prototype_db):
    workspace = await repo.create_workspace(
        owner_user_id=1,
        title="Task 4 Runtime",
        creation_source="prompt",
        runtime_policy={"external_collaborator_profile": "locked_collab"},
        share_policy={"allow_browser_session_resume": True},
    )
    canonical = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_canonical",
        created_by_user_id=1,
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, canonical_preview_status = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        ("snap_canonical", "snap_canonical", "ready", "validated", workspace["id"]),
    )
    prototype_db.commit()
    return workspace, canonical


def test_jobs_module_exports_expected_task_types() -> None:
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    assert {
        "branch_session_bootstrap",
        "preview_boot",
        "snapshot_save",
        "publish_validate_and_promote",
    }.issubset(set(getattr(module, "PROTOTYPE_JOB_TYPES", set())))


@pytest.mark.asyncio
async def test_branch_session_bootstrap_is_idempotent_for_same_request_nonce(
    repo,
    prototype_db,
    prototype_jobs,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)

    first = await prototype_jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="retry-token",
    )
    second = await prototype_jobs.enqueue_branch_session_bootstrap(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="retry-token",
    )

    assert first["id"] == second["id"]
    assert first["job_type"] == "branch_session_bootstrap"
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["payload"]["prototype_workspace_id"] == workspace["id"]
    assert first["payload"]["base_snapshot_id"] == canonical["snapshot_id"]
    assert first["payload"]["request_nonce"] == "retry-token"


@pytest.mark.asyncio
async def test_branch_session_bootstrap_does_not_reuse_expired_session(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    existing = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        expires_at=expired,
    )

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    result = await service.create_or_reuse_branch_session(
        prototype_workspace_id=workspace["id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        request_nonce="fresh-after-expiry",
    )

    assert result["created"] is True
    assert result["session"]["id"] != existing["id"]


@pytest.mark.asyncio
async def test_preview_boot_idempotency_distinguishes_runtime_target_url(
    repo,
    prototype_db,
    prototype_jobs,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )

    first = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9101",
    )
    second = await prototype_jobs.enqueue_preview_boot(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id=canonical["snapshot_id"],
        runtime_target_url="http://127.0.0.1:9102",
    )

    assert first["id"] != second["id"]
    assert first["idempotency_key"] != second["idempotency_key"]


@pytest.mark.asyncio
async def test_revoked_external_collaborator_cannot_reuse_branch_session(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    actor = await repo.create_shared_actor(
        prototype_workspace_id=workspace["id"],
        share_link_id=51,
        display_name="Revoked Stakeholder",
        runtime_policy_profile="locked_collab",
    )
    await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
    )
    prototype_db.execute(
        "UPDATE prototype_shared_actors SET revoked_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), actor["id"]),
    )
    prototype_db.commit()

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="revoked"):
        await service.create_or_reuse_branch_session(
            prototype_workspace_id=workspace["id"],
            actor_type="external_collaborator",
            actor_shared_actor_id=actor["id"],
            request_nonce="after-revoke",
        )


@pytest.mark.asyncio
async def test_expired_session_cannot_save_snapshot(
    repo,
    prototype_db,
):
    workspace, canonical = await _seed_branch_workspace(repo, prototype_db)
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=canonical["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )

    service_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(service_module, "PrototypeWorkspaceService", "PrototypePromotionService")
    service = service_cls(repo=repo)

    with pytest.raises(RuntimeError, match="expired"):
        await service.save_session_snapshot(
            prototype_session_id=session["id"],
            snapshot_id="snap_should_not_save",
        )
