"""Task 4 coverage for candidate promotion and publish validation."""
from __future__ import annotations

import importlib
from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _load_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Module {module.__name__} does not define any of: {', '.join(names)}")


class _AlwaysFailingPublishValidator:
    async def validate(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "reason": "synthetic publish validation failure"}

    async def validate_publish_candidate(self, **kwargs: Any) -> dict[str, Any]:
        return await self.validate(**kwargs)

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return await self.validate(**kwargs)


class _AlwaysPassingPublishValidator:
    async def validate(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    async def validate_publish_candidate(self, **kwargs: Any) -> dict[str, Any]:
        return await self.validate(**kwargs)

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return await self.validate(**kwargs)


@pytest.fixture
def promotion_service(repo):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(module, "PrototypePromotionService", "PrototypeWorkspaceService")
    return service_cls(repo=repo, publish_validator=_AlwaysFailingPublishValidator())


@pytest.fixture
def passing_promotion_service(repo):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(module, "PrototypePromotionService", "PrototypeWorkspaceService")
    return service_cls(repo=repo, publish_validator=_AlwaysPassingPublishValidator())


async def _seed_promotable_workspace(repo, prototype_db):
    workspace = await repo.create_workspace(
        owner_user_id=1,
        title="Task 4 Promotion",
        creation_source="prompt",
        runtime_policy={"external_collaborator_profile": "locked_collab"},
    )
    base_snapshot = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_base",
        created_by_user_id=1,
    )
    session = await repo.create_session(
        prototype_workspace_id=workspace["id"],
        base_snapshot_id=base_snapshot["snapshot_id"],
        actor_type="internal_collaborator",
        actor_user_id=2,
    )
    candidate = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_candidate",
        parent_snapshot_id=base_snapshot["snapshot_id"],
        created_from_session_id=session["id"],
        created_by_user_id=2,
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, canonical_preview_status = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        (base_snapshot["snapshot_id"], base_snapshot["snapshot_id"], "ready", "validated", workspace["id"]),
    )
    prototype_db.commit()
    return workspace, base_snapshot, session, candidate


def test_build_promote_idempotency_key_uses_workspace_candidate_and_canonical_snapshot_ids() -> None:
    jobs_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.jobs")
    assert jobs_module.build_promote_idempotency_key(
        prototype_workspace_id="pw_1",
        candidate_snapshot_id="snap_candidate",
        canonical_snapshot_id="snap_canonical",
    ) == "prototype:promote:pw_1:snap_candidate:snap_canonical"


@pytest.mark.asyncio
async def test_create_workspace_archives_partial_workspace_when_seed_snapshot_fails(
    repo,
    prototype_db,
    monkeypatch,
):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(module, "PrototypeWorkspaceService")
    service = service_cls(repo=repo)

    async def fail_create_snapshot(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("seed snapshot failed")

    monkeypatch.setattr(repo, "create_snapshot", fail_create_snapshot)

    with pytest.raises(RuntimeError, match="seed snapshot failed"):
        await service.create_workspace(
            owner_user_id=1,
            title="Partial seed failure",
            creation_source="prompt",
        )

    row = prototype_db.execute(
        "SELECT archived_at FROM prototype_workspaces WHERE title = ?",
        ("Partial seed failure",),
    ).fetchone()
    assert row is not None
    assert row[0] is not None


@pytest.mark.asyncio
async def test_save_session_snapshot_deletes_snapshot_when_session_state_update_fails(
    repo,
    prototype_db,
    monkeypatch,
):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.service")
    service_cls = _load_attr(module, "PrototypeWorkspaceService")
    service = service_cls(repo=repo)
    workspace, base_snapshot, session, _candidate = await _seed_promotable_workspace(repo, prototype_db)

    async def fail_update_session_state(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("session state update failed")

    monkeypatch.setattr(repo, "update_session_state", fail_update_session_state)

    with pytest.raises(RuntimeError, match="session state update failed"):
        await service.save_session_snapshot(
            prototype_session_id=session["id"],
            snapshot_id="snap_orphaned_save",
            storage_ref="prototype://failed-save",
        )

    assert await repo.get_snapshot("snap_orphaned_save") is None
    updated_workspace = await repo.get_workspace(workspace["id"])
    assert updated_workspace["canonical_snapshot_id"] == base_snapshot["snapshot_id"]


@pytest.mark.asyncio
async def test_promote_candidate_requires_validation(repo, prototype_db, promotion_service):
    workspace, base_snapshot, _session, candidate = await _seed_promotable_workspace(repo, prototype_db)

    result = await promotion_service.promote_candidate(
        prototype_workspace_id=workspace["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        reviewer_user_id=1,
    )

    updated_workspace = await repo.get_workspace(workspace["id"])

    assert result["status"] == "failed"
    assert result["failure_code"] == "publish_validation_failed"
    assert updated_workspace["canonical_snapshot_id"] == base_snapshot["snapshot_id"]
    assert updated_workspace["last_known_good_snapshot_id"] == base_snapshot["snapshot_id"]


@pytest.mark.asyncio
async def test_stale_candidate_blocks_canonical_advance(repo, prototype_db, passing_promotion_service):
    workspace, base_snapshot, _session, candidate = await _seed_promotable_workspace(repo, prototype_db)
    newer_canonical = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_newer_canonical",
        created_by_user_id=1,
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        (newer_canonical["snapshot_id"], newer_canonical["snapshot_id"], "validated", workspace["id"]),
    )
    prototype_db.commit()

    result = await passing_promotion_service.promote_candidate(
        prototype_workspace_id=workspace["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        reviewer_user_id=1,
    )

    updated_workspace = await repo.get_workspace(workspace["id"])

    assert result["status"] == "stale"
    assert result["failure_code"] == "stale_candidate"
    assert updated_workspace["canonical_snapshot_id"] == newer_canonical["snapshot_id"]
    assert updated_workspace["last_known_good_snapshot_id"] == newer_canonical["snapshot_id"]


@pytest.mark.asyncio
async def test_failed_publish_validation_preserves_last_known_good_snapshot_id(
    repo,
    prototype_db,
    promotion_service,
):
    workspace, base_snapshot, _session, candidate = await _seed_promotable_workspace(repo, prototype_db)
    current_canonical = await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_current_canonical",
        created_by_user_id=1,
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        (current_canonical["snapshot_id"], base_snapshot["snapshot_id"], "validated", workspace["id"]),
    )
    prototype_db.commit()

    result = await promotion_service.promote_candidate(
        prototype_workspace_id=workspace["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        reviewer_user_id=1,
    )

    updated_workspace = await repo.get_workspace(workspace["id"])

    assert result["status"] == "failed"
    assert result["failure_code"] == "publish_validation_failed"
    assert updated_workspace["last_known_good_snapshot_id"] == base_snapshot["snapshot_id"]


@pytest.mark.asyncio
async def test_promote_candidate_fails_closed_when_workspace_update_does_not_persist(
    repo,
    prototype_db,
    passing_promotion_service,
    monkeypatch,
):
    workspace, base_snapshot, _session, candidate = await _seed_promotable_workspace(repo, prototype_db)
    original_update = repo.update_workspace_state
    call_count = {"value": 0}

    async def flaky_update(prototype_workspace_id: str, **kwargs: Any):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return await original_update(prototype_workspace_id, **kwargs)
        return None

    monkeypatch.setattr(repo, "update_workspace_state", flaky_update)

    with pytest.raises(RuntimeError, match="failed to persist canonical"):
        await passing_promotion_service.promote_candidate(
            prototype_workspace_id=workspace["id"],
            candidate_snapshot_id=candidate["snapshot_id"],
            reviewer_user_id=1,
        )

    updated_workspace = await repo.get_workspace(workspace["id"])
    assert updated_workspace["canonical_snapshot_id"] == base_snapshot["snapshot_id"]


@pytest.mark.asyncio
async def test_promote_candidate_revokes_preview_and_reverts_workspace_when_request_update_fails(
    repo,
    prototype_db,
    passing_promotion_service,
    monkeypatch,
):
    workspace, base_snapshot, session, candidate = await _seed_promotable_workspace(repo, prototype_db)
    promotion_request = await repo.create_promotion_request(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        candidate_snapshot_id=candidate["snapshot_id"],
        requested_by_user_id=2,
    )
    revoked_handles: list[str] = []
    original_revoke = passing_promotion_service._preview_broker.revoke_preview_handle

    async def recording_revoke(preview_handle: str) -> bool:
        revoked_handles.append(preview_handle)
        return await original_revoke(preview_handle)

    async def fail_update_promotion_request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("promotion request update failed")

    monkeypatch.setattr(
        passing_promotion_service._preview_broker,
        "revoke_preview_handle",
        recording_revoke,
    )
    monkeypatch.setattr(repo, "update_promotion_request", fail_update_promotion_request)

    with pytest.raises(RuntimeError, match="promotion request update failed"):
        await passing_promotion_service.promote_candidate(
            prototype_workspace_id=workspace["id"],
            candidate_snapshot_id=candidate["snapshot_id"],
            reviewer_user_id=1,
            promotion_request_id=promotion_request["id"],
        )

    updated_workspace = await repo.get_workspace(workspace["id"])
    assert updated_workspace["canonical_snapshot_id"] == base_snapshot["snapshot_id"]
    assert updated_workspace["last_known_good_snapshot_id"] == base_snapshot["snapshot_id"]
    assert revoked_handles
