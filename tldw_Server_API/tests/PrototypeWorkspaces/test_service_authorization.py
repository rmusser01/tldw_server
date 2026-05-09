"""Authorization regressions for prototype workspace branch sessions."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Prototype_Workspaces.service import PrototypeWorkspaceService

pytestmark = pytest.mark.unit


async def _seed_workspace(service: PrototypeWorkspaceService) -> dict[str, Any]:
    return await service.create_workspace(
        owner_user_id=1,
        title="Revocation Auth",
        creation_source="prompt",
    )


def _revoked_at() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_create_or_reuse_branch_session_rejects_revoked_at_only_shared_actor(repo, monkeypatch):
    service = PrototypeWorkspaceService(repo=repo)
    workspace = await _seed_workspace(service)

    async def revoked_actor(_shared_actor_id: str) -> dict[str, Any]:
        return {"id": "pactor_revoked", "revoked_at": _revoked_at()}

    async def existing_session(**_kwargs: Any) -> dict[str, Any]:
        return {"id": "pss_existing"}

    monkeypatch.setattr(repo, "get_shared_actor", revoked_actor)
    monkeypatch.setattr(repo, "find_active_session", existing_session)

    with pytest.raises(RuntimeError, match="revoked shared actor"):
        await service.create_or_reuse_branch_session(
            prototype_workspace_id=workspace["id"],
            actor_type="external_collaborator",
            actor_shared_actor_id="pactor_revoked",
        )


@pytest.mark.asyncio
async def test_save_session_snapshot_rejects_revoked_at_only_session(repo, monkeypatch):
    service = PrototypeWorkspaceService(repo=repo)
    workspace = await _seed_workspace(service)

    async def revoked_session(_prototype_session_id: str) -> dict[str, Any]:
        return {
            "id": "pss_revoked",
            "prototype_workspace_id": workspace["id"],
            "base_snapshot_id": workspace["canonical_snapshot_id"],
            "actor_type": "internal_collaborator",
            "actor_user_id": 2,
            "revoked_at": _revoked_at(),
        }

    async def create_snapshot(**_kwargs: Any) -> dict[str, Any]:
        return {"snapshot_id": "psnap_should_not_be_created"}

    async def update_session_state(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(repo, "get_session", revoked_session)
    monkeypatch.setattr(repo, "create_snapshot", create_snapshot)
    monkeypatch.setattr(repo, "update_session_state", update_session_state)

    with pytest.raises(RuntimeError, match="revoked session"):
        await service.save_session_snapshot(
            prototype_session_id="pss_revoked",
            snapshot_id="psnap_blocked",
        )


@pytest.mark.asyncio
async def test_save_session_snapshot_rejects_revoked_at_only_shared_actor(repo, monkeypatch):
    service = PrototypeWorkspaceService(repo=repo)
    workspace = await _seed_workspace(service)

    async def external_session(_prototype_session_id: str) -> dict[str, Any]:
        return {
            "id": "pss_external",
            "prototype_workspace_id": workspace["id"],
            "base_snapshot_id": workspace["canonical_snapshot_id"],
            "actor_type": "external_collaborator",
            "actor_shared_actor_id": "pactor_revoked",
        }

    async def revoked_actor(_shared_actor_id: str) -> dict[str, Any]:
        return {"id": "pactor_revoked", "revoked_at": _revoked_at()}

    async def create_snapshot(**_kwargs: Any) -> dict[str, Any]:
        return {"snapshot_id": "psnap_should_not_be_created"}

    async def update_session_state(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(repo, "get_session", external_session)
    monkeypatch.setattr(repo, "get_shared_actor", revoked_actor)
    monkeypatch.setattr(repo, "create_snapshot", create_snapshot)
    monkeypatch.setattr(repo, "update_session_state", update_session_state)

    with pytest.raises(RuntimeError, match="revoked shared actor"):
        await service.save_session_snapshot(
            prototype_session_id="pss_external",
            snapshot_id="psnap_blocked",
        )
