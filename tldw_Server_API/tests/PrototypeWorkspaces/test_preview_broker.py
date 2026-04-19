"""Task 4 coverage for brokered preview handles and revocation."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

pytestmark = pytest.mark.unit


def _load_attr(module: Any, *names: str) -> Any:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"Module {module.__name__} does not define any of: {', '.join(names)}")


@pytest.fixture
def preview_broker(repo):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.preview_broker")
    broker_cls = _load_attr(module, "PrototypePreviewBroker", "PrototypeWorkspacePreviewBroker")
    return broker_cls(repo=repo)


async def _seed_preview_scope(repo, prototype_db):
    workspace = await repo.create_workspace(
        owner_user_id=1,
        title="Task 4 Preview",
        creation_source="prompt",
        runtime_policy={"external_collaborator_profile": "locked_collab"},
        share_policy={"allow_browser_session_resume": True},
    )
    await repo.create_snapshot(
        prototype_workspace_id=workspace["id"],
        snapshot_id="snap_preview_base",
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
        base_snapshot_id="snap_preview_base",
        actor_type="external_collaborator",
        actor_shared_actor_id=actor["id"],
    )
    prototype_db.execute(
        """
        UPDATE prototype_workspaces
        SET canonical_snapshot_id = ?, last_known_good_snapshot_id = ?, canonical_preview_status = ?, publish_validation_status = ?
        WHERE id = ?
        """,
        ("snap_preview_base", "snap_preview_base", "ready", "validated", workspace["id"]),
    )
    prototype_db.commit()
    return workspace, actor, session


@pytest.mark.asyncio
async def test_preview_broker_keeps_one_active_target_per_scope(repo, prototype_db, preview_broker):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)

    first = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
    )
    second = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9012",
    )

    refreshed_session = await repo.get_session(session["id"])

    assert first["preview_handle"] != second["preview_handle"]
    assert not first["preview_handle"].startswith("http")
    assert not second["preview_handle"].startswith("http")
    assert refreshed_session["preview_handle"] == second["preview_handle"]
    assert refreshed_session["preview_status"] != "uninitialized"


@pytest.mark.asyncio
async def test_revoked_shared_actor_blocks_future_preview_grants(repo, prototype_db, preview_broker):
    workspace, actor, session = await _seed_preview_scope(repo, prototype_db)

    await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
    )
    prototype_db.execute(
        "UPDATE prototype_shared_actors SET revoked_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), actor["id"]),
    )
    prototype_db.commit()

    with pytest.raises(RuntimeError, match="revoked"):
        await preview_broker.issue_preview_grant(
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            snapshot_id="snap_preview_base",
            runtime_target_url="http://127.0.0.1:9013",
        )


@pytest.mark.asyncio
async def test_expired_session_blocks_preview_grants(repo, prototype_db, preview_broker):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)
    prototype_db.execute(
        "UPDATE prototype_sessions SET expires_at = ? WHERE id = ?",
        ((datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(), session["id"]),
    )
    prototype_db.commit()

    with pytest.raises(RuntimeError, match="expired"):
        await preview_broker.issue_preview_grant(
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            snapshot_id="snap_preview_base",
            runtime_target_url="http://127.0.0.1:9014",
        )


@pytest.mark.asyncio
async def test_revoked_actor_invalidates_existing_preview_grant(repo, prototype_db, preview_broker):
    workspace, actor, session = await _seed_preview_scope(repo, prototype_db)
    grant = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9015",
    )
    parsed = urlparse(grant["preview_url"])
    query = parse_qs(parsed.query)
    exp = int(query["exp"][0])

    prototype_db.execute(
        "UPDATE prototype_shared_actors SET revoked_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), actor["id"]),
    )
    prototype_db.commit()

    record = await preview_broker.validate_preview_grant(
        preview_handle=grant["preview_handle"],
        token=grant["token"],
        exp=exp,
        actor_key=f"shared_actor:{actor['id']}",
    )

    assert record is None


@pytest.mark.asyncio
async def test_failed_preview_persistence_restores_previous_active_handle(
    repo,
    prototype_db,
    preview_broker,
    monkeypatch,
):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)
    first = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9016",
    )
    parsed = urlparse(first["preview_url"])
    query = parse_qs(parsed.query)
    exp = int(query["exp"][0])
    original_update = repo.update_session_state

    async def fail_update(*args, **kwargs):
        return None

    monkeypatch.setattr(repo, "update_session_state", fail_update)

    with pytest.raises(RuntimeError, match="failed to persist preview handle"):
        await preview_broker.issue_preview_grant(
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            snapshot_id="snap_preview_base",
            runtime_target_url="http://127.0.0.1:9017",
        )

    restored = await preview_broker.validate_preview_grant(
        preview_handle=first["preview_handle"],
        token=first["token"],
        exp=exp,
        actor_key=f"shared_actor:{session['actor_shared_actor_id']}",
    )
    refreshed_session = await repo.get_session(session["id"])
    monkeypatch.setattr(repo, "update_session_state", original_update)

    assert restored is not None
    assert refreshed_session["preview_handle"] == first["preview_handle"]
