"""Task 4 coverage for brokered preview handles and revocation."""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
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


def test_preview_broker_requires_configured_stable_signing_secret(repo, monkeypatch):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.preview_broker")
    broker_cls = _load_attr(module, "PrototypePreviewBroker", "PrototypeWorkspacePreviewBroker")
    monkeypatch.delenv("PROTOTYPE_PREVIEW_SIGNING_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setattr(
        module,
        "get_settings",
        lambda: SimpleNamespace(JWT_SECRET_KEY=None, SINGLE_USER_API_KEY=None),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="stable signing secret"):
        broker_cls(repo=repo)


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
        share_link_id=11,
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
async def test_preview_broker_reuses_active_handle_for_same_scope_target_retry(
    repo,
    prototype_db,
    preview_broker,
):
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
        runtime_target_url="http://127.0.0.1:9011",
    )
    first_record = await repo.get_preview_handle_record(first["preview_handle"])
    active_record = await repo.get_active_preview_handle_for_scope(f"session:{session['id']}")

    assert second["preview_handle"] == first["preview_handle"]
    assert second["token"]
    assert first_record is not None
    assert active_record is not None
    assert active_record["preview_handle"] == first["preview_handle"]


@pytest.mark.asyncio
async def test_preview_broker_reuses_existing_session_handle_after_workspace_archive(
    repo,
    prototype_db,
    preview_broker,
):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)

    first = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
    )
    await repo.archive_workspace(workspace["id"])
    second = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
    )

    assert second["preview_handle"] == first["preview_handle"]
    assert second["token"]


@pytest.mark.asyncio
async def test_preview_broker_metadata_cannot_override_authoritative_snapshot_id(
    repo,
    prototype_db,
    preview_broker,
):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)

    grant = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
        metadata={"snapshot_id": "snap_spoofed", "runtime_profile_version": "v1"},
    )
    record = await repo.get_preview_handle_record(grant["preview_handle"])
    renewed = await preview_broker.renew_preview_grant(grant["preview_handle"])

    assert grant["snapshot_id"] == "snap_preview_base"
    assert renewed["snapshot_id"] == "snap_preview_base"
    assert record is not None
    assert record["metadata"]["snapshot_id"] == "snap_preview_base"


@pytest.mark.asyncio
async def test_preview_broker_replaces_active_handle_when_runtime_profile_version_changes(
    repo,
    prototype_db,
    preview_broker,
):
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)

    first = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
        metadata={"runtime_profile_version": "v1"},
    )
    second = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
        metadata={"runtime_profile_version": "v2"},
    )
    first_record = await repo.get_preview_handle_record(first["preview_handle"])
    second_record = await repo.get_preview_handle_record(second["preview_handle"])

    assert second["preview_handle"] != first["preview_handle"]
    assert first_record["is_active"] is False
    assert second_record["is_active"] is True
    assert second_record["metadata"]["runtime_profile_version"] == "v2"


@pytest.mark.asyncio
async def test_preview_broker_recovers_preview_record_after_memory_clear(repo, prototype_db, preview_broker):
    module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.preview_broker")
    broker_cls = _load_attr(module, "PrototypePreviewBroker", "PrototypeWorkspacePreviewBroker")
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)
    grant = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9018",
    )
    parsed = urlparse(grant["preview_url"])
    query = parse_qs(parsed.query)
    exp = int(query["exp"][0])

    broker_cls._records.clear()
    broker_cls._active_scope_handles.clear()
    recovered_broker = broker_cls(repo=repo)

    renewed = await recovered_broker.renew_preview_grant(grant["preview_handle"])
    validated = await recovered_broker.validate_preview_grant(
        preview_handle=grant["preview_handle"],
        token=grant["token"],
        exp=exp,
        actor_key=f"shared_actor:{session['actor_shared_actor_id']}",
    )

    assert renewed["preview_handle"] == grant["preview_handle"]
    assert validated is not None
    assert validated["preview_handle"] == grant["preview_handle"]


@pytest.mark.asyncio
async def test_revoked_shared_actor_blocks_future_preview_grants(repo, prototype_db, preview_broker):
    workspace, actor, session = await _seed_preview_scope(repo, prototype_db)

    await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9011",
    )
    await repo.revoke_shared_actor(
        actor["id"],
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )

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
    await repo.update_session_expiry(
        session["id"],
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )

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

    await repo.revoke_shared_actor(
        actor["id"],
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )

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


@pytest.mark.asyncio
async def test_failed_preview_persistence_does_not_restore_previous_after_concurrent_active_change(
    repo,
    prototype_db,
    preview_broker,
    monkeypatch,
):
    models_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.models")
    record_cls = _load_attr(models_module, "PrototypePreviewHandleRecord")
    preview_scope = _load_attr(models_module, "PrototypePreviewScope")
    preview_scope_id = _load_attr(models_module, "preview_scope_id")
    broker_module = importlib.import_module("tldw_Server_API.app.core.Prototype_Workspaces.preview_broker")
    broker_cls = _load_attr(broker_module, "PrototypePreviewBroker", "PrototypeWorkspacePreviewBroker")
    workspace, _actor, session = await _seed_preview_scope(repo, prototype_db)
    first = await preview_broker.issue_preview_grant(
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
        snapshot_id="snap_preview_base",
        runtime_target_url="http://127.0.0.1:9019",
    )
    scope_id = preview_scope_id(
        preview_scope="session",
        prototype_workspace_id=workspace["id"],
        prototype_session_id=session["id"],
    )

    async def fail_update(*_args: Any, **_kwargs: Any) -> None:
        replacement = record_cls(
            handle_id="pph_concurrent_active",
            preview_scope=preview_scope.SESSION,
            scope_id=scope_id,
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            actor_key=f"shared_actor:{session['actor_shared_actor_id']}",
            target_ref="http://127.0.0.1:9020",
            runtime_policy_profile="locked_collab",
            metadata={"snapshot_id": "snap_preview_base"},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with broker_cls._lock:
            previous = broker_cls._records[first["preview_handle"]]
            previous.is_active = False
            previous.revoked_at = "concurrent-revocation"
            broker_cls._records[replacement.handle_id] = replacement
            broker_cls._active_scope_handles[scope_id] = replacement.handle_id
        return None

    monkeypatch.setattr(repo, "update_session_state", fail_update)

    with pytest.raises(RuntimeError, match="failed to persist preview handle"):
        await preview_broker.issue_preview_grant(
            prototype_workspace_id=workspace["id"],
            prototype_session_id=session["id"],
            snapshot_id="snap_preview_base",
            runtime_target_url="http://127.0.0.1:9021",
        )

    assert broker_cls._active_scope_handles[scope_id] == "pph_concurrent_active"
    assert broker_cls._records[first["preview_handle"]].is_active is False
