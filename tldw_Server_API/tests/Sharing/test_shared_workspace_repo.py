"""Unit tests for SharedWorkspaceRepo."""
from __future__ import annotations

import sqlite3

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_create_share(repo):
    share = await repo.create_share(
        workspace_id="ws-1",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=10,
        access_level="view_chat",
        allow_clone=True,
        created_by=1,
    )
    assert share["workspace_id"] == "ws-1"
    assert share["owner_user_id"] == 1
    assert share["share_scope_type"] == "team"
    assert share["share_scope_id"] == 10
    assert share["access_level"] == "view_chat"
    assert share["allow_clone"] is True
    assert share["is_revoked"] is False


@pytest.mark.asyncio
async def test_create_share_duplicate_raises(repo):
    await repo.create_share(
        workspace_id="ws-1", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    with pytest.raises((sqlite3.IntegrityError, Exception)):  # noqa: B017
        await repo.create_share(
            workspace_id="ws-1", owner_user_id=1, share_scope_type="team",
            share_scope_id=10, created_by=1,
        )


@pytest.mark.asyncio
async def test_get_share(repo):
    created = await repo.create_share(
        workspace_id="ws-2", owner_user_id=1, share_scope_type="org",
        share_scope_id=5, created_by=1,
    )
    fetched = await repo.get_share(created["id"])
    assert fetched is not None
    assert fetched["workspace_id"] == "ws-2"
    assert fetched["share_scope_type"] == "org"


@pytest.mark.asyncio
async def test_get_share_not_found(repo):
    result = await repo.get_share(9999)
    assert result is None


@pytest.mark.asyncio
async def test_list_shares_for_workspace(repo):
    await repo.create_share(
        workspace_id="ws-3", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    await repo.create_share(
        workspace_id="ws-3", owner_user_id=1, share_scope_type="org",
        share_scope_id=5, created_by=1,
    )
    shares = await repo.list_shares_for_workspace("ws-3", 1)
    assert len(shares) == 2


@pytest.mark.asyncio
async def test_list_shares_excludes_revoked(repo):
    share = await repo.create_share(
        workspace_id="ws-4", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    await repo.revoke_share(share["id"])
    active = await repo.list_shares_for_workspace("ws-4", 1)
    assert len(active) == 0
    all_shares = await repo.list_shares_for_workspace("ws-4", 1, include_revoked=True)
    assert len(all_shares) == 1


@pytest.mark.asyncio
async def test_list_shares_for_scope(repo):
    await repo.create_share(
        workspace_id="ws-5", owner_user_id=1, share_scope_type="team",
        share_scope_id=42, created_by=1,
    )
    shares = await repo.list_shares_for_scope("team", 42)
    assert len(shares) == 1
    assert shares[0]["workspace_id"] == "ws-5"


@pytest.mark.asyncio
async def test_update_share(repo):
    share = await repo.create_share(
        workspace_id="ws-6", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, access_level="view_chat", created_by=1,
    )
    updated = await repo.update_share(share["id"], access_level="full_edit", allow_clone=False)
    assert updated is not None
    assert updated["access_level"] == "full_edit"
    assert updated["allow_clone"] is False


@pytest.mark.asyncio
async def test_update_share_not_found(repo):
    result = await repo.update_share(9999, access_level="full_edit")
    assert result is None


@pytest.mark.asyncio
async def test_revoke_share(repo):
    share = await repo.create_share(
        workspace_id="ws-7", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    result = await repo.revoke_share(share["id"])
    assert result is True
    fetched = await repo.get_share(share["id"])
    assert fetched["is_revoked"] is True
    assert fetched["revoked_at"] is not None


@pytest.mark.asyncio
async def test_revoke_share_nonexistent_returns_false(repo):
    result = await repo.revoke_share(9999)
    assert result is False


@pytest.mark.asyncio
async def test_revoke_share_already_revoked_returns_true(repo):
    share = await repo.create_share(
        workspace_id="ws-7b", owner_user_id=1, share_scope_type="team",
        share_scope_id=11, created_by=1,
    )
    await repo.revoke_share(share["id"])
    # Second revoke — already revoked, but revoked_at IS NOT NULL so returns True
    result = await repo.revoke_share(share["id"])
    assert result is True


@pytest.mark.asyncio
async def test_revoke_shares_for_workspace(repo):
    await repo.create_share(
        workspace_id="ws-8", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    await repo.create_share(
        workspace_id="ws-8", owner_user_id=1, share_scope_type="org",
        share_scope_id=5, created_by=1,
    )
    count = await repo.revoke_shares_for_workspace("ws-8", 1)
    assert count >= 2
    active = await repo.list_shares_for_workspace("ws-8", 1)
    assert len(active) == 0


@pytest.mark.asyncio
async def test_invalid_scope_type_raises(repo):
    with pytest.raises(ValueError, match="Invalid share_scope_type"):
        await repo.create_share(
            workspace_id="ws-x", owner_user_id=1, share_scope_type="invalid",
            share_scope_id=1, created_by=1,
        )


@pytest.mark.asyncio
async def test_invalid_access_level_raises(repo):
    with pytest.raises(ValueError, match="Invalid access_level"):
        await repo.create_share(
            workspace_id="ws-x", owner_user_id=1, share_scope_type="team",
            share_scope_id=1, access_level="admin", created_by=1,
        )


# ── Token CRUD ──

@pytest.mark.asyncio
async def test_create_and_get_token(repo):
    token = await repo.create_token(
        token_hash="abc123hash",
        token_prefix="abc12345",
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
    )
    assert token["token_hash"] == "abc123hash"
    assert token["resource_type"] == "workspace"
    assert token["use_count"] == 0
    assert token["is_revoked"] is False

    fetched = await repo.get_token(token["id"])
    assert fetched is not None
    assert fetched["token_prefix"] == "abc12345"


@pytest.mark.asyncio
async def test_find_tokens_by_prefix(repo):
    await repo.create_token(
        token_hash="hash1",
        token_prefix="pfx1",
        resource_type="chatbook",
        resource_id="cb-1",
        owner_user_id=1,
    )
    results = await repo.find_tokens_by_prefix("pfx1")
    assert len(results) == 1
    assert results[0]["token_hash"] == "hash1"


@pytest.mark.asyncio
async def test_list_tokens_for_user(repo):
    await repo.create_token(
        token_hash="h1", token_prefix="p1",
        resource_type="workspace", resource_id="ws-1", owner_user_id=1,
    )
    await repo.create_token(
        token_hash="h2", token_prefix="p2",
        resource_type="workspace", resource_id="ws-2", owner_user_id=1,
    )
    tokens = await repo.list_tokens_for_user(1)
    assert len(tokens) == 2


@pytest.mark.asyncio
async def test_increment_token_use_count(repo):
    token = await repo.create_token(
        token_hash="h3", token_prefix="p3",
        resource_type="workspace", resource_id="ws-1", owner_user_id=1,
    )
    await repo.increment_token_use_count(token["id"])
    await repo.increment_token_use_count(token["id"])
    fetched = await repo.get_token(token["id"])
    assert fetched["use_count"] == 2


@pytest.mark.asyncio
async def test_revoke_token(repo):
    token = await repo.create_token(
        token_hash="h4", token_prefix="p4",
        resource_type="workspace", resource_id="ws-1", owner_user_id=1,
    )
    result = await repo.revoke_token(token["id"])
    assert result is True
    fetched = await repo.get_token(token["id"])
    assert fetched["is_revoked"] is True


@pytest.mark.asyncio
async def test_revoke_token_nonexistent_returns_false(repo):
    result = await repo.revoke_token(9999)
    assert result is False


# ── Audit log ──

@pytest.mark.asyncio
async def test_log_and_list_audit_events(repo):
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        actor_user_id=1,
        metadata={"scope": "team"},
    )
    events = await repo.list_audit_events(owner_user_id=1)
    assert len(events) == 1
    assert events[0]["event_type"] == "share.created"
    assert events[0]["metadata"]["scope"] == "team"


@pytest.mark.asyncio
async def test_audit_events_filter_by_resource(repo):
    await repo.log_audit_event(
        event_type="share.created", resource_type="workspace",
        resource_id="ws-1", owner_user_id=1,
    )
    await repo.log_audit_event(
        event_type="token.created", resource_type="chatbook",
        resource_id="cb-1", owner_user_id=1,
    )
    ws_events = await repo.list_audit_events(resource_type="workspace")
    assert len(ws_events) == 1


# ── Config ──

@pytest.mark.asyncio
async def test_get_and_set_config(repo):
    await repo.set_config("default_access_level", "view_chat")
    config = await repo.get_config()
    assert config["default_access_level"] == "view_chat"


@pytest.mark.asyncio
async def test_config_upsert(repo):
    await repo.set_config("allow_clone", "true")
    await repo.set_config("allow_clone", "false")
    config = await repo.get_config()
    assert config["allow_clone"] == "false"


@pytest.mark.asyncio
async def test_config_scoped(repo):
    await repo.set_config("max_shares", "10", scope_type="org", scope_id=1)
    global_config = await repo.get_config()
    assert "max_shares" not in global_config
    org_config = await repo.get_config(scope_type="org", scope_id=1)
    assert org_config["max_shares"] == "10"


# ── Admin ──

@pytest.mark.asyncio
async def test_list_all_shares(repo):
    await repo.create_share(
        workspace_id="ws-a", owner_user_id=1, share_scope_type="team",
        share_scope_id=10, created_by=1,
    )
    all_shares = await repo.list_all_shares()
    assert len(all_shares) == 1


@pytest.mark.asyncio
async def test_list_all_shares_pagination(repo):
    for i in range(5):
        await repo.create_share(
            workspace_id=f"ws-{i}", owner_user_id=1, share_scope_type="team",
            share_scope_id=i + 100, created_by=1,
        )
    page1 = await repo.list_all_shares(limit=2, offset=0)
    page2 = await repo.list_all_shares(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2


@pytest.mark.asyncio
async def test_count_all_shares_respects_revoked_filter(repo):
    """Admin share counts should reflect the full filtered result set."""
    revoked = await repo.create_share(
        workspace_id="ws-revoked-count",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=201,
        created_by=1,
    )
    await repo.create_share(
        workspace_id="ws-active-count",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=202,
        created_by=1,
    )
    await repo.revoke_share(revoked["id"])

    assert await repo.count_all_shares(include_revoked=False) == 1
    assert await repo.count_all_shares(include_revoked=True) == 2


@pytest.mark.asyncio
async def test_count_audit_events_applies_query_filters(repo):
    """Admin audit counts should use the same filters as audit listing."""
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-count-1",
        owner_user_id=1,
    )
    await repo.log_audit_event(
        event_type="token.created",
        resource_type="chatbook",
        resource_id="cb-count-1",
        owner_user_id=1,
    )
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-count-2",
        owner_user_id=2,
    )

    assert await repo.count_audit_events(owner_user_id=1) == 2
    assert await repo.count_audit_events(resource_type="workspace") == 2
    assert await repo.count_audit_events(resource_id="ws-count-1") == 1
