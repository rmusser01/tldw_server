"""Tests for workspace deletion hook."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_repo_revokes_workspace_shares_and_tokens(repo):
    # Create a share and a token
    await repo.create_share(
        workspace_id="ws-del",
        owner_user_id=1,
        share_scope_type="team",
        share_scope_id=10,
        created_by=1,
    )
    await repo.create_token(
        token_hash="delhash",
        token_prefix="delpfx",
        resource_type="workspace",
        resource_id="ws-del",
        owner_user_id=1,
    )

    # Verify active
    shares = await repo.list_shares_for_workspace("ws-del", 1)
    assert len(shares) == 1
    tokens = await repo.list_tokens_for_user(1)
    assert len(tokens) == 1

    # Simulate what the deletion hook does: revoke shares and tokens directly
    await repo.revoke_shares_for_workspace("ws-del", 1)
    await repo.revoke_tokens_for_resource("workspace", "ws-del", 1)

    # Verify revoked
    shares_after = await repo.list_shares_for_workspace("ws-del", 1)
    assert len(shares_after) == 0


@pytest.mark.asyncio
async def test_on_workspace_deleted_hook_swallows_errors():
    """Hook should not raise even if DB is unavailable."""
    from tldw_Server_API.app.core.Sharing.workspace_deletion_hook import on_workspace_deleted

    with patch(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        side_effect=RuntimeError("DB unavailable"),
    ):
        # Should not raise
        await on_workspace_deleted("ws-missing", 999)


@pytest.mark.asyncio
async def test_on_workspace_deleted_hook_failure_log_is_sanitized():
    """Fail-open hook logs should not expose backend exception details."""
    from tldw_Server_API.app.core.Sharing import workspace_deletion_hook

    sensitive_error = RuntimeError(
        "sqlite:///tmp/private/users.db password=supersecret token=abc123"
    )

    with (
        patch(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            new=MagicMock(return_value=MagicMock()),
        ),
        patch(
            "tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo.SharedWorkspaceRepo",
            side_effect=sensitive_error,
        ),
        patch.object(workspace_deletion_hook, "logger") as fake_logger,
    ):
        await workspace_deletion_hook.on_workspace_deleted("ws-missing", 999)

    logged_text = " ".join(
        str(part)
        for call in fake_logger.warning.call_args_list
        for part in call.args
    )
    assert "workspace_deletion_hook failed" in logged_text
    assert "sqlite://" not in logged_text
    assert "/tmp/private/users.db" not in logged_text
    assert "supersecret" not in logged_text
    assert "abc123" not in logged_text
