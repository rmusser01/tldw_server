"""
Hook for revoking shares when a workspace is deleted.

Call `on_workspace_deleted(workspace_id, owner_user_id)` after deleting
a workspace to revoke all active shares and tokens for that workspace.

This module is designed to be called from the workspace deletion endpoint
or service layer, NOT from inside the ChaChaNotes_DB class (to avoid
coupling the DB layer to the sharing module).
"""
from __future__ import annotations

from loguru import logger


async def on_workspace_deleted(workspace_id: str, owner_user_id: int) -> None:
    """
    Revoke all shares and tokens for a deleted workspace.

    Safe to call even if the sharing module is not fully initialized —
    errors are logged but never propagated.
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo

        pool = get_db_pool()
        repo = SharedWorkspaceRepo(db_pool=pool)

        # Revoke all workspace shares
        await repo.revoke_shares_for_workspace(workspace_id, owner_user_id)
        logger.info(f"Revoked shares for deleted workspace {workspace_id}")

        # Revoke all tokens pointing to this workspace
        await repo.revoke_tokens_for_resource("workspace", workspace_id, owner_user_id)
        logger.info(f"Revoked tokens for deleted workspace {workspace_id}")

        # Audit log the event
        from tldw_Server_API.app.core.Sharing.share_audit_service import ShareAuditService
        audit = ShareAuditService(repo)
        await audit.log(
            "share.workspace_deleted",
            resource_type="workspace",
            resource_id=workspace_id,
            owner_user_id=owner_user_id,
            metadata={"trigger": "workspace_deletion"},
        )
    except Exception:
        logger.warning(f"workspace_deletion_hook failed for {workspace_id}")
