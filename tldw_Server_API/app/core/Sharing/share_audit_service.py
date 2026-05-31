"""Service for recording and querying share audit events."""
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import AuditLogError
from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo
from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

# Standard event types
SHARE_CREATED = "share.created"
SHARE_UPDATED = "share.updated"
SHARE_REVOKED = "share.revoked"
SHARE_ACCESSED = "share.accessed"
SHARE_CLONED = "share.cloned"
TOKEN_CREATED = "token.created"  # nosec B105
TOKEN_USED = "token.used"  # nosec B105
TOKEN_REVOKED = "token.revoked"  # nosec B105
TOKEN_PASSWORD_VERIFIED = "token.password_verified"  # nosec B105
TOKEN_PASSWORD_FAILED = "token.password_failed"  # nosec B105


class ShareAuditService:
    """Records sharing events for audit and compliance."""

    def __init__(
        self,
        repo: SharedWorkspaceRepo | None = None,
        writer: UnifiedShareAuditWriter | None = None,
    ) -> None:
        self._repo = repo
        self._writer = writer
        self._owns_writer = False

    async def _ensure_writer(self) -> UnifiedShareAuditWriter:
        """Return the injected writer or lazily create and cache one."""
        if self._writer is not None:
            return self._writer
        self._writer = UnifiedShareAuditWriter()
        self._owns_writer = True
        return self._writer

    async def stop(self) -> None:
        """Shut down the cached writer if we own it."""
        if self._owns_writer and self._writer is not None:
            await self._writer.stop()
            self._writer = None
            self._owns_writer = False

    async def log(
        self,
        event_type: str,
        *,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        actor_user_id: int | None = None,
        share_id: int | None = None,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        try:
            if self._repo is not None and self._writer is None:
                await self._repo.log_audit_event(
                    event_type=event_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    owner_user_id=owner_user_id,
                    actor_user_id=actor_user_id,
                    share_id=share_id,
                    token_id=token_id,
                    metadata=metadata,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
                return

            writer = await self._ensure_writer()
            await writer.log_event(
                event_type=event_type,
                resource_type=resource_type,
                resource_id=resource_id,
                owner_user_id=owner_user_id,
                actor_user_id=actor_user_id,
                share_id=share_id,
                token_id=token_id,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        except Exception as exc:
            logger.error(f"ShareAuditService.log failed; exception_type={type(exc).__name__}")
            raise AuditLogError(f"Failed to log share audit event: {event_type}") from exc

    async def query(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if self._repo is not None and self._writer is None:
            return await self._repo.list_audit_events(
                owner_user_id=owner_user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                limit=limit,
                offset=offset,
            )

        writer = await self._ensure_writer()
        return await writer.query_events(
            owner_user_id=owner_user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            limit=limit,
            offset=offset,
        )

    async def count(
        self,
        *,
        owner_user_id: int | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> int:
        """Return the full count for the same filters accepted by query()."""
        if self._repo is not None and self._writer is None:
            return await self._repo.count_audit_events(
                owner_user_id=owner_user_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )

        writer = await self._ensure_writer()
        return await writer.count_events(
            owner_user_id=owner_user_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
