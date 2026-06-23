from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .sync_adapter import FileSyncWebhookSubscription


class BaseConnector(ABC):
    name: str

    def __init__(self, client_id: str | None = None, client_secret: str | None = None, redirect_base: str | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_base = (redirect_base or "http://localhost:8000").rstrip("/")

    @abstractmethod
    def authorize_url(self, state: str | None = None, scopes: list[str] | None = None, redirect_path: str = "/api/v1/connectors/callback") -> str:
        """Return provider-specific OAuth authorization URL."""
        raise NotImplementedError

    @abstractmethod
    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for tokens and account info."""
        raise NotImplementedError

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh tokens. Default: not implemented in scaffold."""
        return {"access_token": None, "expires_in": None}  # nosec B105 - scaffold placeholder, not a credential

    async def list_sources(
        self,
        account: dict[str, Any],
        parent_remote_id: str | None = None,
        *,
        page_size: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List folders/pages/databases when implemented by a provider."""
        _ = page_size
        _ = cursor
        raise NotImplementedError(f"{self.name} does not support source listing")

    async def list_files(
        self,
        account: dict[str, Any],
        remote_id: str,
        *,
        page_size: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List files under a folder/page when implemented by a provider."""
        _ = page_size
        _ = cursor
        raise NotImplementedError(f"{self.name} does not support file listing")

    async def download_file(self, account: dict[str, Any], file_id: str, **kwargs: Any) -> bytes:
        """Download file contents when implemented by a provider."""
        _ = kwargs
        raise NotImplementedError(f"{self.name} does not support file download")

    async def list_children(
        self,
        account: dict[str, Any],
        parent_remote_id: str,
        *,
        page_size: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Alias file-hosting providers onto the existing list_files contract."""
        return await self.list_files(
            account,
            parent_remote_id,
            page_size=page_size,
            cursor=cursor,
        )

    async def list_changes(
        self,
        account: dict[str, Any],
        *,
        cursor: str | None = None,
        page_size: int = 100,
    ) -> tuple[list[Any], str | None, str | None]:
        """Return provider-native deltas; providers override when they support sync."""
        _ = account
        _ = cursor
        _ = page_size
        raise NotImplementedError(f"{self.name} does not support change listing")

    async def get_item_metadata(
        self,
        account: dict[str, Any],
        remote_id: str,
    ) -> dict[str, Any] | None:
        """Fetch item metadata needed for sync reconciliation."""
        _ = account
        _ = remote_id
        raise NotImplementedError(f"{self.name} does not support metadata lookup")

    async def download_or_export(
        self,
        account: dict[str, Any],
        remote_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        """Bridge the sync adapter contract onto the existing download_file API."""
        download_kwargs: dict[str, Any] = {}
        if metadata:
            mime_type = metadata.get("mime_type") or metadata.get("mimeType")
            export_mime = metadata.get("export_mime")
            if mime_type:
                download_kwargs["mime_type"] = mime_type
            if export_mime:
                download_kwargs["export_mime"] = export_mime
        try:
            return await self.download_file(account, remote_id, **download_kwargs)
        except TypeError:
            return await self.download_file(account, remote_id)

    async def resolve_shared_link(
        self,
        account: dict[str, Any],
        shared_link: str,
    ) -> dict[str, Any] | None:
        """Resolve a shared link to a canonical remote item identifier."""
        _ = account
        _ = shared_link
        return None

    async def subscribe_webhook(
        self,
        account: dict[str, Any],
        *,
        resource: dict[str, Any],
        callback_url: str,
    ) -> FileSyncWebhookSubscription | None:
        """Create a provider webhook subscription when supported."""
        _ = account
        _ = resource
        _ = callback_url
        return None

    async def renew_webhook(
        self,
        account: dict[str, Any],
        *,
        subscription: FileSyncWebhookSubscription,
    ) -> FileSyncWebhookSubscription | None:
        """Renew a provider webhook subscription when supported."""
        _ = account
        _ = subscription
        return None

    async def revoke_webhook(
        self,
        account: dict[str, Any],
        *,
        subscription: FileSyncWebhookSubscription,
    ) -> bool:
        """Revoke a provider webhook subscription when supported."""
        _ = account
        _ = subscription
        return False
