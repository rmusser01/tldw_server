from __future__ import annotations

import pytest

from tldw_Server_API.app.core.External_Sources import connectors_service as svc
from tldw_Server_API.app.core.External_Sources.connector_base import BaseConnector
from tldw_Server_API.app.core.External_Sources.sync_adapter import (
    FileSyncAdapter,
    FileSyncChange,
)


@pytest.mark.unit
def test_file_sync_change_normalizes_required_fields() -> None:
    change = FileSyncChange(
        event_type="content_updated",
        remote_id="abc123",
        remote_name="report.pdf",
    )

    assert change.event_type == "content_updated"
    assert change.remote_id == "abc123"
    assert change.remote_name == "report.pdf"


@pytest.mark.unit
def test_get_file_sync_connector_by_name_restricts_to_file_sync_providers() -> None:
    connector = svc.get_file_sync_connector_by_name("drive")

    assert connector.name == "drive"
    assert isinstance(connector, FileSyncAdapter)

    with pytest.raises(ValueError, match="does not support file sync"):
        svc.get_file_sync_connector_by_name("gmail")


class _MinimalConnector(BaseConnector):
    name = "minimal"

    def authorize_url(
        self,
        state: str | None = None,
        scopes: list[str] | None = None,
        redirect_path: str = "/api/v1/connectors/callback",
    ) -> str:
        return "https://example.test/oauth"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, object]:
        return {"access_token": code, "redirect_uri": redirect_uri}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_base_connector_unsupported_capabilities_raise() -> None:
    connector = _MinimalConnector()

    with pytest.raises(NotImplementedError, match="does not support source listing"):
        await connector.list_sources({})

    with pytest.raises(NotImplementedError, match="does not support file listing"):
        await connector.list_files({}, "parent-1")

    with pytest.raises(NotImplementedError, match="does not support file download"):
        await connector.download_file({}, "file-1")
