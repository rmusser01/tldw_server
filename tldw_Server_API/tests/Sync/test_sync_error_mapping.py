from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint

@pytest.mark.asyncio
async def test_legacy_receive_changes_returns_replaced_gone():
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoint.receive_changes_from_client(
            request=SimpleNamespace(),
            user_id=SimpleNamespace(username="user-1"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE
    assert exc_info.value.detail["error_code"] == "sync_legacy_endpoint_replaced"
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/push"


@pytest.mark.asyncio
async def test_legacy_send_changes_to_client_returns_replaced_gone():
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoint.send_changes_to_client(
            request=SimpleNamespace(),
            user_id=SimpleNamespace(username="user-1"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE
    assert exc_info.value.detail["error_code"] == "sync_legacy_endpoint_replaced"
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/pull"
