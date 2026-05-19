from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.api.v1.schemas.sync_server_models import ClientChangesPayload


def _sync_payload() -> ClientChangesPayload:
    return ClientChangesPayload(
        client_id="client-1",
        changes=[
            {
                "change_id": 1,
                "entity": "Media",
                "entity_uuid": "media-1",
                "operation": "update",
                "timestamp": "2026-04-23T00:00:00Z",
                "client_id": "client-1",
                "version": 2,
                "payload": "{}",
            }
        ],
    )


class _FailingSyncProcessor:
    def __init__(self, *, db, user_id, requesting_client_id):
        self.db = db
        self.user_id = user_id
        self.requesting_client_id = requesting_client_id

    def apply_client_changes_batch(self, changes):
        assert changes
        return False, ["sqlite backend exploded at /private/sync.db"]


class _ValidationFailingSyncProcessor(_FailingSyncProcessor):
    def apply_client_changes_batch(self, changes):
        assert changes
        return False, [
            "unsupported sync entity: Widgets",
            "failed processing change id 1; rolling back batch",
        ]


@pytest.mark.asyncio
async def test_receive_changes_sanitizes_internal_batch_failure(monkeypatch):
    monkeypatch.setattr(sync_endpoint, "ServerSyncProcessor", _FailingSyncProcessor)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoint.receive_changes_from_client(
            payload=_sync_payload(),
            user_id=SimpleNamespace(username="user-1"),
            db=SimpleNamespace(db_path_str="/private/user-media.db"),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == {
        "message": "Failed to apply changes atomically.",
        "errors": ["Internal sync processing failed."],
    }


@pytest.mark.asyncio
async def test_receive_changes_preserves_client_validation_errors(monkeypatch):
    monkeypatch.setattr(sync_endpoint, "ServerSyncProcessor", _ValidationFailingSyncProcessor)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoint.receive_changes_from_client(
            payload=_sync_payload(),
            user_id=SimpleNamespace(username="user-1"),
            db=SimpleNamespace(db_path_str="/private/user-media.db"),
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == {
        "message": "Failed to apply changes atomically.",
        "errors": [
            "unsupported sync entity: Widgets",
            "failed processing change id 1; rolling back batch",
        ],
    }
