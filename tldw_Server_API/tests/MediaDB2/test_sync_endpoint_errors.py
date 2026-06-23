from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoints
from tldw_Server_API.app.api.v1.schemas.sync_server_models import (
    ALLOWED_SYNC_SEND_ENTITIES as SCHEMA_ALLOWED_SYNC_SEND_ENTITIES,
)
from tldw_Server_API.app.api.v1.schemas.sync_server_models import (
    ClientChangesPayload,
    SyncSendLogEntry,
)
from tldw_Server_API.app.core.Sync.Sync_Client import (
    ALLOWED_SYNC_SEND_ENTITIES as CLIENT_ALLOWED_SYNC_SEND_ENTITIES,
)
from tldw_Server_API.app.core.Sync.sync_contract import (
    ALLOWED_SYNC_SEND_ENTITIES as CONTRACT_ALLOWED_SYNC_SEND_ENTITIES,
)


class _DummyUser:
    def __init__(self, username: str):
        self.username = username


def _build_payload_with_entity(entity: str) -> ClientChangesPayload:
    return ClientChangesPayload.model_construct(
        client_id="client_sender_1",
        last_processed_server_id=0,
        changes=[
            {
                "change_id": 1,
                "entity": entity,
                "entity_uuid": "entity-uuid-1",
                "operation": "create",
                "timestamp": "2023-10-27T11:00:00Z",
                "client_id": "client_sender_1",
                "version": 1,
                "payload": '{"uuid":"entity-uuid-1","keyword":"k1"}',
            }
        ],
    )


def test_sync_send_entity_allowlist_is_shared_across_layers() -> None:
    assert SCHEMA_ALLOWED_SYNC_SEND_ENTITIES == CONTRACT_ALLOWED_SYNC_SEND_ENTITIES
    assert CLIENT_ALLOWED_SYNC_SEND_ENTITIES == CONTRACT_ALLOWED_SYNC_SEND_ENTITIES


def test_client_changes_payload_schema_rejects_non_send_entity() -> None:
    with pytest.raises(ValidationError):
        ClientChangesPayload(
            client_id="client_sender_1",
            last_processed_server_id=0,
            changes=[
                SyncSendLogEntry(
                    change_id=1,
                    entity="Transcripts",
                    entity_uuid="entity-uuid-1",
                    operation="create",
                    timestamp="2023-10-27T11:00:00Z",
                    client_id="client_sender_1",
                    version=1,
                    payload='{"uuid":"entity-uuid-1","keyword":"k1"}',
                )
            ],
        )


@pytest.mark.asyncio
async def test_legacy_receive_changes_returns_replaced_gone() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            request=SimpleNamespace(),
            user_id=_DummyUser("sync-user"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE
    assert exc_info.value.detail["error_code"] == "sync_legacy_endpoint_replaced"
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/push"


@pytest.mark.asyncio
async def test_legacy_send_changes_to_client_returns_replaced_gone() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.send_changes_to_client(
            request=SimpleNamespace(),
            user_id=_DummyUser("sync-user"),
        )

    assert exc_info.value.status_code == status.HTTP_410_GONE
    assert exc_info.value.detail["error_code"] == "sync_legacy_endpoint_replaced"
    assert exc_info.value.detail["replacement"] == "/api/v1/sync/pull"


def test_server_sync_processor_rejects_disallowed_entity_with_400_semantics(memory_db_factory) -> None:
    db = memory_db_factory("server-test-client")
    payload = _build_payload_with_entity("Transcripts")
    processor = sync_endpoints.ServerSyncProcessor(
        db=db,
        user_id="sync-user",
        requesting_client_id=payload.client_id,
    )

    success, errors = processor.apply_client_changes_batch(list(payload.changes))

    assert success is False
    assert any("Unsupported sync entity 'Transcripts'" in error for error in errors)
