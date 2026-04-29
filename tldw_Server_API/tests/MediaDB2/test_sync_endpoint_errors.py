# tests/test_sync_endpoint_errors.py
# Description: Regression tests for sync endpoint error handling behavior.
#
# Imports
import sqlite3

import pytest
from fastapi import HTTPException, status
from pydantic import ValidationError

#
# Local Imports
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
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError

#
#######################################################################################################################
#
# Functions:


class _DummyUser:
    def __init__(self, username: str):
        self.username = username


def _capture_sync_endpoint_logs(level: str = "ERROR") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = sync_endpoints.logger.add(
        lambda message: messages.append(str(message)),
        level=level,
        format="{message} {extra}",
    )
    return messages, sink_id


def _build_payload() -> ClientChangesPayload:
    return ClientChangesPayload(
        client_id="client_sender_1",
        last_processed_server_id=0,
        changes=[
            SyncSendLogEntry(
                change_id=1,
                entity="Keywords",
                entity_uuid="kw-uuid-1",
                operation="create",
                timestamp="2023-10-27T11:00:00Z",
                client_id="client_sender_1",
                version=1,
                payload='{"uuid":"kw-uuid-1","keyword":"k1"}',
            )
        ],
    )


def _build_payload_with_entity(entity: str) -> ClientChangesPayload:
    # Bypass pydantic validation so the endpoint's defensive runtime validation
    # path can still be exercised for malformed internal payloads.
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
                {
                    "change_id": 1,
                    "entity": "Transcripts",
                    "entity_uuid": "entity-uuid-1",
                    "operation": "create",
                    "timestamp": "2023-10-27T11:00:00Z",
                    "client_id": "client_sender_1",
                    "version": 1,
                    "payload": '{"uuid":"entity-uuid-1","keyword":"k1"}',
                }
            ],
        )


@pytest.mark.asyncio
async def test_receive_changes_preserves_http_exception_from_processing(memory_db_factory, monkeypatch):
    """A processor-generated HTTPException should not be wrapped into a generic 500."""
    db = memory_db_factory("server-test-client")

    async def _fake_to_thread(*_args, **_kwargs):
        return False, ["payload decode error"]

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            payload=_build_payload(),
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert isinstance(exc_info.value.detail, dict)
    assert "Failed to apply changes atomically." in exc_info.value.detail.get("message", "")


@pytest.mark.asyncio
async def test_receive_changes_sanitizes_unexpected_exception(memory_db_factory, monkeypatch):
    """Unexpected exceptions should return a sanitized 500 detail for /sync/send."""
    db = memory_db_factory("server-test-client")

    async def _fake_to_thread(*_args, **_kwargs):
        raise RuntimeError("sensitive db internals")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            payload=_build_payload(),
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Internal server error while processing sync changes."
    assert "sensitive db internals" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_receive_changes_rejects_disallowed_entity_with_400(memory_db_factory):
    """Disallowed sync entities for /sync/send should fail with 400."""
    db = memory_db_factory("server-test-client")
    payload = _build_payload_with_entity("Transcripts")

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            payload=payload,
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert isinstance(exc_info.value.detail, dict)
    assert "Failed to apply changes atomically." in exc_info.value.detail.get("message", "")
    errors = exc_info.value.detail.get("errors", [])
    assert any("Unsupported sync entity 'Transcripts'" in err for err in errors)


@pytest.mark.asyncio
async def test_receive_changes_internal_invalid_error_stays_500(memory_db_factory, monkeypatch):
    """Server-side failures containing the word 'invalid' must not be misclassified as 400."""
    db = memory_db_factory("server-test-client")

    async def _fake_to_thread(*_args, **_kwargs):
        return False, ["Internal invalid state while writing sync transaction"]

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.receive_changes_from_client(
            payload=_build_payload(),
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_send_changes_sanitizes_unexpected_exception(memory_db_factory, monkeypatch):
    """Unexpected exceptions should return a sanitized 500 detail for /sync/get."""
    db = memory_db_factory("server-test-client")

    async def _fake_to_thread(*_args, **_kwargs):
        raise RuntimeError("raw backend failure text")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.send_changes_to_client(
            client_id="client_sender_1",
            since_change_id=0,
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Internal server error while retrieving sync changes."
    assert "raw backend failure text" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_send_changes_unexpected_error_log_is_sanitized(memory_db_factory, monkeypatch):
    db = memory_db_factory("server-test-client")
    messages, sink_id = _capture_sync_endpoint_logs()

    async def _fake_to_thread(*_args, **_kwargs):
        raise RuntimeError("unexpected sync failed for sync-user client_sender_1 /private/sync.db")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await sync_endpoints.send_changes_to_client(
                client_id="client_sender_1",
                since_change_id=0,
                user_id=_DummyUser("sync-user"),
                db=db,
            )
    finally:
        sync_endpoints.logger.remove(sink_id)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Internal server error while retrieving sync changes."
    joined = "\n".join(messages)
    assert "unexpected sync failed for sync-user client_sender_1 /private/sync.db" not in joined
    assert "sync-user" not in joined
    assert "client_sender_1" not in joined
    assert "/private/sync.db" not in joined
    assert "Unexpected server error getting changes from sync store" in joined
    assert "error_type': 'RuntimeError'" in joined


@pytest.mark.asyncio
async def test_send_changes_preserves_database_error_detail(memory_db_factory, monkeypatch):
    """DatabaseError should keep the dedicated /sync/get 500 detail."""
    db = memory_db_factory("server-test-client")

    async def _fake_to_thread(*_args, **_kwargs):
        raise DatabaseError("sync reader failed")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    with pytest.raises(HTTPException) as exc_info:
        await sync_endpoints.send_changes_to_client(
            client_id="client_sender_1",
            since_change_id=0,
            user_id=_DummyUser("sync-user"),
            db=db,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to retrieve changes from database."


@pytest.mark.asyncio
async def test_send_changes_database_error_log_is_sanitized(memory_db_factory, monkeypatch):
    db = memory_db_factory("server-test-client")
    messages, sink_id = _capture_sync_endpoint_logs()

    async def _fake_to_thread(*_args, **_kwargs):
        raise DatabaseError("sync reader failed for sync-user client_sender_1 /private/sync.db")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await sync_endpoints.send_changes_to_client(
                client_id="client_sender_1",
                since_change_id=0,
                user_id=_DummyUser("sync-user"),
                db=db,
            )
    finally:
        sync_endpoints.logger.remove(sink_id)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to retrieve changes from database."
    joined = "\n".join(messages)
    assert "sync reader failed for sync-user client_sender_1 /private/sync.db" not in joined
    assert "sync-user" not in joined
    assert "client_sender_1" not in joined
    assert "/private/sync.db" not in joined
    assert "Database error getting changes from sync store" in joined
    assert "error_type': 'DatabaseError'" in joined


@pytest.mark.asyncio
async def test_send_changes_sqlite_error_log_is_sanitized(memory_db_factory, monkeypatch):
    db = memory_db_factory("server-test-client")
    messages, sink_id = _capture_sync_endpoint_logs()

    async def _fake_to_thread(*_args, **_kwargs):
        raise sqlite3.OperationalError("sqlite failed for sync-user client_sender_1 /private/sync.db")

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await sync_endpoints.send_changes_to_client(
                client_id="client_sender_1",
                since_change_id=0,
                user_id=_DummyUser("sync-user"),
                db=db,
            )
    finally:
        sync_endpoints.logger.remove(sink_id)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Failed to retrieve changes from database."
    joined = "\n".join(messages)
    assert "sqlite failed for sync-user client_sender_1 /private/sync.db" not in joined
    assert "sync-user" not in joined
    assert "client_sender_1" not in joined
    assert "/private/sync.db" not in joined
    assert "Database error getting changes from sync store" in joined
    assert "error_type': 'OperationalError'" in joined


@pytest.mark.asyncio
async def test_send_changes_returns_partial_success_when_rows_are_invalid(memory_db_factory, monkeypatch):
    """Invalid rows should be skipped while valid rows are still returned."""
    db = memory_db_factory("server-test-client")

    valid_row = {
        "change_id": 10,
        "entity": "Keywords",
        "entity_uuid": "kw-uuid-valid",
        "operation": "create",
        "timestamp": "2023-10-27T11:00:00Z",
        "client_id": "other-client",
        "version": 1,
        "payload": '{"uuid":"kw-uuid-valid","keyword":"valid"}',
    }
    invalid_row = {
        "change_id": 11,
        "entity": "UnsupportedEntity",
        "entity_uuid": "kw-uuid-invalid",
        "operation": "create",
        "timestamp": "2023-10-27T11:01:00Z",
        "client_id": "other-client",
        "version": 1,
        "payload": '{"uuid":"kw-uuid-invalid","keyword":"invalid"}',
    }

    async def _fake_to_thread(*_args, **_kwargs):
        return [valid_row, invalid_row], 11

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    response = await sync_endpoints.send_changes_to_client(
        client_id="client_sender_1",
        since_change_id=0,
        user_id=_DummyUser("sync-user"),
        db=db,
    )
    assert response.latest_change_id == 11
    assert len(response.changes) == 1
    assert response.changes[0].change_id == 10
    assert response.changes[0].entity == "Keywords"


@pytest.mark.asyncio
async def test_send_changes_invalid_row_validation_log_is_sanitized(memory_db_factory, monkeypatch):
    db = memory_db_factory("server-test-client")
    messages, sink_id = _capture_sync_endpoint_logs()

    invalid_row = {
        "change_id": 11,
        "entity": "UnsupportedEntity",
        "entity_uuid": "kw-uuid-invalid",
        "operation": "create",
        "timestamp": "2023-10-27T11:01:00Z",
        "client_id": "other-client",
        "version": 1,
        "payload": '{"uuid":"kw-uuid-invalid","keyword":"invalid"}',
    }

    async def _fake_to_thread(*_args, **_kwargs):
        return [invalid_row], 11

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    try:
        response = await sync_endpoints.send_changes_to_client(
            client_id="client_sender_1",
            since_change_id=0,
            user_id=_DummyUser("sync-user"),
            db=db,
        )
    finally:
        sync_endpoints.logger.remove(sink_id)

    assert response.latest_change_id == 11
    assert response.changes == []
    joined = "\n".join(messages)
    assert "UnsupportedEntity" not in joined
    assert "Input should be" not in joined
    assert "sync-user" not in joined
    assert "Error validating sync log entry data" not in joined
    assert "Invalid sync log entry encountered during /sync/get validation" in joined
    assert "error_type': 'ValidationError'" in joined


@pytest.mark.asyncio
async def test_send_changes_returns_empty_when_all_rows_invalid(memory_db_factory, monkeypatch):
    """All-invalid rows should return 200 with empty changes and latest server id."""
    db = memory_db_factory("server-test-client")

    invalid_row_1 = {
        "change_id": 21,
        "entity": "UnsupportedEntity",
        "entity_uuid": "invalid-1",
        "operation": "create",
        "timestamp": "2023-10-27T11:01:00Z",
        "client_id": "other-client",
        "version": 1,
        "payload": '{"uuid":"invalid-1"}',
    }
    invalid_row_2 = {
        "change_id": 22,
        "entity": "Keywords",
        "entity_uuid": "invalid-2",
        "operation": "invalid-op",
        "timestamp": "2023-10-27T11:02:00Z",
        "client_id": "other-client",
        "version": 1,
        "payload": '{"uuid":"invalid-2"}',
    }

    async def _fake_to_thread(*_args, **_kwargs):
        return [invalid_row_1, invalid_row_2], 22

    monkeypatch.setattr(sync_endpoints.asyncio, "to_thread", _fake_to_thread)

    response = await sync_endpoints.send_changes_to_client(
        client_id="client_sender_1",
        since_change_id=0,
        user_id=_DummyUser("sync-user"),
        db=db,
    )

    assert response.latest_change_id == 22
    assert response.changes == []

#
# End of test_sync_endpoint_errors.py
#######################################################################################################################
