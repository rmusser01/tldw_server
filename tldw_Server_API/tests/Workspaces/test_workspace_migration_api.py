"""Tests for Research Workspace migration protocol endpoints."""
import inspect
from typing import NoReturn, get_type_hints

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspace_migrations
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    StatusResponse,
    WorkspaceMigrationChunkReceiptResponse,
    WorkspaceMigrationResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)


_MANIFEST_HASH = "a" * 64
_CHUNK_HASH = "b" * 64
_OTHER_CHUNK_HASH = "c" * 64


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


@pytest.fixture
def workspace_fastapi_app():
    from tldw_Server_API.app.main import app

    return app


@pytest.fixture
def workspace_client(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            yield client
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _session_payload(**overrides):
    payload = {
        "id": "mig-1",
        "idempotency_key": "legacy-import-1",
        "target_workspace_id": "rw-import-1",
        "target_workspace_name": "Migrated Research Workspace",
        "source_product": "research-workspace-webui",
        "manifest_hash": _MANIFEST_HASH,
        "declared_chunks": [
            {
                "id": "chunk-1",
                "sha256": _CHUNK_HASH,
                "byte_count": 64,
                "chunk_kind": "workspace_bundle",
            }
        ],
        "manifest": {
            "workspace": {"legacy_id": "local-workspace", "name": "Local Workspace"},
            "sources": [{"id": "src-1", "title": "Notebook export"}],
        },
        "diagnostics": {
            "legacy_storage_keys": ["tldw-workspace"],
            "local_inventory_status": "eligible_for_import",
        },
    }
    payload.update(overrides)
    return payload


def _create_session(client: TestClient, **overrides):
    response = client.post("/api/v1/workspaces/migrations", json=_session_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.integration
def test_workspace_migration_session_create_read_and_route_order(workspace_client):
    created = _create_session(workspace_client)

    assert created["id"] == "mig-1"
    assert created["target_workspace_id"] == "rw-import-1"
    assert created["status"] == "created"
    assert created["declared_chunk_count"] == 1
    assert created["accepted_chunk_count"] == 0
    assert created["missing_chunk_ids"] == ["chunk-1"]
    assert created["client_delete_eligible"] is False
    assert created["recovery_manifest"]["can_delete_legacy_storage"] is False

    fetched = workspace_client.get("/api/v1/workspaces/migrations/mig-1")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == "mig-1"

    not_a_workspace = workspace_client.get("/api/v1/workspaces/migrations")
    assert not_a_workspace.status_code != 404, not_a_workspace.text


@pytest.mark.integration
def test_workspace_migration_create_is_idempotent_for_same_manifest(workspace_client):
    first = _create_session(workspace_client)
    second = workspace_client.post("/api/v1/workspaces/migrations", json=_session_payload())

    assert second.status_code == 200, second.text
    assert second.json()["id"] == first["id"]
    assert second.json()["manifest_hash"] == _MANIFEST_HASH


@pytest.mark.integration
def test_workspace_migration_create_conflicts_for_same_id_different_manifest(workspace_client):
    _create_session(workspace_client)

    response = workspace_client.post(
        "/api/v1/workspaces/migrations",
        json=_session_payload(manifest_hash="d" * 64),
    )

    assert response.status_code == 409, response.text


@pytest.mark.integration
def test_workspace_migration_chunk_receipt_is_idempotent_and_conflict_checked(workspace_client):
    _create_session(workspace_client)

    accepted = workspace_client.put(
        "/api/v1/workspaces/migrations/mig-1/chunks/chunk-1",
        json={
            "sha256": _CHUNK_HASH,
            "byte_count": 64,
            "chunk_kind": "workspace_bundle",
            "metadata": {"source_count": 1},
        },
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["id"] == "chunk-1"
    assert accepted.json()["status"] == "accepted"

    duplicate = workspace_client.put(
        "/api/v1/workspaces/migrations/mig-1/chunks/chunk-1",
        json={
            "sha256": _CHUNK_HASH,
            "byte_count": 64,
            "chunk_kind": "workspace_bundle",
            "metadata": {"source_count": 1},
        },
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["sha256"] == _CHUNK_HASH

    conflicting = workspace_client.put(
        "/api/v1/workspaces/migrations/mig-1/chunks/chunk-1",
        json={
            "sha256": _OTHER_CHUNK_HASH,
            "byte_count": 64,
            "chunk_kind": "workspace_bundle",
        },
    )
    assert conflicting.status_code == 409, conflicting.text


@pytest.mark.integration
def test_workspace_migration_finalize_requires_all_chunks_and_keeps_delete_ineligible(workspace_client):
    _create_session(
        workspace_client,
        declared_chunks=[
            {
                "id": "chunk-1",
                "sha256": _CHUNK_HASH,
                "byte_count": 64,
                "chunk_kind": "workspace_bundle",
            },
            {
                "id": "chunk-2",
                "sha256": _OTHER_CHUNK_HASH,
                "byte_count": 32,
                "chunk_kind": "artifact_payloads",
            },
        ],
    )

    premature = workspace_client.post(
        "/api/v1/workspaces/migrations/mig-1/finalize",
        json={"manifest_hash": _MANIFEST_HASH},
    )
    assert premature.status_code == 409, premature.text
    assert premature.json()["detail"]["missing_chunk_ids"] == ["chunk-1", "chunk-2"]

    for chunk_id, chunk_hash, byte_count, kind in (
        ("chunk-1", _CHUNK_HASH, 64, "workspace_bundle"),
        ("chunk-2", _OTHER_CHUNK_HASH, 32, "artifact_payloads"),
    ):
        accepted = workspace_client.put(
            f"/api/v1/workspaces/migrations/mig-1/chunks/{chunk_id}",
            json={
                "sha256": chunk_hash,
                "byte_count": byte_count,
                "chunk_kind": kind,
            },
        )
        assert accepted.status_code == 200, accepted.text

    finalized = workspace_client.post(
        "/api/v1/workspaces/migrations/mig-1/finalize",
        json={"manifest_hash": _MANIFEST_HASH},
    )
    assert finalized.status_code == 200, finalized.text
    body = finalized.json()
    assert body["status"] == "finalized"
    assert body["accepted_chunk_count"] == 2
    assert body["missing_chunk_ids"] == []
    assert body["client_delete_eligible"] is False
    assert body["recovery_manifest"]["can_delete_legacy_storage"] is False

    delete_ack = workspace_client.post(
        "/api/v1/workspaces/migrations/mig-1/client-delete-ack",
        json={"acknowledged_manifest_hash": _MANIFEST_HASH},
    )
    assert delete_ack.status_code == 409, delete_ack.text


@pytest.mark.integration
def test_workspace_migration_accepts_declared_zero_byte_chunk(workspace_client):
    _create_session(
        workspace_client,
        declared_chunks=[
            {
                "id": "chunk-empty",
                "sha256": _CHUNK_HASH,
                "byte_count": 0,
                "chunk_kind": "workspace_bundle",
            }
        ],
    )

    accepted = workspace_client.put(
        "/api/v1/workspaces/migrations/mig-1/chunks/chunk-empty",
        json={
            "sha256": _CHUNK_HASH,
            "byte_count": 0,
            "chunk_kind": "workspace_bundle",
        },
    )

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["byte_count"] == 0


@pytest.mark.integration
def test_workspace_migration_rejects_oversized_chunk_receipt(workspace_client):
    _create_session(workspace_client)

    response = workspace_client.put(
        "/api/v1/workspaces/migrations/mig-1/chunks/chunk-1",
        json={
            "sha256": _CHUNK_HASH,
            "byte_count": 2 * 1024 * 1024 + 1,
            "chunk_kind": "workspace_bundle",
        },
    )

    assert response.status_code == 422, response.text


@pytest.mark.unit
def test_workspace_migration_endpoints_are_sync_handlers_with_return_annotations():
    expected_annotations = {
        workspace_migrations.create_workspace_migration: WorkspaceMigrationResponse,
        workspace_migrations.list_workspace_migrations: list[WorkspaceMigrationResponse],
        workspace_migrations.get_workspace_migration: WorkspaceMigrationResponse,
        workspace_migrations.put_workspace_migration_chunk: WorkspaceMigrationChunkReceiptResponse,
        workspace_migrations.finalize_workspace_migration: WorkspaceMigrationResponse,
        workspace_migrations.acknowledge_workspace_migration_client_delete: StatusResponse,
    }

    for endpoint, expected_return in expected_annotations.items():
        assert not inspect.iscoroutinefunction(endpoint)
        assert get_type_hints(endpoint)["return"] == expected_return


class _BackendErrorConnection:
    def __init__(self, message: str) -> None:
        self.message = message

    def execute(self, *_args, **_kwargs) -> NoReturn:
        raise BackendDatabaseError(self.message)


class _BackendErrorTransaction:
    def __init__(self, message: str) -> None:
        self.message = message

    def __enter__(self) -> _BackendErrorConnection:
        return _BackendErrorConnection(self.message)

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        return False


@pytest.mark.unit
def test_workspace_migration_session_postgres_duplicate_race_returns_existing(db, monkeypatch):
    raced_row = {
        "id": "mig-1",
        "idempotency_key": "legacy-import-1",
        "target_workspace_id": "rw-import-1",
        "target_workspace_name": "Migrated Research Workspace",
        "source_product": "research-workspace-webui",
        "manifest_hash": _MANIFEST_HASH,
        "manifest_json": "{}",
        "diagnostics_json": "{}",
        "declared_chunks_json": "[]",
        "status": "created",
        "client_delete_eligible": False,
        "created_at": "2026-05-24T00:00:00Z",
        "updated_at": "2026-05-24T00:00:00Z",
        "finalized_at": None,
        "recovery_manifest_json": None,
        "_chunks": [],
    }
    reads = iter([None, raced_row])

    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: next(reads))
    monkeypatch.setattr(db, "_get_workspace_migration_session_by_idempotency_key", lambda _key: None)
    monkeypatch.setattr(db, "upsert_workspace", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        db,
        "transaction",
        lambda: _BackendErrorTransaction("duplicate key value violates unique constraint"),
    )

    row, created = db.upsert_workspace_migration_session(_session_payload())

    assert created is False
    assert row == raced_row


@pytest.mark.unit
def test_workspace_migration_session_postgres_backend_error_is_wrapped(db, monkeypatch):
    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: None)
    monkeypatch.setattr(db, "_get_workspace_migration_session_by_idempotency_key", lambda _key: None)
    monkeypatch.setattr(db, "upsert_workspace", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(db, "transaction", lambda: _BackendErrorTransaction("connection reset"))

    with pytest.raises(CharactersRAGDBError, match="workspace migration"):
        db.upsert_workspace_migration_session(_session_payload())


@pytest.mark.unit
def test_workspace_migration_chunk_postgres_duplicate_race_returns_existing(db, monkeypatch):
    session = {
        "id": "mig-1",
        "manifest_hash": _MANIFEST_HASH,
        "status": "created",
        "declared_chunks_json": (
            '[{"id":"chunk-1","sha256":"%s","byte_count":64,"chunk_kind":"workspace_bundle"}]' % _CHUNK_HASH
        ),
        "_chunks": [],
    }
    raced_chunk = {
        "id": "chunk-1",
        "migration_id": "mig-1",
        "sha256": _CHUNK_HASH,
        "byte_count": 64,
        "chunk_kind": "workspace_bundle",
        "metadata_json": "{}",
        "accepted_at": "2026-05-24T00:00:00Z",
    }
    chunk_reads = iter([None, raced_chunk])

    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: session)
    monkeypatch.setattr(db, "_get_workspace_migration_chunk", lambda _migration_id, _chunk_id: next(chunk_reads))
    monkeypatch.setattr(
        db,
        "transaction",
        lambda: _BackendErrorTransaction("duplicate key value violates unique constraint"),
    )

    chunk = db.add_workspace_migration_chunk(
        "mig-1",
        "chunk-1",
        {
            "sha256": _CHUNK_HASH,
            "byte_count": 64,
            "chunk_kind": "workspace_bundle",
        },
    )

    assert chunk == raced_chunk


@pytest.mark.unit
def test_workspace_migration_chunk_postgres_backend_error_is_wrapped(db, monkeypatch):
    session = {
        "id": "mig-1",
        "manifest_hash": _MANIFEST_HASH,
        "status": "created",
        "declared_chunks_json": "[]",
        "_chunks": [],
    }

    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: session)
    monkeypatch.setattr(db, "_get_workspace_migration_chunk", lambda _migration_id, _chunk_id: None)
    monkeypatch.setattr(db, "transaction", lambda: _BackendErrorTransaction("connection reset"))

    with pytest.raises(CharactersRAGDBError, match="workspace migration chunk"):
        db.add_workspace_migration_chunk(
            "mig-1",
            "chunk-1",
            {
                "sha256": _CHUNK_HASH,
                "byte_count": 64,
                "chunk_kind": "workspace_bundle",
            },
        )


@pytest.mark.unit
def test_workspace_migration_finalize_and_delete_ack_wrap_backend_errors(db, monkeypatch):
    finalized_session = {
        "id": "mig-1",
        "target_workspace_id": "rw-import-1",
        "source_product": "research-workspace-webui",
        "manifest_hash": _MANIFEST_HASH,
        "status": "created",
        "declared_chunks_json": "[]",
        "_chunks": [],
    }
    ack_session = {
        **finalized_session,
        "status": "finalized",
        "client_delete_eligible": True,
    }

    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: finalized_session)
    monkeypatch.setattr(db, "transaction", lambda: _BackendErrorTransaction("connection reset"))

    with pytest.raises(CharactersRAGDBError, match="finalize workspace migration"):
        db.finalize_workspace_migration("mig-1", {"manifest_hash": _MANIFEST_HASH})

    monkeypatch.setattr(db, "get_workspace_migration_session", lambda _migration_id: ack_session)

    with pytest.raises(CharactersRAGDBError, match="client delete acknowledgement"):
        db.record_workspace_migration_client_delete_ack(
            "mig-1",
            {"acknowledged_manifest_hash": _MANIFEST_HASH},
        )
