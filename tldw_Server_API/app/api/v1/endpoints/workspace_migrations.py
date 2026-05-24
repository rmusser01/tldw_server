"""Research Workspace migration protocol endpoints."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    StatusResponse,
    WorkspaceMigrationChunkReceiptResponse,
    WorkspaceMigrationChunkUploadRequest,
    WorkspaceMigrationClientDeleteAckRequest,
    WorkspaceMigrationCreateRequest,
    WorkspaceMigrationFinalizeRequest,
    WorkspaceMigrationResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

router = APIRouter()


def _map_chacha_error_to_http(exc: Exception, *, default_detail: str) -> HTTPException:
    if isinstance(exc, ConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, InputError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    logger.error("{}: {}", default_detail, exc)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=default_detail,
    )


def _loads_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _chunk_to_response(chunk: dict[str, Any]) -> WorkspaceMigrationChunkReceiptResponse:
    return WorkspaceMigrationChunkReceiptResponse(
        id=str(chunk["id"]),
        migration_id=str(chunk["migration_id"]),
        sha256=str(chunk["sha256"]),
        byte_count=int(chunk["byte_count"]),
        chunk_kind=str(chunk.get("chunk_kind") or "workspace_bundle"),
        metadata=_loads_json(chunk.get("metadata_json"), {}),
        accepted_at=str(chunk.get("accepted_at", "")),
    )


def _declared_chunks(row: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _loads_json(row.get("declared_chunks_json"), [])
    return parsed if isinstance(parsed, list) else []


def _accepted_chunks(row: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = row.get("_chunks") or []
    return chunks if isinstance(chunks, list) else []


def _missing_chunk_ids(row: dict[str, Any]) -> list[str]:
    accepted_ids = {str(chunk.get("id")) for chunk in _accepted_chunks(row)}
    return [
        str(chunk.get("id"))
        for chunk in _declared_chunks(row)
        if str(chunk.get("id")) not in accepted_ids
    ]


def _default_recovery_manifest(
    row: dict[str, Any],
    *,
    missing_chunk_ids: list[str],
    accepted_chunk_count: int,
) -> dict[str, Any]:
    declared_chunk_count = len(_declared_chunks(row))
    return {
        "migration_id": row["id"],
        "target_workspace_id": row["target_workspace_id"],
        "source_product": row["source_product"],
        "manifest_hash": row["manifest_hash"],
        "status": row.get("status") or "created",
        "declared_chunk_count": declared_chunk_count,
        "accepted_chunk_count": accepted_chunk_count,
        "missing_chunk_ids": missing_chunk_ids,
        "client_delete_eligible": False,
        "can_delete_legacy_storage": False,
    }


def _migration_to_response(row: dict[str, Any]) -> WorkspaceMigrationResponse:
    chunk_rows = _accepted_chunks(row)
    chunks = [_chunk_to_response(chunk) for chunk in chunk_rows]
    missing = _missing_chunk_ids(row)
    recovery = _loads_json(row.get("recovery_manifest_json"), None)
    if not isinstance(recovery, dict):
        recovery = _default_recovery_manifest(
            row,
            missing_chunk_ids=missing,
            accepted_chunk_count=len(chunks),
        )

    return WorkspaceMigrationResponse(
        id=str(row["id"]),
        idempotency_key=str(row["idempotency_key"]),
        target_workspace_id=str(row["target_workspace_id"]),
        target_workspace_name=str(row["target_workspace_name"]),
        source_product=str(row["source_product"]),
        manifest_hash=str(row["manifest_hash"]),
        status=str(row.get("status") or "created"),
        declared_chunk_count=len(_declared_chunks(row)),
        accepted_chunk_count=len(chunks),
        missing_chunk_ids=missing,
        client_delete_eligible=bool(row.get("client_delete_eligible", False)),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
        finalized_at=str(row["finalized_at"]) if row.get("finalized_at") else None,
        recovery_manifest=recovery,
        chunks=chunks,
    )


def _require_migration(db: CharactersRAGDB, migration_id: str) -> dict[str, Any]:
    row = db.get_workspace_migration_session(migration_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Workspace migration not found")
    return row


@router.post(
    "/migrations",
    response_model=WorkspaceMigrationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Create a Research Workspace migration session",
)
async def create_workspace_migration(
    body: WorkspaceMigrationCreateRequest,
    response: Response,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Create or idempotently return a Research Workspace migration session."""
    _ = current_user
    try:
        row, created = db.upsert_workspace_migration_session(body.model_dump(mode="json"))
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to create workspace migration",
        ) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    return _migration_to_response(row)


@router.get(
    "/migrations",
    response_model=list[WorkspaceMigrationResponse],
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List Research Workspace migration sessions",
)
async def list_workspace_migrations(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """List recent durable Research Workspace migration sessions."""
    _ = current_user
    try:
        rows = db.list_workspace_migration_sessions(limit=100)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to list workspace migrations",
        ) from exc
    return [_migration_to_response(row) for row in rows]


@router.get(
    "/migrations/{migration_id}",
    response_model=WorkspaceMigrationResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get a Research Workspace migration session",
)
async def get_workspace_migration(
    migration_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Return a durable migration session and recovery manifest."""
    _ = current_user
    try:
        row = _require_migration(db, migration_id)
    except HTTPException:
        raise
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to fetch workspace migration",
        ) from exc
    return _migration_to_response(row)


@router.put(
    "/migrations/{migration_id}/chunks/{chunk_id}",
    response_model=WorkspaceMigrationChunkReceiptResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Accept a Research Workspace migration chunk receipt",
)
async def put_workspace_migration_chunk(
    migration_id: str,
    chunk_id: str,
    body: WorkspaceMigrationChunkUploadRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Record an idempotent chunk receipt for a migration session."""
    _ = current_user
    try:
        chunk = db.add_workspace_migration_chunk(
            migration_id,
            chunk_id,
            body.model_dump(mode="json"),
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to accept workspace migration chunk",
        ) from exc
    return _chunk_to_response(chunk)


@router.post(
    "/migrations/{migration_id}/finalize",
    response_model=WorkspaceMigrationResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Finalize a Research Workspace migration session",
)
async def finalize_workspace_migration(
    migration_id: str,
    body: WorkspaceMigrationFinalizeRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Finalize a migration only after all declared chunks have receipts."""
    _ = current_user
    try:
        row = _require_migration(db, migration_id)
        if row["manifest_hash"] != body.manifest_hash:
            raise HTTPException(status_code=409, detail="Migration manifest hash mismatch")
        missing = _missing_chunk_ids(row)
        if missing:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Migration cannot be finalized until all declared chunks are accepted.",
                    "missing_chunk_ids": missing,
                },
            )
        finalized = db.finalize_workspace_migration(
            migration_id,
            {"manifest_hash": body.manifest_hash},
        )
    except HTTPException:
        raise
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to finalize workspace migration",
        ) from exc
    return _migration_to_response(finalized)


@router.post(
    "/migrations/{migration_id}/client-delete-ack",
    response_model=StatusResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Acknowledge local legacy deletion for a migration",
)
async def acknowledge_workspace_migration_client_delete(
    migration_id: str,
    body: WorkspaceMigrationClientDeleteAckRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Reject client deletion acknowledgement until deletion eligibility exists."""
    _ = current_user
    try:
        db.record_workspace_migration_client_delete_ack(
            migration_id,
            {"acknowledged_manifest_hash": body.acknowledged_manifest_hash},
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise _map_chacha_error_to_http(
            exc,
            default_detail="Failed to record workspace migration client delete acknowledgement",
        ) from exc
    return StatusResponse(ok=True)
