"""Admit native Data Tables from an authenticated Research Workspace."""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from typing import Annotated, Any
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    require_expected_user_if_present,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.data_tables_deps import data_tables_jobs_queue, get_job_manager
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import (
    DATA_TABLES_MAX_ROWS_LIMIT,
    DataTableColumnHint,
    DataTableGenerateResponse,
)
from tldw_Server_API.app.api.v1.utils.data_tables_responses import table_summary_from_row
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ, MEDIA_UPDATE
from tldw_Server_API.app.core.Data_Tables.jobs_worker import (
    DATA_TABLES_DOMAIN,
    DATA_TABLES_JOB_TYPE,
    read_workspace_table_media,
    workspace_table_route,
)
from tldw_Server_API.app.core.Data_Tables.publication import (
    PROTOCOL_KEY,
    publish_completed_generation,
    read_generation_reservation,
    reservation_for_job,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.exceptions import DataTablesJobError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent
from tldw_Server_API.app.core.Research_Workspace.run_origin import Identifier, Revision, require_workspace

router = APIRouter(dependencies=[
    Depends(require_expected_user_if_present),
    Depends(RequirePermission(MEDIA_READ)),
    Depends(RequirePermission(MEDIA_CREATE)),
    Depends(RequirePermission(MEDIA_UPDATE)),
])
WorkspaceDb = Annotated[CharactersRAGDB, Depends(get_chacha_db_for_user)]
MediaDb = Annotated[Any, Depends(get_media_db_for_user)]
Jobs = Annotated[JobManager, Depends(get_job_manager)]
RequestUser = Annotated[User, Depends(get_request_user)]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
ModelId = Annotated[str, StringConstraints(max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")]
_MAX_SOURCES = 20
_CONTRACT = "research-workspace-data-tables.v1"


class WorkspaceDataTableRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    provider: Identifier
    model: ModelId


class WorkspaceTableSource(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    source_id: Identifier
    catalog_item_id: Revision
    association_version: Revision
    catalog_item_version: Revision


class WorkspaceTableColumnHint(DataTableColumnHint):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: Label
    description: str | None = Field(default=None, max_length=500)
    format: str | None = Field(default=None, max_length=80)


class WorkspaceDataTableRequest(WorkspaceDataTableRoute):
    operation_id: Identifier
    name: Label
    prompt: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    sources: tuple[WorkspaceTableSource, ...] = Field(min_length=1, max_length=_MAX_SOURCES)
    column_hints: tuple[WorkspaceTableColumnHint, ...] = Field(default=(), max_length=24)
    max_rows: int | None = Field(default=None, ge=1, le=DATA_TABLES_MAX_ROWS_LIMIT)

    @model_validator(mode="after")
    def distinct_sources(self):
        if len({source.source_id for source in self.sources}) != len(self.sources) or len(
            {source.catalog_item_id for source in self.sources}
        ) != len(self.sources):
            raise ValueError("Data Table generation requires distinct sources")
        return self


class WorkspaceDataTableCapability(WorkspaceDataTableRoute):
    contract: str = _CONTRACT
    available: bool
    reason: str | None = None
    max_sources: int = _MAX_SOURCES


class WorkspaceDataTableAdmission(DataTableGenerateResponse):
    job_uuid: str
    publication_status: str | None = None
    published_version: int | None = None


def _check_route(route: WorkspaceDataTableRoute, jm: JobManager, queue: str) -> None:
    if route.model.casefold() == "auto":
        raise DataTablesJobError("workspace_table_model_required", retryable=False)
    if not jm.should_enforce_leases():
        raise DataTablesJobError("workspace_table_jobs_unavailable", retryable=True)
    flags = jm.get_queue_flags(DATA_TABLES_DOMAIN, queue)
    if flags["paused"] or flags["drain"]:
        raise DataTablesJobError("workspace_table_jobs_unavailable", retryable=True)
    workspace_table_route(route.provider, route.model)


def _require_workspace(db: CharactersRAGDB, workspace_id: str, owner: str, *, conn=None) -> dict:
    try:
        return require_workspace(db, workspace_id, owner, conn=conn)
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace not found") from None


@router.get("/{workspace_id}/research-data-tables", dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)])
def get_workspace_data_table_capability(
    workspace_id: Identifier, route: Annotated[WorkspaceDataTableRoute, Query()],
    db: WorkspaceDb, user: RequestUser, jm: Jobs,
) -> WorkspaceDataTableCapability:
    """Check the native Data Tables route for this Workspace and model."""
    workspace = _require_workspace(db, workspace_id, str(user.id))
    reason = None
    try:
        if workspace.get("archived") or workspace.get("native_chat_admission_closed"):
            raise DataTablesJobError("workspace_closed", retryable=False)
        _check_route(route, jm, data_tables_jobs_queue())
    except DataTablesJobError as exc:
        reason = str(exc)
    return WorkspaceDataTableCapability(
        provider=route.provider, model=route.model, available=reason is None, reason=reason,
    )


def _selected_sources(
    db: CharactersRAGDB, media: Any, workspace_id: str, owner: str, sources: tuple[WorkspaceTableSource, ...],
) -> list[dict[str, Any]]:
    selected = []
    with db.transaction() as conn:
        workspace = _require_workspace(db, workspace_id, owner, conn=conn)
        if workspace.get("archived") or workspace.get("native_chat_admission_closed"):
            raise DataTablesJobError("workspace_closed", retryable=False)
        for source in sources:
            row = db.workspace_store.get_workspace_source(workspace_id, source.source_id)
            if (
                row is None or not row["selected"]
                or row["media_id"] != source.catalog_item_id or row["version"] != source.association_version
            ):
                raise DataTablesJobError("workspace_table_source_changed", retryable=False)
            _, fingerprint = read_workspace_table_media(
                media, source.catalog_item_id, source.catalog_item_version, owner
            )
            selected.append({
                "source_type": "document", "source_id": str(source.catalog_item_id),
                "title": str(row.get("title") or "")[:200], "media_version": source.catalog_item_version,
                "retrieval_params_json": {"workspace_source_sha256": fingerprint},
            })
    return selected


@router.post(
    "/{workspace_id}/research-data-tables", status_code=202, dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
)
def launch_workspace_data_table(
    workspace_id: Identifier, payload: WorkspaceDataTableRequest, request: Request,
    db: WorkspaceDb, user: RequestUser, media: MediaDb, jm: Jobs,
) -> WorkspaceDataTableAdmission:
    """Reserve one owned table, then submit or recover its exact native job."""
    owner = str(user.id)
    _require_workspace(db, workspace_id, owner)
    scope = json.dumps([owner, workspace_id, payload.operation_id])
    table_uuid = str(uuid5(NAMESPACE_URL, f"research-workspace-table.v1:{scope}"))
    digest = sha256(payload.model_dump_json().encode()).hexdigest()
    key = str(uuid5(NAMESPACE_URL, f"research-workspace-table-job.v1:{table_uuid}:{digest}"))
    try:
        return _admit_table(db, media, jm, request, workspace_id, owner, table_uuid, key, payload)
    except DataTablesJobError as exc:
        raise HTTPException(status_code=503 if exc.retryable else 409, detail=str(exc)) from None
    except (DatabaseError, OSError, RuntimeError, sqlite3.Error):
        logger.warning("Workspace Data Tables admission storage unavailable")
        raise HTTPException(status_code=503, detail="workspace_table_admission_unavailable") from None


def _read_reservation(raw: str | None) -> dict[str, str]:
    try:
        return read_generation_reservation(raw or "{}")
    except (ValueError, TypeError):
        raise DataTablesJobError("workspace_table_operation_conflict", retryable=False) from None


def _admit_table(
    db: CharactersRAGDB, media: Any, jm: JobManager, request: Request, workspace_id: str,
    owner: str, table_uuid: str, key: str, payload: WorkspaceDataTableRequest,
) -> WorkspaceDataTableAdmission:
    queue = data_tables_jobs_queue()
    table = media.get_data_table_by_uuid(table_uuid, include_deleted=True, owner_user_id=owner)
    stored = None
    if table:
        if table["deleted"]:
            raise DataTablesJobError("workspace_table_operation_deleted", retryable=False)
        stored = _read_reservation(table["generation_job_json"])
        if stored["key"] != key or stored["owner"] != owner:
            raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)
        queue = stored["queue"]
    reservation = {"key": key, "owner": owner, "queue": queue}
    job = jm.get_job_or_archived_by_idempotency_key(
        idempotency_key=key, domain=DATA_TABLES_DOMAIN, queue=queue,
        job_type=DATA_TABLES_JOB_TYPE, owner_user_id=owner,
    )
    if not job:
        if table and stored != reservation:
            raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)
        _check_route(payload, jm, queue)
        sources = _selected_sources(db, media, workspace_id, owner, payload.sources)
        hints = [hint.model_dump() for hint in payload.column_hints] or None
        if not table:
            with media.transaction():
                created = media.create_data_table(
                    table_uuid=table_uuid, if_absent=True, name=payload.name, prompt=payload.prompt,
                    workspace_tag=f"workspace:{workspace_id}", column_hints=hints,
                    generation_model=payload.model, generation_job=reservation, owner_user_id=owner,
                )
                if created and media.insert_data_table_sources(created["id"], sources, owner_user_id=owner) != len(sources):
                    raise DataTablesJobError("workspace_table_sources_unavailable", retryable=True)
                table = created or media.get_data_table_by_uuid(table_uuid, include_deleted=True, owner_user_id=owner)
        if not table or table["deleted"]:
            raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)
        current = _read_reservation(table["generation_job_json"])
        if any(current.get(field) != value for field, value in reservation.items()):
            raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)
        expected_evidence = [(source["source_id"], source["retrieval_params_json"]) for source in sources]
        try:
            reserved_evidence = [
                (row["source_id"], json.loads(row["retrieval_params_json"] or "{}"))
                for row in media.list_data_table_sources(table["id"], owner_user_id=owner)
            ]
        except (TypeError, ValueError):
            raise DataTablesJobError("workspace_table_source_changed", retryable=False) from None
        if reserved_evidence != expected_evidence:
            raise DataTablesJobError("workspace_table_source_changed", retryable=False)
        job = jm.create_job(
            domain=DATA_TABLES_DOMAIN, queue=queue, job_type=DATA_TABLES_JOB_TYPE,
            owner_user_id=owner, idempotency_key=key,
            request_id=ensure_request_id(request), trace_id=ensure_traceparent(request) or None,
            payload={
                PROTOCOL_KEY: 1, "generation_key": key, "table_id": table["id"], "table_uuid": table_uuid,
                "user_id": owner, "sources": sources, "prompt": payload.prompt, "model": payload.model,
                "workspace_provider": payload.provider, "column_hints": hints, "max_rows": payload.max_rows,
            },
        )

    job_payload = job.get("payload") or {}
    if isinstance(job_payload, str):
        job_payload = json.loads(job_payload)
    bound = reservation_for_job(job, job_payload)
    if (
        not table or job_payload.get("table_uuid") != table_uuid or job_payload.get("table_id") != table["id"]
        or any(bound.get(field) != value for field, value in reservation.items())
    ):
        raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)

    raw = table.get("generation_job_json")
    if _read_reservation(raw) == reservation:
        # The worker may have bound its exact lease first. That binding is also sufficient.
        media.update_data_table(table["id"], generation_job=bound, expected_generation_job=raw, owner_user_id=owner)
    publication_status = None
    if job["status"] == "completed":
        publication_status = publish_completed_generation(
            jm, job["id"], expected_uuid=job["uuid"], db=media if str(media.client_id) == owner else None,
        )
    table = media.get_data_table_by_uuid(table_uuid, owner_user_id=owner)
    if not table:
        raise DataTablesJobError("workspace_table_operation_deleted", retryable=False)
    current = _read_reservation(table["generation_job_json"])
    if any(bound.get(field) != value for field, value in current.items() if field != "attempt"):
        raise DataTablesJobError("workspace_table_operation_conflict", retryable=False)
    return WorkspaceDataTableAdmission(
        table=table_summary_from_row(table, source_count=len(payload.sources), column_count=0),
        job_id=job["id"], job_uuid=job["uuid"], status=job["status"], publication_status=publication_status,
        published_version=(job.get("result") or {}).get("table_version") if publication_status == "ready" else None,
    )
