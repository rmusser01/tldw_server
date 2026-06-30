"""Data table endpoints and helper utilities for job-driven exports."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Union
from uuid import UUID

from cachetools import LRUCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal, get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import (
    DataTableColumn,
    DataTableContentUpdateRequest,
    DataTableDeleteResponse,
    DataTableDetailResponse,
    DataTableExportFormat,
    DataTableExportResponse,
    DataTableGenerateRequest,
    DataTableGenerateResponse,
    DataTableJobCancelResponse,
    DataTableJobStatus,
    DataTableRegenerateRequest,
    DataTableRow,
    DataTablesListResponse,
    DataTableSource,
    DataTableSummary,
    DataTableUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import (
    AsyncMode,
    ExportMode,
    FileCreateOptions,
    FileCreateRequest,
    FileExportRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    MEDIA_CREATE,
    MEDIA_DELETE,
    MEDIA_READ,
    MEDIA_UPDATE,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.exceptions import (
    FileArtifactsError,
    FileArtifactsValidationError,
    file_artifacts_http_status,
)
from tldw_Server_API.app.core.File_Artifacts.adapters.data_table_adapter import DataTableAdapter
from tldw_Server_API.app.core.File_Artifacts.file_artifacts_service import (
    DEFAULT_MAX_BYTES,
    FileArtifactsService,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

router = APIRouter(prefix="/data-tables", tags=["data-tables"])

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
DATA_TABLE_GENERATION_FAILED_DETAIL = "Data table generation failed"
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _data_tables_jobs_queue() -> str:
    """Return the configured jobs queue for data tables workers."""
    return (os.getenv("DATA_TABLES_JOBS_QUEUE") or "default").strip() or "default"


def _file_artifacts_http_exception(exc: FileArtifactsError) -> HTTPException:
    """Translate file artifact errors into HTTP exceptions with status codes."""
    detail = exc.detail if exc.detail is not None else exc.code
    status_code = file_artifacts_http_status(exc)
    return HTTPException(status_code=status_code, detail=detail)


def _mark_data_table_generate_failed(
    db: Any,
    *,
    table_id: int | None,
    owner_user_id: int | str | None,
    exc: Exception,
) -> None:
    """Best-effort failure marker for partially created data tables."""
    if table_id is None:
        return
    try:
        db.update_data_table(
            table_id,
            status="failed",
            last_error=DATA_TABLE_GENERATION_FAILED_DETAIL,
            owner_user_id=owner_user_id,
        )
    except Exception:
        logger.debug("data_tables.generate: failed to mark table as failed")


def get_job_manager() -> JobManager:
    """Return a cached JobManager instance keyed by JOBS_DB_URL or JOBS_DB_PATH."""
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = f"url:{db_url}" if db_url else f"path:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if db_url:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)
        elif db_path:
            job_manager = JobManager(db_path=Path(db_path))
        else:
            job_manager = JobManager()

        _job_manager_cache[cache_key] = job_manager
        return job_manager


def _parse_json_value(raw: str | None) -> Any:
    """Parse a JSON string when possible, otherwise return the original value."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _model_dump(obj: Any) -> dict[str, Any]:
    """Return a dict representation of a Pydantic model (v1/v2 compatible)."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    dump = getattr(obj, "dict", None)
    if callable(dump):
        return dump()
    return dict(obj)


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _resolve_owner_id(principal: AuthPrincipal, current_user: User) -> int | str | None:
    """Resolve the owner id for data table queries based on auth context."""
    if _principal_has_admin_claims(principal):
        return None
    owner_id = getattr(current_user, "id", None)
    if owner_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="owner_user_id_missing",
        )
    if isinstance(owner_id, int):
        return owner_id
    if isinstance(owner_id, UUID):
        return str(owner_id)
    if isinstance(owner_id, str):
        owner_id = owner_id.strip()
        if not owner_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="owner_user_id_empty",
            )
        try:
            return int(owner_id)
        except ValueError:
            try:
                return str(UUID(owner_id))
            except ValueError:
                return owner_id
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="owner_user_id_invalid",
    )


def _table_summary_from_row(
    row: dict[str, Any],
    *,
    column_count: int | None = None,
    source_count: int | None = None,
) -> DataTableSummary:
    """Build a DataTableSummary response from a database row."""
    return DataTableSummary(
        uuid=str(row.get("uuid") or ""),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        workspace_tag=row.get("workspace_tag"),
        prompt=str(row.get("prompt") or ""),
        column_hints=_parse_json_value(row.get("column_hints_json")),
        status=str(row.get("status") or ""),
        row_count=int(row.get("row_count") or 0),
        column_count=column_count
        if column_count is not None
        else row.get("column_count"),
        generation_model=row.get("generation_model"),
        last_error=row.get("last_error"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_modified=row.get("last_modified"),
        version=int(row.get("version")) if row.get("version") is not None else None,
        source_count=source_count
        if source_count is not None
        else row.get("source_count"),
    )


def _column_from_row(row: dict[str, Any]) -> DataTableColumn:
    """Build a DataTableColumn from a data table column row."""
    return DataTableColumn(
        column_id=str(row.get("column_id") or ""),
        name=str(row.get("name") or ""),
        type=str(row.get("type") or ""),
        description=row.get("description"),
        format=row.get("format"),
        position=int(row.get("position") or 0),
    )


def _row_from_row(row: dict[str, Any]) -> DataTableRow:
    """Build a DataTableRow from a data table row record."""
    return DataTableRow(
        row_id=str(row.get("row_id") or ""),
        row_index=int(row.get("row_index") or 0),
        data=_parse_json_value(row.get("row_json")),
        row_hash=row.get("row_hash"),
    )


def _source_from_row(row: dict[str, Any]) -> DataTableSource:
    """Build a DataTableSource from a data table source row."""
    return DataTableSource(
        source_type=str(row.get("source_type") or ""),
        source_id=str(row.get("source_id") or ""),
        title=row.get("title"),
        snapshot=_parse_json_value(row.get("snapshot_json")),
        retrieval_params=_parse_json_value(row.get("retrieval_params_json")),
    )


def _row_values_from_json(
    row_json: Any,
    *,
    column_ids: list[str],
    column_names: list[str],
) -> list[Any]:
    """Return row values ordered to match column ids or names."""
    if row_json is None:
        return [None] * len(column_ids)
    if isinstance(row_json, dict):
        values = []
        for col_id, col_name in zip(column_ids, column_names, strict=True):
            if col_id in row_json:
                values.append(row_json.get(col_id))
            elif col_name in row_json:
                values.append(row_json.get(col_name))
            else:
                values.append(None)
        return values
    if isinstance(row_json, (list, tuple)):
        row_values = list(row_json)
        if len(row_values) < len(column_ids):
            row_values.extend([None] * (len(column_ids) - len(row_values)))
        elif len(row_values) > len(column_ids):
            row_values = row_values[: len(column_ids)]
        return row_values
    return [row_json] + [None] * max(0, len(column_ids) - 1)


def _collect_export_rows(
    db: Any,
    table_id: int,
    *,
    column_ids: list[str],
    column_names: list[str],
    owner_user_id: int | str | None = None,
) -> list[list[Any]]:
    """Collect all table rows in batches for export."""
    rows: list[list[Any]] = []
    offset = 0
    batch_size = 2000
    while True:
        batch = db.list_data_table_rows(
            table_id,
            limit=batch_size,
            offset=offset,
            owner_user_id=owner_user_id,
        )
        if not batch:
            break
        for row in batch:
            row_json = _parse_json_value(row.get("row_json"))
            rows.append(
                _row_values_from_json(
                    row_json,
                    column_ids=column_ids,
                    column_names=column_names,
                )
            )
        offset += len(batch)
        if len(batch) < batch_size:
            break
    return rows


def _build_export_filename(title: str, export_format: str) -> str:
    """Build a safe export filename using the requested format."""
    base = sanitize_filename(
        title or "data_table",
        max_total_length=80,
        extension=f".{export_format}",
    )
    if not base:
        base = "data_table"
    return f"{base}.{export_format}"


async def _export_structured(adapter, structured: dict[str, Any], export_format: str):
    """Export structured payload, offloading XLSX work to a thread."""
    if export_format == "xlsx":
        return await asyncio.to_thread(adapter.export, structured, format=export_format)
    return adapter.export(structured, format=export_format)


async def _wait_for_job_completion(
    jm: JobManager,
    job_id: int,
    *,
    timeout_seconds: int = 300,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """Poll job status until completion or timeout."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = jm.get_job(int(job_id))
        if not job:
            raise HTTPException(status_code=404, detail="job_not_found")
        status_val = str(job.get("status") or "").lower()
        if status_val in {"completed", "failed", "cancelled", "quarantined"}:
            return job
        await asyncio.sleep(poll_interval)
    raise HTTPException(status_code=408, detail="data_table_job_timeout")


def _build_table_detail_response(
    table_row: dict[str, Any],
    db: Any,
    *,
    rows_limit: int = 200,
    rows_offset: int = 0,
    include_rows: bool = True,
    include_sources: bool = True,
    owner_user_id: int | str | None = None,
) -> DataTableDetailResponse:
    """Assemble a detail response including columns, rows, and sources."""
    table_id = int(table_row.get("id"))
    columns = [
        _column_from_row(row)
        for row in db.list_data_table_columns(table_id, owner_user_id=owner_user_id)
    ]
    rows: list[DataTableRow] = []
    if include_rows:
        rows = [
            _row_from_row(row)
            for row in db.list_data_table_rows(
                table_id,
                limit=rows_limit,
                offset=rows_offset,
                owner_user_id=owner_user_id,
            )
        ]
    sources: list[DataTableSource] = []
    source_count: int | None = table_row.get("source_count")
    if include_sources:
        sources = [
            _source_from_row(row)
            for row in db.list_data_table_sources(table_id, owner_user_id=owner_user_id)
        ]
        source_count = len(sources)
    total_rows = int(table_row.get("row_count") or 0)
    rows_has_more = rows_offset + rows_limit < total_rows
    return DataTableDetailResponse(
        table=_table_summary_from_row(
            table_row,
            column_count=len(columns),
            source_count=source_count,
        ),
        columns=columns,
        rows=rows,
        sources=sources,
        rows_limit=rows_limit,
        rows_offset=rows_offset,
        pagination=build_offset_pagination_meta(
            limit=rows_limit,
            offset=rows_offset,
            total=total_rows,
            count=len(rows),
            has_more=rows_has_more,
        ),
    )


@router.post(
    "/generate",
    response_model=Union[DataTableGenerateResponse, DataTableDetailResponse],
    summary="Submit a data table generation job",
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("data_tables.generate")),
    ],
)
async def generate_data_table(
    req: DataTableGenerateRequest,
    response: Response,
    request: Request,
    wait_for_completion: bool = Query(False, description="Wait for job completion"),
    wait_timeout_seconds: int = Query(300, ge=1, le=1800),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
    jm: JobManager = Depends(get_job_manager),
) -> DataTableGenerateResponse | DataTableDetailResponse:
    """Queue a data table generation job and optionally wait for completion."""
    rid = ensure_request_id(request)
    tp = ensure_traceparent(request)
    table_id = None
    table_uuid = None
    owner_user_id = _resolve_owner_id(principal, current_user)

    try:
        column_hints = None
        if req.column_hints:
            column_hints = [_model_dump(hint) for hint in req.column_hints]

        table_row = db.create_data_table(
            name=req.name.strip(),
            prompt=req.prompt.strip(),
            description=req.description,
            workspace_tag=req.workspace_tag,
            column_hints=column_hints,
            status="queued",
            row_count=0,
            generation_model=req.model,
            owner_user_id=owner_user_id,
        )
        table_id = int(table_row.get("id"))
        table_uuid = str(table_row.get("uuid"))
        table_owner_client_id = str(table_row.get("client_id") or "").strip() or None

        sources_db_payload: list[dict[str, Any]] = []
        job_sources: list[dict[str, Any]] = []
        for source in req.sources:
            src = _model_dump(source)
            job_sources.append(
                {
                    "source_type": src.get("source_type"),
                    "source_id": src.get("source_id"),
                    "title": src.get("title"),
                    "snapshot": src.get("snapshot"),
                    "retrieval_params": src.get("retrieval_params"),
                }
            )
            sources_db_payload.append(
                {
                    "source_type": src.get("source_type"),
                    "source_id": src.get("source_id"),
                    "title": src.get("title"),
                    "snapshot_json": src.get("snapshot"),
                    "retrieval_params_json": src.get("retrieval_params"),
                }
            )
        if sources_db_payload:
            db.insert_data_table_sources(table_id, sources_db_payload, owner_user_id=owner_user_id)

        payload: dict[str, Any] = {
            "table_id": table_id,
            "table_uuid": table_uuid,
            "prompt": req.prompt,
            "sources": job_sources,
            "column_hints": column_hints,
            "model": req.model,
            "max_rows": req.max_rows,
        }
        if table_owner_client_id:
            payload["user_id"] = table_owner_client_id

        job = jm.create_job(
            domain="data_tables",
            queue=_data_tables_jobs_queue(),
            job_type="data_table_generate",
            payload=payload,
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=3,
            request_id=rid,
            trace_id=tp or None,
        )

        if wait_for_completion:
            job_state = await _wait_for_job_completion(jm, int(job.get("id")), timeout_seconds=wait_timeout_seconds)
            job_status = str(job_state.get("status") or "").lower()
            if job_status != "completed":
                raise HTTPException(
                    status_code=409,
                    detail=job_state.get("error_message") or "data_table_job_failed",
                )
            table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
            if not table_row:
                raise HTTPException(status_code=404, detail="data_table_not_found")
            response.status_code = status.HTTP_200_OK
            rows_limit = min(req.max_rows or 2000, 2000)
            return _build_table_detail_response(
                table_row,
                db,
                rows_limit=rows_limit,
                rows_offset=0,
                include_rows=True,
                include_sources=True,
                owner_user_id=owner_user_id,
            )

        response.status_code = status.HTTP_202_ACCEPTED
        return DataTableGenerateResponse(
            job_id=int(job.get("id")),
            job_uuid=job.get("uuid"),
            status=str(job.get("status") or "queued"),
            table=_table_summary_from_row(
                table_row,
                column_count=0,
                source_count=len(sources_db_payload),
            ),
        )
    except InputError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to submit data table job",
        ) from exc
    except HTTPException:
        raise
    except DatabaseError as exc:
        logger.error("data_tables.generate failed")
        _mark_data_table_generate_failed(
            db,
            table_id=table_id,
            owner_user_id=owner_user_id,
            exc=exc,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to submit data table job",
        ) from exc
    except Exception as exc:
        logger.error("data_tables.generate failed")
        _mark_data_table_generate_failed(
            db,
            table_id=table_id,
            owner_user_id=owner_user_id,
            exc=exc,
        )
        raise HTTPException(status_code=500, detail="Failed to submit data table job") from exc


@router.get(
    "",
    response_model=DataTablesListResponse,
    summary="List data tables",
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("data_tables.list")),
    ],
)
async def list_data_tables(
    status_filter: str | None = Query(None, description="Filter by status"),
    search: str | None = Query(None, description="Search by name/description"),
    workspace_tag: str | None = Query(None, description="Filter by workspace tag"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> DataTablesListResponse:
    """List data tables with optional filters and pagination."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        rows = db.list_data_tables(
            status=status_filter,
            search=search,
            workspace_tag=workspace_tag,
            limit=limit,
            offset=offset,
            owner_user_id=owner_user_id,
        )
        total = db.count_data_tables(
            status=status_filter,
            search=search,
            workspace_tag=workspace_tag,
            owner_user_id=owner_user_id,
        )
        table_ids = []
        for row in rows:
            try:
                table_ids.append(int(row.get("id")))
            except (TypeError, ValueError) as exc:
                logger.warning(
                    'data_tables.list: invalid table id row_id={} row={} error={}',
                    row.get("id"),
                    row,
                    exc,
                )
                continue
        counts_map = db.get_data_table_counts(table_ids, owner_user_id=owner_user_id)
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to list data tables",
        ) from exc
    tables = []
    for row in rows:
        table_id = None
        try:
            table_id = int(row.get("id"))
        except (TypeError, ValueError) as exc:
            logger.warning(
                'data_tables.list: invalid table id for summary row_id={} row={} error={}',
                row.get("id"),
                row,
                exc,
            )
            table_id = None
        counts = counts_map.get(table_id or -1, {})
        tables.append(
            _table_summary_from_row(
                row,
                column_count=counts.get("column_count"),
                source_count=counts.get("source_count"),
            )
        )
    return DataTablesListResponse(
        tables=tables,
        count=len(tables),
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=total,
            count=len(tables),
        ),
    )


@router.get(
    "/{table_uuid}",
    response_model=DataTableDetailResponse,
    summary="Get a data table by UUID",
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(check_rate_limit)],
)
async def get_data_table(
    table_uuid: str,
    rows_limit: int = Query(200, ge=1, le=2000),
    rows_offset: int = Query(0, ge=0),
    include_rows: bool = Query(True),
    include_sources: bool = Query(True),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> DataTableDetailResponse:
    """Return a single data table with optional rows and sources."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
        if not table_row:
            raise HTTPException(status_code=404, detail="data_table_not_found")

        return _build_table_detail_response(
            table_row,
            db,
            rows_limit=rows_limit,
            rows_offset=rows_offset,
            include_rows=include_rows,
            include_sources=include_sources,
            owner_user_id=owner_user_id,
        )
    except HTTPException:
        raise
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to fetch data table",
        ) from exc


@router.get(
    "/{table_uuid}/export",
    response_model=DataTableExportResponse,
    responses={
        200: {
            "description": "Data table export metadata or direct download content.",
            "content": {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {},
                "text/csv": {},
            },
        },
    },
    summary="Export a data table",
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("data_tables.export")),
    ],
)
async def export_data_table(
    table_uuid: str,
    response: Response,
    request: Request,
    format: DataTableExportFormat = Query(..., description="Export format (csv|json|xlsx)"),
    async_mode: AsyncMode = Query("auto", description="auto defers large exports; async forces 202"),
    mode: ExportMode = Query("url", description="url returns a download link; inline returns base64 content"),
    download: bool = Query(False, description="Return file content directly"),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
    cdb: CollectionsDatabase = Depends(get_collections_db_for_user),
) -> DataTableExportResponse:
    """Export a data table via file artifacts or direct response."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
        if not table_row:
            raise HTTPException(status_code=404, detail="data_table_not_found")
        if str(table_row.get("status") or "") != "ready":
            raise HTTPException(status_code=409, detail="data_table_not_ready")

        table_id = int(table_row.get("id"))
        column_rows = db.list_data_table_columns(table_id, owner_user_id=owner_user_id)
        if not column_rows:
            raise HTTPException(status_code=409, detail="data_table_missing_columns")

        column_rows = sorted(
            column_rows,
            key=lambda row: (int(row.get("position") or 0), int(row.get("id") or 0)),
        )
        column_ids = [str(row.get("column_id") or "") for row in column_rows]
        column_names = [str(row.get("name") or "") for row in column_rows]
        rows = _collect_export_rows(
            db,
            table_id,
            column_ids=column_ids,
            column_names=column_names,
            owner_user_id=owner_user_id,
        )
    except HTTPException:
        raise
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to export data table",
        ) from exc

    structured = {"columns": column_names, "rows": rows}

    if download:
        adapter = DataTableAdapter()
        try:
            normalized = adapter.normalize(structured)
            export_result = await _export_structured(adapter, normalized, format)
        except FileArtifactsValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileArtifactsError as exc:
            raise _file_artifacts_http_exception(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if export_result.status != "ready" or not export_result.content:
            raise HTTPException(status_code=500, detail="export_failed")
        byte_count = len(export_result.content)
        max_bytes = DEFAULT_MAX_BYTES
        if byte_count > max_bytes:
            raise HTTPException(status_code=422, detail="export_size_exceeded")
        filename = _build_export_filename(str(table_row.get("name") or "data_table"), format)
        return Response(
            content=export_result.content,
            media_type=export_result.content_type or "application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
        )

    options_payload: dict[str, Any] = {"persist": True}
    if rows:
        options_payload["max_rows"] = len(rows)
        options_payload["max_cells"] = len(rows) * len(column_names)
    options = FileCreateOptions(**options_payload)
    export_req = FileExportRequest(format=format, mode=mode, async_mode=async_mode)
    file_req = FileCreateRequest(
        file_type="data_table",
        title=str(table_row.get("name") or "data_table"),
        payload=structured,
        export=export_req,
        options=options,
    )
    request_id = ensure_request_id(request)
    service = FileArtifactsService(cdb, user_id=current_user.id)
    should_async = async_mode == "async" or (
        async_mode == "auto"
        and service._should_export_async("data_table", structured, options, format)
    )
    if should_async:
        try:
            artifact, status_code = await service.create_artifact(file_req, request_id=request_id)
        except FileArtifactsError as exc:
            raise _file_artifacts_http_exception(exc) from exc
        response.status_code = status_code
        return DataTableExportResponse(
            table_uuid=table_uuid,
            file_id=artifact.file_id,
            export=artifact.export,
        )

    try:
        artifact, status_code = await service.create_artifact(file_req, request_id=request_id)
    except FileArtifactsError as exc:
        raise _file_artifacts_http_exception(exc) from exc
    response.status_code = status_code
    return DataTableExportResponse(
        table_uuid=table_uuid,
        file_id=artifact.file_id,
        export=artifact.export,
    )


@router.put(
    "/{table_uuid}/content",
    response_model=DataTableDetailResponse,
    summary="Update data table content",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("data_tables.update_content")),
    ],
)
async def update_data_table_content(
    table_uuid: str,
    req: DataTableContentUpdateRequest,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> DataTableDetailResponse:
    """Update data table columns and rows content."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
    if not table_row:
        raise HTTPException(status_code=404, detail="data_table_not_found")

    table_id = int(table_row.get("id"))
    table_owner_client_id = str(table_row.get("client_id") or "").strip() or None
    mutation_owner_user_id = table_owner_client_id or owner_user_id

    try:
        if not req.columns:
            raise InputError("columns_required")

        columns_payload: list[dict[str, Any]] = []
        seen_names = set()
        seen_ids = set()
        for idx, col in enumerate(req.columns):
            name = (col.name or "").strip()
            if not name:
                raise InputError("column_name_required")
            name_key = name.lower()
            if name_key in seen_names:
                raise InputError("duplicate_column_name")
            source_column_id = (col.column_id or "").strip()
            if source_column_id:
                if source_column_id in seen_ids:
                    raise InputError("duplicate_column_id")
                seen_ids.add(source_column_id)
            # Always generate new IDs to avoid unique constraint collisions on soft-deleted columns.
            column_id = str(uuid.uuid4())
            seen_names.add(name_key)
            columns_payload.append(
                {
                    "column_id": column_id,
                    "source_column_id": source_column_id or None,
                    "name": name,
                    "type": col.type,
                    "description": col.description,
                    "format": col.format,
                    "position": idx,
                }
            )

        rows_payload: list[dict[str, Any]] = []
        seen_row_indexes: set[int] = set()
        for idx, row in enumerate(req.rows or []):
            if not isinstance(row, dict):
                raise InputError("row_payload_invalid")
            row_payload = row.get("row_json") if "row_json" in row else row.get("data", row)
            if isinstance(row_payload, str):
                row_payload = _parse_json_value(row_payload)
            if row_payload is None:
                row_payload = {}
            if not isinstance(row_payload, dict):
                raise InputError("row_payload_invalid")
            row_json: dict[str, Any] = {}
            for column in columns_payload:
                col_id = column["column_id"]
                col_name = column["name"]
                source_id = column.get("source_column_id")
                if source_id and source_id in row_payload:
                    value = row_payload.get(source_id)
                elif col_id in row_payload:
                    value = row_payload.get(col_id)
                elif col_name in row_payload:
                    value = row_payload.get(col_name)
                else:
                    value = None
                row_json[col_id] = value
            row_id = row.get("row_id") if isinstance(row.get("row_id"), str) else None
            row_index = row.get("row_index", idx)
            try:
                row_index = int(row_index) if row_index is not None else idx
            except (TypeError, ValueError):
                row_index = idx
            if row_index in seen_row_indexes:
                raise InputError("duplicate_row_index")
            seen_row_indexes.add(row_index)
            rows_payload.append(
                {
                    "row_id": row_id or str(uuid.uuid4()),
                    "row_index": row_index,
                    "row_json": row_json,
                }
            )
        with db.transaction():
            db.persist_data_table_generation(
                table_id,
                columns=columns_payload,
                rows=rows_payload,
                status="ready",
                row_count=len(rows_payload),
                last_error=None,
                owner_user_id=mutation_owner_user_id,
            )
    except (InputError, DatabaseError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to update data table content",
        ) from exc

    try:
        updated_row = db.get_data_table(table_id, owner_user_id=mutation_owner_user_id) or table_row
        rows_limit = max(1, min(len(rows_payload) or 200, 2000))
        return _build_table_detail_response(
            updated_row,
            db,
            rows_limit=rows_limit,
            rows_offset=0,
            include_rows=True,
            include_sources=True,
            owner_user_id=owner_user_id,
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to update data table content",
        ) from exc


@router.patch(
    "/{table_uuid}",
    response_model=DataTableSummary,
    summary="Update data table metadata",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("data_tables.update")),
    ],
)
async def update_data_table(
    table_uuid: str,
    req: DataTableUpdateRequest,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> DataTableSummary:
    """Update data table metadata such as name and description."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
        if not table_row:
            raise HTTPException(status_code=404, detail="data_table_not_found")
        table_owner_client_id = str(table_row.get("client_id") or "").strip() or None
        mutation_owner_user_id = table_owner_client_id or owner_user_id
        updated = db.update_data_table(
            int(table_row.get("id")),
            name=req.name.strip() if req.name is not None else None,
            description=req.description,
            owner_user_id=mutation_owner_user_id,
        )
        if not updated:
            raise HTTPException(status_code=500, detail="data_table_update_failed")
        counts = db.get_data_table_counts([int(updated.get("id"))], owner_user_id=mutation_owner_user_id).get(
            int(updated.get("id")),
            {},
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update data table") from exc
    return _table_summary_from_row(
        updated,
        column_count=counts.get("column_count"),
        source_count=counts.get("source_count"),
    )


@router.delete(
    "/{table_uuid}",
    response_model=DataTableDeleteResponse,
    summary="Delete a data table",
    dependencies=[
        Depends(RequirePermission(MEDIA_DELETE)),
        Depends(rbac_rate_limit("data_tables.delete")),
    ],
)
async def delete_data_table(
    table_uuid: str,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
) -> DataTableDeleteResponse:
    """Delete a data table."""
    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
        if not table_row:
            raise HTTPException(status_code=404, detail="data_table_not_found")
        table_owner_client_id = str(table_row.get("client_id") or "").strip() or None
        mutation_owner_user_id = table_owner_client_id or owner_user_id
        deleted = db.soft_delete_data_table(
            int(table_row.get("id")),
            owner_user_id=mutation_owner_user_id,
        )
        if not deleted:
            raise HTTPException(status_code=500, detail="data_table_delete_failed")
    except DatabaseError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete data table") from exc
    return DataTableDeleteResponse(success=True)


@router.post(
    "/{table_uuid}/regenerate",
    response_model=Union[DataTableGenerateResponse, DataTableDetailResponse],
    summary="Regenerate a data table from stored sources",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("data_tables.regenerate")),
    ],
)
async def regenerate_data_table(
    table_uuid: str,
    req: DataTableRegenerateRequest,
    response: Response,
    request: Request,
    wait_for_completion: bool = Query(False, description="Wait for job completion"),
    wait_timeout_seconds: int = Query(300, ge=1, le=1800),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_media_db_for_user),
    jm: JobManager = Depends(get_job_manager),
) -> DataTableGenerateResponse | DataTableDetailResponse:
    """Queue a data table regeneration job."""
    rid = ensure_request_id(request)
    tp = ensure_traceparent(request)

    owner_user_id = _resolve_owner_id(principal, current_user)
    try:
        table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
        if not table_row:
            raise HTTPException(status_code=404, detail="data_table_not_found")

        table_id = int(table_row.get("id"))
        table_owner_client_id = str(table_row.get("client_id") or "").strip() or None
        sources_rows = db.list_data_table_sources(table_id, owner_user_id=owner_user_id)
        job_sources = [
            {
                "source_type": row.get("source_type"),
                "source_id": row.get("source_id"),
                "title": row.get("title"),
                "snapshot": _parse_json_value(row.get("snapshot_json")),
                "retrieval_params": _parse_json_value(row.get("retrieval_params_json")),
            }
            for row in sources_rows
        ]
        if not job_sources:
            raise HTTPException(status_code=400, detail="data_table_missing_sources")
    except HTTPException:
        raise
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to regenerate data table",
        ) from exc

    prompt_override = req.prompt.strip() if req.prompt is not None else None
    if prompt_override == "":
        prompt_override = None

    payload: dict[str, Any] = {
        "table_id": table_id,
        "table_uuid": table_uuid,
        "prompt": prompt_override or table_row.get("prompt"),
        "sources": job_sources,
        "column_hints": _parse_json_value(table_row.get("column_hints_json")),
        "model": req.model or table_row.get("generation_model"),
        "max_rows": req.max_rows,
        "regenerate": True,
    }
    if table_owner_client_id:
        payload["user_id"] = table_owner_client_id

    job = jm.create_job(
        domain="data_tables",
        queue=_data_tables_jobs_queue(),
        job_type="data_table_generate",
        payload=payload,
        owner_user_id=str(current_user.id),
        priority=5,
        max_retries=3,
        request_id=rid,
        trace_id=tp or None,
    )
    try:
        db.update_data_table(
            table_id,
            status="queued",
            generation_model=req.model or table_row.get("generation_model"),
            prompt=prompt_override,
            owner_user_id=table_owner_client_id or owner_user_id,
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to regenerate data table",
        ) from exc

    if wait_for_completion:
        job_state = await _wait_for_job_completion(
            jm, int(job.get("id")), timeout_seconds=wait_timeout_seconds
        )
        job_status = str(job_state.get("status") or "").lower()
        if job_status != "completed":
            raise HTTPException(
                status_code=409,
                detail=job_state.get("error_message") or "data_table_job_failed",
            )
        try:
            table_row = db.get_data_table_by_uuid(table_uuid, owner_user_id=owner_user_id)
            if not table_row:
                raise HTTPException(status_code=404, detail="data_table_not_found")
            response.status_code = status.HTTP_200_OK
            rows_limit = min(req.max_rows or 2000, 2000)
            return _build_table_detail_response(
                table_row,
                db,
                rows_limit=rows_limit,
                rows_offset=0,
                include_rows=True,
                include_sources=True,
                owner_user_id=owner_user_id,
            )
        except HTTPException:
            raise
        except DatabaseError as exc:
            raise map_db_error_to_http(
                exc,
                default_detail="Failed to regenerate data table",
            ) from exc

    try:
        response.status_code = status.HTTP_202_ACCEPTED
        counts = db.get_data_table_counts([table_id], owner_user_id=owner_user_id).get(table_id, {})
        return DataTableGenerateResponse(
            job_id=int(job.get("id")),
            job_uuid=job.get("uuid"),
            status=str(job.get("status") or "queued"),
            table=_table_summary_from_row(
                db.get_data_table(table_id, owner_user_id=owner_user_id) or table_row,
                column_count=counts.get("column_count"),
                source_count=counts.get("source_count"),
            ),
        )
    except DatabaseError as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to regenerate data table",
        ) from exc


@router.get(
    "/jobs/{job_id}",
    response_model=DataTableJobStatus,
    summary="Get data table job status",
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("data_tables.jobs.get")),
    ],
)
async def get_data_table_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> DataTableJobStatus:
    """Fetch data table job status."""
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "data_tables":
        raise HTTPException(status_code=404, detail="job_not_found")
    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="not_authorized")
    payload = job.get("payload") or {}
    return DataTableJobStatus(
        id=int(job.get("id")),
        uuid=job.get("uuid"),
        status=job.get("status"),
        job_type=job.get("job_type"),
        owner_user_id=job.get("owner_user_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        cancelled_at=job.get("cancelled_at"),
        cancellation_reason=job.get("cancellation_reason"),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        result=job.get("result"),
        error_message=job.get("error_message"),
        table_uuid=payload.get("table_uuid"),
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=DataTableJobCancelResponse,
    summary="Cancel a data table job",
    dependencies=[
        Depends(RequirePermission(MEDIA_UPDATE)),
        Depends(rbac_rate_limit("data_tables.jobs.cancel")),
    ],
)
async def cancel_data_table_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
    reason: str | None = None,
) -> DataTableJobCancelResponse:
    """Cancel a queued data table job."""
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "data_tables":
        raise HTTPException(status_code=404, detail="job_not_found")
    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="not_authorized")
    status_val = str(job.get("status") or "").lower()
    if status_val in {"completed", "failed", "cancelled", "quarantined"}:
        raise HTTPException(status_code=400, detail="cannot_cancel_terminal_job")
    ok = jm.cancel_job(int(job_id), reason=reason)
    if not ok:
        raise HTTPException(status_code=400, detail="cancellation_failed")
    return DataTableJobCancelResponse(
        success=True,
        job_id=int(job_id),
        status="cancelled",
        message="Job cancellation requested",
    )


__all__ = ["router"]
