from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import FileExportInfo
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

try:
    # Pydantic v2
    from pydantic import model_validator  # type: ignore
except ImportError:  # pragma: no cover - fallback for older environments
    model_validator = None  # type: ignore


ColumnType = Literal["text", "number", "date", "url", "boolean", "currency"]
SourceType = Literal["chat", "document", "rag_query"]
DataTableExportFormat = Literal["csv", "json", "xlsx"]
DataTableRowData = dict[str, Any] | list[Any]
DATA_TABLES_MAX_ROWS_LIMIT = int(os.getenv("DATA_TABLES_MAX_ROWS", "2000") or "2000")


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class DataTableColumnHint(BaseModel):
    """Optional schema hint supplied during generation."""

    name: str = Field(..., min_length=1)
    type: ColumnType | None = None
    description: str | None = None
    format: str | None = None


class DataTableSourceInput(BaseModel):
    """Source reference used to generate a table."""

    source_type: SourceType
    source_id: str = Field(..., min_length=1)
    title: str | None = None
    snapshot: Any | None = None
    retrieval_params: dict[str, Any] | None = None


class DataTableGenerateRequest(BaseModel):
    """Request payload for data table generation."""

    name: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    description: str | None = None
    workspace_tag: str | None = Field(default=None, description="Optional workspace tag (e.g., 'workspace:<slug-or-id>') to associate this table.")
    sources: list[DataTableSourceInput]
    column_hints: list[DataTableColumnHint] | None = None
    model: str | None = None
    max_rows: int | None = Field(default=None, ge=1, le=DATA_TABLES_MAX_ROWS_LIMIT)

    if model_validator is not None:
        @model_validator(mode="after")
        def _validate_payload(self) -> DataTableGenerateRequest:
            if not self.sources:
                raise ValueError("sources are required")
            return self
    else:
        from pydantic import root_validator as _rv  # type: ignore

        @_rv
        def _validate_payload(cls, values: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
            sources = values.get("sources") or []
            if not sources:
                raise ValueError("sources are required")
            return values


class DataTableRegenerateRequest(BaseModel):
    """Optional overrides for table regeneration."""

    prompt: str | None = None
    model: str | None = None
    max_rows: int | None = Field(default=None, ge=1, le=DATA_TABLES_MAX_ROWS_LIMIT)


class DataTableUpdateRequest(BaseModel):
    """Patchable metadata fields for a data table."""

    name: str | None = None
    description: str | None = None

    if model_validator is not None:
        @model_validator(mode="after")
        def _validate_payload(self) -> DataTableUpdateRequest:
            if self.name is None and self.description is None:
                raise ValueError("at least one field is required")
            if self.name is not None and not self.name.strip():
                raise ValueError("name cannot be blank")
            return self
    else:
        from pydantic import root_validator as _rv  # type: ignore

        @_rv
        def _validate_payload(cls, values: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
            if values.get("name") is None and values.get("description") is None:
                raise ValueError("at least one field is required")
            name = values.get("name")
            if name is not None and not str(name).strip():
                raise ValueError("name cannot be blank")
            return values


class DataTableColumn(BaseModel):
    """Column metadata returned for a data table."""

    column_id: str
    name: str
    type: ColumnType
    description: str | None = None
    format: str | None = None
    position: int


class DataTableColumnInput(BaseModel):
    """Column definition used when updating table content."""

    column_id: str | None = None
    name: str = Field(..., min_length=1)
    type: ColumnType
    description: str | None = None
    format: str | None = None
    position: int | None = None


class DataTableRow(BaseModel):
    """Row payload returned for a data table."""

    row_id: str
    row_index: int
    data: DataTableRowData
    row_hash: str | None = None


class DataTableSource(BaseModel):
    """Source metadata associated with a table."""

    source_type: SourceType
    source_id: str
    title: str | None = None
    snapshot: Any | None = None
    retrieval_params: Any | None = None


class DataTableSummary(BaseModel):
    """Summary metadata for a data table."""

    uuid: str
    name: str
    description: str | None = None
    workspace_tag: str | None = None
    prompt: str
    column_hints: Any | None = None
    status: str
    row_count: int
    column_count: int | None = None
    generation_model: str | None = None
    last_error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_modified: str | None = None
    version: int | None = None
    source_count: int | None = None


class DataTablesListResponse(BaseModel):
    """Paginated response containing data table summaries."""

    tables: list[DataTableSummary]
    count: int
    limit: int
    offset: int
    total: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    if model_validator is not None:
        @model_validator(mode="after")
        def _default_pagination_aliases(self) -> DataTablesListResponse:
            return _default_offset_pagination_aliases(self)


class DataTableDetailResponse(BaseModel):
    """Detailed response for a single data table."""

    table: DataTableSummary
    columns: list[DataTableColumn]
    rows: list[DataTableRow]
    sources: list[DataTableSource]
    rows_limit: int
    rows_offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    if model_validator is not None:
        @model_validator(mode="after")
        def _default_pagination_aliases(self) -> DataTableDetailResponse:
            return _default_offset_pagination_aliases(self)


class DataTableContentUpdateRequest(BaseModel):
    """Request payload for updating table columns and rows."""

    columns: list[DataTableColumnInput]
    rows: list[dict[str, Any]]


class DataTableGenerateResponse(BaseModel):
    """Response payload for a table generation job submission."""

    job_id: int
    job_uuid: str | None = None
    status: str
    table: DataTableSummary


class DataTableDeleteResponse(BaseModel):
    """Response payload for table deletion."""

    success: bool


class DataTableJobStatus(BaseModel):
    """Status payload for a data table job."""

    id: int
    uuid: str | None
    status: str
    job_type: str
    owner_user_id: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    cancelled_at: str | None
    cancellation_reason: str | None
    progress_percent: float | None
    progress_message: str | None
    result: dict[str, Any] | None
    error_message: str | None
    table_uuid: str | None = None


class DataTableJobCancelResponse(BaseModel):
    """Response payload for cancelling a data table job."""

    success: bool
    job_id: int
    status: str
    message: str | None = None


class DataTableExportResponse(BaseModel):
    """Response payload for a data table export request."""

    table_uuid: str
    file_id: int
    export: FileExportInfo
