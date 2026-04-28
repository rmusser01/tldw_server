"""Service layer for file artifacts validation, persistence, and exports."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import Any

import aiofiles
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.file_artifacts_schemas import (
    FileArtifact,
    FileCreateOptions,
    FileCreateRequest,
    FileExportInfo,
    FileExportRequest,
    FileValidationResult,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError, StorageError
from tldw_Server_API.app.core.config import get_config_value
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import FileArtifactsError, FileArtifactsValidationError
from tldw_Server_API.app.core.File_Artifacts.adapter_registry import get_registry
from tldw_Server_API.app.core.File_Artifacts.adapters.base import ExportResult, FileAdapter, ValidationIssue
from tldw_Server_API.app.core.File_Artifacts.adapters.image_adapter import (
    reset_image_adapter_request_context,
    set_image_adapter_request_context,
)
from tldw_Server_API.app.core.File_Artifacts.metrics import register_file_artifacts_metrics
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Storage.generated_file_helpers import (
    save_and_register_image,
    save_and_register_spreadsheet,
)

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ROWS = 5000
DEFAULT_MAX_CELLS = 200000
DEFAULT_ASYNC_ROWS = 2000
DEFAULT_ASYNC_CELLS = 100000
INLINE_MAX_BYTES = 256 * 1024
INLINE_MAX_BYTES_UPPER_BOUND = 10 * 1024 * 1024
DEFAULT_EXPORT_TTL_SECONDS = 900

EXPORT_MIME_TYPES = {
    "ics": "text/calendar",
    "md": "text/markdown",
    "html": "text/html",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


class FileArtifactsService:
    def __init__(
        self,
        cdb: CollectionsDatabase,
        *,
        user_id: int | str,
        job_manager: JobManager | None = None,
    ) -> None:
        self._cdb = cdb
        try:
            uid_int = int(user_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid user_id: must be integer or numeric string") from exc
        self._user_id_int = uid_int
        self._user_id = str(uid_int)
        self._registry = get_registry()
        self._jobs_manager: JobManager | None = job_manager
        register_file_artifacts_metrics()

    def get_adapter(self, file_type: str) -> FileAdapter | None:
        """Return the adapter for the requested file type, if available."""
        return self._registry.get_adapter(file_type)

    async def create_artifact(
        self,
        request: FileCreateRequest,
        *,
        request_id: str | None = None,
    ) -> tuple[FileArtifact, int]:
        """Create a file artifact and optionally enqueue an export."""
        adapter = self.get_adapter(request.file_type)
        if adapter is None:
            self._emit_metric("create", "failure", file_type=request.file_type, reason="unsupported_file_type")
            raise FileArtifactsError("unsupported_file_type")

        options = request.options
        if options.persist is not True:
            self._log_validation_failure(request_id, request.file_type, "persist_required")
            self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
            raise FileArtifactsError("persist_required")

        image_context_token = None
        if request.file_type == "image":
            image_context_token = set_image_adapter_request_context(collections_db=self._cdb, user_id=self._user_id)
        try:
            structured = adapter.normalize(request.payload)
        except FileArtifactsValidationError as exc:
            self._log_validation_failure(request_id, request.file_type, exc.code)
            self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
            raise
        except ValueError as exc:
            self._log_validation_failure(request_id, request.file_type, str(exc))
            self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
            raise FileArtifactsValidationError(str(exc)) from exc
        finally:
            if image_context_token is not None:
                reset_image_adapter_request_context(image_context_token)

        issues = adapter.validate(structured)
        errors, warnings = self._split_issues(issues)
        if errors:
            self._log_validation_failure(
                request_id,
                request.file_type,
                {"errors": [issue.code for issue in errors]},
            )
            self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
            raise FileArtifactsValidationError(
                "validation_errors",
                detail={"errors": [self._issue_to_payload(issue) for issue in errors]},
            )

        try:
            self._enforce_limits(request.file_type, structured, options)
        except FileArtifactsValidationError as exc:
            self._log_validation_failure(request_id, request.file_type, exc.code)
            self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
            raise

        if request.export is not None:
            try:
                self._validate_export_request(adapter=adapter, export_req=request.export)
            except FileArtifactsValidationError as exc:
                self._log_validation_failure(request_id, request.file_type, exc.code)
                self._emit_metric("create", "failure", file_type=request.file_type, reason="validation")
                raise

        validation = FileValidationResult(
            ok=True,
            warnings=[self._issue_to_payload(issue) for issue in warnings],
        )
        validation_json = json.dumps(validation.model_dump())
        structured_json = json.dumps(structured)
        title = request.title or self._default_title(request.file_type)
        retention_until = self._normalize_retention_until(options.retention_until)

        row = self._cdb.create_file_artifact(
            file_type=request.file_type,
            title=title,
            structured_json=structured_json,
            validation_json=validation_json,
            retention_until=retention_until,
        )

        export_info = FileExportInfo(status="none")
        status_code = HTTPStatus.OK

        if request.export is not None:
            try:
                export_info, status_code = await self._handle_export(
                    adapter=adapter,
                    structured=structured,
                    file_id=row.id,
                    export_req=request.export,
                    options=options,
                    request_id=request_id,
                )
            except FileArtifactsError as exc:
                self._rollback_artifact(row.id)
                self._log_export_failure(request_id, request.file_type, request.export.format, exc.detail or exc.code)
                self._emit_metric("create", "failure", file_type=request.file_type, reason="export")
                self._emit_metric(
                    "export",
                    "failure",
                    file_type=request.file_type,
                    export_format=request.export.format,
                )
                raise
            except Exception as exc:
                self._rollback_artifact(row.id)
                self._log_export_failure(request_id, request.file_type, request.export.format, str(exc))
                self._emit_metric("create", "failure", file_type=request.file_type, reason="export")
                self._emit_metric(
                    "export",
                    "failure",
                    file_type=request.file_type,
                    export_format=request.export.format,
                )
                raise
            row = self._cdb.get_file_artifact(row.id)

        artifact = self._build_artifact_from_row(row, export_override=export_info)
        self._emit_metric("create", "success", file_type=request.file_type)
        return artifact, status_code

    def get_artifact(self, file_id: int) -> FileArtifact:
        """Fetch a file artifact by id."""
        row = self._cdb.get_file_artifact(file_id)
        return self._build_artifact_from_row(row)

    async def export_artifact_for_job(
        self,
        *,
        adapter: FileAdapter,
        structured: dict[str, Any],
        file_id: int,
        export_format: str,
        options: FileCreateOptions,
    ) -> FileExportInfo:
        """Export and persist a file artifact for the jobs worker."""
        if export_format not in adapter.export_formats:
            raise FileArtifactsValidationError("unsupported_export_format")
        export_req = FileExportRequest(format=export_format, mode="url", async_mode="async")
        export_result = await self._export_sync(adapter, structured, export_format)
        if adapter.file_type == "image" and structured.get("reference_file_id") is not None:
            self._persist_structured_json(file_id, structured)
        return await self._finalize_export(file_id, export_req, export_result, options, file_type=adapter.file_type)

    async def _handle_export(
        self,
        *,
        adapter,
        structured: dict[str, Any],
        file_id: int,
        export_req: FileExportRequest,
        options: FileCreateOptions,
        request_id: str | None = None,
    ) -> tuple[FileExportInfo, int]:
        self._validate_export_request(adapter=adapter, export_req=export_req)
        async_mode = export_req.async_mode

        if async_mode == "async" or (
            async_mode == "auto"
            and self._should_export_async(adapter.file_type, structured, options, export_req.format)
        ):
            export_ttl = self._resolve_export_ttl_seconds(options)
            payload = {
                "file_id": file_id,
                "file_type": adapter.file_type,
                "export_format": export_req.format,
                "user_id": self._user_id,
                "max_bytes": options.max_bytes or DEFAULT_MAX_BYTES,
                "export_ttl_seconds": export_ttl,
            }
            queue = (os.getenv("FILES_JOBS_QUEUE") or "default").strip() or "default"
            try:
                job_row = self._get_jobs_manager().create_job(
                    domain="files",
                    queue=queue,
                    job_type="file_artifact_export",
                    payload=payload,
                    owner_user_id=str(self._user_id),
                    priority=5,
                    max_retries=3,
                    request_id=request_id,
                )
                job_id = str(job_row.get("id") or "")
                if not job_id:
                    raise ValueError("missing_job_id")
                self._cdb.update_file_artifact_export(
                    file_id,
                    export_status="pending",
                    export_format=export_req.format,
                    export_job_id=job_id,
                    export_expires_at=None,
                    export_consumed_at=None,
                )
            except Exception as exc:
                logger.error(
                    "file_artifacts: failed to enqueue export job error_type={}",
                    type(exc).__name__,
                )
                raise FileArtifactsError("export_job_enqueue_failed") from exc
            export_info = FileExportInfo(status="pending", format=export_req.format, job_id=job_id or None)
            self._emit_metric("export", "enqueued", file_type=adapter.file_type, export_format=export_req.format)
            return export_info, HTTPStatus.ACCEPTED

        export_result = await self._export_sync(adapter, structured, export_req.format)
        if adapter.file_type == "image" and structured.get("reference_file_id") is not None:
            self._persist_structured_json(file_id, structured)
        if adapter.file_type == "image" and export_result.content:
            inline_max_bytes = self._resolve_inline_max_bytes_for_file_type(adapter.file_type)
            if len(export_result.content) > inline_max_bytes:
                raise FileArtifactsValidationError("export_size_exceeded")
        export_info = await self._finalize_export(file_id, export_req, export_result, options, file_type=adapter.file_type)
        self._emit_metric("export", "success", file_type=adapter.file_type, export_format=export_req.format)
        return export_info, HTTPStatus.OK

    async def _export_sync(self, adapter, structured: dict[str, Any], export_format: str) -> ExportResult:
        image_context_token = None
        if getattr(adapter, "file_type", None) == "image":
            image_context_token = set_image_adapter_request_context(collections_db=self._cdb, user_id=self._user_id)
        try:
            if export_format == "xlsx" or getattr(adapter, "file_type", None) == "image":
                return await asyncio.to_thread(adapter.export, structured, format=export_format)
            return adapter.export(structured, format=export_format)
        finally:
            if image_context_token is not None:
                reset_image_adapter_request_context(image_context_token)

    def _persist_structured_json(self, file_id: int, structured: dict[str, Any]) -> None:
        """Persist updated structured metadata back onto a file artifact row."""
        updated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        q = (
            "UPDATE file_artifacts SET structured_json = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ? AND deleted = 0"
        )
        params = (json.dumps(structured), updated_at, file_id, self._cdb.user_id)
        try:
            res = self._cdb.backend.execute(q, params)
        except Exception as exc:
            raise FileArtifactsError("storage_persist_failed", detail=str(exc)) from exc
        if res.rowcount <= 0:
            raise FileArtifactsError("storage_persist_failed", detail="file_artifact_not_found")

    def _get_jobs_manager(self) -> JobManager:
        if self._jobs_manager is None:
            self._jobs_manager = jobs_manager_from_env()
        return self._jobs_manager

    async def _finalize_export(
        self,
        file_id: int,
        export_req: FileExportRequest,
        export_result: ExportResult,
        options: FileCreateOptions,
        *,
        file_type: str,
    ) -> FileExportInfo:
        if export_result.status != "ready" or not export_result.content:
            raise FileArtifactsError("export_failed")

        byte_count = len(export_result.content)
        max_bytes = options.max_bytes or DEFAULT_MAX_BYTES
        if byte_count > max_bytes:
            raise FileArtifactsValidationError("export_size_exceeded")

        content_type = export_result.content_type or EXPORT_MIME_TYPES.get(export_req.format)
        content_b64 = None
        inline_max_bytes = self._resolve_inline_max_bytes_for_file_type(file_type)
        inline_ready = export_req.mode == "inline" and inline_max_bytes > 0 and byte_count <= inline_max_bytes
        if inline_ready:
            content_b64 = base64.b64encode(export_result.content).decode("ascii")
        url = None
        expires_at = None
        storage_path: str | None = None
        export_updated = False
        try:
            if export_req.mode == "url" or not inline_ready:
                storage_path, _ = await self._write_export_file(file_id, export_req.format, export_result.content)
                ttl_seconds = self._resolve_export_ttl_seconds(options)
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                self._cdb.update_file_artifact_export(
                    file_id,
                    export_status="ready",
                    export_format=export_req.format,
                    export_storage_path=storage_path,
                    export_bytes=byte_count,
                    export_content_type=content_type,
                    export_job_id=export_result.job_id,
                    export_expires_at=expires_at.replace(microsecond=0).isoformat(),
                    export_consumed_at=None,
                )
                export_updated = True
                url = self._build_export_url(file_id, export_req.format)
            else:
                consumed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                self._cdb.update_file_artifact_export(
                    file_id,
                    export_status="none",
                    export_format=export_req.format,
                    export_storage_path=None,
                    export_bytes=byte_count,
                    export_content_type=content_type,
                    export_job_id=export_result.job_id,
                    export_expires_at=None,
                    export_consumed_at=consumed_at,
                )
                export_updated = True
            if export_req.mode == "url" or not inline_ready:
                # Register generated files for storage/quota tracking (image + spreadsheet exports).
                await self._register_generated_file_export(
                    file_id=file_id,
                    file_type=file_type,
                    export_format=export_req.format,
                    content=export_result.content,
                )
        except Exception:
            if storage_path:
                self._delete_temp_export_file(storage_path)
            if export_updated:
                try:
                    self._cdb.update_file_artifact_export(
                        file_id,
                        export_status="none",
                        export_format=export_req.format,
                        export_storage_path=None,
                        export_bytes=byte_count,
                        export_content_type=content_type,
                        export_job_id=export_result.job_id,
                        export_expires_at=None,
                        export_consumed_at=None,
                    )
                except Exception as reset_exc:
                    logger.warning(
                        "file_artifacts: failed to reset export state error_type={}",
                        type(reset_exc).__name__,
                    )
            raise
        return FileExportInfo(
            status="ready" if export_req.mode == "url" or not inline_ready else "none",
            format=export_req.format,
            url=url,
            content_type=content_type,
            bytes=byte_count,
            job_id=export_result.job_id,
            content_b64=content_b64,
            expires_at=expires_at,
        )

    async def _register_generated_file_export(
        self,
        *,
        file_id: int,
        file_type: str,
        export_format: str,
        content: bytes,
    ) -> None:
        """Persist export outputs into generated files storage for quota tracking."""
        try:
            if file_type == "image":
                await save_and_register_image(
                    user_id=self._user_id_int,
                    image_bytes=content,
                    image_format=export_format,
                    source_prompt=None,
                    model_name=None,
                    check_quota=True,
                )
            elif file_type == "data_table" and export_format in {"xlsx", "csv", "xls"}:
                await save_and_register_spreadsheet(
                    user_id=self._user_id_int,
                    spreadsheet_bytes=content,
                    spreadsheet_format=export_format,
                    original_filename=f"file_{file_id}.{export_format}",
                    source_ref=f"file_artifact:{file_id}",
                    check_quota=True,
                )
        except QuotaExceededError as exc:
            raise FileArtifactsError("storage_quota_exceeded", detail=str(exc)) from exc
        except StorageError as exc:
            raise FileArtifactsError("storage_persist_failed", detail=str(exc)) from exc

    async def _write_export_file(self, file_id: int, export_format: str, content: bytes) -> tuple[str, int]:
        filename = f"file_{file_id}.{export_format}"
        storage_path = self._cdb.resolve_temp_output_storage_path(filename)
        outputs_dir = DatabasePaths.get_user_temp_outputs_dir(self._user_id_int)
        file_path = outputs_dir / storage_path
        async with aiofiles.open(file_path, "wb") as handle:
            await handle.write(content)
        return storage_path, len(content)

    def _delete_temp_export_file(self, storage_path: str) -> None:
        try:
            safe_name = self._cdb.resolve_temp_output_storage_path(storage_path)
            outputs_dir = DatabasePaths.get_user_temp_outputs_dir(self._user_id_int)
            file_path = outputs_dir / safe_name
            if file_path.exists():
                file_path.unlink()
        except Exception as exc:
            logger.warning("file_artifacts: failed to delete export file error_type={}", type(exc).__name__)

    @staticmethod
    def _default_title(file_type: str) -> str:
        return f"{file_type}_artifact"

    @staticmethod
    def _build_export_url(file_id: int, export_format: str) -> str:
        return f"/api/v1/files/{file_id}/export?format={export_format}"

    @staticmethod
    def _issue_to_payload(issue: ValidationIssue) -> dict[str, Any]:
        return {
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
        }

    @staticmethod
    def _log_validation_failure(request_id: str | None, file_type: str, detail: Any) -> None:
        logger.warning(
            "file_artifacts.create validation failed file_type={} detail_type={}",
            file_type,
            type(detail).__name__,
        )

    @staticmethod
    def _log_export_failure(request_id: str | None, file_type: str, export_format: str, detail: Any) -> None:
        logger.warning(
            "file_artifacts.export failed file_type={} format={} detail_type={}",
            file_type,
            export_format,
            type(detail).__name__,
        )

    @staticmethod
    def _emit_metric(
        operation: str,
        status: str,
        *,
        file_type: str | None = None,
        export_format: str | None = None,
        reason: str | None = None,
    ) -> None:
        labels = {"operation": operation, "status": status}
        if file_type:
            labels["file_type"] = file_type
        if export_format:
            labels["export_format"] = export_format
        if reason:
            labels["reason"] = reason
        try:
            get_metrics_registry().increment("file_artifacts_operations_total", labels=labels)
        except Exception:
            logger.debug("metrics increment failed for file_artifacts_operations_total")

    def _rollback_artifact(self, file_id: int) -> None:
        try:
            row = self._cdb.get_file_artifact(file_id, include_deleted=True)
        except KeyError:
            return
        if getattr(row, "export_storage_path", None):
            self._delete_temp_export_file(row.export_storage_path)
        try:
            self._cdb.delete_file_artifact(file_id, hard=True)
        except Exception as exc:
            logger.warning("file_artifacts: failed to rollback file artifact error_type={}", type(exc).__name__)

    @staticmethod
    def _validate_export_request(*, adapter: FileAdapter, export_req: FileExportRequest) -> None:
        if export_req.format not in adapter.export_formats:
            raise FileArtifactsValidationError("unsupported_export_format")
        async_mode = export_req.async_mode
        if async_mode not in {"auto", "sync", "async"}:
            raise FileArtifactsValidationError("invalid_async_mode")
        if adapter.file_type == "image":
            if export_req.mode != "inline":
                raise FileArtifactsValidationError("invalid_export_mode")
            if async_mode != "sync":
                raise FileArtifactsValidationError("invalid_async_mode")

    @staticmethod
    def _split_issues(issues: list[ValidationIssue]) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        for issue in issues:
            if getattr(issue, "level", "error") == "warning":
                warnings.append(issue)
            else:
                errors.append(issue)
        return errors, warnings

    @staticmethod
    def _normalize_retention_until(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _resolve_export_ttl_seconds(options: FileCreateOptions) -> int:
        raw = options.export_ttl_seconds
        if raw is None:
            return DEFAULT_EXPORT_TTL_SECONDS
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_EXPORT_TTL_SECONDS
        return max(1, value)

    @staticmethod
    def _resolve_inline_max_bytes() -> int:
        raw = os.getenv("FILES_INLINE_MAX_BYTES")
        if raw is None:
            raw = get_config_value("Files", "inline_max_bytes")
        if raw is None:
            return INLINE_MAX_BYTES
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return INLINE_MAX_BYTES
        if value <= 0:
            logger.warning(
                "Invalid inline_max_bytes: {}. Using default {} bytes.",
                value,
                INLINE_MAX_BYTES,
            )
            return INLINE_MAX_BYTES
        if value > INLINE_MAX_BYTES_UPPER_BOUND:
            logger.warning(
                "inline_max_bytes capped at {} bytes (configured {}).",
                INLINE_MAX_BYTES_UPPER_BOUND,
                value,
            )
            return INLINE_MAX_BYTES_UPPER_BOUND
        return value

    @staticmethod
    def _resolve_inline_max_bytes_for_file_type(file_type: str) -> int:
        if file_type != "image":
            return FileArtifactsService._resolve_inline_max_bytes()
        raw = get_config_value("Image-Generation", "inline_max_bytes")
        if raw is None:
            return FileArtifactsService._resolve_inline_max_bytes()
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return FileArtifactsService._resolve_inline_max_bytes()
        if value <= 0:
            logger.warning(
                "Invalid image inline_max_bytes: {}. Using default {} bytes.",
                value,
                INLINE_MAX_BYTES,
            )
            return FileArtifactsService._resolve_inline_max_bytes()
        if value > INLINE_MAX_BYTES_UPPER_BOUND:
            logger.warning(
                "image inline_max_bytes capped at {} bytes (configured {}).",
                INLINE_MAX_BYTES_UPPER_BOUND,
                value,
            )
            return INLINE_MAX_BYTES_UPPER_BOUND
        return value

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _build_artifact_from_row(self, row, *, export_override: FileExportInfo | None = None) -> FileArtifact:
        structured = json.loads(row.structured_json)
        validation_data = json.loads(row.validation_json)
        validation = FileValidationResult.model_validate(validation_data)
        export = export_override or self._build_export_info_from_row(row)
        retention = None
        if getattr(row, "retention_until", None):
            try:
                retention = datetime.fromisoformat(row.retention_until)
            except Exception:
                retention = None
        file_type = row.file_type
        if file_type in {"csv_table", "json_table"}:
            file_type = "data_table"
        return FileArtifact(
            file_id=row.id,
            file_type=file_type,
            title=row.title,
            structured=structured,
            validation=validation,
            export=export,
            retention_until=retention,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _build_export_info_from_row(self, row) -> FileExportInfo:
        status_value = row.export_status or "none"
        if status_value == "ready":
            consumed_at = self._parse_iso_datetime(getattr(row, "export_consumed_at", None))
            if consumed_at is not None:
                return FileExportInfo(status="none")
            expires_at = self._parse_iso_datetime(getattr(row, "export_expires_at", None))
            if expires_at is not None and expires_at <= datetime.now(timezone.utc):
                return FileExportInfo(status="none")
            url = None
            if row.export_format and row.export_storage_path:
                url = self._build_export_url(row.id, row.export_format)
            return FileExportInfo(
                status="ready",
                format=row.export_format,
                url=url,
                content_type=row.export_content_type,
                bytes=row.export_bytes,
                job_id=row.export_job_id,
                expires_at=expires_at,
            )
        if status_value == "pending":
            return FileExportInfo(status="pending", format=row.export_format, job_id=row.export_job_id)
        return FileExportInfo(status="none")

    def _enforce_limits(self, file_type: str, structured: dict[str, Any], options: FileCreateOptions) -> None:
        max_rows = options.max_rows or DEFAULT_MAX_ROWS
        max_cells = options.max_cells or DEFAULT_MAX_CELLS
        rows, cells = self._extract_table_shape(file_type, structured)
        if rows > max_rows:
            raise FileArtifactsValidationError("row_limit_exceeded")
        if cells > max_cells:
            raise FileArtifactsValidationError("cell_limit_exceeded")

    def _should_export_async(
        self,
        file_type: str,
        structured: dict[str, Any],
        options: FileCreateOptions,
        export_format: str,
    ) -> bool:
        hard_rows = options.max_rows or DEFAULT_MAX_ROWS
        hard_cells = options.max_cells or DEFAULT_MAX_CELLS
        max_rows = min(hard_rows, DEFAULT_ASYNC_ROWS)
        max_cells = min(hard_cells, DEFAULT_ASYNC_CELLS)
        rows, cells = self._extract_table_shape(file_type, structured)
        if rows > max_rows or cells > max_cells:
            return True
        estimated = self._estimate_export_size(file_type, structured, export_format)
        if estimated is None:
            return False
        max_bytes = options.max_bytes or DEFAULT_MAX_BYTES
        return estimated > max_bytes

    def _estimate_export_size(
        self,
        _file_type: str,
        structured: dict[str, Any],
        export_format: str,
    ) -> int | None:
        if export_format == "ics":
            return self._estimate_ical_bytes(structured)
        if export_format == "xlsx":
            return self._estimate_xlsx_bytes(structured)
        if export_format in {"csv", "md", "html", "json"}:
            return self._estimate_table_bytes(structured, export_format)
        return None

    def _estimate_table_bytes(self, structured: dict[str, Any], export_format: str) -> int:
        tables = list(self._iter_tables(structured))
        total_bytes = 0
        if export_format == "csv":
            for columns, rows in tables:
                total_bytes += self._estimate_csv_bytes(columns, rows)
            return total_bytes
        if export_format == "md":
            for columns, rows in tables:
                total_bytes += self._estimate_markdown_bytes(columns, rows)
            return total_bytes
        if export_format == "html":
            for columns, rows in tables:
                total_bytes += self._estimate_html_bytes(columns, rows)
            return total_bytes
        if export_format == "json":
            for columns, rows in tables:
                total_bytes += self._estimate_json_bytes(columns, rows)
            return total_bytes
        return total_bytes

    def _estimate_csv_bytes(self, columns: list[Any], rows: list[Any]) -> int:
        total_chars = 0
        quote_escape = 0
        formula_prefix = 0
        cell_count = 0
        rows_all = [columns, *rows]
        for row in rows_all:
            if not isinstance(row, list):
                continue
            for cell in row:
                text = self._stringify_cell(cell)
                total_chars += len(text)
                quote_escape += text.count('"')
                if text.lstrip().startswith(("=", "+", "-", "@")):
                    formula_prefix += 1
                cell_count += 1
        sep_count = 0
        if columns:
            sep_count = (len(rows_all)) * max(len(columns) - 1, 0)
        newline_count = len(rows_all)
        quote_wrap = cell_count * 2
        return total_chars + quote_escape + formula_prefix + sep_count + newline_count + quote_wrap

    def _estimate_markdown_bytes(self, columns: list[Any], rows: list[Any]) -> int:
        total_chars = 0
        escape_extra = 0
        row_overhead = 0
        rows_all = [columns, *rows]
        for row in rows_all:
            if not isinstance(row, list):
                continue
            col_count = len(row)
            if col_count:
                row_overhead += 4 + 3 * (col_count - 1)
            for cell in row:
                text = self._stringify_cell(cell)
                total_chars += len(text)
                escape_extra += text.count("|")
        sep_row_overhead = 3 * len(columns) + 4 + 3 * (len(columns) - 1) if columns else 0
        newline_count = len(rows_all) + (1 if columns else 0)
        return total_chars + escape_extra + row_overhead + sep_row_overhead + newline_count

    def _estimate_html_bytes(self, columns: list[Any], rows: list[Any]) -> int:
        total_chars = 0
        escape_extra = 0
        cell_count = 0
        row_count = 0
        rows_all = [columns, *rows]
        for row in rows_all:
            if not isinstance(row, list):
                continue
            row_count += 1
            for cell in row:
                text = self._stringify_cell(cell)
                total_chars += len(text)
                escape_extra += self._count_html_escape_chars(text)
                cell_count += 1
        cell_wrap = cell_count * len("<td></td>")
        header_wrap = len("<table><thead><tr></tr></thead><tbody></tbody></table>")
        row_wrap = row_count * len("<tr></tr>")
        return total_chars + escape_extra + cell_wrap + row_wrap + header_wrap

    def _estimate_json_bytes(self, columns: list[Any], rows: list[Any]) -> int:
        total = len("[]")
        if not isinstance(columns, list):
            columns = []
        key_lengths = [len(self._stringify_cell(c)) for c in columns]
        for row_idx, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            total += len("{}")
            field_count = min(len(columns), len(row))
            for idx in range(field_count):
                key_len = key_lengths[idx]
                value_text = self._stringify_cell(row[idx])
                total += key_len + 3  # quotes around key + colon
                total += len(value_text) + 2  # quoted value
                total += value_text.count('"') + value_text.count("\\")
            if field_count > 1:
                total += field_count - 1
            if row_idx < len(rows) - 1:
                total += 1
        return total

    def _estimate_xlsx_bytes(self, structured: dict[str, Any]) -> int:
        tables = list(self._iter_tables(structured))
        total_chars = 0
        cell_count = 0
        for columns, rows in tables:
            rows_all = [columns, *rows]
            for row in rows_all:
                if not isinstance(row, list):
                    continue
                for cell in row:
                    text = self._stringify_cell(cell)
                    total_chars += len(text)
                    cell_count += 1
        sheet_count = max(len(tables), 1)
        base_overhead = 2048 + sheet_count * 512
        return base_overhead + (cell_count * 12) + (total_chars * 2)

    def _estimate_ical_bytes(self, structured: dict[str, Any]) -> int:
        calendar = structured.get("calendar") or {}
        prodid = self._stringify_cell(calendar.get("prodid") or "-//tldw//files//EN")
        version = self._stringify_cell(calendar.get("version") or "2.0")
        tz = calendar.get("timezone")
        total = len("BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
        total += len("PRODID:") + len(prodid) + len("\r\n")
        total += len("VERSION:") + len(version) + len("\r\n")
        if tz:
            total += len("X-WR-TIMEZONE:") + len(self._stringify_cell(tz)) + len("\r\n")
        events = calendar.get("events") or []
        for event in events:
            if not isinstance(event, dict):
                continue
            total += len("BEGIN:VEVENT\r\nEND:VEVENT\r\n")
            for key in ("uid", "summary", "start", "end", "description", "location"):
                value = event.get(key)
                if value is None:
                    continue
                label = key.upper() if key not in {"start", "end"} else ("DTSTART" if key == "start" else "DTEND")
                total += len(label) + 1 + len(self._stringify_cell(value)) + len("\r\n")
        return total

    @staticmethod
    def _stringify_cell(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _count_html_escape_chars(text: str) -> int:
        extras = {
            "&": 4,   # & -> &amp;
            "<": 3,   # < -> &lt;
            ">": 3,   # > -> &gt;
            "\"": 5,  # " -> &quot;
            "'": 4,   # ' -> &#x27;
        }
        return sum(text.count(ch) * extra for ch, extra in extras.items())

    @staticmethod
    def _iter_tables(structured: dict[str, Any]):
        if "sheets" in structured:
            for sheet in structured.get("sheets") or []:
                yield sheet.get("columns") or [], sheet.get("rows") or []
            return
        yield structured.get("columns") or [], structured.get("rows") or []

    @staticmethod
    def _extract_table_shape(file_type: str, structured: dict[str, Any]) -> tuple[int, int]:
        if "columns" in structured and "rows" in structured:
            rows = structured.get("rows") or []
            columns = structured.get("columns") or []
            return len(rows), len(rows) * len(columns)
        if "sheets" in structured:
            total_rows = 0
            total_cells = 0
            for sheet in structured.get("sheets") or []:
                rows = sheet.get("rows") or []
                cols = sheet.get("columns") or []
                total_rows += len(rows)
                total_cells += len(rows) * len(cols)
            return total_rows, total_cells
        if file_type == "ical":
            events = (structured.get("calendar") or {}).get("events") or []
            return len(events), len(events)
        return 0, 0
