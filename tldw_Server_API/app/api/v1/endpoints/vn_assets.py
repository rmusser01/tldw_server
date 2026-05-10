"""VN asset pack metadata endpoints."""

from __future__ import annotations

import inspect
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetBulkReviewRequest,
    VNAssetCleanupRequest,
    VNAssetCleanupResponse,
    VNAssetGenerationRequest,
    VNAssetGenerationStatusResponse,
    VNAssetItemResponse,
    VNAssetManifestResponse,
    VNAssetPackCreate,
    VNAssetPackResponse,
    VNAssetPackUpdate,
    VNAssetPromptPreviewRequest,
    VNAssetPromptPreviewResponse,
    VNAssetReadinessResponse,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
    VNAssetSlotResponse,
    VNAssetSlotUpdate,
    VNAssetStarterMatricesResponse,
    VNPackExportRequest,
    VNPackExportResponse,
    VNPackImportCommitRequest,
    VNPackImportCommitStartResponse,
    VNPackImportJobResponse,
    VNPackImportPreviewResponse,
    VNPackImportPreviewStartResponse,
    VNPackPortabilityJobResponse,
)
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.jobs import (
    create_pack_export_job,
    create_pack_import_commit_job,
    create_pack_import_preview_job,
)
from tldw_Server_API.app.core.VN_Assets.constants import DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
from tldw_Server_API.app.core.VN_Assets.matrix import expand_starter_matrix
from tldw_Server_API.app.core.VN_Assets.portability.archive import DEFAULT_MAX_ARCHIVE_SIZE_BYTES
from tldw_Server_API.app.core.VN_Assets.portability.constants import VNPACK_EXTENSION
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.storage import (
    VN_ASSET_CONTENT_NOT_FOUND,
    generated_file_matches_vn_asset,
    image_format_from_mime_type,
    resolve_vn_asset_storage_path,
)
from tldw_Server_API.app.core.VN_Platform.errors import (
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_POLICY_BLOCKED,
    vn_error_detail,
)
from tldw_Server_API.app.core.VN_Platform.idempotency import (
    canonical_multipart_payload_hash,
    canonical_payload_hash,
)
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

router = APIRouter(prefix="/vn-assets", tags=["vn-assets"])
CONFLICT_ERROR_CODES = {
    "slot_already_exists",
    "slot_has_dependents",
}
TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "quarantined"}
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


class VNAssetMatrixApplyRequest(BaseModel):
    """Request body for applying a starter matrix to a pack."""

    matrix_key: str = "starter"
    overrides: dict[str, Any] = Field(default_factory=dict)


def _job_manager() -> JobManager:
    return JobManager()


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetPackService:
    owner_user_id = current_user.id_int
    if owner_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_user_id",
        )
    return VNAssetPackService(db, owner_user_id=owner_user_id, jobs_manager=jobs_manager)


async def _generated_files_repo() -> AuthnzGeneratedFilesRepo:
    storage_service = await get_storage_service()
    return await storage_service.get_generated_files_repo()


async def _storage_service() -> Any:
    return await get_storage_service()


def _cleanup_blocker_provider() -> Any | None:
    return None


def _handle_value_error(exc: ValueError) -> HTTPException:
    detail = str(exc) or "invalid_request"
    if "not_found" in detail:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    if detail in CONFLICT_ERROR_CODES:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _idempotency_conflict(
    *,
    scope: str,
    resource_id: str,
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=vn_error_detail(
            ERROR_IDEMPOTENCY_KEY_CONFLICT,
            "Idempotency key was already used with a different payload.",
            details={"scope": scope, "resource_id": resource_id},
        ),
    )


def _idempotency_replay(
    service: VNAssetPackService,
    *,
    owner_user_id: int,
    scope: str,
    resource_id: str,
    idempotency_key: str | None,
    payload_hash: str,
    response_model: type[BaseModel],
) -> BaseModel | None:
    if not idempotency_key:
        return None
    record = service.repo.get_idempotency_record(
        owner_user_id=owner_user_id,
        scope=scope,
        resource_id=resource_id,
        idempotency_key=idempotency_key,
    )
    if record is None:
        return None
    if str(record["payload_hash"]) != payload_hash:
        raise _idempotency_conflict(scope=scope, resource_id=resource_id)
    return response_model.model_validate(json.loads(str(record["response_json"])))


def _record_idempotency_response(
    service: VNAssetPackService,
    *,
    owner_user_id: int,
    scope: str,
    resource_id: str,
    idempotency_key: str | None,
    payload_hash: str,
    response: BaseModel,
) -> None:
    if not idempotency_key:
        return
    service.repo.create_idempotency_record(
        owner_user_id=owner_user_id,
        scope=scope,
        resource_id=resource_id,
        idempotency_key=idempotency_key,
        payload_hash=payload_hash,
        response=response.model_dump(mode="json"),
    )


def _current_user_id(current_user: User) -> int:
    user_id = current_user.id_int
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_user_id",
        )
    return user_id


def _get_vn_asset_upload_max_bytes() -> int:
    raw_value = os.getenv("VN_ASSET_UPLOAD_MAX_BYTES")
    if raw_value is None:
        return DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
    try:
        max_bytes = int(raw_value)
    except ValueError:
        return DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
    return max_bytes if max_bytes > 0 else DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES


def _json_field(row: dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _job_for_portability(jobs_manager: JobManager, job_id: str) -> dict[str, Any] | None:
    try:
        return jobs_manager.get_job(int(job_id))
    except (TypeError, ValueError):
        return None


def _portability_export_or_404(
    service: VNAssetPackService,
    *,
    job_id: str,
    owner_user_id: int,
) -> dict[str, Any]:
    row = service.repo.get_portability_job_by_job_id(str(job_id), owner_user_id=owner_user_id)
    if row is None or row.get("operation") != "export":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export_job_not_found")
    return row


def _portability_import_preview_or_404(
    service: VNAssetPackService,
    *,
    preview_id: int,
    owner_user_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preview = service.repo.get_import_preview(preview_id, owner_user_id=owner_user_id)
    if preview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_preview_not_found")
    portability_job = service.repo.get_portability_job_by_job_id(
        str(preview["job_id"]),
        owner_user_id=owner_user_id,
    )
    if portability_job is None or portability_job.get("operation") != "import_preview":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_preview_job_not_found")
    return preview, portability_job


def _portability_import_commit_or_404(
    service: VNAssetPackService,
    *,
    job_id: str,
    owner_user_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    portability_job = service.repo.get_portability_job_by_job_id(
        str(job_id),
        owner_user_id=owner_user_id,
    )
    if portability_job is None or portability_job.get("operation") != "import_commit":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_commit_job_not_found")
    import_id = portability_job.get("import_id")
    if import_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_journal_not_found")
    journal = service.repo.get_import_journal(int(import_id), owner_user_id=owner_user_id)
    if journal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_journal_not_found")
    return journal, portability_job


def _compose_portability_response(
    service: VNAssetPackService,
    *,
    portability_job: dict[str, Any],
    job: dict[str, Any] | None,
    owner_user_id: int,
) -> VNPackPortabilityJobResponse:
    job_status = str((job or {}).get("status") or portability_job["status"])
    vn_status = str(portability_job["status"])
    if job_status in TERMINAL_JOB_STATUSES and vn_status not in TERMINAL_JOB_STATUSES:
        updated = service.repo.update_portability_job(
            str(portability_job["job_id"]),
            {"status": job_status},
            owner_user_id=owner_user_id,
        )
        if updated is not None:
            portability_job = updated
            vn_status = str(portability_job["status"])

    return VNPackPortabilityJobResponse(
        job_id=str(portability_job["job_id"]),
        portability_job_id=int(portability_job["id"]),
        operation=str(portability_job["operation"]),
        pack_id=portability_job.get("pack_id"),
        status=job_status,
        vn_status=vn_status,
        stage=str(portability_job["stage"]),
        progress=_json_field(portability_job, "progress_json", {}),
        warnings=_json_field(portability_job, "warnings_json", []),
        archive_sha256=portability_job.get("archive_sha256"),
        canonical_payload_fingerprint=portability_job.get("canonical_payload_fingerprint"),
        download_url=portability_job.get("download_url"),
        error_code=portability_job.get("error_code"),
        error_message=portability_job.get("error_message"),
        expires_at=portability_job.get("expires_at"),
    )


def _compose_import_preview_response(
    service: VNAssetPackService,
    *,
    preview: dict[str, Any],
    portability_job: dict[str, Any],
    job: dict[str, Any] | None,
    owner_user_id: int,
) -> VNPackImportPreviewResponse:
    job_status = str((job or {}).get("status") or portability_job["status"])
    preview_status = str(preview["status"])
    if job_status in TERMINAL_JOB_STATUSES:
        if str(portability_job["status"]) not in TERMINAL_JOB_STATUSES:
            updated_job = service.repo.update_portability_job(
                str(portability_job["job_id"]),
                {"status": job_status},
                owner_user_id=owner_user_id,
            )
            if updated_job is not None:
                portability_job = updated_job
        if preview_status not in TERMINAL_JOB_STATUSES:
            updated_preview = service.repo.update_import_preview(
                int(preview["id"]),
                {"status": job_status},
                owner_user_id=owner_user_id,
            )
            if updated_preview is not None:
                preview = updated_preview
                preview_status = str(preview["status"])

    return VNPackImportPreviewResponse(
        preview_id=int(preview["id"]),
        job_id=str(preview["job_id"]),
        portability_job_id=int(portability_job["id"]),
        operation=str(portability_job["operation"]),
        status=job_status,
        vn_status=preview_status,
        stage=str(portability_job["stage"]),
        archive_sha256=preview.get("archive_sha256") or portability_job.get("archive_sha256"),
        canonical_payload_fingerprint=(
            preview.get("canonical_payload_fingerprint")
            or portability_job.get("canonical_payload_fingerprint")
        ),
        schema_version=preview.get("schema_version"),
        bundle_summary=_json_field(preview, "bundle_summary_json", {}),
        validation_warnings=_json_field(preview, "validation_warnings_json", []),
        conflicts=_json_field(preview, "conflicts_json", []),
        proposed_plan=_json_field(preview, "proposed_plan_json", {}),
        quota_estimate=_json_field(preview, "quota_estimate_json", {}),
        required_choices=_json_field(preview, "required_choices_json", []),
        error_code=portability_job.get("error_code"),
        error_message=portability_job.get("error_message"),
        expires_at=preview.get("expires_at") or portability_job.get("expires_at"),
    )


def _compose_import_commit_response(
    service: VNAssetPackService,
    *,
    journal: dict[str, Any],
    portability_job: dict[str, Any],
    job: dict[str, Any] | None,
    owner_user_id: int,
) -> VNPackImportJobResponse:
    job_status = str((job or {}).get("status") or portability_job["status"])
    journal_status = str(journal["status"])
    if job_status in TERMINAL_JOB_STATUSES:
        if str(portability_job["status"]) not in TERMINAL_JOB_STATUSES:
            updated_job = service.repo.update_portability_job(
                str(portability_job["job_id"]),
                {"status": job_status},
                owner_user_id=owner_user_id,
            )
            if updated_job is not None:
                portability_job = updated_job
        if journal_status not in TERMINAL_JOB_STATUSES:
            updated_journal = service.repo.update_import_journal(
                int(journal["id"]),
                {"status": job_status},
                owner_user_id=owner_user_id,
            )
            if updated_journal is not None:
                journal = updated_journal
                journal_status = str(journal["status"])

    return VNPackImportJobResponse(
        job_id=str(portability_job["job_id"]),
        portability_job_id=int(portability_job["id"]),
        operation=str(portability_job["operation"]),
        preview_id=int(journal["preview_id"]),
        import_id=int(journal["id"]),
        status=job_status,
        vn_status=journal_status,
        stage=str(journal.get("stage") or portability_job["stage"]),
        pack_id=journal.get("target_pack_id") or portability_job.get("pack_id"),
        id_maps=_json_field(journal, "id_maps_json", {}),
        created_records=_json_field(journal, "created_records_json", {}),
        cleanup_status=_json_field(journal, "cleanup_status_json", {}),
        warnings=_json_field(journal, "warnings_json", _json_field(portability_job, "warnings_json", [])),
        archive_sha256=journal.get("archive_sha256") or portability_job.get("archive_sha256"),
        canonical_payload_fingerprint=(
            journal.get("canonical_payload_fingerprint")
            or portability_job.get("canonical_payload_fingerprint")
        ),
        error_code=journal.get("error_code") or portability_job.get("error_code"),
        error_message=journal.get("error_message") or portability_job.get("error_message"),
        completed_at=journal.get("completed_at"),
    )


def _vn_pack_export_staging_root(owner_user_id: int) -> Path:
    configured = (os.getenv("VN_PACK_EXPORT_STAGING_ROOT") or "").strip()
    if configured:
        return Path(configured) / str(owner_user_id)
    return DatabasePaths.get_user_temp_outputs_dir(owner_user_id) / "vn_pack_exports"


def _vn_pack_import_preview_staging_root(owner_user_id: int) -> Path:
    configured = (os.getenv("VN_PACK_IMPORT_PREVIEW_STAGING_ROOT") or "").strip()
    if configured:
        return Path(configured) / str(owner_user_id)
    return DatabasePaths.get_user_temp_outputs_dir(owner_user_id) / "vn_pack_import_previews"


def _validated_export_archive_path(owner_user_id: int, archive_path: str | None) -> Path:
    if not archive_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export_archive_not_found")
    root = _vn_pack_export_staging_root(owner_user_id).resolve()
    path = Path(archive_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="export_archive_outside_user_root") from exc
    if path.suffix != VNPACK_EXTENSION:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid_export_archive_type")
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export_archive_not_found")
    return path


async def _save_import_preview_archive(archive: UploadFile, archive_path: Path) -> int:
    total_bytes = 0
    try:
        await aiofiles.os.makedirs(archive_path.parent, exist_ok=True)
        async with aiofiles.open(archive_path, "wb") as output:
            while True:
                chunk = await archive.read(UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > DEFAULT_MAX_ARCHIVE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="import_archive_too_large",
                    )
                await output.write(chunk)
    except Exception:
        await _remove_file_if_exists(archive_path)
        raise

    if total_bytes <= 0:
        await _remove_file_if_exists(archive_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="import_archive_empty")
    return total_bytes


async def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    async with aiofiles.open(path, "rb") as input_file:
        while True:
            chunk = await input_file.read(UPLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


async def _read_upload_file_with_limit(
    upload: UploadFile,
    *,
    max_bytes: int,
    empty_detail: str,
    too_large_detail: str,
) -> bytes:
    total_bytes = 0
    chunks: list[bytes] = []
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=too_large_detail,
            )
        chunks.append(chunk)
    if total_bytes <= 0:
        raise ValueError(empty_detail)
    return b"".join(chunks)


async def _remove_file_if_exists(path: Path) -> None:
    try:
        await aiofiles.os.remove(path)
    except FileNotFoundError:
        return


def _content_not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=VN_ASSET_CONTENT_NOT_FOUND)


async def _touch_generated_file(files_repo: Any, file_id: int) -> None:
    update_accessed_at = getattr(files_repo, "update_accessed_at", None)
    if not callable(update_accessed_at):
        return
    result = update_accessed_at(file_id)
    if inspect.isawaitable(result):
        await result


def _truthy_policy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "blocked", "policy_blocked"}
    return False


def _raise_if_policy_blocked(item: VNAssetItemResponse, file_record: dict[str, Any]) -> None:
    if not _truthy_policy_flag(file_record.get("policy_blocked")):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=vn_error_detail(
            ERROR_POLICY_BLOCKED,
            "VN asset item content is blocked by policy.",
            details={"pack_id": item.pack_id, "item_id": item.id},
        ),
    )


async def _item_file_response(
    *,
    pack_id: int,
    item_id: int,
    service: VNAssetPackService,
    current_user: User,
    files_repo: AuthnzGeneratedFilesRepo,
) -> FileResponse:
    try:
        item = service.get_item_for_pack(pack_id, item_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc

    if item.generated_file_id is None:
        raise _content_not_found()

    user_id = _current_user_id(current_user)
    file_record = await files_repo.get_file_by_id(item.generated_file_id)
    if not file_record or not generated_file_matches_vn_asset(file_record, user_id=user_id, item_id=item_id):
        raise _content_not_found()

    mime_type = str(file_record.get("mime_type") or item.mime_type or "")
    try:
        image_format_from_mime_type(mime_type)
    except ValueError as exc:
        raise _content_not_found() from exc

    file_category = str(file_record.get("file_category") or "").strip().lower()
    if file_category and file_category != "image":
        raise _content_not_found()

    _raise_if_policy_blocked(item, file_record)

    storage_path = str(file_record.get("storage_path") or "")
    try:
        full_path = resolve_vn_asset_storage_path(user_id=user_id, storage_path=storage_path)
    except ValueError as exc:
        raise _content_not_found() from exc

    if not full_path.is_file():
        raise _content_not_found()

    await _touch_generated_file(files_repo, item.generated_file_id)

    raw_filename = file_record.get("original_filename") or file_record.get("filename") or f"vn_asset_item_{item_id}"
    filename = Path(str(raw_filename)).name
    return FileResponse(path=str(full_path), filename=filename, media_type=mime_type)


@router.post(
    "/packs",
    response_model=VNAssetPackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_pack(
    request: VNAssetPackCreate,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetPackResponse:
    try:
        return service.create_pack(request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get("/packs", response_model=list[VNAssetPackResponse])
async def list_packs(
    service: VNAssetPackService = Depends(_service),
) -> list[VNAssetPackResponse]:
    return service.list_packs()


@router.get("/packs/{pack_id}", response_model=VNAssetPackResponse)
async def get_pack(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetPackResponse:
    try:
        return service.get_pack(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.patch("/packs/{pack_id}", response_model=VNAssetPackResponse)
async def update_pack(
    pack_id: int,
    request: VNAssetPackUpdate,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetPackResponse:
    try:
        return service.update_pack(pack_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.delete("/packs/{pack_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pack(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> Response:
    try:
        service.soft_delete_pack(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/packs/{pack_id}/cleanup", response_model=VNAssetCleanupResponse)
async def cleanup_pack(
    pack_id: int,
    request: VNAssetCleanupRequest,
    service: VNAssetPackService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    storage_service: Any = Depends(_storage_service),
    blocker_provider: Any | None = Depends(_cleanup_blocker_provider),
) -> VNAssetCleanupResponse:
    payload_hash = canonical_payload_hash(
        {
            "pack_id": pack_id,
            "request": request.model_dump(mode="json", exclude={"idempotency_key"}),
        }
    )
    replay = None
    if not request.dry_run:
        replay = _idempotency_replay(
            service,
            owner_user_id=service.owner_user_id,
            scope="vn_asset_cleanup",
            resource_id=f"pack:{pack_id}",
            idempotency_key=request.idempotency_key,
            payload_hash=payload_hash,
            response_model=VNAssetCleanupResponse,
        )
    if replay is not None:
        return replay
    try:
        response = await service.cleanup_pack(
            pack_id,
            request,
            files_repo=files_repo,
            unregister_generated_file=storage_service.unregister_generated_file,
            blocker_provider=blocker_provider,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if not request.dry_run:
        _record_idempotency_response(
            service,
            owner_user_id=service.owner_user_id,
            scope="vn_asset_cleanup",
            resource_id=f"pack:{pack_id}",
            idempotency_key=request.idempotency_key,
            payload_hash=payload_hash,
            response=response,
        )
    return response


@router.post(
    "/packs/{pack_id}/export",
    response_model=VNPackExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.export"))],
)
async def start_pack_export(
    pack_id: int,
    request: VNPackExportRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackExportResponse:
    try:
        service.get_pack(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc

    owner_user_id = _current_user_id(current_user)
    export_request = request or VNPackExportRequest()
    request_id = export_request.request_id or uuid.uuid4().hex
    options = export_request.model_dump(exclude={"request_id"})
    payload_hash = canonical_payload_hash({"pack_id": pack_id, "options": options})
    replay = _idempotency_replay(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_export",
        resource_id=f"pack:{pack_id}",
        idempotency_key=export_request.request_id,
        payload_hash=payload_hash,
        response_model=VNPackExportResponse,
    )
    if replay is not None:
        return replay
    job = create_pack_export_job(
        jobs_manager,
        pack_id=pack_id,
        portability_job_id=0,
        request_id=request_id,
        user_id=owner_user_id,
        options=options,
    )
    job_id = str(job["id"])
    portability_job = service.repo.get_portability_job_by_job_id(job_id, owner_user_id=owner_user_id)
    if portability_job is None:
        portability_job = service.repo.create_portability_job(
            owner_user_id=owner_user_id,
            job_id=job_id,
            operation="export",
            status=str(job.get("status") or "queued"),
            stage="queued",
            pack_id=pack_id,
            progress={"request_id": request_id},
        )
    response = _compose_portability_response(
        service,
        portability_job=portability_job,
        job=job,
        owner_user_id=owner_user_id,
    )
    queued_response = VNPackExportResponse(
        job_id=response.job_id,
        portability_job_id=response.portability_job_id,
        operation=response.operation,
        pack_id=response.pack_id,
        status=response.status,
        stage=response.stage,
        download_url=response.download_url,
    )
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_export",
        resource_id=f"pack:{pack_id}",
        idempotency_key=export_request.request_id,
        payload_hash=payload_hash,
        response=queued_response,
    )
    return queued_response


@router.get(
    "/portability/exports/{job_id}",
    response_model=VNPackPortabilityJobResponse,
)
async def get_pack_export_status(
    job_id: str,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackPortabilityJobResponse:
    owner_user_id = _current_user_id(current_user)
    portability_job = _portability_export_or_404(
        service,
        job_id=job_id,
        owner_user_id=owner_user_id,
    )
    job = _job_for_portability(jobs_manager, job_id)
    return _compose_portability_response(
        service,
        portability_job=portability_job,
        job=job,
        owner_user_id=owner_user_id,
    )


@router.get("/portability/exports/{job_id}/download")
async def download_pack_export(
    job_id: str,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> FileResponse:
    owner_user_id = _current_user_id(current_user)
    portability_job = _portability_export_or_404(
        service,
        job_id=job_id,
        owner_user_id=owner_user_id,
    )
    response = _compose_portability_response(
        service,
        portability_job=portability_job,
        job=_job_for_portability(jobs_manager, job_id),
        owner_user_id=owner_user_id,
    )
    if response.status != "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="export_not_completed")

    archive_path = _validated_export_archive_path(owner_user_id, portability_job.get("archive_path"))
    return FileResponse(
        path=str(archive_path),
        filename=archive_path.name,
        media_type="application/zip",
    )


@router.post(
    "/portability/exports/{job_id}/cancel",
    response_model=VNPackPortabilityJobResponse,
)
async def cancel_pack_export(
    job_id: str,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackPortabilityJobResponse:
    owner_user_id = _current_user_id(current_user)
    portability_job = _portability_export_or_404(
        service,
        job_id=job_id,
        owner_user_id=owner_user_id,
    )
    try:
        cancelled = jobs_manager.cancel_job(int(job_id), reason="vn_pack_export_cancel_requested")
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export_job_not_found") from exc
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export_job_not_found")

    updated = service.repo.update_portability_job(
        job_id,
        {"status": "cancelled", "stage": "cancelled"},
        owner_user_id=owner_user_id,
    )
    return _compose_portability_response(
        service,
        portability_job=updated or portability_job,
        job=_job_for_portability(jobs_manager, job_id),
        owner_user_id=owner_user_id,
    )


@router.post(
    "/import/previews",
    response_model=VNPackImportPreviewStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.import"))],
)
async def start_pack_import_preview(
    archive: UploadFile = File(...),
    request_id: str | None = Form(None, min_length=1, max_length=160),
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportPreviewStartResponse:
    owner_user_id = _current_user_id(current_user)
    operation_request_id = request_id or uuid.uuid4().hex
    archive_token = uuid.uuid4().hex
    archive_root = _vn_pack_import_preview_staging_root(owner_user_id)
    archive_path = archive_root / f"{archive_token}{VNPACK_EXTENSION}"
    uploaded_bytes = await _save_import_preview_archive(archive, archive_path)
    archive_sha256 = await _file_sha256(archive_path)
    payload_hash = canonical_multipart_payload_hash(
        {"request_id": request_id},
        file_sha256=archive_sha256,
        filename=archive.filename,
        content_type=archive.content_type,
    )
    try:
        replay = _idempotency_replay(
            service,
            owner_user_id=owner_user_id,
            scope="vn_asset_import_preview",
            resource_id="import_preview",
            idempotency_key=request_id,
            payload_hash=payload_hash,
            response_model=VNPackImportPreviewStartResponse,
        )
    except HTTPException:
        await _remove_file_if_exists(archive_path)
        raise
    if replay is not None:
        await _remove_file_if_exists(archive_path)
        return replay

    preview = service.repo.create_import_preview(
        owner_user_id=owner_user_id,
        job_id=f"pending:{operation_request_id}",
        status="queued",
        archive_path=str(archive_path),
    )
    job = create_pack_import_preview_job(
        jobs_manager,
        preview_id=int(preview["id"]),
        archive_path=str(archive_path),
        request_id=operation_request_id,
        user_id=owner_user_id,
    )
    job_id = str(job["id"])
    preview = service.repo.update_import_preview(
        int(preview["id"]),
        {"job_id": job_id, "status": str(job.get("status") or "queued")},
        owner_user_id=owner_user_id,
    ) or preview
    portability_job = service.repo.create_portability_job(
        owner_user_id=owner_user_id,
        job_id=job_id,
        operation="import_preview",
        status=str(job.get("status") or "queued"),
        stage="queued",
        preview_id=int(preview["id"]),
        archive_path=str(archive_path),
        progress={"request_id": operation_request_id, "uploaded_bytes": uploaded_bytes},
    )

    queued_response = VNPackImportPreviewStartResponse(
        job_id=job_id,
        portability_job_id=int(portability_job["id"]),
        operation=str(portability_job["operation"]),
        preview_id=int(preview["id"]),
        status=str(job.get("status") or portability_job["status"]),
        stage=str(portability_job["stage"]),
    )
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_import_preview",
        resource_id="import_preview",
        idempotency_key=request_id,
        payload_hash=payload_hash,
        response=queued_response,
    )
    return queued_response


@router.get(
    "/import/previews/{preview_id}",
    response_model=VNPackImportPreviewResponse,
)
async def get_pack_import_preview(
    preview_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportPreviewResponse:
    owner_user_id = _current_user_id(current_user)
    preview, portability_job = _portability_import_preview_or_404(
        service,
        preview_id=preview_id,
        owner_user_id=owner_user_id,
    )
    return _compose_import_preview_response(
        service,
        preview=preview,
        portability_job=portability_job,
        job=_job_for_portability(jobs_manager, str(preview["job_id"])),
        owner_user_id=owner_user_id,
    )


@router.post(
    "/import/previews/{preview_id}/cancel",
    response_model=VNPackImportPreviewResponse,
)
async def cancel_pack_import_preview(
    preview_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportPreviewResponse:
    owner_user_id = _current_user_id(current_user)
    preview, portability_job = _portability_import_preview_or_404(
        service,
        preview_id=preview_id,
        owner_user_id=owner_user_id,
    )
    job_id = str(preview["job_id"])
    try:
        cancelled = jobs_manager.cancel_job(
            int(job_id),
            reason="vn_pack_import_preview_cancel_requested",
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_preview_job_not_found") from exc
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_preview_job_not_found")

    updated_preview = service.repo.update_import_preview(
        preview_id,
        {"status": "cancelled"},
        owner_user_id=owner_user_id,
    )
    updated_job = service.repo.update_portability_job(
        job_id,
        {"status": "cancelled", "stage": "cancelled"},
        owner_user_id=owner_user_id,
    )
    return _compose_import_preview_response(
        service,
        preview=updated_preview or preview,
        portability_job=updated_job or portability_job,
        job=_job_for_portability(jobs_manager, job_id),
        owner_user_id=owner_user_id,
    )


@router.delete("/import/previews/{preview_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pack_import_preview(
    preview_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> Response:
    owner_user_id = _current_user_id(current_user)
    preview, portability_job = _portability_import_preview_or_404(
        service,
        preview_id=preview_id,
        owner_user_id=owner_user_id,
    )
    if str(preview["status"]) == "processing":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="import_preview_processing")
    job_id = str(preview["job_id"])
    try:
        jobs_manager.cancel_job(int(job_id), reason="vn_pack_import_preview_delete_requested")
    except (TypeError, ValueError):
        pass
    archive_path = Path(str(preview.get("archive_path") or ""))
    preview_root = _vn_pack_import_preview_staging_root(owner_user_id).resolve()
    try:
        resolved_archive_path = archive_path.resolve()
        resolved_archive_path.relative_to(preview_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="import_preview_archive_outside_user_root") from exc
    if resolved_archive_path.is_file():
        await _remove_file_if_exists(resolved_archive_path)
    service.repo.update_import_preview(
        preview_id,
        {"status": "deleted"},
        owner_user_id=owner_user_id,
    )
    service.repo.update_portability_job(
        str(portability_job["job_id"]),
        {"status": "cancelled", "stage": "deleted"},
        owner_user_id=owner_user_id,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/import/commit",
    response_model=VNPackImportCommitStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.import"))],
)
async def start_pack_import_commit(
    request: VNPackImportCommitRequest,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportCommitStartResponse:
    owner_user_id = _current_user_id(current_user)
    if request.target_mode == "update_existing":
        if request.target_pack_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_pack_required")
        try:
            service.get_pack(int(request.target_pack_id))
        except ValueError as exc:
            raise _handle_value_error(exc) from exc
    if request.character_action == "fail_import":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="primary_character_unresolved")
    if request.character_action == "link_existing_character":
        if request.target_character_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="target_character_required")
        if service.repo.get_character(int(request.target_character_id)) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="target_character_not_found")

    preview = service.repo.get_import_preview(request.preview_id, owner_user_id=owner_user_id)
    if preview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_preview_not_found")
    if str(preview["status"]) != "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="import_preview_not_completed")

    request_id = request.request_id or uuid.uuid4().hex
    payload_hash = canonical_payload_hash(
        {
            "preview_id": request.preview_id,
            "request": request.model_dump(mode="json", exclude={"request_id"}),
        }
    )
    replay = _idempotency_replay(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_import_commit",
        resource_id=f"preview:{request.preview_id}",
        idempotency_key=request.request_id,
        payload_hash=payload_hash,
        response_model=VNPackImportCommitStartResponse,
    )
    if replay is not None:
        return replay
    journal = service.repo.create_import_journal(
        owner_user_id=owner_user_id,
        preview_id=int(preview["id"]),
        job_id=f"pending:{request_id}",
        status="queued",
        stage="queued",
        trust_mode=request.trust_mode,
        target_mode=request.target_mode,
        archive_path=preview.get("archive_path"),
        archive_sha256=preview.get("archive_sha256"),
        canonical_payload_fingerprint=preview.get("canonical_payload_fingerprint"),
    )
    try:
        job = create_pack_import_commit_job(
            jobs_manager,
            import_id=int(journal["id"]),
            preview_id=int(preview["id"]),
            request_id=request_id,
            user_id=owner_user_id,
            trust_mode=request.trust_mode,
            target_mode=request.target_mode,
            character_action=request.character_action,
            target_character_id=request.target_character_id,
            target_pack_id=request.target_pack_id,
            conflict_decisions=request.conflict_decisions,
        )
    except Exception as exc:
        service.repo.update_import_journal(
            int(journal["id"]),
            {
                "status": "failed",
                "stage": "failed",
                "error_code": "import_job_create_failed",
                "error_message": str(exc),
            },
            owner_user_id=owner_user_id,
        )
        raise

    job_id = str(job["id"])
    journal = service.repo.update_import_journal(
        int(journal["id"]),
        {"job_id": job_id, "status": str(job.get("status") or "queued")},
        owner_user_id=owner_user_id,
    ) or journal
    portability_job = service.repo.create_portability_job(
        owner_user_id=owner_user_id,
        job_id=job_id,
        operation="import_commit",
        status=str(job.get("status") or "queued"),
        stage="queued",
        preview_id=int(preview["id"]),
        import_id=int(journal["id"]),
        archive_path=preview.get("archive_path"),
        archive_sha256=preview.get("archive_sha256"),
        canonical_payload_fingerprint=preview.get("canonical_payload_fingerprint"),
        progress={"request_id": request_id},
    )

    queued_response = VNPackImportCommitStartResponse(
        job_id=job_id,
        portability_job_id=int(portability_job["id"]),
        operation=str(portability_job["operation"]),
        preview_id=int(preview["id"]),
        import_id=int(journal["id"]),
        status=str(job.get("status") or portability_job["status"]),
        stage=str(portability_job["stage"]),
    )
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_import_commit",
        resource_id=f"preview:{request.preview_id}",
        idempotency_key=request.request_id,
        payload_hash=payload_hash,
        response=queued_response,
    )
    return queued_response


@router.get(
    "/portability/imports/{job_id}",
    response_model=VNPackImportJobResponse,
)
async def get_pack_import_commit_status(
    job_id: str,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportJobResponse:
    owner_user_id = _current_user_id(current_user)
    journal, portability_job = _portability_import_commit_or_404(
        service,
        job_id=job_id,
        owner_user_id=owner_user_id,
    )
    return _compose_import_commit_response(
        service,
        journal=journal,
        portability_job=portability_job,
        job=_job_for_portability(jobs_manager, job_id),
        owner_user_id=owner_user_id,
    )


@router.post(
    "/portability/imports/{job_id}/cancel",
    response_model=VNPackImportJobResponse,
)
async def cancel_pack_import_commit(
    job_id: str,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportJobResponse:
    owner_user_id = _current_user_id(current_user)
    journal, portability_job = _portability_import_commit_or_404(
        service,
        job_id=job_id,
        owner_user_id=owner_user_id,
    )
    try:
        cancelled = jobs_manager.cancel_job(
            int(job_id),
            reason="vn_pack_import_commit_cancel_requested",
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_commit_job_not_found") from exc
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_commit_job_not_found")

    updated_journal = service.repo.update_import_journal(
        int(journal["id"]),
        {"status": "cancelled", "stage": "cancelled"},
        owner_user_id=owner_user_id,
    )
    updated_job = service.repo.update_portability_job(
        job_id,
        {"status": "cancelled", "stage": "cancelled"},
        owner_user_id=owner_user_id,
    )
    return _compose_import_commit_response(
        service,
        journal=updated_journal or journal,
        portability_job=updated_job or portability_job,
        job=_job_for_portability(jobs_manager, job_id),
        owner_user_id=owner_user_id,
    )


@router.post(
    "/portability/imports/{import_id}/cleanup",
    response_model=VNPackImportJobResponse,
)
async def cleanup_pack_import_commit(
    import_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNPackImportJobResponse:
    owner_user_id = _current_user_id(current_user)
    journal = service.repo.get_import_journal(import_id, owner_user_id=owner_user_id)
    if journal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_journal_not_found")
    if str(journal["status"]) != "failed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="import_cleanup_requires_failed_import")
    portability_job = service.repo.get_portability_job_by_job_id(
        str(journal["job_id"]),
        owner_user_id=owner_user_id,
    )
    if portability_job is None or portability_job.get("operation") != "import_commit":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import_commit_job_not_found")
    if not _json_field(journal, "cleanup_status_json", {}):
        journal = service.repo.update_import_journal(
            import_id,
            {"cleanup_status": {"status": "nothing_to_clean"}},
            owner_user_id=owner_user_id,
        ) or journal
    return _compose_import_commit_response(
        service,
        journal=journal,
        portability_job=portability_job,
        job=_job_for_portability(jobs_manager, str(journal["job_id"])),
        owner_user_id=owner_user_id,
    )


@router.get("/starter-matrices", response_model=VNAssetStarterMatricesResponse)
async def list_starter_matrices() -> VNAssetStarterMatricesResponse:
    slots = expand_starter_matrix(primary_character_id=1, variant_count=1)
    return VNAssetStarterMatricesResponse(
        matrices=[
            {
                "key": "starter",
                "title": "Starter",
                "slot_count": len(slots),
                "planned_output_count": sum(slot.variant_count for slot in slots),
                "asset_types": sorted({slot.asset_type for slot in slots}),
            }
        ]
    )


@router.post("/packs/{pack_id}/matrix/apply", response_model=list[VNAssetSlotResponse])
async def apply_matrix(
    pack_id: int,
    request: VNAssetMatrixApplyRequest,
    service: VNAssetPackService = Depends(_service),
) -> list[VNAssetSlotResponse]:
    try:
        return service.apply_matrix(pack_id, request.matrix_key, request.overrides)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get("/packs/{pack_id}/slots", response_model=list[VNAssetSlotResponse])
async def list_slots(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> list[VNAssetSlotResponse]:
    try:
        return service.list_slots(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post(
    "/packs/{pack_id}/slots",
    response_model=VNAssetSlotResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_slot(
    pack_id: int,
    request: VNAssetSlotCreate,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetSlotResponse:
    try:
        return service.create_slot(pack_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.patch("/packs/{pack_id}/slots/{slot_id}", response_model=VNAssetSlotResponse)
async def update_slot(
    pack_id: int,
    slot_id: int,
    request: VNAssetSlotUpdate,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetSlotResponse:
    try:
        return service.update_slot_for_pack(pack_id, slot_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.delete("/packs/{pack_id}/slots/{slot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_slot(
    pack_id: int,
    slot_id: int,
    service: VNAssetPackService = Depends(_service),
) -> Response:
    try:
        service.delete_slot(pack_id, slot_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/packs/{pack_id}/items", response_model=list[VNAssetItemResponse])
async def list_items(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> list[VNAssetItemResponse]:
    try:
        return service.list_items(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.patch("/packs/{pack_id}/items/{item_id}/review", response_model=VNAssetItemResponse)
async def review_item(
    pack_id: int,
    item_id: int,
    request: VNAssetReviewRequest,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetItemResponse:
    try:
        return service.review_item_for_pack(pack_id, item_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get(
    "/packs/{pack_id}/items/{item_id}/content",
    response_class=FileResponse,
    responses={
        200: {
            "description": "VN asset item content.",
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "image/webp": {},
                "application/octet-stream": {},
            },
        },
    },
)
async def get_item_content(
    pack_id: int,
    item_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
) -> FileResponse:
    return await _item_file_response(
        pack_id=pack_id,
        item_id=item_id,
        service=service,
        current_user=current_user,
        files_repo=files_repo,
    )


@router.get(
    "/packs/{pack_id}/items/{item_id}/preview",
    response_class=FileResponse,
    responses={
        200: {
            "description": "VN asset item preview content.",
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "image/webp": {},
            },
        },
    },
)
async def get_item_preview(
    pack_id: int,
    item_id: int,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
) -> FileResponse:
    return await _item_file_response(
        pack_id=pack_id,
        item_id=item_id,
        service=service,
        current_user=current_user,
        files_repo=files_repo,
    )


@router.post("/packs/{pack_id}/items/bulk-review", response_model=list[VNAssetItemResponse])
async def bulk_review_items(
    pack_id: int,
    request: VNAssetBulkReviewRequest,
    service: VNAssetPackService = Depends(_service),
) -> list[VNAssetItemResponse]:
    try:
        return service.bulk_review_items_for_pack(pack_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post(
    "/packs/{pack_id}/items/upload",
    response_model=VNAssetItemResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.upload"))],
)
async def upload_item(
    pack_id: int,
    slot_id: int = Form(...),
    file: UploadFile = File(...),
    variant_index: int = Form(0),
    idempotency_key: str | None = Form(None, min_length=1, max_length=160),
    service: VNAssetPackService = Depends(_service),
) -> VNAssetItemResponse:
    try:
        mime_type = file.content_type or "application/octet-stream"
        image_format_from_mime_type(mime_type)
        image_bytes = await _read_upload_file_with_limit(
            file,
            max_bytes=_get_vn_asset_upload_max_bytes(),
            empty_detail="vn_asset_upload_empty",
            too_large_detail="vn_asset_upload_too_large",
        )
        payload_hash = canonical_multipart_payload_hash(
            {
                "pack_id": pack_id,
                "slot_id": slot_id,
                "variant_index": variant_index,
            },
            file_sha256=hashlib.sha256(image_bytes).hexdigest(),
            filename=file.filename,
            content_type=mime_type,
        )
        replay = _idempotency_replay(
            service,
            owner_user_id=service.owner_user_id,
            scope="vn_asset_item_upload",
            resource_id=f"pack:{pack_id}:slot:{slot_id}",
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            response_model=VNAssetItemResponse,
        )
        if replay is not None:
            return replay
        response = await service.upload_item(
            pack_id,
            slot_id=slot_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            variant_index=variant_index,
        )
        _record_idempotency_response(
            service,
            owner_user_id=service.owner_user_id,
            scope="vn_asset_item_upload",
            resource_id=f"pack:{pack_id}:slot:{slot_id}",
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            response=response,
        )
        return response
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/packs/{pack_id}/items/{item_id}/preferred", response_model=VNAssetItemResponse)
async def set_preferred_item(
    pack_id: int,
    item_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetItemResponse:
    try:
        return service.set_preferred_item(pack_id, item_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get("/packs/{pack_id}/manifest", response_model=VNAssetManifestResponse)
async def get_manifest(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetManifestResponse:
    try:
        return service.build_manifest(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get("/packs/{pack_id}/readiness", response_model=VNAssetReadinessResponse)
async def get_readiness(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetReadinessResponse:
    try:
        return service.get_readiness(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/packs/{pack_id}/prompt-preview", response_model=VNAssetPromptPreviewResponse)
async def prompt_preview(
    pack_id: int,
    request: VNAssetPromptPreviewRequest,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetPromptPreviewResponse:
    try:
        return service.preview_prompt(pack_id, request)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post(
    "/packs/{pack_id}/generate",
    response_model=VNAssetGenerationStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.generate"))],
)
async def start_generation(
    pack_id: int,
    request: VNAssetGenerationRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetGenerationStatusResponse:
    generation_request = request or VNAssetGenerationRequest()
    owner_user_id = _current_user_id(current_user)
    payload_hash = canonical_payload_hash(
        {
            "pack_id": pack_id,
            "request": generation_request.model_dump(mode="json", exclude={"idempotency_key"}),
        }
    )
    replay = _idempotency_replay(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_generate",
        resource_id=f"pack:{pack_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response_model=VNAssetGenerationStatusResponse,
    )
    if replay is not None:
        return replay
    try:
        response = service.start_generation(
            pack_id,
            generation_request,
            user_id=owner_user_id,
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_generate",
        resource_id=f"pack:{pack_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response=response,
    )
    return response


@router.get("/packs/{pack_id}/generation", response_model=VNAssetGenerationStatusResponse)
async def get_generation_status(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetGenerationStatusResponse:
    try:
        return service.get_generation_status(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/packs/{pack_id}/generation/cancel", response_model=VNAssetGenerationStatusResponse)
async def cancel_generation(
    pack_id: int,
    service: VNAssetPackService = Depends(_service),
) -> VNAssetGenerationStatusResponse:
    try:
        return service.cancel_generation(pack_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post(
    "/packs/{pack_id}/slots/{slot_id}/retry",
    response_model=VNAssetGenerationStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rbac_rate_limit("vn_assets.generate"))],
)
async def retry_slot_generation(
    pack_id: int,
    slot_id: int,
    request: VNAssetGenerationRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetGenerationStatusResponse:
    generation_request = request or VNAssetGenerationRequest()
    owner_user_id = _current_user_id(current_user)
    payload_hash = canonical_payload_hash(
        {
            "pack_id": pack_id,
            "slot_id": slot_id,
            "request": generation_request.model_dump(mode="json", exclude={"idempotency_key"}),
        }
    )
    replay = _idempotency_replay(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_slot_retry",
        resource_id=f"pack:{pack_id}:slot:{slot_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response_model=VNAssetGenerationStatusResponse,
    )
    if replay is not None:
        return replay
    try:
        response = service.retry_slot(
            pack_id,
            slot_id,
            generation_request,
            user_id=owner_user_id,
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_slot_retry",
        resource_id=f"pack:{pack_id}:slot:{slot_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response=response,
    )
    return response


@router.post(
    "/packs/{pack_id}/items/{item_id}/regenerate",
    response_model=VNAssetGenerationStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_item(
    pack_id: int,
    item_id: int,
    request: VNAssetGenerationRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetGenerationStatusResponse:
    generation_request = request or VNAssetGenerationRequest()
    owner_user_id = _current_user_id(current_user)
    payload_hash = canonical_payload_hash(
        {
            "pack_id": pack_id,
            "item_id": item_id,
            "request": generation_request.model_dump(mode="json", exclude={"idempotency_key"}),
        }
    )
    replay = _idempotency_replay(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_item_regenerate",
        resource_id=f"pack:{pack_id}:item:{item_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response_model=VNAssetGenerationStatusResponse,
    )
    if replay is not None:
        return replay
    try:
        response = service.regenerate_item(
            pack_id,
            item_id,
            generation_request,
            user_id=owner_user_id,
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    _record_idempotency_response(
        service,
        owner_user_id=owner_user_id,
        scope="vn_asset_item_regenerate",
        resource_id=f"pack:{pack_id}:item:{item_id}",
        idempotency_key=generation_request.idempotency_key,
        payload_hash=payload_hash,
        response=response,
    )
    return response
