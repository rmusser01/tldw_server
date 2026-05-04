"""VN asset pack metadata endpoints."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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
)
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.constants import DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
from tldw_Server_API.app.core.VN_Assets.matrix import expand_starter_matrix
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Assets.storage import (
    VN_ASSET_CONTENT_NOT_FOUND,
    generated_file_matches_vn_asset,
    image_format_from_mime_type,
    resolve_vn_asset_storage_path,
)
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

router = APIRouter(prefix="/vn-assets", tags=["vn-assets"])
CONFLICT_ERROR_CODES = {
    "slot_already_exists",
    "slot_has_dependents",
}


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


def _handle_value_error(exc: ValueError) -> HTTPException:
    detail = str(exc) or "invalid_request"
    if "not_found" in detail:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    if detail in CONFLICT_ERROR_CODES:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _current_user_id(current_user: User) -> int:
    user_id = current_user.id_int
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_user_id",
        )
    return user_id


def _content_not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=VN_ASSET_CONTENT_NOT_FOUND)


async def _touch_generated_file(files_repo: Any, file_id: int) -> None:
    update_accessed_at = getattr(files_repo, "update_accessed_at", None)
    if not callable(update_accessed_at):
        return
    result = update_accessed_at(file_id)
    if inspect.isawaitable(result):
        await result


def _get_vn_asset_upload_max_bytes() -> int:
    raw_value = os.getenv("VN_ASSET_UPLOAD_MAX_BYTES")
    if raw_value is None:
        return DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
    try:
        max_bytes = int(raw_value)
    except ValueError:
        return DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES
    return max_bytes if max_bytes > 0 else DEFAULT_VN_ASSET_UPLOAD_MAX_BYTES


def _upload_too_large_status_code() -> int:
    return getattr(
        status,
        "HTTP_413_CONTENT_TOO_LARGE",
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    )


async def _read_vn_asset_upload_bytes(file: UploadFile) -> bytes:
    max_bytes = _get_vn_asset_upload_max_bytes()
    image_bytes = await file.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=_upload_too_large_status_code(),
            detail="vn_asset_upload_too_large",
        )
    return image_bytes


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
) -> VNAssetCleanupResponse:
    try:
        return await service.cleanup_pack(
            pack_id,
            request,
            files_repo=files_repo,
            unregister_generated_file=storage_service.unregister_generated_file,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


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
    mime_type = file_record.get("mime_type") or item.mime_type or "application/octet-stream"

    return FileResponse(path=str(full_path), filename=filename, media_type=mime_type)


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
)
async def upload_item(
    pack_id: int,
    slot_id: int = Form(...),
    file: UploadFile = File(...),
    variant_index: int = Form(0),
    service: VNAssetPackService = Depends(_service),
) -> VNAssetItemResponse:
    try:
        mime_type = file.content_type or "application/octet-stream"
        image_format_from_mime_type(mime_type)
        image_bytes = await _read_vn_asset_upload_bytes(file)
        if not image_bytes:
            raise ValueError("vn_asset_upload_empty")
        return await service.upload_item(
            pack_id,
            slot_id=slot_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            variant_index=variant_index,
        )
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
)
async def start_generation(
    pack_id: int,
    request: VNAssetGenerationRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetGenerationStatusResponse:
    try:
        return service.start_generation(
            pack_id,
            request or VNAssetGenerationRequest(),
            user_id=_current_user_id(current_user),
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


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
)
async def retry_slot_generation(
    pack_id: int,
    slot_id: int,
    request: VNAssetGenerationRequest | None = None,
    service: VNAssetPackService = Depends(_service),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VNAssetGenerationStatusResponse:
    try:
        return service.retry_slot(
            pack_id,
            slot_id,
            request or VNAssetGenerationRequest(),
            user_id=_current_user_id(current_user),
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


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
    try:
        return service.regenerate_item(
            pack_id,
            item_id,
            request or VNAssetGenerationRequest(),
            user_id=_current_user_id(current_user),
            jobs_manager=jobs_manager,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
