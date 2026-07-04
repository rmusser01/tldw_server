"""Visual identity expression pack endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user, rbac_rate_limit
from tldw_Server_API.app.api.v1.schemas.visual_identity_schemas import (
    ActorKind,
    VisualIdentityAssetResponse,
    VisualIdentityBindingRequest,
    VisualIdentityBindingResponse,
    VisualIdentityCapabilitiesResponse,
    VisualIdentityDraftActivateRequest,
    VisualIdentityDraftResponse,
    VisualIdentityDraftSlotUpdate,
    VisualIdentityExpressionSlotResponse,
    VisualIdentityGeneratedFileAssetRequest,
    VisualIdentityImportZipStartResponse,
    VisualIdentityPackCreate,
    VisualIdentityPackResponse,
    VisualIdentityPackUpdate,
    VisualIdentityResolveResponse,
)
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.storage import SOURCE_FEATURE_VN_ASSETS
from tldw_Server_API.app.core.Visual_Identities.archive_import import MAX_EXPRESSION_ZIP_BYTES
from tldw_Server_API.app.core.Visual_Identities.constraints import (
    MAX_EXPRESSION_ASSET_BYTES,
    build_visual_identity_capabilities,
    supported_visual_identity_mime_types,
)
from tldw_Server_API.app.core.Visual_Identities.expression_slots import (
    CANONICAL_EXPRESSION_SLOTS,
    EXPRESSION_ALIASES,
    display_label_for_expression_key,
    normalize_expression_key,
)
from tldw_Server_API.app.core.Visual_Identities.jobs import create_visual_identity_import_zip_job
from tldw_Server_API.app.core.Visual_Identities.service import (
    VisualIdentityService,
    VisualIdentityServiceError,
)
from tldw_Server_API.app.core.Visual_Identities.source_context import canonicalize_source_context
from tldw_Server_API.app.core.Visual_Identities.storage import (
    copy_generated_file_record_to_expression_asset,
    resolve_visual_identity_asset_path,
    validate_generated_file_record_for_expression_asset,
    validate_and_store_visual_identity_asset,
)
from tldw_Server_API.app.core.Visual_Identities.vn_bridge import (
    build_vn_visual_identity_source_context,
)
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

router = APIRouter()

_IMMUTABLE_ASSET_CACHE_CONTROL = "private, max-age=31536000, immutable"
_UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
_API_PREFIX = "/api/v1/visual-identities"
_READ_LIMIT = Depends(rbac_rate_limit("visual-identities.read"))
_WRITE_LIMIT = Depends(rbac_rate_limit("visual-identities.write"))
_DELETE_LIMIT = Depends(rbac_rate_limit("visual-identities.delete"))


def _job_manager() -> JobManager:
    return JobManager()


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VisualIdentityService:
    owner_user_id = _current_user_id(current_user)
    return VisualIdentityService(db, owner_user_id=owner_user_id, jobs_manager=jobs_manager)


async def _generated_files_repo() -> AuthnzGeneratedFilesRepo:
    storage_service = await get_storage_service()
    return await storage_service.get_generated_files_repo()


def _current_user_id(current_user: User) -> int:
    user_id = current_user.id_int
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_user_id",
        )
    return user_id


def _repo(service: VisualIdentityService) -> VisualIdentityRepository:
    return service.repository


def _json_mapping(row: dict[str, Any], key: str) -> dict[str, Any]:
    raw_value = row.get(key)
    if isinstance(raw_value, dict):
        return raw_value
    if not raw_value:
        return {}
    try:
        parsed = json.loads(str(raw_value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _pack_response(row: dict[str, Any]) -> VisualIdentityPackResponse:
    return VisualIdentityPackResponse(
        id=int(row["id"]),
        owner_user_id=int(row["owner_user_id"]),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        status=str(row["status"]),
        active_version_id=(
            int(row["active_version_id"]) if row.get("active_version_id") is not None else None
        ),
        default_expression_key=str(row.get("default_expression_key") or "neutral"),
        source_kind=str(row.get("source_kind") or "manual"),
        source_context=_json_mapping(row, "source_context_json"),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
        version=int(row["version"]),
    )


def _asset_response(row: dict[str, Any]) -> VisualIdentityAssetResponse:
    return VisualIdentityAssetResponse(
        id=int(row["id"]),
        owner_user_id=int(row["owner_user_id"]),
        pack_id=int(row["pack_id"]) if row.get("pack_id") is not None else None,
        draft_id=int(row["draft_id"]) if row.get("draft_id") is not None else None,
        pack_version_id=(
            int(row["pack_version_id"]) if row.get("pack_version_id") is not None else None
        ),
        expression_key=str(row["expression_key"]),
        original_expression_key=str(row.get("original_expression_key") or ""),
        display_label=str(row.get("display_label") or ""),
        source_filename=str(row.get("source_filename") or ""),
        content_type=str(row["content_type"]),
        bytes=int(row["bytes"]),
        sha256=str(row["sha256"]),
        width=int(row["width"]),
        height=int(row["height"]),
        is_animated=bool(row.get("is_animated")),
        frame_count=int(row["frame_count"]) if row.get("frame_count") is not None else None,
        duration_ms=int(row["duration_ms"]) if row.get("duration_ms") is not None else None,
        preview_relpath=str(row["preview_relpath"]) if row.get("preview_relpath") else None,
        source_context=_json_mapping(row, "source_context_json"),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
    )


def _draft_response(
    service: VisualIdentityService,
    row: dict[str, Any],
    *,
    pack_version_id: int | None = None,
    asset_ids: list[int] | None = None,
    binding_id: int | None = None,
) -> VisualIdentityDraftResponse:
    assets = _repo(service).list_draft_assets(int(row["id"]), owner_user_id=service.owner_user_id)
    return VisualIdentityDraftResponse(
        id=int(row["id"]),
        owner_user_id=int(row["owner_user_id"]),
        pack_id=int(row["pack_id"]) if row.get("pack_id") is not None else None,
        title=str(row["title"]),
        status=str(row["status"]),
        source_kind=str(row["source_kind"]),
        source_filename=str(row.get("source_filename") or ""),
        import_job_id=str(row["import_job_id"]) if row.get("import_job_id") else None,
        validation_summary=_json_mapping(row, "validation_summary_json"),
        slot_map=_json_mapping(row, "slot_map_json"),
        default_expression_key=str(row.get("default_expression_key") or "neutral"),
        error=_json_mapping(row, "error_json"),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
        version=int(row["version"]),
        assets=[_asset_response(asset) for asset in assets],
        pack_version_id=pack_version_id,
        asset_ids=list(asset_ids or []),
        binding_id=binding_id,
    )


def _binding_response(row: dict[str, Any]) -> VisualIdentityBindingResponse:
    return VisualIdentityBindingResponse(
        id=int(row["id"]),
        owner_user_id=int(row["owner_user_id"]),
        actor_kind=str(row["actor_kind"]),  # type: ignore[arg-type]
        actor_id=str(row["actor_id"]),
        pack_id=int(row["pack_id"]),
        active_version_id=int(row["active_version_id"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]) if row.get("created_at") is not None else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") is not None else None,
        version=int(row["version"]),
    )


def _normalize_expression_or_422(value: str, *, field_name: str = "expression_key") -> str:
    normalized = normalize_expression_key(value)
    if normalized is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid_{field_name}",
        )
    return normalized


def _handle_value_error(exc: ValueError) -> HTTPException:
    detail = str(exc) or "invalid_request"
    if "not_found" in detail or "not_owned" in detail:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)
    if "conflict" in detail or detail in {"visual_identity_draft_not_ready"}:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _idempotency_conflict(*, scope: str, resource_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "idempotency_key_conflict",
            "scope": scope,
            "resource_id": resource_id,
        },
    )


def _handle_idempotency_service_error(
    exc: VisualIdentityServiceError,
    *,
    scope: str,
    resource_id: str,
) -> HTTPException:
    detail = str(exc) or "invalid_request"
    if detail == "idempotency_key_conflict":
        return _idempotency_conflict(scope=scope, resource_id=resource_id)
    if detail == "idempotency_key_in_progress":
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    if detail == "idempotency_response_invalid":
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
    return _handle_value_error(ValueError(detail))


def _require_idempotency_key(idempotency_key: str | None) -> str:
    """Return a trimmed idempotency key or reject the request."""
    normalized = str(idempotency_key or "").strip()
    if normalized:
        return normalized
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="idempotency_key_required",
    )


def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Return the stable SHA-256 hash used for idempotency payload matching."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    """Stream a file into a SHA-256 digest without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(_UPLOAD_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_content_url(*, pack_id: int, asset_id: int) -> str:
    return f"{_API_PREFIX}/packs/{pack_id}/assets/{asset_id}/content"


def _asset_preview_url(*, pack_id: int, asset_id: int) -> str:
    return f"{_API_PREFIX}/packs/{pack_id}/assets/{asset_id}/preview"


def _api_fallback_reason(selection_reason: str | None) -> str | None:
    if selection_reason == "requested":
        return None
    return selection_reason


def _require_pack(service: VisualIdentityService, pack_id: int) -> dict[str, Any]:
    pack = _repo(service).get_pack(pack_id, owner_user_id=service.owner_user_id)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="visual_identity_pack_not_found")
    return pack


def _require_draft(service: VisualIdentityService, draft_id: int) -> dict[str, Any]:
    draft = _repo(service).get_draft(draft_id, owner_user_id=service.owner_user_id)
    if draft is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_draft_not_found",
        )
    return draft


def _require_asset_for_pack(
    service: VisualIdentityService,
    *,
    pack_id: int,
    asset_id: int,
) -> dict[str, Any]:
    asset = _repo(service).get_asset(asset_id, owner_user_id=service.owner_user_id)
    if asset is None or asset.get("pack_id") is None or int(asset["pack_id"]) != int(pack_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_asset_not_found",
        )
    return asset


def _draft_for_asset_upload(
    service: VisualIdentityService,
    pack: dict[str, Any],
    *,
    draft_id: int | None,
) -> dict[str, Any]:
    if draft_id is not None:
        draft = _require_draft(service, draft_id)
        draft_pack_id = draft.get("pack_id")
        if draft_pack_id is not None and int(draft_pack_id) != int(pack["id"]):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="visual_identity_draft_not_found",
            )
        return draft

    return _repo(service).create_draft(
        owner_user_id=service.owner_user_id,
        pack_id=int(pack["id"]),
        title=f"{pack['title']} asset upload",
        source_kind="manual_upload",
        status="ready_for_review",
        default_expression_key=str(pack.get("default_expression_key") or "neutral"),
    )


def _safe_upload_filename(upload: UploadFile, *, fallback: str) -> str:
    filename = Path(str(upload.filename or fallback)).name
    return filename or fallback


async def _upload_to_temp_file(
    upload: UploadFile,
    *,
    max_bytes: int,
) -> tuple[Path, int]:
    suffix = Path(str(upload.filename or "")).suffix
    handle = tempfile.NamedTemporaryFile(
        prefix="visual_identity_upload_",
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(handle.name)
    total = 0
    try:
        with handle:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="file_too_large",
                    )
                await asyncio.to_thread(handle.write, chunk)
        if total <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid_file",
            )
        return temp_path, total
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _create_asset_from_stored_metadata(
    service: VisualIdentityService,
    *,
    pack_id: int,
    draft_id: int,
    expression_key: str,
    source_filename: str,
    stored: Any,
    source_context: dict[str, Any] | None = None,
) -> VisualIdentityAssetResponse:
    asset = _repo(service).create_asset(
        owner_user_id=service.owner_user_id,
        pack_id=pack_id,
        draft_id=draft_id,
        expression_key=expression_key,
        original_expression_key=expression_key,
        display_label=display_label_for_expression_key(expression_key),
        source_filename=source_filename,
        storage_relpath=stored.relpath,
        content_type=stored.content_type,
        bytes=stored.bytes,
        sha256=stored.sha256,
        width=stored.width,
        height=stored.height,
        is_animated=stored.is_animated,
        frame_count=stored.frame_count,
        duration_ms=stored.duration_ms,
        preview_relpath=stored.preview_relpath,
        source_context=source_context or {},
    )
    return _asset_response(asset)


@router.get(
    "/capabilities",
    response_model=VisualIdentityCapabilitiesResponse,
    dependencies=[_READ_LIMIT],
)
async def get_visual_identity_capabilities(
    current_user: User = Depends(get_request_user),
) -> VisualIdentityCapabilitiesResponse:
    _current_user_id(current_user)
    return VisualIdentityCapabilitiesResponse.model_validate(build_visual_identity_capabilities())


@router.get(
    "/expression-slots",
    response_model=list[VisualIdentityExpressionSlotResponse],
    dependencies=[_READ_LIMIT],
)
async def list_visual_identity_expression_slots(
    current_user: User = Depends(get_request_user),
) -> list[VisualIdentityExpressionSlotResponse]:
    _current_user_id(current_user)
    aliases_by_slot: dict[str, list[str]] = {slot: [] for slot in CANONICAL_EXPRESSION_SLOTS}
    for alias, canonical in EXPRESSION_ALIASES.items():
        aliases_by_slot.setdefault(canonical, []).append(alias)
    return [
        VisualIdentityExpressionSlotResponse(
            key=slot,
            label=display_label_for_expression_key(slot),
            canonical=True,
            aliases=sorted(aliases_by_slot.get(slot, [])),
        )
        for slot in CANONICAL_EXPRESSION_SLOTS
    ]


@router.get("/packs", response_model=list[VisualIdentityPackResponse], dependencies=[_READ_LIMIT])
async def list_visual_identity_packs(
    status_filter: str | None = Query(default=None, alias="status"),
    service: VisualIdentityService = Depends(_service),
) -> list[VisualIdentityPackResponse]:
    packs = _repo(service).list_packs(owner_user_id=service.owner_user_id, status=status_filter)
    return [_pack_response(pack) for pack in packs]


@router.post(
    "/packs",
    response_model=VisualIdentityPackResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_WRITE_LIMIT],
)
async def create_visual_identity_pack(
    request: VisualIdentityPackCreate,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityPackResponse:
    default_expression_key = _normalize_expression_or_422(
        request.default_expression_key,
        field_name="default_expression_key",
    )
    try:
        pack = service.create_pack(
            title=request.title,
            description=request.description,
            default_expression_key=default_expression_key,
            source_kind=request.source_kind,
            source_context=request.source_context,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return _pack_response(pack)


@router.get(
    "/packs/{pack_id}",
    response_model=VisualIdentityPackResponse,
    dependencies=[_READ_LIMIT],
)
async def get_visual_identity_pack(
    pack_id: int,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityPackResponse:
    return _pack_response(_require_pack(service, pack_id))


@router.patch(
    "/packs/{pack_id}",
    response_model=VisualIdentityPackResponse,
    dependencies=[_WRITE_LIMIT],
)
async def update_visual_identity_pack(
    pack_id: int,
    request: VisualIdentityPackUpdate,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityPackResponse:
    _require_pack(service, pack_id)
    fields = request.model_dump(exclude_none=True)
    if "default_expression_key" in fields:
        fields["default_expression_key"] = _normalize_expression_or_422(
            str(fields["default_expression_key"]),
            field_name="default_expression_key",
        )
    try:
        pack = _repo(service).update_pack(
            pack_id=pack_id,
            owner_user_id=service.owner_user_id,
            fields=fields,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="visual_identity_pack_not_found")
    return _pack_response(pack)


@router.delete(
    "/packs/{pack_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[_DELETE_LIMIT],
)
async def delete_visual_identity_pack(
    pack_id: int,
    service: VisualIdentityService = Depends(_service),
) -> Response:
    try:
        _repo(service).mark_pack_deleted(pack_id=pack_id, owner_user_id=service.owner_user_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/packs/{pack_id}/assets",
    response_model=VisualIdentityAssetResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_WRITE_LIMIT],
)
async def upload_visual_identity_pack_asset(
    pack_id: int,
    expression_key: str = Form(...),
    draft_id: int | None = Form(default=None),
    file: UploadFile = File(...),
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityAssetResponse:
    pack = _require_pack(service, pack_id)
    normalized_expression = _normalize_expression_or_422(expression_key)
    draft = _draft_for_asset_upload(service, pack, draft_id=draft_id)
    temp_path, _ = await _upload_to_temp_file(file, max_bytes=MAX_EXPRESSION_ASSET_BYTES)
    try:
        stored = validate_and_store_visual_identity_asset(
            source_path=temp_path,
            owner_user_id=service.owner_user_id,
            expression_key=normalized_expression,
            content_type=file.content_type,
            pack_id=f"draft-{draft['id']}",
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return _create_asset_from_stored_metadata(
        service,
        pack_id=int(pack["id"]),
        draft_id=int(draft["id"]),
        expression_key=normalized_expression,
        source_filename=_safe_upload_filename(file, fallback=f"{normalized_expression}.png"),
        stored=stored,
    )


@router.post(
    "/packs/{pack_id}/assets/from-generated-file",
    response_model=VisualIdentityAssetResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_WRITE_LIMIT],
)
async def create_visual_identity_asset_from_generated_file(
    pack_id: int,
    request: VisualIdentityGeneratedFileAssetRequest,
    service: VisualIdentityService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
) -> VisualIdentityAssetResponse:
    pack = _require_pack(service, pack_id)
    normalized_expression = _normalize_expression_or_422(request.expression_key)
    normalized_source_feature = str(request.source_feature or "").strip().lower()
    is_vn_source = normalized_source_feature == SOURCE_FEATURE_VN_ASSETS
    source_feature = SOURCE_FEATURE_VN_ASSETS if is_vn_source else request.source_feature
    try:
        idempotency_source_context = canonicalize_source_context(request.source_context)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc

    scope = "visual_identity_generated_file_asset"
    resource_id = f"pack:{pack_id}:generated-file-asset"
    payload_hash = _canonical_payload_hash(
        {
            "pack_id": pack_id,
            "generated_file_id": request.generated_file_id,
            "expression_key": normalized_expression,
            "draft_id": request.draft_id,
            "source_feature": source_feature,
            "source_context": idempotency_source_context,
        }
    )
    try:
        idempotency_claim = service.claim_or_replay_idempotency(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=request.idempotency_key,
            payload_hash=payload_hash,
        )
    except VisualIdentityServiceError as exc:
        raise _handle_idempotency_service_error(exc, scope=scope, resource_id=resource_id) from exc
    if idempotency_claim.replay_response is not None:
        return VisualIdentityAssetResponse.model_validate(idempotency_claim.replay_response)
    claim_token = idempotency_claim.claim_token
    if claim_token is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="idempotency_claim_missing")

    try:
        generated_file = await files_repo.get_file_by_id(request.generated_file_id)
        if not generated_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="generated_file_not_found",
            )
        if is_vn_source:
            validate_generated_file_record_for_expression_asset(
                owner_user_id=service.owner_user_id,
                generated_file_record=generated_file,
                source_feature=source_feature,
            )
            canonical_context = build_vn_visual_identity_source_context(
                user_id=service.owner_user_id,
                vn_repository=VNAssetPacksRepository.initialized(service.db),
                generated_file_record=generated_file,
                requested_context=request.source_context,
            )
        else:
            validate_generated_file_record_for_expression_asset(
                owner_user_id=service.owner_user_id,
                generated_file_record=generated_file,
                source_feature=source_feature,
            )
            canonical_context = idempotency_source_context
        draft = _draft_for_asset_upload(service, pack, draft_id=request.draft_id)
        stored = copy_generated_file_record_to_expression_asset(
            owner_user_id=service.owner_user_id,
            pack_id=f"draft-{draft['id']}",
            expression_key=normalized_expression,
            generated_file_record=generated_file,
            source_feature=source_feature,
        )
        response = _create_asset_from_stored_metadata(
            service,
            pack_id=int(pack["id"]),
            draft_id=int(draft["id"]),
            expression_key=normalized_expression,
            source_filename=str(generated_file.get("original_filename") or generated_file.get("filename") or ""),
            stored=stored,
            source_context=canonical_context,
        )
    except ValueError as exc:
        service.release_idempotency_claim(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=request.idempotency_key,
            claim_token=claim_token,
        )
        raise _handle_value_error(exc) from exc
    except HTTPException:
        service.release_idempotency_claim(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=request.idempotency_key,
            claim_token=claim_token,
        )
        raise
    except Exception:
        service.release_idempotency_claim(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=request.idempotency_key,
            claim_token=claim_token,
        )
        raise
    try:
        service.record_idempotency_response(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=request.idempotency_key,
            payload_hash=payload_hash,
            response=response.model_dump(mode="json"),
            claim_token=claim_token,
        )
    except VisualIdentityServiceError as exc:
        raise _handle_idempotency_service_error(exc, scope=scope, resource_id=resource_id) from exc
    return response


@router.get(
    "/packs/{pack_id}/assets/{asset_id}/content",
    dependencies=[_READ_LIMIT],
)
async def get_visual_identity_asset_content(
    pack_id: int,
    asset_id: int,
    service: VisualIdentityService = Depends(_service),
) -> FileResponse:
    _require_pack(service, pack_id)
    asset = _require_asset_for_pack(service, pack_id=pack_id, asset_id=asset_id)
    content_type = str(asset.get("content_type") or "").split(";", 1)[0].strip().lower()
    if content_type not in supported_visual_identity_mime_types():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unsupported_mime_type",
        )
    try:
        asset_path = resolve_visual_identity_asset_path(
            owner_user_id=service.owner_user_id,
            relpath=str(asset["storage_relpath"]),
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if not asset_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_asset_content_not_found",
        )
    return FileResponse(
        asset_path,
        media_type=content_type,
        filename=Path(str(asset.get("source_filename") or asset_path.name)).name,
        headers={"Cache-Control": _IMMUTABLE_ASSET_CACHE_CONTROL},
    )


@router.get(
    "/packs/{pack_id}/assets/{asset_id}/preview",
    dependencies=[_READ_LIMIT],
)
async def get_visual_identity_asset_preview(
    pack_id: int,
    asset_id: int,
    service: VisualIdentityService = Depends(_service),
) -> FileResponse:
    _require_pack(service, pack_id)
    asset = _require_asset_for_pack(service, pack_id=pack_id, asset_id=asset_id)
    preview_relpath = str(asset.get("preview_relpath") or "")
    if not preview_relpath:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_asset_preview_not_found",
        )
    try:
        preview_path = resolve_visual_identity_asset_path(
            owner_user_id=service.owner_user_id,
            relpath=preview_relpath,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    if not preview_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_asset_preview_not_found",
        )
    return FileResponse(
        preview_path,
        media_type="image/png",
        filename=preview_path.name,
        headers={"Cache-Control": _IMMUTABLE_ASSET_CACHE_CONTROL},
    )


@router.post(
    "/imports/zip",
    response_model=VisualIdentityImportZipStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[_WRITE_LIMIT],
)
async def start_visual_identity_zip_import(
    archive: UploadFile = File(...),
    title: str = Form(default="Imported Expression Pack"),
    pack_id: int | None = Form(default=None),
    idempotency_key: str = Form(..., min_length=1, max_length=160),
    service: VisualIdentityService = Depends(_service),
    jobs_manager: JobManager = Depends(_job_manager),
) -> VisualIdentityImportZipStartResponse:
    if not str(archive.filename or "").lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_zip_archive",
        )
    if pack_id is not None:
        _require_pack(service, pack_id)
    operation_idempotency_key = _require_idempotency_key(idempotency_key)
    source_filename = _safe_upload_filename(archive, fallback="expressions.zip")
    temp_path, byte_count = await _upload_to_temp_file(archive, max_bytes=MAX_EXPRESSION_ZIP_BYTES)
    file_sha256 = await asyncio.to_thread(_sha256_file, temp_path)
    scope = "visual_identity_zip_import"
    resource_id = f"pack:{pack_id}" if pack_id is not None else "pack:new"
    payload_hash = _canonical_payload_hash(
        {
            "pack_id": pack_id,
            "title": title,
            "source_filename": source_filename,
            "file_sha256": file_sha256,
            "file_size": byte_count,
        }
    )
    try:
        idempotency_claim = service.claim_or_replay_idempotency(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=operation_idempotency_key,
            payload_hash=payload_hash,
        )
        if idempotency_claim.replay_response is not None:
            temp_path.unlink(missing_ok=True)
            return VisualIdentityImportZipStartResponse.model_validate(
                idempotency_claim.replay_response
            )
        claim_token = idempotency_claim.claim_token
        if claim_token is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="idempotency_claim_missing")
    except VisualIdentityServiceError as exc:
        temp_path.unlink(missing_ok=True)
        raise _handle_idempotency_service_error(exc, scope=scope, resource_id=resource_id) from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    upload_path: Path | None = None
    try:
        draft = _repo(service).create_draft(
            owner_user_id=service.owner_user_id,
            pack_id=pack_id,
            title=title,
            source_kind="zip",
            source_filename=source_filename,
            status="importing",
        )
        upload_dir = DatabasePaths.get_user_visual_identities_dir(service.owner_user_id) / "imports" / str(draft["id"])
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / source_filename
        temp_path.replace(upload_path)
        job = create_visual_identity_import_zip_job(
            jobs_manager,
            owner_user_id=service.owner_user_id,
            draft_id=int(draft["id"]),
            upload_path=str(upload_path),
            source_filename=str(draft["source_filename"]),
        )
        job_id = job.get("job_id") or job.get("id")
        if job_id is not None:
            draft = _repo(service).set_draft_import_job_id(
                draft_id=int(draft["id"]),
                owner_user_id=service.owner_user_id,
                import_job_id=str(job_id),
            )
        response = VisualIdentityImportZipStartResponse(
            draft_id=int(draft["id"]),
            job_id=job_id,
            status="queued",
            source_filename=str(draft["source_filename"]),
            import_job_id=str(job_id) if job_id is not None else None,
        )
    except ValueError as exc:
        service.release_idempotency_claim(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=operation_idempotency_key,
            claim_token=claim_token,
        )
        temp_path.unlink(missing_ok=True)
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
        raise _handle_value_error(exc) from exc
    except Exception:
        service.release_idempotency_claim(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=operation_idempotency_key,
            claim_token=claim_token,
        )
        temp_path.unlink(missing_ok=True)
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
        raise
    try:
        service.record_idempotency_response(
            scope=scope,
            resource_id=resource_id,
            idempotency_key=operation_idempotency_key,
            payload_hash=payload_hash,
            response=response.model_dump(mode="json"),
            claim_token=claim_token,
        )
    except VisualIdentityServiceError as exc:
        raise _handle_idempotency_service_error(exc, scope=scope, resource_id=resource_id) from exc
    return response


@router.get(
    "/drafts/{draft_id}",
    response_model=VisualIdentityDraftResponse,
    dependencies=[_READ_LIMIT],
)
async def get_visual_identity_draft(
    draft_id: int,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityDraftResponse:
    return _draft_response(service, _require_draft(service, draft_id))


@router.patch(
    "/drafts/{draft_id}/slots/{slot_key}",
    response_model=VisualIdentityDraftResponse,
    dependencies=[_WRITE_LIMIT],
)
async def update_visual_identity_draft_slot(
    draft_id: int,
    slot_key: str,
    request: VisualIdentityDraftSlotUpdate,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityDraftResponse:
    draft = _require_draft(service, draft_id)
    normalized_slot = _normalize_expression_or_422(slot_key, field_name="slot_key")
    normalized_expression = (
        _normalize_expression_or_422(request.expression_key)
        if request.expression_key is not None
        else normalized_slot
    )
    if request.asset_id is not None:
        asset = _repo(service).get_asset(request.asset_id, owner_user_id=service.owner_user_id)
        if asset is None or asset.get("draft_id") is None or int(asset["draft_id"]) != int(draft_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="visual_identity_asset_not_found",
            )
    slot_map = _json_mapping(draft, "slot_map_json")
    slot_map[normalized_slot] = {
        "expression_key": normalized_expression,
        "asset_id": request.asset_id,
        "display_label": request.display_label
        if request.display_label is not None
        else display_label_for_expression_key(normalized_expression),
        "metadata": request.metadata,
    }
    try:
        updated = _repo(service).update_draft_slot_map(
            draft_id=draft_id,
            owner_user_id=service.owner_user_id,
            slot_map=slot_map,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return _draft_response(service, updated)


@router.post(
    "/drafts/{draft_id}/activate",
    response_model=VisualIdentityDraftResponse,
    dependencies=[_WRITE_LIMIT],
)
async def activate_visual_identity_draft(
    draft_id: int,
    request: VisualIdentityDraftActivateRequest,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityDraftResponse:
    try:
        activation = service.activate_draft(
            draft_id=draft_id,
            actor_kind=request.actor_kind,
            actor_id=request.actor_id,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    draft = _require_draft(service, draft_id)
    return _draft_response(
        service,
        draft,
        pack_version_id=activation.pack_version_id,
        asset_ids=list(activation.asset_ids),
        binding_id=activation.binding_id,
    )


@router.post(
    "/bindings",
    response_model=VisualIdentityBindingResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_WRITE_LIMIT],
)
async def create_visual_identity_binding(
    request: VisualIdentityBindingRequest,
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityBindingResponse:
    pack = _require_pack(service, request.pack_id)
    active_version_id = request.active_version_id
    if active_version_id is None:
        if pack.get("active_version_id") is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="visual_identity_pack_active_version_required",
            )
        active_version_id = int(pack["active_version_id"])
    try:
        actor_id = service._validate_actor_for_binding(request.actor_kind, request.actor_id)
        binding = _repo(service).upsert_binding(
            owner_user_id=service.owner_user_id,
            actor_kind=request.actor_kind,
            actor_id=actor_id,
            pack_id=request.pack_id,
            active_version_id=active_version_id,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return _binding_response(binding)


@router.delete(
    "/bindings/{binding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[_DELETE_LIMIT],
)
async def delete_visual_identity_binding(
    binding_id: int,
    service: VisualIdentityService = Depends(_service),
) -> Response:
    binding = _repo(service).delete_binding(binding_id, owner_user_id=service.owner_user_id)
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="visual_identity_binding_not_found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/bindings/resolve",
    response_model=VisualIdentityResolveResponse,
    dependencies=[_READ_LIMIT],
)
async def resolve_visual_identity_binding(
    actor_kind: ActorKind = Query(...),
    actor_id: int | str = Query(...),
    expression_key: str = Query("neutral"),
    manual_override_expression_key: str | None = Query(default=None),
    mood_expression_key: str | None = Query(default=None),
    role_id: str | None = Query(default=None),
    role_label: str | None = Query(default=None),
    override_pack_id: int | None = Query(default=None, ge=1),
    override_pack_version_id: int | None = Query(default=None, ge=1),
    allow_override_fallback: bool = Query(default=False),
    service: VisualIdentityService = Depends(_service),
) -> VisualIdentityResolveResponse:
    normalized_expression = _normalize_expression_or_422(expression_key)
    try:
        resolved = service.resolve_expression_asset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_expression_key=normalized_expression,
            manual_override_expression_key=manual_override_expression_key,
            mood_expression_key=mood_expression_key,
            role_id=role_id,
            role_label=role_label,
            override_pack_id=override_pack_id,
            override_pack_version_id=override_pack_version_id,
            allow_override_fallback=allow_override_fallback,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    asset_url = resolved.asset_url
    if asset_url is None and resolved.pack_id is not None and resolved.asset_id is not None:
        asset_url = _asset_content_url(pack_id=resolved.pack_id, asset_id=resolved.asset_id)
    preview_url = None
    if (
        resolved.preview_relpath
        and resolved.pack_id is not None
        and resolved.asset_id is not None
    ):
        preview_url = _asset_preview_url(pack_id=resolved.pack_id, asset_id=resolved.asset_id)
    return VisualIdentityResolveResponse(
        actor_kind=resolved.actor_kind,  # type: ignore[arg-type]
        actor_id=resolved.actor_id,
        pack_id=resolved.pack_id,
        pack_version_id=resolved.pack_version_id,
        expression_key=resolved.expression_key,
        requested_expression_key=resolved.requested_expression_key,
        asset_id=resolved.asset_id,
        storage_relpath=resolved.storage_relpath,
        fallback_reason=_api_fallback_reason(resolved.fallback_reason),
        is_animated=resolved.is_animated,
        content_type=resolved.content_type,
        asset_url=asset_url,
        preview_url=preview_url,
        role_id=resolved.role_id,
        role_label=resolved.role_label,
        resolution_source=resolved.resolution_source,
    )
