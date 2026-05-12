"""VN scripts authoring endpoints."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.api.v1.schemas.vn_script_schemas import (
    VNScriptAuthoringCatalogResponse,
    VNScriptCreate,
    VNScriptCreateFromTemplateRequest,
    VNScriptCreateFromTemplateResponse,
    VNScriptDiagnosticsResponse,
    VNScriptDraftPutRequest,
    VNScriptDraftResponse,
    VNScriptListResponse,
    VNScriptManifestSnapshotResponse,
    VNScriptPatch,
    VNScriptPublishRequest,
    VNScriptPublishResponse,
    VNScriptResponse,
    VNScriptSnippetApplyRequest,
    VNScriptSnippetApplyResponse,
    VNScriptSnippetPreviewRequest,
    VNScriptSnippetPreviewResponse,
    VNScriptTemplateListResponse,
    VNScriptTemplateSummary,
    VNScriptValidateRequest,
    VNScriptValidationResponse,
    VNScriptVersionListResponse,
    VNScriptVersionPolicyEvaluateRequest,
    VNScriptVersionPolicyEvaluateResponse,
    VNScriptVersionResponse,
)
from tldw_Server_API.app.core.VN_Scripts.templates import get_template, instantiate_template
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import VNPolicyProfileStore
from tldw_Server_API.app.core.VN_Platform.errors import (
    ERROR_INVALID_REQUEST,
    ERROR_NOT_FOUND,
    vn_error_detail,
)
from tldw_Server_API.app.core.VN_Scripts.authoring_errors import VNScriptAuthoringError
from tldw_Server_API.app.core.VN_Scripts.service import VNScriptService
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

router = APIRouter(prefix="/vn-scripts", tags=["vn-scripts"])
_GENERATION_PROFILE_KEY_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_MAX_GENERATION_PROFILE_MAP_SIZE = 16
_MAX_GENERATION_PROFILE_ID_LENGTH = 80


def _current_user_id(current_user: User) -> int:
    user_id = current_user.id_int
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=vn_error_detail(ERROR_INVALID_REQUEST, "Invalid user id."),
        )
    return user_id


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> VNScriptService:
    return VNScriptService(db, owner_user_id=_current_user_id(current_user))


async def _profile_store(db_pool: DatabasePool = Depends(get_db_pool)) -> VNPolicyProfileStore:
    store = VNPolicyProfileStore(db_pool)
    await store.initialize()
    return store


async def _generated_files_repo() -> AuthnzGeneratedFilesRepo:
    storage_service = await get_storage_service()
    return await storage_service.get_generated_files_repo()


@router.post("/scripts", response_model=VNScriptResponse, status_code=status.HTTP_201_CREATED)
async def create_script(
    request: VNScriptCreate,
    service: VNScriptService = Depends(_service),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptResponse:
    """Create a VN script shell."""
    try:
        await _resolve_request_profiles(
            policy_profile_id=request.policy_profile_id,
            generation_profile_id=request.generation_profile_id,
            generation_profiles=request.generation_profiles,
            profile_store=profile_store,
        )
        row = service.create_script(**request.model_dump(exclude={"generation_profile_ids"}))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNScriptResponse.model_validate(row)


@router.get("/templates", response_model=VNScriptTemplateListResponse)
async def list_templates(service: VNScriptService = Depends(_service)) -> VNScriptTemplateListResponse:
    """List preview-safe VN script starter templates."""
    return VNScriptTemplateListResponse(
        items=[VNScriptTemplateSummary.model_validate(item) for item in service.list_templates()]
    )


@router.get("/vn-authoring-catalog", response_model=VNScriptAuthoringCatalogResponse)
async def get_authoring_catalog(service: VNScriptService = Depends(_service)) -> VNScriptAuthoringCatalogResponse:
    """Return preview-safe VN script operation and snippet metadata."""
    return VNScriptAuthoringCatalogResponse.model_validate(service.get_authoring_catalog())


@router.post(
    "/templates/{template_id}/scripts",
    response_model=VNScriptCreateFromTemplateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_script_from_template(
    template_id: str,
    request: VNScriptCreateFromTemplateRequest,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptCreateFromTemplateResponse:
    """Create a normal VN script and store a validated starter-template draft."""
    try:
        template = get_template(template_id)
        title = request.title or template.default_title
        description = request.description if request.description is not None else template.default_description
        policy_profile, generation_profile, generation_profiles = await _resolve_request_profiles(
            policy_profile_id=request.policy_profile_id,
            generation_profile_id=request.generation_profile_id,
            generation_profiles=request.generation_profiles,
            profile_store=profile_store,
        )
        draft = instantiate_template(
            template_id,
            title=title,
            primary_asset_pack_id=request.primary_asset_pack_id,
            generation_profile_id=request.generation_profile_id,
        )
        audio_refs = await _resolve_accessible_audio_refs(
            draft,
            files_repo=files_repo,
            owner_user_id=service.owner_user_id,
        )
        draft_preview = service.create_script_from_template(
            template_id,
            title=title,
            description=description,
            primary_asset_pack_id=request.primary_asset_pack_id,
            policy_profile_id=request.policy_profile_id,
            generation_profile_id=request.generation_profile_id,
            generation_profiles=request.generation_profiles,
            content_rating=request.content_rating,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            resolved_generation_profiles=generation_profiles,
            audio_refs=audio_refs,
            draft=draft,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNScriptCreateFromTemplateResponse.model_validate(draft_preview)


@router.get("/scripts", response_model=VNScriptListResponse)
async def list_scripts(
    limit: int = 50,
    offset: int = 0,
    service: VNScriptService = Depends(_service),
) -> VNScriptListResponse:
    """List owned VN scripts."""
    bounded_limit, bounded_offset = _bounded_pagination(limit, offset)
    rows, total = service.list_scripts(limit=bounded_limit, offset=bounded_offset)
    return VNScriptListResponse(
        items=[VNScriptResponse.model_validate(row) for row in rows],
        **_pagination_payload(limit=bounded_limit, offset=bounded_offset, total=total),
    )


@router.get("/scripts/{script_id}", response_model=VNScriptResponse)
async def get_script(script_id: int, service: VNScriptService = Depends(_service)) -> VNScriptResponse:
    """Read script metadata."""
    try:
        return VNScriptResponse.model_validate(service.get_script(script_id))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.patch("/scripts/{script_id}", response_model=VNScriptResponse)
async def patch_script(
    script_id: int,
    request: VNScriptPatch,
    service: VNScriptService = Depends(_service),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptResponse:
    """Patch script metadata."""
    try:
        fields = request.model_dump(exclude_unset=True, exclude={"generation_profile_ids"})
        await _resolve_patch_profiles(fields, profile_store=profile_store)
        return VNScriptResponse.model_validate(service.update_script(script_id, fields))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.delete("/scripts/{script_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_script(script_id: int, service: VNScriptService = Depends(_service)) -> Response:
    """Soft-delete a script."""
    try:
        service.delete_script(script_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/scripts/{script_id}/draft", response_model=VNScriptDraftResponse)
async def get_draft(script_id: int, service: VNScriptService = Depends(_service)) -> VNScriptDraftResponse:
    """Read mutable draft."""
    try:
        return VNScriptDraftResponse.model_validate(service.get_draft(script_id))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.put("/scripts/{script_id}/draft", response_model=VNScriptDraftResponse)
async def put_draft(
    script_id: int,
    request: VNScriptDraftPutRequest,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptDraftResponse:
    """Replace whole draft with optimistic revision control."""
    try:
        policy_profile, generation_profile, generation_profiles = await _resolve_script_profiles(
            service.get_script(script_id),
            profile_store=profile_store,
        )
        audio_refs = await _resolve_accessible_audio_refs(
            request.draft,
            files_repo=files_repo,
            owner_user_id=service.owner_user_id,
        )
        return VNScriptDraftResponse.model_validate(
            service.replace_draft(
                script_id,
                if_revision=request.if_revision,
                draft=request.draft,
                audio_refs=audio_refs,
                policy_profile=policy_profile,
                generation_profile=generation_profile,
                generation_profiles=generation_profiles,
            )
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/scripts/{script_id}/draft/validate", response_model=VNScriptValidationResponse)
async def validate_draft(
    script_id: int,
    request: VNScriptValidateRequest,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptValidationResponse:
    """Validate a draft without publishing."""
    try:
        policy_profile, generation_profile, generation_profiles = await _resolve_script_profiles(
            service.get_script(script_id),
            profile_store=profile_store,
        )
        draft = request.draft if request.draft is not None else service.get_draft(script_id)["draft"]
        audio_refs = await _resolve_accessible_audio_refs(
            draft,
            files_repo=files_repo,
            owner_user_id=service.owner_user_id,
        )
        return VNScriptValidationResponse.model_validate(
            service.validate_draft(
                script_id,
                draft,
                audio_refs=audio_refs,
                policy_profile=policy_profile,
                generation_profile=generation_profile,
                generation_profiles=generation_profiles,
            )
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post("/scripts/{script_id}/draft/snippet-preview", response_model=VNScriptSnippetPreviewResponse)
async def preview_snippet(
    script_id: int,
    request: VNScriptSnippetPreviewRequest,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptSnippetPreviewResponse:
    """Preview a snippet patch without mutating stored draft or diagnostics."""
    try:
        build = service.build_snippet_patch(
            script_id,
            request.snippet_id,
            request.anchor.model_dump(exclude_none=True),
            request.parameters,
            draft=request.draft,
            draft_revision=request.draft_revision,
        )
        policy_profile, generation_profile, generation_profiles = await _resolve_script_profiles(
            build["script"],
            profile_store=profile_store,
        )
        patched_draft = build["patch"].draft
        audio_refs = await _resolve_accessible_audio_refs(
            patched_draft,
            files_repo=files_repo,
            owner_user_id=service.owner_user_id,
        )
        return VNScriptSnippetPreviewResponse.model_validate(
            service.preview_snippet_patch(
                script_id,
                request.snippet_id,
                build["base_revision"],
                build["patch"],
                audio_refs=audio_refs,
                policy_profile=policy_profile,
                generation_profile=generation_profile,
                generation_profiles=generation_profiles,
            )
        )
    except VNScriptAuthoringError as exc:
        raise _handle_authoring_error(exc) from exc
    except ValueError as exc:
        raise _handle_value_error(exc, service=service, script_id=script_id) from exc


@router.post("/scripts/{script_id}/draft/snippet-apply", response_model=VNScriptSnippetApplyResponse)
async def apply_snippet(
    script_id: int,
    request: VNScriptSnippetApplyRequest,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptSnippetApplyResponse:
    """Apply a snippet patch to the stored draft using optimistic revision control."""
    try:
        build = service.build_snippet_patch(
            script_id,
            request.snippet_id,
            request.anchor.model_dump(exclude_none=True),
            request.parameters,
            if_revision=request.if_revision,
        )
        policy_profile, generation_profile, generation_profiles = await _resolve_script_profiles(
            build["script"],
            profile_store=profile_store,
        )
        patched_draft = build["patch"].draft
        audio_refs = await _resolve_accessible_audio_refs(
            patched_draft,
            files_repo=files_repo,
            owner_user_id=service.owner_user_id,
        )
        return VNScriptSnippetApplyResponse.model_validate(
            service.apply_snippet_patch_result(
                script_id,
                request.snippet_id,
                request.if_revision,
                build["patch"],
                audio_refs=audio_refs,
                policy_profile=policy_profile,
                generation_profile=generation_profile,
                generation_profiles=generation_profiles,
            )
        )
    except VNScriptAuthoringError as exc:
        raise _handle_authoring_error(exc) from exc
    except ValueError as exc:
        raise _handle_value_error(exc, service=service, script_id=script_id) from exc


@router.get("/scripts/{script_id}/draft/diagnostics", response_model=VNScriptDiagnosticsResponse)
async def get_diagnostics(
    script_id: int,
    service: VNScriptService = Depends(_service),
) -> VNScriptDiagnosticsResponse:
    """Read current author diagnostics."""
    try:
        draft = service.get_draft(script_id)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNScriptDiagnosticsResponse(
        script_id=int(draft["script_id"]),
        revision=int(draft["revision"]),
        diagnostics=dict(draft["diagnostics"]),
    )


@router.post("/scripts/{script_id}/publish", response_model=VNScriptPublishResponse, status_code=status.HTTP_201_CREATED)
async def publish_script(
    script_id: int,
    request: VNScriptPublishRequest,
    api_response: Response,
    service: VNScriptService = Depends(_service),
    files_repo: AuthnzGeneratedFilesRepo = Depends(_generated_files_repo),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNScriptPublishResponse:
    """Validate and publish an immutable script version."""
    try:
        existing = service.get_publish_request_by_key(
            script_id=script_id,
            idempotency_key=request.idempotency_key,
        )
        audio_refs = None
        policy_profile = None
        generation_profile = None
        generation_profiles = None
        if existing is None:
            script = service.get_script(script_id)
            policy_profile, generation_profile, generation_profiles = await _resolve_script_profiles(
                script,
                profile_store=profile_store,
            )
            draft = service.get_draft(script_id)["draft"]
            audio_refs = await _resolve_accessible_audio_refs(
                draft,
                files_repo=files_repo,
                owner_user_id=service.owner_user_id,
            )
        row = service.publish_script(
            script_id,
            draft_revision=request.draft_revision,
            label=request.label,
            idempotency_key=request.idempotency_key,
            acknowledgements=request.acknowledgements,
            audio_refs=audio_refs,
            policy_profile=policy_profile,
            generation_profile=generation_profile,
            generation_profiles=generation_profiles,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    response = VNScriptPublishResponse.model_validate(row)
    if existing is not None:
        api_response.status_code = status.HTTP_200_OK
    return response


@router.get("/scripts/{script_id}/versions", response_model=VNScriptVersionListResponse)
async def list_versions(
    script_id: int,
    limit: int = 50,
    offset: int = 0,
    service: VNScriptService = Depends(_service),
) -> VNScriptVersionListResponse:
    """List published script versions."""
    try:
        bounded_limit, bounded_offset = _bounded_pagination(limit, offset)
        rows, total = service.list_versions(script_id, limit=bounded_limit, offset=bounded_offset)
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNScriptVersionListResponse(
        items=[VNScriptVersionResponse.model_validate(row) for row in rows],
        **_pagination_payload(limit=bounded_limit, offset=bounded_offset, total=total),
    )


@router.get("/scripts/{script_id}/versions/{version_id}", response_model=VNScriptVersionResponse)
async def get_version(
    script_id: int,
    version_id: int,
    service: VNScriptService = Depends(_service),
) -> VNScriptVersionResponse:
    """Read immutable script version."""
    try:
        return VNScriptVersionResponse.model_validate(service.get_version(script_id, version_id))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.get(
    "/scripts/{script_id}/versions/{version_id}/manifest-snapshot",
    response_model=VNScriptManifestSnapshotResponse,
)
async def get_manifest_snapshot(
    script_id: int,
    version_id: int,
    service: VNScriptService = Depends(_service),
) -> VNScriptManifestSnapshotResponse:
    """Inspect pinned manifest snapshot."""
    try:
        return VNScriptManifestSnapshotResponse.model_validate(service.get_manifest_snapshot(script_id, version_id))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


@router.post(
    "/scripts/{script_id}/versions/{version_id}/policy/evaluate",
    response_model=VNScriptVersionPolicyEvaluateResponse,
)
async def evaluate_version_policy(
    script_id: int,
    version_id: int,
    request: VNScriptVersionPolicyEvaluateRequest,
    service: VNScriptService = Depends(_service),
) -> VNScriptVersionPolicyEvaluateResponse:
    """Preflight a published script version."""
    try:
        return VNScriptVersionPolicyEvaluateResponse.model_validate(
            service.evaluate_version_policy(script_id, version_id, context=request.context)
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc


def _bounded_pagination(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(int(limit), 100)), max(0, int(offset))


def _pagination_payload(*, limit: int, offset: int, total: int) -> dict[str, Any]:
    next_offset = offset + limit
    has_more = next_offset < total
    return {
        "limit": limit,
        "offset": offset,
        "total": total,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "pagination": OffsetPaginationMeta(
            limit=limit,
            offset=offset,
            total=total,
            has_more=has_more,
            next_offset=next_offset if has_more else None,
        ),
    }


async def _resolve_request_profiles(
    *,
    policy_profile_id: str,
    generation_profile_id: str,
    generation_profiles: Mapping[str, str] | None = None,
    profile_store: VNPolicyProfileStore,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    policy_profile = await profile_store.get_policy_profile(policy_profile_id)
    generation_profile = await profile_store.get_generation_profile(generation_profile_id)
    if policy_profile is None:
        raise ValueError("policy_profile_not_found")
    if generation_profile is None:
        raise ValueError("generation_profile_not_found")
    _validate_generation_profile_map_shape(generation_profiles)
    resolved_generation_profiles: dict[str, dict[str, Any]] = {"default": generation_profile}
    for profile_key, profile_id in (generation_profiles or {}).items():
        if profile_key == "default":
            raise ValueError("generation_profile_default_reserved")
        profile = await profile_store.get_generation_profile(str(profile_id))
        if profile is None:
            raise ValueError("generation_profile_not_found")
        resolved_generation_profiles[str(profile_key)] = profile
    return policy_profile, generation_profile, resolved_generation_profiles


async def _resolve_patch_profiles(
    fields: Mapping[str, Any],
    *,
    profile_store: VNPolicyProfileStore,
) -> None:
    policy_profile_id = fields.get("policy_profile_id")
    generation_profile_id = fields.get("generation_profile_id")
    generation_profiles = fields.get("generation_profiles")
    if isinstance(policy_profile_id, str) and await profile_store.get_policy_profile(policy_profile_id) is None:
        raise ValueError("policy_profile_not_found")
    if (
        isinstance(generation_profile_id, str)
        and await profile_store.get_generation_profile(generation_profile_id) is None
    ):
        raise ValueError("generation_profile_not_found")
    if isinstance(generation_profiles, Mapping):
        _validate_generation_profile_map_shape(generation_profiles)
        for profile_key, profile_id in generation_profiles.items():
            if profile_key == "default":
                raise ValueError("generation_profile_default_reserved")
            if await profile_store.get_generation_profile(str(profile_id)) is None:
                raise ValueError("generation_profile_not_found")


async def _resolve_script_profiles(
    script: Mapping[str, Any],
    *,
    profile_store: VNPolicyProfileStore,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    return await _resolve_request_profiles(
        policy_profile_id=str(script["policy_profile_id"]),
        generation_profile_id=str(script["generation_profile_id"]),
        generation_profiles={
            key: value
            for key, value in dict(script.get("generation_profiles") or {}).items()
            if key != "default"
        },
        profile_store=profile_store,
    )


def _validate_generation_profile_map_shape(generation_profiles: Mapping[str, Any] | None) -> None:
    if not isinstance(generation_profiles, Mapping):
        return
    if len(generation_profiles) > _MAX_GENERATION_PROFILE_MAP_SIZE:
        raise ValueError("generation_profile_map_too_large")
    for profile_key, profile_id in generation_profiles.items():
        if profile_key == "default":
            raise ValueError("generation_profile_default_reserved")
        if not isinstance(profile_key, str) or not _GENERATION_PROFILE_KEY_RE.fullmatch(profile_key):
            raise ValueError("generation_profile_key_invalid")
        if not isinstance(profile_id, str) or not profile_id or len(profile_id) > _MAX_GENERATION_PROFILE_ID_LENGTH:
            raise ValueError("generation_profile_id_invalid")


async def _resolve_accessible_audio_refs(
    program: Mapping[str, Any],
    *,
    files_repo: AuthnzGeneratedFilesRepo,
    owner_user_id: int,
) -> dict[str, dict[str, Any]]:
    raw_refs = program.get("media_refs")
    if not isinstance(raw_refs, Mapping):
        return {}

    media_ref_ids: dict[str, int] = {}
    for media_ref, metadata in raw_refs.items():
        if not isinstance(media_ref, str) or not isinstance(metadata, Mapping):
            continue
        generated_file_id = metadata.get("generated_file_id")
        if isinstance(generated_file_id, int) and not isinstance(generated_file_id, bool):
            media_ref_ids[media_ref] = generated_file_id

    if not media_ref_ids:
        return {}

    records = await files_repo.get_files_by_ids(list(set(media_ref_ids.values())))
    records_by_id = {
        int(record["id"]): record
        for record in records
        if record.get("id") is not None
    }
    resolved: dict[str, dict[str, Any]] = {}
    for media_ref, generated_file_id in media_ref_ids.items():
        record = records_by_id.get(generated_file_id)
        if not _is_accessible_audio_record(record, owner_user_id=owner_user_id):
            continue
        resolved[media_ref] = {
            "generated_file_id": generated_file_id,
            "mime_type": str(record.get("mime_type") or ""),
            "owner_user_id": owner_user_id,
        }
    return resolved


def _is_accessible_audio_record(record: Mapping[str, Any] | None, *, owner_user_id: int) -> bool:
    if not isinstance(record, Mapping):
        return False
    if int(record.get("user_id") or -1) != int(owner_user_id):
        return False
    if bool(record.get("is_deleted")):
        return False
    return str(record.get("mime_type") or "").startswith("audio/")


def _handle_authoring_error(exc: VNScriptAuthoringError) -> HTTPException:
    status_code = status.HTTP_404_NOT_FOUND if exc.code == "snippet_not_found" else exc.status_code
    details = {"reason": exc.code, **dict(exc.details)}
    return HTTPException(
        status_code=status_code,
        detail=vn_error_detail(
            ERROR_NOT_FOUND if status_code == status.HTTP_404_NOT_FOUND else ERROR_INVALID_REQUEST,
            exc.code,
            details=details,
        ),
    )


def _handle_value_error(
    exc: ValueError,
    *,
    service: VNScriptService | None = None,
    script_id: int | None = None,
) -> HTTPException:
    reason = str(exc) or "invalid_request"
    if reason in {"script_not_found", "script_version_not_found", "template_not_found"}:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=vn_error_detail(ERROR_NOT_FOUND, reason, details={"reason": reason}),
        )
    if reason in {"draft_revision_conflict", "idempotency_key_conflict"}:
        details: dict[str, Any] = {"reason": reason}
        if reason == "draft_revision_conflict":
            current_revision = _current_draft_revision(service, script_id)
            if current_revision is not None:
                details["current_revision"] = current_revision
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=vn_error_detail(ERROR_INVALID_REQUEST, reason, details=details),
        )
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=vn_error_detail(ERROR_INVALID_REQUEST, reason, details={"reason": reason}),
    )


def _current_draft_revision(service: VNScriptService | None, script_id: int | None) -> int | None:
    if service is None or script_id is None:
        return None
    try:
        return int(service.get_draft(script_id)["revision"])
    except (TypeError, ValueError, KeyError):
        return None
