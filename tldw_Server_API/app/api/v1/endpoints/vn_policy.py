"""VN policy profile and preflight endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.api.v1.schemas.vn_policy_schemas import (
    VNGenerationProfileCreate,
    VNGenerationProfileListResponse,
    VNGenerationProfilePatch,
    VNGenerationProfileResponse,
    VNPolicyEvaluateRequest,
    VNPolicyEvaluateResponse,
    VNPolicyProfileCreate,
    VNPolicyProfileListResponse,
    VNPolicyProfilePatch,
    VNPolicyProfileResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import VNPolicyProfileStore
from tldw_Server_API.app.core.VN_Platform.errors import (
    ERROR_INVALID_REQUEST,
    ERROR_NOT_FOUND,
    ERROR_PERMISSION_DENIED,
    vn_error_detail,
)
from tldw_Server_API.app.core.VN_Policy.service import VNPolicyService

router = APIRouter(prefix="/vn-policy", tags=["vn-policy"])


def _current_user_id(current_user: User) -> int:
    user_id = current_user.id_int
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=vn_error_detail(ERROR_INVALID_REQUEST, "Invalid user id."),
        )
    return user_id


async def _profile_store(db_pool: DatabasePool = Depends(get_db_pool)) -> VNPolicyProfileStore:
    store = VNPolicyProfileStore(db_pool)
    await store.initialize()
    return store


def _service(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
    profile_store: VNPolicyProfileStore = Depends(_profile_store),
) -> VNPolicyService:
    return VNPolicyService(db, owner_user_id=_current_user_id(current_user), profile_store=profile_store)


def _admin_user(current_user: User = Depends(get_request_user)) -> User:
    role_names = {
        str(role).strip().lower()
        for role in [current_user.role, *current_user.roles]
        if str(role).strip()
    }
    permissions = {
        str(permission).strip().lower()
        for permission in current_user.permissions
        if str(permission).strip()
    }
    if (
        current_user.is_admin
        or current_user.is_superuser
        or "admin" in role_names
        or "*" in permissions
        or "system.configure" in permissions
    ):
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=vn_error_detail(
            ERROR_PERMISSION_DENIED,
            "Admin privileges are required for VN policy profile mutation.",
        ),
    )


@router.post("/evaluate", response_model=VNPolicyEvaluateResponse)
async def evaluate_policy(
    request: VNPolicyEvaluateRequest,
    service: VNPolicyService = Depends(_service),
) -> VNPolicyEvaluateResponse:
    """Evaluate a VN setup/script/runtime/TTS request against a policy profile."""
    try:
        result = await service.evaluate(
            target_type=request.target_type,
            target_id=request.target_id,
            policy_profile_id=request.policy_profile_id,
            context=request.context,
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNPolicyEvaluateResponse.model_validate(result)


@router.get("/profiles", response_model=VNPolicyProfileListResponse)
async def list_policy_profiles(
    limit: int = 50,
    offset: int = 0,
    service: VNPolicyService = Depends(_service),
) -> VNPolicyProfileListResponse:
    """List usable VN policy profiles."""
    bounded_limit, bounded_offset = _bounded_pagination(limit, offset)
    rows, total = await service.list_policy_profiles(limit=bounded_limit, offset=bounded_offset)
    return VNPolicyProfileListResponse(
        items=[VNPolicyProfileResponse.model_validate(row) for row in rows],
        **_pagination_payload(limit=bounded_limit, offset=bounded_offset, total=total),
    )


@router.get("/profiles/{profile_id}", response_model=VNPolicyProfileResponse)
async def get_policy_profile(
    profile_id: str,
    service: VNPolicyService = Depends(_service),
) -> VNPolicyProfileResponse:
    """Read a usable VN policy profile."""
    row = await service.get_policy_profile(profile_id)
    if row is None:
        raise _not_found("policy_profile_not_found")
    return VNPolicyProfileResponse.model_validate(row)


@router.post("/profiles", response_model=VNPolicyProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_policy_profile(
    request: VNPolicyProfileCreate,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> VNPolicyProfileResponse:
    """Create a VN policy profile. Admin-only."""
    try:
        row = await service.create_policy_profile(
            profile_id=request.profile_id,
            display_name=request.display_name,
            description=request.description,
            definition=request.definition,
            created_by_user_id=_current_user_id(admin_user),
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNPolicyProfileResponse.model_validate(row)


@router.patch("/profiles/{profile_id}", response_model=VNPolicyProfileResponse)
async def patch_policy_profile(
    profile_id: str,
    request: VNPolicyProfilePatch,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> VNPolicyProfileResponse:
    """Patch a VN policy profile. Admin-only."""
    try:
        row = await service.update_policy_profile(
            profile_id,
            display_name=request.display_name,
            description=request.description,
            definition=request.definition,
            updated_by_user_id=_current_user_id(admin_user),
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNPolicyProfileResponse.model_validate(row)


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_policy_profile(
    profile_id: str,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> Response:
    """Disable a VN policy profile. Admin-only."""
    try:
        await service.disable_policy_profile(profile_id, updated_by_user_id=_current_user_id(admin_user))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/generation-profiles", response_model=VNGenerationProfileListResponse)
async def list_generation_profiles(
    limit: int = 50,
    offset: int = 0,
    service: VNPolicyService = Depends(_service),
) -> VNGenerationProfileListResponse:
    """List usable VN generation profiles."""
    bounded_limit, bounded_offset = _bounded_pagination(limit, offset)
    rows, total = await service.list_generation_profiles(limit=bounded_limit, offset=bounded_offset)
    return VNGenerationProfileListResponse(
        items=[VNGenerationProfileResponse.model_validate(row) for row in rows],
        **_pagination_payload(limit=bounded_limit, offset=bounded_offset, total=total),
    )


@router.get("/generation-profiles/{profile_id}", response_model=VNGenerationProfileResponse)
async def get_generation_profile(
    profile_id: str,
    service: VNPolicyService = Depends(_service),
) -> VNGenerationProfileResponse:
    """Read a usable VN generation profile."""
    row = await service.get_generation_profile(profile_id)
    if row is None:
        raise _not_found("generation_profile_not_found")
    return VNGenerationProfileResponse.model_validate(row)


@router.post(
    "/generation-profiles",
    response_model=VNGenerationProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_generation_profile(
    request: VNGenerationProfileCreate,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> VNGenerationProfileResponse:
    """Create a VN generation profile. Admin-only."""
    try:
        row = await service.create_generation_profile(
            profile_id=request.profile_id,
            display_name=request.display_name,
            description=request.description,
            definition=request.definition_payload(),
            created_by_user_id=_current_user_id(admin_user),
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNGenerationProfileResponse.model_validate(row)


@router.patch("/generation-profiles/{profile_id}", response_model=VNGenerationProfileResponse)
async def patch_generation_profile(
    profile_id: str,
    request: VNGenerationProfilePatch,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> VNGenerationProfileResponse:
    """Patch a VN generation profile. Admin-only."""
    existing = await service.profile_store.get_generation_profile(profile_id, include_disabled=True)
    if existing is None:
        raise _not_found("generation_profile_not_found")
    definition = _patched_generation_definition(existing["definition"], request)
    try:
        row = await service.update_generation_profile(
            profile_id,
            display_name=request.display_name,
            description=request.description,
            definition=definition,
            updated_by_user_id=_current_user_id(admin_user),
        )
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return VNGenerationProfileResponse.model_validate(row)


@router.delete("/generation-profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_generation_profile(
    profile_id: str,
    service: VNPolicyService = Depends(_service),
    admin_user: User = Depends(_admin_user),
) -> Response:
    """Disable a VN generation profile. Admin-only."""
    try:
        await service.disable_generation_profile(profile_id, updated_by_user_id=_current_user_id(admin_user))
    except ValueError as exc:
        raise _handle_value_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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


def _patched_generation_definition(
    existing_definition: dict[str, Any],
    request: VNGenerationProfilePatch,
) -> dict[str, Any]:
    definition = dict(existing_definition)
    for field_name in (
        "provider",
        "model",
        "supports_structured_output",
        "temperature_default",
        "temperature_min",
        "temperature_max",
        "max_output_tokens",
        "allowed_content_ratings",
        "max_choices",
        "max_branch_depth",
        "max_model_expansion_scope",
        "tts_allowed",
        "output_persistence_max_days",
        "audit_mode",
    ):
        value = getattr(request, field_name)
        if value is not None:
            definition[field_name] = value
    return definition


def _not_found(code: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=vn_error_detail(ERROR_NOT_FOUND, code, details={"reason": code}),
    )


def _handle_value_error(exc: ValueError) -> HTTPException:
    detail = str(exc) or "invalid_request"
    if detail in {"profile_not_found", "policy_profile_not_found", "generation_profile_not_found", "target_not_found"}:
        return _not_found(detail)
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=vn_error_detail(
            ERROR_INVALID_REQUEST,
            detail,
            details={"reason": detail},
        ),
    )
