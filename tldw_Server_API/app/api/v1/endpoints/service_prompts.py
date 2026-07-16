"""Authenticated per-user access to curated Service Prompt overrides."""

from __future__ import annotations

from typing import NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_api_key_scope
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.schemas.service_prompt_schemas import (
    ServicePromptCatalogItemResponse,
    ServicePromptDetailResponse,
    ServicePromptUpdateRequest,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    DatabaseError,
    PromptsDatabase,
    ServicePromptRevisionConflict,
)
from tldw_Server_API.app.core.Prompt_Management.service_prompts import (
    ResolvedServicePrompt,
    ServicePromptCorruptOverride,
    ServicePromptDefinition,
    ServicePromptValidationError,
    UnknownServicePromptDefinition,
    get_service_prompt_definition,
    list_service_prompt_definitions,
    resolve_service_prompt,
    validate_service_prompt_parts,
)

router = APIRouter()


def _catalog_item(definition: ServicePromptDefinition) -> ServicePromptCatalogItemResponse:
    return ServicePromptCatalogItemResponse(
        id=definition.id,
        label=definition.label,
        description=definition.description,
        parts=[
            {
                "key": part.key,
                "label": part.label,
                "mode": part.mode,
                "required_variables": list(part.required_variables),
            }
            for part in definition.parts
        ],
        affected_workflows=[
            {"id": workflow.id, "label": workflow.label}
            for workflow in definition.affected_workflows
        ],
    )


def _detail(resolved: ResolvedServicePrompt) -> ServicePromptDetailResponse:
    definition = resolved.definition
    effective_parts = dict(resolved.parts)
    return ServicePromptDetailResponse(
        **_catalog_item(definition).model_dump(),
        default_parts=dict(definition.default_parts),
        saved_parts=effective_parts if resolved.source == "user" else None,
        effective_parts=effective_parts,
        source=resolved.source,
        revision=resolved.revision,
    )


def _domain_error(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownServicePromptDefinition):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            headers={"Cache-Control": "no-store"},
            detail={
                "code": "service_prompt_unknown_definition",
                "message": "Service Prompt definition was not found.",
            },
        )
    if isinstance(exc, ServicePromptRevisionConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            headers={"Cache-Control": "no-store"},
            detail={
                "code": "service_prompt_revision_conflict",
                "message": "Service Prompt override changed since it was loaded.",
                "current_revision": exc.current_revision,
            },
        )
    if isinstance(exc, ServicePromptValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            headers={"Cache-Control": "no-store"},
            detail={
                "code": "service_prompt_validation_failed",
                "message": "Service Prompt validation failed.",
                "field_errors": dict(exc.field_errors),
            },
        )
    if isinstance(exc, ServicePromptCorruptOverride):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"Cache-Control": "no-store"},
            detail={
                "code": "service_prompt_corrupt_override",
                "message": "The saved Service Prompt override is corrupt and can be reset.",
                "revision": exc.revision,
                "can_reset": True,
            },
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"Cache-Control": "no-store"},
        detail={
            "code": "service_prompt_store_failed",
            "message": "Service Prompt storage operation failed.",
        },
    )


def _raise_domain_error(exc: Exception) -> NoReturn:
    raise _domain_error(exc) from exc


@router.get(
    "/service-prompts",
    response_model=list[ServicePromptCatalogItemResponse],
    dependencies=[Depends(require_api_key_scope("read"))],
)
async def list_service_prompts(response: Response) -> list[ServicePromptCatalogItemResponse]:
    """Return stable Service Prompt metadata without any prompt bodies."""

    response.headers["Cache-Control"] = "no-store"
    return [_catalog_item(definition) for definition in list_service_prompt_definitions()]


@router.get(
    "/service-prompts/{definition_id}",
    response_model=ServicePromptDetailResponse,
    dependencies=[Depends(require_api_key_scope("read"))],
)
async def get_service_prompt(
    definition_id: str,
    response: Response,
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
) -> ServicePromptDetailResponse:
    """Return the current packaged, saved, and effective prompt state."""

    response.headers["Cache-Control"] = "no-store"
    try:
        return _detail(resolve_service_prompt(db, definition_id))
    except (
        UnknownServicePromptDefinition,
        ServicePromptCorruptOverride,
        DatabaseError,
    ) as exc:
        _raise_domain_error(exc)


@router.put(
    "/service-prompts/{definition_id}",
    response_model=ServicePromptDetailResponse,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def update_service_prompt(
    definition_id: str,
    payload: ServicePromptUpdateRequest,
    response: Response,
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
) -> ServicePromptDetailResponse:
    """Validate and atomically activate one complete prompt override."""

    response.headers["Cache-Control"] = "no-store"
    try:
        definition = get_service_prompt_definition(definition_id)
        validated = validate_service_prompt_parts(definition, payload.parts)
        resolve_service_prompt(db, definition.id)
        db.save_service_prompt_override(
            definition.id,
            validated,
            str(payload.expected_revision) if payload.expected_revision is not None else None,
        )
        return _detail(resolve_service_prompt(db, definition.id))
    except (
        UnknownServicePromptDefinition,
        ServicePromptRevisionConflict,
        ServicePromptValidationError,
        ServicePromptCorruptOverride,
        DatabaseError,
    ) as exc:
        _raise_domain_error(exc)


@router.delete(
    "/service-prompts/{definition_id}",
    response_model=ServicePromptDetailResponse,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def reset_service_prompt(
    definition_id: str,
    response: Response,
    expected_revision: UUID | None = Query(default=None),
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
) -> ServicePromptDetailResponse:
    """Atomically remove an override and return the packaged prompt state."""

    response.headers["Cache-Control"] = "no-store"
    try:
        definition = get_service_prompt_definition(definition_id)
        db.reset_service_prompt_override(
            definition.id,
            str(expected_revision) if expected_revision is not None else None,
        )
        return _detail(
            ResolvedServicePrompt(
                definition=definition,
                parts=definition.default_parts,
                source="packaged",
                revision=None,
            )
        )
    except (
        UnknownServicePromptDefinition,
        ServicePromptRevisionConflict,
        ServicePromptCorruptOverride,
        DatabaseError,
    ) as exc:
        _raise_domain_error(exc)


__all__ = ["router"]
