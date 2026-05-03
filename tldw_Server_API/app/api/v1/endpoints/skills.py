# app/api/v1/endpoints/skills.py
#
# REST API endpoints for skills management
#
"""
Skills API Endpoints
====================

Provides CRUD operations for SKILL.md-based skills:
- List, get, create, update, delete skills
- Import/export skills as files
- Execute skills with argument substitution
"""

from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import CurrentPrincipal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta

# Local Imports
from tldw_Server_API.app.api.v1.schemas.skills_schemas import (
    SkillContextPayload,
    SkillCreate,
    SkillExecuteRequest,
    SkillExecutionResult,
    SkillImportRequest,
    SkillResponse,
    SkillsListResponse,
    SkillSummary,
    SkillUpdate,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillNotFoundError,
    SkillsError,
    SkillValidationError,
)
from tldw_Server_API.app.core.Skills.skill_executor import RequestContext, SkillExecutor
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

router = APIRouter()


async def get_skills_service(
    principal: CurrentPrincipal,
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> SkillsService:
    """
    FastAPI dependency to get the SkillsService instance for the identified user.
    """
    if not isinstance(principal.user_id, int):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed for Skills service.",
        )

    user_id = principal.user_id
    user_base_dir = DatabasePaths.get_user_base_directory(user_id)

    return SkillsService(user_id=user_id, base_path=user_base_dir, db=chacha_db)


def _skill_data_to_response(skill_data: dict) -> SkillResponse:
    """Convert skill data dict to SkillResponse."""
    return SkillResponse(
        id=skill_data["id"],
        name=skill_data["name"],
        description=skill_data.get("description"),
        argument_hint=skill_data.get("argument_hint"),
        disable_model_invocation=skill_data.get("disable_model_invocation", False),
        user_invocable=skill_data.get("user_invocable", True),
        allowed_tools=skill_data.get("allowed_tools"),
        model=skill_data.get("model"),
        context=skill_data.get("context", "inline"),
        content=skill_data["content"],
        supporting_files=skill_data.get("supporting_files"),
        directory_path=skill_data["directory_path"],
        created_at=skill_data["created_at"],
        last_modified=skill_data["last_modified"],
        version=skill_data["version"],
    )


def _metadata_to_summary(metadata) -> SkillSummary:
    """Convert SkillMetadata to SkillSummary."""
    return SkillSummary(
        name=metadata.name,
        description=metadata.description,
        argument_hint=metadata.argument_hint,
        user_invocable=metadata.user_invocable,
        disable_model_invocation=metadata.disable_model_invocation,
        context=metadata.context,
    )


@router.get("/", response_model=SkillsListResponse)
async def list_skills(
    include_hidden: bool = Query(False, description="Include hidden skills (user_invocable=false)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of skills to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    service: SkillsService = Depends(get_skills_service),
):
    """
    List available skills.

    Returns skill summaries (descriptions only, not full content).
    """
    try:
        skills = await service.list_skills(
            include_hidden=include_hidden,
            limit=limit,
            offset=offset,
        )
        total = await service.get_total_count(include_hidden=include_hidden)

        return SkillsListResponse(
            skills=[_metadata_to_summary(s) for s in skills],
            count=len(skills),
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                limit=limit,
                offset=offset,
                total=total,
                count=len(skills),
            ),
        )
    except SkillsError as e:
        logger.error("Error listing skills")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list skills",
        ) from e


@router.get("/context", response_model=SkillContextPayload)
async def get_skills_context(
    service: SkillsService = Depends(get_skills_service),
):
    """
    Get skill context payload for LLM injection.

    Returns formatted skill descriptions suitable for including in chat context.
    """
    try:
        payload = await service.get_context_payload_async()
        return SkillContextPayload(
            available_skills=[
                SkillSummary(**s) for s in payload["available_skills"]
            ],
            context_text=payload["context_text"],
        )
    except SkillsError as e:
        logger.error("Error getting skills context")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skills context",
        ) from e


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(
    skill_name: str,
    service: SkillsService = Depends(get_skills_service),
):
    """
    Get full skill content by name.

    Returns complete skill data including SKILL.md content and supporting files.
    """
    try:
        skill_data = await service.get_skill(skill_name)
        return _skill_data_to_response(skill_data)
    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        ) from None
    except SkillsError as e:
        logger.error("Error getting skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill",
        ) from e


@router.post("/", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def create_skill(
    skill: SkillCreate,
    service: SkillsService = Depends(get_skills_service),
):
    """
    Create a new skill.

    The content should be full SKILL.md content with optional YAML frontmatter.
    """
    try:
        skill_data = await service.create_skill(
            name=skill.name,
            content=skill.content,
            supporting_files=skill.supporting_files,
        )
        return _skill_data_to_response(skill_data)
    except SkillConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except SkillValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except SkillsError as e:
        logger.error("Error creating skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create skill",
        ) from e


@router.put("/{skill_name}", response_model=SkillResponse)
async def update_skill(
    skill_name: str,
    skill: SkillUpdate,
    expected_version: Optional[int] = Header(None, alias="If-Match"),
    service: SkillsService = Depends(get_skills_service),
):
    """
    Update an existing skill.

    Supports optimistic locking via If-Match header with version number.
    """
    try:
        skill_data = await service.update_skill(
            name=skill_name,
            content=skill.content,
            supporting_files=skill.supporting_files,
            expected_version=expected_version,
        )
        return _skill_data_to_response(skill_data)
    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        ) from None
    except SkillConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except SkillValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except SkillsError as e:
        logger.error("Error updating skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update skill",
        ) from e


@router.delete("/{skill_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_skill(
    skill_name: str,
    expected_version: Optional[int] = Header(None, alias="If-Match"),
    service: SkillsService = Depends(get_skills_service),
):
    """
    Delete a skill.

    Supports optimistic locking via If-Match header with version number.
    """
    try:
        await service.delete_skill(skill_name, expected_version=expected_version)
    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        ) from None
    except SkillConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except SkillsError as e:
        logger.error("Error deleting skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete skill",
        ) from e


@router.post("/import", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def import_skill(
    request: SkillImportRequest,
    service: SkillsService = Depends(get_skills_service),
):
    """
    Import a skill from SKILL.md content.

    Set overwrite=true to replace an existing skill with the same name.
    """
    try:
        skill_data = await service.import_skill(
            content=request.content,
            name=request.name,
            supporting_files=request.supporting_files,
            overwrite=request.overwrite,
        )
        return _skill_data_to_response(skill_data)
    except SkillConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except SkillValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except SkillsError as e:
        logger.error("Error importing skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import skill",
        ) from e


@router.post("/import/file", response_model=SkillResponse, status_code=status.HTTP_201_CREATED)
async def import_skill_from_file(
    file: UploadFile = File(..., description="SKILL.md file or zip archive"),
    overwrite: bool = Query(False, description="Overwrite existing skill"),
    service: SkillsService = Depends(get_skills_service),
):
    """
    Import a skill from an uploaded file.

    Accepts either a SKILL.md file or a zip archive containing a skill directory.
    """
    try:
        content = await file.read()

        if file.filename and file.filename.lower().endswith(".zip"):
            # Import from zip
            skill_data = await service.import_from_zip(content, overwrite=overwrite)
        else:
            # Import from SKILL.md content
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                raise SkillValidationError("File must be UTF-8 encoded text or a zip archive") from None

            # Extract name from filename if possible
            name = None
            if file.filename:
                name = Path(file.filename).stem.lower()
                if name == "skill":
                    name = None  # Don't use "skill" as name

            skill_data = await service.import_skill(
                content=text_content,
                name=name,
                overwrite=overwrite,
            )

        return _skill_data_to_response(skill_data)
    except SkillConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except SkillValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except SkillsError as e:
        logger.error("Error importing skill from file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import skill from file",
        ) from e


@router.get(
    "/{skill_name}/export",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {
            "description": "Skill zip archive",
            "content": {
                "application/zip": {
                    "schema": {"type": "string", "format": "binary"},
                },
            },
        },
    },
)
async def export_skill(
    skill_name: str,
    service: SkillsService = Depends(get_skills_service),
):
    """
    Export a skill as a downloadable zip file.

    The zip contains the skill directory with SKILL.md and any supporting files.
    """
    try:
        zip_data = await service.export_skill(skill_name)
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{skill_name}.zip"',
            },
        )
    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        ) from None
    except SkillsError as e:
        logger.error("Error exporting skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export skill",
        ) from e


@router.post("/{skill_name}/execute", response_model=SkillExecutionResult)
async def execute_skill(
    skill_name: str,
    request: SkillExecuteRequest,
    principal: CurrentPrincipal,
    service: SkillsService = Depends(get_skills_service),
):
    """
    Execute a skill with optional arguments.

    Returns the rendered prompt with argument substitution applied.
    This endpoint is useful for testing/previewing skill output.
    """
    try:
        skill_data = await service.get_skill(skill_name)

        executor = SkillExecutor()
        ctx = None
        if isinstance(principal.user_id, int):
            # Populate full RequestContext for fork mode support
            try:
                from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
                default_provider = DEFAULT_LLM_PROVIDER
            except Exception:
                default_provider = "openai"
            try:
                from tldw_Server_API.app.core.config import load_and_log_configs
                app_config = load_and_log_configs()
            except Exception:
                app_config = None
            ctx = RequestContext(
                user_id=principal.user_id,
                default_provider=default_provider,
                app_config=app_config,
                client_id=getattr(service.db, "client_id", None) if getattr(service, "db", None) else None,
            )
        result = await executor.execute(
            skill_data=skill_data,
            arguments=request.args or "",
            context=ctx,
        )

        return SkillExecutionResult(
            skill_name=result.skill_name,
            rendered_prompt=result.rendered_prompt,
            allowed_tools=result.allowed_tools,
            model_override=result.model_override,
            execution_mode=result.execution_mode,
            fork_output=result.fork_output,
        )
    except SkillNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{skill_name}' not found",
        ) from None
    except SkillsError as e:
        logger.error("Error executing skill")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute skill",
        ) from e


@router.post(
    "/seed",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Seed built-in example skills",
    description="Copy built-in example skills into the user's skills directory.",
)
async def seed_builtin_skills(
    overwrite: bool = False,
    service: SkillsService = Depends(get_skills_service),
):
    """Seed built-in example skills.

    Args:
        overwrite: If true, replace existing skills with same names.
    """
    seeded = await service.seed_builtin_skills(overwrite=overwrite)
    return {"seeded": seeded, "count": len(seeded)}


#
# End of skills.py
#######################################################################################################################
