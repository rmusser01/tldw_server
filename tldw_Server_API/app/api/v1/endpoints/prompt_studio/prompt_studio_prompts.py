"""
Prompt Studio Prompts API (with versioning)

Manages prompts and their version history inside a project. Each update
creates a new immutable version to enable evaluation, comparison, and
reproducibility.

Key responsibilities
- Create/list/get prompts in a project
- Update prompts (create new version)
- View prompt version history
- Revert to a previous version (creates new version)

Security
- Read operations require project access
- Write operations require project write access
- Prompt length and content validated against SecurityConfig
"""

import contextlib
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, status
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    PromptStudioDatabase,
    SecurityConfig,
    get_prompt_studio_db,
    get_prompt_studio_user,
    get_security_config,
    require_project_access,
    require_project_write_access,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_page_pagination_meta
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import (
    ListResponse,
    PageListResponse,
    StandardResponse,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_project import (
    PromptCreate,
    StructuredPromptConvertRequest,
    StructuredPromptConvertResponse,
    StructuredPromptPreviewRequest,
    StructuredPromptPreviewResponse,
    PromptResponse,
    PromptUpdate,
    PromptVersion,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import ExecutePromptSimpleRequest
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    ConflictError,
    DatabaseError,
    InputError,
    _prepare_prompt_record_fields,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    PromptDefinition,
    StructuredPromptAssemblyError,
    assemble_prompt_definition,
    convert_legacy_prompt_to_definition,
    extract_legacy_prompt_variables,
    render_legacy_snapshot,
    validate_prompt_definition,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt-studio/prompts",
    tags=["prompt-studio"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)
########################################################################################################################
# Prompt CRUD Endpoints


def _render_definition_legacy_fields(definition: PromptDefinition) -> tuple[str, str]:
    messages = [
        {"role": block.role, "content": block.content}
        for block in sorted(definition.blocks, key=lambda item: item.order)
        if block.enabled
    ]
    legacy = render_legacy_snapshot(messages, definition)
    return legacy.system_prompt, legacy.user_prompt


def _coerce_security_config(security_config: SecurityConfig | Any) -> SecurityConfig:
    if isinstance(security_config, SecurityConfig):
        return security_config
    return get_security_config()


def _coerce_structured_definition(
    *,
    prompt_schema_version: int | None,
    prompt_definition_payload: dict[str, Any] | None,
) -> tuple[PromptDefinition, int]:
    if prompt_schema_version is None:
        raise InputError("Structured prompts require prompt_schema_version.")
    if not isinstance(prompt_definition_payload, dict):
        raise InputError("Structured prompts require prompt_definition.")
    definition = PromptDefinition.model_validate(prompt_definition_payload)
    issues = validate_prompt_definition(definition)
    if issues:
        raise InputError(issues[0].message)

    definition_schema_version = int(definition.schema_version)
    if int(prompt_schema_version) != definition_schema_version:
        raise InputError(
            "prompt_schema_version must match prompt_definition.schema_version."
        )

    return definition, definition_schema_version


def _validate_prompt_lengths(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    security_config: SecurityConfig,
) -> None:
    config = _coerce_security_config(security_config)

    if system_prompt and len(system_prompt) > config.max_prompt_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"System prompt exceeds maximum length of {config.max_prompt_length}"
        )

    if user_prompt and len(user_prompt) > config.max_prompt_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User prompt exceeds maximum length of {config.max_prompt_length}"
        )


def _validate_message_lengths(
    *,
    messages: list[dict[str, str]],
    security_config: SecurityConfig,
) -> None:
    config = _coerce_security_config(security_config)

    for message in messages:
        content = message.get("content") or ""
        if len(content) <= config.max_prompt_length:
            continue
        role = str(message.get("role") or "prompt")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{role.title()} prompt exceeds maximum length of "
                f"{config.max_prompt_length}"
            ),
        )


def _validate_total_message_length(
    *,
    messages: list[dict[str, str]],
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

    total_length = sum(len(message.get("content") or "") for message in messages)
    if total_length <= PromptExecutor.MAX_TOTAL_PROMPT_LENGTH:
        return

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "Structured prompt exceeds maximum total length of "
            f"{PromptExecutor.MAX_TOTAL_PROMPT_LENGTH} and would be truncated during execution"
        ),
    )


def _validation_variables(definition: PromptDefinition) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for variable in definition.variables:
        # Preserve stored defaults during save-time validation so oversized
        # default content is measured consistently with preview/execution.
        if variable.default_value is None:
            variables[variable.name] = ""
    return variables


def _get_signature_for_project(
    *,
    db: PromptStudioDatabase,
    project_id: int,
    signature_id: int | None,
) -> dict[str, Any] | None:
    if signature_id is None:
        return None

    signature = db.get_signature(signature_id)
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signature {signature_id} not found",
        )

    if int(signature.get("project_id") or -1) != int(project_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signature does not belong to the requested project",
        )

    return signature


def _validate_prompt_content(
    *,
    definition: PromptDefinition,
    extras: dict[str, Any],
    security_config: SecurityConfig,
    signature: dict[str, Any] | None = None,
) -> None:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

    assembly = assemble_prompt_definition(
        definition,
        _validation_variables(definition),
        extras=extras,
    )
    messages = assembly.messages
    legacy = assembly.legacy
    if signature is not None:
        messages = PromptExecutor._apply_signature_to_messages(messages, signature)
        legacy = render_legacy_snapshot(messages, definition)
    _validate_prompt_lengths(
        system_prompt=legacy.system_prompt,
        user_prompt=legacy.user_prompt,
        security_config=security_config,
    )
    _validate_message_lengths(
        messages=messages,
        security_config=security_config,
    )
    _validate_total_message_length(messages=messages)


def _coerce_preview_definition(
    *,
    prompt_format: str,
    prompt_schema_version: int | None,
    prompt_definition_payload: dict[str, Any] | None,
    system_prompt: str | None,
    user_prompt: str | None,
) -> tuple[PromptDefinition, str, int | None]:
    if prompt_format == "structured":
        definition, definition_schema_version = _coerce_structured_definition(
            prompt_schema_version=prompt_schema_version,
            prompt_definition_payload=prompt_definition_payload,
        )
        return definition, "structured", definition_schema_version

    definition = convert_legacy_prompt_to_definition(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return definition, "legacy", None

# Compatibility: simple POST on base path returns prompt object directly
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_prompt_simple(
    prompt_data: PromptCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> dict[str, Any]:
    resp = await create_prompt(prompt_data, db, security_config, user_context)  # type: ignore[arg-type]
    # Unwrap StandardResponse regardless of Pydantic/dict
    obj = resp.model_dump() if hasattr(resp, "model_dump") else resp if isinstance(resp, dict) else {}
    data = obj.get("data", obj)
    if hasattr(data, "model_dump"):
        return data.model_dump()
    return data

@router.post(
    "/create",
    response_model=StandardResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Create a prompt",
                            "value": {
                                "project_id": 1,
                                "name": "Summarizer",
                                "system_prompt": "Summarize the content clearly.",
                                "user_prompt": "{{text}}"
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Prompt created",
                "content": {
                    "application/json": {
                        "examples": {
                            "created": {
                                "summary": "Prompt created response",
                                "value": {
                                    "success": True,
                                    "data": {
                                        "id": 12,
                                        "project_id": 1,
                                        "name": "Summarizer",
                                        "version_number": 1,
                                        "system_prompt": "Summarize the content clearly.",
                                        "user_prompt": "{{text}}",
                                        "created_at": "2024-09-20T10:00:00"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def create_prompt(
    prompt_data: PromptCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")
) -> StandardResponse:
    """
    Create a new prompt in a project.

    Args:
        prompt_data: Prompt creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        Created prompt details
    """
    try:
        if not isinstance(idempotency_key, str):
            idempotency_key = None
        normalized_prompt_fields = _prepare_prompt_record_fields(
            prompt_format=prompt_data.prompt_format,
            prompt_schema_version=prompt_data.prompt_schema_version,
            prompt_definition=prompt_data.prompt_definition,
            system_prompt=prompt_data.system_prompt,
            user_prompt=prompt_data.user_prompt,
        )
        _validate_prompt_lengths(
            system_prompt=normalized_prompt_fields["system_prompt"],
            user_prompt=normalized_prompt_fields["user_prompt"],
            security_config=security_config,
        )

        # Ensure write access to the project
        await require_project_write_access(prompt_data.project_id, user_context=user_context, db=db)
        signature = _get_signature_for_project(
            db=db,
            project_id=prompt_data.project_id,
            signature_id=prompt_data.signature_id,
        )

        few_shot_payload = None
        if prompt_data.few_shot_examples:
            few_shot_payload = [model_dump_compat(ex) for ex in prompt_data.few_shot_examples]

        modules_payload = None
        if prompt_data.modules_config:
            modules_payload = [model_dump_compat(mod) for mod in prompt_data.modules_config]

        extras = {
            "few_shot_examples": few_shot_payload or [],
            "modules_config": modules_payload or [],
        }
        if normalized_prompt_fields["prompt_format"] == "structured":
            definition = PromptDefinition.model_validate(
                normalized_prompt_fields["prompt_definition"]
            )
        else:
            definition = convert_legacy_prompt_to_definition(
                system_prompt=normalized_prompt_fields["system_prompt"],
                user_prompt=normalized_prompt_fields["user_prompt"],
            )
        _validate_prompt_content(
            definition=definition,
            extras=extras,
            security_config=security_config,
            signature=signature,
        )

        # Idempotency: if provided, return existing prompt for this key
        user_id_str = str(user_context.get("user_id", "anonymous"))
        if idempotency_key:
            try:
                existing_id = db.lookup_idempotency("prompt", idempotency_key, user_id_str)
                if existing_id:
                    existing = db.get_prompt(existing_id)
                    if existing and int(existing.get("project_id") or -1) == int(prompt_data.project_id):
                        return StandardResponse(success=True, data=PromptResponse(**existing))
                    if existing:
                        logger.warning(
                            "Ignoring idempotency hit with mismatched project for key {} (user {}, requested project {})",
                            idempotency_key,
                            user_id_str,
                            prompt_data.project_id,
                        )
            except Exception:
                logger.debug("Prompt Studio checkpoint sync failed after prompt create")

        prompt_record = db.create_prompt(
            project_id=prompt_data.project_id,
            name=prompt_data.name,
            signature_id=prompt_data.signature_id,
            version_number=1,
            system_prompt=normalized_prompt_fields["system_prompt"],
            user_prompt=normalized_prompt_fields["user_prompt"],
            prompt_format=normalized_prompt_fields["prompt_format"],
            prompt_schema_version=normalized_prompt_fields["prompt_schema_version"],
            prompt_definition=normalized_prompt_fields["prompt_definition"],
            few_shot_examples=few_shot_payload,
            modules_config=modules_payload,
            parent_version_id=prompt_data.parent_version_id,
            change_description=prompt_data.change_description,
            client_id=user_context.get("client_id"),
        )

        if not prompt_record:
            raise DatabaseError("Prompt creation returned empty record")

        logger.info(f"User {user_context['user_id']} created prompt: {prompt_data.name} (project_id={prompt_data.project_id})")
        with contextlib.suppress(Exception):
            logger.info("Created prompt record: {}", prompt_record)

        # Record idempotency mapping if provided
        if idempotency_key and prompt_record.get("id"):
            with contextlib.suppress(Exception):
                db.record_idempotency("prompt", idempotency_key, int(prompt_record["id"]), user_id_str)

        return StandardResponse(
            success=True,
            data=PromptResponse(**prompt_record)
        )

    except ConflictError as e:
        raise map_db_error_to_http(
            e,
            conflict_detail="Prompt with this name already exists in the project",
        ) from e
    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error creating prompt"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to create prompt",
            input_detail_attr="safe_message",
        ) from e

@router.get(
    "/list/{project_id}",
    response_model=PageListResponse,
    openapi_extra={
        "responses": {
            "200": {
                "description": "Prompts",
                "content": {
                    "application/json": {
                        "examples": {
                            "list": {
                                "summary": "Prompt list",
                                "value": {
                                    "success": True,
                                    "data": [
                                        {"id": 12, "name": "Summarizer", "version_number": 2}
                                    ],
                                    "metadata": {
                                        "page": 1,
                                        "per_page": 20,
                                        "total": 1,
                                        "total_pages": 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def list_prompts(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include deleted prompts"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> PageListResponse:
    """
    List prompts in a project.

    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        include_deleted: Include soft-deleted prompts
        db: Database instance

    Returns:
        Paginated list of prompts
    """
    try:
        result = db.list_prompts(
            project_id,
            page=page,
            per_page=per_page,
            include_deleted=include_deleted,
        )
        prompts = [PromptResponse(**prompt) for prompt in result.get("prompts", [])]
        pagination_data = result.get("pagination") or {}
        page_value = int(pagination_data.get("page") or page)
        per_page_value = int(pagination_data.get("per_page") or pagination_data.get("limit") or per_page)
        total_value = int(pagination_data.get("total") if pagination_data.get("total") is not None else len(prompts))
        total_pages_value = pagination_data.get("total_pages")
        if total_pages_value is None:
            total_pages_value = (total_value + per_page_value - 1) // per_page_value if total_value else 0
        total_pages_value = int(total_pages_value)
        metadata = {
            "page": page_value,
            "per_page": per_page_value,
            "total": total_value,
            "total_pages": total_pages_value,
        }

        return PageListResponse(
            success=True,
            data=prompts,
            metadata=metadata,
            pagination=build_page_pagination_meta(
                page=page_value,
                per_page=per_page_value,
                total=total_value,
                total_pages=total_pages_value,
            ),
        )

    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error listing prompts"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to list prompts",
            input_detail_attr="safe_message",
        ) from e

# Simple alias that mirrors the canonical list response shape
@router.get("", response_model=PageListResponse)
async def list_prompts_simple(
    project_id: int = Query(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include deleted prompts"),
    _: bool = Depends(require_project_access),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> PageListResponse:
    return await list_prompts(project_id, page, per_page, include_deleted, True, db)  # type: ignore[arg-type]

# Simple execute endpoint used by tests
@router.post("/execute")
async def execute_prompt_simple(
    payload: ExecutePromptSimpleRequest,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> dict[str, Any]:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
    executor = PromptExecutor(db)
    prompt_id = int(payload.prompt_id)
    inputs = payload.inputs or {}
    provider = payload.provider or "openai"
    model = payload.model or "gpt-3.5-turbo"
    # Support both async executor (normal) and sync mocks in tests
    maybe = executor.execute(prompt_id, inputs=inputs, provider=provider, model=model)
    try:
        import inspect as _inspect
        if _inspect.isawaitable(maybe):
            result = await maybe
        else:
            result = maybe  # test mocks may return plain dict
    except Exception:
        # Fallback to awaiting; if it fails, raise for better visibility
        result = await maybe  # type: ignore[func-returns-value]
    return {
        "output": result.get("raw_output") or result.get("parsed_output") or "",
        "tokens_used": result.get("tokens_used", 0),
        "execution_time": result.get("execution_time_ms", 0) / 1000.0
    }


@router.post("/preview", response_model=StandardResponse)
async def preview_prompt(
    payload: StructuredPromptPreviewRequest,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user),
) -> StandardResponse:
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

    try:
        await require_project_access(payload.project_id, user_context=user_context, db=db)
        definition, prompt_format, prompt_schema_version = _coerce_preview_definition(
            prompt_format=payload.prompt_format,
            prompt_schema_version=payload.prompt_schema_version,
            prompt_definition_payload=payload.prompt_definition,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
        )

        extras = {
            "few_shot_examples": [model_dump_compat(example) for example in (payload.few_shot_examples or [])],
            "modules_config": [model_dump_compat(module) for module in (payload.modules_config or [])],
        }
        assembly = assemble_prompt_definition(definition, payload.variables, extras=extras)
        messages = assembly.messages

        signature = _get_signature_for_project(
            db=db,
            project_id=payload.project_id,
            signature_id=payload.signature_id,
        )
        if signature is not None:
            messages = PromptExecutor(db)._apply_signature_to_messages(messages, signature)

        legacy = render_legacy_snapshot(messages, definition)
        _validate_prompt_lengths(
            system_prompt=legacy.system_prompt,
            user_prompt=legacy.user_prompt,
            security_config=security_config,
        )
        _validate_message_lengths(
            messages=messages,
            security_config=security_config,
        )
        _validate_total_message_length(messages=messages)
        preview_data = StructuredPromptPreviewResponse(
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            assembled_messages=messages,
            legacy_system_prompt=legacy.system_prompt,
            legacy_user_prompt=legacy.user_prompt,
        )
        return StandardResponse(success=True, data=preview_data)
    except HTTPException:
        raise
    except InputError as e:
        raise map_db_error_to_http(
            e,
            input_detail_attr="safe_message",
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid prompt_definition: {e}",
        ) from e
    except StructuredPromptAssemblyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post("/convert", response_model=StandardResponse)
async def convert_prompt(
    payload: StructuredPromptConvertRequest,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user),
) -> StandardResponse:
    await require_project_access(payload.project_id, user_context=user_context, db=db)
    definition = convert_legacy_prompt_to_definition(
        system_prompt=payload.system_prompt,
        user_prompt=payload.user_prompt,
    )
    legacy_system_prompt, legacy_user_prompt = _render_definition_legacy_fields(definition)
    response = StructuredPromptConvertResponse(
        prompt_definition=definition.model_dump(),
        extracted_variables=extract_legacy_prompt_variables(
            payload.system_prompt,
            payload.user_prompt,
        ),
        legacy_system_prompt=legacy_system_prompt,
        legacy_user_prompt=legacy_user_prompt,
    )
    return StandardResponse(success=True, data=response)

@router.get("/get/{prompt_id}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Prompt", "content": {"application/json": {"examples": {"get": {"summary": "Prompt details", "value": {"success": True, "data": {"id": 12, "name": "Summarizer", "version_number": 2}}}}}}}}
})
async def get_prompt(
    prompt_id: int = Path(..., description="Prompt ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Get a specific prompt by ID.

    Args:
        prompt_id: Prompt ID
        db: Database instance

    Returns:
        Prompt details
    """
    try:
        prompt = db.get_prompt_with_project(prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        if prompt.get("project_user_id") != user_context["user_id"] and not user_context["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this prompt"
            )

        # Remove the extra field
        prompt.pop("project_user_id", None)

        return StandardResponse(
            success=True,
            data=PromptResponse(**prompt)
        )

    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error getting prompt"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to get prompt",
            input_detail_attr="safe_message",
        ) from e

@router.put(
    "/update/{prompt_id}",
    response_model=StandardResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "revise": {
                            "summary": "Revise prompt (new version)",
                            "value": {
                                "system_prompt": "Summarize concisely.",
                                "change_description": "Tighten style"
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "New prompt version created",
                "content": {
                    "application/json": {
                        "examples": {
                            "versioned": {
                                "summary": "New version response",
                                "value": {
                                    "success": True,
                                    "data": {
                                        "id": 13,
                                        "project_id": 1,
                                        "name": "Summarizer",
                                        "version_number": 2,
                                        "system_prompt": "Summarize concisely.",
                                        "created_at": "2024-09-21T10:00:00"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def update_prompt(
    prompt_id: int = Path(..., description="Prompt ID"),
    updates: PromptUpdate = ...,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Update a prompt (creates a new version).

    Args:
        prompt_id: Prompt ID
        updates: Fields to update
        db: Database instance
        security_config: Security configuration
        user_context: Current user context

    Returns:
        New prompt version details
    """
    try:
        current_prompt = db.get_prompt_with_project(prompt_id)
        if not current_prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        if current_prompt.get("project_user_id") != user_context["user_id"] and not user_context["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this prompt"
            )
        signature = _get_signature_for_project(
            db=db,
            project_id=int(current_prompt["project_id"]),
            signature_id=current_prompt.get("signature_id"),
        )

        normalized_prompt_fields = _prepare_prompt_record_fields(
            prompt_format=updates.prompt_format,
            prompt_schema_version=updates.prompt_schema_version,
            prompt_definition=updates.prompt_definition,
            system_prompt=updates.system_prompt,
            user_prompt=updates.user_prompt,
            current_prompt=current_prompt,
        )
        _validate_prompt_lengths(
            system_prompt=normalized_prompt_fields["system_prompt"],
            user_prompt=normalized_prompt_fields["user_prompt"],
            security_config=security_config,
        )

        # Create new version
        few_shot_payload = None
        if updates.few_shot_examples is not None:
            few_shot_payload = [model_dump_compat(ex) for ex in updates.few_shot_examples]

        modules_payload = None
        if updates.modules_config is not None:
            modules_payload = [model_dump_compat(mod) for mod in updates.modules_config]

        extras = {
            "few_shot_examples": few_shot_payload or [],
            "modules_config": modules_payload or [],
        }
        if normalized_prompt_fields["prompt_format"] == "structured":
            definition = PromptDefinition.model_validate(
                normalized_prompt_fields["prompt_definition"]
            )
        else:
            definition = convert_legacy_prompt_to_definition(
                system_prompt=normalized_prompt_fields["system_prompt"],
                user_prompt=normalized_prompt_fields["user_prompt"],
            )
        _validate_prompt_content(
            definition=definition,
            extras=extras,
            security_config=security_config,
            signature=signature,
        )

        new_prompt = db.create_prompt_version(
            prompt_id,
            change_description=updates.change_description,
            name=updates.name,
            system_prompt=normalized_prompt_fields["system_prompt"],
            user_prompt=normalized_prompt_fields["user_prompt"],
            prompt_format=normalized_prompt_fields["prompt_format"],
            prompt_schema_version=normalized_prompt_fields["prompt_schema_version"],
            prompt_definition=normalized_prompt_fields["prompt_definition"],
            few_shot_examples=few_shot_payload,
            modules_config=modules_payload,
            client_id=user_context.get("client_id"),
        )

        logger.info(
            'User {} created version {} of prompt {}',
            user_context["user_id"],
            new_prompt.get("version_number"),
            prompt_id,
        )

        return StandardResponse(
            success=True,
            data=PromptResponse(**new_prompt)
        )

    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error updating prompt"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to update prompt",
            input_detail_attr="safe_message",
        ) from e

@router.get("/history/{prompt_id}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "History", "content": {"application/json": {"examples": {"history": {"summary": "Versions", "value": {"success": True, "data": [{"id": 12, "version_number": 1}, {"id": 13, "version_number": 2}]}}}}}}}
})
async def get_prompt_history(
    prompt_id: int = Path(..., description="Prompt ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Get version history for a prompt.

    Args:
        prompt_id: Prompt ID
        db: Database instance

    Returns:
        List of prompt versions
    """
    try:
        prompt = db.get_prompt(prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        project_id = prompt["project_id"]
        await require_project_access(project_id, user_context=user_context, db=db)

        versions = [
            PromptVersion(**version)
            for version in db.list_prompt_versions(project_id, prompt["name"])
        ]

        return StandardResponse(
            success=True,
            data=versions
        )

    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error getting prompt history"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to get prompt history",
            input_detail_attr="safe_message",
        ) from e

@router.post("/revert/{prompt_id}/{version}", response_model=StandardResponse, openapi_extra={
    "responses": {"200": {"description": "Reverted", "content": {"application/json": {"examples": {"reverted": {"summary": "New version", "value": {"success": True, "data": {"id": 14, "version_number": 3}}}}}}}}
})
async def revert_prompt(
    prompt_id: int = Path(..., description="Current prompt ID"),
    version: int = Path(..., description="Version number to revert to"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Revert a prompt to a previous version (creates a new version).

    Args:
        prompt_id: Current prompt ID
        version: Version number to revert to
        db: Database instance
        user_context: Current user context

    Returns:
        New prompt version details
    """
    try:
        current_prompt = db.get_prompt(prompt_id)
        if not current_prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt {prompt_id} not found"
            )
        project_id = current_prompt["project_id"]
        await require_project_write_access(project_id, user_context=user_context, db=db)

        new_prompt = db.revert_prompt_to_version(
            prompt_id,
            version,
            client_id=user_context.get("client_id"),
        )

        logger.info(
            'User {} reverted prompt {} to version {}',
            user_context["user_id"],
            prompt_id,
            version,
        )

        return StandardResponse(
            success=True,
            data=PromptResponse(**new_prompt)
        )

    except (DatabaseError, InputError) as e:
        log_message = "Prompt studio storage error reverting prompt"
        if isinstance(e, InputError):
            logger.warning(
                "{}: {}",
                log_message,
                getattr(e, "original_message", str(e)),
            )
        else:
            logger.error(log_message)
        raise map_db_error_to_http(
            e,
            default_detail="Failed to revert prompt",
            input_detail_attr="safe_message",
        ) from e
