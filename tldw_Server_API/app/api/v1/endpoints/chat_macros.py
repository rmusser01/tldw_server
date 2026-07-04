"""REST endpoints for managing chat macros and macro runs."""

from __future__ import annotations

import re
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Chat_Macros_Deps import get_chat_macros_service
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.schemas.chat_macros import (
    ChatMacroBranchSummary,
    ChatMacroCancelResponse,
    ChatMacroCloneRequest,
    ChatMacroCreateRequest,
    ChatMacroDetail,
    ChatMacroListResponse,
    ChatMacroRunDetailResponse,
    ChatMacroRunRecordResponse,
    ChatMacroRunRequest,
    ChatMacroRunResponse,
    ChatMacroSettingsRequest,
    ChatMacroSettingsResponse,
    ChatMacroSummary,
    ChatMacroUpdateRequest,
    ChatMacroValidateRequest,
    ChatMacroValidateResponse,
)
from tldw_Server_API.app.core.Chat_Macros.exceptions import (
    MacroNotFoundError,
    MacroStorageError,
    MacroValidationError,
)
from tldw_Server_API.app.core.Chat_Macros.models import MacroArgSpec, MacroBranchRecord, MacroDefinition, MacroRunRecord
from tldw_Server_API.app.core.Chat_Macros.service import ChatMacroCatalogItem, ChatMacrosService

router = APIRouter()

_SECRET_BEARER_RE = re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+|bearer\s+)[^\s,;]+")
_SECRET_JSON_RE = re.compile(
    r"""(?ix)
    (?P<prefix>["'](?:api[_-]?key|x-api-key|token|password|secret)["']\s*:\s*)
    ["'][^"']+["']
    """
)
_SECRET_KV_RE = re.compile(
    r"""(?ix)
    (?P<prefix>\b(?:api[_-]?key|x-api-key|token|password|secret)\b\s*[:=]\s*)
    ["']?[^"'\s,;}]+["']?
    """
)
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{6,}")


def _summary(item: ChatMacroCatalogItem) -> ChatMacroSummary:
    return ChatMacroSummary(
        name=item.name,
        command=item.command,
        description=item.description,
        enabled=item.enabled,
        source=item.source,
        immutable=item.immutable,
        digest=item.digest,
        builtin_version=item.builtin_version,
        schema_version=item.definition.schema_version,
    )


def _detail(service: ChatMacrosService, item: ChatMacroCatalogItem) -> ChatMacroDetail:
    raw = yaml.safe_dump(item.definition.model_dump(mode="json"), sort_keys=False)
    supporting_files: dict[str, str] = {}
    if item.source == "user":
        stored = service.storage.read(item.name)
        raw = stored.raw
        supporting_files = stored.supporting_files
    return ChatMacroDetail(
        summary=_summary(item),
        definition=item.definition.model_dump(mode="json"),
        raw=raw,
        supporting_files=supporting_files,
    )


def _macro_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, MacroNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, MacroValidationError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    message = str(exc)
    lowered = message.lower()
    if "already exists" in lowered or "conflict" in lowered:
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)
    if "exceeds" in lowered:
        return HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=message)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def _raise_macro_http(exc: Exception) -> None:
    raise _macro_http_exception(exc) from exc


def _default_arg_value(spec: MacroArgSpec) -> Any:
    if spec.repeated:
        return list(spec.default or [])
    return spec.default


def _matches_arg_type(value: Any, arg_type: str) -> bool:
    if arg_type == "string":
        return isinstance(value, str)
    if arg_type == "boolean":
        return isinstance(value, bool)
    if arg_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if arg_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _normalize_structured_args(definition: MacroDefinition, args: dict[str, Any]) -> dict[str, Any]:
    normalized = {name: _default_arg_value(spec) for name, spec in definition.args.items()}
    for name, value in args.items():
        spec = definition.args.get(name)
        if spec is None:
            raise MacroValidationError(f"unknown macro argument: {name}")
        if spec.repeated:
            if not isinstance(value, list):
                raise MacroValidationError(f"macro argument must be a list: {name}")
            if name == "question" and len(value) > definition.execution.max_branches:
                raise MacroValidationError(f"too many question arguments; max is {definition.execution.max_branches}")
            for item in value:
                if not _matches_arg_type(item, spec.type):
                    raise MacroValidationError(f"macro argument has invalid item type: {name}")
            normalized[name] = list(value)
            continue
        if not _matches_arg_type(value, spec.type):
            raise MacroValidationError(f"macro argument has invalid type: {name}")
        normalized[name] = value
    return normalized


def _safe_error(value: str | None) -> str | None:
    if not value:
        return None
    redacted = _SECRET_BEARER_RE.sub(r"\1[redacted]", value)
    redacted = _SECRET_JSON_RE.sub(r'\g<prefix>"[redacted]"', redacted)
    redacted = _SECRET_KV_RE.sub(r"\g<prefix>[redacted]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[redacted]", redacted)
    return redacted[:500]


def _run_response(run: MacroRunRecord) -> ChatMacroRunResponse:
    return ChatMacroRunResponse(
        run_id=run.run_id,
        status=run.status,
        detail_url=f"/api/v1/chat/macros/runs/{run.run_id}",
        job_id=run.job_id,
    )


def _run_record_response(run: MacroRunRecord) -> ChatMacroRunRecordResponse:
    return ChatMacroRunRecordResponse(
        run_id=run.run_id,
        macro_name=run.macro_name,
        macro_command=run.macro_command,
        macro_source=run.macro_source,
        macro_version=run.macro_version,
        macro_digest=run.macro_digest,
        normalized_args=run.normalized_args,
        status=run.status,
        surface=run.surface,
        conversation_id=run.conversation_id,
        workspace_id=run.workspace_id,
        acp_session_id=run.acp_session_id,
        job_id=run.job_id,
        output_profile=run.output_profile,
        status_message_id=run.status_message_id,
        final_message_id=run.final_message_id,
        final_output=run.final_output,
        final_output_format=run.final_output_format,
        final_post_status=run.final_post_status,
        cancel_requested_at=run.cancel_requested_at,
        error_code=run.error_code,
        error=_safe_error(run.error_message),
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        updated_at=run.updated_at,
    )


def _branch_summary(branch: MacroBranchRecord) -> ChatMacroBranchSummary:
    return ChatMacroBranchSummary(
        branch_id=branch.branch_id,
        step_id=branch.step_id,
        label=branch.label,
        output_name=branch.output_name,
        status=branch.status,
        attempt_count=branch.attempt_count,
        output=branch.output_text,
        retained=branch.retained,
        error_code=branch.error_code,
        error=_safe_error(branch.error_message),
        created_at=branch.created_at,
        started_at=branch.started_at,
        finished_at=branch.finished_at,
    )


def _get_user_run(service: ChatMacrosService, run_id: str) -> MacroRunRecord:
    run = service.repository.get_run(run_id)
    if run is None or run.user_id != service.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Macro run not found")
    return run


@router.get("", response_model=ChatMacroListResponse)
async def list_chat_macros(
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroListResponse:
    try:
        macros = [_summary(item) for item in service.list_macros()]
        return ChatMacroListResponse(macros=macros, count=len(macros))
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)
    except Exception as exc:
        logger.error("Failed to list chat macros")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list chat macros") from exc


@router.post("", response_model=ChatMacroDetail, status_code=status.HTTP_201_CREATED)
async def create_chat_macro(
    request: ChatMacroCreateRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroDetail:
    try:
        item = service.create_macro(request.name, request.raw, request.supporting_files)
        return _detail(service, item)
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)


@router.post("/validate", response_model=ChatMacroValidateResponse)
async def validate_chat_macro(
    request: ChatMacroValidateRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroValidateResponse:
    try:
        definition = service.validate_macro(request.raw)
        return ChatMacroValidateResponse(valid=True, macro=definition.model_dump(mode="json"), error=None)
    except (MacroValidationError, MacroStorageError) as exc:
        return ChatMacroValidateResponse(valid=False, macro=None, error=str(exc))


@router.get("/settings", response_model=ChatMacroSettingsResponse)
async def get_chat_macro_settings(
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroSettingsResponse:
    try:
        return ChatMacroSettingsResponse(settings=service.get_settings())
    except MacroStorageError as exc:
        _raise_macro_http(exc)


@router.put("/settings", response_model=ChatMacroSettingsResponse)
async def update_chat_macro_settings(
    request: ChatMacroSettingsRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroSettingsResponse:
    try:
        return ChatMacroSettingsResponse(settings=service.save_settings(request.settings))
    except (MacroValidationError, MacroStorageError) as exc:
        _raise_macro_http(exc)


@router.post("/run", response_model=ChatMacroRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_chat_macro_run(
    request: ChatMacroRunRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
    job_manager: Any = Depends(try_get_job_manager),
) -> ChatMacroRunResponse:
    try:
        item = service.get_macro(request.macro_name)
        if not item.enabled:
            raise MacroValidationError("macro is disabled")
        normalized_args = _normalize_structured_args(item.definition, request.args)
        normalized_args["mode"] = request.mode
        output_profile = request.output_profile or str(normalized_args.get("output_profile") or item.definition.output_profile)
        resolved_profile = service.resolve_output_profile(output_profile)
        output_profile = resolved_profile.name
        normalized_args["output_profile"] = output_profile
        run = service.repository.create_run(
            user_id=service.user_id,
            macro_name=item.name,
            macro_command=item.command,
            macro_source=item.source,
            macro_version=item.builtin_version,
            macro_digest=item.digest,
            normalized_args=normalized_args,
            status="pending",
            surface=request.surface,
            conversation_id=request.conversation_id,
            workspace_id=request.workspace_id,
            acp_session_id=request.acp_session_id,
            output_profile=output_profile,
            context_snapshot=request.context_snapshot,
            model_selection=request.model_selection,
        )
        if job_manager is None:
            return _run_response(run)
        return _run_response(run)
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)


@router.get("/runs/{run_id}", response_model=ChatMacroRunDetailResponse)
async def get_chat_macro_run(
    run_id: str,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroRunDetailResponse:
    try:
        run = _get_user_run(service, run_id)
        branches = [_branch_summary(branch) for branch in service.repository.list_branches(run_id)]
        return ChatMacroRunDetailResponse(run=_run_record_response(run), branches=branches)
    except HTTPException:
        raise
    except MacroStorageError as exc:
        _raise_macro_http(exc)


@router.post("/runs/{run_id}/cancel", response_model=ChatMacroCancelResponse)
async def cancel_chat_macro_run(
    run_id: str,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroCancelResponse:
    try:
        _get_user_run(service, run_id)
        run = service.repository.request_cancel(run_id)
        return ChatMacroCancelResponse(
            run_id=run.run_id,
            status=run.status,
            cancel_requested_at=run.cancel_requested_at,
        )
    except HTTPException:
        raise
    except MacroStorageError as exc:
        _raise_macro_http(exc)


@router.post("/{name}/clone", response_model=ChatMacroDetail, status_code=status.HTTP_201_CREATED)
async def clone_chat_macro(
    name: str,
    request: ChatMacroCloneRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroDetail:
    try:
        item = service.clone_builtin(name, new_name=request.name, command=request.command)
        return _detail(service, item)
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)


@router.get("/{name}", response_model=ChatMacroDetail)
async def get_chat_macro(
    name: str,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroDetail:
    try:
        return _detail(service, service.get_macro(name))
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)


@router.put("/{name}", response_model=ChatMacroDetail)
async def update_chat_macro(
    name: str,
    request: ChatMacroUpdateRequest,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> ChatMacroDetail:
    try:
        item = service.update_macro(name, request.raw, request.supporting_files)
        return _detail(service, item)
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_macro(
    name: str,
    service: ChatMacrosService = Depends(get_chat_macros_service),
) -> Response:
    try:
        service.delete_macro(name)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except (MacroNotFoundError, MacroStorageError, MacroValidationError) as exc:
        _raise_macro_http(exc)
