# tldw_Server_API/app/api/v1/endpoints/prompts.py
#
#
# Imports
import base64
import contextlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Optional, Union

#
# 3rd-party imports
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    AuthPrincipal,
    TokenScopeGuard,
    User,
    consume_deferred_token_quota,
    enforce_rbac_rate_limit,
    get_auth_principal,
    get_request_user,
)
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.llm_routing_deps import (
    get_request_routing_decision_store,
)
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas import prompt_schemas as schemas
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.llm_budget_guard import enforce_llm_budget
from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.core.Chat.chat_exceptions import set_request_id
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    ConflictError,
    DatabaseError,
    InputError,
    PromptsDatabase,
)
from tldw_Server_API.app.core.LLM_Calls.routing import InMemoryRoutingDecisionStore
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement import (
    PROMPT_IMPROVEMENT_LIMITS,
    PromptImprovementError,
    PromptImprovementInput,
    improve_prompt,
    validate_prompt_improvement_input,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement_dispatch import (
    PromptImprovementDispatchError,
    PromptImprovementDispatchResult,
    dispatch_prompt_improvement,
)

#
# Local Imports
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    db_export_prompt_keywords_to_csv,
    db_export_prompts_formatted,  # Using the standalone function from interop
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
from tldw_Server_API.app.core.testing import env_flag_enabled

from .llm_providers import get_configured_providers

#from tldw_Server_API.app.core.DB_Management.DB_Manager import DBManager
#
#
#######################################################################################################################
#
# Functions:

router = APIRouter()

_TEMPLATE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_MAX_DUPLICATE_NAME_ITERATIONS = 10000

_PROMPTS_LOOKUP_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
)

_PROMPTS_ENDPOINT_EXCEPTIONS = _PROMPTS_LOOKUP_EXCEPTIONS + (
    HTTPException,
)

_PROMPTS_DB_OPERATION_EXCEPTIONS = _PROMPTS_LOOKUP_EXCEPTIONS + (
    DatabaseError,
    ConflictError,
    InputError,
)

_PROMPT_IMPROVEMENT_ERROR_MESSAGES = {
    "invalid_input": "The prompt improvement request is invalid.",
    "missing_model": "Select an active chat model and try again.",
    "unsupported_model": "The selected chat model is not available.",
    "provider_not_configured": "The active provider is not configured for this request.",
    "draft_too_large": "The prompt draft exceeds the configured size limit.",
    "provider_rate_limited": "The active provider is temporarily rate limited.",
    "provider_timeout": "The active provider timed out.",
    "provider_unavailable": "The active provider is temporarily unavailable.",
    "model_refusal": "The active model did not provide an improvement candidate.",
    "invalid_model_output": "The active model returned an unusable response.",
    "preservation_failed": "The candidate could not be presented safely.",
    "internal_error": "The prompt improvement request could not be completed.",
}

_PROMPT_IMPROVEMENT_ERROR_STATUS = {
    "invalid_input": status.HTTP_400_BAD_REQUEST,
    "missing_model": status.HTTP_400_BAD_REQUEST,
    "unsupported_model": status.HTTP_400_BAD_REQUEST,
    "provider_not_configured": status.HTTP_503_SERVICE_UNAVAILABLE,
    "draft_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "provider_rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
    "provider_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "provider_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
    "model_refusal": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_model_output": status.HTTP_502_BAD_GATEWAY,
    "preservation_failed": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "internal_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
}

_PROMPT_IMPROVEMENT_RETRYABLE_CODES = frozenset(
    {"provider_rate_limited", "provider_timeout", "provider_unavailable"}
)

_PROMPT_IMPROVEMENT_ERROR_RESPONSES = {
    status_code: {
        "model": schemas.PromptImproveErrorResponse,
        "description": "Sanitized prompt-improvement failure.",
    }
    for status_code in sorted(set(_PROMPT_IMPROVEMENT_ERROR_STATUS.values()))
}

_PROMPT_IMPROVEMENT_TOKEN_SECURITY = TokenScopeGuard(
    "any",
    require_if_present=True,
    endpoint_id="prompts.improve",
    count_as="call",
    defer_count=True,
)
_PROMPT_IMPROVEMENT_BILLING_CHECK = require_within_limit(
    LimitCategory.API_CALLS_DAY,
    1,
)


def _inline_local_openapi_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic-local definitions before embedding a schema in OpenAPI."""

    definitions = schema.pop("$defs", {})

    def expand(value: Any) -> Any:
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            referenced = definitions.get(name)
            if isinstance(referenced, dict):
                siblings = {key: item for key, item in value.items() if key != "$ref"}
                return expand({**referenced, **siblings})
        return {key: expand(item) for key, item in value.items()}

    return expand(schema)


def _extract_template_variables(template: str) -> list[str]:
    variables: list[str] = []
    for match in _TEMPLATE_VAR_RE.finditer(template or ""):
        var = match.group(1).strip()
        if var and var not in variables:
            variables.append(var)
    return variables


def _render_template(template: str, variables: dict[str, Any]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        if key not in variables:
            raise KeyError(key)
        return str(variables[key])

    return _TEMPLATE_VAR_RE.sub(repl, template)


def _render_definition_legacy_fields(definition: PromptDefinition) -> tuple[str, str]:
    messages = [
        {"role": block.role, "content": block.content}
        for block in sorted(definition.blocks, key=lambda item: item.order)
        if block.enabled
    ]
    legacy = render_legacy_snapshot(messages, definition)
    return legacy.system_prompt, legacy.user_prompt


def _coerce_structured_definition(
    prompt_schema_version: int | None,
    prompt_definition_payload: dict[str, Any] | None,
) -> tuple[PromptDefinition, int]:
    if prompt_schema_version is None:
        raise InputError("Structured prompts require prompt_schema_version.")
    if not isinstance(prompt_definition_payload, dict):
        raise InputError("Structured prompts require prompt_definition.")
    try:
        definition = PromptDefinition.model_validate(prompt_definition_payload)
    except ValidationError as exc:
        raise InputError(f"Invalid prompt_definition: {exc}") from exc
    issues = validate_prompt_definition(definition)
    if issues:
        raise InputError(issues[0].message)

    definition_schema_version = int(definition.schema_version)
    if int(prompt_schema_version) != definition_schema_version:
        raise InputError(
            "prompt_schema_version must match prompt_definition.schema_version."
        )

    return definition, definition_schema_version


def _coerce_preview_definition(
    prompt_format: str,
    prompt_schema_version: int | None,
    prompt_definition_payload: dict[str, Any] | None,
    *,
    system_prompt: str | None,
    user_prompt: str | None,
) -> tuple[PromptDefinition, str, int | None]:
    if prompt_format == "structured":
        definition, definition_schema_version = _coerce_structured_definition(
            prompt_schema_version,
            prompt_definition_payload,
        )
        return definition, "structured", definition_schema_version

    definition = convert_legacy_prompt_to_definition(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return definition, "legacy", None


def _prepare_prompt_storage_payload(
    prompt_data: schemas.PromptCreate | dict[str, Any],
    *,
    existing_prompt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(prompt_data, schemas.PromptCreate):
        payload = prompt_data.model_dump()
    else:
        payload = dict(prompt_data)

    prompt_format = payload.get("prompt_format") or (
        existing_prompt.get("prompt_format") if existing_prompt else "legacy"
    ) or "legacy"
    prompt_schema_version = payload.get("prompt_schema_version")
    if prompt_schema_version is None and existing_prompt and prompt_format == "structured":
        prompt_schema_version = existing_prompt.get("prompt_schema_version")
    prompt_definition_payload = payload.get("prompt_definition")
    if prompt_definition_payload is None and existing_prompt and prompt_format == "structured":
        prompt_definition_payload = existing_prompt.get("prompt_definition")

    if prompt_format == "structured":
        definition, definition_schema_version = _coerce_structured_definition(
            prompt_schema_version,
            prompt_definition_payload,
        )

        system_prompt, user_prompt = _render_definition_legacy_fields(definition)
        payload["prompt_definition"] = definition.model_dump()
        payload["prompt_schema_version"] = definition_schema_version
        payload["prompt_format"] = "structured"
        payload["system_prompt"] = system_prompt
        payload["user_prompt"] = user_prompt
        return payload

    if prompt_definition_payload is not None:
        raise InputError("Legacy prompts cannot include prompt_definition.")
    if prompt_schema_version is not None:
        raise InputError("Legacy prompts cannot include prompt_schema_version.")

    payload["prompt_format"] = "legacy"
    payload["prompt_schema_version"] = None
    payload["prompt_definition"] = None
    return payload


def _generate_unique_prompt_name(base_name: str, used_names: set, name_counts: dict[str, int]) -> str:
    count = name_counts.get(base_name, 0)
    for _ in range(_MAX_DUPLICATE_NAME_ITERATIONS):
        count += 1
        candidate = f"duplicate {count} - {base_name}"
        if candidate not in used_names:
            name_counts[base_name] = count
            return candidate
    raise InputError(f"Could not generate unique name for '{base_name}' after {_MAX_DUPLICATE_NAME_ITERATIONS} attempts.")

def _is_single_user_auth_mode() -> bool:
    if settings.get("SINGLE_USER_MODE") is True:
        return True
    try:
        return get_auth_settings().AUTH_MODE == "single_user"
    except _PROMPTS_LOOKUP_EXCEPTIONS:
        return bool(settings.get("SINGLE_USER_MODE"))


def _get_single_user_api_key() -> Optional[str]:
    if "SINGLE_USER_API_KEY" in settings:
        key = settings.get("SINGLE_USER_API_KEY")
        return key if key else None
    try:
        key = getattr(get_auth_settings(), "SINGLE_USER_API_KEY", None)
    except _PROMPTS_LOOKUP_EXCEPTIONS:
        key = None
    if key:
        return key
    return settings.get("SINGLE_USER_API_KEY")


async def _resolve_prompts_auth_user(
    request: Request,
    Token: Optional[str] = Header(None, alias="Token"),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    Authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[User]:
    """
    Validate the legacy Token header for prompts endpoints.

    Single-user mode validates against SINGLE_USER_API_KEY. Multi-user mode
    defers to the unified AuthNZ path for API keys/JWTs.
    """
    raw_token = None
    for candidate in (Token, x_api_key, Authorization):
        if isinstance(candidate, str) and candidate.strip():
            raw_token = candidate.strip()
            break

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    normalized = raw_token
    if normalized.lower().startswith("bearer "):
        normalized = normalized[len("Bearer ") :].strip()

    if _is_single_user_auth_mode():
        expected = _get_single_user_api_key()
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server authentication misconfigured (API key missing).",
            )
        if normalized != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )
        # Preserve claim-first authorization semantics in downstream checks by
        # returning a synthetic admin-style User rather than branching on mode.
        try:
            user_id = int(getattr(get_auth_settings(), "SINGLE_USER_FIXED_ID", 1))
        except _PROMPTS_LOOKUP_EXCEPTIONS:
            user_id = 1
        return User(
            id=user_id,
            username="single_user",
            role="admin",
            is_active=True,
            is_verified=True,
            is_superuser=True,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
        )

    if request is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    bearer_token = None
    api_key = None
    legacy_header = None

    if isinstance(Authorization, str) and Authorization.strip():
        scheme, _, credential = Authorization.strip().partition(" ")
        if scheme.lower() == "bearer" and credential:
            bearer_token = credential.strip()

    if isinstance(Token, str) and Token.strip():
        legacy_header = Token.strip()
        if legacy_header.lower().startswith("bearer "):
            bearer_token = legacy_header[len("Bearer ") :].strip()
        else:
            api_key = legacy_header

    if isinstance(x_api_key, str) and x_api_key.strip():
        api_key = x_api_key.strip()

    try:
        user = await get_request_user(
            request,
            api_key=api_key,
            token=bearer_token,
            legacy_token_header=legacy_header,
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            ) from exc
        raise

    return user


def _is_prompts_admin_user(user: Optional[User]) -> bool:
    if user is None:
        return False
    try:
        roles = {
            str(role).strip().lower()
            for role in (getattr(user, "roles", []) or [])
            if str(role).strip()
        }
        permissions = {
            str(perm).strip().lower()
            for perm in (getattr(user, "permissions", []) or [])
            if str(perm).strip()
        }
        if "admin" in roles:
            return True
        if "*" in permissions:
            return True
        if "system.configure" in permissions:
            return True
    except _PROMPTS_LOOKUP_EXCEPTIONS:
        return False
    return False

async def verify_prompts_auth(
    request: Request,
    Token: Optional[str] = Header(None, alias="Token"),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    Authorization: Optional[str] = Header(None, alias="Authorization"),
) -> bool:
    """
    Validate the legacy Token header for prompts endpoints.

    Single-user mode validates against SINGLE_USER_API_KEY. Multi-user mode
    defers to the unified AuthNZ path for API keys/JWTs and enforces admin.
    """
    user = await _resolve_prompts_auth_user(
        request=request,
        Token=Token,
        x_api_key=x_api_key,
        Authorization=Authorization,
    )

    if not _is_prompts_admin_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Required role(s): admin",
        )

    return True

async def verify_prompts_user(
    request: Request,
    Token: Optional[str] = Header(None, alias="Token"),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    Authorization: Optional[str] = Header(None, alias="Authorization"),
) -> bool:
    """Authenticate prompts requests for non-admin users.

    Set PROMPTS_REQUIRE_ADMIN=true to force admin-only access for these routes.
    """
    user = await _resolve_prompts_auth_user(
        request=request,
        Token=Token,
        x_api_key=x_api_key,
        Authorization=Authorization,
    )
    require_admin = env_flag_enabled("PROMPTS_REQUIRE_ADMIN")
    if require_admin and not _is_prompts_admin_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Required role(s): admin",
        )
    return True


def _prompt_improvement_error_response(
    code: str,
    *,
    request_id: str,
    retryable: bool | None = None,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    """Build one bounded public failure without echoing untrusted content."""

    public_code = (
        code if code in _PROMPT_IMPROVEMENT_ERROR_MESSAGES else "internal_error"
    )
    should_retry = (
        public_code in _PROMPT_IMPROVEMENT_RETRYABLE_CODES
        if retryable is None
        else bool(retryable)
    )
    if retry_after_seconds is not None:
        try:
            retry_after_seconds = max(0, min(int(retry_after_seconds), 86_400))
        except (TypeError, ValueError):
            retry_after_seconds = None
    error = schemas.PromptImproveErrorResponse(
        code=public_code,
        message=_PROMPT_IMPROVEMENT_ERROR_MESSAGES[public_code],
        retryable=should_retry,
        retry_after_seconds=retry_after_seconds,
        request_id=request_id,
    )
    headers = (
        {"Retry-After": str(retry_after_seconds)}
        if retry_after_seconds is not None
        else None
    )
    return JSONResponse(
        status_code=_PROMPT_IMPROVEMENT_ERROR_STATUS[public_code],
        content=error.model_dump(exclude_none=True),
        headers=headers,
    )


def _parse_prompt_improvement_payload(
    payload: Any,
    *,
    request_id: str,
) -> schemas.PromptImproveRequest | JSONResponse:
    """Validate manually so FastAPI never echoes invalid draft input."""

    if isinstance(payload, Mapping):
        raw_text = payload.get("text")
        if (
            isinstance(raw_text, str)
            and len(raw_text) > PROMPT_IMPROVEMENT_LIMITS.max_draft_chars
        ):
            return _prompt_improvement_error_response(
                "draft_too_large",
                request_id=request_id,
                retryable=False,
            )
    try:
        return schemas.PromptImproveRequest.model_validate(payload)
    except ValidationError:
        return _prompt_improvement_error_response(
            "invalid_input",
            request_id=request_id,
            retryable=False,
        )


async def _read_prompt_improvement_payload(
    request: Request,
    *,
    request_id: str,
) -> schemas.PromptImproveRequest | JSONResponse:
    """Bound raw request parsing so malformed JSON uses the public error contract."""

    content_type = str(request.headers.get("content-type") or "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not (
        media_type == "application/json"
        or (media_type.startswith("application/") and media_type.endswith("+json"))
    ):
        return _prompt_improvement_error_response(
            "invalid_input",
            request_id=request_id,
            retryable=False,
        )

    content_length = request.headers.get("content-length")
    if content_length is not None:
        normalized_length = content_length.strip()
        if not normalized_length.isdecimal():
            return _prompt_improvement_error_response(
                "invalid_input",
                request_id=request_id,
                retryable=False,
            )
        if int(normalized_length) > PROMPT_IMPROVEMENT_LIMITS.max_request_bytes:
            return _prompt_improvement_error_response(
                "draft_too_large",
                request_id=request_id,
                retryable=False,
            )

    chunks: list[bytes] = []
    accumulated = 0
    try:
        async for chunk in request.stream():
            accumulated += len(chunk)
            if accumulated > PROMPT_IMPROVEMENT_LIMITS.max_request_bytes:
                return _prompt_improvement_error_response(
                    "draft_too_large",
                    request_id=request_id,
                    retryable=False,
                )
            if chunk:
                chunks.append(bytes(chunk))
    except (ClientDisconnect, RuntimeError):
        return _prompt_improvement_error_response(
            "invalid_input",
            request_id=request_id,
            retryable=False,
        )

    raw_payload = b"".join(chunks)
    if len(raw_payload) > PROMPT_IMPROVEMENT_LIMITS.max_request_bytes:
        return _prompt_improvement_error_response(
            "draft_too_large",
            request_id=request_id,
            retryable=False,
        )
    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
        return _prompt_improvement_error_response(
            "invalid_input",
            request_id=request_id,
            retryable=False,
        )
    return _parse_prompt_improvement_payload(payload, request_id=request_id)


async def _run_prompt_improvement_post_validation_gates(
    *,
    request: Request,
    response: Response,
    principal: AuthPrincipal,
    x_tldw_org_id: int | None,
    org_id: int | None,
) -> None:
    """Debit and enforce operation gates after bounded domain validation."""

    db_pool = await get_db_pool()
    await consume_deferred_token_quota(request, db_pool=db_pool)
    await enforce_rbac_rate_limit(request, "prompts.improve", db_pool)
    await enforce_llm_budget(request)
    await _PROMPT_IMPROVEMENT_BILLING_CHECK(
        response=response,
        principal=principal,
        x_tldw_org_id=x_tldw_org_id,
        org_id=org_id,
    )


@router.get(
    "/capabilities",
    response_model=schemas.PromptCapabilitiesResponse,
    summary="Discover prompt feature capabilities",
    tags=["prompts"],
)
async def get_prompt_capabilities() -> schemas.PromptCapabilitiesResponse:
    """Return fail-closed Track A/Track B flags and centralized limits."""

    return schemas.PromptCapabilitiesResponse(
        prompt_improvement_v1=schemas.PromptImprovementCapability(
            supported=True,
            limits=schemas.PromptImprovementLimitsResponse(
                **asdict(PROMPT_IMPROVEMENT_LIMITS)
            ),
        ),
        single_text_recipe_v2=schemas.PromptRecipeCapability(supported=False),
    )


@router.post(
    "/improve",
    response_model=schemas.PromptImproveResponse,
    summary="Improve one isolated prompt draft with the active chat model",
    tags=["prompts"],
    responses=_PROMPT_IMPROVEMENT_ERROR_RESPONSES,
    dependencies=[
        Depends(_PROMPT_IMPROVEMENT_TOKEN_SECURITY),
        Depends(get_auth_principal),
    ],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": _inline_local_openapi_refs(
                        schemas.PromptImproveRequest.model_json_schema()
                    )
                }
            },
        }
    },
)
async def improve_prompt_endpoint(
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    routing_decision_store: InMemoryRoutingDecisionStore = Depends(
        get_request_routing_decision_store
    ),
    x_tldw_org_id: int | None = Header(default=None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(default=None),
) -> schemas.PromptImproveResponse | JSONResponse:
    """Generate and validate one draft-only candidate without persistence."""

    request_id = set_request_id()
    try:
        parsed = await _read_prompt_improvement_payload(request, request_id=request_id)
        if isinstance(parsed, JSONResponse):
            return parsed
        improvement_input = PromptImprovementInput(
            target=parsed.target,
            text=parsed.text,
            protected_tokens=tuple(parsed.protected_tokens),
        )
        validate_prompt_improvement_input(improvement_input)
    except PromptImprovementError as exc:
        return _prompt_improvement_error_response(
            exc.code,
            request_id=request_id,
            retryable=False,
        )
    except (AttributeError, KeyError, RecursionError, RuntimeError, TypeError, ValueError):
        return _prompt_improvement_error_response(
            "invalid_input",
            request_id=request_id,
            retryable=False,
        )

    await _run_prompt_improvement_post_validation_gates(
        request=request,
        response=response,
        principal=principal,
        x_tldw_org_id=x_tldw_org_id,
        org_id=org_id,
    )

    dispatch_result: PromptImprovementDispatchResult | None = None

    async def generate(messages: list[dict[str, str]]) -> str:
        nonlocal dispatch_result
        if dispatch_result is not None:
            raise PromptImprovementDispatchError("internal_error")
        dispatch_result = await dispatch_prompt_improvement(
            request=request,
            current_user=current_user,
            routing_decision_store=routing_decision_store,
            selected_model=parsed.model_selection.selected_model,
            provider_hint=parsed.model_selection.provider_hint,
            messages=messages,
            request_id=request_id,
            configured_providers_getter=get_configured_providers,
        )
        return dispatch_result.text

    try:
        result = await improve_prompt(
            improvement_input,
            generate=generate,
        )
        if dispatch_result is None:
            return _prompt_improvement_error_response(
                "internal_error",
                request_id=request_id,
                retryable=False,
            )
        return schemas.PromptImproveResponse(
            operation_id=parsed.operation_id,
            status=result.status,
            improved_text=result.improved_text,
            findings=[
                schemas.PromptImproveFinding(
                    category=finding.category,
                    issue=finding.issue,
                    change=finding.change,
                )
                for finding in result.findings
            ],
            review_required=result.review_required,
            warnings=list(result.warnings),
            resolved_model=schemas.PromptResolvedModel(
                provider=dispatch_result.provider,
                model=dispatch_result.model,
                display_name=dispatch_result.display_name,
            ),
            meta_prompt_version=result.meta_prompt_version,
        )
    except PromptImprovementDispatchError as exc:
        return _prompt_improvement_error_response(
            exc.code,
            request_id=request_id,
            retryable=exc.retryable,
            retry_after_seconds=exc.retry_after_seconds,
        )
    except PromptImprovementError as exc:
        return _prompt_improvement_error_response(
            exc.code,
            request_id=request_id,
            retryable=False,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return _prompt_improvement_error_response(
            "internal_error",
            request_id=request_id,
            retryable=False,
        )


@router.get(
    "/health",
    summary="Prompts service health",
    tags=["prompts"]
)
async def prompts_health():
    """Lightweight health endpoint for the Prompts subsystem."""
    import importlib
    import os
    from pathlib import Path

    from tldw_Server_API.app.core.config import settings

    health = {
        "service": "prompts",
        "status": "healthy",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "components": {}
    }

    try:
        base_dir = settings.get("USER_DB_BASE_DIR")
        exists = Path(base_dir).exists() if base_dir else False
        writable = False
        if exists:
            try:
                test_path = Path(base_dir) / ".prompts_health_check"
                with open(test_path, "w") as f:
                    f.write("ok")
                os.remove(test_path)
                writable = True
            except OSError:
                writable = False

        health["components"]["storage"] = {
            "base_dir": str(base_dir) if base_dir else None,
            "exists": exists,
            "writable": writable
        }

        # Library availability
        try:
            importlib.import_module("tldw_Server_API.app.core.DB_Management.Prompts_DB")
            lib_ok = True
        except ImportError:
            lib_ok = False
            health["components"]["library_error"] = "Prompts DB import failed"

        health["components"]["library"] = {"import_ok": lib_ok}

        if not base_dir or not exists or not lib_ok:
            health["status"] = "degraded"
        if base_dir and exists and not writable:
            health["status"] = "degraded"
    except _PROMPTS_LOOKUP_EXCEPTIONS:
        health["status"] = "unhealthy"
        health["error"] = "Prompts health check failed"

    return health

# --- Sync Log Endpoints ---
@router.get(
    "/sync-log",
    response_model=list[schemas.SyncLogEntryResponse],
    summary="Get sync log entries (admin/debug)",
    dependencies=[Depends(verify_prompts_auth)] # Should be admin-only
)
async def get_sync_log(
    since_change_id: int = Query(0, ge=0),
    limit: Optional[int] = Query(100, ge=1, le=1000),
    db: PromptsDatabase = Depends(get_prompts_db_for_user) # User specific sync log
):
    try:
        entries = db.get_sync_log_entries(since_change_id=since_change_id, limit=limit)
        return [schemas.SyncLogEntryResponse(**entry) for entry in entries]
    except DatabaseError as e:
        raise map_db_error_to_http(e, default_detail="Database error.") from e




# --- Search Endpoints ---
@router.post(
    "/search",
    response_model=schemas.PromptSearchResponse,
    summary="Search prompts",
    dependencies=[Depends(verify_prompts_user)]
)
async def search_all_prompts(
    search_query: str = Query(..., min_length=1, description="Search term(s)"),
    search_fields: Optional[list[str]] = Query(None, description="Fields to search: name, author, details, system_prompt, user_prompt, keywords"),
    page: int = Query(1, ge=1),
    results_per_page: int = Query(20, ge=1, le=100),
    include_deleted: bool = Query(False),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        results_list, total_matches = db.search_prompts(
            search_query=search_query,
            search_fields=search_fields,
            page=page,
            results_per_page=results_per_page,
            include_deleted=include_deleted
        )
        # Convert dicts to PromptSearchResultItem
        items = [schemas.PromptSearchResultItem(**item) for item in results_list]
        return schemas.PromptSearchResponse(
            items=items,
            total_matches=total_matches,
            page=page,
            per_page=results_per_page,
            pagination=build_page_pagination_meta(
                page=page,
                per_page=results_per_page,
                total=total_matches,
                total_pages=(total_matches + results_per_page - 1) // results_per_page,
            ),
        )
    except ValueError as e: # Bad page/per_page
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except DatabaseError as e:
        logger.error(f"Database error searching prompts: {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error during search.") from e


# === Keyword Endpoints ===
@router.post(
    "/keywords/",
    response_model=schemas.KeywordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new keyword",
    dependencies=[Depends(verify_prompts_user)]
)
async def create_keyword(
    keyword_data: schemas.KeywordCreate,
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Step 1: Check if an active keyword with this normalized text already exists.
        # The new DB method handles normalization internally.
        existing_active_keyword = db.get_active_keyword_by_text(keyword_data.keyword_text)

        if existing_active_keyword:
            # If it exists and is active, this endpoint should return a conflict.
            normalized_text = db._normalize_keyword(keyword_data.keyword_text) # For error message
            raise ConflictError(f"Keyword '{normalized_text}' already exists and is active.")

        # Step 2: If not actively existing, proceed to add (which might create or undelete).
        # db.add_keyword is "get or create or undelete".
        kw_id, kw_uuid = db.add_keyword(keyword_data.keyword_text)

        if not kw_id or not kw_uuid: # Should be rare if db.add_keyword is robust
            logger.error(f"db.add_keyword failed to return ID/UUID for '{keyword_data.keyword_text}' after pre-check.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create or retrieve keyword.")

        # Fetch the full details of the (potentially newly created or undeleted) keyword for the response.
        # To do this properly, we might need a get_keyword_by_id or get_keyword_by_uuid
        # For now, constructing from what we have. Prompts_DB.add_keyword normalizes.
        # API contract: return normalized, lowercased keyword text while DB preserves original casing
        final_keyword_text = db._normalize_keyword(keyword_data.keyword_text).lower()

        return schemas.KeywordResponse(
            id=kw_id,
            uuid=kw_uuid,
            keyword_text=final_keyword_text
        )
    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(f"Database error creating keyword: {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error creating keyword: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred") from e


@router.get(
    "/keywords/",
    response_model=list[str], # Just a list of keyword strings
    summary="List all active keywords",
    dependencies=[Depends(verify_prompts_user)]
)
async def list_all_keywords(
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # API contract: return lowercased normalized keyword strings
        try:
            kws = db.fetch_all_keywords(include_deleted=False)
            return [db._normalize_keyword(k).lower() for k in kws]
        except (AttributeError, TypeError):
            # Fallback if normalization method is unavailable
            return db.fetch_all_keywords(include_deleted=False)
    except DatabaseError as e:
        logger.error(f"Database error listing keywords: {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.delete(
    "/keywords/{keyword_text}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft delete a keyword",
    dependencies=[Depends(verify_prompts_user)]
)
async def delete_keyword(
    keyword_text: str,
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
) -> Response:
    try:
        success = db.soft_delete_keyword(keyword_text)
        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Keyword not found or already deleted.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(f"Database error deleting keyword '{keyword_text}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


# === Export Endpoints ===

@router.get(
    "/export",
    response_model=schemas.ExportResponse, # Returns message and base64 content
    summary="Export prompts to CSV or Markdown (as base64 string)",
    dependencies=[Depends(verify_prompts_user)]
)
async def export_prompts_api(
    export_format: str = Query("csv", enum=["csv", "markdown"]),
    filter_keywords: Optional[list[str]] = Query(None),
    include_system: bool = Query(True),
    include_user: bool = Query(True),
    include_details: bool = Query(True),
    include_author: bool = Query(True),
    include_associated_keywords: bool = Query(True),
    markdown_template_name: Optional[str] = Query("Basic Template"),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Use the standalone function from prompts_interop (or Prompts_DB_v2)
        # It needs the db_instance.
        status_msg, file_path_or_content = db_export_prompts_formatted(
            db_instance=db, # Pass the user-specific DB instance
            export_format=export_format,
            filter_keywords=filter_keywords,
            include_system=include_system,
            include_user=include_user,
            include_details=include_details,
            include_author=include_author,
            include_associated_keywords=include_associated_keywords,
            markdown_template_name=markdown_template_name
        )

        if file_path_or_content == "None" or not os.path.exists(file_path_or_content):
            if "No prompts found" in status_msg:
                 return schemas.ExportResponse(message=status_msg, file_content_b64=None)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Export failed")

        with open(file_path_or_content, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode('utf-8')

        # Clean up the temporary file
        try:
            os.remove(file_path_or_content)
        except OSError as e_remove:
            logger.warning(f"Could not remove temporary export file {file_path_or_content}: {e_remove}")

        return schemas.ExportResponse(message=status_msg, file_content_b64=file_b64)

    except ValueError as e: # Invalid export format etc.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except DatabaseError as e:
        logger.error(f"Database error during export: {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error during export.") from e
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error during export: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during export") from e


@router.get(
    "/keywords/export-csv",
    response_model=schemas.ExportResponse,
    summary="Export all prompt keywords with associations to CSV (as base64 string)",
    dependencies=[Depends(verify_prompts_user)]
)
async def export_keywords_api(
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        status_msg, file_path = db_export_prompt_keywords_to_csv(db_instance=db)
        if file_path == "None" or not os.path.exists(file_path):
            if "Successfully exported 0 active prompt keywords" in status_msg or "No active keywords found" in status_msg : # Adjusted condition for empty export
                 return schemas.ExportResponse(message=status_msg, file_content_b64=None)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Keyword export failed")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode('utf-8')
        try:
            os.remove(file_path)
        except OSError as e:
            logger.debug(f"Failed to remove temporary export file {file_path}: {e}")
        return schemas.ExportResponse(message=status_msg, file_content_b64=file_b64)
    except DatabaseError as e:
        raise map_db_error_to_http(e, default_detail="Database error during keyword export.") from e
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error during keyword export: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during keyword export") from e


# === Import Endpoints ===

@router.post(
    "/import",
    response_model=schemas.PromptImportResponse,
    summary="Import prompts from JSON",
    dependencies=[Depends(verify_prompts_user)]
)
async def import_prompts_api(
    payload: schemas.PromptImportRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        try:
            used_names = set(db.fetch_all_prompt_names(include_deleted=True))
        except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
            logger.warning(f"Failed to fetch existing prompt names for import: {e}")
            used_names = set()

        name_counts: dict[str, int] = {}
        imported = 0
        failed = 0
        skipped = 0
        prompt_ids: list[int] = []

        for prompt in payload.prompts:
            base_name = (prompt.name or "").strip()
            details = prompt.details if prompt.details is not None else prompt.content
            if not details and details != "":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each imported prompt must include content or details.",
                )

            if base_name in used_names:
                if payload.skip_duplicates:
                    skipped += 1
                    continue
                candidate_name = _generate_unique_prompt_name(base_name, used_names, name_counts)
            else:
                candidate_name = base_name

            used_names.add(candidate_name)
            try:
                p_id, _uuid, _msg = db.add_prompt(
                    name=candidate_name,
                    author=prompt.author,
                    details=details,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    keywords=prompt.keywords or [],
                    overwrite=False,
                )
                if p_id:
                    imported += 1
                    prompt_ids.append(int(p_id))
                else:
                    failed += 1
            except ConflictError:
                if payload.skip_duplicates:
                    skipped += 1
                else:
                    failed += 1
            except (InputError, DatabaseError) as e:
                logger.warning(f"Import failed for prompt '{base_name}': {e}")
                failed += 1

        return schemas.PromptImportResponse(
            imported=imported,
            failed=failed,
            skipped=skipped,
            prompt_ids=prompt_ids
        )
    except HTTPException:
        raise
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error during import: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during import"
        ) from e


# === Template Processing Endpoints ===

@router.post(
    "/templates/variables",
    response_model=schemas.TemplateVariablesResponse,
    summary="Extract template variables",
    dependencies=[Depends(verify_prompts_user)]
)
async def extract_template_variables_api(
    payload: schemas.TemplateVariablesRequest = Body(...)
):
    variables = _extract_template_variables(payload.template)
    return schemas.TemplateVariablesResponse(variables=variables)


@router.post(
    "/templates/render",
    response_model=schemas.TemplateRenderResponse,
    summary="Render a template with variables",
    dependencies=[Depends(verify_prompts_user)]
)
async def render_template_api(
    payload: schemas.TemplateRenderRequest = Body(...)
):
    try:
        rendered = _render_template(payload.template, payload.variables)
    except KeyError as e:
        missing_key = e.args[0] if e.args else "unknown"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing template variable: {missing_key}"
        ) from e
    return schemas.TemplateRenderResponse(rendered=rendered)


@router.post(
    "/preview",
    response_model=schemas.StructuredPromptPreviewResponse,
    summary="Preview assembled prompt messages",
    dependencies=[Depends(verify_prompts_user)],
)
async def preview_prompt_api(
    payload: schemas.StructuredPromptPreviewRequest = Body(...),
):
    try:
        definition, prompt_format, prompt_schema_version = _coerce_preview_definition(
            payload.prompt_format,
            payload.prompt_schema_version,
            payload.prompt_definition,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
        )
        assembly = assemble_prompt_definition(definition, payload.variables)
    except InputError as e:
        raise map_db_error_to_http(e) from e
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

    return schemas.StructuredPromptPreviewResponse(
        prompt_format=prompt_format,
        prompt_schema_version=prompt_schema_version,
        assembled_messages=assembly.messages,
        legacy_system_prompt=assembly.legacy.system_prompt,
        legacy_user_prompt=assembly.legacy.user_prompt,
    )


@router.post(
    "/convert",
    response_model=schemas.StructuredPromptConvertResponse,
    summary="Convert a legacy prompt to a structured definition",
    dependencies=[Depends(verify_prompts_user)],
)
async def convert_prompt_api(
    payload: schemas.StructuredPromptConvertRequest = Body(...),
):
    definition = convert_legacy_prompt_to_definition(
        system_prompt=payload.system_prompt,
        user_prompt=payload.user_prompt,
    )
    legacy_system_prompt, legacy_user_prompt = _render_definition_legacy_fields(definition)
    return schemas.StructuredPromptConvertResponse(
        prompt_definition=definition.model_dump(),
        extracted_variables=extract_legacy_prompt_variables(
            payload.system_prompt,
            payload.user_prompt,
        ),
        legacy_system_prompt=legacy_system_prompt,
        legacy_user_prompt=legacy_user_prompt,
    )


# === Bulk Operations Endpoints ===

@router.post(
    "/bulk/delete",
    response_model=schemas.PromptBulkDeleteResponse,
    summary="Bulk delete prompts",
    dependencies=[Depends(verify_prompts_user)]
)
async def bulk_delete_prompts(
    payload: schemas.PromptBulkDeleteRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    deleted = 0
    failed_ids: list[int] = []
    for prompt_id in payload.prompt_ids:
        try:
            if db.soft_delete_prompt(prompt_id):
                deleted += 1
            else:
                failed_ids.append(int(prompt_id))
        except (ConflictError, DatabaseError) as e:
            logger.warning(f"Bulk delete failed for prompt {prompt_id}: {e}")
            failed_ids.append(int(prompt_id))
    return schemas.PromptBulkDeleteResponse(
        deleted=deleted,
        failed=len(failed_ids),
        failed_ids=failed_ids
    )


@router.post(
    "/bulk/keywords",
    response_model=schemas.PromptBulkKeywordsResponse,
    summary="Bulk update prompt keywords",
    dependencies=[Depends(verify_prompts_user)]
)
async def bulk_update_prompt_keywords(
    payload: schemas.PromptBulkKeywordsRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    if not payload.add_keywords and not payload.remove_keywords:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of add_keywords or remove_keywords must be provided."
        )
    updated = 0
    failed_ids: list[int] = []

    def _normalize_for_compare(value: str) -> str:
        try:
            return db._normalize_keyword(value).casefold()
        except (AttributeError, TypeError, ValueError):
            return str(value).strip().casefold()

    remove_set = {
        _normalize_for_compare(k)
        for k in payload.remove_keywords
        if isinstance(k, str) and k.strip()
    }

    for prompt_id in payload.prompt_ids:
        try:
            prompt = db.fetch_prompt_details(prompt_id)
            if not prompt:
                failed_ids.append(int(prompt_id))
                continue
            current_keywords = db.fetch_keywords_for_prompt(int(prompt_id), include_deleted=False)
            filtered = [
                k for k in current_keywords
                if _normalize_for_compare(k) not in remove_set
            ]
            existing_norms = {_normalize_for_compare(k) for k in filtered}
            for kw in payload.add_keywords:
                if not isinstance(kw, str) or not kw.strip():
                    continue
                normalized_kw = db._normalize_keyword(kw)
                norm_key = _normalize_for_compare(normalized_kw)
                if norm_key not in existing_norms:
                    filtered.append(normalized_kw)
                    existing_norms.add(norm_key)
            db.update_keywords_for_prompt(int(prompt_id), filtered)
            updated += 1
        except (InputError, DatabaseError) as e:
            logger.warning(f"Bulk keyword update failed for prompt {prompt_id}: {e}")
            failed_ids.append(int(prompt_id))
    return schemas.PromptBulkKeywordsResponse(
        updated=updated,
        failed=len(failed_ids),
        failed_ids=failed_ids
    )


# === Prompt Endpoints ===

# Legacy-compatible create route for tests expecting /api/v1/prompts/create
@router.post(
    "/create",
    summary="Create a prompt (legacy payload)",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_prompts_user)]
)
async def legacy_create_prompt(
    payload: schemas.LegacyPromptCreateRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        name = payload.name
        author = payload.author
        # Legacy tests use "content" instead of details
        details = payload.effective_details
        keywords = payload.keywords or []
        p_id, _uuid, _msg = db.add_prompt(
            name=name,
            author=author,
            details=details,
            system_prompt=payload.system_prompt,
            user_prompt=payload.user_prompt,
            keywords=keywords,
            overwrite=False,
        )
        if not p_id:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create prompt")
        # Legacy response uses prompt_id
        return {"prompt_id": p_id}
    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(f"Database error creating prompt (legacy): {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error creating prompt (legacy): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error.") from e

@router.post(
    "/",
    response_model=schemas.PromptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt",
    dependencies=[Depends(verify_prompts_user)]
)
@router.post(
    "",
    response_model=schemas.PromptResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt [no-slash alias]",
    dependencies=[Depends(verify_prompts_user)]
)
async def create_prompt(
    prompt_data: schemas.PromptCreate,
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        storage_payload = _prepare_prompt_storage_payload(prompt_data)
        # The db.add_prompt method with overwrite=False should raise ConflictError
        # if the name already exists and is active (as per our DB layer modification).
        p_id, p_uuid, db_message = db.add_prompt(  # db_message is returned by add_prompt on success
            name=storage_payload["name"],
            author=storage_payload.get("author"),
            details=storage_payload.get("details"),
            system_prompt=storage_payload.get("system_prompt"),
            user_prompt=storage_payload.get("user_prompt"),
            prompt_format=storage_payload.get("prompt_format", "legacy"),
            prompt_schema_version=storage_payload.get("prompt_schema_version"),
            prompt_definition=storage_payload.get("prompt_definition"),
            keywords=storage_payload.get("keywords"),
            overwrite=False  # For a POST/create, we don't want to overwrite.
        )
        # If add_prompt successfully created or undeleted (if that's its logic for overwrite=False and deleted=True)
        # then p_id and p_uuid will be set.

        # The 'msg' variable was causing the NameError.
        # db.add_prompt returns (id, uuid, message_string)
        # We can use db_message for logging if needed.

        if not p_id or not p_uuid:  # Should ideally not be hit if add_prompt raises on failure
            logger.error(
                f"Failed to create prompt '{prompt_data.name}', add_prompt returned: {p_id}, {p_uuid}, {db_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create prompt",
            )

        created_prompt_dict = db.fetch_prompt_details(p_uuid)  # Fetch by UUID to be sure
        if not created_prompt_dict:
            logger.error(f"Could not fetch newly created prompt by UUID {p_uuid}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Prompt created but could not be retrieved.")

        # Ensure 'deleted' field is populated if the schema expects it
        if 'deleted' not in created_prompt_dict and schemas.PromptResponse.model_fields.get('deleted'):
            created_prompt_dict['deleted'] = False  # Default for new prompts

        return schemas.PromptResponse(**created_prompt_dict)

    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(f"Database error creating prompt: {e}", exc_info=True)
        raise map_db_error_to_http(
            e,
            default_detail="Database error during prompt creation.",
        ) from e
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:  # Catch-all for other unexpected errors
        logger.error(f"Unexpected error creating prompt: {e}", exc_info=True)
        # Avoid leaking the raw 'msg' variable if it was a NameError
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.") from e


# === Collection Endpoints ===

# NOTE:
# Keep these static `/collections*` routes above the dynamic `/{prompt_identifier}`
# route declarations below. FastAPI matches in declaration order; moving the dynamic
# route above these will cause `/collections` to be interpreted as a prompt identifier.

@router.post(
    "/collections/create",
    response_model=schemas.PromptCollectionCreateResponse,
    summary="Create a prompt collection",
    dependencies=[Depends(verify_prompts_user)],
)
async def create_collection(
    payload: schemas.PromptCollectionCreateRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    try:
        created = db.create_prompt_collection(
            name=payload.name,
            description=payload.description,
            prompt_ids=payload.prompt_ids or [],
        )
        return schemas.PromptCollectionCreateResponse(collection_id=created["collection_id"])
    except (InputError, DatabaseError) as e:
        logger.error(f"Database error creating prompt collection '{payload.name}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.get(
    "/collections",
    response_model=schemas.PromptCollectionListResponse,
    summary="List prompt collections",
    dependencies=[Depends(verify_prompts_user)],
)
async def list_collections(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    try:
        items = db.list_prompt_collections(limit=limit, offset=offset)
        total = db.count_prompt_collections()
        return schemas.PromptCollectionListResponse(
            collections=[schemas.PromptCollectionResponse(**item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(items),
            ),
        )
    except (InputError, DatabaseError) as e:
        logger.error("Database error listing prompt collections: {}", e, exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.put(
    "/collections/{collection_id}",
    response_model=schemas.PromptCollectionResponse,
    summary="Update a prompt collection",
    dependencies=[Depends(verify_prompts_user)],
)
async def update_collection(
    collection_id: int,
    payload: schemas.PromptCollectionUpdateRequest = Body(...),
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    try:
        item = db.update_prompt_collection(
            collection_id,
            name=payload.name,
            description=payload.description,
            prompt_ids=payload.prompt_ids,
        )
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        return schemas.PromptCollectionResponse(**item)
    except (InputError, DatabaseError) as e:
        logger.error(f"Database error updating prompt collection '{collection_id}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.get(
    "/collections/{collection_id}",
    response_model=schemas.PromptCollectionResponse,
    summary="Get a prompt collection",
    dependencies=[Depends(verify_prompts_user)],
)
async def get_collection(
    collection_id: int,
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    try:
        item = db.get_prompt_collection_by_id(collection_id)
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        return schemas.PromptCollectionResponse(**item)
    except (InputError, DatabaseError) as e:
        logger.error(f"Database error fetching prompt collection '{collection_id}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.get(
    "/",
    response_model=schemas.PaginatedPromptsResponse,
    summary="List all prompts (paginated)",
    dependencies=[Depends(verify_prompts_user)]
)
@router.get(
    "",
    response_model=schemas.PaginatedPromptsResponse,
    summary="List all prompts (paginated) [no-slash alias]",
    dependencies=[Depends(verify_prompts_user)]
)
async def list_all_prompts(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    include_deleted: bool = Query(False, description="Include soft-deleted prompts"),
    sort_by: str = Query(
        "last_modified",
        description="Sort by: last_modified, name, author, id, usage_count, last_used_at",
    ),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        items_dict_list, total_pages, current_page, total_items = db.list_prompts(
            page=page,
            per_page=per_page,
            include_deleted=include_deleted,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        # Convert list of dicts to list of PromptBriefResponse
        brief_items = [schemas.PromptBriefResponse(**item) for item in items_dict_list]
        return schemas.PaginatedPromptsResponse(
            items=brief_items,
            total_pages=total_pages,
            current_page=current_page,
            total_items=total_items,
            pagination=build_page_pagination_meta(
                page=current_page,
                per_page=per_page,
                total=total_items,
                total_pages=total_pages,
            ),
        )
    except ValueError as e: # For bad page/per_page from DB layer
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except DatabaseError as e:
        logger.error(f"Database error listing prompts: {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error listing prompts.") from e


@router.get(
    "/{prompt_identifier}",
    response_model=schemas.PromptResponse,
    summary="Get a specific prompt by ID, UUID, or Name",
    dependencies=[Depends(verify_prompts_user)]
)
async def get_prompt(
    prompt_identifier: Union[int, str], # Path param will be string, FastAPI can convert to int if possible
    include_deleted: bool = Query(False, description="Include if soft-deleted"),
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        # Attempt to convert to int if it looks like an ID
        processed_identifier: Union[int, str] = prompt_identifier
        try:
            processed_identifier = int(prompt_identifier)
        except ValueError:
            pass # Keep as string if not an int (name or UUID)

        prompt_details = db.fetch_prompt_details(processed_identifier, include_deleted=include_deleted)
        if not prompt_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found.")
        return schemas.PromptResponse(**prompt_details)
    except DatabaseError as e:
        logger.error(f"Database error getting prompt '{prompt_identifier}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.put(
    "/{prompt_identifier}",
    response_model=schemas.PromptResponse,
    summary="Update an existing prompt (or create if name matches and overwrite=true logic used)",
    dependencies=[Depends(verify_prompts_user)]
)
async def update_prompt(
    prompt_identifier: Union[int, str],
    prompt_data: schemas.PromptCreate, # Using PromptCreate for full replacement, or PromptUpdate for partial
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    # This uses add_prompt with overwrite=True logic.
    # For a true PATCH, you'd need a different DB method.
    # The prompt_identifier is used to ensure we are updating the one intended if name changes.
    try:
        # 1. Resolve identifier to actual prompt ID
        target_prompt_dict = db.fetch_prompt_details(prompt_identifier,
                                                     include_deleted=True)  # Allow updating soft-deleted
        if not target_prompt_dict:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Prompt with identifier '{prompt_identifier}' not found.")

        prompt_id_to_update = target_prompt_dict['id']

        # 2. Call the new update method
        # Convert Pydantic model to dict, excluding unset to allow partial-like updates if some fields are optional
        update_input = prompt_data.model_dump(exclude_unset=True)
        update_payload_dict = _prepare_prompt_storage_payload(
            update_input,
            existing_prompt=target_prompt_dict,
        )

        updated_prompt_uuid, msg = db.update_prompt_by_id(prompt_id_to_update, update_payload_dict)

        if not updated_prompt_uuid:
            # This case should be rare if fetch_prompt_details found it, unless db.update_prompt_by_id returns None for "no changes"
            logger.error(
                f"Update for prompt identifier '{prompt_identifier}' (ID: {prompt_id_to_update}) resulted in no UUID: {msg}")
            # Determine appropriate HTTP status based on msg
            if "not found" in msg.lower():  # Should have been caught by fetch_prompt_details
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Prompt update failed")

        # Fetch the fully updated prompt to return
        final_updated_prompt = db.fetch_prompt_details(updated_prompt_uuid)  # Fetch by UUID
        if not final_updated_prompt:
            logger.error(f"Could not retrieve prompt by UUID {updated_prompt_uuid} after update.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Prompt updated but could not be retrieved.")

        if 'deleted' not in final_updated_prompt and hasattr(schemas.PromptResponse, 'deleted'):
            final_updated_prompt['deleted'] = False

        return schemas.PromptResponse(**final_updated_prompt)

    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(f"Database error updating prompt '{prompt_identifier}': {e}", exc_info=True)
        raise map_db_error_to_http(
            e,
            default_detail="Database error during prompt update.",
        ) from e
    except HTTPException:  # Re-raise
        raise
    except _PROMPTS_DB_OPERATION_EXCEPTIONS as e:
        logger.error(f"Unexpected error updating prompt '{prompt_identifier}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during prompt update.") from e


@router.post(
    "/{prompt_identifier}/use",
    response_model=schemas.PromptResponse,
    summary="Record prompt usage",
    dependencies=[Depends(verify_prompts_user)],
)
async def record_prompt_usage(
    prompt_identifier: Union[int, str],
    db: PromptsDatabase = Depends(get_prompts_db_for_user),
):
    try:
        processed_identifier: Union[int, str] = prompt_identifier
        with contextlib.suppress(ValueError):
            processed_identifier = int(prompt_identifier)

        updated_prompt = db.record_prompt_usage(processed_identifier)
        if not updated_prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt not found.",
            )
        if "deleted" not in updated_prompt and hasattr(schemas.PromptResponse, "deleted"):
            updated_prompt["deleted"] = False
        return schemas.PromptResponse(**updated_prompt)
    except (InputError, ConflictError, DatabaseError) as e:
        logger.error(
            f"Database error recording prompt usage for '{prompt_identifier}': {e}",
            exc_info=True,
        )
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.delete(
    "/{prompt_identifier}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Soft delete a prompt",
    dependencies=[Depends(verify_prompts_user)]
)
async def delete_prompt(
    prompt_identifier: Union[int, str],
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
) -> Response:
    try:
        processed_identifier: Union[int, str] = prompt_identifier
        with contextlib.suppress(ValueError):
            processed_identifier = int(prompt_identifier)

        success = db.soft_delete_prompt(processed_identifier)
        if not success:
            # Could be not found or already deleted, DB layer logs warning
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found or already deleted.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except (ConflictError, DatabaseError) as e:
        logger.error(f"Database error deleting prompt '{prompt_identifier}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


# === Version Endpoints ===

@router.get(
    "/{prompt_identifier}/versions",
    response_model=list[schemas.PromptVersionResponse],
    summary="List prompt versions",
    dependencies=[Depends(verify_prompts_user)]
)
async def list_prompt_versions(
    prompt_identifier: Union[int, str],
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        prompt_details = db.fetch_prompt_details(prompt_identifier, include_deleted=True)
        if not prompt_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found.")
        versions = db.get_prompt_versions(int(prompt_details["id"]))
        return [schemas.PromptVersionResponse(**entry) for entry in versions]
    except DatabaseError as e:
        logger.error(f"Database error listing versions for prompt '{prompt_identifier}': {e}", exc_info=True)
        raise map_db_error_to_http(e, default_detail="Database error.") from e


@router.post(
    "/{prompt_identifier}/versions/{version}/restore",
    response_model=schemas.PromptResponse,
    summary="Restore a prompt to a previous version",
    dependencies=[Depends(verify_prompts_user)]
)
async def restore_prompt_version(
    prompt_identifier: Union[int, str],
    version: int,
    db: PromptsDatabase = Depends(get_prompts_db_for_user)
):
    try:
        prompt_details = db.fetch_prompt_details(prompt_identifier, include_deleted=True)
        if not prompt_details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found.")

        updated_uuid, _msg = db.restore_prompt_version(int(prompt_details["id"]), version)
        if updated_uuid:
            updated_prompt = db.fetch_prompt_details(updated_uuid, include_deleted=True)
        else:
            updated_prompt = db.fetch_prompt_details(int(prompt_details["id"]), include_deleted=True)

        if not updated_prompt:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt not found after restore.")

        if 'deleted' not in updated_prompt and hasattr(schemas.PromptResponse, 'deleted'):
            updated_prompt['deleted'] = False

        return schemas.PromptResponse(**updated_prompt)
    except InputError as e:
        raise map_db_error_to_http(
            e,
            default_detail="Database error.",
            not_found_substrings=("not found",),
        ) from e
    except (ConflictError, DatabaseError) as e:
        raise map_db_error_to_http(e, default_detail="Database error.") from e

#
# End of prompts.py
#######################################################################################################################
