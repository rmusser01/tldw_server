"""
Workflows API (v0.1 scaffolding)

Implements minimal definition CRUD and run lifecycle with a no-op engine.
"""

import asyncio
import errno
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, RequirePermission, RequireRole, resolve_user_id_for_request, TokenScopeGuard, User

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_cursor_pagination_meta,
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.workflows import (
    AdhocRunRequest,
    EventResponse,
    RunRequest,
    WorkflowDefinitionCreate,
    WorkflowDefinitionResponse,
    WorkflowPreflightRequest,
    WorkflowPreflightResponse,
    WorkflowRunInvestigationResponse,
    WorkflowRagSearchConfig,
    WorkflowRunListItem,
    WorkflowRunListResponse,
    WorkflowRunResponse,
    WorkflowRunStepsResponse,
    WorkflowStepAttemptsResponse,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
    AuditSeverity,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.permissions import (
    WORKFLOWS_ADMIN,
    WORKFLOWS_RUNS_CONTROL,
    WORKFLOWS_RUNS_READ,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_workflows_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.exceptions import (
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.http_client import RetryPolicy as _RetryPolicy
from tldw_Server_API.app.core.http_client import afetch as _http_afetch
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
from tldw_Server_API.app.core.Resource_Governance.deps import derive_entity_key
from tldw_Server_API.app.core.Resource_Governance.governor import RGRequest
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
    is_truthy,
)
from tldw_Server_API.app.services.workflows_webhook_dlq_service import (
    record_webhook_delivery_event,
)
from tldw_Server_API.app.core.Workflows import RunMode, WorkflowEngine, WorkflowScheduler
from tldw_Server_API.app.core.Workflows.adapters._common import artifacts_base_dir, is_subpath
from tldw_Server_API.app.core.Workflows.capabilities import get_step_capability
from tldw_Server_API.app.core.Workflows.adapters._registry import get_parallelizable
from tldw_Server_API.app.core.Workflows.adapters.research._config import (
    DEEP_RESEARCH_CANONICAL_BUNDLE_FIELDS,
    DeepResearchConfig,
    DeepResearchLoadBundleConfig,
    DeepResearchSelectBundleFieldsConfig,
    DeepResearchWaitConfig,
)
from tldw_Server_API.app.core.Workflows.daily_ledger import (
    backfill_legacy_runs_to_ledger,
    get_workflows_daily_ledger,
    record_workflow_run,
    workflows_ledger_category,
)
from tldw_Server_API.app.core.Workflows.investigation import build_run_investigation
from tldw_Server_API.app.core.Workflows.investigation import list_run_steps as build_run_steps
from tldw_Server_API.app.core.Workflows.investigation import list_step_attempts as build_step_attempts
from tldw_Server_API.app.core.Workflows.registry import StepTypeRegistry

_WORKFLOWS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
    ValidationError,
    HTTPException,
    EgressPolicyError,
    NetworkError,
    RetryExhaustedError,
)

# Best-effort per-process cache for "did we backfill today" keys.
# Keep it bounded to avoid unbounded growth in long-lived workers.
_WORKFLOWS_BACKFILL_CACHE: set[str] = set()
_raw_cache_max = int(os.getenv("WORKFLOWS_BACKFILL_CACHE_MAX", "50000") or "50000")
_WORKFLOWS_BACKFILL_CACHE_MAX = max(1, _raw_cache_max)
_WORKFLOWS_QUOTA_RG_FALLBACK_LOGGED = False


def _log_workflows_quota_rg_fallback_once(*, reason: str, policy_id: str) -> None:
    global _WORKFLOWS_QUOTA_RG_FALLBACK_LOGGED
    if _WORKFLOWS_QUOTA_RG_FALLBACK_LOGGED:
        return
    _WORKFLOWS_QUOTA_RG_FALLBACK_LOGGED = True
    logger.error(
        "Workflows daily-cap RG check unavailable; using diagnostics-only shim (no legacy fallback enforcement). "
        "reason={} policy_id={}",
        reason,
        policy_id,
    )


def _utcnow_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat()


def _normalize_claim_values(raw: Any) -> list[str]:
    values = raw if isinstance(raw, (list, tuple, set)) else ([raw] if raw is not None else [])
    out: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text:
            out.append(text)
    return out


def _is_workflows_admin_user(current_user: User) -> bool:
    """
    Resolve workflows admin authorization from explicit role/permission claims.

    Legacy profile booleans/columns like ``is_admin`` are intentionally not
    trusted for cross-user workflows access.
    """
    try:
        if "admin" in _normalize_claim_values(getattr(current_user, "roles", [])):
            return True
        permission_values = _normalize_claim_values(getattr(current_user, "permissions", []))
        if WORKFLOWS_ADMIN.lower() in permission_values:
            return True
        if "*" in permission_values:
            return True
        if "system.configure" in permission_values:
            return True
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _get_authorized_run_or_404(
    *,
    run_id: str,
    current_user: User,
    db: WorkflowsDatabase,
) -> tuple[Any, bool]:
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    return run, is_admin


router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


def _get_db() -> WorkflowsDatabase:
    backend = get_content_backend_instance()
    return create_workflows_database(backend=backend)


MAX_DEFINITION_BYTES = 256 * 1024
MAX_STEPS = 50
MAX_STEP_CONFIG_BYTES = 32 * 1024

_JSONSCHEMA_REQUIRED_RE = re.compile(r"'([^']+)' is a required property")
_JSONSCHEMA_ADDITIONAL_RE = re.compile(r"'([^']+)' was unexpected")


def _safe_jsonschema_detail(exc: Exception) -> str:
    try:
        from jsonschema.exceptions import ValidationError  # type: ignore
    except ImportError:
        return "schema validation failed"
    if not isinstance(exc, ValidationError):
        return "schema validation failed"

    path = ".".join(str(p) for p in exc.path) if exc.path else ""
    validator = str(exc.validator or "")
    message = None

    if validator == "required":
        missing = None
        match = _JSONSCHEMA_REQUIRED_RE.search(str(exc.message))
        if match:
            missing = match.group(1)
        message = f"missing required field '{missing}'" if missing else "missing required field"
    elif validator == "additionalProperties":
        extra = None
        match = _JSONSCHEMA_ADDITIONAL_RE.search(str(exc.message))
        if match:
            extra = match.group(1)
        message = f"unknown field '{extra}'" if extra else "unknown field"
    elif validator == "type":
        expected = exc.validator_value
        message = f"expected type '{expected}'" if expected else "invalid type"

    if message is None:
        message = " ".join(str(exc.message or "schema validation failed").split())
        if len(message) > 200:
            message = f"{message[:200]}..."

    if path:
        return f"{message} at '{path}'"
    return message


def _pydantic_error_detail(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []) if p is not None)
        msg = err.get("msg") or "invalid"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid config"


def _find_api_key_path(value: Any, path: str = "") -> Optional[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key == "api_key":
                return next_path
            found = _find_api_key_path(child, next_path)
            if found:
                return found
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_path = f"{path}[{idx}]"
            found = _find_api_key_path(child, next_path)
            if found:
                return found
    return None


def _find_signing_secret_path(value: Any, path: str = "") -> Optional[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key == "signing" and isinstance(child, dict) and "secret" in child:
                return f"{next_path}.secret"
            found = _find_signing_secret_path(child, next_path)
            if found:
                return found
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_path = f"{path}[{idx}]"
            found = _find_signing_secret_path(child, next_path)
            if found:
                return found
    return None


def _llm_step_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "LLM step config. Provide provider API keys via runtime secret refs; "
            "api_key is not stored in workflow definitions."
        ),
        "properties": {
            "provider": {"type": "string"},
            "api_provider": {"type": "string"},
            "api_endpoint": {"type": "string"},
            "model": {"type": "string"},
            "model_id": {"type": "string"},
            "prompt": {"type": "string"},
            "input": {"type": "string"},
            "template": {"type": "string"},
            "messages": {"type": ["array", "string"]},
            "messages_payload": {"type": ["array", "string"]},
            "system_message": {"type": "string"},
            "system": {"type": "string"},
            "system_prompt": {"type": "string"},
            "temperature": {"type": "number"},
            "top_p": {"type": "number"},
            "max_tokens": {"type": "integer", "minimum": 1},
            "max_completion_tokens": {"type": "integer", "minimum": 1},
            "stop": {"type": ["string", "array"]},
            "tools": {"type": "array"},
            "tool_choice": {},
            "response_format": {"type": "object"},
            "seed": {"type": "integer"},
            "stream": {"type": "boolean"},
            "include_response": {"type": "boolean"},
            "n": {"type": "integer", "minimum": 1},
            "logit_bias": {"type": "object"},
            "user": {"type": ["string", "integer"]},
        },
        "required": [],
        "additionalProperties": True,
    }


def _rag_search_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "sources": {"type": ["array", "string"], "items": {"type": "string"}},
            "search_mode": {"type": "string", "enum": ["fts", "vector", "hybrid"]},
            "hybrid_alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "expand_query": {"type": "boolean"},
            "expansion_strategies": {
                "type": ["array", "string"],
                "items": {"type": "string", "enum": ["acronym", "synonym", "domain", "entity"]},
            },
            "spell_check": {"type": "boolean"},
            "enable_cache": {"type": "boolean"},
            "cache_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "adaptive_cache": {"type": "boolean"},
            "cache_ttl": {"type": "integer", "minimum": 0},
            "enable_table_processing": {"type": "boolean"},
            "table_method": {"type": "string", "enum": ["markdown", "html", "hybrid"]},
            "include_sibling_chunks": {"type": "boolean"},
            "sibling_window": {"type": "integer", "minimum": 0, "maximum": 20},
            "enable_parent_expansion": {"type": "boolean"},
            "include_parent_document": {"type": "boolean"},
            "parent_max_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
            "enable_reranking": {"type": "boolean"},
            "reranking_strategy": {"type": "string", "enum": ["flashrank", "cross_encoder", "hybrid", "none"]},
            "rerank_top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            "enable_citations": {"type": "boolean"},
            "citation_style": {"type": "string", "enum": ["apa", "mla", "chicago", "harvard", "ieee"]},
            "include_page_numbers": {"type": "boolean"},
            "enable_chunk_citations": {"type": "boolean"},
            "enable_generation": {"type": "boolean"},
            "generation_model": {"type": "string"},
            "generation_prompt": {"type": "string"},
            "max_generation_tokens": {"type": "integer", "minimum": 1, "maximum": 8192},
            "enable_security_filter": {"type": "boolean"},
            "detect_pii": {"type": "boolean"},
            "redact_pii": {"type": "boolean"},
            "sensitivity_level": {"type": "string", "enum": ["public", "internal", "confidential", "restricted"]},
            "content_filter": {"type": "boolean"},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
            "highlight_results": {"type": "boolean"},
            "highlight_query_terms": {"type": "boolean"},
            "track_cost": {"type": "boolean"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _deep_research_step_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Launches a deep research session and returns its run reference; "
            "does not wait for completion."
        ),
        "properties": {
            "query": {
                "type": "string",
                "description": "Templated research query resolved from workflow context",
            },
            "source_policy": {
                "type": "string",
                "enum": [
                    "balanced",
                    "local_first",
                    "external_first",
                    "local_only",
                    "external_only",
                ],
                "default": "balanced",
            },
            "autonomy_mode": {
                "type": "string",
                "enum": ["checkpointed", "autonomous"],
                "default": "checkpointed",
            },
            "limits_json": {
                "type": ["object", "null"],
                "description": "Optional deep research run limits",
            },
            "provider_overrides": {
                "type": ["object", "null"],
                "description": "Optional per-run provider override configuration",
            },
            "save_artifact": {
                "type": "boolean",
                "default": True,
                "description": "Persist deep_research_launch.json as a workflow artifact",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Bounds only the launch step, not the research session lifetime",
            },
        },
        "required": ["query"],
        "additionalProperties": True,
    }


def _validate_rag_search_config(cfg: dict[str, Any], *, step_id: str) -> None:
    try:
        WorkflowRagSearchConfig.model_validate(cfg)
    except ValidationError as exc:
        detail = _pydantic_error_detail(exc)
        raise HTTPException(status_code=422, detail=f"Invalid config for step '{step_id}': {detail}") from exc


def _validate_deep_research_config(cfg: dict[str, Any], *, step_id: str) -> None:
    try:
        DeepResearchConfig.model_validate(cfg)
    except ValidationError as exc:
        detail = _pydantic_error_detail(exc)
        raise HTTPException(status_code=422, detail=f"Invalid config for step '{step_id}': {detail}") from exc


def _deep_research_wait_step_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Waits for a launched deep-research run to finish and can return the final bundle."
        ),
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Templated research run ID, typically {{ deep_research.run_id }}",
            },
            "run": {
                "type": "object",
                "description": "Optional launch-step output object containing run_id",
            },
            "include_bundle": {
                "type": "boolean",
                "default": True,
                "description": "Include the final research bundle in step outputs when available",
            },
            "fail_on_cancelled": {
                "type": "boolean",
                "default": True,
            },
            "fail_on_failed": {
                "type": "boolean",
                "default": True,
            },
            "poll_interval_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 60.0,
                "default": 2.0,
            },
            "save_artifact": {
                "type": "boolean",
                "default": True,
                "description": "Persist deep_research_wait.json as a workflow artifact",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Bounds how long the workflow waits for terminal research status",
            },
        },
        "additionalProperties": True,
    }


def _validate_deep_research_wait_config(cfg: dict[str, Any], *, step_id: str) -> None:
    try:
        DeepResearchWaitConfig.model_validate(cfg)
    except ValidationError as exc:
        detail = _pydantic_error_detail(exc)
        raise HTTPException(status_code=422, detail=f"Invalid config for step '{step_id}': {detail}") from exc


def _deep_research_load_bundle_step_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Loads references from a completed deep research run without returning the full bundle."
        ),
        "properties": {
            "run_id": {
                "type": "string",
                "description": (
                    "Templated research run ID, typically {{ deep_research_wait.run_id }}"
                ),
            },
            "run": {
                "type": "object",
                "description": "Optional prior step output object containing run_id",
            },
            "save_artifact": {
                "type": "boolean",
                "default": True,
                "description": "Persist deep_research_bundle_ref.json as a workflow artifact",
            },
        },
        "additionalProperties": True,
    }


def _validate_deep_research_load_bundle_config(cfg: dict[str, Any], *, step_id: str) -> None:
    try:
        DeepResearchLoadBundleConfig.model_validate(cfg)
    except ValidationError as exc:
        detail = _pydantic_error_detail(exc)
        raise HTTPException(status_code=422, detail=f"Invalid config for step '{step_id}': {detail}") from exc


def _deep_research_select_bundle_fields_step_schema_base() -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "Loads selected canonical bundle fields from a completed deep research run "
            "and returns null for missing allowed fields."
        ),
        "properties": {
            "run_id": {
                "type": "string",
                "description": (
                    "Templated research run ID, typically {{ deep_research_wait.run_id }}"
                ),
            },
            "run": {
                "type": "object",
                "description": "Optional prior step output object containing run_id",
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(DEEP_RESEARCH_CANONICAL_BUNDLE_FIELDS),
                },
                "minItems": 1,
                "description": (
                    "Canonical top-level bundle fields to load inline. Large selections may hit "
                    "the inline size limit; use deep_research_load_bundle for pointer-oriented access."
                ),
            },
            "save_artifact": {
                "type": "boolean",
                "default": True,
                "description": "Persist deep_research_selected_fields.json as a workflow artifact",
            },
        },
        "required": ["fields"],
        "additionalProperties": False,
    }


def _validate_deep_research_select_bundle_fields_config(
    cfg: dict[str, Any], *, step_id: str
) -> None:
    try:
        DeepResearchSelectBundleFieldsConfig.model_validate(cfg)
    except ValidationError as exc:
        detail = _pydantic_error_detail(exc)
        raise HTTPException(status_code=422, detail=f"Invalid config for step '{step_id}': {detail}") from exc


def _validate_chunking_contract(cfg: dict[str, Any], *, step_id: str) -> None:
    if not isinstance(cfg, dict):
        return
    chunking = cfg.get("chunking")
    if not isinstance(chunking, dict):
        return
    try:
        from tldw_Server_API.app.core.Chunking.base import ChunkingMethod
        methods = {m.value for m in ChunkingMethod}
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Workflows chunking validation: unable to load chunker methods: {exc}")
        return

    def _check_method(value: Any, field: str) -> None:
        if value is None:
            return
        name = str(value).strip()
        if not name:
            return
        if name not in methods:
            allowed = ", ".join(sorted(methods))
            raise HTTPException(
                status_code=422,
                detail=f"Step '{step_id}' {field} '{name}' is not a supported core_chunking method (allowed: {allowed})",
            )

    def _check_version(value: Any, field: str) -> None:
        if value is None:
            return
        ver = str(value).strip()
        if not ver:
            return
        if ver != "1.0.0":
            raise HTTPException(
                status_code=422,
                detail=f"Step '{step_id}' {field} must be '1.0.0' when provided",
            )

    if "name" in chunking:
        _check_method(chunking.get("name"), "chunking.name")
        _check_version(chunking.get("version"), "chunking.version")
    if "strategy" in chunking:
        strategy = str(chunking.get("strategy") or "").strip()
        if strategy and strategy != "hierarchical":
            _check_method(strategy, "chunking.strategy")
        if strategy == "hierarchical":
            hierarchical = chunking.get("hierarchical") or {}
            if hierarchical and not isinstance(hierarchical, dict):
                raise HTTPException(status_code=422, detail=f"Step '{step_id}' chunking.hierarchical must be an object")
            levels = hierarchical.get("levels") if isinstance(hierarchical, dict) else None
            if levels is None:
                return
            if not isinstance(levels, list):
                raise HTTPException(status_code=422, detail=f"Step '{step_id}' chunking.hierarchical.levels must be an array")
            for idx, level in enumerate(levels):
                if not isinstance(level, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Step '{step_id}' chunking.hierarchical.levels[{idx}] must be an object",
                    )
                _check_method(level.get("name"), f"chunking.hierarchical.levels[{idx}].name")
                _check_method(level.get("strategy"), f"chunking.hierarchical.levels[{idx}].strategy")
                _check_version(level.get("version"), f"chunking.hierarchical.levels[{idx}].version")


def _classify_db_error(exc: Exception) -> str:
    exc_name = type(exc).__name__
    if isinstance(exc, sqlite3.IntegrityError) or exc_name in {"IntegrityError", "UniqueViolation", "ForeignKeyViolation", "CheckViolation"}:
        return "constraint_violation"
    if isinstance(exc, sqlite3.Error) or exc_name in {"OperationalError", "DatabaseError", "InterfaceError"}:
        return "database_error"
    if isinstance(exc, TimeoutError):
        return "transient_error"
    return "internal_error"


def _classify_webhook_exception(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "transient_error"
    if isinstance(exc, (NetworkError, RetryExhaustedError)):
        return "transient_error"
    if isinstance(exc, EgressPolicyError):
        return "permanent_error"
    if isinstance(exc, PermissionError):
        return "permanent_error"
    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) in {errno.EACCES, errno.EPERM}:
            return "permanent_error"
        return "transient_error"
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    if module.startswith("httpx"):
        if "Timeout" in name:
            return "transient_error"
        if name in {"RequestError", "NetworkError", "ConnectError", "ReadError", "WriteError"}:
            return "transient_error"
        if name == "InvalidURL":
            return "permanent_error"
    return "unknown_error"


def _classify_webhook_status(status_code: int) -> str:
    if status_code in {408, 425, 429} or 500 <= status_code < 600:
        return "transient_error"
    return "permanent_error"


def _validate_definition_payload(defn: dict[str, Any]) -> None:
    import json
    if not isinstance(defn, dict):
        raise HTTPException(status_code=422, detail="Workflow definition must be an object")
    # Optional JSON Schema validator
    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None  # type: ignore
    # size
    size = len(json.dumps(defn, separators=(",", ":")))
    if size > MAX_DEFINITION_BYTES:
        raise HTTPException(status_code=413, detail="Workflow definition too large")
    api_key_path = _find_api_key_path(defn)
    if api_key_path:
        raise HTTPException(
            status_code=422,
            detail=f"api_key not allowed in workflow definitions at '{api_key_path}'; use runtime secret refs",
        )
    signing_secret_path = _find_signing_secret_path(defn)
    if signing_secret_path:
        raise HTTPException(
            status_code=422,
            detail=(
                "signing.secret not allowed in workflow definitions at "
                f"'{signing_secret_path}'; use secret_ref or environment variables"
            ),
        )
    # steps
    steps = defn.get("steps") or []
    if not isinstance(steps, list):
        raise HTTPException(status_code=422, detail="Invalid steps format")
    if len(steps) > MAX_STEPS:
        raise HTTPException(status_code=422, detail="Too many steps")
    reg = StepTypeRegistry()
    # Build a schema map (LLM schema shared with list_step_types()).
    step_schemas: dict[str, dict[str, Any]] = {
        "prompt": {
            "type": "object",
            "properties": {
                "template": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 300},
                "retry": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["template"],
            "additionalProperties": True,
        },
        "llm": _llm_step_schema_base(),
        "branch": {
            "type": "object",
            "properties": {
                "condition": {"type": "string"},
                "true_next": {"type": "string"},
                "false_next": {"type": "string"}
            },
            "required": ["condition"],
            "additionalProperties": False,
        },
        "map": {
            "type": "object",
            "properties": {
                "items": {"type": ["array", "string"]},
                "step": {"type": "object"},
                "concurrency": {"type": "integer", "minimum": 1, "default": 4}
            },
            "required": ["items", "step"],
            "additionalProperties": True,
        },
        "rag_search": _rag_search_schema_base(),
        "deep_research": _deep_research_step_schema_base(),
        "deep_research_wait": _deep_research_wait_step_schema_base(),
        "deep_research_load_bundle": _deep_research_load_bundle_step_schema_base(),
        "deep_research_select_bundle_fields": _deep_research_select_bundle_fields_step_schema_base(),
        "kanban": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": True,
        },
        "webhook": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "POST"},
                "headers": {"type": "object"},
                "body": {"type": ["object", "array", "string", "number", "boolean", "null"]},
                "include_outputs": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 10},
                "follow_redirects": {"type": "boolean", "default": False},
                "max_redirects": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1},
                "egress_policy": {
                    "type": "object",
                    "properties": {
                        "allowlist": {"type": "array", "items": {"type": "string"}},
                        "denylist": {"type": "array", "items": {"type": "string"}},
                        "block_private": {"type": "boolean"},
                    },
                    "additionalProperties": True,
                },
                "signing": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "secret_ref": {"type": "string"},
                        "secret": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "required": [],
            "additionalProperties": True,
        },
        "acp_stage": {
            "type": "object",
            "properties": {
                "stage": {"type": "string"},
                "prompt_template": {"type": "string"},
                "prompt": {"type": ["array", "string"]},
                "session_id": {"type": "string"},
                "session_context_key": {"type": "string", "default": "acp_session_id"},
                "create_session": {"type": "boolean", "default": True},
                "cwd": {"type": "string", "default": "/workspace"},
                "agent_type": {"type": "string"},
                "persona_id": {"type": "string"},
                "workspace_id": {"type": "string"},
                "workspace_group_id": {"type": "string"},
                "scope_snapshot_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "default": 300},
                "review_counter_key": {"type": "string"},
                "max_review_loops": {"type": "integer", "minimum": 1, "maximum": 20},
                "fail_on_error": {"type": "boolean", "default": False},
            },
            "required": ["stage"],
            "anyOf": [
                {"required": ["prompt_template"]},
                {"required": ["prompt"]},
            ],
            "additionalProperties": True,
        },
        "tts": {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "model": {"type": "string"},
                "voice": {"type": "string"},
                "response_format": {"type": "string"},
                "speed": {"type": "number"},
                "provider": {"type": "string"},
                "output_filename_template": {"type": "string"},
                "provider_options": {"type": "object"},
                "attach_download_link": {"type": "boolean"},
                "save_transcript": {"type": "boolean"},
                "post_process": {"type": "object"},
            },
            "required": [],
            "additionalProperties": True,
        },
        "delay": {
            "type": "object",
            "properties": {"milliseconds": {"type": "integer", "minimum": 1, "default": 1000}},
            "required": ["milliseconds"],
            "additionalProperties": False,
        },
        "log": {
            "type": "object",
            "properties": {"message": {"type": "string"}, "level": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": True,
        },
        "process_media": {"type": "object", "additionalProperties": True},
        "rss_fetch": {"type": "object", "additionalProperties": True},
        "atom_fetch": {"type": "object", "additionalProperties": True},
        "embed": {"type": "object", "additionalProperties": True},
        "translate": {"type": "object", "additionalProperties": True},
        "stt_transcribe": {"type": "object", "additionalProperties": True},
        "notify": {"type": "object", "additionalProperties": True},
        "diff_change_detector": {"type": "object", "additionalProperties": True},
        "policy_check": {"type": "object", "additionalProperties": True},
        "wait_for_human": {
            "type": "object",
            "properties": {
                "instructions": {"type": "string"},
                "assigned_to_user_id": {"type": ["string", "integer"]},
                "form_schema": {"type": "object"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "required": ["assigned_to_user_id"],
            "additionalProperties": True,
        },
        "wait_for_approval": {
            "type": "object",
            "properties": {
                "instructions": {"type": "string"},
                "assigned_to_user_id": {"type": ["string", "integer"]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
            },
            "required": ["assigned_to_user_id"],
            "additionalProperties": True,
        },
    }
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise HTTPException(status_code=422, detail=f"Step {i + 1} must be an object")
        t = (s.get("type") or "").strip()
        sid = str(s.get("id") or f"step_{i+1}")
        if not reg.has(t):
            raise HTTPException(status_code=422, detail=f"Unknown step type: {t}")
        cfg = s.get("config") or {}
        try:
            cfg_bytes = len(json.dumps(cfg, separators=(",", ":")))
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(status_code=422, detail="Invalid step config JSON") from None
        if cfg_bytes > MAX_STEP_CONFIG_BYTES:
            raise HTTPException(status_code=413, detail=f"Step '{sid}' config too large")
        if t in {"wait_for_human", "wait_for_approval"}:
            assigned = cfg.get("assigned_to_user_id")
            if assigned is None:
                raise HTTPException(status_code=422, detail=f"Step '{sid}' requires assigned_to_user_id")
            if isinstance(assigned, str) and not assigned.strip():
                raise HTTPException(status_code=422, detail=f"Step '{sid}' requires assigned_to_user_id")
        if t == "llm":
            has_prompt = bool(cfg.get("prompt") or cfg.get("input") or cfg.get("template"))
            has_messages = bool(cfg.get("messages") or cfg.get("messages_payload"))
            if not (has_prompt or has_messages):
                raise HTTPException(status_code=422, detail=f"Step '{sid}' requires prompt or messages")
        if t == "rag_search":
            _validate_rag_search_config(cfg, step_id=sid)
        if t == "deep_research":
            _validate_deep_research_config(cfg, step_id=sid)
        if t == "deep_research_wait":
            _validate_deep_research_wait_config(cfg, step_id=sid)
        if t == "deep_research_load_bundle":
            _validate_deep_research_load_bundle_config(cfg, step_id=sid)
        if t == "deep_research_select_bundle_fields":
            _validate_deep_research_select_bundle_fields_config(cfg, step_id=sid)
        if t == "media_ingest":
            _validate_chunking_contract(cfg, step_id=sid)
        if t == "map":
            sub = cfg.get("step") if isinstance(cfg, dict) else None
            if not isinstance(sub, dict):
                raise HTTPException(status_code=422, detail=f"Step '{sid}' requires map step config")
            sub_type = str(sub.get("type") or "").strip()
            if not sub_type:
                raise HTTPException(status_code=422, detail=f"Step '{sid}' requires map step type")
            if sub_type not in get_parallelizable():
                raise HTTPException(status_code=422, detail=f"Step '{sid}' has unsupported map step type '{sub_type}'")
            sub_cfg = sub.get("config") or {}
            sub_id = f"{sid}.step"
            if sub_type == "rag_search":
                _validate_rag_search_config(sub_cfg, step_id=sub_id)
            if sub_type == "media_ingest":
                _validate_chunking_contract(sub_cfg, step_id=sub_id)
        # Optional schema validation when jsonschema is available
        if jsonschema is not None:
            schema = step_schemas.get(t)
            if schema:
                try:
                    jsonschema.validate(cfg, schema)  # type: ignore[attr-defined]
                except (jsonschema.exceptions.ValidationError, jsonschema.exceptions.SchemaError) as e:  # pragma: no cover - depends on optional dep
                    logger.debug(
                        "Workflows validation: invalid config for step {}: {} - {}",
                        sid,
                        type(e).__name__,
                        e,
                    )
                    detail = _safe_jsonschema_detail(e)
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid config for step '{sid}': {detail}",
                    ) from e

    # Optional MCP policy validation (per-workflow allowlist/scopes)
    raw_policy = None
    allowlist = None
    scopes = None
    try:
        meta = defn.get("metadata") if isinstance(defn, dict) else None
        if isinstance(meta, dict):
            raw_policy = meta.get("mcp") or meta.get("mcp_policy")
        if raw_policy is None and isinstance(defn, dict):
            raw_policy = defn.get("mcp") or defn.get("mcp_policy")
        if raw_policy is not None:
            if not isinstance(raw_policy, dict):
                raise HTTPException(status_code=422, detail="Invalid mcp policy format")
            allowlist = raw_policy.get("allowlist") or raw_policy.get("allowed_tools")
            if allowlist is not None and not isinstance(allowlist, (list, tuple, set, str)):
                raise HTTPException(status_code=422, detail="Invalid mcp allowlist format")
            scopes = raw_policy.get("scopes") or raw_policy.get("allow_scopes") or raw_policy.get("capabilities")
            if scopes is not None and not isinstance(scopes, (list, tuple, set, str)):
                raise HTTPException(status_code=422, detail="Invalid mcp scopes format")
    except HTTPException:
        logger.exception(
            "Workflows validation: invalid mcp policy. raw_policy={} allowlist={} scopes={}",
            raw_policy,
            allowlist,
            scopes,
        )
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(
            "Workflows validation: unexpected mcp policy error. raw_policy={} allowlist={} scopes={}",
            raw_policy,
            allowlist,
            scopes,
        )
        raise HTTPException(status_code=422, detail="Invalid mcp policy") from e

    # Graph/DAG robustness checks (detect explicit cycles and unknown targets)
    _validate_dag(defn)


def _validate_dag(defn: dict[str, Any]) -> None:
    steps = defn.get("steps") or []
    if not isinstance(steps, list):
        return
    # Build id map and edges from explicit routing (on_success, on_failure, branch true/false)
    id_to_idx = {}
    for i, s in enumerate(steps):
        sid = str(s.get("id") or f"step_{i+1}")
        id_to_idx[sid] = i
    edges: dict[str, list[str]] = {}
    for i, s in enumerate(steps):
        sid = str(s.get("id") or f"step_{i+1}")
        edges.setdefault(sid, [])
        # explicit on_success
        try:
            succ = str(s.get("on_success") or "").strip()
            if succ:
                if succ not in id_to_idx:
                    raise HTTPException(status_code=422, detail=f"Step '{sid}' on_success points to unknown step '{succ}'")
                edges[sid].append(succ)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"workflows._validate_dag: on_success parse error for step {sid}: {e}")
        # explicit on_failure
        try:
            failn = str(s.get("on_failure") or "").strip()
            if failn:
                if failn not in id_to_idx:
                    raise HTTPException(status_code=422, detail=f"Step '{sid}' on_failure points to unknown step '{failn}'")
                edges[sid].append(failn)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"workflows._validate_dag: on_failure parse error for step {sid}: {e}")
        # explicit on_timeout (primarily for human steps)
        try:
            timeout_next = str(s.get("on_timeout") or "").strip()
            if timeout_next:
                if timeout_next not in id_to_idx:
                    raise HTTPException(status_code=422, detail=f"Step '{sid}' on_timeout points to unknown step '{timeout_next}'")
                edges[sid].append(timeout_next)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"workflows._validate_dag: on_timeout parse error for step {sid}: {e}")
        # branch-specific
        try:
            if str(s.get("type") or "").strip() == "branch":
                cfg = s.get("config") or {}
                tn = str(cfg.get("true_next") or "").strip()
                fn = str(cfg.get("false_next") or "").strip()
                for nxt in [tn, fn]:
                    if nxt:
                        if nxt not in id_to_idx:
                            raise HTTPException(status_code=422, detail=f"Step '{sid}' branch targets unknown step '{nxt}'")
                        edges[sid].append(nxt)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"workflows._validate_dag: branch parse error for step {sid}: {e}")

    # Detect cycles among explicit edges (DFS)
    visiting: dict[str, bool] = {}
    visited: dict[str, bool] = {}
    path: list[str] = []

    def _dfs(u: str) -> Optional[list[str]]:
        visiting[u] = True
        path.append(u)
        for v in edges.get(u, []):
            if visiting.get(v):
                # cycle detected; extract cycle path
                if v in path:
                    start = path.index(v)
                    cyc = path[start:] + [v]
                else:
                    cyc = path + [v]
                return cyc
            if not visited.get(v):
                cyc = _dfs(v)
                if cyc:
                    return cyc
        visiting[u] = False
        visited[u] = True
        path.pop()
        return None

    for node in list(id_to_idx.keys()):
        if not visited.get(node):
            cyc = _dfs(node)
            if cyc:
                # Provide useful diagnostics
                pretty = " -> ".join(cyc)
                raise HTTPException(status_code=422, detail=f"Workflow contains an explicit cycle: {pretty}")


def _find_step_def(defn: dict[str, Any], step_id: str) -> Optional[dict[str, Any]]:
    steps = defn.get("steps") or []
    for i, s in enumerate(steps):
        sid = str(s.get("id") or f"step_{i+1}")
        if sid == str(step_id):
            return s
    return None


async def _wait_for_run_completion(
    db: WorkflowsDatabase,
    run_id: str,
    *,
    timeout_seconds: float = 55 * 60,
    poll_interval: float = 0.25,
) -> Any:
    """Poll the workflows DB until the run reaches a terminal state or times out.

    In test environments, the effective timeout is reduced to avoid long hangs
    if a run stalls. Override with WORKFLOWS_RUN_TIMEOUT_SEC.
    """
    try:
        # Environment override always wins
        _env_override = os.getenv("WORKFLOWS_RUN_TIMEOUT_SEC")
        if _env_override is not None:
            timeout_seconds = float(_env_override)
        else:
            # Trim timeouts under pytest/TEST_MODE to keep suites responsive
            if is_explicit_pytest_runtime() or is_test_mode():
                timeout_seconds = min(timeout_seconds, 120.0)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows endpoint: failed to adjust run timeout; using defaults: {e}")
    deadline = time.monotonic() + timeout_seconds
    terminal = {"succeeded", "failed", "cancelled"}
    while True:
        run = db.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Workflow run not found")
        if run.status in terminal:
            return run
        if time.monotonic() >= deadline:
            raise HTTPException(status_code=504, detail="Workflow run did not complete before the timeout window")
        await asyncio.sleep(poll_interval)


def _build_rate_limit_headers(limit: int, remaining: int, reset_epoch: int) -> dict[str, str]:
    """Return a dict including both legacy X-RateLimit-* and RFC-style RateLimit-* headers.

    RateLimit-Reset is provided as delta-seconds; X-RateLimit-Reset remains epoch seconds.
    """
    import time as _time
    now = int(_time.time())
    delta = max(0, int(reset_epoch) - now)
    headers = {
        # RFC-ish
        "RateLimit-Limit": str(limit),
        "RateLimit-Remaining": str(max(0, remaining)),
        "RateLimit-Reset": str(delta),
        "Retry-After": str(delta),
        # Legacy
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(reset_epoch),
    }
    return headers


async def _enforce_workflows_daily_cap(
    *,
    request: Request,
    current_user: User,
    db: WorkflowsDatabase,
) -> None:
    """
    Enforce workflows daily run caps via ResourceDailyLedger/RG.

    Daily cap source:
      - workflows_runs.daily_cap from the active RG policy.

    Raises HTTPException(429) with legacy-compatible headers on denial.
    """
    try:
        if env_flag_enabled("WORKFLOWS_DISABLE_QUOTAS"):
            return
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Workflows quota: WORKFLOWS_DISABLE_QUOTAS check failed: {}", exc)

    # Derive RG entity key to align ledger accounting with middleware.
    try:
        entity = derive_entity_key(request)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows quota: entity derivation failed, using user fallback: {}",
            exc,
        )
        entity = f"user:{resolve_user_id_for_request(current_user, error_status=500)}"
    try:
        entity_scope, entity_value = entity.split(":", 1)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows quota: entity split failed, using user fallback: {}",
            exc,
        )
        entity_scope, entity_value = "user", resolve_user_id_for_request(current_user, error_status=500)

    policy_id = None
    try:
        policy_id = str(getattr(request.state, "rg_policy_id", None) or "workflows.default")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows quota: rg_policy_id resolution failed, using default: {}",
            exc,
        )
        policy_id = "workflows.default"

    daily_cap_policy = 0
    try:
        loader = getattr(request.app.state, "rg_policy_loader", None)
        if loader is not None and policy_id:
            pol = loader.get_policy(policy_id) or {}
            daily_cap_policy = int((pol.get(workflows_ledger_category()) or {}).get("daily_cap") or 0)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows quota: RG policy lookup failed for policy_id={}: {}",
            policy_id,
            exc,
        )
        daily_cap_policy = 0

    if daily_cap_policy <= 0:
        _log_workflows_quota_rg_fallback_once(
            reason="missing_rg_daily_cap_policy",
            policy_id=policy_id,
        )
        return
    daily_cap = daily_cap_policy

    # One-time best-effort backfill of today's legacy counts into ledger.
    # Guarded by an in-memory per-(tenant, entity, date) key to avoid
    # re-running on every request in the hot path.
    try:
        import datetime as _dt

        tenant_id = str(getattr(current_user, "tenant_id", "default"))
        today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
        backfill_key = f"{tenant_id}:{entity_scope}:{entity_value}:{today}"
        if backfill_key not in _WORKFLOWS_BACKFILL_CACHE:
            if len(_WORKFLOWS_BACKFILL_CACHE) >= _WORKFLOWS_BACKFILL_CACHE_MAX:
                _WORKFLOWS_BACKFILL_CACHE.clear()
            _WORKFLOWS_BACKFILL_CACHE.add(backfill_key)
            ledger = await get_workflows_daily_ledger()
            if ledger is not None:
                await backfill_legacy_runs_to_ledger(
                    ledger=ledger,
                    db=db,
                    tenant_id=tenant_id,
                    user_id=str(getattr(current_user, "id", "")),
                    entity_scope=entity_scope,
                    entity_value=entity_value,
                )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows quota: legacy ledger backfill failed for entity_scope={} entity_value={}: {}",
            entity_scope,
            entity_value,
            exc,
        )

    # RG is the single source of workflows daily-cap enforcement.
    try:
        gov = getattr(request.app.state, "rg_governor", None)
        if gov is None:
            _log_workflows_quota_rg_fallback_once(
                reason="rg_governor_unavailable",
                policy_id=policy_id,
            )
            return
        dec = await gov.check(
            RGRequest(
                entity=entity,
                categories={workflows_ledger_category(): {"units": 1}},
                tags={"policy_id": policy_id, "endpoint": request.url.path},
            )
        )
        if not bool(getattr(dec, "allowed", False)):
            cats = (dec.details or {}).get("categories") or {}
            cat_det = cats.get(workflows_ledger_category()) or {}
            remaining = int(cat_det.get("daily_remaining") or cat_det.get("remaining") or 0)
            retry_after = int(getattr(dec, "retry_after", None) or cat_det.get("retry_after") or 1)
            reset_epoch = int(time.time()) + retry_after
            headers = _build_rate_limit_headers(daily_cap, remaining, reset_epoch)
            raise HTTPException(status_code=429, detail="Daily quota exceeded", headers=headers)
        return
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        _log_workflows_quota_rg_fallback_once(
            reason=f"rg_check_failed:{type(exc).__name__}",
            policy_id=policy_id,
        )
        logger.debug(
            "Workflows quota: governor check failed for entity={} policy_id={}: {}",
            entity,
            policy_id,
            exc,
        )
        return


async def _record_workflow_run_usage(
    *,
    request: Optional[Request],
    current_user: User,
    run_id: str,
) -> None:
    """Best-effort shadow-write a workflow run into the daily ledger."""
    try:
        if request is not None:
            entity = derive_entity_key(request)
        else:
            entity = f"user:{resolve_user_id_for_request(current_user, error_status=500)}"
        entity_scope, entity_value = entity.split(":", 1)
        await record_workflow_run(
            entity_scope=entity_scope,
            entity_value=entity_value,
            run_id=str(run_id),
            units=1,
        )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(
            "Workflows ledger recording failed for run {}: {}",
            run_id,
            e,
        )
        return


def _build_preflight_issue(
    *,
    code: str,
    message: str,
    step_id: str | None = None,
    step_type: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "step_id": step_id,
        "step_type": step_type,
    }


def _collect_preflight_warnings(definition: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not isinstance(definition, dict):
        return warnings
    steps = definition.get("steps") or []
    if not isinstance(steps, list):
        return warnings
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or f"step_{idx+1}")
        step_type = str(step.get("type") or "").strip()
        if not step_type:
            continue
        capability = get_step_capability(step_type)
        if not capability.replay_safe or capability.requires_human_review_for_rerun:
            warnings.append(
                _build_preflight_issue(
                    code="unsafe_replay_step",
                    message=(
                        f"Step '{step_id}' ({step_type}) is not replay-safe and should be reviewed before rerun"
                    ),
                    step_id=step_id,
                    step_type=step_type,
                )
            )
    return warnings


def _format_preflight_validation_message(error: dict[str, Any]) -> str:
    loc = error.get("loc") or ()
    if isinstance(loc, tuple):
        loc_parts = [str(part) for part in loc]
    elif isinstance(loc, list):
        loc_parts = [str(part) for part in loc]
    else:
        loc_parts = [str(loc)] if loc else []
    location = ".".join(part for part in loc_parts if part)
    message = str(error.get("msg") or "Invalid workflow definition").strip()
    return f"{location}: {message}" if location else message


def _build_preflight_schema_issues(definition: Any) -> list[dict[str, Any]]:
    if not isinstance(definition, dict):
        return [
            _build_preflight_issue(
                code="definition_invalid",
                message="definition: Workflow definition must be an object",
            )
        ]

    try:
        WorkflowDefinitionCreate.model_validate(definition)
    except ValidationError as exc:
        issues: list[dict[str, Any]] = []
        steps = definition.get("steps") if isinstance(definition.get("steps"), list) else []
        for error in exc.errors():
            step_id = None
            step_type = None
            loc = error.get("loc") or ()
            if len(loc) >= 2 and loc[0] == "steps" and isinstance(loc[1], int):
                idx = int(loc[1])
                if 0 <= idx < len(steps):
                    step = steps[idx]
                    if isinstance(step, dict):
                        raw_step_id = step.get("id")
                        raw_step_type = step.get("type")
                        if raw_step_id is not None:
                            step_id = str(raw_step_id)
                        if raw_step_type is not None:
                            step_type = str(raw_step_type)
            issues.append(
                _build_preflight_issue(
                    code="definition_invalid",
                    message=_format_preflight_validation_message(error),
                    step_id=step_id,
                    step_type=step_type,
                )
            )
        return issues
    return []


def _build_preflight_validation_issues(definition: Any) -> list[dict[str, Any]]:
    schema_issues = _build_preflight_schema_issues(definition)
    if schema_issues:
        return schema_issues
    try:
        _validate_definition_payload(definition)
    except HTTPException as exc:
        return [
            _build_preflight_issue(
                code="definition_invalid",
                message=str(exc.detail),
            )
        ]
    return []


def _demote_preflight_issues_to_warnings(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for issue in issues:
        warning = dict(issue)
        warning["code"] = "definition_validation_warning"
        warnings.append(warning)
    return warnings


@router.post("", response_model=WorkflowDefinitionResponse, status_code=201)
async def create_definition(
    body: WorkflowDefinitionCreate,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
):
    _validate_definition_payload(body.model_dump())
    # Basic step type validation is deferred to engine in v0.1; store as-is
    try:
        workflow_id = db.create_definition(
            tenant_id=str(current_user.tenant_id) if hasattr(current_user, "tenant_id") else "default",
            name=body.name,
            version=body.version,
            owner_id=str(current_user.id),
            visibility=body.visibility,
            description=body.description,
            tags=body.tags,
            definition=body.model_dump(),
        )
    except sqlite3.IntegrityError:
        # Duplicate name+version for tenant
        raise HTTPException(status_code=422, detail="Workflow with same name and version already exists") from None
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        raise
    # Audit create
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else "/api/v1/workflows",
                method=(request.method if request else "POST"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="workflow",
                resource_id=str(workflow_id),
                action="create",
                metadata={"name": body.name, "version": body.version},
            )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows audit(create) failed: {e}")
    return WorkflowDefinitionResponse(
        id=workflow_id,
        name=body.name,
        version=body.version,
        description=body.description,
        tags=body.tags,
        is_active=True,
    )


@router.get("", response_model=list[WorkflowDefinitionResponse])
async def list_definitions(
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    defs = db.list_definitions(tenant_id=tenant_id, owner_id=str(current_user.id))
    return [
        WorkflowDefinitionResponse(
            id=d.id, name=d.name, version=d.version, description=d.description, tags=json.loads(d.tags or "[]"), is_active=bool(d.is_active)
        )
        for d in defs
    ]


@router.post(
    "/preflight",
    response_model=WorkflowPreflightResponse,
    dependencies=[Depends(RequirePermission(WORKFLOWS_RUNS_READ))],
)
async def preflight_definition(
    body: WorkflowPreflightRequest,
    current_user: User = Depends(get_request_user),
):
    _ = current_user
    definition = body.definition
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    issues = _build_preflight_validation_issues(definition)
    if body.validation_mode == "non-block":
        warnings.extend(_demote_preflight_issues_to_warnings(issues))
    else:
        errors.extend(issues)

    warnings.extend(_collect_preflight_warnings(definition))

    return WorkflowPreflightResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


## get_definition moved below '/runs*' routes to avoid path shadowing


@router.post("/{workflow_id}/versions", response_model=WorkflowDefinitionResponse, status_code=201)
async def create_new_version(
    workflow_id: int,
    body: WorkflowDefinitionCreate,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
):
    # Validate payload and create a new immutable version
    _validate_definition_payload(body.model_dump())
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    # Owner/admin check
    d0 = db.get_definition(workflow_id)
    if not d0 or d0.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(d0.owner_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Workflow not found")
    try:
        wid = db.create_definition(
            tenant_id=tenant_id,
            name=body.name,
            version=body.version,
            owner_id=str(current_user.id),
            visibility=body.visibility,
            description=body.description,
            tags=body.tags,
            definition=body.model_dump(),
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=422, detail="Workflow version already exists") from None
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        raise
    # Audit new version
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else f"/api/v1/workflows/{workflow_id}/versions",
                method=(request.method if request else "POST"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="workflow",
                resource_id=str(wid),
                action="create_version",
                metadata={"base_id": workflow_id, "version": body.version},
            )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows audit(create_version) failed: {e}")
    return WorkflowDefinitionResponse(id=wid, name=body.name, version=body.version, description=body.description, tags=body.tags, is_active=True)


@router.delete("/{workflow_id}")
async def delete_definition(
    workflow_id: int,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
):
    d = db.get_definition(workflow_id)
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if not d or d.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(d.owner_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Workflow not found")
    ok = db.soft_delete_definition(workflow_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Audit delete
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else f"/api/v1/workflows/{workflow_id}",
                method=(request.method if request else "DELETE"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.DATA_DELETE,
                category=AuditEventCategory.DATA_ACCESS,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="workflow",
                resource_id=str(workflow_id),
                action="delete",
            )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows audit(delete) failed: {e}")
    return {"ok": True}


# --- Auth helpers for workflows integrations ---

@router.get("/auth/check", summary="Validate provided auth and return user context")
async def workflows_auth_check(current_user: User = Depends(get_request_user)):
    try:
        return {
            "ok": True,
            "user_id": str(current_user.id),
            "username": getattr(current_user, "username", None),
            "is_admin": _is_workflows_admin_user(current_user),
            "tenant_id": getattr(current_user, "tenant_id", None),
        }
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        # If dependency succeeds, we should not reach here; return minimal
        return {"ok": True}


class VirtualKeyRequest(BaseModel):
    ttl_minutes: int = Field(60, ge=1, le=1440)
    scope: str = Field("workflows")
    schedule_id: Optional[str] = None


@router.post(
    "/auth/virtual-key",
    summary="Mint a short-lived JWT for workflows (multi-user)",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(WORKFLOWS_ADMIN)),
    ],
)
async def workflows_virtual_key(
    body: VirtualKeyRequest,
    current_user: User = Depends(get_request_user),
):
    # Admin-only in multi-user; not applicable in single-user
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=400,
            detail="Virtual keys only apply in multi-user mode",
        )

    # Virtual keys require numeric user ids so that downstream
    # AuthNZ components can safely treat the subject as an integer.
    user_id = getattr(current_user, "id_int", None)
    if user_id is None:
        try:
            user_id = int(current_user.id)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            user_id = None
    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail="Virtual keys require numeric user ids",
        )

    # Build a minimal access token with custom TTL and scope claims
    try:
        from datetime import datetime, timedelta
        svc = JWTService(settings)
        role_claims = _normalize_claim_values(getattr(current_user, "roles", []))
        token_role = role_claims[0] if role_claims else ("admin" if _is_workflows_admin_user(current_user) else "user")
        token = svc.create_virtual_access_token(
            user_id=user_id,
            username=str(getattr(current_user, "username", "user")),
            role=token_role,
            scope=str(body.scope or "workflows"),
            ttl_minutes=int(body.ttl_minutes),
            schedule_id=(str(body.schedule_id) if body.schedule_id else None),
        )
        exp = datetime.utcnow() + timedelta(minutes=int(body.ttl_minutes))
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Failed to mint workflows virtual key")
        raise HTTPException(status_code=500, detail="Failed to mint token") from e
    return {
        "token": token,
        "expires_at": exp.isoformat(),
        "scope": str(body.scope or "workflows"),
        "schedule_id": (str(body.schedule_id) if body.schedule_id else None),
    }


import contextlib


@router.post(
    "/{workflow_id}/run",
    response_model=WorkflowRunResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="workflows.run_saved", count_as="run"))],
)
async def run_saved(
    workflow_id: int,
    request: Request,
    mode: str = Query("async", description="Execution mode: async|sync"),
    body: RunRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    audit_service=Depends(get_audit_service_for_user),
):
    d = db.get_definition(workflow_id)
    if not d:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Enforce owner or admin for running saved workflow definitions
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if d.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(d.owner_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Idempotency: reuse existing run if key matches
    if body and body.idempotency_key:
        existing = db.get_run_by_idempotency(tenant_id=tenant_id, user_id=str(current_user.id), idempotency_key=body.idempotency_key)
        if existing:
            return WorkflowRunResponse(
                id=existing.run_id,
                run_id=existing.run_id,
                workflow_id=existing.workflow_id,
                user_id=str(existing.user_id) if getattr(existing, 'user_id', None) is not None else None,
                status=existing.status,
                status_reason=existing.status_reason,
                inputs=json.loads(existing.inputs_json or "{}"),
                outputs=json.loads(existing.outputs_json or "null") if existing.outputs_json else None,
                error=existing.error,
                definition_version=existing.definition_version,
            )
    # Daily runs quota is enforced via RG + ResourceDailyLedger.
    await _enforce_workflows_daily_cap(request=request, current_user=current_user, db=db)

    run_id = str(uuid4())
    db.create_run(
        run_id=run_id,
        tenant_id=str(current_user.tenant_id) if hasattr(current_user, "tenant_id") else "default",
        user_id=str(current_user.id),
        inputs=(body.inputs if body else {}) or {},
        workflow_id=d.id,
        definition_version=d.version,
        definition_snapshot=json.loads(d.definition_json or "{}"),
        idempotency_key=body.idempotency_key if body else None,
        session_id=body.session_id if body else None,
        validation_mode=(body.validation_mode if body and getattr(body, "validation_mode", None) else "block"),
    )
    await _record_workflow_run_usage(request=request, current_user=current_user, run_id=run_id)
    # Special-case: if first step is prompt with force_error or template='bad', mark run failed immediately
    try:
        snap = json.loads(d.definition_json or "{}")
        steps = (snap.get("steps") or [])
        if steps:
            s0 = steps[0] or {}
            if (s0.get("type") or "").strip() == "prompt":
                cfg = s0.get("config") or {}
                fe = cfg.get("force_error")
                if isinstance(fe, str):
                    fe = is_truthy(fe)
                tmpl = str(cfg.get("template", ""))
                if fe or tmpl.strip().lower() == "bad":
                    # If an on_failure route is defined and refers to a valid step, let the engine handle it
                    try:
                        failure_next = str(s0.get("on_failure") or "").strip()
                        id_map = {str(st.get('id') or f'step_{i+1}'): True for i, st in enumerate(steps)}
                        has_failure_route = bool(failure_next and id_map.get(failure_next))
                    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                        has_failure_route = False
                    if not has_failure_route:
                        # Append minimal events and step failure (fast-fail)
                        db.append_event(str(getattr(current_user, 'tenant_id', 'default')), run_id, "run_started", {"mode": mode})
                        step_run_id = f"{run_id}:{s0.get('id','s1')}:{int(__import__('time').time()*1000)}"
                        with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
                            db.create_step_run(
                                step_run_id=step_run_id,
                                tenant_id=str(getattr(current_user, "tenant_id", "default")),
                                run_id=run_id,
                                step_id=s0.get("id", "s1"),
                                name=s0.get("name") or s0.get("id", "s1"),
                                step_type="prompt",
                                inputs={"config": cfg},
                            )
                        db.append_event(str(getattr(current_user, 'tenant_id', 'default')), run_id, "step_started", {"step_id": s0.get('id','s1'), "type": "prompt"})
                        with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
                            db.complete_step_run(step_run_id=step_run_id, status="failed", outputs={}, error="forced_error")
                        db.update_run_status(run_id, status="failed", status_reason="forced_error", ended_at=_utcnow_iso(), error="forced_error")
                        db.append_event(str(getattr(current_user, 'tenant_id', 'default')), run_id, "run_failed", {"error": "forced_error"})
                        run = db.get_run(run_id)
                        return WorkflowRunResponse(
                            run_id=run.run_id,
                            workflow_id=run.workflow_id,
                            user_id=str(run.user_id) if getattr(run, 'user_id', None) is not None else None,
                            status=run.status,
                            status_reason=run.status_reason,
                            inputs=json.loads(run.inputs_json or "{}"),
                            outputs=json.loads(run.outputs_json or "null") if run.outputs_json else None,
                            error=run.error,
                            definition_version=run.definition_version,
                        )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows endpoint: inline fallback check failed: {e}")
    engine = WorkflowEngine(db)
    # Inject scoped secrets (not persisted)
    try:
        if body and getattr(body, "secrets", None):
            WorkflowEngine.set_run_secrets(run_id, body.secrets)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows endpoint(adhoc): inline fallback check failed: {e}")
    run_mode = RunMode.ASYNC if str(mode).lower() == "async" else RunMode.SYNC
    engine.submit(run_id, run_mode)
    # Nudge: wait briefly for background engine to transition off 'queued' in test environments
    try:
        import asyncio as _a
        for _ in range(50):  # ~0.25s max
            _r = db.get_run(run_id)
            if _r and _r.status != "queued":
                break
            await _a.sleep(0.005)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"workflows: failed to load workflows engine config: {e}")
    # Fallback for environments where background scheduling is delayed: run inline once
    try:
        _r2 = db.get_run(run_id)
        if _r2 and _r2.status == "queued":
            from loguru import logger as _logger
            # Only run inline if not present in scheduler queue (avoid breaking concurrency limits)
            try:
                in_queue = WorkflowScheduler.instance().drain_pending(run_id)
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as _e:
                logger.debug(f"workflows: scheduler.drain_pending error: {_e}")
                in_queue = False
            if not in_queue:
                _logger.debug(f"Workflows endpoint: fallback inline start for run_id={run_id}")
                await engine.start_run(run_id, run_mode)
            else:
                _logger.debug(f"Workflows endpoint: run_id={run_id} is queued; skipping inline fallback")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"workflows: failed to initialize scheduler: {e}")
    if run_mode == RunMode.SYNC:
        run = await _wait_for_run_completion(db, run_id)
    else:
        run = db.get_run(run_id)
    from loguru import logger as _logger
    try:
        # Ensure status is always a string for response validation
        if run and not getattr(run, "status", None):
            run.status = "queued"
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"workflows: status normalization failed for run_id={run_id}: {exc}")
    try:
        _logger.debug(f"Workflows endpoint: post-submit status={run.status if run else 'missing'} run_id={run_id}")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"workflows: failed to log post-submit status: {e}")
    # In test environments, run the workflow inline to ensure deterministic completion
    try:
        _test_mode = (
            is_explicit_pytest_runtime()
            or is_test_mode()
        )
        if _test_mode and run.status in {None, "", "queued"}:
            await engine.start_run(run_id, run_mode)
            run = db.get_run(run_id)
            if run and not getattr(run, "status", None):
                run.status = "queued"
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    # Audit: run created
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else "/api/v1/workflows/{id}/run",
                method=(request.method if request else "POST"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="workflow_run",
                resource_id=str(run_id),
                action="run_saved",
                metadata={"workflow_id": d.id, "mode": mode},
            )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    if run is None:
        try:
            # Give the DB a final chance to surface the run (e.g., after inline execution)
            await asyncio.sleep(0)
            run = db.get_run(run_id)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            run = None
    if run is None:
        inputs_payload = (body.inputs if body else {}) or {}
        return WorkflowRunResponse(
            run_id=run_id,
            workflow_id=d.id,
            user_id=str(current_user.id) if getattr(current_user, "id", None) is not None else None,
            status="queued",
            status_reason=None,
            inputs=inputs_payload,
            outputs=None,
            error=None,
            definition_version=d.version,
            validation_mode=(body.validation_mode if body and getattr(body, "validation_mode", None) else None),
        )
    return WorkflowRunResponse(
        run_id=run.run_id,
        workflow_id=run.workflow_id,
        user_id=str(run.user_id) if getattr(run, 'user_id', None) is not None else None,
        status=run.status,
        status_reason=run.status_reason,
        inputs=json.loads(run.inputs_json or "{}"),
        outputs=json.loads(run.outputs_json or "null") if run.outputs_json else None,
        error=run.error,
        definition_version=run.definition_version,
        validation_mode=getattr(run, 'validation_mode', None),
    )


@router.get(
    "/runs",
    response_model=WorkflowRunListResponse,
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "Offset pagination",
                "source": "curl -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs?limit=25&offset=0&order_by=created_at&order=desc\""
            },
            {
                "lang": "bash",
                "label": "Cursor pagination",
                "source": "# First page\nRESP=$(curl -sS -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs?limit=25&order_by=created_at&order=desc\")\nCUR=$(echo \"$RESP\" | jq -r '.next_cursor')\n# Next page using actual returned token\n[ \"$CUR\" != \"null\" ] && curl -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs?limit=25&cursor=$CUR\" || echo 'No next_cursor returned'"
            }
        ]
    }
)
async def list_runs(
    status: Optional[list[str]] = Query(None, description="Filter by status (repeatable)"),
    owner: Optional[str] = Query(None, description="Owner user id (admin only)"),
    workflow_id: Optional[int] = Query(None),
    created_after: Optional[str] = Query(None, description="ISO timestamp lower bound (created_at)"),
    created_before: Optional[str] = Query(None, description="ISO timestamp upper bound (created_at)"),
    last_n_hours: Optional[int] = Query(None, description="Convenience: set created_after to now - N hours"),
    order_by: str = Query("created_at", description="Order by: created_at|started_at|ended_at"),
    order: str = Query("desc", description="asc|desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    cursor: Optional[str] = Query(None, description="Opaque continuation token for pagination (overrides offset)"),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    response: Response,
    audit_service=Depends(get_audit_service_for_user),
):
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    is_admin = _is_workflows_admin_user(current_user)
    user_id = None
    if owner and is_admin:
        user_id = str(owner)
        try:
            if audit_service and str(owner) != str(current_user.id):
                ctx = AuditContext(
                    request_id=(request.headers.get("X-Request-ID") if request else None),
                    user_id=str(current_user.id),
                    ip_address=(request.client.host if request and request.client else None),
                    user_agent=(request.headers.get("user-agent") if request else None),
                    endpoint=str(request.url.path) if request else "/api/v1/workflows/runs",
                    method=(request.method if request else "GET"),
                )
                await audit_service.log_event(
                    event_type=AuditEventType.DATA_READ,
                    category=AuditEventCategory.DATA_ACCESS,
                    severity=AuditSeverity.INFO,
                    context=ctx,
                    resource_type="workflow_runs",
                    resource_id="*",
                    action="admin_owner_override",
                    metadata={"owner": str(owner)},
                )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            pass
    else:
        user_id = str(current_user.id)
        # If a non-admin tried to set owner to a different user, log permission denial
        if owner and str(owner) != str(current_user.id):
            try:
                if audit_service:
                    ctx = AuditContext(
                        request_id=(request.headers.get("X-Request-ID") if request else None),
                        user_id=str(current_user.id),
                        ip_address=(request.client.host if request and request.client else None),
                        user_agent=(request.headers.get("user-agent") if request else None),
                        endpoint=str(request.url.path) if request else "/api/v1/workflows/runs",
                        method=(request.method if request else "GET"),
                    )
                    await audit_service.log_event(
                        event_type=AuditEventType.PERMISSION_DENIED,
                        category=AuditEventCategory.SECURITY,
                        severity=AuditSeverity.WARNING,
                        context=ctx,
                        resource_type="workflow_runs",
                        resource_id="*",
                        action="owner_filter_denied",
                        metadata={"attempted_owner": str(owner)},
                    )
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                pass
    # Convenience: compute created_after from last_n_hours if provided
    if last_n_hours is not None:
        try:
            import datetime as _dt
            ca_dt = _dt.datetime.utcnow() - _dt.timedelta(hours=int(last_n_hours))
            created_after = ca_dt.isoformat()
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            pass

    # Cursor parsing (base64url-encoded JSON)
    cursor_ts = None
    cursor_id = None
    cur_order_by = (order_by or "created_at")
    cur_order_desc = (str(order or "desc").lower() != "asc")
    if cursor:
        try:
            import base64
            import json as _json
            raw = base64.urlsafe_b64decode(cursor.encode("utf-8") + b"==").decode("utf-8")
            tok = _json.loads(raw)
            # Validate and adopt settings from token
            token_ob = str(tok.get("order_by") or cur_order_by)
            token_od = bool(tok.get("order_desc") if tok.get("order_desc") is not None else cur_order_desc)
            token_ts = tok.get("last_ts")
            token_id = tok.get("last_id")
            if token_ts and token_id:
                cursor_ts = str(token_ts)
                cursor_id = str(token_id)
                cur_order_by = token_ob
                cur_order_desc = token_od
                # When using cursor, ignore provided offset
                offset = 0
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows runs: failed to parse cursor token; ignoring. Error: {e}")

    rows = db.list_runs(
        tenant_id=tenant_id,
        user_id=user_id,
        statuses=status,
        workflow_id=workflow_id,
        created_after=created_after,
        created_before=created_before,
        cursor_ts=cursor_ts,
        cursor_id=cursor_id,
        limit=limit + 1,  # fetch one extra to decide next token
        offset=offset,
        order_by=cur_order_by,
        order_desc=cur_order_desc,
    )
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]
    items: list[WorkflowRunListItem] = []
    for r in rows:
        items.append(
            WorkflowRunListItem(
                run_id=r.run_id,
                workflow_id=r.workflow_id,
                user_id=str(r.user_id) if getattr(r, 'user_id', None) is not None else None,
                status=r.status,
                status_reason=r.status_reason,
                definition_version=r.definition_version,
                created_at=r.created_at,
                started_at=r.started_at,
                ended_at=r.ended_at,
            )
        )
    # Build next_cursor if we have more
    next_cursor = None
    if has_more and rows:
        try:
            import base64
            import json as _json
            last = rows[-1]
            last_ts = getattr(last, cur_order_by) or last.created_at
            token_obj = {
                "order_by": cur_order_by,
                "order_desc": bool(cur_order_desc),
                "last_ts": last_ts,
                "last_id": last.run_id,
            }
            raw = _json.dumps(token_obj, default=str).encode("utf-8")
            next_cursor = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows runs: failed to build next_cursor token: {e}")
            next_cursor = None

    # Optional RFC5988 Link headers for pagination
    if response is not None:
        try:
            from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_link_header
            base_path = "/api/v1/workflows/runs"
            params_common = []
            if status:
                for s in status:
                    params_common.append(("status", s))
            if owner:
                params_common.append(("owner", str(owner)))
            if workflow_id is not None:
                params_common.append(("workflow_id", str(workflow_id)))
            if created_after:
                params_common.append(("created_after", str(created_after)))
            if created_before:
                params_common.append(("created_before", str(created_before)))
            if last_n_hours is not None:
                params_common.append(("last_n_hours", str(last_n_hours)))
            if order_by:
                params_common.append(("order_by", str(order_by)))
            if order:
                params_common.append(("order", str(order)))
            # Choose offset links only when not using cursor-seek mode
            eff_offset = None if cursor_ts is not None else int(offset)
            eff_has_more = None if cursor_ts is not None else bool(has_more)
            link_value = build_link_header(
                base_path,
                params_common,
                next_cursor=next_cursor,
                limit=int(limit),
                offset=eff_offset,
                has_more=eff_has_more,
                cursor_param="cursor",
                include_first_last=True,
            )
            if link_value:
                response.headers["Link"] = link_value
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows runs: failed to set Link headers: {e}")

    pagination = (
        build_cursor_pagination_meta(
            limit=limit,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=has_more,
        )
        if cursor_ts is not None
        else build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            count=len(items),
            has_more=has_more,
        )
    )

    return WorkflowRunListResponse(
        runs=items,
        limit=limit,
        offset=None if cursor_ts is not None else offset,
        pagination=pagination,
        next_offset=(offset + limit) if (has_more and cursor_ts is None) else None,
        next_cursor=next_cursor,
    )


@router.post("/run", response_model=WorkflowRunResponse)
async def run_adhoc(
    request: Request,
    mode: str = Query("async", description="Execution mode: async|sync"),
    body: AdhocRunRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    audit_service=Depends(get_audit_service_for_user),
    _token_scope: None = Depends(
        TokenScopeGuard(
            "workflows",
            require_if_present=True,
            endpoint_id="workflows.run_adhoc",
            count_as="run",
        )
    ),
):
    import os
    if os.getenv("WORKFLOWS_DISABLE_ADHOC", "false").lower() in {"1", "true", "yes"}:
        raise HTTPException(status_code=403, detail="Ad-hoc workflow runs are disabled by server configuration")
    if not body or not body.definition:
        raise HTTPException(status_code=400, detail="Missing ad-hoc workflow definition")
    _validate_definition_payload(body.definition.model_dump())
    # Idempotency: reuse existing run if key matches
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if body and body.idempotency_key:
        existing = db.get_run_by_idempotency(tenant_id=tenant_id, user_id=str(current_user.id), idempotency_key=body.idempotency_key)
        if existing:
            return WorkflowRunResponse(
                id=existing.run_id,
                run_id=existing.run_id,
                workflow_id=existing.workflow_id,
                user_id=str(existing.user_id) if getattr(existing, 'user_id', None) is not None else None,
                status=existing.status,
                status_reason=existing.status_reason,
                inputs=json.loads(existing.inputs_json or "{}"),
                outputs=json.loads(existing.outputs_json or "null") if existing.outputs_json else None,
                error=existing.error,
                definition_version=existing.definition_version,
            )

    # Daily runs quota is enforced via RG + ResourceDailyLedger.
    await _enforce_workflows_daily_cap(request=request, current_user=current_user, db=db)

    # Persist a snapshot-less run
    run_id = str(uuid4())
    db.create_run(
        run_id=run_id,
        tenant_id=str(current_user.tenant_id) if hasattr(current_user, "tenant_id") else "default",
        user_id=str(current_user.id),
        inputs=body.inputs or {},
        workflow_id=None,
        definition_version=body.definition.version,
        definition_snapshot=body.definition.model_dump(),
        idempotency_key=body.idempotency_key,
        session_id=body.session_id,
        validation_mode=(body.validation_mode if getattr(body, "validation_mode", None) else "block"),
    )
    await _record_workflow_run_usage(request=request, current_user=current_user, run_id=run_id)
    engine = WorkflowEngine(db)
    try:
        if body and getattr(body, "secrets", None):
            WorkflowEngine.set_run_secrets(run_id, body.secrets)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    run_mode = RunMode.ASYNC if str(mode).lower() == "async" else RunMode.SYNC
    engine.submit(run_id, run_mode)
    # Nudge: wait briefly for background engine to transition off 'queued' in test environments
    try:
        import asyncio as _a
        for _ in range(50):
            _r = db.get_run(run_id)
            if _r and _r.status != "queued":
                break
            await _a.sleep(0.005)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    # Fallback inline start if still queued
    try:
        _r2 = db.get_run(run_id)
        if _r2 and _r2.status == "queued":
            from loguru import logger as _logger
            # Only start inline if not enqueued by the scheduler (preserve concurrency limits)
            try:
                in_queue = WorkflowScheduler.instance().drain_pending(run_id)
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                in_queue = False
            if not in_queue:
                _logger.debug(f"Workflows endpoint(adhoc): fallback inline start for run_id={run_id}")
                await engine.start_run(run_id, run_mode)
            else:
                _logger.debug(f"Workflows endpoint(adhoc): run_id={run_id} is queued; skipping inline fallback")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    if run_mode == RunMode.SYNC:
        run = await _wait_for_run_completion(db, run_id)
    else:
        run = db.get_run(run_id)
    from loguru import logger as _logger
    with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
        _logger.debug(f"Workflows endpoint: post-submit (adhoc) status={run.status if run else 'missing'} run_id={run_id}")
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    # Audit: ad-hoc run created
    try:
        if audit_service:
            ctx = AuditContext(
                request_id=(request.headers.get("X-Request-ID") if request else None),
                user_id=str(current_user.id),
                ip_address=(request.client.host if request and request.client else None),
                user_agent=(request.headers.get("user-agent") if request else None),
                endpoint=str(request.url.path) if request else "/api/v1/workflows/run",
                method=(request.method if request else "POST"),
            )
            await audit_service.log_event(
                event_type=AuditEventType.API_REQUEST,
                category=AuditEventCategory.API_CALL,
                severity=AuditSeverity.INFO,
                context=ctx,
                resource_type="workflow_run",
                resource_id=str(run_id),
                action="run_adhoc",
                metadata={"mode": mode},
            )
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    return WorkflowRunResponse(
        id=run.run_id,
        run_id=run.run_id,
        workflow_id=run.workflow_id,
        user_id=str(run.user_id) if getattr(run, 'user_id', None) is not None else None,
        status=run.status,
        status_reason=run.status_reason,
        inputs=json.loads(run.inputs_json or "{}"),
        outputs=json.loads(run.outputs_json or "null") if run.outputs_json else None,
        error=run.error,
        definition_version=run.definition_version,
        validation_mode=getattr(run, 'validation_mode', None),
    )


@router.get(
    "/runs/{run_id}",
    response_model=WorkflowRunResponse,
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def get_run(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    # Tenant isolation
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        try:
            if audit_service:
                ctx = AuditContext(
                    request_id=(request.headers.get("X-Request-ID") if request else None),
                    user_id=str(current_user.id),
                    endpoint=str(request.url.path) if request else "/api/v1/workflows/runs/{id}",
                    method=(request.method if request else "GET"),
                )
                await audit_service.log_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    category=AuditEventCategory.SECURITY,
                    severity=AuditSeverity.WARNING,
                    context=ctx,
                    resource_type="workflow_run",
                    resource_id=str(run_id),
                    action="tenant_mismatch",
                )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows get_run: audit tenant_mismatch failed: {e}")
        raise HTTPException(status_code=404, detail="Run not found")
    # Owner or admin (if attribute available)
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        try:
            if audit_service:
                ctx = AuditContext(
                    request_id=(request.headers.get("X-Request-ID") if request else None),
                    user_id=str(current_user.id),
                    endpoint=str(request.url.path) if request else "/api/v1/workflows/runs/{id}",
                    method=(request.method if request else "GET"),
                )
                await audit_service.log_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    category=AuditEventCategory.SECURITY,
                    severity=AuditSeverity.WARNING,
                    context=ctx,
                    resource_type="workflow_run",
                    resource_id=str(run_id),
                    action="not_owner",
                )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows get_run: audit not_owner failed: {e}")
        raise HTTPException(status_code=404, detail="Run not found")
    return WorkflowRunResponse(
        id=run.run_id,
        run_id=run.run_id,
        workflow_id=run.workflow_id,
        user_id=str(run.user_id) if getattr(run, 'user_id', None) is not None else None,
        status=run.status,
        status_reason=run.status_reason,
        inputs=json.loads(run.inputs_json or "{}"),
        outputs=json.loads(run.outputs_json or "null") if run.outputs_json else None,
        error=run.error,
        definition_version=run.definition_version,
        validation_mode=getattr(run, 'validation_mode', None),
    )


@router.get(
    "/runs/{run_id}/events",
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "cURL",
                "source": "curl -H 'X-API-KEY: $API_KEY' 'http://127.0.0.1:8000/api/v1/workflows/runs/{run_id}/events?limit=100'",
            },
            {
                "lang": "bash",
                "label": "cURL (cursor)",
                "source": "# First page\nRESP=$(curl -sS -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs/{run_id}/events?limit=50\")\nNEXT=$(curl -sSI -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs/{run_id}/events?limit=50\" | awk -F': ' '/^Next-Cursor:/ {print $2}' | tr -d '\r')\n# Next page using the Next-Cursor header value\n[ -n \"$NEXT\" ] && curl -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/runs/{run_id}/events?limit=50&cursor=$NEXT\" || echo 'No Next-Cursor returned'"
            }
        ]
    },
    dependencies=[Depends(RequirePermission(WORKFLOWS_RUNS_READ))],
)
async def get_run_events(
    run_id: str,
    since: Optional[int] = Query(None, description="Return events with seq strictly greater than this value"),
    limit: int = Query(500, ge=1, le=1000),
    types: Optional[list[str]] = Query(None, description="Filter by event types (repeatable)"),
    cursor: Optional[str] = Query(None, description="Opaque continuation token (overrides since)"),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    response: Response,
):
    # Enforce tenant isolation and owner/admin
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    # Normalize types to lower-case for consistency with UI filter chips
    types_norm = [t.strip() for t in (types or []) if str(t).strip()]
    # Cursor token overrides since
    if cursor:
        try:
            import base64
            import json as _json
            pad = "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode((cursor + pad).encode("utf-8")).decode("utf-8")
            tok = _json.loads(raw)
            if isinstance(tok.get("last_seq"), int):
                since = int(tok["last_seq"])  # seek after this seq
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows events: failed to parse cursor token; ignoring. Error: {e}")
    events = db.get_events(run_id, since=since, limit=limit, types=types_norm if types_norm else None)
    out: list[EventResponse] = []
    for e in events:
        out.append(
            EventResponse(
                event_seq=e["event_seq"],
                event_type=e["event_type"],
                payload=e.get("payload_json") or {},
                created_at=e["created_at"],
            )
        )
    # Build next continuation token when we returned a full page
    if response is not None and len(out) == int(limit):
        try:
            last_seq = int(out[-1].event_seq) if hasattr(out[-1], "event_seq") else int(events[-1]["event_seq"])  # type: ignore
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows events: failed to derive last_seq: {e}")
            try:
                last_seq = int(events[-1]["event_seq"]) if events else None  # type: ignore
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e2:
                logger.debug(f"Workflows events: no last_seq available: {e2}")
                last_seq = None
        if last_seq is not None:
            import base64
            import json as _json
            token = {"last_seq": last_seq}
            raw = _json.dumps(token).encode("utf-8")
            nxt = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
            try:
                response.headers["Next-Cursor"] = nxt
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows events: failed to set Next-Cursor header: {e}")
            # Optional RFC5988 Link header for 'next'
            try:
                from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_link_header
                base_path = f"/api/v1/workflows/runs/{run_id}/events"
                params = [("limit", str(limit))]
                if types:
                    for t in types:
                        params.append(("types", t))
                link_value = build_link_header(
                    base_path,
                    params,
                    next_cursor=nxt,
                    limit=int(limit),
                    offset=None,
                    has_more=None,
                )
                if link_value:
                    response.headers["Link"] = link_value
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows events: failed to set Link header: {e}")
    return out


@router.get(
    "/runs/{run_id}/webhooks/deliveries",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def get_run_webhook_deliveries(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    """Return webhook delivery history for a run (derived from events)."""
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    events = db.get_events(run_id, types=["webhook_delivery"]) or []
    deliveries = []
    for e in events:
        p = e.get("payload_json") or {}
        deliveries.append({
            "event_seq": e.get("event_seq"),
            "created_at": e.get("created_at"),
            "host": p.get("host"),
            "status": p.get("status"),
            "code": p.get("code"),
        })
    return {"deliveries": deliveries}


@router.get(
    "/webhooks/dlq",
    dependencies=[
        # Permission-first gate so failures clearly attribute the missing
        # workflows.runs.control permission in error details, even when the
        # principal also lacks the admin role.
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
        Depends(RequireRole("admin")),
    ],
)
async def list_webhook_dlq(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    """Admin: list webhook DLQ entries (all tenants).

    Access is gated via claim-first dependencies (admin role +
    WORKFLOWS_RUNS_CONTROL) to align with other admin surfaces.
    """
    rows = db.list_webhook_dlq_all(limit=limit + 1, offset=offset)
    has_more = len(rows) > limit
    rows = rows[:limit]
    out = []
    for r in rows:
        try:
            body = json.loads(r.get("body_json") or "{}")
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            body = {}
        out.append({
            "id": r.get("id"),
            "tenant_id": r.get("tenant_id"),
            "run_id": r.get("run_id"),
            "url": r.get("url"),
            "attempts": r.get("attempts"),
            "next_attempt_at": r.get("next_attempt_at"),
            "last_error": r.get("last_error"),
            "created_at": r.get("created_at"),
            "body": body,
        })
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        count=len(out),
        has_more=has_more,
    )
    return {
        "items": out,
        "limit": limit,
        "offset": offset,
        "count": len(out),
        "pagination": pagination,
    }


@router.post(
    "/webhooks/dlq/{dlq_id}/replay",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
    ],
)
async def replay_webhook_dlq(
    dlq_id: int,
    _current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    """Admin: attempt immediate replay of a DLQ item.

    Honors the same allow/deny and replay headers as the engine.
    In TEST_MODE, if WORKFLOWS_TEST_REPLAY_SUCCESS=true, the entry is deleted
    without network. Access is gated via claim-first dependencies (admin role
    + WORKFLOWS_RUNS_CONTROL) consistent with other admin endpoints.
    """
    # Find the entry
    rows = db.list_webhook_dlq_all(limit=1, offset=0)
    target = None
    for r in rows:
        if int(r.get("id")) == int(dlq_id):
            target = r
            break
    if not target:
        # Slow path: scan in batches (SQLite-friendly)
        off = 0
        while True:
            batch = db.list_webhook_dlq_all(limit=200, offset=off)
            if not batch:
                break
            for r in batch:
                if int(r.get("id")) == int(dlq_id):
                    target = r
                    break
            if target:
                break
            off += len(batch)
    if not target:
        raise HTTPException(status_code=404, detail="DLQ item not found")

    url = str(target.get("url") or "")
    tenant_id = str(target.get("tenant_id") or "default")
    run_id_str = str(target.get("run_id") or "")
    try:
        body = json.loads(target.get("body_json") or "{}")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.debug(
            "Workflows DLQ replay: failed to parse body_json for dlq_id={}: {} - {}",
            dlq_id,
            type(exc).__name__,
            exc,
        )
        body = {}

    # Test-mode short-circuit
    import os as _os
    if is_test_mode() and is_truthy(_os.getenv("WORKFLOWS_TEST_REPLAY_SUCCESS", "")):
        try:
            record_webhook_delivery_event(
                db,
                tenant_id=tenant_id,
                run_id=run_id_str,
                url=url,
                status="delivered",
                source="dlq_replay",
            )
            db.delete_webhook_dlq(dlq_id=dlq_id)
        except sqlite3.Error as exc:
            logger.debug(
                "Workflows DLQ replay: failed to delete dlq_id={} in test mode: {} - {}",
                dlq_id,
                type(exc).__name__,
                exc,
            )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Workflows DLQ replay: failed to delete dlq_id={} in test mode: {} - {}",
                dlq_id,
                type(exc).__name__,
                exc,
            )
        return {"ok": True, "simulated": True}

    # Policy
    try:
        from tldw_Server_API.app.core.Security.egress import is_webhook_url_allowed_for_tenant as _allow_webhook
        if not _allow_webhook(url, tenant_id):
            raise HTTPException(status_code=400, detail="Denied by egress policy")
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Workflows DLQ replay: egress policy check failed for dlq_id={}: {} - {}",
            dlq_id,
            type(exc).__name__,
            exc,
        )

    # Attempt delivery with the same headers/signing as engine
    try:
        import hashlib
        import hmac
        import time as _time
        secret = _os.getenv("WORKFLOWS_WEBHOOK_SECRET", "")
        ts = str(int(_time.time()))
        headers = {
            "content-type": "application/json",
            "X-Signature-Timestamp": ts,
            "X-Webhook-ID": f"wf-dlq-{dlq_id}-{ts}",
            "X-Workflows-Signature-Version": "v1",
        }
        raw = json.dumps(body)
        if secret:
            sig = hmac.new(secret.encode("utf-8"), f"{ts}.{raw}".encode(), hashlib.sha256).hexdigest()
            headers["X-Workflows-Signature"] = sig
            headers["X-Hub-Signature-256"] = f"sha256={sig}"
        timeout = float(_os.getenv("WORKFLOWS_WEBHOOK_TIMEOUT", "10"))
        resp = await _http_afetch(
            method="POST",
            url=url,
            data=raw,
            headers=headers,
            timeout=timeout,
            retry=_RetryPolicy(attempts=1),
        )
        try:
            status_code = int(getattr(resp, "status_code", 0) or 0)
            logger.debug("DLQ replay POST to {} -> {}", url, status_code)
            if 200 <= status_code < 400:
                try:
                    record_webhook_delivery_event(
                        db,
                        tenant_id=tenant_id,
                        run_id=run_id_str,
                        url=url,
                        status="delivered",
                        code=status_code,
                        source="dlq_replay",
                    )
                    db.delete_webhook_dlq(dlq_id=dlq_id)
                except sqlite3.Error as exc:
                    logger.debug(
                        "Workflows DLQ replay: failed to delete dlq_id={}: {} - {}",
                        dlq_id,
                        type(exc).__name__,
                        exc,
                    )
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Workflows DLQ replay: failed to delete dlq_id={}: {} - {}",
                        dlq_id,
                        type(exc).__name__,
                        exc,
                    )
                return {"ok": True, "status_code": status_code}
            else:
                # Update attempts/backoff minimally
                error_category = _classify_webhook_status(status_code)
                try:
                    record_webhook_delivery_event(
                        db,
                        tenant_id=tenant_id,
                        run_id=run_id_str,
                        url=url,
                        status="failed",
                        code=status_code,
                        reason=f"status={status_code}",
                        source="dlq_replay",
                    )
                    db.update_webhook_dlq_failure(
                        dlq_id=dlq_id,
                        last_error=f"status={status_code}",
                        next_attempt_at_iso=None,
                    )
                except sqlite3.Error as exc:
                    logger.debug(
                        "Workflows DLQ replay: failed to update failure record for dlq_id={}: {} - {}",
                        dlq_id,
                        type(exc).__name__,
                        exc,
                    )
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Workflows DLQ replay: failed to update failure record for dlq_id={}: {} - {}",
                        dlq_id,
                        type(exc).__name__,
                        exc,
                    )
                return {
                    "ok": False,
                    "status_code": status_code,
                    "error": "delivery_failed",
                    "error_category": error_category,
                }
        finally:
            close = getattr(resp, "aclose", None)
            if callable(close):
                await close()
            else:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        error_category = _classify_webhook_exception(e)
        logger.debug(
            "Workflows DLQ replay: delivery failed for dlq_id={}: {} - {}",
            dlq_id,
            type(e).__name__,
            e,
        )
        try:
            error_detail = f"{type(e).__name__}: {e}"
            record_webhook_delivery_event(
                db,
                tenant_id=tenant_id,
                run_id=run_id_str,
                url=url,
                status="failed",
                reason=error_detail,
                source="dlq_replay",
            )
            db.update_webhook_dlq_failure(dlq_id=dlq_id, last_error=error_detail, next_attempt_at_iso=None)
        except sqlite3.Error as exc:
            logger.debug(
                "Workflows DLQ replay: failed to update failure record for dlq_id={}: {} - {}",
                dlq_id,
                type(exc).__name__,
                exc,
            )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Workflows DLQ replay: failed to update failure record for dlq_id={}: {} - {}",
                dlq_id,
                type(exc).__name__,
                exc,
            )
        return {
            "ok": False,
            "error": "delivery_failed",
            "error_type": type(e).__name__,
            "error_category": error_category,
        }


def _artifact_validation_strict(validation_mode: Optional[str]) -> bool:
    env_strict = is_truthy(os.getenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true"))
    run_val = str(validation_mode or "").lower()
    non_block = run_val == "non-block"
    return env_strict and not non_block


def _resolve_artifact_file_path(
    *,
    uri: str,
    workdir: Optional[str],
    validation_mode: Optional[str],
) -> Path:
    if not uri.startswith("file://"):
        raise HTTPException(status_code=400, detail="Only file artifacts are downloadable")
    fpath = uri[len("file://") :]
    p = Path(fpath).resolve()

    allowed_roots: list[Path] = []
    with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
        allowed_roots.append(artifacts_base_dir())
    if workdir:
        try:
            wd = Path(str(workdir).replace("file://", "")).resolve()
            allowed_roots.append(wd)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            pass

    allowed = any(is_subpath(root, p) for root in allowed_roots if root)
    if not allowed:
        if _artifact_validation_strict(validation_mode):
            raise HTTPException(status_code=400, detail="Invalid artifact path scope")
        logger.warning(
            'Workflows artifact path outside allowed roots; proceeding due to non-strict setting: {}',
            p,
        )
    return p


@router.get(
    "/runs/{run_id}/artifacts",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def get_run_artifacts(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    arts = db.list_artifacts_for_run(run_id)
    # Normalize payload
    results = []
    for a in arts:
        results.append({
            "artifact_id": a.get("artifact_id"),
            "type": a.get("type"),
            "uri": a.get("uri"),
            "size_bytes": a.get("size_bytes"),
            "mime_type": a.get("mime_type"),
            "checksum_sha256": a.get("checksum_sha256"),
            "metadata": a.get("metadata_json") or {},
            "created_at": a.get("created_at"),
            "step_run_id": a.get("step_run_id"),
        })
    return results


@router.get(
    "/runs/{run_id}/artifacts/manifest",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def get_run_artifacts_manifest(
    run_id: str,
    verify: bool = Query(False, description="Compute and validate recorded checksums"),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")

    arts = db.list_artifacts_for_run(run_id)
    manifest = []
    mismatches = 0
    for a in arts:
        entry = {
            "artifact_id": a.get("artifact_id"),
            "type": a.get("type"),
            "uri": a.get("uri"),
            "size_bytes": a.get("size_bytes"),
            "mime_type": a.get("mime_type"),
            "checksum_sha256": a.get("checksum_sha256"),
            "created_at": a.get("created_at"),
            "metadata": a.get("metadata_json") or {},
        }
        if verify and entry["uri"] and str(entry["uri"]).startswith("file://") and entry.get("checksum_sha256"):
            try:
                import hashlib as _h
                meta = entry.get("metadata")
                workdir = meta.get("workdir") if isinstance(meta, dict) else None
                fp = _resolve_artifact_file_path(
                    uri=str(entry["uri"]),
                    workdir=workdir,
                    validation_mode=getattr(run, "validation_mode", None),
                )
                if fp.exists() and fp.is_file():
                    h = _h.sha256()
                    with fp.open("rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    calc = h.hexdigest()
                    if calc != entry.get("checksum_sha256"):
                        entry["integrity"] = {"ok": False, "calculated": calc}
                        mismatches += 1
                    else:
                        entry["integrity"] = {"ok": True}
            except HTTPException:
                entry["integrity"] = {"ok": False, "error": "path_outside_allowed_dir"}
                mismatches += 1
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                entry["integrity"] = {"ok": False, "error": "hash_error"}
                mismatches += 1
        manifest.append(entry)

    resp = {"artifacts": manifest}
    if verify:
        resp["integrity_summary"] = {"mismatch_count": mismatches}
    return resp


class VerifyBatchItem(BaseModel):
    artifact_id: str
    expected_sha256: Optional[str] = None


class VerifyBatchRequest(BaseModel):
    items: list[VerifyBatchItem]


@router.post(
    "/runs/{run_id}/artifacts/verify-batch",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def verify_artifacts_batch(
    run_id: str,
    body: VerifyBatchRequest,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")

    import hashlib as _h
    results = []
    for item in (body.items or []):
        aid = str(item.artifact_id)
        a = db.get_artifact(aid)
        if not a:
            results.append({"artifact_id": aid, "ok": False, "error": "not_found"})
            continue
        uri = str(a.get("uri") or "")
        if not uri.startswith("file://"):
            results.append({"artifact_id": aid, "ok": False, "error": "unsupported_uri"})
            continue
        meta = a.get("metadata_json")
        workdir = meta.get("workdir") if isinstance(meta, dict) else None
        try:
            fp = _resolve_artifact_file_path(
                uri=uri,
                workdir=workdir,
                validation_mode=getattr(run, "validation_mode", None),
            )
        except HTTPException:
            results.append({"artifact_id": aid, "ok": False, "error": "path_outside_allowed_dir"})
            continue
        if not (fp.exists() and fp.is_file()):
            results.append({"artifact_id": aid, "ok": False, "error": "file_missing"})
            continue
        try:
            h = _h.sha256()
            with fp.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            calc = h.hexdigest()
            recorded = str(a.get("checksum_sha256") or "").strip() or None
            expected = str(item.expected_sha256 or "").strip() or recorded
            mismatch = (expected is not None and calc != expected)
            results.append({
                "artifact_id": aid,
                "ok": not mismatch,
                "calculated": calc,
                "expected": expected,
                "recorded": recorded,
            })
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(
                "Workflows artifact verify: hash failed for artifact_id={}: {} - {}",
                aid,
                type(e).__name__,
                e,
            )
            results.append({
                "artifact_id": aid,
                "ok": False,
                "error": "hash_error",
                "error_type": type(e).__name__,
            })

    return {"results": results}


@router.get(
    "/artifacts/{artifact_id}/download",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Workflow artifact file download.",
            "content": {"application/octet-stream": {}},
        },
        206: {
            "description": "Partial workflow artifact file download.",
            "content": {"application/octet-stream": {}},
        },
    },
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def download_artifact(
    artifact_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
    *,
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
):
    import os as _os
    art = db.get_artifact(artifact_id)
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    run = db.get_run(str(art.get("run_id"))) if art.get("run_id") else None
    if not run:
        raise HTTPException(status_code=404, detail="Artifact not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        try:
            if audit_service:
                ctx = AuditContext(
                    request_id=(request.headers.get("X-Request-ID") if request else None),
                    user_id=str(current_user.id),
                    endpoint=str(request.url.path) if request else "/api/v1/workflows/artifacts/{id}/download",
                    method=(request.method if request else "GET"),
                )
                await audit_service.log_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    category=AuditEventCategory.SECURITY,
                    severity=AuditSeverity.WARNING,
                    context=ctx,
                    resource_type="artifact",
                    resource_id=str(artifact_id),
                    action="tenant_mismatch",
                )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            pass
        raise HTTPException(status_code=404, detail="Artifact not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        try:
            if audit_service:
                ctx = AuditContext(
                    request_id=(request.headers.get("X-Request-ID") if request else None),
                    user_id=str(current_user.id),
                    endpoint=str(request.url.path) if request else "/api/v1/workflows/artifacts/{id}/download",
                    method=(request.method if request else "GET"),
                )
                await audit_service.log_event(
                    event_type=AuditEventType.PERMISSION_DENIED,
                    category=AuditEventCategory.SECURITY,
                    severity=AuditSeverity.WARNING,
                    context=ctx,
                    resource_type="artifact",
                    resource_id=str(artifact_id),
                    action="not_owner",
                )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            pass
        raise HTTPException(status_code=404, detail="Artifact not found")
    # Only support file:// URIs for direct download
    uri = str(art.get("uri") or "")
    meta = art.get("metadata_json")
    workdir = meta.get("workdir") if isinstance(meta, dict) else None
    p = _resolve_artifact_file_path(
        uri=uri,
        workdir=workdir,
        validation_mode=getattr(run, "validation_mode", None),
    )
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Optional integrity verification when checksum recorded
    try:
        checksum = str(art.get("checksum_sha256") or "").strip()
        if checksum:
            import hashlib as _hashlib
            h = _hashlib.sha256()
            with p.open("rb") as _f:
                for chunk in iter(lambda: _f.read(65536), b""):
                    h.update(chunk)
            calc = h.hexdigest()
            if calc != checksum:
                # Respect validation mode override
                run_val = getattr(run, 'validation_mode', None)
                non_block = isinstance(run_val, str) and run_val.lower() == 'non-block'
                if not non_block and is_truthy(_os.getenv("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", "true")):
                    raise HTTPException(status_code=409, detail="Artifact checksum mismatch")
                else:
                    with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
                        logger.warning(f"Artifact checksum mismatch for {artifact_id}; proceeding due to non-strict mode")
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        # Do not block on hashing errors
        pass
    # Guardrails: max size and allowed MIME
    import mimetypes as _m
    max_bytes = int(_os.getenv("WORKFLOWS_ARTIFACT_MAX_DOWNLOAD_BYTES", "10485760"))
    try:
        if p.stat().st_size > max_bytes:
            raise HTTPException(status_code=413, detail="Artifact too large to download")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    # MIME allowlist
    allowed = [s.strip() for s in (_os.getenv("WORKFLOWS_ARTIFACT_ALLOWED_MIME", "text/plain,text/markdown,application/json,application/pdf,image/png,image/jpeg").split(",")) if s.strip()]
    mime = art.get("mime_type") or _m.guess_type(str(p))[0] or "application/octet-stream"
    if allowed and not any(mime == a or (a.endswith("/*") and mime.startswith(a[:-1])) for a in allowed):
        raise HTTPException(status_code=415, detail=f"MIME not allowed: {mime}")
    # Support Range requests (single range)
    range_header = None
    try:
        range_header = request.headers.get("range") if request else None
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        range_header = None
    # Build a safe Content-Disposition filename to avoid non-ASCII header issues under fuzzing
    def _safe_disp_parts(name: str) -> tuple[str, Optional[str]]:
        try:
            name.encode("ascii")
            return name, None
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            try:
                import urllib.parse as _u
                return "download", _u.quote(name)
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                return "download", None

    if range_header and range_header.lower().startswith("bytes="):
        try:
            total = p.stat().st_size
            rng = range_header.split("=", 1)[1]
            start_str, end_str = (rng.split("-", 1) + [""])[:2]
            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else total - 1
            else:
                # suffix-length: bytes=-N
                length = int(end_str)
                start = max(0, total - length)
                end = total - 1
            if start < 0 or end < start or end >= total:
                raise ValueError("invalid_range")
            length = end - start + 1
            ascii_name, encoded_name = _safe_disp_parts(p.name)
            headers = {
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": mime,
                "Content-Disposition": (
                    f"attachment; filename={ascii_name}"
                    + (f"; filename*=UTF-8''{encoded_name}" if encoded_name else "")
                ),
            }
            def _iter():
                with p.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    chunk = 64 * 1024
                    while remaining > 0:
                        data = f.read(min(chunk, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            return StreamingResponse(_iter(), status_code=206, headers=headers)
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            # 416 Range Not Satisfiable
            raise HTTPException(status_code=416, detail="Invalid Range header") from None
    # Full response
    ascii_name, encoded_name = _safe_disp_parts(p.name)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": (
            f"attachment; filename={ascii_name}" + (f"; filename*=UTF-8''{encoded_name}" if encoded_name else "")
        ),
    }
    return FileResponse(str(p), media_type=mime, headers=headers)


@router.get(
    "/runs/{run_id}/artifacts/download",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Workflow run artifacts ZIP download.",
            "content": {"application/zip": {}},
        },
    },
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def download_run_artifacts_zip(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    import io
    import mimetypes as _m
    import os as _os
    import zipfile

    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")

    arts = db.list_artifacts_for_run(run_id)
    if not arts:
        raise HTTPException(status_code=404, detail="No artifacts for run")

    # Constraints
    max_total = int(_os.getenv("WORKFLOWS_ARTIFACT_BULK_MAX_BYTES", "52428800"))  # 50MB
    allowed = [s.strip() for s in (_os.getenv("WORKFLOWS_ARTIFACT_ALLOWED_MIME", "text/plain,text/markdown,application/json,application/pdf,image/png,image/jpeg").split(",")) if s.strip()]

    # Preselect eligible files and sum sizes
    selected = []
    total = 0
    for a in arts:
        uri = str(a.get("uri") or "")
        if not uri.startswith("file://"):
            continue
        meta = a.get("metadata_json")
        workdir = meta.get("workdir") if isinstance(meta, dict) else None
        try:
            p = _resolve_artifact_file_path(
                uri=uri,
                workdir=workdir,
                validation_mode=getattr(run, "validation_mode", None),
            )
        except HTTPException:
            continue
        if not p.exists() or not p.is_file():
            continue
        try:
            size_b = p.stat().st_size
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
            size_b = 0
        mime = a.get("mime_type") or _m.guess_type(str(p))[0] or "application/octet-stream"
        if allowed and not any(mime == m or (m.endswith("/*") and mime.startswith(m[:-1])) for m in allowed):
            continue
        total += size_b or 0
        if total > max_total:
            raise HTTPException(status_code=413, detail="Total artifact size too large for bulk download")
        selected.append((p, mime))

    if not selected:
        raise HTTPException(status_code=404, detail="No eligible artifacts for bulk download")

    # Build zip in-memory (store)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for p, _mime in selected:
            try:
                zf.write(str(p), arcname=p.name)
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                continue
    buf.seek(0)
    filename = f"artifacts_{run_id}.zip"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


# -------------------- Options Discovery: Chunkers --------------------

@router.get("/options/chunkers")
async def get_chunker_options():
    """Return available chunking methods with defaults and a basic parameter schema.

    This introspects the Chunking module to surface methods and default options
    for UI builders. Version is static for now.
    """
    try:
        from tldw_Server_API.app.core.Chunking import DEFAULT_CHUNK_OPTIONS
        from tldw_Server_API.app.core.Chunking.base import ChunkingMethod
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        raise HTTPException(status_code=500, detail="Chunking module unavailable") from None

    defaults = DEFAULT_CHUNK_OPTIONS.copy()
    methods = [m.value for m in ChunkingMethod]
    # Build a basic param schema (types are indicative)
    schema = {
        "method": {"type": "string", "enum": methods, "default": defaults.get("method")},
        "max_size": {"type": "integer", "default": defaults.get("max_size")},
        "overlap": {"type": "integer", "default": defaults.get("overlap")},
        "language": {"type": "string", "default": defaults.get("language")},
        "adaptive": {"type": "boolean", "default": defaults.get("adaptive")},
        "multi_level": {"type": "boolean", "default": defaults.get("multi_level")},
        "semantic_similarity_threshold": {"type": "number", "default": defaults.get("semantic_similarity_threshold")},
        "semantic_overlap_sentences": {"type": "integer", "default": defaults.get("semantic_overlap_sentences")},
        "json_chunkable_data_key": {"type": "string", "default": defaults.get("json_chunkable_data_key")},
        "summarization_detail": {"type": "number", "default": defaults.get("summarization_detail")},
        "tokenizer_name_or_path": {"type": "string", "default": defaults.get("tokenizer_name_or_path")},
        "proposition_engine": {"type": "string", "enum": ["heuristic", "spacy", "llm", "auto"], "default": defaults.get("proposition_engine")},
        "proposition_aggressiveness": {"type": "integer", "default": defaults.get("proposition_aggressiveness")},
        "proposition_min_proposition_length": {"type": "integer", "default": defaults.get("proposition_min_proposition_length")},
        "proposition_prompt_profile": {"type": "string", "default": defaults.get("proposition_prompt_profile")},
    }

    return {
        "name": "core_chunking",
        "version": "1.0.0",
        "methods": methods,
        "defaults": defaults,
        "parameter_schema": schema,
    }





@router.get(
    "/runs/{run_id}/investigation",
    response_model=WorkflowRunInvestigationResponse,
    dependencies=[Depends(RequirePermission(WORKFLOWS_RUNS_READ))],
)
async def get_run_investigation(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    _run, is_admin = _get_authorized_run_or_404(run_id=run_id, current_user=current_user, db=db)
    investigation = build_run_investigation(
        db,
        run_id=run_id,
        include_operator_detail=is_admin,
    )
    if investigation is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return WorkflowRunInvestigationResponse(**investigation)


@router.get(
    "/runs/{run_id}/steps",
    response_model=WorkflowRunStepsResponse,
    dependencies=[Depends(RequirePermission(WORKFLOWS_RUNS_READ))],
)
async def get_run_steps(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    _run, is_admin = _get_authorized_run_or_404(run_id=run_id, current_user=current_user, db=db)
    steps = build_run_steps(
        db,
        run_id=run_id,
        include_operator_detail=is_admin,
    )
    if steps is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return WorkflowRunStepsResponse(**steps)


@router.get(
    "/runs/{run_id}/steps/{step_id}/attempts",
    response_model=WorkflowStepAttemptsResponse,
    dependencies=[Depends(RequirePermission(WORKFLOWS_RUNS_READ))],
)
async def get_step_attempts(
    run_id: str,
    step_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    _run, is_admin = _get_authorized_run_or_404(run_id=run_id, current_user=current_user, db=db)
    attempts = build_step_attempts(
        db,
        run_id=run_id,
        step_id=step_id,
        include_operator_detail=is_admin,
    )
    if attempts is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return WorkflowStepAttemptsResponse(**attempts)


@router.post(
    "/runs/{run_id}/{action}",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
    ],
)
async def control_run(
    run_id: str,
    action: str,
    request: Request,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    # Enforce tenant + owner/admin
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    engine = WorkflowEngine(db)
    # Admin impersonation audit trail (header opt-in)
    try:
        imp = str(request.headers.get("x-impersonate-user", "")).strip()
        is_admin = _is_workflows_admin_user(current_user)
        if imp and is_admin and str(imp) != str(current_user.id):
            db.append_event(str(getattr(current_user, 'tenant_id', 'default')), run_id, "admin_impersonation", {"actor": str(current_user.id), "target_user_id": imp, "action": action})
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        pass
    result = "applied"
    if action == "pause":
        result = engine.pause(run_id)
    elif action == "resume":
        result = engine.resume(run_id)
    elif action == "cancel":
        result = engine.cancel(run_id)
    elif action == "retry":
        run = db.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        # Delegate to retry behavior
        failed_step = db.get_last_failed_step_id(run_id)
        if failed_step:
            asyncio.create_task(engine.continue_run(run_id, after_step_id=failed_step, last_outputs=None))
        else:
            engine.submit(run_id, RunMode.ASYNC)
    else:
        raise HTTPException(status_code=400, detail="Unsupported action")
    return {"ok": True, "result": result}


@router.get(
    "/step-types",
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "cURL",
                "source": "curl -H 'X-API-KEY: $API_KEY' http://127.0.0.1:8000/api/v1/workflows/step-types",
            }
        ]
    },
)
async def list_step_types():
    """Return available step types with basic JSONSchema, examples, and min engine version.

    This is a lightweight introspection surface to help UIs validate configurations.
    """
    reg = StepTypeRegistry()
    steps = reg.list()
    # Minimal schemas (can be expanded incrementally)
    schemas = {
        "prompt": {
            "type": "object",
            "properties": {
                "template": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 300},
                "retry": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": ["template"],
            "additionalProperties": True,
            "example": {"template": "Hello {{inputs.name}}"},
            "min_engine_version": "0.1.0",
        },
        "llm": {
            **_llm_step_schema_base(),
            "example": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt": "Summarize {{ inputs.topic }}",
                "system": "You are a concise assistant.",
                "seed": 42,
                "max_completion_tokens": 256,
            },
            "min_engine_version": "0.1.1",
        },
        "branch": {
            "type": "object",
            "properties": {
                "condition": {"type": "string"},
                "true_next": {"type": "string"},
                "false_next": {"type": "string"}
            },
            "required": ["condition"],
            "additionalProperties": False,
            "example": {"condition": "{{ inputs.enabled }}", "true_next": "step_b", "false_next": "step_c"},
            "min_engine_version": "0.1.1",
        },
        "map": {
            "type": "object",
            "properties": {
                "items": {"type": ["array", "string"]},
                "step": {"type": "object"},
                "concurrency": {"type": "integer", "minimum": 1, "default": 4}
            },
            "required": ["items", "step"],
            "additionalProperties": True,
            "example": {"items": [1,2,3], "step": {"type":"log", "config": {"message": "Item {{ item }}"}}, "concurrency": 2},
            "min_engine_version": "0.1.1",
        },
        "rag_search": {
            **_rag_search_schema_base(),
            "example": {"query": "large language models safety", "top_k": 8},
            "min_engine_version": "0.1.0",
        },
        "deep_research": {
            **_deep_research_step_schema_base(),
            "example": {
                "query": "Investigate {{ inputs.topic }}",
                "source_policy": "balanced",
                "autonomy_mode": "checkpointed",
                "save_artifact": True,
            },
            "min_engine_version": "0.1.0",
        },
        "deep_research_wait": {
            **_deep_research_wait_step_schema_base(),
            "example": {
                "run_id": "{{ deep_research.run_id }}",
                "include_bundle": True,
                "fail_on_cancelled": True,
                "fail_on_failed": True,
                "poll_interval_seconds": 2.0,
                "save_artifact": True,
            },
            "min_engine_version": "0.1.0",
        },
        "deep_research_load_bundle": {
            **_deep_research_load_bundle_step_schema_base(),
            "example": {
                "run_id": "{{ deep_research_wait.run_id }}",
                "save_artifact": True,
            },
            "min_engine_version": "0.1.0",
        },
        "deep_research_select_bundle_fields": {
            **_deep_research_select_bundle_fields_step_schema_base(),
            "example": {
                "run_id": "{{ deep_research_wait.run_id }}",
                "fields": [
                    "question",
                    "verification_summary",
                    "unsupported_claims",
                ],
                "save_artifact": True,
            },
            "min_engine_version": "0.1.0",
        },
        "kanban": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "board_id": {"type": ["integer", "string"]},
                "list_id": {"type": ["integer", "string"]},
                "card_id": {"type": ["integer", "string"]},
            },
            "required": ["action"],
            "additionalProperties": True,
            "example": {"action": "card.create", "list_id": 123, "title": "Review {{ inputs.topic }}", "client_id": "wf-card-1"},
            "min_engine_version": "0.1.4",
        },
        "webhook": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "include_outputs": {"type": "boolean", "default": True},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 10},
                "follow_redirects": {"type": "boolean", "default": False},
                "max_redirects": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1},
                "egress_policy": {
                    "type": "object",
                    "properties": {
                        "allowlist": {"type": "array", "items": {"type": "string"}},
                        "denylist": {"type": "array", "items": {"type": "string"}},
                        "block_private": {"type": "boolean"},
                    },
                    "additionalProperties": True,
                },
                "signing": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "secret_ref": {"type": "string"},
                        "secret": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "required": [],
            "additionalProperties": True,
            "example": {"url": "https://example.com/hooks/workflow", "follow_redirects": False, "max_bytes": 1048576},
            "min_engine_version": "0.1.0",
        },
        "acp_stage": {
            "type": "object",
            "properties": {
                "stage": {"type": "string"},
                "prompt_template": {"type": "string"},
                "prompt": {"type": ["array", "string"]},
                "session_id": {"type": "string"},
                "session_context_key": {"type": "string", "default": "acp_session_id"},
                "create_session": {"type": "boolean", "default": True},
                "cwd": {"type": "string", "default": "/workspace"},
                "agent_type": {"type": "string"},
                "persona_id": {"type": "string"},
                "workspace_id": {"type": "string"},
                "workspace_group_id": {"type": "string"},
                "scope_snapshot_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600, "default": 300},
                "review_counter_key": {"type": "string"},
                "max_review_loops": {"type": "integer", "minimum": 1, "maximum": 20},
                "fail_on_error": {"type": "boolean", "default": False},
            },
            "required": ["stage"],
            "anyOf": [
                {"required": ["prompt_template"]},
                {"required": ["prompt"]},
            ],
            "additionalProperties": True,
            "example": {
                "stage": "impl",
                "prompt_template": "Implement {{ inputs.task }}",
                "workspace_id": "{{ inputs.workspace_id }}",
                "workspace_group_id": "{{ inputs.workspace_group_id }}",
                "session_context_key": "pipeline_acp_session_id",
            },
            "min_engine_version": "0.1.0",
        },
        "tts": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Templated text. Defaults to last.text or inputs.summary"},
                "model": {"type": "string", "default": "kokoro"},
                "voice": {"type": "string", "default": "af_heart"},
                "response_format": {"type": "string", "enum": ["mp3","wav","opus","flac","aac","pcm"], "default": "mp3"},
                "speed": {"type": "number", "minimum": 0.25, "maximum": 4.0, "default": 1.0},
                "provider": {"type": "string"},
                "output_filename_template": {"type": "string", "description": "e.g., audio_{{timestamp}}_{{voice}}"},
                "provider_options": {"type": "object"},
                "attach_download_link": {"type": "boolean", "default": False},
                "save_transcript": {"type": "boolean", "default": False},
                "post_process": {
                    "type": "object",
                    "properties": {
                        "normalize": {"type": "boolean", "default": False},
                        "target_lufs": {"type": "number", "default": -16.0},
                        "true_peak_dbfs": {"type": "number", "default": -1.5},
                        "lra": {"type": "number", "default": 11.0}
                    }
                }
            },
            "required": [],
            "additionalProperties": True,
            "example": {"input": "Narrate: {{ last.text }}", "model": "kokoro", "voice": "af_heart", "response_format": "mp3"},
            "min_engine_version": "0.1.2",
        },
        "delay": {
            "type": "object",
            "properties": {"milliseconds": {"type": "integer", "minimum": 1, "default": 1000}},
            "required": ["milliseconds"],
            "additionalProperties": False,
            "example": {"milliseconds": 500},
            "min_engine_version": "0.1.0",
        },
        "log": {
            "type": "object",
            "properties": {"message": {"type": "string"}, "level": {"type": "string", "enum": ["debug", "info", "warning", "error"], "default": "info"}},
            "required": ["message"],
            "additionalProperties": True,
            "example": {"message": "step reached", "level": "debug"},
            "min_engine_version": "0.1.0",
        },
        "process_media": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["web_scraping","pdf","ebook","xml","mediawiki_dump","podcast"], "default": "web_scraping"},
                "scrape_method": {"type": "string", "default": "Individual URLs"},
                "url_input": {"type": "string"},
                "url_level": {"type": ["integer", "null"]},
                "max_pages": {"type": "integer", "default": 10},
                "max_depth": {"type": "integer", "default": 3},
                "summarize": {"type": "boolean", "default": False},
                "custom_prompt": {"type": ["string","null"]},
                "api_name": {"type": ["string","null"]},
                "system_prompt": {"type": ["string","null"]},
                "temperature": {"type": "number", "default": 0.7},
                "custom_cookies": {"type": ["array","null"]},
                "user_agent": {"type": ["string","null"]},
                "custom_headers": {"type": ["object","null"]},
                "file_uri": {"type": "string", "description": "file:// path for pdf/ebook/xml/mediawiki_dump"},
                "url": {"type": "string", "description": "Podcast URL (for kind=podcast)"}
            },
            "required": [],
            "additionalProperties": True,
            "example": {"kind": "web_scraping", "scrape_method": "Individual URLs", "url_input": "https://example.com/article"},
            "min_engine_version": "0.1.2"
        },
        "rss_fetch": {
            "type": "object",
            "properties": {
                "urls": {"type": ["array","string"], "description": "RSS/Atom feed URLs (list or string)"},
                "limit": {"type": "integer", "default": 10},
                "include_content": {"type": "boolean", "default": True}
            },
            "required": ["urls"],
            "additionalProperties": True,
            "example": {"urls": ["https://example.com/feed.xml"], "limit": 5},
            "min_engine_version": "0.1.3"
        },
        "atom_fetch": {
            "type": "object",
            "properties": {
                "urls": {"type": ["array","string"]},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["urls"],
            "additionalProperties": True,
            "example": {"urls": ["https://example.com/atom.xml"]},
            "min_engine_version": "0.1.3"
        },
        "embed": {
            "type": "object",
            "properties": {
                "texts": {"type": ["array","string"]},
                "collection": {"type": "string"},
                "model_id": {"type": ["string","null"]},
                "metadata": {"type": ["object","null"]}
            },
            "required": [],
            "additionalProperties": True,
            "example": {"texts": "{{ last.text }}", "collection": "user_1_workflows"},
            "min_engine_version": "0.1.3"
        },
        "translate": {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "target_lang": {"type": "string", "default": "en"},
                "provider": {"type": ["string","null"]},
                "model": {"type": ["string","null"]}
            },
            "required": ["target_lang"],
            "additionalProperties": True,
            "example": {"input": "{{ last.text }}", "target_lang": "fr"},
            "min_engine_version": "0.1.3"
        },
        "stt_transcribe": {
            "type": "object",
            "properties": {
                "file_uri": {"type": "string", "description": "file:// path to audio/video"},
                "model": {"type": "string", "default": "large-v3"},
                "language": {"type": ["string","null"]},
                "diarize": {"type": "boolean", "default": False},
                "word_timestamps": {"type": "boolean", "default": False}
            },
            "required": ["file_uri"],
            "additionalProperties": True,
            "example": {"file_uri": "file:///abs/path/audio.wav", "model": "large-v3"},
            "min_engine_version": "0.1.3"
        },
        "notify": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "message": {"type": "string"},
                "subject": {"type": ["string","null"]},
                "headers": {"type": ["object","null"]}
            },
            "required": ["url","message"],
            "additionalProperties": True,
            "example": {"url": "https://hooks.slack.com/services/...", "message": "{{ last.text }}"},
            "min_engine_version": "0.1.3"
        },
        "diff_change_detector": {
            "type": "object",
            "properties": {
                "current": {"type": "string"},
                "method": {"type": "string", "enum": ["ratio","unified"], "default": "ratio"},
                "threshold": {"type": "number", "default": 0.9}
            },
            "required": [],
            "additionalProperties": True,
            "example": {"current": "{{ last.text }}", "method": "ratio", "threshold": 0.95},
            "min_engine_version": "0.1.3"
        },
        "policy_check": {
            "type": "object",
            "properties": {
                "text_source": {"type": "string", "enum": ["last","inputs","field"], "default": "last"},
                "field": {"type": "string"},
                "block_on_pii": {"type": "boolean", "default": False},
                "block_words": {"type": "array", "items": {"type": "string"}},
                "max_length": {"type": "integer", "minimum": 1},
                "redact_preview": {"type": "boolean", "default": False}
            },
            "required": [],
            "additionalProperties": True,
            "example": {"text_source": "last", "block_on_pii": True, "block_words": ["secret","password"], "max_length": 10000},
            "min_engine_version": "0.1.3"
        },
        "wait_for_human": {
            "type": "object",
            "properties": {
                "instructions": {"type": "string"},
                "assigned_to_user_id": {"type": ["string","integer"]}
            },
            "required": ["assigned_to_user_id"],
            "additionalProperties": True,
            "example": {"instructions": "Review the summary and approve.", "assigned_to_user_id": 1},
            "min_engine_version": "0.1.3"
        },
        "wait_for_approval": {
            "type": "object",
            "properties": {
                "instructions": {"type": "string"},
                "assigned_to_user_id": {"type": ["string","integer"]}
            },
            "required": ["assigned_to_user_id"],
            "additionalProperties": True,
            "example": {"instructions": "Await approval.", "assigned_to_user_id": 1},
            "min_engine_version": "0.1.3"
        },
    }
    out = []
    for s in steps:
        sch = schemas.get(s.name, {"type": "object", "additionalProperties": True, "min_engine_version": "0.1.0"})
        out.append({
            "name": s.name,
            "description": s.description,
            "schema": sch,
            "example": sch.get("example", {}),
            "min_engine_version": sch.get("min_engine_version", "0.1.0"),
            "capabilities": s.capability.to_dict(),
        })
    return out


@router.get("/templates")
async def list_workflow_templates(q: Optional[str] = Query(None, description="Search query on name/title/tags"), tag: Optional[str] = Query(None, description="Filter by tag")) -> list[dict[str, Any]]:
    """List available example workflow templates shipped with the server.

    Returns a list of {name, filename} items. Use GET /api/v1/workflows/templates/{name}
    to retrieve the JSON content.
    """
    try:
        # Locate Samples/Workflows by walking upwards for robustness
        here = Path(__file__).resolve()
        tpl_dir = None
        p = here
        for _ in range(0, 9):
            candidate = p.parent / "Samples" / "Workflows"
            if candidate.exists():
                tpl_dir = candidate
                break
            p = p.parent
        if tpl_dir is None:
            return []
        items: list[dict[str, str]] = []
        if tpl_dir.exists():
            for p in tpl_dir.glob("*.workflow.json"):
                try:
                    # Derive template name without the trailing ".workflow.json"
                    fname = p.name
                    if fname.endswith(".workflow.json"):
                        name = fname[: -len(".workflow.json")]
                    else:
                        # Fallback: stem (may include an extra suffix on some platforms)
                        name = p.stem
                    # Load title from the template JSON's "name" field for a human-friendly label
                    title: str = name
                    try:
                        import json as _json
                        raw = p.read_text(encoding="utf-8")
                        data = _json.loads(raw)
                        tpl_title = str(data.get("name") or "").strip()
                        tags = data.get("tags") or []
                        if tpl_title:
                            title = tpl_title
                        # Normalize tags to list[str]
                        if not isinstance(tags, list):
                            tags = []
                    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                        # Ignore parse errors; fall back to filename-derived name
                        tags = []
                    item = {"name": name, "filename": p.name, "title": title, "tags": tags}
                    items.append(item)
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                    continue
        # Optional search and tag filtering
        def _match(s: str, query: str) -> bool:
            try:
                return query.lower() in s.lower()
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                return False
        if q:
            items = [it for it in items if _match(it.get("name", ""), q) or _match(it.get("title", ""), q) or any(_match(t, q) for t in (it.get("tags") or []))]
        if tag:
            items = [it for it in items if str(tag) in (it.get("tags") or [])]
        return items
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to list workflow templates: {e}")
        return []


@router.get(
    "/templates/tags",
    openapi_extra={
        "x-codeSamples": [
            {
                "lang": "bash",
                "label": "List tags",
                "source": "curl -sS -H 'X-API-KEY: $API_KEY' \"$BASE/api/v1/workflows/templates/tags\" | jq ."
            }
        ]
    },
)
async def list_workflow_template_tags() -> list[str]:
    """Return the unique set of tags from all bundled templates."""
    try:
        # Robust search for Samples/Workflows
        here = Path(__file__).resolve()
        tpl_dir = None
        p = here
        for _ in range(0, 9):
            candidate = p.parent / "Samples" / "Workflows"
            if candidate.exists():
                tpl_dir = candidate
                break
            p = p.parent
        if tpl_dir is None or not tpl_dir.exists():
            return []
        tags_set: set[str] = set()
        for f in tpl_dir.glob("*.workflow.json"):
            try:
                import json as _json
                data = _json.loads(f.read_text(encoding="utf-8"))
                tags = data.get("tags") or []
                if isinstance(tags, list):
                    for t in tags:
                        try:
                            s = str(t).strip()
                            if s:
                                tags_set.add(s)
                        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                            continue
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                continue
        return sorted(tags_set)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to list template tags: {e}")
        return []


@router.get("/templates/{name:path}")
async def get_workflow_template(name: str) -> dict[str, Any]:
    """Return JSON content for a named workflow template (sans extension)."""
    # Disallow traversal and separators (defense-in-depth with unquoting)
    from urllib.parse import unquote as _unquote
    raw = _unquote(name or "").strip()
    if ("/" in raw) or ("\\" in raw) or raw == "":
        raise HTTPException(status_code=400, detail="Invalid template name")
    # Reject path traversal patterns (.. segments)
    if ".." in raw.split("/"):
        raise HTTPException(status_code=400, detail="Invalid template name")
    try:
        # Robust search for Samples/Workflows
        here = Path(__file__).resolve()
        tpl_dir = None
        p = here
        for _ in range(0, 9):
            candidate = p.parent / "Samples" / "Workflows"
            if candidate.exists():
                tpl_dir = candidate
                break
            p = p.parent
        if tpl_dir is None:
            raise HTTPException(status_code=404, detail="Template not found")
        # Support names passed either with or without the optional ".workflow" suffix
        candidates = [
            tpl_dir / f"{raw}.workflow.json",
        ]
        if not raw.endswith(".workflow"):
            candidates.append(tpl_dir / f"{raw}.workflow.workflow.json")
        target = next((c for c in candidates if c.exists() and c.is_file()), None)
        if not (tpl_dir.exists() and target):
            raise HTTPException(status_code=404, detail="Template not found")
        import json as _json
        # Size guard (1MB cap)
        raw = target.read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            raise HTTPException(status_code=413, detail="Template too large")
        return _json.loads(raw)
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to read workflow template {name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load template") from e

@router.get("/templates/_byname/{name:path}")
async def get_workflow_template_legacy(name: str) -> dict[str, Any]:
    """Return JSON content for a named workflow template (sans extension)."""
    # Disallow traversal and separators (defense-in-depth with unquoting)
    from urllib.parse import unquote as _unquote
    raw = _unquote(name or "").strip()
    if ("/" in raw) or ("\\" in raw) or raw == "":
        raise HTTPException(status_code=400, detail="Invalid template name")
    if ".." in raw.split("/"):
        raise HTTPException(status_code=400, detail="Invalid template name")
    try:
        # Robust search for Samples/Workflows
        here = Path(__file__).resolve()
        tpl_dir = None
        p = here
        for _ in range(0, 9):
            candidate = p.parent / "Samples" / "Workflows"
            if candidate.exists():
                tpl_dir = candidate
                break
            p = p.parent
        if tpl_dir is None:
            raise HTTPException(status_code=404, detail="Template not found")
        # Support names passed either with or without the optional ".workflow" suffix
        candidates = [
            tpl_dir / f"{raw}.workflow.json",
        ]
        if not raw.endswith(".workflow"):
            candidates.append(tpl_dir / f"{raw}.workflow.workflow.json")
        target = next((c for c in candidates if c.exists() and c.is_file()), None)
        if not (tpl_dir.exists() and target):
            raise HTTPException(status_code=404, detail="Template not found")
        import json as _json
        # Size guard (1MB cap)
        raw = target.read_text(encoding="utf-8")
        if len(raw.encode("utf-8")) > 1024 * 1024:
            raise HTTPException(status_code=413, detail="Template too large")
        return _json.loads(raw)
    except HTTPException:
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Failed to read workflow template {name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load template") from e


@router.get(
    "/config",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_ADMIN)),
    ],
)
async def get_workflows_config(
    _current_user: User = Depends(get_request_user),
):
    """Return effective Workflows configuration derived from environment and backend (read-only)."""
    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name, "")
        if not v:
            return default
        return is_truthy(v)

    backend_type = "sqlite"
    backend = get_content_backend_instance()
    if backend is not None and getattr(backend, "backend_type", None) == BackendType.POSTGRESQL:
        backend_type = "postgres"

    def _csv(name: str) -> list[str]:
        raw = os.getenv(name, "")
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]

    cfg = {
        "backend": {
            "type": backend_type,
        },
        "rate_limits": {
            "ingress_source": "rg_policy.route_map+requests",
            "quotas_disabled": _env_bool("WORKFLOWS_DISABLE_QUOTAS", False),
            "quota_daily_cap_source": "rg_policy.workflows_runs.daily_cap",
        },
        "engine": {
            "tenant_concurrency": int(os.getenv("WORKFLOWS_TENANT_CONCURRENCY", "2") or 2),
            "workflow_concurrency": int(os.getenv("WORKFLOWS_WORKFLOW_CONCURRENCY", "1") or 1),
        },
        "egress": {
            "profile": (os.getenv("WORKFLOWS_EGRESS_PROFILE", "") or "(auto)").strip(),
            "allowed_ports": _csv("WORKFLOWS_EGRESS_ALLOWED_PORTS") or ["80","443"],
            "allowlist": _csv("WORKFLOWS_EGRESS_ALLOWLIST"),
            "block_private": _env_bool("WORKFLOWS_EGRESS_BLOCK_PRIVATE", True),
        },
        "webhooks": {
            "completion_disabled": _env_bool("WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS", False),
            "secret_set": bool(os.getenv("WORKFLOWS_WEBHOOK_SECRET")),
            "dlq_enabled": _env_bool("WORKFLOWS_WEBHOOK_DLQ_ENABLED", False),
            "allowlist": _csv("WORKFLOWS_WEBHOOK_ALLOWLIST"),
            "denylist": _csv("WORKFLOWS_WEBHOOK_DENYLIST"),
        },
        "artifacts": {
            "validate_strict": _env_bool("WORKFLOWS_ARTIFACT_VALIDATE_STRICT", True),
            "encryption_enabled": _env_bool("WORKFLOWS_ARTIFACT_ENCRYPTION", False),
            "gc_enabled": _env_bool("WORKFLOWS_ARTIFACT_GC_ENABLED", False),
            "retention_days": int(os.getenv("WORKFLOWS_ARTIFACT_RETENTION_DAYS", "30") or 30),
        },
    }
    return cfg


@router.post(
    "/runs/{run_id}/retry",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
    ],
)
async def retry_run(
    run_id: str,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    if str(run.user_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Run not found")
    engine = WorkflowEngine(db)
    # Resume from last failed step if present
    failed_step = db.get_last_failed_step_id(run_id)
    if failed_step:
        asyncio.create_task(engine.continue_run(run_id, after_step_id=failed_step, last_outputs=None))
    else:
        engine.submit(run_id, RunMode.ASYNC)
    return {"ok": True}


@router.get(
    "/{workflow_id}",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_READ)),
    ],
)
async def get_definition(
    workflow_id: int,
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    """Fetch a workflow definition by id.

    Declared after '/runs*' routes to prevent path parameter shadowing.
    """
    d = db.get_definition(workflow_id)
    if not d:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Tenant isolation
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if d.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    # Owner/admin check for definition read (tighten RBAC)
    is_admin = _is_workflows_admin_user(current_user)
    if str(d.owner_id) != str(current_user.id) and not is_admin:
        raise HTTPException(status_code=404, detail="Workflow not found")
    try:
        definition = json.loads(d.definition_json)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        definition = {"error": "invalid_definition_json"}
    return {
        "id": d.id,
        "name": d.name,
        "version": d.version,
        "is_active": bool(d.is_active),
        "definition": definition,
    }


# -------------------- Human-in-the-loop approvals --------------------

class HumanReviewPayload(BaseModel):
    comment: Optional[str] = None
    edited_fields: Optional[dict[str, Any]] = None


@router.post(
    "/runs/{run_id}/steps/{step_id}/approve",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
    ],
)
async def approve_step(
    run_id: str,
    step_id: str,
    payload: HumanReviewPayload = Body(default_factory=HumanReviewPayload),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    step_run = db.get_latest_step_run(run_id=run_id, step_id=step_id)
    if not step_run:
        raise HTTPException(status_code=404, detail="Step run not found")
    assigned_to = step_run.get("assigned_to")
    user_id = str(current_user.id)
    if not is_admin and (not assigned_to or str(assigned_to) != user_id):
        raise HTTPException(status_code=404, detail="Run not found")
    # Update step decision via DB adapter
    try:
        db.approve_step_decision(run_id=run_id, step_id=step_id, approved_by=str(current_user.id), comment=payload.comment or "")
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
        error_category = _classify_db_error(exc)
        logger.exception(
            "Workflows approval: failed to persist decision for run_id={} step_id={} error_category={}",
            run_id,
            step_id,
            error_category,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "approval_failed",
                "message": "Failed to approve step",
                "error_category": error_category,
                "error_type": type(exc).__name__,
            },
        ) from exc

    # Resolve on_success target if defined
    next_step_id = None
    try:
        definition = json.loads(run.definition_snapshot_json or "{}")
        step_def = _find_step_def(definition, step_id)
        next_step_id = str((step_def or {}).get("on_success") or "").strip() or None
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        next_step_id = None
    # Resume run from next step
    engine = WorkflowEngine(db)
    # Pass edited fields as last outputs override
    last_outputs = payload.edited_fields or {}
    last_outputs.setdefault("decision", "approved")
    if payload.comment:
        last_outputs.setdefault("comment", payload.comment)
    asyncio.create_task(engine.continue_run(run_id, after_step_id=step_id, last_outputs=last_outputs, next_step_id=next_step_id))
    return {"ok": True}


@router.post(
    "/runs/{run_id}/steps/{step_id}/reject",
    dependencies=[
        Depends(RequirePermission(WORKFLOWS_RUNS_CONTROL)),
    ],
)
async def reject_step(
    run_id: str,
    step_id: str,
    payload: HumanReviewPayload = Body(default_factory=HumanReviewPayload),
    current_user: User = Depends(get_request_user),
    db: WorkflowsDatabase = Depends(_get_db),
):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    if run.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Run not found")
    is_admin = _is_workflows_admin_user(current_user)
    step_run = db.get_latest_step_run(run_id=run_id, step_id=step_id)
    if not step_run:
        raise HTTPException(status_code=404, detail="Step run not found")
    assigned_to = step_run.get("assigned_to")
    user_id = str(current_user.id)
    if not is_admin and (not assigned_to or str(assigned_to) != user_id):
        raise HTTPException(status_code=404, detail="Run not found")
    failure_next = None
    try:
        definition = json.loads(run.definition_snapshot_json or "{}")
        step_def = _find_step_def(definition, step_id)
        failure_next = str((step_def or {}).get("on_failure") or "").strip() or None
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        failure_next = None
    if db.backend:
        try:
            with db.backend.transaction() as conn:  # type: ignore[union-attr]
                try:
                    db.reject_step_decision(
                        run_id=run_id,
                        step_id=step_id,
                        approved_by=str(current_user.id),
                        comment=payload.comment or "",
                        connection=conn,
                    )
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                    error_category = _classify_db_error(exc)
                    logger.exception(
                        "Workflows rejection: failed to record decision for run_id={} step_id={} error_category={}",
                        run_id,
                        step_id,
                        error_category,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "rejection_failed",
                            "message": "Failed to record step rejection",
                            "error_category": error_category,
                            "error_type": type(exc).__name__,
                        },
                    ) from exc
                if not failure_next:
                    try:
                        db.update_run_status(
                            run_id,
                            status="failed",
                            status_reason="rejected_by_human",
                            ended_at=_utcnow_iso(),
                            connection=conn,
                        )
                    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                        error_category = _classify_db_error(exc)
                        logger.exception(
                            "Workflows rejection: failed to mark run rejected for run_id={} step_id={} error_category={}",
                            run_id,
                            step_id,
                            error_category,
                        )
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "error": "rejection_failed",
                                "message": "Failed to mark run as rejected",
                                "error_category": error_category,
                                "error_type": type(exc).__name__,
                            },
                        ) from exc
                try:
                    db.append_event(
                        str(getattr(current_user, "tenant_id", "default")),
                        run_id,
                        "human_rejected",
                        {"step_id": step_id},
                        connection=conn,
                    )
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                    error_category = _classify_db_error(exc)
                    logger.exception(
                        "Workflows rejection: failed to append event for run_id={} step_id={} error_category={}",
                        run_id,
                        step_id,
                        error_category,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": "rejection_failed",
                            "message": "Failed to append workflow event",
                            "error_category": error_category,
                            "error_type": type(exc).__name__,
                        },
                    ) from exc
        except HTTPException:
            raise
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
            error_category = _classify_db_error(exc)
            logger.exception(
                "Workflows rejection: failed to start transaction for run_id={} step_id={} error_category={}",
                run_id,
                step_id,
                error_category,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rejection_failed",
                    "message": "Failed to record step rejection",
                    "error_category": error_category,
                    "error_type": type(exc).__name__,
                },
            ) from exc
    else:
        try:
            db.reject_step_decision(
                run_id=run_id,
                step_id=step_id,
                approved_by=str(current_user.id),
                comment=payload.comment or "",
            )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
            error_category = _classify_db_error(exc)
            logger.exception(
                "Workflows rejection: failed to record decision for run_id={} step_id={} error_category={}",
                run_id,
                step_id,
                error_category,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rejection_failed",
                    "message": "Failed to record step rejection",
                    "error_category": error_category,
                    "error_type": type(exc).__name__,
                },
            ) from exc
        if not failure_next:
            try:
                db.update_run_status(
                    run_id,
                    status="failed",
                    status_reason="rejected_by_human",
                    ended_at=_utcnow_iso(),
                )
            except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
                error_category = _classify_db_error(exc)
                logger.exception(
                    "Workflows rejection: failed to mark run rejected for run_id={} step_id={} error_category={}",
                    run_id,
                    step_id,
                    error_category,
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "rejection_failed",
                        "message": "Failed to mark run as rejected",
                        "error_category": error_category,
                        "error_type": type(exc).__name__,
                    },
                ) from exc
        try:
            db.append_event(
                str(getattr(current_user, "tenant_id", "default")),
                run_id,
                "human_rejected",
                {"step_id": step_id},
            )
        except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as exc:
            error_category = _classify_db_error(exc)
            logger.exception(
                "Workflows rejection: failed to append event for run_id={} step_id={} error_category={}",
                run_id,
                step_id,
                error_category,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rejection_failed",
                    "message": "Failed to append workflow event",
                    "error_category": error_category,
                    "error_type": type(exc).__name__,
                },
            ) from exc
    if failure_next:
        engine = WorkflowEngine(db)
        last_outputs = payload.edited_fields or {}
        last_outputs.setdefault("decision", "rejected")
        if payload.comment:
            last_outputs.setdefault("comment", payload.comment)
        asyncio.create_task(
            engine.continue_run(
                run_id,
                after_step_id=step_id,
                last_outputs=last_outputs,
                next_step_id=failure_next,
            )
        )
    return {"ok": True}


# -------------------- WebSocket for events (simple polling bridge) --------------------

@router.websocket("/ws")
async def workflows_ws(
    websocket: WebSocket,
    run_id: str,
    token: Optional[str] = Query(None),
    types: Optional[list[str]] = Query(None),
    db: WorkflowsDatabase = Depends(_get_db),
):
    # Extract token: prefer query param; fallback to Authorization header
    if not token:
        auth_hdr = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
        if auth_hdr and auth_hdr.lower().startswith("bearer "):
            token = auth_hdr.split(" ", 1)[1].strip()
    if not token:
        # Force test client to error on connect
        raise RuntimeError("Authentication required")

    # Verify token
    try:
        jwtm = get_jwt_manager()
        token_data = jwtm.verify_token(token)
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
        raise RuntimeError("Invalid token") from None

    # Run-level authorization: owner or admin
    run = db.get_run(run_id)
    if not run:
        raise RuntimeError("Run not found")
    # Enforce run-level ownership: subject must match creator
    if str(token_data.sub) != str(run.user_id):
        raise RuntimeError("Forbidden")

    await websocket.accept()
    # Wrap for metrics and activity tracking; keep domain frames unchanged
    stream = WebSocketStream(
        websocket,
        heartbeat_interval_s=0.0,
        idle_timeout_s=None,
        close_on_done=False,
        labels={"component": "workflows", "endpoint": "workflows_ws"},
    )
    await stream.start()
    # Normalize event types if provided for server-side filtering
    types_norm = [t.strip() for t in (types or []) if str(t).strip()]
    last_seq: Optional[int] = None
    try:
        # On connect, send a snapshot event
        await stream.send_json(
            {
                "type": "snapshot",
                "run": {
                    "run_id": run.run_id,
                    "status": run.status,
                },
            }
        )
        while True:
            events = db.get_events(run_id, since=last_seq, types=(types_norm or None))
            if events:
                for e in events:
                    payload = e.get("payload_json") or {}
                    await stream.send_json({"event_seq": e["event_seq"], "event_type": e["event_type"], "payload": payload, "ts": e["created_at"]})
                    last_seq = e["event_seq"]
            else:
                # Send a lightweight heartbeat so clients using blocking receive_json() don't hang indefinitely
                try:
                    await stream.send_json({"type": "heartbeat", "ts": _utcnow_iso()})
                except _WORKFLOWS_NONCRITICAL_EXCEPTIONS:
                    # If sending heartbeat fails (e.g., client disconnect), let outer exception handling close
                    raise
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("Workflows WS disconnected")
        raise
    except _WORKFLOWS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Workflows WS error: {e}")
        with contextlib.suppress(_WORKFLOWS_NONCRITICAL_EXCEPTIONS):
            await stream.ws.close(code=status.WS_1011_INTERNAL_ERROR)
