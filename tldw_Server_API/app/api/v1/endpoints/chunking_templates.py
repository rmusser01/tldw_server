# chunking_templates.py
"""
API endpoints for managing chunking templates.
"""

import contextlib
import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from loguru import logger
from pydantic import BaseModel

# Dependencies for user-specific database access
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chunking_templates_schemas import (
    ApplyTemplateRequest,
    ApplyTemplateResponse,
    ChunkingTemplateCreate,
    ChunkingTemplateListResponse,
    ChunkingTemplateResponse,
    ChunkingTemplateUpdate,
    TemplateConfig,
    TemplateValidationError,
    TemplateValidationResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.Chunking.regex_safety import check_pattern as _rx_check
from tldw_Server_API.app.core.Chunking.regex_safety import compile_flags as _rx_flags
from tldw_Server_API.app.core.Chunking.regex_safety import warn_ambiguity as _rx_warn
from tldw_Server_API.app.core.Chunking.templates import (
    ChunkingTemplate,
    TemplateClassifier,
    TemplateLearner,
    TemplateProcessor,
    TemplateStage,
)
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

router = APIRouter(prefix="/chunking/templates", tags=["chunking-templates"])

# In-memory fallback store for environments where DB methods are unavailable
# Structure: { user_id: { template_name: record_dict } }
_FALLBACK_TEMPLATES: dict[str, dict[str, dict[str, Any]]] = {}

_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    json.JSONDecodeError,
)

def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
        return ""

def _fb_bucket(user_id: Optional[str]) -> dict[str, dict[str, Any]]:
    uid = str(user_id or "default")
    if uid not in _FALLBACK_TEMPLATES:
        _FALLBACK_TEMPLATES[uid] = {}
    return _FALLBACK_TEMPLATES[uid]

def _supports(obj: Any, method: str) -> bool:
    return hasattr(obj, method) and callable(getattr(obj, method, None))

def _db_class_str(db: Any) -> str:
    try:
        return f"{db.__class__.__module__}.{db.__class__.__name__}"
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
        return str(type(db))

def _emit_db_capability_headers(response: Response, db: Any, required: list[str]) -> None:
    if not isinstance(response, Response):
        return
    try:
        response.headers["X-Template-DB-Class"] = _db_class_str(db)
        missing = [m for m in required if not _supports(db, m)]
        if missing:
            response.headers["X-Template-DB-Capability"] = "fallback"
            response.headers["X-Template-DB-Missing"] = ",".join(missing)
            response.headers["X-Template-DB-Hint"] = (
                "Use a native media DB session that implements chunking-template methods."
            )
            logger.warning(
                f"Chunking Templates DB missing methods {missing} on {response.headers.get('X-Template-DB-Class')} - "
                "using in-memory fallback; hint: use a native media DB session"
            )
        else:
            response.headers["X-Template-DB-Capability"] = "native"
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
        pass


@router.get("/diagnostics")
async def diagnostics(
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
):
    """Diagnostics for chunking templates backend capabilities."""
    required = [
        "list_chunking_templates",
        "get_chunking_template",
        "create_chunking_template",
        "update_chunking_template",
        "delete_chunking_template",
    ]
    if response is not None:
        _emit_db_capability_headers(response, db, required)
        _set_db_capability_gauge(response)
    missing = [m for m in required if not _supports(db, m)]
    return {
        "db_class": _db_class_str(db),
        "capability": "native" if not missing else "fallback",
        "missing_methods": missing,
        "fallback_enabled": _fallback_allowed(),
        "hint": "Use a native media DB session that implements chunking-template methods.",
    }

# Observability helpers (no-op fallbacks)
try:
    from tldw_Server_API.app.core.Metrics import increment_counter, set_gauge
except ImportError:  # pragma: no cover - safety
    def increment_counter(*args, **kwargs):
        return None
    def set_gauge(*args, **kwargs):
        return None

def _fallback_allowed() -> bool:
    val = str(os.getenv("CHUNKING_TEMPLATES_FALLBACK_ENABLED", "1")).lower()
    return val in ("1", "true", "yes")

def _ensure_fallback_policy(db: Any, required: list[str]) -> None:
    """If DB lacks required methods and fallback is disabled, raise a 500 with a hint."""
    missing = [m for m in required if not _supports(db, m)]
    if missing and not _fallback_allowed():
        detail = {
            "success": False,
            "error": "Templates fallback store is disabled",
            "error_code": "FALLBACK_DISABLED",
            "details": [{"field": ",".join(missing), "message": "DB capability missing. Use a native media DB session.", "code": "DB_CAPABILITY"}],
            "hint": "Enable CHUNKING_TEMPLATES_FALLBACK_ENABLED=1 for dev/test or use a native media DB session in production.",
        }
        raise HTTPException(status_code=500, detail=detail)

def _set_db_capability_gauge(response: Response) -> None:
    try:
        cap = (response.headers.get("X-Template-DB-Capability") or "native").lower()
        set_gauge("chunking_templates_db_capability", 1.0 if cap == "native" else 0.0, labels={"capability": cap})
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
        pass


@router.get("", response_model=ChunkingTemplateListResponse)
async def list_templates(
    include_builtin: bool = Query(True, description="Include built-in templates"),
    include_custom: bool = Query(True, description="Include custom templates"),
    tags: Optional[list[str]] = Query(None, description="Filter by tags"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
) -> ChunkingTemplateListResponse:
    """
    List all available chunking templates with optional filtering.

    Returns:
        List of chunking templates matching the filter criteria
    """
    try:
        if response is not None:
            _emit_db_capability_headers(response, db, ["list_chunking_templates"])
            _set_db_capability_gauge(response)
        _ensure_fallback_policy(db, ["list_chunking_templates"])  # Enforce prod safeguard
        if _supports(db, 'list_chunking_templates'):
            templates = db.list_chunking_templates(
                include_builtin=include_builtin,
                include_custom=include_custom,
                tags=tags,
                user_id=user_id,
                include_deleted=False
            )
        else:
            increment_counter("chunking_templates_fallback_list_total", labels={"mode": "fallback"})
            # Fallback: aggregate from in-memory store
            templates = []
            buckets = [_fb_bucket(user_id)] if user_id is not None else list(_FALLBACK_TEMPLATES.values())
            for bucket in buckets:
                for _, rec in bucket.items():
                    if tags and not any(t in (rec.get('tags') or []) for t in tags):
                        continue
                    templates.append(rec)

        # Convert to response format
        template_responses = []
        for template in templates:
            template_responses.append(ChunkingTemplateResponse(
                id=template['id'],
                uuid=template['uuid'],
                name=template['name'],
                description=template['description'],
                template_json=template['template_json'],
                is_builtin=template['is_builtin'],
                tags=template['tags'],
                created_at=template['created_at'],
                updated_at=template['updated_at'],
                version=template['version'],
                user_id=template['user_id']
            ))

        resp = ChunkingTemplateListResponse(
            templates=template_responses,
            total=len(template_responses)
        )
        return resp

    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error listing templates")
        raise HTTPException(status_code=500, detail="Failed to list chunking templates") from e


@router.get("/{template_name}", response_model=ChunkingTemplateResponse)
async def get_template(
    template_name: str,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
) -> ChunkingTemplateResponse:
    """
    Get a specific chunking template by name.

    Args:
        template_name: Name of the template to retrieve

    Returns:
        The requested chunking template

    Raises:
        404: Template not found
    """
    try:
        template = None
        if _supports(db, 'get_chunking_template'):
            template = db.get_chunking_template(name=template_name)
        if not template:
            # Fallback search across in-memory buckets
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    template = bucket[template_name]
                    break

        if not template:
            raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")

        return ChunkingTemplateResponse(
            id=template['id'],
            uuid=template['uuid'],
            name=template['name'],
            description=template['description'],
            template_json=template['template_json'],
            is_builtin=template['is_builtin'],
            tags=template['tags'],
            created_at=template['created_at'],
            updated_at=template['updated_at'],
            version=template['version'],
            user_id=template['user_id']
        )

    except HTTPException:
        raise
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting template")
        raise HTTPException(status_code=500, detail="Failed to get chunking template") from e


@router.post("", response_model=ChunkingTemplateResponse, status_code=201)
async def create_template(
    template_data: ChunkingTemplateCreate,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
) -> ChunkingTemplateResponse:
    """
    Create a new chunking template.

    Args:
        template_data: Template configuration and metadata

    Returns:
        The created chunking template

    Raises:
        400: Invalid template configuration
        409: Template with same name already exists
    """
    try:
        # Convert template config to JSON string
        tmpl_dict = model_dump_compat(template_data.template)
        template_json = json.dumps(tmpl_dict)

        # Emit DB capability headers for diagnostics
        if response is not None:
            _emit_db_capability_headers(response, db, ["create_chunking_template", "get_chunking_template"])
            _set_db_capability_gauge(response)
        _ensure_fallback_policy(db, ["create_chunking_template", "get_chunking_template"])  # Enforce prod safeguard
        # Create template in database (or fallback)
        if _supports(db, 'create_chunking_template'):
            created = db.create_chunking_template(
                name=template_data.name,
                template_json=template_json,
                description=template_data.description,
                is_builtin=False,
                tags=template_data.tags,
                user_id=template_data.user_id
            )
            stored = db.get_chunking_template(name=created['name'])
        else:
            increment_counter("chunking_templates_create_total", labels={"mode": "fallback"})
            # In-memory fallback
            from uuid import uuid4
            uid = str(template_data.user_id or getattr(current_user, 'id', ''))
            bucket = _fb_bucket(uid)
            if template_data.name in bucket:
                raise HTTPException(status_code=409, detail={
                    "success": False,
                    "error": f"Template with name '{template_data.name}' already exists",
                    "error_code": "CONFLICT",
                })
            stored = {
                'id': len(bucket) + 1,
                'uuid': str(uuid4()),
                'name': template_data.name,
                'description': template_data.description,
                'template_json': template_json,
                'is_builtin': False,
                'tags': template_data.tags or [],
                'created_at': _now_iso(),
                'updated_at': _now_iso(),
                'version': 1,
                'user_id': uid
            }
            bucket[template_data.name] = stored
        increment_counter("chunking_templates_create_total", labels={"mode": "native" if _supports(db, 'create_chunking_template') else 'fallback'})
        return ChunkingTemplateResponse(
            id=stored['id'],
            uuid=stored['uuid'],
            name=stored['name'],
            description=stored['description'],
            template_json=stored['template_json'],
            is_builtin=stored['is_builtin'],
            tags=stored['tags'],
            created_at=stored['created_at'],
            updated_at=stored['updated_at'],
            version=stored['version'],
            user_id=stored['user_id']
        )

    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        msg = str(e)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail={"success": False, "error": msg, "error_code": "CONFLICT"}) from e
        elif "Invalid template JSON" in msg:
            raise HTTPException(status_code=400, detail={"success": False, "error": msg, "error_code": "BAD_REQUEST"}) from e
        else:
            logger.error("Error creating template")
            raise HTTPException(
                status_code=500,
                detail={"success": False, "error": "Failed to create template", "error_code": "SERVER_ERROR"},
            ) from e


@router.put("/{template_name}", response_model=ChunkingTemplateResponse)
async def update_template(
    template_name: str,
    template_update: ChunkingTemplateUpdate,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
) -> ChunkingTemplateResponse:
    """
    Update an existing chunking template.

    Args:
        template_name: Name of the template to update
        template_update: Fields to update

    Returns:
        The updated chunking template

    Raises:
        400: Cannot modify built-in templates
        404: Template not found
    """
    try:
        # Emit DB capability headers for diagnostics
        if response is not None:
            _emit_db_capability_headers(response, db, [
                "get_chunking_template",
                "list_chunking_templates",
                "update_chunking_template",
            ])
            _set_db_capability_gauge(response)
        _ensure_fallback_policy(db, ["get_chunking_template", "list_chunking_templates", "update_chunking_template"])  # Enforce prod safeguard
        # Heuristic: protect conventional built-in names eagerly
        if template_name.lower().startswith('builtin_'):
            raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot modify built-in templates", "error_code": "BUILTIN"})
        # Fast path: use list() to detect built-ins deterministically when available
        if _supports(db, 'list_chunking_templates'):
            try:
                matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                if matches:
                    ex0 = matches[0]
                    if ex0.get('is_builtin'):
                        raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot modify built-in templates", "error_code": "BUILTIN"})
            except HTTPException:
                raise
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                pass
        # Find existing first to handle built-ins deterministically
        existing = None
        if _supports(db, 'get_chunking_template'):
            try:
                existing = db.get_chunking_template(name=template_name)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
        if not existing and _supports(db, 'list_chunking_templates'):
            try:
                matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                existing = matches[0] if matches else None
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
        if not existing:
            # Fallback store (best-effort). If still missing, continue; disambiguate later.
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    existing = bucket[template_name]
                    break
        if existing and existing.get('is_builtin'):
            raise HTTPException(status_code=400, detail="Cannot modify built-in templates")

        # Prepare update data
        template_json = None
        if template_update.template:
            try:
                tmpl_dict = model_dump_compat(template_update.template)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as exc:
                logger.exception("Failed to serialise chunking template update payload")
                raise HTTPException(
                    status_code=400,
                    detail={"success": False, "error": "Invalid template payload", "error_code": "BAD_TEMPLATE"},
                ) from exc
            template_json = json.dumps(tmpl_dict)

        # Update template
        if _supports(db, 'update_chunking_template'):
            try:
                success = db.update_chunking_template(
                    name=template_name,
                    template_json=template_json,
                    description=template_update.description,
                    tags=template_update.tags
                )
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                # Conservatively treat DB-layer exceptions as a protected/bad update
                raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot modify built-in templates or invalid update", "error_code": "BAD_REQUEST"}) from None
        else:
            # In-memory update
            success = False
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    rec = bucket[template_name]
                    if rec.get('is_builtin'):
                        raise HTTPException(status_code=400, detail="Cannot modify built-in templates")
                    if template_json is not None:
                        rec['template_json'] = template_json
                    if template_update.description is not None:
                        rec['description'] = template_update.description
                    if template_update.tags is not None:
                        rec['tags'] = template_update.tags
                    rec['updated_at'] = _now_iso()
                    success = True
                    break

        if not success:
            # Attempt to disambiguate failure: not found vs built-in
            existing = None
            try:
                if hasattr(db, 'get_chunking_template'):
                    existing = db.get_chunking_template(name=template_name)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
            if existing is None and hasattr(db, 'list_chunking_templates'):
                try:
                    matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                    existing = matches[0] if matches else None
                except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                    existing = None
            if not existing:
                raise HTTPException(status_code=404, detail={"success": False, "error": f"Template '{template_name}' not found", "error_code": "NOT_FOUND"})
            if existing.get('is_builtin'):
                raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot modify built-in templates", "error_code": "BUILTIN"})
            raise HTTPException(status_code=500, detail={"success": False, "error": "Failed to update template", "error_code": "SERVER_ERROR"})

        # Get updated template
        updated = None
        try:
            if hasattr(db, 'get_chunking_template'):
                updated = db.get_chunking_template(name=template_name)
        except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
            updated = None
        if updated is None and hasattr(db, 'list_chunking_templates'):
            try:
                matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                updated = matches[0] if matches else None
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                updated = None
        if updated is None:
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    updated = bucket[template_name]
                    break

        with contextlib.suppress(_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS):
            increment_counter("chunking_templates_update_total", labels={"mode": "native" if _supports(db, 'update_chunking_template') else 'fallback', "success": str(bool(updated)).lower()})
        return ChunkingTemplateResponse(
            id=updated['id'],
            uuid=updated['uuid'],
            name=updated['name'],
            description=updated['description'],
            template_json=updated['template_json'],
            is_builtin=updated['is_builtin'],
            tags=updated['tags'],
            created_at=updated['created_at'],
            updated_at=updated['updated_at'],
            version=updated['version'],
            user_id=updated['user_id']
        )

    except HTTPException:
        raise
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error updating template")
        raise HTTPException(status_code=500, detail={"success": False, "error": "Failed to update template", "error_code": "SERVER_ERROR"}) from e


@router.delete("/{template_name}", status_code=204)
async def delete_template(
    template_name: str,
    hard_delete: bool = Query(False, description="Permanently delete template"),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
) -> None:
    """
    Delete a chunking template.

    Args:
        template_name: Name of the template to delete
        hard_delete: If true, permanently delete; otherwise soft delete

    Raises:
        400: Cannot delete built-in templates
        404: Template not found
    """
    try:
        # Emit DB capability headers for diagnostics
        if response is not None:
            _emit_db_capability_headers(response, db, [
                "get_chunking_template",
                "list_chunking_templates",
                "delete_chunking_template",
            ])
            _set_db_capability_gauge(response)
        _ensure_fallback_policy(db, ["get_chunking_template", "list_chunking_templates", "delete_chunking_template"])  # Enforce prod safeguard
        # Heuristic: protect conventional built-in names eagerly
        if template_name.lower().startswith('builtin_'):
            raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot delete built-in templates", "error_code": "BUILTIN"})
        # Check existing and built-in first
        existing = None
        if _supports(db, 'get_chunking_template'):
            try:
                existing = db.get_chunking_template(name=template_name)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
        if not existing and _supports(db, 'list_chunking_templates'):
            try:
                matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                existing = matches[0] if matches else None
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
        if not existing:
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    existing = bucket[template_name]
                    break
        if not existing:
            raise HTTPException(status_code=404, detail={"success": False, "error": f"Template '{template_name}' not found", "error_code": "NOT_FOUND"})
        if existing.get('is_builtin'):
            raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot delete built-in templates", "error_code": "BUILTIN"})

        # Delete template
        if _supports(db, 'delete_chunking_template'):
            try:
                success = db.delete_chunking_template(
                    name=template_name,
                    hard_delete=hard_delete
                )
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot delete built-in templates or invalid delete", "error_code": "BAD_REQUEST"}) from None
        else:
            success = False
            for bucket in _FALLBACK_TEMPLATES.values():
                if template_name in bucket:
                    del bucket[template_name]
                    success = True
                    break

        if not success:
            # Attempt to disambiguate failure: not found vs built-in
            existing = None
            try:
                if hasattr(db, 'get_chunking_template'):
                    existing = db.get_chunking_template(name=template_name)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                existing = None
            if existing is None and hasattr(db, 'list_chunking_templates'):
                try:
                    matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == template_name]
                    existing = matches[0] if matches else None
                except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                    existing = None
            if not existing:
                # Probe fallback store
                for bucket in _FALLBACK_TEMPLATES.values():
                    if template_name in bucket:
                        existing = bucket[template_name]
                        break
            if not existing:
                raise HTTPException(status_code=404, detail={"success": False, "error": f"Template '{template_name}' not found", "error_code": "NOT_FOUND"})
            if existing.get('is_builtin'):
                raise HTTPException(status_code=400, detail={"success": False, "error": "Cannot delete built-in templates", "error_code": "BUILTIN"})
            raise HTTPException(status_code=500, detail={"success": False, "error": "Failed to delete template", "error_code": "SERVER_ERROR"})
        with contextlib.suppress(_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS):
            increment_counter("chunking_templates_delete_total", labels={"mode": "native" if _supports(db, 'delete_chunking_template') else 'fallback', "hard": str(bool(hard_delete)).lower(), "success": str(bool(success)).lower()})

    except HTTPException:
        raise
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error deleting template")
        raise HTTPException(status_code=500, detail={"success": False, "error": "Failed to delete template", "error_code": "SERVER_ERROR"}) from e


@router.post("/apply", response_model=ApplyTemplateResponse)
async def apply_template(
    request: ApplyTemplateRequest,
    include_metadata: bool = Query(False, description="Return chunk metadata; if false, return only text list"),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user),
    response: Response = None,
) -> ApplyTemplateResponse:
    """
    Apply a chunking template to text.

    Args:
        request: Template name and text to chunk

    Returns:
        The chunked text results

    Raises:
        404: Template not found
        400: Template application error
    """
    try:
        # Emit DB capability headers for diagnostics
        if response is not None:
            _emit_db_capability_headers(response, db, ["get_chunking_template", "list_chunking_templates"])
            _set_db_capability_gauge(response)
        _ensure_fallback_policy(db, ["get_chunking_template", "list_chunking_templates"])  # Enforce prod safeguard
        # Get template from database or fallback
        template_data = None
        if _supports(db, 'get_chunking_template'):
            try:
                template_data = db.get_chunking_template(name=request.template_name)
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                template_data = None
        if template_data is None and _supports(db, 'list_chunking_templates'):
            try:
                matches = [t for t in db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False) if t.get('name') == request.template_name]
                template_data = matches[0] if matches else None
            except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                template_data = None
        if template_data is None:
            for bucket in _FALLBACK_TEMPLATES.values():
                if request.template_name in bucket:
                    template_data = bucket[request.template_name]
                    break
        if not template_data:
            raise HTTPException(status_code=404, detail={"success": False, "error": f"Template '{request.template_name}' not found", "error_code": "NOT_FOUND"})

        # Parse template JSON
        template_config = json.loads(template_data['template_json'])

        # Create ChunkingTemplate object
        stages = []

        # Add preprocessing stage if exists
        if 'preprocessing' in template_config:
            stages.append(TemplateStage(
                name='preprocess',
                operations=template_config['preprocessing'],
                enabled=True
            ))

        # Add chunking stage
        stages.append(TemplateStage(
            name='chunk',
            operations=[template_config['chunking']],
            enabled=True
        ))

        # Add postprocessing stage if exists
        if 'postprocessing' in template_config:
            stages.append(TemplateStage(
                name='postprocess',
                operations=template_config['postprocessing'],
                enabled=True
            ))

        template = ChunkingTemplate(
            name=template_data['name'],
            description=template_data['description'] or "",
            base_method=template_config['chunking']['method'],
            stages=stages,
            default_options=template_config['chunking'].get('config', {}),
            metadata={'tags': template_data['tags']}
        )

        # Apply template using TemplateProcessor
        processor = TemplateProcessor()

        # Override options if provided
        options = {}
        if request.override_options:
            options.update(request.override_options)

        chunks = processor.process_template(
            text=request.text,
            template=template,
            **options
        )
        # Format according to include_metadata
        if include_metadata:
            out_chunks = chunks  # already List[Dict]
        else:
            out_chunks = [c.get('text', '') if isinstance(c, dict) else str(c) for c in chunks]

        increment_counter("chunking_templates_apply_total", labels={"template": template_data['name']})
        return ApplyTemplateResponse(
            template_name=request.template_name,
            chunks=out_chunks,  # type: ignore[arg-type]
            metadata={
                'chunk_count': len(chunks),
                'template_version': template_data['version']
            }
        )

    except HTTPException:
        raise
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error applying template")
        raise HTTPException(status_code=400, detail={"success": False, "error": f"Template application error: {str(e)}", "error_code": "BAD_REQUEST"}) from e


@router.post("/validate", response_model=TemplateValidationResponse)
async def validate_template(
    template_config: dict[str, Any] = Body(..., description="Template configuration to validate")
) -> TemplateValidationResponse:
    """
    Validate a template configuration without saving it.

    Args:
        template_config: Template configuration to validate

    Returns:
        Validation results with any errors or warnings
    """
    errors = []
    warnings = []

    try:
        # First pass: try structured validation using TemplateConfig.
        # Do not expose 422; convert validation errors into a 200 response payload.
        from pydantic import ValidationError
        try:
            parsed = TemplateConfig.model_validate(template_config)  # type: ignore[arg-type]
            # Use normalized dict for the remaining checks
            template_config = model_dump_compat(parsed)  # type: ignore[assignment]
        except ValidationError as ve:
            for err in ve.errors():
                loc = err.get('loc') or []
                field = '.'.join(str(x) for x in loc) if isinstance(loc, (list, tuple)) else str(loc)
                msg = err.get('msg') or 'Invalid value'
                errors.append(TemplateValidationError(field=field or 'template_config', message=str(msg)))
            return TemplateValidationResponse(valid=False, errors=errors, warnings=warnings or None)

        # Check required fields
        if 'chunking' not in template_config:
            errors.append(TemplateValidationError(
                field='chunking',
                message='Chunking configuration is required'
            ))
        else:
            chunking = template_config['chunking']
            if 'method' not in chunking:
                errors.append(TemplateValidationError(
                    field='chunking.method',
                    message='Chunking method is required'
                ))
            else:
                # Validate chunking method against actual available methods
                try:
                    available_methods = Chunker().get_available_methods()
                except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS:
                    available_methods = ['words', 'sentences', 'paragraphs', 'tokens', 'semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize', 'structure_aware', 'propositions']
                if chunking['method'] not in available_methods:
                    errors.append(TemplateValidationError(
                        field='chunking.method',
                        message=f"Unknown chunking method '{chunking['method']}'. Valid methods: {', '.join(sorted(available_methods))}"
                    ))

        # Validate hierarchical options (either top-level or inside chunking.config)
        def _get_cfg_path(cfg: dict[str, Any], path: list[str]) -> Optional[Any]:
            cur = cfg
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    return None
                cur = cur[key]
            return cur

        hier_flag = _get_cfg_path(template_config, ['chunking', 'config', 'hierarchical'])
        hier_tpl = _get_cfg_path(template_config, ['chunking', 'config', 'hierarchical_template'])
        if hier_flag is not None and not isinstance(hier_flag, bool):
            errors.append(TemplateValidationError(
                field='chunking.config.hierarchical',
                message='hierarchical must be a boolean'
            ))
        # Validate boundaries with limits
        if isinstance(hier_tpl, dict) and 'boundaries' in hier_tpl:
            boundaries = hier_tpl.get('boundaries')
            if not isinstance(boundaries, list):
                errors.append(TemplateValidationError(
                    field='chunking.config.hierarchical_template.boundaries',
                    message='boundaries must be a list'
                ))
            else:
                if len(boundaries) > 20:
                    errors.append(TemplateValidationError(
                        field='chunking.config.hierarchical_template.boundaries',
                        message='Too many boundary rules (max 20)'
                    ))
                for i, rule in enumerate(boundaries[:20]):
                    if not isinstance(rule, dict) or 'pattern' not in rule:
                        errors.append(TemplateValidationError(
                            field=f'chunking.config.hierarchical_template.boundaries[{i}]',
                            message='Each boundary must include a pattern'
                        ))
                        continue
                    pat = str(rule.get('pattern') or '')
                    # Safety check (length + nested quantifier guard + compile test)
                    err = _rx_check(pat, max_len=256)
                    if err:
                        with contextlib.suppress(_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS):
                            increment_counter("chunking_templates_regex_reject_total", labels={"reason": "safety_check"})
                        errors.append(TemplateValidationError(
                            field=f'chunking.config.hierarchical_template.boundaries[{i}].pattern',
                            message=err
                        ))
                    flags_str = str(rule.get('flags') or '').lower()
                    re_flags, ferr = _rx_flags(flags_str)
                    if ferr:
                        with contextlib.suppress(_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS):
                            increment_counter("chunking_templates_regex_reject_total", labels={"reason": "flags"})
                        errors.append(TemplateValidationError(
                            field=f'chunking.config.hierarchical_template.boundaries[{i}].flags',
                            message=ferr
                        ))
                    # Ambiguity warning (non-fatal)
                    w = _rx_warn(pat)
                    if w:
                        warnings.append(TemplateValidationError(
                            field=f'chunking.config.hierarchical_template.boundaries[{i}].pattern',
                            message=w
                        ))

        # Validate classifier
        classifier = template_config.get('classifier') or _get_cfg_path(template_config, ['chunking', 'config', 'classifier'])
        if classifier is not None and not isinstance(classifier, dict):
            errors.append(TemplateValidationError(
                field='classifier',
                message='classifier must be an object'
            ))
        elif isinstance(classifier, dict):
            ms = classifier.get('min_score')
            if ms is not None:
                try:
                    f = float(ms)
                    if f < 0 or f > 1:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(TemplateValidationError(field='classifier.min_score', message='min_score must be in [0,1]'))
            pr = classifier.get('priority')
            if pr is not None and not isinstance(pr, int):
                errors.append(TemplateValidationError(field='classifier.priority', message='priority must be integer'))
            # Validate classifier regex patterns with light guardrails
            for key in ('filename_regex', 'title_regex', 'url_regex'):
                pat = classifier.get(key)
                if pat is None:
                    continue
                if not isinstance(pat, str):
                    errors.append(TemplateValidationError(field=f'classifier.{key}', message='must be a string'))
                    continue
                if len(pat) > 128:
                    errors.append(TemplateValidationError(field=f'classifier.{key}', message='Pattern too long (max 128)'))
                    continue
                perr = _rx_check(pat, max_len=128)
                if perr:
                    with contextlib.suppress(_CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS):
                        increment_counter("chunking_templates_regex_reject_total", labels={"reason": "classifier"})
                    errors.append(TemplateValidationError(field=f'classifier.{key}', message=perr))

        # Validate preprocessing operations
        if 'preprocessing' in template_config:
            if not isinstance(template_config['preprocessing'], list):
                errors.append(TemplateValidationError(
                    field='preprocessing',
                    message='Preprocessing must be a list of operations'
                ))
            else:
                for i, op in enumerate(template_config['preprocessing']):
                    if not isinstance(op, dict) or 'operation' not in op:
                        errors.append(TemplateValidationError(
                            field=f'preprocessing[{i}]',
                            message='Each preprocessing operation must have an "operation" field'
                        ))

        # Validate postprocessing operations
        if 'postprocessing' in template_config:
            if not isinstance(template_config['postprocessing'], list):
                errors.append(TemplateValidationError(
                    field='postprocessing',
                    message='Postprocessing must be a list of operations'
                ))
            else:
                for i, op in enumerate(template_config['postprocessing']):
                    if not isinstance(op, dict) or 'operation' not in op:
                        errors.append(TemplateValidationError(
                            field=f'postprocessing[{i}]',
                            message='Each postprocessing operation must have an "operation" field'
                        ))

        # Try to serialize as JSON to catch any serialization issues
        try:
            json.dumps(template_config)
        except (TypeError, ValueError) as e:
            errors.append(TemplateValidationError(
                field='template_config',
                message=f'Template configuration is not JSON serializable: {str(e)}'
            ))

        return TemplateValidationResponse(
            valid=len(errors) == 0,
            errors=errors if errors else None,
            warnings=warnings if warnings else None
        )

    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error validating template")
        return TemplateValidationResponse(
            valid=False,
            errors=[TemplateValidationError(
                field='template_config',
                message=f'Validation error: {str(e)}'
            )]
        )


@router.post("/match")
async def match_templates(
    media_type: Optional[str] = Query(None),
    title: Optional[str] = Query(None),
    url: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
):
    """Return templates ranked by a simple metadata-based score for auto-apply."""
    try:
        templates = db.list_chunking_templates(include_builtin=True, include_custom=True, tags=None, user_id=None, include_deleted=False)
        ranked = []
        for t in templates:
            try:
                cfg = json.loads(t['template_json']) if isinstance(t.get('template_json'), str) else (t.get('template_json') or {})
            except (TypeError, ValueError, json.JSONDecodeError):
                cfg = {}
            s = TemplateClassifier.score(cfg, media_type=media_type, title=title, url=url, filename=filename)
            if s > 0:
                ranked.append({"name": t['name'], "score": s, "priority": (cfg.get('classifier') or {}).get('priority', 0)})
        # sort by score desc then priority desc
        ranked.sort(key=lambda x: (x['score'], x.get('priority', 0)), reverse=True)
        return {"matches": ranked}
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Failed to match chunking templates") from e


class LearnTemplateRequest(BaseModel):
    name: str
    example_text: Optional[str] = None
    description: Optional[str] = None
    save: bool = False
    classifier: Optional[dict[str, Any]] = None


@router.post("/learn")
async def learn_template(
    req: LearnTemplateRequest,
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
):
    """Learn a basic hierarchical boundary template from an example text and optionally save it."""
    try:
        boundaries = TemplateLearner.learn_boundaries(req.example_text or "")
        tmpl = {
            "name": req.name,
            "description": req.description or "Learned template",
            "chunking": {
                "method": "sentences",
                "config": {
                    "hierarchical": True,
                    "hierarchical_template": boundaries,
                    "classifier": req.classifier or {},
                }
            }
        }
        if req.save:
            uid = str(getattr(current_user, 'id', ''))
            if _supports(db, 'create_chunking_template'):
                db.create_chunking_template(name=req.name, template_json=json.dumps(tmpl), description=req.description or "Learned", is_builtin=False, tags=["learned"], user_id=uid)
            else:
                bucket = _fb_bucket(uid)
                from uuid import uuid4
                bucket[req.name] = {
                    'id': len(bucket) + 1,
                    'uuid': str(uuid4()),
                    'name': req.name,
                    'description': req.description or "Learned",
                    'template_json': json.dumps(tmpl),
                    'is_builtin': False,
                    'tags': ["learned"],
                    'created_at': _now_iso(),
                    'updated_at': _now_iso(),
                    'version': 1,
                    'user_id': uid,
                }
        return {"template": tmpl, "saved": req.save}
    except _CHUNKING_TEMPLATES_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Failed to learn chunking template") from e
