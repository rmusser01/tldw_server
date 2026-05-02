# config_admin.py - Effective config diagnostics (admin-only)
"""Admin-only endpoints for effective configuration diagnostics."""

from __future__ import annotations

import configparser
import os
import re
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole, check_rate_limit
from tldw_Server_API.app.api.v1.schemas.config_schemas import (
    ConfigValue,
    EffectiveConfigResponse,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.config_paths import (
    resolve_config_file,
    resolve_config_root,
    resolve_module_yaml,
    resolve_prompts_dir,
)
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    get_config as get_embeddings_config,
)
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    get_config_sources as get_embeddings_sources,
)
from tldw_Server_API.app.core.Evaluations.config_manager import get_config_snapshot as get_evaluations_snapshot
from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

router = APIRouter(
    prefix="/admin/config",
    tags=["admin", "config"],
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|access[_-]?key|token|secret|password|private[_-]?key|signing[_-]?key|encryption[_-]?key)",
    re.IGNORECASE,
)
_ALLOWED_SOURCES = {"env", "config", "yaml", "default"}
_REDACTED_VALUE = "<redacted>"


def _is_sensitive(path: str) -> bool:
    return bool(_SENSITIVE_KEY_PATTERN.search(path))


def _normalize_source(raw: str | None) -> str:
    if raw in _ALLOWED_SOURCES:
        return raw
    return "default"


def _flatten_values(
    data: Any,
    sources: Any,
    prefix: str,
    out: dict[str, ConfigValue],
) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            branch = sources.get(key) if isinstance(sources, dict) else sources
            _flatten_values(value, branch, path, out)
        return

    if isinstance(data, list):
        for idx, value in enumerate(data):
            path = f"{prefix}[{idx}]"
            _flatten_values(value, sources, path, out)
        return

    if not prefix:
        return

    source = _normalize_source(sources if isinstance(sources, str) else None)
    redacted = _is_sensitive(prefix)
    value = _REDACTED_VALUE if redacted else data
    out[prefix] = ConfigValue(value=value, source=source, redacted=redacted)


def _filter_defaults(
    values: dict[str, ConfigValue],
    include_defaults: bool,
) -> dict[str, ConfigValue]:
    if include_defaults:
        return values
    return {key: val for key, val in values.items() if val.source != "default"}


def _build_config_txt_values() -> dict[str, ConfigValue]:
    values: dict[str, ConfigValue] = {}
    config_parser = load_comprehensive_config()
    if not hasattr(config_parser, "sections"):
        return values

    try:
        sections = config_parser.sections()
    except configparser.Error:
        logger.error("Error reading config sections")
        sections = []

    for section in sections:
        try:
            items = config_parser.items(section)
        except configparser.Error:
            logger.error("Error reading config items")
            items = []
        for key, raw_value in items:
            path = f"{section}.{key}"
            redacted = _is_sensitive(path)
            values[path] = ConfigValue(
                value=_REDACTED_VALUE if redacted else raw_value,
                source="config",
                redacted=redacted,
            )
    return values


def _build_tts_values() -> dict[str, ConfigValue]:
    manager = get_tts_config_manager()
    data = manager.to_dict()
    sources = manager.get_sources()
    values: dict[str, ConfigValue] = {}
    _flatten_values(data, sources, "", values)
    return values


def _build_embeddings_values() -> dict[str, ConfigValue]:
    config = get_embeddings_config()
    data = config.to_dict()
    sources = get_embeddings_sources()
    values: dict[str, ConfigValue] = {}
    _flatten_values(data, sources, "", values)
    return values


def _build_evaluations_values() -> dict[str, ConfigValue]:
    snapshot = get_evaluations_snapshot()
    data = snapshot.get("config", {})
    sources = snapshot.get("sources", {})
    values: dict[str, ConfigValue] = {}
    _flatten_values(data, sources, "", values)
    return values


def _normalize_sections(sections: Iterable[str] | None) -> list[str]:
    if not sections:
        return ["config_txt", "tts", "embeddings", "evaluations"]
    normalized = []
    for section in sections:
        if not section:
            continue
        key = section.strip().lower()
        if key == "config":
            key = "config_txt"
        normalized.append(key)
    return normalized


@router.get(
    "/effective",
    response_model=EffectiveConfigResponse,
    summary="Get effective configuration with redaction",
)
async def get_effective_config(
    sections: list[str] | None = Query(
        None,
        description="Limit response to specific config namespaces (e.g., tts, embeddings)",
    ),
    include_defaults: bool = Query(
        True,
        description="Include default values when true",
    ),
) -> EffectiveConfigResponse:
    """
    Return the effective configuration with sensitive fields redacted.

    Args:
        sections: Optional list of namespaces to include (e.g., tts, embeddings).
        include_defaults: Whether to include default values in the response.

    Returns:
        EffectiveConfigResponse containing the resolved configuration snapshot.

    Raises:
        HTTPException: If config resolution fails.
    """
    try:
        config_root = resolve_config_root()
        config_file = resolve_config_file()
        prompts_dir = resolve_prompts_dir()
    except FileNotFoundError as exc:
        logger.debug("Effective config resolution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve effective configuration",
        ) from exc

    tts_yaml = resolve_module_yaml("tts")
    embeddings_yaml = resolve_module_yaml(
        "embeddings",
        filename_override=os.getenv("EMBEDDINGS_CONFIG_PATH"),
    )
    evaluations_yaml = resolve_module_yaml(
        "evaluations",
        filename_override=os.getenv("EVALUATIONS_CONFIG_PATH"),
    )
    module_yaml = {
        "tts": str(tts_yaml) if tts_yaml else None,
        "embeddings": str(embeddings_yaml) if embeddings_yaml else None,
        "evaluations": str(evaluations_yaml) if evaluations_yaml else None,
    }

    builders = {
        "config_txt": _build_config_txt_values,
        "tts": _build_tts_values,
        "embeddings": _build_embeddings_values,
        "evaluations": _build_evaluations_values,
    }
    selected_sections = _normalize_sections(sections)
    values: dict[str, dict[str, ConfigValue]] = {}
    unknown_sections = sorted({section for section in selected_sections if section not in builders})

    for section in selected_sections:
        builder = builders.get(section)
        if not builder:
            continue
        section_values = builder()
        values[section] = _filter_defaults(section_values, include_defaults)

    return EffectiveConfigResponse(
        config_root=str(config_root),
        config_file=str(config_file) if config_file else None,
        prompts_dir=str(prompts_dir) if prompts_dir else None,
        module_yaml=module_yaml,
        values=values,
        unknown_sections=unknown_sections,
    )


# ---------------------------------------------------------------------------
# Config Profiles & Editing (Items 8, 10, 16)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class ConfigSectionUpdateRequest(BaseModel):
    values: dict[str, str] = Field(..., description="Key-value pairs to set in the section")


class ConfigProfileSaveRequest(BaseModel):
    name: str = Field(..., description="Profile name")
    description: str = Field(default="", description="Optional description")


class ConfigImportRequest(BaseModel):
    sections: dict[str, dict[str, str]] = Field(..., description="Sections to import (section_name -> {key: value})")


@router.get("/profiles")
async def list_config_profiles() -> dict[str, Any]:
    """List saved config profiles."""
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    profiles = await store.list_profiles()
    return {"profiles": profiles}


@router.post("/profiles/snapshot")
async def snapshot_config_profile(payload: ConfigProfileSaveRequest) -> dict[str, Any]:
    """Save the current config.txt as a named profile (snapshot)."""
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    profile = await store.snapshot_current_config(payload.name, payload.description)
    return {"status": "ok", "profile": {"name": profile["name"], "created_at": profile["created_at"]}}


@router.get("/profiles/{profile_name}")
async def get_config_profile(profile_name: str) -> dict[str, Any]:
    """Get a specific config profile."""
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    profile = await store.get_profile(profile_name)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile_not_found")
    return profile


@router.post("/profiles/{profile_name}/restore")
async def restore_config_profile(profile_name: str) -> dict[str, Any]:
    """Restore config.txt from a saved profile.

    Creates a 'pre_restore' snapshot before applying changes.
    """
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    profile = await store.get_profile(profile_name)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile_not_found")

    # Auto-snapshot before restore
    await store.snapshot_current_config(
        f"pre_restore_{profile_name}",
        f"Auto-snapshot before restoring profile '{profile_name}'",
    )

    # Import profile sections
    result = await store.import_config(profile.get("sections", {}))
    return {"status": "ok", "restored_from": profile_name, **result}


@router.delete("/profiles/{profile_name}")
async def delete_config_profile(profile_name: str) -> dict[str, str]:
    """Delete a config profile."""
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    deleted = await store.delete_profile(profile_name)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile_not_found")
    return {"status": "ok"}


@router.put("/sections/{section}")
async def update_config_section(section: str, payload: ConfigSectionUpdateRequest) -> dict[str, Any]:
    """Update values in a config.txt section.

    Creates an auto-snapshot before applying changes.
    """
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    # Auto-snapshot before edit
    await store.snapshot_current_config(
        f"pre_edit_{section}",
        f"Auto-snapshot before editing section [{section}]",
    )
    updated = await store.update_config_section(section, payload.values)
    return {"status": "ok", "section": section, "values": updated}


@router.get("/export")
async def export_config() -> dict[str, Any]:
    """Export the current config.txt as JSON."""
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    return await store.export_config()


@router.post("/import")
async def import_config(payload: ConfigImportRequest) -> dict[str, Any]:
    """Import config sections from JSON, overwriting existing values.

    Creates an auto-snapshot before applying changes.
    """
    from tldw_Server_API.app.services.admin_config_profiles_service import get_config_profile_store

    store = await get_config_profile_store()
    await store.snapshot_current_config("pre_import", "Auto-snapshot before config import")
    result = await store.import_config(payload.sections)
    return {"status": "ok", **result}
