"""Setup endpoints for the first-time configuration flow."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import sys
from configparser import ConfigParser
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    RequireRole,
    check_rate_limit,
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.API_Deps.setup_deps import (
    require_local_setup_access,
    require_shared_audio_installer_access,
)
from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
    AudioBundleOperationResponse,
    AudioPackExportResponse,
    AudioPackImportResponse,
    AudioReadinessResetResponse,
    AudioRecommendationsResponse,
    FirstRunConnectionDiagnostics,
    FirstRunMetadataResponse,
    FirstRunMultiUserExit,
    FirstRunSetupPath,
    FirstRunSkipRequest,
    FirstRunStateResponse,
    FirstRunStepUpdateRequest,
    SetupAssistantResponse,
    SetupCompleteResponse,
    SetupConfigUpdateResponse,
    SetupInstallStatusResponse,
    SetupReadinessPreviewRequest,
    SetupReadinessPreviewResponse,
    SetupReadinessProvisionRequest,
    SetupReadinessProvisionResponse,
    SetupReadinessVerifyRequest,
    SetupReadinessVerifyResponse,
    SetupResetResponse,
    SetupStatusResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Setup import (
    audio_pack_service,
    audio_profile_service,
    audio_readiness_store,
    install_manager,
    readiness_store,
    setup_manager,
)
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    DEFAULT_AUDIO_RESOURCE_PROFILE,
    get_audio_bundle_catalog,
)
from tldw_Server_API.app.core.Setup.first_run_state import (
    FirstRunStateStore,
    FirstRunStatus,
    InvalidFirstRunTransition,
)
from tldw_Server_API.app.core.Setup.install_manager import execute_install_plan
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan
from tldw_Server_API.app.core.Setup.readiness_profiles import build_readiness_profiles
from tldw_Server_API.app.core.Setup.readiness_models import LANE_IDS, LANE_STATUSES, OVERLAY_IDS
from tldw_Server_API.app.core.Setup.readiness_service import preview_readiness_selection, verify_readiness_lanes
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat
from tldw_Server_API.app.services.auth_service import mark_user_verified

router = APIRouter(prefix="/setup", tags=["setup"], include_in_schema=True)

FIRST_RUN_STATE_PATH = setup_manager.resolve_config_root() / "first_run_state.json"
INVALID_AUDIO_BUNDLE_REQUEST_DETAIL = "Invalid audio bundle request"
INVALID_AUDIO_PACK_EXPORT_REQUEST_DETAIL = "Invalid audio pack export request"
AUDIO_BUNDLE_NOT_FOUND_DETAIL = "Audio bundle not found"
_SUSPICIOUS_SETUP_DETAIL_RE = re.compile(
    r"traceback|stack(?:\s*trace)?|exception|\/Users\/|[A-Za-z]:\\|\.py:\d+",
    re.IGNORECASE,
)
_SANITIZED_SETUP_DETAIL_MESSAGE = "Internal setup diagnostics were suppressed."


class ConfigUpdates(BaseModel):
    updates: dict[str, dict[str, Any]] = Field(
        ..., description="Mapping of section -> key/value pairs to persist in config.txt"
    )


class SetupCompleteRequest(BaseModel):
    disable_first_time_setup: bool | None = Field(
        False,
        description="If true, flips enable_first_time_setup to false so the screen stays hidden",
    )
    install_plan: InstallPlan | None = Field(
        None,
        description="Backend installation instructions to execute after setup completes.",
    )


class AssistantQuestion(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question for the setup assistant")


def _legacy_pack_name(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _first_run_store() -> FirstRunStateStore:
    return FirstRunStateStore(FIRST_RUN_STATE_PATH)


async def _require_first_run_write_access(request: Request) -> None:
    await require_local_setup_access(request)
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot.get("enabled"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="setup_disabled",
        )
    if (
        status_snapshot.get("setup_completed")
        or status_snapshot.get("completed")
        or not status_snapshot.get("needs_setup")
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="setup_already_completed",
        )
    state = _first_run_store().load()
    if state.status == FirstRunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="setup_already_completed",
        )


def _read_config_auth_mode() -> str | None:
    try:
        parser = ConfigParser()
        parser.read(setup_manager.get_config_file_path(), encoding="utf-8")
        auth_mode = parser.get("AuthNZ", "auth_mode", fallback=None)
    except Exception as exc:  # noqa: BLE001 - metadata should remain best-effort
        logger.debug("Unable to read auth mode for first-run metadata: {}", type(exc).__name__)
        return None
    return auth_mode.strip() if auth_mode else None


def _resolve_auth_mode(status_snapshot: dict[str, Any]) -> str:
    raw_auth_mode = status_snapshot.get("auth_mode") or os.getenv("AUTH_MODE") or _read_config_auth_mode()
    return str(raw_auth_mode or "single_user").strip() or "single_user"


def _origin_from_request(request: Request) -> str | None:
    try:
        return str(request.base_url).rstrip("/")
    except Exception:  # noqa: BLE001 - diagnostics should not fail setup metadata
        return None


def _frontend_origin_from_headers(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin:
        return origin.rstrip("/")

    referer = request.headers.get("referer")
    if not referer:
        return None
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(referer)
    except Exception:  # noqa: BLE001
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_local_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip().lower().strip("[]")
    return normalized in {"localhost", "127.0.0.1", "::1", "testclient", "testserver"}


def _is_lan_host(host: str | None) -> bool:
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host.strip().lower().strip("[]"))
    except ValueError:
        return False
    return address.is_private


def _classify_browser_access(request: Request) -> str:
    host = request.url.hostname
    client_host = request.client.host if request.client else None

    if _is_local_host(host) or _is_local_host(client_host):
        return "local"
    if _is_lan_host(host) or _is_lan_host(client_host):
        return "lan"
    if host or client_host:
        return "remote"
    return "unknown"


def build_first_run_metadata(request: Request) -> FirstRunMetadataResponse:
    status_snapshot = setup_manager.get_status_snapshot()
    auth_mode = _resolve_auth_mode(status_snapshot)
    browser_access = _classify_browser_access(request)
    setup_completed = bool(status_snapshot.get("setup_completed") or status_snapshot.get("completed"))
    remote_setup_enabled = bool(
        status_snapshot.get("remote_access_active")
        or status_snapshot.get("allow_remote_setup_access")
        or status_snapshot.get("remote_access_env_override")
    )
    bundled_single_user_auth_available = auth_mode == "single_user" and browser_access == "local"

    return FirstRunMetadataResponse(
        auth_mode=auth_mode,
        bundled_single_user_auth_available=bundled_single_user_auth_available,
        manual_auth_required=not bundled_single_user_auth_available,
        setup_required=bool(status_snapshot.get("needs_setup")),
        setup_completed=setup_completed,
        remote_setup_enabled=remote_setup_enabled,
        connection=FirstRunConnectionDiagnostics(
            frontend_origin=_frontend_origin_from_headers(request),
            api_origin=_origin_from_request(request),
            browser_access=browser_access,
        ),
        setup_paths=[
            FirstRunSetupPath(
                key="docker_single_user",
                label="Docker single-user",
                recommended=True,
                guide_path="Docs/Getting_Started/Profile_Docker_Single_User.md",
            ),
            FirstRunSetupPath(
                key="local_single_user",
                label="Local single-user",
                guide_path="Docs/Getting_Started/Profile_Local_Single_User.md",
            ),
            FirstRunSetupPath(
                key="multi_user",
                label="Multi-user",
                guide_path="Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md",
            ),
        ],
        multi_user_exit=FirstRunMultiUserExit(
            guide_path="Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md",
            checklist_path="Docs/User_Guides/Server/Multi-User_Deployment_Guide.md",
        ),
    )


def _sanitize_setup_payload(value: Any) -> Any:
    if isinstance(value, str):
        return (
            _SANITIZED_SETUP_DETAIL_MESSAGE
            if _SUSPICIOUS_SETUP_DETAIL_RE.search(value)
            else value
        )
    if isinstance(value, list):
        return [_sanitize_setup_payload(item) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"details", "exception", "traceback", "stack", "stack_trace"}:
                continue
            sanitized[key] = _sanitize_setup_payload(item)
        return sanitized
    return value


class AudioBundleProvisionRequest(BaseModel):
    bundle_id: str = Field(..., min_length=1, description="Curated audio bundle identifier to provision.")
    resource_profile: str = Field(
        DEFAULT_AUDIO_RESOURCE_PROFILE,
        min_length=1,
        description="Selected resource profile within the curated audio bundle.",
    )
    safe_rerun: bool = Field(
        False,
        description="If true, skip bundle installation only when all expected install steps were previously completed.",
    )
    tts_choice: str | None = Field(
        None,
        description="Optional curated TTS choice for profiles that expose multiple curated TTS engines.",
    )


class AudioBundleVerificationRequest(BaseModel):
    bundle_id: str = Field(..., min_length=1, description="Curated audio bundle identifier to verify.")
    resource_profile: str = Field(
        DEFAULT_AUDIO_RESOURCE_PROFILE,
        min_length=1,
        description="Selected resource profile within the curated audio bundle.",
    )
    tts_choice: str | None = Field(
        None,
        description="Optional curated TTS choice for profiles that expose multiple curated TTS engines.",
    )


class AudioPackExportRequest(BaseModel):
    bundle_id: str = Field(..., min_length=1, description="Curated audio bundle identifier to export.")
    resource_profile: str = Field(
        DEFAULT_AUDIO_RESOURCE_PROFILE,
        min_length=1,
        description="Selected resource profile within the curated audio bundle.",
    )
    pack_name: str | None = Field(
        None,
        description="Optional filename-friendly pack name for the generated audio pack manifest.",
    )
    pack_path: str | None = Field(
        None,
        description="Optional path to write the generated audio pack manifest.",
    )
    tts_choice: str | None = Field(
        None,
        description="Optional curated TTS choice for profiles that expose multiple curated TTS engines.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_pack_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("pack_name") and data.get("pack_path"):
            payload = dict(data)
            payload["pack_name"] = _legacy_pack_name(payload["pack_path"])
            return payload
        return data


class AudioPackImportRequest(BaseModel):
    pack_name: str = Field(
        ...,
        min_length=1,
        description="JSON filename inside the setup-managed audio_packs directory.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_pack_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("pack_name") and data.get("pack_path"):
            payload = dict(data)
            payload["pack_name"] = _legacy_pack_name(payload["pack_path"])
            return payload
        return data


async def require_admin_and_system_configure(
    principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
) -> AuthPrincipal:
    """
    Combined dependency that enforces an admin-style principal and reuses the
    SYSTEM_CONFIGURE permission gate while resolving the AuthPrincipal once.

    Semantics:
    - Principals with ``is_admin=True`` are allowed regardless of an explicit
      SYSTEM_CONFIGURE grant (matching other admin surfaces).
    - Other principals must hold the ``admin`` role and the SYSTEM_CONFIGURE
      permission to pass.
    """
    role_checker = RequireRole("admin")
    perm_checker = RequirePermission(SYSTEM_CONFIGURE)

    principal = await role_checker(principal)
    principal = await perm_checker(principal)
    return principal


def _audio_pack_compatibility(machine_profile: audio_profile_service.MachineProfile) -> dict[str, str]:
    """Project machine-profile data into the portable manifest compatibility shape."""
    return {
        "platform": machine_profile.platform,
        "arch": machine_profile.arch,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def _normalize_audio_pack_name(pack_name: str) -> str:
    """Normalize a managed audio pack filename before resolving it under the setup pack directory."""
    try:
        return audio_pack_service.normalize_audio_pack_name(pack_name)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _raise_audio_bundle_lookup_not_found(exc: KeyError) -> None:
    raise HTTPException(status.HTTP_404_NOT_FOUND, detail=AUDIO_BUNDLE_NOT_FOUND_DETAIL) from exc


@router.get("/status", openapi_extra={"security": []}, response_model=SetupStatusResponse)
async def get_setup_status(_guard: None = Depends(require_local_setup_access)) -> SetupStatusResponse:
    """Return setup availability and placeholder diagnostics."""
    return setup_manager.get_status_snapshot()


@router.get("/first-run/state", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def get_first_run_state(_guard: None = Depends(require_local_setup_access)) -> FirstRunStateResponse:
    return _first_run_store().load()


@router.get("/first-run/metadata", openapi_extra={"security": []}, response_model=FirstRunMetadataResponse)
async def get_first_run_metadata(
    request: Request,
    _guard: None = Depends(require_local_setup_access),
) -> FirstRunMetadataResponse:
    return build_first_run_metadata(request)


@router.post("/first-run/state", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def update_first_run_state(
    payload: FirstRunStepUpdateRequest,
    _guard: None = Depends(_require_first_run_write_access),
) -> FirstRunStateResponse:
    try:
        return _first_run_store().update_step(payload.step, payload.data)
    except InvalidFirstRunTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/first-run/skip", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def skip_first_run(
    payload: FirstRunSkipRequest,
    _guard: None = Depends(_require_first_run_write_access),
) -> FirstRunStateResponse:
    try:
        return _first_run_store().mark_skipped(reason=payload.reason)
    except InvalidFirstRunTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/config", openapi_extra={"security": []})
async def get_setup_config(_guard: None = Depends(require_local_setup_access)) -> dict[str, Any]:
    """Return the current configuration grouped by section for the setup UI."""
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    if not status_snapshot["needs_setup"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Setup already completed. Toggle enable_first_time_setup to revisit the wizard.",
        )

    return setup_manager.get_config_snapshot()


@router.get(
    "/install-status",
    openapi_extra={"security": []},
    response_model=SetupInstallStatusResponse,
)
async def get_install_status(_guard: None = Depends(require_local_setup_access)) -> SetupInstallStatusResponse:
    """Return the current installation plan progress if available."""

    return _get_audio_install_status()


def _ensure_audio_installer_available(*, allow_completed_when_disabled: bool) -> None:
    """Validate whether audio installer actions should remain available."""
    status_snapshot = setup_manager.get_status_snapshot()
    if status_snapshot["enabled"]:
        return

    if allow_completed_when_disabled and (
        status_snapshot.get("setup_completed") or status_snapshot.get("completed")
    ):
        return

    raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")


def _get_audio_install_status(*, allow_completed_when_disabled: bool = False) -> dict[str, Any]:
    """Return the current audio install status payload used by legacy and admin routes."""
    _ensure_audio_installer_available(allow_completed_when_disabled=allow_completed_when_disabled)

    install_status = install_manager.get_install_status_snapshot()
    if not install_status:
        return JSONResponse({"status": "idle"})

    return JSONResponse(install_status)


def _ensure_setup_readiness_available(
    status_snapshot: dict[str, Any],
    *,
    allow_completed_when_disabled: bool = False,
) -> None:
    """Validate whether first-run setup readiness routes should be visible."""

    if status_snapshot["enabled"]:
        if status_snapshot["needs_setup"] or allow_completed_when_disabled:
            return
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Setup already completed. Use the admin setup readiness endpoints.",
        )

    if allow_completed_when_disabled and (
        status_snapshot.get("setup_completed") or status_snapshot.get("completed")
    ):
        return

    raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")


def _build_setup_readiness_profiles_payload(
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Build the read-only first-run setup readiness profile/status payload."""

    status_snapshot = setup_manager.get_status_snapshot()
    _ensure_setup_readiness_available(
        status_snapshot,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    recommendations = _build_audio_recommendations_response(
        prefer_offline_runtime=True,
        allow_hosted_fallbacks=True,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    payload = build_readiness_profiles(
        setup_status=status_snapshot,
        config_snapshot=setup_manager.get_config_snapshot(),
        audio_recommendations=recommendations,
    )
    payload["overlays"] = list(payload.get("active_overlays", []))
    return _merge_setup_readiness_store_payload(payload)


def _build_setup_readiness_status_base_payload(
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Build the lightweight pollable readiness status payload without recommendation work."""

    status_snapshot = setup_manager.get_status_snapshot()
    _ensure_setup_readiness_available(
        status_snapshot,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    setup_mode = "first_run" if status_snapshot.get("needs_setup") else "admin"
    overlays = ["requires_admin"] if not status_snapshot.get("needs_setup") else []
    return {
        "setup_access": {
            "mode": setup_mode,
            "needs_setup": bool(status_snapshot.get("needs_setup")),
            "setup_completed": bool(status_snapshot.get("setup_completed")),
            "remote_access_active": bool(status_snapshot.get("remote_access_active")),
        },
        "lane_ids": list(LANE_IDS),
        "supported_statuses": list(LANE_STATUSES),
        "supported_overlays": list(OVERLAY_IDS),
        "active_overlays": overlays,
        "overlays": list(overlays),
        "readiness_status": "not_started",
        "operation_id": None,
        "operation_status": None,
        "errors": [],
    }


async def _build_setup_readiness_status_payload(
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Build a lightweight readiness status payload for frequent polling."""

    return await _merge_setup_readiness_store_payload_async(
        _build_setup_readiness_status_base_payload(
            allow_completed_when_disabled=allow_completed_when_disabled,
        )
    )


def _merge_setup_readiness_record_payload(payload: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    """Overlay one persisted readiness record onto a profile or status payload."""

    if readiness.get("status") == "not_started":
        return payload

    merged = dict(payload)
    merged["readiness_status"] = readiness.get("status")
    merged["selected_profile_id"] = readiness.get("selected_profile_id")
    merged["operation_id"] = readiness.get("operation_id")
    merged["operation_status"] = readiness.get("operation_status")
    merged["last_preview"] = readiness.get("last_preview")
    merged["last_provision"] = readiness.get("last_provision")
    merged["errors"] = readiness.get("errors", [])
    if readiness.get("lanes"):
        merged["lanes"] = readiness["lanes"]
    merged["overlays"] = list(dict.fromkeys([*merged.get("overlays", []), *readiness.get("overlays", [])]))
    return merged


def _merge_setup_readiness_store_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay persisted preview/provision status onto the profile/status payload."""

    store = readiness_store.get_setup_readiness_store()
    readiness = store.load()
    return _merge_setup_readiness_record_payload(payload, readiness)


async def _load_setup_readiness(
    store: readiness_store.SetupReadinessStore,
) -> dict[str, Any]:
    return await asyncio.to_thread(store.load)


async def _save_setup_readiness(
    store: readiness_store.SetupReadinessStore,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return await asyncio.to_thread(store.save, readiness)


async def _update_setup_readiness(
    store: readiness_store.SetupReadinessStore,
    **fields: Any,
) -> dict[str, Any]:
    return await asyncio.to_thread(store.update, **fields)


async def _update_setup_config(config_updates: dict[str, dict[str, Any]]) -> Path | None:
    return await asyncio.to_thread(setup_manager.update_config, config_updates)


async def _merge_setup_readiness_store_payload_async(payload: dict[str, Any]) -> dict[str, Any]:
    """Async-safe status merge that keeps store writes off the event loop."""

    store = readiness_store.get_setup_readiness_store()
    readiness = await _load_setup_readiness(store)
    readiness = await _refresh_setup_readiness_operation_status_async(readiness, store)
    return _merge_setup_readiness_record_payload(payload, readiness)


def _readiness_status_for_completed_install(
    readiness: dict[str, Any],
    install_status: dict[str, Any],
) -> str:
    """Derive final readiness after installer completion without promoting warnings to ready."""

    if install_status.get("errors") or readiness.get("errors") or readiness.get("overlays"):
        return "ready_with_warnings"
    return "ready"


async def _refresh_setup_readiness_operation_status_async(
    readiness: dict[str, Any],
    store: readiness_store.SetupReadinessStore,
) -> dict[str, Any]:
    """Async-safe installer status refresh for pollable readiness endpoints."""

    if readiness.get("operation_status") not in {"queued", "running"}:
        return readiness

    last_provision = readiness.get("last_provision")
    expected_plan = last_provision.get("install_plan") if isinstance(last_provision, dict) else None
    if not expected_plan:
        return readiness

    install_status = install_manager.get_install_status_snapshot()
    if not install_status or install_status.get("plan") != expected_plan:
        return readiness

    installer_status = install_status.get("status")
    operation_status = readiness.get("operation_status")
    readiness_status = readiness.get("status")
    if installer_status == "in_progress":
        operation_status = "running"
        readiness_status = "provisioning"
    elif installer_status == "completed":
        operation_status = "completed"
        readiness_status = _readiness_status_for_completed_install(readiness, install_status)
    elif installer_status == "failed":
        operation_status = "failed"
        readiness_status = "failed"

    if operation_status == readiness.get("operation_status") and readiness_status == readiness.get("status"):
        return readiness

    last_provision = dict(last_provision)
    last_provision["operation_status"] = operation_status
    last_provision["install_status"] = install_status
    return await _update_setup_readiness(
        store,
        status=readiness_status,
        operation_status=operation_status,
        last_provision=last_provision,
        errors=install_status.get("errors", []),
    )


def _new_setup_readiness_preview_id() -> str:
    return f"preview-{uuid.uuid4().hex}"


def _new_setup_readiness_operation_id() -> str:
    return f"setup-readiness-{uuid.uuid4().hex}"


def _resolve_setup_readiness_profile_selection(
    selection: Any,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Expand a curated profile-only selection into concrete lane inputs."""

    payload = model_dump_compat(selection, exclude_none=True)
    if payload.get("lanes"):
        return payload

    profile_id = str(payload.get("profile_id") or "").strip()
    if not profile_id:
        return payload

    profiles_payload = _build_setup_readiness_profiles_payload(
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    profile = next(
        (
            item
            for item in profiles_payload.get("profiles", [])
            if isinstance(item, dict) and item.get("profile_id") == profile_id
        ),
        None,
    )
    if not profile:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown setup readiness profile: {profile_id}",
        )

    resolved = dict(payload)
    resolved["lanes"] = profile.get("lanes") or {}
    return resolved


def _preview_lanes_as_list(preview: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = preview.get("lanes") or {}
    return [lane for lane in lanes.values() if isinstance(lane, dict)]


def _preview_has_blockers(preview: dict[str, Any]) -> bool:
    return any(lane.get("status") == "blocked" for lane in _preview_lanes_as_list(preview))


def _provision_lanes_from_preview(preview: dict[str, Any], *, install_plan_submitted: bool) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []
    for lane in _preview_lanes_as_list(preview):
        lane_payload = dict(lane)
        if lane_payload.get("status") == "previewed" and install_plan_submitted:
            lane_payload["status"] = "provisioning"
        lanes.append(lane_payload)
    return lanes


def _secret_config_keys_from_preview(preview: dict[str, Any]) -> set[tuple[str, str]]:
    """Return config update keys that correspond to submitted secret fields."""

    secret_keys: set[tuple[str, str]] = set()
    for field in preview.get("secret_fields") or []:
        if not isinstance(field, dict) or field.get("state") != "submitted":
            continue
        section = str(field.get("section") or "").strip()
        key = str(field.get("key") or "").strip()
        if section and key:
            secret_keys.add((section, key))
    return secret_keys


def _raise_if_preview_lost_submitted_secrets(preview: dict[str, Any]) -> None:
    """Reject provisioning previews that acknowledge secrets without retaining values."""

    config_updates = preview.get("config_updates") or {}
    missing = [
        f"{section}.{key}"
        for section, key in sorted(_secret_config_keys_from_preview(preview))
        if key not in (config_updates.get(section) or {})
    ]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                "Submitted secret values are not retained after preview. "
                "Re-submit the readiness selection during provisioning."
            ),
        )


def _stored_setup_readiness_preview(preview: dict[str, Any]) -> dict[str, Any]:
    """Return a preview payload safe for persistence in the readiness store."""

    stored = dict(preview)
    config_updates = {
        section: dict(updates)
        for section, updates in (preview.get("config_updates") or {}).items()
        if isinstance(updates, dict)
    }
    for section, key in _secret_config_keys_from_preview(preview):
        section_updates = config_updates.get(section)
        if not section_updates:
            continue
        section_updates.pop(key, None)
        if not section_updates:
            config_updates.pop(section, None)
    stored["config_updates"] = config_updates
    return stored


def _readiness_status_after_provision(preview: dict[str, Any], *, install_plan_submitted: bool) -> str:
    if install_plan_submitted:
        return "provisioning"
    if preview.get("overlays"):
        return "ready_with_warnings"
    return "ready"


def _resolve_setup_readiness_preview(
    payload: SetupReadinessProvisionRequest,
    store: readiness_store.SetupReadinessStore,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Return the preview payload selected for provisioning."""

    if payload.selection:
        selection = _resolve_setup_readiness_profile_selection(
            payload.selection,
            allow_completed_when_disabled=allow_completed_when_disabled,
        )
        preview = preview_readiness_selection(selection, include_secret_config_updates=True)
        preview["preview_id"] = _new_setup_readiness_preview_id()
        return preview

    preview_id = (payload.preview_id or "").strip()
    stored_preview = store.load().get("last_preview")
    if (
        not preview_id
        or not isinstance(stored_preview, dict)
        or stored_preview.get("preview_id") != preview_id
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="A current readiness preview is required before provisioning.",
        )
    preview = dict(stored_preview)
    _raise_if_preview_lost_submitted_secrets(preview)
    return preview


def _resolve_setup_readiness_verification_selection(
    payload: SetupReadinessVerifyRequest,
    store: readiness_store.SetupReadinessStore,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Return an inline or stored readiness selection for explicit verification."""

    if payload.selection:
        return _resolve_setup_readiness_profile_selection(
            payload.selection,
            allow_completed_when_disabled=allow_completed_when_disabled,
        )

    stored_preview = store.load().get("last_preview")
    preview_id = (payload.preview_id or "").strip()
    if isinstance(stored_preview, dict) and (
        not preview_id or stored_preview.get("preview_id") == preview_id
    ):
        return dict(stored_preview)

    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        detail="A readiness selection or current preview is required before verification.",
    )


@router.get("/audio/recommendations", openapi_extra={"security": []}, response_model=AudioRecommendationsResponse)
async def get_audio_recommendations(
    prefer_offline_runtime: bool = True,
    allow_hosted_fallbacks: bool = True,
    _guard: None = Depends(require_local_setup_access),
) -> AudioRecommendationsResponse:
    """Return machine profile information and ranked audio setup bundle recommendations."""

    return _build_audio_recommendations_response(
        prefer_offline_runtime=prefer_offline_runtime,
        allow_hosted_fallbacks=allow_hosted_fallbacks,
    )


@router.get("/readiness/profiles", openapi_extra={"security": []})
async def get_setup_readiness_profiles(
    _guard: None = Depends(require_local_setup_access),
) -> dict[str, Any]:
    """Return first-run setup readiness profiles and current lane summaries."""

    return _build_setup_readiness_profiles_payload()


@router.get("/readiness/status", openapi_extra={"security": []})
async def get_setup_readiness_status(
    _guard: None = Depends(require_local_setup_access),
) -> dict[str, Any]:
    """Return the current first-run setup readiness status snapshot."""

    return await _build_setup_readiness_status_payload()


@router.post(
    "/readiness/preview",
    openapi_extra={"security": []},
    response_model=SetupReadinessPreviewResponse,
)
async def preview_setup_readiness(
    payload: SetupReadinessPreviewRequest,
    _guard: None = Depends(require_local_setup_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> SetupReadinessPreviewResponse:
    """Preview setup readiness changes without writing config or provisioning assets."""

    return await _preview_setup_readiness(payload)


async def _preview_setup_readiness(
    payload: SetupReadinessPreviewRequest,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Shared first-run/admin readiness preview implementation."""

    status_snapshot = setup_manager.get_status_snapshot()
    _ensure_setup_readiness_available(
        status_snapshot,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    selection = _resolve_setup_readiness_profile_selection(
        payload,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    preview = preview_readiness_selection(selection)
    preview["preview_id"] = _new_setup_readiness_preview_id()
    await _save_setup_readiness(
        readiness_store.get_setup_readiness_store(),
        {
            "status": "previewed",
            "selected_profile_id": preview.get("profile_id"),
            "lanes": _preview_lanes_as_list(preview),
            "overlays": preview.get("overlays", []),
            "last_preview": preview,
        }
    )
    return preview


@router.post(
    "/readiness/provision",
    openapi_extra={"security": []},
    response_model=SetupReadinessProvisionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def provision_setup_readiness(
    payload: SetupReadinessProvisionRequest,
    background_tasks: BackgroundTasks,
    _guard: None = Depends(require_local_setup_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> JSONResponse:
    """Persist previewed config changes and queue any selected setup provisioning work."""

    return await _provision_setup_readiness(payload, background_tasks)


async def _provision_setup_readiness(
    payload: SetupReadinessProvisionRequest,
    background_tasks: BackgroundTasks,
    *,
    allow_completed_when_disabled: bool = False,
) -> JSONResponse:
    """Shared first-run/admin readiness provisioning implementation."""

    status_snapshot = setup_manager.get_status_snapshot()
    _ensure_setup_readiness_available(
        status_snapshot,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    if not payload.confirmed:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Provisioning requires explicit confirmation.")

    store = readiness_store.get_setup_readiness_store()
    preview = _resolve_setup_readiness_preview(
        payload,
        store,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    if _preview_has_blockers(preview):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Resolve blocked readiness lanes before provisioning.",
        )

    backup_path = None
    config_updates = preview.get("config_updates") or {}
    if config_updates:
        backup_path = await _update_setup_config(config_updates)

    install_plan = InstallPlan.model_validate(preview.get("install_plan") or {})
    install_plan_submitted = not install_plan.is_empty()
    install_plan_payload = model_dump_compat(install_plan)
    if install_plan_submitted:
        background_tasks.add_task(execute_install_plan, install_plan_payload)

    operation_id = _new_setup_readiness_operation_id()
    operation_status = "queued" if install_plan_submitted else "completed"
    readiness_status = _readiness_status_after_provision(
        preview,
        install_plan_submitted=install_plan_submitted,
    )
    lanes = _provision_lanes_from_preview(preview, install_plan_submitted=install_plan_submitted)
    stored_preview = _stored_setup_readiness_preview(preview)
    provision_payload = {
        "operation_id": operation_id,
        "operation_status": operation_status,
        "status": readiness_status,
        "preview_id": preview.get("preview_id"),
        "install_plan_submitted": install_plan_submitted,
        "config_updates_applied": bool(config_updates),
        "backup_path": str(backup_path) if backup_path else None,
        "install_plan": install_plan_payload if install_plan_submitted else None,
    }
    saved = await _save_setup_readiness(
        store,
        {
            "status": readiness_status,
            "selected_profile_id": preview.get("profile_id"),
            "lanes": lanes,
            "overlays": preview.get("overlays", []),
            "last_preview": stored_preview,
            "last_provision": provision_payload,
            "operation_id": operation_id,
            "operation_status": operation_status,
        }
    )

    status_url = (
        "/api/v1/setup/admin/readiness/status"
        if allow_completed_when_disabled
        else "/api/v1/setup/readiness/status"
    )
    response = {
        "operation_id": operation_id,
        "operation_status": operation_status,
        "status_url": status_url,
        "status": saved["status"],
        "lanes": saved["lanes"],
        "overlays": saved["overlays"],
        "install_plan_submitted": install_plan_submitted,
        "config_updates_applied": bool(config_updates),
        "backup_path": str(backup_path) if backup_path else None,
    }
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response)


@router.post(
    "/readiness/verify",
    openapi_extra={"security": []},
    response_model=SetupReadinessVerifyResponse,
)
async def verify_setup_readiness(
    payload: SetupReadinessVerifyRequest,
    _guard: None = Depends(require_local_setup_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> SetupReadinessVerifyResponse:
    """Explicitly verify selected setup readiness lanes."""

    return await _verify_setup_readiness(payload)


async def _verify_setup_readiness(
    payload: SetupReadinessVerifyRequest,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Shared first-run/admin readiness verification implementation."""

    status_snapshot = setup_manager.get_status_snapshot()
    _ensure_setup_readiness_available(
        status_snapshot,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )

    store = readiness_store.get_setup_readiness_store()
    selection = _resolve_setup_readiness_verification_selection(
        payload,
        store,
        allow_completed_when_disabled=allow_completed_when_disabled,
    )
    verification = _sanitize_setup_payload(await verify_readiness_lanes(selection))
    await _update_setup_readiness(
        store,
        status=verification["status"],
        selected_profile_id=verification.get("profile_id"),
        lanes=list(verification.get("lanes", {}).values()),
        overlays=verification.get("overlays", []),
        last_verification=verification,
    )
    return verification


@router.get("/admin/readiness/profiles")
async def get_admin_setup_readiness_profiles(
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Return admin-gated setup readiness profiles after first-run setup."""

    return _build_setup_readiness_profiles_payload(allow_completed_when_disabled=True)


@router.get("/admin/readiness/status")
async def get_admin_setup_readiness_status(
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Return admin-gated setup readiness status after first-run setup."""

    return await _build_setup_readiness_status_payload(allow_completed_when_disabled=True)


@router.post("/admin/readiness/preview", response_model=SetupReadinessPreviewResponse)
async def preview_admin_setup_readiness(
    payload: SetupReadinessPreviewRequest,
    _guard: None = Depends(require_shared_audio_installer_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> SetupReadinessPreviewResponse:
    """Preview setup readiness changes through the admin setup surface."""

    return await _preview_setup_readiness(payload, allow_completed_when_disabled=True)


@router.post(
    "/admin/readiness/provision",
    response_model=SetupReadinessProvisionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def provision_admin_setup_readiness(
    payload: SetupReadinessProvisionRequest,
    background_tasks: BackgroundTasks,
    _guard: None = Depends(require_shared_audio_installer_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> JSONResponse:
    """Provision setup readiness changes through the admin setup surface."""

    return await _provision_setup_readiness(
        payload,
        background_tasks,
        allow_completed_when_disabled=True,
    )


@router.post("/admin/readiness/verify", response_model=SetupReadinessVerifyResponse)
async def verify_admin_setup_readiness(
    payload: SetupReadinessVerifyRequest,
    _guard: None = Depends(require_shared_audio_installer_access),
    _rate_limit: None = Depends(check_rate_limit),
) -> SetupReadinessVerifyResponse:
    """Verify setup readiness lanes through the admin setup surface."""

    return await _verify_setup_readiness(payload, allow_completed_when_disabled=True)


def _build_audio_recommendations_response(
    *,
    prefer_offline_runtime: bool,
    allow_hosted_fallbacks: bool,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Build the shared audio recommendations payload."""
    _ensure_audio_installer_available(allow_completed_when_disabled=allow_completed_when_disabled)

    machine_profile = audio_profile_service.detect_machine_profile()
    recommendations = audio_profile_service.recommend_audio_bundles(
        machine_profile,
        prefer_offline_runtime=prefer_offline_runtime,
        allow_hosted_fallbacks=allow_hosted_fallbacks,
    )
    catalog = get_audio_bundle_catalog()
    bundle_lookup = {
        bundle.bundle_id: bundle.model_dump() if hasattr(bundle, "model_dump") else dict(bundle)
        for bundle in catalog.bundles
    }
    for recommendation in recommendations.get("recommendations", []):
        bundle_id = recommendation.get("bundle_id")
        if bundle_id in bundle_lookup:
            recommendation["bundle"] = bundle_lookup[bundle_id]
            resource_profile = recommendation.get("resource_profile")
            if resource_profile:
                recommendation["profile"] = bundle_lookup[bundle_id].get("resource_profiles", {}).get(resource_profile)
    for excluded_bundle in recommendations.get("excluded", []):
        bundle_id = excluded_bundle.get("bundle_id")
        if bundle_id in bundle_lookup:
            excluded_bundle["bundle"] = bundle_lookup[bundle_id]

    return {
        "machine_profile": (
            machine_profile.model_dump()
            if hasattr(machine_profile, "model_dump")
            else dict(machine_profile)
        ),
        "catalog": list(bundle_lookup.values()),
        **recommendations,
    }


@router.get("/audio/readiness", openapi_extra={"security": []}, response_model=audio_readiness_store.AudioReadinessRecord)
async def get_audio_readiness(
    _guard: None = Depends(require_local_setup_access),
) -> audio_readiness_store.AudioReadinessRecord:
    """Return the persisted setup audio readiness snapshot."""

    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    return audio_readiness_store.get_audio_readiness_store().load()


@router.post("/audio/readiness/reset", openapi_extra={"security": []}, response_model=AudioReadinessResetResponse)
async def reset_audio_readiness(
    _guard: None = Depends(require_local_setup_access),
) -> AudioReadinessResetResponse:
    """Reset the persisted setup audio readiness snapshot."""

    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    readiness = audio_readiness_store.get_audio_readiness_store().reset()
    return {
        "success": True,
        "audio_readiness": readiness,
    }


@router.post("/audio/provision", openapi_extra={"security": []}, response_model=AudioBundleOperationResponse)
async def provision_audio_bundle(
    payload: AudioBundleProvisionRequest,
    _guard: None = Depends(require_local_setup_access),
) -> AudioBundleOperationResponse:
    """Expand and provision a curated audio bundle."""

    return await _execute_audio_bundle_provision(payload)


async def _execute_audio_bundle_provision(
    payload: AudioBundleProvisionRequest,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Execute the bundle provisioning flow shared by legacy and admin routes."""
    _ensure_audio_installer_available(allow_completed_when_disabled=allow_completed_when_disabled)

    try:
        return await asyncio.to_thread(
            install_manager.execute_audio_bundle,
            payload.bundle_id,
            resource_profile=payload.resource_profile,
            tts_choice=payload.tts_choice,
            safe_rerun=payload.safe_rerun,
        )
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=INVALID_AUDIO_BUNDLE_REQUEST_DETAIL,
        ) from None
    except KeyError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=AUDIO_BUNDLE_NOT_FOUND_DETAIL,
        ) from None


@router.post("/audio/verify", openapi_extra={"security": []}, response_model=AudioBundleOperationResponse)
async def verify_audio_bundle(
    payload: AudioBundleVerificationRequest,
    _guard: None = Depends(require_local_setup_access),
) -> AudioBundleOperationResponse:
    """Verify the primary STT/TTS paths for a curated audio bundle."""

    result = await _execute_audio_bundle_verification(payload)
    return _sanitize_setup_payload(result)


async def _execute_audio_bundle_verification(
    payload: AudioBundleVerificationRequest,
    *,
    allow_completed_when_disabled: bool = False,
) -> dict[str, Any]:
    """Execute bundle verification shared by legacy and admin routes."""
    _ensure_audio_installer_available(allow_completed_when_disabled=allow_completed_when_disabled)

    try:
        return await install_manager.verify_audio_bundle_async(
            payload.bundle_id,
            resource_profile=payload.resource_profile,
            tts_choice=payload.tts_choice,
        )
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=INVALID_AUDIO_BUNDLE_REQUEST_DETAIL,
        ) from None
    except KeyError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=AUDIO_BUNDLE_NOT_FOUND_DETAIL,
        ) from None


@router.get("/admin/install-status")
async def get_admin_install_status(
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Return installer status for the shared admin audio installer."""

    return _get_audio_install_status(allow_completed_when_disabled=True)


@router.get("/admin/audio/recommendations")
async def get_admin_audio_recommendations(
    prefer_offline_runtime: bool = True,
    allow_hosted_fallbacks: bool = True,
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Return admin-gated audio bundle recommendations for the shared installer UI."""

    return _build_audio_recommendations_response(
        prefer_offline_runtime=prefer_offline_runtime,
        allow_hosted_fallbacks=allow_hosted_fallbacks,
        allow_completed_when_disabled=True,
    )


@router.post("/admin/audio/provision")
async def provision_admin_audio_bundle(
    payload: AudioBundleProvisionRequest,
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Provision a curated audio bundle through the shared admin installer UI."""

    return await _execute_audio_bundle_provision(payload, allow_completed_when_disabled=True)


@router.post("/admin/audio/verify")
async def verify_admin_audio_bundle(
    payload: AudioBundleVerificationRequest,
    _guard: None = Depends(require_shared_audio_installer_access),
) -> dict[str, Any]:
    """Verify a curated audio bundle through the shared admin installer UI."""

    result = await _execute_audio_bundle_verification(
        payload,
        allow_completed_when_disabled=True,
    )
    return _sanitize_setup_payload(result)


@router.post("/audio/packs/export", openapi_extra={"security": []}, response_model=AudioPackExportResponse)
async def export_audio_pack(
    payload: AudioPackExportRequest,
    _guard: None = Depends(require_local_setup_access),
) -> AudioPackExportResponse:
    """Export a v1 audio bundle pack manifest for the selected bundle/profile."""

    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    pack_name = _normalize_audio_pack_name(payload.pack_name) if payload.pack_name else None
    readiness = audio_readiness_store.get_audio_readiness_store().load()
    machine_profile = audio_profile_service.detect_machine_profile()
    compatibility = _audio_pack_compatibility(machine_profile)

    try:
        if pack_name:
            manifest = audio_pack_service.write_audio_pack_manifest(
                pack_name=pack_name,
                bundle_id=payload.bundle_id,
                resource_profile=payload.resource_profile,
                tts_choice=payload.tts_choice,
                installed_assets=readiness.get("installed_asset_manifests"),
                compatibility=compatibility,
            )
        else:
            manifest = audio_pack_service.build_audio_pack_manifest(
                bundle_id=payload.bundle_id,
                resource_profile=payload.resource_profile,
                tts_choice=payload.tts_choice,
                installed_assets=readiness.get("installed_asset_manifests"),
                compatibility=compatibility,
            )
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=INVALID_AUDIO_PACK_EXPORT_REQUEST_DETAIL,
        ) from None
    except KeyError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=AUDIO_BUNDLE_NOT_FOUND_DETAIL,
        ) from None

    return {
        "success": True,
        "manifest": manifest,
        "pack_path": str(Path(audio_pack_service.AUDIO_PACKS_DIRNAME) / pack_name) if pack_name else None,
    }


@router.post("/audio/packs/import", openapi_extra={"security": []}, response_model=AudioPackImportResponse)
async def import_audio_pack(
    payload: AudioPackImportRequest,
    _guard: None = Depends(require_local_setup_access),
) -> AudioPackImportResponse:
    """Validate and register a v1 audio bundle pack manifest."""

    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    pack_name = _normalize_audio_pack_name(payload.pack_name)
    machine_profile = audio_profile_service.detect_machine_profile()
    compatibility = _audio_pack_compatibility(machine_profile)
    machine_profile_payload = (
        machine_profile.model_dump() if hasattr(machine_profile, "model_dump") else dict(machine_profile)
    )
    readiness_store = audio_readiness_store.get_audio_readiness_store()

    try:
        result = audio_pack_service.register_imported_audio_pack(
            pack_name,
            readiness_store=readiness_store,
            machine_profile=machine_profile_payload,
            python_version=compatibility["python_version"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Audio pack not found.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Audio pack manifest is not valid JSON.") from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return result


@router.post("/config", openapi_extra={"security": []}, response_model=SetupConfigUpdateResponse)
async def update_setup_config(
    payload: ConfigUpdates,
    _guard: None = Depends(require_local_setup_access),
) -> SetupConfigUpdateResponse:
    """Persist configuration updates coming from the setup UI."""
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    if not status_snapshot["needs_setup"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Setup already completed. Toggle enable_first_time_setup to make changes here.",
        )

    try:
        backup_path = setup_manager.update_config(payload.updates)
        return {
            "success": True,
            "backup_path": str(backup_path) if backup_path else None,
            "requires_restart": True,
        }
    except ValueError as exc:
        logger.exception("Setup config validation failed via setup endpoint")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to write configuration via setup endpoint")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist setup configuration.",
        ) from exc


@router.post("/complete", openapi_extra={"security": []}, response_model=SetupCompleteResponse)
async def mark_setup_complete(
    payload: SetupCompleteRequest,
    background_tasks: BackgroundTasks,
    _guard: None = Depends(require_local_setup_access),
) -> SetupCompleteResponse:
    """Mark the setup workflow as complete and optionally disable future prompts."""
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["enabled"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Setup flow not enabled in config.txt")

    if not status_snapshot["needs_setup"]:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Setup already marked as complete")

    setup_manager.mark_setup_completed(True)

    plan_requested = False
    if payload.install_plan and not payload.install_plan.is_empty():
        plan_requested = True
        plan_dict = model_dump_compat(payload.install_plan)
        background_tasks.add_task(execute_install_plan, plan_dict)

    if payload.disable_first_time_setup:
        setup_manager.update_config({setup_manager.SETUP_SECTION: {"enable_first_time_setup": False}}, create_backup=False)

    return {
        "success": True,
        "message": "Setup marked as complete. Restart the server to load new configuration.",
        "requires_restart": True,
        "install_plan_submitted": plan_requested,
    }


@router.post("/assistant", openapi_extra={"security": []}, response_model=SetupAssistantResponse)
async def ask_setup_assistant(
    payload: AssistantQuestion,
    _guard: None = Depends(require_local_setup_access),
) -> SetupAssistantResponse:
    """Provide contextual help for setup questions using local configuration knowledge."""
    try:
        return setup_manager.answer_setup_question(payload.question)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(

    "/reset",
    summary="Reset first-time setup flags (admin)",
    description=(
        "Admin-only recovery endpoint to re-enable the guided setup flow by setting "
        "enable_first_time_setup=true and setup_completed=false. Requires server restart."
    ),
    response_model=SetupResetResponse,
)
async def reset_setup_flags(
    _principal: AuthPrincipal = Depends(require_admin_and_system_configure),  # noqa: B008
) -> SetupResetResponse:
    """Admin-only: reset first-time setup flags for recovery.

    Sets `enable_first_time_setup = true` and `setup_completed = false` in config.txt.
    """
    try:
        setup_manager.reset_setup_flags()
    except Exception as exc:
        logger.exception("Failed to reset setup flags")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset setup flags.",
        ) from exc

    return {
        "success": True,
        "message": "Setup flags reset. Restart the server and revisit /setup.",
        "requires_restart": True,
    }


@router.post(

    "/self-verify",
    summary="Mark current user as verified (initial setup)",
    description=(
        "Local-only helper to mark the authenticated user as verified during initial setup. "
        "Requires that the setup wizard is still enabled and not completed. Accepts either "
        "Bearer JWT (Authorization header) or X-API-KEY for multi-user SQLite setups."
    ),
)
async def setup_self_verify(
    principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
    db=Depends(get_db_transaction),
    _guard: None = Depends(require_local_setup_access),
) -> dict[str, Any]:
    """Mark the authenticated account as verified when setup is in progress."""
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot["needs_setup"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="Self-verify is only available while initial setup is in progress.",
        )

    raw_id = principal.user_id
    try:
        user_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid user context",
        ) from exc
    if user_id <= 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid user context")

    try:
        await mark_user_verified(
            db,
            user_id=user_id,
            now_utc=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to self-verify during setup")
        # Avoid leaking raw DB/driver errors to clients; keep detail generic.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark account as verified; please try again.",
        ) from exc

    return {"success": True, "user_id": user_id, "message": "Account marked as verified."}
