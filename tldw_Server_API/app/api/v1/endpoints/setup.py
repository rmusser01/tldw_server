"""Setup endpoints for the first-time configuration flow."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    require_permissions,
    require_roles,
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
    SetupAssistantResponse,
    SetupCompleteResponse,
    SetupConfigUpdateResponse,
    SetupInstallStatusResponse,
    SetupResetResponse,
    SetupStatusResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Setup import install_manager, setup_manager
from tldw_Server_API.app.core.Setup import audio_pack_service
from tldw_Server_API.app.core.Setup import audio_profile_service
from tldw_Server_API.app.core.Setup import audio_readiness_store
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    DEFAULT_AUDIO_RESOURCE_PROFILE,
    get_audio_bundle_catalog,
)
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan
from tldw_Server_API.app.core.Setup.install_manager import execute_install_plan
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat
from tldw_Server_API.app.services.auth_service import mark_user_verified

router = APIRouter(prefix="/setup", tags=["setup"], include_in_schema=True)

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


class AudioPackImportRequest(BaseModel):
    pack_path: str = Field(..., min_length=1, description="Filesystem path to an audio pack manifest JSON file.")


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
    role_checker = require_roles("admin")
    perm_checker = require_permissions(SYSTEM_CONFIGURE)

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
    raise HTTPException(status.HTTP_404_NOT_FOUND, detail=_AUDIO_BUNDLE_LOOKUP_DETAIL) from exc


@router.get("/status", openapi_extra={"security": []}, response_model=SetupStatusResponse)
async def get_setup_status(_guard: None = Depends(require_local_setup_access)) -> SetupStatusResponse:
    """Return setup availability and placeholder diagnostics."""
    return setup_manager.get_status_snapshot()


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
