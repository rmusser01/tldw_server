"""Schemas for setup and first-run workflow endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.core.Setup.audio_readiness_store import AudioReadinessRecord
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import DEFAULT_AUDIO_RESOURCE_PROFILE


def _legacy_pack_name(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


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


class AudioBundleVerificationRequest(BaseModel):
    bundle_id: str = Field(..., min_length=1, description="Curated audio bundle identifier to verify.")
    resource_profile: str = Field(
        DEFAULT_AUDIO_RESOURCE_PROFILE,
        min_length=1,
        description="Selected resource profile within the curated audio bundle.",
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
        description="Optional JSON filename to write under the setup-managed audio_packs directory.",
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


class SetupPlaceholderField(BaseModel):
    section: str
    key: str
    value: str


class SetupStatusResponse(BaseModel):
    enabled: bool
    setup_completed: bool
    needs_setup: bool
    config_path: str
    allow_remote_setup_access: bool
    remote_access_env_override: bool
    remote_access_active: bool
    placeholder_fields: list[SetupPlaceholderField] = Field(default_factory=list)


class SetupInstallStep(BaseModel):
    name: str
    status: str
    detail: str | None = None
    timestamp: str | None = None


class SetupInstallStatusResponse(BaseModel):
    status: str
    plan: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    steps: list[SetupInstallStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AudioRecommendationsResponse(BaseModel):
    machine_profile: dict[str, Any]
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    excluded: list[dict[str, Any]] = Field(default_factory=list)


class AudioReadinessResetResponse(BaseModel):
    success: bool
    audio_readiness: AudioReadinessRecord


class AudioBundleOperationResponse(BaseModel):
    bundle_id: str
    status: str
    resource_profile: str | None = None
    selected_resource_profile: str | None = None
    selection_key: str | None = None
    safe_rerun: bool | None = None
    install_plan: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    machine_profile: dict[str, Any] | None = None
    stt_health: dict[str, Any] | None = None
    tts_health: dict[str, Any] | None = None
    remediation_items: list[Any] = Field(default_factory=list)
    verified_at: str | None = None


class AudioPackExportResponse(BaseModel):
    success: bool
    manifest: dict[str, Any]
    pack_path: str | None = None


class AudioPackImportResponse(BaseModel):
    compatible: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    manifest: dict[str, Any]
    selection_key: str
    bundle_label: str | None = None
    audio_readiness: AudioReadinessRecord


class SetupConfigUpdateResponse(BaseModel):
    success: bool
    backup_path: str | None = None
    requires_restart: bool


class SetupCompleteResponse(BaseModel):
    success: bool
    message: str
    requires_restart: bool
    install_plan_submitted: bool


class SetupAssistantResponse(BaseModel):
    answer: str
    matches: list[dict[str, Any]] = Field(default_factory=list)


class SetupResetResponse(BaseModel):
    success: bool
    message: str
    requires_restart: bool
