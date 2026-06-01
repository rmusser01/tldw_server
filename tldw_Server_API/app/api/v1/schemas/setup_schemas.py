"""Schemas for setup and first-run workflow endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.core.Setup import first_run_models
from tldw_Server_API.app.core.Setup.audio_bundle_catalog import DEFAULT_AUDIO_RESOURCE_PROFILE
from tldw_Server_API.app.core.Setup.audio_readiness_store import AudioReadinessRecord
from tldw_Server_API.app.core.Setup.first_chat_verifier import DEFAULT_FIRST_CHAT_PROMPT
from tldw_Server_API.app.core.Setup.install_schema import InstallPlan

FirstRunChatResult = first_run_models.FirstRunChatResult
FirstRunStateResponse = first_run_models.FirstRunStateResponse
FirstRunStatus = first_run_models.FirstRunStatus
FirstRunStepStatus = first_run_models.FirstRunStepStatus


def _legacy_pack_name(path_value: str) -> str:
    normalized = str(path_value).replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


class ConfigUpdates(BaseModel):
    """Config section updates submitted through setup recovery surfaces."""

    updates: dict[str, dict[str, Any]] = Field(
        ..., description="Mapping of section -> key/value pairs to persist in config.txt"
    )


class SetupReadinessPreviewRequest(BaseModel):
    """Readiness lane selection to preview without applying config changes."""

    profile_id: str | None = Field(
        None,
        description="Curated readiness profile identifier or advanced custom selection.",
    )
    lanes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Lane-specific chat, embeddings/RAG, and speech setup selections.",
    )


class SetupReadinessSecretField(BaseModel):
    """Secret config field status included in readiness previews."""

    section: str
    key: str
    provider: str | None = None
    state: str


class SetupReadinessPreviewResponse(BaseModel):
    """Preview result for setup readiness lanes and planned operations."""

    preview_id: str | None = None
    profile_id: str | None = None
    lane_ids: list[str] = Field(default_factory=list)
    lanes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    overlays: list[str] = Field(default_factory=list)
    config_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    secret_fields: list[SetupReadinessSecretField] = Field(default_factory=list)
    install_plan: dict[str, Any] = Field(default_factory=dict)
    operation_required: bool


class SetupReadinessProvisionRequest(BaseModel):
    """Confirmed readiness selection to persist and optionally provision."""

    preview_id: str | None = Field(
        None,
        description="Identifier returned by /setup/readiness/preview.",
    )
    selection: dict[str, Any] | None = Field(
        None,
        description="Optional inline selection to preview and provision in one request.",
    )
    confirmed: bool = Field(
        False,
        description="Must be true to persist config updates or queue provisioning work.",
    )


class SetupReadinessProvisionResponse(BaseModel):
    """Provisioning operation status returned after readiness changes are queued."""

    operation_id: str
    operation_status: str
    status_url: str
    status: str
    lanes: list[dict[str, Any]] = Field(default_factory=list)
    overlays: list[str] = Field(default_factory=list)
    install_plan_submitted: bool
    config_updates_applied: bool
    backup_path: str | None = None


class SetupReadinessVerifyRequest(BaseModel):
    """Readiness selection or preview identifier to verify explicitly."""

    preview_id: str | None = Field(
        None,
        description="Optional preview identifier to verify.",
    )
    selection: dict[str, Any] | None = Field(
        None,
        description="Optional inline readiness selection to verify.",
    )


class SetupReadinessVerifyResponse(BaseModel):
    """Verification result for selected setup readiness lanes."""

    profile_id: str | None = None
    lane_ids: list[str] = Field(default_factory=list)
    lanes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    overlays: list[str] = Field(default_factory=list)
    status: str
    verified_at: str


class SetupProviderType(str, Enum):
    """Provider setup categories supported by the first-run wizard."""

    HOSTED_API_KEY = "hosted_api_key"
    LOCAL_ENDPOINT = "local_endpoint"


class SetupProviderSaveStatus(str, Enum):
    """Persistence status for a setup provider save request."""

    SAVED = "saved"
    FAILED = "failed"


class SetupProviderCatalogEntry(BaseModel):
    """One provider option exposed to first-run setup."""

    provider_key: str
    label: str
    provider_type: SetupProviderType
    config_section: str
    api_key_field: str | None = None
    base_url_field: str | None = None
    model_field: str | None = None
    default_base_url: str | None = None
    supports_preflight: bool = False
    recommended_for_first_chat: bool = False


class SetupProviderCatalogResponse(BaseModel):
    """Catalog of provider options for the first-run setup UI."""

    providers: list[SetupProviderCatalogEntry] = Field(default_factory=list)


class SetupProviderSaveRequest(BaseModel):
    """Provider settings submitted from first-run setup."""

    provider_key: str
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    make_default: bool = False


class SetupProviderSaveResponse(BaseModel):
    """Sanitized provider settings persistence result."""

    provider_key: str
    status: SetupProviderSaveStatus
    masked_api_key: str | None = None
    credential_configured: bool = False
    base_url: str | None = None
    model: str | None = None
    make_default: bool = False
    requires_restart: bool = False
    failure_category: str | None = None
    message: str | None = None


class SetupProviderValidationResponse(BaseModel):
    """Provider validation result safe to return to unauthenticated setup clients."""

    provider_key: str
    status: str
    failure_category: str | None = None
    message: str | None = None
    models: list[str] = Field(default_factory=list)
    validation_level: str | None = None
    can_gate_first_chat: bool = False


class SetupCompleteRequest(BaseModel):
    """Request to finish setup and optionally queue provisioning work."""

    disable_first_time_setup: bool | None = Field(
        False,
        description="If true, flips enable_first_time_setup to false so the screen stays hidden",
    )
    install_plan: InstallPlan | None = Field(
        None,
        description="Backend installation instructions to execute after setup completes.",
    )


class AssistantQuestion(BaseModel):
    """Natural-language setup assistant question."""

    question: str = Field(..., min_length=1, description="Natural language question for the setup assistant")


class FirstRunStepUpdateRequest(BaseModel):
    """Generic first-run step progress update."""

    step: str = Field(..., min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class FirstRunSkipRequest(BaseModel):
    """Request to skip the focused first-run wizard."""

    reason: str | None = Field(None, max_length=120)


class FirstChatVerifyRequest(BaseModel):
    """First chat verification request used as the onboarding completion gate."""

    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    prompt: str = Field(DEFAULT_FIRST_CHAT_PROMPT, min_length=1, max_length=1000)


class FirstChatVerifyResponse(BaseModel):
    """Sanitized result of a first chat verification attempt."""

    status: str
    provider: str
    model: str
    response_id: str | None = None
    response_text: str | None = None
    failure_category: str | None = None
    message: str | None = None


class FirstRunCompleteRequest(BaseModel):
    """First-run completion request with any final acknowledged steps."""

    acknowledged_steps: list[str] = Field(default_factory=list)


class IngestDefaultsRequest(BaseModel):
    """Default ingest choices captured during first-run setup."""

    allow_local_file_ingest: bool = False
    chunking_profile: str = Field("balanced", min_length=1)
    metadata_mode: str = Field("automatic", min_length=1)
    allowed_local_roots: list[str] = Field(default_factory=list)


class AudioDefaultsRequest(BaseModel):
    """Audio, STT, and TTS defaults captured during first-run setup."""

    mode: str = Field("skip", pattern="^(defaults|configure|skip)$")
    stt_provider: str | None = None
    tts_provider: str | None = None
    tts_voice: str | None = None


class OptionalAdvancedRequest(BaseModel):
    """Optional RAG and storage choices that are non-blocking for first use."""

    rag: str = Field("defer", pattern="^(configure|skip|defer)$")
    storage_paths: str = Field("defer", pattern="^(configure|skip|defer)$")
    values: dict[str, Any] = Field(default_factory=dict)


class FirstRunStepSaveResponse(BaseModel):
    """Acknowledgement returned after a first-run step is saved."""

    status: str
    step: str
    requires_restart: bool = False


class FirstRunSetupPath(BaseModel):
    """Setup path choice displayed in first-run metadata."""

    key: str
    label: str
    recommended: bool = False
    guide_path: str | None = None


class FirstRunMultiUserExit(BaseModel):
    """Documentation links shown when the user chooses multi-user setup."""

    guide_path: str
    checklist_path: str | None = None


class FirstRunConnectionDiagnostics(BaseModel):
    """Connection metadata used to decide local setup assistance."""

    frontend_origin: str | None = None
    api_origin: str | None = None
    browser_access: str | None = None


class FirstRunMetadataResponse(BaseModel):
    """First-run metadata used to shape the solo-user setup flow."""

    auth_mode: str
    bundled_single_user_auth_available: bool
    manual_auth_required: bool
    setup_required: bool
    setup_completed: bool
    remote_setup_enabled: bool
    connection: FirstRunConnectionDiagnostics
    setup_paths: list[FirstRunSetupPath]
    multi_user_exit: FirstRunMultiUserExit


class AudioBundleProvisionRequest(BaseModel):
    """Curated audio bundle provisioning request."""

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
    """Curated audio bundle verification request."""

    bundle_id: str = Field(..., min_length=1, description="Curated audio bundle identifier to verify.")
    resource_profile: str = Field(
        DEFAULT_AUDIO_RESOURCE_PROFILE,
        min_length=1,
        description="Selected resource profile within the curated audio bundle.",
    )


class AudioPackExportRequest(BaseModel):
    """Request to export a portable audio setup pack manifest."""

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
    """Request to import a setup-managed audio pack manifest by name."""

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
    """Placeholder config field that still needs a user-provided value."""

    section: str
    key: str
    value: str


class SetupStatusResponse(BaseModel):
    """Current setup availability and placeholder status."""

    enabled: bool
    setup_completed: bool
    needs_setup: bool
    config_path: str
    allow_remote_setup_access: bool
    remote_access_env_override: bool
    remote_access_active: bool
    placeholder_fields: list[SetupPlaceholderField] = Field(default_factory=list)


class SetupInstallStep(BaseModel):
    """One setup installation step status entry."""

    name: str
    status: str
    detail: str | None = None
    timestamp: str | None = None


class SetupInstallStatusResponse(BaseModel):
    """Current setup installation plan status."""

    status: str
    plan: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    steps: list[SetupInstallStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AudioRecommendationsResponse(BaseModel):
    """Machine profile and recommended audio setup bundles."""

    machine_profile: dict[str, Any]
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    excluded: list[dict[str, Any]] = Field(default_factory=list)


class AudioReadinessResetResponse(BaseModel):
    """Response returned after resetting persisted audio readiness."""

    success: bool
    audio_readiness: AudioReadinessRecord


class AudioBundleOperationResponse(BaseModel):
    """Result of provisioning or verifying a curated audio bundle."""

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
    """Portable audio pack manifest export response."""

    success: bool
    manifest: dict[str, Any]
    pack_path: str | None = None


class AudioPackImportResponse(BaseModel):
    """Audio pack import compatibility and readiness result."""

    compatible: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    manifest: dict[str, Any]
    selection_key: str
    bundle_label: str | None = None
    audio_readiness: AudioReadinessRecord


class SetupConfigUpdateResponse(BaseModel):
    """Result of writing setup configuration changes."""

    success: bool
    backup_path: str | None = None
    requires_restart: bool


class SetupCompleteResponse(BaseModel):
    """Result returned after setup completion is persisted."""

    success: bool
    message: str
    requires_restart: bool
    install_plan_submitted: bool


class SetupAssistantResponse(BaseModel):
    """Answer and source matches returned by the setup assistant."""

    answer: str
    matches: list[dict[str, Any]] = Field(default_factory=list)


class SetupResetResponse(BaseModel):
    """Admin reset result for first-time setup flags."""

    success: bool
    message: str
    requires_restart: bool
