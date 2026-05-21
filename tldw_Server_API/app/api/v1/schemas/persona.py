"""
Pydantic schemas for Persona Agent API.

Scaffold only - minimal models to enable endpoint stubs.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    PersonaVisualRendererSetupStatus,
)
from tldw_Server_API.app.core.Persona.visual_starter_recipe_taxonomy import (
    BUDDY_VISUAL_ANIMATION_OUTPUT_IDS,
    BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS,
)


PersonaMode = Literal["session_scoped", "persistent_scoped"]
PersonaScopeRuleType = Literal["conversation_id", "character_id", "media_id", "media_tag", "note_id"]
PersonaPolicyRuleKind = Literal["mcp_tool", "skill"]
PersonaSessionStatus = Literal["active", "paused", "closed", "archived"]
PersonaLiveLifecycle = Literal["idle", "connecting", "connected", "recovering", "stopping", "stopped", "error"]
PersonaLiveReusePolicy = Literal["resume_compatible", "create_new"]
PersonaExemplarKind = Literal["style", "catchphrase", "boundary", "scenario_demo", "tool_behavior"]
PersonaExemplarSourceType = Literal["manual", "transcript_import", "character_seed", "generated_candidate"]
PersonaExemplarReviewAction = Literal["approve", "reject"]
PersonaConfirmationMode = Literal["always", "destructive_only", "never"]
PersonaWakeBehavior = Literal["one_shot", "continuous", "push_to_talk_after_wake"]
PersonaSetupStatus = Literal["not_started", "in_progress", "completed"]
PersonaSetupStep = Literal["archetype", "persona", "voice", "commands", "safety", "test"]
PersonaSetupTestType = Literal["dry_run", "live_session"]
PersonaVisualPackStatus = Literal["draft", "review", "active", "archived", "failed"]
PersonaVisualRendererType = Literal["sprite_frames", "sprite_sheet", "static_image", "live2d"]
PersonaVisualAssetRole = Literal["frame", "still_pose", "sprite_sheet", "preview", "generated_candidate"]
PersonaVisualStarterComplexityTier = Literal["basic", "intermediate", "intricate"]
PersonaVisualStarterProductionStatus = Literal["scaffold", "art_ready"]
PersonaVisualCandidateStatus = Literal["review", "accepted", "rejected", "failed"]
PersonaVisualCandidateReviewStatus = Literal["accepted", "rejected", "failed"]
PersonaVisualStarterRecipeText = Annotated[str, Field(min_length=1, max_length=320)]
PersonaVisualStarterRecipeItems = Annotated[
    list[PersonaVisualStarterRecipeText],
    Field(min_length=1, max_length=12),
]
PersonaVisualPortabilityOperation = Literal["export", "import_preview", "import_commit"]
PersonaSetupEventType = Literal[
    "setup_started",
    "step_viewed",
    "step_completed",
    "step_error",
    "retry_clicked",
    "detour_started",
    "detour_returned",
    "setup_completed",
    "handoff_action_clicked",
    "handoff_target_reached",
    "handoff_dismissed",
    "first_post_setup_action",
    # First-run assistant setup extensions
    "archetype_selected",
    "archetype_changed",
    "external_server_connected",
    "external_server_failed",
    "connection_test_initiated",
    "setup_skipped",
    "setup_resumed",
]


def _normalize_persona_visual_library_tags(value: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in value:
        tag = str(raw_tag or "").strip().lower()
        if not tag:
            raise ValueError("tags cannot contain empty values")
        if len(tag) > 64:
            raise ValueError("tags must be 64 characters or fewer")
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) > 20:
            raise ValueError("at most 20 tags are allowed")
    return normalized


class PersonaVisualPackCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    manifest: dict[str, Any] = Field(default_factory=dict)


class PersonaVisualPackDuplicateRequest(BaseModel):
    target_persona_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("target_persona_id")
    @classmethod
    def normalize_target_persona_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("target_persona_id is required")
        return normalized

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class PersonaVisualStarterPackCopyRequest(BaseModel):
    target_persona_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("target_persona_id")
    @classmethod
    def normalize_target_persona_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("target_persona_id is required")
        return normalized

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class PersonaVisualLibrarySaveRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("title cannot be empty")
        return normalized

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _normalize_persona_visual_library_tags(value)


class PersonaVisualLibraryUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = None
    expected_version: int | None = Field(default=None, ge=1)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("title cannot be empty")
        return normalized

    @field_validator("notes")
    @classmethod
    def normalize_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _normalize_persona_visual_library_tags(value)


class PersonaVisualLibraryUseRequest(BaseModel):
    target_persona_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("target_persona_id")
    @classmethod
    def normalize_target_persona_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("target_persona_id is required")
        return normalized

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("title cannot be empty")
        return normalized


class PersonaVisualManifestUpdate(BaseModel):
    manifest: dict[str, Any] = Field(default_factory=dict)
    expected_version: int | None = Field(default=None, ge=1)


class PersonaVisualRendererCapabilityResponse(BaseModel):
    renderer_type: PersonaVisualRendererType
    display_name: str
    manifest_versions: list[int] = Field(default_factory=list)
    can_validate: bool
    can_activate: bool
    buddy_runtime_supported: bool
    import_supported: bool
    export_supported: bool
    disabled_reason: str | None = None
    renderer_contract_versions: list[int] = Field(default_factory=list)
    supported_asset_roles: list[str] = Field(default_factory=list)
    required_role_categories: list[str] = Field(default_factory=list)
    role_category_map: dict[str, list[str]] = Field(default_factory=dict)
    allowed_mime_types: list[str] = Field(default_factory=list)
    allowed_extensions: list[str] = Field(default_factory=list)
    max_file_count: int | None = None
    max_total_bytes: int | None = None
    max_texture_width: int | None = None
    max_texture_height: int | None = None
    feature_flag: str | None = None
    setup_status: PersonaVisualRendererSetupStatus = "supported"
    setup_blockers: list[str] = Field(default_factory=list)
    requires_static_fallback: bool = False
    requires_license_ack: bool = False


class PersonaVisualRendererCapabilitiesResponse(BaseModel):
    renderers: list[PersonaVisualRendererCapabilityResponse] = Field(default_factory=list)


class PersonaVisualAssetResponse(BaseModel):
    id: str
    pack_id: str
    persona_id: str
    asset_role: PersonaVisualAssetRole
    storage_key: str
    url: str
    original_filename: str | None = None
    mime_type: str
    byte_size: int
    checksum_sha256: str
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    provenance: str = "uploaded"
    created_at: str
    last_modified: str
    version: int = 1


class PersonaVisualPackResponse(BaseModel):
    id: str
    persona_id: str
    user_id: str
    title: str
    renderer_type: PersonaVisualRendererType
    status: PersonaVisualPackStatus
    manifest_version: int = 1
    manifest: dict[str, Any] = Field(default_factory=dict)
    parent_pack_id: str | None = None
    revision_number: int = 1
    provenance: str = "uploaded"
    active_at: str | None = None
    assets: list[PersonaVisualAssetResponse] = Field(default_factory=list)
    created_at: str
    last_modified: str
    version: int = 1


class PersonaVisualStarterAssetResponse(BaseModel):
    asset_key: str
    filename: str
    mime_type: str
    asset_role: PersonaVisualAssetRole
    byte_size: int


class PersonaVisualStarterProductionRecipeResponse(BaseModel):
    identity_brief: PersonaVisualStarterRecipeText
    neutral_anchor: PersonaVisualStarterRecipeText
    static_sheet: PersonaVisualStarterRecipeText
    animation_outputs: PersonaVisualStarterRecipeItems
    review_checks: PersonaVisualStarterRecipeItems

    @field_validator("animation_outputs")
    @classmethod
    def validate_animation_outputs(cls, value: list[str]) -> list[str]:
        invalid_outputs = [
            output for output in value if output not in BUDDY_VISUAL_ANIMATION_OUTPUT_IDS
        ]
        if invalid_outputs:
            invalid_output_list = ", ".join(sorted(set(invalid_outputs)))
            raise ValueError(
                "animation_outputs must use supported animation output ids. "
                f"Invalid: {invalid_output_list}"
            )
        return value

    @field_validator("review_checks")
    @classmethod
    def validate_review_checks(cls, value: list[str]) -> list[str]:
        if "neutral_identity_consistency" not in value:
            raise ValueError("review_checks must include neutral_identity_consistency")
        return value


class PersonaVisualStarterPackResponse(BaseModel):
    id: str
    title: str
    description: str
    renderer_type: PersonaVisualRendererType
    manifest_version: int = 1
    states_offered: list[str] = Field(default_factory=list)
    asset_count: int = 0
    total_bytes: int = 0
    tags: list[str] = Field(default_factory=list)
    license_label: str = "bundled"
    complexity_tier: PersonaVisualStarterComplexityTier = "basic"
    production_status: PersonaVisualStarterProductionStatus = "scaffold"
    neutral_anchor_required: bool = True
    expected_asset_groups: list[str] = Field(default_factory=list)
    animation_coverage_notes: list[str] = Field(default_factory=list)
    production_recipe: PersonaVisualStarterProductionRecipeResponse

    @field_validator("expected_asset_groups")
    @classmethod
    def validate_expected_asset_groups(cls, value: list[str]) -> list[str]:
        invalid_groups = [
            group for group in value if group not in BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS
        ]
        if invalid_groups:
            invalid_group_list = ", ".join(sorted(set(invalid_groups)))
            raise ValueError(
                "expected_asset_groups must use supported asset group ids. "
                f"Invalid: {invalid_group_list}"
            )
        return value

    @model_validator(mode="after")
    def validate_recipe_outputs_are_expected(self) -> "PersonaVisualStarterPackResponse":
        expected_groups = set(self.expected_asset_groups)
        missing_outputs = sorted(
            output
            for output in self.production_recipe.animation_outputs
            if output not in expected_groups
        )
        if missing_outputs:
            missing_output_list = ", ".join(missing_outputs)
            raise ValueError(
                "production_recipe.animation_outputs must be declared in "
                "expected_asset_groups. "
                f"Missing: {missing_output_list}"
            )
        return self


class PersonaVisualStarterPackDetailResponse(PersonaVisualStarterPackResponse):
    manifest: dict[str, Any] = Field(default_factory=dict)
    assets: list[PersonaVisualStarterAssetResponse] = Field(default_factory=list)


class PersonaVisualStarterPackListResponse(BaseModel):
    starter_packs: list[PersonaVisualStarterPackResponse] = Field(default_factory=list)


class PersonaVisualLibraryItemResponse(BaseModel):
    id: str
    user_id: str
    source_persona_id: str | None = None
    source_pack_id: str | None = None
    title: str
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_persona_name: str | None = None
    source_pack_title: str | None = None
    source_pack_version: int | None = None
    source_current_version: int | None = None
    source_available: bool = False
    source_changed: bool = False
    created_at: str
    last_modified: str
    version: int = 1


class PersonaVisualLibraryListResponse(BaseModel):
    items: list[PersonaVisualLibraryItemResponse] = Field(default_factory=list)


class PersonaVisualLibraryDeleteResponse(BaseModel):
    status: Literal["deleted"]
    item_id: str


class PersonaVisualCandidateReviewRequest(BaseModel):
    status: PersonaVisualCandidateReviewStatus
    failure_reason: str | None = Field(default=None, max_length=1000)


class PersonaVisualGenerationRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=4000)
    target_state: str | None = Field(default=None, max_length=80)
    backend: str | None = Field(default=None, max_length=80)
    starter_pack_id: str | None = Field(default=None, min_length=1, max_length=120)
    recipe_output: str | None = Field(default=None, min_length=1, max_length=320)

    @field_validator("request_id", "starter_pack_id", "recipe_output", "target_state", "backend")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


class PersonaVisualGenerationJobResponse(BaseModel):
    job_id: str
    status: str | None = None
    request_id: str | None = None


class PersonaVisualGenerationReadinessResponse(BaseModel):
    """Preflight state for queueing Persona visual generation jobs.

    ``available`` is true only when the Jobs worker is enabled and the
    selected or default image backend can resolve to an instantiable adapter.
    ``reasons`` contains machine-readable blockers for setup UI copy.
    """

    available: bool
    worker_enabled: bool
    queue: str
    image_backend_available: bool
    default_backend: str | None = None
    requested_backend: str | None = None
    requested_backend_available: bool | None = None
    enabled_backends: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PersonaVisualPackExportRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=120)
    strict: bool = False
    include_full_provenance: bool = False
    warn_for_sharing: bool = True


class PersonaVisualPortabilityJobResponse(BaseModel):
    job_id: str
    portability_job_id: str
    operation: PersonaVisualPortabilityOperation
    persona_id: str | None = None
    pack_id: str | None = None
    status: str
    visual_status: str
    stage: str
    progress: dict[str, Any] = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    archive_sha256: str | None = None
    canonical_payload_fingerprint: str | None = None
    download_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    expires_at: str | None = None


class PersonaVisualPackExportResponse(BaseModel):
    job_id: str
    portability_job_id: str
    operation: Literal["export"]
    persona_id: str
    pack_id: str
    status: str
    stage: str
    download_url: str | None = None


class PersonaVisualImportPreviewStartResponse(BaseModel):
    preview_id: str
    job_id: str
    portability_job_id: str
    operation: Literal["import_preview"]
    target_persona_id: str | None = None
    status: str
    stage: str


class PersonaVisualImportPreviewResponse(BaseModel):
    preview_id: str
    job_id: str
    portability_job_id: str
    operation: Literal["import_preview"]
    target_persona_id: str | None = None
    status: str
    visual_status: str
    stage: str
    archive_sha256: str | None = None
    canonical_payload_fingerprint: str | None = None
    schema_version: str | None = None
    bundle_summary: dict[str, Any] = Field(default_factory=dict)
    validation_warnings: list[Any] = Field(default_factory=list)
    conflicts: list[Any] = Field(default_factory=list)
    proposed_plan: dict[str, Any] = Field(default_factory=dict)
    quota_estimate: dict[str, Any] = Field(default_factory=dict)
    required_choices: list[Any] = Field(default_factory=list)
    target_warnings: list[Any] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    expires_at: str | None = None


class PersonaVisualImportCommitRequest(BaseModel):
    request_id: str | None = Field(default=None, max_length=120)
    trust_mode: Literal["trusted_restore", "untrusted_import"] = "untrusted_import"
    target_mode: Literal["create_new", "replace_draft"] = "create_new"
    target_pack_id: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("request_id", "target_pack_id", "title", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PersonaVisualImportCommitStartResponse(BaseModel):
    job_id: str
    portability_job_id: str
    operation: Literal["import_commit"]
    preview_id: str
    target_persona_id: str
    status: str
    stage: str


class PersonaVisualCandidateResponse(BaseModel):
    id: str
    pack_id: str
    persona_id: str
    user_id: str
    job_id: str | None = None
    status: PersonaVisualCandidateStatus
    proposed_manifest_patch: dict[str, Any] = Field(default_factory=dict)
    generated_asset_ids: list[str] = Field(default_factory=list)
    generated_assets: list[PersonaVisualAssetResponse] = Field(default_factory=list)
    generation_provenance: dict[str, Any] = Field(default_factory=dict)
    prompt: str | None = None
    failure_reason: str | None = None
    created_at: str
    last_modified: str
    version: int = 1


class PersonaVisualCandidateListResponse(BaseModel):
    candidates: list[PersonaVisualCandidateResponse] = Field(default_factory=list)


class PersonaVisualDeactivateResponse(BaseModel):
    status: Literal["deactivated"]
    persona_id: str


class PersonaInfo(BaseModel):
    id: str
    name: str
    description: str | None = None
    voice: str | None = None
    avatar_url: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    default_tools: list[str] = Field(default_factory=list)
    buddy_summary: PersonaBuddySummary | None = None


class PersonaBuddyVisualSummary(BaseModel):
    """Compact visual traits used to render a persona buddy preview."""

    species_id: str
    silhouette_id: str
    palette_id: str
    accessory_id: str | None = None
    eye_style: str | None = None
    expression_profile: str | None = None


class PersonaBuddySummary(BaseModel):
    """Small buddy summary embedded into persona profile and catalog responses."""

    has_buddy: bool = False
    persona_name: str
    role_summary: str | None = None
    visual: PersonaBuddyVisualSummary | None = None


class PersonaSessionRequest(BaseModel):
    persona_id: str
    project_id: str | None = None
    resume_session_id: str | None = None
    surface: str | None = Field(default=None, max_length=120)

    @field_validator("project_id", "resume_session_id", "surface", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None


class PersonaSessionResponse(BaseModel):
    session_id: str
    persona: PersonaInfo
    scopes: list[str] = Field(default_factory=list)
    runtime_mode: PersonaMode | None = None
    scope_snapshot_id: str | None = None
    scope_audit: dict[str, object] = Field(default_factory=dict)


class PersonaSessionSummary(BaseModel):
    session_id: str
    persona_id: str
    created_at: str
    updated_at: str
    turn_count: int = 0
    pending_plan_count: int = 0
    preferences: dict[str, object] = Field(default_factory=dict)
    runtime_mode: PersonaMode | None = None
    status: PersonaSessionStatus | None = None
    reuse_allowed: bool | None = None
    scope_snapshot_id: str | None = None
    scope_audit: dict[str, object] = Field(default_factory=dict)


class PersonaSessionDetail(PersonaSessionSummary):
    turns: list[dict[str, object]] = Field(default_factory=list)


class PersonaLiveSessionCreateRequest(BaseModel):
    persona_id: str = Field(min_length=1, max_length=128)
    reuse_policy: PersonaLiveReusePolicy = "resume_compatible"
    idempotency_key: str | None = Field(default=None, max_length=128)
    surface: str | None = Field(default=None, max_length=120)

    @field_validator("persona_id", mode="before")
    @classmethod
    def _strip_required_persona_id(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("persona_id is required")
        return stripped

    @field_validator("idempotency_key", "surface", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None


class PersonaLiveSessionSummary(BaseModel):
    session_id: str
    persona_id: str
    persona_name: str
    lifecycle: PersonaLiveLifecycle
    status: PersonaSessionStatus | None = None
    is_focused: bool = False
    focused_at: str | None = None
    focus_generation: int | None = None
    last_activity_at: str | None = None
    pending_approval_count: int = 0
    active_tool_name: str | None = None
    error_state: str | None = None
    recovery_hint: str | None = None
    suggested_visual_state: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)


class PersonaLiveSessionListResponse(BaseModel):
    sessions: list[PersonaLiveSessionSummary] = Field(default_factory=list)
    focused_session_id: str | None = None


class PersonaLiveSessionFocusResponse(BaseModel):
    session: PersonaLiveSessionSummary


class PersonaLiveSessionStopResponse(BaseModel):
    session: PersonaLiveSessionSummary


class PersonaVoiceDefaults(BaseModel):
    stt_language: str | None = None
    stt_model: str | None = None
    tts_provider: str | None = None
    tts_voice: str | None = None
    confirmation_mode: PersonaConfirmationMode | None = None
    voice_chat_trigger_phrases: list[str] = Field(default_factory=list)
    wake_behavior: PersonaWakeBehavior | None = None
    auto_resume: bool | None = None
    barge_in: bool | None = None
    auto_commit_enabled: bool | None = None
    vad_threshold: float | None = None
    min_silence_ms: int | None = None
    turn_stop_secs: float | None = None
    min_utterance_secs: float | None = None

    @field_validator("stt_language", "stt_model", "tts_provider", "tts_voice", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("voice_chat_trigger_phrases", mode="before")
    @classmethod
    def _normalize_trigger_phrases(cls, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        seen: set[str] = set()
        normalized: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @field_validator("vad_threshold", "turn_stop_secs", "min_utterance_secs", mode="before")
    @classmethod
    def _normalize_turn_detection_floats(cls, value: Any, info: ValidationInfo) -> float | None:
        if value is None or value == "":
            return None
        bounds = {
            "vad_threshold": (0.0, 1.0),
            "turn_stop_secs": (0.05, 10.0),
            "min_utterance_secs": (0.0, 10.0),
        }
        min_value, max_value = bounds[info.field_name]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(min_value, min(max_value, numeric))

    @field_validator("min_silence_ms", mode="before")
    @classmethod
    def _normalize_min_silence_ms(cls, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        return max(50, min(10_000, numeric))


class PersonaSetupState(BaseModel):
    """Persisted wizard progress for one persona setup run."""

    status: PersonaSetupStatus = "not_started"
    version: int = Field(default=1, ge=1)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    current_step: PersonaSetupStep = "persona"
    completed_steps: list[PersonaSetupStep] = Field(default_factory=list)
    completed_at: str | None = None
    last_test_type: PersonaSetupTestType | None = None


class PersonaSetupEventCreate(BaseModel):
    """Append-only setup analytics event payload sent by the UI."""

    event_id: str = Field(..., min_length=1, max_length=200)
    event_key: str | None = Field(default=None, min_length=1, max_length=200)
    run_id: str = Field(..., min_length=1, max_length=200)
    event_type: PersonaSetupEventType
    step: PersonaSetupStep | None = None
    completion_type: PersonaSetupTestType | None = None
    detour_source: str | None = Field(default=None, min_length=1, max_length=120)
    action_target: str | None = Field(default=None, min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonaSetupEventWriteResponse(BaseModel):
    """Write result for one setup analytics event, including dedupe outcome."""

    event_id: str
    run_id: str
    event_type: PersonaSetupEventType
    deduped: bool = False
    created_at: str | None = None


class PersonaSetupAnalyticsRunSummary(BaseModel):
    """Aggregated setup outcomes for one recorded setup run."""

    run_id: str
    started_at: str | None = None
    completed_at: str | None = None
    completion_type: PersonaSetupTestType | None = None
    terminal_step: PersonaSetupStep | None = None
    handoff_clicked: bool = False
    handoff_target_reached: bool = False
    handoff_dismissed: bool = False
    first_post_setup_action: bool = False


class PersonaSetupAnalyticsSummary(BaseModel):
    """High-level setup funnel and handoff metrics for one persona."""

    total_runs: int = 0
    completed_runs: int = 0
    completion_rate: float = 0.0
    dry_run_completion_count: int = 0
    live_session_completion_count: int = 0
    most_common_dropoff_step: PersonaSetupStep | None = None
    handoff_click_rate: float = 0.0
    handoff_target_reach_rate: float = 0.0
    first_post_setup_action_rate: float = 0.0
    handoff_target_reached_counts: dict[str, int] = Field(default_factory=dict)
    detour_started_counts: dict[str, int] = Field(default_factory=dict)
    detour_returned_counts: dict[str, int] = Field(default_factory=dict)


class PersonaSetupAnalyticsResponse(BaseModel):
    """Persona setup analytics API response with summary and recent runs."""

    persona_id: str
    summary: PersonaSetupAnalyticsSummary = Field(default_factory=PersonaSetupAnalyticsSummary)
    recent_runs: list[PersonaSetupAnalyticsRunSummary] = Field(default_factory=list)


class PersonaProfileCreate(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    archetype_key: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    mode: PersonaMode = "session_scoped"
    system_prompt: str | None = None
    is_active: bool = True
    use_persona_state_context_default: bool = True
    voice_defaults: PersonaVoiceDefaults = Field(default_factory=PersonaVoiceDefaults)
    setup: PersonaSetupState = Field(default_factory=PersonaSetupState)


class PersonaProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    mode: PersonaMode | None = None
    system_prompt: str | None = None
    is_active: bool | None = None
    use_persona_state_context_default: bool | None = None
    voice_defaults: PersonaVoiceDefaults | None = None
    setup: PersonaSetupState | None = None


class PersonaProfileResponse(BaseModel):
    id: str
    name: str
    archetype_key: str | None = Field(default=None, min_length=1, max_length=200)
    character_card_id: int | None = None
    origin_character_id: int | None = None
    origin_character_name: str | None = None
    origin_character_snapshot_at: str | None = None
    mode: PersonaMode
    system_prompt: str | None = None
    is_active: bool = True
    use_persona_state_context_default: bool = True
    voice_defaults: PersonaVoiceDefaults = Field(default_factory=PersonaVoiceDefaults)
    setup: PersonaSetupState = Field(default_factory=PersonaSetupState)
    created_at: str
    last_modified: str
    version: int = 1
    buddy_summary: PersonaBuddySummary | None = None


class PersonaBuddyResolvedProfile(BaseModel):
    """Fully resolved buddy configuration returned by the dedicated buddy endpoint."""

    derivation_version: int
    species_id: str
    silhouette_id: str
    palette_id: str
    behavior_family: str
    expression_profile: str
    accessory_id: str | None = None
    eye_style: str | None = None
    compatibility_status: Literal["exact", "fallback_applied"] = "exact"


class PersonaBuddyResponse(BaseModel):
    """API response for one resolved persona buddy."""

    persona_id: str
    resolved_profile: PersonaBuddyResolvedProfile | None = None
    created_at: str
    last_modified: str


class PersonaScopeRule(BaseModel):
    rule_type: PersonaScopeRuleType
    rule_value: str = Field(..., min_length=1, max_length=2048)
    include: bool = True


class PersonaScopeRulesReplaceRequest(BaseModel):
    rules: list[PersonaScopeRule] = Field(default_factory=list)


class PersonaScopeRulesResponse(BaseModel):
    persona_id: str
    replaced_count: int | None = None
    rules: list[PersonaScopeRule] = Field(default_factory=list)


class PersonaPolicyRule(BaseModel):
    rule_kind: PersonaPolicyRuleKind
    rule_name: str = Field(..., min_length=1, max_length=512)
    allowed: bool = True
    require_confirmation: bool = False
    max_calls_per_turn: int | None = Field(default=None, ge=1)


class PersonaPolicyRulesReplaceRequest(BaseModel):
    rules: list[PersonaPolicyRule] = Field(default_factory=list)


class PersonaPolicyRulesResponse(BaseModel):
    persona_id: str
    replaced_count: int | None = None
    rules: list[PersonaPolicyRule] = Field(default_factory=list)


class PersonaDeleteResponse(BaseModel):
    status: str
    persona_id: str


class PersonaExemplarCreate(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=200)
    kind: PersonaExemplarKind = "style"
    content: str = Field(..., min_length=1, max_length=20_000)
    tone: str | None = Field(default=None, min_length=1, max_length=200)
    scenario_tags: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    source_type: PersonaExemplarSourceType = "manual"
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)


class PersonaExemplarUpdate(BaseModel):
    kind: PersonaExemplarKind | None = None
    content: str | None = Field(default=None, min_length=1, max_length=20_000)
    tone: str | None = Field(default=None, min_length=1, max_length=200)
    scenario_tags: list[str] | None = None
    capability_tags: list[str] | None = None
    priority: int | None = None
    enabled: bool | None = None
    source_type: PersonaExemplarSourceType | None = None
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)


class PersonaExemplarResponse(BaseModel):
    id: str
    persona_id: str
    user_id: str
    kind: PersonaExemplarKind
    content: str
    tone: str | None = None
    scenario_tags: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    source_type: PersonaExemplarSourceType
    source_ref: str | None = None
    notes: str | None = None
    created_at: str
    last_modified: str
    deleted: bool = False
    version: int = 1


class PersonaExemplarImportRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=100_000)
    source_ref: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=10_000)
    max_candidates: int = Field(default=5, ge=1, le=10)


class PersonaExemplarReviewRequest(BaseModel):
    action: PersonaExemplarReviewAction
    notes: str | None = Field(default=None, max_length=10_000)


class PersonaExemplarDeleteResponse(BaseModel):
    status: str
    persona_id: str
    exemplar_id: str


class PersonaStateUpdateRequest(BaseModel):
    soul_md: str | None = Field(default=None, max_length=200_000)
    identity_md: str | None = Field(default=None, max_length=200_000)
    heartbeat_md: str | None = Field(default=None, max_length=200_000)


class PersonaStateResponse(BaseModel):
    persona_id: str
    soul_md: str | None = None
    identity_md: str | None = None
    heartbeat_md: str | None = None
    last_modified: str | None = None


PersonaStateField = Literal["soul_md", "identity_md", "heartbeat_md"]


class PersonaStateHistoryItem(BaseModel):
    entry_id: str
    field: PersonaStateField
    content: str
    is_active: bool = True
    created_at: str | None = None
    last_modified: str | None = None
    version: int = 1


class PersonaStateHistoryResponse(BaseModel):
    persona_id: str
    entries: list[PersonaStateHistoryItem] = Field(default_factory=list)


class PersonaStateRestoreRequest(BaseModel):
    entry_id: str = Field(..., min_length=1, max_length=200)


class PersonaConnectionCreate(BaseModel):
    id: str | None = Field(default=None, min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    base_url: str = Field(..., min_length=1, max_length=2048)
    auth_type: str = Field(default="none", min_length=1, max_length=64)
    secret: str | None = Field(default=None, min_length=1, max_length=8192)
    headers_template: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int = Field(default=15_000, ge=100, le=120_000)


class PersonaConnectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    base_url: str | None = Field(default=None, min_length=1, max_length=2048)
    auth_type: str | None = Field(default=None, min_length=1, max_length=64)
    secret: str | None = Field(default=None, min_length=1, max_length=8192)
    clear_secret: bool = False
    headers_template: dict[str, str] | None = None
    timeout_ms: int | None = Field(default=None, ge=100, le=120_000)


class PersonaConnectionResponse(BaseModel):
    id: str
    persona_id: str
    name: str
    base_url: str
    auth_type: str
    headers_template: dict[str, str] = Field(default_factory=dict)
    timeout_ms: int
    allowed_hosts: list[str] = Field(default_factory=list)
    secret_configured: bool = False
    key_hint: str | None = None
    created_at: str | None = None
    last_modified: str | None = None


class PersonaConnectionDeleteResponse(BaseModel):
    status: str
    persona_id: str
    connection_id: str


class PersonaConnectionTestRequest(BaseModel):
    method: str = Field(default="GET", min_length=1, max_length=16)
    path: str | None = Field(default=None, max_length=2048)
    payload: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    auth_header_name: str | None = Field(default=None, min_length=1, max_length=256)


class PersonaConnectionTestResponse(BaseModel):
    ok: bool
    connection_id: str
    method: str
    url: str
    request_headers: dict[str, str] = Field(default_factory=dict)
    request_payload: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int
    status_code: int | None = None
    body_preview: Any = None
    latency_ms: int | None = None
    error: str | None = None


class PersonaCommandDryRunRequest(BaseModel):
    heard_text: str = Field(..., min_length=1, max_length=20_000)


class PersonaCommandPlannedActionResponse(BaseModel):
    target_type: str
    target_name: str | None = None
    payload_preview: dict[str, Any] = Field(default_factory=dict)


class PersonaCommandSafetyGateResponse(BaseModel):
    classification: str
    requires_confirmation: bool
    reason: str


class PersonaCommandDryRunResponse(BaseModel):
    heard_text: str
    matched: bool
    match_reason: str | None = None
    command_id: str | None = None
    command_name: str | None = None
    connection_id: str | None = None
    connection_status: Literal["ok", "missing"] | None = None
    connection_name: str | None = None
    extracted_params: dict[str, Any] = Field(default_factory=dict)
    planned_action: PersonaCommandPlannedActionResponse | None = None
    safety_gate: PersonaCommandSafetyGateResponse | None = None
    fallback_to_persona_planner: bool = False
    failure_phase: str | None = None


class PersonaVoiceCommandAnalyticsItem(BaseModel):
    command_id: str
    command_name: str | None = None
    total_invocations: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    last_used: str | None = None


class PersonaVoiceFallbackAnalytics(BaseModel):
    total_invocations: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    last_used: str | None = None


class PersonaVoiceAnalyticsSummary(BaseModel):
    total_events: int = 0
    direct_command_count: int = 0
    planner_fallback_count: int = 0
    success_rate: float = 0.0
    fallback_rate: float = 0.0
    avg_response_time_ms: float = 0.0


class PersonaLiveVoiceAnalyticsSummary(BaseModel):
    total_committed_turns: int = 0
    vad_auto_commit_count: int = 0
    manual_commit_count: int = 0
    vad_auto_rate: float = 0.0
    manual_commit_rate: float = 0.0
    degraded_session_count: int = 0


class PersonaLiveVoiceSessionSummary(BaseModel):
    session_id: str
    started_at: str | None = None
    ended_at: str | None = None
    auto_commit_enabled: bool | None = None
    vad_threshold: float | None = None
    min_silence_ms: int | None = None
    turn_stop_secs: float | None = None
    min_utterance_secs: float | None = None
    turn_detection_changed_during_session: bool = False
    total_committed_turns: int = 0
    vad_auto_commit_count: int = 0
    manual_commit_count: int = 0
    manual_mode_required_count: int = 0
    text_only_tts_count: int = 0
    listening_recovery_count: int = 0
    thinking_recovery_count: int = 0


class PersonaLiveVoiceSessionUpdateRequest(BaseModel):
    listening_recovery_count: int = Field(default=0, ge=0)
    thinking_recovery_count: int = Field(default=0, ge=0)
    finalize: bool = False
    ended_at: str | None = None


class PersonaVoiceAnalyticsResponse(BaseModel):
    persona_id: str
    summary: PersonaVoiceAnalyticsSummary
    live_voice: PersonaLiveVoiceAnalyticsSummary = Field(
        default_factory=PersonaLiveVoiceAnalyticsSummary
    )
    recent_live_sessions: list[PersonaLiveVoiceSessionSummary] = Field(default_factory=list)
    commands: list[PersonaVoiceCommandAnalyticsItem] = Field(default_factory=list)
    fallbacks: PersonaVoiceFallbackAnalytics = Field(
        default_factory=PersonaVoiceFallbackAnalytics
    )
