"""Pydantic schemas for VN asset pack APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from tldw_Server_API.app.core.VN_Assets.constants import (
    DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    ITEM_REVIEW_STATUS_APPROVED,
    ITEM_REVIEW_STATUS_DRAFT,
    ITEM_REVIEW_STATUS_HIDDEN,
    ITEM_REVIEW_STATUS_REJECTED,
)

ReviewStatus = Literal["draft", "approved", "rejected", "hidden"]
SlotStatus = Literal[
    "planned",
    "queued",
    "generating",
    "reviewing",
    "approved",
    "failed",
    "skipped",
    "cancelled",
]


class VNAssetPackCreate(BaseModel):
    """Request body for creating VN asset pack metadata."""

    title: str = Field(..., min_length=1, max_length=500)
    primary_character_id: int = Field(..., ge=1)
    description: str | None = None
    content_rating: str = Field(default="general", min_length=1, max_length=100)
    source_world_book_ids: list[int] = Field(default_factory=list)
    scenario_notes: str | None = None
    style_prompt: str | None = None
    negative_prompt: str | None = None
    default_backend: str | None = None
    default_model: str | None = None
    default_dimensions: dict[str, Any] | None = None
    style_lock: dict[str, Any] | None = None
    generation_budget: dict[str, Any] | None = None
    apply_starter_matrix: bool = False
    starter_matrix_variant_count: int = Field(
        default=1,
        strict=True,
        ge=1,
        le=DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    )


class VNAssetPackUpdate(BaseModel):
    """Request body for patching VN asset pack metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = None
    content_rating: str | None = Field(default=None, min_length=1, max_length=100)
    source_world_book_ids: list[int] | None = None
    scenario_notes: str | None = None
    style_prompt: str | None = None
    negative_prompt: str | None = None
    default_backend: str | None = None
    default_model: str | None = None
    default_dimensions: dict[str, Any] | None = None
    style_lock: dict[str, Any] | None = None
    generation_budget: dict[str, Any] | None = None


class VNAssetPackResponse(BaseModel):
    """Serialized VN asset pack metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    title: str
    primary_character_id: int
    description: str | None = None
    status: str
    content_rating: str
    source_world_book_ids: list[int] = Field(default_factory=list)
    scenario_notes: str | None = None
    style_prompt: str | None = None
    negative_prompt: str | None = None
    default_backend: str | None = None
    default_model: str | None = None
    default_dimensions: dict[str, Any] | None = None
    style_lock: dict[str, Any] | None = None
    generation_budget: dict[str, Any] | None = None
    planned_output_count: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None
    version: int
    deleted: bool


class VNAssetSlotCreate(BaseModel):
    """Request body for creating a VN asset slot."""

    model_config = ConfigDict(extra="forbid")

    asset_type: str = Field(..., min_length=1, max_length=100)
    slot_key: str = Field(..., min_length=1, max_length=300)
    labels: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str | None = None
    negative_prompt_template: str | None = None
    variant_count: int = Field(
        default=1,
        strict=True,
        ge=0,
        le=DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    )
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    backend_override: str | None = None
    model_override: str | None = None
    seed_policy: dict[str, Any] | None = None
    requires_review: bool = True
    required_for_runtime: bool = True
    depends_on_slot_id: int | None = Field(default=None, ge=1)
    status: SlotStatus = "planned"
    last_error: str | None = None


class VNAssetSlotUpdate(BaseModel):
    """Request body for patching a VN asset slot."""

    model_config = ConfigDict(extra="forbid")

    asset_type: str | None = Field(default=None, min_length=1, max_length=100)
    slot_key: str | None = Field(default=None, min_length=1, max_length=300)
    labels: dict[str, Any] | None = None
    prompt_template: str | None = None
    negative_prompt_template: str | None = None
    variant_count: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        le=DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    )
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    backend_override: str | None = None
    model_override: str | None = None
    seed_policy: dict[str, Any] | None = None
    requires_review: bool | None = None
    required_for_runtime: bool | None = None
    depends_on_slot_id: int | None = Field(default=None, ge=1)
    status: SlotStatus | None = None
    last_error: str | None = None


class VNAssetSlotResponse(BaseModel):
    """Serialized VN asset slot metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    pack_id: int
    asset_type: str
    slot_key: str
    labels: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str | None = None
    negative_prompt_template: str | None = None
    variant_count: int
    width: int | None = None
    height: int | None = None
    backend_override: str | None = None
    model_override: str | None = None
    seed_policy: dict[str, Any] | None = None
    requires_review: bool
    required_for_runtime: bool
    depends_on_slot_id: int | None = None
    status: str
    last_error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class VNAssetItemResponse(BaseModel):
    """Serialized VN asset item metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    pack_id: int
    slot_id: int
    variant_index: int
    file_artifact_id: str | None = None
    generated_file_id: int | None = None
    storage_ref: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    review_status: str
    preferred: bool
    source: str
    generation_job_id: str | None = None
    depth_kind: str | None = None
    parent_item_id: int | None = None
    has_alpha: bool | None = None
    crop_box: dict[str, Any] | None = None
    anchor: dict[str, float] | None = None
    scale_hint: float | None = None
    trim_status: str
    quality_flags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class VNAssetReviewRequest(BaseModel):
    """Request body for item review transitions."""

    review_status: ReviewStatus
    preferred: bool | None = None


class VNAssetBulkReviewRequest(BaseModel):
    """Request body for applying one review transition to many items."""

    item_ids: list[int] = Field(..., min_length=1)
    review_status: ReviewStatus


class VNAssetGenerationRequest(BaseModel):
    """Request body for planning a generation batch."""

    slot_ids: list[int] = Field(default_factory=list)
    variant_count: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    )
    options: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)


class VNAssetGenerationStatusResponse(BaseModel):
    """Serialized VN asset generation batch status."""

    batch_id: int | None = None
    job_batch_id: str | None = None
    status: str
    total_slots: int = 0
    total_variants: int = 0
    planned_count: int = 0
    enqueued_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    enqueue_error: str | None = None


class VNPackExportRequest(BaseModel):
    """Request body for starting a VN pack backup export."""

    model_config = ConfigDict(extra="forbid")

    include_character_payload: StrictBool = False
    include_world_book_payloads: StrictBool = False
    include_full_provenance: StrictBool = False
    strict: StrictBool = False
    warn_for_sharing: StrictBool = True
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)
    request_id: str | None = Field(default=None, min_length=1, max_length=160)


class VNPackExportResponse(BaseModel):
    """Response returned after a VN pack export job is queued."""

    job_id: str
    portability_job_id: int
    operation: str
    pack_id: int | None = None
    status: str
    stage: str
    download_url: str | None = None


class VNPackPortabilityJobResponse(BaseModel):
    """Composed Jobs lifecycle and VN portability stage response."""

    job_id: str
    portability_job_id: int
    operation: str
    pack_id: int | None = None
    status: str
    vn_status: str
    stage: str
    progress: dict[str, Any] = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    archive_sha256: str | None = None
    canonical_payload_fingerprint: str | None = None
    download_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    expires_at: str | None = None


class VNPackImportPreviewStartResponse(BaseModel):
    """Response returned after a VN pack import preview job is queued."""

    job_id: str
    portability_job_id: int
    operation: str
    preview_id: int
    status: str
    stage: str


class VNPackImportPreviewResponse(BaseModel):
    """Composed preview, Jobs lifecycle, and VN portability stage response."""

    preview_id: int
    job_id: str
    portability_job_id: int
    operation: str
    status: str
    vn_status: str
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
    error_code: str | None = None
    error_message: str | None = None
    expires_at: str | None = None


class VNPackImportCommitRequest(BaseModel):
    """Request body for committing a validated VN pack preview."""

    model_config = ConfigDict(extra="forbid")

    preview_id: int = Field(..., ge=1)
    trust_mode: Literal["trusted_restore", "untrusted_import"]
    target_mode: Literal["create_new", "update_existing"] = "create_new"
    character_action: Literal[
        "import_included_character",
        "link_existing_character",
        "create_placeholder_character",
        "fail_import",
    ]
    target_character_id: int | None = Field(default=None, ge=1)
    target_pack_id: int | None = Field(default=None, ge=1)
    conflict_decisions: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)
    request_id: str | None = Field(default=None, min_length=1, max_length=160)


class VNPackImportCommitStartResponse(BaseModel):
    """Response returned after a VN pack import commit job is queued."""

    job_id: str
    portability_job_id: int
    operation: str
    preview_id: int
    import_id: int
    status: str
    stage: str


class VNPackImportJobResponse(BaseModel):
    """Composed Jobs lifecycle and VN import-journal stage response."""

    job_id: str
    portability_job_id: int
    operation: str
    preview_id: int
    import_id: int
    status: str
    vn_status: str
    stage: str
    pack_id: int | None = None
    id_maps: dict[str, Any] = Field(default_factory=dict)
    created_records: dict[str, Any] = Field(default_factory=dict)
    cleanup_status: dict[str, Any] = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    archive_sha256: str | None = None
    canonical_payload_fingerprint: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    completed_at: str | None = None


class VNAssetPromptPreviewRequest(BaseModel):
    """Request body for prompt preview rendering."""

    slot_id: int = Field(..., ge=1)
    variant_index: int = Field(default=0, ge=0)
    budgets: dict[str, StrictInt] | None = None


class VNAssetPromptPreviewResponse(BaseModel):
    """Rendered prompt preview and diagnostics."""

    prompt: str
    negative_prompt: str = ""
    omitted_source_counts: dict[str, int] = Field(default_factory=dict)
    token_estimates: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class VNAssetReadinessResponse(BaseModel):
    """Pack runtime readiness response."""

    ready: bool
    status: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class VNAssetManifestResponse(BaseModel):
    """Approved-only runtime asset manifest."""

    schema_version: str
    pack_id: int
    title: str
    primary_character_id: int
    content_rating: str
    assets: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class VNAssetCleanupRequest(BaseModel):
    """Request body for VN asset generated-file cleanup previews or execution."""

    dry_run: bool = True
    statuses: list[ReviewStatus] = Field(default_factory=lambda: ["rejected", "hidden"])
    item_statuses: list[ReviewStatus] | None = None
    item_ids: list[int] | None = None
    include_approved: bool = False
    confirmation_text: str | None = None
    confirmation_token: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)


class VNAssetCleanupResponse(BaseModel):
    """Cleanup result summary."""

    dry_run: bool
    removed_item_ids: list[int] = Field(default_factory=list)
    removed_file_count: int = 0
    files_would_delete: int = 0
    files_deleted: int = 0
    skipped_file_ids: list[int] = Field(default_factory=list)
    blocked_count: int = 0
    cleanup_blocked: list[dict[str, Any]] = Field(default_factory=list)
    reclaimed_bytes: int = 0


class VNAssetStarterMatrixSummary(BaseModel):
    """Summary of an available starter matrix."""

    key: str
    title: str
    slot_count: int = Field(..., ge=0)
    planned_output_count: int = Field(..., ge=0)
    asset_types: list[str] = Field(default_factory=list)


class VNAssetStarterMatricesResponse(BaseModel):
    """Available VN asset starter matrices."""

    matrices: list[VNAssetStarterMatrixSummary] = Field(default_factory=list)


VALID_REVIEW_STATUSES = {
    ITEM_REVIEW_STATUS_APPROVED,
    ITEM_REVIEW_STATUS_DRAFT,
    ITEM_REVIEW_STATUS_REJECTED,
    ITEM_REVIEW_STATUS_HIDDEN,
}
