"""Request and response schemas for prompt management APIs."""

# tldw_Server_API/app/api/v1/schemas/prompts_schemas.py
#
# Imports
from datetime import datetime
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    PagePaginationMeta,
    default_offset_pagination_aliases,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement import (
    PROMPT_IMPROVEMENT_LIMITS,
)

#
# Third-party Imports
#
# Local Imports
#
########################################################################################################################
#
# --- Prompt Improvement ---

_MAX_PUBLIC_ERROR_MESSAGE_CHARS = 300
_MAX_REQUEST_ID_CHARS = 128
_MAX_RETRY_AFTER_SECONDS = 86_400
PROMPT_IMPROVEMENT_MAX_REQUEST_BYTES = PROMPT_IMPROVEMENT_LIMITS.max_request_bytes

PromptImproveFindingCategory = Literal[
    "clarity",
    "specificity",
    "structure",
    "constraints",
    "output",
    "consistency",
    "concision",
    "robustness",
    "other",
]
PromptImproveErrorCode = Literal[
    "invalid_input",
    "missing_model",
    "unsupported_model",
    "provider_not_configured",
    "draft_too_large",
    "provider_rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "model_refusal",
    "invalid_model_output",
    "preservation_failed",
    "internal_error",
]
PromptImproveWarning = Annotated[
    str,
    Field(min_length=1, max_length=PROMPT_IMPROVEMENT_LIMITS.max_warning_chars),
]


class PromptImproveModelSelection(BaseModel):
    """Snapshot of the route selected by the chat client."""

    model_config = ConfigDict(extra="forbid")

    selected_model: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_model_chars,
    )
    provider_hint: str | None = Field(
        default=None,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_provider_chars,
    )


class PromptProtectedToken(BaseModel):
    """Bounded preservation hint that must already occur in the target draft."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_protected_token_kind_chars,
    )
    value: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_protected_token_chars,
    )
    occurrences: int = Field(
        ge=1,
        le=PROMPT_IMPROVEMENT_LIMITS.max_protected_token_occurrences,
    )


class PromptImproveRequest(BaseModel):
    """One isolated prompt draft submitted for improvement."""

    model_config = ConfigDict(extra="forbid")

    operation_id: UUID
    target: Literal["system", "user_message"]
    text: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_draft_chars,
    )
    model_selection: PromptImproveModelSelection
    protected_tokens: list[PromptProtectedToken] = Field(
        default_factory=list,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_protected_tokens,
    )


class PromptImproveFinding(BaseModel):
    """Concise provider-authored observation about the candidate."""

    model_config = ConfigDict(extra="forbid")

    category: PromptImproveFindingCategory
    issue: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_finding_text_chars,
    )
    change: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_finding_text_chars,
    )


class PromptResolvedModel(BaseModel):
    """Concrete provider/model used for the improvement call."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_provider_chars,
    )
    model: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_model_chars,
    )
    display_name: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_model_chars,
    )


class PromptImproveResponse(BaseModel):
    """Validated prompt-improvement result returned to clients."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    operation_id: UUID
    status: Literal["improved", "no_change"]
    improved_text: str | None = Field(
        default=None,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_candidate_chars,
    )
    findings: list[PromptImproveFinding] = Field(
        default_factory=list,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_findings,
    )
    review_required: bool
    warnings: list[PromptImproveWarning] = Field(
        default_factory=list,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_warnings,
    )
    resolved_model: PromptResolvedModel
    meta_prompt_version: str = Field(
        min_length=1,
        max_length=PROMPT_IMPROVEMENT_LIMITS.max_meta_prompt_version_chars,
    )

    @model_validator(mode="after")
    def validate_status_candidate_pair(self) -> "PromptImproveResponse":
        """Keep no-change and replacement results unambiguous."""

        if self.status == "improved" and (
            self.improved_text is None or not self.improved_text.strip()
        ):
            raise ValueError("improved_text is required for an improved result")
        if self.status == "no_change" and self.improved_text is not None:
            raise ValueError("improved_text must be null for a no-change result")
        return self


class PromptImproveErrorResponse(BaseModel):
    """Stable, sanitized prompt-improvement failure contract."""

    model_config = ConfigDict(extra="forbid")

    code: PromptImproveErrorCode
    message: str = Field(min_length=1, max_length=_MAX_PUBLIC_ERROR_MESSAGE_CHARS)
    retryable: bool
    retry_after_seconds: int | None = Field(
        default=None,
        ge=0,
        le=_MAX_RETRY_AFTER_SECONDS,
    )
    request_id: str = Field(min_length=1, max_length=_MAX_REQUEST_ID_CHARS)


class PromptImprovementLimitsResponse(BaseModel):
    """Capability bounds sourced from the pure service policy."""

    model_config = ConfigDict(extra="forbid")

    max_request_bytes: int = Field(gt=0)
    max_draft_chars: int = Field(gt=0)
    max_candidate_chars: int = Field(gt=0)
    max_raw_output_chars: int = Field(gt=0)
    max_findings: int = Field(gt=0)
    max_finding_text_chars: int = Field(gt=0)
    max_provider_chars: int = Field(gt=0)
    max_model_chars: int = Field(gt=0)
    max_meta_prompt_version_chars: int = Field(gt=0)
    max_warning_chars: int = Field(gt=0)
    max_warnings: int = Field(gt=0)
    max_protected_tokens: int = Field(gt=0)
    max_protected_token_kind_chars: int = Field(gt=0)
    max_protected_token_chars: int = Field(gt=0)
    max_protected_token_occurrences: int = Field(gt=0)
    max_protected_token_total_chars: int = Field(gt=0)


class PromptImprovementCapability(BaseModel):
    """Availability and limits for Track A prompt improvement."""

    model_config = ConfigDict(extra="forbid")

    supported: bool
    limits: PromptImprovementLimitsResponse


class PromptRecipeCapability(BaseModel):
    """Availability marker for the future Track B recipe contract."""

    model_config = ConfigDict(extra="forbid")

    supported: bool


class PromptCapabilitiesResponse(BaseModel):
    """Versioned prompt feature discovery response."""

    model_config = ConfigDict(extra="forbid")

    prompt_improvement_v1: PromptImprovementCapability
    single_text_recipe_v2: PromptRecipeCapability


# --- Keyword Schemas ---
class KeywordBase(BaseModel):
    keyword_text: str = Field(..., min_length=1, max_length=100, description="The text of the keyword.")


class KeywordCreate(KeywordBase):
    pass


class KeywordResponse(KeywordBase):
    id: int
    uuid: UUID

    # last_modified: datetime # If you want to expose these
    # version: int

    model_config = ConfigDict(from_attributes=True)  # For compatibility if directly mapping from DB model in future


# --- Prompt Schemas ---
class PromptBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique name of the prompt.")
    author: Optional[str] = Field(None, max_length=100, description="Author of the prompt.")
    details: Optional[str] = Field(None, max_length=4000, description="Detailed description or notes about the prompt.")
    system_prompt: Optional[str] = Field(None, max_length=20000, description="The system part of the prompt.")
    user_prompt: Optional[str] = Field(None, max_length=20000, description="The user part of the prompt.")
    prompt_format: Literal["legacy", "structured"] = Field(
        "legacy",
        description="Whether the prompt is stored as legacy text fields or a structured definition.",
    )
    prompt_schema_version: Optional[int] = Field(
        None,
        ge=1,
        description="Structured prompt schema version when prompt_format is 'structured'.",
    )
    prompt_definition: Optional[dict[str, Any]] = Field(
        None,
        description="Structured prompt definition when prompt_format is 'structured'.",
    )


class PromptCreate(PromptBase):
    keywords: Optional[list[str]] = Field(None, description="List of keyword strings to associate with the prompt.")


class PromptUpdate(BaseModel):  # For partial updates if we add a PATCH endpoint
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    author: Optional[str] = Field(None, max_length=100)
    details: Optional[str] = Field(None, max_length=4000)
    system_prompt: Optional[str] = Field(None, max_length=20000)
    user_prompt: Optional[str] = Field(None, max_length=20000)
    keywords: Optional[list[str]] = None  # To update keywords
    usage_count: Optional[int] = Field(None, ge=0)
    last_used_at: Optional[datetime] = None


class PromptResponse(PromptBase):
    id: int
    uuid: UUID
    last_modified: datetime
    version: int
    usage_count: int = Field(0, ge=0, description="Number of times this prompt has been used.")
    last_used_at: Optional[datetime] = Field(None, description="Timestamp of the most recent use.")
    keywords: list[str] = Field(default_factory=list, description="Keywords associated with the prompt.")
    deleted: bool = Field(..., description="Indicates if the prompt is soft-deleted.")

    model_config = ConfigDict(from_attributes=True)


class PromptBriefResponse(BaseModel):
    id: int
    uuid: UUID
    name: str
    author: Optional[str]
    last_modified: datetime
    usage_count: int = 0
    last_used_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedPromptsResponse(BaseModel):
    items: list[PromptBriefResponse]
    total_pages: int
    current_page: int
    total_items: int
    pagination: PagePaginationMeta


class PromptSearchResultItem(PromptResponse):  # Or a more specific search result schema
    relevance_score: Optional[float] = None  # If FTS provides it


class PromptSearchResponse(BaseModel):
    items: list[PromptSearchResultItem]
    total_matches: int
    page: int
    per_page: int
    pagination: PagePaginationMeta


class PromptVersionResponse(BaseModel):
    version: int
    created_at: Optional[datetime] = None
    comment: Optional[str] = None
    name: Optional[str] = None
    author: Optional[str] = None
    details: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ExportResponse(BaseModel):
    message: str
    file_path: Optional[str] = None  # Could be a download link or internal path for admin
    file_content_b64: Optional[str] = None  # For direct download via API


# --- Sync Log (Admin/Debug) ---
class SyncLogEntryResponse(BaseModel):
    change_id: int
    entity: str
    entity_uuid: UUID
    operation: str
    timestamp: datetime
    client_id: str
    version: int
    payload: Optional[dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


# --- Import/Export ---
class PromptImportItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    content: Optional[str] = None
    details: Optional[str] = None
    author: Optional[str] = Field(None, max_length=100)
    system_prompt: Optional[str] = Field(None, max_length=20000)
    user_prompt: Optional[str] = Field(None, max_length=20000)
    keywords: list[str] = Field(default_factory=list)


class PromptImportRequest(BaseModel):
    prompts: list[PromptImportItem] = Field(..., min_length=1)
    skip_duplicates: bool = False


class PromptImportResponse(BaseModel):
    imported: int
    failed: int
    skipped: int
    prompt_ids: list[int] = Field(default_factory=list)


# --- Template Processing ---
class TemplateVariablesRequest(BaseModel):
    template: str = Field(..., min_length=1)


class TemplateVariablesResponse(BaseModel):
    variables: list[str]


class TemplateRenderRequest(BaseModel):
    template: str = Field(..., min_length=1)
    variables: dict[str, Any]


class TemplateRenderResponse(BaseModel):
    rendered: str


# --- Structured Prompt Preview / Conversion ---
class StructuredPromptPreviewRequest(BaseModel):
    prompt_format: Literal["legacy", "structured"] = "legacy"
    system_prompt: Optional[str] = Field(None, max_length=20000)
    user_prompt: Optional[str] = Field(None, max_length=20000)
    prompt_schema_version: Optional[int] = Field(None, ge=1)
    prompt_definition: Optional[dict[str, Any]] = None
    variables: dict[str, Any] = Field(default_factory=dict)


class StructuredPromptPreviewResponse(BaseModel):
    prompt_format: Literal["legacy", "structured"]
    prompt_schema_version: Optional[int] = None
    assembled_messages: list[dict[str, str]] = Field(default_factory=list)
    legacy_system_prompt: str = ""
    legacy_user_prompt: str = ""


class StructuredPromptConvertRequest(BaseModel):
    system_prompt: Optional[str] = Field(None, max_length=20000)
    user_prompt: Optional[str] = Field(None, max_length=20000)


class StructuredPromptConvertResponse(BaseModel):
    prompt_format: Literal["structured"] = "structured"
    prompt_schema_version: int = 1
    prompt_definition: dict[str, Any]
    extracted_variables: list[str] = Field(default_factory=list)
    legacy_system_prompt: str = ""
    legacy_user_prompt: str = ""


# --- Bulk Operations ---
class PromptBulkDeleteRequest(BaseModel):
    prompt_ids: list[int] = Field(..., min_length=1)


class PromptBulkDeleteResponse(BaseModel):
    deleted: int
    failed: int
    failed_ids: list[int] = Field(default_factory=list)


class PromptBulkKeywordsRequest(BaseModel):
    prompt_ids: list[int] = Field(..., min_length=1)
    add_keywords: list[str] = Field(default_factory=list)
    remove_keywords: list[str] = Field(default_factory=list)


class PromptBulkKeywordsResponse(BaseModel):
    updated: int
    failed: int
    failed_ids: list[int] = Field(default_factory=list)

#
# End of prompts_schemas.py
#######################################################################################################################

# --- Legacy/compat request models (for endpoints using simple payloads) ---

class LegacyPromptCreateRequest(BaseModel):
    """Legacy payload for creating prompts.

    Supports both 'details' and legacy 'content'.
    """
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    name: Optional[str] = None
    author: Optional[str] = None
    details: Optional[str] = None
    content: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)

    @property
    def effective_details(self) -> Optional[str]:
        return self.details if (self.details and self.details.strip()) else self.content


class CreatePromptCompatRequest(LegacyPromptCreateRequest):
    """Alias for legacy create payload (compat route)."""
    pass


class PromptCollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    description: Optional[str] = None
    prompt_ids: list[int] = Field(default_factory=list)


class PromptCollectionCreateResponse(BaseModel):
    collection_id: int


class PromptCollectionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = None
    description: Optional[str] = None
    prompt_ids: Optional[list[int]] = None


class PromptCollectionResponse(BaseModel):
    collection_id: int
    name: str
    description: Optional[str] = None
    prompt_ids: list[int] = Field(default_factory=list)


class PromptCollectionListResponse(BaseModel):
    collections: list[PromptCollectionResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 200
    offset: int = 0
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)
