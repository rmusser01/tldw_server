"""Shared contracts for per-occurrence playlist ingest persistence."""

from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Conservative limits for explicitly reviewed metadata mutations.
MAX_METADATA_PATCH_TEXT_LENGTH = 500
MAX_METADATA_PATCH_KEYWORDS = 100
MAX_METADATA_PATCH_KEYWORD_LENGTH = 128
MAX_PLAYLIST_PREFLIGHT_SELECTIONS = 500
MAX_RUN_IDENTITY_LENGTH = 255
MAX_RUN_URL_LENGTH = 8192
MAX_RUN_DISPLAY_TEXT_LENGTH = 2000
MAX_RUN_JSON_LENGTH = 65536
MAX_RUN_JSON_DEPTH = 8
MAX_RUN_JSON_ITEMS = 2000


def _validate_bounded_json(
    value: object,
    *,
    max_length: int = MAX_RUN_JSON_LENGTH,
    max_depth: int = MAX_RUN_JSON_DEPTH,
    max_items: int = MAX_RUN_JSON_ITEMS,
) -> object:
    """Reject non-JSON, non-finite, excessively nested, or oversized values."""
    item_count = 0

    def visit(candidate: object, depth: int) -> None:
        nonlocal item_count
        if depth > max_depth:
            raise ValueError("JSON value is too deeply nested")
        if candidate is None or type(candidate) in {bool, str, int}:
            return
        if type(candidate) is float:
            if not math.isfinite(candidate):
                raise ValueError("JSON numbers must be finite")
            return
        if type(candidate) is list:
            item_count += len(candidate)
            if item_count > max_items:
                raise ValueError("JSON value contains too many items")
            for entry in candidate:
                visit(entry, depth + 1)
            return
        if type(candidate) is dict:
            item_count += len(candidate)
            if item_count > max_items:
                raise ValueError("JSON value contains too many items")
            for key, entry in candidate.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                visit(entry, depth + 1)
            return
        raise ValueError("value must contain only JSON types")

    visit(value, 0)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON serializable") from exc
    if len(encoded) > max_length:
        raise ValueError("JSON value is too large")
    return value


class DuplicatePolicy(str, Enum):
    """Explicit action choices for an existing media item."""

    SKIP = "skip"
    INCLUDE_EXISTING = "include_existing"
    UPDATE_METADATA_ONLY = "update_metadata_only"
    OVERWRITE = "overwrite"


class RunItemState(str, Enum):
    """Server-owned lifecycle states for one ingest occurrence."""

    STAGED = "staged"
    PREPARING = "preparing"
    AWAITING_UPLOAD = "awaiting_upload"
    SUBMIT_PENDING = "submit_pending"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    STATUS_UNAVAILABLE = "status_unavailable"
    TERMINAL = "terminal"


class RunItemOutcome(str, Enum):
    """Terminal results, kept separate from lifecycle state."""

    COMPLETED = "completed"
    INCLUDED_EXISTING = "included_existing"
    METADATA_UPDATED = "metadata_updated"
    SKIPPED_EXISTING = "skipped_existing"
    SUBMIT_FAILED = "submit_failed"
    PROCESSING_FAILED = "processing_failed"
    METADATA_UPDATE_FAILED = "metadata_update_failed"
    CANCELLED = "cancelled"


class MetadataPatch(BaseModel):
    """Explicit, bounded metadata fields reviewed for an existing item."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_TEXT_LENGTH)
    author: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_TEXT_LENGTH)
    keywords_add: list[str] | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_KEYWORDS)

    @field_validator("title", "author", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("metadata patch text must be a string")
        return value.strip()

    @field_validator("keywords_add", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not list:
            raise ValueError("keywords_add must be a list")
        normalized: list[str] = []
        for keyword in value:
            if type(keyword) is not str:
                raise ValueError("keywords_add entries must be strings")
            trimmed = keyword.strip()
            if not trimmed:
                raise ValueError("keywords_add entries must not be blank")
            if len(trimmed) > MAX_METADATA_PATCH_KEYWORD_LENGTH:
                raise ValueError(
                    f"keywords_add entries must be {MAX_METADATA_PATCH_KEYWORD_LENGTH} characters or fewer"
                )
            normalized.append(trimmed)
        return normalized

    @model_validator(mode="after")
    def _require_change(self) -> MetadataPatch:
        if self.title is None and self.author is None and self.keywords_add is None:
            raise ValueError("metadata_patch must contain at least one change")
        return self


class ReviewOverride(BaseModel):
    """Review-time action and optional metadata patch."""

    model_config = ConfigDict(extra="forbid")

    duplicate_policy: DuplicatePolicy
    metadata_patch: MetadataPatch | None = None
    existing_media_id: int | None = Field(default=None, ge=1, strict=True)
    duplicate_of_occurrence_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RUN_IDENTITY_LENGTH,
    )

    @field_validator("duplicate_of_occurrence_id", mode="before")
    @classmethod
    def _strip_duplicate_occurrence_id(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("duplicate_of_occurrence_id must be a string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_policy_patch(self) -> ReviewOverride:
        if self.duplicate_policy is DuplicatePolicy.UPDATE_METADATA_ONLY and self.metadata_patch is None:
            raise ValueError("update_metadata_only requires metadata_patch")
        if (
            self.duplicate_policy in {DuplicatePolicy.SKIP, DuplicatePolicy.INCLUDE_EXISTING}
            and self.metadata_patch is not None
        ):
            raise ValueError(f"{self.duplicate_policy.value} does not allow metadata_patch")
        return self


class ReviewOverrideEnvelope(BaseModel):
    """Bounded raw Review choice whose semantics are checked after refresh."""

    model_config = ConfigDict(extra="forbid")

    duplicate_policy: str = Field(..., min_length=1, max_length=64)
    metadata_patch: dict[str, Any] | None = Field(default=None, max_length=10)
    existing_media_id: int | None = Field(default=None, ge=1, strict=True)
    duplicate_of_occurrence_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_RUN_IDENTITY_LENGTH,
    )

    @field_validator("duplicate_policy", "duplicate_of_occurrence_id", mode="before")
    @classmethod
    def _strip_override_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("review override text must be a string")
        return value.strip()

    @field_validator("metadata_patch", mode="before")
    @classmethod
    def _bound_raw_patch(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not dict:
            raise ValueError("metadata_patch must be an object")
        _validate_bounded_json(value, max_length=8192, max_depth=2, max_items=MAX_METADATA_PATCH_KEYWORDS + 10)
        for patch_value in value.values():
            if isinstance(patch_value, dict):
                raise ValueError("metadata_patch must be shallow")
            if isinstance(patch_value, list) and (
                len(patch_value) > MAX_METADATA_PATCH_KEYWORDS
                or any(isinstance(entry, (dict, list)) for entry in patch_value)
            ):
                raise ValueError("metadata_patch lists must be bounded and shallow")
        return value


class CompactRunDisplayMetadata(BaseModel):
    """Bounded display-only metadata accepted in a run manifest."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_RUN_DISPLAY_TEXT_LENGTH)
    channel_or_uploader: str | None = Field(default=None, min_length=1, max_length=MAX_RUN_DISPLAY_TEXT_LENGTH)
    duration_seconds: int | None = Field(default=None, ge=0, le=315_576_000, strict=True)
    published_at: str | None = Field(default=None, min_length=1, max_length=128)
    thumbnail_url: str | None = Field(default=None, min_length=1, max_length=MAX_RUN_URL_LENGTH)
    playlist_id: str | None = Field(default=None, min_length=1, max_length=MAX_RUN_IDENTITY_LENGTH)
    playlist_title: str | None = Field(default=None, min_length=1, max_length=MAX_RUN_DISPLAY_TEXT_LENGTH)

    @field_validator(
        "title",
        "channel_or_uploader",
        "published_at",
        "thumbnail_url",
        "playlist_id",
        "playlist_title",
        mode="before",
    )
    @classmethod
    def _strip_display_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("display metadata text must be a string")
        return value.strip()


class _RunOccurrenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occurrence_id: str = Field(..., min_length=1, max_length=MAX_RUN_IDENTITY_LENGTH)

    @field_validator("occurrence_id", mode="before")
    @classmethod
    def _strip_input_occurrence_id(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("occurrence_id must be a string")
        return value.strip()


class MaterializedPlaylistItemInput(_RunOccurrenceInput):
    """Reference one server-authoritative materialized playlist occurrence."""

    input_kind: Literal["materialized_playlist_item"]
    materialization_id: str = Field(..., min_length=1, max_length=MAX_RUN_IDENTITY_LENGTH)

    @field_validator("materialization_id", mode="before")
    @classmethod
    def _strip_materialization_id(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("materialization_id must be a string")
        return value.strip()


class DirectUrlInput(_RunOccurrenceInput):
    """One concrete non-playlist URL with display-only client hints."""

    input_kind: Literal["direct_url"]
    url: str = Field(..., min_length=1, max_length=MAX_RUN_URL_LENGTH)
    source_kind: str | None = Field(default=None, min_length=1, max_length=64)
    display_metadata: CompactRunDisplayMetadata = Field(default_factory=CompactRunDisplayMetadata)

    @field_validator("url", "source_kind", mode="before")
    @classmethod
    def _strip_direct_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("direct URL fields must be strings")
        return value.strip()


class FileStubInput(_RunOccurrenceInput):
    """Metadata-only placeholder for bytes supplied after run creation."""

    input_kind: Literal["file_stub"]
    name: str = Field(..., min_length=1, max_length=255)
    content_type: str | None = Field(default=None, min_length=1, max_length=255)
    size_bytes: int = Field(..., ge=0, le=10 * 1024**4, strict=True)
    display_metadata: CompactRunDisplayMetadata = Field(default_factory=CompactRunDisplayMetadata)

    @field_validator("name", "content_type", mode="before")
    @classmethod
    def _strip_file_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("file metadata fields must be strings")
        return value.strip()


PlaylistIngestRunInput = Annotated[
    MaterializedPlaylistItemInput | DirectUrlInput | FileStubInput,
    Field(discriminator="input_kind"),
]


class PlaylistIngestNewCollection(BaseModel):
    """Bounded metadata for one optional server-created playlist collection."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, min_length=1, max_length=2000)
    source_url: str | None = Field(default=None, min_length=1, max_length=2048)
    default_tags: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("name", "description", "source_url", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("collection text fields must be strings")
        return value.strip()

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = urlparse(value)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("collection source_url must be a credential-free HTTP URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("collection source_url must be a credential-free HTTP URL")
        if parsed.fragment:
            raise ValueError("collection source_url must not contain a fragment")
        sensitive_parts = ("auth", "cookie", "credential", "key", "password", "secret", "signature", "token")
        if any(
            any(part in key.casefold() for part in sensitive_parts)
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise ValueError("collection source_url must not contain credential-like query keys")
        return value

    @field_validator("default_tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("default_tags must be a list")
        if len(value) > 50:
            raise ValueError("default_tags cannot contain more than 50 entries")
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in value:
            if type(tag) is not str or not tag.strip() or len(tag.strip()) > 100:
                raise ValueError("default_tags must contain strings between 1 and 100 characters")
            clean = tag.strip()
            folded = clean.casefold()
            if folded not in seen:
                normalized.append(clean)
                seen.add(folded)
        return normalized


class PlaylistIngestRunCreateRequest(BaseModel):
    """Strict, bounded mixed-input manifest validated before run creation."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[PlaylistIngestRunInput] = Field(..., min_length=1, max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS)
    review_overrides: dict[str, ReviewOverrideEnvelope] = Field(
        default_factory=dict, max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS
    )
    processing_options: dict[str, Any] | None = Field(default=None, max_length=100)
    playlist_summaries: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS,
    )
    new_collection: PlaylistIngestNewCollection | None = None

    @field_validator("review_overrides", mode="before")
    @classmethod
    def _validate_override_keys(cls, value: object) -> object:
        if type(value) is not dict:
            raise ValueError("review_overrides must be an object")
        normalized: dict[str, object] = {}
        for occurrence_id, override in value.items():
            if type(occurrence_id) is not str:
                raise ValueError("review override keys must be strings")
            trimmed = occurrence_id.strip()
            if not trimmed or len(trimmed) > MAX_RUN_IDENTITY_LENGTH or trimmed != occurrence_id:
                raise ValueError("review override keys must be canonical occurrence IDs")
            normalized[trimmed] = override
        return normalized

    @field_validator("processing_options", mode="before")
    @classmethod
    def _validate_processing_options(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not dict:
            raise ValueError("processing_options must be an object")
        return _validate_bounded_json(value)

    @field_validator("playlist_summaries", mode="before")
    @classmethod
    def _validate_playlist_summaries(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not list or any(type(summary) is not dict for summary in value):
            raise ValueError("playlist_summaries must be a list of objects")
        return _validate_bounded_json(value)

    @model_validator(mode="after")
    def _validate_manifest(self) -> PlaylistIngestRunCreateRequest:
        occurrence_ids = [item.occurrence_id for item in self.inputs]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("occurrence_id values must be unique")
        return self


class DuplicateEvidence(BaseModel):
    """Safe owner-scoped evidence returned when Review must be repeated."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["library", "in_run", "none"]
    existing_media_id: int | None = Field(default=None, ge=1, strict=True)
    duplicate_of_occurrence_id: str | None = Field(default=None, max_length=MAX_RUN_IDENTITY_LENGTH)


class ReviewRequiredItem(BaseModel):
    """One deterministic occurrence-level Review correction."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str = Field(..., min_length=1, max_length=MAX_RUN_IDENTITY_LENGTH)
    reason: Literal[
        "duplicate_action_required",
        "duplicate_no_longer_present",
        "duplicate_target_changed",
        "invalid_duplicate_override",
        "unknown_review_override",
        "in_run_duplicate_requires_processing_or_skip",
    ]
    evidence: DuplicateEvidence
    allowed_actions: list[DuplicatePolicy] = Field(default_factory=list, max_length=4)


class RunItemSnapshot(BaseModel):
    """Current server snapshot for one immutable run occurrence."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str = Field(..., min_length=1, max_length=255)
    ordinal: int = Field(..., ge=1)
    state: RunItemState
    outcome: RunItemOutcome | None = None
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    progress_message: str | None = Field(default=None, max_length=1000)
    job_id: int | None = Field(default=None, ge=1)
    media_id: int | None = Field(default=None, ge=1)
    planned_collection_item_id: int | None = Field(default=None, ge=1)
    attempt: int = Field(default=1, ge=1)
    retryable: bool = False

    @field_validator("occurrence_id", mode="before")
    @classmethod
    def _strip_occurrence_id(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("occurrence_id must be a string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_terminal_outcome(self) -> RunItemSnapshot:
        if (self.state is RunItemState.TERMINAL) != (self.outcome is not None):
            raise ValueError("outcome is required exactly when state is terminal")
        return self


class PlaylistIngestRunItemResponse(BaseModel):
    """One authoritative owner-scoped run occurrence snapshot."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    input_kind: str
    source_url: str | None = None
    normalized_source_id: str | None = None
    source_kind: str | None = None
    display_metadata: dict[str, Any] = Field(default_factory=dict)
    action: str
    state: RunItemState
    outcome: RunItemOutcome | None = None
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    progress_message: str | None = Field(default=None, max_length=1000)
    job_id: int | None = Field(default=None, ge=1)
    batch_id: str | None = None
    media_id: int | None = Field(default=None, ge=1)
    planned_collection_item_id: int | None = Field(default=None, ge=1)
    attempt: int = Field(default=1, ge=1)
    retryable: bool = False

    @model_validator(mode="after")
    def _validate_outcome(self) -> PlaylistIngestRunItemResponse:
        if (self.state is RunItemState.TERMINAL) != (self.outcome is not None):
            raise ValueError("outcome is required exactly when state is terminal")
        return self


class PlaylistIngestProcessingOccurrence(BaseModel):
    """Authoritative occurrence fields needed for one bounded client submission."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    input_kind: str
    source_url: str | None = None
    source_kind: str | None = None
    display_metadata: dict[str, Any] = Field(default_factory=dict)
    state: Literal["staged", "awaiting_upload"]
    outcome: None = None
    job_id: None = None
    batch_id: None = None
    attempt: int = Field(default=1, ge=1)
    planned_collection_item_id: int | None = Field(default=None, ge=1)


class PlaylistIngestRunSummaryResponse(BaseModel):
    """Bounded aggregate snapshot for one owner-scoped ingest run."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    run_id: str
    status: str
    counts: dict[str, int]
    version: int = Field(..., ge=1)
    collection_id: int | None = Field(default=None, ge=1)
    batch_ids: list[str] = Field(default_factory=list, max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class PlaylistIngestRunCreateResponse(BaseModel):
    """Created run plus authoritative processing occurrences for client chunks."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    run_id: str
    status: str
    version: int = Field(..., ge=1)
    status_url: str
    items_url: str
    events_url: str
    processing_occurrences: list[PlaylistIngestProcessingOccurrence]


class PlaylistIngestRunItemsPageResponse(BaseModel):
    """Bounded immutable-order run occurrence page."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    run_id: str
    version: int = Field(..., ge=1)
    items: list[PlaylistIngestRunItemResponse]
    next_cursor: str | None = None


class PlaylistIngestRunCancelRequest(BaseModel):
    """Cancel selected run occurrences, or the whole run when omitted."""

    model_config = ConfigDict(extra="forbid")

    occurrence_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS,
    )
    reason: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("occurrence_ids", mode="before")
    @classmethod
    def _validate_cancel_occurrences(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not list:
            raise ValueError("occurrence_ids must be a list")
        normalized = []
        for occurrence_id in value:
            if type(occurrence_id) is not str:
                raise ValueError("occurrence_ids entries must be strings")
            candidate = occurrence_id.strip()
            if not candidate or len(candidate) > MAX_RUN_IDENTITY_LENGTH:
                raise ValueError("occurrence_ids entries are invalid")
            normalized.append(candidate)
        if len(set(normalized)) != len(normalized):
            raise ValueError("occurrence_ids must be unique")
        return normalized

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_reason(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("reason must be a string")
        return value.strip()


class PlaylistIngestRunRetryRequest(BaseModel):
    """Request deliberate retries for selected eligible occurrences."""

    model_config = ConfigDict(extra="forbid")

    occurrence_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS,
    )

    _validate_retry_occurrences = field_validator("occurrence_ids", mode="before")(
        PlaylistIngestRunCancelRequest._validate_cancel_occurrences.__func__
    )


class PlaylistIngestRunRetryResponse(BaseModel):
    """Occurrences that won the retry CAS and are ready for resubmission."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    run_id: str
    version: int = Field(..., ge=1)
    processing_occurrences: list[PlaylistIngestProcessingOccurrence]


class PlaylistPreflightCreateRequest(BaseModel):
    """Bounded asynchronous playlist inspection request."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=1, max_length=8192)
    max_items: int = Field(default=100, ge=1, le=500)
    timeout_seconds: int = Field(default=20, ge=1, le=60)

    @field_validator("url", mode="before")
    @classmethod
    def _strip_url(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("url must be a string")
        return value.strip()


class PlaylistPreflightLimits(BaseModel):
    """Safe admission limits advertised with an accepted resource."""

    model_config = ConfigDict(extra="forbid")

    max_items: int = Field(..., ge=1, le=500)
    global_capacity: int = Field(..., ge=1)
    owner_capacity: int = Field(..., ge=1)


class PlaylistPreflightAcceptedResponse(BaseModel):
    """Versioned response returned after durable resource/job binding."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    status: Literal["pending"] = "pending"
    status_url: str
    items_url: str
    expires_at: datetime
    limits: PlaylistPreflightLimits


class PlaylistPreflightSummaryResponse(BaseModel):
    """Owner-scoped asynchronous preflight status."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    status: Literal["pending", "running", "ready", "blocked", "cancelled", "expired"]
    source_url: str
    source_kind: str
    playlist_id: str | None = None
    summary: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class PlaylistPreflightItemResponse(BaseModel):
    """One immutable server-issued playlist occurrence."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    occurrence_index_for_source: int | None = Field(default=None, ge=1)
    source_url: str | None = None
    normalized_source_id: str | None = None
    source_kind: str
    availability: str | None = None
    duplicate_status: str | None = None
    duplicate_of_occurrence_id: str | None = None
    selected_by_default: bool | None = None
    display_metadata: dict[str, Any] = Field(default_factory=dict)


class PlaylistPreflightItemsPageResponse(BaseModel):
    """Bounded immutable preflight item page."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    items: list[PlaylistPreflightItemResponse]
    next_cursor: str | None = None


class PlaylistMaterializationCreateRequest(BaseModel):
    """Only server occurrence identities may be selected for materialization."""

    model_config = ConfigDict(extra="forbid")

    occurrence_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS,
    )

    @field_validator("occurrence_ids", mode="before")
    @classmethod
    def _validate_occurrence_ids(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("occurrence_ids must be a list")
        normalized: list[str] = []
        for occurrence_id in value:
            if type(occurrence_id) is not str:
                raise ValueError("occurrence_ids entries must be strings")
            trimmed = occurrence_id.strip()
            if not trimmed or len(trimmed) > 255:
                raise ValueError("occurrence_ids entries must be between 1 and 255 characters")
            normalized.append(trimmed)
        if len(set(normalized)) != len(normalized):
            raise ValueError("occurrence_ids must be unique")
        return normalized


class PlaylistMaterializationItemResponse(BaseModel):
    """Compact authoritative identity copied from a completed snapshot."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    source_url: str
    normalized_source_id: str | None = None
    source_kind: str
    display_metadata: dict[str, Any] = Field(default_factory=dict)


class PlaylistMaterializationResponse(BaseModel):
    """Owner-bound materialization for a staged Quick Ingest draft."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    materialization_id: str
    preflight_id: str
    status: Literal["ready"] = "ready"
    items: list[PlaylistMaterializationItemResponse]
    expires_at: datetime
