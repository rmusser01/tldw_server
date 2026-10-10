"""Strict H1 request and response envelopes for owner-backed selection endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from tldw_Server_API.app.core.Chat.history_wire import canonical_history_json


def _wire_array(value: object) -> object:
    """Accept a JSON array while storing its validated members as a tuple."""

    return tuple(value) if isinstance(value, list) else value


class HistoryWireModel(BaseModel):
    """Reject unknown wire members and accidental mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AfterMessageCursorV1(HistoryWireModel):
    kind: Literal["after_message"]
    message_id: str = Field(min_length=1)


class BeforeMessageCursorV1(HistoryWireModel):
    kind: Literal["before_message"]
    message_id: str = Field(min_length=1)


class EmptyCursorV1(HistoryWireModel):
    kind: Literal["empty"]


HistoryCursorV1 = Annotated[
    AfterMessageCursorV1 | BeforeMessageCursorV1 | EmptyCursorV1,
    Field(discriminator="kind"),
]


class ParentGraphInterpretationV1(HistoryWireModel):
    kind: Literal["parent_graph_v1"]


class LegacyLinearInterpretationV1(HistoryWireModel):
    kind: Literal["legacy_linear_v1"]
    projection_id: str = Field(min_length=1)


HistoryInterpretationV1 = Annotated[
    ParentGraphInterpretationV1 | LegacyLinearInterpretationV1,
    Field(discriminator="kind"),
]


class LegacyReviewRequiredV1(HistoryWireModel):
    kind: Literal["legacy_review_required"]


class AcceptedLegacyInterpretationV1(LegacyLinearInterpretationV1):
    ordered_path_ids: tuple[str, ...]

    _freeze_path = field_validator("ordered_path_ids", mode="before")(_wire_array)


HistoryInterpretationStatusV1 = Annotated[
    ParentGraphInterpretationV1 | AcceptedLegacyInterpretationV1 | LegacyReviewRequiredV1,
    Field(discriminator="kind"),
]


class HistoryMessageRevisionV1(HistoryWireModel):
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class HistoryFencesV1(HistoryWireModel):
    conversation: str
    history: str
    settings: str


class HistoryRequiredReferenceV1(HistoryWireModel):
    id: str
    revision: str
    kind: str


class HistoryComparisonMetadataV1(HistoryWireModel):
    cluster_id: str
    model_id: str | None
    common: StrictBool


class HistoryNodeV1(HistoryMessageRevisionV1):
    preview: str | None = Field(None, max_length=200)
    legacy_projection_id: str | None = None
    parent_id: str | None
    role: str
    settled: StrictBool
    conversation_id: str | None = None
    metadata: tuple[HistoryRequiredReferenceV1, ...] = ()
    assets: tuple[HistoryRequiredReferenceV1, ...] = ()
    comparison: HistoryComparisonMetadataV1 | None = None

    _freeze_references = field_validator("metadata", "assets", mode="before")(_wire_array)


class HistorySelectedContentV1(HistoryMessageRevisionV1):
    """Composer input bound by ID, order and revision to selected manifest rows."""

    message: str
    images: tuple[str, ...]
    tool_calls: list[dict[str, Any]] | None = None
    extra_metadata: dict[str, Any] | None = None

    _freeze_images = field_validator("images", mode="before")(_wire_array)


class NativeForkContextV1(HistoryWireModel):
    """Digest-bound eligibility for the deliberately limited legacy plain copier."""

    policy: Literal["plain_v1"]
    storage_context_digest: str = Field(min_length=1)
    supported: StrictBool


class HistorySelectionSnapshotV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str = Field(min_length=1)
    conversation_id: str
    fences: HistoryFencesV1
    nodes: tuple[HistoryNodeV1, ...]
    source_digest: str
    interpretation_status: HistoryInterpretationStatusV1
    storage_context_digest: str
    native_fork_context: NativeForkContextV1 | None = None

    _freeze_nodes = field_validator("nodes", mode="before")(_wire_array)

    @model_validator(mode="after")
    def validate_native_fork_context(self) -> HistorySelectionSnapshotV1:
        """Do not accept an eligibility proof from another context capture."""
        if self.native_fork_context and self.native_fork_context.storage_context_digest != self.storage_context_digest:
            raise ValueError("native_fork_context_mismatch")
        return self


class HistoryViewSelectionV1(HistoryWireModel):
    view_session_id: str
    owner_key: str = Field(min_length=1)
    conversation_id: str
    interpretation: HistoryInterpretationV1
    cursor: HistoryCursorV1
    selection_revision: StrictInt = Field(ge=0)


class HistorySelectionV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str = Field(min_length=1)
    conversation_id: str
    interpretation: HistoryInterpretationV1
    cursor: HistoryCursorV1
    selection_revision: StrictInt = Field(ge=0)
    purpose: Literal["send", "fork"]
    messages: tuple[HistoryMessageRevisionV1, ...]
    fences: HistoryFencesV1
    storage_context_digest: str
    request_context_digest: str
    selection_digest: str

    _freeze_messages = field_validator("messages", mode="before")(_wire_array)


class HistorySelectionEnvelopeV1(HistoryWireModel):
    """Finalized selection submitted with an owner admission request."""

    selection: HistorySelectionV1


class CompareHistorySelectionV1(HistoryWireModel):
    """One model's semantic comparison projection and tagged digest."""

    version: Literal[1]
    owner_key: str = Field(min_length=1)
    conversation_id: str
    model_id: str = Field(min_length=1)
    cluster_id: str | None
    cursor: HistoryCursorV1
    messages: tuple[HistoryMessageRevisionV1, ...]
    fences: HistoryFencesV1
    storage_context_digest: str
    request_context_digest: str
    selection_digest: str

    _freeze_messages = field_validator("messages", mode="before")(_wire_array)


class NormalForkInputV1(HistoryWireModel):
    kind: Literal["normal"]
    selection: HistorySelectionV1


class ComparisonForkInputV1(HistoryWireModel):
    kind: Literal["comparison"]
    selection: CompareHistorySelectionV1


ForkInputV1 = Annotated[NormalForkInputV1 | ComparisonForkInputV1, Field(discriminator="kind")]


class ForkRequestV1(HistoryWireModel):
    operation_id: str = Field(min_length=1)
    owner_key: str = Field(min_length=1)
    destination_owner_key: str = Field(min_length=1)
    request_digest: str
    input: ForkInputV1


class HistoryCaptureViewV1(HistoryViewSelectionV1):
    """Only a fresh read-only bootstrap may omit its owner namespace."""

    owner_key: str | None = Field(None, min_length=1)


class HistoryCaptureRequestV1(HistoryWireModel):
    view: HistoryCaptureViewV1
    purpose: Literal["send", "fork"]


class CapturedHistoryV1(HistoryWireModel):
    status: Literal["captured"]
    snapshot: HistorySelectionSnapshotV1
    rows: tuple[HistoryNodeV1, ...]
    selected_content: tuple[HistorySelectedContentV1, ...]
    view: HistoryViewSelectionV1
    purpose: Literal["send", "fork"]
    storage_context_digest: str

    _freeze_rows = field_validator("rows", "selected_content", mode="before")(_wire_array)

    @model_validator(mode="after")
    def validate_content_binding(self) -> CapturedHistoryV1:
        """A strict capture cannot pair content with another manifest revision."""

        members = {row.id: row.revision for row in self.snapshot.nodes}
        if len(members) != len(self.snapshot.nodes):
            raise ValueError("duplicate_message_id")
        if len(self.rows) != len(self.selected_content) or len({row.id for row in self.rows}) != len(self.rows):
            raise ValueError("selected_content_mismatch")
        for row, content in zip(self.rows, self.selected_content):
            if members.get(row.id) != row.revision or row.id != content.id or row.revision != content.revision:
                raise ValueError("selected_content_mismatch")
        return self


HISTORY_BRANCH_FIELD_DESCRIPTION = (
    "[Extension] Branch intent for a `tldw_history_selection_v1` admission. It sits beside the "
    "selection and is not covered by its digest. `false`: the send must extend the latest message; "
    "if the selection's last message (or, for an empty selection, the conversation root) already has "
    "a live child, the owner refuses with 409 `{status: 'stale_selection', code: "
    "'history_branch_changed', conversation_id, parent_message_id, leaf_ids, history_version}` and "
    "writes nothing. `leaf_ids` are the live leaves below that message, oldest first. `true`: an "
    "explicit branch (for example Edit & resend), admitted beside any newer messages. Omitted: no "
    "leaf check (the pre-D7 behaviour). Replaying an already-admitted message id is unaffected."
)


class HistoryFailureV1(HistoryWireModel):
    status: Literal["legacy_review_required", "stale_selection", "invalid_history", "unsupported_history_capability"]
    code: str


class HistoryCaptureFailureV1(HistoryFailureV1):
    """Read failures retain the authenticated complete source for explicit review."""

    snapshot: HistorySelectionSnapshotV1
    view: HistoryViewSelectionV1


HistoryCaptureResultV1 = Annotated[CapturedHistoryV1 | HistoryCaptureFailureV1, Field(discriminator="status")]


class LegacyHistoryProjectionConfirmV1(HistoryWireModel):
    version: Literal[1]
    projection_id: str = Field(min_length=1)
    owner_key: str = Field(min_length=1)
    conversation_id: str
    source_digest: str
    fences: HistoryFencesV1
    source_members: tuple[HistoryMessageRevisionV1, ...]
    ordered_path_ids: tuple[str, ...]
    cursor: HistoryCursorV1
    selection_revision: StrictInt = Field(ge=0)

    _freeze_members = field_validator("source_members", "ordered_path_ids", mode="before")(_wire_array)


class LegacyProjectionConfirmEnvelopeV1(HistoryWireModel):
    confirmation: LegacyHistoryProjectionConfirmV1


class LegacyHistoryProjectionV1(LegacyHistoryProjectionConfirmV1):
    projection_digest: str
    created_at: str


class HistoryAdmissionReferenceV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str = Field(min_length=1)
    conversation_id: str
    input_message_id: str
    input_message_revision: str
    selection_digest: str


class HistoryAdmissionV1(HistoryAdmissionReferenceV1):
    messages: tuple[HistoryMessageRevisionV1, ...]
    originating_selection_revision: StrictInt = Field(ge=0)

    _freeze_messages = field_validator("messages", mode="before")(_wire_array)


def _version_one(value: object) -> object:
    """Do not let JSON booleans or floats masquerade as a wire version."""
    if type(value) is not int or value != 1:
        raise ValueError("unsupported history wire version")
    return value


HistoryVersionV1 = Annotated[Literal[1], BeforeValidator(_version_one)]
Hex64 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
SafeInt = Annotated[StrictInt, Field(ge=0, le=9007199254740991)]


class SourceLinesV1(HistoryWireModel):
    """Inert, ordered source line locator."""

    model_config = ConfigDict(serialize_by_alias=True)
    from_: SafeInt = Field(alias="from")
    to: SafeInt

    @model_validator(mode="after")
    def validate_order(self) -> SourceLinesV1:
        """Reject reversed source line ranges while preserving exact locators."""
        if self.from_ > self.to:
            raise ValueError("source line range is reversed")
        return self


class SourceLocationV1(HistoryWireModel):
    """The only supported source location projection."""

    lines: SourceLinesV1


class SourceMetadataV1(HistoryWireModel):
    """Allowlisted display metadata; optional means absent, never explicit null."""

    source: str = Field(default=None)
    title: str = Field(default=None)
    chunk_id: str = Field(default=None)
    retrieval_strategy: str = Field(default=None)
    source_type: str = Field(default=None)
    selection_reason: str = Field(default=None)
    score: StrictInt | StrictFloat = Field(default=None, allow_inf_nan=False)
    page: SafeInt = Field(default=None)
    media_id: str = Field(default=None)
    author: str = Field(default=None)
    chunk_index: SafeInt = Field(default=None)
    total_chunks: StrictInt = Field(default=None, ge=1, le=9007199254740991)
    start_char: SafeInt = Field(default=None)
    end_char: SafeInt = Field(default=None)
    chunk_start: SafeInt = Field(default=None)
    chunk_end: SafeInt = Field(default=None)
    loc: SourceLocationV1 = Field(default=None)

    @field_validator("score", mode="before")
    @classmethod
    def validate_json_number(cls, value: object) -> object:
        """Reject nonfinite or nonnumeric scores and normalize unsafe integers."""
        if type(value) not in (int, float):
            raise ValueError("source score must be a finite JSON number")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("source score exceeds finite Number domain") from exc
        canonical_history_json(number)
        return number if type(value) is int and abs(value) > 9007199254740991 else value

    @field_validator(
        "source", "title", "chunk_id", "retrieval_strategy", "source_type", "selection_reason", "media_id", "author"
    )
    @classmethod
    def validate_strings(cls, value: str, info: Any) -> str:
        """Require nonblank metadata within each field's UTF-8 byte budget."""
        limit = {"chunk_id": 512, "media_id": 512, "retrieval_strategy": 128, "source_type": 128}.get(
            info.field_name, 1000
        )
        if not value.strip() or len(value.encode("utf-8")) > limit:
            raise ValueError("source metadata string is blank or exceeds UTF-8 budget")
        return value

    @model_validator(mode="after")
    def validate_locators(self) -> SourceMetadataV1:
        """Preserve complete ordered ranges and in-bounds chunk indices."""
        if self.chunk_index is not None and self.total_chunks is not None and self.chunk_index >= self.total_chunks:
            raise ValueError("source chunk index exceeds total chunks")
        for start, end in ((self.start_char, self.end_char), (self.chunk_start, self.chunk_end)):
            if (start is None) != (end is None) or (start is not None and start > end):
                raise ValueError("source character range is incomplete or reversed")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_members(self, handler: Any):
        """Omit wire nulls without a return annotation that replaces the model schema."""
        return {key: value for key, value in handler(self).items() if value is not None}


class SourceV1(HistoryWireModel):
    """Bounded exact projection of an existing RAG presentation source."""

    name: str
    type: str
    mode: Literal["rag"]
    url: str
    pageContent: str = Field(max_length=1000)
    metadata: SourceMetadataV1

    @field_validator("name", "type", "url", "pageContent")
    @classmethod
    def validate_strings(cls, value: str, info: Any) -> str:
        """Bound source text by UTF-8 bytes, allowing only the URL to be blank."""
        limits = {"name": 1000, "type": 128, "url": 2048, "pageContent": 4000}
        if (info.field_name != "url" and not value.strip()) or len(value.encode("utf-8")) > limits[info.field_name]:
            raise ValueError("source string is blank or exceeds UTF-8 budget")
        return value


class HistoryResultPayloadV1(HistoryWireModel):
    """Complete bounded source projection, required even for plain sends."""

    model_config = ConfigDict(
        json_schema_extra={
            "x-tldw-source-bounds": {
                "sources": 20,
                "excerpt_scalars": 1000,
                "excerpt_utf8_bytes": 4000,
                "aggregate_excerpt_scalars": 16000,
                "name_utf8_bytes": 1000,
                "metadata_text_utf8_bytes": 1000,
                "chunk_id_utf8_bytes": 512,
                "compact_label_utf8_bytes": 128,
                "url_utf8_bytes": 2048,
                "result_canonical_utf8_bytes": 65536,
                "distinct_excerpts": True,
                "media_id_utf8_bytes": 512,
                "paired_character_ranges": True,
                "chunk_index_within_total": True,
            }
        }
    )
    version: HistoryVersionV1
    sources: tuple[SourceV1, ...] = Field(max_length=20)

    _freeze_sources = field_validator("sources", mode="before")(_wire_array)

    @model_validator(mode="after")
    def validate_source_budgets(self) -> HistoryResultPayloadV1:
        """Reject duplicate excerpts and enforce aggregate text and wire budgets."""
        excerpts = [source.pageContent for source in self.sources]
        if len(set(excerpts)) != len(excerpts):
            raise ValueError("duplicate source excerpt cannot preserve evidence marker mapping")
        if sum(map(len, excerpts)) > 16000:
            raise ValueError("total source excerpt scalar budget exceeded")
        payload = {"version": self.version, "sources": [source.model_dump(mode="json") for source in self.sources]}
        if len(canonical_history_json(payload).encode("utf-8")) > 65536:
            raise ValueError("canonical result payload exceeds 65536 UTF-8 bytes")
        return self


class DurableHistorySelectionV1(HistoryWireModel):
    """First server-owned single-input admission, bound to a send selection."""

    version: HistoryVersionV1
    kind: Literal["selection"]
    selection: HistorySelectionV1

    @model_validator(mode="after")
    def validate_send(self) -> DurableHistorySelectionV1:
        """Require send authority bound to a lowercase SHA-256 request digest."""
        if self.selection.purpose != "send":
            raise ValueError("selected durable history requires purpose=send")
        if len(self.selection.request_context_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.selection.request_context_digest
        ):
            raise ValueError("selected durable request_context_digest must be Hex64")
        return self


class DurableHistoryAdmissionV1(HistoryWireModel):
    """Accepted-reference Retry with a fresh attempt request-context digest."""

    version: HistoryVersionV1
    kind: Literal["admission"]
    admission: HistoryAdmissionReferenceV1
    request_context_digest: Hex64


DurableHistoryV1 = Annotated[DurableHistorySelectionV1 | DurableHistoryAdmissionV1, Field(discriminator="kind")]


class HistoryResultMetadataV1(HistoryResultPayloadV1):
    """Allowlisted result intent stored alongside exact assistant text."""

    request_context_digest: Hex64


class HistoryResultV1(HistoryResultMetadataV1):
    """Server result receipt, meaningful only after protected settlement."""

    result_message_id: UUID = Field(strict=False, json_schema_extra={"format": "uuid"})
    result_message_revision: Literal["1"]
    admission: HistoryAdmissionReferenceV1

    @field_serializer("result_message_id")
    def serialize_result_id(self, value: UUID) -> str:
        """Emit the server-owned result UUID as its canonical wire string."""
        return str(value)


class WorkspaceHistoryScopeV1(HistoryWireModel):
    """An explicitly pinned workspace namespace."""

    scope_type: Literal["workspace"]
    workspace_id: str = Field(min_length=1)

    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, value: str) -> str:
        """Require a nonblank UTF-8 workspace ID without altering its identity."""
        if not value.strip():
            raise ValueError("workspace_id must be nonblank")
        value.encode("utf-8")
        return value


class GlobalHistoryScopeV1(HistoryWireModel):
    """Global scope requires an explicit null workspace identifier."""

    scope_type: Literal["global"]
    workspace_id: None


HistoryScopeV1 = Annotated[WorkspaceHistoryScopeV1 | GlobalHistoryScopeV1, Field(discriminator="scope_type")]


class HistoryInputVerifiedV1(HistoryWireModel):
    """Owner-read projection of a live protected input admission."""

    version: HistoryVersionV1
    status: Literal["input_verified"]
    scope: HistoryScopeV1
    admission: HistoryAdmissionV1


class HistoryResultVerifiedV1(HistoryWireModel):
    """Owner-read projection of a live protected assistant result."""

    version: HistoryVersionV1
    status: Literal["result_verified"]
    scope: HistoryScopeV1
    result: HistoryResultV1


class HistoryUnverifiedV1(HistoryWireModel):
    """An observation without receipt or write authority."""

    version: HistoryVersionV1
    status: Literal["unverified"]
    code: Literal["no_protected_binding", "live_state_mismatch", "unsupported_projection"]


HistoryRecoveryReadV1 = Annotated[
    HistoryInputVerifiedV1 | HistoryResultVerifiedV1 | HistoryUnverifiedV1,
    Field(discriminator="status"),
]
