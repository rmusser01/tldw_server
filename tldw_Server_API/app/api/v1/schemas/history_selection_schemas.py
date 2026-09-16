"""Strict H1 request and response envelopes for owner-backed selection endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


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

    _freeze_images = field_validator("images", mode="before")(_wire_array)


class HistorySelectionSnapshotV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str
    conversation_id: str
    fences: HistoryFencesV1
    nodes: tuple[HistoryNodeV1, ...]
    source_digest: str
    interpretation_status: HistoryInterpretationStatusV1
    storage_context_digest: str

    _freeze_nodes = field_validator("nodes", mode="before")(_wire_array)


class HistoryViewSelectionV1(HistoryWireModel):
    view_session_id: str
    owner_key: str
    conversation_id: str
    interpretation: HistoryInterpretationV1
    cursor: HistoryCursorV1
    selection_revision: StrictInt = Field(ge=0)


class HistorySelectionV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str
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
    owner_key: str
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
    owner_key: str
    destination_owner_key: str
    request_digest: str
    input: ForkInputV1


class HistoryCaptureRequestV1(HistoryWireModel):
    view: HistoryViewSelectionV1
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


class HistoryFailureV1(HistoryWireModel):
    status: Literal[
        "legacy_review_required", "stale_selection", "invalid_history", "unsupported_history_capability"
    ]
    code: str


HistoryCaptureResultV1 = Annotated[CapturedHistoryV1 | HistoryFailureV1, Field(discriminator="status")]


class LegacyHistoryProjectionConfirmV1(HistoryWireModel):
    version: Literal[1]
    projection_id: str = Field(min_length=1)
    owner_key: str
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
    owner_key: str
    conversation_id: str
    input_message_id: str
    input_message_revision: str
    selection_digest: str


class HistoryAdmissionV1(HistoryAdmissionReferenceV1):
    messages: tuple[HistoryMessageRevisionV1, ...]
    originating_selection_revision: StrictInt = Field(ge=0)

    _freeze_messages = field_validator("messages", mode="before")(_wire_array)
