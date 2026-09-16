"""Strict H1 request and response envelopes for owner-backed selection endpoints."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


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
    ordered_path_ids: list[str]


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


class HistoryNodeV1(HistoryMessageRevisionV1):
    parent_id: str | None
    role: str
    settled: StrictBool
    conversation_id: str | None = None
    metadata: list[HistoryRequiredReferenceV1] = Field(default_factory=list)
    assets: list[HistoryRequiredReferenceV1] = Field(default_factory=list)


class HistorySelectionSnapshotV1(HistoryWireModel):
    version: Literal[1]
    owner_key: str
    conversation_id: str
    fences: HistoryFencesV1
    nodes: list[HistoryNodeV1]
    source_digest: str
    interpretation_status: HistoryInterpretationStatusV1
    storage_context_digest: str


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
    messages: list[HistoryMessageRevisionV1]
    fences: HistoryFencesV1
    storage_context_digest: str
    request_context_digest: str
    selection_digest: str


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
    messages: list[HistoryMessageRevisionV1]
    fences: HistoryFencesV1
    storage_context_digest: str
    request_context_digest: str
    selection_digest: str


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
    rows: list[HistoryNodeV1]
    view: HistoryViewSelectionV1
    purpose: Literal["send", "fork"]
    storage_context_digest: str


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
    source_members: list[HistoryMessageRevisionV1]
    ordered_path_ids: list[str]
    cursor: HistoryCursorV1
    selection_revision: StrictInt = Field(ge=0)


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
    messages: list[HistoryMessageRevisionV1]
    originating_selection_revision: StrictInt = Field(ge=0)
