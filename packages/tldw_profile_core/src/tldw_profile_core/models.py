from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, StrictBool, model_validator

from .canonical import (
    I_JSON_MAX_INTEGER,
    Confidence,
    JsonInteger,
    PortableDateTime,
    VersionOne,
    canonical_bytes,
)
from .enums import (
    AgentVisibility,
    ProposalOperation,
    ProposalState,
    RecordKind,
    RecordState,
    ScopeKind,
    SyncMode,
)
from .payloads import (
    BoundedText,
    FrozenModel,
    ProfilePayload,
    reject_blank,
    reject_secret_material,
)

OpaqueId = Annotated[
    str, Field(min_length=1, max_length=128), AfterValidator(reject_blank)
]
ReasonCode = Annotated[
    str, Field(min_length=1, max_length=128), AfterValidator(reject_blank)
]
EvidenceSpan = Annotated[
    str, Field(min_length=1, max_length=1_000), AfterValidator(reject_blank)
]
Sha256Hash = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


def _timestamps_are_ordered(created_at: datetime, updated_at: datetime) -> bool:
    return (
        created_at.tzinfo is not None
        and updated_at.tzinfo is not None
        and created_at <= updated_at
    )


class ProvenanceSource(StrEnum):
    MANUAL = "manual"
    AGENT = "agent"
    IMPORT = "import"
    MIGRATION = "migration"


class ActorType(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class SemanticKey(FrozenModel):
    namespace: BoundedText
    subject: BoundedText


class ProfileControls(FrozenModel):
    sync_mode: SyncMode
    agent_visibility: AgentVisibility


class ProfileProvenance(FrozenModel):
    source: ProvenanceSource
    actor: ActorType
    reason_code: ReasonCode
    source_references: tuple[OpaqueId, ...] = Field(default=(), max_length=32)
    source_hashes: tuple[Sha256Hash, ...] = Field(default=(), max_length=32)
    derived_from_record_id: OpaqueId | None = None


class ProfileManifest(FrozenModel):
    schema_version: VersionOne = 1
    profile_id: OpaqueId
    revision: JsonInteger = Field(ge=0, le=I_JSON_MAX_INTEGER)
    purge_generation: JsonInteger = Field(ge=0, le=I_JSON_MAX_INTEGER)
    created_at: PortableDateTime
    updated_at: PortableDateTime
    current_version_id: OpaqueId

    @model_validator(mode="after")
    def validate_timestamps(self):
        if not _timestamps_are_ordered(self.created_at, self.updated_at):
            raise ValueError("timestamps must be timezone-aware and ordered")
        return self


class ProfileScope(FrozenModel):
    schema_version: VersionOne = 1
    scope_id: OpaqueId
    profile_id: OpaqueId
    kind: ScopeKind
    version_id: OpaqueId
    created_at: PortableDateTime
    updated_at: PortableDateTime

    @model_validator(mode="after")
    def validate_timestamps(self):
        if not _timestamps_are_ordered(self.created_at, self.updated_at):
            raise ValueError("timestamps must be timezone-aware and ordered")
        return self


class ProfileRecord(FrozenModel):
    schema_version: VersionOne = 1
    profile_id: OpaqueId
    record_id: OpaqueId
    scope_id: OpaqueId
    kind: RecordKind
    payload: ProfilePayload | None = None
    semantic_key: SemanticKey | None = None
    state: RecordState
    controls: ProfileControls
    provenance: ProfileProvenance
    version_id: OpaqueId
    parent_version_id: OpaqueId | None
    created_at: PortableDateTime
    updated_at: PortableDateTime
    expires_at: PortableDateTime | None = None
    no_expiry: StrictBool = False

    @model_validator(mode="after")
    def validate_record(self):
        if not _timestamps_are_ordered(self.created_at, self.updated_at):
            raise ValueError("timestamps must be timezone-aware and ordered")
        if self.expires_at is not None and (
            self.expires_at.tzinfo is None or self.expires_at <= self.updated_at
        ):
            raise ValueError("expiry must be timezone-aware and later than the record")
        if self.state is RecordState.DELETED:
            if any(
                (
                    self.payload is not None,
                    self.semantic_key is not None,
                    self.expires_at is not None,
                    self.no_expiry,
                )
            ):
                raise ValueError("deleted tombstone must be content-free")
            return self
        if self.payload is None or self.payload.kind != self.kind.value:
            raise ValueError("payload kind mismatch")
        if self.kind is RecordKind.WORKING_CONTEXT:
            if (self.expires_at is None) == (not self.no_expiry):
                raise ValueError("working context requires exactly one expiry decision")
        elif self.expires_at is not None or self.no_expiry:
            raise ValueError("expiry is only valid for working context")
        if len(canonical_bytes(self.payload)) > 16 * 1024:
            raise ValueError("payload exceeds 16 KiB canonical limit")
        return self


class ProfileProposal(FrozenModel):
    schema_version: VersionOne = 1
    proposal_id: OpaqueId
    profile_id: OpaqueId
    scope_id: OpaqueId
    operation: ProposalOperation
    target_record_id: OpaqueId | None
    base_version_id: OpaqueId | None
    proposed_record: ProfileRecord | None
    provenance: ProfileProvenance
    confidence: Confidence | None = None
    state: ProposalState = ProposalState.PENDING
    created_at: PortableDateTime
    expires_at: PortableDateTime

    @model_validator(mode="after")
    def validate_proposal(self):
        if (
            not _timestamps_are_ordered(self.created_at, self.expires_at)
            or self.expires_at <= self.created_at
        ):
            raise ValueError("proposal timestamps must be timezone-aware and ordered")
        if self.state is not ProposalState.PENDING:
            if self.proposed_record is not None or self.confidence is not None:
                raise ValueError("resolved proposals are content-free receipts")
        target = self.target_record_id is not None
        base = self.base_version_id is not None
        content = self.proposed_record is not None
        expected = {
            ProposalOperation.CREATE: (
                False,
                False,
                self.state is ProposalState.PENDING,
            ),
            ProposalOperation.UPDATE: (True, True, self.state is ProposalState.PENDING),
            ProposalOperation.ARCHIVE: (True, True, False),
            ProposalOperation.PROMOTE: (True, True, False),
        }[self.operation]
        if (
            self.state is ProposalState.PENDING
            and self.expires_at != self.created_at + timedelta(days=90)
        ):
            raise ValueError("pending proposal expiry must be exactly 90 days")
        if (target, base, content) != expected:
            raise ValueError(f"invalid {self.operation.value} proposal shape")
        if self.proposed_record is None:
            return self
        if (
            self.state is ProposalState.PENDING
            and self.operation in (ProposalOperation.CREATE, ProposalOperation.UPDATE)
            and (
                self.proposed_record.state is not RecordState.ACTIVE
                or self.proposed_record.payload is None
            )
        ):
            raise ValueError(
                "pending create and update proposals require active content"
            )
        if (
            self.proposed_record.profile_id != self.profile_id
            or self.proposed_record.scope_id != self.scope_id
        ):
            raise ValueError("proposal identity mismatch")
        if self.operation is ProposalOperation.CREATE:
            if self.proposed_record.parent_version_id is not None:
                raise ValueError("create proposal cannot have a base version")
        elif self.proposed_record.record_id != self.target_record_id:
            raise ValueError("proposal identity mismatch")
        elif self.proposed_record.parent_version_id != self.base_version_id:
            raise ValueError("proposal base mismatch")
        return self


class ProfileSearchRequest(FrozenModel):
    query: BoundedText
    limit: JsonInteger = Field(default=5, ge=1, le=20)


class ProfileGetRequest(FrozenModel):
    record_id: OpaqueId


class ProfileProposeRequest(FrozenModel):
    operation: Literal[
        ProposalOperation.CREATE, ProposalOperation.UPDATE, ProposalOperation.ARCHIVE
    ]
    target_record_id: OpaqueId | None = None
    base_version_id: OpaqueId | None = None
    proposed_payload: ProfilePayload | None = None
    evidence_span: EvidenceSpan | None = None
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def validate_secrets(self):
        if self.evidence_span is not None:
            reject_secret_material(self.evidence_span)
        if self.proposed_payload is not None:
            reject_secret_material(str(self.proposed_payload.model_dump()))
        return self

    @model_validator(mode="after")
    def validate_shape(self):
        target = self.target_record_id is not None
        base = self.base_version_id is not None
        content = self.proposed_payload is not None
        expected = {
            ProposalOperation.CREATE: (False, False, True),
            ProposalOperation.UPDATE: (True, True, True),
            ProposalOperation.ARCHIVE: (True, True, False),
        }[self.operation]
        if (target, base, content) != expected:
            raise ValueError(f"invalid {self.operation.value} proposal request")
        return self


class ProfileUpdateRequest(FrozenModel):
    record_id: OpaqueId
    base_version_id: OpaqueId
    current_user_message_id: OpaqueId
    evidence_span: EvidenceSpan
    proposed_payload: ProfilePayload

    @model_validator(mode="after")
    def validate_secrets(self):
        reject_secret_material(self.evidence_span)
        reject_secret_material(str(self.proposed_payload.model_dump()))
        return self


class ProfilePromoteRequest(FrozenModel):
    source_record_id: OpaqueId
    base_version_id: OpaqueId
