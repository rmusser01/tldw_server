"""Strict immutable native fork/retention protocol values (no route registration)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, field_serializer, field_validator, model_serializer, model_validator

from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import (
    HistoryCaptureViewV1,
    HistoryWireModel,
    NormalForkInputV1,
    _wire_array,
)

NativeId = Annotated[str, Field(min_length=1, max_length=256)]
NativeRevision = Annotated[str, Field(min_length=1, max_length=128)]
NativeOwnerKey = Annotated[str, Field(min_length=1, max_length=1024)]
NativeHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NativeOperationId = Annotated[str, Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")]
NativeCode = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")]
NativeFidelity = Literal["strict", "allow_unavailable"]
NativeRepresentation = Literal[
    "embedded_image_v1", "document_original_v1", "document_text_v1", "generated_image_revision_v1", "unavailable_v1"
]


class NativeScopeV1(HistoryWireModel):
    """Canonical owner scope; global never accepts a workspace field."""

    kind: Literal["global", "workspace"]
    workspace_id: NativeId | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_scope(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if value.get("kind") == "global" and "workspace_id" in value:
                raise ValueError("global_scope_has_workspace")
            if value.get("kind") == "workspace" and not value.get("workspace_id"):
                raise ValueError("workspace_scope_requires_id")
        return value

    @model_serializer
    def serialize_scope(self) -> dict[str, str | None]:
        """Keep the global wire object closed (no null workspace member)."""
        return {"kind": "global"} if self.kind == "global" else {"kind": "workspace", "workspace_id": self.workspace_id}


class NativeAssetManifestEntryV1(HistoryWireModel):
    """An exact immutable native reference; external paths are never authority."""

    reference_id: NativeId
    reference_revision: NativeRevision
    asset_id: NativeId
    revision: NativeRevision
    representation: NativeRepresentation
    role: Literal["message_image", "document_context", "generated_image", "reference_image"]
    context_enabled: StrictBool
    hash: NativeHash | None
    fidelity_disposition: Literal["retained", "unavailable"]

    @model_validator(mode="after")
    def validate_representation(self) -> NativeAssetManifestEntryV1:
        if self.representation == "unavailable_v1":
            if self.hash is not None or self.fidelity_disposition != "unavailable":
                raise ValueError("invalid_unavailable_asset")
        else:
            allowed = {
                "message_image": {"embedded_image_v1"},
                "document_context": {"document_original_v1", "document_text_v1"},
                "generated_image": {"generated_image_revision_v1"},
                "reference_image": {"embedded_image_v1", "generated_image_revision_v1"},
            }
            if (
                self.hash is None
                or self.fidelity_disposition != "retained"
                or self.representation not in allowed[self.role]
            ):
                raise ValueError("invalid_retained_asset")
        return self


class NativeForkBindingTemplateV1(HistoryWireModel):
    """Validated accepted presentation, before allocation of any child identity."""

    mode: Literal["snapshot_v1"]
    primary_participant_id: NativeId
    display_name: str = Field(min_length=1, max_length=1024)
    snapshot_schema: Literal["1"]
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NativeAssistantBindingV1(NativeForkBindingTemplateV1):
    """Commit-time child identity tied to its retained assistant snapshot."""

    child_id: NativeId
    assistant_id: NativeId
    settings_revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_child_identity(self) -> NativeAssistantBindingV1:
        """Reject a child ID that does not name its immutable snapshot assistant."""
        if self.assistant_id != "snapshot:" + self.child_id:
            raise ValueError("invalid_snapshot_child_identity")
        return self


class NativeForkCaptureRequestV1(HistoryWireModel):
    """Read-only native fork capture request for a selected history view."""

    view: HistoryCaptureViewV1
    fidelity: NativeFidelity = "strict"


class _NativeForkSkeletonV1(HistoryWireModel):
    """Canonical reviewed request skeleton; allocation/title/digest come later."""

    protocol: Literal["native_atomic_v1"]
    projection_version: Literal["native-fork-v1"]
    owner_key: NativeOwnerKey
    scope: NativeScopeV1
    source_conversation_id: NativeId
    input: NormalForkInputV1
    retained_context_digest: NativeHash
    asset_manifest: tuple[NativeAssetManifestEntryV1, ...]
    fidelity: NativeFidelity
    _freeze_arrays = field_validator("asset_manifest", mode="before")(_wire_array)

    @model_validator(mode="after")
    def validate_selection(self) -> _NativeForkSkeletonV1:
        selection = self.input.selection
        if (
            selection.purpose != "fork"
            or selection.owner_key != self.owner_key
            or selection.conversation_id != self.source_conversation_id
        ):
            raise ValueError("fork_selection_mismatch")
        identifiers = [member.id for member in selection.messages]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate_message_id")
        references = [asset.reference_id for asset in self.asset_manifest]
        if len(references) != len(set(references)):
            raise ValueError("duplicate_asset_reference")
        if self.fidelity == "strict" and any(
            asset.fidelity_disposition == "unavailable" for asset in self.asset_manifest
        ):
            raise ValueError("strict_fidelity_requires_assets")
        values = [
            *identifiers,
            getattr(selection.cursor, "message_id", ""),
            getattr(selection.interpretation, "projection_id", ""),
        ]
        if any(len(value) > 256 for value in values) or any(
            len(member.revision) > 128 for member in selection.messages
        ):
            raise ValueError("invalid_identifier_length")
        return self


class NativeForkCaptureV1(_NativeForkSkeletonV1):
    """Validated fork capture with explicit required effects and refusal reasons."""

    required_effects: tuple[NativeCode, ...] = ()
    reasons: tuple[NativeCode, ...] = ()

    _freeze_review = field_validator("required_effects", "reasons", mode="before")(_wire_array)


class NativeForkRequestV1(_NativeForkSkeletonV1):
    """Strict semantic operation input; capabilities/review annotations are not input."""

    operation_id: NativeOperationId
    child_title: str = Field(min_length=1, max_length=200)
    request_digest: NativeHash


class NativeForkResolveRequestV1(HistoryWireModel):
    """Owner-bound lookup for a previously reserved native fork operation."""

    protocol: Literal["native_atomic_v1"]
    projection_version: Literal["native-fork-v1"]
    owner_key: NativeOwnerKey
    scope: NativeScopeV1
    request_digest: NativeHash


class NativeOperationBaseV1(HistoryWireModel):
    """Shared immutable identity returned with every native operation receipt."""

    operation_id: NativeOperationId
    owner_key: NativeOwnerKey
    protocol: Literal["native_atomic_v1"] = "native_atomic_v1"
    projection_version: Literal["native-fork-v1"] = "native-fork-v1"


class NativeForkCommittedV1(NativeOperationBaseV1):
    """Committed child identity and one-to-one source-to-child message map."""

    operation_kind: Literal["native_fork_v1"] = "native_fork_v1"
    state: Literal["committed"]
    child_id: NativeId
    message_map: Mapping[NativeId, NativeId]

    @field_validator("message_map", mode="after")
    @classmethod
    def freeze_map(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Reject duplicate child IDs and freeze the accepted map."""
        if len(set(value.values())) != len(value):
            raise ValueError("duplicate_child_message")
        return MappingProxyType(dict(value))

    @field_serializer("message_map")
    def serialize_map(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Emit a plain JSON-compatible copy of the immutable message map."""
        return dict(value)


class NativeForkPendingV1(NativeOperationBaseV1):
    """Retryable operation receipt with a nonnegative retry delay."""

    operation_kind: Literal["native_fork_v1"] = "native_fork_v1"
    state: Literal["pending"]
    retry_after_seconds: StrictInt = Field(ge=0)


class NativeForkTerminalV1(NativeOperationBaseV1):
    """Terminal operation receipt carrying a stable refusal code."""

    operation_kind: Literal["native_fork_v1"] = "native_fork_v1"
    state: Literal["not_recorded", "rejected", "expired", "gone"]
    code: NativeCode


NativeForkResultV1 = Annotated[
    NativeForkCommittedV1 | NativeForkPendingV1 | NativeForkTerminalV1, Field(discriminator="state")
]


class NativeAssetRetentionRequestV1(HistoryWireModel):
    """Immutable request to retain an exact source asset manifest."""

    operation_kind: Literal["native_asset_retention_v1"]
    operation_id: NativeOperationId
    owner_key: NativeOwnerKey
    scope: NativeScopeV1
    source_conversation_id: NativeId
    expected_asset_manifest_revision: NativeRevision
    asset_manifest: tuple[NativeAssetManifestEntryV1, ...]
    request_digest: NativeHash

    _freeze_manifest = field_validator("asset_manifest", mode="before")(_wire_array)


class NativeRetainedAssetV1(HistoryWireModel):
    """Allocated destination asset identity and revision."""

    asset_id: NativeId
    revision: NativeRevision


class NativeAssetRetentionCommittedV1(NativeOperationBaseV1):
    """Committed asset manifest revision and immutable reference mapping."""

    operation_kind: Literal["native_asset_retention_v1"] = "native_asset_retention_v1"
    state: Literal["committed"]
    asset_manifest_revision: NativeRevision
    asset_map: Mapping[NativeId, NativeRetainedAssetV1]

    @field_validator("asset_map", mode="after")
    @classmethod
    def freeze_map(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        """Freeze the committed reference-to-asset mapping."""
        return MappingProxyType(dict(value))

    @field_serializer("asset_map")
    def serialize_map(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Encode retained asset models as JSON-compatible values."""
        return {key: asset.model_dump(mode="json") for key, asset in value.items()}


class NativeAssetRetentionPendingV1(NativeForkPendingV1):
    """Pending receipt for an asset-retention operation."""

    operation_kind: Literal["native_asset_retention_v1"] = "native_asset_retention_v1"


class NativeAssetRetentionTerminalV1(NativeForkTerminalV1):
    """Terminal receipt for an asset-retention operation."""

    operation_kind: Literal["native_asset_retention_v1"] = "native_asset_retention_v1"


NativeAssetRetentionResultV1 = Annotated[
    NativeAssetRetentionCommittedV1 | NativeAssetRetentionPendingV1 | NativeAssetRetentionTerminalV1,
    Field(discriminator="state"),
]
NativeOperationResultV1 = Annotated[
    NativeForkResultV1 | NativeAssetRetentionResultV1, Field(discriminator="operation_kind")
]


class NativeAssetContextUpdateV1(HistoryWireModel):
    """CAS update for one retained asset's participation in prompt context."""

    expected_asset_manifest_revision: NativeRevision
    target_reference_id: NativeId
    expected_reference_revision: NativeRevision
    expected_asset_revision: NativeRevision
    action: Literal["remove_from_context", "replace", "restore"]
    replacement: NativeRetainedAssetV1 | None = None
    expected_original_hash: NativeHash | None = None

    @model_validator(mode="after")
    def validate_action(self) -> NativeAssetContextUpdateV1:
        """Require replacement fields only for replace or restore actions."""
        if self.action == "remove_from_context":
            if self.replacement is not None or self.expected_original_hash is not None:
                raise ValueError("invalid_remove_context")
        elif self.replacement is None or (self.action == "restore") != (self.expected_original_hash is not None):
            raise ValueError("invalid_asset_replacement")
        return self


class NativeForkLimitsV1(HistoryWireModel):
    """Effective existing destination limits, supplied by capability composition."""

    max_message_image_bytes: StrictInt = Field(gt=0)
    max_selected_image_bytes: StrictInt = Field(gt=0)
    max_document_bytes: StrictInt = Field(gt=0)


class NativeForkCapabilitiesV1(HistoryWireModel):
    """Qualification is explicit; merely importing the schema enables nothing."""

    protocol: Literal["native_atomic_v1"] = "native_atomic_v1"
    projection_version: Literal["native-fork-v1"] = "native-fork-v1"
    enabled: StrictBool = False
    behavior_versions: tuple[NativeCode, ...] = ()
    asset_representations: tuple[NativeRepresentation, ...] = ()
    limits: NativeForkLimitsV1

    _freeze_arrays = field_validator("behavior_versions", "asset_representations", mode="before")(_wire_array)
