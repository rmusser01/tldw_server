"""Pydantic schemas for VN Play runtime APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from tldw_Server_API.app.core.VN_Play.constants import (
    LINKED_CHAT_MODE_READ_ONLY_CONTEXT,
    SESSION_STATUS_ACTIVE,
    TRUST_LEVEL_LOCAL,
)

VNPlayMode = Literal["freeform", "story"]
VNPlaySessionStatus = Literal["active", "paused", "completed", "archived", "failed"]
VNPlayTrustLevel = Literal["local", "trusted_restore", "untrusted_import", "mixed"]
VNPlayLinkedChatMode = Literal["read_only_context"]
VNPlayTurnStatus = Literal[
    "pending",
    "model_calling",
    "model_failed",
    "parse_failed",
    "completed",
    "abandoned",
    "cancelled",
]
VNPlayEventType = Literal[
    "session_started",
    "turn_started",
    "turn_completed",
    "turn_failed",
    "user_turn",
    "model_turn",
    "choice_presented",
    "choice_selected",
    "scene_state_changed",
    "visual_directive_requested",
    "visual_directive_applied",
    "visual_directive_rejected",
    "model_turn_parse_failed",
    "safety_gate_triggered",
    "runtime_gate_failed",
    "session_settings_changed",
    "session_checkpoint_created",
    "session_restored",
]
VNPlayEventSource = Literal["user", "model", "runtime", "system"]


class VNPlaySessionCreate(BaseModel):
    """Request body for creating a VN Play session."""

    model_config = ConfigDict(extra="forbid")

    mode: VNPlayMode = Field(..., description="Runtime mode: freeform or story.")
    title: StrictStr = Field(..., min_length=1, max_length=500)
    primary_character_id: StrictInt = Field(..., ge=1)
    vn_asset_pack_id: StrictInt = Field(..., ge=1)
    additional_character_ids: list[StrictInt] = Field(default_factory=list)
    linked_chat_id: StrictStr | None = Field(default=None, min_length=1)
    asset_manifest_version: StrictStr | None = Field(default=None, min_length=1)
    source_world_book_ids: list[StrictInt] = Field(default_factory=list)
    content_rating: StrictStr = Field(default="general", min_length=1, max_length=100)
    trust_level: VNPlayTrustLevel = TRUST_LEVEL_LOCAL
    linked_chat_mode: VNPlayLinkedChatMode = LINKED_CHAT_MODE_READ_ONLY_CONTEXT
    seed: StrictStr | None = Field(default=None, min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)


class VNPlaySessionUpdate(BaseModel):
    """Request body for patching a VN Play session."""

    model_config = ConfigDict(extra="forbid")

    title: StrictStr | None = Field(default=None, min_length=1, max_length=500)
    status: VNPlaySessionStatus | None = None
    linked_chat_id: StrictStr | None = Field(default=None, min_length=1)
    linked_chat_mode: VNPlayLinkedChatMode | None = None
    settings: dict[str, Any] | None = None
    deleted: StrictBool | None = None


class VNPlayEventResponse(BaseModel):
    """Serialized VN Play event."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    sequence_number: StrictInt
    event_type: VNPlayEventType
    event_payload: dict[str, Any] = Field(default_factory=dict)
    source: VNPlayEventSource = "runtime"
    model_provider: str | None = None
    model_name: str | None = None
    branch_node_id: int | None = None
    created_at: str | None = None


class VNPlaySceneStateResponse(BaseModel):
    """Serialized derived VN Play scene state."""

    model_config = ConfigDict(from_attributes=True)

    session_id: StrictInt
    owner_user_id: StrictInt
    last_event_id: int | None = None
    current_background_item_id: int | None = None
    current_depth_item_id: int | None = None
    active_sprite_items: list[dict[str, Any]] = Field(default_factory=list)
    location_key: str | None = None
    mood: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    active_branch_node_id: int | None = None
    visible_choices: list[dict[str, Any]] = Field(default_factory=list)
    transcript_cursor: int | None = None
    scene_version: StrictInt = Field(default=0, ge=0)
    warnings: list[Any] = Field(default_factory=list)
    updated_at: str | None = None


class VNPlaySessionResponse(BaseModel):
    """Serialized VN Play session metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    owner_user_id: StrictInt
    mode: VNPlayMode
    title: str
    status: VNPlaySessionStatus = SESSION_STATUS_ACTIVE
    primary_character_id: StrictInt
    additional_character_ids: list[int] = Field(default_factory=list)
    linked_chat_id: str | None = None
    vn_asset_pack_id: StrictInt
    asset_manifest_version: str | None = None
    source_world_book_ids: list[int] = Field(default_factory=list)
    content_rating: str = "general"
    trust_level: VNPlayTrustLevel = TRUST_LEVEL_LOCAL
    linked_chat_mode: VNPlayLinkedChatMode = LINKED_CHAT_MODE_READ_ONLY_CONTEXT
    seed: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    scene_version: StrictInt = Field(default=0, ge=0)
    active_turn_request_id: int | None = None
    current_scene: VNPlaySceneStateResponse | None = None
    scene_state: VNPlaySceneStateResponse | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted: bool = False


class VNPlayTurnRequest(BaseModel):
    """Request body for advancing a VN Play session by one turn."""

    model_config = ConfigDict(extra="forbid")

    input_text: StrictStr | None = Field(default=None, min_length=1)
    choice_id: StrictStr | None = Field(default=None, min_length=1)
    custom_action: dict[str, Any] | None = None
    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)
    provider: StrictStr | None = Field(default=None, min_length=1)
    model: StrictStr | None = Field(default=None, min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_exactly_one_input(self) -> VNPlayTurnRequest:
        provided = [
            self.input_text is not None,
            self.choice_id is not None,
            self.custom_action is not None,
        ]
        if sum(provided) != 1:
            raise ValueError("exactly_one_turn_input_required")
        return self


class VNPlayTurnResponse(BaseModel):
    """Response for a submitted VN Play turn."""

    model_config = ConfigDict(from_attributes=True)

    turn_request_id: StrictInt
    status: VNPlayTurnStatus
    scene_version: StrictInt = Field(..., ge=0)
    replayed: bool = False
    session: VNPlaySessionResponse | None = None
    current_scene: VNPlaySceneStateResponse | None = None
    events: list[VNPlayEventResponse] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class VNPlayCheckpointCreate(BaseModel):
    """Request body for creating a named VN Play checkpoint."""

    model_config = ConfigDict(extra="forbid")

    label: StrictStr = Field(..., min_length=1, max_length=300)
    event_id: StrictInt | None = Field(default=None, ge=1)
    scene_version: StrictInt | None = Field(default=None, ge=0)


class VNPlayCheckpointResponse(BaseModel):
    """Serialized VN Play checkpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    label: str
    event_id: int | None = None
    scene_version: StrictInt
    scene_state_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class VNPlayRestoreRequest(BaseModel):
    """Request body for restoring a VN Play checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: StrictInt = Field(..., ge=1)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayRetryTurnRequest(BaseModel):
    """Request body for retrying the last VN Play user turn."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayBranchResponse(BaseModel):
    """Serialized VN Play branch node."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    parent_event_id: int | None = None
    branch_label: str | None = None
    branch_path: list[Any] = Field(default_factory=list)
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None


__all__ = [
    "VNPlayBranchResponse",
    "VNPlayCheckpointCreate",
    "VNPlayCheckpointResponse",
    "VNPlayEventResponse",
    "VNPlayRestoreRequest",
    "VNPlayRetryTurnRequest",
    "VNPlaySceneStateResponse",
    "VNPlaySessionCreate",
    "VNPlaySessionResponse",
    "VNPlaySessionUpdate",
    "VNPlayTurnRequest",
    "VNPlayTurnResponse",
    "VNPlayEventType",
    "VNPlayLinkedChatMode",
    "VNPlayMode",
    "VNPlaySessionStatus",
    "VNPlayTrustLevel",
    "VNPlayTurnStatus",
]
