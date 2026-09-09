"""Bounded independent Buddy profile and attachment contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ResourceId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
BuddyName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class BuddyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StarterSource(BuddyModel):
    kind: Literal["starter"]
    starter_id: ResourceId


class PersonaPackSource(BuddyModel):
    kind: Literal["persona_pack"]
    persona_id: ResourceId
    pack_id: ResourceId


class BuddyCreate(BuddyModel):
    name: BuddyName
    source: Annotated[StarterSource | PersonaPackSource, Field(discriminator="kind")]
    optional_persona_id: ResourceId | None = None
    display_mode: Literal["dynamic", "static"] = "dynamic"


class BuddyUpdate(BuddyModel):
    expected_version: int = Field(ge=1)
    name: BuddyName | None = None
    optional_persona_id: ResourceId | None = None
    display_mode: Literal["dynamic", "static"] | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> BuddyUpdate:
        for field in ("name", "display_mode"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        if self.model_fields_set <= {"expected_version"}:
            raise ValueError("Provide at least one profile change")
        return self


class BuddyAsset(BuddyModel):
    id: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    checksum_sha256: str
    content_url: str


class BuddyProfile(BuddyModel):
    id: str
    name: str
    optional_persona_id: str | None
    optional_persona_available: bool
    display_mode: Literal["dynamic", "static"]
    version: int
    manifest: dict[str, Any]
    attribution: dict[str, Any]
    assets: list[BuddyAsset]


class BuddyList(BuddyModel):
    buddies: list[BuddyProfile]


class BuddyAttachment(BuddyModel):
    buddy_id: ResourceId
    scope_type: Literal["conversation", "workspace"]
    scope_id: ResourceId


class BuddyAttachmentUpdate(BuddyAttachment):
    expected_version: int = Field(ge=0)


class BuddyTarget(BuddyModel):
    title: str
    workspace_id: str | None


class BuddyAttachmentResponse(BuddyModel):
    client_slot: str
    version: int
    attachment: BuddyAttachment | None
    target: BuddyTarget | None = None
    unavailable_reason: Literal["target_unavailable", "buddy_unavailable"] | None = None


class BuddyConversationSummary(BuddyModel):
    id: str
    title: str
    scope_type: Literal["global", "workspace"]
    workspace_id: str | None
    version: int
    assistant_kind: str | None = None
    assistant_id: str | None = None
    assistant_name: str | None = None


class BuddyConversationList(BuddyModel):
    conversations: list[BuddyConversationSummary]
    limit: int
    offset: int


class BuddyResult(BuddyModel):
    id: str
    created_at: str
    content: str


class BuddyActivityItem(BuddyModel):
    conversation_id: str
    title: str
    workspace_id: str | None
    result: BuddyResult
    acknowledged: bool


class BuddyActivityList(BuddyModel):
    items: list[BuddyActivityItem]
    limit: int
    offset: int


class BuddyAcknowledgement(BuddyModel):
    conversation_id: ResourceId
    result_message_id: ResourceId
