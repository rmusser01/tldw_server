"""Bounded independent Buddy profile and attachment contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Buddy.buddy_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    BuddyCreate,
    BuddyModel,
    BuddyName,
    PersonaPackSource,
    ResourceId,
    StarterSource,
)


class BuddyUpdate(BuddyModel):
    """Stage profile changes against a positive optimistic-concurrency version.

    Omitted fields preserve saved values. Explicit null removes the optional
    Persona, while name and display mode must remain non-null when supplied."""

    expected_version: int = Field(ge=1)
    name: BuddyName | None = None
    optional_persona_id: ResourceId | None = None
    display_mode: Literal["dynamic", "static"] | None = None

    @model_validator(mode="after")
    def validate_changes(self) -> BuddyUpdate:
        """Require an actual change and reject null names or display modes.

        Returns:
            The validated update, preserving omitted-versus-null semantics.

        Raises:
            ValueError: No change is supplied or a required value is cleared.
        """
        for field in ("name", "display_mode"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        if self.model_fields_set <= {"expected_version"}:
            raise ValueError("Provide at least one profile change")
        return self


class BuddyAsset(BuddyModel):
    """Describe one owned artwork asset and its authenticated content URL."""

    id: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    checksum_sha256: str
    content_url: str


class BuddyProfile(BuddyModel):
    """Return an owned immutable artwork snapshot with editable identity and version.

    Persona availability is resolved independently of the saved optional Persona ID."""

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
    """Return the bounded collection of profiles visible to the authenticated owner."""

    buddies: list[BuddyProfile]


class BuddyAttachment(BuddyModel):
    """Select an existing conversation or workspace using bounded resource IDs.

    The service rechecks ownership; client-selected identifiers grant no authority."""

    buddy_id: ResourceId
    scope_type: Literal["conversation", "workspace"]
    scope_id: ResourceId


class BuddyAttachmentUpdate(BuddyAttachment):
    """Apply a scoped attachment using its last observed nonnegative version.

    Version zero denotes an attachment slot that has not yet been created."""

    expected_version: int = Field(ge=0)


class BuddyTarget(BuddyModel):
    """Return the authorized target title and its current workspace membership."""

    title: str
    workspace_id: str | None


class BuddyAttachmentResponse(BuddyModel):
    """Return a versioned client preference with freshly checked target availability.

    An unavailable target or Buddy is identified without disclosing foreign resources."""

    client_slot: str
    version: int
    attachment: BuddyAttachment | None
    target: BuddyTarget | None = None
    unavailable_reason: Literal["target_unavailable", "buddy_unavailable"] | None = None


class BuddyConversationSummary(BuddyModel):
    """Describe one accessible conversation, its scope, revision, and explicit identity."""

    id: str
    title: str
    created_at: datetime | None = None
    scope_type: Literal["global", "workspace"]
    workspace_id: str | None
    version: int
    assistant_kind: str | None = None
    assistant_id: str | None = None
    assistant_name: str | None = None


class BuddyConversationList(BuddyModel):
    """Return a bounded conversation page with the applied limit and offset."""

    conversations: list[BuddyConversationSummary]
    limit: int
    offset: int


class BuddyReplySettings(BuddyModel):
    """Project configured reply identifiers without credentials or conversation content."""

    provider: str | None
    model: str | None


class BuddyResult(BuddyModel):
    """Identify the exact persisted assistant result and its creation time and content."""

    id: str
    created_at: str
    content: str


class BuddyActivityItem(BuddyModel):
    """Project one conversation result and its exact acknowledgement status."""

    conversation_id: str
    title: str
    workspace_id: str | None
    result: BuddyResult
    acknowledged: bool


class BuddyActivityList(BuddyModel):
    """Return a bounded activity page with the applied limit and offset."""

    items: list[BuddyActivityItem]
    limit: int
    offset: int


class BuddyAcknowledgement(BuddyModel):
    """Acknowledge one exact owned result using bounded conversation and message IDs."""

    conversation_id: ResourceId
    result_message_id: ResourceId
