"""Core-owned Buddy models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

ResourceId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
BuddyName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class BuddyModel(BaseModel):
    """Closed Buddy API contract; unknown fields fail Pydantic validation."""

    model_config = ConfigDict(extra="forbid")


class StarterSource(BuddyModel):
    """Select immutable bundled artwork by a bounded, path-free catalog identifier."""

    kind: Literal["starter"]
    starter_id: ResourceId


class PersonaPackSource(BuddyModel):
    """Select an owned Persona pack to snapshot; source IDs do not grant access."""

    kind: Literal["persona_pack"]
    persona_id: ResourceId
    pack_id: ResourceId


class BuddyCreate(BuddyModel):
    """Create independent artwork with a nonblank name and one discriminated source.

    Resource IDs and name lengths are bounded. Persona behavior is optional;
    display mode defaults to dynamic. Invalid fields raise Pydantic validation errors."""

    name: BuddyName
    source: Annotated[StarterSource | PersonaPackSource, Field(discriminator="kind")]
    optional_persona_id: ResourceId | None = None
    display_mode: Literal["dynamic", "static"] = "dynamic"


class BuddyTurnCreate(BuddyModel):
    """Accept one plain-text turn for an exact authorized conversation.

    Text is limited to 12,000 characters and excludes blank input and slash
    commands. The request key is bounded and idempotent; attachment version must
    be positive. Optional provider/model overrides have bounded nonempty values."""

    conversation_id: ResourceId
    text: str = Field(min_length=1, max_length=12000)
    client_request_id: ResourceId
    expected_attachment_version: int = Field(ge=1)
    model: str | None = Field(None, min_length=1, max_length=512)
    provider: str | None = Field(None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Validate plain chat text while preserving the submitted content.

        Args:
            value: Length-validated message text.

        Returns:
            The original text, including intentional whitespace.

        Raises:
            ValueError: The message is blank or requests a slash command.
        """
        if not value.strip():
            raise ValueError("Enter a message")
        if value.lstrip().startswith("/"):
            raise ValueError("Use Chat to run slash commands")
        return value
