"""Bounded, reference-only local assistant selection provenance."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

MAX_ASSISTANT_STARTUP_BYTES = 1024


class AssistantStartup(BaseModel):
    """Immutable creation origin, never an assistant snapshot or permission grant."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    source: Literal["workspace_default", "explicit", "explicit_none", "system_fallback", "fork", "unknown"] = "unknown"
    workspace_id: str | None = None
    workspace_version: Annotated[StrictInt, Field(gt=0)] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: object) -> int:
        """Reject bool and numeric coercion at the versioned authority boundary."""
        if type(value) is not int or value != 1:
            raise ValueError("Assistant startup schema version must be integer 1")
        return value

    @field_validator("workspace_id", mode="before")
    @classmethod
    def _validate_workspace_id(cls, value: object) -> str | None:
        """Retain a complete nonempty ID without imposing a new Workspace ID limit."""
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Assistant startup Workspace ID must be a nonempty string")
        return value

    @model_validator(mode="after")
    def _validate_references_and_size(self) -> AssistantStartup:
        """Enforce source/reference relations and the cap even for response-only values."""
        has_id = self.workspace_id is not None
        has_version = self.workspace_version is not None
        if has_id != has_version:
            raise ValueError("Assistant startup Workspace references must be paired")
        if self.source == "workspace_default" and not has_id:
            raise ValueError("Workspace default startup requires Workspace references")
        if self.source not in {"workspace_default", "system_fallback"} and has_id:
            raise ValueError("This assistant startup source forbids Workspace references")
        _canonical_startup_bytes(self)
        return self


def _canonical_startup_bytes(value: AssistantStartup) -> bytes:
    """Serialize deterministically, with bounded Unicode and size validation errors."""
    try:
        encoded = json.dumps(
            value.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    except UnicodeError:
        raise ValueError("Assistant startup must contain valid Unicode") from None
    if len(encoded) > MAX_ASSISTANT_STARTUP_BYTES:
        raise ValueError("Assistant startup exceeds the 1024-byte limit")
    return encoded


def encode_assistant_startup(value: AssistantStartup) -> str:
    """Encode only a validated model, rechecking unchecked copies/construction."""
    if not isinstance(value, AssistantStartup):
        raise TypeError("Assistant startup encoding requires an AssistantStartup value")
    validated = AssistantStartup.model_validate(value.model_dump())
    return _canonical_startup_bytes(validated).decode("utf-8")


def decode_assistant_startup(raw: object) -> AssistantStartup:
    """Project legacy or invalid persisted JSON to a fresh unknown, without input logs."""
    if not isinstance(raw, str) or len(raw) > MAX_ASSISTANT_STARTUP_BYTES:
        return AssistantStartup()
    try:
        if len(raw.encode("utf-8")) > MAX_ASSISTANT_STARTUP_BYTES:
            return AssistantStartup()
        return AssistantStartup.model_validate(json.loads(raw))
    except (ValueError, RecursionError):
        # Includes JSON/Unicode decoding and Pydantic validation, not DB failures.
        return AssistantStartup()
