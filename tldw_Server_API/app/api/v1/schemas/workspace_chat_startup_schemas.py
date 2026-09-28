"""Bounded Workspace startup requests and canonical retry fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import _validate_conversation_state

STARTUP_TEXT_BYTE_LIMITS = {
    "workspace_id": 256,
    "title": 4096,
    "state": 16,
    "topic_label": 1024,
    "cluster_id": 256,
    "source": 256,
    "external_ref": 4096,
}
STARTUP_TEXT_BYTES_MAX = 8192
STARTUP_BODY_BYTES_MAX = 65536
STARTUP_IDEMPOTENCY_KEY_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"


def startup_text_size(value: str, field: str) -> int:
    """Validate raw decoded text and return its bounded UTF-8 byte size."""
    limit = STARTUP_TEXT_BYTE_LIMITS[field]
    if len(value) > limit or "\x00" in value:
        raise ValueError("Startup text exceeds its limit or contains NUL")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        raise ValueError("Startup text must contain valid Unicode") from None
    if size > limit:
        raise ValueError("Startup text exceeds its UTF-8 byte limit")
    return size


class WorkspaceChatStartupRequest(BaseModel):
    """Closed startup input, with an explicit Workspace default selection."""

    model_config = ConfigDict(extra="forbid")

    scope_type: Literal["workspace"]
    workspace_id: str = Field(min_length=1)
    workspace_assistant_selection: Literal["inherit", "none"]
    workspace_assistant_default_version: StrictInt | None = None
    title: str | None = None
    state: str | None = None
    topic_label: str | None = None
    cluster_id: str | None = None
    source: str | None = None
    external_ref: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_supplied_text(cls, data: Any) -> Any:
        """Bound supplied strings before trimming or state normalization."""
        if isinstance(data, Mapping):
            total = sum(
                startup_text_size(data[field], field)
                for field in STARTUP_TEXT_BYTE_LIMITS
                if field in data and isinstance(data[field], str)
            )
            if total > STARTUP_TEXT_BYTES_MAX:
                raise ValueError("Startup text exceeds its combined byte limit")
        return data

    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, value: str) -> str:
        """Require a nonblank Workspace identifier and trim its edges."""
        if not value.strip():
            raise ValueError("Workspace id is required")
        return value.strip()

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: str | None) -> str | None:
        """Reuse the existing conversation lifecycle state semantics."""
        return _validate_conversation_state(value)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Require an inherit version, or forbid any version for none."""
        version = self.workspace_assistant_default_version
        if self.workspace_assistant_selection == "inherit":
            if version is None or version < 1:
                raise ValueError("Inherit requires a positive Workspace version")
        elif "workspace_assistant_default_version" in self.model_fields_set:
            raise ValueError("None does not accept a Workspace version")
        return self


def startup_request_fingerprint(request: WorkspaceChatStartupRequest) -> str:
    """Hash canonical, schema-versioned caller input without generated fields."""
    data = {"schema_version": 1, "body": request.model_dump(mode="json", exclude_unset=True)}
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
