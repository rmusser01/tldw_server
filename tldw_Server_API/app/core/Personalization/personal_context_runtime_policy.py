"""Typed encrypted server-local Personal Context runtime policy values."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROFILE_RUNTIME_POLICY_ID = "__profile__"


class ServerRuntimePolicy(BaseModel):
    """Server-local switch controlling whether profile context may be used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    enabled: bool = False


class WorkspaceRuntimePolicy(BaseModel):
    """Encrypted server-local mapping from canonical scope to workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workspace_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=512)


class RuntimePolicyVersion(BaseModel):
    """Runtime policy value with its optimistic concurrency version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_id: str | None
    enabled: bool
