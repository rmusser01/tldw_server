"""Strict request and bounded collection schemas for Personal Context."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tldw_profile_core import (
    ProfileControls,
    ProfilePayload,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    SemanticKey,
)


class StrictRequest(BaseModel):
    """Base request that refuses accidental or future-unknown fields."""

    model_config = ConfigDict(extra="forbid")


class ProfileCreateRequest(StrictRequest):
    runtime_enabled: bool = False


class WorkspaceScopeCreateRequest(StrictRequest):
    workspace_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=512)


class RecordCreateRequest(StrictRequest):
    scope_id: str = Field(min_length=1, max_length=128)
    payload: ProfilePayload
    semantic_key: SemanticKey | None = None
    controls: ProfileControls
    expires_at: datetime | None = None
    no_expiry: bool = False


class RecordUpdateRequest(StrictRequest):
    expected_version_id: str = Field(min_length=1, max_length=128)
    payload: ProfilePayload | None = None
    semantic_key: SemanticKey | None = None
    controls: ProfileControls | None = None
    expires_at: datetime | None = None
    no_expiry: bool | None = None

    @model_validator(mode="after")
    def require_mutation(self):
        mutable_fields = {
            "payload",
            "semantic_key",
            "controls",
            "expires_at",
            "no_expiry",
        }
        if not self.model_fields_set.intersection(mutable_fields):
            raise ValueError("record update requires at least one changed field")
        return self


class ExpectedVersionRequest(StrictRequest):
    expected_version_id: str = Field(min_length=1, max_length=128)


class ProposalReviewRequest(StrictRequest):
    action: Literal["accept", "reject"]


class RuntimeUpdateRequest(StrictRequest):
    enabled: bool
    expected_version_id: str | None = Field(default=None, max_length=128)


class ExportRequest(StrictRequest):
    mode: Literal["plaintext", "recovery"]
    confirmation: str = Field(min_length=1, max_length=128)
    scope_ids: tuple[str, ...] | None = Field(default=None, max_length=100)
    passphrase: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def validate_mode_fields(self):
        if self.mode == "recovery" and self.passphrase is None:
            raise ValueError("recovery export requires a passphrase")
        if self.mode == "recovery" and self.scope_ids is not None:
            raise ValueError("recovery export is whole-profile only")
        if self.mode == "plaintext" and self.passphrase is not None:
            raise ValueError("plaintext export does not accept a passphrase")
        return self


class PurgeRequest(StrictRequest):
    mode: Literal["local_copy", "everywhere"]
    confirmation: str = Field(min_length=1, max_length=128)
    expected_purge_generation: int = Field(ge=0)


class ScopeListResponse(BaseModel):
    items: tuple[ProfileScope, ...]


class RecordListResponse(BaseModel):
    items: tuple[ProfileRecord, ...]
    limit: int = Field(ge=1, le=20)


class ProposalListResponse(BaseModel):
    items: tuple[ProfileProposal, ...]
    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0, le=1_000)


class ProposalReviewResponse(BaseModel):
    receipt: ProfileProposal
    record: ProfileRecord | None


class ExportResponse(BaseModel):
    mode: Literal["plaintext", "recovery"]
    data: dict[str, Any]
