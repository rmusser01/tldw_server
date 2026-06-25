from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RPGCampaignCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4000)
    default_adapter_key: str = Field(min_length=1, max_length=80)


class RPGCampaignResponse(BaseModel):
    id: int
    title: str
    description: str | None
    default_adapter_key: str
    default_adapter_version: str
    status: str
    version: int


class RPGSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    adapter_key: str = Field(min_length=1, max_length=80)


class RPGSessionResponse(BaseModel):
    id: int
    campaign_id: int
    title: str
    status: str
    adapter_key: str
    adapter_version: str
    current_snapshot_version: int
    last_event_sequence: int
    version: int


class RPGEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1, max_length=120)
    event_payload: dict[str, Any]


class RPGRecordEventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    events: list[RPGEventInput] = Field(min_length=1, max_length=20)


class RPGRecordEventsResponse(BaseModel):
    committed_events: list[dict[str, Any]]
    proposal: dict[str, Any] | None


class RPGProposalApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_last_event_sequence: int = Field(ge=0)
    review_notes: str | None = Field(default=None, max_length=2000)


class RPGProposalRejectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_notes: str | None = Field(default=None, max_length=2000)
