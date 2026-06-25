from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RPGProposalRecord:
    id: int
    session_id: int
    owner_user_id: int
    base_event_sequence: int
    base_snapshot_version: int
    proposed_events: list[dict[str, Any]]
    patch: dict[str, Any] | None
    rationale: str | None
    confidence: float | None
    source_type: str
    source_actor_id: str | None
    model_metadata: dict[str, Any]
    status: str
    review_notes: str | None
    created_at: datetime
    applied_at: datetime | None
    rejected_at: datetime | None
