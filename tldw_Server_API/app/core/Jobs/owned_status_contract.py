"""Jobs-owned bounded projection shared with authenticated status transports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OwnedJobStatus(BaseModel):
    """Bounded status projection; arbitrary Jobs records are never serialized."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    uuid: str
    domain: Literal["claims"] = "claims"
    job_type: Literal["claims_rebuild_clusters"] = "claims_rebuild_clusters"
    status: Literal["queued", "processing", "completed", "failed", "cancelled", "quarantined"]
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    result: dict[str, int | float | str | None] | None = None
    failure_code: str | None = None
