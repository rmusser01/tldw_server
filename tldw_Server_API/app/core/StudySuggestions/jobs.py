"""Job payload helpers for async study-suggestion refreshes."""

from __future__ import annotations

import os
from typing import Any


STUDY_SUGGESTIONS_DOMAIN = "study_suggestions"
STUDY_SUGGESTIONS_REFRESH_JOB_TYPE = "study_suggestions_refresh"


def study_suggestions_jobs_queue() -> str:
    """Return the queue name used by the study-suggestions Jobs worker."""

    queue = (os.getenv("STUDY_SUGGESTIONS_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def build_study_suggestions_job_payload(
    *,
    job_type: str,
    anchor_type: str | None = None,
    anchor_id: int | None = None,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    """Serialize a study-suggestions refresh request into a Jobs payload."""

    payload: dict[str, Any] = {
        "job_type": str(job_type or "").strip(),
    }
    if anchor_type:
        payload["anchor_type"] = str(anchor_type).strip()
    if anchor_id is not None:
        payload["anchor_id"] = int(anchor_id)
    if snapshot_id is not None:
        payload["snapshot_id"] = int(snapshot_id)
    return payload


__all__ = [
    "STUDY_SUGGESTIONS_DOMAIN",
    "STUDY_SUGGESTIONS_REFRESH_JOB_TYPE",
    "build_study_suggestions_job_payload",
    "study_suggestions_jobs_queue",
]
