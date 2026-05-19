"""Job payload helpers for async study-pack generation."""

from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest

STUDY_PACKS_DOMAIN = "study_packs"
STUDY_PACKS_JOB_TYPE = "study_pack_generate"


def study_pack_jobs_queue() -> str:
    """Return the queue name used by the study-pack Jobs worker."""

    queue = (os.getenv("STUDY_PACK_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def build_study_pack_job_payload(
    request: StudyPackCreateJobRequest,
    *,
    regenerate_from_pack_id: int | None = None,
    expected_version: int | None = None,
) -> dict[str, Any]:
    """Serialize a study-pack request into a Jobs payload."""

    payload = request.model_dump(mode="json", exclude_none=True)
    payload["source_items"] = [
        {
            key: value
            for key, value in source_item.items()
            if value not in (None, "", [], {})
        }
        for source_item in payload.get("source_items", [])
        if isinstance(source_item, dict)
    ]
    if regenerate_from_pack_id is not None:
        payload["regenerate_from_pack_id"] = int(regenerate_from_pack_id)
    if expected_version is not None:
        payload["expected_version"] = int(expected_version)
    return payload


def build_study_pack_job_result(
    *,
    pack_id: int,
    deck_id: int,
    deck_name: str | None = None,
    regenerated_from_pack_id: int | None = None,
) -> dict[str, Any]:
    """Serialize the completed job result returned by the worker."""

    result: dict[str, Any] = {
        "pack_id": int(pack_id),
        "deck_id": int(deck_id),
    }
    if deck_name:
        result["deck_name"] = str(deck_name)
    if regenerated_from_pack_id is not None:
        result["regenerated_from_pack_id"] = int(regenerated_from_pack_id)
    return result


def extract_study_pack_source_items(source_bundle_json: Any) -> list[dict[str, str]]:
    """Recover minimal source selections from a persisted study-pack bundle."""

    if isinstance(source_bundle_json, dict):
        items = source_bundle_json.get("items")
    elif isinstance(source_bundle_json, list):
        items = source_bundle_json
    else:
        items = None

    normalized: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        source_type = str(item.get("source_type") or "").strip()
        source_id = str(item.get("source_id") or "").strip()
        if not source_type or not source_id:
            continue
        normalized.append(
            {
                "source_type": source_type,
                "source_id": source_id,
            }
        )
    return normalized


__all__ = [
    "STUDY_PACKS_DOMAIN",
    "STUDY_PACKS_JOB_TYPE",
    "build_study_pack_job_payload",
    "build_study_pack_job_result",
    "extract_study_pack_source_items",
    "study_pack_jobs_queue",
]
