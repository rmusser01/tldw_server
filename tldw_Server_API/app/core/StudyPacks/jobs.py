"""Job payload helpers for async study-pack generation."""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest

STUDY_PACKS_DOMAIN = "study_packs"
STUDY_PACKS_JOB_TYPE = "study_pack_generate"
_DEFAULT_REGENERATION_EXCERPT_CHARS = 12_000
_REGENERATION_EXCERPT_CHARS_ENV = "STUDY_PACK_MAX_EVIDENCE_CHARS_PER_SOURCE"


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


def _clean_text(value: Any) -> str:
    """Coerce optional values into stripped text."""

    if value is None:
        return ""
    return str(value).strip()


def _compact_mapping(value: Any) -> dict[str, Any]:
    """Return a locator mapping without empty fields."""

    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _max_regeneration_excerpt_chars() -> int:
    """Return the maximum persisted evidence hint copied into regeneration jobs."""

    configured_limit = _clean_text(os.getenv(_REGENERATION_EXCERPT_CHARS_ENV))
    if not configured_limit:
        return _DEFAULT_REGENERATION_EXCERPT_CHARS
    try:
        parsed_limit = int(configured_limit)
    except ValueError:
        logger.warning(
            "Ignoring invalid {} value {!r}; using default {}",
            _REGENERATION_EXCERPT_CHARS_ENV,
            configured_limit,
            _DEFAULT_REGENERATION_EXCERPT_CHARS,
        )
        return _DEFAULT_REGENERATION_EXCERPT_CHARS
    if parsed_limit <= 0:
        logger.warning(
            "Ignoring non-positive {} value {}; using default {}",
            _REGENERATION_EXCERPT_CHARS_ENV,
            parsed_limit,
            _DEFAULT_REGENERATION_EXCERPT_CHARS,
        )
        return _DEFAULT_REGENERATION_EXCERPT_CHARS
    return parsed_limit


def _bound_regeneration_excerpt(excerpt_text: str, locator: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Bound regenerated excerpt hints and annotate locator metadata when truncated."""

    max_chars = _max_regeneration_excerpt_chars()
    if len(excerpt_text) <= max_chars:
        return excerpt_text, locator
    bounded_locator = dict(locator)
    bounded_locator["excerpt_truncated"] = True
    bounded_locator["excerpt_original_chars"] = len(excerpt_text)
    return excerpt_text[:max_chars], bounded_locator


def extract_study_pack_source_items(source_bundle_json: Any) -> list[dict[str, Any]]:
    """Recover source selections from a persisted study-pack bundle."""

    bundle_payload = source_bundle_json
    if isinstance(bundle_payload, str):
        try:
            bundle_payload = json.loads(bundle_payload)
        except json.JSONDecodeError:
            bundle_payload = None

    if isinstance(bundle_payload, dict):
        items = bundle_payload.get("items")
    elif isinstance(bundle_payload, list):
        items = bundle_payload
    else:
        items = None

    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        source_type = _clean_text(item.get("source_type"))
        source_id = _clean_text(item.get("source_id"))
        if not source_type or not source_id:
            continue
        source_item: dict[str, Any] = {
            "source_type": source_type,
            "source_id": source_id,
        }
        label = _clean_text(item.get("label"))
        if label:
            source_item["label"] = label
        locator = _compact_mapping(item.get("locator"))
        excerpt_text = _clean_text(item.get("excerpt_text")) or _clean_text(item.get("evidence_text"))
        if excerpt_text:
            excerpt_text, locator = _bound_regeneration_excerpt(excerpt_text, locator)
        if locator:
            source_item["locator"] = locator
        if excerpt_text:
            source_item["excerpt_text"] = excerpt_text
        normalized.append(source_item)
    return normalized


__all__ = [
    "STUDY_PACKS_DOMAIN",
    "STUDY_PACKS_JOB_TYPE",
    "build_study_pack_job_payload",
    "build_study_pack_job_result",
    "extract_study_pack_source_items",
    "study_pack_jobs_queue",
]
