"""Jobs-backed helpers for manuscript scene annotation review."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Writing.manuscript_annotations import (
    VALID_ANNOTATION_CATEGORIES,
    build_scene_review_prompt,
    parse_scene_review_response,
)
from tldw_Server_API.app.core.Writing.manuscript_analysis import _extract_content

WRITING_JOBS_DOMAIN = "writing"
WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE = "writing_scene_annotation_review"
_MAX_SCENE_REVIEW_COMMENTS = 10
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "owner_user_id",
        "user_id",
        "scene_text",
        "content_plain",
        "selected_text",
        "annotation_body",
        "body",
        "suggested_fix",
        "raw_model_output",
    }
)


class WritingAnnotationReviewEnqueueError(RuntimeError):
    """Raised when a scene annotation review cannot be queued."""

    def __init__(
        self,
        message: str = "Failed to enqueue manuscript scene annotation review.",
        *,
        error_code: str = "writing_annotation_review_enqueue_failed",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


def writing_annotation_review_jobs_queue() -> str:
    """Return the Jobs queue name for Writing annotation review work."""
    queue = (os.getenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def build_scene_annotation_review_job_payload(
    *,
    project_id: str,
    scene_id: str,
    scene_version: int,
    provider: str,
    model: str,
    max_comments: int,
    category_filters: list[str] | tuple[str, ...] | None = None,
    review_focus: str | None = None,
) -> dict[str, Any]:
    """Build a sanitized Jobs payload for queued scene annotation review."""
    normalized_max_comments = _coerce_max_comments(max_comments)
    payload: dict[str, Any] = {
        "project_id": str(project_id).strip(),
        "scene_id": str(scene_id).strip(),
        "scene_version": int(scene_version),
        "provider": str(provider).strip(),
        "model": str(model).strip(),
        "max_comments": normalized_max_comments,
        "category_filters": _normalize_category_filters(category_filters),
    }
    normalized_focus = _normalize_optional_text(review_focus)
    if normalized_focus is not None:
        payload["review_focus"] = normalized_focus
    _assert_payload_sanitized(payload)
    return payload


def enqueue_scene_annotation_review_job(
    *,
    job_manager: JobManager,
    owner_user_id: str,
    project_id: str,
    scene_id: str,
    scene_version: int,
    provider: str,
    model: str,
    max_comments: int,
    category_filters: list[str] | tuple[str, ...] | None = None,
    review_focus: str | None = None,
) -> dict[str, Any]:
    """Create a Jobs row for a manuscript scene annotation review."""
    normalized_owner_user_id = str(owner_user_id).strip()
    if not normalized_owner_user_id:
        raise WritingAnnotationReviewEnqueueError(
            "Manuscript scene annotation review requires an owner.",
            error_code="writing_annotation_review_owner_missing",
        )
    payload = build_scene_annotation_review_job_payload(
        project_id=project_id,
        scene_id=scene_id,
        scene_version=scene_version,
        provider=provider,
        model=model,
        max_comments=max_comments,
        category_filters=category_filters,
        review_focus=review_focus,
    )
    try:
        return job_manager.create_job(
            domain=WRITING_JOBS_DOMAIN,
            queue=writing_annotation_review_jobs_queue(),
            job_type=WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE,
            payload=payload,
            owner_user_id=normalized_owner_user_id,
            max_retries=3,
            idempotency_key=_scene_annotation_review_idempotency_key(payload, normalized_owner_user_id),
        )
    except Exception as exc:
        raise WritingAnnotationReviewEnqueueError() from exc


async def process_scene_annotation_review_job(
    *,
    manuscript_db: ManuscriptDBHelper,
    job_payload: dict[str, Any],
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """Run a queued scene annotation review and persist anchored annotations."""
    del job_manager
    payload = _normalize_processor_payload(job_payload)
    scene = manuscript_db.get_scene(payload["scene_id"])
    if not scene or str(scene.get("project_id") or "") != payload["project_id"]:
        return _result_with_diagnostic("scene_not_found", "Scene was not found for annotation review.")
    if int(scene.get("version") or 0) != int(payload["scene_version"]):
        return _result_with_diagnostic(
            "scene_version_mismatch",
            "Scene changed before annotation review started.",
        )

    scene_text = scene.get("content_plain") or ""
    messages = build_scene_review_prompt(
        scene_text=scene_text,
        category_filters=payload["category_filters"],
        max_comments=payload["max_comments"],
        review_focus=payload.get("review_focus"),
    )

    from tldw_Server_API.app.core.Chat import chat_service as _chat_service

    llm_response = await _chat_service.perform_chat_api_call_async(
        messages=messages,
        api_endpoint=payload["provider"],
        model=payload["model"],
        temp=0.2,
    )
    raw_text = _extract_content(llm_response)
    try:
        parsed_annotations = parse_scene_review_response(
            raw_text,
            max_comments=payload["max_comments"],
        )
    except ValueError as exc:
        return _result_with_diagnostic(
            "model_output_invalid",
            "Scene review output could not be parsed.",
            details=str(exc),
        )

    diagnostics: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    allowed_categories = set(payload["category_filters"])
    for entry in parsed_annotations:
        if allowed_categories and entry["category"] not in allowed_categories:
            diagnostics.append(
                {
                    "code": "category_filtered",
                    "message": "Scene review returned an annotation outside requested categories.",
                }
            )
            continue
        anchor = _resolve_scene_review_anchor(scene_text, entry)
        if anchor.get("diagnostic"):
            diagnostics.append(anchor["diagnostic"])
            continue
        candidate = {
            "project_id": payload["project_id"],
            "target_type": "scene",
            "target_id": payload["scene_id"],
            "category": entry["category"],
            "source": "ai_scene_review",
            "body": entry["body"],
            "suggested_fix": entry.get("suggested_fix"),
            "scene_version": payload["scene_version"],
            "anchor_start": anchor["anchor_start"],
            "anchor_end": anchor["anchor_end"],
            "selected_text": anchor["selected_text"],
        }
        candidates.append(candidate)

    candidates = candidates[: payload["max_comments"]]
    retained = manuscript_db.suppress_duplicate_annotation_candidates(payload["project_id"], candidates)
    if len(retained) < len(candidates):
        diagnostics.append(
            {
                "code": "duplicate_annotations_suppressed",
                "message": "Duplicate open annotations were skipped.",
            }
        )

    created_annotation_ids: list[str] = []
    for candidate in retained[: payload["max_comments"]]:
        annotation_id = manuscript_db.create_annotation(
            project_id=candidate["project_id"],
            target_type=candidate["target_type"],
            target_id=candidate["target_id"],
            category=candidate["category"],
            source=candidate["source"],
            body=candidate["body"],
            suggested_fix=candidate.get("suggested_fix"),
            scene_version=candidate["scene_version"],
            anchor_start=candidate["anchor_start"],
            anchor_end=candidate["anchor_end"],
            selected_text=candidate["selected_text"],
        )
        created_annotation_ids.append(annotation_id)
    return {"created_annotation_ids": created_annotation_ids, "diagnostics": diagnostics}


def _normalize_processor_payload(job_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job_payload)
    for key in ("project_id", "scene_id", "provider", "model"):
        payload[key] = str(payload.get(key) or "").strip()
        if not payload[key]:
            raise ValueError(f"missing {key}")
    payload["scene_version"] = int(payload["scene_version"])
    payload["max_comments"] = _coerce_max_comments(payload.get("max_comments", 3))
    payload["category_filters"] = _normalize_category_filters(payload.get("category_filters"))
    payload["review_focus"] = _normalize_optional_text(payload.get("review_focus"))
    _assert_payload_sanitized(payload)
    return payload


def _resolve_scene_review_anchor(text: str, entry: dict[str, Any]) -> dict[str, Any]:
    quote = str(entry.get("quote") or entry.get("selected_text") or "").strip()
    if not quote:
        return {
            "diagnostic": {
                "code": "annotation_anchor_missing",
                "message": "Scene review annotation did not include an anchor quote.",
            }
        }

    start = entry.get("start")
    end = entry.get("end")
    if isinstance(start, int) and isinstance(end, int):
        if 0 <= start < end <= len(text) and text[start:end] == quote:
            return {"anchor_start": start, "anchor_end": end, "selected_text": quote}
        return {
            "diagnostic": {
                "code": "annotation_anchor_invalid",
                "message": "Scene review annotation anchor range did not match the saved scene.",
            }
        }

    matches = _find_all(text, quote)
    if len(matches) == 1:
        anchor_start = matches[0]
        return {
            "anchor_start": anchor_start,
            "anchor_end": anchor_start + len(quote),
            "selected_text": quote,
        }
    if not matches:
        return {
            "diagnostic": {
                "code": "annotation_anchor_not_found",
                "message": "Scene review annotation anchor quote was not found in the saved scene.",
            }
        }
    return {
        "diagnostic": {
            "code": "annotation_anchor_ambiguous",
            "message": "Scene review annotation anchor quote matched multiple saved scene ranges.",
        }
    }


def _scene_annotation_review_idempotency_key(payload: dict[str, Any], owner_user_id: str) -> str:
    owner_digest = hashlib.sha256(f"owner_user_id:{owner_user_id}".encode("utf-8")).hexdigest()[:16]
    key_material = json.dumps(
        {
            "owner_digest": owner_digest,
            "category_filters": payload.get("category_filters") or [],
            "review_focus": payload.get("review_focus") or "",
            "provider": payload["provider"],
            "model": payload["model"],
            "max_comments": payload["max_comments"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
    return (
        "writing-scene-annotation-review:"
        f"owner{owner_digest}:{payload['scene_id']}:v{payload['scene_version']}:"
        f"{payload['provider']}:{payload['model']}:max{payload['max_comments']}:{digest}"
    )


def _coerce_max_comments(value: Any) -> int:
    count = int(value)
    if count < 1 or count > _MAX_SCENE_REVIEW_COMMENTS:
        raise ValueError("max_comments must be between 1 and 10")
    return count


def _normalize_category_filters(value: list[str] | tuple[str, ...] | Any | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("category_filters must be a list")
    normalized: list[str] = []
    for item in value:
        category = str(item).strip()
        if category and category in VALID_ANNOTATION_CATEGORIES and category not in normalized:
            normalized.append(category)
    return normalized


def _normalize_optional_text(value: Any | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _assert_payload_sanitized(payload: dict[str, Any]) -> None:
    forbidden = _FORBIDDEN_PAYLOAD_KEYS.intersection(payload)
    if forbidden:
        raise ValueError("Jobs payload contains fields that must not be persisted")


def _find_all(text: str, needle: str) -> list[int]:
    matches: list[int] = []
    position = text.find(needle)
    while position != -1:
        matches.append(position)
        position = text.find(needle, position + 1)
    return matches


def _result_with_diagnostic(code: str, message: str, *, details: str | None = None) -> dict[str, Any]:
    diagnostic = {"code": code, "message": message}
    if details:
        diagnostic["details"] = details
    return {"created_annotation_ids": [], "diagnostics": [diagnostic]}


__all__ = [
    "WRITING_JOBS_DOMAIN",
    "WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE",
    "WritingAnnotationReviewEnqueueError",
    "build_scene_annotation_review_job_payload",
    "enqueue_scene_annotation_review_job",
    "process_scene_annotation_review_job",
    "writing_annotation_review_jobs_queue",
]
