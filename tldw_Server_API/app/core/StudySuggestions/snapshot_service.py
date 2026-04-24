"""Snapshot helpers for shared study-suggestion status and refresh flows."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Jobs.manager import JobManager

from .flashcard_adapter import build_flashcard_suggestion_context, extract_flashcard_suggestion_evidence
from .jobs import (
    STUDY_SUGGESTIONS_DOMAIN,
    STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
    study_suggestions_jobs_queue,
)
from .quiz_adapter import build_quiz_suggestion_context, extract_quiz_suggestion_evidence
from .topic_aliases import DEFAULT_NAMESPACE, NORMALIZATION_VERSION, topic_key_for
from .topic_pipeline import rank_suggestion_topics, resolve_topic_candidates


def _safe_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _titleize_label(label: str) -> str:
    words = [part for part in str(label or "").strip().split() if part]
    if not words:
        return "Review"
    return " ".join(word.capitalize() for word in words)


def _fallback_topic_item(
    *,
    index: int,
    display_label: str,
    status: str = "candidate",
    selected: bool = False,
) -> dict[str, Any]:
    canonical_label = str(display_label or "review").strip().lower() or "review"
    return {
        "id": f"topic-{index}",
        "display_label": _titleize_label(display_label),
        "type": "derived",
        "status": status,
        "selected": selected,
        "topic_key": topic_key_for(DEFAULT_NAMESPACE, canonical_label),
        "normalization_version": NORMALIZATION_VERSION,
        "canonical_label": canonical_label,
        "evidence_reasons": ["derived_label"],
        "source_count": 0,
    }


def _find_quiz_topic_source(
    topic_labels: Iterable[str],
    source_bundle: list[dict[str, str]],
    *,
    allow_fallback: bool = True,
) -> dict[str, str] | None:
    """Return the source bundle item whose label matches any known topic label."""

    matchable_labels = {
        label.casefold()
        for label in topic_labels
        if isinstance(label, str) and label.strip()
    }
    for source in source_bundle:
        source_label = _safe_text(source.get("label"))
        if source_label and source_label.casefold() in matchable_labels:
            return source
    if allow_fallback and source_bundle:
        return source_bundle[0]
    return None


def _build_quiz_snapshot_payload(note_db: CharactersRAGDB, anchor_id: int) -> tuple[str, str, dict[str, Any]]:
    attempt = note_db.get_attempt(anchor_id, include_questions=True, include_answers=True)
    if not attempt:
        raise ConflictError("Attempt not found", entity="quiz_attempts", identifier=anchor_id)  # noqa: TRY003

    answers = [answer for answer in (attempt.get("answers") or []) if isinstance(answer, Mapping)]
    questions = [question for question in (attempt.get("questions") or []) if isinstance(question, Mapping)]
    correct_answers = sum(1 for answer in answers if answer.get("is_correct") is True)
    question_results = [
        {
            "question_id": answer.get("question_id"),
            "correct": bool(answer.get("is_correct")),
        }
        for answer in answers
    ]
    evidence = extract_quiz_suggestion_evidence(attempt)
    context = build_quiz_suggestion_context(
        quiz_attempt={
            "id": int(attempt["id"]),
            "quiz_id": int(attempt["quiz_id"]),
            "workspace_id": attempt.get("workspace_id"),
            "score": int(attempt.get("score") or 0),
            "correct_answers": correct_answers,
            "total_questions": len(questions),
            "question_results": question_results,
        },
        source_bundle=evidence["source_bundle"],
    )
    candidates = resolve_topic_candidates(
        source_labels=evidence["source_labels"],
        tag_labels=evidence["tag_labels"],
        derived_labels=evidence["derived_labels"],
    )
    ranked_topics = rank_suggestion_topics(
        candidates,
        weakness_labels=evidence["weakness_labels"],
        adjacent_labels=evidence["adjacent_labels"],
        exploratory_labels=evidence["derived_labels"],
    )
    topics: list[dict[str, Any]] = []
    for index, topic in enumerate(ranked_topics[:6], start=1):
        source = _find_quiz_topic_source(
            [*topic.raw_labels, topic.canonical_label, topic.semantic_label],
            context.source_bundle,
            allow_fallback=False,
        )
        raw_label = topic.raw_labels[0] if topic.raw_labels else topic.canonical_label
        item: dict[str, Any] = {
            "id": f"topic-{index}",
            "display_label": _titleize_label(raw_label),
            "type": topic.evidence_class,
            "status": topic.rank_reason,
            "selected": topic.rank_reason == "weakness",
            "topic_key": topic.topic_key,
            "normalization_version": topic.normalization_version,
            "canonical_label": topic.canonical_label,
            "evidence_reasons": list(topic.evidence_reasons),
            "source_count": topic.source_count,
        }
        if source:
            item["source_type"] = source["source_type"]
            item["source_id"] = source["source_id"]
        topics.append(item)

    if not topics:
        topics.append(_fallback_topic_item(index=1, display_label="Review"))

    payload = {
        "summary": {
            "score": int(context.summary_metrics.get("score") or 0),
            "correct_count": int(context.summary_metrics.get("correct_answers") or 0),
            "total_count": int(context.summary_metrics.get("total_questions") or 0),
        },
        "counts": {
            "topic_count": len(topics),
        },
        "topics": topics,
    }
    return context.service, context.activity_type, payload


def _build_flashcard_snapshot_payload(note_db: CharactersRAGDB, anchor_id: int) -> tuple[str, str, dict[str, Any]]:
    session_rollup = note_db.get_flashcard_review_session_rollup(int(anchor_id))
    if not session_rollup:
        raise ConflictError("Flashcard review session not found", entity="flashcard_review_sessions", identifier=anchor_id)  # noqa: TRY003

    provenance: dict[str, Any] = {
        "source_bundle": session_rollup.get("source_bundle") or [],
    }
    if session_rollup.get("study_pack_id") is not None:
        provenance["study_pack_id"] = session_rollup.get("study_pack_id")
        study_pack = note_db.get_study_pack(int(session_rollup["study_pack_id"]))
        if study_pack:
            provenance["study_pack"] = study_pack
    if session_rollup.get("deck_id") is not None:
        deck = note_db.get_deck(int(session_rollup["deck_id"]))
        if deck and deck.get("name"):
            provenance["deck_name"] = str(deck["name"])
    reviewed_cards = note_db.get_flashcard_reviewed_cards(int(anchor_id))
    if reviewed_cards:
        provenance["reviewed_cards"] = reviewed_cards

    context = build_flashcard_suggestion_context(
        session_rollup,
        provenance=provenance,
    )
    evidence = extract_flashcard_suggestion_evidence(
        session_rollup,
        provenance=provenance,
    )

    candidates = resolve_topic_candidates(
        source_labels=evidence["source_labels"],
        tag_labels=evidence["tag_labels"],
        derived_labels=evidence["derived_labels"],
    )
    ranked_topics = rank_suggestion_topics(
        candidates,
        weakness_labels=evidence["weakness_labels"],
        adjacent_labels=evidence["adjacent_labels"],
        exploratory_labels=evidence["derived_labels"],
    )
    topics: list[dict[str, Any]] = []
    for index, topic in enumerate(ranked_topics[:4], start=1):
        source = _find_quiz_topic_source(
            [*topic.raw_labels, topic.canonical_label, topic.semantic_label],
            evidence["source_bundle"],
            allow_fallback=False,
        )
        item: dict[str, Any] = {
            "id": f"topic-{index}",
            "display_label": _titleize_label(topic.raw_labels[0] if topic.raw_labels else topic.canonical_label),
            "type": topic.evidence_class,
            "status": topic.rank_reason,
            "selected": topic.rank_reason == "adjacent",
            "topic_key": topic.topic_key,
            "normalization_version": topic.normalization_version,
            "canonical_label": topic.canonical_label,
            "evidence_reasons": list(topic.evidence_reasons),
            "source_count": topic.source_count,
        }
        if source:
            item["source_type"] = source["source_type"]
            item["source_id"] = source["source_id"]
        topics.append(item)
    if not topics:
        topics.append(_fallback_topic_item(index=1, display_label="Spaced repetition", status="exploratory"))
    payload = {
        "summary": {
            "deck_id": context.summary_metrics.get("deck_id"),
            "correct_count": int(context.summary_metrics.get("correct_count") or 0),
            "total_count": int(context.summary_metrics.get("cards_reviewed") or 0),
        },
        "counts": {
            "topic_count": len(topics),
        },
        "topics": topics,
    }
    return context.service, context.activity_type, payload


def _build_snapshot_payload(note_db: CharactersRAGDB, anchor_type: str, anchor_id: int) -> tuple[str, str, dict[str, Any]]:
    normalized_anchor_type = str(anchor_type or "").strip().lower()
    if normalized_anchor_type == "quiz_attempt":
        return _build_quiz_snapshot_payload(note_db, int(anchor_id))
    if normalized_anchor_type == "flashcard_review_session":
        return _build_flashcard_snapshot_payload(note_db, int(anchor_id))
    raise ValueError("unsupported_study_suggestions_anchor_type")


def _job_matches_anchor(job: Mapping[str, Any], *, anchor_type: str, anchor_id: int) -> bool:
    payload = job.get("payload") or {}
    return (
        str(payload.get("anchor_type") or "").strip().lower() == str(anchor_type or "").strip().lower()
        and int(payload.get("anchor_id") or 0) == int(anchor_id)
    )


def _parse_job_created_at(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    with contextlib.suppress(ValueError):
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    return None


def _find_matching_job(
    *,
    job_manager: JobManager,
    anchor_type: str,
    anchor_id: int,
    statuses: Iterable[str],
    owner_user_id: str | None = None,
) -> dict[str, Any] | None:
    page_size = 100
    for status in statuses:
        created_before = None
        before_id = None
        while True:
            jobs = job_manager.list_jobs(
                domain=STUDY_SUGGESTIONS_DOMAIN,
                queue=study_suggestions_jobs_queue(),
                status=status,
                owner_user_id=owner_user_id,
                job_type=STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
                created_before=created_before,
                before_id=before_id,
                limit=page_size,
            )
            if not jobs:
                break
            for job in jobs:
                if _job_matches_anchor(job, anchor_type=anchor_type, anchor_id=anchor_id):
                    return job
            if len(jobs) < page_size:
                break
            last_job = jobs[-1]
            created_before = _parse_job_created_at(last_job.get("created_at"))
            if created_before is None:
                break
            before_id = int(last_job["id"])
    return None


def get_anchor_status(
    *,
    note_db: CharactersRAGDB,
    job_manager: JobManager,
    anchor_type: str,
    anchor_id: int,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the current suggestion status for a concrete anchor."""

    anchor_id = int(anchor_id)
    pending_job = _find_matching_job(
        job_manager=job_manager,
        anchor_type=anchor_type,
        anchor_id=anchor_id,
        statuses=("processing", "queued"),
        owner_user_id=owner_user_id,
    )
    if pending_job:
        return {
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "status": "pending",
            "job_id": int(pending_job["id"]),
            "snapshot_id": None,
        }

    failed_job = _find_matching_job(
        job_manager=job_manager,
        anchor_type=anchor_type,
        anchor_id=anchor_id,
        statuses=("failed",),
        owner_user_id=owner_user_id,
    )
    if failed_job:
        return {
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "status": "failed",
            "job_id": int(failed_job["id"]),
            "snapshot_id": None,
        }

    snapshots = note_db.list_suggestion_snapshots_for_anchor(anchor_type, anchor_id)
    active_snapshot = next((row for row in snapshots if str(row.get("status") or "").strip().lower() == "active"), None)
    if active_snapshot:
        return {
            "anchor_type": anchor_type,
            "anchor_id": anchor_id,
            "status": "ready",
            "job_id": None,
            "snapshot_id": int(active_snapshot["id"]),
        }

    return {
        "anchor_type": anchor_type,
        "anchor_id": anchor_id,
        "status": "none",
        "job_id": None,
        "snapshot_id": None,
    }


def load_live_evidence_for_snapshot(
    snapshot_row: Mapping[str, Any],
    *,
    note_db: CharactersRAGDB,
    principal: AuthPrincipal,
) -> dict[str, Any]:
    """Hydrate best-effort live evidence for a stored snapshot."""

    _ = principal
    payload = snapshot_row.get("payload_json") or {}
    topics = payload.get("topics") if isinstance(payload, Mapping) else []
    if not isinstance(topics, list):
        return {}

    live_evidence: dict[str, Any] = {}
    for index, topic in enumerate(topics, start=1):
        if not isinstance(topic, Mapping):
            continue
        topic_id = _safe_text(topic.get("id")) or f"topic-{index}"
        source_type = _safe_text(topic.get("source_type"))
        source_id = _safe_text(topic.get("source_id"))
        item: dict[str, Any] = {
            "source_available": bool(source_type and source_id),
        }
        if source_type:
            item["source_type"] = source_type
        if source_id:
            item["source_id"] = source_id
        live_evidence[topic_id] = item
    return live_evidence


def serialize_snapshot(snapshot_row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize DB column names for API responses."""

    return {
        "id": int(snapshot_row["id"]),
        "service": snapshot_row["service"],
        "activity_type": snapshot_row["activity_type"],
        "anchor_type": snapshot_row["anchor_type"],
        "anchor_id": int(snapshot_row["anchor_id"]),
        "suggestion_type": snapshot_row["suggestion_type"],
        "status": snapshot_row["status"],
        "payload": snapshot_row.get("payload_json"),
        "user_selection": snapshot_row.get("user_selection_json"),
        "refreshed_from_snapshot_id": snapshot_row.get("refreshed_from_snapshot_id"),
        "created_at": snapshot_row.get("created_at"),
        "last_modified": snapshot_row.get("last_modified"),
    }


def refresh_snapshot_for_anchor(
    *,
    note_db: CharactersRAGDB,
    anchor_type: str,
    anchor_id: int,
    refreshed_from_snapshot_id: int | None = None,
    principal: AuthPrincipal,
) -> int:
    """Create a fresh snapshot for the provided anchor."""

    _ = principal
    service, activity_type, payload = _build_snapshot_payload(note_db, anchor_type, int(anchor_id))
    return note_db.create_suggestion_snapshot(
        service=service,
        activity_type=activity_type,
        anchor_type=anchor_type,
        anchor_id=int(anchor_id),
        suggestion_type="study_suggestions",
        payload_json=payload,
        refreshed_from_snapshot_id=refreshed_from_snapshot_id,
    )


__all__ = [
    "get_anchor_status",
    "load_live_evidence_for_snapshot",
    "refresh_snapshot_for_anchor",
    "serialize_snapshot",
]
