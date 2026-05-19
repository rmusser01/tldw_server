"""Quiz-specific suggestion context adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .types import SuggestionContext


def _copy_source_bundle(source_bundle: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(item) for item in (source_bundle or [])]


def _safe_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _unique_preserve(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _safe_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _unique_sources(values: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str | None]] = set()
    ordered: list[dict[str, str]] = []
    for value in values:
        source_type = _safe_text(value.get("source_type"))
        source_id = _safe_text(value.get("source_id"))
        label = _safe_text(value.get("label"))
        if not source_type or not source_id:
            continue
        key = (source_type.casefold(), source_id.casefold(), label.casefold() if label else None)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "source_type": source_type,
            "source_id": source_id,
        }
        if label:
            item["label"] = label
        ordered.append(item)
    return ordered


def _normalize_citation_source(citation: Mapping[str, Any]) -> dict[str, str] | None:
    source_type = _safe_text(citation.get("source_type"))
    source_id = _safe_text(citation.get("source_id"))
    label = _safe_text(citation.get("label"))
    if not source_type or not source_id:
        return None
    source: dict[str, str] = {
        "source_type": source_type,
        "source_id": source_id,
    }
    if label:
        source["label"] = label
    return source


def extract_quiz_suggestion_evidence(attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Extract quiz labels and light refs from an attempt row."""

    questions = attempt.get("questions") or []
    answers = attempt.get("answers") or []
    answers_by_question_id = {
        int(answer["question_id"]): answer
        for answer in answers
        if isinstance(answer, Mapping) and answer.get("question_id") is not None
    }

    source_labels: list[str] = []
    tag_labels: list[str] = []
    weakness_labels: list[str] = []
    adjacent_labels: list[str] = []
    source_bundle: list[dict[str, str]] = []

    for question in questions:
        if not isinstance(question, Mapping):
            continue
        question_id = question.get("id")
        answer = answers_by_question_id.get(int(question_id)) if question_id is not None else None
        question_tags = [tag for tag in (question.get("tags") or []) if _safe_text(tag)]
        citations = [item for item in (question.get("source_citations") or []) if isinstance(item, Mapping)]
        citation_labels = [_safe_text(citation.get("label")) for citation in citations]
        citation_labels = [label for label in citation_labels if label]

        tag_labels.extend(question_tags)
        source_labels.extend(citation_labels)
        for citation in citations:
            source = _normalize_citation_source(citation)
            if source:
                source_bundle.append(source)

        is_incorrect = bool(answer) and answer.get("is_correct") is False
        target = weakness_labels if is_incorrect else adjacent_labels
        target.extend(question_tags)
        target.extend(citation_labels)

    derived_labels: list[str] = []
    if not source_labels and not tag_labels:
        incorrect_count = sum(1 for answer in answers if isinstance(answer, Mapping) and answer.get("is_correct") is False)
        derived_labels.append("missed questions" if incorrect_count else "review")

    return {
        "source_labels": _unique_preserve(source_labels),
        "tag_labels": _unique_preserve(tag_labels),
        "derived_labels": _unique_preserve(derived_labels),
        "weakness_labels": _unique_preserve(weakness_labels),
        "adjacent_labels": _unique_preserve(adjacent_labels),
        "source_bundle": _unique_sources(source_bundle),
    }


def build_quiz_suggestion_context(
    *,
    quiz_attempt: Mapping[str, Any],
    source_bundle: Sequence[Mapping[str, Any]] | None = None,
) -> SuggestionContext:
    total_questions = int(quiz_attempt.get("total_questions") or 0)
    correct_answers = int(quiz_attempt.get("correct_answers") or 0)
    incorrect_count = sum(
        1 for result in quiz_attempt.get("question_results", []) if not result.get("correct", False)
    )
    accuracy = (correct_answers / total_questions) if total_questions else 0.0

    return SuggestionContext(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=int(quiz_attempt["id"]),
        workspace_id=quiz_attempt.get("workspace_id"),
        summary_metrics={
            "score": int(quiz_attempt.get("score") or 0),
            "correct_answers": correct_answers,
            "total_questions": total_questions,
        },
        performance_signals={
            "incorrect_count": incorrect_count,
            "accuracy": accuracy,
        },
        source_bundle=_copy_source_bundle(source_bundle),
    )


__all__ = [
    "build_quiz_suggestion_context",
    "extract_quiz_suggestion_evidence",
]
