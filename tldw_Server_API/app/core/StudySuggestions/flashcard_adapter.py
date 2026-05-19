"""Flashcard-specific suggestion context adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
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


def _parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_preserve(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return _unique_preserve([stripped])
        if isinstance(decoded, list):
            return _unique_preserve(decoded)
        if isinstance(decoded, str):
            return _unique_preserve([decoded])
    return []


def _normalize_source_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        nested_items = value.get("items")
        if isinstance(nested_items, list):
            return [item for item in nested_items if isinstance(item, Mapping)]
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _merge_flashcard_provenance(
    session: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(session)
    if not provenance:
        return merged

    for key in ("source_bundle", "study_pack_id", "deck_name", "tag_filter", "workspace_id"):
        if merged.get(key) is None and provenance.get(key) is not None:
            merged[key] = provenance.get(key)
    return merged


def extract_flashcard_suggestion_evidence(
    session: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract flashcard labels and light refs from a merged review-session view."""

    merged_session = _merge_flashcard_provenance(session, provenance)
    source_labels: list[str] = []
    tag_labels: list[str] = []
    derived_labels: list[str] = []
    adjacent_labels: list[str] = []
    source_bundle: list[dict[str, str]] = []

    def add_source_items(items: Any) -> None:
        for item in _normalize_source_items(items):
            source_type = _safe_text(item.get("source_type"))
            source_id = _safe_text(item.get("source_id"))
            label = _safe_text(item.get("label"))
            if not source_type or not source_id:
                continue
            if label:
                source_labels.append(label)
            source_item = {
                "source_type": source_type,
                "source_id": source_id,
            }
            if label:
                source_item["label"] = label
            source_bundle.append(source_item)

    add_source_items(merged_session.get("source_bundle"))

    study_pack = provenance.get("study_pack") if provenance else None
    if isinstance(study_pack, Mapping):
        add_source_items(study_pack.get("source_bundle_json"))
        if _safe_text(study_pack.get("title")):
            derived_labels.append(str(study_pack["title"]))

    tag_filter = _safe_text(merged_session.get("tag_filter"))
    if tag_filter:
        tag_labels.append(tag_filter)
        adjacent_labels.append(tag_filter)

    deck_name = _safe_text(merged_session.get("deck_name"))
    if deck_name:
        derived_labels.append(deck_name)

    reviewed_cards = provenance.get("reviewed_cards") if provenance else None
    if isinstance(reviewed_cards, Sequence):
        for card in reviewed_cards:
            if not isinstance(card, Mapping):
                continue
            tag_labels.extend(_parse_tags(card.get("tags_json")))
            source_type = _safe_text(card.get("source_ref_type"))
            source_id = _safe_text(card.get("source_ref_id"))
            if source_type and source_id and source_type != "manual":
                source_bundle.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                    }
                )

    if tag_labels:
        adjacent_labels.extend(tag_labels)
    elif derived_labels:
        adjacent_labels.extend(derived_labels)
    else:
        derived_labels.append("spaced repetition")
        adjacent_labels.append("spaced repetition")

    return {
        "source_labels": _unique_preserve(source_labels),
        "tag_labels": _unique_preserve(tag_labels),
        "derived_labels": _unique_preserve(derived_labels),
        "weakness_labels": [],
        "adjacent_labels": _unique_preserve(adjacent_labels),
        "source_bundle": _unique_sources(source_bundle),
    }


def is_source_grounded_session(session: Mapping[str, Any]) -> bool:
    source_bundle = session.get("source_bundle") or []
    if session.get("study_pack_id"):
        return True

    for item in source_bundle:
        if item.get("source_type") and item.get("source_id"):
            return True
        if item.get("citation_ordinal") is not None:
            return True
    return False


def build_flashcard_suggestion_context(
    session: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> SuggestionContext:
    merged_session = _merge_flashcard_provenance(session, provenance)
    cards_reviewed = int(merged_session.get("cards_reviewed") or 0)
    correct_count = int(merged_session.get("correct_count") or 0)
    deck_id = int(merged_session["deck_id"]) if merged_session.get("deck_id") is not None else None
    grounded = is_source_grounded_session(merged_session)
    review_mode = str(merged_session.get("review_mode") or "")
    supports_source_aware_adjacency = grounded and review_mode != "manual"

    return SuggestionContext(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=int(merged_session["id"]),
        workspace_id=merged_session.get("workspace_id"),
        summary_metrics={
            "deck_id": deck_id,
            "cards_reviewed": cards_reviewed,
            "correct_count": correct_count,
            "study_pack_id": merged_session.get("study_pack_id"),
        },
        performance_signals={
            "accuracy": (correct_count / cards_reviewed) if cards_reviewed else 0.0,
            "is_source_grounded_session": grounded,
            "supports_source_aware_adjacency": supports_source_aware_adjacency,
        },
        source_bundle=_copy_source_bundle(merged_session.get("source_bundle")),
    )


__all__ = [
    "build_flashcard_suggestion_context",
    "extract_flashcard_suggestion_evidence",
    "is_source_grounded_session",
]
