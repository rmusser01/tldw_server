"""Helpers for study-suggestion follow-up actions and duplicate-safe fingerprints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


DEFAULT_GENERATOR_VERSION = "v1"
LEGACY_NORMALIZATION_VERSION = "legacy"
PENDING_GENERATION_TARGET_PREFIX = "pending:"
FOLLOW_UP_ACTION_CONTRACTS = {
    "follow_up_quiz": {
        "target_service": "quiz",
        "target_type": "quiz",
    },
    "follow_up_flashcards": {
        "target_service": "flashcards",
        "target_type": "deck",
    },
}


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def normalize_selected_topics(selected_topics: Iterable[object]) -> list[str]:
    """Return a stable, deduplicated, sorted list of canonical topic labels."""

    normalized: set[str] = set()
    for topic in selected_topics:
        text = _normalize_text(topic)
        if text:
            normalized.add(text)
    return sorted(normalized)


def _iter_snapshot_topics(snapshot_row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = snapshot_row.get("payload_json") or {}
    topics = payload.get("topics") if isinstance(payload, Mapping) else []
    return [topic for topic in topics if isinstance(topic, Mapping)] if isinstance(topics, list) else []


def _iter_selected_snapshot_topics(
    snapshot_row: Mapping[str, Any],
    *,
    selected_topic_ids: Iterable[object],
    has_explicit_selection: bool = False,
) -> list[Mapping[str, Any]]:
    topics = _iter_snapshot_topics(snapshot_row)
    selected_ids = {str(topic_id).strip() for topic_id in selected_topic_ids if str(topic_id).strip()}
    use_default_selected = not has_explicit_selection and not selected_ids

    selected_topics: list[Mapping[str, Any]] = []
    for topic in topics:
        topic_id = str(topic.get("id") or "").strip()
        if use_default_selected:
            if not bool(topic.get("selected")):
                continue
        elif topic_id not in selected_ids:
            continue
        selected_topics.append(topic)
    return selected_topics


def _resolve_topic_semantic_key(topic: Mapping[str, Any]) -> str:
    for field_name in ("topic_key", "canonical_label", "display_label"):
        value = str(topic.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _topic_identity_tokens(topic: Mapping[str, Any]) -> frozenset[str]:
    identities = {
        normalized
        for field_name in ("topic_key", "canonical_label", "display_label")
        if (normalized := _normalize_text(topic.get(field_name)))
    }
    return frozenset(identities)


def _resolve_normalization_version_values(values: Iterable[object]) -> str:
    normalized = normalize_selected_topics(values)
    return ",".join(normalized) if normalized else LEGACY_NORMALIZATION_VERSION


def _resolve_snapshot_normalization_version(snapshot_row: Mapping[str, Any]) -> str:
    versions = [
        str(topic.get("normalization_version") or "").strip()
        for topic in _iter_snapshot_topics(snapshot_row)
        if str(topic.get("normalization_version") or "").strip()
    ]
    return _resolve_normalization_version_values(versions)


def build_selection_fingerprint(
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selected_topics: Iterable[object],
    action_kind: str,
    generator_version: str | None = None,
    normalization_version: str | None = None,
    include_normalization_version: bool = True,
) -> str:
    """Build a stable, human-readable fingerprint for one follow-up action selection."""

    normalized_topics = normalize_selected_topics(selected_topics)
    resolved_generator_version = _normalize_text(generator_version or DEFAULT_GENERATOR_VERSION) or DEFAULT_GENERATOR_VERSION
    resolved_normalization_version = _resolve_normalization_version_values(
        [normalization_version] if normalization_version is not None else []
    )
    components = [
        f"snapshot_id={int(snapshot_id)}",
        f"target_service={_normalize_text(target_service)}",
        f"target_type={_normalize_text(target_type)}",
        f"topics={','.join(normalized_topics)}",
        f"action_kind={_normalize_text(action_kind)}",
        f"generator_version={resolved_generator_version}",
    ]
    if include_normalization_version:
        components.append(f"normalization_version={resolved_normalization_version}")
    return "|".join(components)


def canonicalize_follow_up_action(
    *,
    target_service: str,
    target_type: str,
    action_kind: str,
) -> dict[str, str]:
    """Resolve one user request into the stable action contract used for dedupe."""

    normalized_action_kind = _normalize_text(action_kind)
    contract = FOLLOW_UP_ACTION_CONTRACTS.get(normalized_action_kind)
    if contract is None:
        raise ValueError("Unsupported study suggestion action kind")

    normalized_service = _normalize_text(target_service)
    normalized_type = _normalize_text(target_type)
    expected_service = str(contract["target_service"])
    expected_type = str(contract["target_type"])

    if normalized_service and normalized_service != expected_service:
        raise ValueError(f"Action {normalized_action_kind} must target {expected_service}/{expected_type}")
    if normalized_type and normalized_type != expected_type:
        raise ValueError(f"Action {normalized_action_kind} must target {expected_service}/{expected_type}")

    return {
        "action_kind": normalized_action_kind,
        "target_service": expected_service,
        "target_type": expected_type,
    }


def resolve_selected_topic_labels(
    snapshot_row: Mapping[str, Any],
    *,
    selected_topic_ids: Iterable[object],
    selected_topic_edits: Iterable[Mapping[str, object]] | None = None,
    manual_topic_labels: Iterable[object] | None = None,
    has_explicit_selection: bool = False,
) -> list[str]:
    """Resolve requested topic ids into display labels from the frozen snapshot payload."""

    edit_labels: dict[str, str] = {}
    for item in selected_topic_edits or []:
        if not isinstance(item, Mapping):
            continue
        topic_id = str(item.get("id") or "").strip()
        topic_label = str(item.get("label") or "").strip()
        if topic_id and topic_label:
            edit_labels[topic_id] = topic_label
    manual_labels = normalize_selected_topics(manual_topic_labels or [])
    labels: list[str] = []
    for topic in _iter_selected_snapshot_topics(
        snapshot_row,
        selected_topic_ids=selected_topic_ids,
        has_explicit_selection=has_explicit_selection,
    ):
        topic_id = str(topic.get("id") or "").strip()
        label = edit_labels.get(topic_id) or str(topic.get("display_label") or "").strip()
        if label:
            labels.append(label)
    labels.extend(manual_labels)
    return normalize_selected_topics(labels)


def resolve_selected_topic_semantic_keys(
    snapshot_row: Mapping[str, Any],
    *,
    selected_topic_ids: Iterable[object],
    manual_topic_labels: Iterable[object] | None = None,
    has_explicit_selection: bool = False,
) -> list[str]:
    """Resolve selected snapshot rows into semantic keys for fingerprinting."""

    manual_labels = normalize_selected_topics(manual_topic_labels or [])
    semantic_keys: list[str] = []
    for topic in _iter_selected_snapshot_topics(
        snapshot_row,
        selected_topic_ids=selected_topic_ids,
        has_explicit_selection=has_explicit_selection,
    ):
        semantic_key = _resolve_topic_semantic_key(topic)
        if semantic_key:
            semantic_keys.append(semantic_key)
    semantic_keys.extend(manual_labels)
    return normalize_selected_topics(semantic_keys)


def resolve_selected_topic_normalization_version(
    snapshot_row: Mapping[str, Any],
    *,
    selected_topic_ids: Iterable[object],
    has_explicit_selection: bool = False,
) -> str:
    """Resolve the deterministic normalization-version contract for one selection."""

    selected_versions = [
        str(topic.get("normalization_version") or "").strip()
        for topic in _iter_selected_snapshot_topics(
            snapshot_row,
            selected_topic_ids=selected_topic_ids,
            has_explicit_selection=has_explicit_selection,
        )
        if str(topic.get("normalization_version") or "").strip()
    ]
    if selected_versions:
        return _resolve_normalization_version_values(selected_versions)
    return _resolve_snapshot_normalization_version(snapshot_row)


def resolve_selected_topic_identity_groups(
    snapshot_row: Mapping[str, Any],
    *,
    selected_topic_ids: Iterable[object],
    has_explicit_selection: bool = False,
) -> list[frozenset[str]]:
    """Return comparable per-topic identity groups for refreshed-lineage matching."""

    identity_groups: list[frozenset[str]] = []
    for topic in _iter_selected_snapshot_topics(
        snapshot_row,
        selected_topic_ids=selected_topic_ids,
        has_explicit_selection=has_explicit_selection,
    ):
        identities = _topic_identity_tokens(topic)
        if identities:
            identity_groups.append(identities)
    return identity_groups


def resolve_lineage_equivalent_topic_selection(
    snapshot_row: Mapping[str, Any],
    *,
    requested_identity_groups: Iterable[Iterable[object]],
) -> tuple[list[str], str] | None:
    """Map the current semantic selection onto one ancestor snapshot's native topic identity."""

    normalized_groups: list[frozenset[str]] = []
    for group in requested_identity_groups:
        normalized_group = normalize_selected_topics(group)
        if normalized_group:
            normalized_groups.append(frozenset(normalized_group))
    if not normalized_groups:
        return None

    topics = _iter_snapshot_topics(snapshot_row)
    matched_semantic_keys: list[str] = []
    matched_versions: list[str] = []
    used_topic_indexes: set[int] = set()

    for identity_group in normalized_groups:
        matched_index: int | None = None
        for index, topic in enumerate(topics):
            if index in used_topic_indexes:
                continue
            if _topic_identity_tokens(topic) & identity_group:
                matched_index = index
                break
        if matched_index is None:
            return None

        matched_topic = topics[matched_index]
        used_topic_indexes.add(matched_index)
        semantic_key = _resolve_topic_semantic_key(matched_topic)
        if semantic_key:
            matched_semantic_keys.append(semantic_key)
        normalization_version = str(matched_topic.get("normalization_version") or "").strip()
        if normalization_version:
            matched_versions.append(normalization_version)

    if len(matched_semantic_keys) != len(normalized_groups):
        return None

    return (
        normalize_selected_topics(matched_semantic_keys),
        _resolve_normalization_version_values(matched_versions)
        if matched_versions
        else _resolve_snapshot_normalization_version(snapshot_row),
    )


def _titleize_label(value: object) -> str:
    words = [part for part in str(value or "").strip().split() if part]
    return " ".join(word.capitalize() for word in words)


def _get_flashcard_review_session(note_db: CharactersRAGDB, *, session_id: int) -> dict[str, Any] | None:
    return note_db.get_flashcard_review_session(int(session_id))


def build_follow_up_flashcard_deck_name(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    selected_topics: Iterable[object],
) -> str:
    """Return a unique deck name for one flashcard follow-up generation.

    Uses an optimistic approach: generate a candidate name, then verify it
    is not taken. If a concurrent worker claims the name between our check
    and the caller's insert, the caller should catch ``ConflictError`` from
    ``add_deck`` and retry with an incremented suffix.
    """

    normalized_topics = normalize_selected_topics(selected_topics)
    topic_fragment = ", ".join(_titleize_label(topic) for topic in normalized_topics[:2])
    base_name = f"Study Suggestions {int(snapshot_id)}"
    if topic_fragment:
        base_name = f"{base_name}: {topic_fragment}"

    candidate = base_name
    suffix = 2
    max_attempts = 20
    for _ in range(max_attempts):
        if note_db.get_deck_by_name(candidate) is None:
            return candidate
        candidate = f"{base_name} ({suffix})"
        suffix += 1
    return candidate


def build_follow_up_flashcard_source_text(
    note_db: CharactersRAGDB,
    *,
    snapshot_row: Mapping[str, Any],
    selected_topics: Iterable[object],
) -> str:
    """Build source text for follow-up flashcard generation from the snapshot anchor."""

    normalized_topics = normalize_selected_topics(selected_topics)
    anchor_type = _normalize_text(snapshot_row.get("anchor_type"))
    anchor_id = int(snapshot_row.get("anchor_id") or 0)

    if anchor_type == "quiz_attempt":
        attempt = note_db.get_attempt(anchor_id, include_questions=True, include_answers=True)
        if not attempt:
            raise ConflictError("Quiz attempt not found", entity="quiz_attempts", identifier=anchor_id)  # noqa: TRY003
        quiz = note_db.get_quiz(int(attempt["quiz_id"])) if attempt.get("quiz_id") is not None else None
        lines = ["Create focused follow-up flashcards from this quiz material."]
        if quiz and quiz.get("name"):
            lines.append(f"Quiz: {str(quiz['name']).strip()}")
        if normalized_topics:
            lines.append(f"Focus topics: {', '.join(normalized_topics)}")
        for question in (attempt.get("questions") or [])[:12]:
            if not isinstance(question, Mapping):
                continue
            question_text = str(question.get("question_text") or "").strip()
            explanation = str(question.get("explanation") or "").strip()
            hint = str(question.get("hint") or "").strip()
            if question_text:
                lines.append(f"Question: {question_text}")
            if explanation:
                lines.append(f"Explanation: {explanation}")
            if hint:
                lines.append(f"Hint: {hint}")
        return "\n".join(line for line in lines if line).strip()[:8000]

    if anchor_type == "flashcard_review_session":
        session = _get_flashcard_review_session(note_db, session_id=anchor_id)
        if not session:
            raise ConflictError(  # noqa: TRY003
                "Flashcard review session not found",
                entity="flashcard_review_sessions",
                identifier=anchor_id,
            )
        lines = ["Create focused follow-up flashcards from this review session."]
        deck_id = session.get("deck_id")
        if deck_id is not None:
            deck = note_db.get_deck(int(deck_id))
            if deck and deck.get("name"):
                lines.append(f"Deck: {str(deck['name']).strip()}")
        if session.get("tag_filter"):
            lines.append(f"Tag filter: {str(session['tag_filter']).strip()}")
        if normalized_topics:
            lines.append(f"Focus topics: {', '.join(normalized_topics)}")
        if deck_id is not None:
            cards = note_db.list_flashcards(
                deck_id=int(deck_id),
                tag=str(session.get("tag_filter") or "").strip() or None,
                due_status="all",
                limit=25,
            )
            for card in cards:
                front = str(card.get("front") or "").strip()
                back = str(card.get("back") or "").strip()
                if front:
                    lines.append(f"Front: {front}")
                if back:
                    lines.append(f"Back: {back}")
        return "\n".join(line for line in lines if line).strip()[:8000]

    topic_text = ", ".join(normalized_topics) or "study suggestions"
    return f"Create focused follow-up flashcards about: {topic_text}"[:8000]


def build_flashcard_generation_payload(
    *,
    deck_id: int,
    selected_topics: Iterable[object],
    raw_flashcards: Iterable[object],
) -> list[dict[str, Any]]:
    """Normalize generated flashcards into DB insert payloads for a persisted deck."""

    normalized_topics = normalize_selected_topics(selected_topics)
    cards: list[dict[str, Any]] = []
    for raw_card in raw_flashcards:
        if not isinstance(raw_card, Mapping):
            continue
        front = str(raw_card.get("front") or "").strip()
        back = str(raw_card.get("back") or "").strip()
        if not front or not back:
            continue
        tags: list[str] = list(normalized_topics)
        raw_tags = raw_card.get("tags")
        if isinstance(raw_tags, list):
            tags.extend(str(tag).strip() for tag in raw_tags if str(tag).strip())
        elif isinstance(raw_tags, str):
            tags.extend(token for token in raw_tags.replace(",", " ").split() if token)
        model_type = str(raw_card.get("model_type") or "basic").strip().lower() or "basic"
        if model_type not in {"basic", "basic_reverse", "cloze"}:
            model_type = "basic"
        cards.append(
            {
                "deck_id": int(deck_id),
                "front": front,
                "back": back,
                "notes": str(raw_card.get("notes") or "").strip() or None,
                "extra": str(raw_card.get("extra") or "").strip() or None,
                "tags_json": json.dumps(normalize_selected_topics(tags)),
                "source_ref_type": "manual",
                "source_ref_id": None,
                "model_type": model_type,
                "reverse": model_type == "basic_reverse",
                "is_cloze": model_type == "cloze",
            }
        )
    return cards


def find_generation_link_by_fingerprint(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selection_fingerprint: str,
) -> dict[str, Any] | None:
    """Find an existing generation link by fingerprint without requiring the target id."""

    return note_db.find_suggestion_generation_link_by_fingerprint(
        snapshot_id=int(snapshot_id),
        target_service=_normalize_text(target_service),
        target_type=_normalize_text(target_type),
        selection_fingerprint=selection_fingerprint,
    )


def build_pending_generation_target_id(selection_fingerprint: str) -> str:
    return f"{PENDING_GENERATION_TARGET_PREFIX}{selection_fingerprint}"


def is_pending_generation_target_id(target_id: object) -> bool:
    return str(target_id or "").startswith(PENDING_GENERATION_TARGET_PREFIX)


def reserve_generation_link(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selection_fingerprint: str,
) -> int:
    """Claim one in-progress generation slot for a selection fingerprint."""

    return note_db.create_suggestion_generation_link(
        snapshot_id=int(snapshot_id),
        target_service=_normalize_text(target_service),
        target_type=_normalize_text(target_type),
        target_id=build_pending_generation_target_id(selection_fingerprint),
        selection_fingerprint=selection_fingerprint,
    )


def finalize_generation_link(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selection_fingerprint: str,
    final_target_id: str,
) -> None:
    """Replace an in-progress reservation target id with the real durable target id."""

    final_target_id = str(final_target_id).strip()
    if not final_target_id:
        raise ValueError("final_target_id must not be empty")

    updated = note_db.finalize_suggestion_generation_link(
        snapshot_id=int(snapshot_id),
        target_service=_normalize_text(target_service),
        target_type=_normalize_text(target_type),
        selection_fingerprint=selection_fingerprint,
        final_target_id=final_target_id,
    )
    if updated == 0:
        raise ConflictError(  # noqa: TRY003
            "Suggestion generation reservation not found",
            entity="suggestion_generation_links",
            identifier=f"{snapshot_id}:{selection_fingerprint}",
        )


def release_generation_link_reservation(
    note_db: CharactersRAGDB,
    *,
    snapshot_id: int,
    target_service: str,
    target_type: str,
    selection_fingerprint: str,
) -> None:
    """Soft-delete an in-progress reservation after generation failure."""

    note_db.release_suggestion_generation_link_reservation(
        snapshot_id=int(snapshot_id),
        target_service=_normalize_text(target_service),
        target_type=_normalize_text(target_type),
        selection_fingerprint=selection_fingerprint,
    )


def soft_delete_deck(note_db: CharactersRAGDB, *, deck_id: int) -> None:
    """Best-effort cleanup for decks created during failed follow-up generation."""

    note_db.soft_delete_deck_by_id(int(deck_id))


__all__ = [
    "DEFAULT_GENERATOR_VERSION",
    "FOLLOW_UP_ACTION_CONTRACTS",
    "LEGACY_NORMALIZATION_VERSION",
    "PENDING_GENERATION_TARGET_PREFIX",
    "build_flashcard_generation_payload",
    "build_follow_up_flashcard_deck_name",
    "build_follow_up_flashcard_source_text",
    "build_pending_generation_target_id",
    "build_selection_fingerprint",
    "canonicalize_follow_up_action",
    "finalize_generation_link",
    "find_generation_link_by_fingerprint",
    "is_pending_generation_target_id",
    "normalize_selected_topics",
    "release_generation_link_reservation",
    "reserve_generation_link",
    "resolve_lineage_equivalent_topic_selection",
    "resolve_selected_topic_labels",
    "resolve_selected_topic_identity_groups",
    "resolve_selected_topic_normalization_version",
    "resolve_selected_topic_semantic_keys",
    "soft_delete_deck",
]
