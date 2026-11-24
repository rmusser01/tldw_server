from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_latest_transcription,
)

MAX_EVIDENCE_TEXT_CHARS = 12000
MAX_EVIDENCE_ITEMS = 5000


def _clip_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _extract_source_parts(source: Any) -> tuple[str, str]:
    if isinstance(source, Mapping):
        source_type = str(source.get("source_type") or "").strip()
        source_id = str(source.get("source_id") or "").strip()
    else:
        source_type = str(getattr(source, "source_type", "") or "").strip()
        source_id = str(getattr(source, "source_id", "") or "").strip()
    if not source_type or not source_id:
        raise ValueError("Each source must include non-empty source_type and source_id")
    return source_type, source_id


def _parse_positive_int(identifier: str, field_name: str) -> int:
    try:
        parsed = int(identifier)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _build_flashcard_text(card: Mapping[str, Any]) -> str:
    parts: list[str] = []
    front = str(card.get("front") or "").strip()
    back = str(card.get("back") or "").strip()
    notes = str(card.get("notes") or "").strip()
    extra = str(card.get("extra") or "").strip()
    if front:
        parts.append(f"Front: {front}")
    if back:
        parts.append(f"Back: {back}")
    if notes:
        parts.append(f"Notes: {notes}")
    if extra:
        parts.append(f"Extra: {extra}")
    return "\n".join(parts).strip()


def _parse_attempt_question_identifier(identifier: str) -> tuple[int, int]:
    raw = str(identifier or "").strip()
    if ":" not in raw:
        raise ValueError("quiz_attempt_question source_id must be formatted as '<attempt_id>:<question_id>'")
    attempt_part, question_part = raw.split(":", 1)
    return (
        _parse_positive_int(attempt_part, "quiz_attempt_question attempt_id"),
        _parse_positive_int(question_part, "quiz_attempt_question question_id"),
    )


def _build_quiz_attempt_question_text(question: Mapping[str, Any], answer: Mapping[str, Any] | None) -> str:
    parts: list[str] = []
    question_text = str(question.get("question_text") or "").strip()
    if question_text:
        parts.append(f"Question: {question_text}")

    user_answer = None if not answer else answer.get("user_answer")
    if user_answer is not None:
        parts.append(f"User answer: {user_answer}")

    if answer and answer.get("is_correct") is not None:
        parts.append(f"Is correct: {bool(answer.get('is_correct'))}")

    correct_answer = answer.get("correct_answer") if answer else question.get("correct_answer")
    if correct_answer is not None:
        parts.append(f"Correct answer: {correct_answer}")

    explanation = str((answer or {}).get("explanation") or question.get("explanation") or "").strip()
    if explanation:
        parts.append(f"Explanation: {explanation}")

    citations = (answer or {}).get("source_citations") or question.get("source_citations") or []
    if citations:
        citation_text = "; ".join(json.dumps(citation, sort_keys=True) for citation in citations if isinstance(citation, Mapping))
        if citation_text:
            parts.append(f"Source citations: {citation_text}")

    return "\n".join(parts).strip()


def _resolve_media_source(
    source_id: str,
    *,
    media_db: Any,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    media_id = _parse_positive_int(source_id, "media source_id")
    media = media_db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        raise ValueError(f"Media {media_id} not found")

    content = str(media.get("content") or "").strip()
    if not content:
        content = (get_latest_transcription(media_db, media_id) or "").strip()
    if not content:
        raise ValueError(f"Media {media_id} has no content to generate quiz from")

    return [
        {
            "source_type": "media",
            "source_id": str(media_id),
            "chunk_id": str(media_id),
            "label": str(media.get("title") or f"Media {media_id}").strip() or f"Media {media_id}",
            "text": _clip_text(content, max_chars_per_item),
        }
    ]


def _resolve_note_source(
    source_id: str,
    *,
    db: CharactersRAGDB,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    note = db.get_note_by_id(source_id)
    if not note:
        raise ValueError(f"Note '{source_id}' not found")

    title = str(note.get("title") or "").strip()
    content = str(note.get("content") or "").strip()
    if not content:
        raise ValueError(f"Note '{source_id}' has no content to generate quiz from")

    note_text = content if not title else f"Title: {title}\n\n{content}"
    return [
        {
            "source_type": "note",
            "source_id": source_id,
            "chunk_id": source_id,
            "label": title or f"Note {source_id}",
            "text": _clip_text(note_text, max_chars_per_item),
        }
    ]


def _resolve_flashcard_deck_source(
    source_id: str,
    *,
    db: CharactersRAGDB,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    deck_id = _parse_positive_int(source_id, "flashcard_deck source_id")
    deck = db.flashcard_store.get_deck(deck_id)
    if not deck:
        raise ValueError(f"Flashcard deck {deck_id} not found")

    cards = db.flashcard_store.list_flashcards(
        deck_id=deck_id,
        include_deleted=False,
        due_status="all",
        limit=10000,
        offset=0,
        order_by="due_at",
    )
    if not cards:
        raise ValueError(f"Flashcard deck {deck_id} has no cards")

    evidence: list[dict[str, Any]] = []
    deck_name = str(deck.get("name") or f"Deck {deck_id}")
    for idx, card in enumerate(cards):
        card_text = _build_flashcard_text(card)
        if not card_text:
            continue
        card_uuid = str(card.get("uuid") or "").strip()
        evidence.append(
            {
                "source_type": "flashcard_deck",
                "source_id": str(deck_id),
                "chunk_id": card_uuid or f"{deck_id}-{idx}",
                "label": deck_name,
                "text": _clip_text(card_text, max_chars_per_item),
            }
        )
    if not evidence:
        raise ValueError(f"Flashcard deck {deck_id} has no usable card content")
    return evidence


def _resolve_flashcard_card_source(
    source_id: str,
    *,
    db: CharactersRAGDB,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    card = db.flashcard_store.get_flashcard(source_id)
    if not card:
        raise ValueError(f"Flashcard '{source_id}' not found")

    card_text = _build_flashcard_text(card)
    if not card_text:
        raise ValueError(f"Flashcard '{source_id}' has no usable content")

    deck_name = str(card.get("deck_name") or "").strip()
    return [
        {
            "source_type": "flashcard_card",
            "source_id": source_id,
            "chunk_id": source_id,
            "label": deck_name or f"Card {source_id}",
            "text": _clip_text(card_text, max_chars_per_item),
        }
    ]


def _resolve_quiz_attempt_source(
    source_id: str,
    *,
    db: CharactersRAGDB,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    attempt_id = _parse_positive_int(source_id, "quiz_attempt source_id")
    attempt = db.get_attempt(attempt_id, include_questions=True, include_answers=True)
    if not attempt:
        raise ValueError(f"Quiz attempt {attempt_id} not found")

    questions = attempt.get("questions") or []
    answers = attempt.get("answers") or []
    answers_by_question_id = {
        int(item.get("question_id")): item
        for item in answers
        if item.get("question_id") is not None
    }

    question_rows: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for question in questions:
        question_id = question.get("id")
        if question_id is None:
            continue
        answer = answers_by_question_id.get(int(question_id))
        if answer and answer.get("is_correct") is False:
            question_rows.append((question, answer))

    if not question_rows:
        for question in questions:
            question_id = question.get("id")
            if question_id is None:
                continue
            question_rows.append((question, answers_by_question_id.get(int(question_id))))

    if not question_rows:
        raise ValueError(f"Quiz attempt {attempt_id} has no question evidence")

    evidence: list[dict[str, Any]] = []
    for question, answer in question_rows:
        question_id = int(question.get("id"))
        evidence.append(
            {
                "source_type": "quiz_attempt",
                "source_id": str(attempt_id),
                "chunk_id": f"{attempt_id}:{question_id}",
                "label": f"Quiz Attempt {attempt_id}",
                "text": _clip_text(_build_quiz_attempt_question_text(question, answer), max_chars_per_item),
            }
        )
    return evidence


def _resolve_quiz_attempt_question_source(
    source_id: str,
    *,
    db: CharactersRAGDB,
    max_chars_per_item: int,
) -> list[dict[str, Any]]:
    attempt_id, question_id = _parse_attempt_question_identifier(source_id)
    attempt = db.get_attempt(attempt_id, include_questions=True, include_answers=True)
    if not attempt:
        raise ValueError(f"Quiz attempt {attempt_id} not found")

    questions = attempt.get("questions") or []
    question = next((item for item in questions if int(item.get("id")) == question_id), None)
    if not question:
        raise ValueError(f"Quiz attempt question {source_id} not found")

    answers = attempt.get("answers") or []
    answer = next((item for item in answers if int(item.get("question_id")) == question_id), None)
    text = _build_quiz_attempt_question_text(question, answer)
    if not text:
        raise ValueError(f"Quiz attempt question {source_id} has no usable evidence")

    return [
        {
            "source_type": "quiz_attempt_question",
            "source_id": source_id,
            "chunk_id": source_id,
            "label": f"Quiz Attempt Question {source_id}",
            "text": _clip_text(text, max_chars_per_item),
        }
    ]


def resolve_quiz_sources(
    sources: Sequence[Any],
    *,
    db: CharactersRAGDB,
    media_db: MediaDatabase,
    max_chars_per_item: int = MAX_EVIDENCE_TEXT_CHARS,
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> list[dict[str, Any]]:
    """Resolve mixed source inputs into normalized evidence items for quiz generation."""
    if not sources:
        raise ValueError("At least one source is required")

    evidence: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    for source in sources:
        source_type, source_id = _extract_source_parts(source)
        source_key = (source_type, source_id)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        if source_type == "media":
            resolved = _resolve_media_source(source_id, media_db=media_db, max_chars_per_item=max_chars_per_item)
        elif source_type == "note":
            resolved = _resolve_note_source(source_id, db=db, max_chars_per_item=max_chars_per_item)
        elif source_type == "flashcard_deck":
            resolved = _resolve_flashcard_deck_source(source_id, db=db, max_chars_per_item=max_chars_per_item)
        elif source_type == "flashcard_card":
            resolved = _resolve_flashcard_card_source(source_id, db=db, max_chars_per_item=max_chars_per_item)
        elif source_type == "quiz_attempt":
            resolved = _resolve_quiz_attempt_source(source_id, db=db, max_chars_per_item=max_chars_per_item)
        elif source_type == "quiz_attempt_question":
            resolved = _resolve_quiz_attempt_question_source(source_id, db=db, max_chars_per_item=max_chars_per_item)
        else:
            raise ValueError(f"Unsupported source_type '{source_type}'")

        evidence.extend(resolved)
        if len(evidence) >= max_items:
            return evidence[:max_items]

    if not evidence:
        raise ValueError("No evidence could be resolved from selected sources")
    return evidence
