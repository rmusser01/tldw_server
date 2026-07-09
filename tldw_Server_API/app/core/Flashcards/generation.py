"""Helpers for flashcard generation request planning and output normalization."""

from typing import Any

from tldw_Server_API.app.api.v1.schemas.flashcards import FlashcardGenerateRequest


class FlashcardGenerationPlanError(ValueError):
    """Raised when generated cards do not satisfy a requested card plan."""


def _truncate_test_mode_flashcard_text(text: str, limit: int = 220) -> str:
    """Return compact source text for deterministic test-mode flashcards."""
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _get_flashcard_generation_plan(payload: FlashcardGenerateRequest) -> list[dict[str, Any]]:
    """Return adapter-ready generation rows for planned or legacy requests."""
    if payload.card_plan is not None:
        return [{"card_type": item.card_type, "count": item.count} for item in payload.card_plan]
    return [{"card_type": payload.card_type or "basic", "count": int(payload.num_cards or 10)}]


def _expected_flashcard_plan_counts(payload: FlashcardGenerateRequest) -> dict[str, int]:
    """Return expected generated-card counts keyed by generation type."""
    return {row["card_type"]: int(row["count"]) for row in _get_flashcard_generation_plan(payload)}


def _storage_model_for_generation_type(generation_type: str) -> str:
    """Map generation-only card types to supported flashcard storage models."""
    return "basic" if generation_type == "true_false" else generation_type


def _build_test_mode_flashcards(payload: FlashcardGenerateRequest) -> list[dict[str, Any]]:
    """Build deterministic flashcards that match the requested generation plan."""
    normalized_text = _truncate_test_mode_flashcard_text(payload.text) or "Workspace study aid coverage."
    tags = [str(topic).strip() for topic in (payload.focus_topics or []) if str(topic).strip()]
    if not tags:
        tags = ["workspace", "study"]

    cards: list[dict[str, Any]] = []
    index = 0
    for row in _get_flashcard_generation_plan(payload):
        generation_type = str(row["card_type"]).strip().lower() or "basic"
        for _ in range(max(1, int(row["count"]))):
            if generation_type == "cloze":
                front = f"{{{{c1::Study point {index + 1}}}}}: {normalized_text}"
                back = f"Study point {index + 1}"
            elif generation_type == "true_false":
                front = f"True or false: Study point {index + 1} is covered by this source."
                back = f"True. {normalized_text}"
            else:
                front = f"What study point {index + 1} should you remember?"
                back = normalized_text

            cards.append(
                {
                    "front": front,
                    "back": back,
                    "tags": tags,
                    "model_type": _storage_model_for_generation_type(generation_type),
                    "generation_type": generation_type,
                    "notes": "Deterministic test-mode flashcard.",
                }
            )
            index += 1
    return cards


def _normalize_generated_flashcards(
    raw_flashcards: Any,
    payload: FlashcardGenerateRequest,
) -> list[dict[str, Any]]:
    """Normalize adapter flashcards and enforce planned generation counts."""
    valid_generation_types = ("basic", "basic_reverse", "cloze", "true_false")
    generated_cards: list[dict[str, Any]] = []
    for raw in raw_flashcards or []:
        if not isinstance(raw, dict):
            continue
        front = str(raw.get("front") or "").strip()
        back = str(raw.get("back") or "").strip()
        if not front or not back:
            continue

        if payload.card_plan is not None:
            raw_generation_type = str(raw.get("generation_type") or "").lower()
            if raw_generation_type not in valid_generation_types:
                raise FlashcardGenerationPlanError(
                    "Generated flashcards did not satisfy requested card plan: "
                    "missing or invalid generation_type"
                )
        else:
            raw_generation_type = str(
                raw.get("generation_type") or raw.get("model_type") or payload.card_type or "basic"
            ).lower()
            if raw_generation_type not in valid_generation_types:
                raw_generation_type = payload.card_type or "basic"

        tags_value = raw.get("tags")
        if isinstance(tags_value, list):
            tags = [str(tag).strip() for tag in tags_value if str(tag).strip()]
        elif isinstance(tags_value, str):
            tags = [token for token in tags_value.replace(",", " ").split() if token]
        else:
            tags = []

        model_type = _storage_model_for_generation_type(raw_generation_type)
        if model_type not in ("basic", "basic_reverse", "cloze"):
            model_type = "basic"

        card = {
            "front": front,
            "back": back,
            "tags": tags,
            "model_type": model_type,
            "generation_type": raw_generation_type,
        }
        notes = raw.get("notes")
        if isinstance(notes, str) and notes.strip():
            card["notes"] = notes
        extra = raw.get("extra")
        if isinstance(extra, str) and extra.strip():
            card["extra"] = extra
        generated_cards.append(card)

    if payload.card_plan is not None:
        expected_counts = _expected_flashcard_plan_counts(payload)
        actual_counts: dict[str, int] = {}
        for card in generated_cards:
            generation_type = str(card.get("generation_type") or "")
            actual_counts[generation_type] = actual_counts.get(generation_type, 0) + 1
        for card_type in list(expected_counts) + [key for key in actual_counts if key not in expected_counts]:
            expected = expected_counts.get(card_type, 0)
            actual = actual_counts.get(card_type, 0)
            if actual != expected:
                raise FlashcardGenerationPlanError(
                    "Generated flashcards did not satisfy requested card plan: "
                    f"{card_type} expected {expected}, got {actual}"
                )

    return generated_cards
