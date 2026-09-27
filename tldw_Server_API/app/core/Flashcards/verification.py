"""Claim construction for verification of generated flashcards."""

from typing import Any

from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import ArtifactVerificationUnit


def build_flashcard_verification_units(cards: list[dict[str, Any]]) -> list[ArtifactVerificationUnit]:
    """Build factual claim units, retaining question context for answer checks."""
    units: list[ArtifactVerificationUnit] = []
    for index, card in enumerate(cards, start=1):
        model_type = str(card.get("model_type") or "basic").lower()
        front = str(card.get("front") or "").strip()
        back = str(card.get("back") or "").strip()
        notes = str(card.get("notes") or "").strip()
        extra = str(card.get("extra") or "").strip()

        # Questions are prompts, not claims. Cloze fronts are factual statements.
        if front and (model_type == "cloze" or not front.endswith("?")):
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"flashcard:{index}:front",
                    text=front,
                    claims=[front],
                )
            )
        if back:
            answer_claim = f"Question: {front}\nAnswer: {back}" if front else back
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"flashcard:{index}:back",
                    text=answer_claim,
                    claims=[answer_claim],
                    metadata={"requires_semantic_verification": True},
                )
            )
        if notes:
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"flashcard:{index}:notes",
                    text=notes,
                    claims=[notes],
                )
            )
        if extra:
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"flashcard:{index}:extra",
                    text=extra,
                    claims=[extra],
                )
            )
    return units
