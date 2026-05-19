"""Immutable internal types used by study-pack resolution and persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_locator(locator: Mapping[str, Any] | None) -> dict[str, Any]:
    if locator is None:
        return {}
    return {str(key): value for key, value in locator.items()}


@dataclass(slots=True, frozen=True)
class StudySourceSelection:
    """Normalized internal representation of a requested study-pack source."""

    source_type: str
    source_id: str
    locator: dict[str, Any] = field(default_factory=dict)
    excerpt_text: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        source_type = _clean_text(self.source_type)
        source_id = _clean_text(self.source_id)
        excerpt_text = _clean_text(self.excerpt_text) or None
        label = _clean_text(self.label) or None
        if not source_type:
            raise ValueError("source_type must not be blank")
        if not source_id:
            raise ValueError("source_id must not be blank")
        if not isinstance(self.locator, Mapping):
            raise ValueError("locator must be a mapping")
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "locator", _normalize_locator(self.locator))
        object.__setattr__(self, "excerpt_text", excerpt_text)
        object.__setattr__(self, "label", label)


@dataclass(slots=True, frozen=True)
class StudySourceBundleItem:
    """Resolved evidence bundle entry ready for generation prompts and citations."""

    source_type: str
    source_id: str
    label: str
    evidence_text: str
    locator: dict[str, Any]

    def __post_init__(self) -> None:
        source_type = _clean_text(self.source_type)
        source_id = _clean_text(self.source_id)
        label = _clean_text(self.label)
        evidence_text = _clean_text(self.evidence_text)
        if not source_type:
            raise ValueError("source_type must not be blank")
        if not source_id:
            raise ValueError("source_id must not be blank")
        if not label:
            raise ValueError("label must not be blank")
        if not evidence_text:
            raise ValueError("evidence_text must not be blank")
        if not isinstance(self.locator, Mapping) or not self.locator:
            raise ValueError("locator must not be empty")
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "evidence_text", evidence_text)
        object.__setattr__(self, "locator", _normalize_locator(self.locator))


@dataclass(slots=True, frozen=True)
class StudySourceBundle:
    """Collection of resolved study sources used to generate one study pack."""

    items: list[StudySourceBundleItem]

    def __post_init__(self) -> None:
        normalized_items = list(self.items)
        for item in normalized_items:
            if not isinstance(item, StudySourceBundleItem):
                raise ValueError("StudySourceBundle items must be StudySourceBundleItem instances")
        object.__setattr__(self, "items", normalized_items)


@dataclass(slots=True, frozen=True)
class StudyCitationDraft:
    """Validated citation draft attached to a generated flashcard."""

    source_type: str
    source_id: str
    citation_text: str
    locator: dict[str, Any]

    def __post_init__(self) -> None:
        source_type = _clean_text(self.source_type)
        source_id = _clean_text(self.source_id)
        citation_text = _clean_text(self.citation_text)
        if not source_type:
            raise ValueError("source_type must not be blank")
        if not source_id:
            raise ValueError("source_id must not be blank")
        if not citation_text:
            raise ValueError("citation_text must not be blank")
        if not isinstance(self.locator, Mapping) or not self.locator:
            raise ValueError("locator must not be empty")
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "citation_text", citation_text)
        object.__setattr__(self, "locator", _normalize_locator(self.locator))


@dataclass(slots=True, frozen=True)
class StudyPackCardDraft:
    """Validated flashcard draft produced by the strict generation flow."""

    front: str
    back: str
    citations: list[StudyCitationDraft]
    model_type: str = "basic"
    notes: str | None = None
    extra: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        front = _clean_text(self.front)
        back = _clean_text(self.back)
        notes = _clean_text(self.notes) or None
        extra = _clean_text(self.extra) or None
        model_type = _clean_text(self.model_type).lower() or "basic"
        if not front:
            raise ValueError("front must not be blank")
        if not back:
            raise ValueError("back must not be blank")
        if model_type not in {"basic", "basic_reverse", "cloze"}:
            raise ValueError("model_type must be one of: basic, basic_reverse, cloze")

        normalized_citations = list(self.citations)
        if not normalized_citations:
            raise ValueError("citations must not be empty")
        for citation in normalized_citations:
            if not isinstance(citation, StudyCitationDraft):
                raise ValueError("citations must be StudyCitationDraft instances")

        normalized_tags: list[str] = []
        for tag in self.tags:
            normalized_tag = _clean_text(tag)
            if normalized_tag:
                normalized_tags.append(normalized_tag)

        object.__setattr__(self, "front", front)
        object.__setattr__(self, "back", back)
        object.__setattr__(self, "citations", normalized_citations)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "extra", extra)
        object.__setattr__(self, "tags", normalized_tags)


@dataclass(slots=True, frozen=True)
class StudyPackGenerationResult:
    """Validated generation output plus repair metadata."""

    cards: list[StudyPackCardDraft]
    raw_response: str | None = None
    repair_attempted: bool = False

    def __post_init__(self) -> None:
        normalized_cards = list(self.cards)
        if not normalized_cards:
            raise ValueError("cards must not be empty")
        for card in normalized_cards:
            if not isinstance(card, StudyPackCardDraft):
                raise ValueError("cards must be StudyPackCardDraft instances")
        object.__setattr__(self, "cards", normalized_cards)
        object.__setattr__(self, "raw_response", _clean_text(self.raw_response) or None)
        object.__setattr__(self, "repair_attempted", bool(self.repair_attempted))


@dataclass(slots=True, frozen=True)
class StudyPackCreationResult:
    """Persistence result returned after a study pack is committed."""

    pack_id: int
    deck_id: int
    deck_name: str
    card_uuids: list[str]
    cards: list[StudyPackCardDraft]
    repair_attempted: bool = False
    regenerated_from_pack_id: int | None = None

    def __post_init__(self) -> None:
        try:
            pack_id = int(self.pack_id)
            deck_id = int(self.deck_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("pack_id and deck_id must be integers") from exc
        if pack_id <= 0 or deck_id <= 0:
            raise ValueError("pack_id and deck_id must be positive")

        deck_name = _clean_text(self.deck_name)
        if not deck_name:
            raise ValueError("deck_name must not be blank")

        normalized_card_uuids = [_clean_text(card_uuid) for card_uuid in self.card_uuids if _clean_text(card_uuid)]
        normalized_cards = list(self.cards)
        if not normalized_card_uuids:
            raise ValueError("card_uuids must not be empty")
        if len(normalized_card_uuids) != len(normalized_cards):
            raise ValueError("card_uuids length must match cards length")
        for card in normalized_cards:
            if not isinstance(card, StudyPackCardDraft):
                raise ValueError("cards must be StudyPackCardDraft instances")

        regenerated_from_pack_id = self.regenerated_from_pack_id
        if regenerated_from_pack_id is not None:
            try:
                regenerated_from_pack_id = int(regenerated_from_pack_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("regenerated_from_pack_id must be an integer or None") from exc
            if regenerated_from_pack_id <= 0:
                raise ValueError("regenerated_from_pack_id must be positive when provided")

        object.__setattr__(self, "pack_id", pack_id)
        object.__setattr__(self, "deck_id", deck_id)
        object.__setattr__(self, "deck_name", deck_name)
        object.__setattr__(self, "card_uuids", normalized_card_uuids)
        object.__setattr__(self, "cards", normalized_cards)
        object.__setattr__(self, "repair_attempted", bool(self.repair_attempted))
        object.__setattr__(self, "regenerated_from_pack_id", regenerated_from_pack_id)


__all__ = [
    "StudyCitationDraft",
    "StudyPackCardDraft",
    "StudyPackCreationResult",
    "StudyPackGenerationResult",
    "StudySourceBundle",
    "StudySourceBundleItem",
    "StudySourceSelection",
]
