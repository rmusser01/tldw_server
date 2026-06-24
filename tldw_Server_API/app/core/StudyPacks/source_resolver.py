"""Resolve study-pack source selections into evidence-backed prompt bundles."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_latest_transcription,
    get_media_by_id,
    get_unvectorized_chunks_in_range,
)

from .types import (
    StudySourceBundle,
    StudySourceBundleItem,
    StudySourceSelection,
)

SUPPORTED_SOURCE_TYPES = frozenset({"note", "media", "message"})
_DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE = 12_000
_MAX_EVIDENCE_CHARS_ENV = "STUDY_PACK_MAX_EVIDENCE_CHARS_PER_SOURCE"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_selection(selection: StudySourceSelection | Mapping[str, Any]) -> StudySourceSelection:
    if hasattr(selection, "model_dump") and callable(selection.model_dump):
        return StudySourceSelection(**selection.model_dump())
    if isinstance(selection, StudySourceSelection):
        return selection
    if isinstance(selection, Mapping):
        return StudySourceSelection(**selection)
    raise ValueError("Each study source selection must be a mapping or StudySourceSelection instance")


def _parse_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _parse_non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _parse_non_negative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return parsed


def _max_evidence_chars_per_source() -> int:
    configured_limit = _clean_text(os.getenv(_MAX_EVIDENCE_CHARS_ENV))
    if not configured_limit:
        return _DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE
    try:
        parsed_limit = int(configured_limit)
    except ValueError:
        logger.warning(
            "Ignoring invalid {} value {!r}; using default {}",
            _MAX_EVIDENCE_CHARS_ENV,
            configured_limit,
            _DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE,
        )
        return _DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE
    if parsed_limit <= 0:
        logger.warning(
            "Ignoring non-positive {} value {}; using default {}",
            _MAX_EVIDENCE_CHARS_ENV,
            parsed_limit,
            _DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE,
        )
        return _DEFAULT_MAX_EVIDENCE_CHARS_PER_SOURCE
    return parsed_limit


def _normalized_for_match(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _candidate_contains_excerpt(candidate: str, excerpt_text: str) -> bool:
    if excerpt_text in candidate:
        return True
    normalized_candidate = _normalized_for_match(candidate)
    normalized_excerpt = _normalized_for_match(excerpt_text)
    return bool(normalized_excerpt and normalized_excerpt in normalized_candidate)


def _pick_evidence_text(selection: StudySourceSelection, *candidates: Any) -> str:
    for candidate in candidates:
        text = _clean_text(candidate)
        if text:
            excerpt_text = _clean_text(selection.excerpt_text)
            if excerpt_text:
                if not _candidate_contains_excerpt(text, excerpt_text):
                    raise ValueError("excerpt_text must match resolved source evidence")
                return excerpt_text
            return text
    return ""


def _bounded_evidence_text(evidence_text: str, locator: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_locator = dict(locator)
    max_chars = _max_evidence_chars_per_source()
    if len(evidence_text) <= max_chars:
        return evidence_text, normalized_locator
    normalized_locator["evidence_truncated"] = True
    normalized_locator["evidence_original_chars"] = len(evidence_text)
    return evidence_text[:max_chars], normalized_locator


def _prepare_evidence(
    selection: StudySourceSelection,
    locator: Mapping[str, Any],
    *candidates: Any,
) -> tuple[str, dict[str, Any]]:
    evidence_text = _pick_evidence_text(selection, *candidates)
    if not evidence_text:
        return "", dict(locator)
    return _bounded_evidence_text(evidence_text, locator)


class StudySourceResolver:
    """Resolves supported workspace objects into stable study-pack evidence bundles."""

    def __init__(self, *, db: Any | None = None, media_db: Any | None = None):
        self.db = db
        self.media_db = media_db

    def resolve(
        self,
        selections: Sequence[StudySourceSelection | Mapping[str, Any]],
    ) -> StudySourceBundle:
        if not selections:
            raise ValueError("At least one study source selection is required")

        items: list[StudySourceBundleItem] = []
        for raw_selection in selections:
            selection = _coerce_selection(raw_selection)
            items.append(self._resolve_selection(selection))
        return StudySourceBundle(items=items)

    def _resolve_selection(self, selection: StudySourceSelection) -> StudySourceBundleItem:
        if selection.source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported study source type: {selection.source_type}")
        if selection.source_type == "note":
            return self._resolve_note(selection)
        if selection.source_type == "media":
            return self._resolve_media(selection)
        return self._resolve_message(selection)

    def _resolve_note(self, selection: StudySourceSelection) -> StudySourceBundleItem:
        if self.db is None or not callable(getattr(self.db, "get_note_by_id", None)):
            raise ValueError("Note source resolution requires a notes-capable db")

        note = self.db.get_note_by_id(selection.source_id)
        if not note:
            raise ValueError(f"Note '{selection.source_id}' not found")

        label = selection.label or _clean_text(note.get("title")) or f"Note {selection.source_id}"
        locator = {**selection.locator, "note_id": selection.source_id}
        evidence_text, locator = _prepare_evidence(selection, locator, note.get("content"))
        if not evidence_text:
            raise ValueError(f"Note '{selection.source_id}' has no evidence text")

        return StudySourceBundleItem(
            source_type="note",
            source_id=selection.source_id,
            label=label,
            evidence_text=evidence_text,
            locator=locator,
        )

    def _resolve_message(self, selection: StudySourceSelection) -> StudySourceBundleItem:
        if self.db is None or not callable(getattr(self.db, "get_message_by_id", None)):
            raise ValueError("Message source resolution requires a chat-capable db")

        message = self.db.get_message_by_id(selection.source_id)
        if not message:
            raise ValueError(f"Message '{selection.source_id}' not found")

        message_id = _clean_text(message.get("id"))
        conversation_id = _clean_text(message.get("conversation_id"))
        if not message_id or not conversation_id:
            raise ValueError(
                "Message source requires both stable message identity and conversation identity"
            )

        label = selection.label or _clean_text(message.get("sender")) or f"Message {message_id}"
        locator = {
            **selection.locator,
            "conversation_id": conversation_id,
            "message_id": message_id,
        }
        evidence_text, locator = _prepare_evidence(selection, locator, message.get("content"))
        if not evidence_text:
            raise ValueError(f"Message '{message_id}' has no evidence text")

        return StudySourceBundleItem(
            source_type="message",
            source_id=message_id,
            label=label,
            evidence_text=evidence_text,
            locator=locator,
        )

    def _resolve_media(self, selection: StudySourceSelection) -> StudySourceBundleItem:
        if self.media_db is None:
            raise ValueError("Media source resolution requires a media db")

        media_id = _parse_positive_int(selection.source_id, "media source_id")
        media = get_media_by_id(
            self.media_db,
            media_id,
            include_deleted=False,
            include_trash=False,
        )
        if not media:
            raise ValueError(f"Media {media_id} not found")

        label = selection.label or _clean_text(media.get("title")) or f"Media {media_id}"
        locator = dict(selection.locator)
        chunk_index = locator.get("chunk_index")
        if chunk_index is not None:
            chunk_index = _parse_non_negative_int(chunk_index, "chunk_index")
            try:
                chunks = get_unvectorized_chunks_in_range(
                    self.media_db,
                    media_id,
                    chunk_index,
                    chunk_index,
                )
            except DatabaseError as exc:
                logger.debug(
                    "Chunk lookup failed for study source media {} chunk {}: {}",
                    media_id,
                    chunk_index,
                    exc,
                )
                chunks = []
            evidence_parts = [
                _clean_text(chunk.get("chunk_text"))
                for chunk in chunks
                if _clean_text(chunk.get("chunk_text"))
            ]
            if evidence_parts:
                chunk_id = _clean_text(chunks[0].get("uuid")) or str(chunk_index)
                chunk_locator = _bounded_evidence_text(
                    _pick_evidence_text(selection, "\n\n".join(evidence_parts)),
                    {
                        "media_id": media_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                )
                evidence_text, normalized_locator = chunk_locator
                return StudySourceBundleItem(
                    source_type="media",
                    source_id=str(media_id),
                    label=label,
                    evidence_text=evidence_text,
                    locator=normalized_locator,
                )

        timestamp_value = locator.get("timestamp_seconds")
        transcript = _clean_text(get_latest_transcription(self.media_db, media_id))
        normalized_locator: dict[str, Any] = {"media_id": media_id}
        if timestamp_value is not None:
            normalized_locator["timestamp_seconds"] = _parse_non_negative_float(
                timestamp_value,
                "timestamp_seconds",
            )
        evidence_text, normalized_locator = _prepare_evidence(selection, normalized_locator, transcript)
        if not evidence_text:
            raise ValueError(f"Media {media_id} has no transcript evidence to resolve")

        return StudySourceBundleItem(
            source_type="media",
            source_id=str(media_id),
            label=label,
            evidence_text=evidence_text,
            locator=normalized_locator,
        )


__all__ = ["SUPPORTED_SOURCE_TYPES", "StudySourceResolver"]
