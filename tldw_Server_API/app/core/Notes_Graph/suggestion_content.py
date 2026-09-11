"""Canonical content and bounded evidence helpers for Notes suggestions."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal

CONTENT_FINGERPRINT_VERSION = b"notes-graph-content-v1"
MAX_SELECTED_NOTE_BYTES = 1_000_000
MAX_CANDIDATE_NOTE_BYTES = 250_000
MAX_SOURCE_WINDOWS = 4
MAX_CANDIDATE_WINDOWS = 2
MAX_WINDOW_CODE_POINTS = 480

EvidenceField = Literal["title", "content"]


@dataclass(frozen=True, slots=True)
class CanonicalNoteContent:
    """Normalized title and Markdown body used for fingerprints and offsets."""

    title: str
    content: str


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One fingerprint-bound half-open excerpt range in a canonical field."""

    note_id: str
    field: EvidenceField
    fingerprint: str
    start_offset: int
    end_offset: int


def normalize_note_text(value: str | None) -> str:
    """Normalize line endings and Unicode exactly once at the content boundary."""

    return unicodedata.normalize("NFC", (value or "").replace("\r\n", "\n").replace("\r", "\n"))


def canonicalize_note_content(title: str | None, content: str | None) -> CanonicalNoteContent:
    """Return the two independently canonicalized Notes fields."""

    return CanonicalNoteContent(
        title=normalize_note_text(title),
        content=normalize_note_text(content),
    )


def canonical_content_bytes(title: str | None, content: str | None) -> bytes:
    """Return the versioned byte sequence used for a Notes content fingerprint."""

    canonical = canonicalize_note_content(title, content)
    return b"\0".join(
        (CONTENT_FINGERPRINT_VERSION, canonical.title.encode("utf-8"), canonical.content.encode("utf-8"))
    )


def content_fingerprint(title: str | None, content: str | None) -> str:
    """Return the canonical SHA-256 content fingerprint with its algorithm prefix."""

    return f"sha256:{hashlib.sha256(canonical_content_bytes(title, content)).hexdigest()}"


def stored_text_utf8_bytes(title: str | None, content: str | None) -> int:
    """Return the stored title/body UTF-8 length before canonicalization."""

    return len((title or "").encode("utf-8")) + len((content or "").encode("utf-8"))


def reconstruct_evidence(
    evidence: EvidenceReference,
    *,
    title: str | None,
    content: str | None,
) -> str | None:
    """Reconstruct one valid excerpt only when its current fingerprint still matches."""

    if content_fingerprint(title, content) != evidence.fingerprint:
        return None
    canonical = canonicalize_note_content(title, content)
    field_text = canonical.title if evidence.field == "title" else canonical.content
    if (
        evidence.start_offset < 0
        or evidence.end_offset <= evidence.start_offset
        or evidence.end_offset > len(field_text)
    ):
        return None
    return field_text[evidence.start_offset : evidence.end_offset]


def split_evidence_windows(
    *,
    note_id: str,
    title: str | None,
    content: str | None,
    max_windows: int,
    max_code_points: int,
) -> tuple[EvidenceReference, ...]:
    """Split canonical fields into bounded, source-ordered, non-overlapping windows."""

    if not 1 <= max_windows <= MAX_SOURCE_WINDOWS:
        raise ValueError(f"max_windows must be between 1 and {MAX_SOURCE_WINDOWS}")
    if not 1 <= max_code_points <= MAX_WINDOW_CODE_POINTS:
        raise ValueError(f"max_code_points must be between 1 and {MAX_WINDOW_CODE_POINTS}")

    canonical = canonicalize_note_content(title, content)
    fingerprint = content_fingerprint(canonical.title, canonical.content)
    windows: list[EvidenceReference] = []
    for field, field_text in (("title", canonical.title), ("content", canonical.content)):
        for start_offset in range(0, len(field_text), max_code_points):
            if len(windows) == max_windows:
                return tuple(windows)
            end_offset = min(start_offset + max_code_points, len(field_text))
            windows.append(
                EvidenceReference(
                    note_id=note_id,
                    field=field,  # type: ignore[arg-type]
                    fingerprint=fingerprint,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )
    return tuple(windows)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens when an exact provider tokenizer is unavailable."""

    return (len(text) + 3) // 4
