"""Plan lossless, bounded Study Pack prompts from saved media sections."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy

from .types import StudyPackSectionError

MAX_STUDY_PACK_PASSES = 256

# Use the Playground's chapter markers only at the start of a line.
_SECTION_HEADING = re.compile(
    r"^[ \t]*(?:\#{1,6}[ \t]+[^\r\n]+|"
    + EbookChapterChunkingStrategy.CHAPTER_PATTERNS["default"].replace(r"\s", r"[ \t]").replace("$", r"(?=\r?$)")
    + r"[^\r\n]*)",
    re.MULTILINE,
)
_MAX_SECTIONS = 512


@dataclass(frozen=True, slots=True)
class MediaSection:
    start: int
    end: int
    title: str


def bounded_spans(text: str, start: int, end: int, limit: int) -> Iterator[tuple[int, int]]:
    """Keep consecutive passages intact where paragraph or line breaks permit."""
    while start < end:
        stop = min(start + limit, end)
        if stop < end:
            floor = start + limit // 2
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, floor, stop)
                if boundary >= floor:
                    stop = boundary + len(separator)
                    break
        yield start, stop
        start = stop


def plan_media_sections(text: str, limit: int) -> tuple[str, list[MediaSection]]:
    """Return the same chapter/section plan for preview and generation."""
    digest = hashlib.sha256(f"{limit}\0".encode())
    digest.update(text.encode("utf-8"))
    if len(text) <= limit:
        return digest.hexdigest(), []

    starts = [0]
    titles: dict[int, str] = {}
    previous_end = None
    for heading in _SECTION_HEADING.finditer(text):
        # Keep adjacent headings with the body, and use the last heading as its title.
        has_body = previous_end is None or bool(text[previous_end:heading.start()].strip())
        if has_body and heading.start() != starts[-1]:
            starts.append(heading.start())
            if len(starts) > _MAX_SECTIONS:
                raise StudyPackSectionError("study_pack_sections_too_many")
        titles[starts[-1]] = heading.group().strip().lstrip("#").strip()
        previous_end = heading.end()

    spans = (
        zip(starts, [*starts[1:], len(text)], strict=True)
        if titles
        else bounded_spans(text, 0, len(text), limit)
    )
    sections: list[MediaSection] = []
    for start, end in spans:
        if not text[start:end].strip():
            continue
        title = titles.get(start, f"Section {len(sections) + 1}")
        sections.append(MediaSection(start, end, title[:120]))
        digest.update(f"\0{start}:{end}".encode())
        if len(sections) > _MAX_SECTIONS:
            raise StudyPackSectionError("study_pack_sections_too_many")
    return digest.hexdigest(), sections


def section_passages(text: str, section: MediaSection, limit: int) -> Iterator[tuple[int, int, str]]:
    """Yield the exact source spans and text used in model prompts."""
    for start, end in bounded_spans(text, section.start, section.end, limit):
        passage = text[start:end]
        trimmed = passage.strip()
        if trimmed:
            first_char = start + len(passage) - len(passage.lstrip())
            yield first_char, first_char + len(trimmed), trimmed


def selected_section_indices(
    locator: Mapping[str, Any], revision: str, sections: list[MediaSection]
) -> list[int]:
    if "section_revision" not in locator and "section_indices" not in locator:
        return list(range(len(sections)))
    indices = locator.get("section_indices")
    if not isinstance(indices, list) or not indices:
        raise StudyPackSectionError("study_pack_sections_invalid")
    if locator.get("section_revision") != revision:
        raise StudyPackSectionError("study_pack_sections_changed")
    if len(indices) > len(sections) or any(
        type(index) is not int or not 0 <= index < len(sections) for index in indices
    ):
        raise StudyPackSectionError("study_pack_sections_invalid")
    return sorted(set(indices))
