from __future__ import annotations

from pathlib import Path
import re

from .base import ParsedDocument, ParsedSection, file_uri

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_markdown(path: Path, text: str, document_type: str) -> ParsedDocument:
    sections: list[ParsedSection] = []
    title = path.stem
    title_set_from_heading = False
    fence_marker: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        fence_match = FENCE_RE.match(line)
        if fence_match and fence_marker is None:
            fence_marker = fence_match.group(1)
            offset += len(line)
            continue
        if fence_match and fence_match.group(1) == fence_marker:
            fence_marker = None
            offset += len(line)
            continue
        if fence_marker is not None:
            offset += len(line)
            continue
        match = HEADING_RE.match(line.strip())
        if match:
            heading = match.group(2).strip()
            if not title_set_from_heading:
                title = heading
                title_set_from_heading = True
            sections.append(
                ParsedSection(
                    heading=heading,
                    level=len(match.group(1)),
                    start_char=offset,
                    end_char=None,
                )
            )
        offset += len(line)

    sections = [
        ParsedSection(
            heading=section.heading,
            level=section.level,
            start_char=section.start_char,
            end_char=sections[index + 1].start_char if index + 1 < len(sections) else offset,
            metadata=section.metadata,
        )
        for index, section in enumerate(sections)
    ]

    return ParsedDocument(
        title=title,
        document_type=document_type,
        text=text,
        sections=sections,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
