from __future__ import annotations

from pathlib import Path
import re

from .base import ParsedDocument, ParsedSection, file_uri

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown(path: Path, text: str, document_type: str) -> ParsedDocument:
    sections: list[ParsedSection] = []
    title = path.stem
    offset = 0
    for line in text.splitlines(keepends=True):
        match = HEADING_RE.match(line.strip())
        if match:
            heading = match.group(2).strip()
            if title == path.stem:
                title = heading
            sections.append(
                ParsedSection(
                    heading=heading,
                    level=len(match.group(1)),
                    start_char=offset,
                    end_char=None,
                )
            )
        offset += len(line)

    return ParsedDocument(
        title=title,
        document_type=document_type,
        text=text,
        sections=sections,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
