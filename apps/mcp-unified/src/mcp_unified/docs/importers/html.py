from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from .base import ParsedDocument, ParsedSection, file_uri


class StaticHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._active_heading: int | None = None
        self._heading_text: list[str] = []
        self.parts: list[str] = []
        self.sections: list[ParsedSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if normalized in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._active_heading = int(normalized[1])
            self._heading_text = []

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._active_heading is not None and normalized == f"h{self._active_heading}":
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                self.sections.append(
                    ParsedSection(
                        heading=heading,
                        level=self._active_heading,
                        start_char=None,
                        end_char=None,
                    )
                )
                self.parts.append(f"\n{heading}\n")
            self._active_heading = None
            self._heading_text = []
        if normalized in {"p", "li", "section", "article", "br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._active_heading is not None:
            self._heading_text.append(data)
            return
        clean = " ".join(data.split())
        if clean:
            self.parts.append(clean)
            self.parts.append(" ")


def parse_html_document(
    *,
    text: str,
    title_hint: str,
    canonical_uri: str,
    source_path: str | None = None,
    source_url: str | None = None,
    extraction_method: str = "static_html",
    warnings: tuple[str, ...] = (),
) -> ParsedDocument:
    parser = StaticHTMLTextParser()
    parser.feed(text)
    body = "\n".join(part.strip() for part in parser.parts if part.strip())
    title = parser.sections[0].heading if parser.sections else title_hint
    return ParsedDocument(
        title=title,
        document_type="html",
        text=body,
        sections=parser.sections,
        canonical_uri=canonical_uri,
        source_path=source_path,
        source_url=source_url,
        extraction_method=extraction_method,
        warnings=warnings,
    )


def parse_html(path: Path, text: str) -> ParsedDocument:
    return parse_html_document(
        text=text,
        title_hint=path.stem,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
