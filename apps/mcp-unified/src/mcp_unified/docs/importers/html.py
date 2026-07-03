from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

from .base import ParsedDocument, ParsedSection, file_uri


class StaticHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._pre_depth = 0
        self._pre_text: list[str] = []
        self.pre_blocks: list[str] = []
        self._active_heading: int | None = None
        self._heading_text: list[str] = []
        self.parts: list[str] = []
        self.sections: list[ParsedSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if normalized == "pre":
            if self._pre_depth == 0:
                self._pre_text = []
            self._pre_depth += 1
            return
        if normalized == "br":
            if self._active_heading is not None:
                self._heading_text.append(" ")
            else:
                self.parts.append("\n")
            return
        if normalized in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            if self._active_heading is not None:
                self._flush_heading()
            self._active_heading = int(normalized[1])
            self._heading_text = []

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if normalized == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if self._pre_depth == 0:
                self._flush_pre_block()
            return
        if self._active_heading is not None and normalized in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush_heading()
            return
        if normalized in {"p", "li", "div", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._pre_depth:
            self._pre_text.append(data)
            return
        if self._active_heading is not None:
            self._heading_text.append(data)
            return
        leading_space = data[:1].isspace()
        trailing_space = data[-1:].isspace()
        clean = " ".join(data.split())
        if clean:
            self._append_text(clean, leading_space=leading_space)
            if trailing_space:
                self.parts.append(" ")

    def close(self) -> None:
        super().close()
        if self._active_heading is not None:
            self._flush_heading()
        if self._pre_depth:
            self._pre_depth = 0
            self._flush_pre_block()

    def _append_text(self, text: str, *, leading_space: bool) -> None:
        if leading_space and self.parts and not self.parts[-1].endswith((" ", "\n")) and text[:1] not in ".,;:!?)]}":
            self.parts.append(" ")
        self.parts.append(text)

    def _flush_heading(self) -> None:
        if self._active_heading is None:
            return
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
            self.parts.append("\n")
            self.parts.append(heading)
            self.parts.append("\n")
        self._active_heading = None
        self._heading_text = []

    def _flush_pre_block(self) -> None:
        block = "".join(self._pre_text).strip("\n")
        if block:
            token = f"@@TLDW_DOCS_PRE_{len(self.pre_blocks)}@@"
            self.pre_blocks.append(block)
            self.parts.append("\n")
            self.parts.append(token)
            self.parts.append("\n")
        self._pre_text = []


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
    parser.close()
    body = _normalize_html_text("".join(parser.parts), parser.pre_blocks)
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


def _normalize_html_text(text: str, pre_blocks: list[str] | None = None) -> str:
    collapsed_spaces = re.sub(r"[ \t\f\v]+", " ", text)
    collapsed_edges = re.sub(r" *\n *", "\n", collapsed_spaces)
    normalized = re.sub(r"\n{3,}", "\n\n", collapsed_edges).strip()
    for index, block in enumerate(pre_blocks or []):
        normalized = normalized.replace(f"@@TLDW_DOCS_PRE_{index}@@", f"\n{block}\n")
    return normalized.strip()
