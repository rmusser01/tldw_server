from __future__ import annotations

import importlib
from urllib.parse import urlsplit

from ..importers.base import ParsedDocument
from ..importers.html import parse_html_document

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
_TEXT_CONTENT_TYPES = {"text/plain", "text/markdown"}


def available_extractors() -> list[str]:
    names: list[str] = []
    if _can_import("trafilatura"):
        names.append("trafilatura")
    if _can_import("bs4"):
        names.append("beautifulsoup")
    names.extend(["static_html", "text"])
    return names


def extract_fetched_document(*, url: str, content_type: str, body: bytes) -> ParsedDocument:
    text = _decode_body(body, content_type)
    media_type = _media_type(content_type)
    if media_type not in _HTML_CONTENT_TYPES:
        return ParsedDocument(
            title=_title_from_url(url),
            document_type="text",
            text=text.strip(),
            sections=[],
            canonical_uri=url,
            source_url=url,
            extraction_method="text",
        )

    static_html = _parse_static_html(url=url, text=text)
    trafilatura_text = _extract_with_trafilatura(text)
    if trafilatura_text:
        return ParsedDocument(
            title=_metadata_title(static_html, url),
            document_type="html",
            text=trafilatura_text,
            sections=static_html.sections,
            canonical_uri=url,
            source_url=url,
            extraction_method="trafilatura",
        )

    soup_text = _extract_with_beautifulsoup(text)
    if soup_text:
        return ParsedDocument(
            title=_metadata_title(static_html, url),
            document_type="html",
            text=soup_text,
            sections=static_html.sections,
            canonical_uri=url,
            source_url=url,
            extraction_method="beautifulsoup",
        )

    return static_html


def _parse_static_html(*, url: str, text: str) -> ParsedDocument:
    return parse_html_document(
        text=text,
        title_hint=_title_from_url(url),
        canonical_uri=url,
        source_url=url,
        extraction_method="static_html",
        warnings=_fallback_warnings(),
    )


def _metadata_title(static_html: ParsedDocument, url: str) -> str:
    return static_html.title or _title_from_url(url)


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _fallback_warnings() -> tuple[str, ...]:
    if not _can_import("trafilatura") and not _can_import("bs4"):
        return ("rich_extractors_unavailable",)
    return ("rich_extractors_fell_back",)


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _decode_body(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";")[1:]:
        key, _separator, value = part.strip().partition("=")
        if key.strip().lower() == "charset" and value:
            charset = value.strip().strip("\"'")
            break
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _title_from_url(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    title = path.rsplit("/", 1)[-1]
    return title or parsed.hostname or "Untitled"


def _extract_with_trafilatura(text: str) -> str | None:
    try:
        trafilatura = importlib.import_module("trafilatura")
    except ImportError:
        return None
    extracted = trafilatura.extract(text, include_comments=False, include_tables=True)
    return extracted.strip() if isinstance(extracted, str) and extracted.strip() else None


def _extract_with_beautifulsoup(text: str) -> str | None:
    try:
        bs4 = importlib.import_module("bs4")
    except ImportError:
        return None
    soup = bs4.BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    extracted = soup.get_text("\n", strip=True)
    return extracted.strip() if extracted.strip() else None


__all__ = ["available_extractors", "extract_fetched_document"]
