from __future__ import annotations

import importlib
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from .acquisition.policy import safe_argument_hash
from .source_utils import redacted_url_for_display

CandidateStatus = Literal["accepted", "duplicate", "denied", "skipped", "ingested", "failed"]


@dataclass(frozen=True)
class DiscoveredURLCandidate:
    url: str
    display_url: str
    status: CandidateStatus
    reason_code: str
    source_kind: str
    parent_url: str
    parent_display_url: str
    safe_argument_hash: str


@dataclass(frozen=True)
class SitemapParseResult:
    status: str
    reason_code: str
    candidates: list[DiscoveredURLCandidate]
    skipped: int = 0


class _LinkHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {name.lower(): value for name, value in attrs if name}
        rel = attr_map.get("rel") or ""
        if "nofollow" in {item.strip().lower() for item in rel.split()}:
            return
        href = attr_map.get("href")
        if not href:
            return
        resolved = _supported_joined_url(self.base_url, href)
        if resolved:
            self.links.append(resolved)


def parse_sitemap_urlset(body: bytes, *, max_pages: int, parent_url: str = "") -> SitemapParseResult:
    upper = body[:4096].upper()
    if b"<!DOCTYPE" in upper:
        return SitemapParseResult(status="denied", reason_code="sitemap_xml_forbidden_doctype", candidates=[])
    if b"<!ENTITY" in upper:
        return SitemapParseResult(status="denied", reason_code="sitemap_xml_forbidden_entity", candidates=[])
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return SitemapParseResult(status="failed", reason_code="sitemap_parse_failed", candidates=[])

    root_name = _local_name(root.tag)
    if root_name == "sitemapindex":
        return SitemapParseResult(status="denied", reason_code="sitemap_index_unsupported", candidates=[])
    if root_name != "urlset":
        return SitemapParseResult(status="failed", reason_code="sitemap_parse_failed", candidates=[])

    locs = [
        (loc.text or "").strip()
        for url_node in root.iter()
        if _local_name(url_node.tag) == "url"
        for loc in list(url_node)
        if _local_name(loc.tag) == "loc" and (loc.text or "").strip()
    ]
    selected = locs[: max(0, max_pages)]
    parent_display = redacted_url_for_display(parent_url) if parent_url else ""
    candidates = [
        DiscoveredURLCandidate(
            url=url,
            display_url=redacted_url_for_display(url),
            status="accepted",
            reason_code="ok",
            source_kind="sitemap",
            parent_url=parent_url,
            parent_display_url=parent_display,
            safe_argument_hash=safe_argument_hash(url),
        )
        for url in selected
    ]
    return SitemapParseResult(
        status="completed",
        reason_code="ok",
        candidates=candidates,
        skipped=max(0, len(locs) - len(selected)),
    )


def extract_page_links(base_url: str, body: bytes) -> list[str]:
    text = body.decode("utf-8", errors="replace")
    links = _extract_links_with_beautifulsoup(base_url, text)
    if links is None:
        parser = _LinkHTMLParser(base_url)
        parser.feed(text)
        links = parser.links
    return _unique_links(links)


def public_candidate(candidate: DiscoveredURLCandidate) -> dict[str, str]:
    return {
        "url": candidate.display_url,
        "status": candidate.status,
        "reason_code": candidate.reason_code,
        "source_kind": candidate.source_kind,
        "parent_url": candidate.parent_display_url,
        "safe_argument_hash": candidate.safe_argument_hash,
    }


def _extract_links_with_beautifulsoup(base_url: str, text: str) -> list[str] | None:
    try:
        bs4 = importlib.import_module("bs4")
    except ImportError:
        return None

    soup = bs4.BeautifulSoup(text, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a"):
        rel = anchor.get("rel") or ()
        rel_values = rel.split() if isinstance(rel, str) else tuple(str(item) for item in rel)
        if "nofollow" in {item.strip().lower() for item in rel_values}:
            continue
        href = anchor.get("href")
        if not href:
            continue
        resolved = _supported_joined_url(base_url, str(href))
        if resolved:
            links.append(resolved)
    return links


def _unique_links(links: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for link in links:
        stripped = _strip_fragment(link)
        if stripped in seen:
            continue
        seen.add(stripped)
        unique.append(stripped)
    return unique


def _supported_joined_url(base_url: str, href: str) -> str | None:
    joined = urljoin(base_url, href.strip())
    parts = urlsplit(joined)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return _strip_fragment(joined)


def _strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
