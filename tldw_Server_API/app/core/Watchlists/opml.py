"""
OPML parser/writer for Watchlists sources.

Supports basic OPML outlines where RSS feeds are represented as <outline> with
attributes xmlUrl (feed URL), optional htmlUrl, and title/text for the name.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from defusedxml import ElementTree as ET

@dataclass
class OPMLSource:
    url: str
    name: str | None = None
    html_url: str | None = None


def _gather_outlines(elem: Any, out: list[OPMLSource]) -> None:
    for child in elem.findall("outline"):
        xml_url = child.attrib.get("xmlUrl") or child.attrib.get("xmlurl")
        title = child.attrib.get("title") or child.attrib.get("text")
        html_url = child.attrib.get("htmlUrl") or child.attrib.get("htmlurl")
        if xml_url:
            out.append(OPMLSource(url=xml_url.strip(), name=(title or None), html_url=(html_url or None)))
        # Recurse into nested outlines
        _gather_outlines(child, out)


def parse_opml(opml_bytes: bytes) -> list[OPMLSource]:
    """Parse OPML content and return a flat list of OPMLSource entries."""
    sources: list[OPMLSource] = []
    try:
        root = ET.fromstring(opml_bytes)
    except Exception:
        return sources
    # Standard path: opml -> body
    body = root.find("body") if root is not None else None
    if body is None:
        return sources
    _gather_outlines(body, sources)
    # Deduplicate by URL preserving order
    seen: set[str] = set()
    uniq: list[OPMLSource] = []
    for s in sources:
        if not s.url or s.url in seen:
            continue
        seen.add(s.url)
        uniq.append(s)
    return uniq


def generate_opml(sources: Iterable[dict[str, Any]]) -> str:
    """Generate a minimal OPML string from iterable of {'name','url','html_url'} dicts."""
    def _escape(text: str | None) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&apos;")
        )

    items: list[str] = []
    for s in sources:
        url = str(s.get("url") or "").strip()
        if not url:
            continue
        name = str(s.get("name") or "").strip() or url
        html_url = str(s.get("html_url") or "").strip()
        attrs = [f'text="{_escape(name)}"', f'title="{_escape(name)}"', f'xmlUrl="{_escape(url)}"']
        if html_url:
            attrs.append(f'htmlUrl="{_escape(html_url)}"')
        items.append(f"    <outline {' '.join(attrs)} />")
    body = "\n".join(["  <body>"] + items + ["  </body>"])
    head = "  <head>\n    <title>Watchlists Export</title>\n  </head>"
    return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<opml version=\"2.0\">\n{head}\n{body}\n</opml>\n"
