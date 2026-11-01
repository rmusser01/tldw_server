"""Bounded discovery from the already-acquired document, never a second fetch."""

from __future__ import annotations

from urllib.parse import parse_qsl, urljoin, urlparse, urlsplit, uses_relative

from bs4 import BeautifulSoup
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from .. import url_utils
from ..contracts.execution import DiscoveredLink, DiscoveryResult
from ..runtime.controls import ParentControls, ParentStop
from .models import CrawlHooks, CrawlPlan


def url_bytes(value: str, plan: CrawlPlan) -> int:
    """Check codepoint length before bounded UTF-8 allocation, then exact bytes."""
    if len(value) > plan.limits.max_url_bytes:
        raise ParentStop("budget_exhausted")
    size = len(value.encode("utf-8"))
    if size > plan.limits.max_url_bytes:
        raise ParentStop("budget_exhausted")
    return size


def charge(controls: ParentControls, counter: str, amount: int = 1) -> None:
    """Charge admitted cumulative work, without a pre-start refund."""
    if not amount:
        controls.check()
        return
    reservation = controls.reserve(counter, amount)
    try:
        reservation.commit()
    finally:
        reservation.close()


def resolve_reference(base: str, reference: str, plan: CrawlPlan) -> str:
    """Pre-admit the derived shape before urllib allocates the resolved URL.

    Splitting bounded inputs only creates bounded components. Account the joined
    path segments without building the amplified path; urllib remains the actual
    resolver, including its scheme, query and dot-segment semantics.
    """
    url_bytes(base, plan)
    url_bytes(reference, plan)
    source, target = urlsplit(base), urlsplit(reference)
    scheme = target.scheme or source.scheme
    if (target.scheme and target.scheme != source.scheme) or scheme not in uses_relative:
        return reference
    host = target.netloc or source.netloc
    query = target.query
    if target.netloc:
        segments = [target.path]
    elif not target.path:
        segments = [source.path]
        query = target.query or source.query
    else:
        if target.path.startswith("/"):
            parts = target.path.split("/")
        else:
            parts = source.path.split("/")[:-1] + target.path.split("/")
            parts[1:-1] = [part for part in parts[1:-1] if part]
        segments = []
        for part in parts:
            if part == "..":
                if segments:
                    segments.pop()
            elif part != ".":
                segments.append(part)
        if parts[-1] in {".", ".."}:
            segments.append("")
        if not segments or segments == [""]:
            segments = ["/"]
    path_size = sum(url_bytes(part, plan) for part in segments) + max(0, len(segments) - 1)
    size = len(scheme) + 1 + (2 + url_bytes(host, plan) if host else 0) + path_size
    size += (1 + url_bytes(query, plan) if query else 0) + (
        1 + url_bytes(target.fragment, plan) if target.fragment else 0
    )
    if size > plan.limits.max_url_bytes:
        raise ParentStop("budget_exhausted")
    resolved = urljoin(base, reference)
    url_bytes(resolved, plan)
    return resolved


def normalize_reference(value: str, base: str, plan: CrawlPlan) -> str:
    """Admit the canonical representation before percent encoding or URL assembly.

    Reuse the established parser/path/tracking rules, but measure quote_plus's
    exact ASCII expansion without creating the query. Raw/resolved strings and
    parsed components are already bounded; Unicode host lowercasing is sized too.
    """
    url_bytes(base, plan)
    url_bytes(value, plan)
    if not value:
        return ""
    resolved = resolve_reference(base, value, plan)
    parsed = urlparse(resolved)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return resolved
    host = parsed.netloc
    port = ":80" if scheme == "http" else ":443"
    if host.endswith(port):
        host = host[: -len(port)]
    host_size = sum(len(character.lower().encode("utf-8")) for character in host)
    path = url_utils._normalize_path(parsed.path)
    items = url_utils._strip_tracking_params(parse_qsl(parsed.query, keep_blank_values=True))

    def quoted_size(text: str) -> int:
        return sum(
            1
            if character.isascii() and (character.isalnum() or character in "-._~ ")
            else 3 * len(character.encode("utf-8"))
            for character in text
        )

    query_size = sum(quoted_size(key) + 1 + quoted_size(item) for key, item in items) + max(0, len(items) - 1)
    size = len(scheme) + 3 + host_size + url_bytes(path, plan) + (1 + query_size if items else 0)
    if size > plan.limits.max_url_bytes:
        raise ParentStop("budget_exhausted")
    query = url_utils.urlencode(items, doseq=True)
    normalized = url_utils.urlunparse((scheme, host.lower(), path, "", query, ""))
    return value if normalized == value else normalized


def discover(
    document: str,
    *,
    request_url: str,
    final_url: str,
    plan: CrawlPlan,
    controls: ParentControls,
    hooks: CrawlHooks | None = None,
) -> DiscoveryResult:
    """Inspect each encounter before filtering or growing the retained collection."""
    controls.check()
    charge(controls, "discovered_url_bytes", url_bytes(request_url, plan) + url_bytes(final_url, plan))
    sitemap = plan.profile.endswith("sitemap")
    legacy_sitemap = plan.profile == "legacy_sitemap"
    if legacy_sitemap:
        try:
            root = ElementTree.fromstring(document)
        except (ElementTree.ParseError, DefusedXmlException):
            return DiscoveryResult(request_url, final_url, ())
        references = (node.text or "" for node in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"))
        base = request_url
    else:
        soup = BeautifulSoup(document, "xml" if sitemap else "html.parser")
        base = plan.request.base_url if plan.profile == "legacy_url_level" else request_url
        if not sitemap and (element := soup.find("base", href=True)) is not None:
            reference = str(element["href"])
            url_bytes(reference, plan)
            if plan.profile == "legacy_recursive":
                base = resolve_reference(request_url, reference, plan)
                url_bytes(base, plan)
        references = (
            (node.get_text().strip() for node in soup.descendants if getattr(node, "name", None) == "loc")
            if sitemap
            else (
                str(node["href"])
                for node in soup.descendants
                if getattr(node, "name", None) == "a" and node.has_attr("href")
            )
        )
    links = []
    for reference in references:
        controls.check()
        charge(controls, "discovered_candidates")
        url_bytes(reference, plan)
        try:
            if sitemap:
                target = reference if legacy_sitemap else reference.strip()
            else:
                target = resolve_reference(base, reference, plan)
                url_bytes(target, plan)
                if plan.profile.startswith("enhanced_"):
                    target = normalize_reference(target, request_url, plan)
            url_bytes(target, plan)
            parsed = urlsplit(target)
            if not parsed.hostname or parsed.scheme not in {"http", "https"}:
                continue
        except (ValueError, UnicodeError):
            continue
        charge(controls, "discovered_url_bytes", url_bytes(target, plan) + url_bytes(request_url, plan))
        if sitemap and hooks and hooks.url_filter and not hooks.url_filter(target):
            continue
        if not sitemap and plan.profile.startswith("enhanced_") and (not reference or reference.startswith("#")):
            continue
        links.append(DiscoveredLink(target, request_url))
        maximum = plan.options.get("max_urls")
        if sitemap and maximum and len(links) >= maximum:
            break
    return DiscoveryResult(request_url, final_url, tuple(links))
