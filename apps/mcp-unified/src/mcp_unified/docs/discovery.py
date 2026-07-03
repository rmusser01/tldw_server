from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from .acquisition.fetcher import URLFetcher
from .acquisition.service import DocsAcquisitionService
from .acquisition.policy import safe_argument_hash
from .acquisition.policy import SourcePolicy
from .models import AccessScope, DiscoverSourceRequest
from .settings import DocsSettings
from .source_utils import redacted_url_for_display, source_defaults_metadata, url_has_query
from .store.sqlite import DocsCatalogStore

CandidateStatus = Literal["accepted", "duplicate", "denied", "skipped", "ingested", "failed"]
_SITEMAP_CONTENT_TYPES = ("application/xml", "text/xml", "application/xhtml+xml")


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
    document_id: int | None = None


@dataclass(frozen=True)
class SitemapParseResult:
    status: str
    reason_code: str
    candidates: list[DiscoveredURLCandidate]
    skipped: int = 0


class DocsSourceDiscoveryService:
    def __init__(
        self,
        *,
        settings: DocsSettings,
        store: DocsCatalogStore,
        resolver: object | None = None,
        transport: object | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.policy = SourcePolicy(
            web_source_profile=settings.web_source_profile,
            preapproved_domains=settings.preapproved_domains,
            allowed_url_prefixes=settings.allowed_url_prefixes,
            denied_domains=settings.denied_domains,
            allow_arbitrary_public_domains=settings.allow_arbitrary_public_domains,
        )
        fetch_settings = replace(
            settings,
            allowed_content_types=_merged_content_types(settings.allowed_content_types, _SITEMAP_CONTENT_TYPES),
        )
        self.fetcher = URLFetcher(settings=fetch_settings, policy=self.policy, resolver=resolver, transport=transport)
        self.acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)

    def discover_source(self, *, scope: AccessScope, request: DiscoverSourceRequest) -> dict[str, Any]:
        if not self.settings.enable_source_discovery:
            return _discovery_response(
                status="denied",
                reason_code="source_discovery_disabled",
                request=request,
                candidates=[],
                warnings=[],
            )
        if not self.settings.enable_web_acquisition:
            return _discovery_response(
                status="denied",
                reason_code="web_acquisition_disabled",
                request=request,
                candidates=[],
                warnings=[],
            )

        validation = _validate_discovery_request(request, self.settings)
        if validation is not None:
            return validation

        fetched = self.fetcher.fetch(request.url)
        if fetched.status != "fetched":
            return _discovery_response(
                status=fetched.status,
                reason_code=fetched.reason,
                request=request,
                candidates=[],
                warnings=[],
                source={
                    "final_url": fetched.final_url,
                    "safe_argument_hash": fetched.safe_argument_hash,
                },
            )

        kind = _resolve_kind(request.kind, request.url, fetched.headers)
        candidates, warnings = self._candidates_for_fetched(kind=kind, request=request, fetched_body=fetched.body)
        candidates, filter_warnings = _filter_candidates(
            candidates,
            seed_url=fetched.canonical_url or request.url,
            policy=self.policy,
            settings=self.settings,
            max_pages=_effective_discovery_page_limit(request, self.settings),
        )
        warnings.extend(filter_warnings)
        source: dict[str, Any] | None = None
        if request.mode == "apply":
            source, candidates, apply_warnings = self._apply_discovery(
                scope=scope,
                kind=kind,
                request=request,
                candidates=candidates,
            )
            warnings.extend(apply_warnings)
        return _discovery_response(
            status="completed",
            reason_code="ok",
            request=request,
            candidates=candidates,
            warnings=warnings,
            source=source,
        )

    def _candidates_for_fetched(
        self,
        *,
        kind: str,
        request: DiscoverSourceRequest,
        fetched_body: bytes,
    ) -> tuple[list[DiscoveredURLCandidate], list[str]]:
        if kind == "sitemap":
            parsed = parse_sitemap_urlset(
                fetched_body,
                max_pages=_effective_discovery_page_limit(request, self.settings),
                parent_url=request.url,
            )
            warnings = [] if parsed.reason_code == "ok" else [parsed.reason_code]
            if parsed.skipped:
                warnings.append("source_discovery_limit_exceeded")
            return parsed.candidates, warnings
        if kind == "page_links":
            parent_display = redacted_url_for_display(request.url)
            candidates = [
                DiscoveredURLCandidate(
                    url=url,
                    display_url=redacted_url_for_display(url),
                    status="accepted",
                    reason_code="ok",
                    source_kind="page_links",
                    parent_url=request.url,
                    parent_display_url=parent_display,
                    safe_argument_hash=safe_argument_hash(url),
                )
                for url in extract_page_links(request.url, fetched_body)
            ]
            return candidates, []
        return [], ["source_discovery_kind_unsupported"]

    def _apply_discovery(
        self,
        *,
        scope: AccessScope,
        kind: str,
        request: DiscoverSourceRequest,
        candidates: list[DiscoveredURLCandidate],
    ) -> tuple[dict[str, Any] | None, list[DiscoveredURLCandidate], list[str]]:
        apply_action = request.apply_action or self.settings.discovery_apply_default
        source: dict[str, Any] | None = None
        warnings: list[str] = []
        if kind == "sitemap" and apply_action in {"register", "register_and_ingest"}:
            source = self._register_sitemap_source(scope=scope, request=request)
            if not self.settings.sitemap_sync_enabled:
                warnings.append("sitemap_sync_disabled")

        if apply_action in {"ingest", "register_and_ingest"}:
            candidates = self._ingest_accepted_candidates(
                scope=scope,
                request=request,
                candidates=candidates,
                source=source,
            )
            if source is not None:
                source = self.store.get_source(scope=scope, source_id=int(source["id"]))

        return _public_source_for_response(source), candidates, warnings

    def _register_sitemap_source(self, *, scope: AccessScope, request: DiscoverSourceRequest) -> dict[str, Any]:
        can_persist_query_uri = self.settings.persist_url_query_strings or not url_has_query(request.url)
        canonical_uri = request.url if can_persist_query_uri else redacted_url_for_display(request.url)
        redacted_source_url = redacted_url_for_display(request.url)
        metadata: dict[str, Any] = source_defaults_metadata(
            keywords=request.keywords,
            collection_names=request.collections,
        )
        metadata.update(
            {
                "discovery_kind": "sitemap",
                "same_origin_only": self.settings.discovery_same_origin_only,
            }
        )
        source_id = self.store.upsert_source(
            scope=scope,
            source_type="url_sitemap",
            canonical_uri=canonical_uri,
            display_name=request.title or redacted_source_url,
            source_path=None,
            source_url=request.url if can_persist_query_uri else None,
            redacted_source_url=redacted_source_url,
            policy_profile=self.settings.web_source_profile,
            sync_enabled=self.settings.sitemap_sync_enabled,
            metadata=metadata,
        )
        source = self.store.get_source(scope=scope, source_id=source_id)
        if source is None:
            raise RuntimeError("Registered sitemap source could not be loaded")
        return source

    def _ingest_accepted_candidates(
        self,
        *,
        scope: AccessScope,
        request: DiscoverSourceRequest,
        candidates: list[DiscoveredURLCandidate],
        source: dict[str, Any] | None,
    ) -> list[DiscoveredURLCandidate]:
        ingested: list[DiscoveredURLCandidate] = []
        for candidate in candidates:
            if candidate.status != "accepted":
                ingested.append(candidate)
                continue
            result = self.acquisition.ingest_url(
                scope=scope,
                url=candidate.url,
                keywords=request.keywords,
                collection_names=request.collections,
            )
            document = result.get("document")
            document_id = int(document["id"]) if isinstance(document, Mapping) and document.get("id") else None
            if result.get("reason_code") == "ok" and document_id is not None:
                if source is not None:
                    stored_document = self.store.get_document(scope, document_id, mode="metadata")
                    self.store.link_source_document(
                        scope=scope,
                        source_id=int(source["id"]),
                        document_id=document_id,
                        source_item_uri=candidate.url,
                        status="active",
                        last_hash=str(stored_document.get("content_hash") or ""),
                        metadata={"importer": "url_sitemap"},
                    )
                ingested.append(
                    _replace_candidate(
                        candidate,
                        candidate.url,
                        candidate.display_url,
                        "ingested",
                        "ok",
                        document_id=document_id,
                    )
                )
                continue
            ingested.append(
                _replace_candidate(
                    candidate,
                    candidate.url,
                    candidate.display_url,
                    "failed",
                    str(result.get("reason_code") or "ingest_failed"),
                )
            )
        return ingested


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


def public_candidate(candidate: DiscoveredURLCandidate) -> dict[str, Any]:
    public = {
        "url": candidate.display_url,
        "status": candidate.status,
        "reason_code": candidate.reason_code,
        "source_kind": candidate.source_kind,
        "parent_url": candidate.parent_display_url,
        "safe_argument_hash": candidate.safe_argument_hash,
    }
    if candidate.document_id is not None:
        public["document_id"] = candidate.document_id
    return public


def _validate_discovery_request(
    request: DiscoverSourceRequest,
    settings: DocsSettings,
) -> dict[str, Any] | None:
    if request.kind not in {"auto", "sitemap", "page_links"}:
        return _discovery_response(
            status="denied",
            reason_code="source_discovery_kind_unsupported",
            request=request,
            candidates=[],
            warnings=[],
        )
    if request.mode not in {"dry_run", "apply"}:
        return _discovery_response(
            status="denied",
            reason_code="source_discovery_request_invalid",
            request=request,
            candidates=[],
            warnings=["invalid_mode"],
        )
    if request.apply_action is not None and request.apply_action not in {"register", "ingest", "register_and_ingest"}:
        return _discovery_response(
            status="denied",
            reason_code="source_discovery_request_invalid",
            request=request,
            candidates=[],
            warnings=["invalid_apply_action"],
        )
    if request.max_pages is not None and request.max_pages < 1:
        return _discovery_response(
            status="denied",
            reason_code="source_discovery_limit_exceeded",
            request=request,
            candidates=[],
            warnings=["max_pages must be positive"],
        )
    if request.max_depth is not None and request.max_depth != settings.max_discovery_depth:
        return _discovery_response(
            status="denied",
            reason_code="source_discovery_request_invalid",
            request=request,
            candidates=[],
            warnings=["max_depth_unsupported"],
        )
    return None


def _filter_candidates(
    candidates: list[DiscoveredURLCandidate],
    *,
    seed_url: str,
    policy: SourcePolicy,
    settings: DocsSettings,
    max_pages: int,
) -> tuple[list[DiscoveredURLCandidate], list[str]]:
    seed_origin = _origin(seed_url)
    seen: set[str] = set()
    accepted = 0
    filtered: list[DiscoveredURLCandidate] = []
    warnings: list[str] = []
    for candidate in candidates:
        candidate_url = _strip_fragment(candidate.url)
        display_url = redacted_url_for_display(candidate_url)
        if candidate_url in seen:
            filtered.append(_replace_candidate(candidate, candidate_url, display_url, "duplicate", "candidate_duplicate"))
            continue
        seen.add(candidate_url)

        if url_has_query(candidate_url) and not settings.persist_url_query_strings:
            filtered.append(
                _replace_candidate(candidate, candidate_url, display_url, "skipped", "candidate_query_not_persisted")
            )
            continue
        if settings.discovery_same_origin_only and _origin(candidate_url) != seed_origin:
            filtered.append(_replace_candidate(candidate, candidate_url, display_url, "denied", "candidate_out_of_scope"))
            continue
        decision = policy.evaluate(candidate_url)
        if decision.status != "allowed":
            filtered.append(_replace_candidate(candidate, candidate_url, display_url, "denied", "candidate_out_of_scope"))
            continue
        if accepted >= max_pages:
            warnings.append("source_discovery_limit_exceeded")
            filtered.append(
                _replace_candidate(candidate, candidate_url, display_url, "skipped", "source_discovery_limit_exceeded")
            )
            continue
        accepted += 1
        filtered.append(_replace_candidate(candidate, candidate_url, display_url, "accepted", "ok"))
    return filtered, warnings


def _discovery_response(
    *,
    status: str,
    reason_code: str,
    request: DiscoverSourceRequest,
    candidates: list[DiscoveredURLCandidate],
    warnings: list[str],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason_code,
        "kind": request.kind,
        "mode": request.mode,
        "source": source,
        "counts": _discovery_counts(candidates),
        "candidates": [public_candidate(candidate) for candidate in candidates],
        "warnings": warnings,
    }


def _discovery_counts(candidates: list[DiscoveredURLCandidate]) -> dict[str, int]:
    counts = _zero_discovery_counts()
    for candidate in candidates:
        if candidate.status == "duplicate":
            counts["duplicates"] += 1
        elif candidate.status in counts:
            counts[candidate.status] += 1
    return counts


def _zero_discovery_counts() -> dict[str, int]:
    return {
        "accepted": 0,
        "duplicates": 0,
        "denied": 0,
        "skipped": 0,
        "ingested": 0,
        "failed": 0,
    }


def _effective_discovery_page_limit(request: DiscoverSourceRequest, settings: DocsSettings) -> int:
    if request.max_pages is None:
        return settings.max_discovery_pages
    return min(request.max_pages, settings.max_discovery_pages)


def _resolve_kind(requested_kind: str, url: str, headers: Mapping[str, str]) -> str:
    if requested_kind != "auto":
        return requested_kind
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in {"application/xml", "text/xml"} or urlsplit(url).path.endswith(".xml"):
        return "sitemap"
    return "page_links"


def _replace_candidate(
    candidate: DiscoveredURLCandidate,
    url: str,
    display_url: str,
    status: CandidateStatus,
    reason_code: str,
    *,
    document_id: int | None = None,
) -> DiscoveredURLCandidate:
    return DiscoveredURLCandidate(
        url=url,
        display_url=display_url,
        status=status,
        reason_code=reason_code,
        source_kind=candidate.source_kind,
        parent_url=candidate.parent_url,
        parent_display_url=candidate.parent_display_url,
        safe_argument_hash=safe_argument_hash(url),
        document_id=document_id,
    )


def _public_source_for_response(source: dict[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    public = dict(source)
    if public.get("source_type") in {"url_page", "url_sitemap"}:
        redacted_uri = public.get("redacted_source_url")
        if not isinstance(redacted_uri, str) or not redacted_uri.strip():
            redacted_uri = redacted_url_for_display(str(public.get("canonical_uri") or public.get("source_url") or ""))
        public.pop("source_url", None)
        public["canonical_uri"] = redacted_uri
        public["display_uri"] = redacted_uri
        public["redacted_source_url"] = redacted_uri
    return public


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port


def _merged_content_types(existing: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for value in (*existing, *additions):
        if value not in merged:
            merged.append(value)
    return tuple(merged)


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
