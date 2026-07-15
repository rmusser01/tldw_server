"""Shadow-only bioRxiv and medRxiv discovery route family."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any

from .contracts import (
    MAX_PAGINATION_CURSOR,
    AccessRoute,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    CredentialRequirement,
    CredentialStatus,
    DiscoveryOutcomeIdentity,
    ExactOrigin,
    ExactQueryValuePolicy,
    ExecutionMode,
    LiteralTermsQueryValuePolicy,
    OperationKind,
    PathSlot,
    PathSlotKind,
    PathTemplate,
    PlannedDispatchGroup,
    PredicateOperator,
    QueryMode,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SourceConstraint,
    SourceDefinition,
    SourcePredicate,
    SourceRouteReference,
)
from .executor import (
    BoundDispatch,
    DiscoveryAdapter,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
)
from .gateway_adapters import (
    MonotonicClock,
    _base_record,
    _checked_response,
    _optional_text,
    _ParseDeadlineExceeded,
    _ParseGuard,
    _ParseLimitExceeded,
    _ParsingProfile,
    _PayloadInvalid,
    _raise_adapter_error,
    _require_dict,
    _require_list,
    _required_text,
    _strict_json,
)
from .identity import build_fingerprint
from .registry import DiscoveryRegistry, foundation_readiness, foundation_registry

SHADOW_CATALOG_VERSION = "research-discovery-v2-biorxiv-medrxiv-shadow"
SHADOW_REGISTRY_VERSION = "research-discovery-v2-biorxiv-medrxiv-shadow-2026-07-15"
SHADOW_READINESS_VERSION = "research-discovery-readiness-v2-biorxiv-medrxiv-shadow"
ROUTE_POLICY_VERSION = "research-discovery-route-policy-v2-biorxiv-medrxiv"
EUROPE_PMC_ADAPTER_ID = "europe_pmc_preprint_v2"
EUROPE_PMC_ADAPTER_VERSION = "europe-pmc-preprint-v2"
DETAILS_ADAPTER_ID = "biorxiv_details_v2"
DETAILS_ADAPTER_VERSION = "biorxiv-details-v2"
DETAILS_DISABLED_REASON = "details_adapter_fixture_pending"

_EUROPE_PMC_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=120,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_FAMILY_PARSING_PROFILES = MappingProxyType({(EUROPE_PMC_ADAPTER_ID, EUROPE_PMC_ADAPTER_VERSION): _EUROPE_PMC_PROFILE})
_PPR_ID_RE = re.compile(r"PPR[1-9][0-9]*\Z", re.ASCII)
_DOI_RE = re.compile(
    r"10\.[0-9]{4,9}/[A-Za-z0-9][-A-Za-z0-9._~!$&'()*+,;=:@]*\Z",
    re.ASCII,
)
_YEAR_RE = re.compile(r"[0-9]{4}\Z", re.ASCII)
_MISSING = object()
_MAX_TITLE_CHARS = 4_096
_MAX_ABSTRACT_CHARS = 65_536
_MAX_AUTHORS = 1_024
_MAX_AUTHOR_CHARS = 512
_MAX_IDENTIFIER_CHARS = 128
_HTML_BLOCK_TAGS = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_HTML_IGNORED_TAGS = frozenset({"script", "style"})

_GENERAL_ROUTE_IDS = (
    "biorxiv_europe_pmc_search_aggregator",
    "medrxiv_europe_pmc_search_aggregator",
)
_DETAILS_ROUTE_IDS = (
    "biorxiv_details_lookup_direct",
    "medrxiv_details_lookup_direct",
    "biorxiv_details_interval_direct",
    "medrxiv_details_interval_direct",
)


def biorxiv_medrxiv_shadow_registry() -> DiscoveryRegistry:
    """Return the foundation plus the isolated bioRxiv/medRxiv family."""
    foundation = foundation_registry()
    sources = tuple(replace(source, catalog_version=SHADOW_CATALOG_VERSION) for source in foundation.sources) + tuple(
        _family_source(source_id) for source_id in ("biorxiv", "medrxiv")
    )
    return DiscoveryRegistry(
        catalog_version=SHADOW_CATALOG_VERSION,
        registry_version=SHADOW_REGISTRY_VERSION,
        sources=sources,
        routes=foundation.routes + _family_routes(),
        backends=foundation.backends
        + (
            BackendDefinition("europe_pmc_rest_api", "Europe PMC REST API"),
            BackendDefinition("biorxiv_details_api", "bioRxiv/medRxiv Details API"),
        ),
    )


def biorxiv_medrxiv_shadow_readiness(execution_mode: ExecutionMode) -> ReadinessOverlay:
    """Return explicit shadow readiness without enabling details execution."""
    foundation = foundation_readiness(execution_mode)
    ready_reason = f"{execution_mode.value}_ready"
    family_entries = tuple(
        RouteReadiness(
            route_id=route_id,
            state=ReadinessState.READY,
            credential_status=CredentialStatus.NOT_REQUIRED,
            reason=ready_reason,
        )
        for route_id in _GENERAL_ROUTE_IDS
    ) + tuple(
        RouteReadiness(
            route_id=route_id,
            state=ReadinessState.DISABLED,
            credential_status=CredentialStatus.NOT_REQUIRED,
            reason=DETAILS_DISABLED_REASON,
        )
        for route_id in _DETAILS_ROUTE_IDS
    )
    return ReadinessOverlay(
        overlay_version=SHADOW_READINESS_VERSION,
        execution_mode=execution_mode,
        routes=foundation.routes + family_entries,
    )


def biorxiv_medrxiv_gateway_adapters(
    *,
    monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]:
    """Return only the fixture-ready adapters owned by this route family."""

    async def europe_pmc_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_europe_pmc_adapter(group, dispatch, monotonic_clock)

    return _compose_adapter_maps({EUROPE_PMC_ADAPTER_ID: europe_pmc_adapter})


def _compose_adapter_maps(*adapter_maps: Mapping[str, DiscoveryAdapter]) -> Mapping[str, DiscoveryAdapter]:
    """Compose a small reviewed adapter set and reject duplicate identities."""
    composed: dict[str, DiscoveryAdapter] = {}
    for adapter_map in adapter_maps:
        for adapter_id, adapter in adapter_map.items():
            if adapter_id in composed:
                raise ValueError(f"duplicate_adapter_id:{adapter_id}")
            composed[adapter_id] = adapter
    return MappingProxyType(composed)


class _PlainTextParser(HTMLParser):
    """Convert a bounded HTML fragment to inert text without resolving links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in _HTML_IGNORED_TAGS:
            self.ignored_depth += 1
        elif not self.ignored_depth and normalized in _HTML_BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in _HTML_IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1
        elif not self.ignored_depth and normalized in _HTML_BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _plain_text(value: str, *, max_chars: int, required: bool) -> str | None:
    if type(value) is not str or len(value) > max_chars:
        raise _ParseLimitExceeded
    if any((ord(character) < 32 and character not in "\t\n\r") or ord(character) == 127 for character in value):
        raise _PayloadInvalid
    try:
        parser = _PlainTextParser()
        parser.feed(value)
        parser.close()
    except Exception as error:
        raise _PayloadInvalid from error
    normalized = " ".join("".join(parser.parts).split())
    if len(normalized) > max_chars:
        raise _ParseLimitExceeded
    if not normalized:
        if required:
            raise _PayloadInvalid
        return None
    return normalized


def _trusted_europe_pmc_inputs(
    group: object,
) -> tuple[PlannedDispatchGroup, _ParsingProfile, int, int]:
    """Validate the exact single-search adapter contract before dispatch."""
    if type(group) is not PlannedDispatchGroup:
        raise DiscoveryAdapterError("provider_payload_invalid")
    profile = _FAMILY_PARSING_PROFILES.get((group.adapter_id, group.adapter_version))
    if (
        group.adapter_id != EUROPE_PMC_ADAPTER_ID
        or group.adapter_version != EUROPE_PMC_ADAPTER_VERSION
        or profile is None
        or group.route_id not in _GENERAL_ROUTE_IDS
        or group.fallback_order != 0
        or group.allowance.physical_dispatches != 1
        or group.allowance.pages != 1
        or group.allowance.redirects != 0
        or group.allowance.retries != 0
        or type(group.intents) is not tuple
        or len(group.intents) != 1
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    intent = group.intents[0]
    limits = group.limits
    if (
        intent.operation_kind is not OperationKind.SEARCH
        or intent.method != "GET"
        or intent.path != "/europepmc/webservices/rest/search"
        or intent.json_body_pairs
        or intent.query_bindings
        or limits.max_pages != 1
        or limits.max_redirects != 0
        or limits.max_retries != 0
        or limits.timeout_ms != 20_000
        or limits.max_response_bytes != 2_097_152
        or limits.max_results != 100
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    pairs = tuple((pair.name, pair.value) for pair in intent.query_pairs)
    suffix = {
        "biorxiv_europe_pmc_search_aggregator": ' AND SRC:PPR AND PUBLISHER:"bioRxiv"',
        "medrxiv_europe_pmc_search_aggregator": ' AND SRC:PPR AND PUBLISHER:"medRxiv"',
    }[group.route_id]
    if (
        len(pairs) != 4
        or tuple(name for name, _value in pairs) != ("query", "format", "resultType", "pageSize")
        or pairs[1:] != (("format", "json"), ("resultType", "core"), pairs[3])
        or type(pairs[0][1]) is not str
        or not pairs[0][1].endswith(suffix)
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    literal_query = pairs[0][1][: -len(suffix)]
    terms = literal_query.split(" AND ")
    if not 1 <= len(terms) <= 16 or any(
        len(term) < 3
        or term[0] != '"'
        or term[-1] != '"'
        or not 1 <= len(term[1:-1]) <= 64
        or not all(character.isalnum() for character in term[1:-1])
        for term in terms
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    raw_page_size = pairs[3][1]
    if (
        type(raw_page_size) is not str
        or not raw_page_size.isascii()
        or not raw_page_size.isdecimal()
        or len(raw_page_size) > profile.max_numeric_token_chars
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    page_size = int(raw_page_size)
    if str(page_size) != raw_page_size or not 1 <= page_size <= min(profile.max_records, limits.max_results, 100):
        raise DiscoveryAdapterError("provider_payload_invalid")
    return (
        group,
        profile,
        min(profile.max_input_bytes, limits.max_response_bytes),
        page_size,
    )


def _authors(record: dict[str, Any], guard: _ParseGuard) -> tuple[str, ...]:
    author_list = record.get("authorList", _MISSING)
    if author_list is _MISSING or author_list is None:
        return ()
    raw_authors = _require_list(_require_dict(author_list).get("author", _MISSING))
    if len(raw_authors) > _MAX_AUTHORS:
        raise _ParseLimitExceeded
    authors: list[str] = []
    for raw_author in raw_authors:
        guard.checkpoint()
        value = _required_text(_require_dict(raw_author), "fullName")
        if len(value) > _MAX_AUTHOR_CHARS:
            raise _ParseLimitExceeded
        if any((ord(character) < 32 and character not in "\t\n\r") or ord(character) == 127 for character in value):
            raise _PayloadInvalid
        normalized = " ".join(value.split())
        if not normalized or len(normalized) > _MAX_AUTHOR_CHARS:
            raise _PayloadInvalid
        authors.append(normalized)
    return tuple(authors)


def _ppr_id(record: dict[str, Any]) -> str:
    value = _required_text(record, "id")
    if len(value) > _MAX_IDENTIFIER_CHARS or _PPR_ID_RE.fullmatch(value) is None:
        raise _PayloadInvalid
    return value


def _doi(record: dict[str, Any]) -> str | None:
    value = _optional_text(record, "doi")
    if value is None:
        return None
    canonical = value.lower()
    if (
        value != value.strip()
        or len(value) > _MAX_IDENTIFIER_CHARS
        or not value.isascii()
        or _DOI_RE.fullmatch(canonical) is None
    ):
        raise _PayloadInvalid
    return canonical


def _publication_values(record: dict[str, Any]) -> tuple[str | None, str | None]:
    raw_date = _optional_text(record, "firstPublicationDate")
    year = _optional_text(record, "pubYear")
    if year is not None:
        if _YEAR_RE.fullmatch(year) is None:
            raise _PayloadInvalid
        try:
            date.fromisoformat(f"{year}-01-01")
        except ValueError as error:
            raise _PayloadInvalid from error
    if raw_date is not None:
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError as error:
            raise _PayloadInvalid from error
        if len(raw_date) != 10 or parsed.isoformat() != raw_date:
            raise _PayloadInvalid
        if year is not None and raw_date[:4] != year:
            raise _PayloadInvalid
        if year is None:
            year = raw_date[:4]
    return raw_date, year


def _source_platform(record: dict[str, Any]) -> str | None:
    if record.get("source") != "PPR":
        return None
    details = record.get("bookOrReportDetails", _MISSING)
    if type(details) is not dict:
        return None
    publisher = details.get("publisher", _MISSING)
    if type(publisher) is not str:
        return None
    return {
        "biorxiv": "biorxiv",
        "medrxiv": "medrxiv",
    }.get(" ".join(publisher.split()).casefold())


def _europe_pmc_record(raw: Any, guard: _ParseGuard) -> dict[str, Any]:
    record = _require_dict(raw)
    ppr_id = _ppr_id(record)
    title = _plain_text(_required_text(record, "title"), max_chars=_MAX_TITLE_CHARS, required=True)
    raw_abstract = _optional_text(record, "abstractText")
    abstract = (
        None if raw_abstract is None else _plain_text(raw_abstract, max_chars=_MAX_ABSTRACT_CHARS, required=False)
    )
    doi = _doi(record)
    published_date, publication_year = _publication_values(record)
    provider_ids = {"europe_pmc_id": ppr_id}
    if doi is not None:
        provider_ids["doi"] = doi
    normalized = _base_record(
        title=title,
        authors=_authors(record, guard),
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=(f"https://doi.org/{doi}" if doi is not None else f"https://europepmc.org/article/PPR/{ppr_id}"),
        pdf_url=None,
        provider="europe_pmc",
        provider_ids=provider_ids,
    )
    normalized.update(
        {
            "published_date": published_date,
            "publication_year": publication_year,
            "ppr_id": ppr_id,
        }
    )
    source_platform = _source_platform(record)
    if source_platform is not None:
        normalized["source_platform"] = source_platform
    return normalized


def _europe_pmc_records(
    payload: Any,
    *,
    guard: _ParseGuard,
    max_records: int,
) -> tuple[dict[str, Any], ...]:
    root = _require_dict(payload)
    hit_count = root.get("hitCount", _MISSING)
    if type(hit_count) is not int or not 0 <= hit_count <= MAX_PAGINATION_CURSOR:
        raise _PayloadInvalid
    result_list = _require_dict(root.get("resultList", _MISSING))
    raw_records = _require_list(result_list.get("result", _MISSING))
    if len(raw_records) > max_records:
        raise _ParseLimitExceeded
    if hit_count < len(raw_records) or (hit_count > 0 and not raw_records):
        raise _PayloadInvalid
    normalized: list[dict[str, Any]] = []
    for raw_record in raw_records:
        guard.checkpoint()
        normalized.append(_europe_pmc_record(raw_record, guard))
    guard.checkpoint()
    return tuple(normalized)


def _deduplicated_candidates(records: tuple[dict[str, Any], ...]) -> tuple[DiscoveryCandidate, ...]:
    staged: list[tuple[str, dict[str, Any]]] = []
    by_candidate_id: dict[str, dict[str, Any]] = {}
    by_ppr_id: dict[str, dict[str, Any]] = {}
    for record in records:
        fingerprint = build_fingerprint(record)
        candidate_id = DiscoveryOutcomeIdentity.from_fingerprint(fingerprint).document_id
        ppr_id = record["ppr_id"]
        candidate_existing = by_candidate_id.get(candidate_id)
        ppr_existing = by_ppr_id.get(ppr_id)
        if (candidate_existing is not None and candidate_existing != record) or (
            ppr_existing is not None and ppr_existing != record
        ):
            raise _PayloadInvalid
        if candidate_existing is not None or ppr_existing is not None:
            continue
        by_candidate_id[candidate_id] = record
        by_ppr_id[ppr_id] = record
        staged.append((candidate_id, record))
    return tuple(DiscoveryCandidate(candidate_id, record) for candidate_id, record in staged)


async def _execute_europe_pmc_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    trusted_group, profile, max_input_bytes, max_records = _trusted_europe_pmc_inputs(group)
    intent = trusted_group.intents[0]
    response = await dispatch(intent)
    checked = _checked_response(response)
    payload, guard = _strict_json(
        checked,
        profile=profile,
        max_input_bytes=max_input_bytes,
        clock=clock,
    )
    try:
        records = _europe_pmc_records(payload, guard=guard, max_records=max_records)
        candidates = _deduplicated_candidates(records)
        guard.checkpoint()
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except DiscoveryAdapterError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    return DiscoveryAdapterResult(candidates)


def _family_source(source_id: str) -> SourceDefinition:
    display_name, alias, host, priority = {
        "biorxiv": ("bioRxiv", "bio_rxiv", "biorxiv.org", 90),
        "medrxiv": ("medRxiv", "med_rxiv", "medrxiv.org", 100),
    }[source_id]
    return SourceDefinition(
        catalog_source_id=source_id,
        display_name=display_name,
        aliases=(alias,),
        categories=("preprints",),
        content_types=("preprints", "papers", "abstracts"),
        surfaces=("standalone_search", "deep_research"),
        route_references=(
            SourceRouteReference(
                f"{source_id}_europe_pmc_search_aggregator",
                SourcePredicate(
                    field_path=("source_platform",),
                    operator=PredicateOperator.EQUALS_ANY,
                    values=(source_id,),
                    case_sensitive=False,
                ),
            ),
            SourceRouteReference(f"{source_id}_details_lookup_direct", None),
            SourceRouteReference(f"{source_id}_details_interval_direct", None),
        ),
        site_hosts=(host,),
        priority=priority,
        catalog_version=SHADOW_CATALOG_VERSION,
    )


def _family_routes() -> tuple[AccessRoute, ...]:
    return (
        _europe_pmc_route("biorxiv", "bioRxiv"),
        _europe_pmc_route("medrxiv", "medRxiv"),
        _details_lookup_route("biorxiv"),
        _details_lookup_route("medrxiv"),
        _details_interval_route("biorxiv"),
        _details_interval_route("medrxiv"),
    )


def _europe_pmc_route(source_id: str, publisher: str) -> AccessRoute:
    return AccessRoute(
        route_id=f"{source_id}_europe_pmc_search_aggregator",
        backend_id="europe_pmc_rest_api",
        adapter_id=EUROPE_PMC_ADAPTER_ID,
        route_kind=RouteKind.AGGREGATOR,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.PROVIDER_SOURCE_FILTER,
        attribution_basis="provider_publisher",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=1,
        adapter_version=EUROPE_PMC_ADAPTER_VERSION,
        policy=RoutePolicy(
            policy_version=ROUTE_POLICY_VERSION,
            origin=ExactOrigin("https", "www.ebi.ac.uk", 443),
            methods=("GET",),
            paths=("/europepmc/webservices/rest/search",),
            allowed_query_keys=("query", "format", "resultType", "pageSize"),
            query_value_policies=(
                LiteralTermsQueryValuePolicy(
                    "query",
                    f' AND SRC:PPR AND PUBLISHER:"{publisher}"',
                    16,
                    64,
                ),
                ExactQueryValuePolicy("format", "json"),
                ExactQueryValuePolicy("resultType", "core"),
                BoundedDecimalQueryValuePolicy("pageSize", 100),
            ),
            limits=_limits(max_pages=1, max_results=100),
        ),
    )


def _details_lookup_route(source_id: str) -> AccessRoute:
    return AccessRoute(
        route_id=f"{source_id}_details_lookup_direct",
        backend_id="biorxiv_details_api",
        adapter_id=DETAILS_ADAPTER_ID,
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.IDENTIFIER_LOOKUP,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=1,
        adapter_version=DETAILS_ADAPTER_VERSION,
        policy=RoutePolicy(
            policy_version=ROUTE_POLICY_VERSION,
            origin=ExactOrigin("https", "api.biorxiv.org", 443),
            methods=("GET",),
            paths=(),
            path_template=PathTemplate(
                (
                    "details",
                    source_id,
                    PathSlot(PathSlotKind.DOI_REGISTRANT, 12),
                    PathSlot(PathSlotKind.DOI_SUFFIX, 128),
                    "na",
                    "json",
                )
            ),
            allowed_query_keys=(),
            limits=_limits(max_pages=1, max_results=30),
        ),
    )


def _details_interval_route(source_id: str) -> AccessRoute:
    return AccessRoute(
        route_id=f"{source_id}_details_interval_direct",
        backend_id="biorxiv_details_api",
        adapter_id=DETAILS_ADAPTER_ID,
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.DATE_INTERVAL, QueryMode.CATEGORY_BROWSE),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_response",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=4,
        adapter_version=DETAILS_ADAPTER_VERSION,
        policy=RoutePolicy(
            policy_version=ROUTE_POLICY_VERSION,
            origin=ExactOrigin("https", "api.biorxiv.org", 443),
            methods=("GET",),
            paths=(),
            path_template=PathTemplate(
                (
                    "details",
                    source_id,
                    PathSlot(PathSlotKind.DATE, 10),
                    PathSlot(PathSlotKind.DATE, 10),
                    PathSlot(PathSlotKind.UINT, 10),
                    "json",
                ),
                pagination_segment_index=4,
            ),
            allowed_query_keys=("category",),
            query_value_policies=(BoundedTextQueryValuePolicy("category", 128),),
            limits=_limits(max_pages=4, max_results=120),
        ),
    )


def _limits(*, max_pages: int, max_results: int) -> RouteLimits:
    return RouteLimits(
        max_pages=max_pages,
        max_redirects=0,
        max_retries=0,
        timeout_ms=20_000,
        max_response_bytes=2_097_152,
        max_results=max_results,
    )
