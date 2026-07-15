"""Shadow-only bioRxiv and medRxiv discovery route family."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any
from urllib.parse import quote_from_bytes

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
    NumericCursor,
)
from .gateway_adapters import (
    MonotonicClock,
    _base_record,
    _canonical_decimal_text,
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

_EUROPE_PMC_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=120,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_DETAILS_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=120,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_FAMILY_PARSING_PROFILES = MappingProxyType(
    {
        (EUROPE_PMC_ADAPTER_ID, EUROPE_PMC_ADAPTER_VERSION): _EUROPE_PMC_PROFILE,
        (DETAILS_ADAPTER_ID, DETAILS_ADAPTER_VERSION): _DETAILS_PROFILE,
    }
)
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
    """Return explicit shadow readiness for the fixture-proven family routes."""
    foundation = foundation_readiness(execution_mode)
    ready_reason = f"{execution_mode.value}_ready"
    family_entries = tuple(
        RouteReadiness(
            route_id=route_id,
            state=ReadinessState.READY,
            credential_status=CredentialStatus.NOT_REQUIRED,
            reason=ready_reason,
        )
        for route_id in _GENERAL_ROUTE_IDS + _DETAILS_ROUTE_IDS
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

    async def details_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_details_adapter(group, dispatch, monotonic_clock)

    return _compose_adapter_maps(
        {EUROPE_PMC_ADAPTER_ID: europe_pmc_adapter},
        {DETAILS_ADAPTER_ID: details_adapter},
    )


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


def _normalized_details_category(value: Any) -> str:
    if type(value) is not str or not value:
        raise _PayloadInvalid
    if len(value) > 128:
        raise _ParseLimitExceeded
    canonical = unicodedata.normalize("NFKC", value).replace("_", " ")
    if any(not character.isalnum() and character not in " -&/" for character in canonical):
        raise _PayloadInvalid
    normalized = " ".join(canonical.split()).casefold()
    if not normalized or not any(character.isalnum() for character in normalized):
        raise _PayloadInvalid
    return normalized


def _details_integer(
    value: Any,
    profile: _ParsingProfile,
    *,
    positive: bool = False,
) -> int:
    if type(value) is int:
        parsed = value
        if parsed < 0 or (positive and parsed == 0) or parsed > MAX_PAGINATION_CURSOR:
            raise _PayloadInvalid
        return parsed
    return _canonical_decimal_text(
        value,
        profile,
        positive=positive,
        maximum=MAX_PAGINATION_CURSOR,
    )


def _trusted_details_inputs(
    group: object,
) -> tuple[
    PlannedDispatchGroup,
    _ParsingProfile,
    int,
    int,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """Validate one exact details plan and expose only its bound request values."""
    if type(group) is not PlannedDispatchGroup:
        raise DiscoveryAdapterError("provider_payload_invalid")
    route_values = {
        "biorxiv_details_lookup_direct": ("biorxiv", "bioRxiv", "doi"),
        "medrxiv_details_lookup_direct": ("medrxiv", "medRxiv", "doi"),
        "biorxiv_details_interval_direct": ("biorxiv", "bioRxiv", "interval"),
        "medrxiv_details_interval_direct": ("medrxiv", "medRxiv", "interval"),
    }.get(group.route_id)
    profile = _FAMILY_PARSING_PROFILES.get((group.adapter_id, group.adapter_version))
    if route_values is None or profile is None:
        raise DiscoveryAdapterError("provider_payload_invalid")
    source_id, response_server, mode = route_values
    limits = group.limits
    expected_pages, expected_results, expected_physical = (1, 30, 1) if mode == "doi" else (4, 120, 4)
    if (
        group.adapter_id != DETAILS_ADAPTER_ID
        or group.adapter_version != DETAILS_ADAPTER_VERSION
        or group.fallback_order != 0
        or group.backend_id != "biorxiv_details_api"
        or group.filters
        or group.allowance.physical_dispatches != expected_physical
        or group.allowance.pages != expected_pages
        or group.allowance.redirects != 0
        or group.allowance.retries != 0
        or limits.max_pages != expected_pages
        or limits.max_redirects != 0
        or limits.max_retries != 0
        or limits.timeout_ms != 20_000
        or limits.max_response_bytes != 2_097_152
        or limits.max_results != expected_results
        or type(group.intents) is not tuple
        or len(group.intents) != 1
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    intent = group.intents[0]
    if (
        intent.route_id != group.route_id
        or intent.policy_digest != group.policy_digest
        or intent.operation_kind is not OperationKind.SEARCH
        or intent.method != "GET"
        or intent.limits != limits
        or intent.json_body_pairs
        or intent.query_bindings
        or type(group.normalized_query) is not str
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")

    doi: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    category: str | None = None
    if mode == "doi":
        doi = group.normalized_query
        if (
            not doi.isascii()
            or doi != doi.lower()
            or doi.count("/") != 1
            or _DOI_RE.fullmatch(doi) is None
            or intent.query_pairs
        ):
            raise DiscoveryAdapterError("provider_payload_invalid")
        registrant, suffix = doi.split("/", 1)
        if len(suffix) > 128:
            raise DiscoveryAdapterError("provider_payload_invalid")
        encoded_suffix = quote_from_bytes(suffix.encode("ascii"), safe="")
        expected_path = f"/details/{source_id}/{registrant}/{encoded_suffix}/na/json"
        if intent.path != expected_path:
            raise DiscoveryAdapterError("provider_payload_invalid")
    else:
        segments = intent.path.split("/")
        if len(segments) != 7 or segments[:3] != ["", "details", source_id] or segments[5:] != ["0", "json"]:
            raise DiscoveryAdapterError("provider_payload_invalid")
        start_date, end_date = segments[3:5]
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError:
            raise DiscoveryAdapterError("provider_payload_invalid") from None
        if (
            start.isoformat() != start_date
            or end.isoformat() != end_date
            or start > end
            or (end - start).days + 1 > 366
        ):
            raise DiscoveryAdapterError("provider_payload_invalid")
        pairs = tuple((pair.name, pair.value) for pair in intent.query_pairs)
        if pairs:
            if (
                len(pairs) != 1
                or pairs[0][0] != "category"
                or type(pairs[0][1]) is not str
                or unicodedata.normalize("NFKC", pairs[0][1]) != pairs[0][1]
                or pairs[0][1] != pairs[0][1].strip()
                or "  " in pairs[0][1]
                or "_" in pairs[0][1]
            ):
                raise DiscoveryAdapterError("provider_payload_invalid")
            try:
                category = _normalized_details_category(pairs[0][1])
            except (_PayloadInvalid, _ParseLimitExceeded):
                raise DiscoveryAdapterError("provider_payload_invalid") from None
        expected_query = f"{start_date}/{end_date}" + (f"/{pairs[0][1]}" if pairs else "")
        if group.normalized_query != expected_query:
            raise DiscoveryAdapterError("provider_payload_invalid")
    return (
        group,
        profile,
        min(profile.max_input_bytes, limits.max_response_bytes),
        min(profile.max_records, limits.max_results),
        source_id,
        response_server,
        mode,
        doi,
        start_date,
        end_date,
        category,
    )


def _details_authors(record: dict[str, Any], guard: _ParseGuard) -> tuple[str, ...]:
    raw = _required_text(record, "authors")
    parts = raw.split(";")
    if len(parts) > _MAX_AUTHORS:
        raise _ParseLimitExceeded
    authors: list[str] = []
    for part in parts:
        guard.checkpoint()
        author = _plain_text(part, max_chars=_MAX_AUTHOR_CHARS, required=True)
        if author is None:
            raise _PayloadInvalid
        authors.append(author)
    return tuple(authors)


def _details_date(record: dict[str, Any]) -> str:
    value = _required_text(record, "date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _PayloadInvalid from error
    if len(value) != 10 or parsed.isoformat() != value:
        raise _PayloadInvalid
    return value


def _published_doi(record: dict[str, Any]) -> str | None:
    value = _optional_text(record, "published")
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.casefold() == "na":
        return None
    canonical = stripped.lower()
    if (
        value != stripped
        or len(value) > _MAX_IDENTIFIER_CHARS
        or not value.isascii()
        or _DOI_RE.fullmatch(canonical) is None
    ):
        raise _PayloadInvalid
    return canonical


def _details_record(
    raw: Any,
    *,
    guard: _ParseGuard,
    source_id: str,
    response_server: str,
    expected_doi: str | None,
    start_date: str | None,
    end_date: str | None,
    expected_category: str | None,
    profile: _ParsingProfile,
) -> dict[str, Any]:
    record = _require_dict(raw)
    if record.get("server") != response_server:
        raise _PayloadInvalid
    doi = _doi(record)
    if doi is None or (expected_doi is not None and doi != expected_doi):
        raise _PayloadInvalid
    title = _plain_text(_required_text(record, "title"), max_chars=_MAX_TITLE_CHARS, required=True)
    if title is None:
        raise _PayloadInvalid
    raw_abstract = _optional_text(record, "abstract")
    abstract = (
        None if raw_abstract is None else _plain_text(raw_abstract, max_chars=_MAX_ABSTRACT_CHARS, required=False)
    )
    published_date = _details_date(record)
    if start_date is not None and end_date is not None and not start_date <= published_date <= end_date:
        raise _PayloadInvalid
    version = _details_integer(record.get("version", _MISSING), profile, positive=True)
    license_value = _plain_text(
        _required_text(record, "license"),
        max_chars=_MAX_IDENTIFIER_CHARS,
        required=True,
    )
    if license_value is None:
        raise _PayloadInvalid
    category = _normalized_details_category(_required_text(record, "category"))
    if expected_category is not None and category != expected_category:
        raise _PayloadInvalid
    normalized = _base_record(
        title=title,
        authors=_details_authors(record, guard),
        abstract=abstract,
        snippet=abstract,
        doi=doi,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=f"https://doi.org/{doi}",
        pdf_url=None,
        provider="biorxiv_details",
        provider_ids={"doi": doi, "version": str(version)},
    )
    normalized.update(
        {
            "published_date": published_date,
            "publication_year": published_date[:4],
            "version": version,
            "license": license_value,
            "category": category,
            "published_doi": _published_doi(record),
            "source_platform": source_id,
        }
    )
    return normalized


def _details_page(
    payload: Any,
    *,
    guard: _ParseGuard,
    profile: _ParsingProfile,
    source_id: str,
    response_server: str,
    mode: str,
    expected_doi: str | None,
    start_date: str | None,
    end_date: str | None,
    expected_category: str | None,
    current_cursor: int,
    seen_records: int,
    expected_total: int | None,
    remaining_records: int,
) -> tuple[tuple[dict[str, Any], ...], int, int | None, bool]:
    root = _require_dict(payload)
    if set(root) != {"messages", "collection"}:
        raise _PayloadInvalid
    messages = _require_list(root.get("messages", _MISSING))
    collection = _require_list(root.get("collection", _MISSING))
    if len(messages) != 1:
        raise _PayloadInvalid
    message = _require_dict(messages[0])
    status = message.get("status", _MISSING)
    if status == "no posts found":
        if message != {"status": "no posts found"} or collection or current_cursor != 0 or seen_records:
            raise _PayloadInvalid
        return (), 0, None, True
    if status != "ok" or len(collection) > remaining_records:
        if len(collection) > remaining_records:
            raise _ParseLimitExceeded
        raise _PayloadInvalid

    if mode == "doi":
        if message != {"status": "ok", "category": "all"} or not collection:
            raise _PayloadInvalid
        normalized = tuple(
            _details_record(
                raw,
                guard=guard,
                source_id=source_id,
                response_server=response_server,
                expected_doi=expected_doi,
                start_date=None,
                end_date=None,
                expected_category=None,
                profile=profile,
            )
            for raw in collection
        )
        guard.checkpoint()
        return normalized, len(normalized), None, True

    if start_date is None or end_date is None:
        raise _PayloadInvalid
    response_category = message.get("category", _MISSING)
    if expected_category is None:
        if response_category != "all":
            raise _PayloadInvalid
    elif _normalized_details_category(response_category) != expected_category:
        raise _PayloadInvalid
    if message.get("interval") != f"{start_date}:{end_date}":
        raise _PayloadInvalid
    response_cursor = _details_integer(message.get("cursor", _MISSING), profile)
    count = _details_integer(message.get("count", _MISSING), profile)
    total = _details_integer(message.get("total", _MISSING), profile)
    if (
        response_cursor != current_cursor
        or count != len(collection)
        or count == 0
        or current_cursor + count > total
        or (expected_total is not None and total != expected_total)
    ):
        raise _PayloadInvalid
    normalized = tuple(
        _details_record(
            raw,
            guard=guard,
            source_id=source_id,
            response_server=response_server,
            expected_doi=None,
            start_date=start_date,
            end_date=end_date,
            expected_category=expected_category,
            profile=profile,
        )
        for raw in collection
    )
    guard.checkpoint()
    return normalized, count, total, current_cursor + count == total


def _details_candidates(records: tuple[dict[str, Any], ...]) -> tuple[DiscoveryCandidate, ...]:
    order: list[str] = []
    versions_by_doi: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        doi = record["doi"]
        version = record["version"]
        versions = versions_by_doi.get(doi)
        if versions is None:
            versions = {}
            versions_by_doi[doi] = versions
            order.append(doi)
        existing = versions.get(version)
        if existing is not None and existing != record:
            raise _PayloadInvalid
        versions[version] = record

    candidates: list[DiscoveryCandidate] = []
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for doi in order:
        versions = versions_by_doi[doi]
        record = versions[max(versions)]
        candidate_id = DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(record)).document_id
        existing = by_candidate_id.get(candidate_id)
        if existing is not None:
            if existing != record:
                raise _PayloadInvalid
            continue
        by_candidate_id[candidate_id] = record
        candidates.append(DiscoveryCandidate(candidate_id, record))
    return tuple(candidates)


async def _execute_details_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    (
        trusted_group,
        profile,
        max_input_bytes,
        max_records,
        source_id,
        response_server,
        mode,
        expected_doi,
        start_date,
        end_date,
        expected_category,
    ) = _trusted_details_inputs(group)
    intent = trusted_group.intents[0]
    current_cursor = 0
    cursor: NumericCursor | None = None
    expected_total: int | None = None
    records: list[dict[str, Any]] = []

    for page_index in range(trusted_group.limits.max_pages):
        response = await dispatch(intent, cursor=cursor)
        checked = _checked_response(response)
        payload, guard = _strict_json(
            checked,
            profile=profile,
            max_input_bytes=max_input_bytes,
            clock=clock,
        )
        try:
            page, count, total, terminal = _details_page(
                payload,
                guard=guard,
                profile=profile,
                source_id=source_id,
                response_server=response_server,
                mode=mode,
                expected_doi=expected_doi,
                start_date=start_date,
                end_date=end_date,
                expected_category=expected_category,
                current_cursor=current_cursor,
                seen_records=len(records),
                expected_total=expected_total,
                remaining_records=max_records - len(records),
            )
            records.extend(page)
            if total is not None and expected_total is None:
                expected_total = total
            guard.checkpoint()
        except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
            _raise_adapter_error(error)
        except DiscoveryAdapterError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError):
            raise DiscoveryAdapterError("provider_payload_invalid") from None

        if terminal or mode == "doi" or len(records) >= max_records or page_index + 1 >= trusted_group.limits.max_pages:
            break
        current_cursor += count
        cursor = NumericCursor(current_cursor)

    try:
        candidates = _details_candidates(tuple(records))
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
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
