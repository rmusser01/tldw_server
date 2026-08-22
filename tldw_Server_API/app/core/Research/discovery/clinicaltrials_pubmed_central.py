"""Shadow-only ClinicalTrials.gov and PubMed Central discovery family."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any, cast

from .contracts import (
    MAX_PAGINATION_CURSOR,
    AccessRoute,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    CredentialRequirement,
    CredentialStatus,
    DeferredNumericCSVQueryBinding,
    DiscoveryOutcomeIdentity,
    DispatchAllowance,
    DispatchIntent,
    ExactOrigin,
    ExactQueryValuePolicy,
    ExecutionMode,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    OperationKind,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
    QueryMode,
    QueryPair,
    ReadinessOverlay,
    ReadinessState,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    RouteReadiness,
    SourceConstraint,
    SourceDefinition,
    SourceRouteReference,
)
from .executor import (
    BoundDispatch,
    DiscoveryAdapter,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    DiscoveryExecutionError,
    OpaqueCursor,
)
from .gateway_adapters import (
    MonotonicClock,
    _base_record,
    _canonical_decimal_text,
    _checked_response,
    _execute_ncbi_esearch_summary,
    _guarded_items,
    _ncbi_json_root,
    _ParseDeadlineExceeded,
    _ParseGuard,
    _ParseLimitExceeded,
    _ParsingProfile,
    _PayloadInvalid,
    _raise_adapter_error,
    _require_dict,
    _require_list,
    _strict_json,
    _TrustedNCBIInputs,
    _validate_ncbi_message_list,
)
from .identity import build_fingerprint, has_unsafe_url_material, normalize_doi
from .registry import DiscoveryRegistry, foundation_readiness, foundation_registry

_MISSING = object()

SHADOW_CATALOG_VERSION = "research-discovery-v2-clinicaltrials-pmc-shadow"
SHADOW_REGISTRY_VERSION = "research-discovery-v2-clinicaltrials-pmc-shadow-2026-08-21"
SHADOW_READINESS_VERSION = "research-discovery-readiness-v2-clinicaltrials-pmc-shadow"
ROUTE_POLICY_VERSION = "research-discovery-route-policy-v2-clinicaltrials-pmc"
CLINICALTRIALS_GOV_ADAPTER_ID = "clinicaltrials_gov_v2"
CLINICALTRIALS_GOV_ADAPTER_VERSION = "clinicaltrials-gov-v2"
PUBMED_CENTRAL_ADAPTER_ID = "pubmed_central_v2"
PUBMED_CENTRAL_ADAPTER_VERSION = "pubmed-central-v2"
PUBMED_IDENTITY_POLICY_VERSION = "research-discovery-route-policy-v2-foundation-pubmed-ncbi-identity-2026-08-21"
PUBMED_IDENTITY_ADAPTER_VERSION = "pubmed-v2-ncbi-identity"
NCBI_TOOL = "tldw_server"
NCBI_EMAIL = "contact@tldwproject.com"
CLINICALTRIALS_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,BriefSummary,OverallStatus,Condition,"
    "InterventionName,LeadSponsorName,StudyType,StartDate,CompletionDate,HasResults"
)

_CLINICALTRIALS_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=50,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_PMC_PROFILE = _ParsingProfile(
    max_input_bytes=2_097_152,
    max_records=100,
    max_depth=16,
    max_nodes=50_000,
    max_string_chars=65_536,
    max_numeric_token_chars=32,
    parse_deadline_ms=500,
)
_FAMILY_PARSING_PROFILES = MappingProxyType(
    {
        (CLINICALTRIALS_GOV_ADAPTER_ID, CLINICALTRIALS_GOV_ADAPTER_VERSION): _CLINICALTRIALS_PROFILE,
        (PUBMED_CENTRAL_ADAPTER_ID, PUBMED_CENTRAL_ADAPTER_VERSION): _PMC_PROFILE,
    }
)

_PUBMED_ROUTE_ID = "pubmed_ncbi_eutils_pubmed_direct"
_PUBMED_OVERLAY_QUERY_KEYS = (
    "db",
    "term",
    "retstart",
    "retmax",
    "retmode",
    "sort",
    "datetype",
    "mindate",
    "maxdate",
    "tool",
    "email",
    "id",
)
_NCT_ID_RE = re.compile(r"NCT[0-9]{8}\Z", re.ASCII)
_RESIDUAL_ENTITY_RE = re.compile(r"&(?:#[0-9]{1,7}|#x[0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{0,31});", re.ASCII)
_COMMONMARK_LINK_RE = re.compile(
    r"!?\[[^\]\r\n]{0,1024}\]\([^\)\r\n]{0,4096}\)|<[^>\r\n]{0,4096}://[^>\r\n]{0,4096}>",
    re.ASCII,
)
_URL_MATERIAL_RE = re.compile(
    r"(?:https?://|ftp://|www\.|mailto:|data:|javascript:)[^\s<>\x00-\x1f]{0,4096}",
    re.IGNORECASE | re.ASCII,
)


def clinicaltrials_pubmed_central_shadow_registry() -> DiscoveryRegistry:
    """Return the foundation, PubMed overlay, and the two shadow family routes."""
    foundation = foundation_registry()
    routes = tuple(
        _pubmed_identity_overlay(route) if route.route_id == _PUBMED_ROUTE_ID else route for route in foundation.routes
    )
    return DiscoveryRegistry(
        catalog_version=SHADOW_CATALOG_VERSION,
        registry_version=SHADOW_REGISTRY_VERSION,
        sources=tuple(replace(source, catalog_version=SHADOW_CATALOG_VERSION) for source in foundation.sources)
        + (_clinicaltrials_source(), _pubmed_central_source()),
        routes=routes + (_clinicaltrials_route(), _pubmed_central_route()),
        backends=foundation.backends
        + (
            BackendDefinition("clinicaltrials_gov_api_v2", "ClinicalTrials.gov API v2"),
            BackendDefinition("ncbi_eutils_pmc", "NCBI Entrez E-utilities for PMC"),
        ),
    )


def clinicaltrials_pubmed_central_shadow_readiness(execution_mode: ExecutionMode) -> ReadinessOverlay:
    """Return explicit readiness for the identity overlay and fixture-proven family."""
    foundation = foundation_readiness(execution_mode)
    reconstructed = tuple(
        RouteReadiness(
            route_id=entry.route_id,
            state=entry.state,
            credential_status=entry.credential_status,
            reason=entry.reason,
        )
        for entry in foundation.routes
    )
    ready_reason = f"{execution_mode.value}_ready"
    return ReadinessOverlay(
        overlay_version=SHADOW_READINESS_VERSION,
        execution_mode=execution_mode,
        routes=reconstructed
        + tuple(
            RouteReadiness(
                route_id=route_id,
                state=ReadinessState.READY,
                credential_status=CredentialStatus.NOT_REQUIRED,
                reason=ready_reason,
            )
            for route_id in (
                "clinicaltrials_gov_studies_search_direct",
                "pubmed_central_esearch_summary_direct",
            )
        ),
    )


def _pubmed_identity_overlay(route: AccessRoute) -> AccessRoute:
    """Return the exact identity-bearing replacement for the foundation PubMed route."""
    return replace(
        route,
        adapter_version=PUBMED_IDENTITY_ADAPTER_VERSION,
        policy=replace(
            route.policy,
            policy_version=PUBMED_IDENTITY_POLICY_VERSION,
            allowed_query_keys=_PUBMED_OVERLAY_QUERY_KEYS,
            pagination_query_key="retstart",
            query_value_policies=(),
            policy_digest="",
        ),
    )


def _clinicaltrials_source() -> SourceDefinition:
    return SourceDefinition(
        catalog_source_id="clinicaltrials_gov",
        display_name="ClinicalTrials.gov",
        aliases=("clinical_trials_gov", "clinical_trials"),
        categories=("biomedical", "clinical_trials"),
        content_types=("clinical_trials", "study_records", "summaries"),
        surfaces=("standalone_search", "deep_research"),
        route_references=(SourceRouteReference("clinicaltrials_gov_studies_search_direct", None),),
        site_hosts=("clinicaltrials.gov",),
        priority=110,
        catalog_version=SHADOW_CATALOG_VERSION,
    )


def _pubmed_central_source() -> SourceDefinition:
    return SourceDefinition(
        catalog_source_id="pubmed_central",
        display_name="PubMed Central",
        aliases=("pmc", "pub_med_central"),
        categories=("biomedical", "open_access"),
        content_types=("papers", "full_text_archive", "biomedical_metadata"),
        surfaces=("standalone_search", "deep_research"),
        route_references=(SourceRouteReference("pubmed_central_esearch_summary_direct", None),),
        site_hosts=("pmc.ncbi.nlm.nih.gov",),
        priority=120,
        catalog_version=SHADOW_CATALOG_VERSION,
    )


def _clinicaltrials_route() -> AccessRoute:
    return AccessRoute(
        route_id="clinicaltrials_gov_studies_search_direct",
        backend_id="clinicaltrials_gov_api_v2",
        adapter_id=CLINICALTRIALS_GOV_ADAPTER_ID,
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native_nct_record",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version=CLINICALTRIALS_GOV_ADAPTER_VERSION,
        policy=RoutePolicy(
            policy_version=ROUTE_POLICY_VERSION,
            origin=ExactOrigin("https", "clinicaltrials.gov", 443),
            methods=("GET",),
            paths=("/api/v2/studies",),
            allowed_query_keys=(
                "query.term",
                "format",
                "markupFormat",
                "fields",
                "pageSize",
                "countTotal",
                "pageToken",
            ),
            limits=RouteLimits(2, 0, 0, 20_000, 2_097_152, 100, 16_384),
            pagination_query_key="pageToken",
            query_value_policies=(
                LiteralTermsQueryValuePolicy("query.term", "", 8, 32),
                ExactQueryValuePolicy("format", "json"),
                ExactQueryValuePolicy("markupFormat", "legacy"),
                ExactQueryValuePolicy("fields", CLINICALTRIALS_FIELDS),
                BoundedDecimalQueryValuePolicy("pageSize", 50),
                ExactQueryValuePolicy("countTotal", "true"),
                OpaqueCursorQueryValuePolicy("pageToken", 1_024, required=False),
            ),
        ),
    )


def _pubmed_central_route() -> AccessRoute:
    return AccessRoute(
        route_id="pubmed_central_esearch_summary_direct",
        backend_id="ncbi_eutils_pmc",
        adapter_id=PUBMED_CENTRAL_ADAPTER_ID,
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="ncbi_pmc_database",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version=PUBMED_CENTRAL_ADAPTER_VERSION,
        policy=RoutePolicy(
            policy_version=ROUTE_POLICY_VERSION,
            origin=ExactOrigin("https", "eutils.ncbi.nlm.nih.gov", 443),
            methods=("GET",),
            paths=("/entrez/eutils/esearch.fcgi", "/entrez/eutils/esummary.fcgi"),
            allowed_query_keys=("db", "term", "retstart", "retmax", "retmode", "tool", "email", "id"),
            limits=RouteLimits(1, 0, 0, 20_000, 2_097_152, 100, 16_384),
            pagination_query_key="retstart",
            query_value_policies=(),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ClinicalTrialsPage:
    total_count: int
    records: tuple[dict[str, Any], ...]
    next_page_token: str | None


def _trusted_clinicaltrials_inputs(
    group: object,
) -> tuple[PlannedDispatchGroup, _ParsingProfile, int, int]:
    """Return exact group, profile, max input bytes, and requested page size."""
    if type(group) is not PlannedDispatchGroup:
        raise DiscoveryAdapterError("provider_payload_invalid")
    profile = _FAMILY_PARSING_PROFILES.get((group.adapter_id, group.adapter_version))
    exact_policy = _clinicaltrials_route().policy
    if (
        group.route_id != "clinicaltrials_gov_studies_search_direct"
        or group.backend_id != "clinicaltrials_gov_api_v2"
        or group.adapter_id != CLINICALTRIALS_GOV_ADAPTER_ID
        or group.adapter_version != CLINICALTRIALS_GOV_ADAPTER_VERSION
        or profile is None
        or group.policy_digest != exact_policy.policy_digest
        or group.fallback_order != 0
        or group.filters != ()
        or group.allowance.pages != 2
        or group.allowance.physical_dispatches != 2
        or group.allowance.redirects != 0
        or group.allowance.retries != 0
        or type(group.intents) is not tuple
        or len(group.intents) != 1
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    limits = group.limits
    if limits != RouteLimits(2, 0, 0, 20_000, 2_097_152, 100, 16_384):
        raise DiscoveryAdapterError("provider_payload_invalid")
    intent = group.intents[0]
    if (
        type(intent.query_pairs) is not tuple
        or type(intent.json_body_pairs) is not tuple
        or intent.json_body_pairs != ()
        or type(intent.query_bindings) is not tuple
        or intent.query_bindings != ()
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    pairs = tuple((pair.name, pair.value) for pair in intent.query_pairs)
    if (
        intent.route_id != group.route_id
        or intent.route_id != "clinicaltrials_gov_studies_search_direct"
        or intent.operation_kind is not OperationKind.SEARCH
        or intent.method != "GET"
        or intent.path != "/api/v2/studies"
        or intent.limits != limits
        or intent.policy_digest != group.policy_digest
        or len(pairs) != 6
        or tuple(name for name, _value in pairs)
        != ("query.term", "format", "markupFormat", "fields", "pageSize", "countTotal")
        or pairs[1:]
        != (
            ("format", "json"),
            ("markupFormat", "legacy"),
            ("fields", CLINICALTRIALS_FIELDS),
            pairs[4],
            ("countTotal", "true"),
        )
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    literal_terms = pairs[0][1].split(" AND ") if type(pairs[0][1]) is str else ()
    if not 1 <= len(literal_terms) <= 8 or any(
        len(term) < 3
        or term[0] != '"'
        or term[-1] != '"'
        or not 1 <= len(term[1:-1]) <= 32
        or not all(character.isalnum() for character in term[1:-1])
        for term in literal_terms
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    raw_page_size = pairs[4][1]
    if (
        type(raw_page_size) is not str
        or not raw_page_size.isascii()
        or not raw_page_size.isdecimal()
        or len(raw_page_size) > profile.max_numeric_token_chars
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    page_size = int(raw_page_size)
    if str(page_size) != raw_page_size or not 1 <= page_size <= 50:
        raise DiscoveryAdapterError("provider_payload_invalid")
    return group, profile, min(profile.max_input_bytes, limits.max_response_bytes), page_size


def _has_forbidden_text_character(value: str) -> bool:
    return any(character == "\ufffd" or unicodedata.category(character) in {"Cc", "Cs"} for character in value)


def _contains_residual_markup(value: str) -> bool:
    return "<" in value or ">" in value or _RESIDUAL_ENTITY_RE.search(value) is not None


def _contains_url_material(value: str) -> bool:
    """Reject any bounded URL/URI token in provider-supplied human text."""
    return _URL_MATERIAL_RE.search(value) is not None or has_unsafe_url_material(value)


def _plain_clinical_text(
    value: Any,
    *,
    max_chars: int,
    required: bool,
) -> str | None:
    """Normalize Unicode whitespace and reject controls, markup, or URL material."""
    invalid = (
        type(value) is not str
        or len(value) > max_chars
        or _has_forbidden_text_character(value)
        or _contains_residual_markup(value)
        or _contains_url_material(value)
    )
    normalized = None if invalid else " ".join(value.split())
    if normalized is not None and (not normalized or len(normalized) > max_chars):
        normalized = None
    if normalized is None and required:
        raise _PayloadInvalid
    return normalized


class _LegacySummaryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def _legacy_summary_text(value: Any) -> str | None:
    """Return at most 16,384 inert characters, or drop unsafe optional content."""
    if type(value) is not str:
        raise _PayloadInvalid
    if (
        len(value) > 65_536
        or _has_forbidden_text_character(value)
        or _COMMONMARK_LINK_RE.search(value) is not None
        or _contains_url_material(value)
    ):
        return None
    try:
        parser = _LegacySummaryParser()
        parser.feed(value)
        parser.close()
    except Exception:  # noqa: BLE001 - hostile optional markup is dropped fail-closed.
        return None
    text = " ".join("".join(parser.parts).split())
    if (
        not text
        or len(text) > 16_384
        or _has_forbidden_text_character(text)
        or _contains_residual_markup(text)
        or _contains_url_material(text)
    ):
        return None
    return text


def _partial_date(value: Any) -> str | None:
    """Return a valid exact partial date, drop invalid strings, and reject wrong types."""
    if type(value) is not str:
        raise _PayloadInvalid
    try:
        if re.fullmatch(r"[0-9]{4}", value):
            date(int(value), 1, 1)
        elif re.fullmatch(r"[0-9]{4}-[0-9]{2}", value):
            year, month = map(int, value.split("-"))
            date(year, month, 1)
        elif re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            if date.fromisoformat(value).isoformat() != value:
                return None
        else:
            return None
    except ValueError:
        return None
    return value


def _optional_container(record: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = record.get(key, _MISSING)
    if value is _MISSING:
        return None
    return _require_dict(value)


def _optional_plain_field(container: dict[str, Any] | None, key: str, max_chars: int) -> str | None:
    if container is None or key not in container:
        return None
    value = container[key]
    if type(value) is not str:
        raise _PayloadInvalid
    return _plain_clinical_text(value, max_chars=max_chars, required=False)


def _optional_text_list(
    container: dict[str, Any] | None,
    key: str,
    *,
    guard: _ParseGuard,
) -> tuple[str, ...] | None:
    if container is None or key not in container:
        return None
    values = _require_list(container[key])
    if len(values) > 64:
        return None
    normalized: list[str] = []
    for value in values:
        guard.checkpoint()
        if type(value) is not str:
            raise _PayloadInvalid
        item = _plain_clinical_text(value, max_chars=512, required=False)
        if item is None:
            return None
        normalized.append(item)
    return tuple(normalized)


def _optional_interventions(
    container: dict[str, Any] | None,
    *,
    guard: _ParseGuard,
) -> tuple[str, ...] | None:
    if container is None or "interventions" not in container:
        return None
    values = _require_list(container["interventions"])
    if len(values) > 64:
        return None
    normalized: list[str] = []
    for value in values:
        guard.checkpoint()
        item = _require_dict(value)
        if "name" not in item or type(item["name"]) is not str:
            raise _PayloadInvalid
        name = _plain_clinical_text(item["name"], max_chars=512, required=False)
        if name is None:
            return None
        normalized.append(name)
    return tuple(normalized)


def _optional_date(status: dict[str, Any] | None, key: str) -> str | None:
    if status is None or key not in status:
        return None
    structure = _require_dict(status[key])
    if "date" not in structure:
        return None
    return _partial_date(structure["date"])


def _clinicaltrials_record(raw: Any, *, guard: _ParseGuard) -> dict[str, Any]:
    """Normalize only the frozen study projection."""
    record = _require_dict(raw)
    protocol = _require_dict(record.get("protocolSection", _MISSING))
    identification = _require_dict(protocol.get("identificationModule", _MISSING))
    nct_id = identification.get("nctId", _MISSING)
    if type(nct_id) is not str or _NCT_ID_RE.fullmatch(nct_id) is None:
        raise _PayloadInvalid
    brief = (
        None
        if "briefTitle" not in identification
        else _plain_clinical_text(identification["briefTitle"], max_chars=1_024, required=False)
    )
    official = (
        None
        if "officialTitle" not in identification
        else _plain_clinical_text(identification["officialTitle"], max_chars=4_096, required=False)
    )
    title = brief or official
    if title is None:
        raise _PayloadInvalid

    description = _optional_container(protocol, "descriptionModule")
    summary: str | None = None
    if description is not None and "briefSummary" in description:
        summary = _legacy_summary_text(description["briefSummary"])
    status = _optional_container(protocol, "statusModule")
    overall_status = _optional_plain_field(status, "overallStatus", 256)
    conditions = _optional_text_list(
        _optional_container(protocol, "conditionsModule"),
        "conditions",
        guard=guard,
    )
    interventions = _optional_interventions(
        _optional_container(protocol, "armsInterventionsModule"),
        guard=guard,
    )
    sponsor_module = _optional_container(protocol, "sponsorCollaboratorsModule")
    lead_sponsor: str | None = None
    if sponsor_module is not None and "leadSponsor" in sponsor_module:
        sponsor = _require_dict(sponsor_module["leadSponsor"])
        lead_sponsor = _optional_plain_field(sponsor, "name", 1_024)
    study_type = _optional_plain_field(_optional_container(protocol, "designModule"), "studyType", 256)
    start_date = _optional_date(status, "startDateStruct")
    completion_date = _optional_date(status, "completionDateStruct")
    has_results: bool | None = None
    if "hasResults" in record:
        if type(record["hasResults"]) is not bool:
            raise _PayloadInvalid
        has_results = record["hasResults"]

    normalized = _base_record(
        title=title,
        authors=(),
        abstract=summary,
        snippet=None if summary is None else summary[:1_024],
        doi=None,
        pmid=None,
        pmcid=None,
        arxiv_id=None,
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        pdf_url=None,
        provider="clinicaltrials_gov",
        provider_ids={"nct_id": nct_id},
    )
    source_metadata: dict[str, Any] = {}
    for key, value in (
        ("brief_title", brief),
        ("official_title", official),
        ("overall_status", overall_status),
        ("conditions", conditions),
        ("interventions", interventions),
        ("lead_sponsor", lead_sponsor),
        ("study_type", study_type),
        ("start_date", start_date),
        ("completion_date", completion_date),
        ("has_results", has_results),
    ):
        if value is not None:
            source_metadata[key] = value
    normalized["source_metadata"] = source_metadata
    return normalized


def _clinicaltrials_page(
    payload: Any,
    *,
    guard: _ParseGuard,
    page_size: int,
) -> _ClinicalTrialsPage:
    """Validate one strict response page without applying cross-page state."""
    root = _require_dict(payload)
    total_count = root.get("totalCount", _MISSING)
    if type(total_count) is not int or total_count < 0:
        raise _PayloadInvalid
    studies = _require_list(root.get("studies", _MISSING))
    if len(studies) > page_size:
        raise _ParseLimitExceeded
    token = root.get("nextPageToken", _MISSING)
    if token is _MISSING:
        next_page_token = None
    elif (
        type(token) is not str
        or not 1 <= len(token) <= 1_024
        or any(not "!" <= character <= "~" for character in token)
    ):
        raise _PayloadInvalid
    else:
        next_page_token = token
    normalized: list[dict[str, Any]] = []
    for study in studies:
        guard.checkpoint()
        normalized.append(_clinicaltrials_record(study, guard=guard))
    return _ClinicalTrialsPage(total_count, tuple(normalized), next_page_token)


async def _execute_clinicaltrials_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    trusted, profile, max_input_bytes, page_size = _trusted_clinicaltrials_inputs(group)
    intent = trusted.intents[0]
    staged_by_nct: dict[str, dict[str, Any]] = {}
    frozen_total: int | None = None
    cumulative_raw = 0
    try:
        response = await dispatch(intent)
        for page_index in range(trusted.limits.max_pages):
            payload, guard = _strict_json(
                _checked_response(response),
                profile=profile,
                max_input_bytes=max_input_bytes,
                clock=clock,
            )
            page = _clinicaltrials_page(payload, guard=guard, page_size=page_size)
            if frozen_total is None:
                frozen_total = page.total_count
            elif page.total_count != frozen_total:
                raise _PayloadInvalid
            if frozen_total > cumulative_raw and not page.records:
                raise _PayloadInvalid

            cumulative_raw += len(page.records)
            if cumulative_raw > frozen_total or cumulative_raw > trusted.limits.max_results:
                raise _PayloadInvalid
            token_required = cumulative_raw < frozen_total
            if token_required != (page.next_page_token is not None):
                raise _PayloadInvalid

            for record in page.records:
                nct_id = cast(str, cast(dict[str, str], record["provider_ids"])["nct_id"])
                previous = staged_by_nct.get(nct_id)
                if previous is not None and previous != record:
                    raise _PayloadInvalid
                staged_by_nct.setdefault(nct_id, record)
            guard.checkpoint()

            capacity_remains = (
                page_index + 1 < trusted.limits.max_pages
                and cumulative_raw < trusted.limits.max_results
                and page_index + 1 < trusted.allowance.physical_dispatches
            )
            if not token_required or not capacity_remains:
                break
            response = await dispatch(intent, cursor=OpaqueCursor(cast(str, page.next_page_token)))

        candidates = tuple(
            DiscoveryCandidate(
                DiscoveryOutcomeIdentity.from_fingerprint(build_fingerprint(record)).document_id,
                record,
            )
            for record in staged_by_nct.values()
        )
    except (_PayloadInvalid, _ParseLimitExceeded, _ParseDeadlineExceeded) as error:
        _raise_adapter_error(error)
    except DiscoveryExecutionError:
        raise
    except DiscoveryAdapterError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError):
        raise DiscoveryAdapterError("provider_payload_invalid") from None
    return DiscoveryAdapterResult(candidates)


def _trusted_pubmed_central_inputs(
    group: object,
) -> _TrustedNCBIInputs:
    """Seal the exact PMC two-intent group and expose bounded values."""
    exact_policy = _pubmed_central_route().policy
    exact_limits = RouteLimits(1, 0, 0, 20_000, 2_097_152, 100, 16_384)
    if (
        type(group) is not PlannedDispatchGroup
        or group.route_id != "pubmed_central_esearch_summary_direct"
        or group.backend_id != "ncbi_eutils_pmc"
        or group.adapter_id != PUBMED_CENTRAL_ADAPTER_ID
        or group.adapter_version != PUBMED_CENTRAL_ADAPTER_VERSION
        or group.policy_digest != exact_policy.policy_digest
        or not _exact_pubmed_central_limits(group.limits, exact_limits)
        or type(group.normalized_query) is not str
        or not group.normalized_query
        or type(group.filters) is not tuple
        or group.filters != ()
        or type(group.logical_attempts) is not tuple
        or len(group.logical_attempts) != 1
        or type(group.logical_attempts[0]) is not PlannedLogicalAttempt
        or group.logical_attempts[0].catalog_source_id != "pubmed_central"
        or type(group.fallback_order) is not int
        or group.fallback_order != 0
        or type(group.intents) is not tuple
        or len(group.intents) != 2
        or type(group.allowance) is not DispatchAllowance
        or any(
            type(value) is not int
            for value in (
                group.allowance.physical_dispatches,
                group.allowance.pages,
                group.allowance.redirects,
                group.allowance.retries,
            )
        )
        or (
            group.allowance.physical_dispatches,
            group.allowance.pages,
            group.allowance.redirects,
            group.allowance.retries,
        )
        != (2, 1, 0, 0)
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")

    search, summary = group.intents
    if type(search) is not DispatchIntent or type(summary) is not DispatchIntent:
        raise DiscoveryAdapterError("provider_payload_invalid")
    if (
        type(search.query_pairs) is not tuple
        or type(summary.query_pairs) is not tuple
        or any(type(pair) is not QueryPair for pair in search.query_pairs + summary.query_pairs)
        or type(search.json_body_pairs) is not tuple
        or search.json_body_pairs != ()
        or type(summary.json_body_pairs) is not tuple
        or summary.json_body_pairs != ()
        or type(search.query_bindings) is not tuple
        or search.query_bindings != ()
        or type(summary.query_bindings) is not tuple
        or len(summary.query_bindings) != 1
        or not _exact_pubmed_central_limits(search.limits, exact_limits)
        or not _exact_pubmed_central_limits(summary.limits, exact_limits)
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")

    search_pairs = tuple((pair.name, pair.value) for pair in search.query_pairs)
    summary_pairs = tuple((pair.name, pair.value) for pair in summary.query_pairs)
    if (
        search.route_id != group.route_id
        or summary.route_id != group.route_id
        or search.policy_digest != group.policy_digest
        or summary.policy_digest != group.policy_digest
        or search.operation_kind is not OperationKind.SEARCH
        or summary.operation_kind is not OperationKind.CONDITIONAL_SUMMARY
        or search.method != "GET"
        or summary.method != "GET"
        or search.path != "/entrez/eutils/esearch.fcgi"
        or summary.path != "/entrez/eutils/esummary.fcgi"
        or len(search_pairs) != 7
        or len(summary_pairs) != 4
        or search_pairs[0] != ("db", "pmc")
        or search_pairs[1][0] != "term"
        or search_pairs[2][0] != "retstart"
        or search_pairs[3][0] != "retmax"
        or search_pairs[4:]
        != (
            ("retmode", "json"),
            ("tool", NCBI_TOOL),
            ("email", NCBI_EMAIL),
        )
        or summary_pairs
        != (
            ("db", "pmc"),
            ("retmode", "json"),
            ("tool", NCBI_TOOL),
            ("email", NCBI_EMAIL),
        )
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")

    expression = search_pairs[1][1]
    literal_terms = expression.split(" AND ") if type(expression) is str else ()
    normalized_terms: list[str] = []
    for term in literal_terms:
        if (
            type(term) is not str
            or len(term) < 3
            or term[0] != '"'
            or term[-1] != '"'
            or not 1 <= len(term[1:-1]) <= 64
            or not all(character.isalnum() for character in term[1:-1])
        ):
            raise DiscoveryAdapterError("provider_payload_invalid")
        normalized_terms.append(term[1:-1])
    if not 1 <= len(normalized_terms) <= 16 or " ".join(normalized_terms) != group.normalized_query:
        raise DiscoveryAdapterError("provider_payload_invalid")

    binding = summary.query_bindings[0]
    if (
        type(binding) is not DeferredNumericCSVQueryBinding
        or binding.binding_id != "pmc_esearch_ids"
        or binding.query_name != "id"
        or type(binding.max_items) is not int
        or type(binding.max_item_chars) is not int
        or binding.max_item_chars != 16
    ):
        raise DiscoveryAdapterError("provider_payload_invalid")
    try:
        retstart = _canonical_decimal_text(
            search_pairs[2][1],
            _PMC_PROFILE,
            maximum=MAX_PAGINATION_CURSOR,
        )
        retmax = _canonical_decimal_text(
            search_pairs[3][1],
            _PMC_PROFILE,
            positive=True,
            maximum=100,
        )
        if retstart != 0 or binding.max_items != retmax:
            raise _PayloadInvalid
    except _PayloadInvalid as error:
        _raise_adapter_error(error)
    return (
        group,
        _PMC_PROFILE,
        min(_PMC_PROFILE.max_input_bytes, exact_limits.max_response_bytes),
        retstart,
        retmax,
        binding,
    )


def _exact_pubmed_central_limits(value: object, expected: RouteLimits) -> bool:
    """Require every PMC route-limit scalar to retain its exact integer type."""
    if type(value) is not RouteLimits:
        return False
    actual_values = (
        value.max_pages,
        value.max_redirects,
        value.max_retries,
        value.timeout_ms,
        value.max_response_bytes,
        value.max_results,
        value.max_request_body_bytes,
    )
    expected_values = (
        expected.max_pages,
        expected.max_redirects,
        expected.max_retries,
        expected.timeout_ms,
        expected.max_response_bytes,
        expected.max_results,
        expected.max_request_body_bytes,
    )
    return all(type(item) is int for item in actual_values) and actual_values == expected_values


def _pmc_uid(value: Any, max_chars: int) -> tuple[str, int]:
    """Return one canonical PMC UID string and its transport-only number."""
    if (
        type(value) is not str
        or type(max_chars) is not int
        or len(value) > max_chars
        or re.fullmatch(r"[1-9][0-9]{0,15}", value, re.ASCII) is None
    ):
        raise _PayloadInvalid
    return value, int(value)


def _pmc_esearch_ids(
    payload: Any,
    *,
    profile: _ParsingProfile,
    guard: _ParseGuard,
    retstart: int,
    retmax: int,
    binding: DeferredNumericCSVQueryBinding,
) -> tuple[tuple[str, int], ...]:
    """Return canonical ordered PMC UIDs plus numeric binding values."""
    root = _ncbi_json_root(payload, "esearch")
    result = _require_dict(root.get("esearchresult", _MISSING))
    if "ERROR" in result:
        raise DiscoveryAdapterError("provider_response_rejected")
    count = _canonical_decimal_text(result.get("count", _MISSING), profile)
    returned_start = _canonical_decimal_text(
        result.get("retstart", _MISSING),
        profile,
        maximum=MAX_PAGINATION_CURSOR,
    )
    returned = _canonical_decimal_text(
        result.get("retmax", _MISSING),
        profile,
        maximum=retmax,
    )
    raw_ids = _require_list(result.get("idlist", _MISSING))
    _validate_ncbi_message_list(result, "errorlist")
    _validate_ncbi_message_list(result, "warninglist")
    if (
        returned_start != retstart
        or returned != len(raw_ids)
        or returned_start + returned > count
        or (count > 0 and returned == 0)
        or len(raw_ids) > binding.max_items
    ):
        raise _PayloadInvalid
    ids = tuple(_pmc_uid(value, binding.max_item_chars) for value in _guarded_items(raw_ids, guard))
    if len({uid for uid, _number in ids}) != len(ids):
        raise _PayloadInvalid
    return ids


def _pmc_identifier_scalar(value: Any, *, max_chars: int) -> str:
    """Validate a bounded identifier scalar without applying human URL rules."""
    if (
        type(value) is not str
        or not 1 <= len(value) <= max_chars
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value)
        or "<" in value
        or ">" in value
    ):
        raise _PayloadInvalid
    return value


def _plain_pmc_text(
    value: Any,
    *,
    max_chars: int,
    required: bool,
) -> str | None:
    """Normalize human text and reject controls, markup, or any URL token."""
    invalid = (
        type(value) is not str
        or len(value) > max_chars
        or _has_forbidden_text_character(value)
        or any(unicodedata.category(character) == "Cf" for character in value)
        or _contains_residual_markup(value)
        or _contains_url_material(value)
    )
    normalized = None if invalid else " ".join(value.split())
    if normalized is not None and (not normalized or len(normalized) > max_chars):
        normalized = None
    if normalized is None and required:
        raise _PayloadInvalid
    return normalized


def _pmc_article_ids(
    raw: Any,
    expected_uid: str,
    guard: _ParseGuard,
) -> tuple[str, str | None, str | None]:
    """Return required PMCID and optional DOI/PMID."""
    article_ids = _require_list(raw)
    if len(article_ids) > 64:
        raise _ParseLimitExceeded
    recognized: dict[str, str] = {}
    for item in _guarded_items(article_ids, guard):
        identifier = _require_dict(item)
        if set(identifier) != {"idtype", "value"}:
            raise _PayloadInvalid
        idtype = _pmc_identifier_scalar(identifier["idtype"], max_chars=32)
        value = _pmc_identifier_scalar(identifier["value"], max_chars=512)
        if idtype in {"pmcid", "doi", "pmid"}:
            if idtype in recognized:
                raise _PayloadInvalid
            recognized[idtype] = value

    pmcid = recognized.get("pmcid")
    if pmcid != f"PMC{expected_uid}" or re.fullmatch(r"PMC[1-9][0-9]{0,15}", pmcid or "", re.ASCII) is None:
        raise _PayloadInvalid
    doi = None if "doi" not in recognized else normalize_doi(recognized["doi"])
    if "doi" in recognized and doi is None:
        raise _PayloadInvalid
    raw_pmid = recognized.get("pmid")
    pmid = None if raw_pmid in {None, "0"} else raw_pmid
    if pmid is not None and re.fullmatch(r"[1-9][0-9]{0,15}", pmid, re.ASCII) is None:
        raise _PayloadInvalid
    return pmcid, doi, pmid


def _pmc_record(raw: Any, expected_uid: str, guard: _ParseGuard) -> dict[str, Any]:
    """Normalize one PMC ESummary record without retaining the numeric UID."""
    record = _require_dict(raw)
    uid = record.get("uid", _MISSING)
    if _pmc_uid(uid, 16)[0] != expected_uid:
        raise _PayloadInvalid
    title = _plain_pmc_text(record.get("title", _MISSING), max_chars=4_096, required=True)
    authors_raw = _require_list(record.get("authors", []))
    if len(authors_raw) > 64:
        raise _ParseLimitExceeded
    authors = tuple(
        cast(
            str,
            _plain_pmc_text(
                _require_dict(author).get("name", _MISSING),
                max_chars=512,
                required=True,
            ),
        )
        for author in _guarded_items(authors_raw, guard)
    )
    pmcid, doi, pmid = _pmc_article_ids(record.get("articleids", _MISSING), expected_uid, guard)
    provider_ids = {"pmcid": pmcid}
    if doi is not None:
        provider_ids["doi"] = doi
    if pmid is not None:
        provider_ids["pmid"] = pmid
    return _base_record(
        title=cast(str, title),
        authors=authors,
        abstract=None,
        snippet=None,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        arxiv_id=None,
        url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        pdf_url=None,
        provider="pubmed_central",
        provider_ids=provider_ids,
    )


def _pmc_summary_records(
    payload: Any,
    *,
    expected_ids: tuple[str, ...],
    guard: _ParseGuard,
) -> tuple[dict[str, Any], ...]:
    """Validate an exact UID-keyed result and restore ESearch ordering."""
    root = _ncbi_json_root(payload, "esummary")
    result = _require_dict(root.get("result", _MISSING))
    raw_uids = _require_list(result.get("uids", _MISSING))
    if (
        len(raw_uids) != len(expected_ids)
        or any(type(uid) is not str for uid in raw_uids)
        or set(raw_uids) != set(expected_ids)
        or set(result) != {"uids", *expected_ids}
    ):
        raise _PayloadInvalid
    records: list[dict[str, Any]] = []
    for expected_id in expected_ids:
        guard.checkpoint()
        records.append(_pmc_record(result[expected_id], expected_id, guard))
    return tuple(records)


async def _execute_pubmed_central_adapter(
    group: object,
    dispatch: BoundDispatch,
    clock: MonotonicClock,
) -> DiscoveryAdapterResult:
    return await _execute_ncbi_esearch_summary(
        group,
        dispatch,
        clock,
        trusted_inputs=_trusted_pubmed_central_inputs,
        parse_esearch_ids=_pmc_esearch_ids,
        parse_summary_records=_pmc_summary_records,
        strict_rate_envelope=True,
    )


def _compose_adapter_maps(
    *adapter_maps: Mapping[str, DiscoveryAdapter],
) -> Mapping[str, DiscoveryAdapter]:
    """Compose the exact reviewed family adapter set without duplicate IDs."""
    composed: dict[str, DiscoveryAdapter] = {}
    for adapter_map in adapter_maps:
        for adapter_id, adapter in adapter_map.items():
            if adapter_id in composed:
                raise ValueError(f"duplicate_adapter_id:{adapter_id}")
            composed[adapter_id] = adapter
    return MappingProxyType(composed)


def clinicaltrials_pubmed_central_gateway_adapters(
    *,
    monotonic_clock: MonotonicClock = time.monotonic,
) -> Mapping[str, DiscoveryAdapter]:
    """Return only the two fixture-ready adapters owned by this family."""

    async def clinicaltrials_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_clinicaltrials_adapter(group, dispatch, monotonic_clock)

    async def pubmed_central_adapter(
        group: PlannedDispatchGroup,
        dispatch: BoundDispatch,
    ) -> DiscoveryAdapterResult:
        return await _execute_pubmed_central_adapter(group, dispatch, monotonic_clock)

    return _compose_adapter_maps(
        {CLINICALTRIALS_GOV_ADAPTER_ID: clinicaltrials_adapter},
        {PUBMED_CENTRAL_ADAPTER_ID: pubmed_central_adapter},
    )
