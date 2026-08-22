"""Shadow-only ClinicalTrials.gov and PubMed Central discovery family."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from html.parser import HTMLParser
from types import MappingProxyType
from typing import Any, cast

from .contracts import (
    AccessRoute,
    BackendDefinition,
    BoundedDecimalQueryValuePolicy,
    CredentialRequirement,
    DiscoveryOutcomeIdentity,
    ExactOrigin,
    ExactQueryValuePolicy,
    LiteralTermsQueryValuePolicy,
    OpaqueCursorQueryValuePolicy,
    OperationKind,
    PlannedDispatchGroup,
    QueryMode,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    SourceConstraint,
    SourceDefinition,
    SourceRouteReference,
)
from .executor import (
    BoundDispatch,
    DiscoveryAdapterError,
    DiscoveryAdapterResult,
    DiscoveryCandidate,
    DiscoveryExecutionError,
    OpaqueCursor,
)
from .gateway_adapters import (
    MonotonicClock,
    _base_record,
    _checked_response,
    _ParseDeadlineExceeded,
    _ParseGuard,
    _ParseLimitExceeded,
    _ParsingProfile,
    _PayloadInvalid,
    _raise_adapter_error,
    _require_dict,
    _require_list,
    _strict_json,
)
from .identity import build_fingerprint, has_unsafe_url_material
from .registry import DiscoveryRegistry, foundation_registry

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
_MISSING = object()
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
    """Return the foundation, PubMed identity overlay, and ClinicalTrials route."""
    foundation = foundation_registry()
    routes = tuple(
        _pubmed_identity_overlay(route) if route.route_id == _PUBMED_ROUTE_ID else route for route in foundation.routes
    )
    return DiscoveryRegistry(
        catalog_version=SHADOW_CATALOG_VERSION,
        registry_version=SHADOW_REGISTRY_VERSION,
        sources=tuple(replace(source, catalog_version=SHADOW_CATALOG_VERSION) for source in foundation.sources)
        + (_clinicaltrials_source(),),
        routes=routes + (_clinicaltrials_route(),),
        backends=foundation.backends + (BackendDefinition("clinicaltrials_gov_api_v2", "ClinicalTrials.gov API v2"),),
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
    pairs = tuple((pair.name, pair.value) for pair in intent.query_pairs)
    if (
        intent.route_id != group.route_id
        or intent.route_id != "clinicaltrials_gov_studies_search_direct"
        or intent.operation_kind is not OperationKind.SEARCH
        or intent.method != "GET"
        or intent.path != "/api/v2/studies"
        or intent.limits != limits
        or intent.policy_digest != group.policy_digest
        or intent.json_body_pairs
        or intent.query_bindings
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
